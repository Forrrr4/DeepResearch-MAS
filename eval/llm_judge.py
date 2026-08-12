"""LLM-as-judge评分脚本，维度定义见docs/04_M1评估集设计说明.md第2节：
accuracy / completeness / citation_validity / honesty，each 0-5分。

judge的角色被严格限制成"检查回答有没有覆盖evaluation_criteria里的要点
清单"，不做二次事实核查（不联网搜索验证，避免judge自己的时效性问题
干扰评分一致性——这是04文档里明确写的设计原则，不是本脚本随意决定的）。

用法：
    python eval/llm_judge.py --eval-set eval/eval_set.jsonl \\
        --results eval/results/m1_baseline_xxx.jsonl \\
        --tag m1_baseline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.infra.model_router import create_message_with_retry, get_client, resolve_model

JUDGE_SYSTEM_PROMPT = """你是一个研究报告质量评审员(judge)，负责对一份AI生成的研究回答打分。

严格规则：
1. 你只依据"evaluation_criteria"（一份好答案应包含的要点清单）来打分，
   不能凭自己的知识去现场核实回答里的事实是否正确——你的角色是"检查
   回答有没有覆盖要点清单"，不是"重新做一遍研究"。
2. 从0-5分别给以下4个维度打分（整数）：
   - accuracy（准确率）：回答内容命中evaluation_criteria里要点的程度，
     命中越多分越高
   - completeness（完整性）：是否有明显遗漏的重要角度（尤其多角度对比
     类问题），结构是否完整
   - citation_validity（引用有效性）：回答中的关键论断是否有来源支撑
     （无论是[来源: URL]还是[S1]这类编号引用），没有来源支撑的确定性
     断言应扣分
   - honesty（诚实度）：遇到不确定/矛盾信息时是否如实说明而非强行给出
     确定结论；这一项对adversarial类问题尤其重要——如果回答对一个业界
     本身有争议的问题强行给出"唯一正确答案"，这一项应给低分
3. 必须调用submit_score工具提交打分，不要用文字回答。
"""

SUBMIT_SCORE_TOOL_SCHEMA = {
    "name": "submit_score",
    "description": "提交对这份回答的评分。",
    "input_schema": {
        "type": "object",
        "properties": {
            "accuracy": {"type": "integer", "minimum": 0, "maximum": 5},
            "completeness": {"type": "integer", "minimum": 0, "maximum": 5},
            "citation_validity": {"type": "integer", "minimum": 0, "maximum": 5},
            "honesty": {"type": "integer", "minimum": 0, "maximum": 5},
            "rationale": {"type": "string", "description": "一到两句话说明打分理由"},
        },
        "required": ["accuracy", "completeness", "citation_validity", "honesty", "rationale"],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def judge_one(
    eval_item: dict[str, Any], result_item: dict[str, Any], semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    answer = result_item.get("answer") or ""
    if not answer.strip():
        # 空回答直接判0分，不需要浪费一次LLM调用去"评审空白"
        return {
            "id": eval_item["id"],
            "category": eval_item["category"],
            "adversarial": eval_item["adversarial"],
            "accuracy": 0,
            "completeness": 0,
            "citation_validity": 0,
            "honesty": 0,
            "rationale": "回答为空，直接判0分（未经过LLM judge）。",
        }

    criteria_text = "\n".join(f"- {c}" for c in eval_item["evaluation_criteria"])
    user_content = (
        f"问题: {eval_item['query']}\n\n"
        f"evaluation_criteria（要点清单）:\n{criteria_text}\n\n"
        f"待评审的回答:\n{answer}"
    )

    async with semaphore:
        client = get_client()
        resp = await create_message_with_retry(
            client,
            model=resolve_model("pro"),
            # 2048而不是1024：DeepSeek端点默认开thinking，评审长回答时thinking
            # 本身会占用不少输出token，1024实测会导致部分长回答（例如
            # eval-011/eval-013这类几千字的报告）judge没能完整生成
            # submit_score调用，评分缺失（复现过2次，和app/agents/sub_agent.py
            # 里同一类max_tokens截断问题同源）。
            max_tokens=2048,
            system=JUDGE_SYSTEM_PROMPT,
            tools=[SUBMIT_SCORE_TOOL_SCHEMA],
            messages=[{"role": "user", "content": user_content}],
        )

    block = next((b for b in resp.content if b.type == "tool_use" and b.name == "submit_score"), None)
    if block is None:
        return {
            "id": eval_item["id"],
            "category": eval_item["category"],
            "adversarial": eval_item["adversarial"],
            "accuracy": None,
            "completeness": None,
            "citation_validity": None,
            "honesty": None,
            "rationale": "judge未能正确调用submit_score工具，打分缺失。",
        }

    return {
        "id": eval_item["id"],
        "category": eval_item["category"],
        "adversarial": eval_item["adversarial"],
        "accuracy": block.input.get("accuracy"),
        "completeness": block.input.get("completeness"),
        "citation_validity": block.input.get("citation_validity"),
        "honesty": block.input.get("honesty"),
        "rationale": block.input.get("rationale", ""),
    }


async def judge_all(
    eval_set: list[dict[str, Any]], results: list[dict[str, Any]], concurrency: int
) -> list[dict[str, Any]]:
    results_by_id = {r["id"]: r for r in results}
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        judge_one(item, results_by_id[item["id"]], semaphore)
        for item in eval_set
        if item["id"] in results_by_id
    ]
    return await asyncio.gather(*tasks)


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    dims = ["accuracy", "completeness", "citation_validity", "honesty"]
    valid = [s for s in scores if s["accuracy"] is not None]
    n = len(valid)
    summary: dict[str, Any] = {"n_scored": n, "n_missing_score": len(scores) - n}
    for dim in dims:
        total = sum(s[dim] for s in valid)
        summary[f"avg_{dim}"] = round(total / n, 2) if n else None

    adversarial_valid = [s for s in valid if s["adversarial"]]
    if adversarial_valid:
        summary["avg_honesty_on_adversarial"] = round(
            sum(s["honesty"] for s in adversarial_valid) / len(adversarial_valid), 2
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out-dir", default="eval/results")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    eval_set = load_jsonl(Path(args.eval_set))
    results = load_jsonl(Path(args.results))
    scores = asyncio.run(judge_all(eval_set, results, args.concurrency))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"judge_{args.tag}_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = summarize(scores)
    summary_path = out_dir / f"judge_{args.tag}_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"打分结果写入: {out_path}")
    print(f"汇总写入: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
