"""M2评估脚本：跑同一份eval_set.jsonl，走完整的LangGraph图
（orchestrator → single_agent | parallel_research），记录路由决策、
耗时、token消耗，用于和M1 baseline做对比（M2验收标准之一）。

用法：
    python eval/run_eval_m2.py [--eval-set eval/eval_set.jsonl] [--concurrency 3]

产出：eval/results/m2_orchestrator_<timestamp>.jsonl + _summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.graph.build_graph import build_graph
from app.graph.state import make_initial_state


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run_one(graph: Any, item: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        start = time.monotonic()
        state = make_initial_state(item["query"], trace_id=item["id"])
        result = await graph.ainvoke(state)
        elapsed = time.monotonic() - start

    return {
        "id": item["id"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "should_trigger_multi_agent": item.get("should_trigger_multi_agent"),
        "query": item["query"],
        "routing_decision": result["routing_decision"],
        "n_subtasks": len(result["subtasks"]),
        "subtask_goals": [s.goal for s in result["subtasks"]],
        "n_findings": len(result["findings"]),
        "subtask_errors": result["subtask_errors"],
        "answer": result["final_report"],
        "elapsed_seconds": round(elapsed, 2),
    }


async def run_eval(eval_set_path: Path, concurrency: int) -> list[dict[str, Any]]:
    items = load_eval_set(eval_set_path)
    graph = build_graph()
    # 图内部（parallel_research）已经有一层SubAgent并发限制，这里的semaphore
    # 限制的是"同时跑几道eval题"，避免多题同时触发多个多agent流程时叠加出
    # 过高的整体并发，两层是不同粒度的限流，不冲突。
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [run_one(graph, item, semaphore) for item in items]
    return await asyncio.gather(*tasks)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    n_multi_agent = sum(1 for r in results if r["routing_decision"] == "multi_agent")
    routing_correct = sum(
        1
        for r in results
        if r.get("should_trigger_multi_agent") is not None
        and (r["routing_decision"] == "multi_agent") == r["should_trigger_multi_agent"]
    )
    n_with_expectation = sum(1 for r in results if r.get("should_trigger_multi_agent") is not None)
    n_subtask_errors = sum(len(r["subtask_errors"]) for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    return {
        "n_questions": n,
        "n_routed_multi_agent": n_multi_agent,
        "n_routed_single_agent": n - n_multi_agent,
        "routing_accuracy_vs_expected": (
            round(routing_correct / n_with_expectation, 3) if n_with_expectation else None
        ),
        "n_with_expectation_label": n_with_expectation,
        "total_subtask_level_errors": n_subtask_errors,
        "sum_elapsed_seconds_across_questions": round(total_time, 1),
        "avg_elapsed_seconds_per_question": round(total_time / n, 2) if n else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.jsonl")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out-dir", default="eval/results")
    parser.add_argument("--tag", default="m2_orchestrator")
    args = parser.parse_args()

    wall_start = time.monotonic()
    results = asyncio.run(run_eval(Path(args.eval_set), args.concurrency))
    wall_elapsed = time.monotonic() - wall_start

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{args.tag}_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(results)
    summary["wall_clock_seconds_total_run"] = round(wall_elapsed, 1)
    summary["concurrency"] = args.concurrency
    summary_path = out_dir / f"{args.tag}_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"结果写入: {out_path}")
    print(f"汇总写入: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
