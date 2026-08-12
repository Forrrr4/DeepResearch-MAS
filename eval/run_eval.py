"""M1 baseline评估脚本：跑 eval_set.jsonl 里的题目，记录真实输出/耗时/
token消耗/是否调用了搜索工具，不做任何LLM-as-judge打分（那是M5的事）。

用法：
    python eval/run_eval.py [--eval-set eval/eval_set.jsonl] [--concurrency 3]

产出：eval/results/m1_baseline_<timestamp>.jsonl（逐题结果）+
      eval/results/m1_baseline_<timestamp>_summary.json（汇总指标）
后续 M5 的对比报告会读取这份M1结果作为baseline。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.agents.baseline_agent import BaselineAgent


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run_one(item: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        agent = BaselineAgent()
        result = await agent.run(query=item["query"], input_summary=item["id"])
    return {
        "id": item["id"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "query": item["query"],
        "answer": result.output,
        "tokens_used": result.tokens_used,
        "tool_calls_used": result.tool_calls_used,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "error": result.error,
    }


async def run_eval(eval_set_path: Path, concurrency: int) -> list[dict[str, Any]]:
    items = load_eval_set(eval_set_path)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [run_one(item, semaphore) for item in items]
    return await asyncio.gather(*tasks)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    n_errors = sum(1 for r in results if r["error"])
    n_used_search = sum(1 for r in results if r["tool_calls_used"] > 0)
    total_tokens = sum(r["tokens_used"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    return {
        "n_questions": n,
        "n_errors": n_errors,
        "n_used_search": n_used_search,
        "total_tokens": total_tokens,
        "avg_tokens_per_question": round(total_tokens / n, 1) if n else 0,
        "sum_elapsed_seconds_across_questions": round(total_time, 1),
        "avg_elapsed_seconds_per_question": round(total_time / n, 2) if n else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.jsonl")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out-dir", default="eval/results")
    parser.add_argument("--tag", default="m1_baseline")
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
