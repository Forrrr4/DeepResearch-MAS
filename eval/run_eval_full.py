"""M5评估脚本：跑当前完整系统（Orchestrator+并行SubAgent+Critic+Citation
Manager+Writer+预算控制，即M1-M4全部能力）在eval_set.jsonl上的表现，
记录真实的路由决策、token消耗、延迟、citation数量、最终报告，供
llm_judge.py打分和comparison_report.md做M1 vs 完整系统的对比。

用法：
    python eval/run_eval_full.py [--eval-set eval/eval_set.jsonl] [--concurrency 2]
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
        "n_findings": len(result["findings"]),
        "n_citations": len(result["citation_map"]),
        "iteration": result["iteration"],
        "subtask_errors": result["subtask_errors"],
        "answer": result["final_report"],
        "tokens_used": result["budget"].get("tokens_used", 0),
        "tool_calls_used": result["budget"].get("tool_calls_used", 0),
        "elapsed_seconds": round(elapsed, 2),
    }


async def run_eval(eval_set_path: Path, concurrency: int) -> list[dict[str, Any]]:
    items = load_eval_set(eval_set_path)
    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [run_one(graph, item, semaphore) for item in items]
    return await asyncio.gather(*tasks)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    n_multi_agent = sum(1 for r in results if r["routing_decision"] == "multi_agent")
    n_converge = sum(1 for r in results if r["routing_decision"] == "converge")
    routing_correct = sum(
        1
        for r in results
        if r.get("should_trigger_multi_agent") is not None
        and (r["routing_decision"] == "multi_agent") == r["should_trigger_multi_agent"]
    )
    n_with_expectation = sum(1 for r in results if r.get("should_trigger_multi_agent") is not None)
    total_tokens = sum(r["tokens_used"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    empty_reports = sum(1 for r in results if not (r["answer"] or "").strip())
    return {
        "n_questions": n,
        "n_routed_multi_agent": n_multi_agent,
        "n_routed_single_agent": n - n_multi_agent - n_converge,
        "n_routed_converge": n_converge,
        "routing_accuracy_vs_expected": (
            round(routing_correct / n_with_expectation, 3) if n_with_expectation else None
        ),
        "total_tokens": total_tokens,
        "avg_tokens_per_question": round(total_tokens / n, 1) if n else 0,
        "avg_elapsed_seconds_per_question": round(total_time / n, 2) if n else 0,
        "n_empty_reports": empty_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.jsonl")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out-dir", default="eval/results")
    parser.add_argument("--tag", default="full_system")
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
