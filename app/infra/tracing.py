"""极简可观测性：每次agent调用生成一条trace，写入本地 logs/trace.jsonl。

按 docs/05_自动化实施指南 第1节的决定，不引入 LangSmith 等第三方依赖，
减少一个外部依赖，自建tracing表即可满足当前阶段需求。
"""
import json
import time
from pathlib import Path
from typing import Any

_TRACE_PATH = Path("logs/trace.jsonl")


def record_trace(
    *,
    trace_id: str,
    agent_name: str,
    input_summary: str,
    output_summary: str,
    tokens_used: int,
    tool_calls: int,
    latency_seconds: float,
    error: str | None = None,
) -> None:
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "trace_id": trace_id,
        "agent_name": agent_name,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "tokens_used": tokens_used,
        "tool_calls": tool_calls,
        "latency_seconds": latency_seconds,
        "error": error,
        "timestamp": time.time(),
    }
    with _TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
