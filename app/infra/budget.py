"""预算控制：CLAUDE.md第3条硬性约束——"全局预算每轮迭代前检查，余量不足
直接进入收敛分支"的具体实现。

这里只做两件确定性的事：记录用量、判断是否超限。"超限之后要不要收敛、
怎么收敛"是图路由层的决策（见app/graph/build_graph.py的orchestrator_node
和route_after_orchestrator），budget.py本身不涉及LLM，也不做任何流程
控制决策，职责边界很窄，方便单测。
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetLimits:
    max_tokens: int
    max_tool_calls: int
    max_wall_clock_seconds: float


def new_budget() -> dict[str, float]:
    return {"tokens_used": 0, "tool_calls_used": 0, "wall_clock_start": time.monotonic()}


def add_usage(budget: dict, *, tokens: int = 0, tool_calls: int = 0) -> dict:
    """返回一个新的budget dict（不原地修改原对象，符合LangGraph节点返回
    增量更新的惯例），累加token/工具调用消耗。"""
    updated = dict(budget)
    updated["tokens_used"] = budget.get("tokens_used", 0) + tokens
    updated["tool_calls_used"] = budget.get("tool_calls_used", 0) + tool_calls
    return updated


def elapsed_seconds(budget: dict) -> float:
    start = budget.get("wall_clock_start")
    if start is None:
        return 0.0
    return time.monotonic() - start


def is_exhausted(budget: dict, limits: BudgetLimits) -> bool:
    return (
        budget.get("tokens_used", 0) >= limits.max_tokens
        or budget.get("tool_calls_used", 0) >= limits.max_tool_calls
        or elapsed_seconds(budget) >= limits.max_wall_clock_seconds
    )
