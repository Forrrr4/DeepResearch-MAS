"""Agent基类：统一封装超时/trace记录，所有具体agent必须继承这里。

CLAUDE.md 硬性约束要求重试/超时/预算检查逻辑在基类里统一实现，不允许
每个agent各写一套。M1阶段预算检查还很简单（只有单agent自己控制的
max_tool_calls），完整的跨agent全局 BudgetTracker 在M4引入，届时
Orchestrator会在调度前查询它，本基类的职责保持不变：超时+trace。

工具调用级别的重试（2次指数退避）在 app/tools/ 各工具内部实现，不在
这里，因为“要不要重试”和“重试几次”是工具语义的一部分，agent只关心
拿到的是不是一个可用结果。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.infra.tracing import record_trace


@dataclass
class AgentRunResult:
    output: str
    tokens_used: int
    tool_calls_used: int
    elapsed_seconds: float = 0.0  # 由 BaseAgent.run() 用wall-clock覆盖，子类不需要自己计时
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base_agent"
    timeout_seconds: float = 90.0

    @abstractmethod
    async def _run(self, **kwargs: Any) -> AgentRunResult:
        """具体agent实现自己的推理/工具调用逻辑。"""

    async def run(
        self, *, trace_id: str | None = None, input_summary: str = "", **kwargs: Any
    ) -> AgentRunResult:
        trace_id = trace_id or str(uuid.uuid4())
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(self._run(**kwargs), timeout=self.timeout_seconds)
            result.elapsed_seconds = time.monotonic() - start
        except asyncio.TimeoutError:
            result = AgentRunResult(
                output="",
                tokens_used=0,
                tool_calls_used=0,
                elapsed_seconds=time.monotonic() - start,
                error=f"timeout after {self.timeout_seconds}s",
            )
        except Exception as exc:  # noqa: BLE001 - agent边界必须捕获+记录，不允许静默失败
            result = AgentRunResult(
                output="",
                tokens_used=0,
                tool_calls_used=0,
                elapsed_seconds=time.monotonic() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

        record_trace(
            trace_id=trace_id,
            agent_name=self.name,
            input_summary=input_summary,
            output_summary=result.output[:200],
            tokens_used=result.tokens_used,
            tool_calls=result.tool_calls_used,
            latency_seconds=result.elapsed_seconds,
            error=result.error,
        )
        return result
