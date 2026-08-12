"""app/agents/base.py 单元测试：验证超时/异常捕获是基类统一行为，不依赖真实API。"""
import asyncio

import pytest

from app.agents.base import AgentRunResult, BaseAgent


class _SlowAgent(BaseAgent):
    name = "slow_test_agent"
    timeout_seconds = 0.05

    async def _run(self, **kwargs):
        await asyncio.sleep(0.5)
        return AgentRunResult(output="should not reach here", tokens_used=0, tool_calls_used=0)


class _FailingAgent(BaseAgent):
    name = "failing_test_agent"

    async def _run(self, **kwargs):
        raise ValueError("boom")


class _OkAgent(BaseAgent):
    name = "ok_test_agent"

    async def _run(self, **kwargs):
        return AgentRunResult(output="ok", tokens_used=5, tool_calls_used=1)


@pytest.mark.asyncio
async def test_base_agent_times_out_instead_of_hanging():
    result = await _SlowAgent().run()
    assert result.error is not None
    assert "timeout" in result.error


@pytest.mark.asyncio
async def test_base_agent_catches_exception_instead_of_raising():
    result = await _FailingAgent().run()
    assert result.error is not None
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_base_agent_success_path_sets_elapsed_time():
    result = await _OkAgent().run()
    assert result.error is None
    assert result.output == "ok"
    assert result.elapsed_seconds >= 0
