"""app/agents/orchestrator.py 单元测试。

check_subtasks_orthogonal / validate_subtask_count 是纯函数，直接测，
不需要mock。Orchestrator._run的LLM交互部分用mock验证重试/降级流程。
"""
import types
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.orchestrator import (
    MAX_REPLAN_ATTEMPTS,
    Orchestrator,
    check_subtasks_orthogonal,
    validate_subtask_count,
)


def test_orthogonal_goals_pass():
    goals = ["调研A公司的技术路线", "调研B公司的融资历史", "调研C市场的监管政策"]
    result = check_subtasks_orthogonal(goals)
    assert result.is_orthogonal
    assert result.overlapping_pairs == []


def test_near_duplicate_goals_flagged_as_overlapping():
    goals = ["调研LangGraph框架的优点", "调研LangGraph框架的好处"]
    result = check_subtasks_orthogonal(goals)
    assert not result.is_orthogonal
    assert len(result.overlapping_pairs) == 1


def test_subtask_count_validation_bounds():
    assert validate_subtask_count(2) is True
    assert validate_subtask_count(6) is True
    assert validate_subtask_count(1) is False
    assert validate_subtask_count(7) is False


def _plan_response(complexity: str, subtasks: list[dict], input_tokens=10, output_tokens=20):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[
            types.SimpleNamespace(
                type="tool_use",
                id="plan_1",
                name="propose_plan",
                input={"complexity": complexity, "subtasks": subtasks},
            )
        ],
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.mark.asyncio
async def test_orchestrator_routes_simple_query_to_single_agent():
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=AsyncMock(return_value=_plan_response("simple", []))
        )
    )
    with patch("app.agents.orchestrator.get_client", return_value=fake_client):
        result = await Orchestrator().run(query="什么是RAG")

    assert result.error is None
    assert result.raw["routing_decision"] == "single_agent"
    assert result.raw["subtasks"] == []


@pytest.mark.asyncio
async def test_orchestrator_routes_complex_query_to_multi_agent_with_valid_plan():
    subtasks = [
        {"id": "sub-1", "goal": "调研LangGraph的状态管理机制"},
        {"id": "sub-2", "goal": "调研AutoGen的消息传递机制"},
        {"id": "sub-3", "goal": "调研CrewAI的角色分工机制"},
    ]
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=AsyncMock(return_value=_plan_response("complex", subtasks))
        )
    )
    with patch("app.agents.orchestrator.get_client", return_value=fake_client):
        result = await Orchestrator().run(query="对比三个框架")

    assert result.error is None
    assert result.raw["routing_decision"] == "multi_agent"
    assert len(result.raw["subtasks"]) == 3
    assert result.raw["subtasks"][0].id == "sub-1"


@pytest.mark.asyncio
async def test_orchestrator_falls_back_to_single_agent_after_max_replan_attempts():
    # 每次都返回重叠严重的子任务，逼迫orchestrator耗尽重试上限后降级
    bad_subtasks = [
        {"id": "sub-1", "goal": "调研LangGraph框架的优点"},
        {"id": "sub-2", "goal": "调研LangGraph框架的好处"},
    ]
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=AsyncMock(return_value=_plan_response("complex", bad_subtasks))
        )
    )
    with patch("app.agents.orchestrator.get_client", return_value=fake_client):
        result = await Orchestrator().run(query="随便一个问题")

    assert result.error is None
    assert result.raw["routing_decision"] == "single_agent"
    assert "fallback_reason" in result.raw
    # 初次 + MAX_REPLAN_ATTEMPTS 次重试
    assert fake_client.messages.create.call_count == MAX_REPLAN_ATTEMPTS + 1
