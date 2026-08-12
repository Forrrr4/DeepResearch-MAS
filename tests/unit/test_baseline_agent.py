"""app/agents/baseline_agent.py 单元测试：mock LLM响应和web_search，不打真实API。"""
import types
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.baseline_agent import BaselineAgent


def _text_response(text: str, input_tokens: int = 10, output_tokens: int = 20):
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text=text)],
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _tool_use_response(
    tool_name: str, tool_input: dict, tool_id: str = "tool_1", input_tokens: int = 10, output_tokens: int = 20
):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[types.SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input)],
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.mark.asyncio
async def test_baseline_agent_answers_without_tool_call_for_simple_question():
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=AsyncMock(return_value=_text_response("RAG是检索增强生成")))
    )
    agent = BaselineAgent()
    with patch("app.agents.baseline_agent.get_client", return_value=fake_client):
        result = await agent.run(query="什么是RAG")

    assert result.error is None
    assert "RAG" in result.output
    assert result.tool_calls_used == 0
    assert result.tokens_used == 30


@pytest.mark.asyncio
async def test_baseline_agent_uses_web_search_then_answers():
    responses = [
        _tool_use_response("web_search", {"query": "最新Claude模型"}),
        _text_response("最新模型是Sonnet 5 [来源: https://example.com]"),
    ]
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=AsyncMock(side_effect=responses))
    )
    fake_search_result = {
        "query": "最新Claude模型",
        "results": [
            {"title": "T", "url": "https://example.com", "content": "c", "published_date": None}
        ],
        "error": None,
    }
    agent = BaselineAgent()
    with patch("app.agents.baseline_agent.get_client", return_value=fake_client), patch(
        "app.agents.baseline_agent.web_search", new=AsyncMock(return_value=fake_search_result)
    ):
        result = await agent.run(query="最新的Claude模型有哪些")

    assert result.error is None
    assert result.tool_calls_used == 1
    assert "https://example.com" in result.output


@pytest.mark.asyncio
async def test_baseline_agent_stops_calling_tools_once_budget_exhausted():
    responses = [
        _tool_use_response("web_search", {"query": "q1"}, tool_id="t1"),
        _text_response("基于已有信息的回答"),
    ]
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=AsyncMock(side_effect=responses))
    )
    agent = BaselineAgent(max_tool_calls=0)
    with patch("app.agents.baseline_agent.get_client", return_value=fake_client), patch(
        "app.agents.baseline_agent.web_search", new=AsyncMock()
    ) as mock_search:
        result = await agent.run(query="任意问题")

    mock_search.assert_not_called()
    assert result.tool_calls_used == 0
    assert result.error is None
