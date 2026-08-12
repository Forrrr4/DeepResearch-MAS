"""app/agents/sub_agent.py 单元测试：mock LLM/web_search，不打真实API。

重点测试 parse_findings 的代码层强制：没有来源就不能是verified状态，
这是CLAUDE.md第2条硬性约束在SubAgent层面的落地。
"""
import types
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.sub_agent import SubAgent, parse_findings
from app.graph.state import SourceRef, SubtaskSpec


def test_parse_findings_downgrades_verified_without_source_to_unverified():
    raw = [{"claim": "无来源的断言", "source_urls": [], "confidence": 0.9, "status": "verified"}]
    known_sources: dict[str, SourceRef] = {}

    findings = parse_findings(raw, "sub-1", known_sources)

    assert findings[0].status == "unverified"
    assert findings[0].source_ids == []


def test_parse_findings_keeps_verified_when_source_present():
    raw = [
        {
            "claim": "有来源的断言",
            "source_urls": ["https://example.com/a"],
            "confidence": 0.8,
            "status": "verified",
        }
    ]
    known_sources: dict[str, SourceRef] = {}

    findings = parse_findings(raw, "sub-1", known_sources)

    assert findings[0].status == "verified"
    assert "https://example.com/a" in known_sources


def test_parse_findings_registers_all_cited_urls_as_sources():
    raw = [
        {
            "claim": "c1",
            "source_urls": ["https://a.com", "https://b.com"],
            "confidence": 0.5,
            "status": "unverified",
        }
    ]
    known_sources: dict[str, SourceRef] = {}

    parse_findings(raw, "sub-1", known_sources)

    assert set(known_sources.keys()) == {"https://a.com", "https://b.com"}


def _submit_response(findings: list[dict], input_tokens=10, output_tokens=20):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[
            types.SimpleNamespace(
                type="tool_use", id="submit_1", name="submit_findings", input={"findings": findings}
            )
        ],
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.mark.asyncio
async def test_sub_agent_submits_findings_without_searching_when_confident():
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=AsyncMock(
                return_value=_submit_response(
                    [{"claim": "c", "source_urls": [], "confidence": 0.3, "status": "unverified"}]
                )
            )
        )
    )
    subtask = SubtaskSpec(id="sub-1", goal="调研一个简单问题")
    with patch("app.agents.sub_agent.get_client", return_value=fake_client):
        result = await SubAgent().run(subtask=subtask)

    assert result.error is None
    assert result.tool_calls_used == 0
    assert len(result.raw["findings"]) == 1


@pytest.mark.asyncio
async def test_sub_agent_errors_when_ending_without_submit_findings():
    no_tool_response = types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text="我直接说完了")],
        usage=types.SimpleNamespace(input_tokens=5, output_tokens=5),
    )
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=AsyncMock(return_value=no_tool_response))
    )
    subtask = SubtaskSpec(id="sub-1", goal="调研一个问题")
    with patch("app.agents.sub_agent.get_client", return_value=fake_client):
        result = await SubAgent().run(subtask=subtask)

    assert result.error is not None
    assert "submit_findings" in result.error
