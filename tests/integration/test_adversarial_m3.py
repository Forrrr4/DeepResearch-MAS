"""M3三个对抗测试用例（对应docs/03_ClaudeCode提示词指南 第4节的要求）：

用例1：mock一个SubAgent返回编造的URL，验证Citation Manager能拦截
用例2：mock两个SubAgent返回互相矛盾的事实，验证Critic能检测出矛盾
用例3：mock Writer尝试引用一个不存在的引用编号，验证系统能拒绝并要求重写

用例1和用例2故意打真实API（不mock网络探测/不mock Critic的LLM调用）——
因为"URL是否真的可达"和"两句话语义上是否矛盾"本质上需要真实的网络/语义
判断能力，纯mock只能验证代码管道对不对，无法验证防护本身是否真的生效。
用例3的核心是代码层的引用校验+重写循环，这部分逻辑是确定性的，mock LLM
即可验证。
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.base import AgentRunResult
from app.agents.critic import Critic
from app.graph.build_graph import writer_node
from app.graph.state import Finding, SourceRef, make_initial_state
from app.tools.citation_manager import CitationManager


@pytest.mark.asyncio
async def test_case_1_citation_manager_catches_fabricated_url():
    """SubAgent"编造"了一个不存在的域名作为来源，Citation Manager的真实
    URL可达性探测应该识别出它不可达，并把原本标记为verified的finding
    代码层强制降级为unverified。"""
    fabricated_url = "https://this-domain-definitely-does-not-exist-9f8e7d6c5b4a.invalid/fake-article"
    real_url = "https://www.anthropic.com"

    cm = CitationManager()
    cm.register_sources(
        {
            fabricated_url: SourceRef(url=fabricated_url, title="编造的来源", fetched_at="now"),
            real_url: SourceRef(url=real_url, title="真实存在的来源", fetched_at="now"),
        }
    )
    await cm.verify_reachability()  # 真实网络探测，不mock

    citation_map = cm.citation_map()
    reachable_by_url = {ref.url: ref.reachable for ref in citation_map.values()}

    fabricated_finding = Finding(
        subtask_id="sub-1",
        claim="一个基于编造来源的断言",
        source_ids=[fabricated_url],
        confidence=0.9,
        status="verified",  # SubAgent"自信地"声称已验证
    )
    real_finding = Finding(
        subtask_id="sub-1",
        claim="一个基于真实来源的断言",
        source_ids=[real_url],
        confidence=0.9,
        status="verified",
    )

    remapped = cm.remap_findings([fabricated_finding, real_finding])

    print(f"\n[用例1结果] 编造URL可达性: {reachable_by_url[fabricated_url]}")
    print(f"[用例1结果] 真实URL可达性: {reachable_by_url[real_url]}")
    print(f"[用例1结果] 编造来源finding降级后状态: {remapped[0].status}")
    print(f"[用例1结果] 真实来源finding状态: {remapped[1].status}")

    assert reachable_by_url[fabricated_url] is False
    assert remapped[0].status == "unverified"  # 被强制降级，拦截成功
    assert remapped[1].status == "verified"  # 真实来源不受影响


@pytest.mark.asyncio
async def test_case_2_critic_detects_cross_subtask_contradiction():
    """两个"来自不同SubAgent"的findings对同一件事给出明显矛盾的数字，
    验证Critic（真实调用DeepSeek-V4-Pro，不mock）能检测出这个矛盾。"""
    findings = [
        Finding(
            subtask_id="sub-langgraph",
            claim="LangGraph于2023年1月首次发布。",
            source_ids=[],
            confidence=0.7,
            status="unverified",
        ),
        Finding(
            subtask_id="sub-langgraph-history",
            claim="LangGraph最早发布于2025年，此前并不存在。",
            source_ids=[],
            confidence=0.7,
            status="unverified",
        ),
    ]

    result = await Critic().run(
        query="LangGraph是什么时候发布的？", findings=findings, subtask_errors={}
    )

    print(f"\n[用例2结果] Critic error: {result.error}")
    print(f"[用例2结果] contradictions: {result.raw.get('contradictions')}")

    assert result.error is None
    contradictions = result.raw.get("contradictions", [])
    assert len(contradictions) >= 1, "Critic应该检测出发布时间的矛盾，但没有检测到"


class _HallucinatingWriter:
    """始终尝试引用一个citation_map里不存在的编号[S99]，模拟Writer幻觉引用。"""

    call_count = 0

    async def run(self, *, query, findings, citation_map, rewrite_feedback=None, input_summary=""):
        _HallucinatingWriter.call_count += 1
        return AgentRunResult(
            output=f"这是一个引用了不存在编号的论点[S99]（第{_HallucinatingWriter.call_count}次尝试）。",
            tokens_used=10,
            tool_calls_used=0,
        )


@pytest.mark.asyncio
async def test_case_3_system_rejects_hallucinated_citation_and_requires_rewrite():
    """Writer坚持引用一个不存在的编号[S99]，验证系统会要求重写（不是
    静默放行），且在耗尽重写次数后代码层兜底清洗，绝不把幻觉引用原样
    输出给用户。"""
    _HallucinatingWriter.call_count = 0
    state = make_initial_state("测试问题", trace_id="adversarial-3")
    state["citation_map"] = {"S1": SourceRef(url="https://real-source.com", title="真实来源", fetched_at="now")}

    with patch("app.graph.build_graph.Writer", _HallucinatingWriter):
        update = await writer_node(state)

    print(f"\n[用例3结果] Writer被调用次数: {_HallucinatingWriter.call_count}")
    print(f"[用例3结果] 最终报告: {update['final_report']}")

    # 系统必须要求过重写（不是第一次就放行），且最终报告里不能残留幻觉引用
    assert _HallucinatingWriter.call_count > 1, "系统应该触发至少一次重写，而不是直接放行幻觉引用"
    assert "[S99]" not in update["final_report"]
    assert "[引用待核实]" in update["final_report"]
