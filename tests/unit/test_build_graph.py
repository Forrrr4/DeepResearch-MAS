"""app/graph/build_graph.py 单元测试：并发限制、单子任务失败不阻塞整体、
路由逻辑（含Critic循环打回的迭代上限）、Writer幻觉引用重写。全部mock
SubAgent/Orchestrator/Critic/Writer，不打真实API。
"""
import asyncio
from unittest.mock import patch

import pytest

from app.agents.base import AgentRunResult
from app.graph.build_graph import (
    assemble_fallback_report,
    orchestrator_node,
    route_after_critic,
    route_after_orchestrator,
    run_subtasks_parallel,
    single_agent_node,
    writer_node,
)
from app.graph.state import Finding, SourceRef, SubtaskSpec, make_initial_state


class _ConcurrencyTrackingAgent:
    current = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def run(self, *, subtask, input_summary=""):
        cls = _ConcurrencyTrackingAgent
        async with cls.lock:
            cls.current += 1
            cls.max_seen = max(cls.max_seen, cls.current)
        await asyncio.sleep(0.05)
        async with cls.lock:
            cls.current -= 1
        return AgentRunResult(
            output="ok",
            tokens_used=1,
            tool_calls_used=1,
            raw={"findings": [], "sources": {}},
        )


@pytest.mark.asyncio
async def test_run_subtasks_parallel_respects_concurrency_limit():
    _ConcurrencyTrackingAgent.current = 0
    _ConcurrencyTrackingAgent.max_seen = 0
    subtasks = [SubtaskSpec(id=f"sub-{i}", goal=f"goal {i}") for i in range(6)]

    with patch("app.graph.build_graph.SubAgent", _ConcurrencyTrackingAgent):
        await run_subtasks_parallel(subtasks, concurrency=2)

    assert _ConcurrencyTrackingAgent.max_seen <= 2


class _MixedResultAgent:
    async def run(self, *, subtask, input_summary=""):
        if subtask.id == "sub-fail":
            return AgentRunResult(output="", tokens_used=1, tool_calls_used=0, error="boom")
        finding = Finding(
            subtask_id=subtask.id, claim=f"claim for {subtask.id}", source_ids=[], confidence=0.5, status="unverified"
        )
        return AgentRunResult(
            output="ok", tokens_used=1, tool_calls_used=1, raw={"findings": [finding], "sources": {}}
        )


@pytest.mark.asyncio
async def test_run_subtasks_parallel_isolates_single_subtask_failure():
    subtasks = [
        SubtaskSpec(id="sub-ok-1", goal="g1"),
        SubtaskSpec(id="sub-fail", goal="g2"),
        SubtaskSpec(id="sub-ok-2", goal="g3"),
    ]

    with patch("app.graph.build_graph.SubAgent", _MixedResultAgent):
        findings, sources, subtask_errors, usage = await run_subtasks_parallel(subtasks, concurrency=3)

    assert subtask_errors == {"sub-fail": "boom"}
    assert {f.subtask_id for f in findings} == {"sub-ok-1", "sub-ok-2"}
    assert usage == {"tokens": 3, "tool_calls": 2}  # 3个子任务各1 token；只有2个成功的各1次tool_call


def test_route_after_orchestrator_single_agent():
    assert route_after_orchestrator({"routing_decision": "single_agent"}) == "single_agent"


def test_route_after_orchestrator_multi_agent():
    assert route_after_orchestrator({"routing_decision": "multi_agent"}) == "parallel_research"


class _TimingOutBaselineAgent:
    async def run(self, *, query, input_summary=""):
        return AgentRunResult(output="", tokens_used=50, tool_calls_used=0, error="timeout after 90.0s")


@pytest.mark.asyncio
async def test_single_agent_node_surfaces_error_instead_of_returning_empty_report():
    """回归测试：M5跑18题eval时真实复现过——BaselineAgent超时后
    single_agent_node会把空字符串原样当final_report返回，用户拿到一份
    空报告却看不出发生了什么，违反CLAUDE.md"禁止静默失败"。"""
    state = make_initial_state("任意问题", trace_id="t1")

    with patch("app.graph.build_graph.BaselineAgent", _TimingOutBaselineAgent):
        update = await single_agent_node(state)

    assert update["final_report"].strip() != ""
    assert "timeout" in update["final_report"]


def test_route_after_orchestrator_converge_skips_research():
    assert route_after_orchestrator({"routing_decision": "converge"}) == "citation_manager"


class _OrchestratorThatShouldNeverBeCalled:
    async def run(self, *, query, critic_feedback=None, input_summary=""):
        raise AssertionError("Orchestrator LLM不应该在预算已耗尽时被调用")


@pytest.mark.asyncio
async def test_orchestrator_node_converges_without_calling_llm_when_budget_exhausted():
    """回归测试：预算检查必须在任何LLM调用之前完成，否则"预算耗尽"这个
    判断本身还要再花一次预算才能生效，自相矛盾。"""
    from app.infra.budget import new_budget

    state = make_initial_state("任意问题", trace_id="t1")
    state["budget"] = {**new_budget(), "tokens_used": 10**9}  # 远超任何合理上限

    with patch("app.graph.build_graph.Orchestrator", _OrchestratorThatShouldNeverBeCalled):
        update = await orchestrator_node(state)

    assert update["routing_decision"] == "converge"


class _FailingOrchestrator:
    async def run(self, *, query, critic_feedback=None, input_summary=""):
        return AgentRunResult(output="", tokens_used=0, tool_calls_used=0, error="BadRequestError: boom")


@pytest.mark.asyncio
async def test_orchestrator_node_surfaces_error_instead_of_silently_defaulting():
    """回归测试：修复前orchestrator_node在Orchestrator出错时会静默把
    routing_decision默认成single_agent，错误信息被丢弃，违反了CLAUDE.md
    "禁止静默失败"的约束。现在必须把错误写进subtask_errors。"""
    state = make_initial_state("任意问题", trace_id="t1")
    with patch("app.graph.build_graph.Orchestrator", _FailingOrchestrator):
        update = await orchestrator_node(state)

    assert update["routing_decision"] == "single_agent"
    assert "orchestrator" in update["subtask_errors"]
    assert "boom" in update["subtask_errors"]["orchestrator"]


def test_route_after_critic_loops_back_when_under_iteration_cap():
    state = {"critic_feedback": {"needs_more_research": True}, "iteration": 1, "max_iterations": 2}
    assert route_after_critic(state) == "orchestrator"


def test_route_after_critic_stops_at_iteration_cap_even_if_critic_wants_more():
    """回归测试：即使Critic一直说needs_more_research=True，达到
    max_iterations上限后也必须强制收敛，这是CLAUDE.md第3条硬性约束
    （迭代上限硬编码为2）在代码层面的强制执行，不依赖Critic自律。"""
    state = {"critic_feedback": {"needs_more_research": True}, "iteration": 2, "max_iterations": 2}
    assert route_after_critic(state) == "citation_manager"


def test_route_after_critic_proceeds_when_coverage_sufficient():
    state = {"critic_feedback": {"needs_more_research": False}, "iteration": 1, "max_iterations": 2}
    assert route_after_critic(state) == "citation_manager"


class _HallucinatingThenCleanWriter:
    """第一次引用一个不存在的编号[S99]，第二次（收到重写反馈后）给出合法引用。"""

    call_count = 0

    async def run(self, *, query, findings, citation_map, rewrite_feedback=None, input_summary=""):
        _HallucinatingThenCleanWriter.call_count += 1
        if rewrite_feedback is None:
            return AgentRunResult(output="这是一个论点[S99]。", tokens_used=10, tool_calls_used=0)
        return AgentRunResult(output="修正后的论点[S1]。", tokens_used=10, tool_calls_used=0)


@pytest.mark.asyncio
async def test_writer_node_retries_on_hallucinated_citation_then_succeeds():
    _HallucinatingThenCleanWriter.call_count = 0
    state = make_initial_state("测试问题", trace_id="t1")
    state["citation_map"] = {"S1": SourceRef(url="https://a.com", title="A", fetched_at="now")}

    with patch("app.graph.build_graph.Writer", _HallucinatingThenCleanWriter):
        update = await writer_node(state)

    assert "[S1]" in update["final_report"]
    assert "[S99]" not in update["final_report"]
    assert _HallucinatingThenCleanWriter.call_count == 2


class _EmptyThenValidWriter:
    """第一次返回空字符串（模拟max_tokens截断导致正文为空），第二次
    （收到重写反馈后）返回正常报告。"""

    call_count = 0

    async def run(self, *, query, findings, citation_map, rewrite_feedback=None, input_summary=""):
        _EmptyThenValidWriter.call_count += 1
        if rewrite_feedback is None:
            return AgentRunResult(output="", tokens_used=10, tool_calls_used=0)
        return AgentRunResult(output="正常生成的报告[S1]。", tokens_used=10, tool_calls_used=0)


@pytest.mark.asyncio
async def test_writer_node_retries_on_empty_report_instead_of_treating_it_as_valid():
    """回归测试：修复前，空字符串报告会被validate_report_citations判定
    为"没有非法引用"从而当作合法结果直接返回，实际复现过一次真实的
    max_tokens截断导致空报告被误放行的案例。现在必须触发重写。"""
    _EmptyThenValidWriter.call_count = 0
    state = make_initial_state("测试问题", trace_id="t1")
    state["citation_map"] = {"S1": SourceRef(url="https://a.com", title="A", fetched_at="now")}

    with patch("app.graph.build_graph.Writer", _EmptyThenValidWriter):
        update = await writer_node(state)

    assert update["final_report"] == "正常生成的报告[S1]。"
    assert _EmptyThenValidWriter.call_count == 2


def test_assemble_fallback_report_lists_findings_with_citation_ids():
    findings = [
        Finding(subtask_id="sub-1", claim="LangGraph用StateGraph管理状态", source_ids=["S1"], confidence=0.8, status="verified"),
        Finding(subtask_id="sub-1", claim="没有来源的推断", source_ids=[], confidence=0.3, status="unverified"),
    ]
    citation_map = {"S1": SourceRef(url="https://a.com", title="A文档", fetched_at="now")}

    report = assemble_fallback_report("LangGraph状态管理", findings, citation_map)

    assert "LangGraph用StateGraph管理状态" in report
    assert "[S1]" in report
    assert "A文档" in report
    assert "没有来源的推断" in report


def test_assemble_fallback_report_handles_no_findings():
    report = assemble_fallback_report("任意问题", [], {})
    assert "未能收集到任何可用信息" in report


class _AlwaysEmptyWriter:
    call_count = 0

    async def run(self, *, query, findings, citation_map, rewrite_feedback=None, input_summary=""):
        _AlwaysEmptyWriter.call_count += 1
        return AgentRunResult(output="", tokens_used=10, tool_calls_used=0)


@pytest.mark.asyncio
async def test_writer_node_falls_back_to_findings_list_when_writer_never_produces_output():
    """回归测试：M5跑18题eval时真实发生过——Writer在findings量大时反复
    只吐出思考内容、正文截断成空，重写3次全部失败。之前的兜底只是一句
    "（Writer未能生成有效报告）"，把已经真实收集、经过来源校验的信息
    全部浪费掉；现在应该改用纯代码拼装的findings列表。"""
    _AlwaysEmptyWriter.call_count = 0
    state = make_initial_state("测试问题", trace_id="t1")
    state["citation_map"] = {"S1": SourceRef(url="https://a.com", title="A", fetched_at="now")}
    state["findings"] = [
        Finding(subtask_id="sub-1", claim="一个已验证的关键结论", source_ids=["S1"], confidence=0.9, status="verified")
    ]

    with patch("app.graph.build_graph.Writer", _AlwaysEmptyWriter):
        update = await writer_node(state)

    assert "一个已验证的关键结论" in update["final_report"]
    assert "未能生成有效报告" not in update["final_report"]


class _AlwaysHallucinatingWriter:
    async def run(self, *, query, findings, citation_map, rewrite_feedback=None, input_summary=""):
        return AgentRunResult(output="顽固的幻觉引用[S99]。", tokens_used=10, tool_calls_used=0)


@pytest.mark.asyncio
async def test_writer_node_sanitizes_report_after_exhausting_rewrite_attempts():
    """回归测试：Writer如果屡教不改，代码层必须兜底清洗，绝不能把带
    幻觉引用的报告原样输出（CLAUDE.md第2条硬性约束）。"""
    state = make_initial_state("测试问题", trace_id="t1")
    state["citation_map"] = {"S1": SourceRef(url="https://a.com", title="A", fetched_at="now")}

    with patch("app.graph.build_graph.Writer", _AlwaysHallucinatingWriter):
        update = await writer_node(state)

    assert "[S99]" not in update["final_report"]
    assert "[引用待核实]" in update["final_report"]
