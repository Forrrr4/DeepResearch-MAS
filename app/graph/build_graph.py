"""LangGraph图构建：

orchestrator → (single_agent → END)
             | (parallel_research → critic →
                 [打回orchestrator补充调研 | citation_manager → writer → END])
             | (预算耗尽 → citation_manager → writer → END，跳过研究直接收尾)

两条独立的硬上限在这里强制执行，都不依赖任何agent的自觉性：
1. Critic打回Orchestrator的迭代上限（ResearchState.max_iterations，默认2）
   在route_after_critic里做代码层检查。
2. 全局预算（token/工具调用/墙钟时间，见app/infra/budget.py）在
   orchestrator_node每次被进入时最先检查，超限直接标记routing_decision=
   "converge"，route_after_orchestrator据此跳过（再一轮）研究直接收尾。
这是CLAUDE.md第3条硬性约束（预算/重试必须有硬上限）的具体落地。
"""
from __future__ import annotations

import asyncio

from langgraph.graph import END, StateGraph

from app.agents.base import AgentRunResult
from app.agents.baseline_agent import BaselineAgent
from app.agents.critic import Critic
from app.agents.orchestrator import Orchestrator
from app.agents.sub_agent import SubAgent
from app.agents.writer import Writer
from app.config import settings
from app.graph.state import Finding, ResearchState, SourceRef, SubtaskSpec
from app.infra.budget import BudgetLimits, add_usage, is_exhausted
from app.tools.citation_manager import CitationManager, sanitize_invalid_citations, validate_report_citations

DEFAULT_CONCURRENCY = 3


def _global_budget_limits() -> BudgetLimits:
    return BudgetLimits(
        max_tokens=settings.global_max_total_tokens,
        max_tool_calls=settings.global_max_tool_calls,
        max_wall_clock_seconds=settings.global_max_wall_clock_seconds,
    )

# Writer因为幻觉引用被要求重写的次数上限，超过后代码层兜底清洗报告，
# 保证流程一定能收敛，不会因为模型反复犯同一个错误而卡死。
MAX_WRITER_REWRITE_ATTEMPTS = 2


async def run_subtasks_parallel(
    subtasks: list[SubtaskSpec], concurrency: int = DEFAULT_CONCURRENCY
) -> tuple[list[Finding], dict[str, SourceRef], dict[str, str], dict[str, int]]:
    """asyncio.gather + Semaphore并行调度SubAgent。

    单个SubAgent失败不会中断其他SubAgent或抛出异常——BaseAgent.run()内部
    已经把异常/超时都捕获成AgentRunResult.error，这里只需要检查error字段，
    把失败的子任务记入subtask_errors，其余正常子任务的结果照常汇总。
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(subtask: SubtaskSpec) -> tuple[str, AgentRunResult]:
        async with semaphore:
            result = await SubAgent().run(subtask=subtask, input_summary=subtask.goal)
        return subtask.id, result

    pairs = await asyncio.gather(*(run_one(s) for s in subtasks))

    findings: list[Finding] = []
    sources: dict[str, SourceRef] = {}
    subtask_errors: dict[str, str] = {}
    usage = {"tokens": 0, "tool_calls": 0}
    for subtask_id, result in pairs:
        usage["tokens"] += result.tokens_used
        usage["tool_calls"] += result.tool_calls_used
        if result.error:
            subtask_errors[subtask_id] = result.error
            continue
        findings.extend(result.raw.get("findings", []))
        sources.update(result.raw.get("sources", {}))
    return findings, sources, subtask_errors, usage


async def orchestrator_node(state: ResearchState) -> dict:
    # 预算检查放在每次进入orchestrator的最前面（架构文档6.1节："Orchestrator
    # 在每轮迭代前检查预算余量"）——尤其关键的是Critic打回重入这里的场景，
    # 这是预算真正会被反复消耗的地方。检查在任何LLM调用之前，超限时直接
    # 返回"converge"，不再发起新一轮调研，用已有findings收尾。
    if is_exhausted(state["budget"], _global_budget_limits()):
        return {"routing_decision": "converge"}

    critic_feedback = state.get("critic_feedback")
    if critic_feedback and critic_feedback.get("supplementary_subtasks"):
        # Critic循环打回：直接用Critic给出的补充子任务，不重新做一次复杂度
        # 判断——复杂度只在query刚进来时判断一次，迭代阶段的意图已经很
        # 明确（需要补充调研），没有必要再问一遍"要不要拆"，也避免Orchestrator
        # 二次判断和Critic的判断打架。
        return {"routing_decision": "multi_agent", "subtasks": critic_feedback["supplementary_subtasks"]}

    result = await Orchestrator().run(
        query=state["query"], critic_feedback=critic_feedback, input_summary=state["query"]
    )
    new_budget = add_usage(state["budget"], tokens=result.tokens_used, tool_calls=result.tool_calls_used)
    if result.error:
        # Orchestrator本身出错时，降级为single_agent保证流程不中断（对应
        # 架构文档6.5节的失败降级策略），但错误必须显式写回state，不能
        # 静默丢弃——否则eval/trace都看不出这次路由到底是"判断为simple"
        # 还是"Orchestrator挂了兜底"，这两种情况的含义完全不同。
        return {
            "routing_decision": "single_agent",
            "subtasks": [],
            "subtask_errors": {"orchestrator": result.error},
            "budget": new_budget,
        }
    routing = result.raw.get("routing_decision", "single_agent")
    return {"routing_decision": routing, "subtasks": result.raw.get("subtasks", []), "budget": new_budget}


async def single_agent_node(state: ResearchState) -> dict:
    result = await BaselineAgent().run(query=state["query"], input_summary=state["query"])
    new_budget = add_usage(state["budget"], tokens=result.tokens_used, tool_calls=result.tool_calls_used)
    # 回归修复：这条路径直接进END，没有writer_node那样的空报告兜底检查，
    # 之前BaselineAgent超时/出错时result.output是""，会被原样当成
    # final_report静默返回——用户拿到的是一份空报告却不知道发生了什么。
    # 在M5跑18题eval时被真实数据发现（eval-008/eval-013两题都是这样），
    # 不是假设的边界情况。
    report = result.output if not result.error else f"（回答生成失败：{result.error}）"
    return {"final_report": report, "budget": new_budget}


async def parallel_research_node(state: ResearchState) -> dict:
    findings, sources, subtask_errors, usage = await run_subtasks_parallel(state["subtasks"])
    # 合并而不是覆盖：Critic打回后的第二轮调研结果要叠加在第一轮之上，
    # 不能把第一轮的findings/sources冲掉。
    merged_findings = [*state.get("findings", []), *findings]
    merged_sources = {**state.get("sources", {}), **sources}
    merged_errors = {**state.get("subtask_errors", {}), **subtask_errors}
    new_budget = add_usage(state["budget"], tokens=usage["tokens"], tool_calls=usage["tool_calls"])
    return {
        "findings": merged_findings,
        "sources": merged_sources,
        "subtask_errors": merged_errors,
        "budget": new_budget,
    }


async def critic_node(state: ResearchState) -> dict:
    result = await Critic().run(
        query=state["query"],
        findings=state["findings"],
        subtask_errors=state["subtask_errors"],
        input_summary=state["query"],
    )
    new_budget = add_usage(state["budget"], tokens=result.tokens_used, tool_calls=result.tool_calls_used)
    if result.error:
        # Critic失败时保守收敛：不再要求更多调研，直接进入收尾，而不是
        # 让一个出错的Critic把整个流程卡在循环里
        feedback = {"needs_more_research": False, "error": result.error}
    else:
        feedback = result.raw
    return {"critic_feedback": feedback, "iteration": state["iteration"] + 1, "budget": new_budget}


async def citation_manager_node(state: ResearchState) -> dict:
    cm = CitationManager()
    cm.register_sources(state["sources"])
    await cm.verify_reachability()
    remapped_findings = cm.remap_findings(state["findings"])
    return {"findings": remapped_findings, "citation_map": cm.citation_map()}


def assemble_fallback_report(
    query: str, findings: list[Finding], citation_map: dict[str, SourceRef]
) -> str:
    """Writer的LLM综合彻底失败（耗尽重写次数仍拿不到有效正文）时的最后
    一道兜底——纯代码按subtask分组罗列findings，不需要语义生成能力，
    所以不调用LLM。

    这不是假设的边界情况：M5跑18题eval时真实发生过——SubAgent/Critic
    产出了40-50条有citation支撑的findings，但Writer因为要综合的信息量
    太大反复超时/截断，最终只能吐出"（Writer未能生成有效报告）"这句
    空话，把已经真实收集到的信息完全浪费掉了。与其让用户拿到一句道歉，
    不如把已经验证过来源的findings原样列出来，至少比空白有用。
    """
    if not findings:
        return f"# 关于「{query}」的报告\n\n很抱歉，本次调研未能收集到任何可用信息。"

    lines = [
        f"# 关于「{query}」的调研结果（自动降级为结构化列表）\n",
        "> 说明：由于本次信息量较大，报告综合环节未能在预算内完成完整叙述性报告，"
        "以下改为直接列出已验证的调研发现。\n",
    ]
    by_subtask: dict[str, list[Finding]] = {}
    for f in findings:
        by_subtask.setdefault(f.subtask_id, []).append(f)

    status_tag = {"verified": "[已验证]", "unverified": "[待验证]", "contradicted": "[有矛盾]"}
    for subtask_id, group in by_subtask.items():
        lines.append(f"## {subtask_id}\n")
        for f in group:
            tag = status_tag.get(f.status, "")
            src = " ".join(f"[{cid}]" for cid in f.source_ids) if f.source_ids else "（无来源）"
            lines.append(f"- {tag} {f.claim} {src}")
        lines.append("")

    if citation_map:
        lines.append("## 参考来源\n")
        for cid, ref in sorted(citation_map.items(), key=lambda kv: int(kv[0][1:])):
            lines.append(f"- {cid}: {ref.title} ({ref.url})")

    return "\n".join(lines)


async def writer_node(state: ResearchState) -> dict:
    citation_map = state["citation_map"]
    findings = state["findings"]
    rewrite_feedback: str | None = None
    last_report = ""
    budget = state["budget"]

    for _ in range(MAX_WRITER_REWRITE_ATTEMPTS + 1):
        result = await Writer().run(
            query=state["query"],
            findings=findings,
            citation_map=citation_map,
            rewrite_feedback=rewrite_feedback,
            input_summary=state["query"],
        )
        budget = add_usage(budget, tokens=result.tokens_used, tool_calls=result.tool_calls_used)
        if result.error:
            break
        last_report = result.output
        if not last_report.strip():
            # 空报告不能算"校验通过"——validate_report_citations对空字符串
            # 天然返回[]（没有引用标记可扫描≠没有幻觉引用），如果不单独拦截
            # 这种情况，一次因max_tokens截断导致输出为空的失败会被误判成
            # "引用全部合法"直接放行，是真实复现过的bug，不是假设的边界情况。
            rewrite_feedback = "你上一次的回复是空的，没有输出任何报告正文，请重新生成完整报告。"
            continue
        invalid_citations = validate_report_citations(last_report, citation_map)
        if not invalid_citations:
            return {"final_report": last_report, "budget": budget}
        rewrite_feedback = (
            f"你引用了不存在的编号: {invalid_citations}，只能使用这些编号: {sorted(citation_map.keys())}"
        )

    # 超过重写上限仍有幻觉引用（或Writer本身出错/持续输出空报告）：
    # 代码层兜底。如果Writer好歹给了点东西，清洗掉幻觉引用后使用；
    # 如果Writer彻底交白卷（last_report为空），改用纯代码拼装的findings
    # 列表兜底，而不是一句空洞的"未能生成有效报告"——保证已经真实收集
    # 到的、经过来源校验的信息不会被平白浪费掉。
    if last_report.strip():
        sanitized = sanitize_invalid_citations(last_report, citation_map)
    else:
        sanitized = assemble_fallback_report(state["query"], findings, citation_map)
    return {"final_report": sanitized, "budget": budget}


def route_after_orchestrator(state: ResearchState) -> str:
    decision = state["routing_decision"]
    if decision == "single_agent":
        return "single_agent"
    if decision == "converge":
        # 预算耗尽的强制收敛：跳过（再一轮）并行调研，直接用已有findings
        # 走citation_manager→writer收尾，而不是无视预算继续跑
        return "citation_manager"
    return "parallel_research"


def route_after_critic(state: ResearchState) -> str:
    feedback = state.get("critic_feedback") or {}
    needs_more = feedback.get("needs_more_research", False)
    if needs_more and state["iteration"] < state["max_iterations"]:
        return "orchestrator"
    return "citation_manager"


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("single_agent", single_agent_node)
    graph.add_node("parallel_research", parallel_research_node)
    graph.add_node("critic", critic_node)
    graph.add_node("citation_manager", citation_manager_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "single_agent": "single_agent",
            "parallel_research": "parallel_research",
            "citation_manager": "citation_manager",
        },
    )
    graph.add_edge("single_agent", END)
    graph.add_edge("parallel_research", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"orchestrator": "orchestrator", "citation_manager": "citation_manager"},
    )
    graph.add_edge("citation_manager", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
