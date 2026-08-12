"""Orchestrator：判断query复杂度 + 生成正交子任务计划。

按docs/03_ClaudeCode提示词指南 第3节的要求，"子任务是否正交"这个判断
逻辑写成独立的纯函数（check_subtasks_orthogonal），不混在prompt里让LLM
隐式判断——这样它可测试、可解释，不是黑盒。LLM只负责有创造性的部分
（怎么拆任务），拆完之后"拆得好不好"用代码校验。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agents.base import AgentRunResult, BaseAgent
from app.graph.state import SubtaskSpec
from app.infra.model_router import create_message_with_retry, get_client, resolve_model

# 重新规划的硬上限：Orchestrator生成的计划校验不通过时最多重试这么多次，
# 超过就强制降级为单agent，保证流程一定能往下走（CLAUDE.md第3条硬性约束）。
MAX_REPLAN_ATTEMPTS = 2

ORCHESTRATOR_SYSTEM_PROMPT = """你是一个研究任务的主管(Orchestrator)，负责判断用户问题的复杂度并做任务分解。

判断标准（关键是"能否拆成2个以上互相独立、可以并行调研的对象"，而不是
"要不要综合多个信息点"——几乎所有问题都需要综合多个信息点，但只有真正
可以拆开独立跑的才值得付出并行协调的开销）：

1. 如果问题是常识性/定义性的，或者虽然需要综合多个事实点、但本质是解释
   一个概念/分析一个机制/说明两者的联系与区别（即使涉及2个概念的对比），
   适合一个agent顺序检索+推理完成，判定complexity为"simple"，subtasks给
   空数组。例如"A和B的区别是什么"如果A、B是需要放在一起理解的相关概念
   （而不是需要分别深入调研的独立主体），仍属于simple。
2. 只有当问题明确要求你**横向比较3个或以上具体命名的独立对象**（如3个及
   以上框架/厂商/产品的多维度对比），或者需要覆盖的独立调研方向本身就
   有明显广度（例如"列举/覆盖至少3家厂商各自的技术路线"），判定为
   "complex"，生成2-6个子任务，每个子任务对应一个独立对象。
3. 拿不准时优先判定为simple——错误地拆分会引入不必要的协调开销和额外
   成本，而如果simple场景下单agent确实答不好，这个代价由后续的Critic
   反馈迭代来弥补，不需要Orchestrator这一步就过度保守地拆分。
4. 每个子任务对应一个独立的调研对象或维度，不要合并成一个大子任务，也不要
   有明显重叠（比如不能拆成"调研A的优点"和"调研A的好处"这种同义改写）。
5. 每个子任务的goal要具体、可独立执行，不要笼统描述。

必须调用 propose_plan 工具提交你的判断，不要用文字回答。
"""

# 不强制tool_choice：DeepSeek端点默认开启thinking模式，thinking模式与
# 强制tool_choice不兼容（实测报 "Thinking mode does not support this
# tool_choice" 400错误）。改为把propose_plan作为可选工具放出去，配合
# 下面_run里"模型没调用工具就要求重试"的逻辑，效果等价但兼容thinking模式。

PROPOSE_PLAN_TOOL_SCHEMA = {
    "name": "propose_plan",
    "description": "提交复杂度判断和任务分解计划。",
    "input_schema": {
        "type": "object",
        "properties": {
            "complexity": {"type": "string", "enum": ["simple", "complex"]},
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "goal": {"type": "string"},
                    },
                    "required": ["id", "goal"],
                },
            },
        },
        "required": ["complexity", "subtasks"],
    },
}


@dataclass
class OrthogonalityCheckResult:
    is_orthogonal: bool
    overlapping_pairs: list[tuple[int, int, float]] = field(default_factory=list)


def _char_shingles(text: str, n: int = 2) -> set[str]:
    """把文本转成字符n-gram集合。用字符级而非词级，因为中英文混排场景下
    分词依赖额外的分词器，字符shingle是一个无需额外依赖、语言无关的
    轻量替代方案，足够用来发现"明显同义改写"级别的重叠。"""
    compact = text.replace(" ", "").replace("\n", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def check_subtasks_orthogonal(
    goals: list[str], overlap_threshold: float = 0.5
) -> OrthogonalityCheckResult:
    """用Jaccard相似度粗粒度检测子任务goal之间是否重叠过高。

    这不是严格的语义去重（那需要embedding，超出了M2的范围），目的是
    拦截Orchestrator偷懒生成的重复/换皮子任务这种明显情况。
    """
    shingle_sets = [_char_shingles(g) for g in goals]
    overlapping_pairs: list[tuple[int, int, float]] = []
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            a, b = shingle_sets[i], shingle_sets[j]
            if not a or not b:
                continue
            similarity = len(a & b) / len(a | b)
            if similarity >= overlap_threshold:
                overlapping_pairs.append((i, j, round(similarity, 3)))
    return OrthogonalityCheckResult(is_orthogonal=not overlapping_pairs, overlapping_pairs=overlapping_pairs)


def validate_subtask_count(n: int, min_n: int = 2, max_n: int = 6) -> bool:
    return min_n <= n <= max_n


class Orchestrator(BaseAgent):
    name = "orchestrator"
    timeout_seconds = 60.0

    async def _run(self, *, query: str, critic_feedback: dict | None = None) -> AgentRunResult:
        client = get_client()
        model = resolve_model("pro")
        tokens_used = 0
        feedback_note = ""
        if critic_feedback:
            feedback_note = (
                f"\n\n上一轮Critic反馈，请据此调整计划：{json.dumps(critic_feedback, ensure_ascii=False)}"
            )

        last_raw_subtasks: list[dict[str, Any]] = []

        for attempt in range(MAX_REPLAN_ATTEMPTS + 1):
            resp = await create_message_with_retry(
                client,
                model=model,
                # 2048而不是1500：同样为了给thinking内容留出余量，避免propose_plan
                # 的JSON在子任务较多时被截断（参考sub_agent.py的max_tokens修复）。
                max_tokens=2048,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=[PROPOSE_PLAN_TOOL_SCHEMA],
                messages=[{"role": "user", "content": query + feedback_note}],
            )
            tokens_used += resp.usage.input_tokens + resp.usage.output_tokens

            plan_block = next((b for b in resp.content if b.type == "tool_use"), None)
            if plan_block is None:
                feedback_note = "\n\n你上次没有调用propose_plan工具，请务必通过该工具提交结果。"
                continue

            complexity = plan_block.input.get("complexity", "complex")
            raw_subtasks = plan_block.input.get("subtasks", [])
            last_raw_subtasks = raw_subtasks

            if complexity == "simple":
                return AgentRunResult(
                    output="routing=single_agent",
                    tokens_used=tokens_used,
                    tool_calls_used=0,
                    raw={"routing_decision": "single_agent", "subtasks": []},
                )

            goals = [s.get("goal", "") for s in raw_subtasks]
            count_ok = validate_subtask_count(len(raw_subtasks))
            ortho_result = (
                check_subtasks_orthogonal(goals) if goals else OrthogonalityCheckResult(True)
            )

            if count_ok and ortho_result.is_orthogonal:
                subtasks = [
                    SubtaskSpec(id=s.get("id") or f"sub-{i + 1}", goal=s["goal"])
                    for i, s in enumerate(raw_subtasks)
                ]
                return AgentRunResult(
                    output=f"routing=multi_agent, {len(subtasks)} subtasks",
                    tokens_used=tokens_used,
                    tool_calls_used=0,
                    raw={"routing_decision": "multi_agent", "subtasks": subtasks},
                )

            problems = []
            if not count_ok:
                problems.append(f"子任务数量应在2-6之间，你给了{len(raw_subtasks)}个")
            if not ortho_result.is_orthogonal:
                pairs_desc = "; ".join(
                    f"({goals[i]!r} vs {goals[j]!r}, 相似度{sim})"
                    for i, j, sim in ortho_result.overlapping_pairs
                )
                problems.append(f"以下子任务重叠度过高，需要重新拆分使其正交: {pairs_desc}")
            feedback_note = "\n\n上次生成的计划未通过校验，请修正后重新生成：" + "；".join(problems)

        # 超过重试上限：强制降级为单agent，保证流程不会卡死
        return AgentRunResult(
            output="routing=single_agent (fallback after max replan attempts)",
            tokens_used=tokens_used,
            tool_calls_used=0,
            raw={
                "routing_decision": "single_agent",
                "subtasks": [],
                "fallback_reason": f"validation failed after {MAX_REPLAN_ATTEMPTS} replans, last_subtasks={last_raw_subtasks}",
            },
        )
