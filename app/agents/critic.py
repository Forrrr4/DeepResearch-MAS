"""Critic Agent：跨子任务矛盾检测 + 覆盖度判断，不足则生成补充调研任务。

和SubAgent自己标注的status="contradicted"不同——那是SubAgent在**自己
子任务内部**发现的矛盾（比如两个搜索结果打架）。Critic要做的是**跨子任务**
的矛盾检测：SubAgent之间互相看不到彼此的结果（星型拓扑，不直接通信），
只有Critic在Orchestrator汇总之后才第一次同时看到全部findings，所以这类
矛盾只能在这一步被发现。
"""
from __future__ import annotations

from typing import Any

from app.agents.base import AgentRunResult, BaseAgent
from app.graph.state import Finding, SubtaskSpec
from app.infra.model_router import create_message_with_retry, get_client, resolve_model

# Critic没有调用propose_plan/submit_review工具时的重试上限（让模型正确
# 走结构化输出格式），不是研究迭代轮数上限——研究迭代轮数上限是图层面的
# ResearchState.max_iterations，两者是不同粒度的"硬上限"，不要混淆。
MAX_TOOL_CALL_RETRIES = 1

CRITIC_SYSTEM_PROMPT = """你是研究结果的批判/校验Agent(Critic)，负责在多个并行子任务的调研结果
汇总之后，做质量把关。

你的职责：
1. 检测不同findings之间是否存在事实矛盾，尤其关注**来自不同子任务**、
   但涉及同一实体/同一事实点的数据或结论冲突（比如两个子任务分别给出
   了不同的版本号/日期/数字）。
2. 判断当前findings整体上是否足以支撑回答原始问题：如果有子任务失败、
   或某个关键角度明显没有被覆盖到，生成1-3个补充调研的子任务；如果已
   经覆盖充分，不要为了"显得严谨"而强行找补充任务。
3. 必须调用submit_review工具提交结果，不要用文字回答。
"""

SUBMIT_REVIEW_TOOL_SCHEMA = {
    "name": "submit_review",
    "description": "提交矛盾检测结果和覆盖度判断。",
    "input_schema": {
        "type": "object",
        "properties": {
            "cross_subtask_contradictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "related_claims": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["description"],
                },
            },
            "coverage_sufficient": {"type": "boolean"},
            "supplementary_subtasks": {
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
        "required": ["cross_subtask_contradictions", "coverage_sufficient", "supplementary_subtasks"],
    },
}


def _format_findings_for_review(findings: list[Finding], subtask_errors: dict[str, str]) -> str:
    lines = ["以下是各子任务调研得到的findings："]
    for f in findings:
        lines.append(f"- [{f.subtask_id}] ({f.status}, confidence={f.confidence}) {f.claim}")
    if subtask_errors:
        lines.append("\n以下子任务失败，没有产出结果：")
        for sid, err in subtask_errors.items():
            lines.append(f"- {sid}: {err}")
    return "\n".join(lines)


class Critic(BaseAgent):
    name = "critic"
    timeout_seconds = 60.0

    async def _run(
        self, *, query: str, findings: list[Finding], subtask_errors: dict[str, str]
    ) -> AgentRunResult:
        client = get_client()
        model = resolve_model("pro")
        findings_text = _format_findings_for_review(findings, subtask_errors)
        tokens_used = 0

        for _ in range(MAX_TOOL_CALL_RETRIES + 1):
            resp = await create_message_with_retry(
                client,
                model=model,
                max_tokens=2048,
                system=CRITIC_SYSTEM_PROMPT,
                tools=[SUBMIT_REVIEW_TOOL_SCHEMA],
                messages=[{"role": "user", "content": f"原始问题: {query}\n\n{findings_text}"}],
            )
            tokens_used += resp.usage.input_tokens + resp.usage.output_tokens

            block = next(
                (b for b in resp.content if b.type == "tool_use" and b.name == "submit_review"), None
            )
            if block is None:
                continue

            raw_supplementary: list[dict[str, Any]] = block.input.get("supplementary_subtasks", [])
            supplementary = [
                SubtaskSpec(id=s.get("id") or f"supp-{i + 1}", goal=s["goal"])
                for i, s in enumerate(raw_supplementary)
            ]
            coverage_sufficient = bool(block.input.get("coverage_sufficient", True))
            contradictions = block.input.get("cross_subtask_contradictions", [])

            return AgentRunResult(
                output=f"coverage_sufficient={coverage_sufficient}, {len(supplementary)} supplementary tasks, {len(contradictions)} contradictions",
                tokens_used=tokens_used,
                tool_calls_used=0,
                raw={
                    "needs_more_research": (not coverage_sufficient) and bool(supplementary),
                    "supplementary_subtasks": supplementary,
                    "contradictions": contradictions,
                },
            )

        # 模型始终没有正确调用submit_review：保守收敛，不再要求更多调研，
        # 避免因为Critic自己输出格式不稳定而卡住整个流程
        return AgentRunResult(
            output="critic did not call submit_review after retries, defaulting to sufficient",
            tokens_used=tokens_used,
            tool_calls_used=0,
            raw={"needs_more_research": False, "supplementary_subtasks": [], "contradictions": []},
        )
