"""ResearchState：全局唯一的显式状态对象。

CLAUDE.md硬性约束第1条：agent间通信只能走这里的字段，不允许靠"对话历史"
隐式传递信息。任何agent需要的输入必须能在这里查到来源；任何agent产生
的、其他节点需要用到的信息，必须写回这里。

字段设计参考架构文档(docs/01_架构设计方案.md)第4.1节，本文件相对原文
做了以下调整，均为M2实现时发现的真实缺口，不是随意改动：

1. 新增 `routing_decision`：原文没有显式字段记录"这次走了单agent还是
   多agent路径"，但Orchestrator第一职责就是做这个判断（架构文档3.1节），
   没有地方存这个判断结果的话，图的条件边和后续eval都无法验证Orchestrator
   分流是否正确。
2. 新增 `subtask_errors`：原文的失败处理策略（架构文档6.5节"子任务失败：
   Orchestrator收到部分结果时仍可继续"）需要知道"哪个子任务失败了、为什么"，
   否则下游没法在报告里标注"该角度信息不足"。放进findings里混着正常结果不
   合适，所以单独开一个字段。
3. `subtasks` 用 `list[SubtaskSpec]`（pydantic模型）而不是原文的
   `list[dict]`——CLAUDE.md代码风格约定要求"不允许裸dict/list作为跨模块
   传参类型"，原文的dict写法在这一点上和代码风格约定冲突，这里按代码风格
   约定优先处理。
4. `SourceRef.reliability_score` 加了默认值0.0并注明：M2阶段SubAgent产出
   source时Critic还没跑（Critic是M3的交付物），打分逻辑不存在，先给一个
   显式默认值而不是留空/None，避免下游误以为"0.0"是Critic给出的低分判断。
5. M3新增 `SourceRef.reachable`（Citation Manager的URL可达性校验结果）
   和 `ResearchState.citation_map`（"引用编号→来源"映射，Writer据此生成
   带编号引用的报告，也是Citation Manager校验Writer幻觉引用的依据）。
"""
from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from app.infra.budget import new_budget


class ToolBudget(BaseModel):
    max_search_calls: int = 6
    max_tokens: int = 8000


class SubtaskSpec(BaseModel):
    id: str
    goal: str
    expected_output_schema: dict[str, str] = Field(
        default_factory=lambda: {"findings": "list", "sources": "list", "confidence": "0-1"}
    )
    tool_budget: ToolBudget = Field(default_factory=ToolBudget)


class SourceRef(BaseModel):
    url: str
    title: str
    fetched_at: str
    reliability_score: float = 0.0  # Critic Agent 打分，M3之前恒为默认值
    reachable: bool | None = None  # M3新增：Citation Manager的URL可达性校验结果，None=未检查


class Finding(BaseModel):
    subtask_id: str
    claim: str
    source_ids: list[str]
    confidence: float
    status: Literal["verified", "unverified", "contradicted"]


class ResearchState(TypedDict):
    query: str
    routing_decision: Literal["single_agent", "multi_agent"]
    subtasks: list[SubtaskSpec]
    findings: list[Finding]
    sources: dict[str, SourceRef]
    citation_map: dict[str, SourceRef]  # M3新增：Citation Manager产出的"引用编号(S1..)→来源"映射，供Writer使用
    subtask_errors: dict[str, str]
    iteration: int
    max_iterations: int
    budget: dict[str, float]  # {tokens_used, tool_calls_used, wall_clock_start}，见app/infra/budget.py
    critic_feedback: dict | None
    final_report: str | None
    trace_id: str


def make_initial_state(query: str, *, trace_id: str, max_iterations: int = 2) -> ResearchState:
    return ResearchState(
        query=query,
        routing_decision="single_agent",
        subtasks=[],
        findings=[],
        sources={},
        citation_map={},
        subtask_errors={},
        iteration=0,
        max_iterations=max_iterations,
        budget=new_budget(),
        critic_feedback=None,
        final_report=None,
        trace_id=trace_id,
    )
