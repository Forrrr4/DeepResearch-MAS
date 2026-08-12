"""Writer Agent：综合findings撰写最终报告。

引用编号必须来自Citation Manager给出的citation_map，不允许引入findings
之外的新事实——这条约束在prompt里声明，但真正的强制执行在
app/graph/build_graph.py的writer_node里，用
app/tools/citation_manager.validate_report_citations做代码层校验，
查不到的引用编号触发重写而不是静默放行（CLAUDE.md第2条硬性约束）。
"""
from __future__ import annotations

from app.agents.base import AgentRunResult, BaseAgent
from app.graph.state import Finding, SourceRef
from app.infra.model_router import create_message_with_retry, get_client, resolve_model

WRITER_SYSTEM_PROMPT = """你是研究报告的撰写Agent(Writer)。

规则：
1. 只能使用下面提供的findings里已有的信息，不能引入新的事实、数字或来源。
2. 每个关键论点后面用形如[S3]的引用编号标注来源，编号必须是"可用引用编号"
   列表里存在的编号，绝对不允许编造引用编号，也不允许引用列表之外的编号。
3. status为'contradicted'的finding要如实呈现矛盾双方，不要挑一个说、也
   不要编造一个折中结论。
4. status为'unverified'的finding要在措辞上体现不确定性（如"有待验证的
   信息显示..."），不要用确定语气陈述。
5. 直接输出Markdown格式的最终报告正文，不要输出解释你在做什么的元话语。
"""


def _format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "（没有可用的findings）"
    lines = []
    for f in findings:
        lines.append(f"- ({f.status}, confidence={f.confidence}, 引用: {f.source_ids}) {f.claim}")
    return "\n".join(lines)


def _format_citation_map(citation_map: dict[str, SourceRef]) -> str:
    if not citation_map:
        return "（没有可用引用）"
    lines = [f"- {cid}: {ref.title} ({ref.url})" for cid, ref in citation_map.items()]
    return "\n".join(lines)


class Writer(BaseAgent):
    name = "writer"
    # 90s而不是60s：findings数量多时（尤其Critic触发过补充调研后）Writer
    # 要综合几十条内容生成长报告，加上thinking开销，实测60s会被自己的
    # 超时打断（复现过一次），和SubAgent的90s上限对齐（架构文档6.4节
    # 给出的参考值）。
    timeout_seconds = 90.0

    async def _run(
        self,
        *,
        query: str,
        findings: list[Finding],
        citation_map: dict[str, SourceRef],
        rewrite_feedback: str | None = None,
    ) -> AgentRunResult:
        client = get_client()
        model = resolve_model("pro")

        user_content = (
            f"原始问题: {query}\n\n"
            f"可用findings:\n{_format_findings(findings)}\n\n"
            f"可用引用编号:\n{_format_citation_map(citation_map)}"
        )
        if rewrite_feedback:
            user_content += f"\n\n上一次生成的报告有问题，请修正：{rewrite_feedback}"

        resp = await create_message_with_retry(
            client,
            model=model,
            # 8192而不是4096：Writer要综合的findings可能有几十条（尤其Critic
            # 触发过补充调研之后），加上thinking内容本身占用输出token，实测
            # 出现过thinking耗尽预算、正文被截断成空字符串的情况（详见
            # writer_node里的空报告兜底检查，这里增大预算是从根源上缓解）。
            max_tokens=8192,
            system=WRITER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        tokens_used = resp.usage.input_tokens + resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        return AgentRunResult(output=text, tokens_used=tokens_used, tool_calls_used=0)
