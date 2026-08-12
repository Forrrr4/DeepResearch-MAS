"""把Writer生成的Markdown报告渲染成安全的HTML，供前端直接展示（而不是
把原始markdown源码当纯文本显示，用户反馈"看起来不像正常文档"）。

纯代码逻辑，不调用LLM（CLAUDE.md第4条硬性约束）。

安全说明：final_report来自LLM生成的内容，而这个项目本身就有明确的
Prompt Injection威胁模型（见app/infra/guardrail_rules.py）——理论上
不能排除网页内容里的注入文本最终经由findings渗透进最终报告文本。
`markdown`库默认会原样保留源文本里的裸HTML（这是Markdown规范本身的
行为，不是bug），如果不做处理直接把转换结果丢进浏览器的innerHTML，
就是一个真实的XSS风险点，不是假设的边界情况。所以这里强制过一遍
bleach白名单清洗，只保留渲染报告需要的标签/属性，其余一律剥离。
"""
from __future__ import annotations

import bleach
import markdown

_ALLOWED_TAGS = [
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "code", "pre",
    "ul", "ol", "li",
    "blockquote",
    "table", "thead", "tbody", "tr", "th", "td",
    "a",
]
_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
}
_ALLOWED_PROTOCOLS = ["http", "https"]


def render_report_html(markdown_text: str) -> str:
    """markdown文本 → 清洗后的安全HTML片段。"""
    if not markdown_text:
        return ""

    raw_html = markdown.markdown(
        markdown_text, extensions=["tables", "fenced_code", "sane_lists"]
    )
    cleaned = bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )
    # 外链统一加target=_blank + rel=noopener，避免新开的标签页能通过
    # window.opener反向操纵原页面（tabnabbing），bleach不会自动加这个。
    cleaned = bleach.linkify(
        cleaned, callbacks=[_add_target_blank], skip_tags=["pre", "code"]
    )
    return cleaned


def _add_target_blank(attrs: dict, new: bool = False) -> dict:
    attrs[(None, "target")] = "_blank"
    attrs[(None, "rel")] = "noopener noreferrer"
    return attrs
