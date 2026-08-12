"""app/tools/report_render.py 单元测试：markdown→HTML渲染 + XSS清洗，
纯函数，不涉及LLM/网络。"""
from app.tools.report_render import render_report_html


def test_renders_headers_bold_and_lists():
    html = render_report_html("# 标题\n\n**加粗**内容\n\n- 项目1\n- 项目2")

    assert "<h1>标题</h1>" in html
    assert "<strong>加粗</strong>" in html
    assert "<li>项目1</li>" in html


def test_renders_tables():
    html = render_report_html("| A | B |\n|---|---|\n| 1 | 2 |")

    assert "<table>" in html
    assert "<th>A</th>" in html
    assert "<td>1</td>" in html


def test_links_get_target_blank_and_noopener():
    html = render_report_html("参考[来源](https://example.com/a)。")

    assert 'href="https://example.com/a"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener' in html


def test_strips_script_tags_but_keeps_text():
    """核心安全测试：即使LLM生成的报告文本里意外/被注入包含<script>标签
    （这个项目本身有明确的Prompt Injection威胁模型），渲染结果里不能
    出现可执行的<script>标签，避免真实XSS。"""
    html = render_report_html("正常内容\n\n<script>alert(document.cookie)</script>")

    assert "<script" not in html
    assert "</script>" not in html


def test_strips_event_handler_attributes():
    html = render_report_html('<img src="x" onerror="alert(1)">')

    assert "onerror" not in html
    assert "<img" not in html  # img不在白名单里，整个标签被剥离


def test_strips_javascript_protocol_links():
    html = render_report_html("[点击](javascript:alert(1))")

    assert "javascript:" not in html


def test_empty_input_returns_empty_string():
    assert render_report_html("") == ""
    assert render_report_html(None) == ""
