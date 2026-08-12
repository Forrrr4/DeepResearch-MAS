"""app/infra/progress.py 单元测试：确认在没有LangGraph图执行上下文时
（比如单元测试里直接调用agent），emit_progress静默跳过而不是抛异常，
不能因为观测性旁路失败就影响agent本身的执行。"""
from unittest.mock import MagicMock, patch

from app.infra.progress import emit_progress


def test_emit_progress_noop_outside_graph_context():
    # 不在任何graph.astream()执行上下文里调用，get_stream_writer()内部
    # 会失败，这里不应该抛出异常
    emit_progress({"type": "test"})


def test_emit_progress_calls_writer_when_available():
    fake_writer = MagicMock()
    with patch("langgraph.config.get_stream_writer", return_value=fake_writer):
        emit_progress({"type": "tool_call", "query": "x"})

    fake_writer.assert_called_once_with({"type": "tool_call", "query": "x"})


def test_emit_progress_swallows_writer_exception():
    fake_writer = MagicMock(side_effect=RuntimeError("boom"))
    with patch("langgraph.config.get_stream_writer", return_value=fake_writer):
        emit_progress({"type": "test"})  # 不应该抛出异常
