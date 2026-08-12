"""细粒度执行进度广播：用LangGraph的custom stream机制，让SubAgent/
BaselineAgent内部的工具调用也能实时推送出去，而不是只能等整个节点跑完
才看到一条粗粒度汇总（架构文档6.7节"前端展示execution graph的实时
状态"的具体落地——之前的实现只到"节点"粒度，用户反馈"看不到壳里在
干什么"，这个模块把粒度下钻到"工具调用"级别）。

这是可选的可观测性增强，不是agent的核心逻辑，所以刻意做成"拿不到
stream writer就静默跳过"，不能因为这层观测失败就影响agent本身的执行
（比如单元测试里直接调用agent、不经过graph.astream时，这里应该什么都
不做，而不是抛异常把测试搞挂）。
"""
from __future__ import annotations

from typing import Any


def emit_progress(event: dict[str, Any]) -> None:
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - 观测性旁路，不能影响主流程
        return
    if writer is None:
        return
    try:
        writer(event)
    except Exception:  # noqa: BLE001 - 同上
        pass
