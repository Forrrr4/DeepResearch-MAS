"""实时进度推送：把图执行过程中每个节点的完成事件广播给对应task_id
的所有WebSocket连接。这是纯粹的连接管理逻辑（不涉及LLM/图编排），
和app/api/main.py里"怎么跑图、怎么产出事件"的业务逻辑分开。
"""
from __future__ import annotations

from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(task_id, []).append(websocket)

    def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        conns = self._connections.get(task_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if task_id in self._connections and not self._connections[task_id]:
            del self._connections[task_id]

    async def broadcast(self, task_id: str, message: dict[str, Any]) -> None:
        """向某个task_id的所有连接推送消息。单个连接发送失败（比如客户端
        已经断开但服务端还没感知到）不应该影响其他连接收到推送，所以
        逐个捕获异常而不是让一次失败中断整个广播。"""
        for ws in list(self._connections.get(task_id, [])):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - 单个连接的发送失败不应中断广播
                self.disconnect(task_id, ws)


manager = ConnectionManager()
