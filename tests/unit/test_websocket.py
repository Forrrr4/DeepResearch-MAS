"""app/api/websocket.py 单元测试：纯连接管理逻辑，用mock WebSocket，
不启动真实服务器。"""
from unittest.mock import AsyncMock

import pytest

from app.api.websocket import ConnectionManager


def _fake_ws() -> AsyncMock:
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_registers_websocket_for_task():
    manager = ConnectionManager()
    ws = _fake_ws()

    await manager.connect("task-1", ws)

    ws.accept.assert_called_once()
    assert ws in manager._connections["task-1"]


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_connections_of_same_task():
    manager = ConnectionManager()
    ws1, ws2 = _fake_ws(), _fake_ws()
    await manager.connect("task-1", ws1)
    await manager.connect("task-1", ws2)

    await manager.broadcast("task-1", {"type": "start"})

    ws1.send_json.assert_awaited_once_with({"type": "start"})
    ws2.send_json.assert_awaited_once_with({"type": "start"})


@pytest.mark.asyncio
async def test_broadcast_does_not_leak_across_different_tasks():
    manager = ConnectionManager()
    ws1, ws2 = _fake_ws(), _fake_ws()
    await manager.connect("task-1", ws1)
    await manager.connect("task-2", ws2)

    await manager.broadcast("task-1", {"type": "start"})

    ws1.send_json.assert_awaited_once()
    ws2.send_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_removes_connection():
    manager = ConnectionManager()
    ws = _fake_ws()
    await manager.connect("task-1", ws)

    manager.disconnect("task-1", ws)

    assert "task-1" not in manager._connections


@pytest.mark.asyncio
async def test_broadcast_survives_one_failed_connection():
    """单个连接发送失败（比如客户端已断开）不应该阻止其他连接收到广播。"""
    manager = ConnectionManager()
    ws_broken, ws_ok = _fake_ws(), _fake_ws()
    ws_broken.send_json.side_effect = RuntimeError("connection closed")
    await manager.connect("task-1", ws_broken)
    await manager.connect("task-1", ws_ok)

    await manager.broadcast("task-1", {"type": "start"})

    ws_ok.send_json.assert_awaited_once()
    assert ws_broken not in manager._connections.get("task-1", [])
