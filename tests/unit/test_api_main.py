"""app/api/main.py 单元测试：mock build_graph()，不打真实API。"""
import asyncio
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import TASKS, app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_get_research_404_for_unknown_task():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/research/nonexistent-task-id")

    assert resp.status_code == 404


class _FakeGraph:
    async def astream(self, state, stream_mode="updates"):
        yield ("custom", {"type": "tool_call", "agent": "m1_baseline_agent", "tool": "web_search", "query": "x"})
        yield ("updates", {"orchestrator": {"routing_decision": "single_agent", "subtasks": []}})
        yield ("updates", {"single_agent": {"final_report": "mock report"}})


@pytest.mark.asyncio
async def test_start_research_streams_events_and_completes():
    with patch("app.api.main.build_graph", return_value=_FakeGraph()):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/research", json={"query": "test query"})
            assert resp.status_code == 200
            task_id = resp.json()["task_id"]

            for _ in range(100):
                await asyncio.sleep(0.02)
                if TASKS[task_id]["status"] != "running":
                    break

            status_resp = await client.get(f"/api/research/{task_id}")

    assert TASKS[task_id]["status"] == "done"
    assert TASKS[task_id]["final_report"] == "mock report"
    assert status_resp.json()["final_report"] == "mock report"
    event_types = [e["type"] for e in TASKS[task_id]["events"]]
    assert event_types[0] == "start"
    assert event_types[-1] == "end"
    assert "node_done" in event_types
    assert "tool_call" in event_types  # custom stream（工具调用级别）事件也要透传


class _FailingGraph:
    async def astream(self, state, stream_mode="updates"):
        raise RuntimeError("graph blew up")
        yield ("updates", {})  # pragma: no cover - 让这是个generator


@pytest.mark.asyncio
async def test_start_research_surfaces_error_instead_of_hanging():
    with patch("app.api.main.build_graph", return_value=_FailingGraph()):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/research", json={"query": "test query"})
            task_id = resp.json()["task_id"]

            for _ in range(100):
                await asyncio.sleep(0.02)
                if TASKS[task_id]["status"] != "running":
                    break

    assert TASKS[task_id]["status"] == "error"
    assert any(e["type"] == "error" for e in TASKS[task_id]["events"])
