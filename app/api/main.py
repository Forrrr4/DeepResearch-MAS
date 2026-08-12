"""FastAPI入口：接收研究请求、后台跑图、通过WebSocket实时推送每个节点
的执行进度（架构文档6.7节"前端展示execution graph的实时状态"）。

任务状态存在进程内内存字典里（TASKS），不做持久化——这是刻意的简化：
M6的目标是"能现场演示、面试官能直观看到多agent协同过程"，不是生产级的
任务队列系统，加Redis/数据库这类持久化会让部署复杂度和这个目标不成
比例，如果之后要支持多进程/重启后恢复任务，才需要升级。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.websocket import manager
from app.graph.build_graph import build_graph
from app.graph.state import make_initial_state
from app.tools.report_render import render_report_html

app = FastAPI(title="DeepResearch-MAS")

TASKS: dict[str, dict[str, Any]] = {}

_FRONTEND_INDEX = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


class ResearchRequest(BaseModel):
    query: str


async def _emit(task_id: str, event: dict[str, Any]) -> None:
    TASKS[task_id]["events"].append(event)
    await manager.broadcast(task_id, event)


def _summarize_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    """把某个节点返回的state更新，抽取成前端进度树关心的几个字段，
    不是把整个ResearchState原样推给前端（避免把大段findings/report文本
    塞进每一条WS消息）。"""
    summary: dict[str, Any] = {}
    if "routing_decision" in update:
        summary["routing_decision"] = update["routing_decision"]
    if "subtasks" in update:
        summary["n_subtasks"] = len(update["subtasks"])
    if "findings" in update:
        summary["n_findings"] = len(update["findings"])
    if "citation_map" in update:
        summary["n_citations"] = len(update["citation_map"])
    if "iteration" in update:
        summary["iteration"] = update["iteration"]
    if "budget" in update:
        summary["tokens_used"] = update["budget"].get("tokens_used")
    if "subtask_errors" in update and update["subtask_errors"]:
        summary["subtask_errors"] = update["subtask_errors"]
    return summary


async def run_research_task(task_id: str, query: str) -> None:
    graph = build_graph()
    state = make_initial_state(query, trace_id=task_id)
    start = time.monotonic()

    await _emit(task_id, {"type": "start", "query": query, "elapsed_seconds": 0.0})

    accumulated: dict[str, Any] = dict(state)
    try:
        # 同时订阅updates（节点粒度：某个agent跑完了、state有哪些字段变化）
        # 和custom（工具调用粒度：SubAgent/BaselineAgent内部每次web_search
        # 调用都能实时推出来，见app/infra/progress.py）——两种粒度合在一起
        # 才能让前端真正看到"壳里在干什么"，而不是等一个节点跑完（可能
        # 50-100秒）才收到一条汇总。
        async for stream_mode_name, chunk in graph.astream(state, stream_mode=["updates", "custom"]):
            if stream_mode_name == "custom":
                await _emit(
                    task_id,
                    {**chunk, "elapsed_seconds": round(time.monotonic() - start, 2)},
                )
                continue
            for node_name, update in chunk.items():
                accumulated.update(update)
                await _emit(
                    task_id,
                    {
                        "type": "node_done",
                        "node": node_name,
                        "elapsed_seconds": round(time.monotonic() - start, 2),
                        **_summarize_update(node_name, update),
                    },
                )
    except Exception as exc:  # noqa: BLE001 - 后台任务边界，必须捕获并上报而不是让task静默卡在running
        TASKS[task_id]["status"] = "error"
        await _emit(
            task_id,
            {"type": "error", "message": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.monotonic() - start, 2)},
        )
        return

    final_report = accumulated.get("final_report")
    final_report_html = render_report_html(final_report)
    TASKS[task_id]["status"] = "done"
    TASKS[task_id]["final_report"] = final_report
    TASKS[task_id]["final_report_html"] = final_report_html
    await _emit(
        task_id,
        {
            "type": "end",
            "final_report": final_report,
            "final_report_html": final_report_html,
            "elapsed_seconds": round(time.monotonic() - start, 2),
        },
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_FRONTEND_INDEX)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def start_research(req: ResearchRequest) -> dict[str, str]:
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "running", "events": [], "final_report": None, "final_report_html": None}
    asyncio.create_task(run_research_task(task_id, req.query))
    return {"task_id": task_id}


@app.get("/api/research/{task_id}")
async def get_research(task_id: str) -> dict[str, Any]:
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str) -> None:
    await manager.connect(task_id, websocket)
    task = TASKS.get(task_id)
    if task is not None:
        # 客户端可能在任务已经跑了一段之后才连上WS，先把历史事件补发一遍，
        # 保证进度树不会"从中间开始"，然后再靠broadcast收实时事件。
        for event in task["events"]:
            await websocket.send_json(event)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id, websocket)
