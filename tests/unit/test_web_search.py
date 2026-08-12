"""app/tools/web_search.py 单元测试：全部mock Tavily客户端，不打真实API。"""
from unittest.mock import AsyncMock, patch

import pytest

from app.tools import web_search as web_search_module


@pytest.mark.asyncio
async def test_web_search_success_formats_results():
    fake_response = {
        "results": [
            {
                "title": "T1",
                "url": "https://a.com",
                "content": "abc",
                "published_date": "2026-01-01",
            }
        ]
    }
    with patch.object(
        web_search_module._client, "search", new=AsyncMock(return_value=fake_response)
    ):
        result = await web_search_module.web_search("test query", max_results=3)

    assert result["error"] is None
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://a.com"


@pytest.mark.asyncio
async def test_web_search_retries_twice_then_returns_error():
    with patch.object(
        web_search_module._client, "search", new=AsyncMock(side_effect=RuntimeError("boom"))
    ) as mock_search, patch.object(web_search_module.asyncio, "sleep", new=AsyncMock()):
        result = await web_search_module.web_search("test query")

    # 初次 + 2次重试 = 3次调用，对应CLAUDE.md第3条硬性约束：工具调用失败重试固定2次
    assert mock_search.call_count == 3
    assert result["error"] is not None
    assert result["results"] == []


@pytest.mark.asyncio
async def test_web_search_empty_results_no_error():
    with patch.object(
        web_search_module._client, "search", new=AsyncMock(return_value={"results": []})
    ):
        result = await web_search_module.web_search("no such thing")

    assert result["error"] is None
    assert result["results"] == []
