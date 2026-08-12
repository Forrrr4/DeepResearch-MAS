"""app/infra/model_router.py 单元测试：模型分层路由 + 顶层LLM调用重试语义。"""
from unittest.mock import AsyncMock, patch

import pytest
from anthropic import APITimeoutError

from app.infra.model_router import create_message_with_retry, resolve_model


def test_resolve_model_maps_pro_and_flash_tiers():
    assert resolve_model("pro") == "deepseek-v4-pro"
    assert resolve_model("flash") == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_create_message_with_retry_succeeds_first_try():
    fake_client = AsyncMock()
    fake_client.messages.create = AsyncMock(return_value="ok")

    result = await create_message_with_retry(fake_client, model="deepseek-v4-pro", messages=[])

    assert result == "ok"
    assert fake_client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_create_message_with_retry_retries_on_timeout_then_succeeds():
    timeout_error = APITimeoutError(request=object())
    fake_client = AsyncMock()
    fake_client.messages.create = AsyncMock(side_effect=[timeout_error, "ok"])

    with patch("app.infra.model_router.asyncio.sleep", new=AsyncMock()):
        result = await create_message_with_retry(fake_client, model="deepseek-v4-pro", messages=[])

    assert result == "ok"
    assert fake_client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_create_message_with_retry_raises_after_exhausting_retries():
    timeout_error = APITimeoutError(request=object())
    fake_client = AsyncMock()
    fake_client.messages.create = AsyncMock(side_effect=timeout_error)

    with patch("app.infra.model_router.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(APITimeoutError):
            await create_message_with_retry(fake_client, model="deepseek-v4-pro", messages=[])

    # 初次 + 2次重试 = 3次调用，对应CLAUDE.md第3条硬性约束
    assert fake_client.messages.create.call_count == 3
