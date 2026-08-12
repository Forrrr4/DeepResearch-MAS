"""DeepSeek-V4-Pro / V4-Flash 分层路由。

DeepSeek 提供与 Anthropic Messages API 兼容的端点
(base_url=https://api.deepseek.com/anthropic)，因此这里直接复用
anthropic SDK，只替换 base_url 和 api_key，agent 代码可以沿用标准的
tool_use 格式，不需要额外适配层。

注意：DeepSeek 没有内置 web_search 工具，任何搜索能力都必须通过
app/tools/web_search.py 显式接入 Tavily，不能假设模型自带搜索。
"""
import asyncio
from typing import Any, Literal

from anthropic import AsyncAnthropic, APIConnectionError, APITimeoutError

from app.config import settings

ModelTier = Literal["pro", "flash"]

_client = AsyncAnthropic(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def get_client() -> AsyncAnthropic:
    return _client


async def create_message_with_retry(client: AsyncAnthropic, **kwargs: Any) -> Any:
    """对顶层LLM调用做和工具调用一致的重试语义（CLAUDE.md第3条：失败重试固定2次+指数退避）。

    只重试网络层瞬时故障（连接/超时），模型返回的业务错误（如参数错误）
    不重试，避免掩盖真实的配置问题。
    """
    last_error: Exception | None = None
    for attempt in range(settings.tool_retry_count + 1):
        try:
            return await client.messages.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
            if attempt < settings.tool_retry_count:
                await asyncio.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def resolve_model(tier: ModelTier) -> str:
    """把角色定位（强推理 vs 轻量任务）映射到具体模型名。

    Orchestrator / Critic / Writer 用 pro；Guardrail规则判断、摘要压缩、
    简单格式化等低复杂度任务用 flash。M1 阶段只有一个 baseline agent，
    统一用 pro（因为它同时要承担理解问题+整合搜索结果的推理工作）。
    """
    return settings.model_pro if tier == "pro" else settings.model_flash
