"""全局配置，统一用 pydantic-settings 从环境变量 / .env 加载。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str
    tavily_api_key: str

    deepseek_base_url: str = "https://api.deepseek.com/anthropic"
    model_pro: str = "deepseek-v4-pro"
    model_flash: str = "deepseek-v4-flash"

    max_search_results: int = 5

    # 单agent级别的默认值（M1阶段baseline agent使用）
    default_max_tool_calls: int = 6
    default_max_tokens: int = 8000
    default_timeout_seconds: int = 90

    tool_retry_count: int = 2

    # M4新增：跨越一次完整研究任务（可能含多轮Critic迭代）的全局预算硬上限，
    # 由app/infra/budget.py消费，见架构文档6.1节。默认值给得比较宽松，
    # 保证正常查询不会被误伤；只有真正跑偏（比如多轮迭代反复补充调研）
    # 才会触发强制收敛。
    global_max_total_tokens: int = 200_000
    global_max_tool_calls: int = 100
    global_max_wall_clock_seconds: float = 600.0


settings = Settings()
