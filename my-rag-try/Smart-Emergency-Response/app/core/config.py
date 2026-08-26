"""集中式配置：所有环境变量在此校验，缺失关键项时启动即失败（fail-fast）。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM（OpenAI 兼容）----
    openai_api_key: str = ""
    openai_api_base: str = "https://api.siliconflow.cn/v1"
    openai_api_model: str = "Qwen/Qwen2.5-7B-Instruct"
    openai_api_embedding_model: str = "BAAI/bge-m3"

    # ---- 服务认证（逗号分隔多 key，支持轮换）----
    api_keys: str = ""

    # ---- 数据目录 ----
    text_data_dir: Path = PROJECT_ROOT / "advance" / "datas" / "文本数据"
    table_data_dir: Path = PROJECT_ROOT / "advance" / "datas" / "表数据+表结构说明+表介绍"

    # ---- 存储（构建产物，全部落在 storage/ 下，已被 .gitignore 排除）----
    storage_dir: Path = PROJECT_ROOT / "storage"
    chroma_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    parquet_dir: Path = PROJECT_ROOT / "storage" / "parquet"
    cache_dir: Path = PROJECT_ROOT / "storage" / "cache"

    # ---- 检索参数 ----
    top_k: int = 6
    score_threshold: float = 0.35  # cosine 相似度阈值，低于则视为无可靠证据

    # ---- 限流 / 超时 / 迭代上限 ----
    rate_limit_per_minute: int = 20
    qa_total_timeout_s: float = 120.0
    sandbox_timeout_s: float = 30.0
    agent_max_iterations: int = 8

    # ---- LLM 调用 ----（重试次数/退避策略固定在 llm/client.py，属代码行为而非运维参数）
    llm_timeout_s: float = 45.0
    embed_batch_size: int = 32

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_list)


@lru_cache
def get_settings() -> Settings:
    return Settings()
