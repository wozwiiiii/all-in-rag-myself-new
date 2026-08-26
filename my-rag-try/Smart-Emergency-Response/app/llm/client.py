"""OpenAI 兼容 LLM 客户端：统一超时、指数退避重试、429 专项等待、embedding 磁盘缓存。

所有业务模块只允许通过本客户端访问 LLM，禁止各自散落 SDK 调用，
这样超时/重试/降级策略只有一份实现，行为可预期。
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from typing import Any

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.errors import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError)


class _Retryable429(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self._s = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # 重试策略由本类统一管理
        )
        self._cache_path = settings.cache_dir / "embeddings.db"
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_cache()

    # ---------------- 缓存 ----------------

    @contextlib.contextmanager
    def _connect(self):
        # closing 确保连接释放；with 事务语义由 closing 内的 commit 完成
        with contextlib.closing(sqlite3.connect(self._cache_path, timeout=10)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()

    def _init_cache(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                " key TEXT PRIMARY KEY, dim INTEGER, vector BLOB)"
            )

    def _cache_key(self, text: str) -> str:
        raw = f"{self._s.openai_api_embedding_model}:{text}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, texts: list[str]) -> dict[int, np.ndarray]:
        found: dict[int, np.ndarray] = {}
        keys = [self._cache_key(t) for t in texts]
        with self._connect() as conn:
            for i, key in enumerate(keys):
                row = conn.execute(
                    "SELECT dim, vector FROM embeddings WHERE key = ?", (key,)
                ).fetchone()
                if row:
                    found[i] = np.frombuffer(row[1], dtype=np.float32)
        return found

    def _cache_put_many(self, texts: list[str], vectors: list[np.ndarray]) -> None:
        rows = [
            (self._cache_key(t), len(v), v.astype(np.float32).tobytes())
            for t, v in zip(texts, vectors)
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
                rows,
            )

    # ---------------- chat ----------------

    @retry(
        retry=retry_if_exception_type((_Retryable429,) + _RETRYABLE + (APIStatusError,)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Any:
        """返回 chat.completions 的 message 对象（含 content / tool_calls）。"""
        kwargs: dict[str, Any] = dict(
            model=self._s.openai_api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                logger.warning("llm_retryable_status", extra={"status": e.status_code})
                raise _Retryable429(str(e)) from e
            raise
        return resp.choices[0].message

    def chat_safe(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """重试耗尽后包装为 LLMError（AppError 子类），保证上层降级/统一错误体生效。"""
        try:
            return self.chat(messages, **kwargs)
        except (RetryError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            raise LLMError(f"LLM 调用失败（重试已耗尽）: {type(e).__name__}: {str(e)[:200]}") from e

    # ---------------- embedding ----------------

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[np.ndarray | None] = [None] * len(texts)
        cached = self._cache_get(texts)
        for i, vec in cached.items():
            vectors[i] = vec

        misses = [i for i, v in enumerate(vectors) if v is None]
        for batch_start in range(0, len(misses), self._s.embed_batch_size):
            batch_idx = misses[batch_start : batch_start + self._s.embed_batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            data = self._embed_api_call(batch_texts)
            if len(data) != len(batch_texts):
                raise LLMError(
                    f"embedding 返回条数 ({len(data)}) 少于请求条数 ({len(batch_texts)})"
                )
            for i, vec in zip(batch_idx, data):
                vectors[i] = vec
            self._cache_put_many(batch_texts, [vectors[i] for i in batch_idx])  # type: ignore[arg-type]

        return [v.tolist() for v in vectors]  # type: ignore[union-attr]

    @retry(
        retry=retry_if_exception_type((_Retryable429,) + _RETRYABLE),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_api_call(self, batch: list[str]) -> list[np.ndarray]:
        try:
            resp = self._client.embeddings.create(
                model=self._s.openai_api_embedding_model, input=batch
            )
        except APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                logger.warning("embedding_retryable_status", extra={"status": e.status_code})
                raise _Retryable429(str(e)) from e
            raise
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [np.asarray(d.embedding, dtype=np.float32) for d in ordered]
