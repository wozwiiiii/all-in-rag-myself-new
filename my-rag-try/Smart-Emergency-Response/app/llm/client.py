"""OpenAI 兼容 LLM 客户端：统一超时、指数退避重试、429 专项等待、embedding 磁盘缓存。

所有业务模块只允许通过本客户端访问 LLM，禁止各自散落 SDK 调用，
这样超时/重试/降级策略只有一份实现，行为可预期。
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import sqlite3
import threading
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
        # usage 观测：实例级线程安全累加器 + 请求级快照差值。
        # 采用累加器而非 contextvar——generator 抽取轮用 ThreadPoolExecutor 并行，
        # contextvar 在池线程 copy-context 写不回父上下文，会丢并行 token。
        # 代价：并发请求间 token 归属有交叉误差，观测用途可接受。
        self._usage_lock = threading.Lock()
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    # ---------------- usage 观测 ----------------

    def _record_usage(self, usage: Any, model: str) -> None:
        """累加一次 LLM 调用的 token 用量（usage 缺失时仅计调用次数）。"""
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", None) or 0
            completion = getattr(usage, "completion_tokens", None) or 0
            total = getattr(usage, "total_tokens", None) or (prompt + completion)
            logger.info(
                "llm_call_usage",
                extra={"model": model, "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total},
            )
        else:
            prompt = completion = total = 0
        with self._usage_lock:
            self._usage["prompt_tokens"] += prompt
            self._usage["completion_tokens"] += completion
            self._usage["total_tokens"] += total
            self._usage["calls"] += 1

    def usage_snapshot(self) -> dict:
        """返回当前累加器的深拷贝快照，供请求入口做差值基线。"""
        with self._usage_lock:
            return copy.deepcopy(self._usage)

    def usage_delta(self, before: dict) -> dict:
        """计算 from 快照到当前的用量增量；无 before 字段时取绝对累计。"""
        cur = self.usage_snapshot()
        return {
            "prompt_tokens": cur["prompt_tokens"] - before.get("prompt_tokens", 0),
            "completion_tokens": cur["completion_tokens"] - before.get("completion_tokens", 0),
            "total_tokens": cur["total_tokens"] - before.get("total_tokens", 0),
            "calls": cur["calls"] - before.get("calls", 0),
        }

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
        self._record_usage(resp.usage, self._s.openai_api_model)
        return resp.choices[0].message

    def chat_safe(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """重试耗尽后包装为 LLMError（AppError 子类），保证上层降级/统一错误体生效。"""
        try:
            return self.chat(messages, **kwargs)
        except (RetryError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            raise LLMError(f"LLM 调用失败（重试已耗尽）: {type(e).__name__}: {str(e)[:200]}") from e

    # ---------------- rerank ----------------

    @retry(
        retry=retry_if_exception_type((_Retryable429,) + _RETRYABLE + (APIStatusError,)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _rerank_api_call(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """直调 /v1/rerank（OpenAI SDK 未封装该端点）。返回 [(原索引, 相关性分数)] 降序。"""
        import httpx

        url = f"{self._s.openai_api_base.rstrip('/')}/rerank"
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self._s.openai_api_key}"},
                json={
                    "model": self._s.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
                timeout=self._s.llm_timeout_s,
            )
        except httpx.HTTPError as e:  # 连接/超时统一映射为可重试
            raise APIConnectionError(message=str(e)) from e  # type: ignore[arg-type]
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Retryable429(f"rerank {resp.status_code}: {resp.text[:200]}")
        if resp.status_code != 200:
            raise LLMError(f"rerank 请求失败 {resp.status_code}: {resp.text[:200]}")
        results = resp.json().get("results", [])
        return [(int(r["index"]), float(r["relevance_score"])) for r in results]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """重试耗尽后抛 LLMError，由调用方决定降级策略（Retriever 会退回向量序）。"""
        try:
            return self._rerank_api_call(query, documents, top_n)
        except (
            RetryError, APIConnectionError, APITimeoutError, APIStatusError, _Retryable429, LLMError,
        ) as e:
            raise LLMError(f"rerank 调用失败（重试已耗尽）: {type(e).__name__}: {str(e)[:200]}") from e

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
