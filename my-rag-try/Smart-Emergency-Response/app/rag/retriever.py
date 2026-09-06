"""融合检索：全量精确 cosine + BM25 → 并集 → 阈值过滤 → rerank 精排（取 top_k）。

坐标系设计（2026-08-31 BM25 混合引入）：
- 启动时一次性 `collection.get(include=[documents, metadatas, embeddings])`
  拉全量建立内存索引，向量分与 BM25 分共享同一"全量文档索引"坐标，
  避免两路候选无法对齐的问题；
- 710 chunk 规模下全量 numpy 矩阵乘（毫秒级）既精确又快于 HNSW 近似检索；
  语料增长到 10^5+ 时再切回 Chroma query + 窗口融合（坐标对齐需换 id 映射）。

设计要点：
- 拒答语义不变：仍按向量 cosine 阈值过滤。BM25 只补充候选与参与排序，
  不参与拒答判定（BM25-only 候选也必须过向量阈值）；
- BM25 补 dense 检索弱项：专有名词/编号（"GB30871""动火作业"）精确匹配；
- rerank 失败或关闭时按 RRF 融合序（而非纯向量序）截断；
- BM25 依赖缺失/构建失败自动降级纯向量路（日志 bm25_failed）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.core.errors import LLMError
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.rag.bm25 import rrf_fuse
from app.rag.ingest import COLLECTION_NAME

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    heading: str
    score: float  # cosine 相似度
    rerank_score: float | None = None  # rerank 相关性分数（未启用/失败时为 None）


class Retriever:
    def __init__(self, settings: Settings, llm: LLMClient):
        import chromadb

        self._s = settings
        self._llm = llm
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            self._collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"向量索引不存在或无法加载，请先执行: python -m scripts.build_index ({e})"
            ) from e

        # 全量内存索引：统一坐标（i ↔ _docs[i]），向量与 BM25 共用
        got = self._collection.get(include=["documents", "metadatas", "embeddings"])
        self._docs: list[str] = got["documents"]
        self._metas: list[dict] = got["metadatas"]
        emb = np.asarray(got["embeddings"], dtype=np.float32)
        # 行归一化后点积即 cosine
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb_normed = emb / norms

        # BM25 索引：jieba 分词 + BM25Okapi（零 API 成本，毫秒级）
        self._bm25 = None
        if settings.bm25_enabled:
            try:
                from app.rag.bm25 import BM25Index

                self._bm25 = BM25Index(self._docs)
            except Exception as e:
                logger.warning("bm25_failed_fallback_vector", extra={"err": str(e)[:200]})

        logger.info(
            "retriever_ready",
            extra={"docs": len(self._docs), "bm25": self._bm25 is not None},
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        # ---- 第一路：全量精确 cosine，取 top recall_k ----
        qvec = np.asarray(self._llm.embed_query(query), dtype=np.float32)
        qn = float(np.linalg.norm(qvec))
        if qn:
            qvec = qvec / qn
        sims = (self._emb_normed @ qvec).tolist()  # 全量 cosine
        vec_ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[
            : self._s.recall_k
        ]

        # ---- 第二路：BM25 召回（专有名词/编号精确匹配），失败降级纯向量 ----
        bm25_ranked: list[int] = []
        if self._bm25 is not None:
            try:
                bm25_ranked = [i for i, _ in self._bm25.search(query, top_k=self._s.recall_k)]
            except Exception as e:
                logger.warning("bm25_search_failed", extra={"err": str(e)[:200]})

        # ---- 并集候选池 + 向量阈值过滤（拒答语义保持纯向量）----
        pool_ids = list(dict.fromkeys(vec_ranked + bm25_ranked))  # 保序去重
        pool_ids = [i for i in pool_ids if sims[i] >= self._s.score_threshold]
        if not pool_ids:
            logger.info("retrieve_no_evidence", extra={"query_len": len(query)})
            return []

        pos_of = {pid: pos for pos, pid in enumerate(pool_ids)}
        pool = [
            RetrievedChunk(
                text=self._docs[i],
                source=self._metas[i].get("source", "?"),
                heading=self._metas[i].get("heading", ""),
                score=round(sims[i], 4),
            )
            for i in pool_ids
        ]

        # ---- 排序：rerank 精排；不可用时 RRF 融合序 ----
        rerank_failed = False
        if self._s.rerank_enabled and len(pool) > self._s.top_k:
            try:
                ranked = self._llm.rerank(query, [c.text for c in pool], top_n=self._s.top_k)
                hits_sorted = [pool[i] for i, _ in ranked]
                for c, (_, r_score) in zip(hits_sorted, ranked):
                    c.rerank_score = round(r_score, 4)
            except LLMError as e:
                rerank_failed = True
                logger.warning("rerank_failed_fallback_rrf", extra={"err": str(e)[:200]})
                hits_sorted = self._rrf_order(pool_ids, pool, sims, bm25_ranked)
        else:
            hits_sorted = self._rrf_order(pool_ids, pool, sims, bm25_ranked)

        hits = hits_sorted[: self._s.top_k]
        logger.info(
            "retrieve",
            extra={"query_len": len(query), "candidates": len(pool_ids), "kept": len(hits),
                   "top_score": hits[0].score if hits else None,
                   "bm25_used": self._bm25 is not None,
                   "rerank_failed": rerank_failed},
        )
        return hits

    def _rrf_order(
        self, pool_ids: list[int], pool: list[RetrievedChunk],
        sims: list[float], bm25_ranked: list[int],
    ) -> list[RetrievedChunk]:
        """无 rerank 时的降级排序：向量路 + BM25 路（pool 内）的 RRF 融合序。"""
        pos_of = {pid: pos for pos, pid in enumerate(pool_ids)}
        vec_ranking = sorted(pool_ids, key=lambda i: -sims[i])  # pool 内向量排名
        bm25_in_pool = [i for i in bm25_ranked if i in pos_of]  # BM25 完整排名的 pool 子序列
        fused = rrf_fuse([vec_ranking, bm25_in_pool])
        return [pool[pos_of[i]] for i in fused if i in pos_of]
