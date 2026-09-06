"""BM25 内存索引：jieba 分词 + rank_bm25，为向量检索补专有名词/编号精确匹配。

设计动因：dense 检索对"GB30871""动火作业""受限空间"等专有名词是弱项
（语义空间中编号与术语区分度低），BM25 的词频精确匹配恰好互补。
全量 710 chunk 一次性载入内存索引，零 API 成本、纯 CPU 毫秒级查询。

边界（如实声明）：
- BM25 只参与候选补充与排序，不参与拒答判定（阈值语义仍由向量 cosine 决定）；
- 依赖 rank-bm25 / jieba 未安装时由 Retriever 降级为纯向量路。
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """jieba 精确模式分词（HMM 关闭保证确定性），过滤空白与单字符标点。"""
    import jieba

    return [t for t in jieba.lcut(text, cut_all=False, HMM=False) if t.strip()]


class BM25Index:
    """全量 chunk 的 BM25 索引。docs 顺序即索引 i，search 返回 (i, score) 降序。"""

    def __init__(self, docs: list[str]):
        from rank_bm25 import BM25Okapi

        self._docs = docs
        self._corpus = [_tokenize(d) for d in docs]
        # BM25Okapi 要求 corpus 非空；空语料退化为恒空检索
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 [(doc_index, bm25_score)] 按 score 降序。查询无有效 token 时返回空。"""
        if self._bm25 is None or top_k <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order[:top_k] if scores[i] > 0]


def rrf_fuse(rankings: list[list[int]], *, k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion：多路排名融合，返回按融合分降序的候选索引。

    score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 起；未出现在某路的候选不贡献该路分数。
    动因：BM25 分与向量 cosine 量纲不可比，RRF 只用排名，无需调权。
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos)
    return sorted(scores, key=lambda d: scores[d], reverse=True)
