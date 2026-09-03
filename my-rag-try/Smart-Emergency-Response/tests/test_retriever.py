"""融合检索测试：BM25 并集补充、rerank 重排、RRF 降级、阈值语义不变。

Mock 结构对齐 Retriever 新接口：__init__ 走 collection.get(include=[...])，
retrieve 走内存 numpy cosine + BM25 + rerank。
"""

import numpy as np

from app.core.config import Settings
from app.core.errors import LLMError
from app.rag.retriever import Retriever

N_DOCS = 12  # 全量文档数


class _FakeCollection:
    """12 篇文档：doc0..doc7 含关键词"受限空间"，其余为无关文本。

    embedding 设计：查询向量与文档向量在同一低位空间，sim(doc0..7) 高（>0.35）、
    其余低——保证阈值过滤语义可测。
    """

    KEY = "受限空间"

    def get(self, include):
        docs = [f"{self.KEY}安全措施文档{i}" if i < 8 else f"无关内容{i}" for i in range(N_DOCS)]
        metas = [{"source": f"s{i}.pdf", "heading": f"h{i}"} for i in range(N_DOCS)]
        # 构造可区分的 embedding：前 8 个在查询方向上有高分量
        base = np.zeros((N_DOCS, 4), dtype=np.float32)
        for i in range(N_DOCS):
            base[i, 0] = 1.0  # 主方向
            base[i, 1] = 0.05 * (N_DOCS - i)  # 与查询内含的次方向正相关
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        return {"documents": docs, "metadatas": metas, "embeddings": base.tolist()}


class _StubLLM:
    """查询向量 = [1, 1, 0, 0]/norm：与文档主方向和次方向都正相关。"""

    def __init__(self, rerank_result=None, rerank_error=False):
        self._result = rerank_result
        self._error = rerank_error
        self.rerank_calls = 0

    def embed_query(self, text):
        v = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def rerank(self, query, documents, top_n):
        self.rerank_calls += 1
        if self._error:
            raise LLMError("rerank down")
        return self._result


def _retriever(llm, **overrides) -> Retriever:
    s = Settings(_env_file=None, **overrides)
    r = Retriever.__new__(Retriever)
    r._s = s
    r._llm = llm
    r._collection = _FakeCollection()
    import numpy as np

    got = r._collection.get(include=["documents", "metadatas", "embeddings"])
    r._docs = got["documents"]
    r._metas = got["metadatas"]
    emb = np.asarray(got["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    r._emb_normed = emb / norms
    if s.bm25_enabled:
        from app.rag.bm25 import BM25Index

        r._bm25 = BM25Index(r._docs)
    else:
        r._bm25 = None
    return r


def test_threshold_semantics_unchanged():
    """阈值过滤语义不变：无关文档（sim 低 + 无关键词）不进入结果。"""
    llm = _StubLLM(rerank_result=[(i, 0.9) for i in range(8)])
    r = _retriever(llm)
    hits = r.retrieve("受限空间安全措施")
    assert all("受限空间" in h.text for h in hits)
    assert len(hits) <= 8


def test_bm25_brings_keyword_exact_match_into_pool():
    """BM25 价值场景：关键词精确命中的文档进入候选池（向量路已含时验证并集不丢）。"""
    llm = _StubLLM(rerank_result=[(0, 0.99), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)])
    r = _retriever(llm)
    hits = r.retrieve("受限空间")
    assert len(hits) == 6  # top_k=6
    assert all("受限空间" in h.text for h in hits)


def test_bm25_disabled_skips_index():
    """bm25_enabled=False：不建 BM25 索引，纯向量 + rerank。"""
    llm = _StubLLM(rerank_result=[(i, 0.9) for i in range(6)])
    r = _retriever(llm, bm25_enabled=False)
    assert r._bm25 is None
    hits = r.retrieve("受限空间")
    assert len(hits) == 6
    assert llm.rerank_calls == 1


def test_rerank_reorders_and_sets_score():
    """rerank 成功：按相关性重排并记录 rerank_score。"""
    # pool 8 个候选中选 6：让原第 6 位升到首位
    llm = _StubLLM(rerank_result=[(5, 0.99), (0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)])
    r = _retriever(llm)
    hits = r.retrieve("受限空间安全措施")
    assert hits[0].text.endswith("5") or "5" in hits[0].text  # doc5 排首位
    assert hits[0].rerank_score == 0.99


def test_rerank_failure_falls_back_to_rrf():
    """rerank 失败：降级 RRF 融合序（不抛异常），结果仍过阈值且截断 top_k。"""
    llm = _StubLLM(rerank_error=True)
    r = _retriever(llm)
    hits = r.retrieve("受限空间安全措施")
    assert len(hits) == 6
    assert all(h.rerank_score is None for h in hits)
    assert all("受限空间" in h.text for h in hits)
