"""BM25 索引与 RRF 融合的单元测试：分词检索、专有名词精确匹配、空防御、融合语义。"""

from app.rag.bm25 import BM25Index, rrf_fuse


def test_bm25_exact_keyword_match():
    """专有名词精确匹配：含目标词的文档应排在无关文档之前。"""
    docs = [
        "化工园区平台架构设计说明",
        "GB30871 危险化学品企业特殊作业安全规范全文",
        "受限空间作业气体检测要求",
        "企业双重预防机制数字化建设指南",
    ]
    idx = BM25Index(docs)
    hits = idx.search("GB30871 特殊作业规范", top_k=3)
    assert hits, "应有命中"
    top_ids = [i for i, _ in hits]
    assert 1 in top_ids[:2]  # 含 GB30871 的文档进前二


def test_bm25_no_hit_returns_empty():
    """无有效 token / 无命中：返回空列表不抛异常。"""
    docs = ["危险化学品管理条例", "安全生产法"]
    idx = BM25Index(docs)
    assert idx.search("", top_k=3) == []
    assert idx.search("红烧肉的做法", top_k=3) == []  # 无词项命中（score=0 过滤）


def test_bm25_empty_corpus():
    """空语料：构建不抛异常，检索恒空。"""
    idx = BM25Index([])
    assert idx.search("任意查询", top_k=3) == []


def test_rrf_fuse_basic():
    """RRF 基本语义：两路都靠前的候选融合分最高。"""
    # 候选 0 在两路均第一 → 融合第一；候选 9 只在 BM25 路第一
    vec = [0, 1, 2, 3, 4]
    bm25 = [9, 0, 5, 6, 7]
    fused = rrf_fuse([vec, bm25])
    assert fused[0] == 0  # 两路都出现且都靠前
    assert set(fused) == set(vec) | set(bm25)


def test_rrf_fuse_single_ranking():
    """单路退化：等价于该路原序。"""
    fused = rrf_fuse([[3, 1, 2]])
    assert fused == [3, 1, 2]


def test_rrf_fuse_empty():
    """全空输入：返回空。"""
    assert rrf_fuse([[], []]) == []
