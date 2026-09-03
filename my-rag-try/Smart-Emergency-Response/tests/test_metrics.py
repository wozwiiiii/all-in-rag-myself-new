"""评测指标单元测试：keyword 命中/别名/中性分、refuse 显式判定、semantic cosine 数学正确性。

2026-09-03 批次1 引入：dataset 补齐 reference 后语义权重生效，
本文件固化 score_answer 的判定口径（含 expected_route 显式化改动）。
"""

import pytest

from eval.metrics import keyword_score, score_answer, semantic_score


class _FakeEmbedLLM:
    """embed_documents 返回按输入顺序的固定向量，不依赖真实 API。"""

    def __init__(self, vectors: list[list[float]]):
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[list[str]] | list[str]):
        self.calls.append(list(texts))
        return self._vectors


# ---------------- keyword_score ----------------


def test_keyword_hit_and_miss():
    assert keyword_score("动火作业分为特级、一级、二级", ["特级", "一级"]) == 1.0
    assert keyword_score("动火作业分级管理", ["特级"]) == 0.0
    assert keyword_score("答案", ["a", "b", "c"]) == 0.0


def test_keyword_alias_hit():
    # "分级"不在答案字面中，但别名"一级/二级/三级"任一命中即视为命中
    assert keyword_score("按照吊物质量分为一级、二级、三级", ["分级"]) == 1.0


def test_keyword_empty_returns_neutral():
    assert keyword_score("任意答案", []) == 0.5


# ---------------- semantic_score ----------------


def test_semantic_score_identical_vectors():
    llm = _FakeEmbedLLM([[1.0, 0.0], [1.0, 0.0]])
    assert semantic_score("答案", "参考", llm) == pytest.approx(1.0)


def test_semantic_score_orthogonal_vectors():
    llm = _FakeEmbedLLM([[1.0, 0.0], [0.0, 1.0]])
    assert semantic_score("答案", "参考", llm) == pytest.approx(0.0)


def test_semantic_score_known_cosine():
    import math

    llm = _FakeEmbedLLM([[1.0, 0.0], [1.0, 1.0]])
    expected = 1.0 / math.sqrt(2)
    assert semantic_score("答案", "参考", llm) == pytest.approx(expected)


def test_semantic_score_missing_reference_is_neutral():
    llm = _FakeEmbedLLM([])
    assert semantic_score("答案", None, llm) == 0.5
    assert llm.calls == []  # 缺参考时不应触发 embedding 调用


# ---------------- score_answer：refuse 显式判定 ----------------


def test_refuse_item_explicit_route_with_refusal():
    llm = _FakeEmbedLLM([])
    s = score_answer(
        answer="该问题超出资料库范围，无法回答",
        refused=True,
        keywords=[],
        reference=None,
        llm=llm,  # type: ignore[arg-type]
        expected_route="refuse",
    )
    assert s["total"] == 1.0 and s["refused"] is True


def test_refuse_item_explicit_route_fabricated_answer():
    llm = _FakeEmbedLLM([])
    s = score_answer(
        answer="巴黎应急预案的第一条是……",
        refused=False,
        keywords=[],
        reference=None,
        llm=llm,  # type: ignore[arg-type]
        expected_route="refuse",
    )
    assert s["total"] == 0.0 and s["refused"] is False


def test_refuse_implicit_fallback_when_no_expected_route():
    """旧数据集兼容：无 expected_route 时退回隐式条件（无关键词且无参考）。"""
    llm = _FakeEmbedLLM([])
    s = score_answer(
        answer="无法回答",
        refused=True,
        keywords=[],
        reference=None,
        llm=llm,  # type: ignore[arg-type]
    )
    assert s["total"] == 1.0


def test_non_refuse_item_with_reference_uses_semantic():
    """有参考答案时总分 = 0.7*kw + 0.3*cos，不再恒为 0.7*x+0.15。"""
    # 关键词全命中 + 语义完全一致 -> 满分 1.0
    llm = _FakeEmbedLLM([[1.0, 0.0], [1.0, 0.0]])
    s = score_answer(
        answer="涉及重氮化工艺的企业数量为4家",
        refused=False,
        keywords=["重氮化"],
        reference="涉及重氮化工艺的企业数量为4家",
        llm=llm,  # type: ignore[arg-type]
        expected_route="table",
    )
    assert s["total"] == pytest.approx(1.0)


def test_non_refuse_item_keyword_miss_semantic_partial():
    """关键词未命中 + 语义正交 -> 0 分（区分度恢复：不再有 0.15 地板分）。"""
    llm = _FakeEmbedLLM([[1.0, 0.0], [0.0, 1.0]])
    s = score_answer(
        answer="答非所问的内容",
        refused=False,
        keywords=["重氮化"],
        reference="涉及重氮化工艺的企业数量为4家",
        llm=llm,  # type: ignore[arg-type]
        expected_route="table",
    )
    assert s["total"] == pytest.approx(0.0)
