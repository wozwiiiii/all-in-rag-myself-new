"""评测指标单元测试：keyword 命中/别名/中性分、refuse 显式判定、semantic cosine 数学正确性。

2026-09-03 批次1 引入：dataset 补齐 reference 后语义权重生效，
本文件固化 score_answer 的判定口径（含 expected_route 显式化改动）。
2026-09-06 批次5 引入：口径 v2（score_answer_v2）——faith 维度
（证据一致性 × reference 数字一致性 × (1-冗余惩罚)）与冗余惩罚。
"""

import pytest

from eval.metrics import (
    evidence_consistency,
    extract_numbers,
    keyword_score,
    redundancy_penalty,
    ref_consistency,
    score_answer,
    score_answer_v2,
    semantic_score,
)


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


# ---------------- 口径 v2：extract_numbers / evidence_consistency ----------------


def test_extract_numbers_dedup_percent_decimal():
    assert extract_numbers("共107家，占10.5%，107家") == ["107", "10.5%"]
    assert extract_numbers("无数字答案") == []


def test_evidence_consistency_hit():
    ev = "统计结果: 107\nObservation:\n[OK]"
    assert evidence_consistency("一般风险企业共107家", ev) == pytest.approx(1.0)


def test_evidence_consistency_miss():
    # 编造数字（证据里只有 17）→ 0
    assert evidence_consistency("一般风险企业共999家", "Observation:\n[result] 17") == pytest.approx(0.0)


def test_evidence_consistency_neutral_no_evidence():
    # 无证据（如 text 路未透传/空）→ 中性 1.0，不罚
    assert evidence_consistency("任意答案 42", "") == 1.0


def test_evidence_consistency_neutral_no_numbers():
    # 纯描述性答案（无数字）→ 中性 1.0
    assert evidence_consistency("按风险等级划分区域管控", "相关条款文本") == 1.0


# ---------------- 口径 v2：ref_consistency（压 Q15 关键子项）----------------


def test_ref_consistency_wrong_field_answer_pressured():
    """Q15 场景：answer 与证据自洽（都是 0）但与 reference 数字（107/3）矛盾 → 0。"""
    reference = "全市风险等级为一般风险的企业数量为107家（risk_grade=3）。"
    assert ref_consistency("一般风险企业数量为0家", reference) == pytest.approx(0.0)


def test_ref_consistency_correct_answer_not_punished():
    """正确答案（107）→ 1.0：reference 侧额外数字（3）不误伤（方向取 answer 为分子）。"""
    reference = "全市风险等级为一般风险的企业数量为107家（risk_grade=3）。"
    assert ref_consistency("一般风险企业共107家", reference) == pytest.approx(1.0)


def test_ref_consistency_neutral_cases():
    reference = "分为三级管控"
    assert ref_consistency("无数字答案", reference) == 1.0  # answer 无数字 → 中性
    assert ref_consistency("答案 107", None) == 1.0  # 无 reference → 中性
    assert ref_consistency("答案 107", "纯文字参考无数字") == 1.0  # reference 无数字 → 中性


# ---------------- 口径 v2：redundancy_penalty ----------------


def test_redundancy_within_free_ratio():
    ans = "\n".join(f"{i}. 无关要点{i}" for i in range(1, 7))  # 6 要点 / 1 命中 = 6 < 10 → 不罚
    assert redundancy_penalty(ans, ["甲"]) == 0.0


def test_redundancy_halfway():
    ans = "\n".join(f"{i}. 无关要点{i}" for i in range(1, 16))  # 15 / 1 = 15 → (15-10)/10 = 0.5
    assert redundancy_penalty(ans, ["甲"]) == pytest.approx(0.5)


def test_redundancy_full_penalty():
    ans = "\n".join(f"{i}. 无关要点{i}" for i in range(1, 31))  # 30 / 1 = 30 → 打满
    assert redundancy_penalty(ans, ["甲"]) == 1.0


def test_redundancy_single_pass_format_not_penalized():
    # 单遍生成无 "N. " 编号行 → 0（报告标注"未计冗余"）
    assert redundancy_penalty("要点甲；要点乙；要点丙", ["甲"]) == 0.0


# ---------------- 口径 v2：score_answer_v2 ----------------


def test_v2_full_marks_still_one():
    """满分仍为 1.0（权重和为 1）：kw 全命中 + 语义一致 + 数字全对 + 无冗余。"""
    llm = _FakeEmbedLLM([[1.0, 0.0], [1.0, 0.0]])
    s = score_answer_v2(
        answer="1. 涉及重氮化工艺的企业数量为4家",
        refused=False,
        keywords=["重氮化"],
        reference="涉及重氮化工艺的企业数量为4家",
        llm=llm,  # type: ignore[arg-type]
        expected_route="table",
        evidence="[result] 4",
    )
    assert s["total"] == pytest.approx(1.0)
    assert s["faithful"] == pytest.approx(1.0)


def test_v2_pressures_q15_style_error():
    """Q15 场景：kw 全命中 + 证据自洽（沙箱也是 0），但 reference 数字矛盾 →
    faithful=0，total 明显低于旧口径（v1 下该错误几乎不丢分）。"""
    llm = _FakeEmbedLLM([[1.0, 0.0], [1.0, 0.0]])
    common = dict(
        answer="1. 一般风险企业数量为0家，区域按等级划分",
        refused=False,
        keywords=["一般风险", "区域", "划分"],
        reference="全市风险等级为一般风险的企业数量为107家（risk_grade=3）。",
        llm=llm,  # type: ignore[arg-type]
        expected_route="hybrid",
    )
    s_v2 = score_answer_v2(evidence="[result] 0", **common)  # type: ignore[arg-type]
    s_v1 = score_answer(**common)  # type: ignore[arg-type]
    assert s_v2["faithful"] == pytest.approx(0.0)  # ref_consistency=0 压掉整个 faith
    assert s_v2["total"] < s_v1["total"]  # 新口径下该错误丢分更多


def test_v2_refuse_item_scores_one_and_zero():
    llm = _FakeEmbedLLM([])
    s = score_answer_v2(
        answer="该问题超出资料库范围，无法回答",
        refused=True,
        keywords=[],
        reference=None,
        llm=llm,  # type: ignore[arg-type]
        expected_route="refuse",
    )
    assert s["total"] == 1.0 and s["faithful"] == 0.0  # faith 维度对 refuse 类不适用
