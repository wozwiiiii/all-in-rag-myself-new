"""口径 v3 单测：结论数字去噪、独立数值项、硬封顶。

2026-09-06 P0-2 引入。验收基准 = 72B 双协议实测（eval/reports/protocol-bench-
20260906-102604.md）暴露的 v2 失效场景：数值全错的答案（"0 家"）在 v2 下拿
0.81-0.90，与正确题无区分度。v3 必须做到：
- id 11/12/13 的真实错误答案 → total < 0.45（被封顶或低分）
- id 9/10 的真实正确答案 → total >= 0.8
测试用真实 dataset reference 文本与真实实测答案原文，mock 仅限 embedding。
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    V3_NUM_MISS_CAP,
    extract_conclusion_numbers,
    numeric_consistency,
    score_answer_v3,
)


class _FakeEmbedLLM:
    """embed_documents 返回按输入顺序的固定向量，不依赖真实 API。"""

    def __init__(self, vectors: list[list[float]]):
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return self._vectors


# ---------------- extract_conclusion_numbers：去噪 ----------------

def test_conclusion_numbers_basic_units():
    assert extract_conclusion_numbers("企业数量为4家") == ["4"]
    assert extract_conclusion_numbers("共登记仓库记录348条，有效记录为273条") == ["348", "273"]
    assert extract_conclusion_numbers("涉及2-丙醇的企业 11 家") == ["11"]


def test_conclusion_numbers_strips_field_assignment():
    """字段赋值（delete_status=0 / check_status=2）不是结论数字——v2 被其稀释的实证。"""
    ans = "企业满足 delete_status=0 且 check_status=2，涉及企业数量为 4 家。"
    assert extract_conclusion_numbers(ans) == ["4"]


def test_conclusion_numbers_strips_dates():
    """日期成分（2025年3月22日）不是结论数字——id11 错误 0 被日期稀释的实证。"""
    ans = "2025年3月22日开停车的企业数量为 0 家。"
    assert extract_conclusion_numbers(ans) == ["0"]


def test_conclusion_numbers_strips_points_numbering():
    assert extract_conclusion_numbers("1. 涉及企业 4 家\n2. 依据 v_process") == ["4"]


def test_conclusion_numbers_no_qty_unit_is_empty():
    """无数量单位的数字（纯日期/字段值）抽不出结论——非数量型问题走中性分支。"""
    assert extract_conclusion_numbers("时间范围为2021年至2023年") == []
    assert extract_conclusion_numbers("风险等级为一般风险（risk_grade=3）") == []


# ---------------- numeric_consistency ----------------

def test_numeric_consistency_hit_and_miss():
    ref = "共有156家企业提交安全承诺，其中处于开停车状态（open_parking=1）的企业数量为4家。"
    assert numeric_consistency("开停车企业数量为4家。", ref) == 1.0
    assert numeric_consistency("开停车的企业数量为 0 家。", ref) == 0.0


def test_numeric_consistency_neutral_cases():
    assert numeric_consistency("答案 4 家", None) == 1.0  # 无 reference
    assert numeric_consistency("无数量结论的答案", "企业数量为4家") == 1.0  # 答案无结论数字
    assert numeric_consistency("答案 4 家", "时间范围为2021年至2023年") == 1.0  # 参考无结论数字


# ---------------- score_answer_v3：验收基准（真实数据） ----------------

# dataset.jsonl 真实 reference（id 9-13）
_REF_9 = "依据v_process工艺表统计，涉及重氮化工艺的企业数量为4家。重氮化属于重点监管危险化工工艺，按国家要求2026年底前须完成全流程自动化改造。"
_REF_10 = "依据v_sds危险化学品登记表统计，涉及危化品2-丙醇的企业数量为11家。2-丙醇为易燃液体，相关企业均已在该表中登记备案。"
_REF_11 = "依据enterprise_promise企业安全承诺表统计，2025年3月22日共有156家企业提交安全承诺，其中处于开停车状态（open_parking=1）的企业数量为4家。"
_REF_12 = "依据enterprise_promise企业安全承诺表统计，全市风险等级为一般风险（risk_grade=3）的企业数量为107家。"
_REF_13 = "依据t_dhzy动火作业记录表统计，剔除已删除记录后，涉及动火作业的企业共50家。动火作业按GB30871-2022分为特级、一级、二级管理。"

# protocol-bench-20260906-102604.md 真实实测答案（react 协议原文）
_ANS_9_OK = '涉及重氮化工艺的企业数量为 4 家。依据：v_process 表 process_name 模糊匹配"重氮化工艺"，企业满足 delete_status=0 且 check_status=2，按 org_code 去重统计。'
_ANS_10_OK = '涉及危化品2-丙醇的企业数量为 11 家。依据：v_sds 表 sds_name 模糊匹配"2-丙醇"，企业满足 delete_status=0 且 check_status=2，按 org_code 去重统计。'
_ANS_11_WRONG = "2025年3月22日开停车的企业数量为 0 家。依据：base_device_draft 表和 base_t_enterprise_draft 表中未找到相关记录。"
_ANS_12_WRONG = '全市风险等级为一般风险的企业数量为 0 家。依据：base_t_enterprise_draft 表中 risk_rating_code 为"一般风险"，企业满足 delete_status=0 且 check_status=2，按 org_code 去重统计。'
_ANS_13_WRONG = "涉及动火作业的企业数量为 0 家。依据：t_dhzy 表中 dhzyjb（动火作业级别）不为空，企业满足 delete_status=0 且 check_status=2，按 org_code 去重统计。"


def _score(ans: str, ref: str, keywords: list[str], vectors: list[list[float]]) -> dict:
    llm = _FakeEmbedLLM(vectors)
    return score_answer_v3(
        answer=ans, refused=False, keywords=keywords, reference=ref, llm=llm,
        expected_route="table",
    )


def test_v3_correct_answers_score_high():
    """正确题（id 9/10 真实答案）必须 ≥0.8。"""
    for ans, ref, kws in (
        (_ANS_9_OK, _REF_9, ["重氮化"]),
        (_ANS_10_OK, _REF_10, ["2-丙醇"]),
    ):
        r = _score(ans, ref, kws, [[1.0, 0.0], [1.0, 0.0]])  # sem=1 上界
        assert r["numeric"] == 1.0
        assert r["total"] >= 0.8, r


def test_v3_wrong_answers_capped_low():
    """错题（id 11/12/13 真实错误答案"0 家"）必须 < 0.45——数值错不能再拿高分。"""
    for ans, ref, kws in (
        (_ANS_11_WRONG, _REF_11, ["开停车"]),
        (_ANS_12_WRONG, _REF_12, ["一般风险"]),
        (_ANS_13_WRONG, _REF_13, ["动火"]),
    ):
        r = _score(ans, ref, kws, [[1.0, 0.0], [1.0, 0.0]])  # sem=1 上界：仍封得住
        assert r["numeric"] == 0.0, r
        assert r["total"] <= V3_NUM_MISS_CAP + 1e-9, r
        assert r["total"] < 0.45, r


def test_v3_wrong_vs_right_gap():
    """同题"对 vs 错"分差 ≥0.4（v2 实测仅 0.01-0.1，无区分度）。"""
    right = _score(_ANS_9_OK, _REF_9, ["重氮化"], [[1.0, 0.0], [1.0, 0.0]])
    wrong = _score(_ANS_11_WRONG, _REF_11, ["开停车"], [[1.0, 0.0], [1.0, 0.0]])
    assert right["total"] - wrong["total"] >= 0.4


def test_v3_cjk_qty_rewrite_exempt_from_cap():
    """中文数字改写（"约五十家"）：规则版查不到 → numeric 中性 1.0（不罚也不奖，
    与 v1/v2 局限声明一致），且豁免硬封顶，分数由 semantic 主导。"""
    r = _score("涉及动火作业的企业约五十家。", _REF_13, ["动火"], [[1.0, 0.0], [0.0, 1.0]])
    assert r["numeric"] == 1.0  # 中性：结论数字抽取对中文数字无感
    assert r["total"] > V3_NUM_MISS_CAP  # 未被封顶


def test_v3_refuse_item():
    llm = _FakeEmbedLLM([])
    r = score_answer_v3(
        answer="抱歉，无法回答该问题。", refused=False, keywords=[],
        reference=None, llm=llm, expected_route="refuse",
    )
    assert r["total"] == 1.0
    assert r["numeric"] == 0.0


def test_v3_text_question_without_numbers_stays_neutral():
    """非数量型 text 题（结论数字双侧皆空）numeric=1.0，不参与拉分。"""
    r = _score("平台采用B/S架构，分为六层。", "平台采用B/S架构，总体划分为六个层次。", ["平台", "架构"], [[1.0, 0.0], [1.0, 0.0]])
    assert r["numeric"] == 1.0
    assert r["total"] == pytest.approx(1.0, abs=1e-6)
