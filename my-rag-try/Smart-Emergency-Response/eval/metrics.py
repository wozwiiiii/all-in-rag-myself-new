"""评测指标：对齐竞赛评分公式的可执行近似。

口径 v1（score_answer）：
score = 0.7 * keyword_score + 0.3 * semantic_score
- keyword_score：期望关键词命中率（竞赛为带权，评测集简化为均权）
- semantic_score：答案与参考描述的 embedding 余弦相似度（参考描述缺失时记 0.5 中性分）
refuse 类：系统明确拒答得 1 分，编造答案得 0 分。
判定来源：dataset 的 expected_route == "refuse"（显式），
或旧数据集的隐式条件（无关键词且无参考描述）。

口径 v2（score_answer_v2，2026-09-06）：v1 只测覆盖不测精确率——
Q15 实证答错字段（risk_rating_code 应为 risk_grade=3）仍拿关键词分。
total = 0.6 * keyword + 0.25 * semantic + 0.15 * faithful
faithful = evidence_consistency × ref_consistency(有 reference 时) × (1 - redundancy_penalty)
- evidence_consistency：answer 抽取的数字在证据文本（召回 chunk / 沙箱 Observation）中
  出现的比例；无证据或答案无数字 → 1.0（中性不罚）
- ref_consistency：answer 数字与 reference 数字的命中比。压 Q15 类"字段误用但数值
  自洽"错误的关键子项——此类错误 answer 与沙箱执行结果一致（都是错的 0），纯证据
  一致性查不出，只有对照 reference 数字才能发现
- redundancy_penalty：要点行数/关键词命中数之比超过 10.0 后线性惩罚、20.0 打满
  （阈值经 2026-09-06 实测校准：text 题要点/命中比常态 8-13，旧值 2.0 会误伤全部
  正常列举型答案；详见 _REDUNDANCY_FREE_RATIO 注释）
规则版已知局限（如实声明）：数字改写（"逾百家"）、区间、单位换算、中文数字均查不到；
只做字面核对，不判断语义等价。

口径 v3（score_answer_v3，2026-09-06）：72B 双协议实测暴露 v2 失效——表格题数值全错
（"0 家" vs 参考 4/107/50）仍拿 0.81–0.90，与正确题几无区分度（错题均 0.854 vs 正确题
均 0.89）。根因（QA 逐题手算坐实）：
(a) keyword 权重 0.6 过高——答案提到关键词即稳拿，与数值对错无关；
(b) ref_consistency 的全量数字抽取被"夹带数字"稀释——答案里的日期（2025/3/22）、
    字段赋值（delete_status=0）与 reference 同款数字撞上，错误主数字（0）的惩罚被分摊
    （id11 faithful 仅降到 0.75 而非 0）；
(c) faithful 权重 0.15 太低——即便扣到 0 也只掉 0.15。
v3 三处修复：
1. 权重重构：0.3*keyword + 0.2*semantic + 0.35*numeric + 0.15*faithful。
   数值对错（numeric）独立成项并成为最大权重信号。
2. 结论数字抽取去噪（extract_conclusion_numbers）：剥离行首编号、字段赋值（xxx=0）、
   日期成分（2025年/3月22日/时:分），只保留带数量单位（家/条/个/人/次/起…）的数字。
   evidence 比对仍用宽松 extract_numbers（证据是自证材料，宽松不引入方向性偏差）。
3. 硬封顶：双方结论数字均非空且完全无交集（数值错）时，total 封顶 ≤0.4；答案含
   中文数字+单位（"约五十家"，即已知局限的改写场景）时豁免封顶，交由 semantic 拉分。
v3 与 v1/v2 总分不可横比（权重与子项定义均不同）；历史 0.8667（R8 文本子集饱和口径）
自 v3 起弃用，全量分数以 v3 重算。
"""

from __future__ import annotations

import math
import re

from app.llm.client import LLMClient

KEYWORD_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3

# 口径 v2 权重（总和 1.0，满分仍为 1.0，历史报告口径见 eval/reports/）
V2_KEYWORD_WEIGHT = 0.6
V2_SEMANTIC_WEIGHT = 0.25
V2_FAITH_WEIGHT = 0.15

# 口径 v3 权重（总和 1.0；数值对错成为最大权重信号，见模块 docstring v3 段）
V3_KEYWORD_WEIGHT = 0.3
V3_SEMANTIC_WEIGHT = 0.2
V3_NUMERIC_WEIGHT = 0.35
V3_FAITH_WEIGHT = 0.15

# 结论数字完全无交集（数值错）时的总分封顶线。
# 校准依据：错题的 0.3*kw(=1) + 0.2*sem(≈0.75-0.9) 天然落在 0.45-0.48，
# 封顶 0.4 保证"数值错"与"数值对"（正常 ≥0.8）拉开 ≥0.4，彻底杜绝错答案拿高分。
V3_NUM_MISS_CAP = 0.4

# 数字抽取：阿拉伯整数/小数/百分数。中文数字与文字化数量（"逾百"）不处理（局限见 docstring）
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

# 冗余惩罚参数：要点行数/关键词命中数超过 FREE_RATIO 起罚，FULL_SPAN 打满。
# 校准依据（2026-09-06 实测）：两遍生成"宁可多抽"下，text 题要点数是关键词
# 标注数的 8-13 倍是常态（如 Q5 抽 39 要点），FREE_RATIO=2 会把全部正常列举型
# 答案打满、faith 维度失去区分度。免罚线放到常态上限之外，只罚极端失衡。
_REDUNDANCY_FREE_RATIO = 10.0
_REDUNDANCY_FULL_SPAN = 20.0

# 两遍生成要点行格式："N. 内容 [i]"（_extract_points 全局重编号）
_POINTS_LINE_RE = re.compile(r"^\s*\d+[.、)]\s", re.MULTILINE)

# 关键词别名词典：答案出现任一别名即视为命中该关键词。
# 动因：字面匹配的假阴性——Q8"吊装作业如何分级管理"的准确答案为
# "按照吊物质量分为一级/二级/三级"，语义完整却不含"分级"字样，实测被判 0.5 分。
# （2026-08-31 引入；引入前的历史报告不含此修正，对比时注意口径）
KEYWORD_ALIASES: dict[str, list[str]] = {
    "分级": ["一级", "二级", "三级", "特级"],
}


def _keyword_hit(answer: str, keyword: str) -> bool:
    if keyword in answer:
        return True
    return any(alias in answer for alias in KEYWORD_ALIASES.get(keyword, []))


def keyword_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.5  # 无关键词要求时给中性分，语义分主导
    hit = sum(1 for k in keywords if _keyword_hit(answer, k))
    return hit / len(keywords)


def semantic_score(answer: str, reference: str | None, llm: LLMClient) -> float:
    if not reference:
        return 0.5
    import numpy as np

    vecs = llm.embed_documents([answer[:2000], reference])
    a, b = np.asarray(vecs[0]), np.asarray(vecs[1])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return max(0.0, float(np.dot(a, b) / denom))


def score_answer(
    *,
    answer: str,
    refused: bool,
    keywords: list[str],
    reference: str | None,
    llm: LLMClient,
    expected_route: str | None = None,
) -> dict:
    # refuse 类判定显式化：优先用 expected_route 字段；
    # 兼容旧数据集（无该字段时退回隐式条件：无关键词且无参考描述）。
    if expected_route is not None:
        is_refuse_item = expected_route == "refuse"
    else:
        is_refuse_item = not keywords and reference is None
    if is_refuse_item:
        # refuse 类题目：关键词为空、无参考描述；API 标志或答案文本判定拒答均可
        text_refused = any(m in answer for m in ("无法回答", "未检索到", "无法给出", "资料库未"))
        effective = refused or text_refused
        return {"total": 1.0 if effective else 0.0, "keyword": 0.0, "semantic": 0.0, "refused": effective}
    kw = keyword_score(answer, keywords) if keywords else 0.0
    sem = semantic_score(answer, reference, llm) if keywords else 0.0
    total = KEYWORD_WEIGHT * kw + SEMANTIC_WEIGHT * sem
    if math.isnan(total):
        total = 0.0
    return {"total": round(total, 4), "keyword": round(kw, 4), "semantic": round(sem, 4), "refused": refused}


# ---------------- 口径 v2：事实一致性 + 冗余惩罚 ----------------


def extract_numbers(text: str) -> list[str]:
    """抽取阿拉伯数字 token（含小数与百分数），去重保序。

    先剥离要点行首编号（'N. '）——编号是排版产物而非语义数字，
    混入会稀释证据/参考一致性分数（正确答案的编号在证据里查不到）。
    """
    text = _POINTS_LINE_RE.sub("", text or "")
    seen: dict[str, None] = {}
    for m in _NUMBER_RE.findall(text):
        seen.setdefault(m, None)
    return list(seen)


def evidence_consistency(answer: str, evidence: str) -> float:
    """answer 中的数字在证据文本中出现的比例；无证据或答案无数字 → 1.0（中性不罚）。"""
    if not evidence:
        return 1.0
    nums = extract_numbers(answer)
    if not nums:
        return 1.0
    hit = sum(1 for n in nums if n in evidence)
    return hit / len(nums)


def ref_consistency(answer: str, reference: str | None) -> float:
    """answer 数字被 reference 数字覆盖的比例；reference 缺失、双方任一无数字 → 1.0。

    方向取 answer 为分子：答案中每个数字都应能在 reference 找到依据——
    编造的数字（如 Q15 错误答案的"0 家"）不在 reference（107/3）中，直接归零；
    而 reference 侧的额外数字（如 risk_grade=3）不会误伤只报了 107 的正确答案。
    """
    if not reference:
        return 1.0
    ref_nums = extract_numbers(reference)
    ans_nums = extract_numbers(answer)
    if not ref_nums or not ans_nums:
        return 1.0
    hit = sum(1 for n in ans_nums if n in ref_nums)
    return hit / len(ans_nums)


def redundancy_penalty(answer: str, keywords: list[str]) -> float:
    """冗余惩罚：要点行数 / max(关键词命中数, 1) 超过 2.0 后线性惩罚，3.0 打满。

    要点行按两遍生成格式（'N. '开头）识别；单遍答案无编号行 → 0（报告侧标注
    '未计冗余'）。防的是'宁可多抽'在覆盖型指标下刷分。
    """
    points = len(_POINTS_LINE_RE.findall(answer or ""))
    if points == 0:
        return 0.0
    hit = sum(1 for k in keywords if _keyword_hit(answer, k)) if keywords else 0
    ratio = points / max(hit, 1)
    if ratio <= _REDUNDANCY_FREE_RATIO:
        return 0.0
    span = _REDUNDANCY_FULL_SPAN - _REDUNDANCY_FREE_RATIO
    return min((ratio - _REDUNDANCY_FREE_RATIO) / span, 1.0)


def score_answer_v2(
    *,
    answer: str,
    refused: bool,
    keywords: list[str],
    reference: str | None,
    llm: LLMClient,
    expected_route: str | None = None,
    evidence: str = "",
) -> dict:
    """口径 v2 评分。公式与局限见模块 docstring；旧 score_answer 保留不动。"""
    # refuse 判定与 v1 完全一致
    if expected_route is not None:
        is_refuse_item = expected_route == "refuse"
    else:
        is_refuse_item = not keywords and reference is None
    if is_refuse_item:
        text_refused = any(m in answer for m in ("无法回答", "未检索到", "无法给出", "资料库未"))
        effective = refused or text_refused
        return {
            "total": 1.0 if effective else 0.0,
            "keyword": 0.0,
            "semantic": 0.0,
            "faithful": 0.0,  # refuse 类不适用事实维度
            "redundancy": 0.0,
            "refused": effective,
        }

    kw = keyword_score(answer, keywords) if keywords else 0.0
    sem = semantic_score(answer, reference, llm) if keywords else 0.0
    ev = evidence_consistency(answer, evidence)
    rc = ref_consistency(answer, reference)
    red = redundancy_penalty(answer, keywords)
    faith = ev * rc * (1.0 - red)
    total = V2_KEYWORD_WEIGHT * kw + V2_SEMANTIC_WEIGHT * sem + V2_FAITH_WEIGHT * faith
    if math.isnan(total):
        total = 0.0
    return {
        "total": round(total, 4),
        "keyword": round(kw, 4),
        "semantic": round(sem, 4),
        "faithful": round(faith, 4),
        "redundancy": round(red, 4),
        "refused": refused,
    }


# ---------------- 口径 v3：结论数字独立成项 + 去噪 + 硬封顶 ----------------

# 数量单位词：结论数字的判定特征（"N 家 / M 条"）。不含 万/亿/% ——
# "3万"会抽出 3 造成误判，百分数场景（text 类占比）由 v2 局限声明覆盖。
_QTY_UNIT = r"(?:家次|家|条|个|人|次|起|间|台|项|份|名|件|辆|栋|吨|起|批次|单)"

# 结论数字 = 紧跟数量单位的阿拉伯数字
_CONCLUSION_NUM_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*{_QTY_UNIT}")

# 去噪模式：这些数字是上下文成分而非结论，混入会稀释比对（v2 实证缺陷 b）。
# 顺序无关，逐个替换为空格防止相邻 token 粘连出伪数字。
_NOISE_PATTERNS = [
    _POINTS_LINE_RE,  # 行首编号 "N. "
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*[=＝]\s*\d+(?:\.\d+)?%?"),  # 字段赋值 delete_status=0
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*[=＝]\s*['\"]?\d+['\"]?"),  # 字段赋值（带引号/冒号变体）
    re.compile(r"\d{4}\s*年"),  # 年份 2025年
    re.compile(r"\d{1,2}\s*月(?:\d{1,2}\s*日)?"),  # 3月 / 3月22日
    re.compile(r"\d{1,2}\s*日"),  # 22日（未与月连写）
    re.compile(r"\d{1,2}\s*[:：点]\d{2}\s*分?"),  # 时刻 9:30 / 9点30分
]

# 中文数字 + 单位（"约五十家"）：已知局限的改写场景，命中则豁免硬封顶。
_CJK_QTY_RE = re.compile(r"[一二两三四五六七八九十百千零〇两]+\s*(?:家次|家|条|个|人|次|起|间|台|项|份|名|件|辆|栋|吨)")


def extract_conclusion_numbers(text: str) -> list[str]:
    """抽取"结论数字"：带数量单位的阿拉伯数字，先剥离日期/字段赋值/编号等上下文噪音。

    与 extract_numbers（宽松，供证据比对）的分工：
    - 结论数字用于 answer↔reference 的数值对错判定，必须严格——v2 实证日期
      （2025/3/22）与字段赋值（delete_status=0）会把错误主数字（0）的惩罚稀释掉；
    - 证据文本是系统自证材料，宽松比对不引入方向性偏差，维持原逻辑。
    去重保序。
    """
    t = text or ""
    for pat in _NOISE_PATTERNS:
        t = pat.sub(" ", t)
    seen: dict[str, None] = {}
    for m in _CONCLUSION_NUM_RE.findall(t):
        seen.setdefault(m, None)
    return list(seen)


def numeric_consistency(answer: str, reference: str | None) -> float:
    """结论数字一致性：answer 结论数字被 reference 结论数字覆盖的比例。

    中性规则与 v2 ref_consistency 一致：无 reference、或任一方抽不到结论数字
    （非数量型问题，如 text 类"时间范围是什么"）→ 1.0，不参与拉分。
    方向取 answer 为分子：答案的每个数量结论都应能在参考中找到依据。
    """
    if not reference:
        return 1.0
    ans_nums = extract_conclusion_numbers(answer)
    ref_nums = extract_conclusion_numbers(reference)
    if not ans_nums or not ref_nums:
        return 1.0
    hit = sum(1 for n in ans_nums if n in ref_nums)
    return hit / len(ans_nums)


def score_answer_v3(
    *,
    answer: str,
    refused: bool,
    keywords: list[str],
    reference: str | None,
    llm: LLMClient,
    expected_route: str | None = None,
    evidence: str = "",
) -> dict:
    """口径 v3 评分。公式、动机与局限见模块 docstring v3 段；v1/v2 保留不动。

    与 v2 的结构差异：faithful 拆为两个独立子项——
    - numeric（0.35）：结论数字一致性（去噪后比对，数值对错的主信号）；
    - faithful（0.15）：evidence_consistency × (1-冗余惩罚)（证据自洽与防刷分）。
    """
    # refuse 判定与 v1/v2 完全一致
    if expected_route is not None:
        is_refuse_item = expected_route == "refuse"
    else:
        is_refuse_item = not keywords and reference is None
    if is_refuse_item:
        text_refused = any(m in answer for m in ("无法回答", "未检索到", "无法给出", "资料库未"))
        effective = refused or text_refused
        return {
            "total": 1.0 if effective else 0.0,
            "keyword": 0.0,
            "semantic": 0.0,
            "numeric": 0.0,  # refuse 类不适用数值维度
            "faithful": 0.0,
            "redundancy": 0.0,
            "refused": effective,
        }

    kw = keyword_score(answer, keywords) if keywords else 0.0
    sem = semantic_score(answer, reference, llm) if keywords else 0.0
    ev = evidence_consistency(answer, evidence)
    red = redundancy_penalty(answer, keywords)
    faith = ev * (1.0 - red)
    num = numeric_consistency(answer, reference)
    total = (
        V3_KEYWORD_WEIGHT * kw
        + V3_SEMANTIC_WEIGHT * sem
        + V3_NUMERIC_WEIGHT * num
        + V3_FAITH_WEIGHT * faith
    )

    # 硬封顶：双方都有结论数字但完全无交集 = 数值错（v2 的核心失效场景）。
    # 中文数字改写（"约五十家"）豁免——这是规则版已知局限，误封顶比漏封顶伤害更大。
    ans_nums = extract_conclusion_numbers(answer)
    ref_nums = extract_conclusion_numbers(reference) if reference else []
    if (
        ans_nums
        and ref_nums
        and not set(ans_nums) & set(ref_nums)
        and not _CJK_QTY_RE.search(answer or "")
    ):
        total = min(total, V3_NUM_MISS_CAP)

    if math.isnan(total):
        total = 0.0
    return {
        "total": round(total, 4),
        "keyword": round(kw, 4),
        "semantic": round(sem, 4),
        "numeric": round(num, 4),
        "faithful": round(faith, 4),
        "redundancy": round(red, 4),
        "refused": refused,
    }
