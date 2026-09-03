"""评测指标：对齐竞赛评分公式的可执行近似。

score = 0.7 * keyword_score + 0.3 * semantic_score
- keyword_score：期望关键词命中率（竞赛为带权，评测集简化为均权）
- semantic_score：答案与参考描述的 embedding 余弦相似度（参考描述缺失时记 0.5 中性分）
refuse 类：系统明确拒答得 1 分，编造答案得 0 分。
判定来源：dataset 的 expected_route == "refuse"（显式），
或旧数据集的隐式条件（无关键词且无参考描述）。
"""

from __future__ import annotations

import math

from app.llm.client import LLMClient

KEYWORD_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3

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
