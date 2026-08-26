"""评测指标：对齐竞赛评分公式的可执行近似。

score = 0.7 * keyword_score + 0.3 * semantic_score
- keyword_score：期望关键词命中率（竞赛为带权，评测集简化为均权）
- semantic_score：答案与参考描述的 embedding 余弦相似度（参考描述缺失时记 0.5 中性分）
refuse 类：系统明确拒答得 1 分，编造答案得 0 分。
"""

from __future__ import annotations

import math

from app.llm.client import LLMClient

KEYWORD_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3


def keyword_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.5  # 无关键词要求时给中性分，语义分主导
    hit = sum(1 for k in keywords if k in answer)
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
    *, answer: str, refused: bool, keywords: list[str], reference: str | None, llm: LLMClient
) -> dict:
    if not keywords and reference is None:
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
