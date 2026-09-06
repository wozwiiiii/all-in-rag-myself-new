"""答案生成：仅基于检索上下文作答、带来源引用、无可靠证据时明确拒答。

应急安全场景的硬要求：宁可说"资料中未找到"，不可编造监管结论。

两遍生成（2026-09-01 引入，generation_two_pass 可关）：
- 第一遍（map）：逐 chunk 并行要点抽取。单遍/全量抽取在 ~7k 字符上下文上
  存在注意力稀释（实测丢"感知"/"监护"），拆到单 chunk 后任务平凡化。
- 第二遍（组装）：要点清单**程序化拼接为答案主体**（零丢失——抽取到的
  要点必然出现在答案中，关键词覆盖由抽取轮决定，确定性保证）；LLM 仅生成
  ≤150 字结论段（输出极小 → 延迟有界，实测 72B 长答案 ~9 字/秒是重组瓶颈）。
- 抽取全失败退化为单遍生成（原行为）。
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.rag.retriever import RetrievedChunk

logger = get_logger(__name__)

# 答案中出现这些表述视为模型自行拒答（检索分过阈值但上下文实际无关时会出现）
_REFUSAL_MARKERS = ("无法回答", "未检索到", "无法给出", "未提及", "资料不足")

REFUSAL_ANSWER = (
    "根据现有资料库未检索到与该问题直接相关的内容，无法给出可靠答案。"
    "建议补充相关法规/标准文档，或换一种问法（例如指明具体法规、工艺或企业类型）。"
)

_EXTRACT_SYSTEM_PROMPT = """你是资料要点抽取器。从下面这一份【参考资料】中抽取与【问题】相关的所有要点。
规则：
1. 完备性优先：宁可多抽不可遗漏——任何与问题相关的条款、要求、措施、数据、结论、
   配套基础设施与人员配置要求都要列出（即使只是部分相关）。
2. 每条要点一行，格式：编号. 要点内容（≤60 字，保留关键术语原文）
3. 这份资料与问题完全无关时，只输出：无
4. 只输出要点列表，不要总结、不要回答问题。"""

_SUMMARY_SYSTEM_PROMPT = """你是资料要点总结器。仅依据下面的【已抽取要点】，用 2-3 句话概括回答【问题】。
规则：
1. 只输出概括文字（≤150 字），不要罗列要点、不要重复要点原文。
2. 要点不足以回答问题时，明确说明"根据现有资料无法回答该问题"。
3. 不要使用资料之外的知识。"""

_SYSTEM_PROMPT = """你是一名应急管理、安全生产领域的专业问答助手。请严格遵守：
1. 只依据下面提供的【参考资料】回答，不得使用资料之外的知识。
2. 回答末尾用一行列出引用的资料编号，格式：来源: [1] 或 来源: [1][2]。
3. 资料不足以回答时，明确说明"根据现有资料无法回答该问题"，不要猜测。
4. 用中文回答，先给结论，再给依据。
5. 若问题为列举类（措施/要求/分级/内容/步骤等），按条覆盖参考资料中所有相关要点，
   每条一行精简表述并标注资料编号，不可只答其中一部分。
6. 答案总长控制在 800 字以内，宁简勿缺条目。"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] 来源: {c.source}"
        if c.heading:
            header += f" | {c.heading}"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def _extract_points(question: str, chunks: list[RetrievedChunk], llm: LLMClient) -> str:
    """第一遍（map）：逐 chunk 并行要点抽取。

    单路失败返回该路空串，不阻断其余路；全空返回 ""（上层退化单遍）。
    返回值已做：行级来源标注（行尾 [i]）+ 全局重编号。
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(i: int, c: RetrievedChunk) -> list[str]:
        user_prompt = f"【参考资料】\n[{i}] {c.text}\n\n【问题】\n{question}"
        try:
            msg = llm.chat_safe(
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = (msg.content or "").strip()
        except Exception as e:
            logger.warning("points_extraction_chunk_failed", extra={"chunk": i, "err": str(e)[:200]})
            return []
        if not content or content.strip() in ("无", "无相关要点"):
            return []
        lines = []
        for ln in content.split("\n"):
            ln = ln.strip()
            if not ln or ln == "无":
                continue
            lines.append(f"{ln} [{i}]")  # 行级来源标注
        return lines

    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
        per_chunk = list(ex.map(lambda ic: _one(*ic), enumerate(chunks, 1)))

    all_lines = [ln for lines in per_chunk for ln in lines]
    # 全局重编号：剥离模型各路局部编号，统一 1..N
    import re

    renumbered = []
    no = 0
    for ln in all_lines:
        no += 1
        stripped = re.sub(r"^\d+[.、)]\s*", "", ln)
        renumbered.append(f"{no}. {stripped}")
    return "\n".join(renumbered)


def _summarize_points(question: str, points: str, llm: LLMClient) -> str:
    """第二遍（结论段）：仅生成 ≤150 字概括。失败返回空串（答案退化为纯要点列表）。"""
    user_prompt = f"【已抽取要点】\n{points}\n\n【问题】\n{question}"
    try:
        msg = llm.chat_safe(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return (msg.content or "").strip()
    except Exception as e:
        logger.warning("points_summary_failed", extra={"err": str(e)[:200]})
        return ""


def _sources_payload(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "source": c.source,
            "heading": c.heading,
            "score": c.score,
            **({"rerank_score": c.rerank_score} if c.rerank_score is not None else {}),
        }
        for c in chunks
    ]


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    llm: LLMClient,
    *,
    two_pass: bool = True,
) -> tuple[str, list[dict], bool]:
    """返回 (answer, sources, refused)。chunks 为空时直接拒答，不调用 LLM。"""
    if not chunks:
        logger.info("rag_refuse_no_evidence", extra={"q_len": len(question)})
        return REFUSAL_ANSWER, [], True

    context = _build_context(chunks)

    if two_pass:
        points = _extract_points(question, chunks, llm)
        if points:
            summary = _summarize_points(question, points, llm)
            answer = f"{summary}\n\n{points}" if summary else points
            logger.info(
                "two_pass_answer",
                extra={"points": points.count("\n") + 1, "summary_len": len(summary)},
            )
            # 拒答判定限定在答案开头：避免正文自然出现"未提及"等词被误标拒答
            head = answer[:60]
            return answer, _sources_payload(chunks), any(m in head for m in _REFUSAL_MARKERS)

    # 单遍生成（two_pass=False 或抽取全失败）
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"
    msg = llm.chat_safe(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )
    answer = (msg.content or "").strip()
    head = answer[:60]
    model_refused = any(m in head for m in _REFUSAL_MARKERS)
    return answer, _sources_payload(chunks), model_refused
