"""问答编排：路由分发 -> 各分支执行（hybrid 并行）-> 融合 -> 统一结果。

对外暴露唯一的异步入口 `answer()`，API 层只与它交互。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.errors import AppError, RetrievalUnavailable
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.orchestrator.guard import LEAK_SAFE_ANSWER, check_prompt_leak
from app.orchestrator.router import IntentRouter
from app.rag.generator import generate_answer
from app.rag.retriever import RetrievedChunk, Retriever
from app.tabular.agent import TabularAgent

logger = get_logger(__name__)


@dataclass
class QAResult:
    answer: str
    route: str = ""  # 由 answer() 统一填充分支结果
    sources: list[dict] = field(default_factory=list)
    refused: bool = False
    latency_ms: int = 0
    detail: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)  # 本次请求 token 用量（由 answer() 填充）


_FUSION_PROMPT = """你是综合答复助手。下面是对同一问题的两路调查结果：
【数据统计结果（来自结构化表，数字以此为准）】
{table_answer}

【政策/文档解读（来自法规库检索）】
{text_answer}

请合并为一份最终答案：先给结论（数据部分以统计结果为准），
再补充政策依据；去除两路结果的重复与冲突表述；中文回答，800 字以内。"""


class QAPipeline:
    def __init__(self, settings: Settings, llm: LLMClient | None = None):
        self._s = settings
        self._llm = llm or LLMClient(settings)
        self._router = IntentRouter(self._llm)
        self._agent = TabularAgent(settings, self._llm)
        self._retriever: Retriever | None = None

    # 懒加载：索引未构建时不阻断服务启动（table 类问题仍可用）
    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            try:
                self._retriever = Retriever(self._s, self._llm)
            except RuntimeError as e:
                raise RetrievalUnavailable(str(e)) from e
        return self._retriever

    async def answer(self, question: str) -> QAResult:
        """同步返回完整答案（API /qa 用）。消费 _answer_stages 取末事件。"""
        result: QAResult | None = None
        async for event in self._answer_stages(question):
            if event["type"] == "answer":
                result = event["result"]
        if result is None:
            raise AppError("问答流程未产出结果", code="qa_no_result", http_status=500)
        return result

    async def answer_stream(self, question: str):
        """阶段事件流（API /qa/stream 用），逐事件透传。"""
        async for event in self._answer_stages(question):
            yield event

    async def _answer_stages(self, question: str):
        """问答编排为阶段事件流。

        事件协议（dict）：
          {"type":"stage","stage":"routing"}     路由判定开始
          {"type":"route","route":<text|table|hybrid>}  已判定路由
          {"type":"stage","stage":"working","branches":[...]}  分支执行中
          {"type":"answer", **QAResult 字段}     最终答案
          {"type":"error","code":...,"message":...}  致命错误（随后停止）
        说明：分支（检索/生成/表分析）是单次阻塞调用，无法在内部细分
        检索/生成阶段，故执行期统一发 working 阶段事件。
        """
        start = time.monotonic()
        deadline = self._s.qa_total_timeout_s
        snap_src = getattr(self._llm, "usage_snapshot", None)
        usage_before = snap_src() if snap_src else {}

        yield {"type": "stage", "stage": "routing"}
        try:
            route = await asyncio.wait_for(
                asyncio.to_thread(self._router.route, question),
                timeout=deadline,
            )
        except asyncio.TimeoutError:
            route = "hybrid"
        logger.info("route_decided", extra={"route": route})
        yield {"type": "route", "route": route}

        if route == "text":
            branches = ["text"]
        elif route == "table":
            branches = ["table"]
        else:
            branches = ["text", "table"]
        yield {"type": "stage", "stage": "working", "branches": branches}

        try:
            if route == "text":
                result = await asyncio.wait_for(self._answer_text(question), timeout=deadline)
            elif route == "table":
                result = await asyncio.wait_for(self._answer_table(question), timeout=deadline)
            else:
                result = await asyncio.wait_for(self._answer_hybrid(question), timeout=deadline)
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "code": "qa_timeout",
                "message": f"问答总耗时超过 {deadline}s 上限，已终止。请缩小问题范围后重试。",
            }
            raise AppError(
                f"问答总耗时超过 {deadline}s 上限，已终止。请缩小问题范围后重试。",
                code="qa_timeout",
                http_status=504,
            )

        result.route = route
        result.latency_ms = int((time.monotonic() - start) * 1000)
        delta_src = getattr(self._llm, "usage_delta", None)
        if delta_src and usage_before:
            result.usage = delta_src(usage_before)

        # 输出侧泄露检测（覆盖全部路由）
        leaked = check_prompt_leak(result.answer)
        if leaked:
            logger.warning(
                "prompt_leak_detected",
                extra={"route": route, "signature": leaked, "latency_ms": result.latency_ms},
            )
            result.answer = LEAK_SAFE_ANSWER
            result.detail["prompt_leak"] = True

        logger.info(
            "qa_done",
            extra={
                "route": route,
                "latency_ms": result.latency_ms,
                "refused": result.refused,
                "usage": result.usage,
            },
        )
        yield {"type": "answer", "result": result}

    # ---------------- 分支实现 ----------------

    async def _answer_text(self, question: str) -> QAResult:
        retriever = self._get_retriever()
        chunks: list[RetrievedChunk] = await asyncio.to_thread(retriever.retrieve, question)
        answer, sources, refused = await asyncio.to_thread(
            generate_answer, question, chunks, self._llm, two_pass=self._s.generation_two_pass
        )
        return QAResult(answer=answer, sources=sources, refused=refused, detail={"branch": "text"})

    async def _answer_table(self, question: str) -> QAResult:
        tab = await asyncio.to_thread(self._agent.run, question)
        sources = [{"source": f"表:{t}", "kind": "table"} for t in tab.tables_used]
        return QAResult(
            answer=tab.answer,
            sources=sources,
            refused=False,
            detail={"branch": "table", "iterations": tab.iterations, "converged": tab.converged},
        )

    async def _answer_hybrid(self, question: str) -> QAResult:
        text_task = self._answer_text(question)
        table_task = self._answer_table(question)
        text_res, table_res = await _gather_first_error(text_task, table_task)
        # 单路失败不阻断：另一路结果照常返回
        if text_res is None or table_res is None:
            survivor = text_res or table_res
            assert survivor is not None
            survivor.detail["branch"] = "hybrid-partial"
            survivor.detail["degraded"] = True
            return survivor

        # 两路都成功：融合
        if table_res.refused or text_res.refused:
            # 出现拒答时以非拒答路为准，避免把"无法回答"融进最终答案
            primary = table_res if not table_res.refused else text_res
            primary.detail["branch"] = "hybrid-single"
            return primary

        # 融合改用 chat_safe + 降级：两路结果已得，融合失败不应让整条请求 500，
        # 退化用表格路答案（数据部分本就以表格路为准）。
        fusion_degraded = False
        try:
            fused = await asyncio.to_thread(
                self._llm.chat_safe,
                [
                    {
                        "role": "user",
                        "content": _FUSION_PROMPT.format(
                            table_answer=table_res.answer, text_answer=text_res.answer
                        ),
                    }
                ],
            )
            fused_text = (fused.content or "").strip() or table_res.answer
        except AppError as e:
            logger.warning("hybrid_fusion_failed_degrade", extra={"err": str(e)[:200]})
            fused_text = table_res.answer
            fusion_degraded = True
        sources = table_res.sources + text_res.sources
        return QAResult(
            answer=fused_text,
            sources=sources,
            refused=False,
            detail={"branch": "hybrid", "fusion_degraded": fusion_degraded},
        )


async def _gather_first_error(t1, t2):
    """gather 的容错版：单路失败（含 LLM SDK 原始异常）返回 None，两路都挂才抛。"""

    async def guard(coro):
        try:
            return await coro
        except Exception as e:  # noqa: BLE001 - 降级保护必须覆盖 SDK 原始异常
            logger.warning("hybrid_branch_failed", extra={"err": str(e)[:200], "type": type(e).__name__})
            return None

    r1, r2 = await asyncio.gather(guard(t1), guard(t2))
    if r1 is None and r2 is None:
        raise AppError("两条问答通路均失败，请稍后重试", code="all_branches_failed", http_status=502)
    return r1, r2
