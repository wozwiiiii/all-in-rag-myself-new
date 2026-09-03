"""业务路由：问答接口（同步 + SSE 流式）+ 存活/就绪探针。

认证依赖由 main.py 在 include_router 时注入（认证配置属于应用装配层）；
pipeline 通过 app.state 注入，便于测试时替换。
"""

from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    HealthResponse,
    QARequest,
    QAResponse,
    QASource,
    ReadyResponse,
    UsageMeta,
)
from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)
qa_router = APIRouter()
probe_router = APIRouter()

VERSION = "0.1.0"


@qa_router.post("/qa", response_model=QAResponse)
async def qa(payload: QARequest, request: Request) -> QAResponse:
    pipeline = request.app.state.pipeline
    # request_id 由 main.py 中间件统一注入
    rid = request_id_var.get()
    result = await pipeline.answer(payload.question)
    return QAResponse(
        answer=result.answer,
        route=result.route,
        sources=[QASource(**s) for s in result.sources],
        refused=result.refused,
        latency_ms=result.latency_ms,
        request_id=rid,
        usage=UsageMeta(**result.usage) if result.usage else None,
    )


def _sse(event: dict) -> str:
    """SSE 单事件编码：`data: <json>\n\n`。事件内的 result 若是 QAResult 先转 dict。"""
    payload = event
    if event.get("type") == "answer" and hasattr(event.get("result"), "__dataclass_fields__"):
        payload = {**event, "result": asdict(event["result"])}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@qa_router.post("/qa/stream")
async def qa_stream(payload: QARequest, request: Request):
    """SSE 流式问答：逐阶段推送事件（routing/route/working/answer 或 error）。

    认证/限流随 qa_router 依赖自动生效；事件协议见 pipeline._answer_stages docstring。
    """
    pipeline = request.app.state.pipeline

    async def event_source():
        rid = request_id_var.get()
        try:
            async for event in pipeline.answer_stream(payload.question):
                if event.get("type") == "answer":
                    event["request_id"] = rid
                yield _sse(event)
        except Exception as e:  # noqa: BLE001 - SSE 响应已开始，无法改状态码，
            # 以 error 事件收尾而非抛给全局 handler（否则流被截断为 500）
            code = getattr(e, "code", "internal_error")
            status = getattr(e, "http_status", 500)
            yield _sse(
                {
                    "type": "error",
                    "code": code,
                    "http_status": status,
                    "message": str(e)[:200],
                }
            )

    return StreamingResponse(event_source(), media_type="text/event-stream")


@probe_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """存活探针（liveness）：进程活着即返回，不做任何依赖检查。"""
    return HealthResponse(status="ok", version=VERSION)


@probe_router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    """就绪探针（readiness）：检查向量索引、表数据、LLM 配置是否就位。"""
    settings = request.app.state.settings
    checks = {
        "llm_configured": bool(settings.openai_api_key),
        "vector_index": (settings.chroma_dir / "chroma.sqlite3").exists(),
        "table_data": any(settings.parquet_dir.glob("*.parquet")),
    }
    ok = all(checks.values())
    return ReadyResponse(status="ok" if ok else "not_ready", checks=checks)
