"""业务路由：问答接口 + 存活/就绪探针。

认证依赖由 main.py 在 include_router 时注入（认证配置属于应用装配层）；
pipeline 通过 app.state 注入，便于测试时替换。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.schemas import (
    HealthResponse,
    QARequest,
    QAResponse,
    QASource,
    ReadyResponse,
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
    )


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
