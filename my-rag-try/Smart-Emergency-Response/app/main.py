"""FastAPI 应用入口。

启动：uvicorn app.main:app --host 0.0.0.0 --port 8000
装配职责：日志、配置校验、认证与限流中间件、request_id 贯穿、
统一错误体（不向外泄漏堆栈）、lifespan 内完成 parquet 预转换与组件装配。
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import probe_router, qa_router
from app.api.security import RateLimiter, make_api_key_dependency
from app.core.config import Settings, get_settings
from app.core.errors import AppError, ConfigError
from app.core.logging import get_logger, request_id_var, setup_logging
from app.llm.client import LLMClient
from app.orchestrator.pipeline import QAPipeline
from app.tabular.loader import convert_excel_to_parquet

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    app.state.settings = settings

    if not settings.openai_api_key:
        logger.error("missing_openai_api_key")
        raise ConfigError(
            "未配置 OPENAI_API_KEY：所有问答路径均依赖 LLM，无法启动。"
            "请复制 .env.example 为 .env 并填写。"
        )

    # Excel -> parquet 预转换（幂等，已有且较新则跳过）
    try:
        convert_excel_to_parquet(settings)
    except Exception as e:
        logger.error("parquet_prepare_failed", extra={"err": str(e)})

    llm = LLMClient(settings)
    app.state.pipeline = QAPipeline(settings, llm)
    logger.info(
        "service_started",
        extra={
            "model": settings.openai_api_model,
            "auth_enabled": settings.auth_enabled,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
        },
    )
    yield
    logger.info("service_stopped")


def add_request_context(app: FastAPI) -> None:
    """每请求生成/透传 X-Request-ID，并注入日志上下文，实现全链路追踪。"""

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response


def register_error_handlers(app: FastAPI) -> None:
    """统一错误体：AppError 映射业务码；未知异常 500 且不泄漏堆栈（完整堆栈只进日志）。"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning("app_error", extra={"code": exc.code, "status": exc.http_status})
        headers = {}
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict(), headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "服务内部错误，请稍后重试"}},
        )


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()

    app = FastAPI(
        title="Smart Emergency QA",
        description="应急管理安全生产智能问答（RAG + Text-to-Pandas，安全沙箱执行）",
        version="0.1.0",
        lifespan=lifespan,
    )

    limiter = RateLimiter(settings.rate_limit_per_minute)
    verify_api_key = make_api_key_dependency(settings, limiter)

    add_request_context(app)
    register_error_handlers(app)

    # 探针免认证（K8s 惯例），业务接口带认证 + 限流
    app.include_router(probe_router, tags=["probes"], prefix="/api/v1")
    app.include_router(
        qa_router,
        tags=["qa"],
        prefix="/api/v1",
        dependencies=[Depends(verify_api_key)],
    )
    return app


app = create_app()
