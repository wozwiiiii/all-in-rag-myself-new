"""API 合约测试：认证、限流、请求校验、健康探针（pipeline 用假实现，不打真实 LLM）。"""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.routes import probe_router, qa_router
from app.api.security import RateLimiter, make_api_key_dependency
from app.core.config import Settings
from app.main import add_request_context, register_error_handlers
from app.orchestrator.pipeline import QAResult


class FakePipeline:
    async def answer(self, question: str) -> QAResult:
        return QAResult(answer=f"echo:{question}", route="text", sources=[], refused=False, latency_ms=1)


def make_app(api_keys: str, rate_limit: int = 100) -> FastAPI:
    settings = Settings(_env_file=None, api_keys=api_keys, rate_limit_per_minute=rate_limit)
    app = FastAPI()
    app.state.pipeline = FakePipeline()
    app.state.settings = settings
    limiter = RateLimiter(settings.rate_limit_per_minute)
    verify = make_api_key_dependency(settings, limiter)
    register_error_handlers(app)
    add_request_context(app)
    app.include_router(probe_router, prefix="/api/v1")
    app.include_router(qa_router, prefix="/api/v1", dependencies=[Depends(verify)])
    return app


HDRS = {"X-API-Key": "test-key-123"}


def test_qa_requires_api_key():
    client = TestClient(make_app("test-key-123"))
    r = client.post("/api/v1/qa", json={"question": "动火作业要求？"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_qa_rejects_wrong_key():
    client = TestClient(make_app("test-key-123"))
    r = client.post("/api/v1/qa", json={"question": "动火作业要求？"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_qa_ok_returns_contract():
    client = TestClient(make_app("test-key-123"))
    r = client.post("/api/v1/qa", json={"question": "动火作业要求？"}, headers=HDRS)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo:动火作业要求？"
    assert body["route"] == "text"
    assert body["latency_ms"] >= 0
    assert body["request_id"]
    assert r.headers.get("X-Request-ID")  # request_id 由 qa 路由生成，响应头由中间件回传


def test_qa_validates_question_length():
    client = TestClient(make_app("test-key-123"))
    assert client.post("/api/v1/qa", json={"question": ""}, headers=HDRS).status_code == 422
    assert client.post("/api/v1/qa", json={"question": "x" * 501}, headers=HDRS).status_code == 422


def test_rate_limit_returns_429():
    client = TestClient(make_app("test-key-123", rate_limit=2))
    assert client.post("/api/v1/qa", json={"question": "问题一"}, headers=HDRS).status_code == 200
    assert client.post("/api/v1/qa", json={"question": "问题二"}, headers=HDRS).status_code == 200
    r = client.post("/api/v1/qa", json={"question": "问题三"}, headers=HDRS)
    assert r.status_code == 429
    assert "限" in r.json()["error"]["message"]


def test_health_open_without_auth():
    client = TestClient(make_app("test-key-123"))
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_missing_index(tmp_path):
    app = make_app("test-key-123")
    # 指向空目录：索引与表数据均不存在
    app.state.settings = Settings(
        _env_file=None,
        api_keys="test-key-123",
        chroma_dir=tmp_path / "chroma",
        parquet_dir=tmp_path / "parquet",
        openai_api_key="sk-test",
    )
    client = TestClient(app)
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["vector_index"] is False
    assert body["checks"]["table_data"] is False
    assert body["checks"]["llm_configured"] is True
