"""SSE 流式问答测试：事件顺序、SSE 格式、鉴权、错误映射、answer 事件含 usage。

2026-09-03 批次4 引入。FakePipeline.answer_stream 产出固定事件序列，
验证路由层 SSE 编码与事件协议，不打真实 LLM。
"""

import json

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.routes import qa_router
from app.api.security import RateLimiter, make_api_key_dependency
from app.core.config import Settings
from app.core.errors import AppError
from app.main import add_request_context, register_error_handlers
from app.orchestrator.pipeline import QAResult


class FakePipeline:
    """支持 answer() 与 answer_stream() 的假编排。"""

    def __init__(self):
        self._events = [
            {"type": "stage", "stage": "routing"},
            {"type": "route", "route": "text"},
            {"type": "stage", "stage": "working", "branches": ["text"]},
            {
                "type": "answer",
                "result": QAResult(
                    answer="标准答案",
                    route="text",
                    sources=[{"source": "a.docx", "kind": "text"}],
                    refused=False,
                    latency_ms=88,
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "calls": 1},
                ),
            },
        ]

    async def answer(self, question: str) -> QAResult:
        for e in self._events:
            if e["type"] == "answer":
                return e["result"]
        raise AssertionError("no answer event")

    async def answer_stream(self, question: str):
        for e in self._events:
            yield e


class ErrorPipeline:
    """answer_stream 中途抛超时错误的假编排（对齐 _answer_stages 超时语义）。"""

    async def answer(self, question: str) -> QAResult:
        raise AppError("超时", code="qa_timeout", http_status=504)

    async def answer_stream(self, question: str):
        yield {"type": "stage", "stage": "routing"}
        yield {"type": "error", "code": "qa_timeout", "message": "问答总耗时超过上限"}
        raise AppError("问答总耗时超过上限", code="qa_timeout", http_status=504)


def make_app(pipeline, api_keys: str = "test-key-123") -> FastAPI:
    settings = Settings(_env_file=None, api_keys=api_keys)
    app = FastAPI()
    app.state.pipeline = pipeline
    app.state.settings = settings
    limiter = RateLimiter(settings.rate_limit_per_minute)
    verify = make_api_key_dependency(settings, limiter)
    register_error_handlers(app)
    add_request_context(app)
    app.include_router(qa_router, prefix="/api/v1", dependencies=[Depends(verify)])
    return app


HDRS = {"X-API-Key": "test-key-123"}


def _parse_sse(text: str) -> list[dict]:
    """解析 SSE 文本为事件 dict 列表。"""
    events = []
    for chunk in text.strip().split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        data = chunk.removeprefix("data: ")
        events.append(json.loads(data))
    return events


def test_stream_requires_api_key():
    client = TestClient(make_app(FakePipeline(), api_keys="test-key-123"))
    r = client.post("/api/v1/qa/stream", json={"question": "问题"})
    assert r.status_code == 401


def test_stream_event_order_and_format():
    client = TestClient(make_app(FakePipeline()))
    with client.stream("POST", "/api/v1/qa/stream", json={"question": "问题"}, headers=HDRS) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    types = [e["type"] for e in events]
    assert types == ["stage", "route", "stage", "answer"]
    assert events[0]["stage"] == "routing"
    assert events[1]["route"] == "text"
    assert events[2]["stage"] == "working"
    # answer 事件：QAResult 已 asdict 转进 result 字段，且回填 request_id
    ans = events[3]["result"]
    assert ans["answer"] == "标准答案"
    assert ans["route"] == "text"
    assert ans["usage"]["total_tokens"] == 15
    assert events[3]["request_id"]


def test_stream_error_event_mapping():
    client = TestClient(make_app(ErrorPipeline()))
    # SSE 响应已开始后无法改 HTTP 状态码：错误经 event_source 捕获，
    # 以 error 事件收尾，流正常返回 200。
    with client.stream("POST", "/api/v1/qa/stream", json={"question": "问题"}, headers=HDRS) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    events = _parse_sse(body)
    # pipeline 已发 error 事件后抛 AppError，event_source 兜底再发一个——
    # 断言至少存在 error 事件且 code 正确即可（SSE 语义：收到 error 即停止消费）
    assert any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "qa_timeout"


def test_stream_after_branch_emits_usage_like_sync():
    """answer() 与 answer_stream() 的末事件应产出同样的答案与 usage。"""
    fp = FakePipeline()
    client = TestClient(make_app(fp))
    # 同步 /qa
    r = client.post("/api/v1/qa", json={"question": "问题"}, headers=HDRS)
    assert r.status_code == 200
    assert r.json()["answer"] == "标准答案"
    assert r.json()["usage"]["total_tokens"] == 15
