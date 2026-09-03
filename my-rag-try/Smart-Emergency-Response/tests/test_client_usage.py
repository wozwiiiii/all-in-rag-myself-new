"""Token usage 可观测测试：累加/快照/差值/并发计数不丢，pipeline 与 API 契约。

2026-09-03 批次2 引入。方案 C：client 实例级线程安全累加器 + 请求级快照差值，
零调用方签名改动，规避 contextvar 在 ThreadPoolExecutor 并行抽取中被 copy-context 阻断的问题。
"""

import threading

from app.core.config import Settings
from app.llm.client import LLMClient
from app.orchestrator.pipeline import QAPipeline, QAResult


class _Msg:
    content = "ok"


class _Choice:
    message = _Msg()


class _Resp:
    def __init__(self, usage):
        self.usage = usage
        self.choices = [_Choice()]


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


def _make_client(monkeypatch) -> LLMClient:
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    client = LLMClient(settings)
    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        lambda **kw: _Resp(_Usage(100, 50)),
    )
    return client


def test_usage_accumulate_and_snapshot(monkeypatch):
    client = _make_client(monkeypatch)
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "hi2"}])
    snap = client.usage_snapshot()
    assert snap["prompt_tokens"] == 200
    assert snap["completion_tokens"] == 100
    assert snap["total_tokens"] == 300
    assert snap["calls"] == 2


def test_usage_delta_between_snapshots(monkeypatch):
    client = _make_client(monkeypatch)
    before = client.usage_snapshot()  # 0
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "hi2"}])
    delta = client.usage_delta(before)
    assert delta == {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300, "calls": 2}
    # 再取一次差值（无新增调用）应为 0
    assert client.usage_delta(client.usage_snapshot())["calls"] == 0


def test_usage_concurrent_no_loss(monkeypatch):
    """多线程并发调用，计数不丢（Lock 保证线程安全）。"""
    client = _make_client(monkeypatch)
    errors = []

    def worker():
        try:
            for _ in range(20):
                client.chat([{"role": "user", "content": "x"}])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    snap = client.usage_snapshot()
    assert snap["calls"] == 100  # 5 线程 × 20 次
    assert snap["prompt_tokens"] == 100 * 100


def test_pipeline_fills_usage_with_real_llm_like(monkeypatch):
    """pipeline.answer 对具备 usage 能力的 llm 应填充 usage。

    路由含 LLM 调用（content 固定 "ok" 使意图无法解析 → 触发启发式兜底），
    但只要走 answer() 入口，路由/分支任一 LLM 调用都会被快照差值计入。
    """
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    client = _make_client(monkeypatch)  # 真实 LLMClient（有 usage_snapshot/delta）
    pipe = QAPipeline(settings, client)

    async def fake_text(q):
        return QAResult(answer="t", refused=False, detail={"branch": "text"})

    async def fake_table(q):
        return QAResult(answer="t", refused=False, detail={"branch": "table"})

    pipe._answer_text = fake_text  # type: ignore[assignment]
    pipe._answer_table = fake_table  # type: ignore[assignment]

    import asyncio

    result = asyncio.run(pipe.answer("一般风险企业数量"))
    # 路由阶段至少有一次 chat 调用（计入 usage）；分支 stub 不产生额外调用
    assert result.usage["calls"] >= 1
    assert result.usage["total_tokens"] >= 150  # 路由 1 次：100+50
    assert "usage" in result.detail or result.usage  # usage 已随 QAResult 返回


def test_pipeline_tolerates_stub_llm_without_usage():
    """无 usage 能力的 stub llm（旧测试替身）不炸，usage 保持空 dict。"""
    settings = Settings(_env_file=None, openai_api_key="sk-test")

    class _Stub:
        def chat_safe(self, messages, **kw):
            class M:
                content = "x"

            return M()

        def chat(self, messages, **kw):
            class M:
                content = "x"

            return M()

    pipe = QAPipeline(settings, _Stub())  # type: ignore[arg-type]

    async def fake_text(q):
        return QAResult(answer="t", refused=False, detail={"branch": "text"})

    async def fake_table(q):
        return QAResult(answer="t", refused=False, detail={"branch": "table"})

    pipe._answer_text = fake_text  # type: ignore[assignment]
    pipe._answer_table = fake_table  # type: ignore[assignment]

    import asyncio

    result = asyncio.run(pipe.answer("一般风险企业数量"))
    assert result.usage == {}  # getattr 防御生效，未报错
