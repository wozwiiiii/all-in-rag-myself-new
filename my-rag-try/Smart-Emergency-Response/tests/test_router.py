"""意图路由测试：LLM 分类正常路径 + 异常降级到启发式。"""

import pytest

from app.orchestrator.router import IntentRouter, _heuristic


class _BrokenLLM:
    def chat_safe(self, *a, **kw):
        raise RuntimeError("llm down")


class _JsonLLM:
    def __init__(self, content: str):
        self._content = content

    def chat_safe(self, *a, **kw):
        class Msg:
            content = self._content

        return Msg()


def test_heuristic_rules():
    assert _heuristic("涉及重氮化工艺的企业数量？") == "table"
    assert _heuristic("动火作业的安全要求是什么？") == "text"
    assert _heuristic("给出一般风险企业数量，并说明如何划分区域。") == "hybrid"
    assert _heuristic("今天天气怎么样") == "hybrid"  # 兜底默认


def test_router_falls_back_when_llm_down():
    router = IntentRouter(_BrokenLLM())  # type: ignore[arg-type]
    assert router.route("涉及重氮化工艺的企业数量？") == "table"
    assert router.route("动火作业的安全要求是什么？") == "text"


def test_router_parses_plain_json():
    router = IntentRouter(_JsonLLM('{"route": "table"}'))  # type: ignore[arg-type]
    assert router.route("任意问题") == "table"


def test_router_parses_fenced_json():
    router = IntentRouter(_JsonLLM('```json\n{"route": "text"}\n```'))  # type: ignore[arg-type]
    assert router.route("任意问题") == "text"


def test_router_invalid_route_falls_back():
    router = IntentRouter(_JsonLLM('{"route": "nonsense"}'))  # type: ignore[arg-type]
    # 无效路由值 -> 异常 -> 启发式（无信号 -> hybrid）
    assert router.route("hello world") == "hybrid"
