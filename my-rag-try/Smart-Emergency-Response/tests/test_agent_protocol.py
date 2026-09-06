"""双协议（react / tool_calls）单测：不依赖真实 API，用脚本化 mock 覆盖关键路径。

覆盖点：
1. tool_calls 正常路径：execute → 回填 observation → finish；
2. 工具参数 JSON 非法时的容错与自愈回填；
3. 既无 tool_calls 又无 content 时记为 format_error；
4. 只有 content 无 tool_calls 时按 tool_choice=auto 语义收尾为答案；
5. 两种协议共用同一套沙箱：越权代码（__class__ / import）都被 REJECTED；
6. agent_protocol 默认是 react，非法值回退 react，配置值可被正确读取；
7. 新增统计字段向后兼容（旧调用方按位置/关键字构造 TabularAnswer 不破）。

注意：不要 `from tests.xxx import ...` —— 本机 user site-packages 下存在一个
无关的第三方 tests 包（其 __init__ 会 import torch 导致段错误），且带 __init__.py
的常规包在 import 时优先于本项目的命名空间包 tests/。mock 一律在本文件内自给。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.config import Settings
from app.llm.client import LLMClient
from app.tabular.agent import (
    VALID_PROTOCOLS,
    TabularAgent,
    TabularAnswer,
    _normalize_protocol,
)


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> Path:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    df.to_parquet(tmp_path / "tiny.parquet", index=False)
    return tmp_path


def _settings(parquet_dir: Path, **kwargs) -> Settings:
    return Settings(parquet_dir=parquet_dir, **kwargs)


def _tc(name: str, args: object, tid: str = "call_1") -> SimpleNamespace:
    """构造带 id / function.name / function.arguments 的 tool_call。

    args 为 dict 时序列化为 JSON 字符串；为 str 时原样使用（用于构造非法 JSON）。
    """
    raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return SimpleNamespace(
        id=tid, type="function", function=SimpleNamespace(name=name, arguments=raw)
    )


class _ScriptedLLM:
    """按调用顺序返回脚本化响应，兼容纯文本（react）与 tool_calls 两种形态。

    responses 元素：
    - str            → 视为 content（react 轮 / 选表轮）
    - dict           → {"content": ..., "tool_calls": [...]}（tool_calls 轮）
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = 0
        self.seen_messages: list[list] = []
        self.kwargs_seen: list[dict] = []

    def chat_safe(self, messages, **kwargs):  # noqa: ANN001, ANN003
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        self.seen_messages.append(messages)
        self.kwargs_seen.append(kwargs)
        r = self._responses[idx]
        if isinstance(r, dict):
            return SimpleNamespace(
                content=r.get("content"), tool_calls=r.get("tool_calls", [])
            )
        return SimpleNamespace(content=r, tool_calls=None)


_SELECT = '{"tables": ["tiny"]}'
_SUM_CODE = "result = dfs['tiny']['v'].sum()\nprint(result)"


# ---------------- 1. tool_calls 正常路径 ----------------

def test_toolcalls_normal_path(parquet_dir: Path):
    llm = _ScriptedLLM(
        [
            _SELECT,
            {"tool_calls": [_tc("execute_pandas", {"code": _SUM_CODE})]},
            {"tool_calls": [_tc("finish", {"answer": "v 列总和为 6。"})]},
        ]
    )
    agent = TabularAgent(_settings(parquet_dir), llm, protocol="tool_calls")  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert res.iterations == 2
    assert res.format_errors == 0
    assert res.tool_errors == 0
    assert res.llm_calls == 3  # 选表 1 + 分析轮 2
    assert res.elapsed_ms >= 0
    assert "6" in res.observations[0]
    assert res.answer == "v 列总和为 6。"

    # 执行结果必须作为 role=tool 消息回填，且带对应的 tool_call_id
    last = llm.seen_messages[-1]
    tool_msgs = [m for m in last if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"
    assert tool_msgs[0]["name"] == "execute_pandas"
    # assistant 消息需原样回带 tool_calls，且 content 为空串而不是 None
    assistant_msgs = [m for m in last if m.get("role") == "assistant"]
    assert assistant_msgs[-1]["content"] == ""
    assert assistant_msgs[-1]["tool_calls"][0]["function"]["name"] == "finish"
    # 工具 Schema 确实下发给 LLM
    assert llm.kwargs_seen[1].get("tools")


# ---------------- 2. 参数 JSON 非法 → 容错自愈（不抛异常） ----------------

def test_toolcalls_bad_json_self_heal(parquet_dir: Path):
    llm = _ScriptedLLM(
        [
            _SELECT,
            # 第一轮 arguments 是非法 JSON → 记 format_error 并回填错误让模型自愈
            {"tool_calls": [_tc("execute_pandas", "{not valid json", tid="call_bad")]},
            {"tool_calls": [_tc("execute_pandas", {"code": _SUM_CODE})]},
            {"tool_calls": [_tc("finish", {"answer": "v 列总和为 6。"})]},
        ]
    )
    agent = TabularAgent(_settings(parquet_dir), llm, protocol="tool_calls")  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert res.format_errors == 1  # 非法 JSON 计为格式错误
    assert res.tool_errors == 0  # 自愈成功，未产生沙箱失败
    # 错误信息回填到了 role=tool 消息里，模型下一轮能看到
    bad_round = llm.seen_messages[2]
    assert any(
        m.get("role") == "tool" and "JSON 解析失败" in m.get("content", "")
        for m in bad_round
    )


# ---------------- 3. 既无 tool_calls 又无 content → format_error ----------------

def test_toolcalls_empty_response_format_error(parquet_dir: Path):
    llm = _ScriptedLLM(
        [
            _SELECT,
            {"content": None, "tool_calls": []},  # 既无工具调用也无文本
            {"tool_calls": [_tc("finish", {"answer": "兜底答案 6。"})]},
        ]
    )
    agent = TabularAgent(_settings(parquet_dir), llm, protocol="tool_calls")  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert res.format_errors == 1
    assert res.answer == "兜底答案 6。"


# ---------------- 4. 只有 content：按 auto 语义收尾 ----------------

def test_toolcalls_content_only_treated_as_answer(parquet_dir: Path):
    llm = _ScriptedLLM([_SELECT, {"content": "无需查表，答案为 6。", "tool_calls": []}])
    agent = TabularAgent(_settings(parquet_dir), llm, protocol="tool_calls")  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert res.format_errors == 0
    assert res.answer == "无需查表，答案为 6。"


# ---------------- 5. tool_calls 模式泄漏 ReAct 协议壳 → 记 format_error 并剥离 ----------------

def test_toolcalls_leak_protocol_shell_recorded_and_stripped(parquet_dir: Path):
    """复现实测缺陷：72B 在 tool_calls 模式下 finish.answer 误带 Thought/Action/Action Input 壳。

    必须：记一次 format_errors（event=protocol_leak_in_finish），并把壳剥离后保留真正答案正文，
    而不是把协议标记直接透传给用户。
    """
    leaked = (
        "Thought: 已得到统计结果，提交答案\n"
        "Action: finish\n"
        "Action Input:\n"
        "系统中有 348 条仓库记录。"
    )
    llm = _ScriptedLLM(
        [
            _SELECT,
            {"tool_calls": [_tc("execute_pandas", {"code": _SUM_CODE})]},
            {"tool_calls": [_tc("finish", {"answer": leaked})]},
        ]
    )
    agent = TabularAgent(_settings(parquet_dir), llm, protocol="tool_calls")  # type: ignore[arg-type]
    res = agent.run("仓库记录有多少条？")

    assert res.converged is True
    assert res.format_errors == 1
    # 壳被剥离，只留真正答案；协议标记不得透传
    assert res.answer == "系统中有 348 条仓库记录。"
    assert "Thought:" not in res.answer
    assert "Action" not in res.answer


# ---------------- 6. 两种协议共用同一套沙箱：越权代码都被 REJECTED ----------------

_BAD_CODES = [
    "x = dfs['tiny'].__class__",  # 下划线属性（AST 层直接拒）
    "import os",  # import 语句（AST 层直接拒）
]


@pytest.mark.parametrize("bad_code", _BAD_CODES)
@pytest.mark.parametrize("protocol", ["tool_calls", "react"])
def test_both_protocols_share_sandbox(parquet_dir: Path, protocol: str, bad_code: str):
    if protocol == "tool_calls":
        responses: list = [
            _SELECT,
            {"tool_calls": [_tc("execute_pandas", {"code": bad_code})]},
            {"tool_calls": [_tc("finish", {"answer": "被拒。"})]},
        ]
    else:
        responses = [
            _SELECT,
            f"Thought: 越权\nAction: execute_pandas\nAction Input:\n{bad_code}",
            "Thought: 收尾\nAction: finish\nAction Input:\n被拒。",
        ]

    llm = _ScriptedLLM(responses)
    agent = TabularAgent(_settings(parquet_dir), llm, protocol=protocol)  # type: ignore[arg-type]
    res = agent.run("越权访问")

    assert res.tool_errors == 1, res.observations
    assert res.format_errors == 0  # 越权是工具错误，不是协议格式错误
    assert any("[REJECTED]" in o for o in res.observations)


# ---------------- 6. 协议配置：默认值 / 非法值 / 从 settings 读取 ----------------

def test_protocol_default_is_react(parquet_dir: Path):
    llm = _ScriptedLLM([_SELECT])
    agent = TabularAgent(_settings(parquet_dir), llm)  # type: ignore[arg-type]
    assert agent._protocol == "react"


def test_invalid_protocol_falls_back_to_react(parquet_dir: Path):
    llm = _ScriptedLLM([_SELECT])
    for bad in ("bogus", "", None, "REACT "):
        agent = TabularAgent(_settings(parquet_dir), llm, protocol=bad)  # type: ignore[arg-type]
        # 大小写/空白被归一，"REACT " 合法；其余非法值回退 react
        assert agent._protocol in VALID_PROTOCOLS
    assert TabularAgent(_settings(parquet_dir), llm, protocol="bogus")._protocol == "react"  # type: ignore[arg-type]


def test_protocol_read_from_settings(parquet_dir: Path):
    llm = _ScriptedLLM([_SELECT])
    settings = _settings(parquet_dir, agent_protocol="tool_calls")
    assert TabularAgent(settings, llm)._protocol == "tool_calls"  # type: ignore[arg-type]
    # 构造参数优先于配置
    assert TabularAgent(settings, llm, protocol="react")._protocol == "react"  # type: ignore[arg-type]


def test_config_default_protocol_is_react():
    assert Settings().agent_protocol == "react"
    assert _normalize_protocol("tool_calls") == "tool_calls"
    assert _normalize_protocol("anything") == "react"
    assert _normalize_protocol(None) == "react"


# ---------------- 7. 新增统计字段向后兼容 ----------------

def test_tabular_answer_new_fields_default_zero():
    ans = TabularAnswer(answer="x")
    assert ans.format_errors == 0
    assert ans.llm_calls == 0
    assert ans.tool_errors == 0
    assert ans.elapsed_ms == 0
    assert ans.observations == []
    assert ans.converged is False


def test_react_counts_format_error(parquet_dir: Path):
    """react 下正则匹配失败 → format_error +1，并提示重试后继续。"""
    llm = _ScriptedLLM(
        [
            _SELECT,
            "我想先随便说两句，不按协议输出。",
            "Thought: 收尾\nAction: finish\nAction Input:\n答案 6",
        ]
    )
    agent = TabularAgent(_settings(parquet_dir), llm)  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert res.format_errors == 1
    assert res.llm_calls == 3
