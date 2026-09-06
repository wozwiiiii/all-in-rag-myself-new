"""表格代理单元测试：TabularAnswer 观测留存（评测侧事实一致性的证据来源）。

2026-09-06 批次5 引入 observations 字段：run() 每轮沙箱 Observation 留存
（保留最后 3 轮），供 QAResult.evidence 透传与 eval 侧核对答案数值出处。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.config import Settings
from app.llm.client import LLMClient
from app.tabular.agent import TabularAgent, TabularAnswer


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> Path:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    df.to_parquet(tmp_path / "tiny.parquet", index=False)
    return tmp_path


class _ScriptedLLM:
    """按调用顺序返回预设响应（选表 → ReAct 轮…），不依赖真实 API。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    def chat_safe(self, messages, **kwargs):  # noqa: ANN001, ANN003
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[idx])


def _settings(parquet_dir: Path) -> Settings:
    return Settings(parquet_dir=parquet_dir)


def test_tabular_answer_default_observations_empty():
    """默认空列表：不破任何按旧签名列名构造 TabularAnswer 的调用方。"""
    ans = TabularAnswer(answer="x")
    assert ans.observations == []
    assert ans.converged is False


def test_run_collects_observations(parquet_dir: Path):
    """execute 轮的 Observation 被留存，finish 返回时随结果带出。"""
    llm = _ScriptedLLM(
        [
            '{"tables": ["tiny"]}',
            "Thought: 求和\nAction: execute_pandas\nAction Input:\n"
            "result = dfs['tiny']['v'].sum()\nprint(result)",
            "Thought: 已得到结果\nAction: finish\nAction Input:\nv 列总和为 6。",
        ]
    )
    agent = TabularAgent(_settings(parquet_dir), llm)  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert res.iterations == 2
    assert res.observations and len(res.observations) == 1
    assert "[OK]" in res.observations[0]
    assert "6" in res.observations[0]  # 执行结果数值在证据中


def test_observations_capped_at_last_3(parquet_dir: Path):
    """多轮执行只保留最后 3 轮 Observation。"""
    execute = (
        "Thought: 再算一轮\nAction: execute_pandas\nAction Input:\n"
        f"result = dfs['tiny']['v'].sum()\nprint(result)"
    )
    llm = _ScriptedLLM(
        ['{"tables": ["tiny"]}']
        + [execute] * 4  # 4 轮 execute
        + ["Thought: 提交\nAction: finish\nAction Input:\n答案 6"]
    )
    agent = TabularAgent(_settings(parquet_dir), llm)  # type: ignore[arg-type]
    res = agent.run("tiny 表 v 列总和是多少？")

    assert res.converged is True
    assert len(res.observations) == 3  # 4 轮只留尾部 3 轮
