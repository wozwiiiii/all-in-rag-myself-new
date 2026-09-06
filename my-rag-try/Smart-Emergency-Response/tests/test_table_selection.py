"""选表增强的离线测试：字段级提示注入、选表规则、字段校验兜底。

背景（protocol-bench-20260906 实测归因）：6 道表格题中 3 题第一阶段就选错表
（如"开停车企业数"选了 base_device_draft，正确答案在 enterprise_promise.open_parking），
根因是选表 prompt 只有表介绍文本、没有字段级语义。本文件固化三项增强的行为：
1. 表清单按问题注入相关字段提示（_relevant_columns / _table_line）；
2. _SELECT_PROMPT 含基于表结构文档的选表规则；
3. LLM 漏选列语义强命中的表时，纯本地兜底补选（_augment_selection，不增 LLM 调用）。

测试通过 monkeypatch 替换 table_intros / table_columns，不依赖真实 docx 与 17 张表。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.config import Settings
from app.tabular.agent import (
    _SELECT_PROMPT,
    TabularAgent,
    _relevant_columns,
)


# ---------------- 测试环境 ----------------

# 模拟真实场景的最小表集：
# - promise_daily：列描述含"开停车/企业编码"（对应 enterprise_promise 的区分性语义）
# - device_log：列描述与业务问题无语义交集（对应被误选的 base_device_draft）
# - base_t_enterprise_draft：企业基础表（有效性过滤规则兜底用）
_COLMAPS = {
    "promise_daily": {
        "org_code": "企业编码",
        "open_parking": "是否处于开停车状态，0：否，1：是",
        "risk_grade": "风险级别 0未填报 4低风险 3一般风险 2较大风险 1重大风险",
        "create_date": "承诺日期",
    },
    "device_log": {"device_id": "设备编号", "workshop": "所属车间"},
    "base_t_enterprise_draft": {"org_code": "企业编码", "name": "企业名称"},
}
_INTROS = {
    "promise_daily": "企业安全生产责任承诺表，储存企业安全承诺信息。",
    "device_log": "设备检修日志表。",
    "base_t_enterprise_draft": "企业基础信息表。",
}
_ALL = sorted(_COLMAPS)


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> Path:
    for name, cols in {
        "promise_daily": {"org_code": ["e1", "e2"], "open_parking": [1, 0]},
        "device_log": {"device_id": ["d1"], "workshop": ["w1"]},
        "base_t_enterprise_draft": {"org_code": ["e1"], "name": ["甲"]},
    }.items():
        pd.DataFrame(cols).to_parquet(tmp_path / f"{name}.parquet", index=False)
    return tmp_path


class _CaptureLLM:
    """记录收到的 messages 并按序返回预设响应。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0
        self.messages_seen: list[list[dict]] = []

    def chat_safe(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.messages_seen.append(messages)
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[idx])


def _agent(parquet_dir: Path, llm: _CaptureLLM, monkeypatch) -> TabularAgent:
    monkeypatch.setattr(
        "app.tabular.agent.table_intros", lambda s: dict(_INTROS)
    )
    monkeypatch.setattr(
        "app.tabular.agent.table_columns", lambda s: {k: dict(v) for k, v in _COLMAPS.items()}
    )
    return TabularAgent(Settings(parquet_dir=parquet_dir), llm)  # type: ignore[arg-type]


# ---------------- _relevant_columns：问题与列描述的语义匹配 ----------------


def test_relevant_columns_ranks_semantic_hits_first():
    """列描述与问题共享 2-gram 的字段被选中并按命中数降序。"""
    q = "2025年3月22日处于开停车状态的企业有多少家"
    rel = _relevant_columns(q, _COLMAPS["promise_daily"])
    cols = [c for _, c, _ in rel]
    assert cols[0] == "open_parking"  # 开停/停车/车的/处于 强命中
    assert "create_date" not in cols  # "承诺日期"与问题无重合
    assert "device_id" not in {c for _, c, _ in _relevant_columns(q, _COLMAPS["device_log"])}


def test_relevant_columns_empty_colmap():
    assert _relevant_columns("任何问题", {}) == []


# ---------------- 选表 prompt：字段提示 + 选表规则 ----------------


def test_select_prompt_contains_field_hints_and_rules(parquet_dir, monkeypatch):
    """表清单按问题注入相关字段提示，system prompt 含选表规则。"""
    llm = _CaptureLLM(['{"tables": ["promise_daily"]}'])
    agent = _agent(parquet_dir, llm, monkeypatch)
    agent._select_tables("处于开停车状态的企业有多少家")

    system = llm.messages_seen[0][0]["content"]
    assert "open_parking=是否处于开停车状态" in system  # 字段级提示注入
    assert "enterprise_promise" in system  # 选表规则提到承诺表
    assert "risk_rating_code" in system  # 编码字段 vs 枚举列的消歧规则
    assert llm.calls == 1  # 选表只花一次 LLM 调用


# ---------------- _augment_selection：字段校验兜底 ----------------


def test_augment_adds_missed_table_by_column_semantics(parquet_dir, monkeypatch):
    """LLM 漏选列语义强命中的表时被补选（对应 id 11 实测失效场景）。"""
    llm = _CaptureLLM(['{"tables": ["device_log"]}'])  # 复现实测：误选设备表
    agent = _agent(parquet_dir, llm, monkeypatch)
    picked = agent._select_tables("2025年3月22日处于开停车状态的企业有多少家")

    assert "promise_daily" in picked  # open_parking 列描述强命中，兜底补选
    assert "device_log" in picked  # 只增不删：LLM 的选择保留
    assert llm.calls == 1  # 兜底是纯本地计算，不增加 LLM 调用


def test_augment_no_false_positive_when_scores_low(parquet_dir, monkeypatch):
    """无强命中时不补选，避免 prompt 膨胀。"""
    llm = _CaptureLLM(['{"tables": ["device_log"]}'])
    agent = _agent(parquet_dir, llm, monkeypatch)
    picked = agent._select_tables("设备车间今日运行情况")

    assert "promise_daily" not in picked
    assert "base_t_enterprise_draft" not in picked  # 问题不含企业/家/单位


def test_augment_adds_base_enterprise_table_for_entity_count(parquet_dir, monkeypatch):
    """"企业数量"类问题漏选企业基础表时按业务规则补上（有效性过滤依赖）。"""
    llm = _CaptureLLM(['{"tables": ["promise_daily"]}'])
    agent = _agent(parquet_dir, llm, monkeypatch)
    picked = agent._select_tables("涉及动火作业的企业有多少家")

    assert "base_t_enterprise_draft" in picked
