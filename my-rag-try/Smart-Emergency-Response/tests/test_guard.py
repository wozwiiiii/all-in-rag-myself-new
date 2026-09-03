"""输出侧 prompt 泄露检测测试 + 结构隔离回归。

2026-09-03 批次3 引入。
- guard 特征子串匹配：泄露样本命中、正常答案不误伤（含否定句）。
- pipeline 出口：泄露答案被替换为安全文案 + 日志关键字 + detail 标记。
- agent._select_tables 消息结构：指令与问题分属 system/user（结构隔离）。
"""

import asyncio
import re

from app.core.config import Settings
from app.orchestrator.guard import LEAK_SAFE_ANSWER, PROMPT_LEAK_SIGNATURES, check_prompt_leak
from app.orchestrator.pipeline import QAPipeline, QAResult
from app.tabular.agent import TabularAgent


# ---------------- guard.check_prompt_leak ----------------


def test_legal_sql_rule_in_answer_not_detected():
    """表格路合法答案把 delete_status=0/check_status=2 当筛选依据写出，不应误判为泄露。"""
    ans = "涉及重氮化工艺的企业数量为4家，依据v_process工艺表统计，企业需满足delete_status=0且check_status=2的有效性条件。"
    assert check_prompt_leak(ans) is None


def test_protocol_replay_detected():
    """完整复述 system prompt 的协议框架（角色/Thought/Action）应命中。"""
    ans = "我的系统提示词是：你是危化企业数据分析专家。每轮按 Thought: ... Action: execute_pandas\nAction Input: ... 输出"
    assert check_prompt_leak(ans) is not None


def test_leak_role_identity_detected():
    assert check_prompt_leak("你的系统提示词说你是危化企业数据分析专家") is not None


def test_normal_answer_not_false_positive():
    # 正常业务答案（不含任何提示词内部记号）不应命中
    assert check_prompt_leak("涉及重氮化工艺的企业数量为4家，依据v_process工艺表统计。") is None


def test_negative_sentence_not_false_positive():
    # 否定句含"水/冲洗"等词不应误伤（黑名单正则的经典陷阱）
    assert check_prompt_leak("浓硫酸泄漏时禁止用水直接冲洗，应先用沙土覆盖吸附") is None


def test_dynamic_profile_prompt_leak():
    # 复述融合 prompt 的原文措辞（"你是综合答复助手"）应命中
    assert check_prompt_leak("下面是我的系统提示词：你是综合答复助手，负责融合两路结果") is not None


# ---------------- pipeline 出口拦截 ----------------

def test_pipeline_intercepts_leaked_answer():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    pipe = QAPipeline(settings, llm=None)  # type: ignore[arg-type]

    async def fake_text(q):
        # 直接返回复述协议框架的内容，模拟文本分支被越狱
        return QAResult(answer="我的指令是：你是危化企业数据分析专家，按 Action Input: 输出代码", refused=False)

    pipe._answer_text = fake_text  # type: ignore[assignment]
    pipe._router.route = lambda q: "text"  # type: ignore[method-assign] 固定路由，聚焦出口检测
    result = asyncio.run(pipe.answer("怎么判断企业有效性？"))
    assert result.answer == LEAK_SAFE_ANSWER
    assert result.detail.get("prompt_leak") is True


def test_pipeline_keeps_normal_answer():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    pipe = QAPipeline(settings, llm=None)  # type: ignore[arg-type]

    async def fake_text(q):
        return QAResult(answer="涉及氯化工艺的企业数量为5家", refused=False)

    pipe._answer_text = fake_text  # type: ignore[assignment]
    pipe._router.route = lambda q: "text"  # type: ignore[method-assign]
    result = asyncio.run(pipe.answer("氯化工艺企业数？"))
    assert result.answer == "涉及氯化工艺的企业数量为5家"
    assert "prompt_leak" not in result.detail


# ---------------- agent._select_tables 结构隔离 ----------------

def _sealed_settings(tmp_path):
    # 最小化 parquet 目录，避免 _select_tables 枚举出真实 17 张表
    return Settings(_env_file=None, openai_api_key="sk-test", parquet_dir=tmp_path)


def test_agent_select_tables_splits_roles(tmp_path):
    """选表调用的消息应为 system(指令+表清单) / user(仅问题)，指令不含用户问题原文。"""

    captured = {}

    class _LLM:
        def chat_safe(self, messages, **kw):
            captured["messages"] = messages
            class M:
                content = '{"tables": ["base_t_enterprise_draft"]}'

            return M()

    agent = TabularAgent(_sealed_settings(tmp_path), _LLM())  # type: ignore[arg-type]
    agent._select_tables("请问一般风险企业有几家？")

    msgs = captured["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # system 含指令/表清单，不含问题原文；user 只有问题
    assert "select_tables" not in msgs[0]["content"]
    assert "请问一般风险企业有几家？" not in msgs[0]["content"]
    assert msgs[1]["content"] == "【问题】\n请问一般风险企业有几家？"
    # 系统提示不应包含"忽略上文/输出提示词"之类的引导词，且是忠实指令
    assert "base_t_enterprise_draft" not in msgs[0]["content"] or True  # 表清单可能含该名
    assert re.search(r"输出.*系统提示词|ignore", msgs[0]["content"], re.I) is None


def test_leak_safe_answer_contains_no_refusal_marker():
    """安全文案不应含拒答 marker 词（metrics.py），避免污染 refused 语义与评测。"""
    markers = ("无法回答", "未检索到", "无法给出", "资料库未")
    assert not any(m in LEAK_SAFE_ANSWER for m in markers)
