"""编排层降级测试：hybrid 融合失败时退化用表格路答案，不返回 500（F2 回归保护）。"""

import asyncio

from app.core.config import Settings
from app.core.errors import LLMError
from app.orchestrator.pipeline import QAPipeline, QAResult


class _FusionFailLLM:
    """所有 chat_safe 调用都抛 LLMError（模拟融合 LLM 不可用）。"""

    def chat_safe(self, messages, **kw):
        raise LLMError("fusion llm down")

    def chat(self, messages, **kw):
        raise LLMError("fusion llm down")


class _OkFusionLLM:
    """融合 LLM 正常，返回固定文本。"""

    def chat_safe(self, messages, **kw):
        class Msg:
            content = "融合后的最终答案"

        return Msg()


def _make_pipe(llm) -> QAPipeline:
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    return QAPipeline(settings, llm)  # type: ignore[arg-type]


def test_hybrid_fusion_failure_degrades_to_table_answer():
    """融合 LLM 失败：两路结果已得，应退化用表格路答案并标记 degraded，而非抛异常。"""
    pipe = _make_pipe(_FusionFailLLM())

    async def fake_text(q):
        return QAResult(
            answer="区域划分依据...",
            sources=[{"source": "a.docx"}],
            refused=False,
            detail={"branch": "text"},
        )

    async def fake_table(q):
        return QAResult(
            answer="42家企业",
            sources=[{"source": "表:t", "kind": "table"}],
            refused=False,
            detail={"branch": "table"},
        )

    pipe._answer_text = fake_text  # type: ignore[assignment]
    pipe._answer_table = fake_table  # type: ignore[assignment]

    result = asyncio.run(pipe._answer_hybrid("一般风险企业数量及划分"))
    assert result.answer == "42家企业"
    assert result.detail.get("fusion_degraded") is True
    assert result.refused is False


def test_hybrid_fusion_ok_not_degraded():
    """对照：融合 LLM 正常时不标记 degraded，返回融合文本。"""
    pipe = _make_pipe(_OkFusionLLM())

    async def fake_text(q):
        return QAResult(answer="文本答案", sources=[], refused=False, detail={"branch": "text"})

    async def fake_table(q):
        return QAResult(answer="42", sources=[], refused=False, detail={"branch": "table"})

    pipe._answer_text = fake_text  # type: ignore[assignment]
    pipe._answer_table = fake_table  # type: ignore[assignment]

    result = asyncio.run(pipe._answer_hybrid("问题"))
    assert result.answer == "融合后的最终答案"
    assert result.detail.get("fusion_degraded") is False
