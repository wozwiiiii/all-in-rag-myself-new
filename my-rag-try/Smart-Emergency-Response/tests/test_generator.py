"""生成器拒答判定测试：拒答标记仅匹配答案前 60 字符（AUDIT M4 修复点的回归保护）。"""

from app.rag.generator import generate_answer
from app.rag.retriever import RetrievedChunk


class _StubLLM:
    def __init__(self, content):
        self._content = content

    def chat_safe(self, messages, **kw):
        class Msg:
            content = self._content

        return Msg()


def _chunk():
    return [RetrievedChunk(text="上下文", source="a.docx", heading="h", score=0.9)]


def test_empty_chunks_refuse_without_llm_call():
    """无检索证据：直接拒答，不调用 LLM。"""
    ans, src, refused = generate_answer("问题", [], _StubLLM("不应被调用"))
    assert refused is True
    assert src == []
    assert "未检索到" in ans or "无法给出" in ans


def test_refusal_marker_in_head_detected():
    """拒答标记出现在答案开头（前 60 字符）：判定为拒答。"""
    answer = "无法回答该问题，资料不足。"
    _, _, refused = generate_answer("问题", _chunk(), _StubLLM(answer))
    assert refused is True


def test_refusal_marker_in_body_not_misjudged():
    """拒答标记出现在 60 字符之后的正文：不应误判为拒答（M4 修复核心）。"""
    # 一句约 23 字符，重复 3 次 > 60 字符，前 60 字符内无任何拒答标记
    prefix = "根据《规范》要求，动火作业分为特级、一级、二级。" * 3
    answer = prefix + "其他未提及的细则见附录。"
    ans, _, refused = generate_answer("动火作业分级", _chunk(), _StubLLM(answer))
    assert refused is False
    assert "未提及" in ans  # 确实出现在答案正文中，但不应触发拒答
