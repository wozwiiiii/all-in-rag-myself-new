"""生成器测试：双层拒答判定 + 两遍生成（逐 chunk 并行抽取 → 程序化拼接 + 结论段）。"""

import threading

from app.rag.generator import generate_answer
from app.rag.retriever import RetrievedChunk


class _StubLLM:
    """按 system prompt 区分抽取轮/总结轮/单遍轮，返回各自预设内容；记录调用。"""

    def __init__(self, extract="1. 监护人全程监护", summary="概括结论。", single="单遍答案",
                 fail_extract=False):
        self._extract = extract
        self._summary = summary
        self._single = single
        self._fail_extract = fail_extract
        self.extract_calls = 0
        self.summary_calls = 0
        self.single_calls = 0
        self.summary_prompts: list[str] = []
        self._lock = threading.Lock()  # 并行抽取下计数安全

    def chat_safe(self, messages, **kw):
        system = messages[0]["content"]
        kind = (
            "extract" if "要点抽取器" in system
            else "summary" if "要点总结器" in system
            else "single"
        )
        with self._lock:
            if kind == "extract":
                self.extract_calls += 1
            elif kind == "summary":
                self.summary_calls += 1
                self.summary_prompts.append(messages[-1]["content"])
            else:
                self.single_calls += 1

        class Msg:
            pass

        m = Msg()
        if kind == "extract" and self._fail_extract:
            raise RuntimeError("extraction down")
        m.content = {"extract": self._extract, "summary": self._summary, "single": self._single}[kind]
        return m


def _chunk(n=1):
    return [
        RetrievedChunk(text=f"上下文{i}", source="a.docx", heading="h", score=0.9)
        for i in range(n)
    ]


def test_empty_chunks_refuse_without_llm_call():
    """无检索证据：直接拒答，不调用 LLM。"""
    llm = _StubLLM()
    ans, src, refused = generate_answer("问题", [], llm)
    assert refused is True
    assert src == []
    assert "未检索到" in ans or "无法给出" in ans
    assert llm.extract_calls + llm.summary_calls + llm.single_calls == 0


def test_refusal_marker_in_head_detected():
    """拒答标记出现在答案开头（前 60 字符）：判定为拒答（两遍路：总结段开头）。"""
    llm = _StubLLM(summary="无法回答该问题，资料不足。")
    _, _, refused = generate_answer("问题", _chunk(), llm)
    assert refused is True


def test_refusal_marker_in_body_not_misjudged():
    """拒答标记出现在 60 字符之后：不应误判为拒答（M4 修复核心，单遍路）。"""
    prefix = "根据《规范》要求，动火作业分为特级、一级、二级。" * 3
    answer = prefix + "其他未提及的细则见附录。"
    llm = _StubLLM(fail_extract=True, single=answer)  # 抽取失败退化单遍
    ans, _, refused = generate_answer("动火作业分级", _chunk(), llm)
    assert refused is False
    assert "未提及" in ans  # 确实出现在答案正文中，但不应触发拒答


# ---------------- 两遍生成 ----------------


def test_two_pass_points_guaranteed_in_answer():
    """两遍生成核心保证：抽取到的要点原文进入答案（程序化拼接，零丢失）。"""
    chunks = _chunk(3)
    llm = _StubLLM(extract="1. 监护人全程监护\n2. 气体检测分析")
    ans, _, _ = generate_answer("安全措施", chunks, llm)
    assert llm.extract_calls == 3  # 逐 chunk 各一次
    assert llm.summary_calls == 1
    # 每路要点都在答案中（3 路 × 2 条 = 6 处出现）
    assert ans.count("监护人全程监护") == 3
    assert ans.count("气体检测分析") == 3
    assert ans.startswith("概括结论。")  # 结论段在前


def test_two_pass_renumbering_and_source_tags():
    """行级来源标注 + 全局重编号：答案行形如 'N. 内容 [i]'。"""
    llm = _StubLLM(extract="1. 要点甲")
    ans, _, _ = generate_answer("问题", _chunk(2), llm)
    lines = [ln for ln in ans.split("\n") if ln.startswith(("1.", "2.", "3."))]
    assert len(lines) == 2  # 2 chunk × 1 要点，全局编号 1、2
    assert lines[0].startswith("1. 要点甲 [1]")
    assert lines[1].startswith("2. 要点甲 [2]")


def test_two_pass_irrelevant_chunk_skipped():
    """无关 chunk（输出'无'）不产生要点行。"""
    class _MixedLLM(_StubLLM):
        def chat_safe(self, messages, **kw):
            if "要点抽取器" in messages[0]["content"]:
                with self._lock:
                    self.extract_calls += 1
                    if self.extract_calls > 1:  # 第 2 路无关
                        m = type("M", (), {"content": "无"})()
                        return m
            return super().chat_safe(messages, **kw)

    llm = _MixedLLM(extract="1. 要点甲")
    ans, _, _ = generate_answer("问题", _chunk(2), llm)
    assert "要点甲" in ans
    assert ans.count("1. ") == 1  # 只有第 1 路的要点被编号


def test_two_pass_summary_failure_tolerated():
    """总结失败：答案退化为纯要点列表，仍可用。"""
    class _NoSummaryLLM(_StubLLM):
        def chat_safe(self, messages, **kw):
            if "要点总结器" in messages[0]["content"]:
                raise RuntimeError("summary down")
            return super().chat_safe(messages, **kw)

    llm = _NoSummaryLLM()
    ans, _, _ = generate_answer("问题", _chunk(1), llm)
    assert ans.startswith("1. 监护人全程监护")  # 无结论段，直接要点


def test_two_pass_all_extraction_failed_degrades_to_single():
    """全部抽取失败：退化为单遍生成。"""
    llm = _StubLLM(fail_extract=True)
    ans, _, refused = generate_answer("问题", _chunk(2), llm)
    assert ans == "单遍答案"
    assert refused is False
    assert llm.single_calls == 1


def test_two_pass_disabled_single_call():
    """two_pass=False：只有单遍调用。"""
    llm = _StubLLM()
    ans, _, _ = generate_answer("问题", _chunk(), llm, two_pass=False)
    assert llm.extract_calls == 0
    assert llm.single_calls == 1
    assert ans == "单遍答案"
