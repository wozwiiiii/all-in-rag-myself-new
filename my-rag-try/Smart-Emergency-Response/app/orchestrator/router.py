"""意图路由：LLM 单次分类（text / table / hybrid），失败时降级为关键词启发式。

可靠性设计：路由是单点，必须永不抛异常——分类 LLM 挂了也要给出可用路由。
"""

from __future__ import annotations

import json

from app.core.logging import get_logger
from app.llm.client import LLMClient

logger = get_logger(__name__)

VALID_ROUTES = {"text", "table", "hybrid"}

_ROUTER_PROMPT = """你是问答系统的流量分发器。判断问题应走哪条数据通路，只输出 JSON：
{"route": "text"}    -> 政策/法规/标准/规范/办法的解读、说明类问题（知识在文档里）
{"route": "table"}   -> 企业数量、统计、查询具体数据类问题（答案在结构化表里）
{"route": "hybrid"}  -> 既需要查数据又需要解释政策/标准的问题

只输出上述 JSON，不要输出任何其他内容。"""

_TABLE_HINTS = ("数量", "多少", "几", "统计", "列出", "占比", "比例", "排名", "哪些企业")
_TEXT_HINTS = ("规定", "要求", "如何", "办法", "标准", "规范", "依据", "架构", "划分", "包括哪些", "措施", "流程")


def _heuristic(question: str) -> str:
    has_table = any(h in question for h in _TABLE_HINTS)
    has_text = any(h in question for h in _TEXT_HINTS)
    if has_table and has_text:
        return "hybrid"
    if has_table:
        return "table"
    if has_text:
        return "text"
    return "hybrid"


class IntentRouter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def route(self, question: str) -> str:
        try:
            msg = self._llm.chat_safe(
                [
                    {"role": "system", "content": _ROUTER_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=32,
            )
            content = (msg.content or "").strip()
            # 容忍模型输出 ```json 包裹
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            route = str(json.loads(content).get("route", ""))
            if route in VALID_ROUTES:
                return route
            raise ValueError(f"invalid route: {route}")
        except Exception as e:
            fallback = _heuristic(question)
            logger.warning("router_fallback_to_heuristic", extra={"err": str(e), "fallback": fallback})
            return fallback
