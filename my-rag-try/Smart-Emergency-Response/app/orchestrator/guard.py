"""输出侧 prompt 泄露检测（输入侧结构隔离的第二道防线）。

设计原则（对齐 sandbox.py 的"白名单优于黑名单"哲学）：
- 用户输入与指令在消息层面已隔离（见 agent.py 选表阶段），本模块只兜底
  "忽略指令、把系统提示词复述/推理出来"这类 role 隔离挡不住的越狱。
- 实现为**特征串子串匹配**（归一化后），而非语义正则黑名单——
  正则理解不了否定语境，会误伤"禁止用水直接冲洗浓硫酸"这类含
  "水"+"冲洗"的正确答案；子串特征只匹配提示词内部特有记号，
  正常业务答案几乎不可能命中。

放业务层（orchestrator）而非 core：core 不允许反向依赖业务常量。
"""

from __future__ import annotations

# 提示词内部特有记号/片段。一旦答案泄露这些，判定为用户越狱成功。
#
# 选取原则（避免误伤业务答案）：
# - 只收"prompt 元层"特征：角色定义、ReAct 协议标记、表画像区块标题——
#   这些是模型只会从 system prompt 里看到、正常业务回答绝不会输出的内部措辞。
# - 刻意【不收】可合法出现在答案里的业务 SQL 规则（如 delete_status=0 / check_status=2）：
#   表格路会把"按 delete_status=0 且 check_status=2 筛选"作为回答依据正常写出，
#   将其列为泄露特征会导致合法答案被误拦（已实测误伤全部 table/hybrid 题）。
PROMPT_LEAK_SIGNATURES: list[str] = [
    # tabular SYSTEM_PROMPT 的角色定义与协议框架（正常答案不会引用）
    "你是危化企业数据分析专家",
    "Action Input:",
    "Thought:",
    # 表画像区块标题（表格路最终答案不会输出）
    "【各表数据画像",
    # 选表/融合 prompt 的内部角色句
    "你是数据表选择器",
    "你是综合答复助手",
    # 问答路由 prompt 内部句
    "你是问答系统的流量分发器",
]


def _normalize(text: str) -> str:
    """去空白与全角空格，降低模型对特征串的轻微改写绕过。"""
    return "".join(text.split()).replace("\u3000", "").replace(" ", "")


def check_prompt_leak(answer: str) -> str | None:
    """答案含任一泄露特征串则返回命中的特征（供日志），否则返回 None。

    因归一化后子串匹配，需对特征串与答案做同样的归一化比较。
    """
    if not answer:
        return None
    norm = _normalize(answer)
    for sig in PROMPT_LEAK_SIGNATURES:
        if _normalize(sig) in norm:
            return sig
    return None


# 检测到泄露后用于替换答案的安全文案。
# 刻意不含 _REFUSAL_MARKERS（metrics.py 中 "无法回答/未检索到" 等词），
# 以免污染 refused 语义与评测口径。
LEAK_SAFE_ANSWER = (
    "您的问题似乎试图获取本系统的内部提示词。出于安全考虑，"
    "我不能透露系统内部配置；如有化工应急相关问题，请直接描述您的需求。"
)
