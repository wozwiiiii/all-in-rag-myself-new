"""表格 ReAct 代理（两阶段）：先由 LLM 选择相关表，再只注入所选表 schema 执行分析。

两阶段的动因：17 张表全量 schema 约 2.3 万 token，7B 模型推理必然超时；
选表后 prompt 缩小一个数量级，延迟与成功率同时改善。

协议选择：使用经典 ReAct 文本协议（Thought/Action/Action Input）而非
OpenAI function calling —— 实测硅基流动 Qwen2.5-7B 的 tool_calls 输出
不稳定（伪 JSON 乱码、无法解析），文本协议对 7B 模型更可靠且完全可控。

安全与可靠性：
1. LLM 生成的代码全部经 AST 白名单 + 子进程沙箱执行（见 sandbox.py）；
2. 执行错误 traceback 回填下一轮，形成自愈闭环；
3. 选表失败自动降级为启发式选表，不阻断请求。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.errors import SandboxViolation
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.tabular.sandbox import run_code
from app.tabular.schema import build_table_profiles, table_intros

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是危化企业数据分析专家，通过编写 pandas 代码回答问题。

【执行环境（重要）】
- 代码在受限沙箱运行：只有 pd / np / dfs 可用；禁止 import、文件读写、下划线属性（如 __class__）。
- 所有表已预载入字典 dfs，key 为表名（见下方表清单），不要使用 pd.read_*。
- 将需要返回的结果赋值给变量 result，或用 print 输出。

【业务规则（必须遵守）】
1. 凡涉及企业编码（org_code）的查询，必须基于有效企业：企业基础表 base_t_enterprise_draft 中 delete_status=0（未删除）且 check_status=2（已审核）。
2. 主表（enterprise_promise 除外）若含"删除状态/审核状态"字段，同样只统计有效数据。
3. "特殊作业"时间范围由两阶段构成：计划阶段 start_time~end_time；实施阶段 zysskssj~zyssjssj。涉及某日是否有作业时，两个时间窗都要检查。
4. 禁止把多张表合并成一张表处理：需要跨表信息时用 merge/join 按主键关联。

【输出协议（每轮必须严格遵守，不要输出协议之外的任何内容）】
Thought: <一句话说明你这一步要做什么>
Action: execute_pandas
Action Input:
<完整可执行的 Python 代码>

或者，当你已确认最终答案时：
Thought: <一句话总结结论>
Action: finish
Action Input:
<最终答案：中文，先结论，再一句依据（涉及哪些表、什么筛选条件）>

收到执行结果后，我会以 "Observation:" 开头返回给你；若代码报错，请阅读 traceback 修正代码再试。

【完整示例（严格模仿此格式与代码风格，只用真实存在的列名）】
问题：涉及危化品2-丙醇的企业数量？

Thought: 统计 v_sds 中 sds_name 含"2-丙醇"的去重企业数，只统计有效企业
Action: execute_pandas
Action Input:
sds = dfs['v_sds']
ent = dfs['base_t_enterprise_draft']
valid = ent[(ent['delete_status'] == 0) & (ent['check_status'] == 2)]['org_code']
merged = sds[sds['org_code'].isin(set(valid))]
result = merged[merged['sds_name'].astype(str).str.contains('2-丙醇', na=False)]['org_code'].nunique()
print(result)

Observation:
[OK]
[result] 17

Thought: 已得到统计结果，提交答案
Action: finish
Action Input:
涉及危化品2-丙醇的企业数量为 17 家。依据：v_sds 表 sds_name 模糊匹配"2-丙醇"，企业满足 delete_status=0 且 check_status=2，按 org_code 去重统计。"""

_SELECT_PROMPT = """你是数据表选择器。根据问题从下列业务表中选出回答问题所需的表（含关联所需的中间表），只输出 JSON：
{{"tables": ["表名1", "表名2"]}}

可用表：
{table_list}

问题：{question}

注意：只输出 JSON；一般选 1-4 张表；不确定时宁可多选关联表。"""


@dataclass
class TabularAnswer:
    answer: str
    tables_used: list[str] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


_DFS_RE = re.compile(r"dfs\[['\"]([^\]'\"]+)['\"]\]")

_ACTION_RE = re.compile(
    r"Action:\s*(execute_pandas|finish)\s*\n\s*Action Input:\s*\n?(.+?)"
    r"(?=\n[ \t]*(?:Thought:|Action:|Observation:)\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^```[\w+-]*[ \t]*\n?([\s\S]*?)\n?```[ \t]*$")

_FORMAT_ERROR = (
    "格式错误：你的回复未遵循 Thought/Action/Action Input 协议。"
    "请重新输出，且 Action 只能是 execute_pandas 或 finish，"
    "Action Input 后换行紧跟代码或最终答案，其后不要再输出其他内容。"
)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


class TabularAgent:
    def __init__(self, settings: Settings, llm: LLMClient):
        self._s = settings
        self._llm = llm

    # ---------------- 阶段一：表选择 ----------------

    def _select_tables(self, question: str) -> list[str] | None:
        intros = table_intros(self._s)
        all_names = sorted(p.stem for p in self._s.parquet_dir.glob("*.parquet"))
        table_list = "\n".join(
            f"- {n}: {intros.get(n, '（见列名）')}" for n in all_names
        )
        try:
            msg = self._llm.chat_safe(
                [
                    {
                        "role": "user",
                        "content": _SELECT_PROMPT.format(table_list=table_list, question=question),
                    }
                ],
                temperature=0.0,
                max_tokens=200,
            )
            content = (msg.content or "").strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            tables = [t for t in json.loads(content).get("tables", []) if t in all_names]
            if tables:
                logger.info("tables_selected", extra={"tables": tables})
                return tables
        except Exception as e:
            logger.warning("table_selection_fallback_to_heuristic", extra={"err": str(e)})
        picked = self._heuristic_select(question, intros, all_names)
        logger.info("tables_selected_heuristic", extra={"tables": picked})
        return picked

    @staticmethod
    def _heuristic_select(
        question: str, intros: dict[str, str], all_names: list[str], top: int = 5
    ) -> list[str]:
        """LLM 选表失败时的降级：问题与表介绍的 2-gram 重合度排序。"""
        bigrams = {question[i : i + 2] for i in range(len(question) - 1)}

        def score(name: str) -> int:
            text = intros.get(name, "") + " " + name
            return sum(1 for bg in bigrams if bg in text)

        ranked = sorted(all_names, key=score, reverse=True)
        picked = [n for n in ranked[:top] if score(n) > 0] or all_names
        if picked != all_names and "base_t_enterprise_draft" not in picked:
            picked.append("base_t_enterprise_draft")  # 企业有效性过滤常需要
        return picked

    # ---------------- 阶段二：ReAct 分析 ----------------

    def _build_system_prompt(self, tables: list[str] | None) -> str:
        profiles = build_table_profiles(self._s, tables)
        return f"{SYSTEM_PROMPT}\n\n【各表数据画像（行数/列/示例）】\n{profiles}"

    def run(self, question: str) -> TabularAnswer:
        selected = self._select_tables(question)
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(selected)},
            {"role": "user", "content": f"【问题】\n{question}"},
        ]
        used_tables: set[str] = set(selected or [])

        for i in range(1, self._s.agent_max_iterations + 1):
            force_finish = i == self._s.agent_max_iterations
            if force_finish:
                messages.append(
                    {
                        "role": "user",
                        "content": "已达到迭代上限。请立即输出 Action: finish 并给出基于已有观察的最终答案；若确实无法得出，在答案中如实说明原因。",
                    }
                )

            msg = self._llm.chat_safe(messages, temperature=0.1)
            text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": text})

            m = _ACTION_RE.search(text)
            if not m:
                logger.warning("react_format_error", extra={"iter": i})
                messages.append({"role": "user", "content": _FORMAT_ERROR})
                continue

            action = m.group(1).lower()
            action_input = _strip_code_fence(m.group(2))

            if action == "finish":
                return TabularAnswer(
                    answer=action_input or "（模型未给出答案）",
                    tables_used=sorted(used_tables),
                    iterations=i,
                    converged=True,
                )

            if force_finish:
                # 最后一轮禁止再执行代码（最多 30s 沙箱执行，结果无处使用）
                messages.append(
                    {
                        "role": "user",
                        "content": "[LIMIT] 已达迭代上限，代码不再执行。请立即输出 Action: finish 给出基于已有观察的最终答案。",
                    }
                )
                continue

            used_tables.update(_DFS_RE.findall(action_input))
            observation = self._execute(action_input)
            logger.info(
                "tabular_step",
                extra={"iter": i, "ok": observation.startswith("[OK")},
            )
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        return TabularAnswer(
            answer="已达到最大迭代次数仍未获得最终答案，请缩小问题范围后重试。",
            tables_used=sorted(used_tables),
            iterations=self._s.agent_max_iterations,
            converged=False,
        )

    def _execute(self, code: str) -> str:
        try:
            res = run_code(
                code,
                parquet_dir=self._s.parquet_dir,
                timeout_s=self._s.sandbox_timeout_s,
            )
        except SandboxViolation as e:
            logger.warning("sandbox_violation", extra={"reason": e.message})
            return f"[REJECTED] {e.message}\n请按规则改写：只用 pd/np/dfs，不 import、不读写文件、不用下划线属性。"
        if res.ok:
            out = res.output.strip() or "（无输出）"
            return f"[OK]\n{out}"
        return (
            f"[ERROR] 代码执行失败，请阅读以下 traceback 修正代码后重试：\n{res.error}"
        )
