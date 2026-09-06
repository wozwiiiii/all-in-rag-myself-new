"""双协议对比 bench：同一模型、同一温度下 react 与 tool_calls 的实测对照。

    python eval/bench_protocol.py --model Qwen/Qwen2.5-72B-Instruct          # 默认 6 道 table 题
    python eval/bench_protocol.py --model Qwen/Qwen2.5-7B-Instruct --limit 3 # 冒烟
    python eval/bench_protocol.py --ids 9,10                                 # 指定题目
    python eval/bench_protocol.py --category hybrid                          # 换题型（默认 table）

设计要点（决定数据能不能上简历，务必保持）：
1. 同一题先后用两种协议各跑一遍，共享同一个 LLMClient、同一个模型、同一温度（0.1），
   只有"动作解析/下发"这一层不同；
2. 直接调 TabularAgent.run()，不经过 QAPipeline —— 后者会走意图路由、并可能并行两路，
   既干扰计时也让 token 无法归因到具体协议；
3. token 用 LLMClient.usage_snapshot()/usage_delta() 做差值（客户端已有累加器）；
4. 跑挂的题如实记为 error 并计入统计分母，**不删除、不重跑挑好的、不美化口径**；
   report 里每个数字都来自本次真实执行。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.tabular.agent import TabularAgent  # noqa: E402
from app.tabular.loader import convert_excel_to_parquet  # noqa: E402

logger = get_logger("bench")

PROTOCOLS = ("react", "tool_calls")
DEFAULT_CATEGORY = "table"
# 与 TabularAgent 内部一致：分析轮 temperature=0.1、选表轮 0.0
ANALYSIS_TEMPERATURE = 0.1


@dataclass
class Row:
    id: int
    question: str
    protocol: str
    converged: bool = False
    iterations: int = 0
    format_errors: int = 0
    tool_errors: int = 0
    llm_calls: int = 0
    elapsed_ms: int = 0
    wall_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_calls: int = 0
    answer: str = ""
    error: str = ""
    observations: list[str] = field(default_factory=list)


def load_dataset(category: str, ids: list[int] | None, limit: int | None) -> list[dict]:
    items: list[dict] = []
    with open(EVAL_DIR / "dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("category") != category:
                continue
            if ids and item["id"] not in ids:
                continue
            items.append(item)
    if limit:
        items = items[:limit]
    return items


def run_one(settings, llm: LLMClient, item: dict, protocol: str) -> Row:
    """跑单题单协议；异常不吞也不重试，如实落进 Row.error。"""
    row = Row(id=item["id"], question=item["question"], protocol=protocol)
    before = llm.usage_snapshot()
    t0 = time.monotonic()
    try:
        ans = TabularAgent(settings, llm, protocol=protocol).run(item["question"])
        row.converged = ans.converged
        row.iterations = ans.iterations
        row.format_errors = ans.format_errors
        row.tool_errors = ans.tool_errors
        row.llm_calls = ans.llm_calls
        row.elapsed_ms = ans.elapsed_ms
        row.answer = ans.answer
        row.observations = list(ans.observations)
    except Exception as e:  # noqa: BLE001 - bench 需要记录任何失败而不是中断整轮
        row.error = f"{type(e).__name__}: {str(e)[:300]}"
        logger.error("bench_item_failed", extra={"id": item["id"], "protocol": protocol, "err": row.error})
    row.wall_ms = int((time.monotonic() - t0) * 1000)

    delta = llm.usage_delta(before)
    row.prompt_tokens = delta["prompt_tokens"]
    row.completion_tokens = delta["completion_tokens"]
    row.total_tokens = delta["total_tokens"]
    row.usage_calls = delta["calls"]
    return row


def _mean(rows: list[Row], attr: str) -> float:
    return sum(getattr(r, attr) for r in rows) / len(rows) if rows else 0.0


def _rate(rows: list[Row]) -> float:
    return sum(1 for r in rows if r.converged) / len(rows) if rows else 0.0


def _esc(text: str) -> str:
    """markdown 表格内换行/竖线转义，避免答案把表格撑坏。"""
    return text.replace("|", "\\|").replace("\n", " <br> ")


def build_report(rows: list[Row], model: str, category: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    by_proto: dict[str, list[Row]] = {p: [r for r in rows if r.protocol == p] for p in PROTOCOLS}
    question_ids: list[int] = []
    for r in rows:
        if r.id not in question_ids:
            question_ids.append(r.id)

    lines = [
        "# 表格代理双协议对比（react vs tool_calls）",
        "",
        f"- 生成时间: {ts}",
        f"- 模型: `{model}`",
        f"- 题型: {category}（题目 id: {', '.join(str(i) for i in question_ids)}）",
        f"- 温度: 分析轮 {ANALYSIS_TEMPERATURE} / 选表轮 0.0（两协议一致）",
        "- 调用方式: 直接 `TabularAgent.run()`，不经 QAPipeline（避免路由与并行干扰计时与 token 归因）",
        "- 同一题先后跑两协议，共享同一 LLMClient；token 由 `usage_snapshot()/usage_delta()` 差值得到",
        "- 口径说明: `format_errors` = 协议解析失败（react 正则未命中 / fc 无 tool_calls 且无 content 或参数 JSON 非法）；"
        "`tool_errors` = 沙箱执行失败（[ERROR] 异常 / [REJECTED] 越权）；两者互斥统计",
        "",
        "## 汇总对比",
        "",
        "| 协议 | 题数 | 收敛率 | 平均迭代轮次 | 平均格式错误 | 平均工具错误 | 平均延迟(ms) | 平均总 token | 平均 LLM 调用 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in PROTOCOLS:
        rs = by_proto[p]
        if not rs:
            continue
        lines.append(
            f"| {p} | {len(rs)} | {_rate(rs) * 100:.1f}% | {_mean(rs, 'iterations'):.2f} | "
            f"{_mean(rs, 'format_errors'):.2f} | {_mean(rs, 'tool_errors'):.2f} | "
            f"{_mean(rs, 'elapsed_ms'):.0f} | {_mean(rs, 'total_tokens'):.0f} | {_mean(rs, 'llm_calls'):.2f} |"
        )

    lines += [
        "",
        "## 逐题明细",
        "",
        "| id | 协议 | 收敛 | 迭代 | 格式错误 | 工具错误 | LLM 调用 | 延迟(ms) | 总 token | 答案摘要 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        summary = _esc(r.answer[:80]) if not r.error else f"**失败** {_esc(r.error[:80])}"
        lines.append(
            f"| {r.id} | {r.protocol} | {'✓' if r.converged else '✗'} | {r.iterations} | "
            f"{r.format_errors} | {r.tool_errors} | {r.llm_calls} | {r.elapsed_ms} | "
            f"{r.total_tokens} | {summary} |"
        )

    lines += ["", "## 答案文本对照", ""]
    for qid in question_ids:
        same_q = [r for r in rows if r.id == qid]
        question = same_q[0].question if same_q else ""
        lines += [f"### [{qid}] {question}", ""]
        for p in PROTOCOLS:
            r = next((x for x in same_q if x.protocol == p), None)
            if r is None:
                continue
            body = _esc(r.error or r.answer or "（空答案）") if not r.error else f"**执行失败**：{_esc(r.error)}"
            lines += [
                f"- **{p}**（收敛={'是' if r.converged else '否'}，迭代 {r.iterations}，"
                f"格式错误 {r.format_errors}，工具错误 {r.tool_errors}，延迟 {r.elapsed_ms}ms）",
                f"  - 答案：{body}",
            ]
            if r.observations:
                lines.append(f"  - 末轮证据：`{_esc(r.observations[-1][:300])}`")
        lines.append("")

    lines += [
        "## 说明",
        "",
        "- 表中所有数字均来自本次脚本真实执行；失败用例以「执行失败」原样保留，未剔除、未重跑挑选。",
        "- 延迟为 `TabularAgent.run()` 内部计时（含选表与沙箱执行），wall clock 另记于脚本日志。",
        "- 单次运行样本量小（默认 6 题），只能看趋势，不能当显著性结论；需要更稳的结论请多次重复。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="react vs tool_calls 双协议对比 bench")
    parser.add_argument("--model", default=None, help="覆盖 settings.openai_api_model（如 Qwen/Qwen2.5-72B-Instruct）")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="dataset.jsonl 的 category 过滤（默认 table）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--ids", default=None, help="只跑指定 id，逗号分隔，如 9,10")
    parser.add_argument("--timeout", type=float, default=None,
                        help="覆盖 LLM 单次调用超时（秒）；慢模型（如 7B 在拥挤网关）可适当调大")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    if args.model:
        settings.openai_api_model = args.model
    if args.timeout:
        settings.llm_timeout_s = args.timeout
    if not settings.openai_api_key:
        print("错误: 未配置 OPENAI_API_KEY（项目根 .env）", file=sys.stderr)
        return 1

    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    dataset = load_dataset(args.category, ids, args.limit)
    if not dataset:
        print("错误: 没有匹配的题目", file=sys.stderr)
        return 1

    # 表格路只需要 parquet；缺则现转（不跑 build_index，避免烧 embedding）
    if not sorted(settings.parquet_dir.glob("*.parquet")):
        convert_excel_to_parquet(settings, force=False)

    llm = LLMClient(settings)
    print(
        f"模型={settings.openai_api_model} 题型={args.category} 题数={len(dataset)} "
        f"协议={'/'.join(PROTOCOLS)} 迭代上限={settings.agent_max_iterations}",
        flush=True,
    )

    rows: list[Row] = []
    for item in dataset:
        for protocol in PROTOCOLS:
            row = run_one(settings, llm, item, protocol)
            rows.append(row)
            print(
                f"[{row.id:>2}] {protocol:<10} converged={row.converged} iter={row.iterations} "
                f"fmt_err={row.format_errors} tool_err={row.tool_errors} calls={row.llm_calls} "
                f"{row.elapsed_ms}ms tok={row.total_tokens}"
                + (f" ERROR={row.error[:80]}" if row.error else ""),
                flush=True,
            )

    for p in PROTOCOLS:
        rs = [r for r in rows if r.protocol == p]
        if rs:
            print(
                f"== {p}: 收敛率 {_rate(rs) * 100:.1f}% 平均迭代 {_mean(rs, 'iterations'):.2f} "
                f"平均格式错误 {_mean(rs, 'format_errors'):.2f} 平均工具错误 {_mean(rs, 'tool_errors'):.2f} "
                f"平均延迟 {_mean(rs, 'elapsed_ms'):.0f}ms 平均 token {_mean(rs, 'total_tokens'):.0f}",
                flush=True,
            )

    reports_dir = EVAL_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = reports_dir / f"protocol-bench-{ts}.md"
    out.write_text(build_report(rows, settings.openai_api_model, args.category), encoding="utf-8")
    print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
