"""评测 CLI：跑通整条问答链路并产出 markdown 报告，用于改动前后对比（A/B）。

    python eval/run_eval.py                  # 全量（默认口径 v3）
    python eval/run_eval.py --limit 5        # 前 5 题（冒烟）
    python eval/run_eval.py --category table # 只跑表格类
    python eval/run_eval.py --metrics v2     # 旧口径（历史报告对比用）

口径 v3（2026-09-06）：0.3*keyword + 0.2*semantic + 0.35*numeric + 0.15*faithful。
numeric = 结论数字一致性（去噪后比对"X 家/Y 条"类数量结论）；
双方结论数字完全无交集（数值错）时总分封顶 ≤0.4（中文数字改写豁免）。
动因：v2 下表格题数值全错仍拿 0.81-0.90（keyword 0.6 权重过高 + 全量数字
抽取被日期/字段赋值稀释 + faithful 0.15 太低），错题与正确题无区分度。
已知局限：规则版数字核对查不到改写/区间/单位换算/中文数字（封顶豁免）；
refuse 类不适用。v3 与 v1/v2 总分不可横比。
口径 v2（2026-09-06）：0.6*keyword + 0.25*semantic + 0.15*faithful，
faithful = 证据一致性 × reference 数字一致性 × (1-冗余惩罚)。
与 v1 历史报告对比时注意口径不同，总分不可直接横比。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.orchestrator.pipeline import QAPipeline  # noqa: E402
from eval.metrics import (  # noqa: E402
    score_answer,
    score_answer_v2,
    score_answer_v3,
)

logger = get_logger("eval")


def load_dataset() -> list[dict]:
    items = []
    with open(EVAL_DIR / "dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run(limit: int | None, category: str | None, metrics: str = "v3") -> int:
    setup_logging()
    settings = get_settings()
    if not settings.openai_api_key:
        print("错误: 未配置 OPENAI_API_KEY", file=sys.stderr)
        return 1

    scorers = {"v3": score_answer_v3, "v2": score_answer_v2, "v1": score_answer}
    scorer = scorers[metrics]

    dataset = load_dataset()
    if category:
        dataset = [d for d in dataset if d["category"] == category]
    if limit:
        dataset = dataset[:limit]

    llm = LLMClient(settings)
    pipeline = QAPipeline(settings, llm)

    results = []
    for item in dataset:
        t0 = time.monotonic()
        evidence = ""
        try:
            qa = await pipeline.answer(item["question"])
            answer, refused, route = qa.answer, qa.refused, qa.route
            evidence = getattr(qa, "evidence", "") or ""
        except Exception as e:
            logger.error("eval_item_failed", extra={"id": item["id"], "err": str(e)})
            answer, refused, route = f"执行失败: {e}", False, "error"
        elapsed = int((time.monotonic() - t0) * 1000)

        score_kwargs = dict(
            answer=answer,
            refused=refused,
            keywords=item.get("keywords", []),
            reference=item.get("reference"),
            llm=llm,
            expected_route=item.get("expected_route"),
        )
        if metrics in ("v2", "v3"):  # v1 的 score_answer 无 evidence 参数
            score_kwargs["evidence"] = evidence
        scores = scorer(**score_kwargs)
        scores.update(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": answer[:300],
                "expected_route": item.get("expected_route", item["category"]),
                "actual_route": route,
                "latency_ms": elapsed,
            }
        )
        results.append(scores)
        print(f"[{item['id']:>2}] {item['category']:<7} total={scores['total']:.3f} "
              f"route={route} {elapsed}ms")

    write_report(results, metrics)
    mean = sum(r["total"] for r in results) / max(len(results), 1)
    print(f"\n平均分: {mean:.4f}（{len(results)} 题，口径 {metrics}）")
    return 0


def _route_correct(r: dict) -> bool:
    """路由判定：普通题比 actual/expected；refuse 题 router 结构上不输出
    refuse 路由（拒答发生在下游），故以 refused 标志判定。"""
    if r["expected_route"] == "refuse":
        return bool(r.get("refused"))
    return r["actual_route"] == r["expected_route"]


def write_report(results: list[dict], metrics: str = "v3") -> None:
    reports_dir = EVAL_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"report-{ts}.md"

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    metric_desc = {
        "v3": ("0.3*keyword + 0.2*semantic + 0.35*numeric + 0.15*faithful；"
               "numeric=结论数字一致性（去噪后比对数量结论）；结论数字完全无交集（数值错）"
               "时总分封顶 ≤0.4（中文数字改写豁免）。与 v1/v2 总分不可横比。"),
        "v2": ("0.6*keyword + 0.25*semantic + 0.15*faithful；"
               "faithful=证据一致性×reference数字一致性×(1-冗余惩罚)。规则版局限：查不到数字改写/区间/单位换算/中文数字"),
        "v1": "0.7*keyword + 0.3*semantic（历史口径）",
    }
    lines = [
        "# 评测报告",
        f"- 时间: {ts}",
        f"- 评分口径: {metrics}（{metric_desc.get(metrics, '未知')}）",
        f"- 题目数: {len(results)}",
        f"- 总平均分: {sum(r['total'] for r in results) / max(len(results), 1):.4f}",
        "",
        "## 分类汇总",
        "",
        "| 类别 | 题数 | 平均分 | 路由正确数 |",
        "|---|---|---|---|",
    ]
    for cat, rows in sorted(by_cat.items()):
        mean = sum(r["total"] for r in rows) / len(rows)
        routed = sum(1 for r in rows if _route_correct(r))
        lines.append(f"| {cat} | {len(rows)} | {mean:.3f} | {routed}/{len(rows)} |")

    is_v2 = "faithful" in (results[0] if results else {})
    is_v3 = "numeric" in (results[0] if results else {})
    if is_v2:
        header = (
            "| ID | 类别 | 得分 | numeric | faithful | 冗余 | 路由 | 路由判定 | 耗时 |"
            if is_v3
            else "| ID | 类别 | 得分 | faithful | 冗余 | 路由 | 路由判定 | 耗时 |"
        )
        lines += ["", "## 明细", "", header, "|---" * (9 if is_v3 else 8) + "|"]
        for r in results:
            row = (
                f"| {r['id']} | {r['category']} | {r['total']:.3f} | "
                f"{r.get('numeric', 0):.3f} | {r.get('faithful', 0):.3f} | "
                f"{r.get('redundancy', 0):.2f} | "
            ) if is_v3 else (
                f"| {r['id']} | {r['category']} | {r['total']:.3f} | "
                f"{r.get('faithful', 0):.3f} | {r.get('redundancy', 0):.2f} | "
            )
            lines.append(
                row
                + f"{r['actual_route']}（期望 {r['expected_route']}） | {'✓' if _route_correct(r) else '✗'} | {r['latency_ms']}ms |"
            )
    else:
        lines += ["", "## 明细", "", "| ID | 类别 | 得分 | 路由 | 路由判定 | 耗时 |", "|---|---|---|---|---|---|"]
        for r in results:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['total']:.3f} | "
                f"{r['actual_route']}（期望 {r['expected_route']}） | {'✓' if _route_correct(r) else '✗'} | {r['latency_ms']}ms |"
            )

    lines += ["", "## 低分用例（< 0.5）", ""]
    for r in results:
        if r["total"] < 0.5:
            lines.append(f"### [{r['id']}] {r['question']}")
            lines.append(f"- 路由: {r['actual_route']}，拒答: {r['refused']}，得分: {r['total']}")
            if is_v2 and r["category"] != "refuse":
                lines.append(
                    f"- 分项: keyword={r.get('keyword', 0):.3f} semantic={r.get('semantic', 0):.3f} "
                    f"numeric={r.get('numeric', 0):.3f} faithful={r.get('faithful', 0):.3f}"
                    f"（冗余罚 {r.get('redundancy', 0):.2f}）"
                )
            lines.append(f"- 答案摘录: {r['answer'][:200]}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", choices=["text", "table", "hybrid", "refuse"], default=None)
    parser.add_argument(
        "--metrics", choices=["v1", "v2", "v3"], default="v3",
        help="评分口径（默认 v3；v2/v1 供历史对比）",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.limit, args.category, args.metrics)))
