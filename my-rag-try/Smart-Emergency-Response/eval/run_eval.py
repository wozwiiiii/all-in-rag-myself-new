"""评测 CLI：跑通整条问答链路并产出 markdown 报告，用于改动前后对比（A/B）。

    python eval/run_eval.py                  # 全量
    python eval/run_eval.py --limit 5        # 前 5 题（冒烟）
    python eval/run_eval.py --category table # 只跑表格类
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
from eval.metrics import score_answer  # noqa: E402

logger = get_logger("eval")


def load_dataset() -> list[dict]:
    items = []
    with open(EVAL_DIR / "dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run(limit: int | None, category: str | None) -> int:
    setup_logging()
    settings = get_settings()
    if not settings.openai_api_key:
        print("错误: 未配置 OPENAI_API_KEY", file=sys.stderr)
        return 1

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
        try:
            qa = await pipeline.answer(item["question"])
            answer, refused, route = qa.answer, qa.refused, qa.route
        except Exception as e:
            logger.error("eval_item_failed", extra={"id": item["id"], "err": str(e)})
            answer, refused, route = f"执行失败: {e}", False, "error"
        elapsed = int((time.monotonic() - t0) * 1000)

        scores = score_answer(
            answer=answer,
            refused=refused,
            keywords=item.get("keywords", []),
            reference=item.get("reference"),
            llm=llm,
        )
        scores.update(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": answer[:300],
                "expected_route": item["category"],
                "actual_route": route,
                "latency_ms": elapsed,
            }
        )
        results.append(scores)
        print(f"[{item['id']:>2}] {item['category']:<7} total={scores['total']:.3f} "
              f"route={route} {elapsed}ms")

    write_report(results)
    mean = sum(r["total"] for r in results) / max(len(results), 1)
    print(f"\n平均分: {mean:.4f}（{len(results)} 题）")
    return 0


def write_report(results: list[dict]) -> None:
    reports_dir = EVAL_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"report-{ts}.md"

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    lines = [
        "# 评测报告",
        f"- 时间: {ts}",
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
        routed = sum(1 for r in rows if r["actual_route"] == r["expected_route"])
        lines.append(f"| {cat} | {len(rows)} | {mean:.3f} | {routed}/{len(rows)} |")

    lines += ["", "## 明细", "", "| ID | 类别 | 得分 | 路由 | 耗时 |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['total']:.3f} | "
            f"{r['actual_route']}（期望 {r['expected_route']}） | {r['latency_ms']}ms |"
        )

    lines += ["", "## 低分用例（< 0.5）", ""]
    for r in results:
        if r["total"] < 0.5:
            lines.append(f"### [{r['id']}] {r['question']}")
            lines.append(f"- 路由: {r['actual_route']}，拒答: {r['refused']}，得分: {r['total']}")
            lines.append(f"- 答案摘录: {r['answer'][:200]}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", choices=["text", "table", "hybrid", "refuse"], default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.limit, args.category)))
