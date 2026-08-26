"""索引构建 CLI（与服务启动分离）：

    python -m scripts.build_index           # 常规构建（embedding 有缓存，增量快）
    python -m scripts.build_index --force   # 强制全量重建
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.llm.client import LLMClient
from app.rag.ingest import build_index
from app.tabular.loader import convert_excel_to_parquet


def main() -> int:
    setup_logging()
    logger = get_logger("build_index")
    parser = argparse.ArgumentParser(description="构建向量索引与 parquet 表数据")
    parser.add_argument("--force", action="store_true", help="忽略缓存与已有产物，全量重建")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openai_api_key:
        print("错误: 未配置 OPENAI_API_KEY（请复制 .env.example 为 .env 并填写）", file=sys.stderr)
        return 1

    convert_excel_to_parquet(settings, force=args.force)
    llm = LLMClient(settings)
    report = build_index(settings, llm, force=args.force)

    print(f"\n构建完成：{report.files_indexed} 个文件，{report.chunks_indexed} 个分块")
    if report.skipped:
        print("跳过的文件：")
        for s in report.skipped:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
