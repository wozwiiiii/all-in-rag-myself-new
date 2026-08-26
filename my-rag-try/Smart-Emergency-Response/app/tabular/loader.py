"""Excel 预转换：xlsx 一次性转 parquet，服务与沙箱后续只读 parquet。

xlsx 解析慢（openpyxl），parquet 读取快一个数量级；且沙箱子进程按表惰性加载，
避免每次执行都重复解析全部 17 张表。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def convert_excel_to_parquet(settings: Settings, *, force: bool = False) -> list[Path]:
    import pandas as pd

    settings.parquet_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    xlsx_files = sorted(settings.table_data_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise RuntimeError(f"表数据目录中没有 xlsx 文件: {settings.table_data_dir}")

    for src in xlsx_files:
        dst = settings.parquet_dir / f"{src.stem}.parquet"
        if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        df = pd.read_excel(src)
        df.to_parquet(dst, index=False)
        converted.append(dst)
        logger.info("excel_converted", extra={"table": src.stem, "rows": len(df), "cols": len(df.columns)})

    all_parquets = sorted(settings.parquet_dir.glob("*.parquet"))
    logger.info("parquet_ready", extra={"tables": len(all_parquets), "converted_now": len(converted)})
    return all_parquets


def table_names(settings: Settings) -> list[str]:
    return sorted(p.stem for p in settings.parquet_dir.glob("*.parquet"))
