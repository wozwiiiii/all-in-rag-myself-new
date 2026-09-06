"""索引构建：文本提取 → 语义分块 → embedding（带缓存）→ Chroma 持久化。

构建与服务启动完全分离：`python -m scripts.build_index` 一次构建，
服务端只读加载，重启不再重复计算 embedding（缓存 + 持久化双重保证）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.rag.chunking import Chunk, SUPPORTED_EXTS, chunk_blocks, extract_file

logger = get_logger(__name__)

COLLECTION_NAME = "emergency_docs"


@dataclass
class IngestReport:
    files_indexed: int = 0
    chunks_indexed: int = 0
    skipped: list[str] = field(default_factory=list)


def _chunk_id(chunk: Chunk) -> str:
    raw = f"{chunk.source}|{chunk.heading_path}|{chunk.index}|{hashlib.md5(chunk.text.encode('utf-8')).hexdigest()[:8]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_index(settings: Settings, llm: LLMClient, *, force: bool = False) -> IngestReport:
    import chromadb

    report = IngestReport()

    files = sorted(
        p for p in settings.text_data_dir.rglob("*") if p.is_file()
    )
    all_chunks: list[Chunk] = []
    for path in files:
        rel = str(path.relative_to(settings.text_data_dir)).replace("\\", "/")
        if path.suffix.lower() not in SUPPORTED_EXTS:
            report.skipped.append(f"{rel}（不支持的格式 {path.suffix}，已知限制：.wps/.doc 暂无法解析）")
            continue
        try:
            blocks = extract_file(path)
        except Exception as e:  # 单文件失败不阻断整体构建
            logger.warning("ingest_extract_failed", extra={"file": rel, "err": str(e)})
            report.skipped.append(f"{rel}（解析失败: {e}）")
            continue
        chunks = chunk_blocks(blocks, source=rel)
        all_chunks.extend(chunks)
        report.files_indexed += 1
        logger.info("ingest_file", extra={"file": rel, "chunks": len(chunks)})

    if not all_chunks:
        raise RuntimeError(f"数据目录中没有可索引的文档: {settings.text_data_dir}")

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    # 构建即全量重建（幂等：embedding 有磁盘缓存，重复构建成本低）
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in all_chunks]
    vectors = llm.embed_documents(texts)
    collection.add(
        ids=[_chunk_id(c) for c in all_chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[
            {"source": c.source, "heading": c.heading_path, "chunk_index": c.index}
            for c in all_chunks
        ],
    )
    report.chunks_indexed = len(all_chunks)
    logger.info(
        "ingest_done",
        extra={"files": report.files_indexed, "chunks": report.chunks_indexed, "skipped": len(report.skipped)},
    )
    return report
