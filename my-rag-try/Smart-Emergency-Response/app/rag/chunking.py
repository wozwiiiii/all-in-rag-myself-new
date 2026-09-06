"""文档提取与语义分块：按标题层级切 section，超长 section 按段落二次切分（带重叠）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTS = {".docx", ".pdf"}

MAX_CHARS = 1200
OVERLAP_CHARS = 150


@dataclass
class Block:
    level: int  # 0 = 正文；1-4 = 标题层级
    text: str


@dataclass
class Chunk:
    text: str
    source: str  # 相对数据目录的文件路径
    heading_path: str  # 形如 "某文件 > 第一章 > 1.2 节"
    index: int = 0
    tags: list[str] = field(default_factory=list)


def extract_docx(path: Path) -> list[Block]:
    import docx

    doc = docx.Document(str(path))
    blocks: list[Block] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style is not None else ""
        if style.startswith("Heading"):
            digits = "".join(c for c in style.split()[-1] if c.isdigit())
            level = min(int(digits) if digits else 1, 4)
            blocks.append(Block(level=level, text=text))
        else:
            blocks.append(Block(level=0, text=text))
    # 表格按行拼接为正文（保留结构信息）
    for table in doc.tables:
        lines = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            lines.append(" | ".join(cells))
        if lines:
            blocks.append(Block(level=0, text="\n".join(lines)))
    return blocks


def extract_pdf(path: Path) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(Block(level=0, text=text))
        else:
            # 扫描版页面（无文本层）→ OCR 兜底；依赖缺失时跳过并记录
            ocr_text = _ocr_pdf_page(path, i)
            if ocr_text:
                blocks.append(Block(level=0, text=ocr_text))
    return blocks


_OCR_ENGINE = None  # 模型加载昂贵，进程内单例复用


def _ocr_pdf_page(path: Path, page_index: int, *, scale: float = 2.0) -> str:
    """栅格化单页并 OCR。pypdfium2 渲染 + RapidOCR（PP-OCRv4 中文模型，CPU）。

    依赖为可选安装（pip install ".[ocr]"），缺失时返回空串并只打一次警告，
    不阻断索引构建（该页按无内容处理）。
    """
    global _OCR_ENGINE
    try:
        import numpy as np  # noqa: F401
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "ocr_deps_missing",
            extra={"file": str(path), "page": page_index},
        )
        return ""
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    page = pdfium.PdfDocument(str(path))[page_index]
    pil = page.render(scale=scale).to_pil()
    result, _ = _OCR_ENGINE(np.asarray(pil)[:, :, ::-1].copy())  # RGB->BGR
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def extract_file(path: Path) -> list[Block]:
    if path.suffix.lower() == ".docx":
        return extract_docx(path)
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"不支持的文件类型: {path.suffix}")


def _merge_overflow(paragraphs: list[str], heading: str, start: int) -> list[tuple[str, int]]:
    """将超长 section 的段落聚合为多个 (text, chunk_no)，相邻块间保留尾部重叠。"""
    pieces: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_len = 0
    no = start
    for p in paragraphs:
        # 单段自身超长：硬切
        if len(p) > MAX_CHARS:
            for i in range(0, len(p), MAX_CHARS - OVERLAP_CHARS):
                pieces.append((p[i : i + MAX_CHARS], no))
                no += 1
            continue
        if buf_len + len(p) > MAX_CHARS and buf:
            pieces.append(("\n".join(buf), no))
            no += 1
            tail = buf[-1]
            buf = [tail[-OVERLAP_CHARS:]] if len(tail) > OVERLAP_CHARS else [tail]
            buf_len = len(buf[0])
        buf.append(p)
        buf_len += len(p) + 1
    if buf:
        pieces.append(("\n".join(buf), no))
    return pieces


def chunk_blocks(blocks: list[Block], source: str) -> list[Chunk]:
    """按标题层级分块；每个 chunk 记录完整标题路径，便于引用来源与后续父子检索扩展。"""
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    current_paras: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal current_paras, counter
        if not current_paras:
            return
        heading = " > ".join(t for _, t in heading_stack) or source
        body = "\n".join(current_paras)
        if len(body) <= MAX_CHARS:
            chunks.append(Chunk(text=body, source=source, heading_path=heading, index=counter))
            counter += 1
        else:
            for text, _ in _merge_overflow(current_paras, heading, counter):
                chunks.append(Chunk(text=text, source=source, heading_path=heading, index=counter))
                counter += 1
        current_paras = []

    for block in blocks:
        if block.level > 0:
            flush()
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.text))
        else:
            current_paras.append(block.text)
    flush()
    return chunks
