"""分块逻辑测试：标题路径保留、超长切分、docx 提取。"""

from pathlib import Path

from app.rag.chunking import Block, Chunk, chunk_blocks, extract_docx


def test_heading_path_preserved():
    blocks = [
        Block(level=1, text="第一章 总则"),
        Block(level=0, text="为了加强安全管理，制定本规范。"),
        Block(level=2, text="适用范围"),
        Block(level=0, text="适用于危险化学品企业。"),
    ]
    chunks = chunk_blocks(blocks, source="a.docx")
    assert len(chunks) == 2
    assert chunks[0].heading_path == "第一章 总则"
    assert chunks[1].heading_path == "第一章 总则 > 适用范围"
    assert all(c.source == "a.docx" for c in chunks)


def test_long_section_split_with_overlap():
    paras = [f"第{i}段落内容，" + "字" * 200 for i in range(10)]
    blocks = [Block(level=1, text="标题"), *[Block(level=0, text=p) for p in paras]]
    chunks = chunk_blocks(blocks, source="b.pdf")
    assert len(chunks) > 1
    # 均不超过上限（拼接单段硬切场景下有 overlap 冗余，允许少量超出）
    for c in chunks:
        assert len(c.text) <= 1200 + 200


def test_empty_blocks_no_chunks():
    assert chunk_blocks([], source="x") == []


def test_extract_docx(tmp_path: Path):
    import docx

    doc = docx.Document()
    doc.add_heading("规章标题", level=1)
    doc.add_paragraph("正文内容一。")
    doc.add_heading("细则", level=2)
    doc.add_paragraph("正文内容二。")
    path = tmp_path / "sample.docx"
    doc.save(str(path))

    blocks = extract_docx(path)
    assert (blocks[0].level, blocks[0].text) == (1, "规章标题")
    assert any(b.level == 0 and "正文内容一" in b.text for b in blocks)

    chunks = chunk_blocks(blocks, source="sample.docx")
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
