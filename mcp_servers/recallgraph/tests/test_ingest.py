"""解析层测试：章节识别、切块约束、PDF 端到端。

原则：能不用真 PDF 就不用；需要验证 PDF 路径时，用 matplotlib 现场生成一页纸，
不往仓库里塞二进制文件。
"""

from __future__ import annotations

import pytest

from recallgraph.ingest.chunker import ChunkSettings, build_chunks
from recallgraph.ingest.sections import Block, detect_heading, split_blocks
from recallgraph.ingest.source import PageText, strip_running_lines


def pages(*texts: str) -> list[PageText]:
    return [PageText(i, text) for i, text in enumerate(texts, start=1)]


# ── 章节识别 ────────────────────────────────────────────────────────────────


def test_detect_numbered_and_known_headings() -> None:
    assert detect_heading("3.2 Retrieval-Augmented Generation") == (2, "3.2 Retrieval-Augmented Generation")
    assert detect_heading("## Method", markdown=True) == (2, "Method")
    assert detect_heading("References") == (1, "References")


def test_sentence_starting_with_number_is_not_heading() -> None:
    assert detect_heading("2020. We propose a new method for retrieval.") is None
    assert detect_heading("This is a normal sentence that goes on and on and on for a while.") is None


def test_section_stack_tracks_nesting() -> None:
    text = "3 Experiments\nWe run three benchmarks.\n3.1 Setup\nWe use BEIR as the benchmark suite."
    result = split_blocks(pages(text))
    paths = [block.section_path for block in result.blocks]
    assert paths[0] == ("3 Experiments",)
    assert paths[1] == ("3 Experiments", "3.1 Setup")


def test_references_section_is_dropped() -> None:
    text = "1 Introduction\nBody text here.\nReferences\n[1] A paper. 2020."
    result = split_blocks(pages(text))
    assert [block.text for block in result.blocks] == ["Body text here."]
    assert any("参考文献" in warning for warning in result.warnings)


# ── 页眉页脚清理 ────────────────────────────────────────────────────────────


def test_repeated_running_header_is_removed() -> None:
    body = "Machine Learning Survey, 2024"
    cleaned = strip_running_lines(
        pages(
            f"{body}\nPage one content about retrieval.",
            f"{body}\nPage two content about ranking.",
            f"{body}\nPage three content about memory.",
        )
    )
    assert all(body not in page.text for page in cleaned.pages)
    assert "Page two content about ranking." in cleaned.pages[1].text
    assert any(body in line for line in cleaned.removed_lines)


# ── 切块 ────────────────────────────────────────────────────────────────────


def _block(text: str, page: int, section: tuple[str, ...] = ("Method",), start: int = 0) -> Block:
    return Block(
        text=text,
        page_start=page,
        page_end=page,
        section_path=section,
        char_start=start,
        char_end=start + len(text),
    )


def test_chunks_never_cross_sections() -> None:
    blocks = [
        _block("A" * 300, 1, ("Method",)),
        _block("B" * 300, 1, ("Results",), start=300),
    ]
    chunks = build_chunks(blocks, version_id="v1", title="T", settings=ChunkSettings(max_chars=1000))
    assert [chunk.section_path for chunk in chunks] == [["Method"], ["Results"]]


def test_chunks_never_span_more_than_one_page() -> None:
    blocks = [_block("A" * 200, 2), _block("B" * 200, 5, start=200)]
    chunks = build_chunks(blocks, version_id="v1", title="T")
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.page_end - chunk.page_start <= 1


def test_oversized_block_is_split_on_sentence_boundary() -> None:
    sentence = "这是用来测试切块边界的句子。" * 10
    blocks = [_block(sentence * 6, 1)]
    chunks = build_chunks(blocks, version_id="v1", title="T", settings=ChunkSettings(max_chars=500))
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)
    assert all(chunk.text.endswith("。") for chunk in chunks)


def test_context_header_carries_title_and_section() -> None:
    blocks = [_block("Some evidence about hybrid retrieval.", 1, ("3 Experiments", "3.1 Setup"))]
    chunk = build_chunks(blocks, version_id="v1", title="RAG Revisited")[0]
    assert chunk.context_header == "RAG Revisited ｜ 3 Experiments ｜ 3.1 Setup"


def test_chunk_ids_follow_version_and_seq() -> None:
    blocks = [_block("x" * 400, 1), _block("y" * 400, 1, ("Results",), start=400)]
    chunks = build_chunks(blocks, version_id="abc", title="T")
    assert [chunk.id for chunk in chunks] == [f"abc:{i:04d}" for i in range(len(chunks))]


# ── 端到端 ──────────────────────────────────────────────────────────────────


def test_ingest_markdown_note(tmp_path) -> None:
    from recallgraph.ingest.pipeline import ingest

    note = tmp_path / "note.md"
    note.write_text(
        "# 我的笔记\n\nRAG 的核心是检索与生成的解耦。\n\n## 待验证\n\n这条结论还没有证据。\n",
        encoding="utf-8",
    )
    result = ingest(note)
    assert result.version.page_count == 1
    assert result.document.title == "我的笔记"
    assert result.chunks
    assert all(chunk.text for chunk in result.chunks)


def test_ingest_pdf_end_to_end(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from recallgraph.ingest.pipeline import ingest

    path = tmp_path / "paper.pdf"
    with PdfPages(path) as pdf:
        for body in (
            "Hybrid Retrieval for Knowledge Agents\nAbstract\nWe study retrieval over papers.",
            "2 Method\nOur retriever combines BM25 with dense vectors.\nIt reranks the top candidates.",
        ):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.95
            for line in body.splitlines():
                fig.text(0.06, y, line, fontsize=11)
                y -= 0.04
            pdf.savefig(fig)
            plt.close(fig)

    result = ingest(path)
    assert result.version.page_count == 2
    assert len(result.version.content_hash) == 64
    assert result.chunks
    for chunk in result.chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 2
        assert chunk.page_end - chunk.page_start <= 1
        assert chunk.context_header
        assert chunk.char_end > chunk.char_start

# ── 真实 PDF 暴露出的噪声形态（回归） ────────────────────────────────────────


def test_axis_ticks_are_not_headings() -> None:
    assert detect_heading("10 20 30 40 50") is None
    assert detect_heading("BART 15.1 19.7 38.2 41.6 64.0 81.1") is None


def test_figure_text_block_is_dropped() -> None:
    figure = "KR e t r i e v e dD o c s 44NQ Exact Match RAG-Tok RAG-Seq"
    result = split_blocks(pages(f"1 Introduction\n{figure}"))
    assert result.dropped_blocks == 1
    assert "[0]" not in " ".join(block.text for block in result.blocks)


def test_table_row_block_is_dropped() -> None:
    row = "RAG-Tok. 17.3 22.2 40.1 41.5 72.5 89.5 14.7 21.4 40.8 44.2"
    result = split_blocks(pages(f"4 Results\n{row}"))
    assert result.dropped_blocks == 1


def test_wrapped_title_is_joined(tmp_path) -> None:
    from recallgraph.ingest.pipeline import ingest

    note = tmp_path / "paper.md"
    note.write_text("Retrieval-Augmented Generation for\nAnalysis of Knowledge Tasks\n", encoding="utf-8")
    assert ingest(note).document.title == "Retrieval-Augmented Generation for Analysis of Knowledge Tasks"


def test_pages_after_references_are_not_reported_missing(tmp_path) -> None:
    from recallgraph.ingest.pipeline import ingest

    note = tmp_path / "paper.md"
    note.write_text("1 Introduction\nSome body text about retrieval.\n\nReferences\n[1] Someone. 2020.\n", encoding="utf-8")
    warnings = ingest(note).warnings
    assert not any("没有任何 chunk" in warning for warning in warnings)


def test_table_rows_are_not_headings() -> None:
    assert detect_heading("18 layers 27.94 27.88") is None
    assert detect_heading("LAYERS # PARAMS") is None
    assert detect_heading("3.2 Identity Mapping by Shortcuts") == (2, "3.2 Identity Mapping by Shortcuts")
    assert detect_heading("3 方法") == (1, "3 方法")


def test_title_with_digits_is_still_a_heading() -> None:
    assert detect_heading("4.2. CIFAR-10 and Analysis") == (2, "4.2. CIFAR-10 and Analysis")


def test_numeric_prose_is_kept() -> None:
    prose = (
        "We construct 101-layer and 152-layer ResNets using 3-layer blocks. "
        "The 152-layer model has 11.3 billion FLOPs, lower than VGG-16/19 nets (15.3/19.6 billion). "
        "Results improve by 2.1 points on ImageNet top-1 accuracy."
    )
    result = split_blocks(pages(f"4.2. CIFAR-10 and Analysis\n{prose}"))
    assert result.dropped_blocks == 0
    assert any("11.3 billion" in block.text for block in result.blocks)
