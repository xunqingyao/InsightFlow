"""解析层主管线：文件 → Document / DocumentVersion / Chunk。

这是整个系统唯一会读原文的地方。跑完之后，后续所有环节只碰 chunks 表，
不再打开 PDF——这是"离线编译、在线查询"的分界线。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from recallgraph.ingest.chunker import ChunkSettings, build_chunks
from recallgraph.ingest.sections import split_blocks
from recallgraph.ingest.source import PageText, extract_pages, strip_running_lines
from recallgraph.schema import Chunk, Document, DocumentVersion
from recallgraph.vocabulary import DocType

PARSER_VERSION = "recallgraph-pdf-0.1"

_ARXIV_HEADER = ("arxiv:", "doi:", "http://", "https://", "@", "proceedings of", "preprint")


class IngestResult(BaseModel):
    """一次解析的全部产物。可以直接落库，也可以先给人看一眼。"""

    document: Document
    version: DocumentVersion
    chunks: list[Chunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)


def ingest(
    path: str | Path,
    *,
    chunk_settings: ChunkSettings | None = None,
    stop_at_references: bool = True,
) -> IngestResult:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    pages = extract_pages(file_path)
    if not pages:
        raise ValueError(f"没能从 {file_path.name} 里读到任何页")

    cleaning = strip_running_lines(pages)
    clean_pages = list(cleaning.pages)
    is_markdown = file_path.suffix.lower() in {".md", ".markdown"}
    blocks_result = split_blocks(
        clean_pages,
        stop_at_references=stop_at_references,
        markdown=is_markdown,
    )

    title = guess_title(clean_pages, file_path)
    document = Document(
        title=title,
        doc_type=DocType.PAPER if file_path.suffix.lower() == ".pdf" else DocType.NOTE,
    )
    version = DocumentVersion(
        document_id=document.id,
        content_hash=digest,
        file_path=str(file_path),
        page_count=len(pages),
        parser_version=PARSER_VERSION,
    )
    chunks = build_chunks(
        blocks_result.blocks,
        version_id=version.id,
        title=title,
        settings=chunk_settings,
    )

    warnings = list(blocks_result.warnings)
    if blocks_result.dropped_blocks:
        warnings.append(f"已跳过 {blocks_result.dropped_blocks} 段疑似表格/图表乱码（不含正文）")
    warnings.extend(_coverage_warnings(clean_pages, chunks, blocks_result.stopped_at_page))
    if cleaning.removed_lines:
        warnings.append(f"已删除 {len(cleaning.removed_lines)} 条跨页重复的页眉/页脚")

    total_chars = sum(len(chunk.text) for chunk in chunks)
    return IngestResult(
        document=document,
        version=version,
        chunks=chunks,
        warnings=warnings,
        stats={
            "pages": float(len(pages)),
            "blocks": float(len(blocks_result.blocks)),
            "chunks": float(len(chunks)),
            "chars": float(total_chars),
            "avg_chunk_chars": round(total_chars / len(chunks), 1) if chunks else 0.0,
            "chars_per_page": round(total_chars / len(pages), 1),
        },
    )


def guess_title(pages: list[PageText], file_path: Path) -> str:
    """取标题：Markdown 用一级标题；PDF 用首页第一行"像标题"的文本。

    "像标题"的排除项：arXiv 头、链接、邮箱、全大写短词（英文论文的页眉常见）、
    过短或过长的行（前者是页码/编号，后者是正文段落）。
    """
    first_page = pages[0].text
    for line in first_page.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if 2 <= len(heading) <= 200:
                return heading

    lines = [" ".join(line.split()) for line in first_page.splitlines()]
    for index, candidate in enumerate(lines):
        if not (8 <= len(candidate) <= 200):
            continue
        lowered = candidate.lower()
        if any(lowered.startswith(prefix) or prefix in lowered for prefix in _ARXIV_HEADER):
            continue
        if candidate.isascii() and candidate.isupper() and len(candidate.split()) <= 3:
            continue
        return _continue_title(candidate, lines[index + 1 :])
    return file_path.stem


_DANGLING = {"for", "of", "and", "the", "in", "on", "to", "with", "via", "using", "a", "an", "towards", "about"}


def _continue_title(title: str, following: list[str]) -> str:
    """标题被 PDF 折行时补回下一行，例如 "… for" + "Analysis"。

    只在当前行以虚词结尾、下一行足够短且不以标点开头时才拼接，避免把作者行并进标题。
    """
    words = title.split()
    if not words or words[-1].lower().strip(":,") not in _DANGLING:
        return title
    for line in following:
        if not line or len(line) > 80:
            break
        if line[0] in ",.;:(" or line.lower().startswith(("abstract", "†", "‡")):
            break
        return f"{title} {line}"[:200]
    return title


def _coverage_warnings(
    pages: list[PageText], chunks: list[Chunk], stopped_at_page: int | None
) -> list[str]:
    """报告"有页没进 chunk"。参考文献之后的页面不算缺失，那是主动跳过的。"""
    warnings: list[str] = []
    if stopped_at_page:
        pages = [page for page in pages if page.page < stopped_at_page]
    empty_pages = [page.page for page in pages if not page.text.strip()]
    if empty_pages:
        sample = ", ".join(str(p) for p in empty_pages[:5])
        more = " 等" if len(empty_pages) > 5 else ""
        warnings.append(f"有 {len(empty_pages)} 页没有抽出文本（第 {sample}{more} 页），可能是扫描页或图表页")
    if pages and chunks:
        covered = {page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)}
        missing = sorted({page.page for page in pages} - covered)
        if missing:
            warnings.append(f"有 {len(missing)} 页没有任何 chunk（第 {', '.join(map(str, missing[:5]))} 页）")
    return warnings


def summarize(result: IngestResult) -> str:
    lines = [
        f"文档：{result.document.title}",
        f"版本：{result.version.id}  hash={result.version.content_hash[:12]}…  页数={result.version.page_count}",
        (
            f"切块：{int(result.stats['chunks'])} 块，共 {int(result.stats['chars'])} 字，"
            f"平均 {result.stats['avg_chunk_chars']} 字/块"
        ),
        "",
    ]
    for warning in result.warnings:
        lines.append(f"⚠︎ {warning}")
    if result.warnings:
        lines.append("")
    for chunk in result.chunks[:3]:
        preview = chunk.text.replace("\n", " ")
        lines.append(f"[{chunk.seq}] p{chunk.page_start}-{chunk.page_end} · {chunk.context_header}")
        lines.append(f"    {preview[:80]}…")
    if len(result.chunks) > 3:
        lines.append(f"… 其余 {len(result.chunks) - 3} 块省略")
    return "\n".join(lines)