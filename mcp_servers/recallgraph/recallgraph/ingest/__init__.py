"""解析层：文件 → Document / DocumentVersion / Chunk。"""

from recallgraph.ingest.chunker import ChunkSettings, build_chunks
from recallgraph.ingest.pipeline import PARSER_VERSION, IngestResult, ingest, summarize
from recallgraph.ingest.sections import Block, detect_heading, split_blocks
from recallgraph.ingest.source import (
    Cleaning,
    PageText,
    UnsupportedSource,
    extract_pages,
)

__all__ = [
    "PARSER_VERSION",
    "Block",
    "ChunkSettings",
    "Cleaning",
    "IngestResult",
    "PageText",
    "UnsupportedSource",
    "build_chunks",
    "detect_heading",
    "extract_pages",
    "ingest",
    "split_blocks",
    "summarize",
]