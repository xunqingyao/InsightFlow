"""原文读取：把文件变成"按页的纯文本"，顺便清掉页眉页脚。

只做两件事，且都保持可解释：
1. 抽取：PDF 逐页取文本；Markdown/文本按换页符切。
2. 清洗：删除在多数页面重复出现的短行（页眉、页脚、页码）。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

_PAGE_NUMBER = re.compile(r"^\d{1,4}$")


class UnsupportedSource(ValueError):
    """不支持的文件类型。"""


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class Cleaning:
    pages: tuple[PageText, ...]
    removed_lines: tuple[str, ...]


def extract_pages(path: Path) -> list[PageText]:
    """读取文件，返回逐页文本。页码从 1 开始。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_pages(path)
    if suffix in SUPPORTED_SUFFIXES:
        return _text_pages(path)
    raise UnsupportedSource(f"不支持的文件类型：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")


# pypdf 会把"缺 fontTools"这类字体提示打到 stderr，对解析结果无影响，这里降噪。
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _pdf_pages(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 —— 单页损坏不应中断整篇
            text = ""
        pages.append(PageText(index, text))
    return pages


def _text_pages(path: Path) -> list[PageText]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parts = raw.split("\f")
    return [PageText(i, part) for i, part in enumerate(parts, start=1)]


def strip_running_lines(pages: list[PageText]) -> Cleaning:
    """删除跨页重复的短行（页眉/页脚/页码）。

    只处理 3 页以上的文档；判据是"同一行出现在 ≥60% 的页面上"，
    或者"整页只有页码"这种一眼可辨的情况。宁可漏删，不可误删正文。
    """
    if len(pages) < 3:
        return Cleaning(tuple(pages), ())

    threshold = max(3, math.ceil(len(pages) * 0.6))
    seen: dict[str, set[int]] = {}
    for page in pages:
        for line in page.text.splitlines():
            key = " ".join(line.split())
            if key:
                seen.setdefault(key, set()).add(page.page)

    repeated = {
        line
        for line, page_ids in seen.items()
        if len(page_ids) >= threshold and len(line) <= 80 and not _PAGE_NUMBER.match(line)
    }

    cleaned: list[PageText] = []
    for page in pages:
        kept = []
        for line in page.text.splitlines():
            key = " ".join(line.split())
            if key in repeated or (len(pages) >= 5 and _PAGE_NUMBER.match(key)):
                continue
            kept.append(line)
        cleaned.append(PageText(page.page, "\n".join(kept)))
    return Cleaning(tuple(cleaned), tuple(sorted(repeated)))