"""切块：段落 → 可引用的 Chunk。

三条硬约束（对应 schema.Chunk 的校验）：
- 不跨节：一个 chunk 只属于一条章节路径，否则引用会指向错误的节；
- 不跨两页以上：page_end - page_start <= 1，保证页码可指认；
- 不超长：超过 max_chars 必须在句子边界切开。

剩下的都是软约束：目标长度、相邻小段落合并、尾部过短时回并。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from recallgraph.ingest.sections import Block
from recallgraph.schema import Chunk

_SENTENCE_END = re.compile(r"[。！？；!?;]|(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkSettings:
    target_chars: int = 500
    min_chars: int = 120
    max_chars: int = 1000
    max_context_chars: int = 240

    def __post_init__(self) -> None:
        if not (self.min_chars < self.target_chars <= self.max_chars):
            raise ValueError("需要满足 min_chars < target_chars <= max_chars")


def build_chunks(
    blocks: list[Block],
    *,
    version_id: str,
    title: str,
    settings: ChunkSettings | None = None,
) -> list[Chunk]:
    cfg = settings or ChunkSettings()
    pieces = _split_oversized(blocks, cfg)
    groups = _group(pieces, cfg)

    chunks: list[Chunk] = []
    for seq, group in enumerate(groups):
        text = "\n\n".join(block.text for block in group)
        if not text.strip():
            continue
        section_path = list(group[0].section_path)
        chunks.append(
            Chunk(
                version_id=version_id,
                seq=len(chunks),
                page_start=min(block.page_start for block in group),
                page_end=max(block.page_end for block in group),
                section_path=section_path,
                text=text,
                context_header=make_context_header(title, section_path, cfg.max_context_chars),
                char_start=group[0].char_start,
                char_end=group[-1].char_end,
                token_count=estimate_tokens(text),
            )
        )
    return chunks


def make_context_header(title: str, section_path: list[str], limit: int) -> str:
    """拼给模型/检索器看的前缀。长文档里没有它，模型会不知道这段话在讲哪一节。"""
    seen = {_fold(title)}
    trimmed = [title.strip()] if title.strip() else []
    for part in section_path:
        cleaned = part.strip()
        if cleaned and _fold(cleaned) not in seen:
            seen.add(_fold(cleaned))
            trimmed.append(cleaned)
    header = " ｜ ".join(trimmed)
    if len(header) > limit:
        header = header[: limit - 1] + "…"
    return header or "未命名文档"


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def _split_oversized(blocks: list[Block], cfg: ChunkSettings) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if len(block.text) <= cfg.max_chars:
            out.append(block)
            continue
        for piece, offset in _split_text(block.text, cfg.max_chars):
            out.append(
                Block(
                    text=piece,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    section_path=block.section_path,
                    char_start=block.char_start + offset,
                    char_end=block.char_start + offset + len(piece),
                )
            )
    return out


def _split_text(text: str, max_chars: int) -> list[tuple[str, int]]:
    """在句子边界切分超长段落；找不到边界就硬切。返回 (片段, 相对偏移)。"""
    pieces: list[tuple[str, int]] = []
    start = 0
    while len(text) - start > max_chars:
        window = text[start : start + max_chars]
        cut = 0
        for match in _SENTENCE_END.finditer(window):
            edge = match.end()
            if edge >= max_chars * 0.5:
                cut = edge
        if cut == 0:
            cut = max_chars
        piece = text[start : start + cut].strip()
        if piece:
            pieces.append((piece, start))
        start += cut
    tail = text[start:].strip()
    if tail:
        pieces.append((tail, start))
    return pieces


def _group(blocks: list[Block], cfg: ChunkSettings) -> list[list[Block]]:
    """贪心合并：同节、页码跨度 ≤1、总长不超 max；达到目标长度就收口。"""
    groups: list[list[Block]] = []
    current: list[Block] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        if current:
            groups.append(current)
        current = []
        length = 0

    for block in blocks:
        if not current:
            current = [block]
            length = len(block.text)
            continue

        same_section = block.section_path == current[-1].section_path
        page_span = max(block.page_end, max(b.page_end for b in current)) - min(
            block.page_start, min(b.page_start for b in current)
        )
        fits = length + len(block.text) + 2 <= cfg.max_chars

        if same_section and page_span <= 1 and fits:
            current.append(block)
            length += len(block.text) + 2
            if length >= cfg.target_chars:
                flush()
        else:
            flush()
            current = [block]
            length = len(block.text)

    flush()
    return _merge_short_tail(groups, cfg)


def _merge_short_tail(groups: list[list[Block]], cfg: ChunkSettings) -> list[list[Block]]:
    """末尾过短的块回并到前一块（同节且页码放得下时），避免产生一句话的碎片。"""
    if len(groups) < 2:
        return groups
    last = groups[-1]
    prev = groups[-2]
    last_len = sum(len(b.text) for b in last)
    prev_len = sum(len(b.text) for b in prev)
    if last_len >= cfg.min_chars:
        return groups

    same_section = last[0].section_path == prev[-1].section_path
    page_span = max(last[-1].page_end, prev[-1].page_end) - min(last[0].page_start, prev[0].page_start)
    if same_section and page_span <= 1 and prev_len + last_len + 2 <= cfg.max_chars:
        prev.extend(last)
        groups.pop()
    return groups


_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中日韩字符按 1 个算，其余按词算再乘 1.3（英文子词平均切分率）。

    只用于预算与统计，不参与正确性判断；真要精确值，接分词器时替换这里。
    """
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    return max(1, int(cjk + words * 1.3 + 0.5))
