"""章节识别与段落切分。

输入是逐页文本，输出是"带章节路径的段落块"。章节路径是引用可定位的关键：
读者要能回答"这句话出自第 3.2 节"，而不只是"来自某篇 PDF"。

识别规则全部是显式的，宁可漏判也不臆造：
1. Markdown 标题（# 的个数即层级）；
2. 编号标题（1 / 2.3 / 3.2.1 开头，短、不以句号结尾）；
3. 常见章节名（Abstract / Introduction / References 等）；
4. 全大写短行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from recallgraph.ingest.source import PageText

_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+(\S.*)$")
# 图表轴标签会把每个字符拆成独立的"词"，例如 "R e t r i e v e d"。
_SPACED_LETTERS = re.compile(r"(?:\b\w\b\s+){6,}")
_NUMERIC_TOKEN = re.compile(r"^\d+(?:[.,]\d+)?%?$")

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experiment",
    "results",
    "evaluation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
    "appendix",
    "摘要",
    "引言",
    "相关工作",
    "方法",
    "实验",
    "结果",
    "讨论",
    "结论",
    "参考文献",
}

REFERENCES_TITLES = {"references", "bibliography", "参考文献"}


@dataclass(frozen=True)
class Block:
    """一个段落。页码、章节、字符区间都记录在案，供后续切块与引用。"""

    text: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    char_start: int
    char_end: int


@dataclass
class BlocksResult:
    blocks: list[Block] = field(default_factory=list)
    full_text: str = ""
    warnings: list[str] = field(default_factory=list)
    dropped_blocks: int = 0
    stopped_at_page: int | None = None


def detect_heading(line: str, *, markdown: bool = False) -> tuple[int, str] | None:
    """判断一行是不是标题，返回 (层级, 标题)。

    `markdown` 只在源文件确实是 Markdown 时开启：PDF 表格里常出现 "#"，
    若对 PDF 也套用 Markdown 规则，会把 "# layers # params" 当成标题。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None

    if markdown:
        md = _MD_HEADING.match(stripped)
        if md:
            return len(md.group(1)), md.group(2).strip()

    if stripped.endswith((".", "。", ",", "；")) or len(stripped.split()) > 12:
        return None

    numbered = _NUMBERED.match(stripped)
    if numbered and _looks_like_section_number(stripped, numbered):
        digits = numbered.group(1)
        return digits.count(".") + 1, stripped

    lowered = stripped.lower().strip(" :：")
    if lowered in KNOWN_HEADINGS and len(stripped) <= 40:
        return 1, stripped

    # 全大写只认"像词组"的行：不含数字、至少两个词（或本身是常见章节名）。
    # 表格里的 MARCO / BART 15.1 19.7 这类行靠这条排除。
    if (
        stripped.isupper()
        and 3 <= len(stripped) <= 60
        and any(c.isalpha() for c in stripped)
        and not any(c.isdigit() for c in stripped)
        and not any(c in stripped for c in "#|%&=+<>")
        and (len(stripped.split()) >= 2 or lowered in KNOWN_HEADINGS)
    ):
        return 1, stripped.title()

    return None


def _looks_like_section_number(line: str, match: re.Match[str]) -> bool:
    """编号标题的额外约束：编号后的内容必须是词（不能是"10 20 30"这种轴刻度）。"""
    digits = match.group(1)
    rest = match.group(2)
    if len(line) > 90 or _looks_like_sentence(line):
        return False
    if int(digits.split(".")[0]) > 30:
        return False
    # 标题正文必须是"词"开头且不含数字：表格行 "18 layers 27.94 27.88" 据此排除。
    if not rest or not (rest[0].isupper() or ord(rest[0]) > 0x2E80):
        return False
    # 允许 "CIFAR-10" 这类带数字的标题，但拒绝纯数字词（表格行 "27.94 27.88"）。
    if any(_NUMERIC_TOKEN.match(token.strip(".,;()")) for token in rest.split()):
        return False
    return len(rest.split()) <= 10


def _looks_like_layout_noise(text: str) -> bool:
    """识别图表/表格被抽成文本后的乱码。只在两种强特征同时/明确时判定为噪声。"""
    if _SPACED_LETTERS.search(text):
        return True
    tokens = text.split()
    if len(tokens) < 8:
        return False
    # 表格/坐标轴几乎没有句末标点，而数字密度很高；正文段落正好相反。
    sentence_marks = len(re.findall(r"[.。!?！？](?:\s|$)", text))
    if sentence_marks > 2:
        return False
    numeric = sum(1 for token in tokens if _NUMERIC_TOKEN.match(token.strip(".,;()")))
    return numeric >= 5 and numeric / len(tokens) >= 0.4


def _looks_like_sentence(line: str) -> bool:
    """编号标题通常是名词短语；含多个动词性虚词的多半是正文。"""
    lowered = line.lower()
    return any(token in lowered for token in (" we ", " this ", " that ", " are ", " was "))


def split_blocks(
    pages: list[PageText],
    *,
    stop_at_references: bool = True,
    markdown: bool = False,
    max_block_chars: int = 1800,
) -> BlocksResult:
    """把逐页文本切成段落块，并维护章节栈。"""
    result = BlocksResult()
    stack: list[str] = []
    buffer: list[tuple[str, int, int]] = []  # (行, 字符偏移, 页码)
    buffered_chars = 0
    cursor = 0
    stopped = False

    def flush() -> None:
        nonlocal buffer, buffered_chars
        if not buffer:
            return
        # 段落可能跨页；超过一页的部分要拆开，否则引用页码会失真。
        segments: list[list[tuple[str, int, int]]] = []
        for item in buffer:
            if segments and item[2] - segments[-1][0][2] <= 1:
                segments[-1].append(item)
            else:
                segments.append([item])

        for segment in segments:
            text = _join_lines([line for line, _, _ in segment])
            if not text.strip():
                continue
            if _looks_like_layout_noise(text):
                result.dropped_blocks += 1
                continue
            start = segment[0][1]
            result.blocks.append(
                Block(
                    text=text,
                    page_start=segment[0][2],
                    page_end=segment[-1][2],
                    section_path=tuple(stack),
                    char_start=start,
                    char_end=start + len(text),
                )
            )
        buffer = []
        buffered_chars = 0

    for page in pages:
        for line in page.text.splitlines():
            if stopped:
                cursor += len(line) + 1
                continue
            offset = cursor
            cursor += len(line) + 1

            if not line.strip():
                flush()
                continue

            heading = detect_heading(line, markdown=markdown)
            if heading is None:
                buffer.append((line, offset, page.page))
                buffered_chars += len(line) + 1
                if buffered_chars >= max_block_chars:
                    flush()
                continue

            flush()
            level, title = heading
            if stop_at_references and title.lower().strip(" :：") in REFERENCES_TITLES:
                result.warnings.append(f"在第 {page.page} 页遇到“{title}”，其后内容为参考文献，已跳过")
                result.stopped_at_page = page.page
                stopped = True
                continue
            stack[:] = stack[: level - 1] + [title]

    flush()
    result.full_text = _rebuild_text(pages)
    if not result.blocks:
        result.warnings.append("未抽到任何正文段落（可能是扫描版 PDF 或纯图片页）")
    return result


def _join_lines(lines: list[str]) -> str:
    """把硬换行还原成段落：普通行用空格拼接，列表项保留换行。"""
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_item = bool(re.match(r"^([-*•]|\d+[.)])\s", stripped))
        if out and not is_item:
            prev = out[-1]
            if not prev.endswith(("\n",)) and not re.search(r"[。！？；.!?:：]$", prev):
                out[-1] = prev + " " + stripped
                continue
        out.append(stripped)
    return "\n".join(out)


def _rebuild_text(pages: list[PageText]) -> str:
    return "\n".join(page.text for page in pages)