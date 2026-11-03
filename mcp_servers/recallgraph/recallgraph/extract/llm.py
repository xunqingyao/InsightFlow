"""真模型抽取器：让 LLM 从段落里读出（实体, 断言, 证据）。

和桩的分工：桩负责"链路能跑"，这里负责"抽得对"。契约不变——模型输出仍要过
`validate_result`：引文必须是原文子串、页码必须落在块的页范围内，否则拒收。

几个刻意的选择：

- **只让模型给 quote，页码我们自己算**。模型数页码会瞎编；把引文在块里的位置
  换算成页（按字符比例映射到 page_start..page_end）至少是可复现的。
- **引文做"吸附"**：模型常把引文改一个标点或合并两行。先精确匹配，不中再按前
  24 个字符定位，然后向前后扩到句界——不中的直接丢弃，不硬凑。
- **谓词是闭集**：只接受词表里的 12 个，写别的整条丢，不做模糊映射。
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Sequence

from recallgraph.extract.contract import ClaimDraft, EntityDraft, EvidenceDraft, ExtractionResult, normalize_for_match
from recallgraph.schema import Chunk
from recallgraph.vocabulary import Confidence, EntityKind, Predicate

CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
NAME = "qwen-plus-1"
DEFAULT_MODEL = "qwen-plus"
MAX_CLAIMS = 8

_SENTENCE_END = re.compile(r"[.!?。！？\n]")

SYSTEM_PROMPT = """你是论文与笔记的知识抽取器。给定一段原文，抽取出其中**明确陈述**的事实性断言。

输出 JSON，结构固定：
{"entities":[{"canonical_name":"...","kind":"method|dataset|metric|concept|paper|author|task|organization|other","aliases":["..."]}],
 "claims":[{"subject":"...","predicate":"...","object":"...","conditions":"","claim_text":"...","quote":"..."}]}

规则（违反会被程序拒收）：
1. quote 必须从原文**逐字复制**的一段连续文字，不许改写、不许翻译、不许拼接两句。长度 10~200 字。
2. predicate 只能取：is_a, part_of, uses, requires, causes, improves, outperforms, replaces, evaluates_on, supports, contradicts, achieves。
3. subject/object 用名词短语，不超过 20 字；不要用"本文/我们/该方法/它"这类代词，要写出具体名称。
4. subject/object 必须在 quote 里原样出现（否则会被丢弃）。
5. 不要抽：作者对本文贡献的自我介绍（We propose…）、图表编号、纯公式、参考文献条目、致谢。
6. 一段最多 {max_claims} 条；没有可抽的就返回 {{"entities":[],"claims":[]}}。"""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{max_claims}", str(MAX_CLAIMS))

_PREDICATE_BY_VALUE = {predicate.value: predicate for predicate in Predicate}
_KIND_BY_VALUE = {kind.value: kind for kind in EntityKind}


class LlmExtractionError(RuntimeError):
    """模型调用失败或返回无法解析。"""


class LlmClaimExtractor:
    """百炼 qwen 抽取器。"""

    name = NAME

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 45.0,
        max_claims: int = MAX_CLAIMS,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.timeout = timeout
        self.max_claims = max_claims

    def extract(self, chunk: Chunk) -> ExtractionResult:
        try:
            payload = self._request(chunk)
        except LlmExtractionError:
            # 长段落偶尔会把 max_tokens 撑爆、返回截断的 JSON；用前半段再试一次。
            if not chunk.text:
                raise
            payload = self._request(chunk, text=chunk.text[: max(len(chunk.text) // 2, 200)])
        return self._to_result(chunk, payload)

    # ── 模型调用 ──────────────────────────────────────────────────────────

    def _request(self, chunk: Chunk, *, text: str | None = None) -> dict:
        if not self.api_key:
            raise LlmExtractionError("DASHSCOPE_API_KEY 未设置，无法调用抽取模型")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"标题：{chunk.context_header}\n\n原文：\n{(text or chunk.text)[:3000]}",
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 2400,
        }
        request = urllib.request.Request(
            CHAT_URL,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last = LlmExtractionError(f"抽取接口 HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404):
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last = LlmExtractionError(f"抽取接口失败: {exc}")
            time.sleep(0.8 * (attempt + 1))
        raise last or LlmExtractionError("抽取接口失败（未知原因）")

    # ── 结果整形 ──────────────────────────────────────────────────────────

    def _to_result(self, chunk: Chunk, payload: dict) -> ExtractionResult:
        entities = [self._entity(item) for item in payload.get("entities") or []]
        claims: list[ClaimDraft] = []
        for item in (payload.get("claims") or [])[: self.max_claims]:
            draft = self._claim(chunk, item)
            if draft is not None:
                claims.append(draft)
        return ExtractionResult(
            entities=[entity for entity in entities if entity is not None],
            claims=claims,
        )

    @staticmethod
    def _entity(item: dict) -> EntityDraft | None:
        name = str(item.get("canonical_name") or "").strip()
        if not name:
            return None
        kind = _KIND_BY_VALUE.get(str(item.get("kind") or "").strip().lower(), EntityKind.CONCEPT)
        aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
        return EntityDraft(canonical_name=name[:120], kind=kind, aliases=aliases[:4])

    def _claim(self, chunk: Chunk, item: dict) -> ClaimDraft | None:
        predicate = _PREDICATE_BY_VALUE.get(str(item.get("predicate") or "").strip().lower())
        subject = str(item.get("subject") or "").strip()
        if predicate is None or not subject:
            return None
        quote = _snap_quote(chunk.text, str(item.get("quote") or ""))
        if quote is None:
            return None
        object_text = str(item.get("object") or "").strip() or None
        if object_text and normalize_for_match(object_text) == normalize_for_match(subject):
            return None
        page = _page_for(chunk, chunk.text.find(quote))
        try:
            return ClaimDraft(
                subject=subject[:120],
                predicate=predicate,
                object=object_text[:120] if object_text else None,
                conditions=str(item.get("conditions") or "").strip()[:200],
                # 断言文本直接用吸附后的原句，而不是模型的转述：
                # 转述会带进原文没有的数字/比较级（"outperforms … by 2.6 Bleu" 就这么来的）。
                claim_text=_one_line(quote)[:400],
                confidence=Confidence.EXTRACTED,
                evidence=[EvidenceDraft(chunk_id=chunk.id, quote=quote, page=page)],
            )
        except ValueError:
            return None


def _one_line(text: str) -> str:
    """claim_text 必须是一句话：把引文里的换行压成空格。引文本身保持原样（要做子串校验）。"""
    return " ".join(text.split())


def _page_for(chunk: Chunk, position: int) -> int:
    """按引文在块内的相对位置，把页码映射到块的页范围。"""
    if position < 0 or chunk.page_end == chunk.page_start or not chunk.text:
        return chunk.page_start
    ratio = position / len(chunk.text)
    span = chunk.page_end - chunk.page_start + 1
    return min(chunk.page_end, chunk.page_start + int(ratio * span))


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """返回（归一化文本, 每个字符对应的原文下标）。归一化=NFKC+小写+空白折叠。"""
    chars: list[str] = []
    index: list[int] = []
    last_space = False
    for position, raw in enumerate(text):
        if raw.isspace():
            if not last_space:
                chars.append(" ")
                index.append(position)
                last_space = True
            continue
        normalized = unicodedata.normalize("NFKC", raw).casefold()
        if not normalized:
            continue
        for char in normalized:
            chars.append(char)
            index.append(position)
        last_space = False
    return "".join(chars), index


def _snap_quote(text: str, quote: str) -> str | None:
    """把模型给的引文吸附到原文的句界上；对不上就返回 None（宁可丢，不许编）。"""
    quote = quote.strip().strip('"“”')
    if len(quote) < 5:
        return None
    start, end = _locate(text, quote)
    if start < 0:
        return None
    # 只在引文头/尾不在句界时才扩展，避免把下一句也吞进来
    if start > 0 and text[start - 1] not in ".!?。！？\n；;":
        line_start = text.rfind("\n", 0, start) + 1
        for match in _SENTENCE_END.finditer(text, line_start, start):
            start = max(start, match.end())
    tail_char = text[end - 1] if end - 1 < len(text) else ""
    if tail_char not in ".!?。！？\n；;":
        tail = _SENTENCE_END.search(text, end)
        end = tail.end() if tail else len(text)
    snapped = text[start:end].strip()
    return snapped if len(snapped) >= 5 else None


def _locate(text: str, quote: str) -> tuple[int, int]:
    """在原文里找引文，返回原文的 [start, end)。先按空格折叠比，再按去空格比。"""
    variants = [
        (_normalized_with_map(text), normalize_for_match(quote), False),
        (_compact_with_map(text), normalize_for_match(quote).replace(" ", ""), True),
    ]
    for (normalized, index), needle, compact in variants:
        if not needle:
            continue
        found = normalized.find(needle)
        if found < 0:
            found = normalized.find(needle[:24]) if len(needle) > 24 else -1
        if found < 0:
            continue
        last = min(found + len(needle) - 1, len(index) - 1)
        return index[found], index[last] + 1
    return -1, -1


def _compact_with_map(text: str) -> tuple[str, list[int]]:
    """归一化并去掉所有空白，用于容忍中英混排里的空格差异。"""
    normalized, index = _normalized_with_map(text)
    chars: list[str] = []
    kept: list[int] = []
    for char, position in zip(normalized, index, strict=True):
        if char == " ":
            continue
        chars.append(char)
        kept.append(position)
    return "".join(chars), kept


def extract_all(extractor: LlmClaimExtractor, chunks: Sequence[Chunk]) -> list[tuple[Chunk, ExtractionResult]]:
    """顺序抽取，单块失败不拖垮整批，返回已成功的结果。"""
    results: list[tuple[Chunk, ExtractionResult]] = []
    for chunk in chunks:
        try:
            results.append((chunk, extractor.extract(chunk)))
        except LlmExtractionError:
            continue
    return results