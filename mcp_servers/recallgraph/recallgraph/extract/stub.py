"""桩抽取器：不调模型，用句式规则从段落里抽断言。

它的定位很清楚——**先让整条链路（契约 → 校验 → 落库 → 检索）跑起来**，
等接入真模型时，只换这一个实现，别的都不动。

规则抽出来的断言质量有限（只认显式句式，不认识同义改写），但这正好用来标定
"下限"：如果 LLM 抽取的准确率还不如这套规则，那说明 prompt 或模型选错了。
"""

from __future__ import annotations

import re

from recallgraph.extract.contract import ClaimDraft, EntityDraft, EvidenceDraft, ExtractionResult
from recallgraph.schema import Chunk
from recallgraph.vocabulary import Confidence, EntityKind, Predicate, Stance

NAME = "stub-0.1"

MAX_SENTENCE_CHARS = 320
MAX_CLAIMS_PER_CHUNK = 8
MAX_SUBJECT_CHARS = 60
MAX_OBJECT_CHARS = 70

_SENTENCE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?")

# 英文动词 → 谓词。顺序即优先级，先匹配更具体的。
_EN_PATTERNS: list[tuple[Predicate, re.Pattern[str]]] = [
    (Predicate.OUTPERFORMS, re.compile(r"\b(?:outperform(?:s|ed)?|surpass(?:es|ed)?|beat(?:s)?|exceed(?:s|ed)?)\b", re.IGNORECASE)),
    (Predicate.IMPROVES, re.compile(r"\b(?:improv(?:e|es|ed)|increas(?:e|es|ed)|boost(?:s|ed)?|enhanc(?:e|es|ed)|reduc(?:e|es|ed))\b", re.IGNORECASE)),
    (Predicate.REPLACES, re.compile(r"\b(?:replace(?:s|d)?|substitut(?:e|es|ed))\b", re.IGNORECASE)),
    (Predicate.REQUIRES, re.compile(r"\b(?:requires?|required|needs?|needed)\b", re.IGNORECASE)),
    (Predicate.CAUSES, re.compile(r"\b(?:causes?|caused|leads? to|results? in)\b", re.IGNORECASE)),
    (Predicate.PART_OF, re.compile(r"\b(?:contains?|includes?|consists? of)\b", re.IGNORECASE)),
    (Predicate.USES, re.compile(r"\b(?:uses?|used|leverages?|employs?|relies? on)\b", re.IGNORECASE)),
    (Predicate.IS_A, re.compile(r"\b(?:is|are)\s+(?:a|an)\b", re.IGNORECASE)),
]

_CN_PATTERNS: list[tuple[Predicate, re.Pattern[str]]] = [
    (Predicate.OUTPERFORMS, re.compile(r"(优于|超过|胜过|超越)")),
    (Predicate.IMPROVES, re.compile(r"(提升|提高|改善|增强|降低|减少)")),
    (Predicate.REPLACES, re.compile(r"(取代|替代|替换)")),
    (Predicate.REQUIRES, re.compile(r"(需要|必须|要求)")),
    (Predicate.CAUSES, re.compile(r"(导致|引起|带来)")),
    (Predicate.PART_OF, re.compile(r"(包含|包括)")),
    (Predicate.USES, re.compile(r"(使用|采用|利用|依赖|基于)")),
    (Predicate.IS_A, re.compile(r"是")),
]

_LEADING_NOISE = frozenset(
    ["and", "or", "but", "which", "that", "who", "whom", "whose", "where", "when", "while", "although", "however", "thus", "therefore", "because", "if", "so", "then", "also", "there", "here", "it", "its", "this", "these", "those"]
)

_DANGLING_TAIL = frozenset(["the", "a", "an", "of", "to", "in", "on", "with", "and", "or", "which", "that"])

_MATH_NOISE = re.compile(r"[()=⊤∝η∑|\[\]{}<>]")

_PRONOUNS = frozenset(
    ["it", "this", "that", "these", "those", "they", "we", "you", "he", "she", "they", "i", "there", "here", "its", "their", "our", "his", "her"]
)

_METHOD_HINTS = re.compile(r"(net(work)?|model|rag|bart|bert|gpt|transformer|retriever|reader|encoder|decoder)", re.IGNORECASE)
_DATASET_HINTS = re.compile(r"(imagenet|cifar|squad|fever|ms ?marco|natural questions|\bnq\b|wikitext)", re.IGNORECASE)
_METRIC_HINTS = re.compile(r"(accuracy|f1|em\b|bleu|rouge|top-1|top-5|precision|recall)", re.IGNORECASE)


class StubExtractor:
    name = NAME

    def extract(self, chunk: Chunk) -> ExtractionResult:
        claims: list[ClaimDraft] = []
        entities: dict[str, EntityDraft] = {}

        for offset, sentence in _sentences(chunk.text):
            if len(sentence) > MAX_SENTENCE_CHARS or _is_meta_sentence(sentence):
                continue
            for predicate, pattern in _patterns(sentence):
                draft = _draft_from_sentence(chunk, sentence, offset, predicate, pattern)
                if draft is None:
                    continue
                claims.append(draft)
                for name in (draft.subject, draft.object or ""):
                    if name:
                        entity = _entity(name)
                        entities.setdefault(entity.canonical_name.casefold(), entity)
                if len(claims) >= MAX_CLAIMS_PER_CHUNK:
                    break
            if len(claims) >= MAX_CLAIMS_PER_CHUNK:
                break

        return ExtractionResult(entities=list(entities.values()), claims=claims)


def _is_meta_sentence(sentence: str) -> bool:
    """作者自述（"We propose…"）讲的是这篇论文做了什么，不是关于世界的断言。"""
    head = sentence[:12].lower()
    return head.startswith(("we ", "our ", "in this paper", "in this work", "this paper "))


def _patterns(sentence: str) -> list[tuple[Predicate, re.Pattern[str]]]:
    if re.search(r"[\u4e00-\u9fff]", sentence):
        return _CN_PATTERNS
    return _EN_PATTERNS


def _sentences(text: str) -> list[tuple[int, str]]:
    """切句并保留原始偏移（页码估算要用），同时把换行折叠成空格。"""
    result: list[tuple[int, str]] = []
    for match in _SENTENCE.finditer(text):
        raw = match.group(0).strip()
        flat = " ".join(raw.split())
        if len(flat) >= 20:
            result.append((match.start(), flat))
    return result


def _draft_from_sentence(
    chunk: Chunk,
    sentence: str,
    offset: int,
    predicate: Predicate,
    pattern: re.Pattern[str],
) -> ClaimDraft | None:
    match = pattern.search(sentence)
    if match is None:
        return None

    subject = _clean_phrase(sentence[: match.start()])
    obj = _clean_phrase(sentence[match.end() :].split(";")[0])
    if not subject or not obj:
        return None
    if len(subject) > MAX_SUBJECT_CHARS or len(obj) > MAX_OBJECT_CHARS * 2:
        return None
    if _MATH_NOISE.search(subject) or _MATH_NOISE.search(obj):
        return None
    if subject.split(" ")[0].lower() in _PRONOUNS and len(subject.split()) <= 3:
        return None
    words = subject.split()
    if words and words[-1].lower().strip(".,;:") in _DANGLING_TAIL:
        return None
    if words and words[0][:1].isdigit():
        return None
    if obj.lower() in _PRONOUNS or obj.lower() == subject.lower():
        return None
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", obj):
        return None

    if predicate is Predicate.ACHIEVES:
        return None  # 桩不猜数值

    page = _estimate_page(chunk, offset)
    # 桩不做改写：这条断言就是这句话本身，claims 表里存原句，读起来才不别扭
    claim_text = sentence if len(sentence) <= 380 else sentence[:380].rsplit(" ", 1)[0]

    return ClaimDraft(
        subject=" ".join(subject.split()),
        predicate=predicate,
        object=_clip(obj, MAX_OBJECT_CHARS),
        claim_text=claim_text,
        confidence=Confidence.EXTRACTED,
        evidence=[EvidenceDraft(chunk_id=chunk.id, quote=sentence, page=page, stance=Stance.SUPPORT)],
    )


def _clean_phrase(text: str) -> str:
    phrase = text.strip(" ,，;；:：()（）[]【】\"'“”‘’—-–•….")
    phrase = phrase.rstrip(" .。！？!?")
    # 长句取最后一段：主语通常紧贴动词，前面的从句是噪声
    for separator in (",", ";", "，", "；", "：", ":"):
        if separator in phrase:
            phrase = phrase.rsplit(separator, 1)[-1]

    words = phrase.split()
    while words and words[0].lower().strip(".,;:") in _LEADING_NOISE:
        words.pop(0)
    phrase = " ".join(words)
    # 相对从句截断：主语通常到关系代词为止
    phrase = re.split(r"\s+(?:which|that|who|where|whose|because|while|unless)\b", phrase, maxsplit=1)[0]
    return " ".join(phrase.split())


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]


def _estimate_page(chunk: Chunk, offset: int) -> int:
    """块跨页时按字符位置比例估页码。单页块直接返回。"""
    span = chunk.page_end - chunk.page_start
    if span <= 0 or not chunk.text:
        return chunk.page_start
    ratio = min(max(offset / len(chunk.text), 0.0), 1.0)
    return chunk.page_start + int(ratio * span + 0.5)


def _entity(name: str) -> EntityDraft:
    kind = EntityKind.CONCEPT
    if _DATASET_HINTS.search(name):
        kind = EntityKind.DATASET
    elif _METRIC_HINTS.search(name):
        kind = EntityKind.METRIC
    elif _METHOD_HINTS.search(name):
        kind = EntityKind.METHOD
    return EntityDraft(canonical_name=name, kind=kind)
