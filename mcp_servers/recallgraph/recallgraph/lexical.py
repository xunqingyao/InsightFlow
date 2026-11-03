"""关键词通道：分词 → 倒排表 → BM25 打分。

两个设计取舍：
1. **中文必须分词**。Postgres 自带的分词器不切中文，`jieba` 是事实标准；
   英文按正则切词。分词结果落库存 tf，查询时只读库，不再碰原文。
2. **BM25 在 SQL 里算**。df、平均长度这些都是聚合量，交给数据库比拉回 Python 更省内存，
   也方便以后按文档/时间过滤后再排序。
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass

import psycopg

with warnings.catch_warnings():
    # jieba 0.42 的字符串里有未转义的 "\u"/"\."，在 3.12 上会刷 SyntaxWarning，与使用无关。
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba

jieba.setLogLevel(20)  # 关掉加载词典时的 INFO 输出，只保留警告

from recallgraph.ingest.chunker import estimate_tokens
from recallgraph.stemming import stem

# 只保留对检索有区分度的词：英文/数字用正则，中文交给 jieba，标点与空白直接丢。
_WORD = re.compile(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*|[a-z][a-z0-9]*")
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

STOPWORDS: frozenset[str] = frozenset(
    {
        # 英文功能词
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does",
        "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
        "may", "more", "most", "not", "of", "on", "or", "our", "over", "such", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this", "to", "using", "was",
        "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with",
        "within", "would", "you",
        # 中文虚词与高频套话（jieba 会把它们切成独立词）
        "的", "了", "和", "与", "及", "或", "是", "在", "中", "对", "也", "就", "都", "而",
        "被", "把", "让", "给", "从", "到", "由", "为", "以", "其", "该", "这", "那",
        "一个", "一种", "我们", "你", "他", "她", "它", "他们", "可以", "需要", "进行", "通过",
        "以及", "但是", "如果", "因为", "所以", "这个", "那个", "这些", "那些", "什么", "怎么",
    }
)

BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索用的词项序列（保留重复，用于算 tf）。"""
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []

    for match in _WORD.finditer(normalized):
        token = match.group(0)
        if len(token) >= 2 and token not in STOPWORDS:
            # 英文做词干还原：retrieval/retrieving/retrieve 在倒排表里是同一个词。
            # 中文不动——中文的粒度问题在分词，不在形态。
            tokens.append(stem(token) if token.isascii() and token.isalpha() else token)

    for run in _CJK_RUNS(normalized):
        for word in jieba.cut_for_search(run):
            word = word.strip()
            if len(word) >= 2 and word not in STOPWORDS:
                tokens.append(word)


    return tokens


def _CJK_RUNS(text: str) -> list[str]:
    """把连续的中日韩片段切出来，避免把整句丢给 jieba 时把英文也搅进去。"""
    runs: list[str] = []
    current: list[str] = []
    for char in text:
        if _CJK.match(char):
            current.append(char)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def term_frequencies(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def index_version(conn: psycopg.Connection, version_id: str) -> tuple[int, int]:
    """为一个版本重建倒排表，返回（块数, 词项行数）。已在事务外则需调用方提交。"""
    conn.execute("DELETE FROM chunk_terms WHERE chunk_id IN (SELECT id FROM chunks WHERE version_id = %s)", (version_id,))
    rows = conn.execute("SELECT id, text FROM chunks WHERE version_id = %s ORDER BY seq", (version_id,)).fetchall()

    payload: list[tuple[str, str, int]] = []
    for chunk_id, text in rows:
        for term, tf in term_frequencies(tokenize(text)).items():
            payload.append((chunk_id, term, tf))

    if payload:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunk_terms (chunk_id, term, tf) VALUES (%s, %s, %s)",
                payload,
            )
    return len(rows), len(payload)


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    score: float
    version_id: str
    title: str
    seq: int
    page_start: int
    page_end: int
    section_path: list[str]
    text: str


_BM25_SQL = """
WITH q (term) AS (SELECT DISTINCT unnest(%(terms)s::text[])),
     corpus AS (
         SELECT count(*)::float8 AS n,
                GREATEST(coalesce(avg(token_count), 1), 1)::float8 AS avgdl
         FROM chunks
     ),
     df AS (
         SELECT ct.term, count(*)::float8 AS df
         FROM chunk_terms ct JOIN q ON q.term = ct.term
         GROUP BY ct.term
     ),
     scored AS (
         SELECT ct.chunk_id,
                sum(
                    ln(1.0 + (corpus.n - df.df + 0.5) / (df.df + 0.5))
                    * (ct.tf * (%(k1)s + 1.0))
                    / (ct.tf + %(k1)s * (1.0 - %(b)s + %(b)s * GREATEST(c.token_count, 1) / corpus.avgdl))
                ) AS score
         FROM chunk_terms ct
         JOIN q ON q.term = ct.term
         JOIN df ON df.term = ct.term
         JOIN chunks c ON c.id = ct.chunk_id
         CROSS JOIN corpus
         GROUP BY ct.chunk_id
     )
SELECT s.chunk_id, s.score, c.version_id, d.title, c.seq, c.page_start, c.page_end, c.section_path, c.text
FROM scored s
JOIN chunks c ON c.id = s.chunk_id
JOIN document_versions v ON v.id = c.version_id
JOIN documents d ON d.id = v.document_id
WHERE (%(version_ids)s::text[] IS NULL OR c.version_id = ANY(%(version_ids)s::text[]))
ORDER BY s.score DESC, c.version_id, c.seq
LIMIT %(limit)s
"""


def search_lexical(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 5,
    version_ids: list[str] | None = None,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[LexicalHit]:
    terms = sorted(set(tokenize(query)))
    if not terms:
        return []

    rows = conn.execute(
        _BM25_SQL,
        {
            "terms": terms,
            "k1": k1,
            "b": b,
            "limit": top_k,
            "version_ids": version_ids,
        },
    ).fetchall()

    return [
        LexicalHit(
            chunk_id=row[0],
            score=float(row[1]),
            version_id=row[2],
            title=row[3],
            seq=row[4],
            page_start=row[5],
            page_end=row[6],
            section_path=list(row[7]),
            text=row[8],
        )
        for row in rows
    ]


def index_stats(conn: psycopg.Connection) -> dict[str, float]:
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    terms = conn.execute("SELECT count(*) FROM chunk_terms").fetchone()[0]
    vocab = conn.execute("SELECT count(DISTINCT term) FROM chunk_terms").fetchone()[0]
    avg_terms = terms / chunks if chunks else 0.0
    return {
        "chunks": float(chunks),
        "term_rows": float(terms),
        "vocab": float(vocab),
        "avg_terms_per_chunk": round(avg_terms, 1),
    }


__all__ = [
    "BM25_B",
    "BM25_K1",
    "STOPWORDS",
    "LexicalHit",
    "estimate_tokens",
    "index_stats",
    "index_version",
    "search_lexical",
    "term_frequencies",
    "tokenize",
]