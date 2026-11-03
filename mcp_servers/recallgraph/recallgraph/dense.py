"""向量通道：把段落编码成向量，按余弦相似度召回。

模型选 bge-small-en-v1.5（384 维，ONNX，CPU 可跑）：
和 docs/01 里定的 BGE 系列同源，但只有 130 MB，评测能离线复现。

**一个必须守住的约束**：向量与模型绑定。换模型必须整批重算，
否则新旧向量混在一张表里，相似度没有意义——所以每行都记 model。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import psycopg

DEFAULT_MODELS = {
    "unified": "text-embedding-v3",  # 百炼（DashScope）：中英同模型，1024 维
    "en": "BAAI/bge-small-en-v1.5",  # 离线兜底：384 维
    "zh": "BAAI/bge-small-zh-v1.5",
}
API_MODELS = frozenset({"text-embedding-v3", "text-embedding-v4"})
DASHSCOPE_URL = os.environ.get(
    "DASHSCOPE_EMBEDDING_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
)
# 显式指定优先于默认：中文语料要用中文模型，见 docs/07-中文体检.md
MODEL_NAME = os.environ.get(
    "RECALLGRAPH_EMBEDDING_MODEL",
    DEFAULT_MODELS["unified"] if os.environ.get("DASHSCOPE_API_KEY") else DEFAULT_MODELS["en"],
)
DIM = 384
CACHE_DIR = os.environ.get(
    "FASTEMBED_CACHE_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "fastembed")
)


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Iterable[np.ndarray]:
        ...


class FastEmbedder:
    """fastembed 的懒加载包装：不在 import 时下载模型。"""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            # HF 的 xet 下载后端在受限沙箱里会报 "Operation not permitted"，
            # 关掉它走普通 HTTP 下载；模型本身不变。
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name, cache_dir=CACHE_DIR)
        return self._model

    def embed(self, texts: Sequence[str]) -> Iterable[np.ndarray]:
        return self._load().embed(list(texts))


class ApiEmbedder:
    """百炼（DashScope）的 OpenAI 兼容 embedding 接口，中英同一个模型。

    批大小 10 是 text-embedding-v3 的单请求上限；失败重试三次，指数退避。
    返回顺序与输入一致（按响应里的 index 重排，不依赖服务端顺序）。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODELS["unified"],
        *,
        api_key: str | None = None,
        batch_size: int = 10,
        dimensions: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置，无法调用百炼 embedding 接口")
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + self.batch_size]]
            payload = json.dumps(
                {"model": self.model_name, "input": batch, "dimensions": self.dimensions, "encoding_format": "float"}
            ).encode()
            request = urllib.request.Request(
                DASHSCOPE_URL,
                data=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        body = json.loads(response.read())
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f"embedding 请求失败：{exc}") from exc
                    time.sleep(2**attempt)
            rows = sorted(body["data"], key=lambda item: item["index"])
            if len(rows) != len(batch):
                raise RuntimeError(f"embedding 返回条数不符：{len(rows)} != {len(batch)}")
            vectors.extend(np.asarray(row["embedding"], dtype=np.float32) for row in rows)
        return vectors


def make_embedder(model_name: str | None = None) -> Embedder:
    """按模型名选实现：百炼走 API，其余走本地 fastembed。"""
    name = model_name or MODEL_NAME
    return ApiEmbedder(name) if name in API_MODELS else FastEmbedder(name)


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    score: float  # 余弦相似度，1 表示同向
    version_id: str
    page_start: int
    seq: int
    text: str


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.6f}" for value in vector) + "]"


def encode_version(
    conn: psycopg.Connection,
    version_id: str,
    *,
    embedder: Embedder | None = None,
    batch_size: int = 32,
    model: str = MODEL_NAME,
) -> int:
    """为一个版本重建向量，返回写入行数。调用方负责提交事务。"""
    embedder = embedder or make_embedder(model)
    rows = conn.execute("SELECT id, text FROM chunks WHERE version_id = %s ORDER BY seq", (version_id,)).fetchall()
    if not rows:
        return 0
    conn.execute(
        "DELETE FROM chunk_vectors WHERE model = %s AND chunk_id = ANY(%s)",
        (model, [row[0] for row in rows]),
    )

    written = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = list(embedder.embed([row[1] for row in batch]))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunk_vectors (chunk_id, model, embedding) VALUES (%s, %s, %s::vector)",
                [(row[0], model, _vector_literal(vector)) for row, vector in zip(batch, vectors, strict=True)],
            )
        written += len(batch)
    return written


def search_dense(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 20,
    embedder: Embedder | None = None,
    model: str = MODEL_NAME,
    version_ids: list[str] | None = None,
) -> list[DenseHit]:
    embedder = embedder or make_embedder(model)
    vector = next(iter(embedder.embed([query])))
    rows = conn.execute(
        """
        SELECT v.chunk_id, 1 - (v.embedding <=> %s::vector) AS score,
               c.version_id, c.page_start, c.seq, c.text
        FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id
        WHERE v.model = %s
          AND (%s::text[] IS NULL OR c.version_id = ANY(%s::text[]))
        ORDER BY v.embedding <=> %s::vector
        LIMIT %s
        """,
        (_vector_literal(vector), model, version_ids, version_ids, _vector_literal(vector), top_k),
    ).fetchall()
    return [
        DenseHit(
            chunk_id=row[0],
            score=float(row[1]),
            version_id=row[2],
            page_start=row[3],
            seq=row[4],
            text=row[5],
        )
        for row in rows
    ]


def available_models(conn: psycopg.Connection, version_ids: list[str] | None = None) -> dict[str, int]:
    """库里已有向量的模型及行数，按行数从多到少。"""
    rows = conn.execute(
        """
        SELECT v.model, count(*)
        FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id
        WHERE (%s::text[] IS NULL OR c.version_id = ANY(%s::text[]))
        GROUP BY v.model ORDER BY count(*) DESC
        """,
        (version_ids, version_ids),
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def vector_stats(conn: psycopg.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT model, count(*) FROM chunk_vectors GROUP BY model").fetchall()
    return {row[0]: row[1] for row in rows}


__all__ = [
    "DEFAULT_MODELS",
    "DIM",
    "MODEL_NAME",
    "ApiEmbedder",
    "DenseHit",
    "Embedder",
    "FastEmbedder",
    "available_models",
    "encode_version",
    "make_embedder",
    "search_dense",
    "vector_stats",
]
