"""向量通道的纯函数测试：融合与向量字面量（不加载模型、不连库）。"""

from __future__ import annotations

import numpy as np

from recallgraph.dense import DEFAULT_MODELS, _vector_literal
from recallgraph.retriever import DENSE_WEIGHT, RRF_K, _fuse


def test_vector_literal_shape() -> None:
    literal = _vector_literal(np.array([0.5, -1.0, 0.0]))
    assert literal.startswith("[") and literal.endswith("]")
    assert len(literal.split(",")) == 3
    assert "0.500000" in literal


def test_fuse_prefers_dense_when_weight_high() -> None:
    lexical = {"a": 1, "b": 2}
    dense = {"c": 1, "b": 2}
    fused = _fuse(lexical, dense, dense_weight=10.0)
    # b 两路都命中，分最高；c 只在 dense 第 1 名，排第二；a 只在 lexical，垫底
    assert list(fused) == ["b", "c", "a"]
    assert fused["a"] == 3


def test_fuse_is_deterministic_on_ties() -> None:
    fused = _fuse({"z": 1}, {"z": 1})
    assert fused == {"z": 1}


def test_fuse_respects_rrf_constant() -> None:
    assert RRF_K == 60
    assert DENSE_WEIGHT >= 1.0

def test_resolve_model_prefers_explicit_over_env(monkeypatch):
    from recallgraph.retriever import resolve_embedding_model

    monkeypatch.setenv("RECALLGRAPH_EMBEDDING_MODEL", DEFAULT_MODELS["zh"])
    assert resolve_embedding_model(None, [], explicit=DEFAULT_MODELS["en"]) == DEFAULT_MODELS["en"]


def test_resolve_model_uses_env_when_no_explicit(monkeypatch):
    from recallgraph.retriever import resolve_embedding_model

    monkeypatch.setenv("RECALLGRAPH_EMBEDDING_MODEL", DEFAULT_MODELS["zh"])
    assert resolve_embedding_model(None, []) == DEFAULT_MODELS["zh"]
