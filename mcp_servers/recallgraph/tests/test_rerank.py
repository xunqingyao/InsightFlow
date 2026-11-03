from __future__ import annotations

import pytest

from recallgraph.rerank import DashScopeReranker, RerankError


class FakeReranker(DashScopeReranker):
    def __init__(self, scores):
        super().__init__(api_key="test")
        self._scores = scores
        self.calls = []

    def score(self, query, documents):
        if documents:
            self.calls.append((query, list(documents)))
        return self._scores[: len(documents)]


def test_rerank_orders_by_score_and_keeps_ids():
    reranker = FakeReranker([0.1, 0.9, 0.5])
    ordered = reranker.rerank("q", [("a", "文本一"), ("b", "文本二"), ("c", "文本三")])
    assert [ident for ident, _ in ordered] == ["b", "c", "a"]
    assert reranker.calls == [("q", ["文本一", "文本二", "文本三"])]


def test_rerank_empty_input_does_not_call_api():
    reranker = FakeReranker([])
    assert reranker.rerank("q", []) == []
    assert reranker.calls == []


def test_missing_api_key_raises():
    reranker = DashScopeReranker(api_key="")
    with pytest.raises(RerankError):
        reranker.score("q", ["文本"])


def test_scores_align_by_returned_index(monkeypatch):
    """服务端结果不一定按输入顺序返回，必须按 index 回填。"""
    reranker = DashScopeReranker(api_key="test", batch_size=3)

    def fake_post(request):
        return {"output": {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}}

    monkeypatch.setattr(reranker, "_post", fake_post)
    scores = reranker.score("q", ["a", "b", "c"])
    assert scores == [0.2, 0.0, 0.9]


def test_rerank_batches_large_input(monkeypatch):
    import json

    reranker = DashScopeReranker(api_key="test", batch_size=2)
    calls = []

    def fake_post(request):
        batch = json.loads(request.data)["input"]["documents"]
        calls.append(len(batch))
        return {"output": {"results": [{"index": i, "relevance_score": 0.1 * (i + 1)} for i in range(len(batch))]}}

    monkeypatch.setattr(reranker, "_post", fake_post)
    scores = reranker.score("q", ["a", "b", "c"])
    assert calls == [2, 1]
    assert scores == [0.1, 0.2, 0.1]
