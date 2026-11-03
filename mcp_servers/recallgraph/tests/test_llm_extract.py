from __future__ import annotations

from recallgraph.extract.llm import LlmClaimExtractor, _page_for, _snap_quote
from recallgraph.schema import Chunk
from recallgraph.vocabulary import Predicate


def make_chunk(text: str, page_start: int = 3, page_end: int = 3, seq: int = 0) -> Chunk:
    version_id = "01TESTVERSION000000000000"
    return Chunk(
        id=f"{version_id}:{seq:04d}",
        version_id=version_id,
        seq=seq,
        page_start=page_start,
        page_end=page_end,
        text=text,
        context_header="测试 ｜ 方法",
        char_start=0,
        char_end=len(text),
    )


def test_snap_quote_exact_substring():
    text = "RAG 的训练是端到端微调。检索器冻结。"
    assert _snap_quote(text, "RAG 的训练是端到端微调。") == "RAG 的训练是端到端微调。"


def test_snap_quote_tolerates_whitespace_and_fullwidth():
    text = "RAG的训练是端到端微调：只更新生成器（BART）的参数。"
    snapped = _snap_quote(text, "RAG 的训练是端到端微调: 只更新生成器(BART)的参数。")
    assert snapped is not None
    assert "只更新生成器" in snapped


def test_snap_quote_rejects_invented_text():
    text = "RAG 的训练是端到端微调。"
    assert _snap_quote(text, "RAG 使用了强化学习来训练。") is None


def test_page_for_maps_position_across_page_span():
    chunk = make_chunk("a" * 100, page_start=5, page_end=6)
    assert _page_for(chunk, 0) == 5
    assert _page_for(chunk, 95) == 6


def test_extractor_drops_invalid_predicate(monkeypatch):
    extractor = LlmClaimExtractor(api_key="test")
    chunk = make_chunk("RAG 的训练是端到端微调。")
    monkeypatch.setattr(
        extractor,
        "_request",
        lambda _chunk: {
            "claims": [
                {"subject": "RAG", "predicate": "随便编的", "object": "x", "claim_text": "这是一条测试断言", "quote": "RAG 的训练是端到端微调。"},
                {"subject": "RAG", "predicate": "uses", "object": "BART", "claim_text": "这是一条测试断言", "quote": "RAG 的训练是端到端微调。"},
            ]
        },
    )
    result = extractor.extract(chunk)
    assert [claim.predicate for claim in result.claims] == [Predicate.USES]


def test_extractor_drops_hallucinated_quote(monkeypatch):
    extractor = LlmClaimExtractor(api_key="test")
    chunk = make_chunk("RAG 的训练是端到端微调。")
    monkeypatch.setattr(
        extractor,
        "_request",
        lambda _chunk: {
            "claims": [
                {"subject": "RAG", "predicate": "uses", "object": "BART", "claim_text": "这是一条测试断言", "quote": "RAG 用了强化学习。"}
            ]
        },
    )
    assert extractor.extract(chunk).claims == []


def test_extractor_rejects_self_loop(monkeypatch):
    extractor = LlmClaimExtractor(api_key="test")
    chunk = make_chunk("RAG 是一个方法。")
    monkeypatch.setattr(
        extractor,
        "_request",
        lambda _chunk: {
            "claims": [
                {"subject": "RAG", "predicate": "is_a", "object": "rag", "claim_text": "这是一条测试断言", "quote": "RAG 是一个方法。"}
            ]
        },
    )
    assert extractor.extract(chunk).claims == []


def test_claim_text_is_verbatim_quote(monkeypatch):
    """断言文本用原句，模型的转述不采信。"""
    extractor = LlmClaimExtractor(api_key="test")
    chunk = make_chunk("RAG 的训练是端到端微调。")
    monkeypatch.setattr(
        extractor,
        "_request",
        lambda _chunk: {
            "claims": [
                {
                    "subject": "RAG",
                    "predicate": "uses",
                    "object": "BART",
                    "claim_text": "RAG 只训练 BART，不训练编码器",
                    "quote": "RAG 的训练是端到端微调。",
                }
            ]
        },
    )
    claims = extractor.extract(chunk).claims
    assert claims and claims[0].claim_text == "RAG 的训练是端到端微调。"
