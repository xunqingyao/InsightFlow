"""检索契约测试：五条规则是否真的能挡住错误用法。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recallgraph.retrieval import (
    EXPANSION_PREDICATES,
    ChunkHit,
    Citation,
    ClaimHit,
    QueryFilters,
    RetrievalQuery,
    RetrievalResult,
    RetrievalScore,
    estimate_tokens,
    fit_budget,
)
from recallgraph.schema import Claim, Entity
from recallgraph.vocabulary import ClaimStatus, Confidence, EntityKind, Predicate


def make_claim(predicate: Predicate = Predicate.OUTPERFORMS) -> Claim:
    rag = Entity(canonical_name="RAG", kind=EntityKind.METHOD)
    ft = Entity(canonical_name="fine-tuning", kind=EntityKind.METHOD)
    return Claim(
        version_id="v" * 26,
        subject_id=rag.id,
        predicate=predicate,
        object_id=ft.id,
        claim_text="RAG 在开放域问答上优于参数微调",
        confidence=Confidence.EXTRACTED,
        status=ClaimStatus.ACTIVE,
    )


def make_citation(**overrides) -> Citation:
    base = {
        "chunk_id": "v" * 26 + ":0001",
        "version_id": "v" * 26,
        "document_id": "d" * 26,
        "page": 5,
        "quote": "RAG outperforms fine-tuning on open-domain QA.",
    }
    base.update(overrides)
    return Citation(**base)


def test_query_defaults() -> None:
    q = RetrievalQuery(text="RAG 和微调哪个好")
    assert q.mode == "hybrid"
    assert q.top_k == 20 and q.max_chunks == 12 and q.max_tokens == 6000
    assert q.filters.claim_status == (ClaimStatus.ACTIVE,)


@pytest.mark.parametrize(
    "field,value",
    [("top_k", 0), ("top_k", 51), ("expand_hops", 3), ("max_tokens", 100)],
)
def test_query_rejects_out_of_range(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(text="q", **{field: value})


def test_filter_rejects_inverted_year_range() -> None:
    with pytest.raises(ValidationError, match="year_min"):
        QueryFilters(year_min=2024, year_max=2020)


def test_citation_requires_page_and_quote() -> None:
    with pytest.raises(ValidationError):
        make_citation(page=0)
    with pytest.raises(ValidationError):
        make_citation(quote="短")


def test_claim_hit_requires_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        ClaimHit(claim=make_claim(), citations=())


def test_result_ok_without_content_is_illegal() -> None:
    with pytest.raises(ValidationError, match="status=ok 但结果为空"):
        RetrievalResult(status="ok", query=RetrievalQuery(text="q"))


def test_result_empty_with_content_is_illegal() -> None:
    hit = ClaimHit(claim=make_claim(), citations=(make_citation(),))
    with pytest.raises(ValidationError, match="status=empty 但存在结果"):
        RetrievalResult(status="empty", query=RetrievalQuery(text="q"), claims=(hit,))


def test_result_ambiguous_needs_candidates() -> None:
    with pytest.raises(ValidationError, match="ambiguous"):
        RetrievalResult(status="ambiguous", query=RetrievalQuery(text="q"))


def test_result_ok_roundtrip() -> None:
    hit = ClaimHit(claim=make_claim(), citations=(make_citation(),), channels=("lexical", "dense"))
    result = RetrievalResult(status="ok", query=RetrievalQuery(text="q"), claims=(hit,))
    assert result.claims[0].citations[0].page == 5


def test_estimate_tokens_counts_cjk_per_char() -> None:
    assert estimate_tokens("检索增强生成") == 6
    assert estimate_tokens("retrieval augmented") >= 1


def test_fit_budget_truncates_and_flags() -> None:
    def hit(i: int, text: str) -> ChunkHit:
        return ChunkHit(chunk_id=f"c{i}", version_id="v", page=1, text=text)

    chunks = [hit(0, "甲" * 100), hit(1, "乙" * 100), hit(2, "丙" * 100)]
    kept, truncated = fit_budget(chunks, max_tokens=250)
    assert len(kept) == 2 and truncated is True

    kept_all, truncated_all = fit_budget(chunks, max_tokens=10_000)
    assert len(kept_all) == 3 and truncated_all is False


def test_score_sort_key_prefers_evidence_then_confidence() -> None:
    weak = RetrievalScore(rrf=0.5, evidence_count=3, confidence=Confidence.INFERRED, claim_id="b")
    strong = RetrievalScore(rrf=0.5, evidence_count=3, confidence=Confidence.EXTRACTED, claim_id="a")
    fewer = RetrievalScore(rrf=0.5, evidence_count=1, confidence=Confidence.EXTRACTED, claim_id="c")
    ordered = sorted([fewer, weak, strong], key=RetrievalScore.sort_key)
    # 证据条数优先于置信度：两条证据的 INFERRED 排在一证据的 EXTRACTED 之前
    assert [s.evidence_count for s in ordered] == [3, 3, 1]
    assert [s.confidence for s in ordered] == [
        Confidence.EXTRACTED,
        Confidence.INFERRED,
        Confidence.EXTRACTED,
    ]


def test_expansion_whitelist_excludes_metadata_predicates() -> None:
    assert Predicate.CONTRADICTS not in EXPANSION_PREDICATES
    assert Predicate.PROPOSES not in EXPANSION_PREDICATES
    assert Predicate.EVALUATES_ON not in EXPANSION_PREDICATES
    assert len(EXPANSION_PREDICATES) == 8