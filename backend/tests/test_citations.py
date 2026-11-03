from __future__ import annotations

from datetime import UTC, datetime

from insightflow.research.quality.citations import EvidenceRegistry, validate_report
from insightflow.research.schemas import (
    CheckStatus,
    ClaimType,
    Finding,
    Report,
    Source,
    SourceRole,
    SupportStatus,
)


def _source(source_id: str, url: str) -> Source:
    return Source(
        source_id=source_id,
        url=url,
        canonical_url=url,
        title="标题",
        role=SourceRole.REPRINT,
        retrieved_at=datetime(2026, 10, 27, tzinfo=UTC),
    )


def _report(findings: list[Finding]) -> Report:
    return Report(
        report_id="rep_test",
        run_id="run_test",
        created_at=datetime(2026, 10, 27, tzinfo=UTC),
        title="t",
        scope_text="s",
        summary="sum",
        findings=findings,
    )


def test_missing_evidence_id_fails() -> None:
    source = _source("src_1", "https://example.invalid/a")
    report = _report(
        [
            Finding(
                finding_id="f1",
                text="某结论",
                claim_type=ClaimType.PUBLISHER_CLAIM,
                evidence_ids=["ev_not_registered"],
                support_status=SupportStatus.SUPPORTED,
            )
        ]
    )
    checks = validate_report(
        report,
        EvidenceRegistry(),
        {source.source_id: source},
        window_start=datetime(2026, 10, 20, tzinfo=UTC),
        window_end=datetime(2026, 10, 27, tzinfo=UTC),
    )
    statuses = {check.check: check.status for check in checks}
    assert statuses["evidence_refs"] is CheckStatus.FAIL


def test_reprint_cannot_be_supported() -> None:
    source = _source("src_1", "https://example.invalid/a")
    registry = EvidenceRegistry()
    evidence = registry.register(source, "转载报道声称某模型表现更好，但未提供独立测试数据。", "正文")
    report = _report(
        [
            Finding(
                finding_id="f1",
                text="转载报道声称某模型更好",
                claim_type=ClaimType.PUBLISHER_CLAIM,
                evidence_ids=[evidence.evidence_id],
                support_status=SupportStatus.SUPPORTED,
            )
        ]
    )
    report.cited_source_ids = [source.source_id]
    checks = validate_report(
        report,
        registry,
        {source.source_id: source},
        window_start=datetime(2026, 10, 20, tzinfo=UTC),
        window_end=datetime(2026, 10, 27, tzinfo=UTC),
    )
    statuses = {check.check: check.status for check in checks}
    assert statuses["reprint_independence"] is CheckStatus.WARN
