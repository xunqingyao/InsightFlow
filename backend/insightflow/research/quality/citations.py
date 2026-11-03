"""证据登记与引用校验（设计文档 §7.3、§13）。

原则：报告里的每条结论只能引用已登记的 evidence_id；引用不到证据的结论
要么降级为 system_inference，要么删掉。校验结果会写进报告的 checks 字段，
并发给用户，而不是只在日志里。
"""

from __future__ import annotations

from datetime import datetime

from insightflow.research.schemas import (
    CheckResult,
    CheckStatus,
    ClaimType,
    Evidence,
    Finding,
    Report,
    Source,
    SourceRole,
    stable_id,
)
from insightflow.research.time_window import in_window

MIN_QUOTE_CHARS = 12


class EvidenceRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def __len__(self) -> int:
        return len(self._items)

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def register(
        self, source: Source, quote: str, locator: str, scope: str | None = None
    ) -> Evidence:
        evidence_id = stable_id("ev", source.source_id, locator, quote[:120])
        existing = self._items.get(evidence_id)
        if existing:
            return existing
        item = Evidence(
            evidence_id=evidence_id,
            source_id=source.source_id,
            locator=locator,
            quote=quote.strip(),
            scope=scope,
            created_at=source.retrieved_at,
        )
        self._items[evidence_id] = item
        return item

    def register_existing(self, item: Evidence) -> Evidence:
        existing = self._items.get(item.evidence_id)
        if existing:
            return existing
        self._items[item.evidence_id] = item
        return item

    def has_source(self, source_id: str) -> bool:
        return any(item.source_id == source_id for item in self._items.values())


def _check(name: str, status: CheckStatus, detail: str) -> CheckResult:
    return CheckResult(check=name, status=status, detail=detail)


def validate_report(
    report: Report,
    registry: EvidenceRegistry,
    sources: dict[str, Source],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. 每条结论都必须挂上已登记的 evidence_id
    missing: list[str] = []
    for finding in report.findings:
        if not finding.evidence_ids:
            missing.append(f"{finding.finding_id}（无引用）")
            continue
        for evidence_id in finding.evidence_ids:
            if registry.get(evidence_id) is None:
                missing.append(f"{finding.finding_id}→{evidence_id}")
    checks.append(
        _check(
            "evidence_refs",
            CheckStatus.FAIL if missing else CheckStatus.PASS,
            "存在未登记引用：" + "、".join(missing[:5]) if missing else "所有结论均引用已登记证据",
        )
    )

    # 2. 引用的证据必须属于本报告已登记来源
    unknown_sources = sorted(
        {
            registry.get(evidence_id).source_id  # type: ignore[union-attr]
            for finding in report.findings
            for evidence_id in finding.evidence_ids
            if registry.get(evidence_id) is not None
            and registry.get(evidence_id).source_id not in sources  # type: ignore[union-attr]
        }
    )
    checks.append(
        _check(
            "evidence_source_known",
            CheckStatus.FAIL if unknown_sources else CheckStatus.PASS,
            "引用了未登记来源：" + "、".join(unknown_sources[:5])
            if unknown_sources
            else "证据来源均在来源清单内",
        )
    )

    # 3. cited_source_ids 必须覆盖实际引用到的来源
    referenced = {
        registry.get(evidence_id).source_id  # type: ignore[union-attr]
        for finding in report.findings
        for evidence_id in finding.evidence_ids
        if registry.get(evidence_id) is not None
    }
    declared = set(report.cited_source_ids)
    uncovered = sorted(referenced - declared)
    checks.append(
        _check(
            "cited_sources",
            CheckStatus.FAIL if uncovered else CheckStatus.PASS,
            "cited_source_ids 缺少：" + "、".join(uncovered[:5])
            if uncovered
            else f"报告声明引用 {len(declared)} 个来源",
        )
    )

    # 4. 引文不能是空串或过短（防止"引用了一个空格"）
    thin = [
        item.evidence_id
        for item in registry.all()
        if len(item.quote.strip()) < MIN_QUOTE_CHARS
    ]
    checks.append(
        _check(
            "quote_length",
            CheckStatus.WARN if thin else CheckStatus.PASS,
            f"{len(thin)} 条引文短于 {MIN_QUOTE_CHARS} 字符，无法支撑结论"
            if thin
            else "引文长度均达到最小要求",
        )
    )

    # 5. 旧闻新报：来源发布时间落在窗口外，或者干脆没有时间
    outside: list[str] = []
    unknown_time: list[str] = []
    for source in sources.values():
        if source.published_at is None:
            unknown_time.append(source.source_id)
        elif not in_window(source.published_at, window_start, window_end):
            outside.append(source.source_id)
    if outside:
        checks.append(
            _check(
                "time_window",
                CheckStatus.WARN,
                f"{len(outside)} 个来源的发布时间在窗口外，报告中已标注为背景信息："
                + "、".join(outside[:5]),
            )
        )
    else:
        checks.append(_check("time_window", CheckStatus.PASS, "所有来源的发布时间均在窗口内"))
    if unknown_time:
        checks.append(
            _check(
                "published_at_unknown",
                CheckStatus.WARN,
                f"{len(unknown_time)} 个来源没有可确认的发布时间，未按“本周新增”处理",
            )
        )

    # 6. 转载不能充当独立验证
    reprints = {
        source.source_id
        for source in sources.values()
        if source.role in {SourceRole.REPRINT, SourceRole.SECONDARY}
    }
    claimed_supported = [
        finding.finding_id
        for finding in report.findings
        if finding.support_status.value == "supported"
        and any(
            registry.get(evidence_id) is not None
            and registry.get(evidence_id).source_id in reprints  # type: ignore[union-attr]
            for evidence_id in finding.evidence_ids
        )
    ]
    checks.append(
        _check(
            "reprint_independence",
            CheckStatus.WARN if claimed_supported else CheckStatus.PASS,
            f"{len(claimed_supported)} 条结论的支撑来源包含转载/二手报道，独立性不足"
            if claimed_supported
            else "没有把转载当作独立验证",
        )
    )

    # 7. 结论类型与来源类型的一致性
    weak_facts = [
        finding.finding_id
        for finding in report.findings
        if finding.claim_type is ClaimType.OBSERVABLE_FACT
        and not any(
            registry.get(evidence_id) is not None
            and registry.get(evidence_id).source_id in sources  # type: ignore[union-attr]
            for evidence_id in finding.evidence_ids
        )
    ]
    checks.append(
        _check(
            "claim_provenance",
            CheckStatus.WARN if weak_facts else CheckStatus.PASS,
            f"{len(weak_facts)} 条 observable_fact 缺少可核对的来源"
            if weak_facts
            else "结论类型与证据来源匹配",
        )
    )

    return checks


def finalize_findings(
    findings: list[Finding], registry: EvidenceRegistry, sources: dict[str, Source]
) -> list[Finding]:
    """剔除引用缺失的结论，其余保持原样（不悄悄改写用户可见内容）。"""
    kept: list[Finding] = []
    for finding in findings:
        if not finding.evidence_ids:
            continue
        if all(registry.get(evidence_id) is not None for evidence_id in finding.evidence_ids):
            kept.append(finding)
    return kept


def cited_source_ids(findings: list[Finding], registry: EvidenceRegistry) -> list[str]:
    seen: list[str] = []
    for finding in findings:
        for evidence_id in finding.evidence_ids:
            evidence = registry.get(evidence_id)
            if evidence and evidence.source_id not in seen:
                seen.append(evidence.source_id)
    return seen