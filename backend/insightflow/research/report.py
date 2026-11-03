"""把结构化 Report 渲染成 Markdown。

渲染只做排版，不改事实：没有发布时间就写“未知”，没有证据就不写结论。
"""

from __future__ import annotations

from insightflow.research.schemas import (
    CheckStatus,
    ClaimType,
    Evidence,
    Report,
    Source,
    SupportStatus,
)

ROLE_LABELS = {
    "original": "一手",
    "official": "官方",
    "reprint": "转载",
    "secondary": "二手",
    "unknown": "角色未知",
}

FETCH_SCOPE_LABELS = {
    "none": "未抓取",
    "snippet": "仅摘要",
    "partial": "部分正文",
    "full": "全文",
}

CLAIM_LABELS: dict[ClaimType, str] = {
    ClaimType.OBSERVABLE_FACT: "可观察事实",
    ClaimType.PUBLISHER_CLAIM: "发布方说法",
    ClaimType.RESEARCH_RESULT: "研究结论",
    ClaimType.EXTERNAL_TEST: "第三方测试",
    ClaimType.SYSTEM_INFERENCE: "系统推断",
}

SUPPORT_LABELS: dict[SupportStatus, str] = {
    SupportStatus.SUPPORTED: "已支撑",
    SupportStatus.PARTIAL: "部分支撑",
    SupportStatus.CONFLICTED: "存在冲突",
    SupportStatus.INSUFFICIENT: "证据不足",
}

_STATUS_ICON = {CheckStatus.PASS: "通过", CheckStatus.WARN: "注意", CheckStatus.FAIL: "未通过"}


def _fmt_time(value: object) -> str:
    if value is None:
        return "未知"
    return value.astimezone().strftime("%Y-%m-%d %H:%M %Z")  # type: ignore[union-attr]


def _fmt_date(value: object) -> str:
    if value is None:
        return "未知"
    return value.astimezone().strftime("%Y-%m-%d")  # type: ignore[union-attr]


def render_markdown(report: Report, sources: list[Source], evidence: list[Evidence]) -> str:
    by_id = {source.source_id: source for source in sources}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    lines: list[str] = []

    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"> {report.scope_text}")
    lines.append("")
    lines.append(report.summary.strip())
    lines.append("")

    if report.findings:
        lines.append("## 关键发现")
        lines.append("")
    for index, finding in enumerate(report.findings, start=1):
        label = CLAIM_LABELS.get(finding.claim_type, str(finding.claim_type))
        support = SUPPORT_LABELS.get(finding.support_status, str(finding.support_status))
        marks = "".join(
            f"[{i + 1}]" for i, item in enumerate(evidence) if item.evidence_id in finding.evidence_ids
        )
        lines.append(f"### {index}. {finding.text.strip()}")
        lines.append("")
        lines.append(f"- 结论类型：{label}｜支撑状态：{support}{('｜证据 ' + marks) if marks else ''}")
        for evidence_id in finding.evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                continue
            source = by_id.get(item.source_id)
            ref = _ref_index(evidence, evidence_id)
            where = f" — {source.title}" if source else ""
            lines.append(f"- [{ref}] {item.locator}{where}")
        for limitation in finding.limitations:
            lines.append(f"- 限制：{limitation}")
        lines.append("")

    if report.coverage:
        lines.append("## 覆盖情况")
        lines.append("")
        for item in report.coverage:
            lines.append(f"- {item}")
        lines.append("")

    if report.checks:
        lines.append("## 质量校验")
        lines.append("")
        lines.append("| 检查项 | 结果 | 说明 |")
        lines.append("| --- | --- | --- |")
        for check in report.checks:
            lines.append(
                f"| {check.check} | {_STATUS_ICON.get(check.status, check.status)} | {check.detail} |"
            )
        lines.append("")

    if report.limitations:
        lines.append("## 局限与未知")
        lines.append("")
        for limitation in report.limitations:
            lines.append(f"- {limitation}")
        lines.append("")

    if evidence:
        lines.append("## 证据与来源")
        lines.append("")
        # 同一个来源常被引用多次，来源清单要按来源去重，否则一份报告会列出好几个
        # 一模一样的小标题，读者以为读了多个来源。
        ordered: list[Source] = []
        for item in evidence:
            source = by_id.get(item.source_id)
            if source is not None and all(s.source_id != source.source_id for s in ordered):
                ordered.append(source)
        source_no = {source.source_id: index for index, source in enumerate(ordered, start=1)}

        lines.append(f"**来源**（{len(ordered)} 个）")
        lines.append("")
        for index, source in enumerate(ordered, start=1):
            lines.append(
                f"{index}. [{source.title}]({source.url})｜{_fmt_date(source.published_at)}"
                f"｜角色 {ROLE_LABELS.get(source.role.value, source.role.value)}"
                f"｜抓取范围 {FETCH_SCOPE_LABELS.get(source.fetch_scope.value, source.fetch_scope.value)}"
                f"｜阅读深度 {source.read_depth.value}"
            )
        lines.append("")

        lines.append("**引文**")
        lines.append("")
        for index, item in enumerate(evidence, start=1):
            number = source_no.get(item.source_id)
            if number is None:
                continue
            lines.append(f"{index}. 来源 {number} — > {item.quote.strip()[:300]}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"报告 ID：`{report.report_id}`｜版本 {report.version}｜状态 {report.status}"
        f"｜生成时间 {_fmt_time(report.created_at)}"
    )
    lines.append(
        "说明：本报告只引用已登记证据；来源发布时间未知的未计入“新增”；"
        "转载与二手报道不视为独立验证。"
    )
    return "\n".join(lines).strip() + "\n"


def _ref_index(evidence: list[Evidence], evidence_id: str) -> int:
    for index, item in enumerate(evidence, start=1):
        if item.evidence_id == evidence_id:
            return index
    return 0


def render_title(request_query: str, task_type: str) -> str:
    cleaned = " ".join(request_query.split())
    suffix = {
        "briefing": "动态简报",
        "topic_research": "专题研究",
        "comparison": "对比分析",
        "data_analysis": "数据分析",
    }.get(task_type, "研究简报")
    return f"{cleaned}｜{suffix}" if cleaned else suffix