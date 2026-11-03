"""最小研究链路：规划 → 检索 → 抓取 → 证据 → 报告。

刻意保持顺序、可重放：每一步都把中间结果写库，失败也能看出停在哪里。
不引入 LangGraph 图（第 1 天验收只要求一条能跑通的最小链路）；图化在预算、
中断恢复需求出现后再做，避免先写一个只有节点的空壳。
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

from insightflow.config.settings import Settings
from insightflow.research.agents.model import (
    Decision,
    GapBrief,
    LLMError,
    OfflineResearchModel,
    ResearchModel,
    ResearchState,
    SourceBrief,
    SourceMaterial,
    Synthesis,
    build_model,
)
from insightflow.research import events
from insightflow.research.budget import Budget, BudgetExceeded, BudgetGuard
from insightflow.research.coverage import THEME_LABELS, Gap, coverage_lines, find_gaps
from insightflow.research.quality.citations import (
    EvidenceRegistry,
    cited_source_ids,
    finalize_findings,
    validate_report,
)
from insightflow.research.report import render_markdown, render_title
from insightflow.research.schemas import (
    CheckStatus,
    FetchScope,
    Finding,
    ReadDepth,
    Report,
    RunCounters,
    RunRecord,
    RunStatus,
    Source,
    SourceStatus,
    stable_id,
)
from insightflow.research.time_window import describe_window, in_window
from insightflow.storage.db import Database
from insightflow.storage.files import SnapshotStore
from insightflow.tools.fetch import FetchedSource, fetch_source
from insightflow.tools.replay import ReplayProvider
from insightflow.tools.search import SearchHit, search_web
from insightflow.tools.urls import canonicalize

MAX_QUERIES = 3
MAX_HITS_PER_QUERY = 5
MAX_SOURCES = 6
MODEL_CHAR_LIMIT = 12_000
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text)


class NoSourcesError(RuntimeError):
    pass


class RunCancelled(RuntimeError):
    """运行在阶段边界发现已被取消。"""



def _final_counters(guard: BudgetGuard, model: Any) -> RunCounters:
    """收尾时的计数：预算用量 + 模型 token（模型侧统计）。"""
    usage = guard.usage()
    model_usage = getattr(model, "usage", None)
    if model_usage is not None:
        usage["input_tokens"] = int(getattr(model_usage, "input_tokens", 0))
        usage["output_tokens"] = int(getattr(model_usage, "output_tokens", 0))
    return RunCounters(**usage)


class ResearchEngine:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        model: ResearchModel | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self._model_override = model
        self.budget = budget or Budget()

    # ------------------------------------------------------------------ run

    def run(self, run_id: str) -> RunRecord:
        record = self.db.get_run(run_id)
        if record is None:
            raise LookupError(f"运行不存在：{run_id}")
        if record.status not in {RunStatus.QUEUED, RunStatus.RECOVERABLE}:
            return record

        request = record.request
        # 条件更新：排队期间被取消的运行不应该被工作线程“复活”。
        if not self.db.update_run(
            run_id,
            expected=record.status,
            status=RunStatus.RUNNING,
            updated_at=datetime.now(UTC),
        ):
            return self.db.get_run(run_id) or record
        self._emit(
            run_id,
            events.RUN_STARTED,
            {
                "mode": request.mode.value,
                "resumed": record.counters.resumes > 0,
                "resumes": record.counters.resumes,
            },
        )

        # 预算用量跟着 run 走：重启后从上次停下的地方继续计数。
        def _save_usage(usage: dict[str, int]) -> None:
            self.db.update_run(run_id, counters=RunCounters(**usage))

        guard = BudgetGuard(self.budget, used=record.counters.model_dump(), on_change=_save_usage)
        source_text: dict[str, str] = {}
        sources: list[Source] = []
        # 续跑时复用已经抓到的正文，不重复花抓取预算，也不再请求网络。
        reused = {
            source.canonical_url: source
            for source in self.db.list_sources(run_id)
            if source.status is SourceStatus.FETCHED
        }
        store = SnapshotStore(self.settings.snapshots_dir)
        for source in reused.values():
            if source.snapshot_path:
                try:
                    source_text[source.source_id] = store.read_text(source.snapshot_path)
                except OSError:
                    reused.pop(source.canonical_url, None)
        limitations: list[str] = []
        stop_reason: str | None = None
        gaps: list[Gap] = []
        model = self._model_override or build_model(
            self.settings, prefer_offline=request.mode.value == "replay"
        )

        try:
            self._ensure_active(run_id)
            queries = self._plan(model, record, guard)
            self._ensure_active(run_id)
            hits, search_failures = self._search(record, queries, guard)
            seen = {hit.url.strip() for hit in hits}
            self._ensure_active(run_id)
            fetched = self._fetch(record, hits, guard, source_text, reused=reused)
            sources = fetched
            readable = [item for item in fetched if item.status is SourceStatus.FETCHED]
            if not readable:
                stop_reason = "search_failed" if search_failures else "no_sources"
                limitations.append(
                    f"本次尝试 {len(fetched)} 个来源，均未成功抓取正文，无法形成带证据的结论。"
                    if fetched
                    else (
                        "检索未成功（" + "、".join(sorted(set(search_failures)))
                        + "），本次没有可读来源；这不等于“没有新动态”。"
                    )
                    if search_failures
                    else "检索成功但没有匹配来源，本次无法形成带证据的结论。"
                )
                synthesis = Synthesis(summary="未能获取可读来源，报告不包含结论。")
            else:
                synthesis, stop_reason, gaps = self._research_rounds(
                    model,
                    record,
                    guard,
                    sources,
                    source_text,
                    seen,
                    stop_reason,
                    limitations,
                )

            self._ensure_active(run_id)
            report, checks, registry = self._build_report(
                record,
                sources,
                synthesis,
                source_text,
                limitations,
                gaps=gaps,
                supplement_rounds=guard.supplement_rounds,
            )
            self._ensure_active(run_id)
            self.db.replace_sources(run_id, sources)
            self.db.add_evidence(run_id, registry.all())
            self.db.save_report(report, report.findings)
            status = RunStatus.COMPLETED if report.status == "published" else RunStatus.PARTIAL
            self._emit(
                run_id,
                events.REPORT_READY,
                {
                    "report_id": report.report_id,
                    "status": str(report.status),
                    "findings": len(report.findings),
                    "cited_sources": len(report.cited_source_ids),
                    "checks": [{"check": check.check, "status": str(check.status)} for check in checks],
                },
            )
            written = self.db.update_run(
                run_id,
                expected=RunStatus.RUNNING,
                status=status,
                report_id=report.report_id,
                stop_reason=stop_reason,
                updated_at=datetime.now(UTC),
                counters=_final_counters(guard, model),
            )
            if written:
                self._emit(
                    run_id,
                    events.RUN_FINISHED,
                    {
                        "status": str(status),
                        "report_id": report.report_id,
                        "stop_reason": stop_reason,
                        "counters": _final_counters(guard, model).model_dump(),
                    },
                )
        except RunCancelled:
            # 取消事件由发起方写入，这里只负责停下来，不改状态（避免覆盖 cancel 的时间戳）。
            return self.db.get_run(run_id) or record
        except BudgetExceeded as exc:
            stop_reason = f"budget:{exc.dimension}"
            limitations.append(str(exc))
            written = self.db.update_run(
                run_id,
                expected=RunStatus.RUNNING,
                status=RunStatus.PARTIAL,
                stop_reason=stop_reason,
                updated_at=datetime.now(UTC),
                error=str(exc),
            )
            if written:
                self._emit(
                    run_id,
                    events.RUN_FINISHED,
                    {
                        "status": str(RunStatus.PARTIAL),
                        "stop_reason": stop_reason,
                        "error": str(exc),
                    },
                )
        except Exception as exc:  # noqa: BLE001 - 顶层兜底，状态必须落库
            written = self.db.update_run(
                run_id,
                expected=RunStatus.RUNNING,
                status=RunStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                updated_at=datetime.now(UTC),
            )
            if written:
                self._emit(
                    run_id,
                    events.RUN_FAILED,
                    {"status": str(RunStatus.FAILED), "error": f"{type(exc).__name__}: {exc}"},
                )
            raise
        return self.db.get_run(run_id) or record

    # ---------------------------------------------------------- cancellation

    def _ensure_active(self, run_id: str) -> None:
        """在阶段边界检查是否已被取消；取消是一种正常结束，不抛到 API 层。"""
        current = self.db.get_run(run_id)
        if current is None or current.status is RunStatus.CANCELLED:
            raise RunCancelled(run_id)

    def _emit(self, run_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
        # 阶段事件只在运行仍活跃时写入；终态事件（finished/failed/cancelled）无条件写入。
        self.db.append_event(
            run_id,
            event_type,
            data,
            only_if_active=event_type not in events.TERMINAL_EVENTS,
        )

    # --------------------------------------------------------------- stages

    def _plan(self, model: ResearchModel, record: RunRecord, guard: BudgetGuard) -> list[str]:
        request = record.request
        guard.take_llm()
        try:
            queries = model.plan(request)
        except LLMError:
            queries = OfflineResearchModel().plan(request)
            self._note_tool(record.run_id, "plan", ok=False, error_code="llm_error")
        deduped: list[str] = []
        for query in queries:
            if query not in deduped:
                deduped.append(query)
        planned = deduped[:MAX_QUERIES] or [request.query]
        self._emit(record.run_id, events.PLAN_READY, {"queries": planned})
        return planned

    def _search(
        self,
        record: RunRecord,
        queries: list[str],
        guard: BudgetGuard,
        *,
        seen: set[str] | None = None,
    ) -> tuple[list[SearchHit], list[str]]:
        request = record.request
        replay = self._replay(request.fixture_id) if request.mode.value == "replay" else None
        seen = set() if seen is None else seen
        hits: list[SearchHit] = []
        failures: list[str] = []
        for query in queries:
            guard.take_search()
            started = time.perf_counter()
            if replay is not None:
                result = replay.search(query, max_results=MAX_HITS_PER_QUERY)
            else:
                result = search_web(
                    query,
                    api_key=self.settings.tavily_api_key,
                    max_results=MAX_HITS_PER_QUERY,
                    timeout=self.settings.http_timeout_seconds,
                )
            duration = int((time.perf_counter() - started) * 1000)
            self._note_tool(
                record.run_id,
                "search",
                ok=result.ok,
                error_code=str(result.error.code) if result.error else None,
                duration_ms=duration,
                request={"query": query},
                metadata=result.metadata,
            )
            if not result.ok or result.data is None:
                code = str(result.error.code) if result.error else "unknown_error"
                failures.append(code)
                self._emit(
                    record.run_id, events.SEARCH_FAILED, {"query": query, "error_code": code}
                )
                continue
            before = len(hits)
            for hit in result.data.hits:
                key = hit.url.strip()
                if key and key not in seen:
                    seen.add(key)
                    hits.append(hit)
            new_hits = len(hits) - before
            self._emit(
                record.run_id,
                events.SEARCH_COMPLETED,
                {
                    "query": query,
                    "hits": len(result.data.hits),
                    "new_hits": new_hits,
                    "deduped": len(result.data.hits) - new_hits,
                },
            )
        return hits, failures

    def _fetch(
        self,
        record: RunRecord,
        hits: list[SearchHit],
        guard: BudgetGuard,
        source_text: dict[str, str],
        *,
        reused: dict[str, Source] | None = None,
    ) -> list[Source]:
        request = record.request
        replay = self._replay(request.fixture_id) if request.mode.value == "replay" else None
        store = SnapshotStore(self.settings.snapshots_dir)
        reused = reused or {}
        sources: list[Source] = list(reused.values())
        for source in reused.values():
            self._emit(
                record.run_id,
                events.SOURCE_FETCHED,
                {
                    "source_id": source.source_id,
                    "url": source.url,
                    "title": source.title,
                    "kind": source.kind.value,
                    "role": source.role.value,
                    "published_at": source.published_at.isoformat() if source.published_at else None,
                    "fetch_scope": source.fetch_scope.value,
                    "read_depth": source.read_depth.value,
                    "reused": True,
                },
            )
        for hit in hits:
            if len(sources) >= MAX_SOURCES:
                break
            key = canonicalize(hit.url)
            if key in reused:
                continue
            guard.take_fetch()
            started = time.perf_counter()
            if replay is not None:
                result = replay.fetch(hit.url)
            else:
                result = fetch_source(
                    hit.url,
                    timeout=self.settings.http_timeout_seconds,
                    max_bytes=self.settings.fetch_max_bytes,
                    user_agent=self.settings.user_agent,
                    allow_private_addresses=self.settings.allow_private_addresses,
                )
            duration = int((time.perf_counter() - started) * 1000)
            self._note_tool(
                record.run_id,
                "fetch",
                ok=result.ok,
                error_code=str(result.error.code) if result.error else None,
                duration_ms=duration,
                request={"url": hit.url},
            )
            if not result.ok or result.data is None:
                failed = self._failed_source(hit, result, record.run_id)
                sources.append(failed)
                self._emit(
                    record.run_id,
                    events.SOURCE_FAILED,
                    {
                        "url": hit.url,
                        "error_code": failed.error_code,
                        "retryable": bool(result.error and result.error.retryable),
                    },
                )
                continue
            source, text = self._to_source(result.data, hit, store, record.run_id)
            source_text[source.source_id] = text
            sources.append(source)
            self._emit(
                record.run_id,
                events.SOURCE_FETCHED,
                {
                    "source_id": source.source_id,
                    "url": source.url,
                    "title": source.title,
                    "kind": str(source.kind),
                    "role": str(source.role),
                    "published_at": source.published_at.isoformat() if source.published_at else None,
                    "fetch_scope": str(source.fetch_scope),
                    "read_depth": str(source.read_depth),
                },
            )
        return sources

    # ------------------------------------------------------- supplement loop

    def _decide(
        self, model: ResearchModel, state: ResearchState
    ) -> tuple[Decision, str, str]:
        """让模型决定下一步；模型不可用时退回规则决策，并如实记录降级原因。"""
        try:
            return model.decide(state), "model", ""
        except LLMError as exc:
            fallback = OfflineResearchModel().decide(state)
            return fallback, "fallback", str(exc)

    def _research_rounds(
        self,
        model: ResearchModel,
        record: RunRecord,
        guard: BudgetGuard,
        sources: list[Source],
        source_text: dict[str, str],
        seen: set[str],
        stop_reason: str | None,
        limitations: list[str],
    ) -> tuple[Synthesis, str | None, list[Gap]]:
        """合成 → 查缺口 →（有缺口且还有预算）补搜 → 再合成，最多 max_supplement_rounds 轮。

        缺口随返回值传出，不放引擎实例上——引擎被多个工作线程共用，放实例上会让
        并发运行的缺口串味。
        """
        request = record.request
        synthesis = Synthesis(summary="无摘要。")
        gaps: list[Gap] = []

        while True:
            self._ensure_active(record.run_id)
            synthesis, degraded = self._synthesize(
                model, record, sources, source_text, guard
            )
            if degraded:
                stop_reason = stop_reason or "model_fallback"
                limitations.append("模型调用失败，已降级为模板抽取；结论范围以原文摘录为准。")
            limitations.extend(synthesis.limitations)

            gaps = find_gaps(request, sources)
            self._emit(
                record.run_id,
                events.GAPS_FOUND,
                {
                    "round": guard.supplement_rounds,
                    "gaps": [gap.as_event() for gap in gaps],
                },
            )
            if not gaps:
                break

            # 要不要补搜由模型判断（缺口只是给它看的事实），代码只守预算与轮次上限。
            state = ResearchState(
                request=request,
                sources=[
                    SourceBrief(
                        source_id=source.source_id,
                        title=source.title,
                        kind=str(source.kind),
                        role=str(source.role),
                        published_at=source.published_at,
                        fetch_scope=str(source.fetch_scope),
                    )
                    for source in sources
                    if source.status is SourceStatus.FETCHED
                ],
                gaps=[
                    GapBrief(
                        kind=gap.kind,
                        theme=gap.theme,
                        label=THEME_LABELS.get(gap.theme or "", gap.theme or ""),
                        reason=gap.reason,
                        queries=gap.queries,
                    )
                    for gap in gaps
                ],
                findings=len(synthesis.findings),
                rounds_used=guard.supplement_rounds,
                rounds_left=self.budget.max_supplement_rounds - guard.supplement_rounds,
                searches_left=guard.remaining_searches(),
            )
            decision, decided_by, decision_error = self._decide(model, state)
            if decision_error:
                limitations.append(
                    f"模型决策失败（{decision_error}），已退回规则决策：按缺口模板补充检索。"
                )
            self._emit(
                record.run_id,
                events.DECISION,
                {
                    "action": decision.action,
                    "reason": decision.reason,
                    "queries": decision.queries,
                    "decided_by": decided_by,
                    "rounds_left": state.rounds_left,
                    "searches_left": state.searches_left,
                },
            )

            if decision.action == "stop":
                if gaps:
                    # 模型说停，但要分清是“判断够了”还是“没预算了”，后者必须如实归因。
                    unresolved = "；".join(gap.reason for gap in gaps)
                    if state.rounds_left <= 0:
                        stop_reason = stop_reason or "supplement_limit"
                        limitations.append(
                            f"补搜已达上限（{self.budget.max_supplement_rounds} 轮），"
                            f"仍有 {len(gaps)} 项覆盖缺口：{unresolved}"
                        )
                    elif state.searches_left <= 0:
                        stop_reason = stop_reason or "budget:max_search_calls"
                        limitations.append("检索预算已用完，无法继续补搜。")
                    else:
                        stop_reason = stop_reason or "model_declined"
                        limitations.append(
                            f"模型判断不再补搜：{decision.reason}；未解决缺口：{unresolved}"
                        )
                break

            if guard.supplement_rounds >= self.budget.max_supplement_rounds:
                stop_reason = stop_reason or "supplement_limit"
                limitations.append(
                    f"补搜已达上限（{self.budget.max_supplement_rounds} 轮），"
                    f"仍有 {len(gaps)} 项覆盖缺口："
                    + "；".join(gap.reason for gap in gaps)
                )
                self._emit(
                    record.run_id,
                    events.SUPPLEMENT_STOPPED,
                    {"reason": "limit", "round": guard.supplement_rounds, "gaps": [g.as_event() for g in gaps]},
                )
                break

            if guard.remaining_searches() <= 0:
                stop_reason = stop_reason or "budget:max_search_calls"
                limitations.append("检索预算已用完，无法继续补搜。")
                self._emit(
                    record.run_id,
                    events.SUPPLEMENT_STOPPED,
                    {"reason": "budget", "round": guard.supplement_rounds},
                )
                break

            guard.take_supplement()
            queries = decision.queries
            if not queries:
                self._emit(
                    record.run_id,
                    events.SUPPLEMENT_STOPPED,
                    {"reason": "no_queries", "round": guard.supplement_rounds},
                )
                break
            self._emit(
                record.run_id,
                events.SUPPLEMENT_STARTED,
                {
                    "round": guard.supplement_rounds,
                    "queries": queries,
                    "reasons": [gap.reason for gap in gaps],
                },
            )

            self._ensure_active(record.run_id)
            new_hits, failures = self._search(record, queries, guard, seen=seen)
            if failures:
                limitations.append(
                    "补搜检索未成功：" + "、".join(sorted(set(failures))) + "。"
                )
            if not new_hits:
                stop_reason = stop_reason or "supplement_no_new_sources"
                limitations.append("补搜没有找到新的来源，停止补搜。")
                self._emit(
                    record.run_id,
                    events.SUPPLEMENT_STOPPED,
                    {"reason": "no_new_sources", "round": guard.supplement_rounds},
                )
                break

            added = self._fetch(record, new_hits, guard, source_text)
            sources.extend(added)
            usable = [item for item in added if item.status is SourceStatus.FETCHED]
            self._emit(
                record.run_id,
                events.SUPPLEMENT_COMPLETED,
                {
                    "round": guard.supplement_rounds,
                    "new_sources": len(usable),
                    "attempted": len(added),
                    "total_sources": len(sources),
                },
            )
        return synthesis, stop_reason, gaps

    def _synthesize(
        self,
        model: ResearchModel,
        record: RunRecord,
        sources: list[Source],
        source_text: dict[str, str],
        guard: BudgetGuard,
    ) -> tuple[Synthesis, bool]:
        materials = [
            SourceMaterial(
                source_id=source.source_id,
                title=source.title,
                url=source.url,
                kind=str(source.kind),
                role=str(source.role),
                published_at=source.published_at,
                text=source_text.get(source.source_id, "")[:MODEL_CHAR_LIMIT],
            )
            for source in sources
            if source.status is SourceStatus.FETCHED
        ]
        guard.take_llm()
        try:
            synthesis = model.synthesize(record.request, materials)
            degraded = False
        except LLMError:
            self._note_tool(record.run_id, "synthesize", ok=False, error_code="llm_error")
            synthesis = OfflineResearchModel().synthesize(record.request, materials)
            degraded = True
        self._emit(
            record.run_id,
            events.SYNTHESIS_READY,
            {"findings": len(synthesis.findings), "degraded": degraded},
        )
        return synthesis, degraded

    # ----------------------------------------------------------------- aide

    def _build_report(
        self,
        record: RunRecord,
        sources: list[Source],
        synthesis: Synthesis,
        source_text: dict[str, str],
        limitations: list[str],
        *,
        gaps: list[Gap] | None = None,
        supplement_rounds: int = 0,
    ) -> tuple[Report, list[Any], EvidenceRegistry]:
        request = record.request
        registry = EvidenceRegistry()
        source_map = {source.source_id: source for source in sources}
        findings: list[Finding] = []
        dropped: list[str] = []

        for index, draft in enumerate(synthesis.findings):
            evidence_ids: list[str] = []
            for ref in draft.evidence:
                source = source_map.get(ref.source_id)
                if source is None:
                    continue
                text = source_text.get(ref.source_id, "")
                if not text or _norm(ref.quote) not in _norm(text):
                    dropped.append(ref.source_id)
                    continue
                evidence = registry.register(source, ref.quote, ref.locator, scope="quote")
                evidence_ids.append(evidence.evidence_id)
            if not evidence_ids:
                continue
            findings.append(
                Finding(
                    finding_id=stable_id("find", record.run_id, str(index), draft.text[:60]),
                    text=draft.text.strip(),
                    claim_type=draft.claim_type,
                    evidence_ids=evidence_ids,
                    support_status=draft.support_status,
                    limitations=draft.limitations,
                )
            )

        single_source = [
            item
            for item in findings
            if len({registry.get(eid).source_id for eid in item.evidence_ids if registry.get(eid)}) == 1
        ]
        if single_source:
            limitations.append(
                f"{len(single_source)} 条结论仅由单一来源支撑，未做独立交叉验证。"
            )
        if dropped:
            limitations.append(
                f"{len(dropped)} 处模型引文未能在原文中逐字找到，已剔除对应结论（不接受编造引用）。"
            )

        kept = finalize_findings(findings, registry, source_map)
        cited = cited_source_ids(kept, registry)
        fetched_count = sum(1 for item in sources if item.status is SourceStatus.FETCHED)
        scope_text = (
            f"时间窗口 {describe_window(request)}"
            f"｜as_of {request.as_of.isoformat()}"
            f"｜成功读取 {fetched_count}/{len(sources)} 个来源"
            f"｜模式 {request.mode.value}"
        )

        report = Report(
            report_id=stable_id("rep", record.run_id),
            run_id=record.run_id,
            created_at=datetime.now(UTC),
            title=render_title(request.query, str(request.task_type)),
            scope_text=scope_text,
            summary=synthesis.summary.strip() or "无摘要。",
            findings=kept,
            limitations=list(dict.fromkeys(item for item in limitations if item)),
            cited_source_ids=cited,
            coverage=coverage_lines(request, sources, list(gaps or []), supplement_rounds),
            supplement_rounds=supplement_rounds,
        )
        checks = validate_report(
            report,
            registry,
            source_map,
            window_start=request.window_start,
            window_end=request.window_end,
        )
        report.checks = checks
        outside = sum(
            1
            for item in sources
            if item.status is SourceStatus.FETCHED
            and not in_window(item.published_at, request.window_start, request.window_end)
        )
        if supplement_rounds:
            report.scope_text += f"｜补搜 {supplement_rounds} 轮"
        if gaps:
            report.scope_text += f"｜未解决覆盖缺口 {len(gaps)} 项"
        if outside:
            report.scope_text += f"｜窗口外来源 {outside} 个（不计为窗口内新增）"
            report.limitations.append(
                f"{outside} 个来源的发布时间早于窗口起点或缺失，相关内容不构成本窗口内的新增动态。"
            )
        hard_fail = any(check.status is CheckStatus.FAIL for check in checks)
        warn = any(check.status is CheckStatus.WARN for check in checks)
        insufficient = any(
            finding.support_status.value == "insufficient" for finding in kept
        )
        if not kept or hard_fail or insufficient or gaps:
            report.status = "partial"
        elif warn:
            report.status = "partial"
        else:
            report.status = "published"
        report.markdown = render_markdown(report, sources, registry.all())
        return report, checks, registry

    def _replay(self, fixture_id: str | None) -> ReplayProvider:
        if not fixture_id:
            raise ValueError("回放模式缺少 fixture_id")
        return ReplayProvider(self.settings.fixtures_dir, fixture_id)

    def _to_source(
        self, fetched: FetchedSource, hit: SearchHit, store: SnapshotStore, run_id: str
    ) -> tuple[Source, str]:
        source_id = stable_id("src", run_id, fetched.canonical_url)
        snapshot = store.save_text(run_id, source_id, fetched.text)
        text = fetched.text
        source = Source(
            source_id=source_id,
            url=fetched.url,
            canonical_url=fetched.canonical_url,
            title=fetched.title or hit.title or fetched.canonical_url,
            kind=fetched.kind,
            role=fetched.role,
            published_at=fetched.published_at or hit.published_at,
            updated_at=fetched.updated_at,
            retrieved_at=fetched.fetched_at,
            snapshot_path=snapshot,
            content_hash=fetched.content_hash,
            fetch_scope=fetched.fetch_scope,
            read_depth=fetched.read_depth(MODEL_CHAR_LIMIT),
            status=SourceStatus.FETCHED,
            provider=fetched.provider,
        )
        return source, text

    def _failed_source(self, hit: SearchHit, result: Any, run_id: str) -> Source:
        return Source(
            source_id=stable_id("src", run_id, hit.url),
            url=hit.url,
            canonical_url=hit.url,
            title=hit.title or hit.url,
            published_at=hit.published_at,
            retrieved_at=datetime.now(UTC),
            fetch_scope=FetchScope.NONE,
            read_depth=ReadDepth.NONE,
            status=SourceStatus.FAILED,
            error_code=str(result.error.code) if result.error else "unknown_error",
            provider="replay" if result.metadata.get("fixture") else "http",
        )

    def _note_tool(
        self,
        run_id: str,
        tool: str,
        *,
        ok: bool,
        error_code: str | None = None,
        duration_ms: int = 0,
        request: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.db.log_tool_call(
            run_id,
            tool,
            ok=ok,
            error_code=error_code,
            started_at=datetime.now(UTC),
            duration_ms=duration_ms,
            request=request,
            metadata=metadata,
        )
