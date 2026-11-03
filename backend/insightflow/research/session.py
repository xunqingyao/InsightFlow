"""给宿主 Agent 用的研究工具层：这里**没有编排**。

一句话说清这个文件在做什么：把"检索、读原文、登记证据、校验引文、出报告"变成
一组可以单独调用的工具，由外部的 agent（DeerFlow 的 lead agent、Claude Desktop、
任何 MCP 客户端）决定什么时候调用哪一个。

所以这里刻意不做的事：
- 不决定搜几条、读几篇、要不要补搜——那是调用方（agent）的判断；
- 不假装模型在场——没有 LLM 调用，也不需要 API key；
- 不在内部循环——每次调用只做一件事，返回结构化结果。

调用方仍然拿得到质量保证：来源必须真的抓取成功才登记、引文必须逐字出现在快照里
才收下、报告只能引用本次登记过的证据、预算耗尽时明确回 `budget_exceeded`。

典型调用顺序（也是 skill 里教给 agent 的顺序）：

    start -> search_web / arxiv / github -> read_source -> submit_finding
          -> coverage -> (再搜) -> publish

进程内以 run_id 持有会话；来源、证据、报告落 SQLite，正文落文件快照，
所以工具进程重启后 `attach(run_id)` 仍能继续校验与出报告。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from insightflow.config.settings import Settings
from insightflow.research import events
from insightflow.research.budget import Budget, BudgetExceeded, BudgetGuard
from insightflow.research.coverage import Gap, find_gaps, required_themes
from insightflow.research.quality.citations import EvidenceRegistry, validate_report
from insightflow.research.report import render_markdown, render_title
from insightflow.research.schemas import (
    ClaimType,
    Evidence,
    FetchScope,
    Finding,
    ReadDepth,
    Report,
    ResearchRequest,
    RunCounters,
    RunMode,
    RunRecord,
    RunStatus,
    Source,
    SourceKind,
    SourceRole,
    SourceStatus,
    SupportStatus,
    stable_id,
)
from insightflow.research.time_window import describe_window, in_window, normalize, now_in
from insightflow.storage.db import Database
from insightflow.storage.files import SnapshotStore
from insightflow.tools.arxiv import search_arxiv
from insightflow.tools.contracts import ToolResult
from insightflow.tools.fetch import fetch_source
from insightflow.tools.github import get_releases
from insightflow.tools.replay import ReplayProvider
from insightflow.tools.search import search_web
from insightflow.tools.urls import canonicalize

MAX_QUOTE_CHARS = 2000
MIN_QUOTE_CHARS = 12
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text or "")


def _log(db: Database, run_id: str, tool: str, *, ok: bool, request: str = "", error: str | None = None) -> None:
    now = now_in("UTC")
    db.log_tool_call(
        run_id,
        tool,
        ok=ok,
        error_code=error,
        started_at=now,
        duration_ms=0,
        request=request,
    )


def ok(data: Any, **metadata: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "metadata": metadata}


def fail(code: str, message: str, *, retryable: bool = False, **metadata: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message, "retryable": retryable, "retry_after": None},
        "metadata": metadata,
    }


@dataclass
class Session:
    """一次研究运行的上下文；工具函数都拿它当第一个参数。"""

    run_id: str
    request: ResearchRequest
    settings: Settings
    db: Database
    guard: BudgetGuard
    store: SnapshotStore
    replay: ReplayProvider | None = None
    sources: dict[str, Source] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    evidence: EvidenceRegistry = field(default_factory=EvidenceRegistry)
    report: Report | None = None

    # ---- 内部小工具 -------------------------------------------------------
    def source(self, source_id: str) -> Source | None:
        return self.sources.get(source_id)

    def register_source(self, source: Source, text: str = "") -> Source:
        self.sources[source.source_id] = source
        if text:
            self.texts[source.source_id] = text
        self.db.replace_sources(self.run_id, list(self.sources.values()))
        return source

    def emit(self, type_: str, data: dict[str, Any]) -> None:
        self.db.append_event(self.run_id, type_, data)

    def _budget_left(self) -> dict[str, Any]:
        return {
            "searches_left": self.guard.remaining_searches(),
            "max_search_calls": self.guard.budget.max_search_calls,
            "max_source_fetches": self.guard.budget.max_source_fetches,
            "max_llm_calls": self.guard.budget.max_llm_calls,
        }




def _counters(guard: BudgetGuard) -> RunCounters:
    usage = guard.usage()
    usage["llm_calls"] = 0  # 会话层不调模型，模型调用在宿主侧
    return RunCounters(**usage)


def _persist_usage(db: Database, run_id: str):
    """把预算用量写回 run，重启/换进程后能接着算。"""

    def _save(usage: dict[str, int]) -> None:
        db.update_run(run_id, counters=RunCounters(**usage))

    return _save


def _guard_for(db: Database, record: RunRecord, budget: Budget | None = None) -> BudgetGuard:
    used = record.counters.model_dump() if record.counters else {}
    return BudgetGuard(budget or Budget(), used=used, on_change=_persist_usage(db, record.run_id))


def start(
    settings: Settings,
    question: str,
    *,
    as_of: datetime | None = None,
    days: int = 7,
    focus: list[str] | None = None,
    mode: RunMode = RunMode.ONLINE,
    fixture_id: str | None = None,
    budget: Budget | None = None,
) -> tuple[Session, dict[str, Any]]:
    """开一次运行：固定 as_of 与窗口，建立库记录，返回 run_id 与下一步提示。"""
    db = Database(settings.db_path)
    db.init_schema()
    resolved_as_of = as_of or now_in(settings.timezone)
    try:
        request = normalize(
            question,
            as_of=resolved_as_of,
            window=timedelta(days=days),
            focus=focus,
            mode=mode,
            fixture_id=fixture_id,
            timezone=settings.timezone,
        )
    except ValueError as exc:
        return None, fail("invalid_input", str(exc))  # type: ignore[return-value]

    run_id = stable_id("run", request.query, request.as_of.isoformat(), request.window_start.isoformat())
    if db.get_run(run_id) is not None:
        return None, fail(
            "duplicate_run",
            "同样的问题和 as_of 已经跑过；换 as_of 或复用已有 run_id。",
            run_id=run_id,
        )

    created = now_in(settings.timezone)
    db.create_run(
        RunRecord(
            run_id=run_id,
            status=RunStatus.RUNNING,
            request=request,
            created_at=created,
            updated_at=created,
        )
    )
    db.append_event(run_id, events.RUN_STARTED, {"mode": "tools", "query": request.query})

    replay = None
    if request.mode is RunMode.REPLAY:
        fixture = request.fixture_id or ""
        try:
            replay = ReplayProvider(settings.fixtures_dir, fixture)
        except (OSError, ValueError) as exc:
            db.update_run(run_id, status=RunStatus.FAILED, error=f"fixture 不可用：{exc}")
            return None, fail("fixture_unavailable", f"回放资料不可用：{exc}", run_id=run_id)

    session = Session(
        run_id=run_id,
        request=request,
        settings=settings,
        db=db,
        guard=_guard_for(db, db.get_run(run_id), budget),  # type: ignore[arg-type]
        store=SnapshotStore(settings.snapshots_dir),
        replay=replay,
    )
    return session, ok(
        {
            "run_id": run_id,
            "query": request.query,
            "mode": request.mode.value,
            "window": describe_window(request),
            "window_start": request.window_start.isoformat(),
            "window_end": request.window_end.isoformat(),
            "as_of": request.as_of.isoformat(),
            "timezone": settings.timezone,
            "required_themes": required_themes(request),
            "budget": session._budget_left(),
            "next": (
                "用 search_web / arxiv_search / github_releases 找候选来源，再 read_source 读正文；"
                "结论用 submit_finding 提交（引文必须逐字来自 read_source 返回的正文），最后 publish_report。"
            ),
        },
        run_id=run_id,
    )


def attach(settings: Settings, run_id: str) -> tuple[Session | None, dict[str, Any]]:
    """重新挂上已有的 run（工具进程重启后继续用）。"""
    db = Database(settings.db_path)
    record = db.get_run(run_id)
    if record is None:
        return None, fail("not_found", f"没有这个运行：{run_id}")
    session = Session(
        run_id=run_id,
        request=record.request,
        settings=settings,
        db=db,
        guard=_guard_for(db, record),
        store=SnapshotStore(settings.snapshots_dir),
        replay=(
            ReplayProvider(settings.fixtures_dir, record.request.fixture_id or "")
            if record.request.mode is RunMode.REPLAY
            else None
        ),
    )
    for source in db.list_sources(run_id):
        session.sources[source.source_id] = source
        if source.snapshot_path:
            try:
                session.texts[source.source_id] = session.store.read_text(source.snapshot_path)
            except OSError:
                pass
    for item in db.list_evidence(run_id):
        session.evidence.register_existing(item)
    session.findings = db.list_draft_findings(run_id)
    return session, ok(
        {
            "run_id": run_id,
            "sources": len(session.sources),
            "evidence": len(session.evidence.all()),
            "findings": len(session.findings),
        }
    )



def resume(settings: Settings, run_id: str) -> tuple[Session | None, dict[str, Any]]:
    """把停在 `recoverable`/`running` 的 run 重新挂起来继续用。

    状态先做一次条件迁移（只有一个进程能抢到），再用 attach 恢复来源、证据、
    草稿结论与预算计数。真正的“继续做事”由宿主决定（接着调工具）。
    """
    db = Database(settings.db_path)
    record = db.get_run(run_id)
    if record is None:
        return None, fail("not_found", f"没有这个运行：{run_id}")
    if record.status in {RunStatus.COMPLETED, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.CANCELLED}:
        return None, fail(
            "invalid_state",
            f"运行已经结束（{record.status.value}），没有可续的部分；要重跑请用同一个问题重新发起。",
            status=record.status.value,
        )
    # 工具模式下 `running` 只代表“这个 run 还没结束”，不代表进程还活着：
    # 每次 MCP 调用都是新进程，所以 running 直接重挂，不改状态。
    if record.status is not RunStatus.RUNNING:
        if not db.update_run(run_id, status=RunStatus.RUNNING, expected=record.status):
            current = db.get_run(run_id)
            return None, fail(
                "conflict",
                "另一个进程正在接手这个运行",
                status=current.status.value if current else None,
            )
    db.update_run(run_id, counters=RunCounters(**{**record.counters.model_dump(), "resumes": record.counters.resumes + 1}))
    session, payload = attach(settings, run_id)
    if session is None:
        return None, payload
    session.emit(events.RUN_STARTED, {"resumed": True, "from": record.status.value})
    payload["resumed"] = True
    payload["budget"] = {
        "searches_left": session.guard.remaining_searches(),
        "fetches_left": session.guard.remaining_fetches(),
        "llm_calls_left": session.guard.remaining_llm_calls(),
        "supplement_rounds_used": session.guard.supplement_rounds,
    }
    return session, ok(payload)


def search(session: Session, query: str, *, max_results: int = 5) -> dict[str, Any]:
    """检索候选来源。返回的是候选，不是证据；正文要靠 read_source 才拿到。"""
    try:
        session.guard.take_search()
    except BudgetExceeded as exc:
        return fail(
            "budget_exceeded",
            f"检索次数已用完（上限 {exc.limit}）；请基于已有来源出报告，或在局限里写明。",
            dimension=exc.dimension,
        )

    try:
        if session.replay is not None:
            result = session.replay.search(query, max_results=max_results)
        else:
            result = search_web(
                query,
                api_key=session.settings.tavily_api_key,
                max_results=max_results,
                timeout=session.settings.http_timeout_seconds,
            )
    except Exception as exc:  # 适配器已把常见错误变成信封，这里兜底
        return fail("upstream_error", f"检索失败：{exc}", retryable=True)

    _log(session.db, session.run_id, "search_web", ok=result.ok, request=query)
    if not result.ok:
        error = result.error
        session.emit(events.SEARCH_FAILED, {"query": query, "error_code": error.code.value if error else "unknown"})
        return fail(
            error.code.value if error else "upstream_error",
            error.message if error else "检索失败",
            retryable=bool(error and error.retryable),
        )

    bundle = result.data
    session.emit(events.SEARCH_COMPLETED, {"query": query, "hits": len(bundle.hits)})
    hits = []
    for hit in bundle.hits:
        source_id = stable_id("src", session.run_id, canonicalize(hit.url))
        already = source_id in session.sources
        hits.append(
            {
                "url": hit.url,
                "title": hit.title,
                "snippet": hit.snippet,
                "published_at": hit.published_at.isoformat() if hit.published_at else None,
                "already_read": already,
                "source_id_if_read": source_id if already else None,
            }
        )
    return ok(
        hits,
        query=query,
        hint="snippet 只能当线索；引用必须来自 read_source 返回的正文。",
        budget=session._budget_left(),
    )


def read(session: Session, url: str) -> dict[str, Any]:
    """读一篇来源的正文：抓取、存快照、登记来源，返回可引用的正文。"""
    try:
        session.guard.take_fetch()
    except BudgetExceeded as exc:
        return fail(
            "budget_exceeded",
            f"抓取次数已用完（上限 {exc.limit}）；请基于已读来源出报告。",
            dimension=exc.dimension,
        )

    canonical = canonicalize(url)
    source_id = stable_id("src", session.run_id, canonical)
    if source_id in session.sources and source_id in session.texts:
        source = session.sources[source_id]
        return ok(_source_payload(session, source, session.texts[source_id]), cached=True)

    result: ToolResult[Any]
    if session.replay is not None:
        result = session.replay.fetch(url)
    else:
        result = fetch_source(
            url,
            timeout=session.settings.http_timeout_seconds,
            max_bytes=session.settings.fetch_max_bytes,
            user_agent=session.settings.user_agent,
            allow_private_addresses=session.settings.allow_private_addresses,
        )

    _log(session.db, session.run_id, "read_source", ok=result.ok, request=url)
    if not result.ok:
        error = result.error
        code = error.code.value if error else "upstream_error"
        failed = Source(
            source_id=source_id,
            canonical_url=canonical,
            url=url,
            title=url,
            status=SourceStatus.FAILED,
            error_code=code,
            fetch_scope=FetchScope.NONE,
            read_depth=ReadDepth.NONE,
            retrieved_at=now_in(session.settings.timezone),
        )
        session.register_source(failed)
        session.emit(events.SOURCE_FAILED, {"url": url, "error_code": code})
        return fail(code, error.message if error else "抓取失败", retryable=bool(error and error.retryable))

    fetched = result.data
    text = fetched.text
    source = Source(
        source_id=source_id,
        canonical_url=canonical,
        url=fetched.final_url or fetched.url,
        title=fetched.title or fetched.url,
        kind=fetched.kind,
        role=fetched.role,
        published_at=fetched.published_at,
        status=SourceStatus.FETCHED,
        fetch_scope=fetched.fetch_scope,
        read_depth=fetched.read_depth(12_000),
        retrieved_at=now_in(session.settings.timezone),
        content_hash=fetched.content_hash,
        provider=fetched.provider,
    )
    source.snapshot_path = session.store.save_text(session.run_id, source_id, text)
    session.register_source(source, text)
    session.emit(
        events.SOURCE_FETCHED,
        {
            "source_id": source_id,
            "url": source.url,
            "title": source.title,
            "published_at": source.published_at.isoformat() if source.published_at else None,
            "fetch_scope": source.fetch_scope.value,
        },
    )
    return ok(_source_payload(session, source, text))


def _source_payload(session: Session, source: Source, text: str, *, limit: int = 12_000) -> dict[str, Any]:
    in_scope = in_window(source.published_at, session.request)
    return {
        "source_id": source.source_id,
        "title": source.title,
        "url": source.url,
        "kind": source.kind.value,
        "role": source.role.value,
        "published_at": source.published_at.isoformat() if source.published_at else None,
        "in_window": in_scope,
        "fetch_scope": source.fetch_scope.value,
        "read_depth": source.read_depth.value,
        "text": text[:limit],
        "truncated_for_prompt": len(text) > limit,
        "hint": "引用时必须从 text 里原样复制一段连续文字，不能改写、翻译或拼接。",
    }


def coverage(session: Session) -> dict[str, Any]:
    """报告覆盖缺口：哪些主题没有来源、可读来源是否太少、是否还有时间补。"""
    sources = list(session.sources.values())
    gaps = find_gaps(session.request, sources)
    readable = [s for s in sources if s.status is SourceStatus.FETCHED]
    return ok(
        {
            "readable_sources": len(readable),
            "failed_sources": len(sources) - len(readable),
            "gaps": [gap.as_event() for gap in gaps],
            "suggested_queries": [q for gap in gaps for q in gap.queries][:4],
            "budget": session._budget_left(),
            "next": (
                "有缺口且还有检索预算就再 search + read；没有就 publish_report，"
                "缺口会写进报告的覆盖情况，状态为 partial。"
            ),
        },
        run_id=session.run_id,
    )


def _prepare_finding(
    session: Session,
    text: str,
    *,
    claim_type: str,
    support_status: str,
    evidence: list[dict[str, Any]] | None,
    limitations: list[str] | None,
) -> tuple[Finding | None, str | None, list[dict[str, Any]], str]:
    """校验一条结论，返回 (finding, error_code, problems, message)。"""
    text = (text or "").strip()
    if not text:
        return None, "invalid_input", [], "结论正文不能为空。"
    try:
        claim = ClaimType(claim_type)
        support = SupportStatus(support_status)
    except ValueError as exc:
        return None, "invalid_input", [], f"claim_type / support_status 取值不合法：{exc}"

    items = evidence or []
    if not items:
        return None, "evidence_required", [], "结论必须带至少一条证据；没有证据的结论会被丢弃。"

    problems: list[dict[str, Any]] = []
    prepared: list[Evidence] = []
    for index, item in enumerate(items, start=1):
        source_id = str(item.get("source_id") or "")
        quote = str(item.get("quote") or "").strip()
        locator = str(item.get("locator") or "正文").strip() or "正文"
        source = session.source(source_id)
        if source is None:
            problems.append({"item": index, "problem": "source_not_in_run", "source_id": source_id})
            continue
        if source.status is not SourceStatus.FETCHED or source_id not in session.texts:
            problems.append({"item": index, "problem": "source_not_readable", "source_id": source_id})
            continue
        if len(quote) < MIN_QUOTE_CHARS:
            problems.append({"item": index, "problem": "quote_too_short", "min_chars": MIN_QUOTE_CHARS})
            continue
        if len(quote) > MAX_QUOTE_CHARS:
            problems.append({"item": index, "problem": "quote_too_long", "max_chars": MAX_QUOTE_CHARS})
            continue
        if _norm(quote) not in _norm(session.texts[source_id]):
            problems.append(
                {
                    "item": index,
                    "problem": "quote_not_verbatim",
                    "source_id": source_id,
                    "hint": "从 read_source 返回的 text 里复制连续原文，不要改写或翻译。",
                }
            )
            continue
        prepared.append(
            session.evidence.register(
                source,
                quote=quote,
                locator=locator,
                scope=source.fetch_scope.value,
            )
        )

    if problems:
        return None, "evidence_mismatch", problems, "有证据不满足要求，整条结论未收下；请按 details 修正后重新提交。"

    session.db.add_evidence(session.run_id, prepared)
    finding = Finding(
        finding_id=stable_id("fnd", session.run_id, text[:200], str(len(session.findings))),
        text=text,
        claim_type=claim,
        evidence_ids=[item.evidence_id for item in prepared],
        support_status=support,
        limitations=[line for line in (limitations or []) if line.strip()],
    )
    session.findings.append(finding)
    session.db.replace_draft_findings(session.run_id, session.findings)
    return finding, None, [], ""


def submit_finding(
    session: Session,
    text: str,
    *,
    claim_type: str = "publisher_claim",
    support_status: str = "partial",
    evidence: list[dict[str, Any]] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """提交一条结论。引文逐字对不上就整条退回，并告诉调用方具体哪一条错。"""
    finding, code, problems, message = _prepare_finding(
        session,
        text,
        claim_type=claim_type,
        support_status=support_status,
        evidence=evidence,
        limitations=limitations,
    )
    if finding is None:
        extra: dict[str, Any] = {"details": problems} if problems else {}
        if code == "invalid_input":
            extra = {
                "allowed_claims": [item.value for item in ClaimType],
                "allowed_support": [item.value for item in SupportStatus],
            }
        return fail(code or "invalid_input", message, **extra)
    return ok(
        {
            "finding_id": finding.finding_id,
            "evidence_ids": finding.evidence_ids,
            "position": len(session.findings),
        },
        total_findings=len(session.findings),
        next="继续 submit_finding，或 publish_report 出报告。",
    )


def submit_findings(session: Session, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """一次提交多条结论：逐条独立校验，好的收下，坏的按条退回。

    逐条独立是刻意的——一条引文写错不该让另外九条白交；调用方只需要重交被退回的那几条。
    """
    if not findings:
        return fail("invalid_input", "findings 为空：至少给一条结论。")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(findings, start=1):
        if not isinstance(item, dict):
            rejected.append({"index": index, "code": "invalid_input", "message": "每条结论必须是对象"})
            continue
        finding, code, problems, message = _prepare_finding(
            session,
            str(item.get("text") or ""),
            claim_type=str(item.get("claim_type") or "publisher_claim"),
            support_status=str(item.get("support_status") or "partial"),
            evidence=item.get("evidence") or [],
            limitations=item.get("limitations") or [],
        )
        if finding is None:
            rejected.append({"index": index, "code": code, "message": message, "details": problems})
            continue
        accepted.append(
            {
                "index": index,
                "finding_id": finding.finding_id,
                "evidence_ids": finding.evidence_ids,
                "position": len(session.findings),
            }
        )

    if not accepted:
        return fail(
            "evidence_mismatch",
            f"{len(rejected)} 条结论都没有通过校验，一条也没收下。",
            details=rejected,
            rejected=len(rejected),
        )
    return ok(
        {"accepted": accepted, "rejected": rejected},
        total_findings=len(session.findings),
        next=(
            f"还有 {len(rejected)} 条被退回，按 details 改完可以只重交这几条；"
            "否则继续检索，或 publish_report 出报告。"
            if rejected
            else "全部收下，继续检索或 publish_report 出报告。"
        ),
    )


_NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*(%|％|个百分点|percent)")
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_QUANTITY = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|[零一二两三四五六七八九十百千]{1,6})\s*"
    r"(?P<unit>个百分点|%|％|步|个|篇|年|个月)"
)


def _cn_to_number(text: str) -> float | None:
    """把“一百二十”“十一”这类中文数字转成数值；认不出就返回 None。"""
    if not text or any(ch not in _CN_DIGITS and ch not in "十百千" for ch in text):
        return None
    total, section, digit = 0, 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
            continue
        factor = {"十": 10, "百": 100, "千": 1000}[ch]
        section += (digit or 1) * factor
        digit = 0
    total += section + digit
    return float(total)


def _figures(text: str, limit: int = 8) -> list[dict[str, str]]:
    """抽出可对照的量：百分点、百分比、步数、数量，各带一小段上下文。

    中文数字也认（“十一个百分点”），否则中文材料里几乎抽不到东西。
    """
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _QUANTITY.finditer(text):
        before = text[: match.start()].rstrip()
        if before.endswith("第") or (before and before[-1] in "每单同逐半"):
            continue  # “第五步”“每一步”“单步”是序数或修饰，不是可对照的量
        raw, unit = match.group("num"), match.group("unit")
        number = float(raw) if raw[0].isdigit() else _cn_to_number(raw)
        if number is None:
            continue
        unit = "%" if unit == "％" else unit
        value = f"{number:g}{unit}" if unit not in {"个百分点", "个", "篇", "年", "步"} else f"{number:g} {unit}"
        if value in seen:
            continue
        seen.add(value)
        context = text[max(0, match.start() - 40) : match.end() + 40].replace("\n", " ").strip()
        found.append({"kind": "quantity", "value": value, "context": context})
        if len(found) >= limit:
            return found

    for match in _VERSION.finditer(text):
        value = match.group(0)
        if value in seen:
            continue
        seen.add(value)
        context = text[max(0, match.start() - 40) : match.end() + 40].replace("\n", " ").strip()
        found.append({"kind": "version", "value": value, "context": context})
        if len(found) >= limit:
            break
    return found


def compare_sources(session: Session, source_ids: list[str] | None = None) -> dict[str, Any]:
    """把几个来源里的数字与版本号摆在一起，看谁说的是同一个数。

    这是文本层面的对照，不是事实核查：它只能说明两个来源有没有引用同一个数字，
    不能说明这个数字对不对。
    """
    readable = [sid for sid, source in session.sources.items() if source.status is SourceStatus.FETCHED]
    picked = [sid for sid in (source_ids or readable[:3]) if sid in session.texts]
    if len(picked) < 2:
        return fail(
            "insufficient_sources",
            "至少要有两个已读来源才能对照；先 read_source。",
            readable=len(readable),
        )

    per_source: list[dict[str, Any]] = []
    owners: dict[str, set[str]] = {}
    for source_id in picked:
        source = session.sources[source_id]
        figures = _figures(session.texts[source_id])
        per_source.append(
            {
                "source_id": source_id,
                "title": source.title,
                "role": source.role.value,
                "figures": figures,
            }
        )
        for figure in figures:
            owners.setdefault(figure["value"], set()).add(source_id)

    shared = sorted(value for value, who in owners.items() if len(who) >= 2)
    only_one = [
        {"value": value, "source_ids": sorted(who)} for value, who in owners.items() if len(who) == 1
    ]
    return ok(
        {"sources": per_source, "shared_values": shared, "only_in_one": only_one},
        caveat=(
            "这是文本层面的数字对照，不是事实核查：shared_values 只说明两个来源出现了同一个数字，"
            "转载常直接抄数字，不代表独立验证；only_in_one 也不能说明那个来源错了。"
        ),
    )


def _stop_reason(failed: list[Any], unresolved: list[Any], warned: list[Any]) -> str:
    """降级原因分开写：校验失败和“覆盖不够”是两件事，别混成一个词。"""
    if failed:
        return "checks_failed"
    if unresolved:
        return "unresolved_coverage"
    if warned:
        return "checks_warned"
    return ""

def publish(
    session: Session,
    summary: str,
    *,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """出报告：登记过的证据 + 七项校验 + Markdown。没有结论时拒绝出版。"""
    if not session.findings:
        return fail("no_findings", "还没有任何结论，不能出报告；先 submit_finding。")

    summary = (summary or "").strip()
    if not summary:
        return fail("invalid_input", "摘要不能为空。")

    sources = list(session.sources.values())
    gaps = find_gaps(session.request, sources)
    evidence = session.evidence.all()
    created = now_in(session.settings.timezone)

    # 先算“实际引用到的来源”，再交给校验：否则 cited_sources 永远看到一份空清单，
    # 任何真实报告都会被判 fail —— 这是曾经发生过的缺陷。
    cited: list[str] = []
    for finding in session.findings:
        for evidence_id in finding.evidence_ids:
            item = session.evidence.get(evidence_id)
            if item and item.source_id not in cited:
                cited.append(item.source_id)
    report = Report(
        report_id=stable_id("rep", session.run_id, str(len(session.findings)), created.isoformat()),
        run_id=session.run_id,
        version=1,
        created_at=created,
        title=render_title(session.request.query, session.request.task_type.value),
        scope_text=_scope_text(session, gaps),
        summary=summary,
        findings=list(session.findings),
        limitations=[line for line in (limitations or []) if line.strip()],
        coverage=_coverage_lines(session, gaps),
        cited_source_ids=cited,
        checks=[],
    )
    checks = validate_report(
        report,
        session.evidence,
        {source.source_id: source for source in sources},
        window_start=session.request.window_start,
        window_end=session.request.window_end,
    )
    report = report.model_copy(update={"checks": checks})

    failed = [check for check in checks if check.status.value == "fail"]
    warned = [check for check in checks if check.status.value == "warn"]
    unresolved = [gap for gap in gaps if gap.kind in {"theme_missing", "no_sources"}]
    if failed:
        report = report.model_copy(update={"status": "partial"})
        report.limitations.append("有质量校验未通过，报告状态降为 partial；详见校验表。")
    elif unresolved or warned:
        report = report.model_copy(update={"status": "partial"})
    else:
        report = report.model_copy(update={"status": "published"})

    report = report.model_copy(update={"markdown": render_markdown(report, sources, evidence)})

    session.store.save_report(
        session.settings.reports_dir, session.run_id, report.report_id, report.markdown
    )
    session.db.save_report(report, report.findings)
    session.db.update_run(
        session.run_id,
        status=RunStatus.COMPLETED if report.status == "published" else RunStatus.PARTIAL,
        report_id=report.report_id,
        stop_reason=_stop_reason(failed, unresolved, warned),
        counters=_counters(session.guard),
    )
    session.report = report
    session.emit(
        events.REPORT_READY,
        {
            "report_id": report.report_id,
            "status": report.status,
            "findings": len(report.findings),
            "checks": [{"check": c.check, "status": c.status.value} for c in checks],
        },
    )
    session.emit(
        events.RUN_FINISHED,
        {"status": "completed" if report.status == "published" else "partial", "report_id": report.report_id},
    )
    return ok(
        {
            "report_id": report.report_id,
            "status": report.status,
            "findings": len(report.findings),
            "checks": [{"check": c.check, "status": c.status.value, "detail": c.detail} for c in checks],
            "cited_sources": cited,
            "markdown": report.markdown,
        },
        run_id=session.run_id,
    )


def _scope_text(session: Session, gaps: list[Gap]) -> str:
    request = session.request
    scope = describe_window(request)
    readable = [s for s in session.sources.values() if s.status is SourceStatus.FETCHED]
    scope += f"｜来源 {len(readable)} 个"
    outside = [
        s
        for s in readable
        if s.published_at is not None and not in_window(s.published_at, request)
    ]
    if outside:
        scope += f"｜窗口外来源 {len(outside)} 个（不计为窗口内新增）"
    if gaps:
        scope += f"｜未解决覆盖缺口 {len(gaps)} 项"
    return scope


def _coverage_lines(session: Session, gaps: list[Gap]) -> list[str]:
    from insightflow.research.coverage import coverage_lines

    return coverage_lines(session.request, list(session.sources.values()), gaps, 0)


def status(session: Session) -> dict[str, Any]:
    """当前状态：读了几篇、几条结论、预算还剩多少、能否出报告。

    报告是否存在以数据库为准，不能只看进程里的 `session.report`——否则换一个
    进程挂上已经发布的 run，会误报成 `running`。
    """
    stored = session.db.latest_report(session.run_id)
    report_status = stored.status if stored is not None else "running"
    finding_count = len(stored.findings) if stored is not None else len(session.findings)
    return ok(
        {
            "run_id": session.run_id,
            "status": report_status,
            "report_id": stored.report_id if stored is not None else None,
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "published_at": source.published_at.isoformat() if source.published_at else None,
                    "status": source.status.value,
                }
                for source in session.sources.values()
            ],
            "findings": finding_count,
            "budget": session._budget_left(),
            "counters": session.guard.usage(),
            "can_publish": bool(session.findings),
        }
    )


def arxiv_tool(session: Session, query: str, *, max_results: int = 5) -> dict[str, Any]:
    """检索 arXiv 元数据（不含全文）；要引用摘要得先用返回的 abs_url 走 read_source。"""
    try:
        session.guard.take_search()
    except BudgetExceeded as exc:
        return fail("budget_exceeded", f"检索次数已用完（上限 {exc.limit}）。", dimension=exc.dimension)
    result = search_arxiv(query, max_results=max_results, timeout=session.settings.http_timeout_seconds)
    _log(session.db, session.run_id, "arxiv_search", ok=result.ok, request=query)
    if not result.ok:
        error = result.error
        return fail(error.code.value if error else "upstream_error", error.message if error else "检索失败")
    session.emit(events.SEARCH_COMPLETED, {"query": f"arxiv:{query}", "hits": len(result.data)})
    return ok(
        [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "published_at": paper.published_at.isoformat() if paper.published_at else None,
                "abs_url": paper.abs_url,
                "pdf_url": paper.pdf_url,
            }
            for paper in result.data
        ],
        budget=session._budget_left(),
    )


def github_tool(session: Session, repo: str, *, limit: int = 5) -> dict[str, Any]:
    """读 Releases；没有 Release 不等于没有更新。"""
    try:
        session.guard.take_search()
    except BudgetExceeded as exc:
        return fail("budget_exceeded", f"检索次数已用完（上限 {exc.limit}）。", dimension=exc.dimension)
    result = get_releases(
        repo,
        limit=limit,
        token=session.settings.github_token,
        api_base=session.settings.github_api_base,
        timeout=session.settings.http_timeout_seconds,
    )
    _log(session.db, session.run_id, "github_releases", ok=result.ok, request=repo)
    if not result.ok:
        error = result.error
        return fail(
            error.code.value if error else "upstream_error",
            error.message if error else "读取失败",
            retryable=bool(error and error.retryable),
        )
    session.emit(events.SEARCH_COMPLETED, {"query": f"github:{repo}", "hits": len(result.data)})
    return ok(
        [
            {
                "tag": item.tag_name,
                "name": item.name,
                "published_at": item.published_at.isoformat() if item.published_at else None,
                "html_url": item.html_url,
                "body": item.body,
                "prerelease": item.prerelease,
            }
            for item in result.data
        ],
        budget=session._budget_left(),
    )


__all__ = [
    "Session",
    "arxiv_tool",
    "attach",
    "coverage",
    "fail",
    "github_tool",
    "ok",
    "publish",
    "read",
    "search",
    "start",
    "status",
    "submit_finding",
]