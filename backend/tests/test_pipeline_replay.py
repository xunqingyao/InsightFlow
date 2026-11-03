from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

from insightflow.config.settings import Settings
from insightflow.research.agents.model import (
    LLMError,
    OfflineResearchModel,
    Synthesis,
    SynthesisEvidence,
    SynthesisFinding,
)
from insightflow.research.pipeline import ResearchEngine
from insightflow.research.schemas import (
    CheckStatus,
    ClaimType,
    RunMode,
    RunRecord,
    RunStatus,
    SourceRole,
    SourceStatus,
    SupportStatus,
    stable_id,
)
from insightflow.research.time_window import normalize
from insightflow.storage.db import Database

FIXTURE_ID = "briefing-2026-10-27"
REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"


def _engine(tmp_path: Path, **overrides) -> tuple[Database, ResearchEngine, Settings]:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(REPO_FIXTURES, fixtures, dirs_exist_ok=True)
    settings = Settings(
        data_dir=tmp_path / "data", fixtures_root=fixtures, tavily_api_key=None, **overrides
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()
    engine = ResearchEngine(settings, db, model=OfflineResearchModel())
    return db, engine, settings


def _seed(
    db: Database,
    *,
    fixture_id: str = FIXTURE_ID,
    window: timedelta | None = None,
    as_of: datetime | None = None,
) -> str:
    request = normalize(
        "最近一周 AI Agent 评测动态",
        as_of=as_of,
        window=window or timedelta(days=7),
        mode=RunMode.REPLAY,
        fixture_id=fixture_id,
    )
    run_id = stable_id("run", request.query, request.as_of.isoformat(), fixture_id)
    db.create_run(
        RunRecord(
            run_id=run_id,
            status=RunStatus.QUEUED,
            request=request,
            created_at=request.as_of,
            updated_at=request.as_of,
        )
    )
    return run_id


def test_replay_run_produces_cited_report(tmp_path: Path) -> None:
    db, engine, _ = _engine(tmp_path)
    run_id = _seed(db)
    result = engine.run(run_id)

    assert result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL}
    report = db.latest_report(run_id)
    assert report is not None
    assert report.findings, "回放链路应至少产出一条带证据的结论"

    evidence = {item.evidence_id: item for item in db.list_evidence(run_id)}
    sources = {source.source_id: source for source in db.list_sources(run_id)}
    for finding in report.findings:
        assert finding.evidence_ids
        for evidence_id in finding.evidence_ids:
            assert evidence_id in evidence
            assert evidence[evidence_id].source_id in sources
    assert report.markdown.startswith("# ")
    assert "质量校验" in report.markdown


def test_quotes_exist_verbatim_in_snapshot(tmp_path: Path) -> None:
    db, engine, settings = _engine(tmp_path)
    run_id = _seed(db)
    engine.run(run_id)

    sources = {source.source_id: source for source in db.list_sources(run_id)}
    for item in db.list_evidence(run_id):
        source = sources[item.source_id]
        assert source.snapshot_path is not None
        text = (settings.snapshots_dir / source.snapshot_path).read_text(encoding="utf-8")
        assert item.quote.strip() in text, "证据引文必须逐字出现在来源快照中"


def test_sources_outside_window_are_flagged(tmp_path: Path) -> None:
    db, engine, _ = _engine(tmp_path)
    run_id = _seed(db, window=timedelta(hours=1))
    engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    statuses = {check.check: check.status for check in report.checks}
    assert statuses["time_window"].value == "warn"
    assert report.status == "partial"
    assert "窗口外来源" in report.scope_text
    assert any("不构成本窗口内的新增动态" in item for item in report.limitations)


def test_missing_pages_are_recorded_as_failed_sources(tmp_path: Path) -> None:
    db, engine, settings = _engine(tmp_path)
    manifest = json.loads((settings.fixtures_dir / FIXTURE_ID / "manifest.json").read_text("utf-8"))
    manifest["fixture_id"] = "broken-fixture"
    for item in manifest["items"]:
        item["page"] = "pages/nope.html"
    broken = settings.fixtures_dir / "broken-fixture"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    run_id = _seed(db, fixture_id="broken-fixture")
    engine.run(run_id)

    sources = db.list_sources(run_id)
    assert sources and all(source.status is SourceStatus.FAILED for source in sources)
    report = db.latest_report(run_id)
    assert report is not None
    assert report.findings == []
    assert report.status == "partial"
    assert any("未成功抓取" in item for item in report.limitations)


def test_duplicate_run_id_not_recreated(tmp_path: Path) -> None:
    db, engine, _ = _engine(tmp_path)
    run_id = _seed(db)
    engine.run(run_id)
    first = db.get_run(run_id)
    assert first is not None
    engine.run(run_id)  # 已完成，直接返回，不应被重置
    second = db.get_run(run_id)
    assert second is not None
    assert second.status == first.status


def test_same_source_can_appear_in_two_runs(tmp_path: Path) -> None:
    """同一 URL 出现在两次运行里时，来源 ID 必须按运行隔离，否则第二次写入会撞主键。"""
    db, engine, _ = _engine(tmp_path)
    tz = ZoneInfo("Asia/Shanghai")
    first = _seed(db, as_of=datetime(2026, 10, 27, 12, tzinfo=tz))
    second = _seed(db, as_of=datetime(2026, 10, 27, 18, tzinfo=tz))
    assert first != second

    engine.run(first)
    engine.run(second)

    urls_a = {source.canonical_url for source in db.list_sources(first)}
    urls_b = {source.canonical_url for source in db.list_sources(second)}
    assert urls_a == urls_b and urls_a
    ids_a = {source.source_id for source in db.list_sources(first)}
    ids_b = {source.source_id for source in db.list_sources(second)}
    assert ids_a.isdisjoint(ids_b)


def test_missing_search_key_is_reported_not_hidden(tmp_path: Path) -> None:
    """没有搜索凭据时不能写成“没有检索到来源”，要说清是检索失败。"""
    db, engine, _ = _engine(tmp_path)
    request = normalize("最近一周 AI Agent 评测动态", window=timedelta(days=7))
    run_id = stable_id("run", "no-search-key", request.as_of.isoformat())
    db.create_run(
        RunRecord(
            run_id=run_id,
            status=RunStatus.QUEUED,
            request=request,
            created_at=request.as_of,
            updated_at=request.as_of,
        )
    )
    result = engine.run(run_id)

    assert result.status is RunStatus.PARTIAL
    assert result.stop_reason == "search_failed"
    report = db.latest_report(run_id)
    assert report is not None
    assert report.findings == []
    assert any("检索未成功" in item for item in report.limitations)


class _BrokenModel:
    """模拟模型服务不可用：规划与综合都失败，应回退模板并留下痕迹。"""

    name = "broken"
    provider = "test"

    @property
    def usage(self):
        return []

    def plan(self, request):
        raise LLMError("模型不可用")

    def synthesize(self, request, materials):
        raise LLMError("模型不可用")


def test_model_failure_falls_back_to_template(tmp_path: Path) -> None:
    db, _, _ = _engine(tmp_path)
    engine = ResearchEngine(
        Settings(data_dir=tmp_path / "data", fixtures_root=tmp_path / "fixtures"),
        db,
        model=_BrokenModel(),
    )
    run_id = _seed(db)
    result = engine.run(run_id)

    assert result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL}
    assert result.stop_reason == "model_fallback"
    report = db.latest_report(run_id)
    assert report is not None
    assert report.findings, "回退模板仍应产出可核查的摘录结论"
    assert any("模板" in item for item in report.limitations)


def test_replay_preserves_source_role(tmp_path: Path) -> None:
    """角色丢失会让“转载不算独立验证”变成摆设，这条守住它。"""
    db, engine, _ = _engine(tmp_path)
    run_id = _seed(db)
    engine.run(run_id)

    by_url = {source.url: source for source in db.list_sources(run_id)}
    assert by_url["https://example.invalid/lab/eval-suite-x"].role is SourceRole.OFFICIAL
    assert by_url["https://example.invalid/media/report-eval-suite-x"].role is SourceRole.REPRINT
    assert by_url["https://example.invalid/preprint/lhb"].role is SourceRole.ORIGINAL


class _ReprintModel:
    """把转载当成唯一强支撑：应被 reprint_independence 拦下，并降级为部分完成。"""

    @property
    def usage(self):
        return []

    def plan(self, request):
        return ["评测"]

    def synthesize(self, request, materials):
        reprint = next(item for item in materials if item.role == "reprint")
        quote = reprint.text.split("。")[0].strip() + "。"
        return Synthesis(
            summary="某媒体称评测框架已发布。",
            findings=[
                SynthesisFinding(
                    text="某实验室已发布 Agent 评测框架 X。",
                    claim_type=ClaimType.OBSERVABLE_FACT,
                    support_status=SupportStatus.SUPPORTED,
                    evidence=[SynthesisEvidence(source_id=reprint.source_id, quote=quote)],
                )
            ],
        )


def test_reprint_cannot_count_as_independent_support(tmp_path: Path) -> None:
    db, _, _ = _engine(tmp_path)
    engine = ResearchEngine(
        Settings(data_dir=tmp_path / "data", fixtures_root=tmp_path / "fixtures"),
        db,
        model=_ReprintModel(),
    )
    run_id = _seed(db)
    result = engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    checks = {check.check: check.status for check in report.checks}
    assert checks["reprint_independence"] is CheckStatus.WARN
    assert result.status is RunStatus.PARTIAL
