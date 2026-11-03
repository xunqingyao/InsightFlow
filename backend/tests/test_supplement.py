"""覆盖缺口与补搜：缺什么、为什么补、补到哪一步停，都要能复核。"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from insightflow.config.settings import Settings
from insightflow.research.agents.model import OfflineResearchModel
from insightflow.research.budget import Budget
from insightflow.research.coverage import find_gaps, required_themes
from insightflow.research.pipeline import ResearchEngine
from insightflow.research.schemas import (
    RunMode,
    RunRecord,
    RunStatus,
    Source,
    SourceKind,
    SourceRole,
    SourceStatus,
    stable_id,
)
from insightflow.research.time_window import normalize
from insightflow.storage.db import Database

REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"
QUERY = "最近一周 AI Agent 评测动态"


def _engine(tmp_path: Path, *, budget: Budget | None = None) -> tuple[Database, ResearchEngine]:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(REPO_FIXTURES, fixtures, dirs_exist_ok=True)
    settings = Settings(data_dir=tmp_path / "data", fixtures_root=fixtures, tavily_api_key=None)
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()
    engine = ResearchEngine(settings, db, model=OfflineResearchModel(), budget=budget)
    return db, engine


def _seed(db: Database, fixture_id: str, *, query: str = QUERY) -> str:
    request = normalize(
        query,
        as_of=datetime(2026, 10, 27, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        window=timedelta(days=7),
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


def _types(db: Database, run_id: str) -> list[str]:
    return [event["type"] for event in db.list_events(run_id, limit=200)]


# ---------------------------------------------------------------- unit: gaps


def _source(title: str) -> Source:
    return Source(
        source_id=stable_id("src", title),
        url="https://example.invalid/x",
        canonical_url="https://example.invalid/x",
        title=title,
        retrieved_at=datetime(2026, 10, 25, tzinfo=ZoneInfo("UTC")),
        status=SourceStatus.FETCHED,
        kind=SourceKind.OFFICIAL,
        role=SourceRole.OFFICIAL,
    )


def test_theme_gap_and_thin_coverage(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", fixtures_root=tmp_path)
    request = normalize(QUERY, mode=RunMode.REPLAY, fixture_id="x")
    assert required_themes(request) == ["agents"]

    gaps = find_gaps(request, [_source("示例：推理模型 M2 发布")])
    kinds = {gap.kind for gap in gaps}
    assert "theme_missing" in kinds
    assert "thin_coverage" in kinds
    theme_gap = next(gap for gap in gaps if gap.kind == "theme_missing")
    assert theme_gap.theme == "agents"
    assert theme_gap.queries

    covered = find_gaps(request, [_source("示例：智能体框架 A"), _source("示例：智能体框架 B")])
    assert covered == []


def test_no_readable_source_is_its_own_gap(tmp_path: Path) -> None:
    request = normalize(QUERY, mode=RunMode.REPLAY, fixture_id="x")
    gaps = find_gaps(request, [])
    assert [gap.kind for gap in gaps] == ["no_sources"]
    assert gaps[0].queries == [request.query]


def test_explicit_focus_wins_over_inference(tmp_path: Path) -> None:
    request = normalize(
        "最近一周的进展", focus=["papers"], mode=RunMode.REPLAY, fixture_id="x"
    )
    assert required_themes(request) == ["papers"]


# ------------------------------------------------------------- integration


def test_gap_triggers_one_supplement_round(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path)
    run_id = _seed(db, "supplement-2026-10-27")
    result = engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    assert result.status is RunStatus.COMPLETED
    assert report.supplement_rounds == 1
    assert report.status == "published", "补搜补齐主题后应能发布"
    titles = [source.title for source in db.list_sources(run_id)]
    assert any("智能体" in title for title in titles), "补搜应带回智能体主题的来源"

    types = _types(db, run_id)
    assert "gaps_found" in types
    assert "supplement_started" in types
    assert "supplement_completed" in types
    assert any(item.startswith("覆盖情况") for item in report.coverage) or any(
        "补搜轮次" in item for item in report.coverage
    )
    assert "## 覆盖情况" in report.markdown
    assert "补搜 1 轮" in report.scope_text


def test_supplement_stops_when_nothing_new(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path)
    run_id = _seed(db, "thin-2026-10-27")
    result = engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    assert result.status is RunStatus.PARTIAL
    assert result.stop_reason == "supplement_no_new_sources"
    assert report.status == "partial"
    assert any("未解决缺口" in line for line in report.coverage)
    stopped = [
        event
        for event in db.list_events(run_id, limit=200)
        if event["type"] == "supplement_stopped"
    ]
    assert stopped and stopped[0]["data"]["reason"] == "no_new_sources"


def test_supplement_limit_is_recorded(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path, budget=Budget(max_supplement_rounds=0))
    run_id = _seed(db, "supplement-2026-10-27")
    result = engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    assert result.stop_reason == "supplement_limit"
    assert report.status == "partial"
    assert report.supplement_rounds == 0
    assert any("补搜已达上限" in line for line in report.limitations)


def test_complete_coverage_skips_supplement(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path)
    run_id = _seed(db, "briefing-2026-10-27")
    engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    assert report.supplement_rounds == 0
    assert report.status == "published"
    assert "supplement_started" not in _types(db, run_id)