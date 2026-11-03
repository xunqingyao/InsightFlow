"""重启续跑与跨进程预算计数。

约束：预算用量跟着 run 走，不跟着进程走；续跑时已经抓到的正文不再重复抓。
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from insightflow.api.app import create_app
from insightflow.config.settings import Settings
from insightflow.research import session as S
from insightflow.research.budget import Budget
from insightflow.research.pipeline import ResearchEngine
from insightflow.research.schemas import (
    ReadDepth,
    RunCounters,
    RunMode,
    RunRecord,
    RunStatus,
    Source,
    SourceKind,
    SourceStatus,
    FetchScope,
    stable_id,
)
from insightflow.research.time_window import normalize
from insightflow.storage.db import Database
from insightflow.storage.files import SnapshotStore

REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"
AS_OF = datetime(2026, 10, 27, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
FIXTURE_ID = "briefing-2026-10-27"


def _settings(tmp_path: Path) -> Settings:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(REPO_FIXTURES, fixtures, dirs_exist_ok=True)
    settings = Settings(data_dir=tmp_path / "data", fixtures_root=fixtures, tavily_api_key=None)
    settings.ensure_dirs()
    return settings


def _open(tmp_path: Path):
    settings = _settings(tmp_path)
    session, started = S.start(
        settings,
        "最近一周 AI Agent 评测动态",
        as_of=AS_OF,
        days=7,
        mode=RunMode.REPLAY,
        fixture_id=FIXTURE_ID,
    )
    assert started["ok"], started
    assert session is not None
    return settings, session


def test_usage_survives_reattach(tmp_path: Path) -> None:
    settings, session = _open(tmp_path)
    assert S.search(session, "AI Agent 评测")["ok"]
    assert S.search(session, "Agent benchmark")["ok"]
    hits = S.search(session, "评测框架")["data"]
    assert S.read(session, hits[0]["url"])["ok"]

    db = Database(settings.db_path)
    db.init_schema()
    counters = db.get_run(session.run_id).counters
    assert counters.search_calls == 3
    assert counters.source_fetches == 1

    # 相当于工具进程被杀掉、换了一个进程重新挂上同一个 run
    resumed, payload = S.attach(settings, session.run_id)
    assert resumed is not None and payload["ok"]
    assert resumed.guard.search_calls == 3
    assert resumed.guard.source_fetches == 1
    assert resumed.guard.remaining_searches() == 9

    assert S.search(resumed, "AI Agent 评测")["ok"]
    again = Database(settings.db_path).get_run(session.run_id).counters
    assert again.search_calls == 4  # 继续累加，不是从零开始


def test_resume_marks_run_running_and_counts_resumes(tmp_path: Path) -> None:
    settings, session = _open(tmp_path)
    db = Database(settings.db_path)
    db.init_schema()
    db.update_run(session.run_id, status=RunStatus.RECOVERABLE, stop_reason="process_restart")

    resumed, payload = S.resume(settings, session.run_id)
    assert resumed is not None and payload["ok"], payload
    record = db.get_run(session.run_id)
    assert record.status is RunStatus.RUNNING
    assert record.counters.resumes == 1
    assert payload["data"]["budget"]["searches_left"] == 12

    # 工具模式下 running 也能重挂（MCP 每次调用都是新进程）
    db.update_run(session.run_id, status=RunStatus.RUNNING)
    running, payload2 = S.resume(settings, session.run_id)
    assert running is not None and payload2["ok"], payload2
    assert db.get_run(session.run_id).status is RunStatus.RUNNING
    assert db.get_run(session.run_id).counters.resumes == 2

    # 已经结束的运行不能再续
    db.update_run(session.run_id, status=RunStatus.COMPLETED)
    _, failed = S.resume(settings, session.run_id)
    assert failed["ok"] is False
    assert failed["error"]["code"] == "invalid_state"


def test_pipeline_resume_reuses_snapshots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    db.init_schema()
    request = normalize(
        "最近一周 AI Agent 评测动态",
        as_of=AS_OF,
        window=timedelta(days=7),
        mode=RunMode.REPLAY,
        fixture_id=FIXTURE_ID,
        timezone=settings.timezone,
    )
    run_id = stable_id("run", request.query, request.as_of.isoformat())
    now = datetime.now(UTC)
    db.create_run(
        RunRecord(
            run_id=run_id,
            status=RunStatus.RECOVERABLE,
            request=request,
            created_at=now,
            updated_at=now,
            counters=RunCounters(search_calls=2, source_fetches=1, resumes=1),
        )
    )

    # 上次已经读到的正文：模拟"进程在抓完之后挂掉"
    source = Source(
        source_id=stable_id("src", run_id, "https://example.invalid/lab/eval-suite-x"),
        kind=SourceKind.MEDIA,
        title="已抓到的正文",
        url="https://example.invalid/lab/eval-suite-x",
        canonical_url="https://example.invalid/lab/eval-suite-x",
        status=SourceStatus.FETCHED,
        published_at=AS_OF - timedelta(days=2),
        retrieved_at=now,
        fetch_scope=FetchScope.FULL,
        read_depth=ReadDepth.FULL,
    )
    store = SnapshotStore(settings.snapshots_dir)
    path = store.save_text(run_id, source.source_id, "某实验室发布评测框架 X。" * 20)
    source = source.model_copy(update={"snapshot_path": path})
    db.replace_sources(run_id, [source])

    engine = ResearchEngine(settings, db, budget=Budget())
    result = engine.run(run_id)

    assert result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL}
    counters = db.get_run(run_id).counters
    assert counters.search_calls > 2  # 从 2 接着累加，不是重置
    # 已读过的那个 URL 不再抓；另外两个候选照常抓
    assert counters.source_fetches == 3
    assert counters.resumes == 1  # 续跑次数不会被收尾覆盖
    fetched_events = [
        item["data"] for item in db.list_events(run_id, after_seq=0, limit=500)
        if item["type"] == "source_fetched"
    ]
    assert [e["url"] for e in fetched_events if e.get("reused")] == [
        "https://example.invalid/lab/eval-suite-x"
    ]
    assert db.latest_report(run_id) is not None
    urls = [item.canonical_url for item in db.list_sources(run_id)]
    assert urls.count("https://example.invalid/lab/eval-suite-x") == 1


def test_api_recovers_interrupted_runs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    db.init_schema()
    request = normalize(
        "最近一周 AI Agent 评测动态",
        as_of=AS_OF,
        window=timedelta(days=7),
        mode=RunMode.REPLAY,
        fixture_id=FIXTURE_ID,
        timezone=settings.timezone,
    )
    run_id = stable_id("run", request.query, request.as_of.isoformat())
    now = datetime.now(UTC)
    db.create_run(
        RunRecord(run_id=run_id, status=RunStatus.RUNNING, request=request, created_at=now, updated_at=now)
    )

    app = create_app(settings)
    with TestClient(app) as client:
        record = app.state.db.get_run(run_id)
        assert record.status is RunStatus.RECOVERABLE
        assert record.stop_reason == "process_restart"

        response = client.post(f"/api/research/runs/{run_id}/resume")
        assert response.status_code == 202, response.text
        assert response.json()["metadata"]["resumes"] == 1

        deadline = time.monotonic() + 20
        status = None
        while time.monotonic() < deadline:
            status = client.get(f"/api/research/runs/{run_id}").json()["data"]["run"]["status"]
            if status in {"completed", "partial", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status in {"completed", "partial"}, status

        events = client.get(f"/api/research/runs/{run_id}/events").json()["data"]
        types = [item["type"] for item in events]
        assert "run_recoverable" in types
        assert types.count("run_started") >= 1
        assert json.dumps(events)  # 事件必须是可序列化的

def test_status_reads_report_from_database(tmp_path: Path) -> None:
    settings, session = _open(tmp_path)
    hits = S.search(session, "AI Agent 评测")["data"]
    read = S.read(session, hits[0]["url"])
    assert read["ok"], read
    text = read["data"]["text"]
    quote = next(line.strip() for line in text.splitlines() if len(line.strip()) >= 20)[:60]
    submitted = S.submit_finding(
        session,
        text=f"发布方给出了一个可引用的说法：{quote[:20]}",
        evidence=[{"source_id": read["data"]["source_id"], "quote": quote}],
    )
    assert submitted["ok"], submitted
    published = S.publish(session, summary="一次最小发布。")
    assert published["ok"], published

    # 换一个进程挂上来：报告状态必须来自数据库，而不是内存里的 session.report
    resumed, payload = S.attach(settings, session.run_id)
    assert resumed is not None and payload["ok"]
    state = S.status(resumed)["data"]
    assert state["status"] in {"published", "partial"}
    assert state["report_id"]
    assert state["findings"] >= 1
