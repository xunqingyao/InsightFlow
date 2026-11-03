"""事件流、取消与幂等。

这三件事的共同前提是：运行过程中发生的事情必须可被外部看到、可被打断、可被重放。
测试都走真实 HTTP 与真实后台线程，不 mock 引擎。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from insightflow.api.app import create_app
from insightflow.config.settings import Settings
from insightflow.research.pipeline import ResearchEngine

FIXTURE_ID = "briefing-2026-10-27"
REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"
TERMINAL = {"completed", "partial", "failed", "cancelled"}


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", fixtures_root=REPO_FIXTURES)


def _create_replay_run(client: TestClient) -> str:
    response = client.post(
        "/api/research/runs",
        json={
            "query": "最近一周 AI Agent 评测动态",
            "days": 7,
            "mode": "replay",
            "fixture_id": FIXTURE_ID,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/api/research/runs/{run_id}").json()["data"]
        if payload["run"]["status"] in TERMINAL:
            return payload
        time.sleep(0.05)
    raise AssertionError("运行未在超时内结束")


def _read_sse(
    client: TestClient, run_id: str, *, last_event_id: int | None = None
) -> list[dict[str, Any]]:
    headers = {"accept": "text/event-stream"}
    if last_event_id is not None:
        headers["Last-Event-ID"] = str(last_event_id)
    events: list[dict[str, Any]] = []
    frame: dict[str, str] = {}
    with client.stream(
        "GET", f"/api/research/runs/{run_id}/events", headers=headers
    ) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line:
                if frame:
                    events.append(json.loads(frame["data"]))
                    frame = {}
                continue
            if line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            frame[key] = value
    return events


def test_events_are_monotonic_and_cover_each_stage(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        run_id = _create_replay_run(client)
        _wait_terminal(client, run_id)

        body = client.get(f"/api/research/runs/{run_id}/events").json()
        events = body["data"]
        types = [event["type"] for event in events]

        assert body["ok"] is True
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        for expected in (
            "run_queued",
            "run_started",
            "plan_ready",
            "search_completed",
            "source_fetched",
            "synthesis_ready",
            "report_ready",
            "run_finished",
        ):
            assert expected in types, (expected, types)
        assert types[-1] == "run_finished"
        assert body["metadata"]["max_seq"] == len(events)

        fetched = [event for event in events if event["type"] == "source_fetched"]
        assert len(fetched) == 3
        # 事件里只放 ID 与元数据，不放正文：正文在快照里，事件表不该变成第二份存储。
        assert set(fetched[0]["data"]) >= {"source_id", "url", "title", "read_depth"}
        assert "text" not in fetched[0]["data"]


def test_events_can_be_replayed_from_last_event_id(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        run_id = _create_replay_run(client)
        _wait_terminal(client, run_id)
        full = client.get(f"/api/research/runs/{run_id}/events").json()["data"]
        assert len(full) > 4

        # JSON 快照的游标语义
        tail = client.get(
            f"/api/research/runs/{run_id}/events", params={"after": full[2]["seq"]}
        ).json()
        assert [event["seq"] for event in tail["data"]] == [
            event["seq"] for event in full[3:]
        ]

        # SSE 重连：只补发断点之后的事件，且以终态事件收尾
        resumed = _read_sse(client, run_id, last_event_id=full[2]["seq"])
        assert [event["seq"] for event in resumed] == [event["seq"] for event in full[3:]]
        assert resumed[-1]["type"] == "run_finished"


def test_sse_stream_delivers_full_history_when_run_already_finished(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        run_id = _create_replay_run(client)
        _wait_terminal(client, run_id)

        events = _read_sse(client, run_id)
        assert events[0]["type"] == "run_queued"
        assert events[-1]["type"] == "run_finished"
        seqs = [event["seq"] for event in events]
        assert seqs == sorted(set(seqs))


class _BlockingModel:
    """卡在 plan 阶段，给测试留出“取消已经发生”的时间窗。"""

    provider = "test"
    name = "blocking-model"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.gate = threading.Event()
        self.synthesized = False

    def plan(self, request: Any) -> list[str]:
        self.started.set()
        self.gate.wait(timeout=10)
        return ["AI Agent 评测"]

    def synthesize(self, request: Any, materials: list[Any]) -> Any:
        self.synthesized = True
        raise AssertionError("取消后不应进入合成阶段")


def test_cancel_stops_run_and_is_not_overwritten(tmp_path: Path) -> None:
    model = _BlockingModel()
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        app.state.engine = ResearchEngine(
            app.state.settings, app.state.db, model=model
        )
        created = client.post(
            "/api/research/runs",
            json={"query": "最近一周 AI Agent 评测动态", "days": 7},
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        assert model.started.wait(timeout=10), "引擎没有进入 plan 阶段"

        cancelled = client.post(f"/api/research/runs/{run_id}/cancel")
        assert cancelled.status_code == 202, cancelled.text
        assert cancelled.json()["data"]["status"] == "cancelled"

        model.gate.set()
        time.sleep(0.4)  # 给引擎线程足够时间走到下一个阶段边界

        detail = client.get(f"/api/research/runs/{run_id}").json()["data"]
        assert detail["run"]["status"] == "cancelled"
        assert detail["run"]["report_id"] is None
        assert model.synthesized is False
        assert app.state.db.latest_report(run_id) is None

        events = client.get(f"/api/research/runs/{run_id}/events").json()["data"]
        assert events[-1]["type"] == "run_cancelled"
        assert "run_finished" not in [event["type"] for event in events]

        again = client.post(f"/api/research/runs/{run_id}/cancel")
        assert again.status_code == 409


def test_cancelling_finished_run_conflicts(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        run_id = _create_replay_run(client)
        _wait_terminal(client, run_id)
        response = client.post(f"/api/research/runs/{run_id}/cancel")
        assert response.status_code == 409


def test_idempotency_key_returns_same_run(tmp_path: Path) -> None:
    body = {
        "query": "最近一周 AI Agent 评测动态",
        "days": 7,
        "mode": "replay",
        "fixture_id": FIXTURE_ID,
    }
    with TestClient(create_app(_settings(tmp_path))) as client:
        first = client.post("/api/research/runs", json=body, headers={"Idempotency-Key": "k-1"})
        second = client.post("/api/research/runs", json=body, headers={"Idempotency-Key": "k-1"})
        assert first.status_code == 202 and second.status_code == 202
        assert first.json()["run_id"] == second.json()["run_id"]

        runs = client.get("/api/research/runs").json()
        assert runs["metadata"]["count"] == 1

        conflict = client.post(
            "/api/research/runs",
            json={**body, "days": 1},
            headers={"Idempotency-Key": "k-1"},
        )
        assert conflict.status_code == 409
        assert "Idempotency-Key" in conflict.json()["detail"]