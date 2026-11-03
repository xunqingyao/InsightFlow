from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from insightflow.api.app import create_app
from insightflow.config.settings import Settings

FIXTURE_ID = "briefing-2026-10-27"
REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"

TERMINAL = {"completed", "partial", "failed", "cancelled"}


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(data_dir=tmp_path / "data", fixtures_root=REPO_FIXTURES)
    return TestClient(create_app(settings))


def _wait(client: TestClient, run_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/research/runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()["data"]
        if payload["run"]["status"] in TERMINAL:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"运行未在 {timeout}s 内结束：{payload}")


def test_health(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        body = client.get("/api/health").json()
        assert body["ok"] is True
        assert body["data"]["runs"] == 0


def test_replay_run_via_api(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = client.post(
            "/api/research/runs",
            json={
                "query": "最近一周 AI Agent 评测动态",
                "days": 7,
                "mode": "replay",
                "fixture_id": FIXTURE_ID,
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]

        detail = _wait(client, run_id)
        assert detail["run"]["status"] in {"completed", "partial"}
        assert len(detail["sources"]) == 3
        assert all(item["status"] == "fetched" for item in detail["sources"])

        report = client.get(f"/api/research/runs/{run_id}/report")
        assert report.status_code == 200
        data = report.json()["data"]
        assert data["report"]["findings"]
        assert data["markdown"].startswith("# ")

        source_id = detail["sources"][0]["source_id"]
        source = client.get(f"/api/sources/{source_id}")
        assert source.status_code == 200
        assert source.json()["data"]["source"]["source_id"] == source_id


def test_duplicate_run_conflicts(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        payload = {
            "query": "最近一周 AI Agent 评测动态",
            "days": 7,
            "mode": "replay",
            "fixture_id": FIXTURE_ID,
        }
        assert client.post("/api/research/runs", json=payload).status_code == 202
        assert client.post("/api/research/runs", json=payload).status_code == 409


def test_invalid_request_rejected(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/research/runs", json={"query": "x", "mode": "replay", "fixture_id": None}
        )
        assert response.status_code == 422

        missing = client.post(
            "/api/research/runs", json={"query": "测试一下", "mode": "replay"}
        )
        assert missing.status_code == 422


def test_unknown_report_is_404(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/reports/rep_missing").status_code == 404
        assert client.get("/api/research/runs/run_missing").status_code == 404
