"""网页入口：静态页、回放清单、报告 HTML 渲染。

界面本身不做业务判断，只把接口给的东西摆出来；所以测试覆盖三件事：
页面能拿到、清单能显示、报告能渲染成 HTML（且用户输入不会被当成标签执行）。
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from insightflow.api.app import create_app
from insightflow.api.render import render_markdown
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
        payload = client.get(f"/api/research/runs/{run_id}").json()["data"]
        if payload["run"]["status"] in TERMINAL:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"运行未在 {timeout}s 内结束：{payload}")


def test_index_page_is_served(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        body = response.text
        assert "InsightFlow" in body
        assert "开始研究" in body
        # 页面引用的脚本与样式必须真的能取到，否则线上就是白屏。
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200


def test_fixture_listing(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        body = client.get("/api/fixtures").json()
        assert body["ok"] is True
        ids = {item["id"] for item in body["data"]}
        assert FIXTURE_ID in ids
        entry = next(item for item in body["data"] if item["id"] == FIXTURE_ID)
        assert entry["items"] == 3
        assert entry["synthetic"] is True


def test_report_has_html(tmp_path: Path) -> None:
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
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        _wait(client, run_id)

        payload = client.get(f"/api/research/runs/{run_id}/report").json()["data"]
        assert payload["markdown"].startswith("# ")
        html = payload["html"]
        assert "<h1>" in html
        assert "<table>" in html  # 质量校验表
        assert "<script>" not in html


def test_markdown_escapes_user_text() -> None:
    rendered = render_markdown("# 标题\n\n<script>alert(1)</script>\n\n| a | b |\n| - | - |\n| 1 | 2 |\n")
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<table>" in rendered