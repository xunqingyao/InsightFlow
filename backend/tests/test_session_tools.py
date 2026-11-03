"""工具层：宿主 agent 调什么、拿到什么、被拒绝时看到什么。

这里没有任何编排——测试也是"假装自己是一个 agent"，一步一步调工具。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from insightflow.config.settings import Settings
from insightflow.research import session as S
from insightflow.research.budget import Budget
from insightflow.research.schemas import ClaimType, SupportStatus

REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"
AS_OF = datetime(2026, 10, 27, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def _settings(tmp_path: Path) -> Settings:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(REPO_FIXTURES, fixtures, dirs_exist_ok=True)
    settings = Settings(data_dir=tmp_path / "data", fixtures_root=fixtures, tavily_api_key=None)
    settings.ensure_dirs()
    return settings


def _open(tmp_path: Path, *, budget: Budget | None = None):
    settings = _settings(tmp_path)
    session, started = S.start(
        settings,
        "最近一周 AI Agent 评测动态",
        as_of=AS_OF,
        days=7,
        mode=S.RunMode.REPLAY,
        fixture_id="briefing-2026-10-27",
        budget=budget,
    )
    assert started["ok"], started
    assert session is not None
    return settings, session, started


def test_start_returns_window_and_next_step(tmp_path: Path) -> None:
    _, _, started = _open(tmp_path)
    data = started["data"]
    assert data["run_id"].startswith("run_")
    assert data["window_start"] < data["window_end"]
    assert data["budget"]["searches_left"] == 12
    assert "search_web" in data["next"]


def test_search_then_read_returns_verbatim_text(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    found = S.search(session, "AI Agent 评测", max_results=5)
    assert found["ok"], found
    hits = found["data"]
    assert hits and hits[0]["url"].startswith("https://")
    assert hits[0]["already_read"] is False

    read = S.read(session, hits[0]["url"])
    assert read["ok"], read
    payload = read["data"]
    assert payload["source_id"].startswith("src_")
    assert len(payload["text"]) > 100
    assert payload["fetch_scope"] in {"full", "partial"}
    assert "原样复制" in payload["hint"]

    again = S.read(session, hits[0]["url"])
    assert again["ok"] and again.get("metadata", {}).get("cached") is True


def test_bad_quote_is_rejected_with_reason(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    read = S.read(session, S.search(session, "AI Agent")["data"][0]["url"])["data"]

    result = S.submit_finding(
        session,
        "官方发布了新框架。",
        claim_type=ClaimType.PUBLISHER_CLAIM.value,
        support_status=SupportStatus.PARTIAL.value,
        evidence=[{"source_id": read["source_id"], "quote": "这句话是我编的，原文里没有"}],
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "quote_not_verbatim" or result["error"]["code"] == "evidence_mismatch"
    detail = result["metadata"]["details"][0]
    assert detail["problem"] in {"quote_not_verbatim", "quote_too_short"}
    assert len(session.findings) == 0, "被拒绝的结论不能进报告"


def test_evidence_from_unread_source_is_rejected(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    result = S.submit_finding(
        session,
        "仓库发布了新版本。",
        evidence=[{"source_id": "src_fake", "quote": "一段看起来很像原文的引用文字"}],
    )
    assert result["ok"] is False
    assert result["metadata"]["details"][0]["problem"] == "source_not_in_run"


def test_happy_path_publishes_report_with_checks(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    urls = [hit["url"] for hit in S.search(session, "AI Agent 评测")["data"][:2]]
    for url in urls:
        assert S.read(session, url)["ok"]

    for source in session.sources.values():
        text = session.texts[source.source_id]
        quote = next(
            line.strip()
            for line in text.splitlines()
            if 30 <= len(line.strip()) <= 160 and "虚构示例" not in line
        )
        submitted = S.submit_finding(
            session,
            f"{source.title}：这是一条可核对的陈述。",
            evidence=[{"source_id": source.source_id, "quote": quote, "locator": "正文"}],
        )
        assert submitted["ok"], submitted
        assert submitted["data"]["evidence_ids"]

    coverage = S.coverage(session)
    assert coverage["ok"] and coverage["data"]["readable_sources"] == 2

    published = S.publish(session, "本窗口内有两个来源，记录其说法，未做独立验证。")
    assert published["ok"], published
    payload = published["data"]
    assert payload["status"] in {"published", "partial"}
    assert payload["findings"] == 2
    assert "#" in payload["markdown"]
    assert payload["checks"], "报告必须带质量校验结果"

    state = S.status(session)
    assert state["data"]["can_publish"] is True
    assert len(state["data"]["sources"]) == 2


def test_publish_without_findings_is_refused(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    result = S.publish(session, "空报告")
    assert result["ok"] is False
    assert result["error"]["code"] == "no_findings"


def test_budget_is_enforced_by_tools(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path, budget=Budget(max_search_calls=1))
    assert S.search(session, "AI Agent")["ok"]
    second = S.search(session, "AI Agent 评测")
    assert second["ok"] is False
    assert second["error"]["code"] == "budget_exceeded"


def test_attach_restores_sources_and_evidence(tmp_path: Path) -> None:
    settings, session, _ = _open(tmp_path)
    read = S.read(session, S.search(session, "AI Agent")["data"][0]["url"])["data"]
    text = session.texts[read["source_id"]]
    quote = next(line.strip() for line in text.splitlines() if 30 <= len(line.strip()) <= 160)
    assert S.submit_finding(
        session, "一条结论。", evidence=[{"source_id": read["source_id"], "quote": quote}]
    )["ok"]

    reopened, attached = S.attach(settings, session.run_id)
    assert attached["ok"]
    assert reopened is not None
    assert read["source_id"] in reopened.sources
    assert reopened.texts[read["source_id"]]
    assert len(reopened.evidence.all()) == 1
    assert reopened.evidence.get(reopened.evidence.all()[0].evidence_id) is not None

def _read_first(session, query: str = "AI Agent") -> dict:
    hit = S.search(session, query)["data"][0]
    return S.read(session, hit["url"])["data"]


def _quote(session, source_id: str) -> str:
    text = session.texts[source_id]
    return next(line.strip() for line in text.splitlines() if 30 <= len(line.strip()) <= 160)


def test_batch_submit_keeps_good_and_returns_bad(tmp_path: Path) -> None:
    """一条引文写错，不该让另一条白交。"""
    _, session, _ = _open(tmp_path)
    page = _read_first(session)
    quote = _quote(session, page["source_id"])

    result = S.submit_findings(
        session,
        [
            {"text": "能对上的结论。", "evidence": [{"source_id": page["source_id"], "quote": quote}]},
            {"text": "对不上的结论。", "evidence": [{"source_id": page["source_id"], "quote": "原文里没有这句话"}]},
        ],
    )
    assert result["ok"] is True
    assert [item["index"] for item in result["data"]["accepted"]] == [1]
    assert [item["index"] for item in result["data"]["rejected"]] == [2]
    assert result["data"]["rejected"][0]["code"] == "evidence_mismatch"
    assert result["metadata"]["total_findings"] == 1


def test_batch_submit_all_bad_is_refused(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    page = _read_first(session)
    result = S.submit_findings(
        session,
        [{"text": "只有摘要没有正文。", "evidence": [{"source_id": page["source_id"], "quote": "无"}]}],
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "evidence_mismatch"
    assert session.findings == []


def test_compare_sources_marks_shared_and_unique_figures(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    hits = S.search(session, "AI Agent 评测", max_results=5)["data"]
    pages = [S.read(session, hit["url"])["data"] for hit in hits[:2]]

    result = S.compare_sources(session, [page["source_id"] for page in pages])
    assert result["ok"], result
    data = result["data"]
    assert len(data["sources"]) == 2
    assert "shared_values" in data and "only_in_one" in data
    assert "不是事实核查" in result["metadata"]["caveat"]
    for source in data["sources"]:
        for figure in source["figures"]:
            assert figure["context"], "每个数字都要带上下文，否则无法核对"


def test_compare_sources_needs_two_readable(tmp_path: Path) -> None:
    _, session, _ = _open(tmp_path)
    page = _read_first(session)
    result = S.compare_sources(session, [page["source_id"]])
    assert result["ok"] is False
    assert result["error"]["code"] == "insufficient_sources"

def test_publish_counts_cited_sources_before_validation(tmp_path: Path) -> None:
    """回归：先校验、后填 cited_source_ids，会让任何真实报告都被判 fail。"""
    _, session, _ = _open(tmp_path)
    urls = [hit["url"] for hit in S.search(session, "AI Agent 评测")["data"]]
    for url in urls:
        assert S.read(session, url)["ok"]
    assert len(urls) == 3

    for source in session.sources.values():
        text = session.texts[source.source_id]
        quote = next(
            line.strip()
            for line in text.splitlines()
            if 30 <= len(line.strip()) <= 160 and "虚构示例" not in line
        )
        assert S.submit_finding(
            session,
            f"{source.title} 的说法需要核对。",
            evidence=[{"source_id": source.source_id, "quote": quote, "locator": "正文"}],
        )["ok"]

    published = S.publish(session, "三个来源各记一条说法。")
    assert published["ok"], published
    payload = published["data"]
    checks = {item["check"]: item for item in payload["checks"]}
    assert checks["cited_sources"]["status"] == "pass", checks["cited_sources"]
    assert "3" in checks["cited_sources"]["detail"]
    assert payload["status"] == "published", payload["status"]


def test_report_lists_each_source_once(tmp_path: Path) -> None:
    """回归：同一来源被引用多次时，来源清单曾经重复列出同一篇文章。"""
    _, session, _ = _open(tmp_path)
    urls = [hit["url"] for hit in S.search(session, "AI Agent 评测")["data"]]
    for url in urls:
        assert S.read(session, url)["ok"]

    official = next(iter(session.sources.values()))
    text = session.texts[official.source_id]
    lines = [
        line.strip()
        for line in text.splitlines()
        if 30 <= len(line.strip()) <= 160 and "虚构示例" not in line
    ]
    for quote in lines[:3]:
        assert S.submit_finding(
            session,
            f"同源多条：{quote[:12]}",
            evidence=[{"source_id": official.source_id, "quote": quote, "locator": "正文"}],
        )["ok"]

    published = S.publish(session, "同一来源被引用三次。")
    assert published["ok"], published
    markdown = published["data"]["markdown"]
    head, _, _ = markdown.partition("**引文**")
    assert head.count(official.url) == 1, "来源清单应去重"
    assert "**来源**（1 个）" in markdown, "只引用一个来源时，来源清单只应有一个"


def test_github_tool_serializes_real_releases(tmp_path: Path, monkeypatch) -> None:
    """回归：字段名写错（tag vs tag_name）时，只有“有 Release”的仓库才会崩。"""
    from datetime import UTC

    from insightflow.tools.contracts import ToolResult
    from insightflow.tools.github import GithubRelease

    _, session, _ = _open(tmp_path)
    release = GithubRelease(
        repo="o/r",
        tag_name="v1.2.3",
        name="1.2.3",
        prerelease=False,
        draft=False,
        published_at=datetime(2026, 10, 25, tzinfo=UTC),
        body="notes",
        html_url="https://github.com/o/r/releases/tag/v1.2.3",
    )
    monkeypatch.setattr(
        "insightflow.research.session.get_releases",
        lambda *a, **k: ToolResult[list[GithubRelease]](ok=True, data=[release]),
    )
    result = S.github_tool(session, "o/r", limit=1)
    assert result["ok"], result
    assert result["data"][0]["tag"] == "v1.2.3"
    assert result["data"][0]["published_at"].startswith("2026-10-25")


def test_findings_survive_reattach(tmp_path: Path) -> None:
    """MCP 每次调用都可能换一个进程：结论落库后，新进程也要能接着 publish。"""
    settings, session, _ = _open(tmp_path)
    page = _read_first(session)
    quote = _quote(session, page["source_id"])
    assert S.submit_finding(
        session, "一条结论。", evidence=[{"source_id": page["source_id"], "quote": quote}]
    )["ok"]

    reopened, attached = S.attach(settings, session.run_id)
    assert attached["ok"] and attached["data"]["findings"] == 1
    assert reopened is not None
    published = S.publish(reopened, summary="窗口内只有一条可核对的新动态。")
    assert published["ok"], published
    assert published["data"]["status"] in {"published", "partial"}
    assert published["data"]["findings"] == 1
