"""补搜由模型决定：代码只守预算、轮次与证据，不替模型做“够不够”的判断。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import respx

from insightflow.config.settings import Settings
from insightflow.research.agents.model import (
    Decision,
    LLMError,
    OfflineResearchModel,
    OpenAIResearchModel,
    ResearchState,
)
from insightflow.research.budget import Budget
from insightflow.research.pipeline import ResearchEngine
from insightflow.research.schemas import RunMode, RunRecord, RunStatus, stable_id
from insightflow.research.time_window import normalize
from insightflow.storage.db import Database

REPO_FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures"
QUERY = "最近一周 AI Agent 评测动态"


def _engine(
    tmp_path: Path, *, model=None, budget: Budget | None = None
) -> tuple[Database, ResearchEngine]:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(REPO_FIXTURES, fixtures, dirs_exist_ok=True)
    settings = Settings(data_dir=tmp_path / "data", fixtures_root=fixtures, tavily_api_key=None)
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()
    engine = ResearchEngine(
        settings, db, model=model or OfflineResearchModel(), budget=budget
    )
    return db, engine


def _seed(db: Database, fixture_id: str) -> str:
    request = normalize(
        QUERY,
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


def _events(db: Database, run_id: str, type_: str) -> list[dict]:
    return [
        event
        for event in db.list_events(run_id, limit=200)
        if event["type"] == type_
    ]


class _DecliningModel(OfflineResearchModel):
    """有缺口也说“够了”：用来验证停止确实是模型的决定，而不是缺口一票否决。"""

    def decide(self, state: ResearchState) -> Decision:
        return Decision(action="stop", reason="现有两个来源足以回答问题")


class _BrokenDecisionModel(OfflineResearchModel):
    def decide(self, state: ResearchState) -> Decision:
        raise LLMError("decision endpoint 502")


def test_decision_event_comes_from_model(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path)
    run_id = _seed(db, "supplement-2026-10-27")
    engine.run(run_id)

    decisions = _events(db, run_id, "decision")
    assert decisions, "补搜前必须有一条模型决策事件"
    first = decisions[0]["data"]
    assert first["action"] == "supplement"
    assert first["decided_by"] == "model"
    assert first["queries"], "模型要求补搜时必须给出检索式"


def test_model_can_decline_despite_gaps(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path, model=_DecliningModel())
    run_id = _seed(db, "thin-2026-10-27")
    result = engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    assert result.stop_reason == "model_declined"
    assert report.supplement_rounds == 0
    assert not _events(db, run_id, "supplement_started"), "模型说停就不该继续搜"
    assert report.status == "partial"
    assert any("模型判断不再补搜" in line for line in report.limitations)
    assert _events(db, run_id, "decision")[0]["data"]["action"] == "stop"


def test_decision_failure_falls_back_to_rules(tmp_path: Path) -> None:
    db, engine = _engine(tmp_path, model=_BrokenDecisionModel())
    run_id = _seed(db, "supplement-2026-10-27")
    engine.run(run_id)

    report = db.latest_report(run_id)
    assert report is not None
    decision = _events(db, run_id, "decision")[0]["data"]
    assert decision["decided_by"] == "fallback"
    assert decision["action"] == "supplement"
    assert report.supplement_rounds == 1
    assert any("模型决策失败" in line for line in report.limitations)


@respx.mock
def test_openai_decision_is_parsed() -> None:
    route = respx.post("https://llm.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "supplement",
                                    "reason": "没有覆盖智能体主题",
                                    "queries": ["智能体框架 发布", "智能体框架 发布"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )
    )
    model = OpenAIResearchModel(
        base_url="https://llm.example/v1", api_key="k", model="m"
    )
    state = ResearchState(
        request=normalize(
            QUERY,
            as_of=datetime(2026, 10, 27, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            window=timedelta(days=7),
            mode=RunMode.REPLAY,
            fixture_id="supplement-2026-10-27",
        ),
        rounds_used=0,
        rounds_left=2,
        searches_left=9,
    )
    decision = model.decide(state)

    assert route.called
    assert decision.action == "supplement"
    assert decision.queries == ["智能体框架 发布"], "重复检索式应去重"
    assert model.usage is not None and model.usage.total_tokens == 120


@respx.mock
def test_openai_decision_rejects_illegal_output() -> None:
    respx.post("https://llm.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"action": "supplement"}'}}]},
        )
    )
    model = OpenAIResearchModel(base_url="https://llm.example/v1", api_key="k", model="m")
    state = ResearchState(
        request=normalize(QUERY, mode=RunMode.REPLAY, fixture_id="x"),
        rounds_left=1,
        searches_left=5,
    )
    try:
        model.decide(state)
    except LLMError as exc:
        assert "没有给出检索式" in str(exc)
    else:  # pragma: no cover - 明确失败比含糊通过好
        raise AssertionError("非法决策应抛 LLMError，交给编排层退回规则决策")