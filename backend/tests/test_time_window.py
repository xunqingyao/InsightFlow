from __future__ import annotations

from datetime import UTC, datetime, timedelta

from insightflow.research.schemas import ResearchRequest, RunMode, TaskType
from insightflow.research.time_window import in_window, infer_window, normalize


def _request(start: datetime, end: datetime) -> ResearchRequest:
    return ResearchRequest(
        query="测试",
        task_type=TaskType.BRIEFING,
        as_of=end,
        timezone="Asia/Shanghai",
        window_start=start,
        window_end=end,
    )


def test_window_is_left_closed_right_open() -> None:
    start = datetime(2026, 10, 20, tzinfo=UTC)
    end = datetime(2026, 10, 27, tzinfo=UTC)
    request = _request(start, end)
    assert in_window(start, request) is True
    assert in_window(end - timedelta(microseconds=1), request) is True
    assert in_window(end, request) is False
    assert in_window(None, request) is False


def test_normalize_pins_as_of_and_window_end() -> None:
    as_of = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    request = normalize("最近一周的大模型动态", as_of=as_of)
    assert request.window_end == request.as_of
    assert request.window_end - request.window_start == timedelta(days=7)
    assert request.as_of.utcoffset() is not None


def test_infer_window_variants() -> None:
    assert infer_window("最近一个月") == timedelta(days=30)
    assert infer_window("最近三天") == timedelta(days=3)
    assert infer_window("最近 24 小时") == timedelta(hours=24)
    assert infer_window("随便看看") == timedelta(days=7)


def test_replay_requires_fixture() -> None:
    import pytest

    with pytest.raises(ValueError):
        normalize("测试", mode=RunMode.REPLAY)
