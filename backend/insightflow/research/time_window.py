"""请求归一化：`as_of`、时间窗口、主题与任务类型。

约定（设计文档 §6.2 / §7.2）：
- 请求进入即固定 `as_of`，窗口是左闭右开区间 `[window_start, window_end)`，`window_end == as_of`；
- 内部统一 UTC 存储，展示用请求时区；
- 无法判断时使用**可见的默认值**，不反复追问普通偏好。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from insightflow.research.schemas import ResearchRequest, RunMode, TaskType

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_WINDOW = timedelta(days=7)  # 7×24 小时

_WINDOW_PATTERNS: tuple[tuple[re.Pattern[str], timedelta], ...] = (
    (re.compile(r"最近\s*(?:一|1)\s*个?\s*月|近\s*(?:一|1)\s*个?\s*月|过去\s*(?:一|1)\s*个?\s*月"), timedelta(days=30)),
    (re.compile(r"最近\s*(?:两|二|2)\s*个?\s*周"), timedelta(days=14)),
    (re.compile(r"最近\s*(?:一|1)\s*个?\s*周|近\s*(?:一|1)\s*个?\s*周|过去\s*(?:一|1)\s*个?\s*周|本周"), timedelta(days=7)),
    (re.compile(r"最近\s*(?:三|3)\s*天|近\s*(?:三|3)\s*天"), timedelta(days=3)),
    (re.compile(r"最近\s*24\s*小时|近\s*24\s*小时|今天"), timedelta(hours=24)),
    (re.compile(r"最近\s*(?:一|1)\s*年|过去\s*(?:一|1)\s*年"), timedelta(days=365)),
)

_TASK_PATTERNS: tuple[tuple[TaskType, re.Pattern[str]], ...] = (
    (TaskType.DATA_ANALYSIS, re.compile(r"数据|统计|人口|指标|图表|csv|CSV|趋势分析")),
    (TaskType.COMPARISON, re.compile(r"对比|比较|横向|选型|哪个更好")),
    (TaskType.TOPIC_RESEARCH, re.compile(r"专题|综述|调研|梳理|研究进展|值得阅读")),
    (TaskType.BRIEFING, re.compile(r"动态|简报|最近|新闻|更新|发布")),
)

# 主题词表：既用于从请求里识别重点，也用于把重点映射成检索词
THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "models": ("模型", "大模型", "llm", "gpt", "基础模型"),
    "agents": ("agent", "智能体", "代理", "工具调用"),
    "open_source": ("开源", "open source", "github", "仓库"),
    "papers": ("论文", "paper", "arxiv", "研究"),
}


def now_in(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz)).replace(microsecond=0)


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("时间必须带时区")
    return value.astimezone(UTC)


def normalize(
    query: str,
    *,
    as_of: datetime | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    window: timedelta | None = None,
    task_type: TaskType | None = None,
    focus: list[str] | None = None,
    language: str = "zh-CN",
    output_format: str = "markdown",
    budget_profile: str = "standard",
    mode: RunMode = RunMode.ONLINE,
    fixture_id: str | None = None,
) -> ResearchRequest:
    """把用户输入归一化为可执行请求。

    显式传入的参数优先；其余按请求文本做启发式判断并写入 `focus`，便于界面展示"本次研究范围"。
    """
    tz = timezone
    if as_of is None:
        as_of = now_in(tz)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=ZoneInfo(tz))
    as_of = as_of.astimezone(ZoneInfo(tz)).replace(microsecond=0)

    if window is None:
        window = infer_window(query)
    if window <= timedelta(0):
        raise ValueError("时间窗口必须为正")

    return ResearchRequest(
        query=query,
        task_type=task_type or infer_task_type(query),
        as_of=as_of,
        timezone=tz,
        window_start=as_of - window,
        window_end=as_of,
        focus=focus if focus is not None else infer_focus(query),
        language=language,
        output_format=output_format,
        budget_profile=budget_profile,
        mode=mode,
        fixture_id=fixture_id,
    )


def infer_window(query: str) -> timedelta:
    for pattern, delta in _WINDOW_PATTERNS:
        if pattern.search(query):
            return delta
    return DEFAULT_WINDOW


def infer_task_type(query: str) -> TaskType:
    for task_type, pattern in _TASK_PATTERNS:
        if pattern.search(query):
            return task_type
    # 无法判断时采用可见默认值，而不是反复澄清
    return TaskType.BRIEFING


def infer_focus(query: str) -> list[str]:
    lowered = query.lower()
    return [theme for theme, words in THEME_KEYWORDS.items() if any(w in lowered for w in words)]


def in_window(
    value: datetime | None,
    request_or_start: ResearchRequest | datetime,
    window_end: datetime | None = None,
) -> bool:
    """左闭右开。时间未知一律不判入窗口，交由调用方标注。

    接受 `(value, request)` 或 `(value, start, end)` 两种调用方式，避免上层为了
    取两个时间点而构造请求对象。
    """
    if value is None:
        return False
    if value.tzinfo is None:
        raise ValueError("待判断时间必须带时区")
    if isinstance(request_or_start, ResearchRequest):
        start, end = request_or_start.window_start, request_or_start.window_end
    else:
        if window_end is None:
            raise ValueError("缺少 window_end")
        start, end = request_or_start, window_end
    return start <= value < end


def describe_window(
    request_or_start: ResearchRequest | datetime,
    window_end: datetime | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> str:
    if isinstance(request_or_start, ResearchRequest):
        start, end, tz_name = (
            request_or_start.window_start,
            request_or_start.window_end,
            request_or_start.timezone,
        )
    else:
        if window_end is None:
            raise ValueError("缺少 window_end")
        start, end, tz_name = request_or_start, window_end, timezone
    tz = ZoneInfo(tz_name)
    start_local = start.astimezone(tz)
    end_local = end.astimezone(tz)
    days = (end - start) / timedelta(days=1)
    span = f"{days:g} 天" if days >= 1 else f"{days * 24:g} 小时"
    return (
        f"{start_local:%Y-%m-%d %H:%M} → {end_local:%Y-%m-%d %H:%M}（{tz_name}，"
        f"左闭右开，跨度 {span}）"
    )


def window_bounds_utc(request: ResearchRequest) -> tuple[datetime, datetime]:
    return to_utc(request.window_start), to_utc(request.window_end)