"""覆盖与证据缺口检查（设计文档 §5.1 的“检查覆盖与证据缺口”一步）。

补搜不能凭空发起：先要说清“缺什么、为什么缺、打算查什么”，否则就是无边界地
多搜几轮。这里的产出直接写进事件和报告，补搜轮次与停止原因也可复核。

只检查能从已有数据判断的缺口，不假装能判断“内容是否说清楚了”：
- 用户关心的主题有没有来源（按来源标题判断，标题没提就等于没覆盖，偏保守）；
- 可读来源是否过少；
- 一个可读来源都没有。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from insightflow.research.schemas import ResearchRequest, Source, SourceStatus
from insightflow.research.time_window import infer_focus

# 每个主题的一组关键词：标题里出现任意一个就算覆盖。
THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "models": ("模型", "model", "大模型"),
    "agents": ("智能体", "agent", "代理", "工作流"),
    "open_source": ("开源", "open source", "github"),
    "papers": ("论文", "paper", "arxiv", "预印本"),
}

THEME_LABELS: dict[str, str] = {
    "models": "模型",
    "agents": "智能体/Agent",
    "open_source": "开源项目",
    "papers": "论文",
}

MIN_READABLE_SOURCES = 2


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["theme_missing", "no_sources", "thin_coverage"]
    theme: str | None = None
    reason: str
    queries: list[str] = Field(default_factory=list)

    def as_event(self) -> dict[str, str | None]:
        """事件里只放判断依据，不放正文。"""
        return {
            "kind": self.kind,
            "theme": self.theme,
            "label": THEME_LABELS.get(self.theme or "", self.theme or ""),
            "reason": self.reason,
        }


def _mentions_theme(source: Source, theme: str) -> bool:
    keywords = THEME_KEYWORDS.get(theme, ())
    title = source.title.lower()
    return any(keyword.lower() in title for keyword in keywords)


def required_themes(request: ResearchRequest) -> list[str]:
    """用户明确点到的主题优先；没点名就按问题里出现的主题词推断。"""
    if request.focus:
        return [theme for theme in request.focus if theme in THEME_KEYWORDS]
    return [theme for theme in infer_focus(request.query) if theme in THEME_KEYWORDS]


def find_gaps(
    request: ResearchRequest, sources: list[Source], *, min_sources: int = MIN_READABLE_SOURCES
) -> list[Gap]:
    readable = [source for source in sources if source.status is SourceStatus.FETCHED]
    if not readable:
        return [
            Gap(
                kind="no_sources",
                reason="没有抓取到任何可读来源",
                queries=[request.query],
            )
        ]

    gaps: list[Gap] = []
    for theme in required_themes(request):
        if any(_mentions_theme(source, theme) for source in readable):
            continue
        label = THEME_LABELS.get(theme, theme)
        # 同一主题用两个不同关键词各查一次：中文唤词和英文项目名经常命中的不是同一批页面。
        queries = [f"{request.query} {keyword}" for keyword in THEME_KEYWORDS[theme][:2]]
        gaps.append(
            Gap(
                kind="theme_missing",
                theme=theme,
                reason=f"没有覆盖「{label}」主题的来源",
                queries=queries,
            )
        )

    if len(readable) < min_sources:
        gaps.append(
            Gap(
                kind="thin_coverage",
                reason=f"只有 {len(readable)} 个可读来源，少于下限 {min_sources} 个",
                queries=[request.query],
            )
        )
    return gaps


def coverage_lines(
    request: ResearchRequest, sources: list[Source], gaps: list[Gap], rounds: int
) -> list[str]:
    """写进报告“覆盖情况”一节的短句，和缺口一一对应。"""
    readable = [source for source in sources if source.status is SourceStatus.FETCHED]
    failed = len(sources) - len(readable)
    lines = [f"可读来源 {len(readable)} 个" + (f"，抓取失败 {failed} 个" if failed else "")]
    themes = required_themes(request)
    if themes:
        covered: list[str] = []
        missing: list[str] = []
        for theme in themes:
            label = THEME_LABELS.get(theme, theme)
            if any(_mentions_theme(source, theme) for source in readable):
                covered.append(label)
            else:
                missing.append(label)
        lines.append("主题覆盖：" + ("、".join(covered) if covered else "无"))
        if missing:
            lines.append("主题缺口：" + "、".join(missing))
    if gaps:
        lines.append("未解决缺口：" + "；".join(gap.reason for gap in gaps))
    lines.append(f"补搜轮次：{rounds}")
    return lines


__all__ = [
    "Gap",
    "MIN_READABLE_SOURCES",
    "THEME_KEYWORDS",
    "THEME_LABELS",
    "coverage_lines",
    "find_gaps",
    "required_themes",
]