"""运行事件的类型与语义约定。

事件是“过程”的唯一真相：状态机（runs.status）只描述此刻，事件流描述怎么走到这里。
因此每个事件的 `data` 里都要带足够重建前端的字段，但**不放正文**——正文在快照里，
事件只给 ID 与计数，避免事件表变成第二份存储、也避免泄漏抓取内容。

`seq` 由数据库在同一 run 内单调递增分配，客户端用 `Last-Event-ID` 断点续传时只认
`seq > last`，重复投递不会造成重复处理。
"""

from __future__ import annotations

RUN_QUEUED = "run_queued"
RUN_STARTED = "run_started"
PLAN_READY = "plan_ready"
SEARCH_COMPLETED = "search_completed"
SEARCH_FAILED = "search_failed"
SOURCE_FETCHED = "source_fetched"
SOURCE_FAILED = "source_failed"
DECISION = "decision"
AGENT_STEP = "agent_step"
TOOL_CALLED = "tool_called"
TOOL_FAILED = "tool_failed"
REPORT_REJECTED = "report_rejected"
GAPS_FOUND = "gaps_found"
SUPPLEMENT_STARTED = "supplement_started"
SUPPLEMENT_COMPLETED = "supplement_completed"
SUPPLEMENT_STOPPED = "supplement_stopped"
SYNTHESIS_READY = "synthesis_ready"
REPORT_READY = "report_ready"
RUN_FINISHED = "run_finished"
RUN_CANCELLED = "run_cancelled"
RUN_FAILED = "run_failed"

TERMINAL_EVENTS = frozenset({RUN_FINISHED, RUN_CANCELLED, RUN_FAILED})

__all__ = [
    "AGENT_STEP",
    "DECISION",
    "GAPS_FOUND",
    "REPORT_REJECTED",
    "PLAN_READY",
    "REPORT_READY",
    "RUN_CANCELLED",
    "RUN_FAILED",
    "RUN_FINISHED",
    "RUN_QUEUED",
    "RUN_STARTED",
    "TOOL_CALLED",
    "TOOL_FAILED",
    "SEARCH_COMPLETED",
    "SEARCH_FAILED",
    "SOURCE_FAILED",
    "SOURCE_FETCHED",
    "SUPPLEMENT_COMPLETED",
    "SUPPLEMENT_STARTED",
    "SUPPLEMENT_STOPPED",
    "SYNTHESIS_READY",
    "TERMINAL_EVENTS",
]