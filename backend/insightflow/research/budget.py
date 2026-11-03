"""预算控制（设计文档 §8）。

预算是硬上限：调用方在每次花钱前问 `allow_*()`，引擎在触发上限时把运行标记为
`partial` 并写清是哪一项耗尽，而不是悄悄降级。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    def __init__(self, dimension: str, limit: int | float) -> None:
        super().__init__(f"预算耗尽：{dimension} 已达上限 {limit}")
        self.dimension = dimension
        self.limit = limit


@dataclass(frozen=True)
class Budget:
    max_search_calls: int = 12
    max_source_fetches: int = 24
    max_supplement_rounds: int = 2
    max_review_revisions: int = 1
    max_llm_calls: int = 30
    max_active_walltime_seconds: float = 300.0


class BudgetGuard:
    """预算计数器。

    计数器可以带着已有用量恢复（`restore`），并在每次消耗后回调 `on_change`，
    让上层把用量写进数据库——这样进程重启、换一个进程接手同一个 run 时，
    预算不会从零重算。
    """

    def __init__(
        self,
        budget: Budget | None = None,
        *,
        used: dict[str, int] | None = None,
        on_change: Callable[[dict[str, int]], None] | None = None,
    ) -> None:
        self.budget = budget or Budget()
        used = used or {}
        self.search_calls = int(used.get("search_calls", 0))
        self.source_fetches = int(used.get("source_fetches", 0))
        self.llm_calls = int(used.get("llm_calls", 0))
        self.supplement_rounds = int(used.get("supplement_rounds", 0))
        self.review_revisions = int(used.get("review_revisions", 0))
        self.active_seconds_before = float(used.get("active_seconds", 0))
        self.resumes = int(used.get("resumes", 0))
        self.on_change = on_change
        self._started = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return self.active_seconds_before + (time.monotonic() - self._started)

    def usage(self) -> dict[str, int]:
        return {
            "search_calls": self.search_calls,
            "source_fetches": self.source_fetches,
            "llm_calls": self.llm_calls,
            "supplement_rounds": self.supplement_rounds,
            "review_revisions": self.review_revisions,
            "active_seconds": int(self.elapsed_seconds),
            "resumes": self.resumes,
        }

    def _persist(self) -> None:
        if self.on_change is not None:
            self.on_change(self.usage())

    def _check_walltime(self) -> None:
        if self.elapsed_seconds > self.budget.max_active_walltime_seconds:
            raise BudgetExceeded("walltime", self.budget.max_active_walltime_seconds)

    def take_search(self) -> None:
        self._check_walltime()
        if self.search_calls >= self.budget.max_search_calls:
            raise BudgetExceeded("search_calls", self.budget.max_search_calls)
        self.search_calls += 1
        self._persist()

    def take_fetch(self) -> None:
        self._check_walltime()
        if self.source_fetches >= self.budget.max_source_fetches:
            raise BudgetExceeded("source_fetches", self.budget.max_source_fetches)
        self.source_fetches += 1
        self._persist()

    def take_llm(self) -> None:
        self._check_walltime()
        if self.llm_calls >= self.budget.max_llm_calls:
            raise BudgetExceeded("llm_calls", self.budget.max_llm_calls)
        self.llm_calls += 1
        self._persist()

    def take_supplement(self) -> None:
        if self.supplement_rounds >= self.budget.max_supplement_rounds:
            raise BudgetExceeded("supplement_rounds", self.budget.max_supplement_rounds)
        self.supplement_rounds += 1
        self._persist()

    def take_review(self) -> None:
        """审校/退回次数也算预算（`max_review_revisions`）。"""
        if self.review_revisions >= self.budget.max_review_revisions:
            raise BudgetExceeded("review_revisions", self.budget.max_review_revisions)
        self.review_revisions += 1
        self._persist()

    def remaining_searches(self) -> int:
        return max(0, self.budget.max_search_calls - self.search_calls)

    def remaining_fetches(self) -> int:
        return max(0, self.budget.max_source_fetches - self.source_fetches)

    def remaining_llm_calls(self) -> int:
        return max(0, self.budget.max_llm_calls - self.llm_calls)