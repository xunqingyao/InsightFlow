"""抽取层：把段落变成（实体, 断言, 证据）。

对外只暴露两样东西：`get_extractor(name)` 和 `extract_version(...)`。
上层（CLI、Agent 工具、评测）不需要知道背后是规则还是模型。
"""

from __future__ import annotations

from recallgraph.extract.llm import LlmClaimExtractor
from recallgraph.extract.stub import StubExtractor

_EXTRACTORS = {
    "stub": StubExtractor,
    "llm": LlmClaimExtractor,
}


def get_extractor(name: str = "stub"):
    key = (name or "stub").strip().lower()
    if key not in _EXTRACTORS:
        raise KeyError(f"未知抽取器：{name}（可选：{'/'.join(_EXTRACTORS)}）")
    return _EXTRACTORS[key]()


__all__ = ["LlmClaimExtractor", "StubExtractor", "get_extractor"]