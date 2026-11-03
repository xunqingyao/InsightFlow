from __future__ import annotations

import asyncio

from research_tools.server import server

EXPECTED = {
    "start_research",
    "search_web",
    "read_source",
    "arxiv_search",
    "github_releases",
    "submit_finding",
    "submit_findings",
    "compare_sources",
    "coverage_check",
    "publish_report",
    "run_status",
}


def test_tools_are_discoverable() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert EXPECTED <= set(tools)
    for tool in tools.values():
        assert tool.description, f"{tool.name} 缺少描述，模型无法判断何时调用"


def test_failures_come_back_as_envelope() -> None:
    """工具失败必须走信封（ok=false），不能抛异常给 MCP 客户端。"""
    result = asyncio.run(server.call_tool("run_status", {"run_id": "run_missing"}))
    payload = result.structured_content
    assert payload is not None
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == "not_found"
