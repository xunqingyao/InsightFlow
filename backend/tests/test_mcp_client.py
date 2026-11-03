"""MCP 客户端契约一致性。

引擎内部直接调用 `insightflow.tools`（同步、错误类型精确、好测试），MCP 服务是给
外部 agent 的入口。两者必须有同一份信封，否则“同一工具内部成功、外部看不懂”。

这个测试起真实 stdio 子进程握手，验证：工具名可发现、失败返回 `ok:false` 信封而不是
协议错误、字段形状与进程内调用一致。全部离线（不给 key），不依赖网络。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from insightflow.tools.contracts import ToolErrorCode
from insightflow.tools.search import search_web

ENVELOPE_KEYS = {"ok", "data", "error", "metadata"}


def _server_params() -> StdioServerParameters:
    scratch = tempfile.mkdtemp(prefix="insightflow-mcp-")
    env = {
        **os.environ,
        "TAVILY_API_KEY": "",
        "GITHUB_TOKEN": "",
        "INSIGHTFLOW_DATA_DIR": scratch,
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "research_tools.server"],
        env=env,
    )


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    if payload is None:
        payload = getattr(result, "structuredContent", None)
    assert payload is not None, f"工具没有返回结构化内容：{result}"
    return payload


def test_mcp_client_sees_same_envelope_as_in_process() -> None:
    async def scenario() -> None:
        async with stdio_client(_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                info = await session.initialize()
                assert info.server_info.name == "research-tools"

                listing = await session.list_tools()
                names = {tool.name for tool in listing.tools}
                assert {
                    "start_research",
                    "search_web",
                    "read_source",
                    "submit_finding",
                    "publish_report",
                } <= names
                assert all(tool.description for tool in listing.tools)

                started = _structured(
                    await session.call_tool(
                        "start_research",
                        {
                            "question": "最近一周 AI Agent 评测动态",
                            "as_of": "2026-10-27T12:00:00+08:00",
                            "mode": "replay",
                            "fixture_id": "briefing-2026-10-27",
                        },
                    )
                )
                assert started["ok"] is True
                run_id = started["data"]["run_id"]

                remote = _structured(
                    await session.call_tool("search_web", {"run_id": run_id, "query": "AI Agent 评测"})
                )
                assert set(remote) == ENVELOPE_KEYS
                assert remote["ok"] is True
                assert remote["data"][0]["url"]

                missing = _structured(
                    await session.call_tool("run_status", {"run_id": "run_missing"})
                )
                assert missing["ok"] is False
                assert missing["error"]["code"] == "not_found"

                online = _structured(
                    await session.call_tool(
                        "start_research",
                        {"question": "在线检索没有 key", "as_of": "2026-10-27T12:00:00+08:00"},
                    )
                )
                assert online["ok"] is True
                unauth = _structured(
                    await session.call_tool(
                        "search_web", {"run_id": online["data"]["run_id"], "query": "AI Agent"}
                    )
                )
                assert set(unauth) == ENVELOPE_KEYS
                assert unauth["ok"] is False
                assert unauth["error"]["code"] == ToolErrorCode.UNAUTHORIZED.value

                local = search_web("AI Agent 评测", api_key=None).model_dump(mode="json")
                assert local["error"]["retryable"] == unauth["error"]["retryable"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))