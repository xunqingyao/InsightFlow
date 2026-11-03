"""用真模型驱动一次研究工具：验证“工具 + skill”能不能被模型正确使用。

这不是产品代码，也不属于服务端——它是一个最小的宿主（host）：接 OpenAI 兼容的
/chat/completions，把 InsightFlow 的 MCP 工具原样交给模型，看它按不按 skill 的顺序走、
在哪里会跑偏。跑完把轨迹和报告打出来，用来判断该改工具描述还是改 skill。

    uv run python evaluation/runners/live_model.py --question "最近一周 AI Agent 评测动态"

需要 `.env` 里的 MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME。默认走 replay 资料，
不联网、不消耗搜索额度；加 `--online` 才走真实检索（需要 TAVILY_API_KEY）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from insightflow.config.settings import Settings  # noqa: E402

SKILL = REPO / "skills" / "research-briefing" / "SKILL.md"
MAX_STEPS = 24
TOOL_RESULT_CHARS = 4000


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = tool.input_schema or {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


class Host:
    def __init__(
        self,
        settings: Settings,
        session: ClientSession,
        tools: list[Any],
        *,
        max_searches: int = 4,
    ) -> None:
        self.settings = settings
        self.session = session
        self.max_searches = max_searches
        self.searches = 0
        self.schemas = [_tool_schema(tool) for tool in tools]
        self.messages: list[dict[str, Any]] = []
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.client = httpx.Client(timeout=httpx.Timeout(180.0, connect=20.0))

    def _post(self, *, temperature: float = 0.2) -> dict[str, Any]:
        payload = {
            "model": self.settings.model_name,
            "messages": self.messages,
            "tools": self.schemas,
            "tool_choice": "auto",
            "temperature": temperature,
        }
        url = self.settings.model_base_url.rstrip("/") + "/chat/completions"
        response = self.client.post(
            url,
            headers={"Authorization": f"Bearer {self.settings.model_api_key}"},
            json=payload,
        )
        if response.status_code >= 400:
            raise SystemExit(f"模型调用失败 HTTP {response.status_code}: {response.text[:400]}")
        body = response.json()
        usage = body.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens") or 0
        self.completion_tokens += usage.get("completion_tokens") or 0
        self.calls += 1
        return body["choices"][0]["message"]

    async def run(self, question: str) -> None:
        skill = SKILL.read_text(encoding="utf-8")
        self.messages = [
            {
                "role": "system",
                "content": (
                    "你是一个研究助理，只能通过给定的工具查资料和写报告。\n"
                    "严格遵守下面的作业说明；不确定时先调工具，不要凭记忆作答。\n\n" + skill
                ),
            },
            {"role": "user", "content": question},
        ]

        for step in range(1, MAX_STEPS + 1):
            message = self._post()
            content = (message.get("content") or "").strip()
            calls = message.get("tool_calls") or []
            self.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls or None,
                }
            )
            if content:
                print(f"[{step:>2}] 模型说：{content.splitlines()[0][:120]}")
            if not calls:
                print(f"[{step:>2}] 没有工具调用，循环结束")
                return

            for call in calls:
                name = call["function"]["name"]
                raw = call["function"].get("arguments") or "{}"
                try:
                    arguments = json.loads(raw)
                except ValueError:
                    arguments = {}
                started = time.monotonic()
                if name == "search_web":
                    # 演示宿主自己加的一道闸：Tavily 免费额度有限，别让模型无限搜。
                    self.searches += 1
                    if self.searches > self.max_searches:
                        payload = {
                            "ok": False,
                            "data": None,
                            "error": {
                                "code": "budget_exceeded",
                                "message": f"演示宿主限制：本次最多 {self.max_searches} 次联网检索。",
                                "retryable": False,
                            },
                            "metadata": {},
                        }
                        print(f"[{step:>2}] -> search_web 被宿主额度拦下")
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id") or "",
                                "content": json.dumps(payload, ensure_ascii=False),
                            }
                        )
                        continue
                result = await self.session.call_tool(name, arguments)
                payload = result.structured_content
                elapsed = time.monotonic() - started
                summary = _summarize(name, payload)
                print(f"[{step:>2}] -> {name}({_brief(arguments)}) {elapsed:0.1f}s  {summary}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or "",
                        "content": json.dumps(payload, ensure_ascii=False)[:TOOL_RESULT_CHARS],
                    }
                )

        print(f"达到最大步数 {MAX_STEPS}，停止")


def _brief(arguments: dict[str, Any]) -> str:
    text = json.dumps(arguments, ensure_ascii=False)
    return text if len(text) <= 90 else text[:87] + "..."


def _summarize(name: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)[:80]
    if not payload.get("ok"):
        error = payload.get("error") or {}
        return f"FAILED {error.get('code')}: {str(error.get('message'))[:80]}"
    data = payload.get("data")
    if isinstance(data, list):
        return f"ok，{len(data)} 条"
    if isinstance(data, dict):
        keys = ["run_id", "report_id", "status", "total_findings", "accepted", "rejected"]
        bits = [f"{k}={data[k]}" for k in keys if k in data and not isinstance(data[k], (list, dict))]
        if "findings" in data and isinstance(data["findings"], int):
            bits.append(f"findings={data['findings']}")
        if "accepted" in data and isinstance(data["accepted"], list):
            bits.append(f"accepted={len(data['accepted'])}")
        if "rejected" in data and isinstance(data["rejected"], list):
            bits.append(f"rejected={len(data['rejected'])}")
        return "ok " + " ".join(bits) if bits else "ok"
    return "ok"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="最近一周 AI Agent 评测动态")
    parser.add_argument("--online", action="store_true", help="走真实检索（需要 TAVILY_API_KEY）")
    parser.add_argument("--fixture", default="briefing-2026-10-27")
    parser.add_argument("--as-of", default="2026-10-27T12:00:00+08:00")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-searches", type=int, default=4)
    args = parser.parse_args()

    settings = Settings()
    if not settings.model_configured:
        print("没有配置模型：在 .env 里设置 MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME", file=sys.stderr)
        return 2
    print(f"模型 {settings.model_name} @ {settings.model_base_url}")

    data_dir = os.environ.get("INSIGHTFLOW_DATA_DIR") or str(
        REPO / "data" / ("live-online" if args.online else "live-model")
    )
    env = {
        **os.environ,
        "INSIGHTFLOW_DATA_DIR": data_dir,
        "MODEL_API_KEY": settings.model_api_key or "",
        "MODEL_NAME": settings.model_name,
        "MODEL_BASE_URL": settings.model_base_url,
    }
    mode = "online" if args.online else "replay"
    params = StdioServerParameters(command=str(REPO / ".venv" / "bin" / "research-tools"), env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"MCP {info.server_info.name}：{len(tools)} 个工具\n")

            host = Host(settings, session, tools, max_searches=args.max_searches)
            question = (
                f"{args.question}（mode=online，as_of={args.as_of}，days={args.days}，"
                f"最多 {args.max_searches} 次检索）"
                if args.online
                else f"{args.question}（用回放资料：mode=replay，fixture_id={args.fixture}，"
                f"as_of={args.as_of}，days={args.days}）"
            )
            started = time.monotonic()
            await host.run(question)
            elapsed = time.monotonic() - started

            print(
                f"\n模型调用 {host.calls} 次，用时 {elapsed:0.1f}s，"
                f"tokens in={host.prompt_tokens} out={host.completion_tokens}"
            )
            last = _last_report(host.messages)
            if last:
                print(f"报告 {last.get('report_id')}｜状态 {last.get('status')}｜结论 {last.get('findings')}")
    return 0


def _last_report(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("ok"):
            data = payload.get("data")
            if isinstance(data, dict) and data.get("report_id"):
                return data
    return None


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))