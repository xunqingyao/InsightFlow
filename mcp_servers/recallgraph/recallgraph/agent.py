"""最小 Agent 循环：模型自己决定调哪个工具、调几次、什么时候停。

这里没有编排好的工作流——只有工具和一份说明书。检索、多跳、读原文的顺序全由模型决定，
我们只负责：把工具调用的结果原样递回去、把每一步记下来（--trace 可看）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import psycopg

from recallgraph.env import load_dotenv
from recallgraph.retriever import SqlRetriever
from recallgraph.tools import dispatch, tool_specs

CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen-plus"
MAX_TOOL_RESULT_CHARS = 12000

SYSTEM_PROMPT = """你是 RecallGraph，一个建立在个人知识库上的研究助手。

知识库里是论文和笔记，组织成图：断言（claim）连接实体，每条断言都带原文引文和页码。
你的工具：
- search_materials：检索，返回断言和段落预览。这是索引，不是原文。
- query_knowledge_graph：从实体出发多跳，看关系网。
- read_evidence：读原文全文。引用前用它核对。

工作方式（你自己判断，不必机械照做）：
1. 先搞清楚“有没有、在哪里”，需要细节再回来读原文。
2. 引用必须来自工具返回的内容，写出文档标题和页码；没找到就说“知识库里没有”。
3. 明确区分“论文里写的”和“我的推断”，推断要标注。
4. 证据够了就停，不要重复调用同一个工具凑步骤。
5. 如果用户的说法/记忆与知识库证据冲突，不要顺着说：以证据为准，指出"你记的是 X，但《标题》第 N 页写的是 Y"。
6. 用户说"我记得""我之前说过"时，先调 recall_memories 看他当时的说法，再拿证据对照。
7. 不要仅凭 claims 为空就说"知识库里没有"：看 coverage.uncovered_chunks，必要时读原文。
8. 只用工具返回的内容作答：邮箱、数字、日期、作者名这类事实，工具没给就说"知识库里没有"，绝不能用你自己知道的去补。
"""


def chat(payload: dict[str, Any], *, api_key: str, timeout: float = 120.0, retries: int = 2) -> dict[str, Any]:
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"模型请求失败：{last}")


def run(
    conn: psycopg.Connection,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = 8,
    trace: bool = False,
    retriever: SqlRetriever | None = None,
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[dict[str, Any]] = []

    for _ in range(max_steps):
        body = chat(
            {"model": model, "messages": messages, "tools": tool_specs(), "tool_choice": "auto", "temperature": 0.2},
            api_key=api_key,
        )
        message = body["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return {"answer": message.get("content", ""), "steps": steps, "messages": messages}

        for call in calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            try:
                result, note = dispatch(conn, name, arguments, retriever=retriever)
            except (psycopg.Error, ValueError, RuntimeError) as exc:  # 工具失败要告诉模型，而不是让它卡住
                # 数据库报错会把事务留在 aborted 状态，不 rollback 的话后面所有工具都会连环失败。
                if conn is not None and not conn.closed:
                    conn.rollback()
                result, note = {"error": str(exc)}, {"tool": name, "error": str(exc)}
            steps.append({"tool": name, "arguments": arguments, **note})
            if trace:
                print(f"  · {name}({json.dumps(arguments, ensure_ascii=False)}) -> {json.dumps(note, ensure_ascii=False)}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS],
                }
            )

    # 步数用完还没收敛：摘掉工具再问一次，让它基于已有证据作答或明确说证据不足。
    messages.append(
        {
            "role": "user",
            "content": "（评测提示）工具调用次数已到上限。请直接基于上面已经拿到的证据作答；证据不足的部分明确说明缺什么，不要再要求调用工具。",
        }
    )
    body = chat({"model": model, "messages": messages, "temperature": 0.2}, api_key=api_key)
    return {
        "answer": body["choices"][0]["message"].get("content", ""),
        "steps": steps,
        "messages": messages,
        "max_steps_reached": True,
    }