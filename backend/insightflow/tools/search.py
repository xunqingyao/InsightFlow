"""搜索适配器：首版接 Tavily（设计文档 §6.1）。

搜索只负责"发现候选来源"。时间参数只作为检索提示，是否属于本期事件
仍由应用按 `event_at` 判断，不能把搜索引擎的日期当成事件时间。
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field

from insightflow.tools.contracts import ToolErrorCode, ToolResult

DEFAULT_ENDPOINT = "https://api.tavily.com/search"


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str = ""
    score: float | None = None
    published_at: datetime | None = None
    provider: str = "tavily"


class SearchBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    provider: str
    searched_at: datetime
    raw_count: int = 0


def _parse_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def search_web(
    query: str,
    *,
    api_key: str | None,
    max_results: int = 5,
    days: int | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 20.0,
    client: httpx.Client | None = None,
) -> ToolResult[SearchBundle]:
    if not api_key:
        return ToolResult.failure(
            ToolErrorCode.UNAUTHORIZED,
            "未配置 TAVILY_API_KEY，无法执行在线搜索",
            retryable=False,
            provider="tavily",
        )
    payload: dict[str, object] = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(max_results, 20)),
        "search_depth": "basic",
        "include_answer": False,
        "topic": "general",
    }
    if days:
        payload["days"] = days

    own_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.post(endpoint, json=payload)
    except httpx.TimeoutException as exc:
        return ToolResult.failure(ToolErrorCode.TIMEOUT, f"搜索超时：{exc}", retryable=True)
    except httpx.HTTPError as exc:
        return ToolResult.failure(ToolErrorCode.NETWORK_ERROR, f"搜索网络错误：{exc}", retryable=True)
    finally:
        if own_client:
            http.close()

    if response.status_code in {401, 403}:
        return ToolResult.failure(
            ToolErrorCode.UNAUTHORIZED, "搜索服务拒绝凭据", retryable=False, provider="tavily"
        )
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        return ToolResult.failure(
            ToolErrorCode.RATE_LIMITED,
            "搜索服务限流",
            retryable=True,
            retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            provider="tavily",
        )
    if response.status_code >= 400:
        return ToolResult.failure(
            ToolErrorCode.UPSTREAM_ERROR,
            f"搜索服务返回 HTTP {response.status_code}",
            retryable=response.status_code >= 500,
            provider="tavily",
        )

    try:
        body = response.json()
    except ValueError:
        return ToolResult.failure(
            ToolErrorCode.UPSTREAM_ERROR, "搜索响应不是合法 JSON", retryable=False
        )

    raw_results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(raw_results, list):
        return ToolResult.failure(
            ToolErrorCode.UPSTREAM_ERROR, "搜索响应缺少 results", retryable=False
        )

    hits: list[SearchHit] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        hits.append(
            SearchHit(
                title=str(item.get("title") or url),
                url=url,
                snippet=str(item.get("content") or ""),
                score=float(item["score"]) if isinstance(item.get("score"), (int, float)) else None,
                published_at=_parse_date(item.get("published_date")),
                provider="tavily",
            )
        )

    if not hits:
        return ToolResult.failure(
            ToolErrorCode.EMPTY_RESULT, "搜索未返回可用结果", retryable=True, provider="tavily"
        )

    bundle = SearchBundle(
        query=query,
        hits=hits,
        provider="tavily",
        searched_at=datetime.now(UTC),
        raw_count=len(raw_results),
    )
    return ToolResult.success(bundle, provider="tavily", query=query)