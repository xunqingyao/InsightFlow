"""GitHub Releases 读取。

注意（设计文档 §6.1）：没有关联 Release 的普通 tag 不会出现在这里，
因此"没有新 Release"不能被推断成"仓库没有新版本"。
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict

from insightflow.tools.contracts import ToolErrorCode, ToolResult


class GithubRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    tag_name: str
    name: str
    prerelease: bool
    draft: bool
    published_at: datetime | None = None
    body: str = ""
    html_url: str


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def get_releases(
    repo: str,
    *,
    limit: int = 5,
    token: str | None = None,
    api_base: str = "https://api.github.com",
    timeout: float = 20.0,
    client: httpx.Client | None = None,
) -> ToolResult[list[GithubRelease]]:
    """`repo` 形如 `owner/name`。"""
    if "/" not in repo:
        return ToolResult.failure(
            ToolErrorCode.INVALID_INPUT, "repo 需形如 owner/name", retryable=False
        )
    headers = {
        "accept": "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
        "user-agent": "InsightFlow/0.1",
    }
    if token:
        headers["authorization"] = f"Bearer {token}"

    own_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.get(
            f"{api_base.rstrip('/')}/repos/{repo}/releases",
            params={"per_page": max(1, min(limit, 30))},
            headers=headers,
        )
    except httpx.TimeoutException as exc:
        return ToolResult.failure(ToolErrorCode.TIMEOUT, f"GitHub 请求超时：{exc}", retryable=True)
    except httpx.HTTPError as exc:
        return ToolResult.failure(
            ToolErrorCode.NETWORK_ERROR, f"GitHub 网络错误：{exc}", retryable=True
        )
    finally:
        if own_client:
            http.close()

    if response.status_code == 404:
        return ToolResult.failure(ToolErrorCode.NOT_FOUND, f"仓库不存在：{repo}", retryable=False)
    if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
        reset = response.headers.get("x-ratelimit-reset")
        return ToolResult.failure(
            ToolErrorCode.RATE_LIMITED,
            "GitHub 未认证请求额度用尽",
            retryable=True,
            provider="github",
            details={"reset": reset},
        )
    if response.status_code >= 400:
        return ToolResult.failure(
            ToolErrorCode.UPSTREAM_ERROR,
            f"GitHub 返回 HTTP {response.status_code}",
            retryable=response.status_code >= 500,
            provider="github",
        )
    try:
        items = response.json()
    except ValueError:
        return ToolResult.failure(ToolErrorCode.UPSTREAM_ERROR, "GitHub 响应不是合法 JSON")
    if not isinstance(items, list):
        return ToolResult.failure(ToolErrorCode.UPSTREAM_ERROR, "GitHub 响应结构异常")

    releases: list[GithubRelease] = []
    for item in items:
        if not isinstance(item, dict) or item.get("draft"):
            continue
        releases.append(
            GithubRelease(
                repo=repo,
                tag_name=str(item.get("tag_name") or ""),
                name=str(item.get("name") or item.get("tag_name") or ""),
                prerelease=bool(item.get("prerelease")),
                draft=bool(item.get("draft")),
                published_at=_parse_time(item.get("published_at")),
                body=str(item.get("body") or "")[:4000],
                html_url=str(item.get("html_url") or ""),
            )
        )
    if not releases:
        return ToolResult.failure(
            ToolErrorCode.EMPTY_RESULT,
            "该仓库没有已发布 Release（不代表没有新 tag）",
            retryable=False,
            provider="github",
        )
    return ToolResult.success(releases, provider="github", count=len(releases))