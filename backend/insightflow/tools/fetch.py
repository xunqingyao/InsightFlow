"""网页正文读取与元数据抽取。

只做"抓取 + 结构化"，不做摘要。抓到的正文是否交给模型、交给多少，
由编排层决定并记录 `read_depth`；本模块只报告事实：抓了多大、截断没有、时间来自哪里。
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ConfigDict, Field

from insightflow.research.schemas import FetchScope, ReadDepth, SourceKind, SourceRole
from insightflow.tools.contracts import ToolErrorCode, ToolResult
from insightflow.tools.net import BlockedURLError, InvalidURLError, Resolver, http_get
from insightflow.tools.urls import canonicalize

ACCEPTED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_WHITESPACE = re.compile(r"[ \t\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_META_TIME_KEYS = (
    ("property", "article:published_time"),
    ("name", "article:published_time"),
    ("itemprop", "datePublished"),
    ("name", "publish-date"),
    ("name", "pubdate"),
    ("name", "date"),
    ("property", "og:updated_time"),
)


class FetchedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str
    canonical_url: str
    title: str
    text: str
    content_hash: str
    fetch_scope: FetchScope
    status_code: int
    content_type: str
    truncated: bool
    fetched_at: datetime
    published_at: datetime | None = None
    updated_at: datetime | None = None
    published_at_source: str | None = None
    provider: str = "http"
    kind: SourceKind = SourceKind.MEDIA
    role: SourceRole = SourceRole.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)

    def read_depth(self, char_limit: int = 12_000) -> ReadDepth:
        """按"给模型的字符上限"判断实际阅读深度。"""
        length = len(self.text.strip())
        if length == 0:
            return ReadDepth.NONE
        if self.truncated or length > char_limit:
            return ReadDepth.PARTIAL
        return ReadDepth.FULL


def _clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _pick_container(soup: BeautifulSoup) -> Tag:
    for selector in ("article", "main", "[role=main]", "#content", ".post-content", ".article-body"):
        node = soup.select_one(selector)
        if isinstance(node, Tag) and len(node.get_text(strip=True)) > 200:
            return node
    return soup.body or soup


def _block_text(container: Tag) -> str:
    for junk in container.select("script, style, noscript, svg, iframe, form, nav, footer, aside"):
        junk.decompose()
    blocks: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "pre", "td"]):
        chunk = node.get_text(" ", strip=True)
        if not chunk:
            continue
        if blocks and blocks[-1] == chunk:
            continue
        blocks.append(chunk)
    if not blocks:
        return _clean_text(container.get_text("\n", strip=True))
    return _clean_text("\n".join(blocks))


def _extract_title(soup: BeautifulSoup) -> str:
    for attrs in ({"property": "og:title"}, {"name": "twitter:title"}):
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content"):
            return str(node["content"]).strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def _parse_iso(value: str) -> datetime | None:
    text = value.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            if fmt is None:
                parsed = datetime.fromisoformat(text)
            else:
                parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _extract_times(soup: BeautifulSoup) -> tuple[datetime | None, datetime | None, str | None]:
    published: datetime | None = None
    updated: datetime | None = None
    source: str | None = None
    for attr, key in _META_TIME_KEYS:
        node = soup.find("meta", attrs={attr: key})
        if not node or not node.get("content"):
            continue
        parsed = _parse_iso(str(node["content"]))
        if parsed is None:
            continue
        if key == "og:updated_time":
            updated = parsed
        elif published is None:
            published, source = parsed, f"meta:{key}"
    if published is None:
        node = soup.find("time")
        if isinstance(node, Tag) and node.get("datetime"):
            parsed = _parse_iso(str(node["datetime"]))
            if parsed is not None:
                published, source = parsed, "element:time[datetime]"
    return published, updated, source


def _looks_like_challenge(soup: BeautifulSoup) -> bool:
    title = (soup.title.string or "").lower() if soup.title and soup.title.string else ""
    return any(token in title for token in ("access denied", "just a moment", "captcha"))


def parse_html(html: str, *, url: str, fetched_at: datetime, truncated: bool, status: int) -> Any:
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)
    published, updated, source = _extract_times(soup)
    container = _pick_container(soup)
    text = _block_text(container)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "title": title,
        "text": text,
        "content_hash": content_hash,
        "published_at": published,
        "updated_at": updated,
        "published_at_source": source,
        "metadata": {
            "challenge_page": _looks_like_challenge(soup),
            "char_count": len(text),
            "status": status,
        },
        "truncated": truncated,
        "fetched_at": fetched_at,
    }


def fetch_source(
    url: str,
    *,
    timeout: float = 20.0,
    max_bytes: int = 3_000_000,
    user_agent: str = "InsightFlow/0.1",
    resolver: Resolver | None = None,
    allow_private_addresses: bool = False,
    client: httpx.Client | None = None,
) -> ToolResult[FetchedSource]:
    fetched_at = datetime.now(UTC)
    try:
        payload = http_get(
            url,
            timeout=timeout,
            max_bytes=max_bytes,
            user_agent=user_agent,
            resolver=resolver,
            allow_private_addresses=allow_private_addresses,
            client=client,
        )
    except InvalidURLError as exc:
        return ToolResult.failure(ToolErrorCode.INVALID_INPUT, str(exc), retryable=False)
    except BlockedURLError as exc:
        return ToolResult.failure(ToolErrorCode.BLOCKED_ADDRESS, str(exc), retryable=False)
    except httpx.TimeoutException as exc:
        return ToolResult.failure(ToolErrorCode.TIMEOUT, f"读取超时：{exc}", retryable=True)
    except httpx.TooManyRedirects as exc:
        return ToolResult.failure(ToolErrorCode.UPSTREAM_ERROR, str(exc), retryable=False)
    except httpx.HTTPError as exc:
        return ToolResult.failure(ToolErrorCode.NETWORK_ERROR, f"网络错误：{exc}", retryable=True)

    if payload.status_code == 429:
        retry_after = payload.headers.get("retry-after")
        return ToolResult.failure(
            ToolErrorCode.RATE_LIMITED,
            "来源方限流",
            retryable=True,
            retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            status=payload.status_code,
        )
    if payload.status_code in {401, 403}:
        return ToolResult.failure(
            ToolErrorCode.UNAUTHORIZED,
            f"访问被拒绝（HTTP {payload.status_code}）",
            retryable=False,
            status=payload.status_code,
        )
    if payload.status_code == 404:
        return ToolResult.failure(ToolErrorCode.NOT_FOUND, "来源不存在（404）", retryable=False)
    if payload.status_code >= 400:
        return ToolResult.failure(
            ToolErrorCode.UPSTREAM_ERROR,
            f"来源返回 HTTP {payload.status_code}",
            retryable=payload.status_code >= 500,
            status=payload.status_code,
        )

    if payload.content_type not in ACCEPTED_CONTENT_TYPES:
        return ToolResult.failure(
            ToolErrorCode.UNSUPPORTED_CONTENT,
            f"暂不支持的内容类型：{payload.content_type or 'unknown'}"
            "（PDF 解析列入 P1，不会假装已读）",
            retryable=False,
            content_type=payload.content_type,
        )

    parsed = parse_html(
        payload.text(),
        url=payload.url,
        fetched_at=fetched_at,
        truncated=payload.truncated,
        status=payload.status_code,
    )
    result = FetchedSource(
        url=url,
        final_url=payload.url,
        canonical_url=canonicalize(payload.url),
        fetch_scope=FetchScope.PARTIAL if payload.truncated else FetchScope.FULL,
        status_code=payload.status_code,
        content_type=payload.content_type,
        **parsed,
    )
    if not result.text:
        return ToolResult.failure(
            ToolErrorCode.EMPTY_RESULT, "未抽取到正文", retryable=False, url=payload.url
        )
    return ToolResult.success(result, provider="http")