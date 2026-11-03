"""arXiv 元数据查询（Atom API）。

只取元数据与摘要：没有全文解析能力时，方法与实验设置的结论必须保持"未确认"，
不能因为读到了标题就当成读过论文。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field

from insightflow.tools.contracts import ToolErrorCode, ToolResult

API_ENDPOINT = "https://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_WS = re.compile(r"\s+")


class ArxivPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: str | None = None
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    abs_url: str
    pdf_url: str | None = None
    read_depth: str = "metadata"


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return _WS.sub(" ", node.text).strip()


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def parse_feed(xml_text: str) -> list[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = _text(entry.find("atom:id", _NS))
        match = re.search(r"abs/(?P<id>[^v]+)(?P<version>v\d+)?$", raw_id)
        arxiv_id = match.group("id") if match else raw_id
        version = match.group("version") if match else None
        pdf_url = None
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                version=version,
                title=_text(entry.find("atom:title", _NS)),
                abstract=_text(entry.find("atom:summary", _NS)),
                authors=[
                    _text(author.find("atom:name", _NS))
                    for author in entry.findall("atom:author", _NS)
                ],
                categories=[
                    c.get("term", "")
                    for c in entry.findall("atom:category", _NS)
                    if c.get("term")
                ],
                published_at=_parse_time(_text(entry.find("atom:published", _NS))),
                updated_at=_parse_time(_text(entry.find("atom:updated", _NS))),
                abs_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
            )
        )
    return papers


def search_arxiv(
    query: str,
    *,
    max_results: int = 5,
    timeout: float = 20.0,
    user_agent: str = "InsightFlow/0.1",
    client: httpx.Client | None = None,
) -> ToolResult[list[ArxivPaper]]:
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max(1, min(max_results, 20)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    own_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.get(API_ENDPOINT, params=params, headers={"user-agent": user_agent})
    except httpx.TimeoutException as exc:
        return ToolResult.failure(ToolErrorCode.TIMEOUT, f"arXiv 查询超时：{exc}", retryable=True)
    except httpx.HTTPError as exc:
        return ToolResult.failure(
            ToolErrorCode.NETWORK_ERROR, f"arXiv 网络错误：{exc}", retryable=True
        )
    finally:
        if own_client:
            http.close()

    if response.status_code == 429:
        return ToolResult.failure(
            ToolErrorCode.RATE_LIMITED, "arXiv 限流", retryable=True, provider="arxiv"
        )
    if response.status_code >= 400:
        return ToolResult.failure(
            ToolErrorCode.UPSTREAM_ERROR,
            f"arXiv 返回 HTTP {response.status_code}",
            retryable=response.status_code >= 500,
            provider="arxiv",
        )
    try:
        papers = parse_feed(response.text)
    except ET.ParseError:
        return ToolResult.failure(ToolErrorCode.UPSTREAM_ERROR, "arXiv 响应不是合法 XML")
    if not papers:
        return ToolResult.failure(ToolErrorCode.EMPTY_RESULT, "arXiv 未返回结果", retryable=True)
    return ToolResult.success(papers, provider="arxiv", count=len(papers))