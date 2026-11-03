"""离线回放：只使用冻结的本地资料，不访问公网。

回放模式用于稳定演示与可复现测试（设计文档 §12.1）。它不冒充实时检索结果，
`provider` 与报告中都会标明数据来自固定快照。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from insightflow.research.schemas import FetchScope, SourceKind, SourceRole
from insightflow.tools.contracts import ToolErrorCode, ToolResult
from insightflow.tools.fetch import FetchedSource, parse_html
from insightflow.tools.search import SearchBundle, SearchHit
from insightflow.tools.urls import canonicalize


class FixtureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    snippet: str = ""
    kind: str = "media"
    role: str = "unknown"
    published_at: datetime | None = None
    page: str
    queries: list[str] = Field(default_factory=list)


class FixtureManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    description: str = ""
    synthetic: bool = True
    items: list[FixtureItem]


class ReplayProvider:
    """把 `evaluation/fixtures/<id>/manifest.json` 当成一个只读的搜索 + 抓取服务。"""

    def __init__(self, fixtures_dir: Path, fixture_id: str) -> None:
        self.fixture_id = fixture_id
        self.root = Path(fixtures_dir) / fixture_id
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"回放资料不存在：{manifest_path}")
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = FixtureManifest.model_validate(data)

    @property
    def items(self) -> list[FixtureItem]:
        return self.manifest.items

    def _match(self, query: str) -> list[FixtureItem]:
        lowered = query.lower()
        return [
            item
            for item in self.items
            if not item.queries or any(token.lower() in lowered for token in item.queries)
        ]

    def search(self, query: str, *, max_results: int = 5) -> ToolResult[SearchBundle]:
        hits = [
            SearchHit(
                title=item.title,
                url=item.url,
                snippet=item.snippet,
                published_at=item.published_at,
                provider="replay",
            )
            for item in self._match(query)[:max_results]
        ]
        if not hits:
            return ToolResult.failure(
                ToolErrorCode.EMPTY_RESULT, "回放资料中没有匹配结果", retryable=False
            )
        bundle = SearchBundle(
            query=query,
            hits=hits,
            provider="replay",
            searched_at=datetime.now(UTC),
            raw_count=len(hits),
        )
        return ToolResult.success(bundle, provider="replay", fixture=self.fixture_id)

    def fetch(self, url: str) -> ToolResult[FetchedSource]:
        item = next((i for i in self.items if canonicalize(i.url) == canonicalize(url)), None)
        if item is None:
            return ToolResult.failure(
                ToolErrorCode.NOT_FOUND, "回放资料中没有该 URL", retryable=False
            )
        page_path = self.root / item.page
        if not page_path.is_file():
            return ToolResult.failure(
                ToolErrorCode.NOT_FOUND, f"回放页面缺失：{item.page}", retryable=False
            )
        html = page_path.read_text(encoding="utf-8")
        fetched_at = datetime.now(UTC)
        parsed = parse_html(html, url=item.url, fetched_at=fetched_at, truncated=False, status=200)
        parsed.update(
            {
                "title": item.title or parsed["title"],
                "published_at": item.published_at or parsed["published_at"],
            }
        )
        media_kind = SourceKind[item.kind.upper()] if item.kind.upper() in SourceKind.__members__ else SourceKind.MEDIA
        # 角色必须一路带到报告：转载/二手报道不能算作独立验证，丢了它校验就成了摆设。
        role = (
            SourceRole[item.role.upper()]
            if item.role.upper() in SourceRole.__members__
            else SourceRole.UNKNOWN
        )
        result = FetchedSource(
            url=item.url,
            final_url=item.url,
            canonical_url=canonicalize(item.url),
            fetch_scope=FetchScope.FULL,
            status_code=200,
            content_type="text/html",
            provider="replay",
            kind=media_kind,
            role=role,
            **parsed,
        )
        return ToolResult.success(result, provider="replay", fixture=self.fixture_id)


__all__ = ["FixtureItem", "FixtureManifest", "ReplayProvider"]