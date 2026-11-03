"""受限 HTTP 访问：协议校验、公网地址校验、重定向逐跳复验、体积上限。

设计文档 §8.3：URL 只允许 http/https 与公网地址，重定向后再次校验，并设置体积和读取超时。
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

Resolver = Callable[[str], list[str]]
MAX_REDIRECTS = 5


class BlockedURLError(ValueError):
    """URL 不满足抓取策略（协议、主机、目标地址）。"""


class InvalidURLError(BlockedURLError):
    """URL 本身不成立（缺 scheme、缺主机名、端口非法）。

    与 `BlockedURLError` 分开：前者是“调用方给错了，改参数可修”，后者是“目标被策略
    拒绝，换参数没用”。错误码要能指导模型下一步动作，所以不能都归成 blocked_address。
    """


def _default_resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [info[4][0] for info in infos]


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(
    url: str,
    *,
    resolver: Resolver | None = None,
    allow_private_addresses: bool = False,
) -> str:
    """校验并返回规范化 URL；不合法直接抛 `BlockedURLError`。"""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise InvalidURLError(f"仅允许 http/https，收到 {parts.scheme or '空 scheme'}")
    host = parts.hostname
    if not host:
        raise InvalidURLError("URL 缺少主机名")
    try:
        parts.port
    except ValueError as exc:
        raise InvalidURLError(f"URL 端口非法：{url}") from exc
    if host.lower() in {"localhost", "localhost.localdomain"}:
        raise BlockedURLError("禁止访问本机地址")

    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        resolve = resolver or _default_resolve
        try:
            raw = resolve(host)
        except OSError as exc:  # DNS 失败属于网络错误，交给调用方转成可重试错误
            raise BlockedURLError(f"域名解析失败：{host}") from exc
        addresses = []
        for item in raw:
            try:
                addresses.append(ipaddress.ip_address(item))
            except ValueError:
                continue
    if not addresses:
        raise BlockedURLError(f"无法解析主机：{host}")
    for ip in addresses:
        if not _is_public(ip) and not allow_private_addresses:
            raise BlockedURLError(f"禁止访问非公网地址：{host} -> {ip}")
    return url


@dataclass(slots=True)
class HttpPayload:
    url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    truncated: bool

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def charset(self) -> str:
        for part in self.headers.get("content-type", "").split(";")[1:]:
            key, _, value = part.strip().partition("=")
            if key.lower() == "charset" and value:
                return value.strip('"')
        return "utf-8"

    def text(self) -> str:
        try:
            return self.body.decode(self.charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


def http_get(
    url: str,
    *,
    timeout: float = 20.0,
    max_bytes: int = 3_000_000,
    user_agent: str = "InsightFlow/0.1",
    headers: dict[str, str] | None = None,
    resolver: Resolver | None = None,
    allow_private_addresses: bool = False,
    client: httpx.Client | None = None,
) -> HttpPayload:
    """带逐跳校验的 GET。

    超过 `max_bytes` 时截断并置 `truncated=True`，由调用方把阅读深度标为 partial。
    """
    merged_headers = {"user-agent": user_agent, "accept": "*/*"}
    if headers:
        merged_headers.update(headers)

    own_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_url(
                current, resolver=resolver, allow_private_addresses=allow_private_addresses
            )
            with http.stream("GET", current, headers=merged_headers) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise httpx.HTTPStatusError(
                            "重定向缺少 location", request=response.request, response=response
                        )
                    current = urljoin(current, location)
                    continue
                chunks: list[bytes] = []
                total = 0
                truncated = False
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        chunks.append(chunk[: max_bytes - (total - len(chunk))])
                        truncated = True
                        break
                    chunks.append(chunk)
                return HttpPayload(
                    url=current,
                    status_code=response.status_code,
                    headers={k.lower(): v for k, v in response.headers.items()},
                    body=b"".join(chunks),
                    truncated=truncated,
                )
        raise httpx.TooManyRedirects(f"重定向超过 {MAX_REDIRECTS} 次")
    finally:
        if own_client:
            http.close()