from __future__ import annotations

from insightflow.tools.contracts import ToolErrorCode, ToolResult


def test_success_has_no_error() -> None:
    result = ToolResult[int].success(3, provider="test")
    assert result.ok is True
    assert result.error is None
    assert result.data == 3
    assert result.metadata["provider"] == "test"


def test_failure_has_error_and_no_data() -> None:
    result = ToolResult[int].failure(ToolErrorCode.TIMEOUT, "超时", retryable=True)
    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TIMEOUT
    assert result.error.retryable is True


def test_cannot_mark_success_with_error() -> None:
    import pytest

    with pytest.raises(ValueError):
        ToolResult[int](ok=True, data=1, error={"code": "timeout", "message": "x"})


def test_fetch_scope_matches_design_values() -> None:
    """设计文档规定 fetch_scope 取值 none/snippet/partial/full，不能自创枚举。"""
    from insightflow.research.schemas import FetchScope, ReadDepth

    assert {item.value for item in FetchScope} == {"none", "snippet", "partial", "full"}
    assert {item.value for item in ReadDepth} == {"none", "snippet", "partial", "full"}


def test_private_address_guard_can_be_relaxed_for_transparent_proxies() -> None:
    from insightflow.tools.net import BlockedURLError, validate_public_url

    fake_dns = lambda host: ["198.19.0.69"]  # noqa: E731  # 透明代理常见的“假公网”地址

    try:
        validate_public_url("https://example.com/a", resolver=fake_dns)
    except BlockedURLError as exc:
        assert "非公网" in str(exc)
    else:
        raise AssertionError("默认必须拦截私有网段")

    assert validate_public_url(
        "https://example.com/a", resolver=fake_dns, allow_private_addresses=True
    ) == "https://example.com/a"

    # 开关只放宽 IP 段，不放宽本机与非法 scheme
    for url in ("http://localhost:8000/x", "ftp://example.com/x"):
        try:
            validate_public_url(url, resolver=fake_dns, allow_private_addresses=True)
        except BlockedURLError:
            pass
        else:
            raise AssertionError(f"应当拒绝 {url}")
