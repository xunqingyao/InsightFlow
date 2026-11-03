"""工具统一结果与错误约定（设计文档 §8.1）。

所有适配器返回 `ToolResult`：`ok / data / error / metadata`。
失败必须显式返回，不允许把"没抓到"包装成"读过了"。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

T = TypeVar("T")


class ToolErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    UPSTREAM_ERROR = "upstream_error"
    BLOCKED_ADDRESS = "blocked_address"
    TOO_LARGE = "too_large"
    UNSUPPORTED_CONTENT = "unsupported_content"
    NOT_FOUND = "not_found"
    EMPTY_RESULT = "empty_result"


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ToolErrorCode
    message: str
    retryable: bool = False
    retry_after: float | None = None
    details: dict[str, Any] = {}


class ToolResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: T | None = None
    error: ToolError | None = None
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _consistent(self) -> ToolResult[T]:
        if self.ok and self.error is not None:
            raise ValueError("成功结果不能携带 error")
        if not self.ok and self.error is None:
            raise ValueError("失败结果必须携带 error")
        if not self.ok and self.data is not None:
            raise ValueError("失败结果不能携带 data")
        return self

    @classmethod
    def success(cls, data: T, **metadata: Any) -> ToolResult[T]:
        return cls(ok=True, data=data, metadata=metadata)

    @classmethod
    def failure(
        cls,
        code: ToolErrorCode,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ToolResult[T]:
        return cls(
            ok=False,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
                retry_after=retry_after,
                details=details or {},
            ),
            metadata=metadata,
        )

    def envelope(self) -> dict[str, Any]:
        """给 MCP / HTTP 边界用的纯 JSON 形态。"""
        return self.model_dump(mode="json")