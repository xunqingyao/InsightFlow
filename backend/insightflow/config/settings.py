"""运行配置。

凭据只从服务端环境（含本地 .env）读取，绝不写入数据库、报告或导出文件。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # 允许用字段名直接构造（测试与 CLI 覆盖），同时保留 env 别名
        populate_by_name=True,
    )

    env: str = Field(default="local", validation_alias="INSIGHTFLOW_ENV")
    data_dir: Path = Field(default=Path("./data"), validation_alias="INSIGHTFLOW_DATA_DIR")
    timezone: str = Field(default="Asia/Shanghai", validation_alias="INSIGHTFLOW_TIMEZONE")

    # 模型（OpenAI 兼容 /chat/completions）
    model_base_url: str = Field(
        default="https://api.deepseek.com/v1", validation_alias="MODEL_BASE_URL"
    )
    model_api_key: str | None = Field(default=None, validation_alias="MODEL_API_KEY")
    model_name: str = Field(default="deepseek-chat", validation_alias="MODEL_NAME")
    model_timeout_seconds: float = Field(default=120.0, validation_alias="MODEL_TIMEOUT_SECONDS")

    # 搜索
    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    tavily_search_endpoint: str = Field(
        default="https://api.tavily.com/search", validation_alias="TAVILY_SEARCH_ENDPOINT"
    )

    # GitHub
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    github_api_base: str = Field(
        default="https://api.github.com", validation_alias="GITHUB_API_BASE"
    )

    # 网络
    user_agent: str = Field(
        default="InsightFlow/0.1 (+research agent; contact: you@example.com)",
        validation_alias="USER_AGENT",
    )
    http_timeout_seconds: float = Field(default=20.0, validation_alias="HTTP_TIMEOUT_SECONDS")
    fetch_max_bytes: int = Field(default=3_000_000, validation_alias="FETCH_MAX_BYTES")
    # 某些开发环境（企业透明代理、容器 DNS）会把公网域名解析到私有网段，
    # 此时 IP 级 SSRF 校验无法生效。打开这个开关会把“非公网 IP”降级为放行，
    # 仍然阻止 localhost、私有主机名与非法 scheme。只应在受控环境里打开。
    allow_private_addresses: bool = Field(
        default=False, validation_alias="INSIGHTFLOW_ALLOW_PRIVATE_ADDRESSES"
    )

    # 回放资料根目录；不设时按 CWD/仓库根自动探测
    fixtures_root: Path | None = Field(default=None, validation_alias="INSIGHTFLOW_FIXTURES_DIR")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "insightflow.db"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def fixtures_dir(self) -> Path:
        """回放资料目录：显式配置 > CWD > 仓库根（backend 的上一级）。"""
        if self.fixtures_root is not None:
            return Path(self.fixtures_root)
        candidates = [
            Path.cwd() / "evaluation" / "fixtures",
            Path.cwd().parent / "evaluation" / "fixtures",
            Path(__file__).resolve().parents[3] / "evaluation" / "fixtures",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]

    @property
    def model_configured(self) -> bool:
        return bool(self.model_api_key)

    @property
    def search_configured(self) -> bool:
        return bool(self.tavily_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.snapshots_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """测试用：丢弃已缓存的设置。"""
    get_settings.cache_clear()