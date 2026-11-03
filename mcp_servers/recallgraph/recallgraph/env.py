"""极简 .env 加载：只在进程里补默认值，不覆盖已有环境变量。

密钥不写进代码、不进仓库：`.env` 已在 .gitignore 里；CI/评测用真实环境变量。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    target = path or REPO_ROOT / ".env"
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
