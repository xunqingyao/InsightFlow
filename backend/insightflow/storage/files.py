"""文件快照存储：原文与报告都先写临时文件再原子替换（设计文档 §9.4）。

目录布局：
    <data>/snapshots/<run_id>/<source_id>.txt      来源原文纯文本快照
    <data>/snapshots/<run_id>/<source_id>.html     原始 HTML（若抓到）
    <data>/reports/<run_id>/<report_id>.md         报告 Markdown
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError(f"路径越界：{relative}")
        return target

    def save_text(self, run_id: str, source_id: str, text: str, suffix: str = ".txt") -> str:
        relative = f"{run_id}/{source_id}{suffix}"
        _atomic_write(self._resolve(relative), text.encode("utf-8"))
        return relative

    def save_bytes(self, run_id: str, name: str, payload: bytes) -> str:
        relative = f"{run_id}/{name}"
        path = self._resolve(relative)
        _atomic_write(path, payload)
        return relative

    def read_text(self, relative: str) -> str:
        return self._resolve(relative).read_text(encoding="utf-8")

    def exists(self, relative: str) -> bool:
        try:
            return self._resolve(relative).is_file()
        except ValueError:
            return False

    def save_report(self, reports_root: Path, run_id: str, report_id: str, markdown: str) -> str:
        target = Path(reports_root).resolve() / run_id / f"{report_id}.md"
        _atomic_write(target, markdown.encode("utf-8"))
        return str(target.relative_to(Path(reports_root).resolve()))