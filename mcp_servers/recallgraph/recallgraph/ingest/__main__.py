"""命令行：python -m recallgraph.ingest <文件> [--json 输出路径]

只做一件事：把文件解析成 chunks 并打印摘要。方便在接入数据库之前先肉眼验货。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recallgraph.ingest.pipeline import ingest, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m recallgraph.ingest", description="解析一篇论文/笔记，输出切块结果")
    parser.add_argument("path", type=Path, help="PDF / Markdown / 文本文件")
    parser.add_argument("--json", type=Path, default=None, help="把完整结果写成 JSON")
    parser.add_argument("--target-chars", type=int, default=500, help="目标块长（默认 500 字）")
    parser.add_argument("--max-chars", type=int, default=1000, help="单块上限（默认 1000 字）")
    parser.add_argument("--keep-references", action="store_true", help="保留参考文献部分（默认跳过）")
    args = parser.parse_args(argv)

    from recallgraph.ingest.chunker import ChunkSettings

    settings = ChunkSettings(target_chars=args.target_chars, max_chars=args.max_chars)
    result = ingest(args.path, chunk_settings=settings, stop_at_references=not args.keep_references)

    print(summarize(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())