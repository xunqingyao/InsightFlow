"""命令行入口：不依赖 HTTP 也能跑通一次研究。

用法：
    uv run python -m insightflow.cli run --query "最近一周 AI Agent 动态" --days 7
    uv run python -m insightflow.cli run --query "..." --replay briefing-2026-10-27 --offline
    uv run python -m insightflow.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from insightflow.config.settings import get_settings
from insightflow.research.agents.model import OfflineResearchModel, build_model
from insightflow.research.budget import Budget
from insightflow.research.pipeline import ResearchEngine
from insightflow.research.schemas import ResearchRequest, RunMode, RunRecord, RunStatus, stable_id
from insightflow.research.time_window import normalize
from insightflow.storage.db import Database


def _build_request(args: argparse.Namespace) -> ResearchRequest:
    window = timedelta(days=args.days) if args.days else None
    return normalize(
        args.query,
        as_of=datetime.fromisoformat(args.as_of) if args.as_of else None,
        timezone=args.timezone,
        window=window,
        mode=RunMode.REPLAY if args.replay else RunMode.ONLINE,
        fixture_id=args.replay,
    )


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()
    try:
        request = _build_request(args)
    except ValueError as exc:
        print(f"请求不合法：{exc}", file=sys.stderr)
        return 2

    now = datetime.now(tz=request.as_of.tzinfo)
    record = RunRecord(
        run_id=stable_id("run", request.query, request.as_of.isoformat()),
        status=RunStatus.QUEUED,
        request=request,
        created_at=now,
        updated_at=now,
    )
    if db.get_run(record.run_id):
        if not args.force:
            print(f"已存在的运行：{record.run_id}（加 --force 可重跑并覆盖）", file=sys.stderr)
            existing = db.latest_report(record.run_id)
            if existing:
                print(existing.markdown)
                return 0
        db.delete_run(record.run_id)
    db.create_run(record)

    model = OfflineResearchModel() if args.offline else None
    engine = ResearchEngine(settings, db, model=model, budget=Budget())
    print(f"运行 {record.run_id}", file=sys.stderr)
    print(f"窗口 {request.window_start.isoformat()} → {request.window_end.isoformat()}", file=sys.stderr)
    result = engine.run(record.run_id)
    report = db.latest_report(record.run_id)

    if report is None:
        print(f"运行结束但未生成报告：status={result.status} error={result.error}", file=sys.stderr)
        return 1
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.markdown, encoding="utf-8")
        print(f"报告已写入 {target}", file=sys.stderr)
    else:
        print(report.markdown)
    print(
        f"状态 {result.status}｜校验 "
        + json.dumps({c.check: str(c.status) for c in report.checks}, ensure_ascii=False),
        file=sys.stderr,
    )
    return 0 if result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL} else 1



def cmd_resume(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()
    record = db.get_run(args.run_id)
    if record is None:
        print(f"没有这个运行：{args.run_id}", file=sys.stderr)
        return 2
    takeover = record.status is RunStatus.RUNNING and args.force
    if record.status not in {RunStatus.RECOVERABLE, RunStatus.QUEUED} and not takeover:
        print(
            f"当前状态 {record.status.value}，只有 recoverable/queued 可以续跑；"
            "如果确认占用它的进程已经死了（running 但没人在跑），加 --force 接管。",
            file=sys.stderr,
        )
        return 1
    if takeover:
        record = record.model_copy(update={"stop_reason": "process_restart"})
    if not db.update_run(record.run_id, expected=record.status, status=RunStatus.QUEUED):
        print("另一个进程抢先接手了这个运行", file=sys.stderr)
        return 1
    counters = record.counters.model_copy(update={"resumes": record.counters.resumes + 1})
    db.update_run(record.run_id, counters=counters)
    print(
        f"续跑 {record.run_id}（第 {counters.resumes} 次续跑，已有检索 {counters.search_calls} 次、"
        f"抓取 {counters.source_fetches} 次）",
        file=sys.stderr,
    )
    model = OfflineResearchModel() if args.offline else None
    engine = ResearchEngine(settings, db, model=model, budget=Budget())
    result = engine.run(record.run_id)
    report = db.latest_report(record.run_id)
    if report is None:
        print(f"续跑结束但未生成报告：status={result.status} error={result.error}", file=sys.stderr)
        return 1
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.markdown, encoding="utf-8")
        print(f"报告已写入 {target}", file=sys.stderr)
    else:
        print(report.markdown)
    print(f"状态 {result.status}｜续跑次数 {counters.resumes}", file=sys.stderr)
    return 0 if result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL} else 1


def cmd_list(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = Database(settings.db_path)
    db.init_schema()
    for run in db.list_runs(limit=args.limit):
        print(f"{run.run_id}\t{run.status}\t{run.created_at.isoformat()}\t{run.request.query}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("insightflow.api.app:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insightflow", description="InsightFlow 研究 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行一次研究并输出 Markdown 报告")
    run.add_argument("--query", required=True)
    run.add_argument("--days", type=int, default=None, help="时间窗口天数，默认按问题推断（7 天）")
    run.add_argument("--as-of", dest="as_of", default=None, help="ISO 时间，默认当前时间")
    run.add_argument("--timezone", default="Asia/Shanghai")
    run.add_argument("--replay", default=None, help="回放资料 ID（离线）")
    run.add_argument("--offline", action="store_true", help="强制使用模板模型（不调用 LLM）")
    run.add_argument("--out", default=None, help="报告输出路径")
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="续跑一个中断的运行（复用已抓到的正文）")
    resume.add_argument("--run-id", dest="run_id", required=True)
    resume.add_argument("--offline", action="store_true")
    resume.add_argument("--out", default=None)
    resume.add_argument("--force", action="store_true", help="接管一个停在 running 但没人跑的 run")
    resume.set_defaults(func=cmd_resume)

    listing = sub.add_parser("list", help="列出最近的运行")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(func=cmd_list)

    serve = sub.add_parser("serve", help="启动 HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())