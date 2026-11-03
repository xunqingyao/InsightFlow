"""跑 evaluation/tasks 下的开发题，打印并保存结果。

用法：
    uv run python evaluation/runners/run_tasks.py            # 全部
    uv run python evaluation/runners/run_tasks.py dev-002    # 指定任务
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "evaluation" / "tasks"
RESULTS = ROOT / "evaluation" / "results"

sys.path.insert(0, str(ROOT / "backend"))

from insightflow.config.settings import Settings  # noqa: E402
from insightflow.research.pipeline import ResearchEngine  # noqa: E402
from insightflow.research.schemas import RunMode, RunRecord, RunStatus, stable_id  # noqa: E402
from insightflow.research.time_window import normalize  # noqa: E402
from insightflow.storage.db import Database  # noqa: E402


def run_task(task: dict) -> dict:
    task_id = task["task_id"]
    data_dir = RESULTS / task_id / "runs"
    settings = Settings(
        data_dir=data_dir,
        fixtures_root=ROOT / "evaluation" / "fixtures",
        tavily_api_key=None,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()

    request = normalize(
        task["query"],
        as_of=datetime.fromisoformat(task["as_of"]),
        timezone=task.get("timezone"),
        window=None if task.get("days") is None else timedelta(days=task["days"]),
        mode=RunMode(task["mode"]),
        fixture_id=task.get("fixture_id"),
    )
    run_id = stable_id("run", task_id, request.as_of.isoformat())
    if db.get_run(run_id):
        db.delete_run(run_id)
    db.create_run(
        RunRecord(
            run_id=run_id,
            status=RunStatus.QUEUED,
            request=request,
            created_at=request.as_of,
            updated_at=request.as_of,
        )
    )

    engine = ResearchEngine(settings, db)
    record = engine.run(run_id)
    report = db.latest_report(run_id)
    sources = db.list_sources(run_id)
    assert report is not None

    checks = {check.check: check.status.value for check in report.checks}
    failures: list[str] = []
    expect = task.get("expect", {})
    if expect.get("status") and str(record.status) not in expect["status"]:
        failures.append(f"status={record.status} 不在 {expect['status']}")
    if expect.get("min_sources") and len(sources) < expect["min_sources"]:
        failures.append(f"来源 {len(sources)} < {expect['min_sources']}")
    if expect.get("supplement_rounds") is not None and report.supplement_rounds != expect["supplement_rounds"]:
        failures.append(
            f"补搜轮次 {report.supplement_rounds} != {expect['supplement_rounds']}"
        )
    if expect.get("stop_reason") and record.stop_reason != expect["stop_reason"]:
        failures.append(f"stop_reason={record.stop_reason} != {expect['stop_reason']}")
    if expect.get("min_findings") and len(report.findings) < expect["min_findings"]:
        failures.append(f"结论 {len(report.findings)} < {expect['min_findings']}")
    if expect.get("checks_all_pass") and any(v != "pass" for v in checks.values()):
        failures.append(f"校验未全通过：{checks}")
    for name in expect.get("warn_checks", []):
        if checks.get(name) != "warn":
            failures.append(f"{name} 期望 warn，实际 {checks.get(name)}")
    for finding in report.findings:
        if not finding.evidence_ids:
            failures.append(f"结论缺证据：{finding.finding_id}")
            break

    out_dir = RESULTS / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report.markdown, encoding="utf-8")

    return {
        "task_id": task_id,
        "purpose": task.get("purpose"),
        "run_id": run_id,
        "status": str(record.status),
        "report_status": report.status,
        "stop_reason": record.stop_reason,
        "sources": len(sources),
        "findings": len(report.findings),
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def main(argv: list[str]) -> int:
    wanted = set(argv)
    summaries = []
    for path in sorted(TASKS.glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if wanted and task["task_id"] not in wanted:
            continue
        summary = run_task(task)
        summaries.append(summary)
        mark = "PASS" if summary["passed"] else "FAIL"
        print(
            f"[{mark}] {summary['task_id']}｜{summary['status']}/报告 {summary['report_status']}"
            f"｜来源 {summary['sources']}｜结论 {summary['findings']}"
        )
        for failure in summary["failures"]:
            print(f"   - {failure}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if all(item["passed"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
