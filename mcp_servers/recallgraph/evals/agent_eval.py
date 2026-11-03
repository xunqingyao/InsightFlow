"""端到端评测：让 Agent 自己调工具答题，然后两层判分。

规则层查"该有的有没有"（关键结论、页码引用、该不该拒答）；
模型层让 qwen-plus 当裁判，查"说的是不是真的、有没有编"。
两层都过才算通过——只靠规则会放过"引用了但说错"，只靠模型会放过"没引用"。

用法：
    python -m evals.agent_eval            # 全量 20 题
    python -m evals.agent_eval --only f1,mem2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.agent_cases import CASES, seed_memories
from recallgraph import agent
from recallgraph.env import load_dotenv
from recallgraph.storage import connect

CITATION = re.compile(r"第\s*\d+\s*页|《[^》]+》")
REFUSAL = re.compile(r"没有|未提及|未涉及|未找到|无法(?:回答|确认|判断)|不确定|不在(?:知识库|材料)中|无相关")
JUDGE_MODEL = "qwen-plus"

JUDGE_PROMPT = """你是严格的评测裁判。题目来自一个"个人知识库问答 Agent"的端到端评测。

判断标准：
1. 事实是否正确：回答里的关键结论必须与给定标准一致；说错了算 fail。
2. 是否编造：回答中出现标准之外、且知识库里不可能有的具体数字/人名/结论，算 fail。
3. 拒答题：标准是"知识库里没有答案"。如果回答明确说明没有/无法回答，且没有编造，算 pass；编造了算 fail。
4. 不要因为措辞、格式、长度扣分；回答比标准更详细但内容正确，算 pass。
5. 标准里没写、但回答里出现的“额外细节”，只有在它明显与事实矛盾或凭空捏造时才算 fail。

只输出 JSON：{"verdict": "pass" 或 "fail", "reason": "一句话说明"}"""


def rule_check(case: dict[str, Any], answer: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    low = answer.lower()
    for needle in case.get("must_contain", []):
        if needle.lower() not in low:
            reasons.append(f"缺少「{needle}」")
    for group in case.get("any_of", []):
        if not any(alt.lower() in low for alt in group):
            reasons.append(f"缺少任一：{'/'.join(group)}")
    if case.get("need_citation") and not CITATION.search(answer):
        reasons.append("没有页码或文献引用")
    if case.get("must_refuse") and not REFUSAL.search(answer):
        reasons.append("没有拒答")
    return not reasons, reasons


def judge(case: dict[str, Any], answer: str, *, api_key: str) -> tuple[bool, str]:
    standard = case.get("expected") or "关键结论：" + "；".join(
        case.get("must_contain", []) + [alt for group in case.get("any_of", []) for alt in group]
    )
    if case.get("must_refuse"):
        standard += "（知识库里没有这个信息，正确行为是明确说明没有，不得编造。）"
    body = agent.chat(
        {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"问题：{case['question']}\n\n标准：{standard}\n\n回答：{answer}",
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        api_key=api_key,
    )
    content = body["choices"][0]["message"]["content"]
    try:
        verdict = json.loads(content)
    except json.JSONDecodeError:
        return False, f"裁判输出无法解析：{content[:80]}"
    return verdict.get("verdict") == "pass", str(verdict.get("reason", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端评测")
    parser.add_argument("--only", default="", help="只跑指定 id，逗号分隔")
    parser.add_argument("--out", default="", help="报告路径")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY 未设置")

    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    cases = [case for case in CASES if not wanted or case["id"] in wanted]

    results: list[dict[str, Any]] = []
    with connect() as conn:
        seed_memories(conn)
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            out = agent.run(conn, case["question"], max_steps=8, trace=False)
            elapsed = time.perf_counter() - started
            conn.rollback()  # 单题失败不要污染后续题目的事务
            answer = out["answer"]
            rule_ok, rule_reasons = rule_check(case, answer)
            try:
                judge_ok, judge_reason = judge(case, answer, api_key=api_key)
            except (RuntimeError, KeyError) as exc:
                judge_ok, judge_reason = False, f"裁判调用失败：{exc}"
            passed = rule_ok and judge_ok
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "passed": passed,
                    "rule_ok": rule_ok,
                    "rule_reasons": rule_reasons,
                    "judge_ok": judge_ok,
                    "judge_reason": judge_reason,
                    "tools": len(out["steps"]),
                    "tool_names": [step["tool"] for step in out["steps"]],
                    "seconds": round(elapsed, 1),
                    "answer": answer,
                }
            )
            flag = "PASS" if passed else "FAIL"
            print(f"[{index}/{len(cases)}] {flag} {case['id']:<5} {elapsed:5.1f}s tools={len(out['steps'])} {judge_reason[:40]}")

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    categories = sorted({item["category"] for item in results})
    with_citation = [item for item in results if item["id"].startswith(("f", "c", "m")) and not item["id"].startswith("mem")]
    citation_rate = sum(1 for item in with_citation if CITATION.search(item["answer"])) / max(len(with_citation), 1)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 3),
        "by_category": {
            category: {
                "passed": sum(1 for item in results if item["category"] == category and item["passed"]),
                "total": sum(1 for item in results if item["category"] == category),
            }
            for category in categories
        },
        "citation_rate": round(citation_rate, 3),
        "avg_tools": round(sum(item["tools"] for item in results) / max(total, 1), 2),
        "avg_seconds": round(sum(item["seconds"] for item in results) / max(total, 1), 1),
        "results": results,
    }

    out_path = Path(args.out or f"evals/reports/agent-{datetime.now():%Y-%m-%d}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, out_path)
    out_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n通过 {passed}/{total}（{report['pass_rate']:.0%}），引用率 {citation_rate:.0%}，平均 {report['avg_tools']} 次工具调用")
    print(f"报告：{out_path}")


def write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Agent 端到端评测（{report['generated_at']}）",
        "",
        f"- 题量：{report['total']}，通过 {report['passed']}（{report['pass_rate']:.0%}）",
        f"- 引用率：{report['citation_rate']:.0%}（需要引用的题里，回答含页码/文献名的比例）",
        f"- 平均工具调用：{report['avg_tools']} 次，平均耗时 {report['avg_seconds']}s",
        f"- 裁判：{report['judge_model']}（规则层 + 模型层，两层都过才算通过）",
        "",
        "## 分类通过率",
        "",
        "| 类别 | 通过 | 题数 |",
        "| --- | --- | --- |",
    ]
    for category, stat in report["by_category"].items():
        lines.append(f"| {category} | {stat['passed']} | {stat['total']} |")
    lines += ["", "## 逐题结果", ""]
    for item in report["results"]:
        flag = "✅" if item["passed"] else "❌"
        lines.append(f"### {flag} {item['id']}（{item['category']}，{item['tools']} 次工具，{item['seconds']}s）")
        lines.append("")
        lines.append(f"- 问题：{item['question']}")
        if item["rule_reasons"]:
            lines.append(f"- 规则层：{'；'.join(item['rule_reasons'])}")
        lines.append(f"- 裁判：{item['judge_reason']}")
        lines.append(f"- 工具链：{' → '.join(item['tool_names']) or '（未调工具）'}")
        lines.append("")
        lines.append("<details><summary>回答</summary>")
        lines.append("")
        lines.append(item["answer"].strip()[:1500])
        lines.append("")
        lines.append("</details>")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()