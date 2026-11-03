"""基线对照：同一批题、同一个裁判，只改"检索和作答方式"，看 Agent 循环值多少。

三档：
- B0 普通 RAG：混合检索 top-8 段落 → 一次模型调用作答（没有断言、没有图、没有重排、没有工具循环）；
- B1 证据包：检索 → 断言 + 引文（我们的检索层，但一次调用、不调工具）；
- B2 Agent：模型自己调工具（数字取自 evals/reports/agent-*-v3.json）。

用法：
    python -m evals.agent_baseline --mode chunks
    python -m evals.agent_baseline --mode claims
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.agent_cases import CASES
from evals.agent_eval import CITATION, judge, rule_check
from recallgraph import agent
from recallgraph.env import load_dotenv
from recallgraph.retrieval import RetrievalQuery
from recallgraph.retriever import SqlRetriever
from recallgraph.storage import connect

ANSWER_PROMPT = """你是一个论文知识库助手。只根据下面给出的材料回答用户问题。

要求：
1. 材料里没有的内容，直接说"知识库里没有"，不要用自己的知识补。
2. 关键结论后面注明出处：《标题》第 N 页。
3. 中文回答，简洁，先说结论。"""


def build_context(conn, question: str, mode: str) -> tuple[str, int]:
    retriever = SqlRetriever(conn, reranker=None)
    if mode == "chunks":
        result = retriever.retrieve(RetrievalQuery(text=question, expand_hops=0, top_k=8, max_chunks=8))
        ids = [hit.chunk_id for hit in result.chunks][:8]
        if not ids:
            return "", 0
        rows = conn.execute(
            """
            SELECT c.text, c.page_start, d.title
            FROM chunks c JOIN document_versions v ON v.id = c.version_id
            JOIN documents d ON d.id = v.document_id
            WHERE c.id = ANY(%s)
            """,
            (ids,),
        ).fetchall()
        blocks = [
            f"[材料{i}] 《{title}》第 {page} 页：\n{text[:900]}"
            for i, (text, page, title) in enumerate(rows, 1)
        ]
        return "\n\n".join(blocks), len(blocks)

    result = retriever.retrieve(RetrievalQuery(text=question, expand_hops=1, top_k=8, max_chunks=8))
    ids = [hit.chunk_id for hit in result.chunks][:8]
    texts = {}
    if ids:
        rows = conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()
        texts = {row[0]: row[1] for row in rows}
    blocks = []
    used_chunks: set[str] = set()
    for claim in result.claims[:8]:
        citation = claim.citations[0] if claim.citations else None
        source = f"《{citation.document_title}》第 {citation.page} 页" if citation else ""
        excerpt = texts.get(citation.chunk_id, "")[:400] if citation else ""
        if citation:
            used_chunks.add(citation.chunk_id)
        blocks.append(f"[材料{len(blocks) + 1}] 断言：{claim.claim.claim_text}（{source}）\n原文：{excerpt}")
    # 断言没覆盖到的段落也要给——"断言是桥，不是准入条件"。
    for chunk_id, text in texts.items():
        if len(blocks) >= 8:
            break
        if chunk_id in used_chunks:
            continue
        blocks.append(f"[材料{len(blocks) + 1}] 原文片段：{text[:600]}")
    return "\n\n".join(blocks), len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chunks", "claims"], required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY 未设置")

    results: list[dict[str, Any]] = []
    with connect() as conn:
        for index, case in enumerate(CASES, 1):
            started = time.perf_counter()
            context, used = build_context(conn, case["question"], args.mode)
            if not context:
                answer = "知识库里没有找到相关材料。"
            else:
                body = agent.chat(
                    {
                        "model": "qwen-plus",
                        "messages": [
                            {"role": "system", "content": ANSWER_PROMPT},
                            {"role": "user", "content": f"材料：\n{context}\n\n问题：{case['question']}"},
                        ],
                        "temperature": 0.2,
                    },
                    api_key=api_key,
                )
                answer = body["choices"][0]["message"]["content"]
            elapsed = time.perf_counter() - started
            rule_ok, rule_reasons = rule_check(case, answer)
            try:
                judge_ok, judge_reason = judge(case, answer, api_key=api_key)
            except (RuntimeError, KeyError) as exc:
                judge_ok, judge_reason = False, f"裁判失败：{exc}"
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "passed": rule_ok and judge_ok,
                    "rule_reasons": rule_reasons,
                    "judge_reason": judge_reason,
                    "materials": used,
                    "seconds": round(elapsed, 1),
                    "answer": answer,
                }
            )
            print(f"[{index}/{len(CASES)}] {'PASS' if results[-1]['passed'] else 'FAIL'} {case['id']:<5} {elapsed:5.1f}s {judge_reason[:36]}")

    passed = sum(1 for item in results if item["passed"])
    citations = sum(1 for item in results if CITATION.search(item["answer"]))
    summary = {
        "mode": args.mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 3),
        "citation_rate": round(citations / len(results), 3),
        "avg_seconds": round(sum(item["seconds"] for item in results) / len(results), 1),
        "results": results,
    }
    out = Path(args.out or f"evals/reports/baseline-{args.mode}-2026-10-30.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 基线对照：{args.mode}（{summary['generated_at']}）",
        "",
        f"- 通过 {passed}/{len(results)}（{summary['pass_rate']:.0%}），引用率 {summary['citation_rate']:.0%}，平均 {summary['avg_seconds']}s",
        "",
        "| 题 | 类别 | 结果 | 裁判理由 |",
        "| --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['id']} | {item['category']} | {'✅' if item['passed'] else '❌'} | {item['judge_reason'][:80].replace('|', '/')} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{args.mode}: {passed}/{len(results)}（{summary['pass_rate']:.0%}），引用率 {summary['citation_rate']:.0%}")
    print(f"报告：{out}")


if __name__ == "__main__":
    main()