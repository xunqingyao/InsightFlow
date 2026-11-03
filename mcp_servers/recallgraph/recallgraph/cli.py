"""命令行入口：初始化数据库、把文件灌进库、看库里有什么。

    recallgraph init-db
    recallgraph ingest papers/*.pdf notes/*.md
    recallgraph stats
    recallgraph show <version_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from recallgraph.dense import encode_version, make_embedder, vector_stats
from recallgraph.extract import get_extractor
from recallgraph.extract.writer import extract_version
from recallgraph.ingest.pipeline import ingest
from recallgraph.lexical import index_stats, index_version, search_lexical, tokenize
from recallgraph.retrieval import RetrievalQuery
from recallgraph.retriever import SqlRetriever
from recallgraph.storage import connect, counts, ensure_schema, load_ingest, schema_ready
from recallgraph.storage.db import SQL_DIR


def _cmd_init_db(_: argparse.Namespace) -> int:
    with connect() as conn:
        applied = ensure_schema(conn)
        total = len(list(SQL_DIR.glob("*.sql")))
        print(f"数据库就绪，已应用 {len(applied)} 个新脚本（共 {total} 个）")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    with connect() as conn:
        ensure_schema(conn)
        totals = {"new": 0, "deduped": 0, "chunks": 0, "warnings": 0}
        for raw in args.paths:
            path = Path(raw)
            if not path.exists():
                print(f"[跳过] 不存在：{path}", file=sys.stderr)
                continue
            result = ingest(path, stop_at_references=not args.keep_references)
            outcome = load_ingest(conn, result)
            conn.commit()
            mark = "重复" if outcome.deduped else "新增"
            totals["deduped" if outcome.deduped else "new"] += 1
            totals["chunks"] += outcome.chunks_written
            totals["warnings"] += len(result.warnings)
            print(
                f"[{mark}] {result.document.title[:48]:<48} "
                f"页={result.version.page_count:>3} 块={len(result.chunks):>3} "
                f"版本={outcome.version_id} 警告={len(result.warnings)}"
            )
        print(
            f"\n完成：新增 {totals['new']} 篇，重复 {totals['deduped']} 篇，"
            f"写入 {totals['chunks']} 块，警告 {totals['warnings']} 条"
        )
    return 0


def _cmd_stats(_: argparse.Namespace) -> int:
    with connect() as conn:
        if not schema_ready(conn):
            print("表还没建，先跑 `recallgraph init-db`", file=sys.stderr)
            return 1
        for table, value in counts(conn).items():
            print(f"{table:<18}{value:>8}")
        lexical = index_stats(conn)
        print(f"{'词项行数':<16}{int(lexical['term_rows']):>8}")
        print(f"{'词表大小':<16}{int(lexical['vocab']):>8}")
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    with connect() as conn:
        ensure_schema(conn)
        version_ids = [row[0] for row in conn.execute("SELECT id FROM document_versions ORDER BY ingested_at")]
        if args.version_id:
            version_ids = [v for v in version_ids if v == args.version_id]
        total_terms = 0
        for version_id in version_ids:
            _, terms = index_version(conn, version_id)
            total_terms += terms
            print(f"[索引] {version_id} 词项 {terms}")
        conn.commit()
        print(f"\n完成：{len(version_ids)} 个版本，共 {total_terms} 个词项行")
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    model = args.model or os.environ.get("RECALLGRAPH_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
    with connect() as conn:
        if not schema_ready(conn):
            print("表还没建，先跑 `recallgraph init-db`", file=sys.stderr)
            return 1
        if args.version_id:
            versions = [args.version_id]
        else:
            versions = [row[0] for row in conn.execute("SELECT DISTINCT version_id FROM chunks ORDER BY version_id")]
        embedder = make_embedder(model)
        total = 0
        for version_id in versions:
            written = encode_version(conn, version_id, embedder=embedder, model=model)
            total += written
            print(f"{version_id[:8]}  {written} 块")
        conn.commit()
        print(f"模型 {model}：写入 {total} 条；库里现有 {vector_stats(conn)}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    with connect() as conn:
        if not schema_ready(conn):
            print("表还没建，先跑 `recallgraph init-db`", file=sys.stderr)
            return 1
        terms = tokenize(args.query)
        hits = search_lexical(conn, args.query, top_k=args.top_k)
        if not hits:
            print(f"没有命中。分词结果：{terms}", file=sys.stderr)
            return 1
        print(f"查询：{args.query}")
        print(f"分词：{' '.join(terms)}\n")
        for rank, hit in enumerate(hits, start=1):
            span = f"p{hit.page_start}" if hit.page_start == hit.page_end else f"p{hit.page_start}-{hit.page_end}"
            section = " > ".join(hit.section_path) or "(无节)"
            print(f"{rank}. [{hit.score:6.2f}] {hit.title[:44]} · {span} · {section}")
            print(f"   {_snippet(hit.text, terms, 220)}\n")
        return 0


def _snippet(text: str, terms: list[str], limit: int) -> str:
    """裁出包含查询词的片段，让人一眼看到命中处。"""
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if term in lowered]
    start = max(0, (min(positions) if positions else 0) - 60)
    piece = " ".join(text[start : start + limit].split())
    return ("…" if start else "") + piece + ("…" if start + limit < len(text) else "")


def _cmd_extract(args: argparse.Namespace) -> int:
    with connect() as conn:
        if not schema_ready(conn):
            print("表还没建，先跑 `recallgraph init-db`", file=sys.stderr)
            return 1
        ensure_schema(conn)
        conn.commit()

        if args.version_id:
            targets = [(args.version_id, "")]
        else:
            rows = conn.execute(
                """
                SELECT v.id, d.title
                FROM document_versions v JOIN documents d ON d.id = v.document_id
                ORDER BY d.title
                """
            ).fetchall()
            targets = [(row[0], row[1]) for row in rows]
            if not targets:
                print("库里还没有文档，先跑 `recallgraph ingest <文件>`", file=sys.stderr)
                return 1

        extractor = get_extractor(args.extractor)
        for version_id, title in targets:
            stats = extract_version(conn, version_id, extractor, force=args.force, retire_old=args.retire_old)
            conn.commit()
            label = title[:40] if title else version_id
            print(f"{label:<42} {stats.summary()}")
            for error in stats.errors[:5]:
                print(f"    ! {error}")
            if len(stats.errors) > 5:
                print(f"    … 另有 {len(stats.errors) - 5} 条被拒")

        if args.show:
            _print_sample(conn, args.show)
        return 0


def _print_sample(conn, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT c.claim_text, c.confidence, c.predicate, e.page, e.quote, d.title
        FROM claims c
        JOIN evidence e ON e.claim_id = c.id
        JOIN document_versions v ON v.id = c.version_id
        JOIN documents d ON d.id = v.document_id
        WHERE c.status = 'active'
        ORDER BY c.created_at DESC, c.id
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return
    print("\n抽查（最新写入的断言）：")
    for claim_text, confidence, predicate, page, quote, title in rows:
        print(f"  · [{confidence}] {claim_text}  ({title[:28]} p{page}, {predicate})")
        print(f"    证据：{_one_line(quote, 120)}")


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _cmd_ask(args: argparse.Namespace) -> int:
    query = RetrievalQuery(
        text=args.question,
        max_chunks=args.max_chunks,
        max_tokens=args.max_tokens,
        expand_hops=args.expand_hops,
    )
    with connect() as conn:
        if not schema_ready(conn):
            print("表还没建，先跑 `recallgraph init-db`", file=sys.stderr)
            return 1
        result = SqlRetriever(conn, embedding_model=args.embedding_model, rerank=not args.no_rerank).retrieve(query)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0 if result.status != "empty" else 1

    print(f"问题：{query.text}\n")
    if result.status == "empty":
        print("没有找到相关断言。")
        for warning in result.warnings:
            print(f"  · {warning}")
        return 1

    stats = result.stats
    print(
        f"命中 {stats.claims_total} 条断言 · 证据 {stats.chunks_total} 块 · "
        f"约 {stats.tokens_used} tokens · {stats.elapsed_ms:.0f} ms\n"
    )

    for hit in result.claims:
        citation = hit.citations[0]
        section = " › ".join(citation.section_path) or "(无节)"
        channels = " + ".join(hit.channels)
        print(f"{hit.rank:>2}. [{hit.claim.confidence.value}] {hit.claim.claim_text}")
        print(f"    出处：{citation.document_title[:40]} p{citation.page} · {section}")
        print(f"    引文：{_one_line(citation.quote, 160)}")
        print(f"    通道：{channels}（rrf {hit.rrf:.4f}，跳数 {hit.hop}，证据 {len(hit.citations)} 条）")
        print()

    if result.chunks:
        print("命中的段落（按融合名次，原文走 read_evidence 取全块）：")
        for chunk in result.chunks[:3]:
            section = " › ".join(chunk.section_path) or "(无节)"
            print(f"  p{chunk.page} [{'+'.join(chunk.channels)}] {section}：{_one_line(chunk.text, 90)}")
        print()

    stages = " → ".join(f"{step.stage} {step.candidates_out}" for step in result.trace)
    print(f"轨迹：{stages}")
    for warning in result.warnings:
        print(f"注意：{warning}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT seq, page_start, page_end, section_path, context_header, text
            FROM chunks WHERE version_id = %s ORDER BY seq LIMIT %s
            """,
            (args.version_id, args.limit),
        ).fetchall()
        if not rows:
            print("没有找到这个版本的段落", file=sys.stderr)
            return 1
        for seq, page_start, page_end, section_path, header, text in rows:
            span = f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}"
            print(f"#{seq:04d} {span} {' > '.join(section_path) or '(无节)'}")
            print(f"  头: {header}")
            print(f"  文: {text[:160]}{'…' if len(text) > 160 else ''}\n")
    return 0


def _cmd_memory_list(args: argparse.Namespace) -> int:
    from recallgraph import memory

    with connect() as conn:
        items = memory.list_keys(conn)
    if not items:
        print("（还没有记忆）")
        return 0
    for item in items:
        when = item["valid_from"].astimezone().strftime("%m-%d %H:%M")
        print(f"{item['key']:<24} {item['value']}  [{when}]")
    return 0


def _cmd_memory_set(args: argparse.Namespace) -> int:
    from recallgraph import memory

    with connect() as conn:
        proposal = memory.propose(
            conn, key=args.key, value=args.value, reason=args.reason, namespace=args.namespace
        )
        if proposal["operation"] != "noop" and proposal["previous_value"]:
            print(f"将覆盖：{proposal['previous_value']}")
        print(memory.commit(conn, proposal, namespace=args.namespace))
    return 0


def _cmd_memory_history(args: argparse.Namespace) -> int:
    from recallgraph import memory

    with connect() as conn:
        items = memory.history(conn, args.key)
    if not items:
        print("（没有这条记忆）")
        return 0
    for item in items:
        when = item["recorded_at"].astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"{when}  {item['status']:<11} {item['value']}")
    return 0


def _cmd_memory_forget(args: argparse.Namespace) -> int:
    from recallgraph import memory

    with connect() as conn:
        print(memory.retract(conn, args.key, reason=args.reason))
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    from recallgraph import agent as agent_mod

    with connect() as conn:
        outcome = agent_mod.run(conn, args.question, model=args.model, max_steps=args.max_steps, trace=args.trace)
    if args.json:
        print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
        return 0
    print(outcome["answer"])
    tools = [step.get("tool") for step in outcome["steps"]]
    print(f"\n[工具调用] {len(tools)} 次：{' → '.join(tools) if tools else '无'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recallgraph", description="RecallGraph 命令行")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-db", help="建表（幂等）")
    init.set_defaults(func=_cmd_init_db)

    ing = sub.add_parser("ingest", help="解析文件并入库")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--keep-references", action="store_true", help="保留参考文献之后的段落")
    ing.set_defaults(func=_cmd_ingest)

    stat = sub.add_parser("stats", help="各表行数与词表规模")
    stat.set_defaults(func=_cmd_stats)

    show = sub.add_parser("show", help="看某个版本的前几块")
    show.add_argument("version_id")
    show.add_argument("--limit", type=int, default=3)
    show.set_defaults(func=_cmd_show)

    reindex = sub.add_parser("reindex", help="重建某个/全部版本的倒排表")
    reindex.add_argument("version_id", nargs="?")
    reindex.set_defaults(func=_cmd_reindex)

    ext = sub.add_parser("extract", help="从段落抽取断言与证据")
    ext.add_argument("version_id", nargs="?", help="不给则处理库里全部版本")
    ext.add_argument("--extractor", default="stub", help="stub（规则）或 llm（百炼 qwen）")
    ext.add_argument("--retire-old", action="store_true", help="换抽取器后把旧抽取器的断言标为撤回")
    ext.add_argument("--force", action="store_true", help="重跑：旧断言标为 retracted")
    ext.add_argument("--show", type=int, default=3, help="跑完后抽查几条（0 关闭）")
    ext.set_defaults(func=_cmd_extract)

    ask = sub.add_parser("ask", help="检索断言证据包（关键词 + 图扩展）")
    ask.add_argument("question")
    ask.add_argument("--max-chunks", type=int, default=8)
    ask.add_argument("--max-tokens", type=int, default=6000)
    ask.add_argument("--expand-hops", type=int, default=1, choices=[0, 1])
    ask.add_argument("--json", action="store_true", help="输出完整 JSON（给评测/调试用）")
    ask.add_argument("--embedding-model", default=None, help="向量模型（默认取库里最多的那个）")
    ask.add_argument("--no-rerank", action="store_true", help="关闭百炼 gte-rerank 重排（默认开，仅重排头部 10 个）")
    ask.set_defaults(func=_cmd_ask)

    enc = sub.add_parser("encode", help="为一个/全部版本生成向量")
    enc.add_argument("version_id", nargs="?")
    enc.add_argument("--model", default=None, help="默认 BAAI/bge-small-en-v1.5，可用 BAAI/bge-small-zh-v1.5")
    enc.set_defaults(func=_cmd_encode)

    mem = sub.add_parser("memory", help="记忆：列出/写入/查看历史/作废")
    mem_sub = mem.add_subparsers(dest="memory_command", required=True)
    mem_sub.add_parser("list", help="当前有效的记忆").set_defaults(func=_cmd_memory_list)
    mem_set = mem_sub.add_parser("set", help="写入或更新一条记忆（确认提案用这个）")
    mem_set.add_argument("namespace", nargs="?", default="user")
    mem_set.add_argument("key")
    mem_set.add_argument("value")
    mem_set.add_argument("--reason", default="")
    mem_set.set_defaults(func=_cmd_memory_set)
    mem_show = mem_sub.add_parser("history", help="一条记忆的完整历史")
    mem_show.add_argument("key")
    mem_show.set_defaults(func=_cmd_memory_history)
    mem_forget = mem_sub.add_parser("forget", help="作废当前值（保留历史）")
    mem_forget.add_argument("key")
    mem_forget.add_argument("--reason", default="")
    mem_forget.set_defaults(func=_cmd_memory_forget)

    agent = sub.add_parser("agent", help="Agent 对话：模型自己决定调哪些工具")
    agent.add_argument("question")
    agent.add_argument("--model", default="qwen-plus", help="百炼模型名")
    agent.add_argument("--max-steps", type=int, default=8)
    agent.add_argument("--trace", action="store_true", help="打印每次工具调用")
    agent.add_argument("--json", action="store_true", help="输出完整 JSON")
    agent.set_defaults(func=_cmd_agent)

    search = sub.add_parser("search", help="关键词检索（BM25）")
    search.add_argument("query")
    search.add_argument("-k", "--top-k", type=int, default=5)
    search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())