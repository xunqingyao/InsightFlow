"""SciFact（BEIR）外部评测：只测检索层的关键词通道。

为什么用外部数据集：自己的话问自己库，怎么测都是自证。
SciFact 的语料、问题、相关性判定全是别人标注的，我们只负责"灌进去、查出来、算分"。

    python -m evals.scifact load     # 建库 + 灌 5183 篇摘要
    python -m evals.scifact eval     # 跑 test 集，输出指标

数据：https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
参照基线：BEIR 论文里 BM25（flat）在 SciFact 上 nDCG@10 = 0.664，均值 0.613。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import psycopg

from evals.metrics import mrr, ndcg, recall
from recallgraph.dense import encode_version, make_embedder, search_dense, vector_stats
from recallgraph.ids import new_ulid
from recallgraph.lexical import BM25_B, BM25_K1, search_lexical, tokenize
from recallgraph.rerank import DashScopeReranker
from recallgraph.retriever import RERANK_HEAD
from recallgraph.storage import ensure_schema

DATA_DIR = Path(__file__).resolve().parent / "data" / "scifact"
SCIFACT_DSN = "postgresql://recallgraph:recallgraph@127.0.0.1:5434/recallgraph_scifact"
DOC_MAP = DATA_DIR / "doc_map.json"

PARSER = "beir-scifact-1.0"


# ── 灌库 ────────────────────────────────────────────────────────────────────


def load(dsn: str, *, reset: bool = False, with_vectors: bool = True) -> None:
    corpus = _read_jsonl(DATA_DIR / "corpus.jsonl")
    print(f"语料 {len(corpus)} 篇")

    with psycopg.connect(dsn) as conn:
        ensure_schema(conn, dsn)
        conn.commit()
        if conn.execute("SELECT count(*) FROM chunks").fetchone()[0]:
            if not reset:
                print("库里已有数据；重建倒排表请用 --reset")
                return
            conn.execute("TRUNCATE chunk_terms, chunks, document_versions, documents CASCADE")

        document_id, version_id = new_ulid(), new_ulid()
        raw = (DATA_DIR / "corpus.jsonl").read_bytes()
        conn.execute(
            "INSERT INTO documents (id, title, title_normalized, doc_type, external_ids) "
            "VALUES (%s, %s, %s, 'paper', %s)",
            (document_id, "SciFact (BEIR) corpus", "scifact (beir) corpus", json.dumps({"beir": "scifact"})),
        )
        conn.execute(
            "INSERT INTO document_versions (id, document_id, content_hash, file_path, page_count, parser_version, language) "
            "VALUES (%s, %s, %s, %s, 1, %s, 'en')",
            (version_id, document_id, hashlib.sha256(raw).hexdigest(), str(DATA_DIR), PARSER),
        )

        doc_map: dict[str, int] = {}
        with (
            conn.cursor() as cur,
            cur.copy(
                "COPY chunks (id, version_id, seq, page_start, page_end, section_path, text, context_header, char_start, char_end, token_count) "
                "FROM STDIN"
            ) as copy,
        ):
                for seq, entry in enumerate(corpus, start=1):
                    text = _compose(entry)
                    doc_map[entry["_id"]] = seq
                    copy.write_row(
                        (
                            f"{version_id}:{seq:04d}",
                            version_id,
                            seq,
                            1,
                            1,
                            [],
                            text,
                            entry.get("title", "")[:120] or "SciFact",
                            0,
                            len(text),
                            max(1, len(text) // 4),
                        )
                    )

        terms: Counter[tuple[str, str]] = Counter()
        for entry in corpus:
            chunk_id = f"{version_id}:{doc_map[entry['_id']]:04d}"
            for term in set(tokenize(_compose(entry))):
                terms[(chunk_id, term)] += 1
        with conn.cursor() as cur, cur.copy("COPY chunk_terms (chunk_id, term, tf) FROM STDIN") as copy:
            for (chunk_id, term), tf in terms.items():
                copy.write_row((chunk_id, term, tf))

        conn.commit()
        if with_vectors:
            written = encode_version(conn, version_id)
            conn.commit()
            print(f"向量已写入 {written} 条（{vector_stats(conn)}）")

    DOC_MAP.write_text(json.dumps(doc_map), encoding="utf-8")
    print(f"已灌入 1 个版本 / {len(corpus)} 个段落 / {len(terms)} 个词项行")


def _compose(entry: dict) -> str:
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or "").strip()
    return f"{title}. {text}" if title and not text.startswith(title) else text


# ── 评测 ────────────────────────────────────────────────────────────────────


def tune_adaptive(dsn: str, *, k1: float = 0.9, b: float = 0.4) -> None:
    """在 dev 上扫"覆盖度阈值"：低于阈值就关掉关键词通道。"""
    queries = {row["_id"]: row["text"] for row in _read_jsonl(DATA_DIR / "queries.jsonl")}
    qrels = _read_qrels(DATA_DIR / "qrels" / "train.tsv")
    doc_map = json.loads(DOC_MAP.read_text(encoding="utf-8"))
    seq_to_doc = {seq: doc for doc, seq in doc_map.items()}
    embedder = make_embedder()

    rows: list[tuple[str, float, float]] = []
    with psycopg.connect(dsn) as conn:
        for tau, soft in [(None, False), (0.0, False), (0.2, True), (0.4, True)]:
            ndcgs, recalls = [], []
            for query_id, relevant in qrels.items():
                text = queries.get(query_id)
                if not text:
                    continue
                chunk_ids = _search_chunks(
                    conn, text, top_k=100, channel="hybrid", k1=k1, b=b, embedder=embedder,
                    dense_weight=6.0, adaptive_tau=tau, adaptive_soft=soft,
                )
                ranked = [seq_to_doc[int(chunk_id.rsplit(":", 1)[1])] for chunk_id in chunk_ids]
                ndcgs.append(ndcg(ranked, relevant, 10))
                recalls.append(recall(ranked, relevant, 10))
            label = "fixed(6.0)" if tau is None else (f"soft{tau}" if soft else f"tau={tau}")
            rows.append((label, sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)))
            print(f"{label:<12} dev nDCG@10={rows[-1][1]:.4f} R@10={rows[-1][2]:.4f}", flush=True)

    best = max(rows, key=lambda row: row[1])
    print(f"\n最佳 dev 配置：{best[0]} → nDCG@10={best[1]:.4f}")


def tune_weights(dsn: str, *, k1: float = 0.9, b: float = 0.4) -> None:
    """在 train 分片上扫 dense 权重（混合通道专用）。

    权重必须和 k1/b 一样在 dev 上选：直接拿 test 上最好的权重去报数，
    等于把 test 当成了验证集。
    """
    queries = {row["_id"]: row["text"] for row in _read_jsonl(DATA_DIR / "queries.jsonl")}
    qrels = _read_qrels(DATA_DIR / "qrels" / "train.tsv")
    doc_map = json.loads(DOC_MAP.read_text(encoding="utf-8"))
    seq_to_doc = {seq: doc for doc, seq in doc_map.items()}

    embedder = make_embedder()
    rows: list[tuple[float, float, float]] = []
    with psycopg.connect(dsn) as conn:
        for weight in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            ndcgs, recalls = [], []
            for query_id, relevant in qrels.items():
                text = queries.get(query_id)
                if not text:
                    continue
                chunk_ids = _search_chunks(
                    conn,
                    text,
                    top_k=100,
                    channel="hybrid",
                    k1=k1,
                    b=b,
                    embedder=embedder,
                    dense_weight=weight,
                )
                ranked = [seq_to_doc[int(chunk_id.rsplit(":", 1)[1])] for chunk_id in chunk_ids]
                ndcgs.append(ndcg(ranked, relevant, 10))
                recalls.append(recall(ranked, relevant, 10))
            rows.append((weight, sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)))
            print(f"dense_weight={weight:<4} dev nDCG@10={rows[-1][1]:.4f} R@10={rows[-1][2]:.4f}", flush=True)

    best = max(rows, key=lambda row: row[1])
    print(f"\n最佳 dev 权重：dense_weight={best[0]} → nDCG@10={best[1]:.4f}")


def tune(dsn: str) -> None:
    """在 train 分片上扫 k1/b，选 dev 上最好的，再去 test 报一次。

    为什么必须这样做：拿 test 调参等于偷看答案，报出来的数没有任何意义。
    """
    queries = {row["_id"]: row["text"] for row in _read_jsonl(DATA_DIR / "queries.jsonl")}
    qrels = _read_qrels(DATA_DIR / "qrels" / "train.tsv")
    doc_map = json.loads(DOC_MAP.read_text(encoding="utf-8"))
    seq_to_doc = {seq: doc for doc, seq in doc_map.items()}

    from recallgraph.lexical import search_lexical

    rows: list[tuple[float, float, float]] = []
    with psycopg.connect(dsn) as conn:
        for k1 in (0.9, 1.2, 1.5):
            for b in (0.4, 0.6, 0.75):
                ndcgs = []
                for query_id, relevant in qrels.items():
                    text = queries.get(query_id)
                    if not text:
                        continue
                    hits = search_lexical(conn, text, top_k=100, k1=k1, b=b)
                    ranked = [seq_to_doc[int(hit.chunk_id.rsplit(":", 1)[1])] for hit in hits]
                    ndcgs.append(ndcg(ranked, relevant, 10))
                rows.append((k1, b, sum(ndcgs) / max(len(ndcgs), 1)))
                print(f"k1={k1:<4} b={b:<5} dev nDCG@10={rows[-1][2]:.4f}", flush=True)

    best = max(rows, key=lambda row: row[2])
    print(f"\n最佳 dev 配置：k1={best[0]}, b={best[1]} → nDCG@10={best[2]:.4f}")
    print(f"下一步：python -m evals.scifact eval --k1 {best[0]} --b {best[1]}")


def _coverage(text: str, hits) -> float:
    """查询词项有多大比例出现在关键词召回的前几条里。

    这是**查询时可见**的信号：不需要知道哪篇文档相关，
    只问"BM25 自己找回来的东西，到底覆盖了我的多少词"。
    覆盖低 → 它召回的多半只是碰巧共享常见词的噪声。
    """
    from recallgraph.lexical import tokenize

    wanted = set(tokenize(text))
    if not wanted:
        return 0.0
    seen: set[str] = set()
    for hit in hits[:10]:
        seen.update(tokenize(hit.text))
    return len(wanted & seen) / len(wanted)


def _search_chunks(
    conn: psycopg.Connection,
    text: str,
    *,
    top_k: int,
    channel: str,
    k1: float,
    b: float,
    embedder,
    dense_weight: float = 1.0,
    adaptive_tau: float | None = None,
    adaptive_soft: bool = False,
) -> list[str]:
    """按通道召回并做 RRF 融合，只返回 chunk_id 排名。

    RRF：每个通道按名次给 1/(60+rank) 分，多通道相加。不用加权和——
    权重需要标注集标定，而没有标注时任何权重都是拍的。
    """
    scores: dict[str, float] = {}
    lexical_hits = search_lexical(conn, text, top_k=top_k, k1=k1, b=b) if channel in ("lexical", "hybrid") else []
    dense_hits = search_dense(conn, text, top_k=top_k, embedder=embedder) if channel in ("dense", "hybrid") else []
    use_lexical = channel in ("lexical", "hybrid") and bool(lexical_hits)
    if adaptive_tau is not None and use_lexical and dense_hits:
        # 动态融合：两路 top-10 重合度低 = 它们对"相关"的判断不一致，
        # 而语料上向量更可信，于是让关键词通道退出，避免噪声挤掉正确结果。
        top_lexical = {hit.chunk_id for hit in lexical_hits[:10]}
        top_dense = {hit.chunk_id for hit in dense_hits[:10]}
        use_lexical = len(top_lexical & top_dense) / 10 >= adaptive_tau
    lexical_weight = 1.0
    if adaptive_soft and use_lexical and dense_hits:
        top_lexical = {hit.chunk_id for hit in lexical_hits[:10]}
        top_dense = {hit.chunk_id for hit in dense_hits[:10]}
        lexical_weight = len(top_lexical & top_dense) / 10
    if use_lexical:
        for rank, hit in enumerate(lexical_hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + lexical_weight / (60 + rank)
    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + dense_weight / (60 + rank)
    return [chunk_id for chunk_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def evaluate(
    dsn: str,
    top_k: int = 100,
    k1: float | None = None,
    b: float | None = None,
    channel: str = "lexical",
    dense_weight: float = 1.0,
    rerank: bool = False,
) -> dict:

    queries = {row["_id"]: row["text"] for row in _read_jsonl(DATA_DIR / "queries.jsonl")}
    qrels = _read_qrels(DATA_DIR / "qrels" / "test.tsv")
    doc_map = json.loads(DOC_MAP.read_text(encoding="utf-8"))
    seq_to_doc = {seq: doc for doc, seq in doc_map.items()}

    embedder = make_embedder() if channel in ("dense", "hybrid") else None
    reranker = DashScopeReranker() if rerank else None
    per_query: list[dict] = []
    with psycopg.connect(dsn) as conn:
        for query_id, relevant in qrels.items():
            text = queries.get(query_id)
            if not text:
                continue
            chunk_ids = _search_chunks(
                conn,
                text,
                top_k=top_k,
                channel=channel,
                k1=BM25_K1 if k1 is None else k1,
                b=BM25_B if b is None else b,
                embedder=embedder,
                dense_weight=dense_weight,
            )
            if reranker is not None and chunk_ids:
                pool = chunk_ids[:50]
                texts = {
                    row[0]: row[1]
                    for row in conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (pool,)).fetchall()
                }
                items = [(chunk_id, texts[chunk_id]) for chunk_id in pool if chunk_id in texts]
                rr = reranker.rerank(text, items)
                # 与产品一致：只重排头部 RERANK_HEAD 个，尾部顺序不动
                head = [chunk_id for chunk_id, _ in rr[:RERANK_HEAD]]
                tail = [chunk_id for chunk_id in chunk_ids if chunk_id not in set(head)]
                chunk_ids = head + tail
            ranked = [seq_to_doc[int(chunk_id.rsplit(":", 1)[1])] for chunk_id in chunk_ids]
            per_query.append(
                {
                    "query_id": query_id,
                    "n_relevant": len(relevant),
                    "recall_5": recall(ranked, relevant, 5),
                    "recall_10": recall(ranked, relevant, 10),
                    "recall_100": recall(ranked, relevant, 100),
                    "ndcg_10": ndcg(ranked, relevant, 10),
                    "mrr_10": mrr(ranked, relevant, 10),
                    "hits": len(ranked),
                }
            )

    n = len(per_query)
    summary = {
        "channel": channel,
        "queries": n,
        "corpus": len(doc_map),
        "ndcg_10": _mean(per_query, "ndcg_10"),
        "recall_5": _mean(per_query, "recall_5"),
        "recall_10": _mean(per_query, "recall_10"),
        "recall_100": _mean(per_query, "recall_100"),
        "mrr_10": _mean(per_query, "mrr_10"),
        "empty_result_rate": sum(1 for row in per_query if row["hits"] == 0) / max(n, 1),
    }
    return {"summary": summary, "per_query": per_query}


def report_buckets(
    dsn: str,
    *,
    k1: float | None,
    b: float | None,
    channel: str = "lexical",
    dense_weight: float = 1.0,
) -> None:
    """按"查询词有多少比例出现在相关文档里"分桶。

    这是给后续向量通道立的标尺：词面重合度低的那一桶，词面匹配天然无解，
    向量通道如果要证明自己有用，必须把这一桶的分数拉起来，而不是靠整体均值。
    """

    queries = {row["_id"]: row["text"] for row in _read_jsonl(DATA_DIR / "queries.jsonl")}
    qrels = _read_qrels(DATA_DIR / "qrels" / "test.tsv")
    doc_map = json.loads(DOC_MAP.read_text(encoding="utf-8"))
    seq_to_doc = {seq: doc for doc, seq in doc_map.items()}
    texts = {
        entry["_id"]: _compose(entry) for entry in _read_jsonl(DATA_DIR / "corpus.jsonl")
    }

    bucket_rows: dict[str, list[dict]] = {label: [] for label in BUCKET_LABELS}
    embedder = make_embedder() if channel in ("dense", "hybrid") else None
    with psycopg.connect(dsn) as conn:
        for query_id, relevant in qrels.items():
            text = queries.get(query_id)
            if not text:
                continue
            chunk_ids = _search_chunks(
                conn,
                text,
                top_k=100,
                channel=channel,
                k1=BM25_K1 if k1 is None else k1,
                b=BM25_B if b is None else b,
                embedder=embedder,
                dense_weight=dense_weight,
            )
            ranked = [seq_to_doc[int(chunk_id.rsplit(":", 1)[1])] for chunk_id in chunk_ids]
            query_terms = set(tokenize(text))
            overlap = max(
                (
                    len(query_terms & set(tokenize(texts[doc_id]))) / max(len(query_terms), 1)
                    for doc_id in relevant
                    if doc_id in texts
                ),
                default=0.0,
            )
            bucket_rows[_bucket_of(overlap)].append(
                {
                    "overlap": overlap,
                    "ndcg_10": ndcg(ranked, relevant, 10),
                    "recall_10": recall(ranked, relevant, 10),
                }
            )

    print(f"通道：{channel}")
    print(f"{'词面重合度':<12}{'查询数':>6}{'nDCG@10':>10}{'Recall@10':>12}{'平均重合':>10}")
    for label, rows in bucket_rows.items():
        if not rows:
            continue
        print(
            f"{label:<12}{len(rows):>6}{_mean(rows, 'ndcg_10'):>10.4f}"
            f"{_mean(rows, 'recall_10'):>12.4f}{_mean(rows, 'overlap'):>10.2f}"
        )
    total = sum(len(rows) for rows in bucket_rows.values())
    print(f"{'合计':<12}{total:>6}")


BUCKET_LABELS = ("0–25%", "25–50%", "50–75%", "75–100%")


def _bucket_of(overlap: float) -> str:
    if overlap < 0.25:
        return BUCKET_LABELS[0]
    if overlap < 0.5:
        return BUCKET_LABELS[1]
    if overlap < 0.75:
        return BUCKET_LABELS[2]
    return BUCKET_LABELS[3]


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


# ── IO ──────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("query-id"):
            handle.seek(0)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, corpus_id, score = parts[0], parts[1], float(parts[2])
            if score > 0:
                qrels.setdefault(query_id, set()).add(corpus_id)
    return qrels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.scifact")
    parser.add_argument("command", choices=["load", "tune", "tune-weights", "tune-adaptive", "eval", "buckets", "encode"])
    parser.add_argument("--dsn", default=SCIFACT_DSN)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--channel", default="lexical", choices=["lexical", "dense", "hybrid"])
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rerank", action="store_true", help="融合后取前 50 交给百炼重排")
    parser.add_argument("--no-vectors", action="store_true", help="灌语料时跳过向量编码")
    parser.add_argument("--b", type=float, default=None)
    args = parser.parse_args(argv)

    if not (DATA_DIR / "corpus.jsonl").exists():
        print(f"缺少数据：{DATA_DIR}（先下载 scifact.zip 解压）", file=sys.stderr)
        return 1

    if args.command == "load":
        load(args.dsn, reset=args.reset, with_vectors=not args.no_vectors)
        return 0

    if args.command == "tune-adaptive":
        tune_adaptive(args.dsn)
        return 0

    if args.command == "tune-weights":
        tune_weights(args.dsn)
        return 0

    if args.command == "tune":
        tune(args.dsn)
        return 0

    if args.command == "buckets":
        report_buckets(args.dsn, k1=args.k1, b=args.b, channel=args.channel, dense_weight=args.dense_weight)
        return 0

    if args.command == "encode":
        with psycopg.connect(args.dsn) as conn:
            ensure_schema(conn, args.dsn)
            conn.commit()
            rows = conn.execute("SELECT DISTINCT version_id FROM chunks").fetchall()
            for (version_id,) in rows:
                written = encode_version(conn, version_id)
                conn.commit()
                print(f"版本 {version_id}: {written} 条向量")
            print(f"向量统计：{vector_stats(conn)}")
        return 0

    result = evaluate(
        args.dsn,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        channel=args.channel,
        dense_weight=args.dense_weight,
        rerank=args.rerank,
    )
    summary = result["summary"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())