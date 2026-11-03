"""中文/跨语言检索体检：跑人工标注的 gold（evals/labels_zh.json）。

指标用 evals/metrics.py，别在这里另写一份——上次 zh_tune 自己抄了个错公式，
第一名只给 0.63 分，把调参结论带偏了。

分两层看：
- 候选召回 Recall@50：正确的段落在不在候选池里（通道的问题）
- 最终排序 nDCG@10：在候选池里但排不上去（排序/融合的问题）

用法：python -m evals.zh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.metrics import ndcg, recall
from recallgraph.dense import make_embedder, search_dense
from recallgraph.lexical import search_lexical
from recallgraph.retrieval import RetrievalQuery
from recallgraph.retriever import DENSE_WEIGHT, RRF_K, SqlRetriever, resolve_embedding_model
from recallgraph.storage import connect

LABELS = Path(__file__).with_name("labels_zh.json")
TOP_K = 50
EN_PREFIXES = ("01M5YXRH",)  # 两篇英文论文的版本 ULID 前缀


def load_labels(path: Path = LABELS) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def is_cross_language(item: dict) -> bool:
    return any(key.startswith(EN_PREFIXES) for key in item["relevant"])


def fuse(lexical: list[str], dense: list[str], weight: float = DENSE_WEIGHT) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank)
    for rank, chunk_id in enumerate(lexical, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def main() -> int:
    rerank = "--rerank" in sys.argv
    items = load_labels()
    with connect() as conn:
        model = resolve_embedding_model(conn, [], explicit=None)
        print(f"模型：{model}｜题数：{len(items)}（跨语言 {sum(map(is_cross_language, items))}）")
        embedder = make_embedder(model)
        retriever = SqlRetriever(conn, embedding_model=model)  # 产品默认：开重排
        reranked = SqlRetriever(conn, embedding_model=model, rerank=True) if rerank else None

        rows = []
        for item in items:
            gold = {key: float(value) for key, value in item["relevant"].items()}
            query = item["query"]
            lex = [hit.chunk_id for hit in search_lexical(conn, query, top_k=TOP_K)]
            den = [hit.chunk_id for hit in search_dense(conn, query, top_k=TOP_K, embedder=embedder, model=model)]
            fused = fuse(lex, den)
            product = [c.chunk_id for c in retriever.retrieve(RetrievalQuery(text=query, max_chunks=10)).chunks]
            runs = {"keyword": lex, "dense": den, "hybrid": fused, "product": product}
            if reranked is not None:
                runs["rerank"] = [c.chunk_id for c in reranked.retrieve(RetrievalQuery(text=query, max_chunks=10)).chunks]
            rows.append((item, gold, runs))

        header = f"{'通道':<8}{'候选召回@50':>12}{'nDCG@10':>10}"
        for label, group in (
            ("全部", rows),
            ("中文题", [r for r in rows if not is_cross_language(r[0])]),
            ("跨语言", [r for r in rows if is_cross_language(r[0])]),
        ):
            print(f"\n[{label}] {len(group)} 题\n{header}")
            for channel in ("keyword", "dense", "hybrid", "product", *(("rerank",) if rerank else ())):
                cand = sum(recall(r[2][channel][:TOP_K], r[1], TOP_K) for r in group) / len(group)
                final = sum(ndcg(r[2][channel], r[1], 10) for r in group) / len(group)
                print(f"{channel:<8}{cand:>12.3f}{final:>10.3f}")

        print("\n低分题（hybrid nDCG@10 < 0.2）：")
        for item, gold, runs in rows:
            score = ndcg(runs["hybrid"], gold, 10)
            if score < 0.2:
                top = runs["hybrid"][:2]
                print(f"  {score:.2f} {item['query']}｜gold={len(gold)}｜top2={top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())