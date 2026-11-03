"""在人工标注集上扫 dense_weight：dev 选参数，test 只跑一次。

题目来自 evals/labels_zh.json；split 按 query 的哈希固定，保证可复现。
指标统一走 evals/metrics.py。

用法：python -m evals.zh_tune
"""

from __future__ import annotations

import hashlib

from evals.metrics import ndcg
from evals.zh import fuse, is_cross_language, load_labels
from recallgraph.dense import make_embedder, search_dense
from recallgraph.lexical import search_lexical
from recallgraph.storage import connect

WEIGHTS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
TOP_K = 50


def split_key(query: str) -> float:
    return int(hashlib.sha256(query.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> int:
    items = load_labels()
    dev = [item for item in items if split_key(item["query"]) < 0.6]
    test = [item for item in items if split_key(item["query"]) >= 0.6]
    print(f"dev {len(dev)} 题 / test {len(test)} 题（跨语言 {sum(map(is_cross_language, items))}）")

    with connect() as conn:
        embedder = make_embedder("text-embedding-v3")
        lexical = {item["query"]: [h.chunk_id for h in search_lexical(conn, item["query"], top_k=TOP_K)] for item in items}
        dense = {
            item["query"]: [
                h.chunk_id
                for h in search_dense(conn, item["query"], top_k=TOP_K, embedder=embedder, model="text-embedding-v3")
            ]
            for item in items
        }

        def score(group: list[dict], weight: float) -> float:
            values = [
                ndcg(
                    fuse(lexical[item["query"]], dense[item["query"]], weight),
                    {k: float(v) for k, v in item["relevant"].items()},
                    10,
                )
                for item in group
            ]
            return sum(values) / len(values)

        rows = [(weight, score(dev, weight)) for weight in WEIGHTS]
        for weight, value in rows:
            print(f"  weight={weight:<5} dev nDCG@10={value:.4f}")
        best = max(rows, key=lambda row: row[1])
        print(f"\n最佳 {best[0]}（dev {best[1]:.4f}）→ test 只跑一次：{score(test, best[0]):.4f}")
        print(f"对照：默认 6.0 → test {score(test, 6.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())