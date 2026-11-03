"""检索指标的**唯一实现**。

以前三个脚本各抄了一份，其中 evals/zh_tune.py 那份把序号写成 1 起、
折损又用 log2(i+2)，第一名也只有 1/log2(3)=0.6309 分。中文调参的
0.288/0.294 都是这个错误公式下的数，必须重算。

这里的约定：
- 序号从 1 开始，折损 log2(rank + 1) → 第一名折损为 1。
- gold 可以是 set（相关性=1）或 dict{id: 相关性等级}（0/1/2）。
- 缺 gold 一律返回 0，不抛异常，方便评估脚本批量跑。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

Relevance = set[str] | Mapping[str, float]


def gain(item: str, gold: Relevance) -> float:
    if isinstance(gold, Mapping):
        return float(gold.get(item, 0.0))
    return 1.0 if item in gold else 0.0


def dcg(ranked: Sequence[str], gold: Relevance, k: int = 10) -> float:
    return sum(gain(item, gold) / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], start=1))


def ideal_dcg(gold: Relevance, k: int = 10) -> float:
    grades = sorted((float(v) for v in gold.values()), reverse=True) if isinstance(gold, Mapping) else [1.0] * len(gold)
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades[:k], start=1))


def ndcg(ranked: Iterable[str], gold: Relevance, k: int = 10) -> float:
    if not gold:
        return 0.0
    ideal = ideal_dcg(gold, k)
    return dcg(list(ranked), gold, k) / ideal if ideal else 0.0


def recall(ranked: Iterable[str], gold: Relevance, k: int = 10) -> float:
    """命中数 / gold 数。gold 若给了等级，只把等级 >= 1 的算作相关。"""
    relevant = {key for key, value in gold.items() if value >= 1} if isinstance(gold, Mapping) else set(gold)
    if not relevant:
        return 0.0
    return len(set(list(ranked)[:k]) & relevant) / len(relevant)


def mrr(ranked: Iterable[str], gold: Relevance, k: int = 10) -> float:
    for rank, item in enumerate(list(ranked)[:k], start=1):
        if gain(item, gold) > 0:
            return 1.0 / rank
    return 0.0


__all__ = ["dcg", "gain", "ideal_dcg", "mrr", "ndcg", "recall"]