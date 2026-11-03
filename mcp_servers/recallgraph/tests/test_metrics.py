"""指标测试：这里锁死的是"第一名必须得 1.0"这类不能再退化的性质。"""

from __future__ import annotations

import math

import pytest

from evals.metrics import dcg, ideal_dcg, mrr, ndcg, recall


def test_recall_counts_only_relevant_in_top_k() -> None:
    assert recall(["a", "b", "c", "d"], {"b", "d"}, 2) == 0.5
    assert recall(["a", "b", "c", "d"], {"b", "d"}, 4) == 1.0


def test_nothing_relevant_retrieved_is_zero() -> None:
    assert ndcg(["x", "y"], {"a"}) == 0.0


def test_single_relevant_at_rank_one_scores_one() -> None:
    assert ndcg(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_single_relevant_at_rank_two_matches_log_discount() -> None:
    assert ndcg(["b", "a"], {"a"}) == pytest.approx(1 / math.log2(3))


def test_order_matters() -> None:
    gold = {"a", "b"}
    early = ndcg(["a", "b", "z"], gold)
    late = ndcg(["z", "a", "b"], gold)
    assert early > late


def test_graded_relevance_prefers_better_grade() -> None:
    gold = {"a": 2.0, "b": 1.0}
    assert ndcg(["a", "b"], gold) > ndcg(["b", "a"], gold)


def test_ideal_is_upper_bound() -> None:
    gold = {str(i) for i in range(5)}
    assert ndcg(list(gold), gold) == pytest.approx(1.0)
    assert ndcg(list(gold), gold, k=3) == pytest.approx(1.0)
    assert ideal_dcg(gold, k=3) > ideal_dcg(gold, k=1)


def test_empty_gold_is_zero_not_crash() -> None:
    assert ndcg(["a"], set()) == 0.0
    assert recall(["a"], set()) == 0.0
    assert mrr(["a"], set()) == 0.0


def test_recall_and_mrr() -> None:
    assert recall(["x", "a", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)
    assert mrr(["x", "a"], {"a"}) == pytest.approx(0.5)


def test_k_truncates() -> None:
    assert dcg(["a"], {"a"}, k=0) == 0.0
    assert recall(["z", "a"], {"a"}, k=1) == 0.0