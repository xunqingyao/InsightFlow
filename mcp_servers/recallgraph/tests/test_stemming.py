"""词干还原的回归测试。词干是索引层的地基，错一个词就是全库召回错。"""

from __future__ import annotations

import pytest

from recallgraph.stemming import stem


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Porter 论文里的标准例子
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("ties", "ti"),
        ("caress", "caress"),
        ("cats", "cat"),
        ("feed", "feed"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("motoring", "motor"),
        ("sing", "sing"),
        ("conflated", "conflat"),
        ("troubled", "troubl"),
        ("sized", "size"),
        ("hopping", "hop"),
        ("tanned", "tan"),
        ("falling", "fall"),
        ("filing", "file"),
        ("happy", "happi"),
        ("relational", "relat"),
        ("conditional", "condit"),
        ("rational", "ration"),
        ("digitizer", "digit"),
        ("vietnamization", "vietnam"),
        ("predication", "predic"),
        ("operator", "oper"),
        ("feudalism", "feudal"),
        ("decisiveness", "decis"),
        ("triplicate", "triplic"),
        ("formalize", "formal"),
        ("electrical", "electr"),
        ("goodness", "good"),
        ("adjustment", "adjust"),
        ("adoption", "adopt"),
        ("effective", "effect"),
        ("bowdlerize", "bowdler"),
        ("probate", "probat"),
        ("controll", "control"),
        # 本项目真正关心的形态变化
        ("retrieval", "retriev"),
        ("retrieve", "retriev"),
        ("retrieving", "retriev"),
        ("documents", "document"),
    ],
)
def test_stem_matches_porter(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_short_and_non_alpha_words_pass_through() -> None:
    assert stem("is") == "is"
    assert stem("rag") == "rag"
    assert stem("bm25") == "bm25"
    assert stem("0-dimensional") == "0-dimensional"
    assert stem("检索") == "检索"


def test_stem_is_idempotent() -> None:
    for word in ("retrieval", "documents", "electrical", "hopping"):
        assert stem(stem(word)) == stem(word)
