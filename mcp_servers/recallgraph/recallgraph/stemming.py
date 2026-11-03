"""Porter 词干还原（Porter, 1980），只为让 BM25 不被词形变化拖累。

为什么不直接用 nltk / snowballstemmer：它们会带进 numpy 之外的一整条依赖链，
而这里只要一个 120 行的确定性函数——评测要可复现，实现要能讲清楚每一步。

只处理 ASCII 单词；中文不做词干（中文的问题在分词，不在形态）。
"""

from __future__ import annotations

_VOWELS = frozenset("aeiou")


def stem(word: str) -> str:
    """还原英文词干；非纯字母或过短的词原样返回。"""
    if len(word) <= 2 or not word.isascii() or not word.isalpha():
        return word
    word = word.lower()

    if len(word) > 2:
        word = _step1a(word)
        word = _step1b(word)
        word = _step1c(word)
        word = _step2(word)
        word = _step3(word)
        word = _step4(word)
        word = _step5(word)
    return word


def _is_cons(word: str, index: int) -> bool:
    char = word[index]
    if char in _VOWELS:
        return False
    if char == "y":
        return index == 0 or not _is_cons(word, index - 1)
    return True


def _measure(word: str) -> int:
    """m：VC 序列的个数。Porter 用它区分"短词"和"长词"。"""
    count = 0
    previous_vowel = False
    for index in range(len(word)):
        if _is_cons(word, index):
            if previous_vowel:
                count += 1
            previous_vowel = False
        else:
            previous_vowel = True
    return count


def _has_vowel(word: str) -> bool:
    return any(not _is_cons(word, index) for index in range(len(word)))


def _ends_double_cons(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_cons(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    if len(word) < 3:
        return False
    return (
        _is_cons(word, len(word) - 3)
        and not _is_cons(word, len(word) - 2)
        and _is_cons(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem_word = word[: -len(suffix)]
            if not _has_vowel(stem_word):
                return word
            return _step1b_cleanup(stem_word)
    return word


def _step1b_cleanup(word: str) -> str:
    if word.endswith(("at", "bl", "iz")):
        return word + "e"
    if _ends_double_cons(word) and word[-1] not in "lsz":
        return word[:-1]
    if _measure(word) == 1 and _ends_cvc(word):
        return word + "e"
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
)

_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def _apply(word: str, table: tuple[tuple[str, str], ...]) -> str:
    for suffix, replacement in table:
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _measure(base) > 0:
                return base + replacement
            return word
    return word


def _step2(word: str) -> str:
    return _apply(word, _STEP2)


def _step3(word: str) -> str:
    return _apply(word, _STEP3)


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _measure(base) > 1 and not (suffix == "ion" and (not base or base[-1] not in "st")):
                return base
            return word
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        base = word[:-1]
        m = _measure(base)
        if m > 1 or (m == 1 and not _ends_cvc(base)):
            word = base
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


__all__ = ["stem"]