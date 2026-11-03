"""ID 生成与归一化工具。

ULID：26 位 Crockford Base32，前 10 位是毫秒时间戳，天然按创建时间有序，
便于 PostgreSQL 上按主键范围扫描；后 16 位是 80 位随机数，无中心协调也能唯一。
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """返回一个新的 26 位 ULID。"""
    ts = int(time.time() * 1000)
    ts_part = "".join(_CROCKFORD[(ts >> shift) & 0x1F] for shift in range(45, -1, -5))
    rand = int.from_bytes(os.urandom(10), "big")
    rand_part = "".join(_CROCKFORD[(rand >> shift) & 0x1F] for shift in range(75, -1, -5))
    return ts_part + rand_part


def normalize_name(name: str) -> str:
    """实体名归一化：全角转半角、小写、折叠空白、去掉首尾标点。

    只做"确定性、可解释"的归一化。同义词归并靠 aliases，不靠相似度。
    """
    import unicodedata

    text = unicodedata.normalize("NFKC", name).casefold()
    text = " ".join(text.split())
    return text.strip(" \t\r\n.,;:!?\"'()[]{}<>·-—_")