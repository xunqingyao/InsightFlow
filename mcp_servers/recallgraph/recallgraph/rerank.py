"""百炼重排（gte-rerank-v2）：把候选段落按"和问题的相关度"重新打分。

为什么要单独一层：本项目的候选召回率已经 0.94（26 题标注集），但 nDCG@10 只有 0.61，
说明正确段落进了池子却没被排到前面——这是排序问题，不是召回问题。重排模型看
（问题, 段落）这一对，比 BM25 看词频、比 embedding 看句子整体向量都更直接。

接口：DashScope 的 text-rerank，单请求最多 500 段，这里按 100 段一批。
失败重试三次；仍失败就抛 RerankError，由调用方决定降级（不吞错）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
DEFAULT_RERANK_MODEL = "gte-rerank-v2"
BATCH_SIZE = 100


class RerankError(RuntimeError):
    """重排接口失败（重试后仍失败）。"""


class DashScopeReranker:
    """把 (问题, 候选文本) 送进百炼重排，返回按相关度降序的 (下标, 分数)。"""

    def __init__(
        self,
        *,
        model: str = DEFAULT_RERANK_MODEL,
        api_key: str | None = None,
        batch_size: int = BATCH_SIZE,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("DASHSCOPE_API_KEY", "")
        self.batch_size = batch_size
        self.timeout = timeout

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """返回与 documents 等长的分数，顺序与输入一致（不依赖服务端顺序）。"""
        if not documents:
            return []
        if not self.api_key:
            raise RerankError("DASHSCOPE_API_KEY 未设置，无法调用重排接口")

        scores: list[float] = [0.0] * len(documents)
        for start in range(0, len(documents), self.batch_size):
            batch = list(documents[start : start + self.batch_size])
            payload = json.dumps(
                {
                    "model": self.model,
                    "input": {"query": query, "documents": batch},
                    "parameters": {"return_documents": False, "top_n": len(batch)},
                }
            ).encode()
            request = urllib.request.Request(
                RERANK_URL,
                data=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            body = self._post(request)
            for item in body["output"]["results"]:
                scores[start + int(item["index"])] = float(item["relevance_score"])
        return scores

    def _post(self, request: urllib.request.Request) -> dict:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = RerankError(f"重排接口 HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404):
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = RerankError(f"重排接口失败: {exc}")
            time.sleep(0.5 * (attempt + 1))
        raise last_error or RerankError("重排接口失败（未知原因）")

    def rerank(self, query: str, items: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """items 是 (标识, 文本)，返回按分数降序的 (标识, 分数)；分数相同保持原顺序。"""
        texts = [text for _, text in items]
        scores = self.score(query, texts)
        paired = list(zip(items, scores, strict=True))
        paired.sort(key=lambda pair: -pair[1])
        return [(ident, score) for (ident, _), score in paired]