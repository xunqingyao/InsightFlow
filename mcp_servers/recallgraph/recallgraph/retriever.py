"""检索实现：把 BM25 召回的段落，翻成可核对的断言证据包。

一条查询走五步（对应 docs/02-检索契约.md 的八步，先落地前五步）：

    关键词召回段落 → 段落映射断言 → 图扩展一跳 → 组装引用 → 按预算裁剪

向量通道已接入（百炼 text-embedding-v3 或本地 bge，由 resolve_embedding_model 决定）；
重排默认开启（百炼 gte-rerank-v2）：第一阶段 BM25+向量融合后的头部 `RERANK_HEAD` 个交给重排模型再排一次，
尾部顺序不动——这是“重排”的本义（对已有排序的头部再排），不是多一路召回。
实测中文同语言题会掉（0.689→0.603），跨语言题大涨（0.445→0.643）；产品的主场景是中文提问、英文论文，所以默认开。
缺通道一律在 warnings 里说清楚。
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import psycopg

from recallgraph.dense import MODEL_NAME, Embedder, available_models, make_embedder, search_dense
from recallgraph.lexical import LexicalHit, search_lexical
from recallgraph.rerank import DashScopeReranker, RerankError
from recallgraph.retrieval import (
    EXPANSION_PREDICATES,
    ChunkHit,
    Citation,
    ClaimHit,
    NoteHit,
    QueryFilters,
    RetrievalQuery,
    RetrievalResult,
    RetrievalScore,
    RetrievalStats,
    TraceStep,
    estimate_tokens,
    fit_budget,
)
from recallgraph.schema import Claim, Note
from recallgraph.vocabulary import ClaimStatus, NoteRelation, Predicate

RRF_K = 60
RERANK_HEAD = 10  # 只重排头部这么多：K=10 与 K=50 效果相同，成本差 5 倍
# 向量通道在融合里的权重。SciFact dev 分片扫出 6.0（1.0/2.0/3.0/4.0/6.0 中最好），
# 意味着在这份英文语料上"语义"比"字面"更可靠；中文场景未验证，配置项留着。
# 试过按查询动态调权重（覆盖度/两路分歧，硬门控与软加权），dev 上全部不如固定——
# 信号分离度不够，见 evals/reports/scifact-2026-10-28.md 的"动态融合"一节。
DENSE_WEIGHT = float(os.environ.get("RECALLGRAPH_DENSE_WEIGHT", "6.0"))
CHUNK_PREVIEW_CHARS = 600
# 关键词命中低于最高分这个比例就当噪声丢掉：中文常见词（"连接""模型"）会误撞，
# 加进融合后反而把真正的向量命中挤下去。
LEXICAL_FLOOR_RATIO = float(os.environ.get("RECALLGRAPH_LEXICAL_FLOOR", "0.3"))
EXPANSION_DECAY = 0.5
# 开重排时两路各取多少候选再融合。池子太小重排没有腾挪空间，太大又费钱；
# 50 是候选召回率（0.94）开始饱和的位置。
RERANK_POOL = int(os.environ.get("RECALLGRAPH_RERANK_POOL", "50"))


class SqlRetriever:
    """基于 PostgreSQL 的检索器。构造时传一个连接，不自建连接池。"""

    name = "sql-v1"

    def __init__(
        self,
        conn: psycopg.Connection,
        *,
        embedding_model: str | None = None,
        dense_weight: float = DENSE_WEIGHT,
        rerank: bool | None = None,
        reranker: DashScopeReranker | None = None,
        candidate_pool: int = RERANK_POOL,
    ) -> None:
        self.conn = conn
        self.embedding_model = embedding_model
        self.dense_weight = dense_weight
        self.use_lexical = True
        # None = 自动：有 key 就开重排（主场景是中文提问、英文论文）。
        if rerank is None:
            self.rerank = bool(os.environ.get("DASHSCOPE_API_KEY"))
        else:
            self.rerank = rerank
        self._reranker = reranker or (DashScopeReranker() if self.rerank else None)
        self.candidate_pool = candidate_pool
        self._embedders: dict[str, Embedder] = {}

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        started = time.perf_counter()
        trace: list[TraceStep] = []
        warnings: list[str] = []

        # ① 关键词召回段落
        t0 = time.perf_counter()
        versions = list(query.filters.version_ids)
        recall_k = self.candidate_pool if self.rerank else query.top_k
        lexical = (
            search_lexical(self.conn, query.text, top_k=recall_k, version_ids=versions or None)
            if self.use_lexical
            else []
        )
        if lexical:
            # 相对地板：低于最高分 30% 的命中当噪声（中文常见词会误撞）。
            floor = LEXICAL_FLOOR_RATIO * lexical[0].score
            lexical = [hit for hit in lexical if hit.score >= floor]
        lexical_rank = {hit.chunk_id: rank for rank, hit in enumerate(lexical, start=1)}
        trace.append(
            TraceStep(
                stage="lexical",
                elapsed_ms=_ms(t0),
                candidates_in=1,
                candidates_out=len(lexical),
                detail={"channel": "bm25", "terms": len(query.text.split())},
            )
        )
        # ①b 向量召回（有向量才走；模型不可用不能让关键词通道陪葬）
        dense_rank: dict[str, int] = {}
        t0 = time.perf_counter()
        model = resolve_embedding_model(self.conn, versions, explicit=self.embedding_model)
        present = available_models(self.conn, versions or None)
        if len(present) > 1 and not (self.embedding_model or os.environ.get("RECALLGRAPH_EMBEDDING_MODEL")):
            warnings.append(f"库里有多个向量模型 {list(present)}，已默认用 {model}；要换请传 --embedding-model")
        if model is None:
            warnings.append("向量通道未接入：所选版本还没有向量")
        else:
            try:
                embedder = self._embedders.setdefault(model, make_embedder(model))
                dense = search_dense(
                    self.conn, query.text, top_k=recall_k, embedder=embedder, model=model, version_ids=versions or None
                )
                dense_rank = {hit.chunk_id: rank for rank, hit in enumerate(dense, start=1)}
                if not dense_rank:
                    warnings.append(f"向量通道无结果：模型 {model} 在所选版本上没有向量")
            except Exception as exc:  # noqa: BLE001 - 模型下载/推理失败都降级
                warnings.append(f"向量通道不可用，已降级为关键词：{exc}")
        trace.append(
            TraceStep(
                stage="dense",
                elapsed_ms=_ms(t0),
                candidates_in=1,
                candidates_out=len(dense_rank),
                detail={"channel": model or "none", "weight": DENSE_WEIGHT},
            )
        )

        chunk_rank = _fuse(lexical_rank, dense_rank, dense_weight=self.dense_weight)
        if not chunk_rank:
            return _empty(query, trace, started, "关键词与向量都没有命中任何段落")

        # ①c 重排（可选）：把融合后的候选整体交给重排模型重新打分
        reranked_ids: set[str] = set()
        if self.rerank and self._reranker is not None:
            t0 = time.perf_counter()
            ordered = sorted(chunk_rank.items(), key=lambda item: item[1])
            pool_ids = [chunk_id for chunk_id, _ in ordered[: self.candidate_pool]]
            texts = {
                row[0]: row[1]
                for row in self.conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (pool_ids,)).fetchall()
            }
            items = [(chunk_id, texts[chunk_id]) for chunk_id in pool_ids if chunk_id in texts]
            if items:
                try:
                    ranked_pool = self._reranker.rerank(query.text, items)
                    # 只替换头部 RERANK_HEAD 个的顺序，尾部保持第一阶段的排序
                    head = [chunk_id for chunk_id, _ in ranked_pool[:RERANK_HEAD]]
                    tail = [chunk_id for chunk_id, _ in ordered if chunk_id not in set(head)]
                    merged = head + tail
                    chunk_rank = {chunk_id: rank for rank, chunk_id in enumerate(merged, 1)}
                    reranked_ids = set(texts)
                    trace.append(
                        TraceStep(
                            stage="rerank",
                            elapsed_ms=_ms(t0),
                            candidates_in=len(items),
                            candidates_out=len(ranked_pool),
                            detail={"model": self._reranker.model, "pool": len(items)},
                        )
                    )
                except RerankError as exc:
                    warnings.append(f"重排不可用，已回退到融合排序：{exc}")

        # ② 段落 → 断言
        t0 = time.perf_counter()
        hits = self._claims_from_chunks(chunk_rank, query.filters, lexical, lexical_rank, dense_rank)
        trace.append(
            TraceStep(
                stage="map_entities",
                elapsed_ms=_ms(t0),
                candidates_in=len(lexical),
                candidates_out=len(hits),
                detail={"per_chunk": round(len(hits) / max(len(lexical), 1), 2)},
            )
        )
        if not hits:
            # 断言是加分项不是准入条件：段落检索到了就返回原文，
            # 这样"检索没找到"和"抽取器没抽出来"能分开评估。
            warnings.append("命中的段落里没有已抽取的断言，本次只返回原文证据")

        # ③ 图扩展一跳（共享实体的邻近断言）
        if hits and query.expand_hops > 0:
            t0 = time.perf_counter()
            before = len(hits)
            hits = self._expand(hits, query.filters, seed_rrf=max((h.rrf for h in hits), default=0.0))
            trace.append(
                TraceStep(
                    stage="expand",
                    elapsed_ms=_ms(t0),
                    candidates_in=before,
                    candidates_out=len(hits),
                    detail={"hops": 1, "added": len(hits) - before},
                )
            )

        # ④ 排序 + 组装引用
        t0 = time.perf_counter()
        def _order(hit: ClaimHit) -> tuple[int, tuple[float, int, int, str]]:
            best = min((chunk_rank[c.chunk_id] for c in hit.citations if c.chunk_id in chunk_rank), default=10**6)
            return (best, _score_key(hit))

        ranked = sorted(hits, key=_order) if hits else []
        selected = ranked[: query.max_chunks]
        ranked_out: list[ClaimHit] = []
        for rank, hit in enumerate(selected, start=1):
            ranked_out.append(hit.model_copy(update={"rank": rank}))
        chunks = self._chunk_previews(chunk_rank, lexical_rank, dense_rank, query.max_chunks, reranked_ids=reranked_ids)
        trace.append(
            TraceStep(
                stage="assemble",
                elapsed_ms=_ms(t0),
                candidates_in=len(hits),
                candidates_out=len(ranked_out),
                detail={"chunks": len(chunks)},
            )
        )

        # ⑤ 预算裁剪
        t0 = time.perf_counter()
        kept_chunks, truncated = fit_budget(chunks, query.max_tokens)
        tokens_used = sum(estimate_tokens(chunk.text) for chunk in kept_chunks)
        if truncated:
            warnings.append(f"证据超出 {query.max_tokens} token 预算，已截断 {len(chunks) - len(kept_chunks)} 块")
        trace.append(
            TraceStep(
                stage="budget",
                elapsed_ms=_ms(t0),
                candidates_in=len(chunks),
                candidates_out=len(kept_chunks),
                detail={"tokens_used": tokens_used, "truncated": truncated},
            )
        )

        unsupported = _unsupported_filters(query.filters)
        if unsupported:
            warnings.append(f"以下过滤条件尚未实现，已忽略：{'、'.join(unsupported)}")

        notes = self._notes(ranked_out, query.filters) if query.filters.include_notes else []
        if query.mode in ("hybrid", "dense") and not dense_rank:
            warnings.append("向量通道未生效：本次仅关键词召回 + 图扩展")

        return RetrievalResult(
            status="ok" if (ranked_out or chunks) else "empty",
            query=query,
            claims=tuple(ranked_out),
            chunks=tuple(kept_chunks),
            notes=tuple(notes),
            trace=tuple(trace),
            stats=RetrievalStats(
                elapsed_ms=_ms(started),
                tokens_used=tokens_used,
                claims_total=len(ranked_out),
                chunks_total=len(kept_chunks),
                truncated=truncated,
            ),
            warnings=tuple(warnings),
        )

    # ── 内部步骤 ──────────────────────────────────────────────────────────

    def _claims_from_chunks(
        self,
        chunk_rank: dict[str, int],
        filters: QueryFilters,
        lexical: Sequence[LexicalHit],
        lexical_rank: dict[str, int] | None = None,
        dense_rank: dict[str, int] | None = None,
    ) -> list[ClaimHit]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.version_id, c.subject_id, c.predicate, c.object_id, c.object_text,
                   c.conditions, c.claim_text, c.confidence, c.status, c.created_at,
                   e.chunk_id, e.quote, e.page,
                   d.id, d.title, ch.section_path, ch.text,
                   s.canonical_name, o.canonical_name
            FROM claims c
            JOIN evidence e ON e.claim_id = c.id
            JOIN chunks ch ON ch.id = e.chunk_id
            JOIN document_versions v ON v.id = c.version_id
            JOIN documents d ON d.id = v.document_id
            JOIN entities s ON s.id = c.subject_id
            LEFT JOIN entities o ON o.id = c.object_id
            WHERE c.status = ANY(%s) AND e.chunk_id = ANY(%s)
            """,
            ([status.value for status in filters.claim_status], list(chunk_rank)),
        ).fetchall()

        grouped: dict[str, dict] = {}
        for row in rows:
            claim_id = row[0]
            if claim_id not in grouped:
                grouped[claim_id] = {"row": row, "evidence": []}
            grouped[claim_id]["evidence"].append(row[11:18])

        hits: list[ClaimHit] = []
        for claim_id, item in grouped.items():
            row = item["row"]
            claim = Claim(
                id=claim_id,
                version_id=row[1],
                subject_id=row[2],
                predicate=Predicate(row[3]),
                object_id=row[4],
                object_text=row[5],
                conditions=row[6],
                claim_text=row[7],
                confidence=row[8],
                status=ClaimStatus(row[9]),
                created_at=row[10],
            )
            citations = tuple(
                Citation(
                    chunk_id=evidence[0],
                    version_id=claim.version_id,
                    document_id=evidence[3],
                    document_title=evidence[4],
                    page=evidence[2],
                    quote=evidence[1],
                    section_path=tuple(evidence[5] or ()),
                )
                for evidence in item["evidence"]
            )
            if filters.entity_ids and claim.subject_id not in filters.entity_ids and claim.object_id not in filters.entity_ids:
                continue
            best = min(chunk_rank[c.chunk_id] for c in citations if c.chunk_id in chunk_rank)
            used = []
            if lexical_rank and any(c.chunk_id in lexical_rank for c in citations):
                used.append("lexical")
            if dense_rank and any(c.chunk_id in dense_rank for c in citations):
                used.append("dense")
            hits.append(
                ClaimHit(
                    claim=claim,
                    citations=citations,
                    channels=tuple(used) or ("lexical",),
                    hop=0,
                    rrf=1.0 / (RRF_K + best),
                )
            )
        return hits

    def _expand(self, hits: list[ClaimHit], filters: QueryFilters, seed_rrf: float) -> list[ClaimHit]:
        known = {hit.claim.id for hit in hits}
        entity_ids = sorted({hit.claim.subject_id for hit in hits} | {hit.claim.object_id for hit in hits if hit.claim.object_id})
        if not entity_ids:
            return hits

        rows = self.conn.execute(
            """
            SELECT c.id, c.version_id, c.subject_id, c.predicate, c.object_id, c.object_text,
                   c.conditions, c.claim_text, c.confidence, c.status, c.created_at,
                   e.chunk_id, e.quote, e.page,
                   d.id, d.title, ch.section_path
            FROM claims c
            JOIN evidence e ON e.claim_id = c.id
            JOIN chunks ch ON ch.id = e.chunk_id
            JOIN document_versions v ON v.id = c.version_id
            JOIN documents d ON d.id = v.document_id
            WHERE c.status = ANY(%s)
              AND c.predicate = ANY(%s)
              AND (c.subject_id = ANY(%s) OR c.object_id = ANY(%s))
            LIMIT %s
            """,
            (
                [status.value for status in filters.claim_status],
                [predicate.value for predicate in sorted(EXPANSION_PREDICATES, key=lambda p: p.value)],
                entity_ids,
                entity_ids,
                len(hits) * 3,
            ),
        ).fetchall()

        expanded: list[ClaimHit] = []
        for row in rows:
            if row[0] in known:
                continue
            known.add(row[0])
            claim = Claim(
                id=row[0],
                version_id=row[1],
                subject_id=row[2],
                predicate=Predicate(row[3]),
                object_id=row[4],
                object_text=row[5],
                conditions=row[6],
                claim_text=row[7],
                confidence=row[8],
                status=ClaimStatus(row[9]),
                created_at=row[10],
            )
            citation = Citation(
                chunk_id=row[11],
                version_id=claim.version_id,
                document_id=row[14],
                document_title=row[15],
                page=row[13],
                quote=row[12],
                section_path=tuple(row[16] or ()),
            )
            expanded.append(
                ClaimHit(
                    claim=claim,
                    citations=(citation,),
                    channels=("graph",),
                    hop=1,
                    rrf=seed_rrf * EXPANSION_DECAY,
                )
            )
        return hits + expanded

    def _chunk_previews(
        self,
        chunk_rank: dict[str, int],
        lexical_rank: dict[str, int],
        dense_rank: dict[str, int],
        limit: int,
        *,
        reranked_ids: set[str] | None = None,
    ) -> list[ChunkHit]:
        """证据块清单：按段落融合名次取前 limit 个，不要求它先有断言。

        早先这里按"被断言引用的块"构造，结果没有断言的段落直接消失——桩在中文上
        抽不准，等于把最好的块筛掉（产品路径 nDCG 0.31 vs 单通道 0.49 就是这么丢的）。
        """
        ordered = sorted(chunk_rank.items(), key=lambda item: item[1])[:limit]
        if not ordered:
            return []
        ids = [chunk_id for chunk_id, _ in ordered]
        rows = {
            row[0]: row[1:]
            for row in self.conn.execute(
                "SELECT id, version_id, page_start, section_path, text FROM chunks WHERE id = ANY(%s)", (ids,)
            ).fetchall()
        }
        previews: list[ChunkHit] = []
        for chunk_id, _ in ordered:
            row = rows.get(chunk_id)
            if row is None:
                continue
            channels: list[str] = []
            score = 0.0
            if chunk_id in lexical_rank:
                channels.append("lexical")
                score += 1.0 / (RRF_K + lexical_rank[chunk_id])
            if chunk_id in dense_rank:
                channels.append("dense")
                score += DENSE_WEIGHT / (RRF_K + dense_rank[chunk_id])
            if reranked_ids and chunk_id in reranked_ids:
                channels.append("rerank")
            previews.append(
                ChunkHit(
                    chunk_id=chunk_id,
                    version_id=row[0],
                    page=row[1],
                    section_path=tuple(row[2] or ()),
                    text=_clip(row[3], CHUNK_PREVIEW_CHARS),
                    channels=tuple(channels) or ("lexical",),
                    rrf=score,
                    rank=len(previews) + 1,
                )
            )
        return previews

    def _notes(self, hits: Sequence[ClaimHit], filters: QueryFilters) -> list[NoteHit]:
        claim_ids = [hit.claim.id for hit in hits]
        if not claim_ids:
            return []
        rows = self.conn.execute(
            """
            SELECT n.id, n.author, n.title, n.body, n.linked_version_ids, n.created_at, n.updated_at,
                   nc.claim_id, nc.relation
            FROM notes n JOIN note_claims nc ON nc.note_id = n.id
            WHERE nc.claim_id = ANY(%s)
            """,
            (claim_ids,),
        ).fetchall()
        hits_out: list[NoteHit] = []
        seen: set[str] = set()
        for row in rows:
            if row[0] in seen:
                continue
            seen.add(row[0])
            note = Note(
                id=row[0],
                author=row[1],
                title=row[2],
                body=row[3],
                linked_version_ids=list(row[4] or []),
                created_at=row[5],
                updated_at=row[6],
            )
            hits_out.append(
                NoteHit(note=note, relation=NoteRelation(row[8]), claim_id=row[7])
            )
        return hits_out


def resolve_embedding_model(
    conn: psycopg.Connection, versions: Sequence[str], *, explicit: str | None = None
) -> str | None:
    """决定用哪个向量模型：显式参数 > 环境变量 > 库里行数最多的那个。

    不做语言猜测——猜错比不猜更糟（中文查询配英文模型，见 docs/07）。
    """
    if explicit:
        return explicit
    env = os.environ.get("RECALLGRAPH_EMBEDDING_MODEL")
    if env:
        return env
    models = available_models(conn, list(versions) or None)
    if not models:
        return None
    # 统一模型（百炼 text-embedding-v3）有向量时就优先用它，避免多模型并存时按行数随机选。
    if MODEL_NAME in models:
        return MODEL_NAME
    return next(iter(models))


def _fuse(lexical_rank: dict[str, int], dense_rank: dict[str, int], *, dense_weight: float = DENSE_WEIGHT) -> dict[str, int]:
    """RRF 融合两路召回，返回 chunk_id → 融合名次。"""
    scores: dict[str, float] = {}
    for chunk_id, rank in lexical_rank.items():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for chunk_id, rank in dense_rank.items():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {chunk_id: rank for rank, (chunk_id, _) in enumerate(ordered, start=1)}


def _score_key(hit: ClaimHit) -> tuple[float, int, int, str]:
    return RetrievalScore(
        rrf=hit.rrf,
        evidence_count=len(hit.citations),
        confidence=hit.claim.confidence,
        claim_id=hit.claim.id,
    ).sort_key()


def _unsupported_filters(filters: QueryFilters) -> list[str]:
    pending: list[str] = []
    if filters.document_ids:
        pending.append("document_ids")
    if filters.entity_kinds:
        pending.append("entity_kinds")
    if filters.year_min is not None or filters.year_max is not None:
        pending.append("年份")
    if filters.languages:
        pending.append("语言")
    return pending


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _empty(query: RetrievalQuery, trace: list[TraceStep], started: float, reason: str) -> RetrievalResult:
    return RetrievalResult(
        status="empty",
        query=query,
        trace=tuple(trace),
        stats=RetrievalStats(elapsed_ms=_ms(started)),
        warnings=(reason,),
    )


__all__ = ["SqlRetriever"]