"""把抽取结果写进知识表：entities / claims / evidence / chunk_entities。

幂等规则：同一个 (版本, 抽取器) 只跑一次；要重跑，先把旧断言标记为 retracted
（证据表只追加、禁止删除，所以"撤回"就是留痕，不是抹掉），再用新结果覆盖"当前有效"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg

from recallgraph.extract.contract import (
    ClaimDraft,
    EntityDraft,
    ExtractionError,
    ExtractionResult,
    validate_result,
)
from recallgraph.ids import new_ulid, normalize_name
from recallgraph.schema import Chunk
from recallgraph.vocabulary import ClaimStatus, EntityKind


@dataclass
class ExtractionStats:
    version_id: str
    extractor: str
    claims_written: int = 0
    evidence_written: int = 0
    entities_written: int = 0
    skipped_reason: str = ""
    retracted: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped_reason:
            return f"跳过（{self.skipped_reason}）"
        return (
            f"断言 {self.claims_written} · 证据 {self.evidence_written} · "
            f"实体 {self.entities_written} · 撤回旧断言 {self.retracted} · 拒收 {len(self.errors)}"
        )


def extract_version(
    conn: psycopg.Connection,
    version_id: str,
    extractor,
    *,
    force: bool = False,
    retire_old: bool = False,
) -> ExtractionStats:
    stats = ExtractionStats(version_id=version_id, extractor=extractor.name)

    chunks = _load_chunks(conn, version_id)
    if not chunks:
        stats.skipped_reason = "版本不存在或没有段落"
        return stats

    existing = conn.execute(
        "SELECT count(*) FROM claims WHERE version_id = %s AND extractor = %s",
        (version_id, extractor.name),
    ).fetchone()[0]

    if existing and not force:
        stats.skipped_reason = f"已抽取过 {existing} 条（用 --force 重跑）"
        return stats

    if existing and force:
        stats.retracted = conn.execute(
            """
            UPDATE claims SET status = %s
            WHERE version_id = %s AND extractor = %s AND status <> %s
            """,
            (ClaimStatus.RETRACTED.value, version_id, extractor.name, ClaimStatus.RETRACTED.value),
        ).rowcount

    drafts: list[ClaimDraft] = []
    entities: dict[str, EntityDraft] = {}
    for chunk in chunks:
        try:
            result = extractor.extract(chunk)
        except ExtractionError as exc:
            # 单段失败不该毁掉整个版本：记下来，继续抽下一段。
            stats.errors.append(f"{chunk.id}: {exc}")
            continue
        for entity in result.entities:
            entities.setdefault(normalize_name(entity.canonical_name), entity)
        drafts.extend(result.claims)

    checked, errors = validate_result(ExtractionResult(claims=_merge(drafts)), chunks)
    stats.errors.extend(errors)

    # 模型不一定把断言里出现的每个名字都列进 entities，缺的补成 concept，
    # 否则 _entity_id 会 KeyError（桩会自己生成实体，这个洞直到接模型才暴露）。
    for item in checked:
        for name in (item.draft.subject, item.draft.object):
            if name and normalize_name(name) not in entities:
                entities[normalize_name(name)] = EntityDraft(canonical_name=name, kind=EntityKind.CONCEPT)

    entity_ids = _write_entities(conn, list(entities.values()), stats)

    seen_claims: set[tuple[str, str, str]] = set()
    for item in checked:
        draft = item.draft
        key = (normalize_name(draft.subject), draft.predicate.value, normalize_name(draft.object or ""))
        if key in seen_claims:
            continue
        seen_claims.add(key)

        claim_id = new_ulid()
        conn.execute(
            """
            INSERT INTO claims
                (id, version_id, subject_id, predicate, object_id, object_text, conditions,
                 claim_text, confidence, status, extractor)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                claim_id,
                version_id,
                _entity_id(entity_ids, draft.subject),
                draft.predicate.value,
                None if draft.predicate.value == "achieves" else _entity_id(entity_ids, draft.object or ""),
                draft.object if draft.predicate.value == "achieves" else None,
                draft.conditions,
                draft.claim_text,
                draft.confidence.value,
                ClaimStatus.ACTIVE.value,
                extractor.name,
            ),
        )
        stats.claims_written += 1

        for evidence in draft.evidence:
            conn.execute(
                """
                INSERT INTO evidence (id, claim_id, chunk_id, quote, page, stance)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (new_ulid(), claim_id, evidence.chunk_id, evidence.quote, evidence.page, evidence.stance.value),
            )
            stats.evidence_written += 1
            _link_chunk_entity(conn, evidence.chunk_id, draft.subject, entity_ids)
            if draft.object:
                _link_chunk_entity(conn, evidence.chunk_id, draft.object, entity_ids)

    if retire_old and stats.claims_written:
        # 换抽取器时，旧抽取器的断言作废但保留（证据只增不删，撤回是留痕）
        stats.retracted += conn.execute(
            """
            UPDATE claims SET status = %s
            WHERE version_id = %s AND extractor <> %s AND status <> %s
            """,
            (ClaimStatus.RETRACTED.value, version_id, extractor.name, ClaimStatus.RETRACTED.value),
        ).rowcount

    return stats


def _merge(drafts: list[ClaimDraft]) -> list[ClaimDraft]:
    """同一版本内合并重复断言：保留第一条，把其它条的支持证据并进来。"""
    merged: dict[tuple[str, str, str], ClaimDraft] = {}
    for draft in drafts:
        key = (normalize_name(draft.subject), draft.predicate.value, normalize_name(draft.object or ""))
        if key not in merged:
            merged[key] = draft
            continue
        first = merged[key]
        quotes = {e.quote for e in first.evidence}
        extra = [e for e in draft.evidence if e.quote not in quotes]
        if extra:
            merged[key] = first.model_copy(update={"evidence": first.evidence + extra})
    return list(merged.values())


def _load_chunks(conn: psycopg.Connection, version_id: str) -> list[Chunk]:
    rows = conn.execute(
        "SELECT id, version_id, seq, page_start, page_end, printed_page_start, printed_page_end, "
        "section_path, text, context_header, char_start, char_end, token_count "
        "FROM chunks WHERE version_id = %s ORDER BY seq",
        (version_id,),
    ).fetchall()
    return [
        Chunk(
            id=row[0],
            version_id=row[1],
            seq=row[2],
            page_start=row[3],
            page_end=row[4],
            printed_page_start=row[5],
            printed_page_end=row[6],
            section_path=list(row[7]),
            text=row[8],
            context_header=row[9],
            char_start=row[10],
            char_end=row[11],
            token_count=row[12],
        )
        for row in rows
    ]


def _write_entities(conn: psycopg.Connection, drafts: list[EntityDraft], stats: ExtractionStats) -> dict[str, str]:
    """按 normalized_name 去重复用已有实体；新实体才插入。"""
    ids: dict[str, str] = {}
    for draft in drafts:
        normalized = normalize_name(draft.canonical_name)
        if not normalized:
            continue
        row = conn.execute("SELECT id FROM entities WHERE normalized_name = %s", (normalized,)).fetchone()
        if row is not None:
            ids[normalized] = row[0]
            continue

        entity_id = new_ulid()
        conn.execute(
            "INSERT INTO entities (id, canonical_name, normalized_name, kind, aliases, description) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (entity_id, draft.canonical_name, normalized, draft.kind.value, list(draft.aliases), draft.description),
        )
        ids[normalized] = entity_id
        stats.entities_written += 1
    return ids


def _entity_id(ids: dict[str, str], name: str) -> str:
    key = normalize_name(name)
    if key not in ids:
        raise KeyError(f"实体未写入：{name}")
    return ids[key]


def _link_chunk_entity(conn: psycopg.Connection, chunk_id: str, name: str, ids: dict[str, str]) -> None:
    entity_id = ids.get(normalize_name(name))
    if entity_id is None:
        return
    offset = conn.execute("SELECT strpos(text, %s) - 1 FROM chunks WHERE id = %s", (name, chunk_id)).fetchone()
    if offset is None or offset[0] < 0:
        offset = (0,)
    conn.execute(
        "INSERT INTO chunk_entities (chunk_id, entity_id, mention_text, char_offset) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING",
        (chunk_id, entity_id, name, offset[0]),
    )


__all__ = ["ExtractionStats", "extract_version"]