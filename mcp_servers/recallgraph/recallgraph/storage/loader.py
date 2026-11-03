"""把解析结果写进数据库。

两处去重，对应"同一份文件不要重复干活"这个诉求：
- 版本按 content_hash 去重：文件没变就不再解析、不再写 chunk；
- 文档按 (归一化标题, 首作者, 年份) 去重：同一篇论文的 v1/v2 会指向同一个 Document。
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from recallgraph.ingest.pipeline import IngestResult
from recallgraph.lexical import index_version


@dataclass(frozen=True)
class LoadOutcome:
    document_id: str
    version_id: str
    chunks_written: int
    deduped: bool


def load_ingest(conn: psycopg.Connection, result: IngestResult) -> LoadOutcome:
    """写入一次解析结果。已是同一份文件时直接返回已有版本。"""
    existing = conn.execute(
        "SELECT id, document_id FROM document_versions WHERE content_hash = %s",
        (result.version.content_hash,),
    ).fetchone()
    if existing is not None:
        version_id, document_id = existing
        # 老数据可能是在倒排表上线前导入的；重复导入时顺手补齐，避免"查不到"。
        indexed = conn.execute(
            "SELECT count(*) FROM chunk_terms WHERE chunk_id LIKE %s", (f"{version_id}:%",)
        ).fetchone()[0]
        if indexed == 0:
            index_version(conn, version_id)
        return LoadOutcome(document_id=document_id, version_id=version_id, chunks_written=0, deduped=True)

    document_id = _upsert_document(conn, result)
    conn.execute(
        """
        INSERT INTO document_versions
            (id, document_id, content_hash, file_path, page_count, published_at,
             ingested_at, language, parser_version, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            result.version.id,
            document_id,
            result.version.content_hash,
            result.version.file_path,
            result.version.page_count,
            result.version.published_at,
            result.version.ingested_at,
            result.version.language,
            result.version.parser_version,
            result.version.status.value,
        ),
    )

    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks
                (id, version_id, seq, page_start, page_end, printed_page_start, printed_page_end,
                 section_path, text, context_header, char_start, char_end, token_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.id,
                    chunk.version_id,
                    chunk.seq,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.printed_page_start,
                    chunk.printed_page_end,
                    list(chunk.section_path),
                    chunk.text,
                    chunk.context_header,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.token_count,
                )
                for chunk in result.chunks
            ],
        )

    # 先写块，再建倒排表：检索只读 chunk_terms，所以两者必须在同一事务里一起成功。
    index_version(conn, result.version.id)

    return LoadOutcome(
        document_id=document_id,
        version_id=result.version.id,
        chunks_written=len(result.chunks),
        deduped=False,
    )


def _upsert_document(conn: psycopg.Connection, result: IngestResult) -> str:
    """先按去重键找回已有文档，找不到再插入。"""
    document = result.document
    row = conn.execute(
        """
        SELECT id FROM documents
        WHERE title_normalized = %s
          AND coalesce(first_author, '') = %s
          AND coalesce(year, 0) = %s
        """,
        (document.title_normalized, document.first_author or "", document.year or 0),
    ).fetchone()
    if row is not None:
        return row[0]

    conn.execute(
        """
        INSERT INTO documents (id, title, title_normalized, authors, first_author, year, external_ids, doc_type, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            document.id,
            document.title,
            document.title_normalized,
            list(document.authors),
            document.first_author,
            document.year,
            psycopg.types.json.Jsonb(document.external_ids),
            document.doc_type.value,
            document.created_at,
        ),
    )
    return document.id