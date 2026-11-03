-- RecallGraph 初始建表脚本（PostgreSQL 15+）
-- 对应文档：docs/01-数据模型.md
-- 三条铁律在库层面的落法：
--   1) 版本/段落/证据只写不改  -> 触发器 forbid_mutation
--   2) 断言必须有证据          -> 应用层 ClaimBundle 校验 + 视图 v_active_claims 过滤
--   3) 记忆只追加              -> memory_events 仅允许更新 status 字段
--
-- 向量列默认注释掉；启用 pgvector 后再执行文件末尾的 ALTER。

BEGIN;

-- ── 1. 身份 ────────────────────────────────────────────────────────────────

CREATE TABLE documents (
    id                text PRIMARY KEY,
    title             text NOT NULL,
    title_normalized  text NOT NULL,
    authors           text[] NOT NULL DEFAULT '{}',
    first_author      text,
    year              integer CHECK (year BETWEEN 1900 AND 2100),
    external_ids      jsonb NOT NULL DEFAULT '{}'::jsonb,
    doc_type          text NOT NULL DEFAULT 'paper'
                      CHECK (doc_type IN ('paper', 'note', 'book', 'web')),
    created_at        timestamptz NOT NULL DEFAULT now()
);

-- 去重键：标题 + 首作者 + 年份（大量 PDF 没有 DOI）
CREATE UNIQUE INDEX documents_dedup_idx
    ON documents (title_normalized, coalesce(first_author, ''), coalesce(year, 0));

CREATE TABLE document_versions (
    id              text PRIMARY KEY,
    document_id     text NOT NULL REFERENCES documents (id) ON DELETE RESTRICT,
    content_hash    text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    file_path       text NOT NULL,
    page_count      integer NOT NULL CHECK (page_count >= 1),
    published_at    timestamptz,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    language        text,
    parser_version  text NOT NULL,
    status          text NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'superseded', 'failed'))
);

CREATE UNIQUE INDEX document_versions_hash_idx ON document_versions (content_hash);
CREATE INDEX document_versions_doc_idx ON document_versions (document_id, ingested_at DESC);

-- ── 2. 原文 ────────────────────────────────────────────────────────────────

CREATE TABLE chunks (
    id                    text PRIMARY KEY,
    version_id            text NOT NULL REFERENCES document_versions (id) ON DELETE RESTRICT,
    seq                   integer NOT NULL CHECK (seq >= 0),
    page_start            integer NOT NULL CHECK (page_start >= 1),
    page_end              integer NOT NULL CHECK (page_end >= page_start),
    printed_page_start    integer,
    printed_page_end      integer,
    section_path          text[] NOT NULL DEFAULT '{}',
    text                  text NOT NULL CHECK (length(text) > 0),
    context_header        text NOT NULL CHECK (length(context_header) > 0),
    char_start            integer NOT NULL CHECK (char_start >= 0),
    char_end              integer NOT NULL CHECK (char_end > char_start),
    token_count           integer CHECK (token_count > 0),
    CONSTRAINT chunks_page_span CHECK (page_end - page_start <= 1),
    CONSTRAINT chunks_id_format CHECK (id = version_id || ':' || lpad(seq::text, 4, '0'))
);

CREATE UNIQUE INDEX chunks_version_seq_idx ON chunks (version_id, seq);

CREATE TABLE entities (
    id               text PRIMARY KEY,
    canonical_name   text NOT NULL,
    normalized_name  text NOT NULL,
    kind             text NOT NULL
                     CHECK (kind IN ('method', 'dataset', 'metric', 'concept', 'person', 'org', 'paper')),
    aliases          text[] NOT NULL DEFAULT '{}',
    description      text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX entities_normalized_name_idx ON entities (normalized_name);
CREATE INDEX entities_aliases_gin_idx ON entities USING gin (aliases);
CREATE INDEX entities_kind_idx ON entities (kind);

CREATE TABLE chunk_entities (
    chunk_id     text NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    entity_id    text NOT NULL REFERENCES entities (id) ON DELETE RESTRICT,
    mention_text text NOT NULL,
    char_offset  integer NOT NULL CHECK (char_offset >= 0),
    PRIMARY KEY (chunk_id, entity_id, char_offset)
);

CREATE INDEX chunk_entities_entity_idx ON chunk_entities (entity_id);

-- ── 3. 知识 ────────────────────────────────────────────────────────────────

CREATE TABLE claims (
    id           text PRIMARY KEY,
    version_id   text NOT NULL REFERENCES document_versions (id) ON DELETE RESTRICT,
    subject_id   text NOT NULL REFERENCES entities (id) ON DELETE RESTRICT,
    predicate    text NOT NULL
                 CHECK (predicate IN (
                     'improves', 'outperforms', 'achieves', 'replaces', 'requires',
                     'uses', 'causes', 'is_a', 'part_of', 'proposes', 'evaluates_on',
                     'contradicts'
                 )),
    object_id    text REFERENCES entities (id) ON DELETE RESTRICT,
    object_text  text,
    conditions   text NOT NULL DEFAULT '',
    claim_text   text NOT NULL CHECK (length(claim_text) > 0),
    confidence   text NOT NULL DEFAULT 'EXTRACTED'
                 CHECK (confidence IN ('EXTRACTED', 'INFERRED')),
    status       text NOT NULL DEFAULT 'draft'
                 CHECK (status IN ('draft', 'active', 'retracted')),
    created_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT claims_subject_object_differ CHECK (object_id IS NULL OR object_id <> subject_id),
    CONSTRAINT claims_object_presence CHECK (
        (predicate = 'achieves' AND object_id IS NULL AND object_text IS NOT NULL)
        OR (predicate <> 'achieves' AND object_id IS NOT NULL)
    )
);

CREATE INDEX claims_version_idx ON claims (version_id);
CREATE INDEX claims_subject_idx ON claims (subject_id);
CREATE INDEX claims_object_idx ON claims (object_id);
CREATE INDEX claims_predicate_idx ON claims (predicate);
CREATE INDEX claims_active_idx ON claims (version_id) WHERE status = 'active';

CREATE TABLE evidence (
    id          text PRIMARY KEY,
    claim_id    text NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
    chunk_id    text NOT NULL REFERENCES chunks (id) ON DELETE RESTRICT,
    quote       text NOT NULL CHECK (length(quote) >= 5),
    page        integer NOT NULL CHECK (page >= 1),
    stance      text NOT NULL DEFAULT 'support'
                CHECK (stance IN ('support', 'refute', 'context')),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX evidence_claim_idx ON evidence (claim_id);
CREATE INDEX evidence_chunk_idx ON evidence (chunk_id);

-- ── 4. 笔记 ────────────────────────────────────────────────────────────────

CREATE TABLE notes (
    id                  text PRIMARY KEY,
    author              text NOT NULL DEFAULT 'user',
    title               text NOT NULL,
    body                text NOT NULL,
    linked_version_ids  text[] NOT NULL DEFAULT '{}',
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz
);

CREATE TABLE note_claims (
    note_id   text NOT NULL REFERENCES notes (id) ON DELETE CASCADE,
    claim_id  text NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
    relation  text NOT NULL DEFAULT 'cites'
              CHECK (relation IN ('agrees', 'disagrees', 'extends', 'cites')),
    PRIMARY KEY (note_id, claim_id, relation)
);

-- ── 5. 记忆 ────────────────────────────────────────────────────────────────

CREATE TABLE memory_events (
    id              text PRIMARY KEY,
    user_id         text NOT NULL,
    namespace       text NOT NULL,
    memory_key      text NOT NULL,
    operation       text NOT NULL CHECK (operation IN ('add', 'update', 'retract', 'noop')),
    value           text NOT NULL,
    valid_from      timestamptz NOT NULL,
    valid_to        timestamptz,
    recorded_at     timestamptz NOT NULL DEFAULT now(),
    source_turn_id  text,
    source_chunk_id text,
    supersedes_id   text REFERENCES memory_events (id) ON DELETE RESTRICT,
    status          text NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'superseded', 'retracted')),
    CONSTRAINT memory_valid_range CHECK (valid_to IS NULL OR valid_to >= valid_from),
    CONSTRAINT memory_update_needs_supersedes CHECK (
        operation NOT IN ('update', 'retract') OR supersedes_id IS NOT NULL
    ),
    CONSTRAINT memory_add_no_supersedes CHECK (
        operation <> 'add' OR supersedes_id IS NULL
    ),
    CONSTRAINT memory_retract_status CHECK (
        operation <> 'retract' OR status = 'retracted'
    )
);

CREATE INDEX memory_lookup_idx ON memory_events (user_id, namespace, memory_key, recorded_at DESC);
CREATE INDEX memory_active_idx ON memory_events (user_id, namespace)
    WHERE status = 'active';
CREATE INDEX memory_supersedes_idx ON memory_events (supersedes_id);

-- ── 6. 不可变约束（铁律落到数据库）─────────────────────────────────────────

CREATE OR REPLACE FUNCTION recallgraph_forbid_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION '% 是只追加表，禁止删除（用 status 或 valid_to 打点）', TG_TABLE_NAME;
    END IF;

    -- memory_events 允许唯一例外：只改 status
    IF TG_TABLE_NAME = 'memory_events'
       AND (to_jsonb(NEW) - 'status') = (to_jsonb(OLD) - 'status') THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION '% 是不可变表，禁止修改（请生成新版本）', TG_TABLE_NAME;
END;
$$;

CREATE TRIGGER document_versions_immutable
    BEFORE UPDATE OR DELETE ON document_versions
    FOR EACH ROW EXECUTE FUNCTION recallgraph_forbid_mutation();

CREATE TRIGGER chunks_immutable
    BEFORE UPDATE OR DELETE ON chunks
    FOR EACH ROW EXECUTE FUNCTION recallgraph_forbid_mutation();

CREATE TRIGGER evidence_immutable
    BEFORE UPDATE OR DELETE ON evidence
    FOR EACH ROW EXECUTE FUNCTION recallgraph_forbid_mutation();

CREATE TRIGGER memory_events_append_only
    BEFORE UPDATE OR DELETE ON memory_events
    FOR EACH ROW EXECUTE FUNCTION recallgraph_forbid_mutation();

-- ── 7. 常用视图 ────────────────────────────────────────────────────────────

-- 只返回"有证据支撑"的断言：检索与评测都走这个视图。
CREATE VIEW v_active_claims AS
SELECT c.*
FROM claims c
WHERE c.status = 'active'
  AND EXISTS (
      SELECT 1 FROM evidence e
      WHERE e.claim_id = c.id AND e.stance IN ('support', 'refute')
  );

COMMIT;

-- ── 8. 向量召回（需要 pgvector）────────────────────────────────────────────
-- 扩展必须在建向量列之前存在；镜像自带，普通 Postgres 需自行安装。
CREATE EXTENSION IF NOT EXISTS vector;
-- 向量列单独放 chunk_vectors（见 004_dense.sql），不再直接挂在 chunks 上：
-- 换 embedding 模型时整表重建，不必 ALTER 主表。
-- ALTER TABLE chunks ADD COLUMN embedding vector(1024);   -- BGE-M3
-- CREATE INDEX chunks_embedding_idx ON chunks
--     USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);