-- 词项倒排表：chunk → 词项 → 词频。
-- 用途：BM25 关键词通道。检索时不再读 PDF，只查这张表 + chunks。
-- 与 chunks 不同，这张表是"派生物"，可以删掉重建（reindex），因此不加不可变触发器。

BEGIN;

CREATE TABLE chunk_terms (
    chunk_id text NOT NULL REFERENCES chunks (id) ON DELETE RESTRICT,
    term     text NOT NULL CHECK (length(term) > 0 AND term = lower(term)),
    tf       integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (chunk_id, term)
);

CREATE INDEX chunk_terms_term_idx ON chunk_terms (term);
CREATE INDEX chunk_terms_chunk_idx ON chunk_terms (chunk_id);

COMMIT;
