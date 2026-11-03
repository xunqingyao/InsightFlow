-- 向量通道：每个段落一条向量。
-- model 字段决定"向量是谁算的"：换模型时旧向量必须整批重算，不能混用。
BEGIN;

CREATE TABLE chunk_vectors (
    chunk_id   text PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    model      text NOT NULL,
    embedding  vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- HNSW：近似最近邻。小库上也可以走顺序扫描，但先建好省得以后改表。
CREATE INDEX chunk_vectors_embedding_idx
    ON chunk_vectors USING hnsw (embedding vector_cosine_ops);

COMMIT;
