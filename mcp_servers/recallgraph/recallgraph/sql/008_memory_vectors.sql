-- 记忆的向量索引：让“改主意”能召回“调整方向”这类同义说法。
-- 一条记忆事件一个向量（key + value 拼起来编码）；重建时按 model 整批删除重写。
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_event_id text PRIMARY KEY REFERENCES memory_events (id) ON DELETE RESTRICT,
    model           text NOT NULL,
    embedding       vector NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_vectors_model_idx ON memory_vectors (model);