-- 允许同一段落保存多个模型的向量，便于 A/B（英文模型 vs 中文模型）。
-- 主键从 chunk_id 改为 (chunk_id, model)。
BEGIN;

ALTER TABLE chunk_vectors DROP CONSTRAINT chunk_vectors_pkey;
ALTER TABLE chunk_vectors ADD CONSTRAINT chunk_vectors_pkey PRIMARY KEY (chunk_id, model);

COMMIT;
