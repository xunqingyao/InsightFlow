-- 不同模型的向量维度不同（bge-small-en=384，bge-small-zh=512），
-- 固定 vector(384) 装不下中文模型。改为不指定维度 + 放弃 HNSW：
-- 当前规模（万级段落）顺序扫描足够快，且换模型不再需要改表。
BEGIN;

ALTER TABLE chunk_vectors ALTER COLUMN embedding TYPE vector;
DROP INDEX IF EXISTS chunk_vectors_embedding_idx;

COMMIT;
