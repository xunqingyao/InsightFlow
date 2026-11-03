-- 记录每条断言由哪个抽取器产生。
-- 目的：桩抽取器 → LLM 抽取器切换时，旧断言可以标记为 retracted 并保留审计，
-- 新断言用新 extractor 值追加，不需要删除历史。

BEGIN;

ALTER TABLE claims ADD COLUMN extractor text NOT NULL DEFAULT 'unknown';

CREATE INDEX claims_extractor_idx ON claims (version_id, extractor);

COMMIT;
