-- 记忆：一条 key 同时只能有一个 active 版本。
-- 靠数据库保证，不靠代码自觉：并发写或写错逻辑都会在这里撞墙。
CREATE UNIQUE INDEX IF NOT EXISTS memory_one_active_idx
    ON memory_events (user_id, namespace, memory_key)
    WHERE status = 'active';