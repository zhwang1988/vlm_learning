"""
数据库模型（PostgreSQL + pgvector）。

对应讲义 docs/11-shopify.md「店铺数据模型」，对应计划 Day 38。

用 SQL 而非 ORM，是为了让你看清每一列为什么存在。
生产环境可以换 SQLAlchemy / Prisma。

重点看三处：
  · shops.access_token        生产环境必须加密
  · messages.image_urls       客服场景要存图，注意隐私
  · usage_records.idempotency_key  唯一约束 = 防重复计费的最后一道防线
"""

from __future__ import annotations

import os
from pathlib import Path

SCHEMA = """
-- 扩展：向量检索
CREATE EXTENSION IF NOT EXISTS vector;
-- 扩展：UUID
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 店铺
-- ============================================================
CREATE TABLE IF NOT EXISTS shops (
    id              SERIAL PRIMARY KEY,
    shop_domain     TEXT UNIQUE NOT NULL,
    access_token    TEXT NOT NULL,          -- ⚠️ 生产必须加密（Fernet/KMS）
    scope           TEXT,
    plan            TEXT DEFAULT 'free',
    shopify_shop_id TEXT,
    currency        TEXT DEFAULT 'CNY',
    installed_at    TIMESTAMPTZ DEFAULT now(),
    uninstalled_at  TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_shops_domain ON shops(shop_domain);

-- ============================================================
-- 会话
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,        -- 我们自己的 session id
    shop_id         INT REFERENCES shops(id) ON DELETE CASCADE,
    customer_id     TEXT,                    -- Shopify customer id（可能为空）
    source          TEXT DEFAULT 'widget',   -- widget / admin / api
    user_agent      TEXT,
    started_at      TIMESTAMPTZ DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    n_messages      INT DEFAULT 0,
    resolved        BOOLEAN,                 -- 是否解决（人工标注或推断）
    escalated       BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_sessions_shop ON sessions(shop_id, started_at DESC);

-- ============================================================
-- 消息
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,           -- user / assistant / tool / system
    content         TEXT,
    image_urls      TEXT[],                  -- ⚠️ 隐私：用户上传图可能有脸
    tool_calls      JSONB,                   -- 工具调用记录（审计用）
    visual_evidence JSONB,                   -- 提取的视觉证据
    tokens_in       INT DEFAULT 0,
    tokens_out      INT DEFAULT 0,
    latency_ms      INT,
    model           TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- ============================================================
-- 用量记录（计费）
-- ============================================================
CREATE TABLE IF NOT EXISTS usage_records (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT REFERENCES shops(id) ON DELETE CASCADE,
    -- ⭐ 唯一约束是防重复计费的最后一道防线
    idempotency_key TEXT UNIQUE NOT NULL,
    session_id      TEXT,
    amount          NUMERIC(10,4) DEFAULT 0,
    reported        BOOLEAN DEFAULT FALSE,   -- 是否已上报给 Shopify
    reported_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_usage_shop ON usage_records(shop_id, created_at DESC);

-- ============================================================
-- 订阅
-- ============================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id                     BIGSERIAL PRIMARY KEY,
    shop_id                INT REFERENCES shops(id) ON DELETE CASCADE,
    shopify_subscription_id TEXT,
    plan                   TEXT,
    status                 TEXT,             -- active/cancelled/expired/frozen
    trial_ends_at          TIMESTAMPTZ,
    current_period_end     TIMESTAMPTZ,
    capped_amount          NUMERIC(10,2) DEFAULT 0,
    created_at             TIMESTAMPTZ DEFAULT now(),
    updated_at             TIMESTAMPTZ DEFAULT now(),
    UNIQUE(shop_id, shopify_subscription_id)
);

-- ============================================================
-- 商品向量索引
-- ============================================================
CREATE TABLE IF NOT EXISTS product_index (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT REFERENCES shops(id) ON DELETE CASCADE,
    product_id      TEXT NOT NULL,
    title           TEXT,
    category        TEXT,
    image_url       TEXT,
    local_image     TEXT,
    attributes      JSONB,
    embedding       vector(512),             -- CLIP ViT-B/32 的维度
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(shop_id, product_id, image_url)
);
-- 近似最近邻索引（数据量大时必需）
CREATE INDEX IF NOT EXISTS idx_product_embedding
    ON product_index USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_product_shop ON product_index(shop_id);

-- ============================================================
-- 知识库
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_docs (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT REFERENCES shops(id) ON DELETE CASCADE,
    doc_id          TEXT,
    title           TEXT,
    content         TEXT NOT NULL,
    source          TEXT,                    -- policy / faq / size_chart
    embedding       vector(512),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(shop_id, doc_id)
);

-- ============================================================
-- Webhook 幂等
-- ============================================================
CREATE TABLE IF NOT EXISTS webhook_events (
    id              BIGSERIAL PRIMARY KEY,
    webhook_id      TEXT UNIQUE NOT NULL,
    topic           TEXT,
    shop_domain     TEXT,
    payload         JSONB,
    processed       BOOLEAN DEFAULT FALSE,
    processed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- 人工工单
-- ============================================================
CREATE TABLE IF NOT EXISTS tickets (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT REFERENCES shops(id) ON DELETE CASCADE,
    session_id      TEXT REFERENCES sessions(id),
    reason          TEXT,
    status          TEXT DEFAULT 'open',
    assigned_to     TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    resolved_at     TIMESTAMPTZ
);

-- ============================================================
-- 审计日志（合规要求）
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT,
    actor           TEXT,
    action          TEXT,                    -- install/uninstall/data_export/redact
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);
"""

# GDPR 数据删除需要动的表（顺序重要：先删叶子表）
REDACT_ORDER = [
    ("messages", "session_id IN (SELECT id FROM sessions WHERE shop_id=%s AND customer_id=%s)"),
    ("sessions", "shop_id=%s AND customer_id=%s"),
]

SHOP_REDACT_ORDER = [
    "messages", "usage_records", "product_index", "knowledge_docs",
    "tickets", "subscriptions", "sessions", "shops",
]


def init_db(dsn: str | None = None):
    """建表。需要 psycopg。"""
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        print("缺少 DATABASE_URL。示例：")
        print("  DATABASE_URL=postgresql://user:pass@localhost:5432/cx")
        return False
    try:
        import psycopg
    except ImportError:
        print("需要 psycopg: pip install 'psycopg[binary]'")
        return False

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()
    print("✓ 建表完成")
    return True


def save_schema(out: str | Path = "configs/schema.sql"):
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(SCHEMA, encoding="utf-8")
    print(f"✓ schema → {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="导出 schema.sql")
    ap.add_argument("--init", action="store_true", help="建表")
    args = ap.parse_args()

    if args.dump:
        save_schema()
    elif args.init:
        init_db()
    else:
        print(SCHEMA)
        print()
        print("=" * 70)
        print("三个关键设计（Day 38 的验收点）")
        print("=" * 70)
        print("""
1. usage_records.idempotency_key 有 UNIQUE 约束
   → 即使应用层幂等失败，数据库也会拦住重复计费。
   → 这是最后一道防线。

2. messages.image_urls 存的是数组
   → 多图样本很常见（用户可能一次发 3 张）。
   → GDPR 删除时要连图片文件一起删（face 是个人数据）。

3. product_index 用 pgvector 的 ivfflat 索引
   → 商品几万个时，暴力检索会慢。
   → ivfflat 是近似最近邻，牺牲一点召回换几十倍速度。
   → lists 参数约等于 sqrt(行数)。
""")
