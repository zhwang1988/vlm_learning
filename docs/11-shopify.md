# 11 · Shopify App 产品化

> 对应 Day 37–Day 45
>
> ⚠️ Shopify 平台更新较快，本篇讲的是**稳定的架构概念与流程**。具体字段、审核条款、API 版本请以 https://shopify.dev 官方文档为准（Day 37 第一件事就是打开它对照本篇）。

## 概念

### 三种 App 类型

| 类型 | 面向 | 上架 | 适用 |
|---|---|---|---|
| **Public App** | 任意店铺 | 需 App Store 审核 | 做 SaaS 卖给别人 ← 本项目目标 |
| **Custom App** | 单个店铺 | 无需审核 | 自用、大客户定制 |
| Private App | 单个店铺 | 已逐步弃用 | 用 Custom App 替代 |

**重要**：Public App 审核比较严格（功能完整、性能、隐私政策、卸载 webhook 齐全）。**不要一上来就冲审核**——先在开发店跑通，把功能做扎实。

### 开发环境

```
工具链：
  Shopify CLI          —— 创建/运行/部署 App 项目
  Partner Account      —— 免费注册，创建开发商店
  Development Store    —— 无限期的免费测试店，可装未审核 App
  Ngrok / Cloudflare Tunnel —— 本地开发时把 localhost 暴露成 HTTPS
```

**Day 37 第一件事**：
1. 注册 Partner 账号
2. 创建一个 Development Store（选「测试数据」）
3. 往店里塞 10–20 个商品，每个商品多张图（这是你的数据源）
4. `shopify app init` 生成项目骨架

### 认证链路（必须彻底搞懂）

```
① 商家点击「安装 App」
   ↓
② 你的 App 收到 /auth?shop=xxx.myshopify.com
   ↓
③ 重定向到 Shopify 授权页（带 client_id, scope, redirect_uri, state, nonce）
   ↓
④ 商家同意 → 回调到你的 redirect_uri，带 code 和 hmac
   ↓
⑤ 你校验 HMAC（用 client_secret 对 query 参数签名验证）
   ↓
⑥ 用 code 换 access_token（POST /admin/oauth/access_token）
   ↓
⑦ 存 token 到数据库（关联 shop domain）
   ↓
⑧ 后续所有 API 调用带 X-Shopify-Access-Token
```

**关键概念区分**（面试常问）：

| | Access Token | Session Token |
|---|---|---|
| 用在哪 | 服务端调 Admin API | 前端调你的后端 |
| 生命周期 | 长期（除非卸载） | 短期（JWT，约 1 分钟） |
| 载体 | HTTP header | JWT |
| 用途 | 代表店铺的身份 | 代表「当前这个商家用户」的身份 |

**现代做法**：前端用 App Bridge 自动在请求里带 session token，后端用 `shopify-app-*` 库校验 JWT。这样不需要自己管理 session cookie。

### Admin API：用 GraphQL，别用 REST

```
REST（legacy）：/admin/api/2024-10/products.json
   限流：漏桶，每秒 2 请求（基础版），超了返回 429

GraphQL（推荐）：/admin/api/2024-10/graphql.json
   限流：计算成本（calculated cost），基于查询复杂度
   返回 extensions.cost 告诉你本次消耗和剩余
```

**GraphQL 的实践要点**：

```graphql
# 必须处理分页（默认最多返回 50/250 条）
query Products($cursor: String) {
  products(first: 100, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      title
      description
      variants(first: 50) { nodes { id title price inventoryQuantity } }
      images(first: 20) { nodes { url altText width height } }
    }
  }
}
```

**限流处理**：读 `extensions.cost.throttleStatus.currentlyAvailable`，低于阈值就 sleep。**不要用固定 sleep 硬编码**。

### 三个关键技术组件

```
① App Bridge          前端 SDK，让嵌入在 Shopify 后台的 iframe 里也能正常导航/弹窗/拿 session token
② Theme App Extension 往店铺前台主题里注入 UI 的官方方式（app block / embed block）
                     —— 你的客服聊天挂件就用这个
③ Polaris             Shopify 的官方 UI 组件库，让后台界面风格统一（审核加分）
```

**Day 39 的聊天挂件**属于 Theme App Extension，用 Liquid + JS 实现：

```
extensions/chat-widget/
├── blocks/chat_widget.liquid      # app block 定义
├── assets/chat.js                 # 挂件逻辑（上传图、调你的 API）
├── assets/chat.css
├── locales/
└── shopify.extension.toml
```

### Webhook：让数据保持同步

你的 RAG 索引必须和店铺商品保持一致。靠定时轮询又慢又费配额，正确做法是订阅 webhook：

| 主题 | 用途 |
|---|---|
| `products/create` / `products/update` / `products/delete` | 增量更新商品向量索引 ⭐ |
| `orders/create` | 关联订单到会话 |
| `orders/updated` / `refunds/create` | 更新会话状态 |
| `app/uninstalled` | **必须**：清理店铺数据 |
| `customers/data_request` | **GDPR 必须** |
| `customers/redact` | **GDPR 必须** |
| `shop/redact` | **GDPR 必须** |

**两个必须处理的点**：

1. **HMAC 校验**：用 `X-Shopify-Hmac-Sha256` 头 + client_secret 验证请求真的来自 Shopify。不校验等于把接口裸奔。
2. **幂等 + 快速响应**：Shopify 要求 5 秒内返回 200，超时会重试。所以**收到后立刻入队，异步处理**，用 `X-Shopify-Webhook-Id` 做幂等键。

### Billing API

```
计费模型（Shopify 支持的类型）：
  - 一次性收费        appPurchaseOneTimeCreate
  - 订阅制（月/年）    appSubscriptionCreate         ← 最常用
  - 按用量计费        usage-based + capped amount   ← 适合 AI（按会话数）

流程：
  ① 后台点「升级套餐」
  ② App 调 appSubscriptionCreate → 拿到 confirmationUrl
  ③ 跳转商家确认 → 回调回你的 App
  ④ 你必须校验：商家确实接受了（查 currentAppInstallation.activeSubscriptions）
  ⑤ 用量计费：appUsageRecordCreate 上报用量
```

**按用量计费的关键坑**：
- 有 **capped amount**（上限），商户可设置
- 上报要**幂等**（用唯一 key），否则重复扣费
- 一个计费周期结束后才能结算，不是实时扣

**Day 41 的简化方案**：先做「免费 14 天 + 标准版 ¥xx/月（含 N 次会话）+ 超出部分按用量」。够用了。

### GDPR 与隐私

Public App 必须实现三个强制 webhook，且必须有隐私政策 URL。

**客服场景的额外合规负担**（比一般 App 重）：
- 用户聊天里会有**订单号、手机号、地址**
- 上传的图片可能有**人脸**（模特图、用户自拍）
- 聊天记录属于**个人数据**

**Day 44 要做的事**：
- PII 脱敏：日志里订单号/手机号/地址必须打码（`138****5678`）
- 图片不落地，或加密存储 + 定期清理
- 数据保留策略：明确告诉商家存多久
- 用户数据删除接口（配合 `customers/redact`）

## 工程细节

### 项目结构（Python 后端方案）

Shopify 官方模板是 Node/Remix，但**你的 AI 栈是 Python**，所以走「前端 Remix/原生 JS + 后端 Python」的混合方案：

```
src/shopify/
├── app.py                 # FastAPI 主应用
├── auth.py                # OAuth + session token 校验
├── client.py              # Admin GraphQL 客户端（分页/重试/限流）
├── webhooks.py            # webhook 接收与分发
├── billing.py             # 订阅与用量上报
├── indexer.py             # 商品 → 向量索引同步
├── models.py              # DB 模型（shop, session, subscription, message）
├── extensions/
│   └── chat-widget/       # Theme App Extension
└── admin-ui/              # 后台界面（可用 React + Polaris）
```

### OAuth 的 HMAC 校验（最容易写错的地方）

```python
import hmac, hashlib
from urllib.parse import parse_qsl

def verify_hmac(query_string: str, secret: str) -> bool:
    params = dict(parse_qsl(query_string))
    received = params.pop("hmac", None)
    if not received:
        return False
    # 关键：把剩余参数按 key 排序后拼接，不是用原始 query string
    message = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    digest = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, received)
```

**常见错误**：直接把整个 query string 拿去算 HMAC。必须先把 `hmac` 参数摘掉，且**排序后重新拼接**。

（注意：OAuth 首次回调的校验规则和后续「verify request」的规则有细微差别，以官方文档为准。）

### 店铺数据模型

```sql
CREATE TABLE shops (
    id              SERIAL PRIMARY KEY,
    shop_domain     TEXT UNIQUE NOT NULL,
    access_token    TEXT NOT NULL,        -- 生产环境要加密
    scope           TEXT,
    plan            TEXT DEFAULT 'free',
    installed_at    TIMESTAMPTZ DEFAULT now(),
    uninstalled_at  TIMESTAMPTZ
);

CREATE TABLE sessions (
    id              TEXT PRIMARY KEY,
    shop_id         INT REFERENCES shops(id),
    customer_id     TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE messages (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    role            TEXT,                 -- user / assistant / tool
    content         TEXT,
    image_urls      TEXT[],
    tool_calls      JSONB,
    tokens_in       INT,
    tokens_out      INT,
    latency_ms      INT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE usage_records (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT REFERENCES shops(id),
    idempotency_key TEXT UNIQUE,           -- 防重复计费
    session_id      TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE product_index (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         INT REFERENCES shops(id),
    product_id      TEXT,
    embedding       vector(768),           -- pgvector
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(shop_id, product_id)
);
```

**`idempotency_key` 唯一约束**是防重复计费的最后一道防线。

### 上线检查清单（Day 45）

```
功能
  ☐ 安装流程顺畅，无需手动步骤
  ☐ 卸载后数据清理（app/uninstalled webhook）
  ☐ 前台挂件在主流主题（Dawn）下正常显示
  ☐ 移动端可用
  ☐ 首次进入有引导（不能是空白页）

合规
  ☐ 隐私政策页面（公网可访问）
  ☐ 三个 GDPR webhook 实现并返回 200
  ☐ HMAC 校验全部到位
  ☐ 请求的 scope 最小化（只要真的需要的）

性能
  ☐ 后台页面首屏 < 3s
  ☐ 前台挂件不拖慢店铺（异步加载、体积 < 100KB）
  ☐ API 调用有缓存、有限流退避

AI 特有
  ☐ 明确告知用户「你在和 AI 对话」
  ☐ 有转人工入口
  ☐ 有免责声明（AI 可能出错）
  ☐ 敏感问题（退款金额争议等）自动转人工
  ☐ 日志中 PII 已脱敏
```

**AI 相关的四条是审核的新增关注点**，很多 AI App 在这上面被拒。**主动加免责声明和转人工入口**。

## 自检问题

1. Public / Custom / Private 三种 App 的区别是什么？本项目做哪种？
2. Access Token 和 Session Token 的区别？各自用在哪？
3. 为什么推荐 GraphQL 而不是 REST？限流机制有什么不同？
4. OAuth 的 HMAC 校验为什么不能直接用原始 query string？
5. 三个强制 GDPR webhook 是什么？
6. Webhook 收到后为什么要「立刻返回 200 再异步处理」？
7. 按用量计费的两个关键坑是什么？
8. AI 类 App 审核时额外要注意哪四点？

---

**上一篇**：[10-agent.md](10-agent.md) · **下一篇**：[12-papers.md](12-papers.md)
