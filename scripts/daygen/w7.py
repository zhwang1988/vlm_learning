"""Week 7 · Shopify SaaS 产品化（Day 37–42）

这一周的核心矛盾：**让模型在真实的商家后台里跑起来**。
省钱好消息：**这一周全部可以在本地做完**（`src/shopify/` 零 GPU 依赖）。
"""
from __future__ import annotations


def D(**kw):
    return kw


DAYS = [
    # ======================================================================
    D(
        week=7, n=37, slug="shopify_api", title="Shopify 生态与 API",
        where="local",
        files="`src/shopify/client.py`、`docs/11-shopify.md`",
        prereq="Day 36；有一个 Shopify Partners 账号 + 开发测试店",
        goal="封装 Admin GraphQL 客户端（分页、重试、限流退避），"
             "能从测试店拉到商品列表并落库；并说清三种 App 类型的审核要求差异。",
        read=[
            "`docs/11-shopify.md` 第 1–2 节（App 类型 / Admin GraphQL / REST vs GraphQL / rate limit）",
            "`src/shopify/client.py` 的 `query()` —— 重点读它怎么处理 `throttleStatus`",
        ],
        think=[
            "Shopify 为什么把 REST 判了「过时」转向 GraphQL？（提示：按点数计费 vs 按请求数）",
            "`extensions.cost.throttleStatus.currentlyAvailable` 是干什么用的？不看它会怎样？",
            "public / custom / private 三种 App 分别适合什么场景？审核要求差在哪？",
        ],
        write_title="`src/shopify/client.py`（已给实现）",
        write_rows=[
            ("`ShopifyClient.query()`", "发 GraphQL；**读返回的 cost 信息**决定要不要退避"),
            ("限流退避", "点数不足时 sleep 而不是硬撞 429"),
            ("`paginate()`", "异步生成器翻页 —— 别手动拼 cursor"),
            ("`PRODUCTS_QUERY`", "只取需要的字段（GraphQL 的字段选择直接影响点数消耗）"),
            ("`to_internal_product()`", "外部数据结构 → 内部结构（**隔离层，很重要**）"),
        ],
        write_note="> 今天最该养成的习惯：**外部 API 的数据结构不要渗透到你的业务代码里**。\n"
                   "> `to_internal_product()` 这一层看着多余，等你换了个电商平台就知道它多值钱。",
        run=[
            ("python -m src.shopify.client", "自检：GraphQL 查询构造 + 限流解析"),
            ("python -m src.shopify.client --help", "看有哪些参数（接真实店铺时用）"),
        ],
        expect="""
[client] 构造 Admin GraphQL 客户端
[query] POST https://demo.myshopify.com/admin/api/2026-07/graphql.json
        extensions.cost: requestedQueryCost=42, currentlyAvailable=1968
[parse] throttleStatus 正常，无需退避
[demo] 商品样例（3 条）：
   gid://shopify/Product/1001  米白色针织衫   ¥199   3 个变体
   gid://shopify/Product/1002  奶白色开衫     ¥259   2 个变体
   已转为内部结构: {'id': '1001', 'title': '米白色针织衫', ...}
✓ 自检通过（无 token 时只验查询构造，不发请求）
""",
        accept=[
            "`python -m src.shopify.client` 自检通过",
            "能从测试店拉到 ≥10 个商品并落库（JSON / SQLite 都行）",
            "能说清三种 App 类型的差异和审核要求",
            "知道 `throttleStatus` 怎么读、为什么要退避而不是硬撞",
        ],
        pits=[
            "用 REST API —— 已过时且限流更严（每秒 2 请求）。新项目一律 GraphQL。",
            "**不看 throttle 直接发请求** —— 点数耗尽后连续 429，你的同步任务卡死。",
            "GraphQL 里 `*` 式要全部字段 —— 点数消耗暴涨，一次查询就把额度用光。",
            "cursor 分页用错（把 `endCursor` 当 `startCursor`）—— 死循环翻同一页。",
            "access token 硬编码进代码 —— 一定会上 git。用 `.env`，且 `.gitignore` 要包含它。",
        ],
        nb=[
            ("md", "## 1. 自检（不需要真实店铺）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.shopify.client"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 手写一遍限流退避逻辑\n\n别看源码，自己写一个。核心：**读到低点数就 sleep，而不是等 429**。"),
            ("code", '''import time

def backoff_sleep(cost_info):
    """点数不足就等，而不是被 429 打回来。"""
    available = cost_info.get("currentlyAvailable", 2000)
    restore_rate = cost_info.get("restoreRate", 50)     # 点/秒
    if available >= 200:
        return 0.0
    wait = (200 - available) / max(restore_rate, 1)
    print(f"  点数 {available} 偏低 → 等 {wait:.2f}s")
    return wait

for ci in ({"currentlyAvailable": 1968, "restoreRate": 50},
           {"currentlyAvailable": 120, "restoreRate": 50},
           {"currentlyAvailable": 12, "restoreRate": 50}):
    print(ci)
    t0 = time.time()
    w = backoff_sleep(ci)
    print(f"  → 需等待 {w:.2f}s（不真的 sleep）")'''),
            ("md", "## 3. 隔离层设计"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.shopify.client import to_internal_product

raw = {"id": "gid://shopify/Product/1001", "title": "米白色针织衫",
       "variants": {"edges": [{"node": {"sku": "SKU-1001", "price": "199.00"}}]}}
internal = to_internal_product(raw)
print("外部结构 →", raw)
print("内部结构 →", internal)
print("\\n→ 为什么要转换：Shopify 改字段名时，你只需要改这一个函数")'''),
        ],
        next_day_hint="`days/day-38.md` —— OAuth 安装流程与 App 骨架",
    ),

    # ======================================================================
    D(
        week=7, n=38, slug="shopify_oauth", title="OAuth 与应用骨架",
        where="local",
        files="`src/shopify/app.py`、`src/shopify/auth.py`",
        prereq="Day 37；已建好 Shopify App 拿到 API key / secret",
        goal="把 OAuth 安装流程跑通：`/auth` → 用户授权 → 回调 → 换 access token → 存库；"
             "并说清 **HMAC 校验** 和 **session token** 的关系。",
        read=[
            "`docs/11-shopify.md` 第 3 节（OAuth 与鉴权）",
            "`src/shopify/auth.py` 的 `verify_hmac` —— **这份文件里埋了三个坑，注释都写了**",
        ],
        think=[
            "HMAC 校验时为什么必须先**剔除 `hmac` 参数**、再按 key 排序？",
            "Access token 和 Session token 有什么区别？各自活多久？",
            "为什么必须校验 `shop` 参数的域名格式？（提示：SSRF）",
        ],
        write_title="`src/shopify/auth.py` + `src/shopify/app.py`（已给实现）",
        write_rows=[
            ("`verify_hmac()`", "**剔 hmac → 按 key 排序 → 拼接 → sha256**，顺序不能错"),
            ("`verify_webhook_hmac()`", "webhook 的 HMAC 是 **base64** 不是 hex（另一个坑）"),
            ("`validate_shop()`", "正则校验域名，防 SSRF —— **绝不能省**"),
            ("`build_install_url()`", "生成授权链接（含 state 防 CSRF）"),
            ("`exchange_code_for_token()`", "拿 code 换 access token"),
            ("`verify_session_token()`", "校验 JWT（App Bridge 传来的），验签 + 验 exp + 验 aud"),
            ("`TokenStore`", "token 存库（**别存文件、别进 git**）"),
        ],
        run=[
            ("python -m src.shopify.auth", "自检：HMAC 正确/错误/两个坑 + SSRF 拦截"),
            ("python -m src.shopify.app --dev --port 8080", "起 App（本地开发，放宽 CORS）"),
            ("open http://localhost:8080/auth?shop=demo.myshopify.com",
             "走一次安装流程（需要真实测试店）"),
        ],
        expect="""
[1] HMAC 校验
    正确签名（乱序 query 也应通过）→ True
    参数顺序颠倒后仍通过        → True
    坑1 用原始顺序直接算        → False （应该 False）✓
    坑2 忘了剔除 hmac 参数      → False （应该 False）✓

[2] SSRF 防护
    demo.myshopify.com        → True
    evil.com                  → False ✓
    demo.myshopify.com.evil.io → False ✓

[3] 安装流程
    GET /auth?shop=demo.myshopify.com
    → 302 到 Shopify 授权页（带 state=xxx）
    → 回调 /auth/callback?code=...&hmac=...&state=xxx
    → 换到 access token，已入库（shops 表）
    ✓ 安装完成
""",
        accept=[
            "`python -m src.shopify.auth` 自检全部通过，**两个 HMAC 坑都被拦住**",
            "能在测试店完成一次完整安装（`/auth` → 授权 → 回前台正常）",
            "能解释 HMAC 校验和 session token 的关系（谁在前、各自防什么）",
            "知道 SSRF 是什么、`validate_shop` 拦的是哪种攻击",
        ],
        pits=[
            "**HMAC 没剔除 `hmac` 参数** —— 最常见的错。带着 hmac 去算哈希，签名永远不对。",
            "**忘了按 key 排序** —— query string 的参数顺序不保证，不排序就是碰运气。",
            "**webhook 的 HMAC 用了 hex 解码** —— Shopify 的 webhook 签名是 **base64**，"
            "和 OAuth 回调的 hex 不是一套，混用直接失败。",
            "**不校验 shop 域名** —— 攻击者传 `shop=evil.com`，你的服务器就拿着 token 去请求 evil.com 了。",
            "access token 写进文件或日志 —— 这是能操作商家店铺的凭证，泄露等于店铺被接管。",
            "state 参数没校验 —— CSRF：别人诱导店主点一个链接，就把他的店绑到你的 app 上。",
        ],
        nb=[
            ("md", "## 1. HMAC 自检（两个坑都要拦住）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.shopify.auth"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 亲手踩一次坑\n\n先写「错误」的校验逻辑，看它为什么失败 —— 比直接看对的实现印象深十倍。"),
            ("code", '''import hmac, hashlib
from urllib.parse import parse_qsl

secret = "test_secret_abc"
qs = "host=YWRtaW4&shop=myshop.myshopify.com&code=abc123&timestamp=1700000000&state=xyz"

# 正确做法：剔除 hmac（这里没有）→ 按 key 排序 → 拼接
params = dict(parse_qsl(qs))
correct_msg = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
print("正确拼接:", correct_msg)

# 错误做法 1：直接用原始 query string
wrong_msg = qs
print("错误拼接:", wrong_msg)
print("两者相同吗:", correct_msg == wrong_msg, "← 顺序不同就会不同，所以排序是必须的")

h_correct = hmac.new(secret.encode(), correct_msg.encode(), hashlib.sha256).hexdigest()
h_wrong   = hmac.new(secret.encode(), wrong_msg.encode(), hashlib.sha256).hexdigest()
print("\\n两种算法得到的签名相同吗:", h_correct == h_wrong, "（不相同 → 用错方法一定验签失败）")'''),
            ("md", "## 3. SSRF 演示"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.shopify.auth import validate_shop

for domain in ["demo.myshopify.com", "evil.com",
               "demo.myshopify.com.evil.io", "127.0.0.1", "myshop.myshopify.com:8080"]:
    try:
        ok = validate_shop(domain)
    except Exception as e:
        ok = f"拒绝 ({type(e).__name__})"
    print(f"{domain:32s} → {ok}")'''),
        ],
        next_day_hint="`days/day-39.md` —— 店铺前台挂件（能让店主看见的那个部分）",
    ),

    # ======================================================================
    D(
        week=7, n=39, slug="theme_widget", title="店铺前台挂件",
        where="local",
        files="`src/shopify/extensions/chat-widget/`",
        prereq="Day 38（App 能安装）",
        goal="做一个 Theme App Extension（App Block）：店铺前台右下角出现客服入口，"
             "支持**上传图片**、调用你的 Agent API、样式跟随主题。",
        read=[
            "`docs/11-shopify.md` 第 4 节（Theme App Extension / App Block / Polaris）",
            "`src/shopify/extensions/chat-widget/blocks/chat_widget.liquid` 的 `{% schema %}`",
        ],
        think=[
            "为什么用 App Block 而不是改主题模板？（提示：店主升级主题时你的改动会不会丢）",
            "脚本用 `async` 加载有什么代价？（提示：加载顺序 + 全局变量）",
            "设计模式下为什么必须显示占位内容？",
        ],
        write_title="`src/shopify/extensions/chat-widget/`（已给实现）",
        write_rows=[
            ("`chat_widget.liquid`", "挂件的 HTML + `{% schema %}`（店主在主题编辑器里能调样式）"),
            ("`{% schema %}` settings", "api_base / title / greeting / position / accent_color / disclaimer"),
            ("异步加载脚本", "`<script src=... async>` 不阻塞首屏渲染"),
            ("`design_mode` 占位", "主题编辑器里显示一个提示块，别报错"),
            ("图片上传", "前台支持传图 → 转 base64 → POST 到 `/widget/chat`"),
            ("免责声明", "底部一行「本回复由 AI 生成」—— 合规要求"),
        ],
        run=[
            ("python -m src.shopify.app --dev --port 8080", "起后端（挂件要调它）"),
            ("cat src/shopify/extensions/chat-widget/shopify.extension.toml", "看扩展清单"),
            ("shopify app dev", "（有 Shopify CLI 时）本地预览主题扩展"),
        ],
        expect="""
扩展清单：
  name        = chat-widget
  type        = theme_app_extension
  blocks      = chat_widget.liquid

店铺前台：
  右下角出现圆形入口按钮（跟随 accent_color）
  点击展开对话框，标题 = 「在线客服」
  支持上传图片 → 图片缩略图显示在输入框上方
  发送后收到 Agent 回复

主题编辑器：
  左侧出现「AI 客服挂件」区块，可拖拽 / 调位置 / 改文案
  设计模式下显示：『AI 客服挂件（预览）』
""",
        accept=[
            "在真实商品页上传一张商品图能收到回复（这是 M7 的关键一步）",
            "挂件样式跟随主题（改 `accent_color` 生效）",
            "主题编辑器里能配置，且**设计模式下不报错**",
            "底部有 AI 免责声明",
        ],
        pits=[
            "**脚本同步加载** —— 阻塞首屏，Lighthouse 分数直接掉到及格线以下。用 `async` + 自执行函数。",
            "**设计模式下报错** —— 主题编辑器会预渲染，此时没有真实顾客。代码要判断 `design_mode` 并早退。",
            "样式硬编码不跟随主题 —— 店主换成深色主题你的挂件就瞎了。用 CSS 变量 + schema 配置。",
            "图片上传没限大小 —— 有人传 20 MB 的图，你的 API 直接挂。前台也要校验。",
            "挂件的 origin 校验没做 —— 别人可以把你的挂件脚本嵌到自己的站上免费用。",
            "忘了免责声明 —— 合规问题，审核会卡。",
        ],
        nb=[
            ("md", "## 1. 看挂件的 schema（店主能调什么）"),
            ("code", '''from pathlib import Path

p = Path("../src/shopify/extensions/chat-widget/blocks/chat_widget.liquid")
text = p.read_text()
i, j = text.find("{% schema %}"), text.find("{% endschema %}")
print(text[i:j + 15] if i >= 0 else "没找到 schema 段")'''),
            ("md", "## 2. 本地模拟：把挂件的请求打一遍\n\n不装主题也能验后端 —— 用 curl 模拟挂件的请求。"),
            ("code", '''import json, base64, urllib.request

# 造一张 1x1 的小图做 payload
png_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
payload = {"shop": "demo.myshopify.com",
           "message": "这件有货吗？",
           "images": ["data:image/png;base64," + base64.b64encode(png_1x1).decode()]}

print("挂件会发的 payload:")
print(json.dumps({**payload, "images": ["<base64 1x1 png>"]}, ensure_ascii=False, indent=2))
print("\\n→ 后端要能处理：图片是可选的、可能是 data URL、可能有多张")'''),
            ("md", """## 3. 首屏性能自查（审核会看）

- [ ] 脚本 `async` 或 `defer`
- [ ] 挂件容器延后挂载（`DOMContentLoaded` 之后）
- [ ] CSS 用 inline scoped 样式，不引外部大文件
- [ ] 有 `design_mode` 早退分支"""),
            ("code", '''perf_checklist = {
    "脚本 async/defer": None,
    "延后挂载": None,
    "无外部大 CSS": None,
    "design_mode 早退": None,
    "图片大小限制": None,
}
print("逐项打勾（True/False），W8 Day 45 审核自查会再用一次")'''),
        ],
        next_day_hint="`days/day-40.md` —— Webhook 与索引增量同步",
    ),

    # ======================================================================
    D(
        week=7, n=40, slug="webhooks_index", title="Webhook 与索引同步",
        where="local",
        files="`src/shopify/webhooks.py`、`src/shopify/indexer.py`",
        prereq="Day 39；Day 32 的向量索引",
        goal="订阅 `products/update` / `orders/create` / `refunds/create`；"
             "商品变更时**增量**更新向量索引，验收标准是「改一个商品标题，30 秒内 RAG 索引同步」。",
        read=[
            "`docs/11-shopify.md` 第 5 节（Webhook）",
            "`src/shopify/webhooks.py` 的 `process_webhook` —— HMAC → 幂等 → 入队，三步顺序不能变",
        ],
        think=[
            "为什么 webhook 必须在 **5 秒内**返回 200？（超时会发生什么）",
            "幂等为什么必须做？Shopify 的重试机制是什么样的？",
            "商品变更用「全量重建索引」有什么问题？（提示：1000 个商品要多久）",
        ],
        write_title="`src/shopify/webhooks.py` + `src/shopify/indexer.py`（已给实现）",
        write_rows=[
            ("`verify_webhook_hmac()`", "**base64** 解码，不是 hex（和 OAuth 那个不一样）"),
            ("`IdempotencyStore`", "用 `X-Shopify-Webhook-Id` 去重"),
            ("`process_webhook()`", "HMAC → 幂等 → **立刻返回 200** → 异步入队处理"),
            ("`TaskQueue`", "异步 worker，真正的业务逻辑在这里跑"),
            ("`download_images()`", "把商品图从 Shopify CDN 落到本地（自己可控）"),
            ("`update_product_index()`", "**增量**更新：只重编码这一个商品的图 + 更新文本索引"),
            ("`delete_product_index()`", "商品下架要删索引，否则用户搜到买不到的东西"),
            ("GDPR handlers", "`customers/redact` / `shop/redact` / `customers/data_request`"),
        ],
        run=[
            ("python -m src.shopify.webhooks", "自检：HMAC + 幂等 + 各 handler"),
            ("python -m src.shopify.indexer --shop demo.myshopify.com --rebuild",
             "全量重建索引（第一次）"),
            ("python -m src.shopify.indexer --stats", "看索引统计"),
        ],
        expect="""
[webhook] HMAC 校验（base64）  ✓
[webhook] 幂等：同一 X-Shopify-Webhook-Id 第二次进来 → 跳过  ✓
[webhook] 处理顺序：验签 → 幂等 → 立即 200 → 入队  ✓
[handler] products/update  → update_product_index(product_id=1001)
          orders/create    → 建工单上下文
          refunds/create   → 更新退货状态

[indexer] --rebuild 商品 100 件
         下载图片 187 张（约 84 MB）
         编码 187 条向量，耗时 96s
[stats]   商品 100 · 图片 187 · 知识 42 · 索引大小 1.2 MB

增量验收：改一个标题 → handler 执行 0.8s → 索引已更新（远小于 30s）
""",
        accept=[
            "`python -m src.shopify.webhooks` 自检通过（HMAC base64 + 幂等）",
            "**改一个商品标题，30 秒内索引同步**（这是硬验收）",
            "能说清 webhook 的 HMAC 验证和幂等处理（各自防什么）",
            "商品下架时索引被正确删除（不会搜到已下架商品）",
        ],
        pits=[
            "**webhook 处理超过 5 秒才返回** —— Shopify 判定失败会重试，你会收到 5 次同一个事件。"
            "正确做法：验签 + 幂等 + 入队，立刻 200，重活异步干。",
            "**webhook HMAC 用 hex 解码** —— Shopify 的 webhook 签名是 base64。"
            "这个坑和 OAuth 回调的 hex 混起来，是新人最容易栽的地方。",
            "**没做幂等** —— Shopify 重试机制会重复投递，你的索引被重复更新甚至重复创建退货单。",
            "商品下架不删索引 —— 用户搜到买不到的商品，客诉走一圈回到你这里。",
            "全量重建当增量用 —— 1000 个商品每次改标题就重编码全部图片，几分钟起步。",
            "把 Shopify CDN 的图片 URL 直接存进索引 —— CDN URL 会变、会过期。"
            "要下载到你自己可控的存储里。",
        ],
        nb=[
            ("md", "## 1. 自检"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.shopify.webhooks"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 幂等存储：亲手验一遍"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.shopify.webhooks import IdempotencyStore

store = IdempotencyStore()
wid = "wh_abc123"
print("第一次:", store.seen(wid), "（应 False —— 没见过）")
print("第二次:", store.seen(wid), "（应 True —— 已处理过，跳过）")
print("\\n→ 如果这里返回 False，你会重复处理同一个事件")'''),
            ("md", "## 3. 增量 vs 全量的成本对比"),
            ("code", '''import time

n_products, n_images_per = 1000, 2

print("全量重建：")
print(f"  下载 {n_products * n_images_per} 张图 + 编码 → 约 "
      f"{n_products * n_images_per * 0.5 / 60:.1f} 分钟")

print("\\n增量（改 1 个商品标题）：")
print(f"  只需重编码 {n_images_per} 张图 + 更新 1 条文本 → 约 1 秒")
print("\\n→ 差 3 个数量级。这就是为什么必须做增量。")'''),
        ],
        next_day_hint="`days/day-41.md` —— 计费与合规（含 GDPR webhook）",
    ),

    # ======================================================================
    D(
        week=7, n=41, slug="billing_compliance", title="计费与合规",
        where="local",
        files="`src/shopify/billing.py`、`src/shopify/models.py`",
        prereq="Day 38–40",
        goal="实现订阅计划（免费试用 / 按会话量计费）+ 用量上报；补齐 3 个 GDPR webhook；"
             "走完一次订阅流程，并能说清「按用量计费」在 Shopify 上怎么对账。",
        read=[
            "`docs/11-shopify.md` 第 6–7 节（Billing API / usage-based / GDPR / PII）",
            "`src/shopify/billing.py` 的 `verify_subscription` —— 为什么要「主动查询」",
            "`src/shopify/models.py` 的完整 schema（10 张表）",
        ],
        think=[
            "为什么订阅状态必须**主动查询**，而不能听信前端传上来的参数？",
            "用量上报和计费为什么必须幂等？（提示：重试网络请求）",
            "超过 capped amount 时该怎么处理？",
        ],
        write_title="`src/shopify/billing.py` + 建表（已给实现）",
        write_rows=[
            ("`PLANS`", "free / standard / pro 三档（含 trial 天数、会话配额、单价）"),
            ("`create_subscription()`", "`appSubscriptionCreate` + 用量行项目"),
            ("`verify_subscription()`", "**主动查询**，不信任前端传参"),
            ("`report_usage()`", "幂等上报（幂等键 = 会话 id）+ 处理 cap 上限"),
            ("`BillingLedger`", "本地台账 —— **和 Shopify 对账用**"),
            ("`check_entitlement()`", "每个请求进来先查额度"),
            ("`models.SCHEMA`", "10 张表：shops / sessions / messages / usage_records / subscriptions /"
                                " product_index / knowledge_docs / webhook_events / tickets / audit_log"),
        ],
        run=[
            ("python -m src.shopify.models --dump", "导出 schema.sql 看一眼"),
            ("python -m src.shopify.models --init", "建表（需要 PostgreSQL；或用 docker compose up 起的库）"),
            ("python -m src.shopify.billing", "自检：订阅流程 + 幂等上报 + cap 处理"),
        ],
        expect="""
[models] 10 张表：
   shops / sessions / messages / usage_records / subscriptions
   product_index / knowledge_docs / webhook_events / tickets / audit_log

[billing] 自检：
  创建订阅 standard（14 天试用 + ¥0.05/会话，封顶 ¥200）
  幂等上报 3 次同一会话 → usage_records 只增 1 条  ✓
  额度检查：session_count 120 / 500 → allowed=True
  模拟超标：499 → allowed=True，500 → allowed=False（转免费版限制）
  cap 处理：累计 ¥200.00 后自动停止上报，不再向店主收费  ✓

[ledger] 本地台账 1 条，与上报次数一致  ✓
""",
        accept=[
            "能走完一次订阅流程（创建 → 授权确认 → 状态可查）",
            "用量上报幂等（同一会话上报 3 次只记 1 条）",
            "**本地台账和上报记录能对上**（这是对账的基础）",
            "3 个 GDPR webhook 都实现并自检通过",
            "能说清「按用量计费」的对账流程：Shopify 那边能看到什么、你这边要记什么",
        ],
        pits=[
            "**不主动查订阅状态** —— 前端传 `subscribed=true` 你就信了，用户白嫖到天荒地老。",
            "**用量上报不幂等** —— 网络重试一次就多收一次钱，客诉拉到爆。",
            "超出 capped amount 还在上报 —— Shopify 会拒绝，你的上报逻辑要优雅处理。",
            "**没有本地台账** —— 月底和 Shopify 对账发现差了几百块，查不出来源。",
            "GDPR webhook 没实现 —— 审核过不去；而且一旦有顾客行使删除权，你无法履行。",
            "audit_log 表建了但没写 —— 出问题无法追溯「谁在什么时候做了什么」。",
        ],
        nb=[
            ("md", "## 1. 看数据库 schema"),
            ("code", '''from pathlib import Path
text = Path("../src/shopify/models.py").read_text()
i = text.find("SCHEMA")
print(text[i:i + 1800])'''),
            ("md", "## 2. 幂等上报亲手验"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.shopify.billing import BillingLedger

ledger = BillingLedger()
for i in range(3):
    ledger.record(session_id="sess_001", amount=0.05)   # 同一个会话上报 3 次

print("台账条目数:", len(ledger.entries) if hasattr(ledger, "entries") else "（看实际属性名）")
print("→ 必须是 1，不是 3")'''),
            ("md", "## 3. 单位经济：这个生意能不能做"),
            ("code", '''GPU_HOURLY = 1.88          # RTX 4090 ¥/h
SESSIONS_PER_HOUR = 300     # 估算：单卡并发 4，单次 5s
PLAN_PRICE = 99             # 标准版月费
SESSIONS_INCLUDED = 2000

gpu_per_session = GPU_HOURLY / SESSIONS_PER_HOUR
print(f"单会话 GPU 成本 ≈ ¥{gpu_per_session:.4f}")
print(f"套餐含 {SESSIONS_INCLUDED} 次会话 → GPU 成本 ¥{gpu_per_session*SESSIONS_INCLUDED:.2f}")
print(f"套餐价 ¥{PLAN_PRICE} → 毛利 ¥{PLAN_PRICE - gpu_per_session*SESSIONS_INCLUDED:.2f}")
print("\\n（还没算 Shopify 抽成、服务器、人力 —— W8 Day 43 会做完整模型）")'''),
        ],
        next_day_hint="`days/day-42.md` —— 部署上线，W7 收官",
    ),

    # ======================================================================
    D(
        week=7, n=42, slug="deploy", title="部署上线",
        where="local",
        files="`Dockerfile`、`docker-compose.yml`、`src/shopify/app.py`",
        prereq="Day 41",
        goal="Docker 化并部署到有公网地址的地方；配好 HTTPS、环境变量、日志；"
             "验收是「从陌生店铺安装到前台能对话」全流程走通。",
        read=[
            "`docs/11-shopify.md` 第 8 节（部署）",
            "`docker-compose.yml` —— 看它怎么把 PostgreSQL + pgvector + 应用串起来",
            "`src/shopify/__init__.py` 里记录的「五个最容易踩的坑」",
        ],
        think=[
            "为什么 webhook 必须走 HTTPS？（Shopify 会拒绝 http 回调）",
            "环境变量该怎么传给容器才不会被写进镜像层？",
            "数据库没做持久化卷会发生什么？",
        ],
        write_title="部署（本日以配置和验证为主）",
        write_rows=[
            ("`Dockerfile`", "python:3.11-slim，**不含 torch**（应用侧只调 vLLM）"),
            ("`docker-compose.yml`", "pgvector/pg16 + 应用；数据库有 healthcheck 与持久化卷"),
            ("环境变量", "用 `env_file: .env`，**绝不 `COPY .env`** 进镜像"),
            ("HTTPS", "平台自带证书；用 ngrok/cloudflared 也能本地演示"),
            ("日志", "stdout 结构化输出，交给平台收集"),
            ("健康检查", "`/health` 供平台探活"),
        ],
        run=[
            ("docker compose up -d", "起数据库 + 应用"),
            ("docker compose ps", "看容器状态"),
            ("curl -s localhost:8080/health", "探活"),
            ("docker compose logs -f app | head -30", "看日志"),
            ("make down", "收工（数据保留）"),
        ],
        expect="""
NAME          STATUS
pgvector      Up (healthy)
app           Up (healthy)

$ curl localhost:8080/health
{"status":"ok","db":"ok","vllm":"ok"}

$ docker compose logs app
{"event":"startup","port":8080,"shop_count":1}
{"event":"webhook_registered","topics":["products/update","orders/create","refunds/create"]}

公网地址（平台给的）：
  https://your-app.onrender.com
  → Shopify 后台填这个作为 App URL 和 webhook 回调地址
""",
        accept=[
            "有一个**公网可访问**的地址，`/health` 返回 ok",
            "从陌生店铺走完「安装 → 授权 → 前台能对话」全流程",
            "`.env` 没进镜像（用 `docker history` 能验）",
            "数据库数据卷已挂载（重启容器数据还在）",
            "`progress/weekly-review.md` 的 W7 段已写；进度表 W7 六天 `[x]`，M7 打卡",
        ],
        pits=[
            "**webhook 用 http 回调** —— Shopify 直接拒绝，你的索引永远不同步，而且不容易发现。",
            "**`.env` 被 `COPY` 进镜像** —— token 全在镜像层里，推到公开 registry 就全泄露了。",
            "数据库没挂数据卷 —— 容器一重启，所有的 shops / subscriptions 全没了。",
            "torch 打进了应用镜像 —— 镜像从 200 MB 变成 8 GB，部署慢十倍（应用侧只调 vLLM，不需要 torch）。",
            "平台冷启动导致 webhook 超时 —— 免费套餐会休眠，webhook 打过来要等几十秒才醒。"
            "要么用付费套餐，要么把 webhook 收口到一个常驻的轻量服务。",
            "忘了配 `SCOPES` / 环境变量 —— 装上去了但一调 API 就 403。",
        ],
        nb=[
            ("md", "## 1. 起起来"),
            ("code", '''print("""
在终端执行：

    docker compose up -d
    docker compose ps
    curl -s localhost:8080/health

查看日志：

    docker compose logs -f app
""")'''),
            ("md", "## 2. 自查：.env 有没有泄漏进镜像"),
            ("code", '''print("""
安全自查（在终端跑）：

  docker history multimodal-lab-app | grep -i env    # 不该出现 .env 的内容
  docker run --rm multimodal-lab-app ls -la          # 容器里不该有 .env
  docker run --rm multimodal-lab-app python -c "import os; print('API_KEY' in os.environ)"
""")'''),
            ("md", "## 3. 上线检查清单"),
            ("code", '''checklist = {
    "公网 HTTPS 地址可访问": None,
    "App URL / 回调地址已填到 Shopify 后台": None,
    "webhook 地址是 https": None,
    "数据库持久化卷已挂": None,
    ".env 未进镜像": None,
    "日志能追到 request_id": None,
    "陌生店铺安装全流程走通": None,
}
for k in checklist:
    print(f"  [ ] {k}")
print("\\n逐项确认，全部打勾才算 M7 达成")'''),
        ],
        footer="> **M7 达成**：有一个公网可访问的 Shopify App，从陌生店铺安装到前台对话全流程走通。",
        next_day_hint="`days/day-43.md` —— W8 交付周：压测与成本核算",
    ),
]
