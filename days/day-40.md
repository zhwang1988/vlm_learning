# Day 40 · Webhook 与索引同步

> 预计 3–4h ｜ 📓 `notebooks/day-40_webhooks_index.ipynb` ｜ 💻 本地 · `src/shopify/webhooks.py`、`src/shopify/indexer.py`
> 前置：Day 39；Day 32 的向量索引

## 今日目标（一句话）

订阅 `products/update` / `orders/create` / `refunds/create`；商品变更时**增量**更新向量索引，验收标准是「改一个商品标题，30 秒内 RAG 索引同步」。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 5 节（Webhook）
- `src/shopify/webhooks.py` 的 `process_webhook` —— HMAC → 幂等 → 入队，三步顺序不能变

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么 webhook 必须在 **5 秒内**返回 200？（超时会发生什么）
2. 幂等为什么必须做？Shopify 的重试机制是什么样的？
3. 商品变更用「全量重建索引」有什么问题？（提示：1000 个商品要多久）

## 二、写（100 min）

`src/shopify/webhooks.py` + `src/shopify/indexer.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `verify_webhook_hmac()` | **base64** 解码，不是 hex（和 OAuth 那个不一样） |
| `IdempotencyStore` | 用 `X-Shopify-Webhook-Id` 去重 |
| `process_webhook()` | HMAC → 幂等 → **立刻返回 200** → 异步入队处理 |
| `TaskQueue` | 异步 worker，真正的业务逻辑在这里跑 |
| `download_images()` | 把商品图从 Shopify CDN 落到本地（自己可控） |
| `update_product_index()` | **增量**更新：只重编码这一个商品的图 + 更新文本索引 |
| `delete_product_index()` | 商品下架要删索引，否则用户搜到买不到的东西 |
| GDPR handlers | `customers/redact` / `shop/redact` / `customers/data_request` |

## 三、跑（本地（无需 GPU））

```bash
# 自检：HMAC + 幂等 + 各 handler
python -m src.shopify.webhooks
# 全量重建索引（第一次）
python -m src.shopify.indexer --shop demo.myshopify.com --rebuild
# 看索引统计
python -m src.shopify.indexer --stats
```

期望输出（节选）：
```
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
```

## 四、验收清单

- [ ] `python -m src.shopify.webhooks` 自检通过（HMAC base64 + 幂等）
- [ ] **改一个商品标题，30 秒内索引同步**（这是硬验收）
- [ ] 能说清 webhook 的 HMAC 验证和幂等处理（各自防什么）
- [ ] 商品下架时索引被正确删除（不会搜到已下架商品）

## 五、容易踩的坑

1. **webhook 处理超过 5 秒才返回** —— Shopify 判定失败会重试，你会收到 5 次同一个事件。正确做法：验签 + 幂等 + 入队，立刻 200，重活异步干。
2. **webhook HMAC 用 hex 解码** —— Shopify 的 webhook 签名是 base64。这个坑和 OAuth 回调的 hex 混起来，是新人最容易栽的地方。
3. **没做幂等** —— Shopify 重试机制会重复投递，你的索引被重复更新甚至重复创建退货单。
4. 商品下架不删索引 —— 用户搜到买不到的商品，客诉走一圈回到你这里。
5. 全量重建当增量用 —— 1000 个商品每次改标题就重编码全部图片，几分钟起步。
6. 把 Shopify CDN 的图片 URL 直接存进索引 —— CDN URL 会变、会过期。要下载到你自己可控的存储里。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
