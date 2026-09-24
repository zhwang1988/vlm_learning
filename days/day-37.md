# Day 37 · Shopify 生态与 API

> 预计 3–4h ｜ 📓 `notebooks/day-37_shopify_api.ipynb` ｜ 💻 本地 · `src/shopify/client.py`、`docs/11-shopify.md`
> 前置：Day 36；有一个 Shopify Partners 账号 + 开发测试店

## 今日目标（一句话）

封装 Admin GraphQL 客户端（分页、重试、限流退避），能从测试店拉到商品列表并落库；并说清三种 App 类型的审核要求差异。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 1–2 节（App 类型 / Admin GraphQL / REST vs GraphQL / rate limit）
- `src/shopify/client.py` 的 `query()` —— 重点读它怎么处理 `throttleStatus`

思考题（先自己想，答案在讲义或代码注释里）：
1. Shopify 为什么把 REST 判了「过时」转向 GraphQL？（提示：按点数计费 vs 按请求数）
2. `extensions.cost.throttleStatus.currentlyAvailable` 是干什么用的？不看它会怎样？
3. public / custom / private 三种 App 分别适合什么场景？审核要求差在哪？

## 二、写（100 min）

`src/shopify/client.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `ShopifyClient.query()` | 发 GraphQL；**读返回的 cost 信息**决定要不要退避 |
| 限流退避 | 点数不足时 sleep 而不是硬撞 429 |
| `paginate()` | 异步生成器翻页 —— 别手动拼 cursor |
| `PRODUCTS_QUERY` | 只取需要的字段（GraphQL 的字段选择直接影响点数消耗） |
| `to_internal_product()` | 外部数据结构 → 内部结构（**隔离层，很重要**） |

> 今天最该养成的习惯：**外部 API 的数据结构不要渗透到你的业务代码里**。
> `to_internal_product()` 这一层看着多余，等你换了个电商平台就知道它多值钱。

## 三、跑（本地（无需 GPU））

```bash
# 自检：GraphQL 查询构造 + 限流解析
python -m src.shopify.client
# 看有哪些参数（接真实店铺时用）
python -m src.shopify.client --help
```

期望输出（节选）：
```
[client] 构造 Admin GraphQL 客户端
[query] POST https://demo.myshopify.com/admin/api/2026-07/graphql.json
        extensions.cost: requestedQueryCost=42, currentlyAvailable=1968
[parse] throttleStatus 正常，无需退避
[demo] 商品样例（3 条）：
   gid://shopify/Product/1001  米白色针织衫   ¥199   3 个变体
   gid://shopify/Product/1002  奶白色开衫     ¥259   2 个变体
   已转为内部结构: {'id': '1001', 'title': '米白色针织衫', ...}
✓ 自检通过（无 token 时只验查询构造，不发请求）
```

## 四、验收清单

- [ ] `python -m src.shopify.client` 自检通过
- [ ] 能从测试店拉到 ≥10 个商品并落库（JSON / SQLite 都行）
- [ ] 能说清三种 App 类型的差异和审核要求
- [ ] 知道 `throttleStatus` 怎么读、为什么要退避而不是硬撞

## 五、容易踩的坑

1. 用 REST API —— 已过时且限流更严（每秒 2 请求）。新项目一律 GraphQL。
2. **不看 throttle 直接发请求** —— 点数耗尽后连续 429，你的同步任务卡死。
3. GraphQL 里 `*` 式要全部字段 —— 点数消耗暴涨，一次查询就把额度用光。
4. cursor 分页用错（把 `endCursor` 当 `startCursor`）—— 死循环翻同一页。
5. access token 硬编码进代码 —— 一定会上 git。用 `.env`，且 `.gitignore` 要包含它。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
