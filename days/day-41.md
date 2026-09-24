# Day 41 · 计费与合规

> 预计 3–4h ｜ 📓 `notebooks/day-41_billing_compliance.ipynb` ｜ 💻 本地 · `src/shopify/billing.py`、`src/shopify/models.py`
> 前置：Day 38–40

## 今日目标（一句话）

实现订阅计划（免费试用 / 按会话量计费）+ 用量上报；补齐 3 个 GDPR webhook；走完一次订阅流程，并能说清「按用量计费」在 Shopify 上怎么对账。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 6–7 节（Billing API / usage-based / GDPR / PII）
- `src/shopify/billing.py` 的 `verify_subscription` —— 为什么要「主动查询」
- `src/shopify/models.py` 的完整 schema（10 张表）

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么订阅状态必须**主动查询**，而不能听信前端传上来的参数？
2. 用量上报和计费为什么必须幂等？（提示：重试网络请求）
3. 超过 capped amount 时该怎么处理？

## 二、写（100 min）

`src/shopify/billing.py` + 建表（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `PLANS` | free / standard / pro 三档（含 trial 天数、会话配额、单价） |
| `create_subscription()` | `appSubscriptionCreate` + 用量行项目 |
| `verify_subscription()` | **主动查询**，不信任前端传参 |
| `report_usage()` | 幂等上报（幂等键 = 会话 id）+ 处理 cap 上限 |
| `BillingLedger` | 本地台账 —— **和 Shopify 对账用** |
| `check_entitlement()` | 每个请求进来先查额度 |
| `models.SCHEMA` | 10 张表：shops / sessions / messages / usage_records / subscriptions / product_index / knowledge_docs / webhook_events / tickets / audit_log |

## 三、跑（本地（无需 GPU））

```bash
# 导出 schema.sql 看一眼
python -m src.shopify.models --dump
# 建表（需要 PostgreSQL；或用 docker compose up 起的库）
python -m src.shopify.models --init
# 自检：订阅流程 + 幂等上报 + cap 处理
python -m src.shopify.billing
```

期望输出（节选）：
```
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
```

## 四、验收清单

- [ ] 能走完一次订阅流程（创建 → 授权确认 → 状态可查）
- [ ] 用量上报幂等（同一会话上报 3 次只记 1 条）
- [ ] **本地台账和上报记录能对上**（这是对账的基础）
- [ ] 3 个 GDPR webhook 都实现并自检通过
- [ ] 能说清「按用量计费」的对账流程：Shopify 那边能看到什么、你这边要记什么

## 五、容易踩的坑

1. **不主动查订阅状态** —— 前端传 `subscribed=true` 你就信了，用户白嫖到天荒地老。
2. **用量上报不幂等** —— 网络重试一次就多收一次钱，客诉拉到爆。
3. 超出 capped amount 还在上报 —— Shopify 会拒绝，你的上报逻辑要优雅处理。
4. **没有本地台账** —— 月底和 Shopify 对账发现差了几百块，查不出来源。
5. GDPR webhook 没实现 —— 审核过不去；而且一旦有顾客行使删除权，你无法履行。
6. audit_log 表建了但没写 —— 出问题无法追溯「谁在什么时候做了什么」。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
