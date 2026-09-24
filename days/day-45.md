# Day 45 · Shopify 审核对齐

> 预计 3–4h ｜ 📓 `notebooks/day-45_shopify_review.ipynb` ｜ 💻 本地 · `reports/shopify_checklist.md`
> 前置：Day 44

## 今日目标（一句话）

逐项补齐 Shopify App Store 的审核要求：隐私政策、卸载 webhook、性能（Lighthouse）、无障碍；产出全绿的 `reports/shopify_checklist.md`，提交审核或至少完成自查。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 9 节（审核清单）
- Shopify 官方的 App Store requirements 页面（重点看 2026 版新增项）

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么「卸载 webhook」是硬要求？（提示：不留残余数据）
2. Lighthouse 性能要求卡在哪一项最容易被拒？
3. 审核被拒最常见的三个原因是什么？

## 二、写（100 min）

自查清单（补齐 + 验证）

| 函数 / 文件 | 你要做什么 |
|---|---|
| 隐私政策页 | 公网可访问的隐私政策 URL（写清收集什么、存多久、怎么删） |
| `app/uninstalled` webhook | 卸载时清理数据（或标记待清理） |
| 3 个 GDPR webhook | customers/data_request / customers/redact / shop/redact |
| Lighthouse | Performance / Accessibility / Best Practices 都要达标 |
| 无障碍 | 挂件可键盘操作、有 aria-label、对比度达标 |
| 卸载流程实测 | 真的卸载一次，看数据有没有按预期处理 |
| 审核材料 | App 描述、截图、演示视频、定价说明 |

## 三、跑（本地（无需 GPU））

```bash
# 看清单现状
ls reports/shopify_checklist.md && head -50 reports/shopify_checklist.md
# 确认 GDPR / uninstall handler 都注册了
python -m src.shopify.webhooks
```

期望输出（节选）：
```
reports/shopify_checklist.md

  基础
  [x] 公网 HTTPS 地址可访问
  [x] OAuth 安装流程完整（含 state 防 CSRF）
  [x] 卸载 webhook 已实现并实测
  [x] 3 个 GDPR webhook 已实现

  合规
  [x] 隐私政策 URL 可访问
  [x] AI 生成内容有免责声明
  [x] 最小权限 scope（只申请需要的）
  [x] 数据保留策略写明

  性能
  [x] Lighthouse Performance ≥ 70
  [x] 挂件不阻塞首屏（async）
  [x] Accessibility ≥ 90

  提交材料
  [x] App 描述 / 截图 / 演示视频
  [ ] 提交审核（或：完成自查，暂不提交）
```

## 四、验收清单

- [ ] `reports/shopify_checklist.md` 绝大部分打勾（未打的写清原因和计划）
- [ ] **卸载流程实测过**（真卸载一次，数据按预期处理）
- [ ] Lighthouse 跑过，Performance / Accessibility 有具体分数
- [ ] 要么提交了审核，要么明确记录「为什么现在不提交」

## 五、容易踩的坑

1. **隐私政策缺失或不可访问** —— 审核必卡，而且没得商量。
2. **卸载 webhook 没实现** —— 硬要求。不实现的话，店主卸载后数据还在你这里，属于违规留存。
3. **Lighthouse 被挂件拖垮** —— 挂件脚本同步加载、图片没压缩，Performance 掉到 40。W7 Day 39 的清单在这里要再过一遍。
4. 无障碍完全没做 —— 键盘无法操作、屏幕阅读器读不出，Accessibility 分数过低也会被拒。
5. 演示视频用真实店铺数据 —— 审核员看得到顾客信息，这是隐私问题。用假数据。
6. 为了过审申请了过大的 scope —— 审核员会问「你为什么需要这个权限」。最小权限。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
