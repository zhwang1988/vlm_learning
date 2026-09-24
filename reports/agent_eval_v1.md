# Agent 评测报告 v1

- 任务数: **7**
- **任务完成率: 100.0%** （目标 ≥ 65%）
- 工具选择准确率: 100.0%
- 参数填充准确率: 100.0%
- 平均交互轮数: 1.9
- 不必要工具调用总数: 2（平均 0.29 次/任务）
- 延迟 P50 / P95: 0ms / 0ms
- Token 消耗: 6,000 in / 430 out
- 估算成本: ¥0.0073（7 个任务，约 ¥0.0010/任务）

## 逐任务结果

| 任务 | 结果 | 轮数 | 调用工具 | 说明 |
|---|---|---:|---|---|
| t01 | ✅ | 2 | lookup_order, shipping_status | ✓ 命中 ['快递', '运输'] |
| t02 | ✅ | 3 | lookup_order, check_return_eligibility, start_return | ✓ 调用了 check_return_eligibility |
| t03 | ✅ | 2 | check_stock | ✓ 调用了 check_stock |
| t04 | ✅ | 1 | - | ✓ 命中 ['线头', '工艺', '正常'] |
| t05 | ✅ | 3 | lookup_order, check_return_eligibility, start_return | ✓ 调用了 start_return |
| t06 | ✅ | 1 | - | ✓ 命中 ['抱歉', '转接', '人工', '理解'] |
| t07 | ✅ | 1 | - | ✓ 命中 ['标注', '棉'] |

## 优化方向

- ✓ 完成率达标
