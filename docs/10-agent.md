# 10 · 多模态 Agent

> 对应 Day 31–Day 36

## 概念

### 为什么客服场景必须用 Agent，而不是纯模型

纯模型：`(图, 文) → 回答`。它能说「根据退货政策可以退」，但它**不知道这个订单的真实状态**。

Agent：`(图, 文) → 计划 → 调工具 → 观察 → 回答`。

客服的真实任务里，**大部分有价值的动作需要外部信息**：

| 用户问 | 需要什么 | 纯模型能做吗 |
|---|---|---|
| 「我的快递到哪了」 | 订单号、物流 API | ❌ 只能瞎猜 |
| 「这个有 M 码吗」 | 实时库存 | ❌ |
| 「我要退货」 | 创建退货单（有副作用） | ❌ |
| 「这个线头算质量问题吗」 | 图片理解 + 政策知识 | ✅ 模型就够了 |
| 「和图片上不一样」 | 以图搜图定位商品 | ⚠️ 需要检索 |

**结论**：图片理解能力是基础，但完整解决用户问题必须靠 Agent。

### Agent 的三个流派

```
① ReAct（Reason + Act）        ← 本项目选它
   Thought → Action → Observation → Thought → ...
   每步都显式思考再行动，可解释、易调试
   缺点：多一次 LLM 调用，慢

② Plan-Execute
   先一次性生成完整计划，再逐步执行
   优点：调用少、快
   缺点：计划错了全错，中途不能适应

③ Function Calling / Tool Use（现代 API 原生）
   模型直接输出结构化的 tool_calls，由 runtime 执行
   优点：格式可靠（有约束解码）、快
   缺点：黑盒、调试难、对本地小模型支持参差

实践建议：用 ③ 的格式（结构化 tool call），但保留 ① 的显式推理步骤。
```

这就是所谓 **"structured ReAct"**：模型每步输出

```json
{
  "thought": "用户想知道快递到哪了，需要先拿到订单号",
  "action": "lookup_order",
  "action_input": {"session_id": "s_123"},
  "response": null
}
```

能解析、能调试、能记录、能评测。

### 工具设计原则

**原则 1：工具要「任务级」不要「数据级」**

```
❌ 差：query_database(sql)          —— 太底层，模型写 SQL 会出错
❌ 差：get_order(order_id)          —— 太原子，一个流程要调 5 次
✅ 好：lookup_order_by_session()    —— 一个调用拿到该用户的所有订单
✅ 好：check_return_eligibility(order_id)  —— 一次调用返回能否退 + 原因
```

**原则 2：参数要给默认值和枚举**

```json
{
  "name": "check_stock",
  "description": "查询商品库存。当用户询问某尺码/颜色是否有货时使用。",
  "parameters": {
    "type": "object",
    "properties": {
      "sku_id": {"type": "string", "description": "商品 SKU，可从 product_id + 尺码/颜色推导"},
      "size": {"type": "string", "enum": ["XS","S","M","L","XL","XXL"], "description": "留空则查全部尺码"},
      "color": {"type": "string", "description": "留空则查全部颜色"}
    },
    "required": ["sku_id"]
  }
}
```

**原则 3：有副作用的工具必须幂等**

```
无幂等：用户点两次「申请退货」→ 创建两张退货单 → 财务炸了
有幂等：带 idempotency_key（hash(session_id + action + 时间窗口)）
       第二次调用返回第一次的结果
```

这是 Day 33 的核心任务。

**原则 4：错误返回要「可被模型理解」**

```json
// ❌ 差
{"error": "500 Internal Server Error"}

// ✅ 好
{
  "success": false,
  "error_code": "ORDER_NOT_FOUND",
  "message": "未找到该会话关联的订单",
  "suggestion": "请用户提供订单号，或确认是否在本店下单"
}
```

`suggestion` 字段让模型知道下一步该干什么，而不是卡死或编造。

### 多模态 RAG：以图搜图

客服场景的核心检索需求：

```
用户上传一张图 → 这是哪个商品？→ 才能查库存/政策/价格
```

**方案对比**：

| 方案 | 做法 | 准确率 | 延迟 |
|---|---|---|---|
| 先描述再搜文 | VLM 生成描述 → 文本检索 | 低（描述会丢信息、会和库里描述不匹配） | 中 |
| **CLIP 双塔直搜** | 用户图 → CLIP image emb → 向量库 | **中高** | 低 |
| CLIP 粗排 + VLM 精排 | 先取 top-20，再用 VLM 逐对判断 | **高** | 高 |

**推荐做法**：CLIP 粗排取 top-20 + 规则过滤（类目、价格区间）+ VLM 精排取 top-3。

**注意 CLIP 的局限**：它匹配的是「整体语义」，对「同一款不同颜色」这种细粒度差异不敏感。所以：

- 粗排召回可以，**精排必须靠 VLM**
- 或者训练一个领域专用的商品图 embedding（这是加餐方向）

**双索引设计**（Day 32 的代码任务）：

```
索引 1 · 商品图库    CLIP image embedding   → 以图搜图
索引 2 · 知识库      文本 embedding          → 政策/FAQ/尺码表检索
索引 3 · 商品属性    结构化过滤              → 类目/价格/库存状态
```

检索时三个索引协同：图搜定位商品 → 结构化查属性 → 文本检索相关知识。

### Agent 主循环

```
输入: 用户消息（图 + 文）+ 会话历史
  ↓
① 意图识别（轻量模型或规则）—— 分类到 8 大意图
  ↓
② 上下文组装
   - 会话历史（最近 N 轮）
   - 若有图：CLIP 检索找商品候选
   - 检索相关知识与政策
  ↓
③ 循环（最多 max_steps 步）
   ├─ LLM 输出 { thought, action, action_input }
   ├─ 若是 final_answer → 跳出
   ├─ 执行工具 → 得到 observation
   ├─ 检查：死循环？成本超限？步数超限？
   └─ 把 (action, observation) 追加到上下文
  ↓
④ 后处理
   - 输出过滤（PII、承诺性语言）
   - 若失败 → 降级到人工
  ↓
输出: 回复 + 工具调用记录（用于评测和审计）
```

**三个必须有的护栏**：

1. **max_steps**（如 6）——防止无限循环
2. **检测重复 action**——同一个工具同样参数调两次 → 强制跳出
3. **成本上限**——累计 token 超阈值 → 转人工

### Agent 评测的不同之处

传统模型评测：单个输出对不对。
Agent 评测：**整条轨迹对不对**。

```
指标：
  任务完成率 (Task Success Rate)    ← 主指标，目标 ≥65%
  工具选择准确率 (Tool Selection)
  参数填充准确率 (Slot Filling)
  平均交互轮数 (Avg Turns)          ← 越少越好
  P95 延迟
  单次会话成本
  不必要工具调用率                  ← 应尽量低
```

**任务完成率的判定**需要**环境状态检查**，不只看回复内容：

```python
def is_task_done(task, final_state):
    """不只看回复，看真实副作用"""
    if task.type == "return":
        return final_state.has_return_request(task.order_id)
    if task.type == "inquiry":
        return check_answer_contains_facts(final_state.reply, task.gold_facts)
```

这是 Agent 评测和模型评测最大的区别——**要看世界状态变了没有**。

## 工程细节

### 关键失败模式

| 失败模式 | 表现 | 修法 |
|---|---|---|
| 工具幻觉 | 调用不存在的工具 | 严格 schema 校验 + 拒绝未知 action |
| 参数幻觉 | 编造订单号 | 参数只允许来自已验证上下文 |
| 死循环 | 反复查同一个订单 | 重复检测 + max_steps |
| 过度调用 | 简单问题也查一堆工具 | 加「不必要调用」的惩罚奖励 |
| 中途丢失用户意图 | 查完库存忘了原本的问题 | 每步都带上原始 query |
| 副作用重复 | 重复创建退货单 | 幂等键 |
| 图丢了 | 工具调用后不再引用图片 | 图片特征保持在上下文里 |

**最后一条很隐蔽**：很多 Agent 在「调用工具 → 拿到文本结果」后，就不再关注原始图片了，导致最终回答和图片无关。解法是**把视觉证据显式写进中间状态**：

```python
state.visual_evidence = {
    "detected": ["浅蓝色上衣", "有线头", "尺码标签写着 M"],
    "confidence": 0.85,
}
# 后续每步都把 visual_evidence 放进 prompt
```

## 自检问题

1. 什么情况下纯模型够用，什么情况下必须上 Agent？用客服场景举例。
2. ReAct 和 Plan-Execute 的取舍是什么？「structured ReAct」是什么意思？
3. 工具设计里「任务级 vs 数据级」为什么重要？举例说明。
4. 为什么有副作用的工具必须幂等？怎么实现？
5. 以图搜图的三个方案，准确率和延迟怎么权衡？CLIP 的局限是什么？
6. Agent 主循环必须有的三个护栏是什么？
7. Agent 评测为什么不能只看回复内容？「任务完成」怎么判定？
8. 「图丢了」这个失败模式的成因和解法？

---

**上一篇**：[09-inference.md](09-inference.md) · **下一篇**：[11-shopify.md](11-shopify.md)
