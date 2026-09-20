# 07 · 偏好对齐：DPO / ORPO / KTO / GRPO

> 对应 Day 25–Day 28

## 概念

### 为什么 SFT 之后还要对齐

SFT 教模型「什么样的回答是对的」，但它不教「两个都能接受的回答里哪个更好」。结果是：

- 回答正确但语气生硬，不像客服
- 遇到模糊情况不会主动澄清，硬答
- 偶尔输出不该输出的内容（透露内部信息、承诺无法兑现的事）

**对齐（alignment）** 用「偏好比较」作信号来修这些。

### 从 RLHF 到 DPO

```
经典 RLHF（三阶段，复杂）
  ① SFT  →  ② 训 Reward Model（用人类偏好对）  →  ③ PPO 优化策略
  问题：要同时养四个模型（policy / reward / ref / value），极不稳定

DPO（Direct Preference Optimization，2023）
  数学上证明：可以跳过 Reward Model，直接用偏好对优化策略
  只需要两个模型（policy + frozen reference）
  稳定、简单、效果好 → 已成为主流
```

**DPO 的核心 insight**（值得理解，不只是背）：

PPO 的目标里，最优策略有闭式解：

```
π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
```

把它反解出 reward，代回 Bradley-Terry 偏好模型，reward 就被消掉了，只剩下一个**直接用 policy 和 reference 的对数概率比**构造的损失：

```
L_DPO = - E [ log σ( β · ( log(π_θ(y_w)/π_ref(y_w)) − log(π_θ(y_l)/π_ref(y_l)) ) ) ]
                     └──── chosen ────┘   └──── rejected ────┘
```

**这个式子的直觉**：

- `log(π_θ/π_ref)` 是「相对基座，策略对这个回答的偏好提升了多少」
- DPO 要让 **chosen 的相对提升大于 rejected 的相对提升**
- 除以 `π_ref` 很关键：防止模型为了拉高 chosen 的绝对概率而破坏语言模型本身（**用 reference 做锚**）

**β 的作用**：控制「偏离 reference 的惩罚」。β 大 → 保守，变化小；β 小 → 激进，容易崩。典型值 **0.1–0.5**。

### DPO 家族速查

| 方法 | 数据需求 | 需要 reference 模型 | 特点 |
|---|---|---|---|
| **DPO** | 成对 (chosen, rejected) | 是 | 标准，稳，本项目选它 |
| **ORPO** | 成对 | **否** | 把 SFT loss 和偏好 loss 合到一步，省显存 |
| **KTO** | 只要单条 + 好/坏标签 | 是 | 不需要配对，数据好收集 |
| **SimPO** | 成对 | **否** | 不需要 reference，用长度归一化 |
| **GRPO** | 只要 prompt + 奖励函数 | 是 | 组内相对优势，适合可验证任务 |

**选型建议**：

- 你手上有成对数据 → **DPO**
- 显存紧张（无卡环境）→ **ORPO / SimPO**（省掉 reference 模型，显存减半）
- 你能写出规则化奖励（格式对不对、工具调用对不对）→ **GRPO**

### 多模态 DPO 的特殊性

文本 DPO 只有一种「chosen/rejected」维度。多模态有三种，而且**难点完全不同**：

```
类型 1 · 同图，不同答
  图: 一件有轻微线头的衣服
  chosen:   "从图上看这是一处轻微的缝线不齐，属于正常工艺范围，不影响穿着..."
  rejected: "这是严重的质量问题，必须退货！"
  → 修的是「判断校准」和「语气」。最容易构造，收益最大。

类型 2 · 同答，不同图（contrastive image）
  prompt: "这件衣服有质量问题吗？"
  chosen 配: 真的有线头的图
  rejected 配: 完好无损的图
  → 修的是「视觉 grounding」。作用是让模型真的看图，而不是靠语言先验瞎猜。
  → 这类数据是治幻觉的利器。

类型 3 · 同图同答，不同格式/工具调用
  chosen:   {"action": "lookup_order", "order_id": "..."}   ← 可解析
  rejected: "我帮你查一下订单，请稍等..."                    ← 不可解析
  → 修的是 Agent 场景的工具调用稳定性。
```

**类型 2 是被低估的杀器**：随机把图和答案错配，就得到 rejected。这个构造几乎免费，但能显著降低幻觉（模型学会「我必须看这张图才能回答」）。

**数据量经验**：多模态 DPO 通常 **1k–5k 对** 就够看到效果。不需要像 SFT 那样上万。原因是 DPO 只需要「相对偏好」，信号密度更高。

## 工程细节

### 一个可读的 DPO loss 实现

```python
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps,    # (B,)  策略对 chosen 的对数概率和
             policy_rejected_logps,  # (B,)
             ref_chosen_logps,       # (B,)  reference 模型
             ref_rejected_logps,     # (B,)
             beta=0.1):
    # 相对 reference 的「隐含奖励」
    chosen_rewards   = beta * (policy_chosen_logps   - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # 让 chosen 的隐含奖励高于 rejected
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # 诊断指标：accuracy = chosen 赢的比例，应稳定上升趋近 1
    acc = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards, rejected_rewards, acc
```

其中 `logps` 是**只对回答部分求和**的对数概率（同样要 mask 掉 prompt）：

```python
def get_batch_logps(logits, labels, label_mask):
    # logits: (B, L, V)  labels: (B, L)
    per_token = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (per_token * label_mask).sum(-1)     # (B,)
```

**注意**：DPO 训练时**必须**对 chosen 和 rejected 都跑 forward。如果 rejected 的图不同，那就是 4 次 forward（policy×2 + ref×2），显存压力大。**无卡环境的实用技巧**：把 reference 换成同一个模型 + LoRA 关闭（adapter 可以动态开关），省一份权重。

### GRPO 与可验证奖励

GRPO（Group Relative Policy Optimization）适合**答案可验证**的任务。做法是：同一个 prompt 采样 G 个回答，算各自的奖励，用组内均值/标准差做基线：

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)      # 组内相对优势
L = -E[ A_i · log π(y_i|x) ]  + β · KL(π || π_ref)
```

**客服场景的可验证奖励设计**（Day 28 的代码任务）：

```python
def reward_format(completion) -> float:
    """输出是否是合法 JSON / 是否包含必需字段"""
    try:
        obj = json.loads(extract_json(completion))
        return 1.0 if "action" in obj else 0.0
    except Exception:
        return 0.0

def reward_tool_call(completion, expected_tool) -> float:
    """工具选择是否正确"""
    return 1.0 if get_tool(completion) == expected_tool else 0.0

def reward_grounding(completion, image_regions) -> float:
    """提到的视觉证据是否真实存在于图中（用检测器验证）"""
    mentioned = extract_objects(completion)
    return len(mentioned & image_regions) / max(len(mentioned), 1)

def reward_hallucination_penalty(completion, image_regions) -> float:
    """提到图中不存在的物体，扣分"""
    phantom = extract_objects(completion) - image_regions
    return -1.0 * len(phantom)
```

`reward_hallucination_penalty` 是治幻觉最直接的手段——**让「编造」在数学上真的亏**。

## 自检问题

1. DPO 为什么不需要 Reward Model？它用什么东西替代了 reward？
2. DPO loss 里的 `log(π_θ/π_ref)` 在直觉上是什么？除以 `π_ref` 为什么重要？
3. β 太大会怎样？太小会怎样？
4. 多模态 DPO 的三种数据类型分别修什么问题？哪种最适合治幻觉？
5. 显存紧张时，DPO 可以换成哪个方法？省在哪？
6. 为什么客服场景适合用可验证奖励而不是纯人类偏好？
7. 设计一个「让模型学会说不知道」的奖励函数。

---

**上一篇**：[06-sft-training.md](06-sft-training.md) · **下一篇**：[08-evaluation.md](08-evaluation.md)
