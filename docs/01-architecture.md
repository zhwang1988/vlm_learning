# 01 · VLM 架构：三大件

> 对应 Day 1–D5

## 概念

### 全景

一个现代 VLM 就是三个模块串起来：

```
┌─────────────┐   ┌───────────────┐   ┌──────────────┐
│  视觉编码器  │ → │    连接器      │ → │   语言主干    │
│ Vision Enc. │   │  Connector    │   │  LLM Backbone│
├─────────────┤   ├───────────────┤   ├──────────────┤
│ SigLIP/ViT  │   │ MLP / Q-Former│   │ Qwen2.5      │
│ ~400M–6B    │   │ / Perceiver   │   │ 0.5B–72B     │
│ 冻结 or 微调 │   │ 必训          │   │ 冻结 or LoRA │
└─────────────┘   └───────────────┘   └──────────────┘
     ↑                    ↑                   ↑
  学"看"              学"翻译"             学"推理表达"
```

**训练阶段的分工**（这是最重要的心智模型）：

| 阶段 | 视觉编码器 | 连接器 | LLM | 目的 |
|---|---|---|---|---|
| Stage 0 对齐 | 冻结 | **训练** | 冻结 | 让视觉特征能映射到 LLM 的语义空间 |
| Stage 1 预训练 | 可选解冻 | 训练 | 训练 | 多任务大规模图文学习 |
| Stage 2 SFT | 冻结 | 训练 | LoRA/全参 | 指令跟随，学会按人要求回答 |
| Stage 3 对齐 | 冻结 | 训练 | LoRA | 偏好对齐，更helpful/更安全 |

**本项目走的是 Stage 2 + Stage 3**：拿已经做完 Stage 0/1 的 Qwen2.5-VL 官方权重，在上面做领域 SFT 和 DPO。这也叫「后训练（post-training）」。

> 为什么不做 Stage 0/1？因为那需要千万级图文对和几百卡天。后训练是个人和小团队唯一现实的路径，而且**在垂直场景上收益往往比通用的 Stage 1 更大**——因为领域数据是别人没有的。

### 三大件各自的取舍

**视觉编码器**

- 越大越细：InternVL 用 6B 的 InternViT，能读清小字和细纹理；Qwen2.5-VL 用约 675M 的 ViT + 窗口注意力。
- 分辨率 > 参数量：对 OCR / 商品细节来说，把图放大比把模型加大更有效。
- 冻结 vs 微调：**小数据（<100k）下冻结更稳**。视觉塔一解冻，很容易把预训练学到的通用视觉能力训崩（灾难性遗忘），这在小数据集上特别明显。

**连接器**

细节见 `03-connector.md`。核心结论：**MLP 最简单且不差，Perceiver/Q-Former 能压 token 数**。

**语言主干**

- 参数量决定「世界知识」和「推理能力」上限。
- 3B 够用来做领域 SFT（因为领域知识你灌进数据里了），7B 在通用推理上明显更强。
- 主干的选择还决定了 tokenizer 和 chat template，这两样是踩坑重灾区。

### 一个容易忽略的点：图像在序列中的位置

```
方案 1（LLaVA）：  <image> 用户问题
方案 2（Qwen）：    用户问题 <image> 补充说明
方案 3（交错）：     用户问 <image1> 这是A吗 <image2> 那这个呢
```

同一个模型，图像放前放后效果可能差很多。原因是 causal attention：图像在前的 token 无法看到后面的文字，反之亦然。

**客服场景的实践结论**：把图像放在**用户问题之后**通常更好——模型先知道「要回答什么」，再去看图，注意力更聚焦。这一点你在 Day 12 构造数据时要保持一致。

## 工程细节

### 视觉特征是怎么「插」进文本序列的

以 Qwen2.5-VL 为例（简化）：

```python
# 1. 文本里有个占位符
text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>这是什么材质？<|im_end|>"

# 2. tokenizer 把 <|image_pad|> 切成若干个 image_pad token
#    Qwen 用 <|image_pad|> 重复 N 次的方式，N = visual token 数
input_ids = [..., VISION_START, PAD, PAD, ..., PAD, VISION_END, ...]
#                                 └── N 个 ──┘

# 3. 模型 forward 时，把 embedding 矩阵里 PAD 位置替换成视觉特征
inputs_embeds = embed(input_ids)                        # (B, L, D)
mask = input_ids == IMAGE_PAD_ID                        # (B, L)
inputs_embeds[mask] = visual_features.reshape(-1, D)     # 替换

# 4. 正常走 LLM
out = llm(inputs_embeds=inputs_embeds, attention_mask=...)
```

**所以「visual token 数」必须是事先算好的**，且必须和 `input_ids` 里 PAD 的个数严格一致。这是多模态训练最经典的 bug 来源：processor 算出的 N 和实际特征数的 N 对不上，报一个 shape mismatch，你要去翻 processor 源码。

### 三类常见报错和解法

| 报错 | 原因 | 解法 |
|---|---|---|
| `shape mismatch in index_put_` | visual token 数和 PAD 数不一致 | 检查 processor 是否被正确调用，图像是否被 resize 过两次 |
| loss 一直是 0 / 不降 | label mask 把 assistant 部分也 mask 掉了 | 检查 label 里非 -100 的位置是否覆盖 assistant 回复 |
| 生成全是重复 | chat template 错配（用了 base 模板而非 instruct 模板） | 对比官方 `apply_chat_template` 输出 |

### 冻结策略的实际影响

```
全冻结（最保守）
  显存最小、最稳，但只能学「怎么说」，学不会「怎么看」
  → 适合：数据 <10k、场景与预训练分布接近

Connector + LLM 的 LoRA（本项目选择）
  平衡点：能学新知识、能改表达风格、显存可控
  → 适合：10k–200k 数据、垂直领域

Connector + LLM 全参
  效果上限最高，但 3B 全参 SFT 至少要 40G+ 显存（bf16 + gradient checkpointing）
  → 适合：数据 >100k、有 A100/H100

再加视觉塔 LoRA
  能学领域特有的视觉特征（如特定商品的瑕疵形态）
  风险：小数据下容易过拟合、破坏通用视觉能力
  → 本项目在 Day 46 消融里试一下，不作为默认
```

### 参数量和显存的经验公式

```
训练显存 ≈ 模型参数量 × (2 [bf16权重] + 2 [梯度] + 8 [AdamW 优化器状态]) 字节
          × 1/可训练比例
        + 激活值显存
```

- 全参 bf16 + AdamW：约 **12 字节/参数**
- 3B 全参 → 36 GB，仅权重和优化器状态，还没算激活值 → 实际需要 60G+
- 3B LoRA (r=16)：可训练参数约 0.5%，优化器状态几乎为 0 → 权重 6G + 激活值 → 16G 卡勉强能跑，开 gradient checkpointing

**QLoRA 的关键**：把冻结的 base 权重量化成 4bit（6G → 1.5G），进一步省。

Day 13 的 `scripts/estimate_vram.py` 会把这些算清楚。

## 自检问题

1. VLM 训练分为几个阶段？每个阶段解冻哪些模块？为什么？
2. 为什么个人做后训练应该从 Stage 2 起步，而不是从头做 Stage 0？
3. 「图像放在问题前面还是后面」为什么会影响效果？用 causal attention 解释。
4. `inputs_embeds` 替换法里，visual token 数为什么要提前算准？算错的报错长什么样？
5. 3B 模型全参 bf16 + AdamW 大约需要多少显存？LoRA 能省在哪一部分？

---

**上一篇**：[00-orientation.md](00-orientation.md) · **下一篇**：[02-vision-encoder.md](02-vision-encoder.md)
