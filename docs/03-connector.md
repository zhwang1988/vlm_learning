# 03 · 连接器：模态对齐的那一层

> 对应 Day 3

## 概念

### 连接器要解决什么

视觉塔输出的特征是 `(N_patch, D_vision)`，LLM 要的是 `(N_token, D_llm)`。两者维度不同、语义空间不同、长度也不同。连接器干三件事：

1. **维度对齐**：`D_vision → D_llm`（如 1152 → 896）
2. **语义对齐**：把「视觉特征」翻译成「LLM 能理解的 token 表示」
3. **长度压缩**（可选）：`N_patch → N_vis`，把几千个 patch 压到几百个 token

第 3 点是连接器设计里最有价值的部分，也是最容易被忽视的。

### 四种主流方案

```
方案 1 · 线性投影（Linear）
   (N, D_v) --W--> (N, D_l)
   最早期的方案（如早期 MiniGPT-4 用一层 Linear）
   优点：几乎没有参数
   缺点：表达力不足，对齐质量差
   现代已淘汰

方案 2 · MLP（2 层 + 激活）              ← LLaVA / Qwen2.5-VL / InternVL
   (N, D_v) --Linear--> --GELU--> --Linear--> (N, D_l)
   优点：简单、稳定、效果好、参数量小（几千万）
   缺点：不压缩 token 数
   事实标准

方案 3 · Q-Former / Perceiver Resampler   ← BLIP-2 / Flamingo / Idefics
   (N, D_v) 作为 K,V
   M 个可学习的 query 作为 Q
        ↓ 多层 cross-attention
   (M, D_l)，其中 M << N
   优点：能把 2000 个 patch 压到 32/64 个 query token
   缺点：训练难、需要两阶段训练、压缩会丢细节

方案 4 · 交叉注意力直接插进 LLM 层        ← Flamingo
   在 LLM 的每 k 层插入一个 gated cross-attention 层，
   直接让文本 token attend 到视觉特征
   优点：信息交互最深
   缺点：改动 LLM 结构，训练不稳、显存高
```

### 为什么 LLaVA 用最笨的 MLP 反而赢了

这是个经典问题。答案是：

> **图像塔和 LLM 都已经预训练得很好了，连接器只需要做一个「坐标变换」，不需要复杂的表达能力。**

- 视觉塔已经学过「什么是猫」，LLM 已经学过「猫这个词是什么意思」。连接器的任务只是把前者的表示搬到后者的坐标系里。
- 任务越简单，越不容易训崩。Q-Former 的可学习 query 反而引入了额外的训练难度。
- MLP 的参数量小（约 20M）→ 小数据上不易过拟合。

**这正是「后训练」思路的体现**：不要重新发明架构，最大化复用已有能力。

### 但 MLP 不压缩 token —— 这就成了新的瓶颈

MLP 保真但贵。一张 1024×1024 的图 → 1332 个 visual token（Qwen2.5-VL 已做 2×2 merge）。

```
成本对照（假设 batch=8，每个样本 1 图 + 100 字问题）：
  文本部分：8 × 150 tok = 1.2k
  视觉部分：8 × 1332 tok = 10.6k      ← 占 90%
  
  序列长度主要是被图撑起来的。
```

所以工业界有两个方向：

- **往下压**：在 MLP 之前做 pooling / pixel shuffle（InternVL 的 pixel shuffle 把 4 个 patch 合成 1 个，token 数 /4）
- **往上堆**：不改连接器，直接吃长上下文（Qwen2.5-VL 支持到 128k，代价是显存）

**工程取舍**：客服场景的图通常不大（商品图、瑕疵特写、快递单），单图 1000 token 左右可以接受。**不要过早优化 token 数，先跑通质量，第 5 周再考虑压缩。**

## 工程细节

### 一个最小的 MLP 连接器

```python
import torch.nn as nn

class MLPConnector(nn.Module):
    def __init__(self, d_vision=1152, d_llm=896, hidden=None):
        super().__init__()
        hidden = hidden or d_vision          # LLaVA 用视觉维度做隐藏层
        self.proj = nn.Sequential(
            nn.Linear(d_vision, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_llm),
        )

    def forward(self, x):                    # x: (B, N, D_v)
        return self.proj(x)                  # (B, N, D_l)
```

**注意**：Qwen2.5-VL 实际用的是 `nn.Linear` 的堆叠，没有 LayerNorm。不要自作聪明加 LN——和官方权重不匹配。

### Perceiver Resampler

```python
class PerceiverResampler(nn.Module):
    def __init__(self, d_vision, d_llm, n_query=64, n_head=8, n_layer=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_query, d_llm) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverLayer(d_llm, n_head) for _ in range(n_layer)
        ])
        self.proj_in = nn.Linear(d_vision, d_llm) if d_vision != d_llm else nn.Identity()

    def forward(self, x):                    # (B, N, D_v)
        x = self.proj_in(x)                  # (B, N, D_l)
        lat = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for layer in self.layers:
            lat = layer(lat, x)              # query=lat, kv=x
        return lat                           # (B, n_query, D_l)
```

关键在 `PerceiverLayer`：**query 是 latent，key/value 是视觉特征**。这样输出长度恒等于 `n_query`，与输入图大小无关。

**代价**：64 个 query 要装下整张图的信息，细节必然损失。所以纯 Resampler 方案在 OCR 上表现差——这也是 Qwen 选择不做压缩的原因。

### 三种连接器的实测对比（你在 Day 3 要跑）

| 连接器 | 参数量 | 输出 token（1024²图） | 显存 | 预期效果 |
|---|---|---|---|---|
| Linear | ~1M | 1332 | 最低 | 差 |
| MLP (2 层) | ~2.5M | 1332 | 低 | **好** |
| Perceiver (64 query) | ~28M | 64 | **最低** | 中，细节差 |
| Perceiver (256 query) | ~110M | 256 | 低 | 中上 |

**记住这个结论**：参数量和效果不是正相关的，**输出 token 数才是效果的关键**。这条经验在后面做数据构造（要不要把图缩小）和推理优化（要不要降分辨率）时会反复用到。

### LayerNorm 的位置陷阱

不同实现里 LN 的位置不一样，加载预训练权重时容易错配：

```
LLaVA / Qwen 系：  Linear → GELU → Linear          （无 LN）
一些实现：          LayerNorm → Linear → GELU → Linear
```

加载官方权重时，**直接打印 state_dict 的 key 名和 shape 对齐**，不要凭记忆写结构。这是 Day 3 代码任务里的一个必做检查。

## 自检问题

1. 连接器做的三件事分别是什么？
2. 为什么 MLP 这种「笨」方案在后训练场景下反而最好？
3. Perceiver Resampler 的 query 和 key/value 分别是什么？输出长度由谁决定？
4. 一张 1024² 的图在 MLP 连接器下产生多少 visual token？在整个序列里占比如何？
5. 如果你的场景是「读一张带小字的价格标签」，你会选 MLP 还是 Perceiver？为什么？

---

**上一篇**：[02-vision-encoder.md](02-vision-encoder.md) · **下一篇**：[04-qwen25vl.md](04-qwen25vl.md)
