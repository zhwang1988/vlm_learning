# 04 · Qwen2.5-VL 架构精读

> 对应 Day 4（也是本项目基座模型的说明书）

## 概念

### 为什么单独给它一篇

接下来 7 周，你所有代码都跑在这个模型上。它的三个特性会直接影响你的数据构造、显存估算和推理优化：

1. **原生动态分辨率** → 你的数据里图片尺寸是自由的，不用手工 resize 到固定尺寸
2. **M-RoPE（多模态旋转位置编码）** → 图像、视频、文本共享一套位置编码机制
3. **窗口注意力 + 全注意力混合** → 决定了显存曲线不是简单的 O(n²)

### 整体结构

```
                    ┌──────────────────────────────┐
图片 (任意尺寸)  →   │  ViT (窗口注意力 + 全注意力混合) │
                    │   patch 14×14, 32 层           │
                    │   输出 (N, 1280)               │
                    └───────────┬──────────────────┘
                                │
                    ┌───────────▼──────────────────┐
                    │  2×2 patch merging            │  ← token 数 /4
                    │  再过一个 MLP 压缩 (merge MLP) │
                    └───────────┬──────────────────┘
                                │
                    ┌───────────▼──────────────────┐
                    │  MLP Connector                │
                    │  (1280 → 3584 或 2048)        │  ← 对齐 LLM 维度
                    └───────────┬──────────────────┘
                                │
文本 → tokenizer → embedding ───┴───→ Qwen2.5 LLM → 生成
```

### 特性一：原生动态分辨率（Naive Dynamic Resolution）

**做法**：不 resize 到固定尺寸。而是：

1. 把图**按原比例**缩放到「总像素数不超过阈值」（3B 版默认总视觉 token 数上限约 1280）
2. 用 ViT 处理，patch 数随图大小变化
3. 输出**变长**的 visual token 序列

```
原图 1200×800
  ↓ 按上限缩放（保持比例）
1260×840（假设）
  ↓ patch 14×14
90 × 60 = 5400 个 patch
  ↓ ViT
5400 × 1280
  ↓ 2×2 merging
1350 × 1280
  ↓ 上限裁剪到 ~1280
1280 × 1280
  ↓ MLP
1280 × 2048   ← 送进 LLM 的 visual token
```

**直接后果（对你很重要）**：

| 现象 | 说明 |
|---|---|
| visual token 数是变量 | 同一个 batch 里每张图的 token 数不同 → 必须 padding，浪费显存 |
| 图片比例被保留 | 好处：长图详情页、宽 banner 都不会变形 |
| 小图省 token | 一张 200×200 的小图只花几十个 token，很便宜 |
| 显存不可预测 | 一张 4000×3000 的高清图和一张 800×600 的图差 25 倍 |

**数据构造启示（Day 11）**：训练时最好把图片按 token 数**分桶（bucketing）**打包，让同一 batch 的图大小接近，否则 padding 会浪费一半显存。

### 特性二：M-RoPE（Multimodal Rotary Position Embedding）

普通 RoPE 只给序列位置一个索引 `pos`。M-RoPE 把它拆成三个：

```
文本 token:  (t, h, w) = (pos, pos, pos)      ← 三个维度和序列位置一致
图像 token:  (t, h, w) = (0, row, col)        ← t 固定，h/w 是二维网格坐标
视频 token:  (t, h, w) = (frame_idx, row, col) ← t 是帧号
```

**为什么这么设计**：

- 文本只关心一维顺序 → 三维退化成同一维，行为和普通 RoPE 一致
- 图像需要有**空间关系**：第 (3,5) 个 patch 和第 (3,6) 个是相邻的，第 (3,5) 和第 (8,5) 是上下关系。用 (t=0, h=3, w=5) 编码就能让注意力感知这种二维邻近性
- 视频是三维的（时间 + 空间），天然适合

**工程实现上**：M-RoPE 把 `head_dim` 分成三段，分别用 t/h/w 的频率做旋转。默认比例通常是 16:24:24（共 64 维）。

**这对你有什么影响**：

- 图像的空间关系被隐式建模了 → 模型能回答「图里左边那个和右边那个哪个贵」这类问题
- 你在做数据时**不需要显式加坐标信息**，模型自己知道
- 但如果你把图切成了多个 tile 分开送进去，**tile 之间的空间关系就丢了**。这是切图方案（AnyRes）的固有缺陷。

### 特性三：混合注意力（窗口 + 全注意力）

ViT 里 32 层，大部分层是**窗口注意力**（window attention），少数层是**全注意力**（full attention）。

```
全注意力：每个 patch 都看其他所有 patch     → O(N²)，贵，但能建模全局关系
窗口注意力：每个 patch 只看邻域窗口内       → O(N·w²)，便宜，但只有局部视野

Qwen2.5-VL 的做法（简化）：
  大部分层：窗口注意力（如 8×8 patch 的窗口内）
  每 4 层插入 1 层全注意力（+ 可学习的窗口间链接）
```

**效果**：既保留全局建模能力，又把视觉塔的计算量降下来。这让你能在 Colab 上处理 1000+ token 的图。

**对你的影响**：视觉塔的计算量不是显存瓶颈，**LLM 部分才是**。所以优化时优先动 LLM（LoRA、量化、减少 visual token 数）。

### 特性四：视频的 3D 卷积与绝对时间

Qwen2.5-VL 把视频帧的时间维也纳入 patch 化（3D conv，temporal patch = 2 帧合并），并且引入了**绝对时间编码**——模型知道第 5 秒的事件在第 3 秒之后。

**注意**：第二版 Qwen2.5-VL 支持动态帧率（dynamic FPS），可以按事件密度自适应采样，而不是固定每秒 1 帧。本项目不用视频，知道有这回事即可（加餐方向会用到）。

## 工程细节

### 计算 visual token 数

这是 Day 4 的核心代码任务。逻辑（简化版）：

```python
def compute_visual_tokens(h, w, patch=14, merge=2, min_pixels, max_pixels):
    # 1. 保持比例，把总像素限制在 [min_pixels, max_pixels]
    h_bar = round(h / patch) * patch
    w_bar = round(w / patch) * patch

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / patch) * patch
        w_bar = math.floor(w / beta / patch) * patch
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / patch) * patch
        w_bar = math.ceil(w * beta / patch) * patch

    # 2. patch 数
    grid_h, grid_w = h_bar // patch, w_bar // patch

    # 3. merge 后 token 数
    n_tokens = (grid_h // merge) * (grid_w // merge)
    return n_tokens, (grid_h, grid_w)
```

**验证方法**：拿官方 `AutoProcessor` 处理同一张图，对比 `input_ids` 里 `<|image_pad|>` 的个数。**必须完全一致**，否则你后面一定会踩 shape mismatch。

### 关键超参数（3B-Instuct 默认值）

| 参数 | 值 | 含义 |
|---|---|---|
| `patch_size` | 14 | ViT patch 边长 |
| `spatial_merge_size` | 2 | 2×2 合并 |
| `min_pixels` | 56×56 (≈3136) | 图再小也放大到这个量 |
| `max_pixels` | 1280×28×28 (≈1,003,520) | 单图 token 上限约 1280 |
| `temporal_patch_size` | 2 | 视频时间维合并 |
| `hidden_size` (ViT) | 1280 | 视觉特征维度 |
| `hidden_size` (LLM) | 2048 (3B) / 3584 (7B) | 主干维度 |
| `num_layers` (ViT) | 32 | |

### 调整 max_pixels 的权衡

```python
# 想要更高清（读小字、看瑕疵）
processor.image_processor.max_pixels = 2560 * 28 * 28   # token 数翻倍

# 想要更快更省
processor.image_processor.max_pixels = 640 * 28 * 28    # token 数减半
```

**实验建议**（Day 21 评测时会做）：在你的客服评测集上跑三档 `max_pixels`，画出「准确率 vs 延迟」曲线。**大概率你会发现有个拐点**——过了某个分辨率，准确率不再涨但延迟继续涨。这个拐点就是你的生产配置。

### 官方 processor 的正确用法

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": "这件衣服是什么材质？"},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
# inputs["input_ids"] 里数 <|image_pad|> 的个数 == 你算出来的 n_tokens
```

**注意**：图像在 `content` 里的顺序 = 在最终序列里的顺序。上面这个例子图像在文字前面。想试试「文字在前」的变体（Day 12 会做），把两个 dict 换个位置即可。

### 3B vs 7B 的取舍（对你的算力条件）

| | 3B | 7B |
|---|---|---|
| bf16 权重 | ~7 GB | ~16 GB |
| LoRA 训练显存 (seq 2048, gc on) | **约 14–16 GB** | 约 26–32 GB |
| QLoRA 训练显存 | 约 8–10 GB | 约 14–18 GB |
| Colab T4 (16G) | **勉强可以**（QLoRA 稳） | 不行 |
| Colab A100 (40G) | 很轻松 | 可以 |
| 通用推理能力 | 中 | 明显更好 |
| 领域 SFT 后 | **接近 7B 基座** | 更强 |

**建议**：**主用 3B**。原因很简单——你的领域优势来自数据，不来自参数量。把 20h/周投在数据和评测上的收益，远大于纠结 3B 还是 7B。7B 留到第 5 周有云算力时再试。

## 自检问题

1. Qwen2.5-VL 的「原生动态分辨率」和 LLaVA 的「AnyRes 切图」有什么本质区别？各自丢失了什么？
2. M-RoPE 里 t/h/w 三个维度分别编码什么？图像 token 的 t 为什么固定为 0？
3. 一张 800×600 的图，经过 2×2 merging 后大约有多少 visual token？（自己算，patch=14）
4. 为什么说「视觉塔不是瓶颈，LLM 才是」？
5. `max_pixels` 调大一倍会带来什么？什么时候值得调？
6. 3B 和 7B 在你的无卡条件下分别要选什么训练方案？

---

**上一篇**：[03-connector.md](03-connector.md) · **下一篇**：[05-data-engineering.md](05-data-engineering.md)
