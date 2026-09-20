# 06 · SFT 训练工程

> 对应 Day 13–Day 18

## 概念

### SFT 在做什么

给模型看成千上万条 `(对话上下文, 理想回复)`，最小化理想回复的负对数似然：

```
L = - Σ log P(y_t | y_<t, x)      # x = 图文上下文, y = 理想回复
```

就这一个损失，没有玄机。**SFT 的质量差异 90% 来自数据，10% 来自超参。**

但「简单」不等于「容易跑对」。多模态 SFT 的坑比纯文本多一倍，因为多了图像预处理、视觉 token 对齐、变长序列三件事。

### 三个必须搞清楚的显存去向

训练显存由四块组成：

```
① 模型权重        params × 2 bytes (bf16)
② 梯度            params × 2 bytes (bf16)  ← 只对可训练参数
③ 优化器状态      params × 8 bytes (AdamW: fp32 的 m 和 v)
④ 激活值          batch × seq_len × hidden × layers × 系数
```

**关键结论**：

- ①②③ 是**静态**的，和 batch 无关
- ④ 是**动态**的，是唯一能用 batch size / gradient checkpointing / 序列长度调节的
- 全参训练时 ③ 最大（8 字节/参数），LoRA 时 ③ 几乎为 0

**全参 bf16 + AdamW ≈ 12 字节/参数**：

| 模型 | ①+②+③ | 加激活值（seq 2048） | 加激活值（seq 4096） |
|---|---|---|---|
| Qwen2.5-VL-3B | 36 GB | ~50 GB | ~70 GB |
| Qwen2.5-VL-7B | 84 GB | ~105 GB | ~140 GB |

**LoRA (r=16，约 0.5% 可训练)**：

| 模型 | 静态部分 | seq 2048 + gc | seq 4096 + gc |
|---|---|---|---|
| 3B | ~7 GB | **~14 GB** | ~18 GB |
| 7B | ~17 GB | ~26 GB | ~34 GB |

**QLoRA（base 4bit）**：

| 模型 | 静态 | seq 2048 + gc |
|---|---|---|
| 3B | ~2.5 GB | **~8 GB** |
| 7B | ~6 GB | ~14 GB |

> 结论：**没有本地卡也能训**。Colab T4 (16G) 跑 3B QLoRA、Colab A100 (40G) 跑 3B LoRA 甚至 7B QLoRA，都可行。Day 13 的 `scripts/estimate_vram.py` 会把公式写清楚。

### LoRA 到底在做什么

冻结原权重 W，旁路一个低秩更新：

```
h = W x + ΔW x = W x + (B A) x

W: (d_out, d_in)    冻结
A: (r, d_in)        初始化 N(0, σ)
B: (d_out, r)       初始化 0        ← 关键是 B 初始为 0，所以 ΔW 初始为 0
```

**为什么 B 初始化为 0**：训练开始时 ΔW = 0，模型行为与基座完全一致，不会破坏已学知识。这是 LoRA 稳定的关键。

**四个超参**：

| 超参 | 作用 | 推荐值 | 调参直觉 |
|---|---|---|---|
| `r` | 低秩维度，控制容量 | 8–64 | 领域差距大就大一点。16 是安全默认 |
| `alpha` | 缩放因子，`ΔW × alpha/r` | 通常 = 2r | 只调 alpha 不改 r 等价于调学习率 |
| `dropout` | LoRA 层 dropout | 0.05–0.1 | 小数据（<5k）用 0.1 |
| `target_modules` | 加在哪些线性层 | 见下 | **影响最大** |

**target_modules 的选择**（对 VLM 尤其重要）：

```python
# 保守：只加注意力投影
["q_proj", "v_proj"]                          # 参数最少，效果一般

# 标准：注意力 + MLP
["q_proj", "k_proj", "v_proj", "o_proj",
 "gate_proj", "up_proj", "down_proj"]         # 推荐

# 激进：再加上连接器
+ ["merger.mlp.0", "merger.mlp.2"]            # ← VLM 特有！
```

**最后一行是 VLM 的独门技巧**：把 LoRA 也加在连接器上。这样模型能同时调整「怎么看」和「怎么说」，在领域数据上通常有额外收益。**本项目默认开启**。

注意不要加在视觉塔（`visual.`）上——除非你有 50k+ 数据，否则会损害通用视觉能力。

### 学习率、调度、epoch

| 超参 | LoRA | 全参 | 说明 |
|---|---|---|---|
| learning_rate | 1e-4 ~ 2e-4 | 1e-5 ~ 2e-5 | **差一个数量级**，别搞错 |
| lr_scheduler | cosine | cosine | |
| warmup_ratio | 0.03 | 0.03 | 防止前期梯度爆炸 |
| weight_decay | 0 | 0.01 | LoRA 通常不衰减 |
| num_train_epochs | 2–3 | 2–3 | 1 epoch 欠拟合，>3 易过拟合 |
| per_device_batch_size | 1–2 | 1 | VLM 的 batch 开不大 |
| gradient_accumulation | 8–16 | 16 | 有效 batch = bsz × accum × n_gpu |
| max_grad_norm | 1.0 | 1.0 | 梯度裁剪 |

**VLM 的学习率有个经验**：比纯文本微调**再低一点**。因为视觉部分的信息更「脆」，学习率高了容易把图文对齐关系训坏。

**模型选择的依据**：不要只看最后一步。**按 eval loss 最低的 checkpoint 选**，而且要看 eval loss 的形态——如果 eval loss 在第 1 个 epoch 后就上升，说明过拟合，立刻停。

## 工程细节

### 三条曲线的正常形态

```
train loss
  1.8 |●
      | ●●
  1.2 |   ●●●
      |      ●●●●●●●●●●  ← 平滑下降，尾部趋缓
  0.6 |                
      +----------------→ step
      正常：单调下降，尾部变平。抖动能看出噪点但趋势清晰

grad_norm
  2.0 |  ●
      | ●            ← 初期有小尖峰正常
  0.5 |  ●●●●●●●●●●  ← 稳定在低位
      +----------------→ step
      异常：持续 >10 或突然飙升 → 梯度爆炸，降 lr 或加 warmup

learning_rate
      |●
      | ●●
      |   ●●●
      |       ●●●●●●  ← cosine 衰减到接近 0
      +----------------→ step
```

**loss 不降的五个原因**（按排查顺序）：

1. **label 全被 mask 了** → 打印一条样本，确认 label 里有非 -100 的 token
2. **学习率太小** → LoRA 用 1e-5 基本不降，要 1e-4
3. **数据格式错** → chat template 或 image pad 数量不对，模型在学噪声
4. **梯度没传** → 检查 `requires_grad`，确认 LoRA 层真的可训练
5. **数据本身太乱** → 回到 Day 10 的清洗报告

### 常见报错的解法

**报错 1：`RuntimeError: The size of tensor a (1332) must match tensor b (1330)`**

视觉 token 数不匹配。原因通常是：
- 自己 resize 过图，又让 processor resize 了一次
- 用 `max_pixels` 改了 processor 参数，但数据里的图已经按旧参数处理过

解法：统一在 processor 里做预处理，不要在数据构造阶段 resize。

**报错 2：`CUDA out of memory`**

按代价从低到高依次尝试：

```python
per_device_train_batch_size=1          # 先降到 1
gradient_checkpointing=True            # 用时间换显存，省 30-50%
gradient_accumulation_steps=16         # 补回有效 batch
max_pixels=640*28*28                   # 降图像分辨率，token 减半
model.config.use_cache=False           # 必须关，否则和 gc 冲突
optim="paged_adamw_8bit"               # bitsandbytes 分页优化器
bf16=True                              # 确认没在用 fp32
# 最后手段：换 QLoRA
```

**报错 3：生成结果重复或乱码**

`use_cache=False` 之后忘了在推理时开回来，或者 chat template 用错。检查：
```python
model.config.use_cache = True   # 推理前必须设回 True
```

### LLaMA-Factory 路线 vs 原生 transformers 路线

| | LLaMA-Factory | 原生 transformers + peft |
|---|---|---|
| 上手速度 | 快，改 yaml 就行 | 慢，要自己写 Trainer |
| 可控性 | 中 | 高 |
| 多模态支持 | 好（有内置 template） | 自己拼 |
| 调试友好度 | 差（黑盒） | 好 |
| 适合 | 快速跑通、对比实验 | 理解原理、自定义 loss/数据 |

**建议**：**Day 15 用 LLaMA-Factory 快速跑通**（建立信心、拿到基线），**Day 17–18 读一遍原生实现**（理解细节）。两条路都走一遍，收获最大。

### LLaMA-Factory 配置示例（Day 15 用）

```yaml
# configs/sft_lora_3b.yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
do_train: true

# 数据集
dataset: cx_vlm_sft_v0
dataset_dir: data/processed
template: qwen2_vl
cutoff_len: 4096
overwrite_cache: true

# LoRA
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all              # 或显式列出 + merger.mlp

# 训练
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
max_grad_norm: 1.0

# 评估
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
save_steps: 100
logging_steps: 10
load_best_model_at_end: true
metric_for_best_model: eval_loss

# 输出
output_dir: outputs/qwen25vl3b-cx-lora-v0
report_to: tensorboard
```

### 分布式：ZeRO 三个阶段

单卡跑不动时往上走。DeepSpeed ZeRO 的思路是**把训练状态切分到多卡**：

```
ZeRO-1：切分优化器状态        显存 ↓ 4x    通信 低
ZeRO-2：再切分梯度            显存 ↓ 8x    通信 中       ← 最常用
ZeRO-3：再切分权重            显存 ↓ Nx     通信 高       ← 大模型必须
```

**VLM 特有的坑**：视觉塔和 LLM 参数量差距巨大（675M vs 3B）。ZeRO-3 切分时如果按层切，会有严重的负载不均（有些卡几乎没活干）。DeepSpeed 的 `stage3_prefetch_bucket_size` 之类的参数需要调。**但这是 7B 多卡才遇到的问题，本项目可以不管。**

**offload**：ZeRO-3 + `offload_optimizer` 把优化器状态放 CPU 内存，能进一步省显存，但慢 2–3 倍。Colab 上不可用（CPU 内存太小）。

## 自检问题

1. 全参 bf16 训练大约多少字节/参数？这 12 字节分别是什么？
2. LoRA 为什么要把 B 矩阵初始化为 0？
3. `alpha` 和 `r` 的关系是什么？只调 alpha 等效于调什么？
4. VLM 的 LoRA 为什么要额外加在 `merger` 上？为什么不能加在视觉塔上？
5. LoRA 的学习率比全参高还是低？差多少？
6. loss 不降的五个原因，按排查顺序排列。
7. 3B 模型在 16G 卡上训练，你要开哪几个显存优化？按代价排序。
8. `gradient_checkpointing` 和 `use_cache` 为什么冲突？

---

**上一篇**：[05-data-engineering.md](05-data-engineering.md) · **下一篇**：[07-alignment.md](07-alignment.md)
