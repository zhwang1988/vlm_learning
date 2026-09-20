# configs/ —— 配置文件说明

所有配置文件都带中文注释解释「为什么这么设」，而不是只给一个数字。

## 训练配置

| 文件 | 用途 | 何时用 |
| --- | --- | --- |
| `sft_lora_3b.yaml` | **主配置**：3B + LoRA bf16 | 默认路径。24 GB 卡能跑，质量最好 |
| `sft_qlora_3b.yaml` | 省显存：3B + QLoRA 4bit | 16 GB 卡 / Colab T4 / 想挤多实验 |
| `dpo_3b.yaml` | 偏好对齐（DPO） | Day 27–29，**必须在 SFT 之后** |
| `llamafactory_sft_3b.yaml` | LLaMA-Factory 版 SFT | 正式跑训练（快、稳）；读代码用 `src/train/sft_peft.py` |

两个训练入口产出的 adapter 格式一致（都是 PEFT 格式），可以互相接。

## 推荐的跑法顺序

```bash
# 1. 先干跑，只检查数据格式，不加载模型（10 秒，必做）
python -m src.train.sft_peft --config configs/sft_lora_3b.yaml --dry-run

# 2. 用 50 条数据试 20 步，确认 loss 在动、显存够
python -m src.train.sft_peft --config configs/sft_lora_3b.yaml \
    --max-steps 20 --limit 50

# 3. 正式跑（跑完自动关机，省钱）
python -m src.train.sft_peft --config configs/sft_lora_3b.yaml \
    && /usr/bin/shutdown
```

> 第 2 步千万别跳过。直接上全量跑 3 小时然后发现 label mask 写错了，
> 是这条路上最经典的浪费时间方式。

## 分布式配置（DeepSpeed）

| 文件 | 分片什么 | 显存效果 |
| --- | --- | --- |
| `deepspeed_zero2.json` | 优化器状态 + 梯度 | 3B 全参 bf16：单卡 56 GB → 2 卡各约 30 GB |
| `deepspeed_zero3.json` | 权重也分片 | 能装下远超单卡显存的模型，但通信开销大 |

**本项目大概率用不上它们。** 说清楚边界，免得你过早优化：

- ✅ 该用：多卡全参微调、或者单卡实在放不下激活值
- ❌ 不该用：
  - **单卡** —— ZeRO-2 在单卡上没有分片对象，纯增加复杂度
  - **QLoRA** —— NF4 量化权重和 ZeRO 分片会冲突报错；QLoRA 单卡用 `device_map` 就够
  - **3B QLoRA** —— 显存本来就够（8–12 GB），上分布式是自找麻烦

怎么用：

```bash
# 方式一：accelerate（推荐，它帮你算 batch size 和进程数）
accelerate config      # 交互式问一遍，写在 ~/.cache/huggingface/accelerate/default_config.yaml
accelerate launch -m src.train.sft_peft --config configs/sft_lora_3b.yaml

# 方式二：直接给 transformers 传
# 在 sft_peft.py 的 TrainingArguments 里加：
#   deepspeed="configs/deepspeed_zero2.json"
```

### 两个必踩的坑

1. **batch size 必须是卡数的倍数**
   `train_batch_size = per_device_batch * grad_accum * world_size`。
   不整除的话最后一步各卡步数不一致，ZeRO 会**直接 hang**，而且不报错 —— 你只会看到
   训练卡在那里不动。这是分布式最气人的一类问题。

2. **开了 ZeRO 也要开 gradient_checkpointing**
   两者治的是不同的显存：ZeRO 分片的是权重/梯度/优化器状态，
   检查点省的是激活值。多模态下激活值（尤其视觉塔那部分）占比很高，不能省。

3. **NCCL 超时先怀疑 dataloader**
   图像解码比文本慢得多。如果 `preprocessing_num_workers` 太小，
   所有卡都在等数据，看起来像通信挂了。先把 workers 调大试试（4–8）。

## 相关

- `docs/06-sft-training.md` —— 显存怎么算、LoRA 怎么配、三条曲线怎么看
- `docs/07-alignment.md` —— DPO 的 β 到底在调什么
- `scripts/estimate_vram.py` —— 换配置前先算一遍显存
- `.env.example` —— 环境变量（API key、数据库等）
