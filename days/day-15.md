# Day 15 · 跑通第一次 LoRA SFT

> 预计 3–4h ｜ 📓 `notebooks/day-15_first_lora_sft.ipynb` ｜ ☁️ 云 GPU · `configs/sft_lora_3b.yaml`、`src/train/sft_peft.py`
> 前置：Day 14；`data/processed/sft_train.jsonl` 已在数据盘；模型已下载

## 今日目标（一句话）

把训练真正跑起来 —— 先用 `--dry-run` 验数据与显存，再正式开训，亲眼看到 loss 在 50 步内开始下降，checkpoint 落盘。

## 一、读（60 min）

材料：
- `configs/sft_lora_3b.yaml` 逐项 —— 每一项都对应 Day 13/14 学过的一个概念
- `src/train/sft_peft.py` 的 `MultimodalSFTDataset` 与 `MultimodalCollator`
- `docs/06-sft-training.md` 第 5 节（第一次训练的观测清单）

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么要 `remove_unused_columns=False`？（多模态 batch 里有非标准字段，默认会被删掉）
2. `gradient_checkpointing=True` 时**必须** `use_cache=False`，为什么？
3. 为什么只保存 LoRA adapter，而不是每次存全量权重？（3B 全量 ~6 GB vs adapter ~60 MB）

## 二、写（100 min）

改配置 → dry-run → 开训（今天不写新代码，全是配置与观测）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `configs/sft_lora_3b.yaml` | 改 `max_pixels` / `grad_accum` / `num_epochs` 适配你的卡和预算 |
| `MultimodalSFTDataset` | 确认它把 assistant 段落标成了有效 label（其余 -100） |
| `MultimodalCollator` | 确认 `pixel_values` / `image_grid_thw` 被正确拼批 |
| `--dry-run` | 不加载模型，只验数据格式 + 打印一个 batch 的形状 |
| 产出目录 | `outputs/qwen25vl3b-cx-lora-v0/`（在**数据盘**上，别放系统盘） |

## 三、跑（在云 GPU 上）

```bash
# 先验数据，30 秒出结果
python -m src.train.sft_peft --config configs/sft_lora_3b.yaml --dry-run
# 正式开训。另开一个终端盯显存
python -m src.train.sft_peft --config configs/sft_lora_3b.yaml
# （另开终端）盯显存和利用率
watch -n 5 nvidia-smi
```

期望输出（节选）：
```
[data] train 1840 / eval 210 条
[sample] 图片 1 张 → 1369 visual tokens，最长序列 2048
[collator] batch 形状: input_ids (1,2048) labels (1,2048) pixel_values (1369,1176)
[lora] 可训练 30.4 M / 3043 M (1.00%)，注入 12 类模块
[dry-run] 一切正常。去掉 --dry-run 开始训练。
----------------------------------------------------------------
{'loss': 2.8412, 'grad_norm': 1.83, 'learning_rate': 1e-05, 'epoch': 0.01}
{'loss': 2.3977, 'grad_norm': 1.21, 'learning_rate': 1.98e-05, 'epoch': 0.02}
{'loss': 1.9015, 'grad_norm': 0.94, 'learning_rate': 2.95e-05, 'epoch': 0.03}
```

## 四、验收清单

- [ ] `--dry-run` 通过，且打印的 visual token 数与 Day 4 算的一致
- [ ] 正式训练启动，`loss` 在前 50 步内开始下降（不必降到很低，只要趋势向下）
- [ ] `outputs/qwen25vl3b-cx-lora-v0/` 下出现 `adapter_model.safetensors`
- [ ] 能解释日志三条曲线的**正常形态**：loss 缓降、lr 先升后降（warmup+cosine）、grad_norm 平稳

## 五、容易踩的坑

1. **chat template 错配**（头号杀手）—— 训练时手工拼字符串，推理时用 `apply_chat_template`，两边不一致 → 模型在推理时胡说。训练和推理必须走同一个模板函数。
2. **label mask 错** —— 模型学会复述问题、或学会抢答 system。这是 Day 11 的坑在训练期的复现。
3. `use_cache=True` + gradient checkpointing → 报 warning 甚至直接 OOM。
4. 显存不够时不要先动 batch，**先调小 `max_pixels`**（图片 token 是序列长度的主要贡献者）。
5. checkpoint 存到了系统盘 —— 租的机器关机可能被清空。全部放 `/root/autodl-tmp/`。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。

> 训练跑起来后**别关机器**，但可以去读 `days/day-16.md` 的准备材料。
> 今天烧的钱大约 ¥2–4，盯一眼 `nvidia-smi` 的利用率，低于 70% 说明数据加载是瓶颈。
