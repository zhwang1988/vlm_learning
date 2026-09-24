# Day 14 · LoRA / QLoRA 原理与实现

> 预计 3–4h ｜ 📓 `notebooks/day-14_lora_principles.ipynb` ｜ ☁️ 云 GPU · `src/train/lora_utils.py`
> 前置：Day 13（显存账已算清）

## 今日目标（一句话）

手写一个最小 LoRA 层（低秩 A/B 矩阵），验证「B 零初始化 → 输出与基座逐元素相等」，并搞清 `r` / `alpha` / `target_modules` 在 VLM 里怎么选。

## 一、读（60 min）

材料：
- `docs/06-sft-training.md` 第 4 节
- LoRA 原论文的关键 5 页（`docs/12-papers.md` 标了阅读重点，别通读）
- `src/train/lora_utils.py` 顶部注释 —— 里面写了 target_modules 的选取逻辑

思考题（先自己想，答案在讲义或代码注释里）：
1. B 为什么必须**零初始化**？如果随机初始化，训练第一步会发生什么？
2. `scaling = alpha / r` 的作用是什么？把 r 从 8 改成 64，alpha 要不要跟着改？
3. 为什么 `FORBIDDEN = ["visual"]` —— 视觉塔禁止注入 LoRA？

## 二、写（100 min）

`src/train/lora_utils.py`

| 函数 / 文件 | 你要做什么 |
|---|---|
| `LoRALinear.__init__` | A 用 kaiming 初始化、**B 用 zeros**；`scaling = alpha / r` |
| `LoRALinear.forward` | `base(x) + dropout(x) @ A.T @ B.T * scaling` |
| `LoRALinear.merge` | 把 BA 加回 base 权重 —— 推理时零额外开销 |
| `resolve_target_modules` | `q/k/v/o` + `gate/up/down` + **`merger.mlp`** |
| `FORBIDDEN` | 视觉塔永不注入（冻结 + 不注入，双保险） |
| `estimate_training_vram` | 和 Day 13 的 `estimate_vram.py` 交叉验证，两个数应该对得上 |

> **今天最该记住的一句**：LoRA 不是「小模型」，是「给大模型加了一小块可训练的旁路」，原权重全程冻结、只更新 A 和 B。

## 三、跑（在云 GPU 上）

```bash
# B 零初始化后，输出应与基座逐元素相等
python -m src.train.lora_utils --demo
# r=8/16/64 的参数量与显存对照
python -m src.train.lora_utils --table
```

期望输出（节选）：
```
[1] LoRALinear 数学正确性
    基座输出与加 LoRA 后输出 allclose = True   ← B 是零，必须相等
    手动 merge 后 allclose = True

[2] 注入计划
    可注入 (12 类) : q_proj k_proj v_proj o_proj gate_proj up_proj down_proj merger.mlp.0 merger.mlp.2 ...
    禁止注入       : visual.*   ← 视觉塔
    可训练参数     : 30.4 M  (0.86%)

[3] r 的影响（以 q_proj 为例）
     r=8    →  0.05 M/层      r=16  →  0.10 M/层      r=64  →  0.39 M/层
     r 翻 4 倍，参数量也翻 4 倍；alpha/r 会把这个差异抵消掉一部分
```

## 四、验收清单

- [ ] `--demo` 全部断言通过（**B=0 时 `torch.allclose` 必须为 True**）
- [ ] 能说清 `r` / `alpha` / `dropout` / `target_modules` 各自作用和调参直觉
- [ ] 能说出「为什么客服场景要优先注入 `merger.mlp`」（视觉和语言的对齐层）
- [ ] 知道 `merge_and_unload()` 之后推理为什么和全量模型一样快

## 五、容易踩的坑

1. **B 用了随机初始化** —— 训练一开始 loss 就炸，而且很难查（不是数据问题，是初始化问题）。
2. 忘了乘 `scaling` —— r 变大时有效学习率跟着变，超参搜索白做。
3. **漏掉 `merger.mlp`** —— 只注入 LLM 内部层，视觉和语言的对齐层不更新，SFT 收益大打折扣。这是 VLM 版 LoRA 和纯文本版最大的差别。
4. 在 fp16 下 merge —— 会掉精度。merge 要在 fp32/bf16 做，然后转回目标精度。
5. r 一味调大 —— 参数量线性涨，收益却很快饱和。客服场景 r=16 通常够，先固定 r 调数据。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
