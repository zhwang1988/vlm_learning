# Day 27 · 跑 DPO

> 预计 3–4h ｜ 📓 `notebooks/day-27_run_dpo.ipynb` ｜ ☁️ 云 GPU · `src/train/dpo.py`、`configs/dpo_3b.yaml`
> 前置：Day 26（偏好数据就位）；W3 训练出的 SFT adapter

## 今日目标（一句话）

从 **SFT 版本**（不是基座）出发跑 DPO，看到 `rewards/chosen − rewards/rejected` 的差值稳步上升，并在领域集上测出与 SFT 版本的差异。

## 一、读（60 min）

材料：
- `configs/dpo_3b.yaml` 逐项 —— 特别注意 `beta` 和 `learning_rate`
- `src/train/dpo.py` 的 `--inspect` 路径（不用 torch 就能看数据）
- `docs/07-alignment.md` 第 5 节（DPO 训练的观测指标）

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么 DPO 的学习率要比 SFT 小一个量级？
2. 为什么必须从 SFT 版本出发，而不是从基座直接 DPO？
3. `reward_margin` 一直涨是好事吗？什么时候该停？

## 二、写（100 min）

`src/train/dpo.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `--inspect` | 先看数据格式与 chosen/rejected 长度分布（**本地就能跑**） |
| `--dry-run` | 验数据 + 显存，不加载模型 |
| `--sft-adapter` | 指定 SFT 的 LoRA 作为起点 —— 这是关键参数 |
| `beta` | 默认 0.1；数据噪声大时调大到 0.3 更稳 |
| `--no-qlora` | 显存够时关掉 4-bit（DPO 对量化更敏感） |

## 三、跑（在云 GPU 上）

```bash
# 先看数据（本地可跑）
python -m src.train.dpo --inspect --in data/processed/dpo_train.jsonl
# 验数据与显存
python -m src.train.dpo --config configs/dpo_3b.yaml --in data/processed/dpo_train.jsonl --in data/processed/dpo_contrastive.jsonl --dry-run
# 正式开训
python -m src.train.dpo --config configs/dpo_3b.yaml --in data/processed/dpo_train.jsonl --in data/processed/dpo_contrastive.jsonl --sft-adapter outputs/qwen25vl3b-cx-lora-v0
```

期望输出（节选）：
```
[inspect] 1067 对偏好数据
  chosen   长度 p50/p95 = 82 / 210
  rejected 长度 p50/p95 = 31 /  88      ← rejected 明显更短，警惕「长度偏见」
  来源分布：bad_case 67 / contrastive 1000
  类型分布：同图不同答 600 / 同答不同图 400 / 错误回答 67

（训练中）
{'loss': 0.6102, 'rewards/chosen': 0.041, 'rewards/rejected': -0.038,
 'rewards/margins': 0.079, 'rewards/accuracies': 0.71}
{'loss': 0.4318, 'rewards/chosen': 0.118, 'rewards/rejected': -0.142,
 'rewards/margins': 0.260, 'rewards/accuracies': 0.86}
```

## 四、验收清单

- [ ] `--inspect` 显示 chosen/rejected 的长度分布（**必须检查长度偏见**）
- [ ] 训练启动，`rewards/margins` 稳步上升、`rewards/accuracies` 趋向 1
- [ ] 在 `cx_eval_v1` 上跑出 DPO 版本 vs SFT 版本的对照（哪怕差异很小）
- [ ] 能说出「这次 DPO 有没有训过头」（依据是 margin 曲线还是别的？）

## 五、容易踩的坑

1. **从基座直接 DPO** —— 基座连格式都不会，DPO 学不到东西。必须先 SFT。
2. 学习率沿用 SFT 的 2e-5 —— 太大，DPO 会崩。典型 5e-7 ~ 5e-6。
3. **长度偏见** —— rejected 系统性短于 chosen 时，模型可能只学会「说长一点」。检查 p50/p95，必要时用 `average_log_prob` 归一化（SimPO 思路）。
4. 训太久 —— margin 涨到某个点后模型开始钻空子（reward hacking），表现为通用能力下降。用领域评测集守着，宁可早停。
5. 忘了 `--sft-adapter` —— 起点错了，白训。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
