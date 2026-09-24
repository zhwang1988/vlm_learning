# Day 25 · DPO 家族原理

> 预计 3–4h ｜ 📓 `notebooks/day-25_dpo_family.ipynb` ｜ ☁️ 云 GPU · `src/train/dpo_loss.py`、`docs/07-alignment.md`
> 前置：Day 24（评测报告 v1）

## 今日目标（一句话）

手写 DPO loss（含 reference 项）并用玩具数据验证数值 —— 重点是跑通那个 **`ln 2` 检查**：chosen 与 rejected 完全相同时 loss 必须等于 0.6931。

## 一、读（60 min）

材料：
- `docs/07-alignment.md` 第 1–3 节（RLHF → DPO → ORPO → KTO → GRPO 的关系图）
- DPO 原论文的公式推导部分（`docs/12-papers.md` 标了重点）

思考题（先自己想，答案在讲义或代码注释里）：
1. DPO 的闭式解是怎么推出来的？为什么可以不需要 reward model？
2. β 控制什么？β=0.01 和 β=1.0 分别会发生什么？
3. 为什么 `log(π_θ/π_ref)` 里的 π_ref 是「锚」—— 去掉它会发生什么？

## 二、写（100 min）

`src/train/dpo_loss.py` Part A（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `get_batch_logps()` | 只要被监督部分的 log 概率和；注意别把 prompt 也算进去 |
| `dpo_loss()` | 四项 logps 进，loss + 指标出（含 `reward_margin`） |
| `label_smoothing` | cDPO：对偏好标签做平滑，抗噪声标注 |
| `verify_dpo_loss()` | **4 个场景 + ln2 检查** —— 今天必须全部通过 |
| `_ensure_torch()` | 惰性导入：让 Part B 的造数据路径在本地也能跑 |

## 三、跑（在云 GPU 上）

```bash
# 数值验证（要 torch，不需要 GPU）
python -m src.train.dpo_loss --verify
```

期望输出（节选）：
```
DPO Loss 数值验证
==============================================================================
[1] chosen 明显优于 rejected
    loss = 0.0472   accuracy = 1.00   margin = 3.20     ← 应该是很小的 loss

[2] chosen 与 rejected 完全相同  ← 关键检查
    loss = 0.6931   ← 必须等于 ln 2 = 0.693147
    accuracy = 0.00

[3] chosen 略优于 rejected
    loss = 0.5128   accuracy = 1.00

[4] label_smoothing=0.1（cDPO）
    loss = 0.5823   ← 比场景 1 大：平滑让 loss 不会压到 0
==============================================================================
✓ 4 个场景全部通过
```

## 四、验收清单

- [ ] `--verify` 四个场景断言全部通过，**特别是 ln2 = 0.6931 那一项**
- [ ] 能自己推出 DPO 的闭式解（不用翻论文）
- [ ] 能说清 β 的作用，以及为什么典型值是 0.1–0.5
- [ ] 知道 ORPO / SimPO / KTO 分别省掉了什么（reference / 配对假设 / 正负样本）

## 五、容易踩的坑

1. **忘了 reference 要 detach** —— π_ref 有梯度的话，训练目标就变了，怎么调都不对。
2. **logps 把 prompt 部分也算进去了** —— 那算的是「整段对话的概率」而不是「回答的概率」，DPO 立刻失效。label mask 必须只覆盖 assistant 段（Day 11 的老坑换了个地方）。
3. β 设得太大 —— 模型被锚死在 reference 附近，学不动；太小则语言能力崩坏。
4. reference model 用了和 policy 同一个实例 —— 等价于没除，DPO 退化成毫无意义的损失。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
