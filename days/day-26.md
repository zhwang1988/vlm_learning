# Day 26 · 多模态偏好数据构造

> 预计 3–4h ｜ 📓 `notebooks/day-26_preference_data.ipynb` ｜ 💻 本地 · `src/train/dpo_loss.py` Part B
> 前置：Day 23 的 `reports/bad_cases.jsonl`；Day 25

## 今日目标（一句话）

从 W4 的 bad case 反向构造 (chosen, rejected) 对，凑够 3k 对；并亲手构造**同图不同答**和**同答不同图**两种多模态特有的偏好对。

## 一、读（60 min）

材料：
- `docs/07-alignment.md` 第 4 节（多模态偏好数据的三种类型）
- `src/train/dpo_loss.py` 的 `build_preference_from_badcases` / `build_contrastive_pairs`

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么「模型真实犯过的错」比「人造的错误回答」更适合做 rejected？
2. **同图不同答**治的是什么病？**同答不同图**又治的是什么病？
3. 如果 chosen 和 rejected 只差语气（一个礼貌一个冷淡），模型能学到东西吗？

## 二、写（100 min）

`src/train/dpo_loss.py` Part B（已给实现，你要扩来源）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `build_preference_from_badcases()` | 字段契约：`query` / `model_answer` / `reference`（由 error_analysis 输出） |
| `build_contrastive_pairs()` | 同图不同答 / 同答不同图 —— **多模态版的幻觉解药** |
| 抽检脚本 | 随机抽 30 对，判断「rejected 是否真的更差，且差异可学习」 |
| 去重 | 同一 (图, 问) 不要出现多对，否则某一类被过度加权 |
| 配比 | 幻觉类 / 漏信息类 / 语气类 大致 4:4:2 |

## 三、跑（本地（无需 GPU））

```bash
# 从 W4 的 bad case 造（主力来源）
python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl
# 造对比式偏好对（幻觉解药）
python -m src.train.dpo_loss --contrastive data/processed/clean.jsonl --n-pairs 1000
# 看两个文件的量
wc -l data/processed/dpo_train.jsonl data/processed/dpo_contrastive.jsonl
```

期望输出（节选）：
```
读入 71 条 bad case
⚠️  跳过 4 条：缺 model_answer 或 reference 字段
✓ 写出 67 对偏好数据 → data/processed/dpo_train.jsonl

从 1840 条样本构造对比式偏好对
  同图不同答： 600 对
  同答不同图： 400 对
✓ 写出 1000 对 → data/processed/dpo_contrastive.jsonl

data/processed/dpo_train.jsonl         67 行
data/processed/dpo_contrastive.jsonl  1000 行
```

## 四、验收清单

- [ ] `dpo_train.jsonl` 有实打实的偏好对（不是 0 对）
- [ ] 抽检 30 对，逐对判断「rejected 确实更差，且差异是**可学习的**」
- [ ] 三种类型的偏好对都有：同图不同答 / 同答不同图 / 格式或工具调用
- [ ] 能说清哪一类偏好对治的是「幻觉」这个具体病

## 五、容易踩的坑

1. **silent 0** —— 字段名不对时脚本会写出 0 对。现在会告警，但你要认得这个信号（正确字段是 `query` / `model_answer` / `reference`）。
2. chosen 与 rejected 只差语气 —— 这类对占太多，模型只学会「更啰嗦」而没学会「更准确」。
3. **同图不同答时图没对齐** —— 两张图张冠李戴，模型学到的是噪声。构造时必须校验图 ID。
4. rejected 全部来自幻觉类 —— 偏好数据分布单一，DPO 会过度惩罚「不确定」的表达。
5. 不抽检直接开训 —— DPO 对数据质量极其敏感，垃圾进，崩坏出（比 SFT 更脆）。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
