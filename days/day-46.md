# Day 46 · 对比实验与消融

> 预计 3–4h ｜ 📓 `notebooks/day-46_ablation.ipynb` ｜ ☁️ 云 GPU · `src/eval/report.py --ablation`
> 前置：Day 21/24 的评测流水线；四个版本的模型

## 今日目标（一句话）

在**同一个评测集**上跑齐 4 组：基座 / SFT / SFT+DPO / SFT+DPO+Agent，算出每一阶段贡献了多少；并明确说出「哪一步最值」「哪一步可以省」。

## 一、读（60 min）

材料：
- `src/eval/report.py` 的 `--ablation` 参数
- 回看 Day 24 的 `reports/eval_v1.md`（基线报告）

思考题（先自己想，答案在讲义或代码注释里）：
1. 如果 SFT 提升 8 分、DPO 只提升 1.5 分，DPO 还值得做吗？（提示：看不只看总分）
2. Agent 层的「提升」和模型层的提升，度量方式一样吗？
3. 消融实验最容易被忽略的控制变量是什么？

## 二、写（100 min）

跑 4 组实验（本日以执行 + 分析为主）

| 函数 / 文件 | 你要做什么 |
|---|---|
| 第 1 组 基座 | `Qwen/Qwen2.5-VL-3B-Instruct` 原版 |
| 第 2 组 SFT | W3 的 LoRA 合并版本 |
| 第 3 组 SFT+DPO | W5 的 DPO 版本 |
| 第 4 组 +Agent | W6 的 Agent 系统（用 agent_eval 的指标） |
| 统一评测集 | `data/eval/cx_eval_v1.jsonl` —— **四组必须完全一致** |
| `--ablation` 报告 | 自动生成增量贡献表 |

## 三、跑（在云 GPU 上）

```bash
# 第 1 组
python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct --eval data/eval/cx_eval_v1.jsonl --tag ab1
# 第 2 组（自动对比）
python -m src.eval.run_eval --model outputs/qwen25vl3b-cx-merged-v0 --eval data/eval/cx_eval_v1.jsonl --tag ab2 --baseline reports/eval_ab1_raw.jsonl
# 第 3 组
python -m src.eval.run_eval --model outputs/qwen25vl3b-cx-dpo-v0 --eval data/eval/cx_eval_v1.jsonl --tag ab3 --baseline reports/eval_ab2_raw.jsonl
# 生成消融报告
python -m src.eval.report --ablation reports/eval_ab1_raw.jsonl reports/eval_ab2_raw.jsonl reports/eval_ab3_raw.jsonl --out reports/ablation.md
```

期望输出（节选）：
```
[ablation] 载入 3 个模型 run + 1 个 Agent run

步骤           规则命中   judge  L4 命中  Agent成功率   增量
------------------------------------------------------------------
基座            71.2%     3.42    47.9%      —          —
+SFT            78.1%     3.88    52.1%      —        +6.9  ★最大贡献
+DPO            80.4%     4.11    55.2%      —        +2.3  ★性价比最高
+Agent           —         —       —        68.8%     +?（度量不同）

按难度看谁贡献最大：
  L1: SFT +9.4 / DPO +1.2
  L3: SFT +7.8 / DPO +3.4     ← DPO 在难题上更有用
  L4: SFT +4.2 / DPO +3.1

结论（自动生成）：
  · SFT 贡献最大，DPO 在 L3/L4 上边际收益明显高于 L1
  · Agent 层的提升主要体现在「可执行任务」，不能和模型分直接相加
  · 如果时间只够做一件事：做 SFT
```

## 四、验收清单

- [ ] `reports/ablation.md` 已产出，含每一步的增量贡献
- [ ] 四组用的是**同一个评测集**（这条必须确保，否则结论无效）
- [ ] 能明确回答「哪一步最值、哪一步可以省」
- [ ] 能解释 Agent 层的提升为什么不能和模型分直接相加

## 五、容易踩的坑

1. **四组用了不同的评测集** —— 结论直接作废。这是消融最常见的低级错误。
2. 只报总分 —— 必须分层，因为 DPO 的收益主要在难题上，总分看不出这个规律。
3. 没控制随机种子 / 解码参数 —— 生成式评测有随机性，参数不一致时差异可能是噪声。
4. 把 Agent 的成功率和模型分混在一起算「总提升」—— 两者量纲不同，不能相加。
5. 结论写「都很有用」—— 那就等于没说。必须有取舍判断。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
