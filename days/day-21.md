# Day 21 · 自动评测流水线

> 预计 3–4h ｜ 📓 `notebooks/day-21_eval_pipeline.ipynb` ｜ ☁️ 云 GPU · `src/eval/run_eval.py`、`src/eval/judge.py`
> 前置：Day 20（评测集已冻结）

## 今日目标（一句话）

一条命令跑完全套评测并出 markdown 报告；把 LLM-as-judge 的**位置偏见**校准掉，让 judge 与人工打分的相关性 ≥ 0.7。

## 一、读（60 min）

材料：
- `docs/08-evaluation.md` 第 5 节（LLM-as-judge 的偏见与校准）
- `src/eval/judge.py` 的 `JUDGE_SYSTEM` —— 看它怎么显式要求「忽略长度和格式」
- `src/eval/metrics.py` 的规则打分 —— 有规则就别用 LLM，便宜且稳定

思考题（先自己想，答案在讲义或代码注释里）：
1. 规则打分和 LLM judge 各自适合什么题？为什么要**先规则后 judge**？
2. 位置偏见怎么测？为什么必须做 A/B 交换的两次打分？
3. judge 和被评模型同源（都是 Qwen 系）会带来什么偏见？

## 二、写（100 min）

`src/eval/run_eval.py` + `src/eval/judge.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `metrics.evaluate_rules()` | must_contain / must_not_contain / 拒答 / 超承诺 / PII 五类规则 |
| `Judge.score()` | 三个维度：事实性 / 帮助性 / 语气，各自 1–5 分 |
| `pairwise_with_calibration()` | A/B 与 B/A 各打一次，抵消位置偏见 |
| `calibrate()` | 算 judge 与人工打分的 Spearman 相关，≥0.7 才算可用 |
| `build_report()` | 按难度层 / 意图分组，附失败模式与**自动生成的下一步建议** |

## 三、跑（在云 GPU 上）

```bash
# 先用规则打分跑基座（不花钱）
python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct --eval data/eval/cx_eval_v1.jsonl --tag base --no-judge
# 跑你的版本，直接与基线对比出报告
python -m src.eval.run_eval --model outputs/qwen25vl3b-cx-merged-v0 --eval data/eval/cx_eval_v1.jsonl --tag lora --baseline reports/eval_base_raw.jsonl
# 校准 judge（需要你手工打 30 条分）
python -m src.eval.judge --calibrate reports/judge_sheet.jsonl
```

期望输出（节选）：
```
[eval] data/eval/cx_eval_v1.jsonl  320 条
[model] outputs/qwen25vl3b-cx-merged-v0
[rule] 规则打分：must_contain 命中 78.1% | 格式合规 96.2% | 越界承诺 3 条
[judge] 三维度均分：事实性 3.82 / 帮助性 4.01 / 语气 4.13
[run] 320/320 完成，耗时 21 min

按难度分层：
  L1  规则命中 91.7%   judge 4.2
  L2  规则命中 80.4%   judge 3.9
  L3  规则命中 66.3%   judge 3.5     ← 掉在这里
  L4  规则命中 43.8%   judge 2.8     ← 最弱

对比基线（base）：
  规则命中 71.2% → 78.1%   Δ +6.9
  L4 反而下降 4.1  ← 需要关注

[out] reports/eval_lora.md
```

## 四、验收清单

- [ ] 一条命令跑完全套评测，输出 markdown 报告
- [ ] 报告**按难度层和意图分组**，能看出弱在哪一层（不能只有总分）
- [ ] judge 与人工打分相关性 ≥ 0.7（做 30 条人工标注校准）
- [ ] 位置偏见已通过 A/B 交换抵消，且你能说出抵消的原理

## 五、容易踩的坑

1. **judge 偏爱长答案** —— 这是最常见的偏见。`JUDGE_SYSTEM` 里显式要求「忽略长度」，但你还得用几个长短对照样本验证它真的照做了。
2. **位置偏见** —— A 放前面 consistently 得分高。不做交换打分，pairwise 结果不可信。
3. **自偏好** —— judge 和被测模型同源会系统性偏高。换一个不同家的 judge 交叉验证一次。
4. 能写规则判的却用了 LLM judge —— 贵、慢、还不稳定。规则优先。
5. 报告只给总分 —— 没有分层分组的报告，看完不知道下一步该干什么。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
