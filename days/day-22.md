# Day 22 · 幻觉评测与缓解

> 预计 3–4h ｜ 📓 `notebooks/day-22_hallucination_eval.ipynb` ｜ 💻 本地 · `src/eval/hallucination.py`
> 前置：Day 21

## 今日目标（一句话）

用 POPE 式诱导问题测出幻觉率，**同时测漏答率**，再实现「不确定就说不确定」的缓解策略并做 A/B 对比。

## 一、读（60 min）

材料：
- `docs/08-evaluation.md` 第 6 节 + `docs/12-papers.md` 里 POPE 那一行
- `src/eval/hallucination.py` 里的 `MITIGATION_STRATEGIES`（4 个策略）

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么**只看幻觉率会骗人**？「全答没有」的模型幻觉率是多少？
2. 存在的幻觉（说了图里没有的东西）和漏报（图里有的却说没有）—— 哪个对客服伤害更大？
3. 「不确定就说不确定」的策略，副作用是什么？（提示：用户满意度）

## 二、写（100 min）

`src/eval/hallucination.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `build_probes()` | 对每张图构造「存在/不存在」两类问题（POPE 式） |
| `parse_yes_no()` | 解析模型的回答 —— 要处理「是的」「有的」「并没有」这些变体 |
| `compute_hallucination_rate()` | **同时返回幻觉率和漏答率**（两个都要报） |
| `MITIGATION_STRATEGIES` | 4 个策略：明确允许说不确定 / 要求先描述再回答 / 降低温度 / 两轮自检 |
| `run_mitigation_experiment()` | 控制变量，一次只改一个策略 |

## 三、跑（本地（无需 GPU））

```bash
# 自检：构造 probes + 统计逻辑
python -m src.eval.hallucination
# 生成正式的诱导问题集
python -m src.eval.hallucination --build-probes --out data/eval/probes_v1.jsonl
# 在你的模型上实测幻觉率
python -m src.eval.run_eval --model outputs/qwen25vl3b-cx-merged-v0 --eval data/eval/probes_v1.jsonl --tag halluc
```

期望输出（节选）：
```
[probes] 100 张图 × 2 类问题 = 200 条
         存在的物体 100 条 / 不存在的物体 100 条
[strategy] 基线（无缓解）
  幻觉率 = 23.0%   （问了不存在的物体，答"有"）
  漏答率 =  9.0%   （问了存在的物体，答"没有"）

[strategy] 明确允许说不确定
  幻觉率 = 11.0%  ↓12.0     漏答率 = 16.0%  ↑7.0    ← 代价
[strategy] 要求先描述再回答
  幻觉率 = 17.0%  ↓ 6.0     漏答率 = 10.0%  ↑1.0    ← 最划算
```

## 四、验收清单

- [ ] 幻觉率与漏答率**两列一起报**（只报一个的结论一律不采信）
- [ ] 能说清三类幻觉（物体存在性 / 属性 / 关系）的成因差异
- [ ] 4 个缓解策略里至少实测 2 个，并用控制变量方式对比
- [ ] 能解释「降低幻觉率往往抬高漏答率」这个 trade-off

## 五、容易踩的坑

1. **只看幻觉率** —— 一个永远回答「没有」的模型幻觉率是 0%，但完全没用。必须同时看漏答率。
2. probe 里的物体太离谱 —— 比如给一张衣服图问「有飞机吗」。模型不可能答错，测不出东西。要选**视觉上可能混淆**的物体（同色系、同类目、背景里出现过的）。
3. 缓解策略对比时同时改了多个变量 —— 结论无效。一次只改一个。
4. 属性幻觉和关系幻觉用同一套 probe —— 它们是不同题型，要分开造。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
