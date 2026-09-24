# Day 19 · 通用 VLM 评测全景

> 预计 3–4h ｜ 📓 `notebooks/day-19_benchmark_overview.ipynb` ｜ ☁️ 云 GPU · `src/eval/run_benchmark.py`（今日新增）、`docs/08-evaluation.md`
> 前置：Day 18（有了合并后的权重）

## 今日目标（一句话）

在自己的模型上跑 1–2 个通用榜的子集，拿到真实分数；并说清「通用榜单高分 ≠ 你的客服场景好用」的三个具体原因。

## 一、读（60 min）

材料：
- `docs/08-evaluation.md` 第 1–2 节 —— 每个榜测什么能力，一张表看完
- MMMU / MMBench / OCRBench / POPE / HallusionBench / DocVQA / MathVista 的**官方说明页**
- `docs/12-papers.md` 里 POPE 那一行（今天先扫一眼，Day 22 精读）

思考题（先自己想，答案在讲义或代码注释里）：
1. OCRBench 和 DocVQA 都在测「看图读字」，区别是什么？
2. 为什么 MMMU 分数高，不代表退货政策问答答得好？
3. 榜单数据可能已经被模型在预训练里见过（数据污染）—— 怎么粗略判断？

## 二、写（100 min）

`src/eval/run_benchmark.py`（今日新增，已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `SUITES` 注册表 | 定义每个子集的名字、数据源、评分函数 —— 想加榜就加一行 |
| `load_suite()` | 从本地缓存读子集（不联网也能跑） |
| `score()` | 按题型走不同评分：选择题精确匹配 / OCR 用归一化编辑距离 |
| `run()` | 统一走 `LocalVLM` 接口（和 `run_eval.py` 复用同一个推理后端） |
| 报告 | 输出「分数 + 用了多少条 + 是否子集」—— **子集必须标注**，别冒充全量 |

## 三、跑（在云 GPU 上）

```bash
# 先跑 OCR 子集 100 条
python -m src.eval.run_benchmark --model outputs/qwen25vl3b-cx-merged-v0 --suite ocrbench --n 100
# 同参数跑基座，作为对照
python -m src.eval.run_benchmark --model Qwen/Qwen2.5-VL-3B-Instruct --suite ocrbench --n 100
# 看看支持哪些榜
python -m src.eval.run_benchmark --list
```

期望输出（节选）：
```
[suite] ocrbench  子集 100/1000 条（⚠️ 这是子集，不能和榜单数字直接比）
[model] outputs/qwen25vl3b-cx-merged-v0
[run] 100/100 完成，耗时 412s，峰值显存 9.1 GB
[score] 归一化编辑距离均值 = 0.612  → 近似准确率 61.2%

对照（同一子集）：
  基座 Qwen2.5-VL-3B-Instruct : 63.8%
  你的 SFT 版本               : 61.2%   Δ = -2.6
  → 微调在通用 OCR 上略降是**正常**的（能力被收窄到客服域）
```

## 四、验收清单

- [ ] 至少 1 个榜单子集跑出分数，且**标注了这是子集**
- [ ] 基座与微调版本在同一子集上的对照数字
- [ ] 能说出「通用榜单高分 ≠ 场景好用」的**三个具体原因**（不是空话）
- [ ] 能解释「微调后通用能力略降」为什么是正常现象

## 五、容易踩的坑

1. benchmark 的 prompt 格式与训练格式不一致 —— 你用它考模型，模型却没见过这种问法。
2. 拿子集分数冒充全量榜单 —— 报告里必须写清 `子集 N/M`，这是学术诚信问题。
3. 忘了跑基座对照 —— 没有 baseline 的分数毫无意义。
4. 忽略数据污染 —— 有些榜的答案可能已在预训练语料里，分数虚高。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
