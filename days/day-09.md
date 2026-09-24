# Day 9 · 数据合成

> 预计 3–4h（含等待 API 返回）｜ 📓 `notebooks/day-09_data_synthesis.ipynb` ｜ 💻 `src/data/synth.py`
> 今天第一次花钱（API 费）。**先 `--limit 20` 试跑，人工抽检通过再放量。**

## 今日目标（一句话）

用强模型（qwen-vl-plus）按 Day 8 的矩阵批量合成图文客服对话，
跑出第一批 2k 条，人工抽检合格率 ≥ 70%。

## 一、读（45 min）

- `docs/05-data-engineering.md` 第 5 节（合成三范式 + 质量陷阱）
- `.env.example` 第 1 节注释（API 提供商怎么配）
- `src/data/synth.py` 头部：`SYSTEM_PROMPT` 里的**负向约束**（禁「首先/其次/最后」）为什么存在

## 二、写（75 min）

`src/data/synth.py`（骨架已给）：

| 组件 | 作用 | 你的动作 |
|---|---|---|
| `PERSONAS` × `EMOTIONS` | 反同质化：说话人设 × 情绪随机组合 | 各加 3 个你从真实客服里见过的风格 |
| `CLARIFY_INSTRUCTION` | 让模型学会追问缺失信息 | 造 2 个「信息不全」的场景验证 |
| `ESCALATE_INSTRUCTION` | 让模型学会转人工 | 同上 |
| `validate_sample` | 规则校验（长度/空答/复读） | 跑通，故意喂一条坏数据确认能拦住 |
| `Synthesizer.run` | 断点续跑主循环 | 用 `--limit 20` 先跑 |

**费用控制**（重要）：
```bash
# 第一步：20 条试跑，几分钱
python -m src.data.synth --limit 20 --out data/synthetic/pilot.jsonl
# 人工抽检这 20 条 → 通过后再放量到 2000 条（约几块钱）
```

## 三、跑（本地即可，不占 GPU）

```bash
jupyter lab   # notebooks/day-09_data_synthesis.ipynb
# notebook：读 pilot.jsonl → 打印样本 → 统计意图/情绪分布 → 人工抽检打分表
```

## 四、验收清单

- [ ] 20 条试跑，人工抽检 ≥ 70% 合格（不合格先改 prompt 再放量，**不要硬着头皮跑全量**）
- [ ] 2k 条正式跑完，`data/synthetic/` 有 jsonl + 断点文件
- [ ] 分布对照 Day 8 矩阵：偏差最大的意图记进打卡（Day 10 清洗时补偿）
- [ ] 抽 5 条「多轮追问」的样本，确认追问自然（不是模板复读）

## 五、容易踩的坑

1. **同质化** —— 所有回答都是「亲，这款……」。对策：personas/emotions 随机化 + 温度 ≥ 0.9。
2. **幻觉参数** —— 强模型会编造「面料是 95% 棉」，但图里根本没有标签。对策：合成 prompt 里只允许引用图中可见信息。
3. **API 限流** —— 并发别开太高；断点文件让你随时 kill 重跑。
4. 别用 GPT-4o 合成「中文客服」语感 —— qwen-vl-plus 便宜且味儿正。

## 六、打卡

三行照旧 + 记一笔：**今天花了多少钱、合成了多少条、单条成本**。
（W2 结束时你要能报出「数据集总成本」这个数。）
