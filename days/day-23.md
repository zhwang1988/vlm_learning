# Day 23 · 错误分析与 Bad Case 归类

> 预计 3–4h ｜ 📓 `notebooks/day-23_error_analysis.ipynb` ｜ 💻 本地 · `src/eval/error_analysis.py`
> 前置：Day 21/22（有 `reports/eval_*_raw.jsonl` 结果文件）

## 今日目标（一句话）

把失败样本自动聚类 + 关键词归类，输出 top 10 失败模式，并导出 `bad_cases.jsonl` —— 这份文件是 Day 26 构造 DPO 数据的原料。

## 一、读（60 min）

材料：
- （今天没有新讲义）—— 全部时间用来读失败样本
- `src/eval/error_analysis.py` 的 `ERROR_RULES` 和 `KEYWORD_GROUPS`

思考题（先自己想，答案在讲义或代码注释里）：
1. 自动化归类结果如果不可解释，还有价值吗？怎么让它可解释？
2. 「答非所问」和「答得不全」是两类问题，修复手段一样吗？
3. top 10 失败模式里，哪几类其实源于同一个根因？

## 二、写（100 min）

`src/eval/error_analysis.py`（已给实现，你要扩规则）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `ERROR_RULES` | 规则归类：漏信息 / 幻觉参数 / 拒答不当 / 格式错 / 越界承诺 / PII |
| `KEYWORD_GROUPS` | 关键词组 —— **今天最值得你扩充的地方**（业务词表） |
| `semantic_group()` | 语义聚类，把规则覆盖不到的聚成新簇 |
| `classify()` | 单条样本 → （错误类型, 证据片段） |
| `analyze()` | 输出 `error_analysis.md` + **`bad_cases.jsonl`**（带 `error_type` 标签） |

## 三、跑（本地（无需 GPU））

```bash
# ⭐ 没有 GPU 也先跑这个：造「好 / 坏 / 全拒答」三种预测，验证归类和报告能不能区分
make eval-fake
# 真实数据跑归类，出报告 + bad_cases.jsonl
python -m src.eval.error_analysis --in reports/eval_lora_raw.jsonl --out-dir reports/
# （离线）用假预测跑一遍，确认 bad_cases 格式正确
python -m src.eval.error_analysis --in reports/eval_fake_bad_raw.jsonl --out-dir reports/
# 看 bad case 的格式（Day 26 要用）
head -3 reports/bad_cases.jsonl
```

期望输出（节选）：
```
读入 273 条评测记录
✓ 报告 → reports/error_analysis.md
✓ Bad cases → reports/bad_cases.jsonl  (225 条)
  下一步（Day 26）：python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl
  （注意：bad_cases 里的 reference 需要有参考答案才能构造偏好对）

======================================================================
失败模式 Top 5（语义主题）
======================================================================
  要素缺失          223  ████████████████████
  质量不达标         20  ██
```

## 四、验收清单

- [ ] 错误分类表已产出，每类**至少 2 条典型样本**
- [ ] 能明确指出**下一轮该补哪三类数据**（不是「都补」）
- [ ] `bad_cases.jsonl` 已按 `error_type` 打标（Day 26 直接消费）
- [ ] 能指出哪几类失败其实同源（比如「漏信息」和「不当拒答」可能都是数据不足）

## 五、容易踩的坑

1. 只统计不读样本 —— 分类表再漂亮，不读原始样本也发现不了真问题。
2. 聚类结果不可解释 —— 一堆「簇 3、簇 7」没人看得懂。要么加关键词命名，要么别用。
3. `bad_cases.jsonl` 忘了打标签 —— 没有 `error_type` 就没法分层构造 DPO 数据。
4. 把「模型答得啰嗦」也当失败 —— 风格问题不是错误，别污染 bad case 池。
5. ⭐ **「失败」的定义只能有一套。**
   实测踩过：`report.py` 里有的地方用 `rule["passed"]` 判失败、
   有的地方用 `rule["score"] < 0.5`，两套口径不一致 ——
   于是同一份报告能同时打印出「规则命中 64.8%」和「没有失败样本」。
   **一个报告里出现两种『失败』定义，比不报错更危险**：
   它不会崩，只会让你对着自相矛盾的数字猜。
   修法是把判据收进一个函数（`is_fail()`），所有地方都调它。
6. ⭐ **「0 泄漏 / 0 失败 / 幻觉率 0%」必须带上分母。**
   只看分子的话，「真的没问题」和「根本没检查」长得一模一样。
   报告里凡是出现 0，都要能回答：分母是多少？怎么算出来的？

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
