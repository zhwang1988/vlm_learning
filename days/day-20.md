# Day 20 · 构建客服领域评测集

> 预计 3–4h ｜ 📓 `notebooks/day-20_build_domain_eval.ipynb` ｜ 💻 本地 · `src/eval/build_domain_eval.py`
> 前置：Day 12 的 `data/processed/clean.jsonl`；Day 19（知道通用榜的局限了）

## 今日目标（一句话）

从真实商品与对话里构造 **300+ 条**客服评测样本，分 4 个难度层，并写一份标注手册 —— 这是整个项目**最重要的一份资产**。

## 一、读（60 min）

材料：
- `docs/08-evaluation.md` 第 3–4 节（标注规范、一致性、难度分层）
- `src/eval/build_domain_eval.py` 里的 `L1`–`L4` 模板

思考题（先自己想，答案在讲义或代码注释里）：
1. L1（直接问答）和 L4（多约束+拒答判断）各自的典型样本长什么样？
2. 为什么标准答案不能用「必须一字不差」来判？（会惩罚正确的同义表达）
3. 评测集和训练集的**图像级泄漏**为什么比对文本去重更致命？

## 二、写（100 min）

`src/eval/build_domain_eval.py`（已给实现，你要改配额与模板）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `EvalSample` | question / image / must_contain / must_not_contain / tier / intent |
| L1–L4 模板 | 四个难度层的出题模板 —— **今天最该改的就是这一块** |
| `build_eval_set()` | 按配额抽样：L1 30% / L2 35% / L3 25% / L4 10% |
| `generate_eval_card()` | 输出 `EVAL_CARD.md`，含泄漏检查 + 冻结声明 |
| `check_leakage()` | 用 pHash 查训练集与评测集的图像重叠 —— **必须为 0** |

> 评测集一旦生成就**冻结**，之后不许再改（改了就变成「对着答案调模型」）。
> 要改就出 `cx_eval_v2`，v1 的分数永远保留。

## 三、跑（本地（无需 GPU））

```bash
# 先跑离线自检：模板静态检查 + 四个难度层都必须非空
python -m src.eval.build_domain_eval --selftest
# 生成 320 条评测样本 + 卡片
python -m src.eval.build_domain_eval --source data/processed/clean.jsonl --n 320 --out data/eval/cx_eval_v1.jsonl --card
# （上一条的 --card 只生成卡片；这条重跑一遍确认四层分布）
python -m src.eval.build_domain_eval --source data/processed/clean.jsonl --out data/eval/cx_eval_v1.jsonl
# 肉眼抽查格式
head -20 data/eval/cx_eval_v1.jsonl
```

期望输出（节选）：
```
✓ 写出 273 条评测样本 → data/eval/cx_eval_v1.jsonl

难度分布:
  L1:   94  █████████████
  L2:   69  ██████████
  L3:   80  ███████████
  L4:   30  ████

⚠️  L4 样本必须人工复核！(30 条)
✓ 评测集卡片 → data/eval/EVAL_CARD.md
```

## 四、验收清单

- [ ] `cx_eval_v1.jsonl` 含 300±50 条，**4 个难度层都有量**，8 类意图全部覆盖
- [ ] `python -m src.eval.build_domain_eval --selftest` 全绿
- [ ] `EVAL_CARD.md` 里有泄漏检查结果（必须为 0）和**冻结声明**
- [ ] 标注手册写清了「怎么判对」—— 关键是同义表达算对
- [ ] 隔一周自己重标 30 条，一致性 ≥ 85%（这条需要留时间做）

## 五、容易踩的坑

1. **图像级泄漏** —— 同一张商品图既在训练集又在评测集，分数虚高。pHash 检查必须过。
2. ⭐ **`must_contain` 里写「类别描述」而不是字面关键词。**
   判分函数 `check_must_contain` 做的是**子串匹配**：
   `hits = [k for k in must_contain if k in answer]`。
   写「颜色类关键词」的话，字面上永远不会出现在任何回答里 →
   命中数恒为 0 → **这批样本对任何模型都判失败**。
   后果是报告上「规则通过率」偏低，看起来像「模型不行」——
   实际上是评测集自己坏了。这种错误不抛异常、不发告警。
   本项目实测踩过：L1 的 4 个模板里 2 个写的是「颜色类关键词」/
   「款式类关键词」。现在 `--selftest` 的 `_lint_templates()` 会拦住。
3. ⭐ **模板的分组键必须和源数据字段对得上。**
   L2 的模板是按 **intent**（`quality_issue` / `color_mismatch` /
   `material`）分组的，而查候选池时用了 `by_type`（image_type）——
   两边的键一个都对不上，`by_type.get(...)` 全返回空，
   于是 **L2 整整一层恒为 0 条**（35% 的评测样本静默消失）。
   报告上只表现为「L2: 0」一行，看上去像个正常的空桶。
   现在空层是**硬错误**（`SystemExit`），不再是统计数字。
4. 只造简单题 —— L1 占 90%、L4 一条没有。配额是硬约束，不要妥协。
5. 标准答案写成「必须一字不差」—— 会把正确的同义表达判成错，评测结果失去意义。
6. 评测集没有冻结 —— 一边调模型一边改答案，等于自己骗自己。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
