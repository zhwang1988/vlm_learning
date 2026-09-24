# Day 11 · 数据打包与对话模板（W2 技术含量最高的一天）

> 预计 3–4h ｜ 📓 `notebooks/day-11_build_sft.ipynb` ｜ 💻 `src/data/build_sft.py`
> 本地跑。今天出错的样本，Day 14 训练时会以「loss 不收敛」的形式报复你。

## 今日目标（一句话）

把清洗后的对话转成训练格式：正确的 chat template、正确的多图占位、
**label mask 只盖住 assistant 段（含 `<|im_end|>`）**，
**按图片整体分组切分**，并通过图像级泄漏检查。

## 一、读（60 min）

- `docs/05-data-engineering.md` 第 8 节（label mask 三种错误画法）
- `docs/06-sft-training.md` 里 label mask 一节
- Qwen2.5-VL chat template 源码（notebook 里直接打印出来看）：
  `<|im_start|>role\n...<|im_end|>` 结构 + `<|vision_start|><|image_pad|><|vision_end|>`

## 二、写（90 min）

`src/data/build_sft.py`（骨架已给）：

| 函数 | 作用 | 你的动作 |
|---|---|---|
| `to_llamafactory_format` | 统一中间格式 | 跑通 |
| `build_labeled_sample` | **手工构造 label_mask** | 逐 token 检查：assistant 起止在哪 |
| `split_by_image` | ⭐ **按图片分组**切分 train/val/test | 确认同一张图不跨集合 |
| `check_split_by_image` | 路径级自检（不读图，必定能跑） | 理解它和 `check_leakage` 的分工 |
| `check_leakage` | pHash 跨集泄漏检查 | 造一对泄漏样本，确认它能拦住 |

**label mask 的三种错误**（今天第二个考点，notebook 里各造一个反例）：
1. 把 system prompt 也算进 loss（模型学会抢答）
2. 漏掉 `<|im_end|>`（模型学不会停 → 推理时说个没完）
3. 把 user 问题算进 loss（模型学会复述问题）

### ⭐ 今天的**最大**考点：切分必须按图片分组

先说清一件事：**同一张商品图配很多个不同的提问，那是有效样本** ——
Day 10 的去重专门要「同图不同文必须保留」。

但到了切分，同一张图就**只能整体进一个集合**。否则这张图的一个提问在 train、
另一个在 test，你测的就是「模型有没有记住这张图」，分数必然虚高，
而且**从数字上完全看不出来**。

容易犯的两个错：

| 错误写法 | 后果 | 实测现象 |
|---|---|---|
| `random.shuffle` 后直接切片 | 同图跨集合 → 评测虚高 | 泄漏检查报 10 处 `d=0` 的「自己配自己」 |
| 分层后**在层内**分配 | 每层只剩一两个组，`累计到 target 就切` 恒成立 | **train 0 / val 1 / test 135**，全跑 test 去了 |

第二个错的修法是：按层**轮转取组**得到交替序列（保住分布），
再按**全局**目标配比把组分给三个集合（`split_by_image` 就是这么做的）。

## 三、跑（本地）

```bash
# ① 离线验证「同图不同文必须保留」和四种有效样本不误删
make demo-check

# ② 真实数据打包
python -m src.data.build_sft --in data/processed/clean.jsonl --out-dir data/processed
python -m src.data.build_sft --inspect data/processed/sft_train.jsonl 0   # 肉眼看第一条
jupyter lab notebooks/day-11_build_sft.ipynb
```

⚠️ 输入是 **`data/processed/clean.jsonl`**（Day 10 的输出），不是 `data/clean/clean.jsonl`。

期望输出：
```
读入 1712 条
按图片分组：640 组（其中 180 组含多条样本，最大一组 6 条）
分层切分: train 1370 / val 171 / test 171

图片级泄漏检查...
  ✓ 检查 128 张评测图 × 512 张训练图，无泄漏
```

`--inspect` 期望输出：
```
=== 样本 0 ===
system   : [masked]  (loss=-100)
user     : [masked]  (loss=-100)  + 1 张图 → 428 visual tokens
assistant: 「您好，M 码有现货……」 (loss=正常)  含 <|im_end|> ✓
```

### 泄漏检查要读三个数，不是一个

```
"n_eval_checked"    : 128      ← 真的比对过的评测图
"n_eval_skipped"    : 0        ← 路径读不到、退出的
"coverage"          : 1.0      ← 覆盖率
"n_leak_pairs"      : 0        ← 泄漏**对**数
"n_leak_eval_images": 0        ← 涉及泄漏的评测**图**数
"leak_rate"         : 0.0      ← 上一个 ÷ n_eval_checked
```

**「0 泄漏」有两种含义，差别是天壤之别：**

| 情况 | n_eval_checked | n_leak | 结论 |
|---|---|---|---|
| 真的没泄漏 | 128（很大） | 0 | 可以放心评测 |
| 一张图都没读到 | **0** | 0 | **这个「通过」不能当数** |

第二种在上云训练时极其常见：写好的 `image_path` 在云机器上不存在，
或者训完把数据拉回本地后路径全变了。
**分母塌成 0 时，任何比率都好看。**

另外注意 `leak_rate` 的分子分母必须是同一种东西。
早先的写法是「泄漏**对数** ÷ 评测**图数**」——
一张评测图撞上 3 张训练图就贡献 3 个泄漏对，
于是 9 张评测图能算出 10 个泄漏对，`leak_rate = 111%`。
**比率超过 100% 是个一眼就该看出不对的信号**，别放过它，
更别因为「数字大 = 问题严重」就默认它是对的。

## 四、验收清单

- [ ] `make demo-check` 三张对账表全绿
- [ ] 抽 5 条样本 decode 出来肉眼检查，label mask 三种错误一个都没有
- [ ] `split_by_image` 生效：`check_split_by_image` 返回空列表（同图不跨集合）
- [ ] `check_leakage` 报 0 泄漏，**且 `n_eval_checked` 是个合理的正数**、`coverage = 1.0`
- [ ] `leak_rate ≤ 1.0`（超过 1 就说明分子分母口径错了）
- [ ] train/val/test 分布一致：8 个意图占比偏差 < 5%
- [ ] `sft_train.jsonl` + `sft_eval.jsonl` + `sft_test.jsonl` 落盘
- [ ] `python -m src.data.build_sft --selftest` 全绿

## 五、容易踩的坑

1. 多图样本：每张图占一个 `<|vision_start|>` 块，`<|image_pad|>` 数量各不相同 —— 数错就是 shape mismatch。

2. tokenizer 的 `add_special_tokens` 参数：chat template 已经含 `<|im_start|>`，别再让 tokenizer 加一遍。

3. padding 侧方向：训练 batch 用左 padding 还是右 padding，和推理要一致（Qwen 默认左）。

4. **占位符不能当分组键。** `__no_image__` 这类占位符全都长一样，
   拿来当分组键会把整份数据并成一组（然后全落进 train）。
   而且它是**非空字符串**，`p or f"__solo_{i}"` 这种兜底写法不会生效 ——
   必须显式判 `startswith("__")`。没有真实图片的样本各自成组。

5. **统计口径要和实际行为用同一个函数。**
   实测翻过车：切分用的是 `__solo_i`（20 组），
   而打印统计时用了 `p or ...`（`__no_image__` 非空 → 全部同一个 key），
   于是打出「按图片分组：1 组」。数字和实际行为对不上，比不打印更糟。
   修法是把 key 计算抽成 `_group_key()` 一个函数，两边共用。

6. **`check_split_by_image` 和 `check_leakage` 两个都要有。**
   前者只比路径、**不读图**，所以永远有效；
   后者读图算 pHash，图片读不到时会退化成「什么也没检查」。
   前者兜底，后者抓「内容近似但路径不同」的隐性泄漏。

## 六、打卡

三行照旧。今天卡的地方大概率是 mask 边界和切分分组 ——
把踩的坑写成 2 行代码注释留在 `build_labeled_sample` 和 `split_by_image` 里。
