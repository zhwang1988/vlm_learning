# Day 3 · 连接器：模态对齐的关键那一层

> 预计 3–4h ｜ 📓 `notebooks/day-03_connector_compare.ipynb` ｜ ☁️ 云 GPU · `src/minivlm/connector.py`、`docs/03-connector.md`
> 前置：Day 2 完成（手写 ViT 能跑通）

## 今日目标（一句话）

实现三种连接器（单层 MLP / 两层 MLP+GELU / Perceiver Resampler），对比它们的参数量和输出 token 数，并回答「为什么 LLaVA 用最笨的 MLP 反而效果好」。

## 一、读（60 min）

材料：
- `docs/03-connector.md` —— MLP / Q-Former / Perceiver / Cross-Attn 四种方案的对比表
- LLaVA 论文的 architecture 部分（只看图 1 + 连接器那一段）
- BLIP-2 论文里 Q-Former 的设计动机（**重点看它为什么要固定数量的 query**）

思考题（先自己想，答案在讲义或代码注释里）：
1. 连接器要解决的**根本矛盾**是什么？（提示：视觉 token 太多，语言模型序列装不下）
2. Perceiver Resampler 用固定数量的 query，好处是什么？代价是什么？
3. **关键一问**：LLaVA 用一层 MLP 就打赢了 Q-Former，这说明「对齐」这件事到底难在哪、不难在哪？
4. 如果连接器只做线性投影，为什么还需要非线性（GELU）？

## 二、写（100 min）

`src/minivlm/connector.py`（三种连接器，参数化可切换）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `MLPConnector` | 单层线性投影 `D_vision → D_llm`。最笨，但是 LLaVA 的基线 |
| `TwoLayerMLPConnector` | 加一层 GELU 隐层。**LLaVA-1.5 用的就是它**，比单层明显好 —— 这个结论反直觉，值得记 |
| `PerceiverResampler` | 固定 K 个可学习 query + 交叉注意力。把 N 个视觉 token 压成 K 个，**这是 Flamingo 的方案** |
| `count_params()` | 统计每种连接器的参数量 —— 你会发现它相对 LLM 小得可怜 |
| `compare_table()` | 输出的对比表：参数量 / 输出 token 数 / 是否有信息压缩 |

> 连接器的参数量通常只占整个模型的 **0.1%–2%**，但它决定了「视觉信息以什么形式进入语言模型」。
> 参数量小 ≠ 不重要。这是这一天的核心认知。

## 三、跑（在云 GPU 上）

```bash
# 自检 + 三种连接器对比表
python -m src.minivlm.connector
# 模拟真实规模：1024 个视觉 token 压到 64 个
python -m src.minivlm.connector --n 1024 --k 64
```

期望输出（节选）：
```
$ python -m src.minivlm.connector
                    参数量        输入token   输出token   信息压缩
MLP                 21.2 M         1024        1024       无
2-layer MLP+GELU    22.3 M         1024        1024       无
Perceiver (K=64)    26.9 M         1024          64       16×

d_model: vision 1152 → llm 2048

结论：三者参数量同一量级（20–27 M），差的是**压缩能力**。
      LLaVA 选 2-layer MLP：不压缩，但靠更强的 LLM 吃下全部 token。
      Flamingo 选 Perceiver：必须压缩，因为要吃几十张图 + 视频帧。
```

## 四、验收清单

- [ ] 能画出三种连接器的结构图，并说清 query 的数量各自是多少
- [ ] 能回答「LLaVA 为什么用 MLP 反而好」——**答案和 LLM 的强度有关**
- [ ] 能说出 Perceiver 的 K 调大调小分别会怎样（K 太小丢信息，K 太大失去压缩意义）
- [ ] 知道连接器参数量占比很小，但决定信息以什么形式进 LLM

## 五、容易踩的坑

1. **以为连接器是「翻译官」** —— 它不是把视觉特征翻译成语言，它是把视觉特征**投影到 LLM 能读的维度**。真正「对齐语义」是 SFT 阶段做的事。
2. **Perceiver 的 query 初始化** —— 全零初始化会让注意力均匀分布，导致训练初期几乎不更新。用 `randn * 0.02`。
3. **忘了 mask** —— Cross-Attention 里视觉侧不需要 mask，但如果加了 causal mask 会丢掉一半视觉信息，且不会有任何报错。
4. **参数量算错** —— bias 别漏。`nn.Linear(1152, 2048)` 是 `1152*2048 + 2048 = 2.36M`，不是 `1152*2048`。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
