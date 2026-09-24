# Day 5 · 从零手搭 Mini-VLM

> 预计 3–4h ｜ 📓 `notebooks/day-05_minivlm_assembly.ipynb` ｜ 💻 `src/minivlm/model.py` `generate.py`
> 前置：Day 2（vision.py）、Day 3（connector.py）已完成

## 今日目标（一句话）

把 SigLIP（视觉）+ MLP 连接器 + Qwen2.5-0.5B（语言）三块拼成一个能 forward 的 MiniVLM，
并亲眼看到 **视觉 embedding 是如何替换掉 `<image>` 占位符** 进入 LLM 的。

## 一、读（60 min）

材料：
- `docs/01-architecture.md` 第 4 节「inputs_embeds 替换」—— 今天唯一的新概念
- 复读你自己 Day 2/3 写的 `vision.py` / `connector.py` 头部注释

思考题（答案都在代码注释里，先自己想）：
1. 为什么不能把视觉 embedding 拼在文本 embedding 后面就行，而要「替换占位符」？
2. `inputs_embeds` 和 `input_ids` 传给模型，哪个优先？
3. 替换时 dtype 不一致会发生什么？（提示：bf16 vs fp32）

## 二、写（100 min）

`src/minivlm/model.py`（已给骨架，你要补/读懂的函数）：

| 函数/方法 | 你要做什么 |
|---|---|
| `MiniVLMConfig` | 读懂每个字段，改 connector kind 试试 |
| `MiniVLM.__init__` | 三块组件的实例化顺序 |
| `MiniVLM.merge_visual_embeds_expand` | **核心**：逐样本找 `<image>` 位置，切成 文本段+视觉段 交替 |
| `MiniVLM.forward` | vision → connector → 替换 → LLM，串起来 |

验收性质的小实验（在 notebook 里做）：把 `merge_visual_embeds_expand` 里的视觉段
替换成全零向量，看输出还能不能算 loss —— 能，说明形状对了但信息没了。

## 三、跑（在云 GPU 上）

```bash
python -m src.minivlm.model        # 自检：形状逐层打印
jupyter lab                        # 打开 notebooks/day-05_minivlm_assembly.ipynb
```

期望输出（节选）：
```
visual_embeds : [1, 577, 1152]  → connector → [1, 577, 896]
text_embeds   : [1, 26, 896]
merged        : [1, 602, 896]        # 26 - 1(占位符) + 577 = 602
logits        : [1, 602, 151936]
```

## 四、验收清单

- [ ] `python -m src.minivlm.model` 无报错
- [ ] notebook 里 merged 序列长度 = 文本长度 - 占位符数 + 视觉 token 数
- [ ] 能画出数据流草图（存 `assets/day5-flow.png`）
- [ ] 不查资料回答：LLM 看到的到底是什么？（答：一串连续向量，其中若干个来自图像）

## 五、容易踩的坑

1. **dtype 不匹配** —— SigLIP 输出 fp32，Qwen 是 bf16。`.to(emb.dtype)` 一行不能漏。
2. **attention_mask 长度没跟着扩** —— 替换后序列变长了，mask 还是旧的 → 静默错误。
3. **多个 `<image>` 占位符** —— 逐样本循环时要用 `nonzero` 找所有位置，不是只找第一个。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
