# Day 6 · 复盘：一次完整的图文推理（W1 收官）

> 预计 3–4h ｜ 📓 `notebooks/day-06_full_inference_review.ipynb` ｜ 💻 `src/minivlm/generate.py`
> 本日无新知识，全是把 W1 串起来。**目标只有一个：M1 达成。**

## 今日目标（一句话）

用官方 Qwen2.5-VL 和你手搭的 MiniVLM 各跑一次完整推理，
对照两者差异，白板画出完整数据流 —— 这是里程碑 M1 的验收动作。

## 一、读（40 min）

- 回看自己 W1 的 4 条打卡（`progress/daily-log.md`）
- `docs/glossary.md` 里 W1 相关词条自测：patch merge、visual token、M-RoPE、占位符替换

## 二、写（120 min）

### 1. 对比实验（notebook 里有现成格子）

| 实验 | 命令/函数 | 你要观察什么 |
|---|---|---|
| 官方模型推理 | `qwen_infer(image, question)` | 回答质量、视觉 token 数 |
| 图片顺序敏感性 | `--compare-order` | image 在前 vs text 在前，输出差异 |
| MiniVLM 粗推理 | `minivlm_infer(...)` | 0.5B 小模型答成什么样（不要求好） |
| token 数对照 | `compute_visual_tokens` vs 官方 `<\|image_pad\|>` | 必须一致 |

### 2. 周复盘（写进 `progress/weekly-review.md` 的 W1 段）

三个固定问题：
1. W1 最反直觉的一个知识点是什么？
2. 如果重做 Day 2-5，你会在哪一步换个做法？
3. 你现在能不能给一个完全不懂的人讲清「多模态模型怎么看见图」？

## 三、跑

```bash
# 云上
python -m src.minivlm.generate --compare-order
jupyter lab   # notebooks/day-06_full_inference_review.ipynb
```

## 四、验收清单（= M1 验收）

- [ ] 白板/纸上画出完整数据流：图 → patch → ViT → connector → 替换占位符 → LLM → 文字
  （拍照存 `assets/day6-whiteboard.png`）
- [ ] 能指出这条流水线里**最容易出 bug 的一步**，并说为什么
- [ ] 官方 vs MiniVLM 的输出差异，你能给出 2 个原因（模型大小 / 训练量）
- [ ] 周复盘已写，进度表 W1 六天全部 `[x]`

## 五、容易踩的坑

- notebook 跑不动时先查是不是 GPU 会话断了（Colab/云机 90min 超时）
- `--compare-order` 结果偶尔一样不是 bug —— 对强模型，顺序影响的概率分布很小

## 六、打卡 + 里程碑

打卡三行之外，额外写一句：**「M1 我认为 达成/未达成，因为____」**。
未达成就列出还差哪一条，Day 7 早上先补。
