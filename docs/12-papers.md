# 12 · 论文精读清单

> 全程配套。**不要通读所有论文**——每篇都标了「读到什么程度就够」。

## 阅读原则

1. **先读摘要 + 图 1 + 结论，再决定要不要读正文。** 图 1 通常就是整篇的核心 idea。
2. **分三次读**：第一次抓 idea（15 min），第二次抓方法细节（45 min），第三次抓实验和消融（30 min）。
3. **读的时候必须动手**：画架构图、写下你不懂的问题、去代码里找对应实现。
4. **不要读超过 3 天前没动手的论文**——会忘光。

推荐工具：`arxiv-sanity`、`paperswithcode`、`Zotero` 管文献。

---

## 第一梯队：必读（Week 1–3 完成）

### 1. An Image is Worth 16x16 Words (ViT, 2020)
- arXiv: 2010.11929
- **读到什么程度**：精读方法部分。理解 patch embedding、位置编码、为什么需要大数据。
- **对本项目的用处**：Day 2 手写 ViT 时对照。
- **重点**：图 1（架构图）、Section 3.1、Section 4.2（数据量对比实验）

### 2. Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)
- arXiv: 2103.00020
- **读到什么程度**：精读 Section 2.1–2.3（方法）。伪代码要能默写出来。
- **重点**：对称对比损失、prompt engineering 为什么有用、zero-shot 怎么做的
- **不必读**：超长附录、所有实验表

### 3. Sigmoid Loss for Language Image Pre-Training (SigLIP, 2023)
- arXiv: 2303.15343
- **读到什么程度**：读方法即可（3 页）。理解它和 CLIP 的唯一区别和带来的好处。
- **重点**：sigmoid loss 公式、小 batch 实验

### 4. Visual Instruction Tuning (LLaVA, 2023)
- arXiv: 2304.08485
- **读到什么程度**：**精读全文**。这是最简洁的 VLM 范式，你手搭 Mini-VLM 就是照它来的。
- **重点**：两阶段训练（alignment + instruction tuning）、数据构造方法（用 GPT-4 把图文对转成对话）、MLP 连接器
- **为什么必读**：整个「外挂式」路线的教科书

### 5. Improved Baselines with Visual Instruction Tuning (LLaVA-1.5, 2023)
- arXiv: 2310.03744
- **读到什么程度**：读 3 页。它做的是消融——MLP vs Linear、加不加学术数据、分辨率的影响。
- **重点**：这三个消融的结论直接回答了 `03-connector.md` 里的问题

### 6. Qwen2-VL (2024) / Qwen2.5-VL 技术报告
- arXiv: 2409.12191（Qwen2-VL）；Qwen2.5-VL 见官方博客与技术报告
- **读到什么程度**：**精读架构部分**。这是你的基座，必须懂。
- **重点**：Naive Dynamic Resolution、M-RoPE、视频处理、ViT 的窗口注意力设计
- **配合**：读的同时打开 HF 上的 `modeling_qwen2_5_vl.py` 对照

---

## 第二梯队：做数据和对齐时读（Week 2、Week 5）

### 7. BLIP-2 (2023)
- arXiv: 2301.12597
- **读到什么程度**：读 Q-Former 部分。
- **重点**：两阶段预训练、可学习 query 的设计动机
- **用处**：理解为什么后来大家都不用 Q-Former 了

### 8. Direct Preference Optimization (DPO, 2023)
- arXiv: 2305.18290
- **读到什么程度**：**精读 Section 4（推导）**。这个推导值得花 2 小时。
- **重点**：从 PPO 目标到 DPO loss 的完整推导、β 的作用
- **配合**：`07-alignment.md`

### 9. ORPO (2024) / SimPO (2024)
- arXiv: 2403.07691（ORPO）；2405.14734（SimPO）
- **读到什么程度**：读摘要 + 方法 1 页。知道「不需要 reference 模型」这个核心卖点即可。
- **用处**：显存不够时的备选方案

### 10. DeepSeek-R1 / GRPO
- arXiv: 2501.12948
- **读到什么程度**：读 GRPO 部分（不需要读完整篇 RL 推理的内容）。
- **重点**：组内相对优势怎么算、为什么能省掉 value model
- **配合**：`07-alignment.md` 的可验证奖励设计

### 11. MagicBrush / 多模态 DPO 相关工作
- 搜 "multimodal DPO" / "vision language model DPO alignment"
- **读到什么程度**：挑 2–3 篇的摘要和方法，看别人的偏好数据怎么构造的
- **重点**：contrastive image 的构造方式（治幻觉的关键）

---

## 第三梯队：做评测时读（Week 4）

### 12. POPE: Polling-based Object Probing Evaluation (2023)
- arXiv: 2305.10355
- **读到什么程度**：读方法，看它怎么构造负样本
- **用处**：Day 22 幻觉评测直接照搬

### 13. MMMU (2023) / MMBench (2023) / MMStar (2024)
- arXiv: 2311.16502（MMMU）；2307.06281（MMBench）；2403.20330（MMStar）
- **读到什么程度**：看摘要 + 评测维度设计
- **用处**：知道通用榜单在测什么、为什么不代表业务

### 14. Judging LLM-as-a-Judge with MT-Bench (2023)
- arXiv: 2306.05685
- **读到什么程度**：**精读**。位置偏见、长度偏见、自我偏好都在这里定义。
- **用处**：Day 21 的 judge 校准

### 15. HallusionBench (2023)
- arXiv: 2310.14566
- **读到什么程度**：读方法，看「视觉依赖」和「幻觉」怎么解耦
- **用处**：Day 22

---

## 第四梯队：做 Agent 时读（Week 6）

### 16. ReAct: Synergizing Reasoning and Acting (2022)
- arXiv: 2210.03629
- **读到什么程度**：精读。这是 Agent 的奠基工作。

### 17. Toolformer (2023)
- arXiv: 2302.04761
- **读到什么程度**：读方法。看怎么让模型自己学会何时调工具。

### 18. Reflexion / Tree of Thoughts / Plan-and-Solve
- **读到什么程度**：读摘要，知道有这些范式即可

### 19. Voyager / SWE-agent / 各种 Agent 工程实践
- **读到什么程度**：挑 1–2 篇，重点看它们的**失败分析**部分（比方法更有价值）

---

## 第五梯队：工程与优化（Week 5、Week 7）

### 20. LoRA (2021) / QLoRA (2023)
- arXiv: 2106.09685（LoRA）；2305.14314（QLoRA）
- **读到什么程度**：LoRA 精读（很重要）；QLoRA 读核心的三种量化技术（NF4、双重量化、分页优化器）

### 21. Efficient Memory Management for LLM Serving (PagedAttention/vLLM, 2023)
- arXiv: 2309.06180
- **读到什么程度**：读 Section 3–4。理解分页的思路。

### 22. AWQ (2023) / GPTQ (2022)
- arXiv: 2306.00978（AWQ）；2210.17323（GPTQ）
- **读到什么程度**：读方法。理解「为什么激活值比权重更值得保护」

### 23. ZeRO (2019) / PyTorch FSDP
- arXiv: 1910.02054
- **读到什么程度**：读 Section 3 三个阶段的划分。理解显存砍在哪。

---

## 按周的精读安排（建议）

| 周 | 必读 | 选读 |
|---|---|---|
| W1 | ViT、CLIP、LLaVA、LLaVA-1.5、Qwen2.5-VL | SigLIP、BLIP-2 |
| W2 | LLaVA 的数据构造部分（重读） | BLIP-2、ALIGN |
| W3 | LoRA、QLoRA | ZeRO |
| W4 | POPE、MMBench、MT-Bench judge | MMMU、MMStar |
| W5 | DPO、GRPO | ORPO、SimPO、AWQ |
| W6 | ReAct、Toolformer | Reflexion、SWE-agent |
| W7 | PagedAttention、AWQ | — |
| W8 | 无（全在动手） | — |

**总计约 23 篇**。8 周平均每周 3 篇，其中「精读」的只有 8 篇左右。**完全可行，前提是不要试图读完参考文献。**

---

## 找论文的几个技巧

- **顺藤摸瓜**：读 A 论文时，重点看它的相关工作里哪几篇被反复引用
- **看后续**：在 Google Scholar 上点「被引用」，按时间排序看它启发了什么
- **看批判**：搜「XXX limitations」「XXX revisited」，批判性文章往往比原文更能帮你理解边界
- **看代码优先**：如果一篇论文有高质量开源代码，直接读代码 + 摘要，比读论文快
- **不要追新**：arXiv 每天几百篇，追不完。读经典 + 读你真正要用的。

---

**上一篇**：[11-shopify.md](11-shopify.md) · **返回**：[README.md](README.md)
