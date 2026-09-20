# docs · 讲义索引

13 篇讲义，按学习顺序排列。每篇对应 `PLAN.md` 里的具体哪一天，标在标题后。

| # | 文件 | 对应天 | 一句话 |
|---|---|---|---|
| 00 | [00-orientation.md](00-orientation.md) | D1 | 多模态到底在解决什么问题，三条技术路线 |
| 01 | [01-architecture.md](01-architecture.md) | D1–D5 | VLM 三大件：视觉编码器 / 连接器 / 语言主干 |
| 02 | [02-vision-encoder.md](02-vision-encoder.md) | D2 | ViT 到 CLIP 到 SigLIP，以及动态分辨率 |
| 03 | [03-connector.md](03-connector.md) | D3 | MLP / Q-Former / Perceiver / Cross-Attn 怎么选 |
| 04 | [04-qwen25vl.md](04-qwen25vl.md) | D4 | Qwen2.5-VL 精读：M-RoPE、窗口注意力、原生分辨率 |
| 05 | [05-data-engineering.md](05-data-engineering.md) | D7–D12 | 图文数据：构造、合成、清洗、去重、打包 |
| 06 | [06-sft-training.md](06-sft-training.md) | D13–D18 | SFT 全流程、LoRA/QLoRA、显存估算、踩坑清单 |
| 07 | [07-alignment.md](07-alignment.md) | D25–D28 | DPO / ORPO / KTO / GRPO 与可验证奖励 |
| 08 | [08-evaluation.md](08-evaluation.md) | D19–D24 | 榜单全景、领域评测集、LLM-as-judge、幻觉 |
| 09 | [09-inference.md](09-inference.md) | D29–D30 | vLLM / SGLang、量化、服务化、单位经济 |
| 10 | [10-agent.md](10-agent.md) | D31–D36 | 多模态 Agent：工具、RAG、循环、评测 |
| 11 | [11-shopify.md](11-shopify.md) | D37–D45 | Shopify App 全流程 |
| 12 | [12-papers.md](12-papers.md) | 全程 | 论文精读清单，附「读到什么程度就够」 |
| — | [glossary.md](glossary.md) | 随时 | 术语表 |

## 怎么读

**不要按顺序从头啃到尾。** 每篇讲义分三部分：

- `## 概念` —— 必须理解的，读的时候要在脑子里能画出图
- `## 工程细节` —— 实现时才会用到的，第一遍扫读，用到了再回来精读
- `## 自检问题` —— 读完立刻回答，答不出来就说明没读懂，回去重读对应段

读讲义的原则：**一次只搞懂一个概念**。一篇文章信息密度高，别指望一次全吸收。第一遍只抓 `## 概念` 和 `## 自检问题`。

## 配套资源

- 论文清单和阅读重点：`12-papers.md`
- 遇到不懂的术语：`glossary.md`
- 代码里看不懂的实现：直接去 `src/` 对应目录，代码注释写得比讲义还细
