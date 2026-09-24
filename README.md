# multimodal-lab · 8 周多模态 VLM 后训练与客服 Agent 实战

> 从「看懂 VLM 架构」到「训出自己的一版模型」，再到「让它在 Shopify 店铺里接客」。
> 周期：8 周 × 6 天 = 48 个学习单元，每天 3–4 小时（约 20h/周）

---

## 一、这个仓库是什么

一条**闭环链路**，不是一堆散装教程：

```
架构原理  →  数据工程  →  SFT 后训练  →  偏好对齐  →  评测体系
                                                      ↓
                              Shopify App  ←  多模态 Agent  ←  推理服务
```

- **垂直场景**：电商客服（语言 + 图像）。选它是因为：图文并茂天然是多模态刚需、bad case 一眼可判定、Shopify 上有真实商品图和真实业务流、效果好不好能量化。
- **基座模型**：`Qwen2.5-VL-3B-Instruct`（主力）与 `Qwen2.5-VL-7B-Instruct`（进阶）。选它的理由见 `docs/01`。
- **算力策略**：本地无卡，全程走 Colab / 云按量。代码按「可迁移到集群」的规范写，不写只在 Colab 活的胶水代码。

## 二、怎么用

```bash
# 1. 看总纲，知道 8 周怎么排
open PLAN.md

# 2. 今天该干什么 —— 打开进度总表
open progress/progress-tracker.md

# 3. 学理论
open docs/README.md

# 4. 跑代码
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # 建个空配置文件放着，现在什么都不用填。
                          # W2 数据合成才要 API key（阿里云百炼，注册送额度），
                          # W7 Shopify 才要 SHOPIFY_API_KEY。见 .env 里的注释。
```

**每天的固定节奏**（3–4h）：

| 时段 | 时长 | 干什么 |
|---|---|---|
| 读 | 60–75 min | 当天的 `docs/` 讲义 + 指定论文/源码段落，边读边在 `progress/daily-log.md` 写笔记 |
| 写 | 90–120 min | 当天的代码任务，跑通、看日志、改参数 |
| 记 | 20–30 min | 填当日打卡：产出物、卡住的地方、明天的第一步 |

**不要跳过的一件事**：每天结束时，在 `progress/daily-log.md` 追加一条。8 周后这份日志本身就是你能拿出手的东西。

## 三、目录说明

```
multimodal-lab/
├── README.md                  # 你在这里
├── PLAN.md                    # ⭐ 8 周总纲（索引）：每个 Day 指向 days/ 与 notebooks/
├── days/                      # ⭐ 每日讲义：一天一个 md（目标/读/写/跑/验收/坑）
│   ├── README.md              #    day 索引表
│   ├── day-05.md              #    手搭 Mini-VLM
│   ├── day-06.md              #    完整推理复盘（M1）
│   └── day-07.md ... day-12.md #    W2 数据工程六天
├── notebooks/                 # ⭐ 每日操作台：一天一个 ipynb
│   ├── day-05_minivlm_assembly.ipynb
│   ├── day-06_full_inference_review.ipynb
│   ├── day-07_image_preprocess.ipynb
│   ├── day-08_taxonomy_matrix.ipynb
│   ├── day-09_data_synthesis.ipynb
│   ├── day-10_cleaning_dedup.ipynb
│   ├── day-11_build_sft.ipynb
│   └── day-12_dataset_card.ipynb
├── docs/                      # 讲义：原理 / 工程 / 业务，13 篇
│   ├── README.md              #    阅读顺序与索引
│   ├── 00-orientation.md      #    技术版图：多模态到底在解决什么问题
│   ├── 01-architecture.md     #    VLM 三大件：编码器 / 连接器 / 主干
│   ├── 02-vision-encoder.md   #    ViT、CLIP、SigLIP、动态分辨率
│   ├── 03-connector.md        #    MLP / Q-Former / Perceiver 对比
│   ├── 04-qwen25vl.md         #    Qwen2.5-VL 架构精读（M-RoPE、窗口注意力）
│   ├── 05-data-engineering.md #    图文数据构造、清洗、去重、打包
│   ├── 06-sft-training.md     #    SFT 全流程、LoRA/QLoRA、显存估算、踩坑
│   ├── 07-alignment.md        #    DPO / ORPO / GRPO 与多模态可验证奖励
│   ├── 08-evaluation.md       #    通用榜单 + 领域评测集构建 + LLM-as-judge
│   ├── 09-inference.md        #    vLLM / SGLang、量化、服务化、成本
│   ├── 10-agent.md            #    多模态 Agent：ReAct、工具、多模态 RAG
│   ├── 11-shopify.md          #    Shopify App 全流程：OAuth/App Block/Billing/审核
│   ├── 12-papers.md           #    论文精读清单（含阅读顺序和精读重点）
│   ├── 13-hardware-and-cost.md#    云 GPU 选型、本地/云端分工、省钱策略
│   └── glossary.md            #    术语表，随时查
├── src/
│   ├── minivlm/               # 从零手搭一个能跑的最小 VLM（Week 1）
│   ├── data/                  # 客服图文数据：taxonomy / 合成 / 去重 / 打包
│   ├── train/                 # SFT (LoRA/QLoRA) + DPO + 训练脚本
│   ├── eval/                  # 评测框架 + 客服领域评测集
│   ├── serve/                 # vLLM 部署 + FastAPI 网关
│   ├── agent/                 # 多模态客服 Agent（工具 + RAG + 循环）
│   └── shopify/               # Shopify App 后端骨架
├── configs/                   # 训练 / 评测 / 服务配置
├── progress/                  # ⭐ 每日打卡 + 48 天进度总表 + 周复盘
├── scripts/                   # 环境准备、模型下载、notebook 生成等脚本
├── docker-compose.yml         # W7 本地一键环境（PostgreSQL+pgvector+应用）
└── Dockerfile                 # 应用侧镜像（不含 torch，~200MB）
```

## 四、8 周地图（一句话版）

| 周 | 主题 | 周末你会得到 |
|---|---|---|
| W1 | VLM 架构解剖 | 手搭的 Mini-VLM 跑出第一句图文回答 |
| W2 | 视觉表示与数据工程 | 1 万条自己的客服图文指令数据集 v0 |
| W3 | SFT 训练工程 | 一版 LoRA 微调过的 Qwen2.5-VL-3B |
| W4 | 评测体系 | 客服领域评测集 + 一份对比评测报告 |
| W5 | 偏好对齐与推理优化 | DPO 版本 + 一个能打的推理服务 |
| W6 | 多模态 Agent | 端到端客服 Agent（图 + 文 + 工具 + RAG） |
| W7 | Shopify SaaS 产品化 | 装在自己测试店里的 Shopify App |
| W8 | 打磨与交付 | 开源仓库 + 技术报告 + demo |

详细到每天的安排见 **`PLAN.md`**。

## 五、验收标准（8 周结束时你该有什么）

1. 一个**能讲清楚** VLM 训练全链路的自己（面试/分享能讲 40 分钟）。
2. 一个 **1k+ star 潜力的开源仓库**：Mini-VLM 实现 + 客服 SFT 全流程 + Agent。
3. 一版**自己后训练的模型**，在自建客服评测集上相对基座有明确提升（目标：意图准确率 +8pt 以上，图片问题回答准确率 +10pt 以上）。
4. 一个**部署在 Shopify 测试店里的 App**，能真实处理商品图 + 文本的客服咨询。
5. 一份**技术报告**（中英各一版），包含数据、方法、消融、失败案例。

## 六、两个诚实提醒

- **8 周做完全部内容偏紧**，尤其是 W5 的 DPO 和 W7 的 Shopify 审核走流程。真跑不完时，砍 **W5 的 GRPO** 和 **W7 的 Billing**，这两块最容易吃掉一周却收益最低。保底方案见 `PLAN.md` 末尾的「止损线」。
- **评测永远优先于训练**。没有评测集的 SFT 就是玄学。W4 如果推迟，后面全都是盲调。
