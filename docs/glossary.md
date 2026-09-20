# 术语表

> 随时查阅。按主题分组。

## 架构

**VLM (Vision-Language Model)** · 视觉语言模型。能同时处理图像和文本的模型。

**Vision Encoder / Visual Tower** · 视觉编码器 / 视觉塔。把图像变成特征向量序列的模块。常见的有 CLIP ViT、SigLIP、InternViT。

**Connector / Projector / Adapter** · 连接器。把视觉特征投影到语言模型维度的模块。常见的有 MLP、Q-Former、Perceiver Resampler。

**LLM Backbone** · 语言主干。VLM 里负责推理和生成的语言模型部分。

**Patch** · 图像被切成的固定大小方块。ViT 的标准 patch 是 14×14 或 16×16。

**Patch Embedding** · 把每个 patch 拉平后线性投影成向量的操作，等价于一个 kernel=stride=patch_size 的卷积。

**Visual Token** · 视觉 token。视觉编码器输出、送进 LLM 的每个向量。**数量决定成本和显存**。

**Naive Dynamic Resolution** · 原生动态分辨率。Qwen2-VL 系列的做法：不 resize 到固定尺寸，按原图比例处理，产生变长的 visual token 序列。

**AnyRes** · LLaVA-NeXT / InternVL 的做法：把图切成多个固定大小的 tile 分别处理。

**Pixel Shuffle** · 把相邻的 n×n 个 patch 在通道维拼接，从而把 token 数除以 n² 的操作。InternVL 用它压缩。

**M-RoPE (Multimodal RoPE)** · 多模态旋转位置编码。把位置拆成 t/h/w 三个维度，分别编码时间、行、列。

**Window Attention** · 窗口注意力。每个 token 只 attend 到局部窗口内的 token，降低计算量（O(N·w²) 而不是 O(N²)）。

**3D Convolution (temporal patch)** · 时间维卷积。把连续 N 帧在时间维合并成一个 patch，用于视频。

**Cross-Attention Layer** · 交叉注意力层。Flamingo 在 LLM 层间插入的、让文本 attend 到视觉特征的层。

**Q-Former** · BLIP-2 提出的连接器。用一组可学习的 query 通过 cross-attention 从视觉特征里抽取信息。

**Perceiver Resampler** · Flamingo 用的类似结构。可学习 latent 作为 query，视觉特征作为 key/value。

---

## 训练

**SFT (Supervised Fine-Tuning)** · 监督微调。用「输入-理想输出」对训练。

**Post-training / 后训练** · 在已经预训练好的模型上做 SFT / 对齐。本项目的工作范围。

**Stage 0 / 1 / 2 / 3** · VLM 训练的四个阶段：对齐、多任务预训练、指令微调、偏好对齐。

**LoRA (Low-Rank Adaptation)** · 冻结原权重，旁路一个低秩更新 `ΔW = BA`。

**QLoRA** · Quantized LoRA。把冻结的 base 权重量化成 4bit，再叠 LoRA。

**r / rank** · LoRA 的低秩维度，控制可训练容量。

**alpha** · LoRA 的缩放因子。`ΔW × alpha/r`。只调 alpha 等效于调学习率。

**target_modules** · LoRA 加在哪些层上。VLM 要额外考虑 `merger`（连接器）和视觉塔。

**Gradient Checkpointing** · 梯度检查点。前向时不保存中间激活，反向时重算，用时间换显存（省 30–50%）。

**Gradient Accumulation** · 梯度累积。多次前向累积梯度再一次更新，等效于增大 batch。

**ZeRO (Zero Redundancy Optimizer)** · DeepSpeed 的分片策略。三个 stage 分别切分优化器状态、梯度、权重。

**FSDP (Fully Sharded Data Parallel)** · PyTorch 原生的分片数据并行，思路类似 ZeRO-3。

**Catastrophic Forgetting / 灾难性遗忘** · 只训领域数据导致通用能力退化。用 replay 数据混合训练来防。

**Replay / 混合训练** · 在领域数据里混入通用数据，防止能力退化。

**Label Mask** · 把不需要计算 loss 的位置（如 system、user 部分）的 label 置为 -100。

**Chat Template** · 把对话结构转成模型输入格式的模板。Qwen 系用 ChatML 风格。

**Packing** · 把多条短样本拼成一个长序列，提高训练效率。

**Bucketing** · 按长度分组，让同一 batch 内样本长度接近，减少 padding。

---

## 对齐

**RLHF (Reinforcement Learning from Human Feedback)** · 基于人类反馈的强化学习。三阶段：SFT → Reward Model → PPO。

**DPO (Direct Preference Optimization)** · 直接偏好优化。跳过 reward model，用偏好对直接优化。

**β (beta)** · DPO 里控制偏离 reference 模型惩罚强度的系数。典型 0.1–0.5。

**Reference Model** · DPO 里冻结的参考模型，通常是 SFT 后的模型。

**Implicit Reward** · 隐含奖励。DPO 里 `β · log(π_θ/π_ref)` 这个量。

**ORPO / SimPO** · 不需要 reference 模型的偏好优化方法，省显存。

**KTO** · 只需要单条样本 + 好坏标签的偏好优化方法。

**GRPO (Group Relative Policy Optimization)** · 组相对策略优化。同 prompt 采多个答案，用组内相对优势做基线。

**Verifiable Reward** · 可验证奖励。用规则/程序判定的奖励，如格式是否合法、工具调用是否正确。

**Bradley-Terry Model** · 偏好概率模型。`P(y_w > y_l) = σ(r(y_w) - r(y_l))`。DPO 推导的起点。

---

## 数据

**Taxonomy** · 分类体系。本项目指「意图 × 图像类型」的二维矩阵。

**pHash (Perceptual Hash)** · 感知哈希。把图缩到 8×8 算 DCT 得到 64 bit 指纹，用于图片去重。

**Hamming Distance** · 汉明距离。两个 bit 串不同位的个数。pHash 距离 < 5 通常视为重复。

**Semantic Dedup** · 语义去重。用 embedding 余弦相似度判断文本重复。

**Image-level Leakage** · 图片级泄漏。评测集的图出现在训练集里，导致分数虚高。

**Dataset Card** · 数据集卡片。记录规模、分布、来源、已知缺陷的文档。

**Persona** · 人设。合成数据时模拟的用户类型（急躁/理性/新手），用来抗同质化。

**Difficulty Tier (L1–L4)** · 难度分层。L1 单图单事实 → L4 需澄清/应拒答。

---

## 评测

**MMMU / MMBench / MMVet** · 通用多模态榜单。分别测学科推理、细粒度能力、综合主观质量。

**OCRBench / DocVQA / ChartQA** · OCR 与文档图表理解榜单。

**POPE** · 物体存在性幻觉评测。问图里不存在的东西，看模型是否说有。

**HallusionBench** · 幻觉 + 视觉依赖的解耦评测。

**MMStar** · 抗数据泄漏的纯净评测集。

**LLM-as-a-Judge** · 用强模型给答案打分。有位置偏见、长度偏见、自我偏好、格式偏见。

**Position Bias** · 位置偏见。judge 偏爱排在前面的答案。用交换顺序各评一次来校准。

**Spearman Correlation** · 斯皮尔曼相关系数。judge 与人工打分的一致性度量，目标 ≥ 0.7。

**Existence Hallucination** · 存在性幻觉。说有图里没有的东西。

**Attribute Hallucination** · 属性幻觉。颜色/材质/数字判断错误。

**Relation Hallucination** · 关系幻觉。空间关系判断错误。

**Task Success Rate** · 任务成功率。Agent 主指标。

---

## 推理与服务

**vLLM** · 高吞吐推理引擎。核心是 PagedAttention。

**PagedAttention** · 把 KV cache 切成固定大小 block 按需分配的技术，类比操作系统虚拟内存。

**Continuous Batching** · 连续批处理。不同请求可以随时插入，不必等最长的结束。

**KV Cache** · 缓存已计算的 key/value，避免重复计算。显存占用的主要来源之一。

**Prefill / Decode** · 推理的两个阶段。Prefill 处理输入（可并行），Decode 逐 token 生成（串行）。

**TTFT (Time To First Token)** · 首 token 延迟。流式输出时用户感知的主要指标。

**AWQ / GPTQ** · 两种主流的 int4 权重量化方法。

**FP8** · 8 位浮点。H100/L40S 原生支持，质量损失极小。

**Prefix Caching** · 前缀缓存。相同的 prompt 前缀只算一次 prefill。system prompt 固定时收益大。

**SGLang** · 另一个高性能推理框架，特点是 RadixAttention（前缀树缓存）。

**Calculated Cost** · Shopify GraphQL 的限流计量单位。

---

## Agent

**ReAct** · Reason + Act。交替进行思考和行动的 Agent 范式。

**Plan-Execute** · 先规划再执行的范式。快但不灵活。

**Function Calling / Tool Use** · 模型输出结构化工具调用请求，由 runtime 执行。

**Structured ReAct** · 结合两者：显式推理步骤 + 结构化输出。

**Tool Schema** · 工具的描述（名称、描述、参数 JSON Schema）。

**Idempotency Key** · 幂等键。保证同一操作重复调用只生效一次。

**Multi-modal RAG** · 多模态检索增强生成。图文双索引检索。

**Contrastive Image** · 对比图像。DPO 里同答案配不同图，用于治幻觉。

**Max Steps** · Agent 最大循环步数，防止死循环。

**Slot Filling** · 槽位填充。把用户话语里的参数（订单号、尺码）抽取出来。

---

## 平台与业务

**Shopify App** · 三种类型：Public（需审核）、Custom（单店）、Private（弃用）。

**OAuth** · 授权流程。商家授权 → 拿 code → 换 access token。

**Access Token** · 长期凭据，服务端调 Admin API 用。

**Session Token** · 短期 JWT，前端调自己后端用。

**App Bridge** · Shopify 前端 SDK，处理 iframe 内导航、弹窗、token。

**Theme App Extension** · 往店铺前台主题注入 UI 的官方方式。

**App Block** · Theme App Extension 里的一个可拖拽组件。

**Polaris** · Shopify 官方 UI 组件库。

**Admin GraphQL API** · 管理店铺数据的接口（推荐）。

**Webhook** · 事件推送。商品更新、订单创建、卸载等。

**HMAC** · 哈希消息认证码。用于验证 webhook 和 OAuth 回调确实来自 Shopify。

**Billing API** · 计费接口。订阅制、一次性、按用量。

**Usage-based Billing / Capped Amount** · 按用量计费及其上限。

**GDPR Webhook** · 三个强制 webhook：`customers/data_request`、`customers/redact`、`shop/redact`。

**Development Store** · 免费的测试店铺，可装未审核 App。

---

**返回**：[README.md](README.md)
