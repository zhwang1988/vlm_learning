# PLAN.md · 8 周执行总纲（索引）

> 每天 3–4 小时。每周 6 天（第 7 天留给复盘或还债）。
> 每一天的结构固定为：**读（理论）→ 写（代码）→ 记（打卡）**。
>
> **这个文件只是索引。** 每天真正要看的材料在这两个地方：
>
> | 文件 | 是什么 | 怎么用 |
> |---|---|---|
> | `days/day-XX.md` | 当天详细讲义：目标 / 读什么章节 / 写什么函数 / 跑什么命令 / 期望输出 / 坑 / 验收清单 | **每天开工先打开这个** |
> | `notebooks/day-XX_*.ipynb` | 当天的操作台：驱动 `src/` 代码跑通 + 可视化 + 自测题 | 云上或本地按当天说明跑 |
> | `src/**` | 当天要写/读的工程代码 | 打开对应模块 |
>
> 打卡写 `progress/daily-log.md`，勾选写 `progress/progress-tracker.md`。
>
> 每一天下面都有 📄/📓 直链 —— **48 天，一天一个 md、一天一个 notebook**，正文都在 `days/` 里。
> Day 1–4 也已拆出来（`days/day-01.md` ~ `day-04.md`），不再只存在于本文件。

---

## 先搞清楚：哪些事在本地做，哪些在云上做

这条决定了你 8 周到底花 ¥300 还是 ¥3000。

| | 本地（你的 Mac） | 云（租的 GPU 机器） |
|---|---|---|
| **干什么** | 读讲义、写/改代码、跑纯 Python 自检、复盘 | 真实推理、数据合成、训练、评测、vLLM 服务 |
| **需要什么** | Python + numpy（可选） | torch + CUDA + 显存 |
| **计时费吗** | 不计 | 按秒计，**开着机不干活就是烧钱** |

实测结论（在 Mac 上逐模块跑出来的）：

- **零依赖就能在本地跑**：`src/eval/` 的 `metrics` `judge` `hallucination` `agent_eval`；
  `src/shopify/` 全部（`auth` `webhooks` `billing` `client`）
- **本地装个 numpy 就能跑**：`src/data/` 全部、`src/agent/` 全部
- **必须在云上跑**：`src/minivlm/` 全部、`src/train/` 全部（要 torch）

也就是说 —— **W4（评测）和 W7（Shopify）这两周的代码可以在本地全部写完调通，不用开机。**

环境体检脚本两边都用同一个，它自己判断环境：

```bash
make check           # 本地：torch/CUDA 缺失只标黄，退出码 0
make check-cloud     # 云上：同样缺失就是硬伤，退出码 1
```

详细分工和理由见 `docs/13-hardware-and-cost.md` 的「五之二」。

---

## 全局里程碑

| 里程碑 | 时间 | 判定标准 |
|---|---|---|
| M1 读懂架构 | 第 1 周末 | 能白板画出 VLM 数据流，手搭 Mini-VLM 出句子 |
| M2 数据自主 | 第 2 周末 | 10k 条自建客服图文指令集，通过去重和质量过滤 |
| M3 跑通训练 | 第 3 周末 | Qwen2.5-VL-3B LoRA 权重训练完成，loss 曲线正常 |
| M4 可量化 | 第 4 周末 | 领域评测集 + 基座/微调对比报告 |
| M5 对齐+服务 | 第 5 周末 | DPO 版本 + 可被外部调用的推理 API |
| M6 Agent 闭环 | 第 6 周末 | Agent 在自建测试集上任务成功率 ≥ 65% |
| M7 产品化 | 第 7 周末 | Shopify 测试店能装、能用、能计费 |
| M8 交付 | 第 8 周末 | 开源仓库 + 技术报告 + demo 视频 |

---

# Week 1 · VLM 架构解剖

> 目标：把「多模态模型 = 视觉编码器 + 连接器 + 语言主干」这句话，从记住变成理解。
> 交付：`src/minivlm/` 能加载一张图 + 一句话，输出回答。

### Day 1 — 技术版图与问题定义

> 📄 `days/day-01.md`　·　📓 `notebooks/day-01_environment_and_first_inference.ipynb`

- **读**：`docs/00-orientation.md`；论文清单里的 CLIP、Flamingo、BLIP-2 摘要页（`docs/12-papers.md` 给了阅读重点，不要通读）
- **写**：① 本地跑 `make check`，torch/CUDA 标黄是正常的；② **上云后下载模型**：`python scripts/download_model.py --model 3b-instruct`（自动选 ModelScope，国内 2–5 分钟）；③ `make check-cloud` 全绿；④ 跑通 `notebooks/01_first_vlm_inference.ipynb`，让 Qwen2.5-VL 描述一张商品图
- **产出**：一张自己画的「多模态技术树」草图（可手绘拍照存 `assets/`）；模型本地路径截图（`download_model.py --status` 输出）
- **验收**：能不看资料说出三条技术路线（对比学习式 / 融合式 / 外挂式）的差异；能在云上让 7B 模型描述一张图

### Day 2 — 视觉编码器：ViT 到 SigLIP

> 📄 `days/day-02.md`　·　📓 `notebooks/day-02_vit_from_scratch.ipynb`

- **读**：`docs/02-vision-encoder.md` 前半；ViT 原文 key idea；CLIP 的 InfoNCE loss 推导
- **写**：`src/minivlm/vision.py` —— 手写 patch embedding + attention block，用 timm 的 SigLIP 权重，打印每层输出 shape
- **产出**：一段代码，输入 `(B,3,448,448)` 输出 `(B, N, D)` 并正确算出 N
- **验收**：能解释为什么 ViT 需要位置编码、CLIP 和 SigLIP 的 loss 差在哪

### Day 3 — 连接器：模态对齐的关键那一层

> 📄 `days/day-03.md`　·　📓 `notebooks/day-03_connector_compare.ipynb`

- **读**：`docs/03-connector.md`（MLP / Q-Former / Perceiver / Cross-Attn 四种方案的对比表）
- **写**：`src/minivlm/connector.py` —— 实现三种连接器（MLP、2-layer MLP+GELU、Perceiver Resampler），参数化可切换
- **产出**：对比实验脚本，输出三种连接器的参数量和输出 token 数
- **验收**：能回答「为什么 LLaVA 用最笨的 MLP 反而效果好」这个问题

### Day 4 — Qwen2.5-VL 架构精读

> 📄 `days/day-04.md`　·　📓 `notebooks/day-04_visual_token_budget.ipynb`

- **读**：`docs/04-qwen25vl.md`（重点：native dynamic resolution、M-RoPE、window attention、视频 3D conv）
- **写**：`src/minivlm/processor.py` —— 复现「按原图比例切 patch、算 visual token 数」的逻辑，与官方 processor 输出对齐校验
- **产出**：一个表格：不同尺寸图片 → visual token 数，与官方一致
- **验收**：能画图解释 M-RoPE 里 t/h/w 三个维度的旋转位置是怎么分配的

### Day 5 — 从零手搭 Mini-VLM

> 📄 `days/day-05.md`　·　📓 `notebooks/day-05_minivlm_assembly.ipynb`

- **读**：`src/minivlm/model.py` 的注释走一遍；回看 `docs/01-architecture.md`
- **写**：`src/minivlm/model.py` —— 拼装 SigLIP + Connector + Qwen2.5-0.5B，替换 input embedding 为「文本嵌入 + 视觉嵌入拼接」
- **产出**：`src/minivlm/generate.py` 能跑出「这张图里有什么」的粗回答（不要求质量，要求跑通）
- **验收**：模型能 forward 不报错，且 visual token 确实进入了 LLM 的 attention

### Day 6 — 复盘：一次完整的图文推理

> 📄 `days/day-06.md`　·　📓 `notebooks/day-06_full_inference_review.ipynb`

- **读**：总结本周讲义，回看自己的打卡笔记
- **写**：整理 `notebooks/day-06_full_inference_review.ipynb`，把 Mini-VLM 的 forward 过程可视化（token 序列、attention mask）
- **产出**：**周复盘**写入 `progress/weekly-review.md` W1 段
- **验收**：白板画出完整数据流并讲给自己听一遍，能指出哪一步最容易出 bug

---

# Week 2 · 视觉表示与数据工程

> 目标：从「会用别人数据」到「能造自己的数据」。
> 交付：10k 条电商客服图文指令数据集，含质量报告。

### Day 7 — 图像预处理全链路

> 📄 `days/day-07.md`　·　📓 `notebooks/day-07_image_preprocess.ipynb`

- **读**：`docs/05-data-engineering.md` 第 1–3 节
- **写**：`src/data/image_utils.py` —— resize 策略（保持比例 / smart resize）、归一化、pHash 计算、EXIF 纠正、CMYK/透明通道处理
- **产出**：对 200 张商品图跑一遍，输出尺寸/宽高比/token 分布直方图
- **验收**：能解释「为什么不能简单粗暴 resize 到 448×448」

### Day 8 — 客服场景数据 Taxonomy 设计

> 📄 `days/day-08.md`　·　📓 `notebooks/day-08_taxonomy_matrix.ipynb`

- **读**：`docs/05-data-engineering.md` 第 4 节；`src/data/taxonomy.py` 的初版定义
- **写**：`src/data/taxonomy.py` —— 定义 8 大意图 × 图像类型的二维矩阵（如：尺码咨询×上身图、质量问题×瑕疵特写、搭配建议×商品图、物流×快递单）
- **产出**：一张二维矩阵表，每个空格标注预估样本量
- **验收**：矩阵里每个格子都有至少一个真实对话例子（自己去电商平台逛出来的）

### Day 9 — 数据合成

> 📄 `days/day-09.md`　·　📓 `notebooks/day-09_data_synthesis.ipynb`

- **读**：`docs/05-data-engineering.md` 第 5 节（合成数据的三种范式与质量陷阱）
- **写**：`src/data/synth.py` —— 用强模型（GPT-4o / Qwen-VL-Max）按 taxonomy 批量生成图文多轮对话，含 system prompt 模板
- **产出**：跑出第一批 2k 条样本（用你 `.env` 里的 API）
- **验收**：人工抽检 30 条，合格率 ≥ 70%

### Day 10 — 数据清洗与去重

> 📄 `days/day-10.md`　·　📓 `notebooks/day-10_cleaning_dedup.ipynb`

- **读**：`docs/05-data-engineering.md` 第 6–7 节
- **写**：`src/data/dedup.py` —— pHash 图片去重 + 文本 embedding 去重 + 规则过滤（长度/拒绝语/格式）
- **产出**：清洗报告：输入 N 条 → 各环节淘汰多少 → 剩余多少
- **验收**：能说出「语义去重」和「字面去重」各自会漏掉什么

### Day 11 — 数据打包与对话模板

> 📄 `days/day-11.md`　·　📓 `notebooks/day-11_build_sft.ipynb`

- **读**：`docs/05-data-engineering.md` 第 8 节；Qwen2.5-VL 官方 chat template 源码
- **写**：`src/data/build_sft.py` —— 转成训练格式，正确处理 `<|vision_start|>...<|vision_end|>`、label mask（只对 assistant 部分算 loss）、多图样本
- **产出**：`data/processed/sft_train.jsonl`（8k）+ `sft_eval.jsonl`（2k）
- **验收**：抽一条样本 decode 出来肉眼检查，确认 label mask 正确

### Day 12 — 数据集 v0 交付

> 📄 `days/day-12.md`　·　📓 `notebooks/day-12_dataset_card.ipynb`

- **读**：复盘 W2 全部讲义
- **写**：`src/data/report.py` 生成数据集卡片（规模、分布、来源、已知缺陷）
- **产出**：`data/processed/DATASET_CARD.md` + **周复盘**
- **验收**：把数据集推上 HuggingFace（私有也行），有完整说明

---

# Week 3 · SFT 训练工程

> 目标：真的训出一个模型，而不是看懂别人怎么训。
> 交付：Qwen2.5-VL-3B LoRA 权重 + 训练日志分析。

### Day 13 — 训练环境与显存工程

> 📄 `days/day-13.md`　·　📓 `notebooks/day-13_training_env_vram.ipynb`

- **读**：`docs/06-sft-training.md` 第 1–3 节（含显存估算公式）
- **写**：`scripts/estimate_vram.py` —— 输入模型大小/精度/batch/序列长度/LoRA rank，估算显存
- **产出**：一张表：3B / 7B × 全参/LoRA/QLoRA × seq_len 2048/4096 的显存需求
- **验收**：能推导出「为什么 3B QLoRA 在 16G 卡上要开 gradient checkpointing」

### Day 14 — LoRA / QLoRA 原理与实现

> 📄 `days/day-14.md`　·　📓 `notebooks/day-14_lora_principles.ipynb`

- **读**：`docs/06-sft-training.md` 第 4 节；LoRA 原文 + QLoRA 原文关键段
- **写**：`src/train/lora_utils.py` —— 手写一个最小的 LoRA 层（低秩 A/B 矩阵），并打印出 target_modules 的选取逻辑
- **产出**：对比 `r=8/16/64` 的参数量与显存
- **验收**：能说清 `r`、`alpha`、`dropout`、`target_modules` 各自的作用和调参直觉

### Day 15 — 跑通第一次 LoRA SFT

> 📄 `days/day-15.md`　·　📓 `notebooks/day-15_first_lora_sft.ipynb`

- **读**：`configs/sft_lora_3b.yaml` 每一项的含义
- **写**：用 `src/train/sft_llamafactory.sh`（LLaMA-Factory 路线）或 `src/train/sft_peft.py`（原生 transformers 路线）在 Colab/云上启动训练
- **产出**：训练成功启动，看到 loss 开始下降
- **验收**：能解释日志里 `grad_norm`、`learning_rate`、`loss` 三条曲线的正常形态

### Day 16 — 训练日志与踩坑

> 📄 `days/day-16.md`　·　📓 `notebooks/day-16_training_log_debug.ipynb`

- **读**：`docs/06-sft-training.md` 第 5–7 节（常见坑清单：loss 不降、梯度爆炸、OOM、chat template 错配）
- **写**：`src/train/monitor.py` —— 解析 trainer log，画 loss/lr/grad_norm 三联图
- **产出**：自己这次训练的三联图 + 一段解读
- **验收**：至少主动制造并修复一个 bug（比如故意错配 template，看会怎样）

### Day 17 — 规模化训练：DeepSpeed / FSDP

> 📄 `days/day-17.md`　·　📓 `notebooks/day-17_deepspeed_fsdp.ipynb`

- **读**：`docs/06-sft-training.md` 第 8 节（ZeRO 三阶段、FSDP、多模态特有的不平衡问题）
- **写**：`configs/deepspeed_zero2.json` / `zero3.json`；把一个 7B 全参配置写好（先不跑，Colab 跑不动）
- **产出**：两份可用的 DeepSpeed 配置 + 注释说明
- **验收**：能说清 ZeRO-2 和 ZeRO-3 的取舍，以及为什么 VLM 训练里视觉塔通常冻结

### Day 18 — 第一次训练收官

> 📄 `days/day-18.md`　·　📓 `notebooks/day-18_first_training_wrap.ipynb`

- **读**：复盘 W3
- **写**：合并 LoRA 权重 → 导出 → 用 `src/eval/quick_eval.py` 做 20 条人工抽检
- **产出**：**base vs 你的 LoRA 版本** 的 20 条并排对比 + **周复盘**
- **验收**：能指出至少 3 个你的版本明显更好的 case 和 2 个更差的 case

---

# Week 4 · 评测体系

> 目标：把「感觉变好了」变成「提升了多少，在哪类上提升」。
> 交付：客服领域评测集 + 完整评测报告。

### Day 19 — 通用 VLM 评测全景

> 📄 `days/day-19.md`　·　📓 `notebooks/day-19_benchmark_overview.ipynb`

- **读**：`docs/08-evaluation.md` 第 1–2 节（MMMU / MMBench / MMVet / OCRBench / POPE / HallusionBench / DocVQA / MathVista）
- **写**：`src/eval/run_benchmark.py` —— 接一个开源榜（如 lmms-eval 或 VLMEvalKit）跑 OCRBench 的子集
- **产出**：你的模型在 1–2 个通用榜上的分数
- **验收**：能说出「通用榜单高分 ≠ 你的场景好用」的三个具体原因

### Day 20 — 构建客服领域评测集

> 📄 `days/day-20.md`　·　📓 `notebooks/day-20_build_domain_eval.ipynb`

- **读**：`docs/08-evaluation.md` 第 3–4 节（标注规范、一致性、难度分层）
- **写**：`src/eval/build_domain_eval.py` —— 从真实商品/对话构造 300 条评测样本，含 4 个难度层
- **产出**：`data/eval/cx_eval_v1.jsonl` + 标注手册
- **验收**：两人（或你两次隔一周）标注一致性 ≥ 85%

### Day 21 — 自动评测流水线

> 📄 `days/day-21.md`　·　📓 `notebooks/day-21_eval_pipeline.ipynb`

- **读**：`docs/08-evaluation.md` 第 5 节（LLM-as-judge 的偏见与校准）
- **写**：`src/eval/judge.py` —— 规则打分（意图命中/格式合规/拒答检测）+ LLM judge（事实性/帮助性/语气），做 position bias 校准
- **产出**：一条命令跑完全套评测并出 markdown 报告
- **验收**：judge 与人工打分相关性 ≥ 0.7

### Day 22 — 幻觉评测与缓解

> 📄 `days/day-22.md`　·　📓 `notebooks/day-22_hallucination_eval.ipynb`

- **读**：`docs/08-evaluation.md` 第 6 节；POPE 论文思路
- **写**：`src/eval/hallucination.py` —— 构造「图片里没有的东西」的诱导性问题，统计幻觉率；实现「不确定就说不确定」的 prompt 策略对比
- **产出**：幻觉率对比表（基座 vs 微调 vs 加缓解策略）
- **验收**：能解释幻觉的三种类型（物体存在性/属性/关系）各自成因

### Day 23 — 错误分析与 Bad Case 归类

> 📄 `days/day-23.md`　·　📓 `notebooks/day-23_error_analysis.ipynb`

- **读**：无（今天全在动手）
- **写**：`src/eval/error_analysis.py` —— 把失败样本自动聚类（embedding + 关键词），输出 top 10 失败模式
- **产出**：错误分类表 + 每个类别的典型样本
- **验收**：能明确指出下一轮该补哪三类数据

### Day 24 — 评测报告 v1

> 📄 `days/day-24.md`　·　📓 `notebooks/day-24_eval_report_v1.ipynb`

- **读**：复盘 W4
- **写**：`src/eval/report.py` 生成完整报告
- **产出**：`reports/eval_v1.md`：基座 vs LoRA 在领域集 + 通用集 + 幻觉集上的全对比 + **周复盘**
- **验收**：报告里每个结论都有数据支撑，没有「感觉」

---

# Week 5 · 偏好对齐与推理优化

> 目标：让模型不只「答对」，还要「答得像人、答得安全、答得快」。
> 交付：DPO 版本 + 可对外服务的推理 API。

### Day 25 — DPO 家族原理

> 📄 `days/day-25.md`　·　📓 `notebooks/day-25_dpo_family.ipynb`

- **读**：`docs/07-alignment.md` 第 1–3 节（RLHF vs DPO vs ORPO vs KTO vs GRPO 的关系图）
- **写**：`src/train/dpo_loss.py` —— 手写 DPO loss（含 reference model 项）并用玩具数据验证数值
- **产出**：一条 loss 曲线，能看出 chosen 概率上升
- **验收**：能推出 DPO 的闭式解，并说清 β 的作用

### Day 26 — 多模态偏好数据构造

> 📄 `days/day-26.md`　·　📓 `notebooks/day-26_preference_data.ipynb`

- **读**：`docs/07-alignment.md` 第 4 节
- **写**：`src/train/build_preference.py` —— 从 W4 的 bad case 反向构造 (chosen, rejected) 对；同图不同答、同答不同图两种模式
- **写出** 3k 对偏好数据
- **产出**：`data/processed/dpo_train.jsonl`
- **验收**：抽检 30 对，确认 rejected 确实比 chosen 差且差异是「可学习的」

### Day 27 — 跑 DPO

> 📄 `days/day-27.md`　·　📓 `notebooks/day-27_run_dpo.ipynb`

- **读**：`configs/dpo_3b.yaml`
- **写**：`src/train/dpo_train.py`（TRL 路线）或 LLaMA-Factory 的 DPO 配置，启动训练
- **产出**：DPO 权重 + `rewards/chosen` 与 `rewards/rejected` 差值曲线
- **验收**：在领域评测集上，DPO 版本 vs SFT 版本有可测差异（哪怕很小）

### Day 28 — 可验证奖励与 GRPO（进阶，可跳过）

> 📄 `days/day-28.md`　·　📓 `notebooks/day-28_verifiable_rewards.ipynb`

- **读**：`docs/07-alignment.md` 第 5–6 节
- **写**：`src/train/rewards.py` —— 实现三个可验证奖励：格式合规（JSON 可解析）、工具调用正确、引用图片区域正确
- **产出**：奖励函数单元测试通过
- **验收**：能说清「客服场景下为什么可验证奖励比人偏好更划算」

### Day 29 — 量化与推理加速

> 📄 `days/day-29.md`　·　📓 `notebooks/day-29_quantize_accelerate.ipynb`

- **读**：`docs/09-inference.md` 第 1–4 节（AWQ / GPTQ / FP8 / KV cache / continuous batching）
- **写**：`src/serve/quantize.py` —— 对 SFT 后模型做 AWQ 量化；`src/serve/vllm_server.sh` 起服务
- **产出**：量化前后：显存占用 / 首 token 延迟 / 吞吐（tok/s）对比表
- **验收**：能解释为什么量化对 VLM 的视觉塔尤其敏感

### Day 30 — 端到端推理服务

> 📄 `days/day-30.md`　·　📓 `notebooks/day-30_inference_service.ipynb`

- **读**：`docs/09-inference.md` 第 5–6 节
- **写**：`src/serve/api.py` —— FastAPI 网关：接收 base64/URL 图片 + 文本，转发 vLLM，带超时、重试、限流、结构化日志
- **产出**：`curl` 能打通的 `/v1/chat` 接口 + **周复盘**
- **验收**：并发 10 请求下不炸，且日志能追溯到单次耗时

---

# Week 6 · 多模态 Agent

> 目标：从「一个会看图说话的模型」到「一个能解决问题的客服」。
> 交付：端到端多模态客服 Agent，任务成功率 ≥ 65%。

### Day 31 — Agent 范式与工具协议

> 📄 `days/day-31.md`　·　📓 `notebooks/day-31_agent_tools.ipynb`

- **读**：`docs/10-agent.md` 第 1–3 节（ReAct / Plan-Execute / function calling schema）
- **写**：`src/agent/tools.py` —— 定义工具 schema：`lookup_order` / `check_stock` / `start_return` / `shipping_status` / `product_qa`，全部先 mock
- **产出**：工具能被模型正确调用（用 20 条测试 query 验证）
- **验收**：能说清「让模型输出 JSON」和「用 function calling」在鲁棒性上的差别

### Day 32 — 多模态 RAG

> 📄 `days/day-32.md`　·　📓 `notebooks/day-32_multimodal_rag.ipynb`

- **读**：`docs/10-agent.md` 第 4 节
- **写**：`src/agent/retriever.py` —— 商品图文双索引：CLIP 向量做图搜（用户上传图 → 找到对应商品）+ 文本向量做知识检索（退货政策、尺码表）；实现混合检索 + 重排
- **产出**：图搜 top-3 准确率 ≥ 70%（在 100 张商品图上测）
- **验收**：能解释为什么「以图搜图」比「先描述再搜文本」更适合商品场景

### Day 33 — 工具链完善与状态管理

> 📄 `days/day-33.md`　·　📓 `notebooks/day-33_tools_state_mgmt.ipynb`

- **读**：`docs/10-agent.md` 第 5 节（会话状态、幂等、失败回滚）
- **写**：补全真实业务逻辑（接 Shopify mock 数据）；加幂等键、事务边界、失败降级话术
- **产出**：所有工具在异常输入下不崩溃且有合理回复
- **验收**：能说出「下单类工具为什么必须幂等」

### Day 34 — Agent 骨架

> 📄 `days/day-34.md`　·　📓 `notebooks/day-34_agent_loop.ipynb`

- **读**：`docs/10-agent.md` 第 6 节
- **写**：`src/agent/agent.py` —— 主循环：意图理解 → 检索 → 规划 → 工具调用 → 生成回复；含最大步数、死循环检测、成本上限
- **产出**：Agent 能处理单轮图文咨询
- **验收**：连续 10 条 query 不出死循环、不超时

### Day 35 — 端到端联调

> 📄 `days/day-35.md`　·　📓 `notebooks/day-35_agent_demo.ipynb`

- **读**：无（动手日）
- **写**：`src/agent/demo.py` —— Gradio/Streamlit 界面，能上传图 + 打字，看到 Agent 的思考过程和工具调用
- **产出**：一个能录屏 demo 的界面
- **验收**：在 20 条真实场景 query 上跑通，人工判定成功率

### Day 36 — Agent 评测

> 📄 `days/day-36.md`　·　📓 `notebooks/day-36_agent_eval.ipynb`

- **读**：复盘 W6
- **写**：`src/eval/agent_eval.py` —— 指标：任务完成率、工具选择准确率、平均轮数、P95 延迟、单次成本
- **产出**：`reports/agent_eval_v1.md` + **周复盘**
- **验收**：任务成功率 ≥ 65%，且能定位失败集中在哪类任务

---

# Week 7 · Shopify SaaS 产品化

> 目标：让模型在真实的商家后台里跑起来。
> 交付：可安装在测试店的 Shopify App。

### Day 37 — Shopify 生态与 API

> 📄 `days/day-37.md`　·　📓 `notebooks/day-37_shopify_api.ipynb`

- **读**：`docs/11-shopify.md` 第 1–2 节（App 类型、Admin GraphQL、REST vs GraphQL、rate limit）
- **写**：`src/shopify/client.py` —— Admin GraphQL 客户端封装（分页、重试、限流退避）
- **产出**：能从测试店拉到商品列表并落库
- **验收**：能说清 App 的三种类型（public / custom / private）和各自审核要求

### Day 38 — OAuth 与应用骨架

> 📄 `days/day-38.md`　·　📓 `notebooks/day-38_shopify_oauth.ipynb`

- **读**：`docs/11-shopify.md` 第 3 节
- **写**：`src/shopify/app.py` —— OAuth 安装流程（`/auth` → 回调 → 换 access token → 存库）、session 校验、App Bridge 前端入口
- **产出**：能在测试店完成安装
- **验收**：能解释 HMAC 校验和 session token 的关系

### Day 39 — 店铺前台挂件

> 📄 `days/day-39.md`　·　📓 `notebooks/day-39_theme_widget.ipynb`

- **读**：`docs/11-shopify.md` 第 4 节（Theme App Extension / App Block）
- **写**：`src/shopify/extensions/chat-widget/` —— 前台聊天挂件：支持上传图片、调用你的 Agent API、样式跟随主题
- **产出**：店铺前台右下角出现能用的客服入口
- **验收**：在真实商品页上传一张商品图能收到回复

### Day 40 — Webhook 与索引同步

> 📄 `days/day-40.md`　·　📓 `notebooks/day-40_webhooks_index.ipynb`

- **读**：`docs/11-shopify.md` 第 5 节
- **写**：`src/shopify/webhooks.py` —— 订阅 `products/update`、`orders/create`、`refunds/create`；商品变更时增量更新向量索引
- **产出**：改一个商品标题，30 秒内 RAG 索引同步
- **验收**：能说清 webhook 的 HMAC 验证和幂等处理

### Day 41 — 计费与合规

> 📄 `days/day-41.md`　·　📓 `notebooks/day-41_billing_compliance.ipynb`

- **读**：`docs/11-shopify.md` 第 6–7 节（Billing API、GDPR webhook、PII）
- **写**：`src/shopify/billing.py` —— 订阅计划（免费试用 / 按会话量计费）、用量上报；补 3 个 GDPR webhook
- **产出**：能走完一次订阅流程
- **验收**：能说清「按用量计费」在 Shopify 上怎么实现和怎么对账

### Day 42 — 部署上线

> 📄 `days/day-42.md`　·　📓 `notebooks/day-42_deploy.ipynb`

- **读**：`docs/11-shopify.md` 第 8 节
- **写**：Docker 化 + 部署到云（Render/Railway/Fly.io 或你自己的服务器）；配置 HTTPS、环境变量、日志
- **产出**：**一个公网可访问的 App** + **周复盘**
- **验收**：从「陌生店铺安装」到「前台能对话」全流程走通

---

# Week 8 · 打磨与交付

> 目标：把 7 周的碎成果，变成一个能拿出去讲的整体。

### Day 43 — 全链路压测与成本核算

> 📄 `days/day-43.md`　·　📓 `notebooks/day-43_loadtest_cost.ipynb`

- **写**：`scripts/loadtest.py`；算清单位经济：每千次会话的 GPU 成本 + API 成本 + 基础设施
- **产出**：`reports/cost_model.md`：定价建议 + 毛利测算
- **验收**：能回答「这个 SaaS 打不打得平」

### Day 44 — 安全、合规与可靠性

> 📄 `days/day-44.md`　·　📓 `notebooks/day-44_security_reliability.ipynb`

- **写**：PII 脱敏（订单号/手机号/地址）、prompt 注入防护、输出过滤、降级策略、监控告警
- **产出**：安全清单逐项打勾 + 一份威胁模型
- **验收**：能说清客服场景下三类最危险的数据泄露路径

### Day 45 — Shopify 审核对齐

> 📄 `days/day-45.md`　·　📓 `notebooks/day-45_shopify_review.ipynb`

- **读**：`docs/11-shopify.md` 第 9 节（审核清单）
- **写**：补齐审核要求：隐私政策、卸载 webhook、性能要求（Lighthouse）、无障碍
- **产出**：`reports/shopify_checklist.md` 全绿
- **验收**：提交审核或至少完成自查

### Day 46 — 对比实验与消融

> 📄 `days/day-46.md`　·　📓 `notebooks/day-46_ablation.ipynb`

- **写**：跑齐 4 组对比：基座 / SFT / SFT+DPO / SFT+DPO+Agent，在同一评测集上
- **产出**：`reports/ablation.md`：每一阶段贡献了多少
- **验收**：能明确说出「哪一步最值」「哪一步可以省」

### Day 47 — 技术报告

> 📄 `days/day-47.md`　·　📓 `notebooks/day-47_tech_report.ipynb`

- **写**：中英各一版技术报告：背景、数据、方法、实验、消融、失败案例、成本、局限
- **产出**：`reports/tech_report_zh.md` / `_en.md`
- **验收**：一个不懂的人读完能复现你的流程

### Day 48 — 开源与交付

> 📄 `days/day-48.md`　·　📓 `notebooks/day-48_open_source_ship.ipynb`

- **写**：README 重写（含 demo GIF）、清理代码、写贡献指南、上传模型和数据、录 demo 视频
- **产出**：**完整开源仓库 + demo 视频 + 技术报告 + 8 周复盘**
- **验收**：陌生人 clone 后照 README 能在 30 分钟内跑通推理

---

## 止损线（跑不完时按这个顺序砍）

1. **砍 Day 28 的 GRPO**（省 3–4h）
2. **砍 Day 41 的 Billing**，用免费版上线（省 6h）
3. **砍 Day 17 的多卡配置**，只留单卡（省 4h）
4. **Day 39 的挂件改成独立网页 Demo**，不接主题（省 8h）
5. **Day 46 的消融改成 3 组**，去掉 Agent 组（省 4h）

**绝不能砍的**：Day 20–24（评测体系）、Day 34–36（Agent 闭环）、Day 42（真部署）。这三块是这个项目的「有」和「没有」的分界。

## 如果时间充裕，加餐方向

- 视频理解：把商品视频接进来（Qwen2.5-VL 原生支持视频，改 processor 即可）
- 多图推理：用户上传「上身图 + 尺码表 + 瑕疵特写」三张图，判断是否属于质量问题
- 语音：ASR 转文字后接进 Agent，做语音客服
- 蒸馏：用 7B 的结果蒸馏 3B，看能不能在保持效果的同时降本
- 自建 RLHF：找人标 300 条偏好数据，跑完整 GRPO
