# 进度总表 · 48 天

> 每天 3–4 小时，每周 6 天。第 7 天留给复盘或还债。
> 用法：做完一天就把 `[ ]` 改成 `[x]`，并在 `daily-log.md` 里补三行打卡。
>
> **状态图例**：`[ ]` 未开始 · `[~]` 在做 · `[x]` 完成 · `[-]` 主动跳过（记下原因）

---

## 里程碑看板

| # | 里程碑 | 时间 | 判定标准 | 状态 |
|---|---|---|---|---|
| M1 | 读懂架构 | 第 1 周末 | 能白板画出 VLM 数据流，手搭 Mini-VLM 出句子 | `[ ]` |
| M2 | 数据自主 | 第 2 周末 | 10k 条自建客服图文指令集，通过去重和质量过滤 | `[ ]` |
| M3 | 跑通训练 | 第 3 周末 | Qwen2.5-VL-3B LoRA 训练完成，loss 曲线正常 | `[ ]` |
| M4 | 可量化 | 第 4 周末 | 领域评测集 + 基座/微调对比报告 | `[ ]` |
| M5 | 对齐 + 服务 | 第 5 周末 | DPO 版本 + 可被外部调用的推理 API | `[ ]` |
| M6 | Agent 闭环 | 第 6 周末 | Agent 在自建测试集上任务成功率 ≥ 65% | `[ ]` |
| M7 | 产品化 | 第 7 周末 | Shopify 测试店能装、能用、能计费 | `[ ]` |
| M8 | 交付 | 第 8 周末 | 开源仓库 + 技术报告 + demo 视频 | `[ ]` |

**当前进度**：（每天更新）

```
第 __ 周 / 共 8 周      完成 __ / 48 天     最近里程碑：M__
```

---

## Week 0 · 开工前（半天）

- [ ] 装 `requirements-core.txt`，跑通 `make check`
- [ ] 按 `docs/13-hardware-and-cost.md` 第三节租好卡（AutoDL 4090）
- [ ] 在实例上跑 `bash scripts/cloud_bootstrap.sh`，看到冒烟测试通过
- [ ] `cp .env.example .env`，填上 `SYNTH_API_KEY`
- [ ] 建 git 仓库，`git init && git add . && git commit -m "init"`

> 这半天做完，后面 8 周每天都能「打开就能干活」，不会在环境上反复卡住。

---

## Week 1 · VLM 架构解剖

> 目标：把「多模态模型 = 视觉编码器 + 连接器 + 语言主干」从记住变成理解。
> 交付：`src/minivlm/` 能加载一张图 + 一句话，输出回答。

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 1 | 技术版图与问题定义 | [`day-01`](../days/day-01.md) · `docs/00-orientation.md` + CLIP/Flamingo/BLIP-2 摘要 | 写下客服场景的 5 个图文问题 | 能说出三条技术路线的差别 | `[x]` |
| 2 | 视觉编码器：ViT → SigLIP | [`day-02`](../days/day-02.md) · `docs/02-vision-encoder.md` | `src/minivlm/vision.py` | 手写 ViT 跑通，打印各层形状 | `[x]` |
| 3 | 连接器：模态对齐那一层 | [`day-03`](../days/day-03.md) · `docs/03-connector.md` | `src/minivlm/connector.py` | 对比 4 种连接器的参数量与输出 token | `[x]` |
| 4 | Qwen2.5-VL 架构精读 | [`day-04`](../days/day-04.md) · `docs/04-qwen25vl.md` | `src/minivlm/processor.py` | **`make day4` 通过**（算出 token 数 == 官方 `<\|image_pad\|>` 数） | `[x]` |
| 5 | 从零手搭 Mini-VLM | [`day-05`](../days/day-05.md) | `src/minivlm/model.py` | MiniVLM 能前向，视觉 embedding 正确替换占位符 | `[~]` |
| 6 | 复盘：一次完整的图文推理 | [`day-06`](../days/day-06.md) | `src/minivlm/generate.py` + notebook | **M1 达成**：白板画出完整数据流 | `[ ]` |

**W1 交付物**：`src/minivlm/` 六个文件全部可运行 + notebook 01

**本周最容易卡住的地方**：
- 视觉 token 数对不上 → 十有八九是 `smart_resize` 里的对齐取整没做（要能被 `patch*merge` 整除）
- 这是本周唯一的硬骨头。卡超过 1 小时就去看 `processor.py` 里的 `check_against_official`

---

## Week 2 · 数据工程

> 目标：不依赖任何公开数据集，自己造出 10k 条能用的客服图文指令数据。
> 交付：`data/processed/` 下的训练/验证集 + DATASET_CARD.md

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 7 | 图像预处理全链路 | [`day-07`](../days/day-07.md) | `src/data/image_utils.py` | EXIF 旋转、透明底、pHash 三个测试通过 | `[ ]` |
| 8 | 客服数据 Taxonomy 设计 | [`day-08`](../days/day-08.md) | `src/data/taxonomy.py` | `make data-plan` 打出 8×6 配比矩阵 | `[ ]` |
| 9 | 数据合成 | [`day-09`](../days/day-09.md) | `src/data/synth.py` | 试跑 20 条，人工抽查语气不雷同 | `[ ]` |
| 10 | 数据清洗与去重 | [`day-10`](../days/day-10.md) | `src/data/dedup.py` | CleaningReport 里每一条都能解释 | `[ ]` |
| 11 | 数据打包与对话模板 | [`day-11`](../days/day-11.md) | `src/data/build_sft.py` | **label mask 只覆盖 assistant 段**，跑 leakage 检查 | `[ ]` |
| 12 | 数据集 v0 交付 | [`day-12`](../days/day-12.md) | `src/data/report.py` | **M2 达成**：DATASET_CARD.md 生成，含「已知缺陷」 | `[ ]` |

**W2 交付物**：`data/processed/sft_train.jsonl` + `sft_eval.jsonl` + `DATASET_CARD.md`

**本周坑点预警（这两个坑会让你 W3 白跑）**：
1. **图像级泄漏** —— 同一张图不能同时出现在 train 和 eval。按图去重，不是按样本去重。
   `build_sft.py` 的 `check_leakage` 会拦你，但别绕过它。
2. **label mask 漏了 `<|im_end|>`** —— 模型学不会"停"。
   症状是推理时话说不完、或者答完继续编。`inspect_sample` 会打印被训的 token。

---

## Week 3 · 第一次 SFT

> 目标：让 Qwen2.5-VL-3B 学会用客服的口吻回答电商图文问题。
> 交付：`outputs/qwen25vl3b-cx-lora-v0` + 训练日志与曲线

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 13 | 训练环境与显存工程 | [`day-13`](../days/day-13.md) · `docs/06` 第一节 + `docs/13` | `scripts/estimate_vram.py` | 能算出自己卡的可用 batch，说出全参 16 bytes/param 的来历 | `[ ]` |
| 14 | LoRA / QLoRA 原理与实现 | [`day-14`](../days/day-14.md) · `docs/06` 第二三节 | `src/train/lora_utils.py` | 手写 LoRALinear，验证 **B 零初始化时输出与基座完全一致** | `[ ]` |
| 15 | 跑通第一次 LoRA SFT | [`day-15`](../days/day-15.md) · `docs/06` 第四节 | `src/train/sft_peft.py` | `make train-dry` → `make train-try` 20 步 loss 在降 | `[ ]` |
| 16 | 训练日志与踩坑 | [`day-16`](../days/day-16.md) · `docs/06` 三条曲线 | `src/train/monitor.py` | 能区分「正常收敛 / 过拟合 / 学不动」三种曲线 | `[ ]` |
| 17 | 规模化训练 | [`day-17`](../days/day-17.md) · `configs/README.md` + DeepSpeed 文档 | 读懂 ZeRO 三个 stage | 能说出本项目为什么**多半用不上** DeepSpeed | `[ ]` |
| 18 | 第一次训练收官 | [`day-18`](../days/day-18.md) · — | 全量训练 + `make sync-down` | **M3 达成**：checkpoint 已拉回本地，曲线存进 `reports/` | `[ ]` |

**W3 交付物**：LoRA adapter + `reports/training_curves.png` + 训练日志

**省钱提醒**：训练命令记得接 `&& /usr/bin/shutdown`。忘关一次，一周的收益就没了。

---

## Week 4 · 让它可被量化

> 目标：从"感觉好点了"变成"在 N 个指标上好了 X 个点"。
> 交付：领域评测集 + 基座/微调对比报告

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 19 | 通用 VLM 评测全景 | [`day-19`](../days/day-19.md) · `docs/08` 前三节 | 读 MMMU / MMBench / POPE 的指标定义 | 能说出「MMMU 高不代表客服场景好」为什么 | `[ ]` |
| 20 | 构建客服领域评测集 | [`day-20`](../days/day-20.md) · `docs/08` 领域评测一节 | `src/eval/build_domain_eval.py` | 生成 EVAL_CARD.md，L1–L4 配额合理 | `[ ]` |
| 21 | 自动评测流水线 | [`day-21`](../days/day-21.md) · `docs/08` 规则指标一节 | `src/eval/metrics.py` + `run_eval.py` | 基座 vs 微调跑出对比表 | `[ ]` |
| 22 | 幻觉评测与缓解 | [`day-22`](../days/day-22.md) · `docs/08` 幻觉一节 | `src/eval/hallucination.py` | **同时报 hallucination_rate 和 miss_rate**（只报一个会自欺） | `[ ]` |
| 23 | 错误分析与 Bad Case 归类 | [`day-23`](../days/day-23.md) · `docs/08` LLM 裁判偏差 | `src/eval/error_analysis.py` | 产出 `error_analysis.md` + `bad_cases.jsonl` | `[ ]` |
| 24 | 评测报告 v1 | [`day-24`](../days/day-24.md) · — | `reports/eval_v0.md` | **M4 达成**：报告里有失败模式和「下一步建议」 | `[ ]` |

**W4 交付物**：`data/eval/domain_eval.jsonl` + `reports/` 下三份报告

**本周最有价值的产出其实是 `bad_cases.jsonl`** —— W5 的 DPO 直接吃它。

---

## Week 5 · 对齐 + 推理服务

> 目标：把模型偏好掰正，并且能被人调用。
> 交付：DPO 版本 + 推理 API

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 25 | DPO 家族原理 | [`day-25`](../days/day-25.md) · `docs/07` 前两节 | `src/train/dpo_loss.py` | **`make dpo-verify` 通过**（含 ln2 校验） | `[ ]` |
| 26 | 多模态偏好数据构造 | [`day-26`](../days/day-26.md) · `docs/07` 三种 DPO 数据类型 | `--from-badcases` / `--contrastive` | 能手造「同图不同答」和「同答不同图」两类对 | `[ ]` |
| 27 | 跑 DPO | [`day-27`](../days/day-27.md) · `docs/07` β 那一节 | `configs/dpo_3b.yaml` | rewards/accuracies 在涨，且 eval 没有变差 | `[ ]` |
| 28 | 可验证奖励与 GRPO（进阶，可跳过） | [`day-28`](../days/day-28.md) · `docs/07` GRPO 一节 | `src/train/rewards.py` | 6 个奖励函数自测通过；**跳过也不算欠债** | `[ ]` |
| 29 | 量化与推理加速 | [`day-29`](../days/day-29.md) · `docs/09` 量化分档 | `src/serve/quantize.py` | 能说出**为什么视觉塔不能一起量化** | `[ ]` |
| 30 | 端到端推理服务 | [`day-30`](../days/day-30.md) · `docs/09` 部署一节 | `src/serve/api.py` + `vllm_server.sh` | **M5 达成**：curl 能打通 `/v1/chat`，含图片 | `[ ]` |

**W5 交付物**：DPO adapter + 可用的推理 API + 量化对比数据

**Danger zone**：在训练环境里 `pip install vllm` 会换掉 torch。
用 `bash scripts/cloud_bootstrap.sh --with-vllm` 建独立环境（`docs/13` 坑 2）。

---

## Week 6 · Agent

> 目标：从"会回答"升级到"会办事"。
> 交付：能查订单、能看图、能升级人工的客服 Agent + 测试集成功率 ≥ 65%

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 31 | Agent 范式与工具协议 | [`day-31`](../days/day-31.md) · `docs/10` 前两节 | `src/agent/tools.py` | 工具幂等性通过，**幻觉工具名被拦住** | `[ ]` |
| 32 | 多模态 RAG | [`day-32`](../days/day-32.md) · `docs/10` 检索一节 | `src/agent/retriever.py` | CLIP 粗排 + VLM 精排跑通 | `[ ]` |
| 33 | 工具链与状态管理 | [`day-33`](../days/day-33.md) · `docs/10` guardrails 一节 | 完善 tools + 状态持久化 | 三条护栏都生效（max_steps / 重复检测 / 成本上限） | `[ ]` |
| 34 | Agent 骨架 | [`day-34`](../days/day-34.md) · `docs/10` 主循环一节 | `src/agent/agent.py` | **视觉证据提取**生效（不会"图丢了"） | `[ ]` |
| 35 | 端到端联调 | [`day-35`](../days/day-35.md) · — | `src/agent/demo.py` | Gradio 界面能上传图 + 多轮对话 | `[ ]` |
| 36 | Agent 评测 | [`day-36`](../days/day-36.md) · `docs/10` 失败模式 | `src/eval/agent_eval.py` | **M6 达成**：任务成功率 ≥ 65%，失败案例已归类 | `[ ]` |

**W6 交付物**：Agent + `reports/agent_eval.md`

**本周最容易忽略的一件事**：**把视觉证据写进状态**。
多轮对话里如果不显式保存"用户传过什么图、图里有什么"，第 3 轮模型就"忘"了图，
然后开始瞎编。`extract_visual_evidence` 就是干这个的。

---

## Week 7 · 产品化

> 目标：让一个真实的店主能装上、能用、能付费。
> 交付：Shopify 测试店能装能用能计费

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 37 | Shopify 生态与 API | [`day-37`](../days/day-37.md) · `docs/11` 前三节 | `src/shopify/client.py` | 能说出 Admin GraphQL 和 REST 的取舍、cost-based 限流 | `[ ]` |
| 38 | OAuth 与应用骨架 | [`day-38`](../days/day-38.md) · `docs/11` OAuth 一节 | `src/shopify/auth.py` + `app.py` | **HMAC 校验通过 + SSRF 被拦**（两个测试都要过） | `[ ]` |
| 39 | 店铺前台挂件 | [`day-39`](../days/day-39.md) · `docs/11` 主题扩展一节 | `extensions/chat-widget/` | 测试店前台出现挂件，能上传图 | `[ ]` |
| 40 | Webhook 与索引同步 | [`day-40`](../days/day-40.md) · `docs/11` webhook 一节 | `src/shopify/webhooks.py` + `indexer.py` | **改商品后 ≤30 秒生效**；重复投递只处理一次 | `[ ]` |
| 41 | 计费与合规 | [`day-41`](../days/day-41.md) · `docs/11` 计费与 GDPR | `src/shopify/billing.py` + `models.py` | 订阅能创建；usage 上报幂等且不超过上限 | `[ ]` |
| 42 | 部署上线 | [`day-42`](../days/day-42.md) · `docs/11` 上线清单 | 部署 + `scripts/demo_up.sh` | 公网可访问，`/health` 正常 | `[ ]` |

**W7 交付物**：可安装的 Shopify App + 部署好的公网地址

**三个会卡住你的点**（`src/shopify/__init__.py` 里也记了）：
1. OAuth HMAC 要**先剔除 hmac 参数、再按 key 排序**重新拼接
2. Webhook HMAC 是 **base64**，不是 hex（和 OAuth 那个不一样，很容易混）
3. Webhook 必须**先返回 200 再处理业务**，超 5 秒 Shopify 会重试，然后你就收到重复事件

---

## Week 8 · 交付

> 目标：把 8 周的东西讲清楚，让别人能复现。
> 交付：开源仓库 + 技术报告 + demo 视频

| Day | 主题 | 读 | 笔/代码 | 验收 | 状态 |
|---|---|---|---|---|---|
| 43 | 全链路压测与成本核算 | [`day-43`](../days/day-43.md) · `docs/09` 单位经济一节 | `scripts/loadtest.py` | 出 QPS/延迟/单位成本三张表 | `[ ]` |
| 44 | 安全、合规与可靠性 | [`day-44`](../days/day-44.md) · `docs/11` GDPR + 敏感场景 | Agent 敏感工具禁用 + PII 脱敏 | 敏感场景不调 `start_return`；日志里没有 PII | `[ ]` |
| 45 | Shopify 审核对齐 | [`day-45`](../days/day-45.md) · `docs/11` 上线清单 | 逐条过 checklist | 知道哪些会被拒、怎么改 | `[ ]` |
| 46 | 对比实验与消融 | [`day-46`](../days/day-46.md) · — | 跑 4 组对照 | 能回答「LoRA vs QLoRA 差几个点」「加 DPO 值不值」 | `[ ]` |
| 47 | 技术报告 | [`day-47`](../days/day-47.md) · — | `reports/FINAL_REPORT.md` | **M8 部分达成**：含方法、数据、结果、失败、复现步骤 | `[ ]` |
| 48 | 开源与交付 | [`day-48`](../days/day-48.md) · — | 仓库整理 + demo 视频 | **M8 达成**：别人能按 README 复现 | `[ ]` |

**W8 交付物**：`README` 可复现 + `FINAL_REPORT.md` + demo 视频

**消融实验是这个项目最容易被跳过、但最能加分的一环。**
「我训了一个模型」和「我知道 LoRA 比 QLoRA 在幻觉指标上差 1.8 个点」是两个层次的结论。

---

## 止损线（跑不完时按这个顺序砍）

时间不够是正常的。按优先级从低到高砍，**保住左边**：

```
保住 ←──────────────────────────────────────────────→ 先砍

M1 架构  M2 数据  M3 训练  M4 评测  M5 对齐+服务  M6 Agent  M7 Shopify  M8 交付

Day 28 GRPO          ← 第一个砍（进阶，标了可跳过）
Day 46 消融实验       ← 第二个砍（加分项，非必要）
Day 44 安全加固       ← 第三个砍（但 PII 脱敏要留）
Day 42 真实部署       ← 第四个砍（本地跑通也算完成）
Day 41 计费           ← 第五个砍（能演示即可）
Day 17 DeepSpeed     ← 第六个砍（单卡用不上）
```

**绝对不能砍的四件事**：
1. **W2 的数据质量** —— 垃圾数据训出来的模型，后面所有周都白做
2. **W2 的图像级去重** —— 有泄漏的评测集，所有结论都不可信
3. **W4 的评测** —— 没有评测，你根本不知道自己有没有变好
4. **`make sync-down`** —— 云上丢了就真没了

---

## 打卡统计

| 周 | 计划天数 | 实际完成 | 周复盘已写 | 备注 |
|---|---|---|---|---|
| W1 | 6 | | `[ ]` | |
| W2 | 6 | | `[ ]` | |
| W3 | 6 | | `[ ]` | |
| W4 | 6 | | `[ ]` | |
| W5 | 6 | | `[ ]` | |
| W6 | 6 | | `[ ]` | |
| W7 | 6 | | `[ ]` | |
| W8 | 6 | | `[ ]` | |
| **合计** | **48** | | | |

---

相关：`PLAN.md`（每天详细任务）· `progress/daily-log.md`（每日打卡）· `progress/weekly-review.md`（周复盘）
