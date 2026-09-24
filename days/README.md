# days/ · 每日材料索引

**一天一个 md，一天一个 notebook。** `PLAN.md` 只做总纲索引，正文都在这里。

每个 `day-XX.md` 固定六节：**今日目标 → 读 → 写 → 跑 → 验收 → 坑 → 打卡**。

> ⚠️ 这里是**生成物**。要改内容改 `scripts/daygen/w*.py`，然后：
> ```bash
> python scripts/gen_days.py           # 重新生成全部
> python scripts/gen_days.py --week 5  # 只重生第 5 周
> ```

---

## Week 1 · VLM 架构解剖

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 1 | 技术版图与问题定义 | [`day-01.md`](day-01.md) | [`day-01_environment_and_first_inference.ipynb`](../notebooks/day-01_environment_and_first_inference.ipynb) | 💻/☁️ 两可 |
| 2 | 视觉编码器：ViT → SigLIP | [`day-02.md`](day-02.md) | [`day-02_vit_from_scratch.ipynb`](../notebooks/day-02_vit_from_scratch.ipynb) | ☁️ 云 GPU |
| 3 | 连接器：模态对齐那一层 | [`day-03.md`](day-03.md) | [`day-03_connector_compare.ipynb`](../notebooks/day-03_connector_compare.ipynb) | ☁️ 云 GPU |
| 4 | Qwen2.5-VL 架构精读 | [`day-04.md`](day-04.md) | [`day-04_visual_token_budget.ipynb`](../notebooks/day-04_visual_token_budget.ipynb) | ☁️ 云 GPU |

## Week 2 · 数据工程

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 5 | 从零手搭 Mini-VLM | [`day-05.md`](day-05.md) | [`day-05_minivlm_assembly.ipynb`](../notebooks/day-05_minivlm_assembly.ipynb) | ☁️ 云 GPU |
| 6 | 复盘：一次完整的图文推理（M1） | [`day-06.md`](day-06.md) | [`day-06_full_inference_review.ipynb`](../notebooks/day-06_full_inference_review.ipynb) | ☁️ 云 GPU |
| 7 | 图像预处理全链路 | [`day-07.md`](day-07.md) | [`day-07_image_preprocess.ipynb`](../notebooks/day-07_image_preprocess.ipynb) | 💻 本地 |
| 8 | 客服数据 Taxonomy 设计 | [`day-08.md`](day-08.md) | [`day-08_taxonomy_matrix.ipynb`](../notebooks/day-08_taxonomy_matrix.ipynb) | 💻 本地 |
| 9 | 数据合成 | [`day-09.md`](day-09.md) | [`day-09_data_synthesis.ipynb`](../notebooks/day-09_data_synthesis.ipynb) | 💻 本地 + API |
| 10 | 数据清洗与去重 | [`day-10.md`](day-10.md) | [`day-10_cleaning_dedup.ipynb`](../notebooks/day-10_cleaning_dedup.ipynb) | 💻 本地 |
| 11 | 数据打包与对话模板 | [`day-11.md`](day-11.md) | [`day-11_build_sft.ipynb`](../notebooks/day-11_build_sft.ipynb) | 💻 本地 |
| 12 | 数据集 v0 交付（M2） | [`day-12.md`](day-12.md) | [`day-12_dataset_card.ipynb`](../notebooks/day-12_dataset_card.ipynb) | 💻 本地 |

## W3 · SFT 训练工程

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 13 | 训练环境与显存工程 | [`days/day-13.md`](day-13.md) | [`notebooks/day-13_training_env_vram.ipynb`](../notebooks/day-13_training_env_vram.ipynb) | ☁️ 云 GPU |
| 14 | LoRA / QLoRA 原理与实现 | [`days/day-14.md`](day-14.md) | [`notebooks/day-14_lora_principles.ipynb`](../notebooks/day-14_lora_principles.ipynb) | ☁️ 云 GPU |
| 15 | 跑通第一次 LoRA SFT | [`days/day-15.md`](day-15.md) | [`notebooks/day-15_first_lora_sft.ipynb`](../notebooks/day-15_first_lora_sft.ipynb) | ☁️ 云 GPU |
| 16 | 训练日志与踩坑 | [`days/day-16.md`](day-16.md) | [`notebooks/day-16_training_log_debug.ipynb`](../notebooks/day-16_training_log_debug.ipynb) | ☁️ 云 GPU |
| 17 | 规模化训练：DeepSpeed / FSDP | [`days/day-17.md`](day-17.md) | [`notebooks/day-17_deepspeed_fsdp.ipynb`](../notebooks/day-17_deepspeed_fsdp.ipynb) | 💻 本地 |
| 18 | 第一次训练收官 | [`days/day-18.md`](day-18.md) | [`notebooks/day-18_first_training_wrap.ipynb`](../notebooks/day-18_first_training_wrap.ipynb) | ☁️ 云 GPU |

## W4 · 评测体系

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 19 | 通用 VLM 评测全景 | [`days/day-19.md`](day-19.md) | [`notebooks/day-19_benchmark_overview.ipynb`](../notebooks/day-19_benchmark_overview.ipynb) | ☁️ 云 GPU |
| 20 | 构建客服领域评测集 | [`days/day-20.md`](day-20.md) | [`notebooks/day-20_build_domain_eval.ipynb`](../notebooks/day-20_build_domain_eval.ipynb) | 💻 本地 |
| 21 | 自动评测流水线 | [`days/day-21.md`](day-21.md) | [`notebooks/day-21_eval_pipeline.ipynb`](../notebooks/day-21_eval_pipeline.ipynb) | ☁️ 云 GPU |
| 22 | 幻觉评测与缓解 | [`days/day-22.md`](day-22.md) | [`notebooks/day-22_hallucination_eval.ipynb`](../notebooks/day-22_hallucination_eval.ipynb) | 💻 本地 |
| 23 | 错误分析与 Bad Case 归类 | [`days/day-23.md`](day-23.md) | [`notebooks/day-23_error_analysis.ipynb`](../notebooks/day-23_error_analysis.ipynb) | 💻 本地 |
| 24 | 评测报告 v1 | [`days/day-24.md`](day-24.md) | [`notebooks/day-24_eval_report_v1.ipynb`](../notebooks/day-24_eval_report_v1.ipynb) | 💻 本地 |

## W5 · 偏好对齐与推理优化

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 25 | DPO 家族原理 | [`days/day-25.md`](day-25.md) | [`notebooks/day-25_dpo_family.ipynb`](../notebooks/day-25_dpo_family.ipynb) | ☁️ 云 GPU |
| 26 | 多模态偏好数据构造 | [`days/day-26.md`](day-26.md) | [`notebooks/day-26_preference_data.ipynb`](../notebooks/day-26_preference_data.ipynb) | 💻 本地 |
| 27 | 跑 DPO | [`days/day-27.md`](day-27.md) | [`notebooks/day-27_run_dpo.ipynb`](../notebooks/day-27_run_dpo.ipynb) | ☁️ 云 GPU |
| 28 | 可验证奖励与 GRPO（进阶，可跳过） | [`days/day-28.md`](day-28.md) | [`notebooks/day-28_verifiable_rewards.ipynb`](../notebooks/day-28_verifiable_rewards.ipynb) | 💻 本地 |
| 29 | 量化与推理加速 | [`days/day-29.md`](day-29.md) | [`notebooks/day-29_quantize_accelerate.ipynb`](../notebooks/day-29_quantize_accelerate.ipynb) | ☁️ 云 GPU |
| 30 | 端到端推理服务 | [`days/day-30.md`](day-30.md) | [`notebooks/day-30_inference_service.ipynb`](../notebooks/day-30_inference_service.ipynb) | ☁️ 云 GPU |

## W6 · 多模态 Agent

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 31 | Agent 范式与工具协议 | [`days/day-31.md`](day-31.md) | [`notebooks/day-31_agent_tools.ipynb`](../notebooks/day-31_agent_tools.ipynb) | 💻 本地 |
| 32 | 多模态 RAG | [`days/day-32.md`](day-32.md) | [`notebooks/day-32_multimodal_rag.ipynb`](../notebooks/day-32_multimodal_rag.ipynb) | ☁️ 云 GPU |
| 33 | 工具链完善与状态管理 | [`days/day-33.md`](day-33.md) | [`notebooks/day-33_tools_state_mgmt.ipynb`](../notebooks/day-33_tools_state_mgmt.ipynb) | 💻 本地 |
| 34 | Agent 骨架 | [`days/day-34.md`](day-34.md) | [`notebooks/day-34_agent_loop.ipynb`](../notebooks/day-34_agent_loop.ipynb) | ☁️ 云 GPU |
| 35 | 端到端联调 | [`days/day-35.md`](day-35.md) | [`notebooks/day-35_agent_demo.ipynb`](../notebooks/day-35_agent_demo.ipynb) | ☁️ 云 GPU |
| 36 | Agent 评测 | [`days/day-36.md`](day-36.md) | [`notebooks/day-36_agent_eval.ipynb`](../notebooks/day-36_agent_eval.ipynb) | 💻 本地 |

## W7 · Shopify SaaS 产品化

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 37 | Shopify 生态与 API | [`days/day-37.md`](day-37.md) | [`notebooks/day-37_shopify_api.ipynb`](../notebooks/day-37_shopify_api.ipynb) | 💻 本地 |
| 38 | OAuth 与应用骨架 | [`days/day-38.md`](day-38.md) | [`notebooks/day-38_shopify_oauth.ipynb`](../notebooks/day-38_shopify_oauth.ipynb) | 💻 本地 |
| 39 | 店铺前台挂件 | [`days/day-39.md`](day-39.md) | [`notebooks/day-39_theme_widget.ipynb`](../notebooks/day-39_theme_widget.ipynb) | 💻 本地 |
| 40 | Webhook 与索引同步 | [`days/day-40.md`](day-40.md) | [`notebooks/day-40_webhooks_index.ipynb`](../notebooks/day-40_webhooks_index.ipynb) | 💻 本地 |
| 41 | 计费与合规 | [`days/day-41.md`](day-41.md) | [`notebooks/day-41_billing_compliance.ipynb`](../notebooks/day-41_billing_compliance.ipynb) | 💻 本地 |
| 42 | 部署上线 | [`days/day-42.md`](day-42.md) | [`notebooks/day-42_deploy.ipynb`](../notebooks/day-42_deploy.ipynb) | 💻 本地 |

## W8 · 打磨与交付

| Day | 主题 | 讲义 | notebook | 在哪跑 |
|---|---|---|---|---|
| 43 | 全链路压测与成本核算 | [`days/day-43.md`](day-43.md) | [`notebooks/day-43_loadtest_cost.ipynb`](../notebooks/day-43_loadtest_cost.ipynb) | ☁️ 云 GPU |
| 44 | 安全、合规与可靠性 | [`days/day-44.md`](day-44.md) | [`notebooks/day-44_security_reliability.ipynb`](../notebooks/day-44_security_reliability.ipynb) | 💻 本地 |
| 45 | Shopify 审核对齐 | [`days/day-45.md`](day-45.md) | [`notebooks/day-45_shopify_review.ipynb`](../notebooks/day-45_shopify_review.ipynb) | 💻 本地 |
| 46 | 对比实验与消融 | [`days/day-46.md`](day-46.md) | [`notebooks/day-46_ablation.ipynb`](../notebooks/day-46_ablation.ipynb) | ☁️ 云 GPU |
| 47 | 技术报告 | [`days/day-47.md`](day-47.md) | [`notebooks/day-47_tech_report.ipynb`](../notebooks/day-47_tech_report.ipynb) | 💻 本地 |
| 48 | 开源与交付 | [`days/day-48.md`](day-48.md) | [`notebooks/day-48_open_source_ship.ipynb`](../notebooks/day-48_open_source_ship.ipynb) | 💻 本地 |

---

## 在哪跑？（省钱的关键）

| 徽标 | 含义 |
|---|---|
| ☁️ 云 GPU | 必须租机器（训练 / 推理 / 服务）。什么时候开、开多久见 `docs/13-hardware-and-cost.md` |
| 💻 本地 | 你的 Mac 就能跑，不开 GPU，零成本 |
| 💻 本地 + API | 本地跑，但要 `.env` 里的 API key（合成 / 裁判模型） |

**规律**：W2（数据）、W4（评测）、W7（Shopify）几乎全在本地；
只有 W3（训练）和 W5（对齐+推理）需要持续开卡。

## 每天开工第一条命令

```bash
make day N=25      # 打印当天：讲义路径 + notebook + 代码入口 + 命令
make list-days     # 列出全部日材料
```
