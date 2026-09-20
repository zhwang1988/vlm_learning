# =============================================================================
# Makefile —— 常用命令的短名字
#
# 存在的理由：这个项目的命令有点长（python -m src.train.sft_peft --config ...），
# 记不住就容易偷懒跳过（比如跳过 --dry-run）。给它们短名字，你就愿意跑了。
#
#   make            # 看所有可用命令
#   make check      # 环境体检（最先跑这个）
#   make day1       # Day 1 的验收
# =============================================================================

PY ?= python
CONFIG_SFT   ?= configs/sft_lora_3b.yaml
CONFIG_QLORA ?= configs/sft_qlora_3b.yaml
CONFIG_DPO   ?= configs/dpo_3b.yaml
# 模型权重放哪。默认数据盘第一挂点（Coabs/AutoDL 等都通用）。
MODEL_ROOT ?= /root/autodl-tmp/models

.PHONY: help check check-cloud day1 day4 smoke \
        data-plan data-synth data-clean data-build \
        train-dry train-try train-qloa train \
        dpo-verify dpo-build dpo \
        eval domain-eval error-analysis \
        serve vllm demo demo-down demo-cost \
        clean clean-reports sync-down

help:
	@echo ""
	@echo "  multimodal-lab —— 常用命令"
	@echo "  ============================================================"
	@echo ""
	@echo "  上手"
	@echo "    make check          环境体检（第一步就跑这个）"
	@echo "    make check-cloud    同上，但按云训练机的标准卡（在云机器上跑）"
	@echo "    make model-status   看模型下载状态"
	@echo "    make model-3b       下载 Qwen2.5-VL-3B-Instruct（云上 Day 1）"
	@echo "    make model-7b       下载 Qwen2.5-VL-7B-Instruct（可选）"
	@echo "    make day1           Day 1 验收：跑通一次 VLM 推理"
	@echo "    make day4           Day 4 验收：视觉 token 数对不对"
	@echo "    make smoke          全模块自检（跑所有 __main__）"
	@echo ""
	@echo "  数据（Day 8-12）"
	@echo "    make data-plan      看数据配比矩阵"
	@echo "    make data-synth     合成数据（先 --limit 20 试跑）"
	@echo "    make data-clean     清洗去重"
	@echo "    make data-build     打包成训练格式 + 切分"
	@echo ""
	@echo "  训练（Day 14-18）"
	@echo "    make train-dry      只检查数据格式（10 秒，必做）"
	@echo "    make train-try      50 条数据试 20 步（验证 loss 在动）"
	@echo "    make train-qloa     QLoRA 正式训练"
	@echo "    make train          LoRA 正式训练（主配置）"
	@echo ""
	@echo "  对齐（Day 27-29）"
	@echo "    make dpo-verify     验证 DPO loss 实现"
	@echo "    make dpo-build      从 bad cases 构造偏好对"
	@echo "    make dpo            DPO 训练"
	@echo ""
	@echo "  评测（Day 20-24）"
	@echo "    make domain-eval    构造域内评测集"
	@echo "    make eval           跑评测出报告"
	@echo "    make error-analysis 错误归因 -> bad_cases.jsonl"
	@echo ""
	@echo "  服务与演示（Day 25+ / Day 39+）"
	@echo "    make vllm           起 vLLM"
	@echo "    make demo           起完整演示会话"
	@echo "    make demo-down      关掉（省钱！）"
	@echo "    make demo-cost      这次烧了多少钱"
	@echo ""
	@echo "  维护"
	@echo "    make sync-down      把云上产物拉回本地（重要！）"
	@echo "    make clean          清临时文件"
	@echo ""

# ---------------------------------------------------------------------------
# 上手
# ---------------------------------------------------------------------------

check:
	@echo "本地跑：torch/CUDA 缺失在这里只是提示，不影响你读文档改代码。"
	$(PY) scripts/env_check.py

check-cloud:
	@echo "云机器跑：torch/CUDA/显存 缺失在这里都是硬伤（退出码 1）。"
	$(PY) scripts/env_check.py --mode cloud

# 模型权重下载（云上 Day 1 必跑）
model-status:
	$(PY) scripts/download_model.py --status --root $(MODEL_ROOT)

model-3b:
	@echo "下载 Qwen2.5-VL-3B-Instruct（约 6.2 GB，2-5 分钟）"
	$(PY) scripts/download_model.py --model 3b-instruct --root $(MODEL_ROOT)

model-7b:
	@echo "下载 Qwen2.5-VL-7B-Instruct（约 14.5 GB，5-10 分钟）"
	$(PY) scripts/download_model.py --model 7b-instruct --root $(MODEL_ROOT)

model-verify:
	$(PY) scripts/download_model.py --verify --model 3b-instruct --root $(MODEL_ROOT)

day1:
	$(PY) -c "import torch; from transformers import AutoProcessor; print('依赖 OK'); print('CUDA:', torch.cuda.is_available())"
	@echo ""
	@echo "  下一步：跑一次真实推理，确认模型看得见图"
	$(PY) -m src.minivlm.generate --compare-order

day4:
	@echo "Day 4 验收：我们算出的视觉 token 数，是否等于官方的 <|image_pad|> 数量"
	@echo "（这个测试通过 = 你真的搞懂了 Qwen2.5-VL 的分辨率处理）"
	$(PY) -m src.minivlm.processor --check

smoke:
	@echo "=== minivlm: 从零搭的 ViT ==="
	-$(PY) -m src.minivlm.vision
	@echo ""
	@echo "=== minivlm: 四种连接器对比 ==="
	-$(PY) -m src.minivlm.connector
	@echo ""
	@echo "=== data: 图像工具（EXIF/透明底/pHash）==="
	-$(PY) -m src.data.image_utils
	@echo ""
	@echo "=== train: LoRA 实现（零初始化验证）==="
	-$(PY) -m src.train.lora_utils
	@echo ""
	@echo "=== train: DPO loss（4 个场景 + ln2 校验）==="
	-$(PY) -m src.train.dpo_loss --verify
	@echo ""
	@echo "=== eval: 规则指标 ==="
	-$(PY) -m src.eval.metrics
	@echo ""
	@echo "=== agent: 工具幂等性 + 防工具幻觉 ==="
	-$(PY) -m src.agent.tools
	@echo ""
	@echo "=== shopify: HMAC 校验 + SSRF 防护 ==="
	-$(PY) -m src.shopify.auth
	@echo ""
	@echo "=== shopify: webhook 幂等 ==="
	-$(PY) -m src.shopify.webhooks

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

data-plan:
	$(PY) -m src.data.taxonomy --print

data-synth:
	@echo "先小批量试跑，确认 prompt 效果和花费："
	@echo "  $(PY) -m src.data.synth --limit 20 --dry-run"
	@echo ""
	$(PY) -m src.data.synth --limit 50

data-clean:
	$(PY) -m src.data.dedup --in data/raw/synth.jsonl --out data/interim/clean.jsonl

data-build:
	$(PY) -m src.data.build_sft --in data/interim/clean.jsonl --out-dir data/processed
	$(PY) -m src.data.report --data data/processed/sft_train.jsonl

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

train-dry:
	@echo "只检查数据格式，不加载模型。跳过这一步是浪费时间的最经典方式。"
	$(PY) -m src.train.sft_peft --config $(CONFIG_SFT) --dry-run

train-try:
	@echo "50 条数据跑 20 步：确认 loss 在下降、显存够、没有 NaN"
	$(PY) -m src.train.sft_peft --config $(CONFIG_SFT) --limit 50 --max-steps 20

train-qloa:
	$(PY) -m src.train.sft_peft --config $(CONFIG_QLORA)

train:
	$(PY) -m src.train.sft_peft --config $(CONFIG_SFT)

# ---------------------------------------------------------------------------
# 对齐
# ---------------------------------------------------------------------------

dpo-verify:
	$(PY) -m src.train.dpo_loss --verify

dpo-build:
	$(PY) -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl

dpo:
	$(PY) -m src.train.dpo --config $(CONFIG_DPO)

# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

domain-eval:
	$(PY) -m src.eval.build_domain_eval --out data/eval/domain_eval.jsonl

eval:
	$(PY) -m src.eval.run_eval --data data/eval/domain_eval.jsonl --out reports/

error-analysis:
	$(PY) -m src.eval.error_analysis --in reports/eval_results.jsonl --out reports/

# ---------------------------------------------------------------------------
# 服务与演示
# ---------------------------------------------------------------------------

vllm:
	bash src/serve/vllm_server.sh

serve:
	$(PY) -m uvicorn src.serve.api:app --host 0.0.0.0 --port 8080

demo:
	bash scripts/demo_up.sh

demo-down:
	bash scripts/demo_up.sh --down

demo-cost:
	bash scripts/demo_up.sh --cost

# ---------------------------------------------------------------------------
# 维护
# ---------------------------------------------------------------------------

sync-down:
	@if [ -z "$(HOST)" ]; then \
		echo "用法: make sync-down HOST=<ip> PORT=<端口>"; \
		echo "  例: make sync-down HOST=connect.xxx.seetacloud.com PORT=12345"; \
		echo ""; \
		echo "先看要拉什么: bash scripts/sync_down.sh --host <ip> --port <端口> --dry-run"; \
	else \
		bash scripts/sync_down.sh --host $(HOST) --port $(PORT); \
	fi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f .demo_session_start 2>/dev/null || true
	rm -rf .demo_pids 2>/dev/null || true
	@echo "临时文件已清理（data/ outputs/ reports/ 都保留了）"

clean-reports:
	@echo "这会删掉 reports/ 下的所有报告。"
	@printf "确认？(y/N) "; read -r a; [ "$$a" = "y" ] && rm -rf reports/* && echo "已清空" || echo "取消"
