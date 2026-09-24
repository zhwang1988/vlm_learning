# =============================================================================
# Makefile —— 常用命令的短名字
#
# 存在的理由：这个项目的命令有点长（python -m src.train.sft_peft --config ...），
# 记不住就容易偷懒跳过（比如跳过 --dry-run）。给它们短名字，你就愿意跑了。
#
#   make                 # 看所有可用命令
#   make check           # 环境体检（最先跑这个）
#   make selfcheck       # 代码自检（确认代码本身没坏）
#   make day N=13        # 第 N 天的讲义 + notebook + 代码入口
#
# ⚠️ 改这里的命令时，请顺手跑一次：
#      make selfcheck
#    下面每一条都对应 src/ 里真实存在的入口。之前 `smoke` 用 `-` 前缀吞掉了
#    退出码，命令写错了也看不出来 —— 现在统一交给 scripts/selfcheck.py，
#    它用子进程跑并检查退出码，写错的命令会当场暴露。
# =============================================================================

PY ?= python

# 模型：本地路径或 HF id。训练/评测命令都要它。
MODEL        ?= Qwen/Qwen2.5-VL-3B-Instruct
ADAPTER      ?= outputs/sft_lora_3b

CONFIG_SFT   ?= configs/sft_lora_3b.yaml
CONFIG_QLORA ?= configs/sft_qlora_3b.yaml
CONFIG_DPO   ?= configs/dpo_3b.yaml

# 模型权重放哪。默认数据盘第一挂点（Coabs/AutoDL 等都通用）。
MODEL_ROOT ?= /root/autodl-tmp/models

# 评测集 / 报告的路径（全项目统一，别再各写各的）
EVAL_SET   ?= data/eval/cx_eval_v1.jsonl
CLEAN_DATA ?= data/processed/clean.jsonl
RUNS       ?= reports/eval_base_raw.jsonl
LIMIT      ?= 0

# 上游原始数据。默认是 Day 9 真调模型合成的产物；
# `RAW=data/fixtures/demo_synth.jsonl` 可以切成离线演示数据，
# 让 Day 10–12 在没有 API key 的机器上也能跑。
RAW        ?= data/raw/synth_v0.jsonl

.PHONY: help check check-cloud selfcheck smoke \
        day list-days model-status model-3b model-7b model-verify day1 day4 \
        data-plan data-synth data-clean data-build data-card \
        demo-data demo-check demo-clean \
        train-dry train-try train-qlora train-qloa train \
        dpo-verify dpo-build dpo \
        domain-eval eval eval-fake quick-eval benchmark error-analysis report \
        serve vllm up down psql demo demo-down demo-cost \
        clean clean-reports sync-down

help:
	@echo ""
	@echo "  multimodal-lab —— 常用命令"
	@echo "  ============================================================"
	@echo ""
	@echo "  上手"
	@echo "    make check          环境体检（第一步就跑这个）"
	@echo "    make check-cloud    同上，但按云训练机的标准卡（在云机器上跑）"
	@echo "    make selfcheck      代码自检：全模块跑一遍（找不到环境问题就找代码问题）"
	@echo "    make smoke          = selfcheck --core，零依赖子集，最快"
	@echo "    make day N=13       看第 N 天的讲义路径 + notebook + 代码入口"
	@echo "    make list-days      列出所有已拆出的每日材料"
	@echo "    make model-status   看模型下载状态"
	@echo "    make model-3b       下载 Qwen2.5-VL-3B-Instruct（云上 Day 1）"
	@echo "    make model-7b       下载 Qwen2.5-VL-7B-Instruct（可选）"
	@echo "    make day1           Day 1 验收：跑通一次 VLM 推理"
	@echo "    make day4           Day 4 验收：视觉 token 数对不对"
	@echo ""
	@echo "  数据（Day 8-12）"
	@echo "    make data-plan      看数据配比矩阵"
	@echo "    make data-synth     合成数据（先 --limit 20 试跑）"
	@echo "    make data-clean     清洗去重"
	@echo "    make data-build     打包成训练格式 + 切分 + 泄漏检查"
	@echo "    make data-card      生成数据集卡片（Day 12 交付物）"
	@echo ""
	@echo "  离线演示数据（没 GPU / 没 API key 也能跑 Day 9-12）"
	@echo "    make demo-check     造数据 + 清洗 + 和 ground truth 对账（最推荐）"
	@echo "    make demo-data      只造演示数据（含 49 张真图和 ID 级 label）"
	@echo "    make demo-clean     用演示数据跑清洗，看 reports/cleaning_report.md"
	@echo "    （也可以给别的目标喂演示数据：make data-clean RAW=data/fixtures/demo_synth.jsonl）"
	@echo ""
	@echo "  训练（Day 14-18）"
	@echo "    make train-dry      只检查数据格式（10 秒，必做）"
	@echo "    make train-try      50 条数据试 20 步（验证 loss 在动）"
	@echo "    make train-qlora    QLoRA 正式训练（显存小走这条）"
	@echo "    make train          LoRA 正式训练（主配置）"
	@echo ""
	@echo "  对齐（Day 25-29）"
	@echo "    make dpo-verify     验证 DPO loss 实现（含 ln2 数值检查）"
	@echo "    make dpo-build      从 bad cases 构造偏好对"
	@echo "    make dpo            DPO 训练"
	@echo ""
	@echo "  评测（Day 19-24）"
	@echo "    make domain-eval    构造域内评测集"
	@echo "    make eval           跑域内评测（需要 GPU）"
	@echo "    make eval-fake      ⭐ 离线验证评测链路：造好/坏/全拒答三种预测再对比"
	@echo "    make quick-eval     训练前后抽 20 条并排对比（最快的手感检查）"
	@echo "    make benchmark      通用榜（--list 看有哪些，不需要 GPU）"
	@echo "    make report         把 raw 结果合成报告"
	@echo "    make error-analysis 错误归因 -> bad_cases.jsonl"
	@echo ""
	@echo "  服务与演示（Day 30+ / Day 42）"
	@echo "    make vllm           起 vLLM"
	@echo "    make serve          起推理网关（FastAPI）"
	@echo "    make up             Docker 一键起：PostgreSQL(pgvector) + 应用"
	@echo "    make down           停 Docker 环境"
	@echo "    make psql           进数据库命令行"
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

# 代码自检：把 src/ 下所有 __main__ 跑一遍。
# 用子进程逐个跑 —— 模块的自检会改全局状态（清 mock 表、推进计数器），
# 同进程连跑会出现「单独跑 OK、一起跑 FAIL」的假故障。
selfcheck:
	$(PY) scripts/selfcheck.py

smoke:
	@echo "零依赖子集：一个包都不装也应该是全绿的。"
	$(PY) scripts/selfcheck.py --core

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

# ---------------------------------------------------------------- 每日材料入口
#   make day N=13    打印第 N 天的讲义路径、notebook 路径、当天代码入口
day:
	@test -n "$(N)" || (echo "用法: make day N=13"; exit 1)
	@n=$(N); nn=$$(printf "%02d" $$n); \
	 echo "=================== Day $$n ==================="; \
	 echo "📄 讲义      : days/day-$$nn.md"; \
	 echo "📓 notebook  : $$(ls notebooks/day-$$nn*.ipynb 2>/dev/null || echo '(暂无，按讲义手写)')"; \
	 echo "💻 代码      : 见 days/day-$$nn.md 「二、写」一节"; \
	 echo; \
	 head -14 days/day-$$nn.md 2>/dev/null || echo "(day-$$nn.md 还没写，Day 1–4 见 PLAN.md)"

#   make list-days 列出所有已拆出的日文件
list-days:
	@echo "已拆出的每日讲义:"; ls -1 days/day-*.md 2>/dev/null | sed 's/^/  📄 /'; \
	 echo; echo "已拆出的每日 notebook:"; ls -1 notebooks/day-*.ipynb 2>/dev/null | sed 's/^/  📓 /'

# ---------------------------------------------------------------------------
# 数据（Day 8-12）
# ---------------------------------------------------------------------------

data-plan:
	$(PY) -m src.data.taxonomy --print

data-synth:
	@echo "先小批量试跑，确认 prompt 效果和花费："
	@echo "  $(PY) -m src.data.synth --limit 20 --dry-run"
	@echo ""
	$(PY) -m src.data.synth

# 注意：dedup 的默认输出就是 data/processed/clean.jsonl，
# 而 build_sft / build_domain_eval 的默认输入也是它。全项目统一这个路径，
# 别改成 data/interim/ —— 那样下游四个脚本都会找不到文件。
data-clean:
	$(PY) -m src.data.dedup --in $(RAW) --out $(CLEAN_DATA)

data-build:
	$(PY) -m src.data.build_sft --in $(CLEAN_DATA) --out-dir data/processed
	$(PY) -m src.data.report --data-dir data/processed

data-card:
	$(PY) -m src.data.report --data-dir data/processed --name CX-VLM-SFT-v0

# 离线演示数据（不需要 GPU / 不需要 API key）
#   make demo-data     造数据 + ground truth 清单
#   make demo-check    跑清洗并对账，PASS/FAIL 直接给结论
#   make demo-clean    只跑清洗（用演示数据当输入），看 reports/cleaning_report.md
#
# 这一组命令的意义：Day 9 的数据合成要调模型（要钱），Day 10–12 又要读它的产物，
# 于是没有 key 的机器上第 9 天之后全断了。演示数据把这个断点补上 ——
# 并且因为带了 ground truth，它检验的不只是「跑得动」，还有「跑得对」。
demo-data:
	$(PY) scripts/make_demo_data.py

demo-check:
	$(PY) scripts/make_demo_data.py
	$(PY) scripts/reconcile_demo.py

demo-clean:
	@echo "⚠️  下面这条命令用**演示数据**当输入，但输出路径和真实流水线是同一个："
	@echo "      $(CLEAN_DATA)"
	@echo "    也就是说它会覆盖掉你用真实数据清洗出来的结果。"
	@echo "    想保留真实结果，就先备份，或者换个输出路径："
	@echo "      make demo-clean CLEAN_DATA=data/processed/demo_clean.jsonl"
	@echo ""
	$(PY) -m src.data.dedup --in data/fixtures/demo_synth.jsonl --out $(CLEAN_DATA)

# ---------------------------------------------------------------------------
# 训练（Day 14-18）
# ---------------------------------------------------------------------------

train-dry:
	@echo "只检查数据格式，不加载模型。跳过这一步是浪费时间的最经典方式。"
	$(PY) -m src.train.sft_peft --config $(CONFIG_SFT) --dry-run

train-try:
	@echo "50 条数据跑 20 步：确认 loss 在下降、显存够、没有 NaN"
	$(PY) -m src.train.sft_peft --config $(CONFIG_SFT) --limit 50 --max-steps 20

# 名字容易打错（qlora 不是 qloa），保留旧名的别名免得你踩空
train-qlora:
	$(PY) -m src.train.sft_peft --config $(CONFIG_QLORA)

train-qloa: train-qlora

train:
	$(PY) -m src.train.sft_peft --config $(CONFIG_SFT)

# ---------------------------------------------------------------------------
# 对齐（Day 25-29）
# ---------------------------------------------------------------------------

dpo-verify:
	$(PY) -m src.train.dpo_loss --verify

dpo-build:
	$(PY) -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl

dpo:
	$(PY) -m src.train.dpo --config $(CONFIG_DPO)

# ---------------------------------------------------------------------------
# 评测（Day 19-24）
# ---------------------------------------------------------------------------

domain-eval:
	$(PY) -m src.eval.build_domain_eval --source $(CLEAN_DATA) --out $(EVAL_SET)

eval:
	@test -f $(EVAL_SET) || (echo "缺少评测集 $(EVAL_SET)，先跑 make domain-eval"; exit 1)
	$(PY) -m src.eval.run_eval --model $(MODEL) --eval $(EVAL_SET) --tag base \
		$(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),)

# 离线验证评测链路（不需要 GPU / 不需要模型）
#   oracle 是**可通行性探针**：它会拼出一个必然满足全部规则的回答。
#   如果连 oracle 都过不了，问题在评测集，不在模型 —— 这种错误在真实评测里
#   只会表现为「模型怎么训都上不去」，很容易误判成模型能力问题。
eval-fake:
	@test -f $(EVAL_SET) || (echo "缺少评测集 $(EVAL_SET)，先跑 make domain-eval"; exit 1)
	$(PY) scripts/fake_eval.py --mode good   --tag fake_good   --eval $(EVAL_SET)
	$(PY) scripts/fake_eval.py --mode bad    --tag fake_bad    --eval $(EVAL_SET)
	$(PY) scripts/fake_eval.py --mode silent --tag fake_silent --eval $(EVAL_SET)
	$(PY) -m src.eval.report --runs reports/eval_fake_good_raw.jsonl \
		reports/eval_fake_bad_raw.jsonl reports/eval_fake_silent_raw.jsonl
	@echo ""
	@echo "  看 reports/eval_v1.md：好的应该 > 坏的 > 全拒答的。"
	@echo "  silent（一律回答『我无法确定』）用来验伪指标设计："
	@echo "  它幻觉率 0%，但漏答率极高 —— 报告必须能把它判成最差的那个。"

quick-eval:
	$(PY) -m src.eval.quick_eval --base $(MODEL) --eval $(EVAL_SET) --n 20

benchmark:
	$(PY) -m src.eval.run_benchmark --list

report:
	$(PY) -m src.eval.report --runs $(RUNS)

error-analysis:
	$(PY) -m src.eval.error_analysis --in $(RUNS) --out-dir reports

# ---------------------------------------------------------------------------
# 服务与演示（Day 30+ / Day 42）
# ---------------------------------------------------------------------------

vllm:
	bash src/serve/vllm_server.sh

serve:
	$(PY) -m uvicorn src.serve.api:create_app --factory --host 0.0.0.0 --port 8080

# Docker 一键环境（W7）：PostgreSQL(pgvector) + 应用，本地不用装数据库
up:
	docker compose up -d --build
	@echo "  应用: http://localhost:8001/health   数据库: make psql"

down:
	docker compose down

psql:
	docker compose exec db psql -U mmlab -d mmlab

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
