"""Week 3 · SFT 训练工程（Day 13–18）

这一周的核心矛盾：**显存**。全部 6 天都在围绕「怎么把 3B 模型塞进一张卡并真的训出东西」。
Day 13–14 在云上（要 torch），Day 15–17 云上训练，Day 18 云上收尾。
"""
from __future__ import annotations


def D(**kw):
    return kw


DAYS = [
    # ======================================================================
    D(
        week=3, n=13, slug="training_env_vram", title="训练环境与显存工程",
        where="cloud",
        files="`scripts/estimate_vram.py`、`docs/06-sft-training.md`",
        prereq="Day 12（数据集 v0 已交付）；云机器已租好，`bash scripts/cloud_bootstrap.sh` 跑过",
        goal="把「显存四分账」——权重 / 梯度 / 优化器状态 / 激活值 —— 变成一张能直接指导"
             "租卡的表格：3B·7B × 全参·LoRA·QLoRA × seq 2048·4096。",
        read=[
            "`docs/06-sft-training.md` 第 1–3 节 —— 今天的全部理论都在这里",
            "`scripts/estimate_vram.py` 源码 —— 边读边和讲义里的公式对一遍",
            "回看 `docs/01-architecture.md` 第 5 节（冻结策略），今天要回答视觉塔为什么冻结",
        ],
        think=[
            "全参 bf16 + AdamW 为什么按约 12 字节/参数算？（提示：权重 2 + 梯度 2 + 一阶动量 4 + 二阶动量 4）",
            "gradient checkpointing 省的是四本账里的哪一本？代价是什么？",
            "视觉塔已经冻结、不产生梯度，为什么它的前向激活值还占显存？",
        ],
        write_title="`scripts/estimate_vram.py`（已给实现，你要读懂并改参数）",
        write_rows=[
            ("`estimate()`", "把显存拆成 权重 / 梯度 / 优化器 / 激活 四项，各自可单独打印"),
            ("激活值分支", "`grad_checkpointing=True` 只存层边界，否则存全部中间量"),
            ("视觉塔激活", "冻结也要算前向；开了 checkpointing 只存层边界 + 重算峰值"),
            ("`suggest()`", "**用经验区间上沿**选卡，不要用解析估算值（解析值天然偏乐观）"),
            ("`--suggest`", "输出「最小可行 / 舒适选择」，并对不支持 bf16 的卡单独告警"),
        ],
        write_note="> 这一版我修过两个真 bug，你读的时候留意对应注释：\n"
                   "> ① 激活值曾被按 4-bit 算 —— 权重 4-bit 了，激活**永远是 bf16**；\n"
                   "> ② 选卡曾推荐 16 GB T4 且不留余量 —— 现在按经验区间上沿 ×1.15 来选。",
        run=[
            ("python scripts/estimate_vram.py --all", "全矩阵：3B/7B × 全参/LoRA/QLoRA"),
            ("python scripts/estimate_vram.py --model 3b --method qlora --suggest",
             "只算 3B QLoRA，并给出选卡建议"),
        ],
        expect="""
  全矩阵  (解析估算 GB / 实测经验区间 GB)
  条件: batch=1, seq=2048, 1 张 1024² 图, 梯度检查点开, flash-attn 开
==============================================================================================
  模型                               全参bf16           LoRA          QLoRA         推理bf16
----------------------------------------------------------------------------------------------
  0.5B（教学用）                  13 / 12 – 16      4 / 6 – 9      3 / 4 – 6    2 / 1.5 – 3
  3B                         63 / 48 – 60   11 / 16 – 20     5 / 8 – 12     7 / 8 – 10
  7B                       136 / 100 – 130   20 / 30 – 38    8 / 16 – 22   15 / 15 – 18
  32B                      550 / 420 – 520  74 / 80 – 100   23 / 45 – 60   61 / 62 – 70
==============================================================================================

  一句话结论：本项目的甜点区是 3B QLoRA + 24 GB 卡。
""",
        accept=[
            "`--all` 矩阵能说清每一格为什么差这么多（尤其「3B 全参 63 GB vs QLoRA 5 GB」）",
            "`--suggest` 能给出卡型建议，并且知道为什么推荐 3090/4090 而不是 T4",
            "能不看资料推导出「3B QLoRA 在 16 GB 卡上必须开 gradient checkpointing」",
            "**租卡决定已经做了**（写进今日打卡：准备租哪张卡、为什么、预算多少）",
        ],
        pits=[
            "解析估算**天然偏乐观** —— 真实框架还有 padding、flash-attn workspace、"
            "dataloader buffer。选卡一定按经验区间上沿，脚本里按 0.92 可用率折算。",
            "**激活值精度搞错** —— 权重 4-bit 不代表激活 4-bit。QLoRA 的激活仍是 bf16，"
            "这是最容易把显存估少一半的地方。",
            "忘了算 logits —— `vocab=151936 × seq × batch × 2 bytes`，seq 到 4096 时它是 1 GB 级。",
            "T4 / V100 **不支持 bf16**，配置里得写 `bf16: false, fp16: true`。",
        ],
        nb=[
            ("md", "## 1. 跑全矩阵"),
            ("code", '''import subprocess, sys
out = subprocess.run([sys.executable, "../scripts/estimate_vram.py", "--all"],
                     capture_output=True, text=True)
print(out.stdout or out.stderr)'''),
            ("md", """## 2. 手算一遍，验证脚本没骗你

显存四分账里，**静态三项**（权重/梯度/优化器）是可以手算的。
激活值必须靠经验区间 —— 这就是为什么脚本要同时打印两个数。"""),
            ("code", '''P = 3.0e9                      # 3B 总参数
GB = 1024 ** 3
trainable_ratio = 0.005        # LoRA r=16 大约 0.5% 可训练参数

weights = P * 0.5 / GB                      # 4-bit 权重
lora    = P * trainable_ratio * 2 / GB      # A/B 矩阵，bf16
grad    = lora                              # 只有 LoRA 参数有梯度
opt     = P * trainable_ratio * 8 / GB      # AdamW 两个状态 × 4 字节

print(f"权重(4bit) {weights:6.2f} GB")
print(f"LoRA A/B   {lora:6.3f} GB")
print(f"梯度       {grad:6.3f} GB")
print(f"优化器     {opt:6.3f} GB")
print(f"静态合计   {weights+lora+grad+opt:6.2f} GB   ← 剩下的全给激活值 + logits + KV")'''),
            ("md", """## 3. 自己改参数做敏感性实验

把 `seq_len` 从 2048 拉到 4096、把图片从 1 张改成 2 张，看哪一项涨得最凶。
**结论应该指向：图片 token 数比 batch 更影响显存** —— 这是 VLM 训练和纯文本训练最大的不同。"""),
            ("code", '''import subprocess, sys
for args in (["--model", "3b", "--method", "qlora", "--seq", "2048"],
             ["--model", "3b", "--method", "qlora", "--seq", "4096"],
             ["--model", "7b", "--method", "qlora", "--seq", "2048"]):
    r = subprocess.run([sys.executable, "../scripts/estimate_vram.py", *args],
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if "解析估算合计" in l or "实测经验区间" in l]
    print(" ".join(args[-2:]), "→", " | ".join(t.strip() for t in tail))'''),
        ],
        next_day_hint="`days/day-14.md` —— LoRA 原理，手写低秩 A/B 矩阵",
    ),

    # ======================================================================
    D(
        week=3, n=14, slug="lora_principles", title="LoRA / QLoRA 原理与实现",
        where="cloud",
        files="`src/train/lora_utils.py`",
        prereq="Day 13（显存账已算清）",
        goal="手写一个最小 LoRA 层（低秩 A/B 矩阵），验证「B 零初始化 → 输出与基座逐元素相等」，"
             "并搞清 `r` / `alpha` / `target_modules` 在 VLM 里怎么选。",
        read=[
            "`docs/06-sft-training.md` 第 4 节",
            "LoRA 原论文的关键 5 页（`docs/12-papers.md` 标了阅读重点，别通读）",
            "`src/train/lora_utils.py` 顶部注释 —— 里面写了 target_modules 的选取逻辑",
        ],
        think=[
            "B 为什么必须**零初始化**？如果随机初始化，训练第一步会发生什么？",
            "`scaling = alpha / r` 的作用是什么？把 r 从 8 改成 64，alpha 要不要跟着改？",
            "为什么 `FORBIDDEN = [\"visual\"]` —— 视觉塔禁止注入 LoRA？",
        ],
        write_title="`src/train/lora_utils.py`",
        write_rows=[
            ("`LoRALinear.__init__`", "A 用 kaiming 初始化、**B 用 zeros**；`scaling = alpha / r`"),
            ("`LoRALinear.forward`", "`base(x) + dropout(x) @ A.T @ B.T * scaling`"),
            ("`LoRALinear.merge`", "把 BA 加回 base 权重 —— 推理时零额外开销"),
            ("`resolve_target_modules`", "`q/k/v/o` + `gate/up/down` + **`merger.mlp`**"),
            ("`FORBIDDEN`", "视觉塔永不注入（冻结 + 不注入，双保险）"),
            ("`estimate_training_vram`", "和 Day 13 的 `estimate_vram.py` 交叉验证，两个数应该对得上"),
        ],
        write_note="> **今天最该记住的一句**：LoRA 不是「小模型」，是「给大模型加了一小块可训练的旁路」，"
                   "原权重全程冻结、只更新 A 和 B。",
        run=[
            ("python -m src.train.lora_utils --demo", "B 零初始化后，输出应与基座逐元素相等"),
            ("python -m src.train.lora_utils --table", "r=8/16/64 的参数量与显存对照"),
        ],
        expect="""
[1] LoRALinear 数学正确性
    基座输出与加 LoRA 后输出 allclose = True   ← B 是零，必须相等
    手动 merge 后 allclose = True

[2] 注入计划
    可注入 (12 类) : q_proj k_proj v_proj o_proj gate_proj up_proj down_proj merger.mlp.0 merger.mlp.2 ...
    禁止注入       : visual.*   ← 视觉塔
    可训练参数     : 30.4 M  (0.86%)

[3] r 的影响（以 q_proj 为例）
     r=8    →  0.05 M/层      r=16  →  0.10 M/层      r=64  →  0.39 M/层
     r 翻 4 倍，参数量也翻 4 倍；alpha/r 会把这个差异抵消掉一部分
""",
        accept=[
            "`--demo` 全部断言通过（**B=0 时 `torch.allclose` 必须为 True**）",
            "能说清 `r` / `alpha` / `dropout` / `target_modules` 各自作用和调参直觉",
            "能说出「为什么客服场景要优先注入 `merger.mlp`」（视觉和语言的对齐层）",
            "知道 `merge_and_unload()` 之后推理为什么和全量模型一样快",
        ],
        pits=[
            "**B 用了随机初始化** —— 训练一开始 loss 就炸，而且很难查（不是数据问题，是初始化问题）。",
            "忘了乘 `scaling` —— r 变大时有效学习率跟着变，超参搜索白做。",
            "**漏掉 `merger.mlp`** —— 只注入 LLM 内部层，视觉和语言的对齐层不更新，"
            "SFT 收益大打折扣。这是 VLM 版 LoRA 和纯文本版最大的差别。",
            "在 fp16 下 merge —— 会掉精度。merge 要在 fp32/bf16 做，然后转回目标精度。",
            "r 一味调大 —— 参数量线性涨，收益却很快饱和。客服场景 r=16 通常够，先固定 r 调数据。",
        ],
        nb=[
            ("md", """## 1. 十行手写一遍 LoRA，确认你真的懂

不看 `lora_utils.py`，自己写一遍最小版本。核心只有一行：
`y = Wx + (alpha/r) · B(Ax)`，其中 **B 初始化为零**。"""),
            ("code", '''import torch, torch.nn as nn

class MiniLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False          # 原权重冻结
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(r, d_in)); nn.init.kaiming_uniform_(self.A)
        self.B = nn.Parameter(torch.zeros(d_out, r))     # ★ 零初始化
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

base = nn.Linear(64, 32)
lora = MiniLoRA(base)
x = torch.randn(4, 64)
print("B=0 时与基座相等:", torch.allclose(lora(x), base(x), atol=1e-6))
print("可训练参数:", sum(p.numel() for p in lora.parameters() if p.requires_grad),
      "/", sum(p.numel() for p in lora.parameters()))'''),
            ("md", "## 2. 真实注入计划（对着模型看）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.lora_utils", "--table"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", """## 3. 思考题落笔

在下面这个格子写下你的答案（写完就是打卡素材）：

1. 为什么 LoRA 能省显存，而 **QLoRA 还能再省一半**？（提示：基座权重存成什么精度）
2. 客服场景里，一个「看得懂图」的 LoRA，最该更新的是哪些层？"""),
            ("code", '''my_answer = """
1.
2.
"""
print(my_answer)'''),
        ],
        next_day_hint="`days/day-15.md` —— 真正开训，第一次看到 loss 下降",
    ),

    # ======================================================================
    D(
        week=3, n=15, slug="first_lora_sft", title="跑通第一次 LoRA SFT",
        where="cloud",
        files="`configs/sft_lora_3b.yaml`、`src/train/sft_peft.py`",
        prereq="Day 14；`data/processed/sft_train.jsonl` 已在数据盘；模型已下载",
        goal="把训练真正跑起来 —— 先用 `--dry-run` 验数据与显存，再正式开训，"
             "亲眼看到 loss 在 50 步内开始下降，checkpoint 落盘。",
        read=[
            "`configs/sft_lora_3b.yaml` 逐项 —— 每一项都对应 Day 13/14 学过的一个概念",
            "`src/train/sft_peft.py` 的 `MultimodalSFTDataset` 与 `MultimodalCollator`",
            "`docs/06-sft-training.md` 第 5 节（第一次训练的观测清单）",
        ],
        think=[
            "为什么要 `remove_unused_columns=False`？（多模态 batch 里有非标准字段，默认会被删掉）",
            "`gradient_checkpointing=True` 时**必须** `use_cache=False`，为什么？",
            "为什么只保存 LoRA adapter，而不是每次存全量权重？（3B 全量 ~6 GB vs adapter ~60 MB）",
        ],
        write_title="改配置 → dry-run → 开训（今天不写新代码，全是配置与观测）",
        write_rows=[
            ("`configs/sft_lora_3b.yaml`", "改 `max_pixels` / `grad_accum` / `num_epochs` 适配你的卡和预算"),
            ("`MultimodalSFTDataset`", "确认它把 assistant 段落标成了有效 label（其余 -100）"),
            ("`MultimodalCollator`", "确认 `pixel_values` / `image_grid_thw` 被正确拼批"),
            ("`--dry-run`", "不加载模型，只验数据格式 + 打印一个 batch 的形状"),
            ("产出目录", "`outputs/qwen25vl3b-cx-lora-v0/`（在**数据盘**上，别放系统盘）"),
        ],
        run=[
            ("python -m src.train.sft_peft --config configs/sft_lora_3b.yaml --dry-run",
             "先验数据，30 秒出结果"),
            ("python -m src.train.sft_peft --config configs/sft_lora_3b.yaml",
             "正式开训。另开一个终端盯显存"),
            ("watch -n 5 nvidia-smi", "（另开终端）盯显存和利用率"),
        ],
        expect="""
[data] train 1840 / eval 210 条
[sample] 图片 1 张 → 1369 visual tokens，最长序列 2048
[collator] batch 形状: input_ids (1,2048) labels (1,2048) pixel_values (1369,1176)
[lora] 可训练 30.4 M / 3043 M (1.00%)，注入 12 类模块
[dry-run] 一切正常。去掉 --dry-run 开始训练。
----------------------------------------------------------------
{'loss': 2.8412, 'grad_norm': 1.83, 'learning_rate': 1e-05, 'epoch': 0.01}
{'loss': 2.3977, 'grad_norm': 1.21, 'learning_rate': 1.98e-05, 'epoch': 0.02}
{'loss': 1.9015, 'grad_norm': 0.94, 'learning_rate': 2.95e-05, 'epoch': 0.03}
""",
        accept=[
            "`--dry-run` 通过，且打印的 visual token 数与 Day 4 算的一致",
            "正式训练启动，`loss` 在前 50 步内开始下降（不必降到很低，只要趋势向下）",
            "`outputs/qwen25vl3b-cx-lora-v0/` 下出现 `adapter_model.safetensors`",
            "能解释日志三条曲线的**正常形态**：loss 缓降、lr 先升后降（warmup+cosine）、grad_norm 平稳",
        ],
        pits=[
            "**chat template 错配**（头号杀手）—— 训练时手工拼字符串，推理时用 `apply_chat_template`，"
            "两边不一致 → 模型在推理时胡说。训练和推理必须走同一个模板函数。",
            "**label mask 错** —— 模型学会复述问题、或学会抢答 system。这是 Day 11 的坑在训练期的复现。",
            "`use_cache=True` + gradient checkpointing → 报 warning 甚至直接 OOM。",
            "显存不够时不要先动 batch，**先调小 `max_pixels`**（图片 token 是序列长度的主要贡献者）。",
            "checkpoint 存到了系统盘 —— 租的机器关机可能被清空。全部放 `/root/autodl-tmp/`。",
        ],
        nb=[
            ("md", "## 1. 先看配置，再看数据"),
            ("code", '''import yaml, json, itertools
from pathlib import Path

cfg = yaml.safe_load(Path("../configs/sft_lora_3b.yaml").read_text())
for k, v in cfg.items():
    print(f"{k:22s} {v}")'''),
            ("md", "## 2. 抽一个 batch 肉眼检查 label mask\n\n**这是今天最值钱的一步**：亲眼看 `-100` 落在了哪里。"),
            ("code", '''import json
from pathlib import Path

p = Path("../data/processed/sft_train.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()][:2]

for r in rows:
    print("=" * 70)
    print("图片:", r.get("images"))
    convo = r.get("conversations") or r.get("messages") or []
    for turn in convo:
        print(f"[{turn.get('from', turn.get('role'))}] {str(turn.get('value', turn.get('content')))[:120]}")'''),
            ("md", "## 3. dry-run（先别烧 GPU）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.sft_peft",
                    "--config", "configs/sft_lora_3b.yaml", "--dry-run"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-3000:] or r.stderr[-3000:])'''),
            ("md", """## 4. 正式开训

**在终端里跑，不要在 notebook 里跑** —— 训练要几个小时，notebook 断了就白跑。
```bash
cd /root/autodl-tmp/multimodal-lab
nohup python -m src.train.sft_peft --config configs/sft_lora_3b.yaml \\
      > outputs/train_w3.log 2>&1 &
tail -f outputs/train_w3.log
```
用 `nohup` + `tail -f`，这样 SSH 断了训练还在。"""),
        ],
        footer="> 训练跑起来后**别关机器**，但可以去读 `days/day-16.md` 的准备材料。\n"
               "> 今天烧的钱大约 ¥2–4，盯一眼 `nvidia-smi` 的利用率，低于 70% 说明数据加载是瓶颈。",
        next_day_hint="`days/day-16.md` —— 读日志、制造 bug、画三联图",
    ),

    # ======================================================================
    D(
        week=3, n=16, slug="training_log_debug", title="训练日志与踩坑",
        where="cloud",
        files="`src/train/monitor.py`",
        prereq="Day 15（至少跑过 ~200 步）",
        goal="把 trainer log 变成三联图（loss / lr / grad_norm），**并主动制造一个 bug 再修好** —— "
             "后者才是今天真正的产出。",
        read=[
            "`docs/06-sft-training.md` 第 5–7 节（loss 不降的 5 类原因 / 梯度爆炸 / OOM / template 错配）",
            "`src/train/monitor.py` 的 `diagnose()` —— 每类病因的判据都写在里面",
        ],
        think=[
            "loss 不降有 5 类原因，怎么用三条曲线把它们区分开？",
            "`grad_norm` 突然出现尖刺（比如从 0.9 跳到 50）说明什么？该做什么？",
            "`eval_loss` 开始抬头、`train_loss` 还在降 —— 这是 bug 还是正常现象？",
        ],
        write_title="`src/train/monitor.py`",
        write_rows=[
            ("`parse_trainer_log`", "从文本日志里正则抽取 loss / lr / grad_norm 三列"),
            ("`diagnose`", "五类病因判据：loss 平、loss 炸、grad_norm 尖刺、过拟合、lr 调度异常"),
            ("`plot`", "三联图输出 PNG"),
            ("你的 bug 实验", "**故意改坏一处**（推荐：把 chat template 换成错的），再诊断出来"),
        ],
        run=[
            ("python -m src/train/monitor outputs/qwen25vl3b-cx-lora-v0",
             "默认读取该目录下的 trainer log 并出图"),
            ("ls -la outputs/qwen25vl3b-cx-lora-v0/*.png", "确认三联图已生成"),
        ],
        expect="""
[parse] 从 trainer_state.json / train.log 解析到 380 个 step
[diagnose]
  loss      : 2.84 → 1.12   趋势正常（缓降，无平台期）
  lr        : warmup 20 步 → 峰值 3e-5 → cosine 衰减   正常
  grad_norm : 均值 0.98，最大 3.21，无尖刺             正常
  结论：这次训练没有明显异常。建议再跑 1 个 epoch 看 eval_loss 拐点。
[plot] 已保存 outputs/qwen25vl3b-cx-lora-v0/train_curves.png
""",
        accept=[
            "三联图已生成，并写了一段**解读**（不是贴图了事）",
            "**主动制造并修复了至少 1 个 bug**，能说清「症状 → 判据 → 根因 → 修法」",
            "能指出你这次训练的过拟合拐点大概在第几步（或说明为什么还没到）",
            "知道 `eval_loss` 抬头不一定是 bug —— 要先看 `train_loss` 是否还在降",
        ],
        pits=[
            "只盯 loss 不看 lr 调度 —— lr 调错时 loss 的表现和「数据有问题」一模一样。",
            "`grad_norm` 长期 > 1 却不处理 —— 加大 warmup 或降 lr，必要时开梯度裁剪。",
            "把 `eval_loss` 上升当成 bug —— 可能只是正常过拟合，先看 train/eval 的差值。",
            "日志没落盘就重启了机器 —— 训练一定重定向到文件（`nohup ... > log 2>&1 &`）。",
        ],
        nb=[
            ("md", "## 1. 解析日志并画三联图"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.monitor",
                    "outputs/qwen25vl3b-cx-lora-v0"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", """## 2. 反例：人为造一个坏日志，训练你的诊断直觉

下面直接构造三段「病态曲线」，让 `diagnose()` 判一遍。
你要做的是**先自己猜病因，再看脚本的判断**。"""),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.train.monitor import diagnose

cases = {
    "病态A": [{"step": i, "loss": 2.5 + 0.001 * i, "grad_norm": 0.9, "learning_rate": 3e-5}
              for i in range(100)],
    "病态B": [{"step": i, "loss": 2.5 if i < 10 else float("nan"),
               "grad_norm": 0.9, "learning_rate": 3e-5} for i in range(100)],
    "病态C": [{"step": i, "loss": 1.0 if i < 50 else 0.2,
               "grad_norm": 12.0 if i % 17 == 0 else 0.8,
               "learning_rate": 3e-5} for i in range(100)],
}
for name, log in cases.items():
    print("=" * 60)
    print(name)
    try:
        print(diagnose(log))
    except Exception as e:
        print("diagnose 需要不同的输入结构 →", e)'''),
            ("md", """## 3. 今日的 bug 实验（必做）

三种改坏方式，选一种真的改，跑 50 步，观察症状：

| 改坏方式 | 症状 |
|---|---|
| chat template 换成错的 | loss 降得下去，但**推理时胡说**（最阴的一种） |
| label mask 包含 user 段 | 模型学会复述问题，loss 虚低 |
| lr 调大 10 倍 | loss 抖动/爆炸，grad_norm 尖刺 |

把「症状 → 根因 → 修法」三行写进打卡 —— 这三行比三联图值钱。"""),
            ("code", '''bug_note = """
症状：
根因：
修法：
"""
print(bug_note)'''),
        ],
        next_day_hint="`days/day-17.md` —— DeepSpeed / FSDP 配置（本地就能做完）",
    ),

    # ======================================================================
    D(
        week=3, n=17, slug="deepspeed_fsdp", title="规模化训练：DeepSpeed / FSDP",
        where="local",
        files="`configs/deepspeed_zero2.json`、`configs/deepspeed_zero3.json`、`configs/README.md`",
        prereq="Day 16",
        goal="搞清 ZeRO-1/2/3 与 FSDP 的取舍，写出两份能直接用的 DeepSpeed 配置；"
             "并回答「为什么 VLM 训练里视觉塔通常冻结」。",
        read=[
            "`docs/06-sft-training.md` 第 8 节（ZeRO 三阶段、FSDP、多模态特有的不平衡问题）",
            "`configs/README.md` —— 两份配置的设计取舍都写在这里",
        ],
        think=[
            "ZeRO-2 切了梯度和优化器状态，ZeRO-3 连**权重**也切 —— 代价分别是什么？",
            "为什么单卡训练开 ZeRO-3 是纯亏？（通信开销 > 省下的显存）",
            "VLM 训练里视觉塔冻结的三个理由是什么？（答案在 docs/06 第 8 节）",
        ],
        write_title="两份 DeepSpeed 配置（今天不跑训练，是「写配置」日）",
        write_rows=[
            ("`configs/deepspeed_zero2.json`", "切梯度 + 优化器状态，单机多卡的主力选择"),
            ("`configs/deepspeed_zero3.json`", "连权重也切 —— 为 7B/32B 全参预留，本项目先不用"),
            ("`bf16` 段", "`\"bf16\": {\"enabled\": true}` 必须写明，否则默认可能走 fp16"),
            ("`zero_optimization`", "注意 `stage3_gather_16bit_weights_on_model_save` 这个坑"),
            ("`configs/README.md`", "把你的取舍理由写进去（面试时这一页很值钱）"),
        ],
        run=[
            ('python -c "import json; [json.load(open(f)) for f in '
             "['configs/deepspeed_zero2.json','configs/deepspeed_zero3.json']]; "
             'print(\'✓ 两份配置 JSON 合法\')"',
             "先验 JSON 合法（DeepSpeed 的解析很严格，多余逗号都会炸）"),
            ("deepspeed --num_gpus=1 src/train/sft_peft.py --config configs/sft_lora_3b.yaml "
             "--deepspeed configs/deepspeed_zero2.json",
             "单卡其实不需要 ZeRO，先确认「它能不能起来」就够了"),
        ],
        expect="""
✓ 两份配置 JSON 合法

（deepspeed 启动后节选）
[INFO] DeepSpeed Flops Profiler 未启用
[INFO] Using /root/.cache/torch_extensions as PyTorch extensions root
[INFO] ZeRO stage 2, offload=False, contiguous_gradients=True
[INFO] 可训练参数 30.4 M，优化器状态已分片
""",
        accept=[
            "两份配置 JSON 合法，且每项都有注释说明为什么这么写",
            "能说清 ZeRO-2 和 ZeRO-3 的取舍（省显存 vs 通信开销）",
            "能说出 VLM 训练里视觉塔冻结的理由，以及「冻结 ≠ 不占显存」",
            "知道 `stage3_gather_16bit_weights_on_model_save` 是 ZeRO-3 存模型时的必踩坑",
        ],
        pits=[
            "JSON 里写了注释 —— DeepSpeed 用严格 JSON 解析器，注释和尾逗号都会报错（注释只能写在 README 里）。",
            "ZeRO-3 + LoRA 忘了 `stage3_gather_16bit_weights_on_model_save: true` —— 存出来的权重加载不了。",
            "CPU offload 看着很香 —— 实际拖慢 2–3 倍，本项目在 24 GB 卡上完全不需要。",
            "单卡开 ZeRO-3 —— 纯亏，通信没有收益。",
        ],
        nb=[
            ("md", "## 1. 校验两份配置"),
            ("code", '''import json
from pathlib import Path

for name in ("deepspeed_zero2.json", "deepspeed_zero3.json"):
    p = Path("../configs") / name
    cfg = json.loads(p.read_text())
    print("=" * 60)
    print(name)
    print("  stage       :", cfg.get("zero_optimization", {}).get("stage"))
    print("  offload     :", cfg.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none"))
    print("  bf16        :", cfg.get("bf16", {}).get("enabled"))
    print("  gather_16bit:", cfg.get("zero_optimization", {}).get("stage3_gather_16bit_weights_on_model_save"))'''),
            ("md", """## 2. 显存账：切了之后各卡还剩下多少

ZeRO-2 把**梯度和优化器状态**按卡数摊分，ZeRO-3 连权重也摊。
手算一下：3 张卡开 ZeRO-3，单卡静态显存是多少？和 Day 13 的估算对一下。"""),
            ("code", '''P = 3.0e9
GB = 1024 ** 3
n_gpu = 3

for stage, w_share, g_share, o_share in [
    ("ZeRO-1", 1.0, 1.0, 1 / n_gpu),
    ("ZeRO-2", 1.0, 1 / n_gpu, 1 / n_gpu),
    ("ZeRO-3", 1 / n_gpu, 1 / n_gpu, 1 / n_gpu),
]:
    w = P * 2 / GB * w_share        # bf16 全参权重
    g = P * 2 / GB * g_share
    o = P * 8 / GB * o_share
    print(f"{stage}: 权重 {w:6.1f} + 梯度 {g:6.1f} + 优化器 {o:6.1f} = {w+g+o:6.1f} GB/卡（3 卡）")'''),
            ("md", """## 3. 落笔：为什么视觉塔冻结

把三个理由写下来（提示：参数量占比 / 预训练已充分 / 梯度不稳定对底层特征的破坏）。"""),
            ("code", '''why_freeze = """
1.
2.
3.
"""
print(why_freeze)'''),
        ],
        next_day_hint="`days/day-18.md` —— 合并 LoRA、20 条并排抽检，W3 收官",
    ),

    # ======================================================================
    D(
        week=3, n=18, slug="first_training_wrap", title="第一次训练收官",
        where="cloud",
        files="`src/eval/quick_eval.py`（本周新增）、LoRA 合并脚本",
        prereq="Day 15–17",
        goal="合并 LoRA → 导出可独立加载的权重，然后做 **20 条 base vs 你的版本的人工并排抽检** —— "
             "第一次能回答「我训的模型到底有没有变好」。",
        read=[
            "`docs/06-sft-training.md` 第 9 节（合并与导出）",
            "`src/eval/quick_eval.py` —— 今天新增的小工具，先读它怎么挑样本",
        ],
        think=[
            "为什么抽检样本必须覆盖 L1–L4 四个难度层？（只挑简单的会自我感觉良好）",
            "`merge_and_unload()` 之后，模型的参数量和显存占用变成多少？",
            "如果 20 条里变好的只有「语气更客气」，这算不算成功？",
        ],
        write_title="合并 + 抽检（工具已给，你要做的是**判断**）",
        write_rows=[
            ("`PeftModel.from_pretrained(...)`", "把 adapter 挂到 base 上"),
            ("`merge_and_unload()`", "合并成一份可直接加载的权重"),
            ("`torch_dtype=torch.bfloat16`", "别忘了指定精度，否则默认 fp32 显存翻倍"),
            ("`quick_eval.py`", "按 4 个难度层分层抽样 N 条，两条并排输出 markdown"),
            ("你的判断", "逐条标注「变好 / 变差 / 没变」，并写下理由"),
        ],
        run=[
            ("python -m src.eval.quick_eval --base Qwen/Qwen2.5-VL-3B-Instruct "
             "--adapter outputs/qwen25vl3b-cx-lora-v0 --n 20 --out reports/quick_eval_w3.md",
             "生成 20 条并排对比"),
            ("head -60 reports/quick_eval_w3.md", "打开看结果"),
        ],
        expect="""
[load] base = Qwen/Qwen2.5-VL-3B-Instruct
[load] adapter = outputs/qwen25vl3b-cx-lora-v0
[merge] 合并完成，参数量 3.75 B，dtype bfloat16
[sample] 分层抽样 20 条：L1 6 / L2 7 / L3 5 / L4 2
[run] 20/20 完成，耗时 143s，峰值显存 8.9 GB
[out] reports/quick_eval_w3.md

  | # | 难度 | 问题 | base | 你的版本 | 你的判断 |
  |---|---|---|---|---|---|
  | 1 | L1 | 这件有货吗？ | 有的哦~ | 您好，M/L 码现货，S 码 3 天内补 | ? |
  ...
""",
        accept=[
            "合并后的权重能用**一个独立的 `from_pretrained` 直接加载**（不挂 adapter）",
            "20 条并排对比已生成，且**你逐条写了判断**",
            "能指出至少 **3 个变好的 case** 和 **2 个变差的 case**，并给出原因假设",
            "`progress/weekly-review.md` 的 W3 段已写；进度表 W3 六天 `[x]`，M3 打卡",
        ],
        pits=[
            "忘了 `merge_and_unload()` —— 只保存 adapter 就以为「训练完了」，换个加载方式就报错。",
            "不指定 dtype —— 默认 fp32，3B 模型显存直接翻倍到 12 GB 以上。",
            "抽检全是容易的样本 —— 结论会过于乐观。**必须分层抽**。",
            "只看「变好的」不看「变差的」—— 变差的 case 才是下一轮 DPO 的素材（Day 26 要用）。",
        ],
        nb=[
            ("md", "## 1. 合并 LoRA 并导出"),
            ("code", '''import sys; sys.path.insert(0, "..")
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "outputs/qwen25vl3b-cx-lora-v0"
OUT = "outputs/qwen25vl3b-cx-merged-v0"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()          # ★ 关键一步
model.save_pretrained(OUT)
AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
print("已导出 →", OUT)'''),
            ("md", "## 2. 生成 20 条并排对比"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.quick_eval",
                    "--base", "Qwen/Qwen2.5-VL-3B-Instruct",
                    "--adapter", "outputs/qwen25vl3b-cx-lora-v0",
                    "--n", "20", "--out", "reports/quick_eval_w3.md"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-2000:] or r.stderr[-2000:])'''),
            ("md", """## 3. 逐条打分表格（今天的核心产出）

把这 20 条复制到下面，逐条写判断。**建议直接看 `reports/quick_eval_w3.md` 的表格**，
在这里只填你的结论汇总。"""),
            ("code", '''verdict = {
    "变好": [],   # 例: ["L1-有货吗-答出了具体码数", ...]
    "变差": [],
    "没变": [],
}
for k, v in verdict.items():
    print(f"{k}: {len(v)} 条")

hypothesis = """
变好的共性：
变差的共性：
下一轮要补的数据：
"""
print(hypothesis)'''),
        ],
        footer="""> **M3 达成条件**：有一份合并后的权重 + 20 条带人工判断的对比 + W3 周复盘。\n"""
                 "> 没达成也别急 —— Day 28 的止损线允许你把 W3 的目标降为「跑通流程」。",
        next_day_hint="`days/day-19.md` —— W4 评测周，今天起可以回本地做（省钱）",
    ),
]
