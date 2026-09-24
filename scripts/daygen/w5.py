"""Week 5 · 偏好对齐与推理优化（Day 25–30）

这一周的核心矛盾：**让模型不只「答对」，还要「答得像人、答得安全、答得快」**。
省钱提示：Day 26 / 28 在本地就能做完（`dpo_loss.py` 的造数据路径已做惰性 torch 导入）。
"""
from __future__ import annotations


def D(**kw):
    return kw


DAYS = [
    # ======================================================================
    D(
        week=5, n=25, slug="dpo_family", title="DPO 家族原理",
        where="cloud",
        files="`src/train/dpo_loss.py`、`docs/07-alignment.md`",
        prereq="Day 24（评测报告 v1）",
        goal="手写 DPO loss（含 reference 项）并用玩具数据验证数值 —— "
             "重点是跑通那个 **`ln 2` 检查**：chosen 与 rejected 完全相同时 loss 必须等于 0.6931。",
        read=[
            "`docs/07-alignment.md` 第 1–3 节（RLHF → DPO → ORPO → KTO → GRPO 的关系图）",
            "DPO 原论文的公式推导部分（`docs/12-papers.md` 标了重点）",
        ],
        think=[
            "DPO 的闭式解是怎么推出来的？为什么可以不需要 reward model？",
            "β 控制什么？β=0.01 和 β=1.0 分别会发生什么？",
            "为什么 `log(π_θ/π_ref)` 里的 π_ref 是「锚」—— 去掉它会发生什么？",
        ],
        write_title="`src/train/dpo_loss.py` Part A（已给实现）",
        write_rows=[
            ("`get_batch_logps()`", "只要被监督部分的 log 概率和；注意别把 prompt 也算进去"),
            ("`dpo_loss()`", "四项 logps 进，loss + 指标出（含 `reward_margin`）"),
            ("`label_smoothing`", "cDPO：对偏好标签做平滑，抗噪声标注"),
            ("`verify_dpo_loss()`", "**4 个场景 + ln2 检查** —— 今天必须全部通过"),
            ("`_ensure_torch()`", "惰性导入：让 Part B 的造数据路径在本地也能跑"),
        ],
        run=[
            ("python -m src.train.dpo_loss --verify", "数值验证（要 torch，不需要 GPU）"),
        ],
        expect="""
DPO Loss 数值验证
==============================================================================
[1] chosen 明显优于 rejected
    loss = 0.0472   accuracy = 1.00   margin = 3.20     ← 应该是很小的 loss

[2] chosen 与 rejected 完全相同  ← 关键检查
    loss = 0.6931   ← 必须等于 ln 2 = 0.693147
    accuracy = 0.00

[3] chosen 略优于 rejected
    loss = 0.5128   accuracy = 1.00

[4] label_smoothing=0.1（cDPO）
    loss = 0.5823   ← 比场景 1 大：平滑让 loss 不会压到 0
==============================================================================
✓ 4 个场景全部通过
""",
        accept=[
            "`--verify` 四个场景断言全部通过，**特别是 ln2 = 0.6931 那一项**",
            "能自己推出 DPO 的闭式解（不用翻论文）",
            "能说清 β 的作用，以及为什么典型值是 0.1–0.5",
            "知道 ORPO / SimPO / KTO 分别省掉了什么（reference / 配对假设 / 正负样本）",
        ],
        pits=[
            "**忘了 reference 要 detach** —— π_ref 有梯度的话，训练目标就变了，怎么调都不对。",
            "**logps 把 prompt 部分也算进去了** —— 那算的是「整段对话的概率」而不是「回答的概率」，"
            "DPO 立刻失效。label mask 必须只覆盖 assistant 段（Day 11 的老坑换了个地方）。",
            "β 设得太大 —— 模型被锚死在 reference 附近，学不动；太小则语言能力崩坏。",
            "reference model 用了和 policy 同一个实例 —— 等价于没除，DPO 退化成毫无意义的损失。",
        ],
        nb=[
            ("md", "## 1. 手工推一遍公式，再跑验证"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.dpo_loss", "--verify"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", """## 2. 自己实现一遍 DPO loss（不看源码）

只有三行：
```python
chosen_r  = beta * (pi_chosen  - ref_chosen)
rejected_r = beta * (pi_rejected - ref_rejected)
loss = -F.logsigmoid(chosen_r - rejected_r).mean()
```"""),
            ("code", '''import math

def my_dpo(pc, pr, rc, rr, beta=0.1):
    cr = beta * (pc - rc)
    rj = beta * (pr - rr)
    d = cr - rj
    return -math.log(1 / (1 + math.exp(-d)))     # -log sigmoid(d)

# 场景 1：chosen 明显更好
print("chosen 好 →", round(my_dpo(-1.0, -2.0, -1.0, -2.0), 4))
# 场景 2：完全相同 → 应该正好是 ln2
print("完全一样 →", round(my_dpo(-1.5, -1.5, -1.0, -1.0), 4), " (ln2 =", round(math.log(2), 4), ")")
# 思考：β 变大 10 倍，场景 1 的 loss 怎么变？为什么？
for beta in (0.01, 0.1, 1.0):
    print(f"  beta={beta:<5} loss={my_dpo(-1.0, -2.0, -1.0, -2.0, beta):.6f}")'''),
            ("md", "## 3. 落笔：β 的直觉\n\nβ 大 → 保守（贴近 reference）／β 小 → 激进（易崩）。写下你的理解和一个具体的调参方案。"),
            ("code", '''beta_note = """
β 的物理意义：
客服场景我选 β = ___，理由是：
"""
print(beta_note)'''),
        ],
        next_day_hint="`days/day-26.md` —— 造偏好数据（本地可跑，不用开 GPU）",
    ),

    # ======================================================================
    D(
        week=5, n=26, slug="preference_data", title="多模态偏好数据构造",
        where="local",
        files="`src/train/dpo_loss.py` Part B",
        prereq="Day 23 的 `reports/bad_cases.jsonl`；Day 25",
        goal="从 W4 的 bad case 反向构造 (chosen, rejected) 对，凑够 3k 对；"
             "并亲手构造**同图不同答**和**同答不同图**两种多模态特有的偏好对。",
        read=[
            "`docs/07-alignment.md` 第 4 节（多模态偏好数据的三种类型）",
            "`src/train/dpo_loss.py` 的 `build_preference_from_badcases` / `build_contrastive_pairs`",
        ],
        think=[
            "为什么「模型真实犯过的错」比「人造的错误回答」更适合做 rejected？",
            "**同图不同答**治的是什么病？**同答不同图**又治的是什么病？",
            "如果 chosen 和 rejected 只差语气（一个礼貌一个冷淡），模型能学到东西吗？",
        ],
        write_title="`src/train/dpo_loss.py` Part B（已给实现，你要扩来源）",
        write_rows=[
            ("`build_preference_from_badcases()`", "字段契约：`query` / `model_answer` / `reference`（由 error_analysis 输出）"),
            ("`build_contrastive_pairs()`", "同图不同答 / 同答不同图 —— **多模态版的幻觉解药**"),
            ("抽检脚本", "随机抽 30 对，判断「rejected 是否真的更差，且差异可学习」"),
            ("去重", "同一 (图, 问) 不要出现多对，否则某一类被过度加权"),
            ("配比", "幻觉类 / 漏信息类 / 语气类 大致 4:4:2"),
        ],
        run=[
            ("python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl",
             "从 W4 的 bad case 造（主力来源）"),
            ("python -m src.train.dpo_loss --contrastive data/processed/clean.jsonl --n-pairs 1000",
             "造对比式偏好对（幻觉解药）"),
            ("wc -l data/processed/dpo_train.jsonl data/processed/dpo_contrastive.jsonl",
             "看两个文件的量"),
        ],
        expect="""
读入 71 条 bad case
⚠️  跳过 4 条：缺 model_answer 或 reference 字段
✓ 写出 67 对偏好数据 → data/processed/dpo_train.jsonl

从 1840 条样本构造对比式偏好对
  同图不同答： 600 对
  同答不同图： 400 对
✓ 写出 1000 对 → data/processed/dpo_contrastive.jsonl

data/processed/dpo_train.jsonl         67 行
data/processed/dpo_contrastive.jsonl  1000 行
""",
        accept=[
            "`dpo_train.jsonl` 有实打实的偏好对（不是 0 对）",
            "抽检 30 对，逐对判断「rejected 确实更差，且差异是**可学习的**」",
            "三种类型的偏好对都有：同图不同答 / 同答不同图 / 格式或工具调用",
            "能说清哪一类偏好对治的是「幻觉」这个具体病",
        ],
        pits=[
            "**silent 0** —— 字段名不对时脚本会写出 0 对。现在会告警，但你要认得这个信号"
            "（正确字段是 `query` / `model_answer` / `reference`）。",
            "chosen 与 rejected 只差语气 —— 这类对占太多，模型只学会「更啰嗦」而没学会「更准确」。",
            "**同图不同答时图没对齐** —— 两张图张冠李戴，模型学到的是噪声。构造时必须校验图 ID。",
            "rejected 全部来自幻觉类 —— 偏好数据分布单一，DPO 会过度惩罚「不确定」的表达。",
            "不抽检直接开训 —— DPO 对数据质量极其敏感，垃圾进，崩坏出（比 SFT 更脆）。",
        ],
        nb=[
            ("md", "## 1. 从 bad case 造偏好对"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.dpo_loss",
                    "--from-badcases", "reports/bad_cases.jsonl"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 造对比式偏好对（多模态特有两种）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.dpo_loss",
                    "--contrastive", "data/processed/clean.jsonl", "--n-pairs", "1000"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-1500:] or r.stderr[-1500:])'''),
            ("md", """## 3. 抽 5 对肉眼检查「差异是可学习的吗」

**这是今天最关键的判断**。差异太小 → 学不到；差异太大（一个全错一个全对）→ 也没用，
模型会去学那些显而易见的差异而不是你想要的细节。"""),
            ("code", '''import json, random
from pathlib import Path

for name in ("dpo_train", "dpo_contrastive"):
    p = Path(f"../data/processed/{name}.jsonl")
    if not p.exists():
        print(f"{name}: 还没生成"); continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print("=" * 72)
    print(f"{name}  共 {len(rows)} 对")
    for r in random.sample(rows, min(2, len(rows))):
        print("-" * 72)
        print("图  :", r.get("images"))
        print("问  :", str(r.get("prompt"))[:80])
        print("✓chosen  :", str(r.get("chosen"))[:110])
        print("✗rejected:", str(r.get("rejected"))[:110])
        print("来源:", r.get("source"), "| 类型:", r.get("error_type", r.get("pair_type", "-")))'''),
            ("md", """## 4. 落笔：三行判断

- 哪些对「差异太小，学不到」？
- 哪些对「差异太大，学到的是废话」？
- 你想补哪一类？"""),
            ("code", '''judgement = """
差异太小的例子：
差异太大的例子：
我打算补的数据：
"""
print(judgement)'''),
        ],
        next_day_hint="`days/day-27.md` —— 真正跑 DPO（回云上）",
    ),

    # ======================================================================
    D(
        week=5, n=27, slug="run_dpo", title="跑 DPO",
        where="cloud",
        files="`src/train/dpo.py`、`configs/dpo_3b.yaml`",
        prereq="Day 26（偏好数据就位）；W3 训练出的 SFT adapter",
        goal="从 **SFT 版本**（不是基座）出发跑 DPO，看到 `rewards/chosen − rewards/rejected` "
             "的差值稳步上升，并在领域集上测出与 SFT 版本的差异。",
        read=[
            "`configs/dpo_3b.yaml` 逐项 —— 特别注意 `beta` 和 `learning_rate`",
            "`src/train/dpo.py` 的 `--inspect` 路径（不用 torch 就能看数据）",
            "`docs/07-alignment.md` 第 5 节（DPO 训练的观测指标）",
        ],
        think=[
            "为什么 DPO 的学习率要比 SFT 小一个量级？",
            "为什么必须从 SFT 版本出发，而不是从基座直接 DPO？",
            "`reward_margin` 一直涨是好事吗？什么时候该停？",
        ],
        write_title="`src/train/dpo.py`（已给实现）",
        write_rows=[
            ("`--inspect`", "先看数据格式与 chosen/rejected 长度分布（**本地就能跑**）"),
            ("`--dry-run`", "验数据 + 显存，不加载模型"),
            ("`--sft-adapter`", "指定 SFT 的 LoRA 作为起点 —— 这是关键参数"),
            ("`beta`", "默认 0.1；数据噪声大时调大到 0.3 更稳"),
            ("`--no-qlora`", "显存够时关掉 4-bit（DPO 对量化更敏感）"),
        ],
        run=[
            ("python -m src.train.dpo --inspect --in data/processed/dpo_train.jsonl",
             "先看数据（本地可跑）"),
            ("python -m src.train.dpo --config configs/dpo_3b.yaml "
             "--in data/processed/dpo_train.jsonl --in data/processed/dpo_contrastive.jsonl --dry-run",
             "验数据与显存"),
            ("python -m src.train.dpo --config configs/dpo_3b.yaml "
             "--in data/processed/dpo_train.jsonl --in data/processed/dpo_contrastive.jsonl "
             "--sft-adapter outputs/qwen25vl3b-cx-lora-v0",
             "正式开训"),
        ],
        expect="""
[inspect] 1067 对偏好数据
  chosen   长度 p50/p95 = 82 / 210
  rejected 长度 p50/p95 = 31 /  88      ← rejected 明显更短，警惕「长度偏见」
  来源分布：bad_case 67 / contrastive 1000
  类型分布：同图不同答 600 / 同答不同图 400 / 错误回答 67

（训练中）
{'loss': 0.6102, 'rewards/chosen': 0.041, 'rewards/rejected': -0.038,
 'rewards/margins': 0.079, 'rewards/accuracies': 0.71}
{'loss': 0.4318, 'rewards/chosen': 0.118, 'rewards/rejected': -0.142,
 'rewards/margins': 0.260, 'rewards/accuracies': 0.86}
""",
        accept=[
            "`--inspect` 显示 chosen/rejected 的长度分布（**必须检查长度偏见**）",
            "训练启动，`rewards/margins` 稳步上升、`rewards/accuracies` 趋向 1",
            "在 `cx_eval_v1` 上跑出 DPO 版本 vs SFT 版本的对照（哪怕差异很小）",
            "能说出「这次 DPO 有没有训过头」（依据是 margin 曲线还是别的？）",
        ],
        pits=[
            "**从基座直接 DPO** —— 基座连格式都不会，DPO 学不到东西。必须先 SFT。",
            "学习率沿用 SFT 的 2e-5 —— 太大，DPO 会崩。典型 5e-7 ~ 5e-6。",
            "**长度偏见** —— rejected 系统性短于 chosen 时，模型可能只学会「说长一点」。"
            "检查 p50/p95，必要时用 `average_log_prob` 归一化（SimPO 思路）。",
            "训太久 —— margin 涨到某个点后模型开始钻空子（reward hacking），"
            "表现为通用能力下降。用领域评测集守着，宁可早停。",
            "忘了 `--sft-adapter` —— 起点错了，白训。",
        ],
        nb=[
            ("md", "## 1. 先 inspect 数据（不用 GPU）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.dpo",
                    "--inspect", "--in", "data/processed/dpo_train.jsonl"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 长度偏见自检\n\n**这是 DPO 最隐蔽的坑**：如果 rejected 一律比 chosen 短，模型可能只学到「变啰嗦」。"),
            ("code", '''import json, statistics
from pathlib import Path

p = Path("../data/processed/dpo_train.jsonl")
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    cl = [len(r["chosen"]) for r in rows]
    rl = [len(r["rejected"]) for r in rows]
    print(f"chosen   中位长度 {statistics.median(cl):6.1f}")
    print(f"rejected 中位长度 {statistics.median(rl):6.1f}")
    ratio = statistics.median(cl) / max(statistics.median(rl), 1)
    print(f"比值 {ratio:.2f}")
    if ratio > 1.8:
        print("⚠️  chosen 明显更长 —— 模型可能只学会『说长一点』，考虑 average_log_prob 归一化")
    else:
        print("✓ 长度分布还算平衡")
else:
    print("先跑 Day 26 造数据")'''),
            ("md", "## 3. 训练完看这两条曲线"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.monitor",
                    "outputs/qwen25vl3b-cx-dpo-v0"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
        ],
        next_day_hint="`days/day-28.md` —— 可验证奖励（本地可跑，进阶可选）",
    ),

    # ======================================================================
    D(
        week=5, n=28, slug="verifiable_rewards", title="可验证奖励与 GRPO（进阶，可跳过）",
        where="local",
        files="`src/train/rewards.py`",
        prereq="Day 27",
        goal="实现三个**可验证**奖励（JSON 格式可解析 / 工具调用正确 / 引用图片区域正确），"
             "跑通单元测试；并回答「客服场景下为什么可验证奖励比人偏好更划算」。",
        read=[
            "`docs/07-alignment.md` 第 5–6 节（可验证奖励 / GRPO / RLVR）",
            "`src/train/rewards.py` 的 `RewardWeights` —— 权重怎么配是今天的核心问题",
        ],
        think=[
            "「格式合规」可以被程序验证，「语气友好」不能 —— 这两类奖励该怎么配权重？",
            "为什么说可验证奖励是**免费的数据标注**？",
            "奖励函数写错了会怎样？（比数据错更可怕，因为模型会全力钻空子）",
        ],
        write_title="`src/train/rewards.py`（已给实现）",
        write_rows=[
            ("`reward_format()`", "JSON 能否解析 + 字段齐全 + 值域合法"),
            ("`reward_tool_call()`", "工具名是否在白名单 + 参数是否齐全 + **是否重复调用**"),
            ("`reward_slot_filling()`", "订单号 / 尺码 / 颜色 等槽位是否被正确提取"),
            ("`reward_grounding()`", "回答里的视觉描述是否与图对上（引用区域）"),
            ("`reward_hallucination_penalty()`", "出现图里没有的物体 → 扣分"),
            ("`reward_refusal()`", "该拒答时拒答（而非瞎编）"),
        ],
        write_note="> 奖励函数是**代码**，不是数据。它的 bug 会被模型以最激进的方式利用 —— \n"
                   "> 所以每一个奖励都要有单元测试，且要专门写「对抗样本」测试它。",
        run=[
            ("python -m src.train.rewards", "跑全部奖励函数的单元测试（本地，无 torch）"),
        ],
        expect="""
奖励函数单元测试
==============================================================================
[格式] 合法 JSON            → 1.000   ✓
[格式] 缺字段               → 0.300   ✓
[格式] 不是 JSON            → 0.000   ✓
[工具] 正确调用查订单        → 1.000   ✓
[工具] 编造工具名 get_weather → 0.000   ✓   ← 白名单拦住了
[工具] 漏参数               → 0.400   ✓
[槽位] 订单号提取正确        → 1.000   ✓
[定位] 描述的物体图里有       → 1.000   ✓
[定位] 描述的物体图里没有     → 0.000   ✓   ← 幻觉惩罚
[拒答] 该拒答时拒答          → 1.000   ✓
==============================================================================
✓ 全部通过（含 3 个对抗样本）
""",
        accept=[
            "奖励函数单元测试全部通过，**且包含对抗样本测试**",
            "能说清「客服场景下为什么可验证奖励比人偏好更划算」（三个理由）",
            "能说出奖励配比不当会怎样（比如格式奖励权重过高 → 只输出格式不解决问题）",
            "知道 GRPO 和 DPO 的最根本差别（要不要 reward model / 要不要配对数据）",
        ],
        pits=[
            "**格式奖励权重过高** —— 模型学会输出完美 JSON 但内容空洞。这是最经典的 reward hacking。",
            "工具调用奖励不校验参数 —— 模型学会「调用工具但不填参数」，形式上通过。",
            "幻觉惩罚误伤诚实的「不确定」 —— 模型学会硬编一个答案，反而不说不知道了。",
            "奖励函数没有对抗测试 —— 你自己想不出钻空子的方式，模型能。",
            "跳过这一天 —— **这是允许的**（止损线第 1 条）。但「可验证奖励」的思想必须理解，"
            "它是 Day 31–36 工具调用的理论基础。",
        ],
        nb=[
            ("md", "## 1. 跑单元测试"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.train.rewards"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 写一个「对抗样本」：你能想到的钻空子方式\n\n**这一步比读懂奖励函数更重要。**"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.train.rewards import reward_format, reward_tool_call, RewardContext

# 对抗样本 1：完美 JSON，但内容是空洞的
hack1 = '{"intent": "咨询", "reply": "好的", "slots": {}}'
print("空洞但格式完美 →", reward_format(hack1))

# 对抗样本 2：调用了正确工具但参数是空的
ctx = RewardContext(available_tools=["lookup_order", "check_stock"])
print("调用对但没填参数 →", reward_tool_call("lookup_order", {}, ctx))

# 对抗样本 3：编一个很像的工具名
print("编造工具名     →", reward_tool_call("get_order_info", {"order_id": "A1"}, ctx))'''),
            ("md", """## 3. 落笔：为什么可验证奖励更划算

三个理由（提示：**成本**、**规模**、**可复现性**）。"""),
            ("code", '''why_verifiable = """
1. 成本：
2. 规模：
3. 可复现性：
"""
print(why_verifiable)'''),
        ],
        footer="> **可跳过**：如果进度落后，直接看讲义 + 跑一遍单元测试就够了，不必深挖 GRPO。",
        next_day_hint="`days/day-29.md` —— 量化与推理加速",
    ),

    # ======================================================================
    D(
        week=5, n=29, slug="quantize_accelerate", title="量化与推理加速",
        where="cloud",
        files="`src/serve/quantize.py`、`src/serve/vllm_server.sh`",
        prereq="Day 27（DPO 版本已产出）",
        goal="对 SFT/DPO 后的模型做 AWQ 量化，量出**显存 / 首 token 延迟 / 吞吐**三项指标的前后对比；"
             "并解释为什么量化对 VLM 的**视觉塔**尤其敏感。",
        read=[
            "`docs/09-inference.md` 第 1–4 节（AWQ / GPTQ / FP8 / KV cache / continuous batching）",
            "`src/serve/quantize.py` 的 `VISION_EXCLUDE` —— 这是今天的核心知识点",
        ],
        think=[
            "为什么 `VISION_EXCLUDE = [\"visual\", \"vision_tower\", \"merger\"]`？量化了会怎样？",
            "calibration 数据如果全是纯文本，量化后的模型看图会出什么问题？",
            "AWQ 和 GPTQ 怎么选？（提示：谁对激活值更敏感）",
        ],
        write_title="`src/serve/quantize.py`（已给实现）",
        write_rows=[
            ("`VISION_EXCLUDE`", "**视觉塔和 merger 排除在量化之外** —— 混合精度量化"),
            ("`estimate_savings()`", "只算账不真跑，先看值不值得"),
            ("`build_quant_config()`", "生成 AWQ 配置（含 exclude 列表）"),
            ("`benchmark()`", "三项指标：显存 / 首 token 延迟 / tok/s"),
            ("`src/serve/vllm_server.sh`", "起服务时的坑：`--limit-mm-per-prompt`、别开 128k max-len"),
        ],
        run=[
            ("python -m src.serve.quantize --model outputs/qwen25vl3b-cx-dpo-v0 --estimate",
             "先只算账（快）"),
            ("python -m src.serve.quantize --model outputs/qwen25vl3b-cx-dpo-v0 "
             "--method awq --out outputs/quantized/awq",
             "真跑 AWQ 量化"),
            ("python -m src.serve.quantize --model outputs/quantized/awq --benchmark --n 20 --concurrency 4",
             "跑基准测试"),
            ("bash src/serve/vllm_server.sh", "起 vLLM 服务"),
        ],
        expect="""
[estimate] 原始显存 ≈ 6.4 GB（bf16）→ 量化后 ≈ 2.1 GB，省 67%
           视觉塔保持 bf16，约 0.6 GB 不参与量化
[quantize] calib 数据 128 条图文样本（**必须含图**）
[benchmark]
                      显存      首 token    吞吐
  bf16              6.4 GB     0.42 s    38 tok/s
  awq (混合精度)     2.1 GB     0.31 s    61 tok/s
  awq (全量化, 反例)  1.6 GB     0.29 s    64 tok/s   ← 但看图能力崩了

⚠️  质量警告：视觉塔被量化后，OCR / 细粒度识别准确率下降 20%+
""",
        accept=[
            "三项指标（显存 / 首 token 延迟 / 吞吐）的前后对比表已产出",
            "能解释为什么视觉塔要保持高精度（激活值分布 + 细粒度信息）",
            "calibration 数据里**必须含图** —— 且你能说出为什么",
            "vLLM 服务能用 `curl` 打通，知道 `--limit-mm-per-prompt` 是干嘛的",
        ],
        pits=[
            "**把 vision tower 一起量化了** —— 图片识别能力直接崩，而且你在文本测试里发现不了。"
            "必须专门跑一次图文评测。",
            "calibration 数据全是纯文本 —— 量化参数完全没见到图像激活分布，视觉通路误差被放大。",
            "忘了 `--limit-mm-per-prompt` —— 有人传 10 张图进来，显存瞬间爆掉。",
            "沿用 128k 的 max-len —— KV cache 把显存吃光。客服场景 4k 足够。",
            "只测吞吐不测质量 —— 量化后的模型必须在 `cx_eval_v1` 上再跑一遍，"
            "分数掉超过 3 分就要考虑放弃量化。",
        ],
        nb=[
            ("md", "## 1. 先算账"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.serve.quantize",
                    "--model", "outputs/qwen25vl3b-cx-dpo-v0", "--estimate"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 看排除列表 —— 今天最该记住的 6 行代码"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.serve.quantize import VISION_EXCLUDE, build_quant_config, print_quality_warning

print("不参与量化的模块前缀:")
for name in VISION_EXCLUDE:
    print("   ", name)
print("\\n量化配置:")
try:
    cfg = build_quant_config(exclude=VISION_EXCLUDE)
    print(cfg)
except Exception as e:
    print("（需要 GPU 环境）", e)
print()
print_quality_warning()'''),
            ("md", """## 3. 反例实验设计

设计一个能**证明视觉塔敏感**的小实验：

1. 准备 20 张带文字的图 + 对应 OCR 问题
2. 分别用「混合精度」和「全量化」两个模型跑
3. 对比准确率

**这个实验做出来，你就真的懂了为什么 VLM 量化要区别对待。**"""),
            ("code", '''experiment_plan = """
对照组：awq 混合精度（视觉塔 bf16）
实验组：awq 全量化
样本：20 张带文字的图
指标：OCR 准确率
预期：实验组下降 ___ 个百分点
"""
print(experiment_plan)'''),
        ],
        next_day_hint="`days/day-30.md` —— 端到端推理服务，W5 收官",
    ),

    # ======================================================================
    D(
        week=5, n=30, slug="inference_service", title="端到端推理服务",
        where="cloud",
        files="`src/serve/api.py`",
        prereq="Day 29（vLLM 已能起来）",
        goal="写一个 FastAPI 网关：收 base64/URL 图片 + 文本 → 转发 vLLM → 带超时、重试、限流、"
             "结构化日志；并发 10 请求不炸，且日志能追溯到单次耗时。",
        read=[
            "`docs/09-inference.md` 第 5–6 节（服务化与成本）",
            "`src/serve/api.py` 的 `validate_request` 与 `filter_output`",
        ],
        think=[
            "为什么网关层要做图片校验？（提示：有人会传 20 MB 的图）",
            "超时该设多少？设置依据是什么？（提示：P95 延迟 + 余量）",
            "日志里为什么绝对不能记原图 base64？",
        ],
        write_title="`src/serve/api.py`（已给实现，你要改限流与超时）",
        write_rows=[
            ("`validate_request()`", "图片数量 / 大小 / 格式校验 —— 第一道闸门"),
            ("`VLLMClient`", "转发到 vLLM，带超时与重试"),
            ("`filter_output()`", "输出过滤：PII 掩码（手机号 / 订单号）"),
            ("结构化日志", "request_id / 耗时 / token 数 / 是否降级 —— **可追溯单次请求**"),
            ("`create_app()`", "FastAPI 工厂；uvicorn 要用 `--factory`"),
        ],
        run=[
            ("python -m src.serve.api --port 8080", "起网关"),
            ("curl -s localhost:8080/health", "健康检查"),
            ("curl -s localhost:8080/v1/chat -H 'content-type: application/json' "
             "-d '{\"messages\":[{\"role\":\"user\",\"content\":\"你好，这件有货吗\"}],\"images\":[]}'",
             "打通 /v1/chat"),
            ("python scripts/loadtest.py --url http://localhost:8080/v1/chat --n 10 --concurrency 10",
             "并发 10 压一把（loadtest 在 W8 完善）"),
        ],
        expect="""
INFO  starting gateway on 0.0.0.0:8080  vllm=http://localhost:8000
{"request_id":"a3f2","event":"request","n_images":0,"text_len":9}
{"request_id":"a3f2","event":"response","latency_ms":412,"out_tokens":58,"degraded":false}

$ curl -s localhost:8080/health
{"status":"ok","vllm":"ok","model":"qwen25vl3b-cx-dpo-v0"}

$ curl ... /v1/chat
{"reply":"您好，M/L 码有现货，S 码预计 3 天补货。","request_id":"a3f2"}
""",
        accept=[
            "`curl` 能打通 `/health` 和 `/v1/chat`",
            "并发 10 请求不崩、不超时，且日志能查到每一单的耗时",
            "图片校验生效（传一张 20 MB 的图会被友好拒绝，不是崩）",
            "`progress/weekly-review.md` 的 W5 段已写；进度表 W5 六天 `[x]`，M5 打卡",
        ],
        pits=[
            "**没有超时** —— 一个慢请求把连接池占满，后面所有请求堆积。必须设 connect/read 双超时。",
            "日志里记了 base64 原图 —— 日志文件瞬间膨胀，还涉嫌泄露用户数据。",
            "uvicorn 起不来 —— 因为 `create_app` 是工厂函数，要加 `--factory`。",
            "PII 只做了输出过滤没做输入过滤 —— 用户发的手机号照样进了日志。",
            "降级路径没测过 —— vLLM 挂了你才发现降级分支有语法错误。",
        ],
        nb=[
            ("md", "## 1. 起服务"),
            ("code", '''import subprocess, sys, time
# 在终端里跑更合适：  python -m src.serve.api --port 8080
# notebook 里演示用 Popen（记得最后 terminate）
print("命令：python -m src.serve.api --port 8080")
print("另开终端：curl -s localhost:8080/health")'''),
            ("md", "## 2. 参数校验的边界测试"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.serve.api import validate_request

cases = [
    {"messages": [{"role": "user", "content": "hi"}], "images": []},
    {"messages": [], "images": []},
    {"messages": [{"role": "user", "content": "hi"}], "images": ["x"] * 9},
]
for i, c in enumerate(cases, 1):
    try:
        print(f"case{i}:", validate_request(c))
    except Exception as e:
        print(f"case{i}: 拒绝 → {type(e).__name__}: {e}")'''),
            ("md", "## 3. PII 过滤自检"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.serve.api import filter_output

dirty = "您的订单 20260920123456 已发货，手机号 13812345678，地址 杭州市西湖区某路 1 号。"
print("原文:", dirty)
print("过滤:", filter_output(dirty))
print("\\n→ 客服日志和回流数据都必须过这一层")'''),
        ],
        footer="> **M5 达成**：有一个能对外服务的推理 API + 一份量化对比表 + W5 周复盘。",
        next_day_hint="`days/day-31.md` —— W6 Agent 周：工具协议",
    ),
]
