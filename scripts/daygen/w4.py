"""Week 4 · 评测体系（Day 19–24）

这一周的核心矛盾：**把「感觉变好了」变成「提升了多少、在哪类上提升」**。
好消息：这一周大部分能在本地做完（`src/eval/` 多数模块不依赖 torch）。
"""
from __future__ import annotations


def D(**kw):
    return kw


DAYS = [
    # ======================================================================
    D(
        week=4, n=19, slug="benchmark_overview", title="通用 VLM 评测全景",
        where="cloud",
        files="`src/eval/run_benchmark.py`（今日新增）、`docs/08-evaluation.md`",
        prereq="Day 18（有了合并后的权重）",
        goal="在自己的模型上跑 1–2 个通用榜的子集，拿到真实分数；"
             "并说清「通用榜单高分 ≠ 你的客服场景好用」的三个具体原因。",
        read=[
            "`docs/08-evaluation.md` 第 1–2 节 —— 每个榜测什么能力，一张表看完",
            "MMMU / MMBench / OCRBench / POPE / HallusionBench / DocVQA / MathVista 的**官方说明页**",
            "`docs/12-papers.md` 里 POPE 那一行（今天先扫一眼，Day 22 精读）",
        ],
        think=[
            "OCRBench 和 DocVQA 都在测「看图读字」，区别是什么？",
            "为什么 MMMU 分数高，不代表退货政策问答答得好？",
            "榜单数据可能已经被模型在预训练里见过（数据污染）—— 怎么粗略判断？",
        ],
        write_title="`src/eval/run_benchmark.py`（今日新增，已给实现）",
        write_rows=[
            ("`SUITES` 注册表", "定义每个子集的名字、数据源、评分函数 —— 想加榜就加一行"),
            ("`load_suite()`", "从本地缓存读子集（不联网也能跑）"),
            ("`score()`", "按题型走不同评分：选择题精确匹配 / OCR 用归一化编辑距离"),
            ("`run()`", "统一走 `LocalVLM` 接口（和 `run_eval.py` 复用同一个推理后端）"),
            ("报告", "输出「分数 + 用了多少条 + 是否子集」—— **子集必须标注**，别冒充全量"),
        ],
        run=[
            ("python -m src.eval.run_benchmark --model outputs/qwen25vl3b-cx-merged-v0 "
             "--suite ocrbench --n 100",
             "先跑 OCR 子集 100 条"),
            ("python -m src.eval.run_benchmark --model Qwen/Qwen2.5-VL-3B-Instruct "
             "--suite ocrbench --n 100",
             "同参数跑基座，作为对照"),
            ("python -m src.eval.run_benchmark --list", "看看支持哪些榜"),
        ],
        expect="""
[suite] ocrbench  子集 100/1000 条（⚠️ 这是子集，不能和榜单数字直接比）
[model] outputs/qwen25vl3b-cx-merged-v0
[run] 100/100 完成，耗时 412s，峰值显存 9.1 GB
[score] 归一化编辑距离均值 = 0.612  → 近似准确率 61.2%

对照（同一子集）：
  基座 Qwen2.5-VL-3B-Instruct : 63.8%
  你的 SFT 版本               : 61.2%   Δ = -2.6
  → 微调在通用 OCR 上略降是**正常**的（能力被收窄到客服域）
""",
        accept=[
            "至少 1 个榜单子集跑出分数，且**标注了这是子集**",
            "基座与微调版本在同一子集上的对照数字",
            "能说出「通用榜单高分 ≠ 场景好用」的**三个具体原因**（不是空话）",
            "能解释「微调后通用能力略降」为什么是正常现象",
        ],
        pits=[
            "benchmark 的 prompt 格式与训练格式不一致 —— 你用它考模型，模型却没见过这种问法。",
            "拿子集分数冒充全量榜单 —— 报告里必须写清 `子集 N/M`，这是学术诚信问题。",
            "忘了跑基座对照 —— 没有 baseline 的分数毫无意义。",
            "忽略数据污染 —— 有些榜的答案可能已在预训练语料里，分数虚高。",
        ],
        nb=[
            ("md", "## 1. 看看支持哪些榜"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.run_benchmark", "--list"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", "## 2. 跑对照实验\n\n**两条命令的差别只有一个 `--model`** —— 这就是「消融实验」的最朴素形态。"),
            ("code", '''import subprocess, sys
COMMON = ["-m", "src.eval.run_benchmark", "--suite", "ocrbench", "--n", "100"]
for tag, model in [("base", "Qwen/Qwen2.5-VL-3B-Instruct"),
                   ("sft", "outputs/qwen25vl3b-cx-merged-v0")]:
    r = subprocess.run([sys.executable, *COMMON, "--model", model],
                       capture_output=True, text=True, cwd="..")
    print("=" * 60); print(tag); print(r.stdout[-1200:] or r.stderr[-1200:])'''),
            ("md", """## 3. 落笔：为什么榜单 ≠ 场景

三个原因写下来。提示方向：**题型分布**、**答案形式**、**评测成本**。"""),
            ("code", '''why_benchmark_misleads = """
1. 题型分布：
2. 答案形式：
3. 其他：
"""
print(why_benchmark_misleads)'''),
        ],
        next_day_hint="`days/day-20.md` —— 造自己的客服评测集（今天起回本地做，不开 GPU）",
    ),

    # ======================================================================
    D(
        week=4, n=20, slug="build_domain_eval", title="构建客服领域评测集",
        where="local",
        files="`src/eval/build_domain_eval.py`",
        prereq="Day 12 的 `data/processed/clean.jsonl`；Day 19（知道通用榜的局限了）",
        goal="从真实商品与对话里构造 **300+ 条**客服评测样本，分 4 个难度层，"
             "并写一份标注手册 —— 这是整个项目**最重要的一份资产**。",
        read=[
            "`docs/08-evaluation.md` 第 3–4 节（标注规范、一致性、难度分层）",
            "`src/eval/build_domain_eval.py` 里的 `L1`–`L4` 模板",
        ],
        think=[
            "L1（直接问答）和 L4（多约束+拒答判断）各自的典型样本长什么样？",
            "为什么标准答案不能用「必须一字不差」来判？（会惩罚正确的同义表达）",
            "评测集和训练集的**图像级泄漏**为什么比对文本去重更致命？",
        ],
        write_title="`src/eval/build_domain_eval.py`（已给实现，你要改配额与模板）",
        write_rows=[
            ("`EvalSample`", "question / image / must_contain / must_not_contain / tier / intent"),
            ("L1–L4 模板", "四个难度层的出题模板 —— **今天最该改的就是这一块**"),
            ("`build_eval_set()`", "按配额抽样：L1 30% / L2 35% / L3 25% / L4 10%"),
            ("`generate_eval_card()`", "输出 `EVAL_CARD.md`，含泄漏检查 + 冻结声明"),
            ("`check_leakage()`", "用 pHash 查训练集与评测集的图像重叠 —— **必须为 0**"),
        ],
        write_note="> 评测集一旦生成就**冻结**，之后不许再改（改了就变成「对着答案调模型」）。\n"
                   "> 要改就出 `cx_eval_v2`，v1 的分数永远保留。",
        run=[
            ("python -m src.eval.build_domain_eval --selftest",
             "先跑离线自检：模板静态检查 + 四个难度层都必须非空"),
            ("python -m src.eval.build_domain_eval --source data/processed/clean.jsonl "
             "--n 320 --out data/eval/cx_eval_v1.jsonl --card",
             "生成 320 条评测样本 + 卡片"),
            ("python -m src.eval.build_domain_eval --source data/processed/clean.jsonl "
             "--out data/eval/cx_eval_v1.jsonl",
             "（上一条的 --card 只生成卡片；这条重跑一遍确认四层分布）"),
            ("head -20 data/eval/cx_eval_v1.jsonl", "肉眼抽查格式"),
        ],
        expect="""
✓ 写出 273 条评测样本 → data/eval/cx_eval_v1.jsonl

难度分布:
  L1:   94  █████████████
  L2:   69  ██████████
  L3:   80  ███████████
  L4:   30  ████

⚠️  L4 样本必须人工复核！(30 条)
✓ 评测集卡片 → data/eval/EVAL_CARD.md
""",
        accept=[
            "`cx_eval_v1.jsonl` 含 300±50 条，**4 个难度层都有量**，8 类意图全部覆盖",
            "`python -m src.eval.build_domain_eval --selftest` 全绿",
            "`EVAL_CARD.md` 里有泄漏检查结果（必须为 0）和**冻结声明**",
            "标注手册写清了「怎么判对」—— 关键是同义表达算对",
            "隔一周自己重标 30 条，一致性 ≥ 85%（这条需要留时间做）",
        ],
        pits=[
            "**图像级泄漏** —— 同一张商品图既在训练集又在评测集，分数虚高。pHash 检查必须过。",

            "⭐ **`must_contain` 里写「类别描述」而不是字面关键词。**\n"
            "   判分函数 `check_must_contain` 做的是**子串匹配**：\n"
            "   `hits = [k for k in must_contain if k in answer]`。\n"
            "   写「颜色类关键词」的话，字面上永远不会出现在任何回答里 →\n"
            "   命中数恒为 0 → **这批样本对任何模型都判失败**。\n"
            "   后果是报告上「规则通过率」偏低，看起来像「模型不行」——\n"
            "   实际上是评测集自己坏了。这种错误不抛异常、不发告警。\n"
            "   本项目实测踩过：L1 的 4 个模板里 2 个写的是「颜色类关键词」/\n"
            "   「款式类关键词」。现在 `--selftest` 的 `_lint_templates()` 会拦住。",

            "⭐ **模板的分组键必须和源数据字段对得上。**\n"
            "   L2 的模板是按 **intent**（`quality_issue` / `color_mismatch` /\n"
            "   `material`）分组的，而查候选池时用了 `by_type`（image_type）——\n"
            "   两边的键一个都对不上，`by_type.get(...)` 全返回空，\n"
            "   于是 **L2 整整一层恒为 0 条**（35% 的评测样本静默消失）。\n"
            "   报告上只表现为「L2: 0」一行，看上去像个正常的空桶。\n"
            "   现在空层是**硬错误**（`SystemExit`），不再是统计数字。",

            "只造简单题 —— L1 占 90%、L4 一条没有。配额是硬约束，不要妥协。",
            "标准答案写成「必须一字不差」—— 会把正确的同义表达判成错，评测结果失去意义。",
            "评测集没有冻结 —— 一边调模型一边改答案，等于自己骗自己。",
        ],
        nb=[
            ("md", "## 1. 生成评测集"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.build_domain_eval",
                    "--source", "data/processed/clean.jsonl",
                    "--n", "320", "--out", "data/eval/cx_eval_v1.jsonl", "--card"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-2500:] or r.stderr[-2500:])'''),
            ("md", "## 2. 逐层看样本：L1 vs L4 的差距有多大"),
            ("code", '''import json
from pathlib import Path
from collections import Counter

p = Path("../data/eval/cx_eval_v1.jsonl")
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print("总数:", len(rows))
    print("难度分布:", Counter(r.get("tier") for r in rows))
    print("意图分布:", Counter(r.get("intent") for r in rows))
    for tier in ("L1", "L2", "L3", "L4"):
        sample = next((r for r in rows if r.get("tier") == tier), None)
        if sample:
            print("=" * 70)
            print(f"[{tier}] {sample.get('question')}")
            print("  必须包含:", sample.get("must_contain"))
            print("  不能出现:", sample.get("must_not_contain"))
else:
    print("先跑上面的生成格子")'''),
            ("md", """## 3. 今天的真正作业：改配额，改模板

自动生成的题目**一定有你看着别扭的**。挑 10 条改成你想要的问法，
把改动写进 `src/eval/build_domain_eval.py` 的模板里，重跑一次。
**这一小时是今天最值钱的一小时。**"""),
            ("code", '''my_edits = """
我改了哪几条：
为什么改：
改完之后哪一层的题目质量明显变好了：
"""
print(my_edits)'''),
        ],
        next_day_hint="`days/day-21.md` —— 自动评测流水线 + LLM-as-judge 校准",
    ),

    # ======================================================================
    D(
        week=4, n=21, slug="eval_pipeline", title="自动评测流水线",
        where="cloud",
        files="`src/eval/run_eval.py`、`src/eval/judge.py`",
        prereq="Day 20（评测集已冻结）",
        goal="一条命令跑完全套评测并出 markdown 报告；把 LLM-as-judge 的"
             "**位置偏见**校准掉，让 judge 与人工打分的相关性 ≥ 0.7。",
        read=[
            "`docs/08-evaluation.md` 第 5 节（LLM-as-judge 的偏见与校准）",
            "`src/eval/judge.py` 的 `JUDGE_SYSTEM` —— 看它怎么显式要求「忽略长度和格式」",
            "`src/eval/metrics.py` 的规则打分 —— 有规则就别用 LLM，便宜且稳定",
        ],
        think=[
            "规则打分和 LLM judge 各自适合什么题？为什么要**先规则后 judge**？",
            "位置偏见怎么测？为什么必须做 A/B 交换的两次打分？",
            "judge 和被评模型同源（都是 Qwen 系）会带来什么偏见？",
        ],
        write_title="`src/eval/run_eval.py` + `src/eval/judge.py`（已给实现）",
        write_rows=[
            ("`metrics.evaluate_rules()`", "must_contain / must_not_contain / 拒答 / 超承诺 / PII 五类规则"),
            ("`Judge.score()`", "三个维度：事实性 / 帮助性 / 语气，各自 1–5 分"),
            ("`pairwise_with_calibration()`", "A/B 与 B/A 各打一次，抵消位置偏见"),
            ("`calibrate()`", "算 judge 与人工打分的 Spearman 相关，≥0.7 才算可用"),
            ("`build_report()`", "按难度层 / 意图分组，附失败模式与**自动生成的下一步建议**"),
        ],
        run=[
            ("python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct "
             "--eval data/eval/cx_eval_v1.jsonl --tag base --no-judge",
             "先用规则打分跑基座（不花钱）"),
            ("python -m src.eval.run_eval --model outputs/qwen25vl3b-cx-merged-v0 "
             "--eval data/eval/cx_eval_v1.jsonl --tag lora --baseline reports/eval_base_raw.jsonl",
             "跑你的版本，直接与基线对比出报告"),
            ("python -m src.eval.judge --calibrate reports/judge_sheet.jsonl",
             "校准 judge（需要你手工打 30 条分）"),
        ],
        expect="""
[eval] data/eval/cx_eval_v1.jsonl  320 条
[model] outputs/qwen25vl3b-cx-merged-v0
[rule] 规则打分：must_contain 命中 78.1% | 格式合规 96.2% | 越界承诺 3 条
[judge] 三维度均分：事实性 3.82 / 帮助性 4.01 / 语气 4.13
[run] 320/320 完成，耗时 21 min

按难度分层：
  L1  规则命中 91.7%   judge 4.2
  L2  规则命中 80.4%   judge 3.9
  L3  规则命中 66.3%   judge 3.5     ← 掉在这里
  L4  规则命中 43.8%   judge 2.8     ← 最弱

对比基线（base）：
  规则命中 71.2% → 78.1%   Δ +6.9
  L4 反而下降 4.1  ← 需要关注

[out] reports/eval_lora.md
""",
        accept=[
            "一条命令跑完全套评测，输出 markdown 报告",
            "报告**按难度层和意图分组**，能看出弱在哪一层（不能只有总分）",
            "judge 与人工打分相关性 ≥ 0.7（做 30 条人工标注校准）",
            "位置偏见已通过 A/B 交换抵消，且你能说出抵消的原理",
        ],
        pits=[
            "**judge 偏爱长答案** —— 这是最常见的偏见。`JUDGE_SYSTEM` 里显式要求「忽略长度」，"
            "但你还得用几个长短对照样本验证它真的照做了。",
            "**位置偏见** —— A 放前面 consistently 得分高。不做交换打分，pairwise 结果不可信。",
            "**自偏好** —— judge 和被测模型同源会系统性偏高。换一个不同家的 judge 交叉验证一次。",
            "能写规则判的却用了 LLM judge —— 贵、慢、还不稳定。规则优先。",
            "报告只给总分 —— 没有分层分组的报告，看完不知道下一步该干什么。",
        ],
        nb=[
            ("md", "## 1. 先跑不花钱的规则打分"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.run_eval",
                    "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
                    "--eval", "data/eval/cx_eval_v1.jsonl",
                    "--tag", "base", "--no-judge", "--limit", "20"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-2500:] or r.stderr[-2500:])'''),
            ("md", "## 2. judge 偏见自检（几个手造样本，不用跑模型）"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.eval.judge import JUDGE_SYSTEM, make_calibration_sheet

print(JUDGE_SYSTEM[:900])
print("\\n" + "=" * 70)
try:
    sheet = make_calibration_sheet(n=30)
    print(f"校准表已生成 {len(sheet)} 条 → reports/judge_sheet.jsonl")
    print("→ 打开它，逐条人工打分（1–5），再跑 calibrate")
except Exception as e:
    print("生成校准表：", e)'''),
            ("md", """## 3. 位置偏见的量化

造一对「实际上 A 更好」的答案，分别按 A/B 和 B/A 各打一次。
如果两次结论相反 —— 恭喜，你亲手测出了位置偏见。"""),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.eval.judge import pairwise_with_calibration

answer_short = "M/L 码有现货，S 码预计 3 天补货。"
answer_long  = ("您好呀！非常感谢您的咨询～关于您问的这件商品呢，"
                "我想先跟您说明一下我们的库存情况哦，目前呢……"
                "（此处省略 400 字客套话）")
print("短答案明显更好（信息密度高），但 judge 会怎么选？")
try:
    print(pairwise_with_calibration(answer_short, answer_long))
except Exception as e:
    print("需要配置 JUDGE_API_KEY：", e)'''),
        ],
        next_day_hint="`days/day-22.md` —— 幻觉评测：让模型学会说「不确定」",
    ),

    # ======================================================================
    D(
        week=4, n=22, slug="hallucination_eval", title="幻觉评测与缓解",
        where="local",
        files="`src/eval/hallucination.py`",
        prereq="Day 21",
        goal="用 POPE 式诱导问题测出幻觉率，**同时测漏答率**，再实现「不确定就说不确定」"
             "的缓解策略并做 A/B 对比。",
        read=[
            "`docs/08-evaluation.md` 第 6 节 + `docs/12-papers.md` 里 POPE 那一行",
            "`src/eval/hallucination.py` 里的 `MITIGATION_STRATEGIES`（4 个策略）",
        ],
        think=[
            "为什么**只看幻觉率会骗人**？「全答没有」的模型幻觉率是多少？",
            "存在的幻觉（说了图里没有的东西）和漏报（图里有的却说没有）—— 哪个对客服伤害更大？",
            "「不确定就说不确定」的策略，副作用是什么？（提示：用户满意度）",
        ],
        write_title="`src/eval/hallucination.py`（已给实现）",
        write_rows=[
            ("`build_probes()`", "对每张图构造「存在/不存在」两类问题（POPE 式）"),
            ("`parse_yes_no()`", "解析模型的回答 —— 要处理「是的」「有的」「并没有」这些变体"),
            ("`compute_hallucination_rate()`", "**同时返回幻觉率和漏答率**（两个都要报）"),
            ("`MITIGATION_STRATEGIES`", "4 个策略：明确允许说不确定 / 要求先描述再回答 / 降低温度 / 两轮自检"),
            ("`run_mitigation_experiment()`", "控制变量，一次只改一个策略"),
        ],
        run=[
            ("python -m src.eval.hallucination", "自检：构造 probes + 统计逻辑"),
            ("python -m src.eval.hallucination --build-probes --out data/eval/probes_v1.jsonl",
             "生成正式的诱导问题集"),
            ("python -m src.eval.run_eval --model outputs/qwen25vl3b-cx-merged-v0 "
             "--eval data/eval/probes_v1.jsonl --tag halluc",
             "在你的模型上实测幻觉率"),
        ],
        expect="""
[probes] 100 张图 × 2 类问题 = 200 条
         存在的物体 100 条 / 不存在的物体 100 条
[strategy] 基线（无缓解）
  幻觉率 = 23.0%   （问了不存在的物体，答"有"）
  漏答率 =  9.0%   （问了存在的物体，答"没有"）

[strategy] 明确允许说不确定
  幻觉率 = 11.0%  ↓12.0     漏答率 = 16.0%  ↑7.0    ← 代价
[strategy] 要求先描述再回答
  幻觉率 = 17.0%  ↓ 6.0     漏答率 = 10.0%  ↑1.0    ← 最划算
""",
        accept=[
            "幻觉率与漏答率**两列一起报**（只报一个的结论一律不采信）",
            "能说清三类幻觉（物体存在性 / 属性 / 关系）的成因差异",
            "4 个缓解策略里至少实测 2 个，并用控制变量方式对比",
            "能解释「降低幻觉率往往抬高漏答率」这个 trade-off",
        ],
        pits=[
            "**只看幻觉率** —— 一个永远回答「没有」的模型幻觉率是 0%，但完全没用。必须同时看漏答率。",
            "probe 里的物体太离谱 —— 比如给一张衣服图问「有飞机吗」。模型不可能答错，测不出东西。"
            "要选**视觉上可能混淆**的物体（同色系、同类目、背景里出现过的）。",
            "缓解策略对比时同时改了多个变量 —— 结论无效。一次只改一个。",
            "属性幻觉和关系幻觉用同一套 probe —— 它们是不同题型，要分开造。",
        ],
        nb=[
            ("md", "## 1. 看 probe 是怎么造的"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.eval.hallucination import PLAUSIBLE_OBJECTS, build_probes, MITIGATION_STRATEGIES

print("可混淆物体表（选得越像，测试越有效）:")
for k, v in list(PLAUSIBLE_OBJECTS.items())[:3]:
    print(f"  {k}: {v}")

print("\\n四种缓解策略:")
for i, s in enumerate(MITIGATION_STRATEGIES, 1):
    print(f"  {i}. {s if isinstance(s, str) else s.get('name')}")'''),
            ("md", "## 2. 亲手算一遍「只看幻觉率」的陷阱"),
            ("code", '''# 两个假想模型，在 200 条 probe 上的表现
models = {
    "模型A（爱瞎猜）": {"存在": (91, 100), "不存在": (23, 100)},   # (答对的, 总数)
    "模型B（全说没有）": {"存在": (0, 100), "不存在": (100, 100)},
}
for name, m in models.items():
    hallu = 1 - m["不存在"][0] / m["不存在"][1]     # 不该有的说有 = 幻觉
    miss  = 1 - m["存在"][0] / m["存在"][1]         # 该有的说没有 = 漏答
    acc   = (m["存在"][0] + m["不存在"][0]) / 200
    print(f"{name:16s} 幻觉率 {hallu:5.1%}  漏答率 {miss:5.1%}  总准确 {acc:5.1%}")
print("\\n→ 模型B 的幻觉率是 0%，但漏答率 100%：它其实什么都没在看。")'''),
            ("md", "## 3. 设计你自己的属性类 probe\n\n颜色、材质、尺码标、logo 位置 —— 这些「看起来对但可能错」的属性，才是客服场景的真实幻觉来源。"),
            ("code", '''attribute_probes = {
    # "问题": "该图真实的答案",
    # 例: "这件上衣的领口是什么形状？": "圆领",
}
print("至少写 5 条，Day 23 的错误分析会用到同类型样本。")'''),
        ],
        next_day_hint="`days/day-23.md` —— 错误分析：把失败样本自动归类",
    ),

    # ======================================================================
    D(
        week=4, n=23, slug="error_analysis", title="错误分析与 Bad Case 归类",
        where="local",
        files="`src/eval/error_analysis.py`",
        prereq="Day 21/22（有 `reports/eval_*_raw.jsonl` 结果文件）",
        goal="把失败样本自动聚类 + 关键词归类，输出 top 10 失败模式，"
             "并导出 `bad_cases.jsonl` —— 这份文件是 Day 26 构造 DPO 数据的原料。",
        read=[
            "（今天没有新讲义）—— 全部时间用来读失败样本",
            "`src/eval/error_analysis.py` 的 `ERROR_RULES` 和 `KEYWORD_GROUPS`",
        ],
        think=[
            "自动化归类结果如果不可解释，还有价值吗？怎么让它可解释？",
            "「答非所问」和「答得不全」是两类问题，修复手段一样吗？",
            "top 10 失败模式里，哪几类其实源于同一个根因？",
        ],
        write_title="`src/eval/error_analysis.py`（已给实现，你要扩规则）",
        write_rows=[
            ("`ERROR_RULES`", "规则归类：漏信息 / 幻觉参数 / 拒答不当 / 格式错 / 越界承诺 / PII"),
            ("`KEYWORD_GROUPS`", "关键词组 —— **今天最值得你扩充的地方**（业务词表）"),
            ("`semantic_group()`", "语义聚类，把规则覆盖不到的聚成新簇"),
            ("`classify()`", "单条样本 → （错误类型, 证据片段）"),
            ("`analyze()`", "输出 `error_analysis.md` + **`bad_cases.jsonl`**（带 `error_type` 标签）"),
        ],
        run=[
            ("make eval-fake",
             "⭐ 没有 GPU 也先跑这个：造「好 / 坏 / 全拒答」三种预测，验证归类和报告能不能区分"),
            ("python -m src.eval.error_analysis --in reports/eval_lora_raw.jsonl --out-dir reports/",
             "真实数据跑归类，出报告 + bad_cases.jsonl"),
            ("python -m src.eval.error_analysis --in reports/eval_fake_bad_raw.jsonl --out-dir reports/",
             "（离线）用假预测跑一遍，确认 bad_cases 格式正确"),
            ("head -3 reports/bad_cases.jsonl", "看 bad case 的格式（Day 26 要用）"),
        ],
        expect="""
读入 273 条评测记录
✓ 报告 → reports/error_analysis.md
✓ Bad cases → reports/bad_cases.jsonl  (225 条)
  下一步（Day 26）：python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl
  （注意：bad_cases 里的 reference 需要有参考答案才能构造偏好对）

======================================================================
失败模式 Top 5（语义主题）
======================================================================
  要素缺失          223  ████████████████████
  质量不达标         20  ██
""",
        accept=[
            "错误分类表已产出，每类**至少 2 条典型样本**",
            "能明确指出**下一轮该补哪三类数据**（不是「都补」）",
            "`bad_cases.jsonl` 已按 `error_type` 打标（Day 26 直接消费）",
            "能指出哪几类失败其实同源（比如「漏信息」和「不当拒答」可能都是数据不足）",
        ],
        pits=[
            "只统计不读样本 —— 分类表再漂亮，不读原始样本也发现不了真问题。",
            "聚类结果不可解释 —— 一堆「簇 3、簇 7」没人看得懂。要么加关键词命名，要么别用。",
            "`bad_cases.jsonl` 忘了打标签 —— 没有 `error_type` 就没法分层构造 DPO 数据。",
            "把「模型答得啰嗦」也当失败 —— 风格问题不是错误，别污染 bad case 池。",

            "⭐ **「失败」的定义只能有一套。**\n"
            "   实测踩过：`report.py` 里有的地方用 `rule[\"passed\"]` 判失败、\n"
            "   有的地方用 `rule[\"score\"] < 0.5`，两套口径不一致 ——\n"
            "   于是同一份报告能同时打印出「规则命中 64.8%」和「没有失败样本」。\n"
            "   **一个报告里出现两种『失败』定义，比不报错更危险**：\n"
            "   它不会崩，只会让你对着自相矛盾的数字猜。\n"
            "   修法是把判据收进一个函数（`is_fail()`），所有地方都调它。",

            "⭐ **「0 泄漏 / 0 失败 / 幻觉率 0%」必须带上分母。**\n"
            "   只看分子的话，「真的没问题」和「根本没检查」长得一模一样。\n"
            "   报告里凡是出现 0，都要能回答：分母是多少？怎么算出来的？",
        ],
        nb=[
            ("md", "## 1. 跑归类"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.error_analysis",
                    "--in", "reports/eval_lora_raw.jsonl", "--out-dir", "reports/"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-2500:] or r.stderr[-2500:])'''),
            ("md", "## 2. 扩业务词表（今天的核心动作）\n\n关键词组越贴合你的业务，归类越准。把下面填成你们真实的话术。"),
            ("code", '''import sys; sys.path.insert(0, "..")
from src.eval.error_analysis import KEYWORD_GROUPS

for k, v in KEYWORD_GROUPS.items():
    print(f"{k:12s} {v if isinstance(v, list) else v}")

my_extra = {
    # "物流时效": ["几天到", "什么时候发货", "空运"],
    # "尺码争议": ["偏大", "偏小", "跟平时穿的一样吗"],
}
print("\\n我补充的业务词表:", my_extra)'''),
            ("md", "## 3. 出声读 10 条真实 bad case\n\n这一步不能省。机器归类给你骨架，读样本给你血肉。"),
            ("code", '''import json
from pathlib import Path
p = Path("../reports/bad_cases.jsonl")
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"共 {len(rows)} 条，随机看 10 条：")
    import random
    for r in random.sample(rows, min(10, len(rows))):
        print("-" * 70)
        print("类型:", r.get("error_type"))
        print("问  :", str(r.get("question"))[:90])
        print("答  :", str(r.get("prediction", r.get("answer")))[:160])
else:
    print("先跑归类")'''),
        ],
        next_day_hint="`days/day-24.md` —— 出评测报告 v1，W4 收官",
    ),

    # ======================================================================
    D(
        week=4, n=24, slug="eval_report_v1", title="评测报告 v1",
        where="local",
        files="`src/eval/report.py`（今日新增）",
        prereq="Day 19–23 的全部结果文件",
        goal="把通用集、领域集、幻觉集的所有结果合成一份 `reports/eval_v1.md`，"
             "**每个结论都能追溯到数据**，没有一句「感觉」。",
        read=[
            "回看你 W4 前五天产出的所有 `reports/*.jsonl` 和 `*.md`",
            "`src/eval/report.py` —— 看它怎么把多个 run 合成一张对照表",
        ],
        think=[
            "报告里最容易出现「感觉」的句子是哪一类？怎么改成数据表述？",
            "「L4 上 SFT 反而下降」这件事，是 bug、是数据问题，还是评测集问题？",
            "这份报告给谁看？（自己 / 团队 / 面试官）—— 决定了它该怎么写。",
        ],
        write_title="`src/eval/report.py`（今日新增，已给实现）",
        write_rows=[
            ("`load_runs()`", "读多个 `eval_*_raw.jsonl`，按 tag 归档"),
            ("`compare_table()`", "基座 vs 各版本 × 各难度层的对照表"),
            ("`section_failure()`", "失败模式段落 —— 直接引用 Day 23 的归类结果"),
            ("`section_next()`", "下一步建议 —— **必须由数据推出**，不能是套话"),
            ("`--ablation`", "预留：Day 46 用它生成消融报告"),
        ],
        run=[
            ("make eval-fake",
             "⭐ 不需要 GPU：造好/坏/全拒答三种预测跑通报告，并验证报告**能区分好坏**"),
            ("python -m src.eval.report --runs reports/eval_base_raw.jsonl reports/eval_lora_raw.jsonl "
             "--out reports/eval_v1.md",
             "生成评测报告 v1"),
            ("head -80 reports/eval_v1.md", "检查报告结构"),
        ],
        expect="""
  [load] fake_good               273 条
  [load] fake_bad                273 条
  [load] fake_silent             273 条

✓ 报告 → reports/eval_v1.md  (72 行)

| 指标 | fake_good | fake_bad | fake_silent |
|---|---|---|---|
| 样本数 | 273 | 273 | 273 |
| 规则通过率 | 100.0% | 17.6% | 11.0% |
| 规则均分 | 0.996 | 0.725 | 0.648 |
""",
        accept=[
            "`reports/eval_v1.md` 含：通用集分数 + 领域集分层分 + 幻觉率对照 + 失败模式 + 下一步",
            "`make eval-fake` 能区分好坏：**好 > 坏 > 全拒答**，且三个数不接近",
            "报告里**每一个结论都有数字支撑**，没有「感觉」「似乎」",
            "报告里写明了模型版本、数据版本、评测集版本（可复现）",
            "`progress/weekly-review.md` 的 W4 段已写；进度表 W4 六天 `[x]`，M4 打卡",
        ],
        pits=[
            "只放总分不放分层 —— 看完不知道弱在哪，等于没评。",
            "没记录版本号 —— 两周后回看，不知道这个 78.1% 是哪个模型、哪份数据跑出来的。",
            "下一步建议写成「继续优化数据质量」—— 这不是建议，是废话。要具体到「补哪三类样本」。",

            "⭐ **「规则通过率」和「规则均分」是两个数，别混成一个。**\n"
            "   实测踩过：报告表头写「规则命中」，实际填的是 `rule.score` 的**均值**。\n"
            "   读者会把「平均分 0.65」理解成「65% 的样本通过了」——\n"
            "   而同一份报告下面又用 `score < 0.5` 判失败，\n"
            "   于是能同时打出「规则命中 64.8%」和「没有失败样本」。\n"
            "   一条回答可以拿很高的单项分但最终 `passed=False`，两个数必须分列。",

            "⭐ **版本对比的阈值判断必须取绝对值。**\n"
            "   实测踩过：写成 `if delta <= 0.005: 打印「差异不显著」`——\n"
            "   于是 **-89% 这种暴跌也满足条件**，被描述成「差异不显著」，\n"
            "   还顺带建议「加大训练数据量」。一次严重的性能回归被报告\n"
            "   说成了「没什么变化」。少了 `abs()` 的判断，方向完全反了。\n"
            "   现在退化会明确报「退化了 -x%，先回去核对数据/评测集/超参」。",

            "⭐ **「没有明显弱层」和「最弱的一层是 X」是两种结论，不能二选一硬报。**\n"
            "   实测踩过：四层通过率全是 100%，报告仍输出\n"
            "   「最弱的一层是 L1（通过率 100.0%），优先照顾这一层」——\n"
            "   数据说「没有短板」，报告却给出一个补救动作。\n"
            "   这类无依据的建议比没有建议更糟：它让你把时间花在不存在的问题上。\n"
            "   正确做法：极差很小且都很高时，说「没有明显弱层，应该提高评测难度」。",

            "**Markdown 表格里不许插说明。**\n"
            "   把一段 `>` 引用放在表头和第一行数据之间，表格会被截断 ——\n"
            "   渲染出来是一张只有表头的空表，数据行全丢。说明放在表格之后。",

            "把 W4 的结论和 W3 的抽检混着写 —— 一个是主观抽检一个是客观评测，要分开陈述。",
        ],
        nb=[
            ("md", "## 1. 生成报告"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.eval.report",
                    "--runs", "reports/eval_base_raw.jsonl", "reports/eval_lora_raw.jsonl",
                    "--out", "reports/eval_v1.md"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout[-2000:] or r.stderr[-2000:])'''),
            ("md", "## 2. 逐句检查「有没有感觉词」\n\n自动扫一遍报告里的主观词汇 —— 这是最有效的自查方式。"),
            ("code", '''import re
from pathlib import Path

p = Path("../reports/eval_v1.md")
FUZZY = ["感觉", "似乎", "大概", "可能好一些", "明显更好", "应该", "差不多"]
if p.exists():
    text = p.read_text()
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for w in FUZZY:
            if w in line:
                hits.append((i, w, line.strip()[:80]))
    if hits:
        print(f"⚠️  发现 {len(hits)} 处主观表述：")
        for i, w, line in hits:
            print(f"  L{i} [{w}] {line}")
        print("\\n→ 逐条改成数据表述，例如把「L4 明显更差」写成「L4 规则命中 43.8% vs 基座 47.9%，Δ-4.1」")
    else:
        print("✓ 没有明显的主观表述")
else:
    print("先跑上面的生成格子")'''),
            ("md", """## 3. M4 验收自问

- 如果只能改一件事提升模型，你会改什么？**依据是哪一行数据？**
- 这份报告里，哪一个结论是你最没把握的？为什么？"""),
            ("code", '''self_check = """
最该改的一件事：
依据的数据：
最没把握的结论：
"""
print(self_check)'''),
        ],
        next_day_hint="`days/day-25.md` —— W5 对齐周：DPO 原理",
    ),
]
