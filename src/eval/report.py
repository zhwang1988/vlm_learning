#!/usr/bin/env python3
"""
评测报告合成 —— 把多个 run 的结果汇成一份能给人看的报告。

    # 基座 vs 微调
    python -m src.eval.report --runs reports/eval_base_raw.jsonl \\
                                    reports/eval_lora_raw.jsonl \\
                             --out reports/eval_v1.md

    # 消融（W8 Day 46）
    python -m src.eval.report --ablation reports/eval_ab*_raw.jsonl \\
                             --out reports/ablation.md

设计原则：
  1. **每个结论都要能追溯到数字** —— 报告里不许出现「感觉」「似乎」
  2. 必须分层（难度 / 意图）—— 只有总分等于没评
  3. 下一步建议由数据推出，不是套话
  4. 纯 Python，不依赖 torch，本地就能跑

输入格式：`src/eval/run_eval.py` 写出的 `eval_{tag}_raw.jsonl`
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

# 主观表述黑名单 —— 报告里出现这些词会被自检拦下
FUZZY_WORDS = ["感觉", "似乎", "大概", "可能好一些", "明显更好", "应该差不多", "估计"]

DIFF_ORDER = ["L1", "L2", "L3", "L4"]


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------

def load_run(path: str | Path) -> tuple[str, list[dict]]:
    """读一个 raw.jsonl，tag 从文件名里推（eval_<tag>_raw.jsonl）。"""
    p = Path(path)
    stem = p.stem                                   # eval_base_raw
    tag = stem[len("eval_"):-len("_raw")] if stem.startswith("eval_") and stem.endswith("_raw") else stem
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tag, rows


def expand(patterns: list[str]) -> list[str]:
    """支持 `eval_ab*_raw.jsonl` 这种通配（shell 没展开时也能用）。"""
    out: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        out.extend(hits or [pat])
    return out


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def _num(d: dict | None) -> float | None:
    """judge 字典 → 一个可比的数（优先 overall / score，否则取数值均值）。"""
    if not isinstance(d, dict) or not d:
        return None
    for key in ("overall", "score", "total"):
        v = d.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    vals = [float(v) for v in d.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _avg(xs: list) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    lat = [r.get("latency_ms", 0) for r in ok]
    return {
        "n": len(rows),
        "n_err": len(rows) - len(ok),
        # ⚠️ `passed` 和 `score` 是两个不同的量，必须分开报、分别命名。
        #    早期版本只算 score（均值），却在表头写成「规则命中」——
        #    读者会把「平均分 0.65」理解成「65% 的样本通过了」，
        #    两者在真实数据上可以差很多。而且下面 `is_fail()` 用的是
        #    `score < 0.5`，与 `passed` 又是第三套口径 ——
        #    于是同一份报告里能同时出现「规则命中 64.8%」和
        #    「没有失败样本」，自相矛盾。
        "rule_score": _avg([(r.get("rule") or {}).get("score") for r in ok]),
        "rule_pass": (sum(1 for r in ok if (r.get("rule") or {}).get("passed"))
                      / len(ok) if ok else 0.0),
        "judge": _avg([_num(r.get("judge")) for r in ok]),
        "n_judged": sum(1 for r in ok if _num(r.get("judge")) is not None),
        "p50": _pct(lat, 50),
        "p95": _pct(lat, 95),
    }


def by_key(rows: list[dict], key: str) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r.get(key) or "未知"].append(r)
    return dict(g)


def is_fail(r: dict, judge_thr: float = 3.0) -> bool:
    """这条算不算「失败」。

    ⚠️ 判据只有**一套**，就是这里。报告里所有地方（总览、分层、失败模式、
    复现信息）都必须调这个函数 —— 早期版本有的地方用 `rule["passed"]`、
    有的地方用 `rule["score"] < 0.5`，于是同一份报告里
    「规则命中 64.8%」和「没有失败样本」能同时成立。
    **一个报告里出现两种『失败』定义，比不报错更危险。**

    判据：
      1. 推理出错（error 非空）
      2. 规则判定没通过（rule["passed"] 为 False，含 must_contain /
         must_not_contain / refusal / over_promise / pii 任一项不过）
      3. judge 打了分且低于阈值（judge 为空不算失败 —— 没打分不等于差）
    """
    if r.get("error"):
        return True
    if not (r.get("rule") or {}).get("passed", True):
        return True
    j = _num(r.get("judge"))
    return j is not None and j < judge_thr


def classify_failures(fails: list[dict]) -> dict[str, list[dict]]:
    """能借 error_analysis 的规则就借，借不到就用内置的粗分类。"""
    try:
        from .error_analysis import classify  # type: ignore
    except Exception:
        classify = None  # type: ignore

    g: dict[str, list[dict]] = defaultdict(list)
    for r in fails:
        label = None
        if classify is not None:
            try:
                res = classify(r)
                label = res[0] if isinstance(res, (tuple, list)) else res
            except Exception:
                label = None
        if not label:
            if r.get("error"):
                label = "调用失败"
            elif ((r.get("rule") or {}).get("score") or 0) < 0.2:
                label = "规则命中极低"
            else:
                label = "质量不达标"
        g[str(label)].append(r)
    return dict(g)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def compare_table(runs: dict[str, list[dict]]) -> list[str]:
    L = ["| 指标 | " + " | ".join(runs) + " |",
         "|---" * (len(runs) + 1) + "|"]
    # ⚠️ 「通过率」和「均分」都要列，且列名要说清是哪个。
    #    只有一个数时读者无从判断它是什么口径 —— 早期版本把它们混成一个
    #    「规则命中」，既不是通过率也不是均分，看的人只能猜。
    # ⚠️ 每一项都要带「有效性守卫」（第 4 个字段 = 依赖的分母键）。
    #    没有分母时算出来的 0 不是测量结果，是**空缺**。最典型的是 judge：
    #    一条都没被 judge 打分时 agg() 里的 judge 是 0.0，直接打出来就是
    #    「judge 均分 0.00」—— 读者会理解成「模型被 judge 打了 0 分」，
    #    实际是「judge 根本没跑」。0 又一次同时表示「极差」和「没有」，
    #    和 pHash 哨兵值是同一个病。没有数据就写「—」，不要写一个
    #    长得像测量结果的 0。
    metrics = [("样本数", "n", "{:.0f}", None),
               ("规则通过率", "rule_pass", "{:.1%}", "n"),
               ("规则均分", "rule_score", "{:.3f}", "n"),
               ("judge 均分", "judge", "{:.2f}", "n_judged"),
               ("judge 已打分条数", "n_judged", "{:.0f}", None),
               ("P95 延迟", "p95", "{:.0f} ms", "n")]
    stats = {t: agg(rows) for t, rows in runs.items()}
    suppressed: list[str] = []
    for label, key, fmt, guard in metrics:
        cells = []
        for t in runs:
            if guard is not None and not stats[t].get(guard):
                cells.append("—")
            else:
                cells.append(fmt.format(stats[t][key]))
        if "—" in cells:
            suppressed.append(label)
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> 「规则通过率」是 `rule.passed` 为真的样本占比；"
             "「规则均分」是 `rule.score` 的均值。二者含义不同："
             "一条回答可以拿了很高的单项分但最终 `passed=False`。")
    if suppressed:
        L.append(">")
        L.append(f"> 「—」= 该项**没有可用的分母**，不是 0 分。受影响的行："
                 f"{'、'.join(suppressed)}。"
                 f"最常见的原因是本次评测没跑 LLM judge"
                 f"（没配 API key / 用 `scripts/fake_eval.py` 造的预测），"
                 f"所以 judge 列整列为空 —— 这不代表模型 judge 得分低。")
    return L


def section_difficulty(runs: dict[str, list[dict]]) -> list[str]:
    L = ["### 按难度分层", "",
         "> 「规则通过率」= 该层里 `rule.passed` 为真的比例；"
         "「规则均分」= 该层 `rule.score` 的均值。两个数含义不同，别混读。",
         ""]
    L.append("| 难度 | " + " | ".join(f"{t} 通过率" for t in runs)
             + " | " + " | ".join(f"{t} 均分" for t in runs) + " |")
    L.append("|---" * (len(runs) * 2 + 1) + "|")
    per = {t: by_key(rows, "difficulty") for t, rows in runs.items()}
    tiers = [t for t in DIFF_ORDER if any(t in per[x] for x in per)]
    tiers += [t for t in per[list(runs)[0]] if t not in tiers]
    for tier in tiers:
        cells = [f"{agg(per[t][tier])['rule_pass']:.1%}" if tier in per[t] else "—"
                 for t in runs]
        cells += [f"{agg(per[t][tier])['rule_score']:.3f}" if tier in per[t] else "—"
                  for t in runs]
        L.append(f"| {tier} | " + " | ".join(cells) + " |")
    L.append("")

    # 自动指出最弱的一层。
    # ⚠️ 必须**标明是基于哪个 run** 得出的结论 —— 上面那张表有好几列，
    #    不标的话读者会以为自己看的那一列就是依据（而它其实是第一个 run 的）。
    #    结论和它下面的表格口径不一致，是报告类文档最典型的误导方式。
    #
    # ⚠️ 还有：**没有弱点时不要硬报一个**。
    #    早先版本无条件取 min，四层全是 100% 时会输出
    #    「最弱的一层是 L1（通过率 100.0%），优先照顾这一层」——
    #    数据说「没有短板」，报告却给出一个补救动作。
    #    这类无依据的建议比没有建议更糟：它会让人把时间花在不存在的问题上。
    first = list(runs)[0]
    scored = {t: agg(v)["rule_pass"] for t, v in per[first].items()}
    if scored:
        worst = min(scored, key=lambda t: scored[t])
        best = max(scored.values())
        spread = best - min(scored.values())
        if spread < 0.02 and min(scored.values()) > 0.98:
            L.append(f"→ 各层通过率都很高且差距很小（极差 {spread:.1%}，"
                     f"最低 {scored[worst]:.1%}），**没有明显弱层**。"
                     f"下一步应该去提高评测难度，而不是补数据。")
        else:
            L.append(f"→ **最弱的一层是 {worst}**（基于 `{first}`，"
                     f"通过率 {scored[worst]:.1%}，与最高层差 {spread:.1%}）。"
                     f"下一轮的数据补充优先照顾这一层。")
        L.append("")
    return L


def section_intent(runs: dict[str, list[dict]], top: int = 5) -> list[str]:
    """按意图看哪一类最弱。

    ⚠️ 和 section_difficulty 同一个坑：**没有弱点时不要硬报一个**。
       全部意图并列 100% 时，标题写「最弱的意图」而表格第一行也是 100%，
       读者会去找一个不存在的短板，甚至据此去补某一类的数据 ——
       白干，而且掩盖了真正该做的事（把评测加难）。
    """
    first = list(runs)[0]
    g = by_key(runs[first], "intent")
    if not g:
        return [f"### 各意图通过率（基于 `{first}`）", "",
                f"⚠️ `{first}` 里没有任何带 `intent` 字段的样本，"
                f"无法按意图分层 —— 这不是「各意图表现均衡」。", ""]

    rows = sorted(((k, agg(v)["rule_pass"], len(v)) for k, v in g.items()),
                  key=lambda x: x[1])
    lo, hi = rows[0][1], rows[-1][1]
    tie_all = (hi - lo) < 0.02

    if tie_all:
        L = [f"### 各意图通过率（基于 `{first}`，**无最弱项**）", ""]
    else:
        L = [f"### 最弱的意图（基于 `{first}`，按规则通过率排序）", ""]
    L += ["| 意图 | 通过率 | 样本数 |", "|---|---|---|"]
    for k, score, n in rows[:top]:
        L.append(f"| {k} | {score:.1%} | {n} |")
    L.append("")

    if tie_all:
        if lo > 0.98:
            L += [f"→ 全部 {len(rows)} 个意图通过率最低 {lo:.1%}（极差 "
                  f"{hi - lo:.1%}），**分不出最弱项**。当前评测对这个模型"
                  f"偏简单，该加难度，而不是补某一类的数据。", ""]
        else:
            L += [f"→ 全部 {len(rows)} 个意图通过率并列在 {lo:.1%} 附近（极差 "
                  f"{hi - lo:.1%}），**分不出最弱项** —— 要么样本量不够，"
                  f"要么评测难度不足以区分意图间的差异。", ""]
    elif len(rows) > top:
        L += [f"（只列了最低的 {top} 个，共 {len(rows)} 个意图。）", ""]
    return L


def section_failure(fails: dict[str, list[dict]], total_fail: int,
                    n_total: int, tag: str) -> list[str]:
    L = ["### 失败模式", ""]
    if not fails:
        # ⚠️ 「没有失败样本」这句话必须带上分母和判据。
        #    单独一句「没有失败样本」有两种读法：
        #       真的全都通过了            （n_total 很大）
        #       这个 run 里根本没有记录   （n_total = 0）
        #    和「0 泄漏」「幻觉率 0%」是同一类陷阱：**分母塌成 0 时，
        #    任何『没有问题』的结论都成立。** 所以这里把 n_total 和
        #    判据一起写出来，让读者能自己判断。
        if n_total == 0:
            L += [f"⚠️ 该 run（`{tag}`）**没有任何记录**，"
                  f"所以「没有失败样本」不代表质量好，而是没数据。", ""]
        else:
            L += [f"✓ {n_total} 条样本全部通过（判据：推理无错 且 规则判定通过 且 "
                  f"judge 未低于阈值）。", ""]
        return L
    L += [f"共 {total_fail} / {n_total} 条失败（{total_fail / max(n_total, 1):.1%}）", "",
          "| 失败类型 | 数量 | 占失败样本 | 典型样本 |", "|---|---|---|---|"]
    for label, items in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        sample = str(items[0].get("query", ""))[:32].replace("|", "／")
        L.append(f"| {label} | {len(items)} | {len(items)/max(total_fail,1):.0%} | {sample} |")
    L.append("")
    return L


def section_next(runs: dict[str, list[dict]], fails: dict[str, list[dict]],
                 total_fail: int) -> list[str]:
    """下一步建议 —— 必须由数据推出。"""
    L = ["### 下一步建议", ""]
    hints: list[str] = []

    first = list(runs)[0]
    g = by_key(runs[first], "difficulty")
    scored = {t: agg(v)["rule_pass"] for t, v in g.items()}
    if scored:
        worst = min(scored, key=lambda t: scored[t])
        spread = max(scored.values()) - min(scored.values())
        if spread < 0.02 and min(scored.values()) > 0.98:
            hints.append(f"**各层都没有明显短板**（最低 {worst} {scored[worst]:.1%}，"
                         f"极差 {spread:.1%}）。当前评测对这个模型**偏简单**，"
                         f"应该先加难度（L3 多图对比 / L4 边界场景），"
                         f"而不是继续补同分布的数据。")
        else:
            hints.append(f"**补 {worst} 层的数据**：这一层规则通过率 "
                         f"{scored[worst]:.1%}，是所有层里最低的。")

    if fails:
        top_label, top_items = max(fails.items(), key=lambda kv: len(kv[1]))
        hints.append(f"**针对「{top_label}」做专项**：占失败样本 "
                     f"{len(top_items)/max(total_fail,1):.0%}（{len(top_items)} 条），是最大的一类。")
        hint_map = {
            "幻觉参数": "构造「同图不同答」偏好对，走 Day 26 的 DPO 路径",
            "要素缺失": "检查这类问题的 must_contain 是否过严；若合理，则补该要素的训练样本",
            "不当拒答": "补「该答就答」的正样本，并检查系统提示是否过度保守",
            "规则命中极低": "先人工读 10 条，确认是数据问题还是模型能力问题",
            "质量不达标": "抽读 10 条失败样本，先分清是模型问题还是评测集问题",
            "调用失败": "检查推理服务的稳定性和超时设置",
        }
        if top_label in hint_map:
            # 拼进上一条而不是新起一条：`hints` 最后会被编成有序列表，
            # 新起一条会让「具体做法」变成独立编号的建议，读起来像两件事。
            hints[-1] += f"\n   → 具体做法：{hint_map[top_label]}"

    if len(runs) >= 2:
        tags = list(runs)
        base, last = agg(runs[tags[0]]), agg(runs[tags[-1]])
        delta = last["rule_pass"] - base["rule_pass"]
        # ⚠️ 判断「有没有差别」必须用 **绝对值**。
        #    早先写成 `if delta <= 0.005:`，于是 -89% 这种暴跌也满足条件，
        #    被打印成「两个版本差异不显著」—— 一次严重的性能回归被报告
        #    描述成「没什么变化」，还顺带建议「加大训练数据量」。
        #    少了 abs() 的阈值判断，方向完全反了。
        if abs(delta) <= 0.005:
            hints.append(f"**两个版本差异不显著**（{tags[0]} → {tags[-1]}，"
                         f"Δ{delta:+.1%}）。先确认评测集没有饱和，"
                         f"再考虑加大训练数据量。")
        elif delta > 0:
            hints.append(f"**从 {tags[0]} 到 {tags[-1]} 提升了 {delta:+.1%}**，"
                         f"可以继续沿这条路径推进。")
        else:
            hints.append(f"⚠️ **从 {tags[0]} 到 {tags[-1]} 退化了 {delta:+.1%}** —— "
                         f"这是一次性能回归，先别继续改，回去核对：数据有没有变、"
                         f"评测集有没有改、超参是不是动了。")

    if not hints:
        hints.append("数据不足，先把评测跑起来再谈建议。")
    L += [f"{i}. {h}" for i, h in enumerate(hints, 1)]
    L.append("")
    return L


def ablation_section(runs: dict[str, list[dict]]) -> list[str]:
    """消融：每一步的增量贡献。"""
    tags = list(runs)
    L = ["## 消融分析：每一步贡献了多少", ""]
    L.append("| 步骤 | 规则通过率 | judge | 相对上一步 Δ通过率 | Δjudge |")
    L.append("|---|---|---|---|---|")
    prev = None
    for t in tags:
        s = agg(runs[t])
        if prev is None:
            d1 = d2 = "—"
        else:
            d1 = f"{s['rule_pass'] - prev['rule_pass']:+.1%}"
            d2 = f"{s['judge'] - prev['judge']:+.2f}"
        L.append(f"| {t} | {s['rule_pass']:.1%} | {s['judge']:.2f} | {d1} | {d2} |")
        prev = s
    L.append("")

    # 分层增量
    L += ["### 分层看谁贡献最大", "", "| 难度 | " + " | ".join(tags) + " |", "|---" * (len(tags) + 1) + "|"]
    per = {t: by_key(runs[t], "difficulty") for t in tags}
    for tier in DIFF_ORDER:
        cells = [f"{agg(per[t][tier])['rule_pass']:.1%}" if tier in per[t] else "—" for t in tags]
        L.append(f"| {tier} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> 如果后期步骤的增量集中在 L3/L4，说明它治的是**难题**；"
             "如果集中在上层，可能只是风格拟合。")
    L.append("")
    return L


def build_merged_report(runs: dict[str, list[dict]], ablation: bool = False) -> str:
    tags = list(runs)
    L = [f"# 评测报告 · {' vs '.join(tags)}" if not ablation else "# 消融实验报告", ""]
    L += [f"> 生成自 {len(tags)} 个 run：{', '.join(tags)}",
          "> 数据来源：`src/eval/run_eval.py` 写出的 `eval_*_raw.jsonl`", ""]

    L += ["## 总览", ""] + compare_table(runs) + [""]

    if ablation:
        L += ablation_section(runs)
    else:
        L += ["## 分层结果", ""] + section_difficulty(runs) + section_intent(runs)

    # 失败模式（用最后一个 run）
    last = runs[tags[-1]]
    fails = [r for r in last if is_fail(r)]
    L += ["## 失败分析（基于 `" + tags[-1] + "`）", ""]
    L += section_failure(classify_failures(fails), len(fails), len(last), tags[-1])
    L += section_next(runs, classify_failures(fails), len(fails))

    L += ["## 复现信息", "",
          "| 项 | 值 |", "|---|---|"]
    for t in tags:
        s = agg(runs[t])
        nf = sum(1 for r in runs[t] if is_fail(r))
        L.append(f"| run `{t}` | {s['n']} 条（失败 {nf}，其中调用出错 {s['n_err']}） |")
    # ⚠️ 口径说明必须放在表格**之后**。放在表头和第一行之间会把 markdown
    #    表格截断 —— 渲染出来是一张只有表头的空表 + 一段引用，数据行全丢。
    L += ["", "> 「失败」口径与上方失败分析一致：推理出错 或 规则判定不通过 "
              "或 judge 低于阈值。"]
    L += ["", "> 报告由 `src/eval/report.py` 生成；改数字请重跑 `run_eval`，不要手改本文件。", ""]
    return "\n".join(L)


def lint_fuzzy(text: str) -> list[str]:
    """扫主观表述 —— 报告的自检。"""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for w in FUZZY_WORDS:
            if w in line:
                hits.append(f"L{i} [{w}] {line.strip()[:70]}")
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="评测报告合成")
    ap.add_argument("--runs", nargs="*", default=[], help="多个 eval_*_raw.jsonl")
    ap.add_argument("--ablation", nargs="*", default=[], help="消融模式：按顺序传入各阶段")
    ap.add_argument("--out", default="reports/eval_v1.md")
    ap.add_argument("--no-lint", action="store_true", help="跳过主观表述自检")
    args = ap.parse_args()

    paths = expand((args.ablation or args.runs))
    if not paths:
        ap.print_help()
        print("\n例：python -m src.eval.report --runs "
              "reports/eval_base_raw.jsonl reports/eval_lora_raw.jsonl --out reports/eval_v1.md")
        return 1

    runs: dict[str, list[dict]] = {}
    for p in paths:
        if not Path(p).exists():
            print(f"✗ 找不到 {p}（先跑 run_eval.py）")
            return 1
        tag, rows = load_run(p)
        runs[tag] = rows
        print(f"  [load] {tag:22s} {len(rows):>4} 条")

    report = build_merged_report(runs, ablation=bool(args.ablation))

    hits = lint_fuzzy(report)
    if hits and not args.no_lint:
        print(f"\n⚠️  报告里有 {len(hits)} 处主观表述（建议改成数据表述）：")
        for h in hits[:10]:
            print("    " + h)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n✓ 报告 → {out}  ({len(report.splitlines())} 行)")
    if hits and not args.no_lint:
        print(f"  自检：{len(hits)} 处主观表述（不是错误，但值得改）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
