#!/usr/bin/env python3
"""
快速人工抽检 —— base vs 微调版本并排对比。

    python -m src.eval.quick_eval --base Qwen/Qwen2.5-VL-3B-Instruct \\
        --adapter outputs/qwen25vl3b-cx-lora-v0 --n 20 --out reports/quick_eval_w3.md

这是 W3 Day 18 的收尾工具，也是 W5/W7 想快速看一眼效果时的常用工具。
和 `run_eval.py` 的区别：
  run_eval     —— 自动打分（规则 + judge），出数字
  quick_eval   —— 只出并排文本，**判断交给人**，第一眼看清「到底变好没有」

关键设计：**分层抽样**。只抽 L1 会让你过度乐观，必须四个难度层都覆盖。
需要 GPU（要加载两个模型），所以是云上工具。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

DIFF_ORDER = ["L1", "L2", "L3", "L4"]
DEFAULT_SYSTEM = "你是一名专业的电商客服。回答要准确、简洁，不确定就直说。"


# ---------------------------------------------------------------------------
# 分层抽样
# ---------------------------------------------------------------------------

def stratified_sample(samples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """按难度层等比抽样，保证四层都有。"""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s.get("difficulty", "?")].append(s)

    tiers = [t for t in DIFF_ORDER if t in groups] or list(groups)
    total = sum(len(groups[t]) for t in tiers)
    picked: list[dict] = []
    for t in tiers:
        quota = max(1, round(n * len(groups[t]) / max(total, 1)))
        pool = groups[t][:]
        rng.shuffle(pool)
        picked.extend(pool[:quota])
    rng.shuffle(picked)
    return picked[:n]


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def generate_pair(base_id: str, adapter: Optional[str], samples: list[dict],
                  max_pixels: int = 1280) -> list[dict]:
    """同一批样本，跑两次：基座一次、挂 adapter 一次。"""
    from .run_eval import LocalVLM        # 延迟导入：这里要 torch

    print("[load] base   =", base_id)
    engine_base = LocalVLM(base_id, None, max_pixels)
    engine_tuned = None
    if adapter:
        print("[load] adapter=", adapter)
        engine_tuned = LocalVLM(base_id, adapter, max_pixels)

    rows = []
    for i, s in enumerate(samples, 1):
        try:
            a_base, ms1 = engine_base.generate(s.get("images", []), s["query"], DEFAULT_SYSTEM)
        except Exception as e:
            a_base, ms1 = f"<失败: {type(e).__name__}>", 0
        if engine_tuned:
            try:
                a_tuned, ms2 = engine_tuned.generate(s.get("images", []), s["query"], DEFAULT_SYSTEM)
            except Exception as e:
                a_tuned, ms2 = f"<失败: {type(e).__name__}>", 0
        else:
            a_tuned, ms2 = "（未提供 adapter）", 0
        rows.append({**{k: s.get(k) for k in ("id", "difficulty", "intent", "query", "images")},
                     "base": a_base, "tuned": a_tuned,
                     "base_ms": ms1, "tuned_ms": ms2})
        print(f"  [{i}/{len(samples)}] {s.get('difficulty')} {str(s['query'])[:36]}")
    return rows


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def render_md(rows: list[dict], base_tag: str, tuned_tag: str) -> str:
    n_ok = sum(1 for r in rows if not r["base"].startswith("<失败")
               and not r["tuned"].startswith("<失败"))
    L = [f"# 抽检对比 · {base_tag} vs {tuned_tag}", ""]
    L += [f"> 共 {len(rows)} 条（成功 {n_ok}），**分层抽样**（L1–L4 都覆盖）",
          "> 最后一列留给你手填：变好 / 变差 / 没变 —— 这一列才是这份报告的价值所在。", ""]

    # 分层统计
    by_tier: dict[str, int] = defaultdict(int)
    for r in rows:
        by_tier[r.get("difficulty", "?")] += 1
    L += ["| 难度 | 条数 |", "|---|---|"]
    for t in sorted(by_tier, key=lambda x: DIFF_ORDER.index(x) if x in DIFF_ORDER else 99):
        L.append(f"| {t} | {by_tier[t]} |")
    L.append("")

    L += ["## 逐条对比", "",
          "| # | 难度 | 问题 | 基座 | 微调版 | 你的判断 |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        def cell(x: str, w: int = 90) -> str:
            x = str(x).replace("\n", " ").replace("|", "／")
            return x[:w] + ("…" if len(x) > w else "")
        L.append(f"| {i} | {r.get('difficulty')} | {cell(r['query'], 28)} | "
                 f"{cell(r['base'])} | {cell(r['tuned'])} |  |")
    L.append("")
    L += ["## 汇总（填完手动统计）", "",
          "| 类别 | 条数 |", "|---|---|",
          "| 变好 |  |", "| 变差 |  |", "| 没变 |  |", "",
          "## 判断标准（避免只凭感觉）", "",
          "- **变好**：信息更准确 / 更完整 / 该拒答时拒答了 / 格式对了",
          "- **变差**：出现幻觉 / 漏掉关键约束 / 该答却拒答 / 过度啰嗦",
          "- **没变**：两条回答实质等价（措辞不同不算变好）", "",
          "> 至少找出 3 个「变好」和 2 个「变差」的 case —— 变差的 case 是下一轮 DPO 的原料。",
          ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="base vs 微调 快速人工抽检")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 路径（不给就只跑基座）")
    ap.add_argument("--eval", default="data/eval/cx_eval_v1.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pixels", type=int, default=1280)
    ap.add_argument("--out", default="reports/quick_eval.md")
    args = ap.parse_args()

    p = Path(args.eval)
    if not p.exists():
        print(f"✗ 找不到评测集 {p}（先跑 Day 20 的 build_domain_eval）")
        return 1
    samples = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = stratified_sample(samples, args.n, args.seed)
    print(f"[sample] 分层抽样 {len(picked)}/{len(samples)} 条")

    rows = generate_pair(args.base, args.adapter, picked, args.max_pixels)

    md = render_md(rows, "base", Path(args.adapter).name if args.adapter else "base")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n✓ 抽检报告 → {out}")
    print("  下一步：打开它，把「你的判断」一列填满 —— 这一步不能省。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
