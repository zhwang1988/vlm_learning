"""
错误分析与 Bad Case 归类。

对应计划 Day 23。

**这一步比看数字更有价值。**

做法：把失败样本自动聚类，输出 Top 10 失败模式。
然后你就能明确回答：下一轮该补哪三类数据？

输出的 bad_cases.jsonl 还会被 DPO 的偏好数据构造复用（src/train/dpo_loss.py）。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 错误类型分类器（规则 + 可插拔的语义聚类）
# ---------------------------------------------------------------------------

ERROR_RULES = [
    ("未拒答", lambda r: r.get("difficulty") == "L4"
     and r.get("rule", {}).get("checks", {}).get("refusal", {}).get("passed") is False),
    ("要素缺失", lambda r: r.get("rule", {}).get("checks", {})
     .get("must_contain", {}).get("passed") is False),
    ("出现禁用语", lambda r: r.get("rule", {}).get("checks", {})
     .get("must_not_contain", {}).get("passed") is False),
    ("过度承诺", lambda r: r.get("rule", {}).get("checks", {})
     .get("over_promise", {}).get("passed") is False),
    ("PII 泄露", lambda r: r.get("rule", {}).get("checks", {})
     .get("pii", {}).get("passed") is False),
    ("意图识别错", lambda r: r.get("rule", {}).get("checks", {})
     .get("intent", {}).get("passed") is False),
    ("推理失败", lambda r: bool(r.get("error"))),
    ("回答过短", lambda r: len(r.get("model_answer", "")) < 15),
    ("回答过长", lambda r: len(r.get("model_answer", "")) > 500),
    ("judge 低分", lambda r: (r.get("judge", {}).get("accuracy", 5) <= 2)),
]


def classify(record: dict) -> list[str]:
    return [name for name, fn in ERROR_RULES if fn(record)]


# ---------------------------------------------------------------------------
# 语义聚类（把同样的问题聚在一起）
# ---------------------------------------------------------------------------

KEYWORD_GROUPS = {
    "色差判断": ["色差", "颜色差", "米白", "纯白", "偏黄", "偏灰", "显示器"],
    "瑕疵分级": ["线头", "走线", "瑕疵", "质量", "严重", "轻微", "正常范围"],
    "尺码建议": ["尺码", "身高", "体重", "偏大", "偏小", "胸围", "衣长"],
    "多图对比": ["这张", "哪个", "对比", "不一样", "同款"],
    "OCR/数字": ["厘米", "cm", "码数", "克重", "成分", "百分比"],
    "物流查询": ["快递", "物流", "发货", "到哪", "单号"],
    "退货流程": ["退货", "退款", "换货", "申请"],
    "材质手感": ["材质", "面料", "起球", "透气", "手感", "缩水"],
    "情绪安抚": ["投诉", "欺诈", "生气", "差评", "不满"],
}


def semantic_group(record: dict) -> str:
    text = record.get("query", "") + record.get("model_answer", "")
    for group, kws in KEYWORD_GROUPS.items():
        if any(k in text for k in kws):
            return group
    return "其他"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def analyze(raw_path: str | Path,
            out_dir: str | Path = "reports",
            top_n: int = 10) -> dict:
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in
               raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"读入 {len(records)} 条评测记录")

    ok = [r for r in records if not r.get("error")]
    failures = [r for r in ok if not r.get("rule", {}).get("passed")]

    # --- 按错误类型 ---
    type_counter = Counter()
    type_examples = defaultdict(list)
    for r in failures:
        for t in classify(r):
            type_counter[t] += 1
            if len(type_examples[t]) < 3:
                type_examples[t].append(r)

    # --- 按语义主题 ---
    group_counter = Counter()
    group_examples = defaultdict(list)
    for r in failures:
        g = semantic_group(r)
        group_counter[g] += 1
        if len(group_examples[g]) < 3:
            group_examples[g].append(r)

    # --- 交叉：难度 × 意图 ---
    cross = Counter()
    for r in failures:
        cross[(r.get("difficulty"), r.get("intent"))] += 1

    # --- 生成报告 ---
    lines = ["# 错误分析报告", ""]
    lines.append(f"- 总样本: **{len(records)}**")
    lines.append(f"- 推理失败: **{len(records) - len(ok)}**")
    lines.append(f"- 规则不通过: **{len(failures)}** "
                 f"({len(failures) / max(len(ok), 1):.1%})")
    lines.append("")

    lines.append("## 错误类型分布")
    lines.append("")
    lines.append("| 类型 | 数量 | 占失败比 |")
    lines.append("|---|---:|---:|")
    for t, c in type_counter.most_common(top_n):
        lines.append(f"| {t} | {c} | {c / max(len(failures), 1):.1%} |")
    lines.append("")

    lines.append("## 语义主题分布（失败集中在哪些业务问题上）")
    lines.append("")
    lines.append("| 主题 | 数量 | 占失败比 |")
    lines.append("|---|---:|---:|")
    for g, c in group_counter.most_common(top_n):
        lines.append(f"| {g} | {c} | {c / max(len(failures), 1):.1%} |")
    lines.append("")

    lines.append("## 典型失败样本")
    lines.append("")
    for t, c in type_counter.most_common(6):
        lines.append(f"### {t}（{c} 次）")
        lines.append("")
        for r in type_examples[t]:
            lines.append(f"- **[{r.get('id')}]** {r.get('query', '')[:60]}")
            lines.append(f"  - 模型: {r.get('model_answer', '')[:120]}")
            detail = r.get("rule", {}).get("checks", {})
            bad = [f"{k}: {v['detail']}" for k, v in detail.items()
                   if not v.get("passed")]
            lines.append(f"  - 问题: {'; '.join(bad[:3])}")
        lines.append("")

    lines.append("## 难度 × 意图 交叉（失败次数）")
    lines.append("")
    if cross:
        lines.append("| 难度 \\ 意图 | " +
                     " | ".join(sorted({k[1] for k in cross})) + " |")
        lines.append("|---|" + "---:|" * len({k[1] for k in cross}))
        for d in ("L1", "L2", "L3", "L4"):
            intents = sorted({k[1] for k in cross})
            row = f"| {d} | " + " | ".join(str(cross.get((d, i), 0))
                                          for i in intents) + " |"
            lines.append(row)
    lines.append("")

    # --- 数据补充建议（自动生成，人工复核）---
    lines.append("## 下一轮数据补充建议")
    lines.append("")
    lines.append("根据上面的分布，按优先级排序：")
    lines.append("")
    prio = 1
    for g, c in group_counter.most_common(3):
        lines.append(f"{prio}. **{g}** —— 失败 {c} 次。"
                     f"建议补充 300–500 条该主题的 SFT 样本，"
                     f"并从中构造 200 对偏好数据。")
        prio += 1
    for t, c in type_counter.most_common(2):
        if t in ("未拒答", "过度承诺", "PII 泄露"):
            lines.append(f"{prio}. **{t}** —— 这是安全问题，不是能力问题。"
                         f"优先用规则化奖励（src/train/rewards.py）+ 拒答数据修。")
            prio += 1
    lines.append("")
    lines.append("> 提示：把上面 3–5 条抄进 `progress/daily-log.md`，")
    lines.append("> 作为下一周的数据任务清单。")

    out_md = out_dir / "error_analysis.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 报告 → {out_md}")

    # --- 输出 bad_cases.jsonl（供 DPO 用）---
    bad_path = out_dir / "bad_cases.jsonl"
    with open(bad_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps({
                "id": r.get("id"),
                "query": r.get("query"),
                "images": r.get("images", []),
                "model_answer": r.get("model_answer"),
                "reference": r.get("reference"),
                "error_type": ",".join(classify(r)) or "unknown",
                "semantic_group": semantic_group(r),
                "difficulty": r.get("difficulty"),
                "intent": r.get("intent"),
            }, ensure_ascii=False) + "\n")
    print(f"✓ Bad cases → {bad_path}  ({len(failures)} 条)")
    print(f"  下一步（Day 26）：python -m src.train.dpo_loss "
          f"--from-badcases {bad_path}")
    print(f"  （注意：bad_cases 里的 reference 需要有参考答案才能构造偏好对）")

    print("\n" + "=" * 70)
    print("失败模式 Top 5（语义主题）")
    print("=" * 70)
    for g, c in group_counter.most_common(5):
        bar = "█" * int(40 * c / max(len(failures), 1))
        print(f"  {g:<12} {c:>4}  {bar}")

    return {
        "n_total": len(records), "n_fail": len(failures),
        "error_types": dict(type_counter), "semantic_groups": dict(group_counter),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="错误分析")
    ap.add_argument("--in", dest="inp", required=True,
                    help="run_eval.py 产出的 *_raw.jsonl")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    analyze(args.inp, args.out_dir)
