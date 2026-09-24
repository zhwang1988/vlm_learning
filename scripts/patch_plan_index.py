#!/usr/bin/env python3
"""把 PLAN.md 变成总纲索引：每个 Day 条目下加一行指向 days/day-XX.md + notebook。"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
plan = ROOT / "PLAN.md"

NB = {
    5: "day-05_minivlm_assembly.ipynb",
    6: "day-06_full_inference_review.ipynb",
    7: "day-07_image_preprocess.ipynb",
    8: "day-08_taxonomy_matrix.ipynb",
    9: "day-09_data_synthesis.ipynb",
    10: "day-10_cleaning_dedup.ipynb",
    11: "day-11_build_sft.ipynb",
    12: "day-12_dataset_card.ipynb",
}

lines = plan.read_text().splitlines()
out = []
inserted = 0
for ln in lines:
    out.append(ln)
    m = re.match(r"^### Day (\d+) —", ln)
    if m:
        n = int(m.group(1))
        if n in NB:
            link = f"> 📄 `days/day-{n:02d}.md`　·　📓 `notebooks/{NB[n]}`"
            out.append("")
            out.append(link)
            inserted += 1

plan.write_text("\n".join(out) + "\n")
print(f"已插入 {inserted} 条 day 索引（Day 5–12）")
