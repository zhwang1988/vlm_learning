#!/usr/bin/env python3
"""
生成 Day 1–4 与 Day 13–48 的每日讲义与 notebook。

    python scripts/gen_days.py             # 全部生成 + 重建 PLAN.md 索引 + days/README.md
    python scripts/gen_days.py --week 5    # 只重生第 5 周
    python scripts/gen_days.py --check     # 只校验，不写文件

产物：
    days/day-01.md ... day-04.md
    days/day-13.md ... day-48.md
    notebooks/day-01_*.ipynb ... day-48_*.ipynb
    PLAN.md 里的 📄/📓 索引行（幂等）
    days/README.md 总表（自动重建）

Day 5–12（W2 数据工程）不在这里 —— 那 8 天是早期手写的，直接躺在 days/ 里。

**别手改生成物** —— 改 scripts/daygen/w*.py 后重跑。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from daygen import ALL_DAYS                                   # noqa: E402
from daygen.render import (DAYS_DIR, NB_DIR, WHERE,           # noqa: E402
                           md_name, nb_name, patch_plan_index,
                           render_days_readme, render_md, render_nb,
                           write_all)


def check() -> int:
    """不写文件，只校验内容完整性。"""
    problems = []
    seen = set()
    for d in ALL_DAYS:
        for field in ("n", "slug", "title", "where", "goal", "read", "think",
                      "write_rows", "run", "accept", "pits"):
            if not d.get(field):
                problems.append(f"day-{d.get('n')}: 缺字段 {field}")
        if d["where"] not in WHERE:
            problems.append(f"day-{d['n']}: where 值非法 {d['where']}")
        if d["n"] in seen:
            problems.append(f"day-{d['n']}: 重复")
        seen.add(d["n"])
        if len(d["pits"]) < 3:
            problems.append(f"day-{d['n']}: 坑少于 3 条")
        if len(d["accept"]) < 3:
            problems.append(f"day-{d['n']}: 验收项少于 3 条")

    # daygen 负责的天：Day 1–4（W1）+ Day 13–48（W3–W8）。
    # Day 5–12（W2 数据工程）是早期手写的，直接躺在 days/ 里，不由这里管 ——
    # 所以它们「缺失」是正常的，别把它们算进必检集合。
    owned = {1, 2, 3, 4} | set(range(13, 49))
    missing = sorted(owned - seen)
    extra = sorted(seen - owned)
    if missing:
        problems.append(f"缺这些天：{missing}")
    if extra:
        problems.append(f"多出这些天：{extra}")

    print(f"共 {len(ALL_DAYS)} 天：{sorted(seen)}")
    print(f"（另有 Day 5–12 由手写材料提供，不在此列）")
    if problems:
        print(f"\n✗ {len(problems)} 个问题：")
        for p in problems:
            print("   ", p)
        return 1
    print("✓ 内容完整性校验通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="生成每日讲义与 notebook")
    ap.add_argument("--week", type=int, default=0,
                    help="只生成某一周（1 = Day 1–4，3–8 = 其余）")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    ap.add_argument("--no-index", action="store_true", help="不重建 PLAN.md / days/README.md")
    args = ap.parse_args()

    rc = check()
    if args.check or rc:
        return rc

    days = [d for d in ALL_DAYS if not args.week or d["week"] == args.week]
    if not days:
        print(f"✗ 第 {args.week} 周没有内容")
        return 1

    print(f"\n生成 {len(days)} 天 → {DAYS_DIR.name}/ + {NB_DIR.name}/")
    stats = write_all(days)

    if not args.no_index and not args.week:
        n_ins = patch_plan_index(ALL_DAYS)
        (DAYS_DIR / "README.md").write_text(
            render_days_readme(ALL_DAYS), encoding="utf-8")
        print(f"\n  ✓ PLAN.md 索引行：{n_ins} 条（Day 1–4 与 13–48）")
        print("  ✓ days/README.md 总表已重建")

    print(f"\n完成：{stats['md']} 个 md，{stats['nb']} 个 notebook"
          f"（共 {stats['cells']} 个 cell）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
