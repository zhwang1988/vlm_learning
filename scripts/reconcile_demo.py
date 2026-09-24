#!/usr/bin/env python3
"""演示数据对账：跑一遍清洗流水线，拿 `*_labels.jsonl` 的 ground truth 逐条核对。

为什么需要这个脚本
------------------
`make data-clean` 会打印「① 格式校验后 151 条 / ② 规则过滤后 144 条 /
④ 去重后 136 条」。这些数字**对不对**？光看是看不出来的。

  · 删多了 → 把有效样本当垃圾删了，报告上只表现为数字变小
  · 删少了 → 垃圾留在训练集里，报告上只表现为数字变大

两种错误都不会报错、不会崩、不会有告警。能区分它们的只有 ground truth。
本脚本就是拿 `make_demo_data.py` 造数据时写下的 label 去逐条对账：

    每个阶段，每条样本，是「该留的留了」还是「该删的删了」。

三种断言
--------
1. **逐条**：`expect` 是 format/rules/dedup/keep 的，id 必须落在对应阶段。
2. **组级**：`expect` 是 `dedup_group` 的，组内必须**净剩 1 条**。
   重复组的 ground truth 只能是集合级性质 —— 去重保留先出现的、删后出现的，
   而样本顺序是打乱的，所以「哪条 id 被删」本身就是不确定的。
   硬按 id 断言会得出「1 条该删却留 + 1 条该留却删」这种计数正确、结论错误的结果。
3. **阶段归属**：一条标了 `dedup` 的样本如果在 ① 就被拦下，也算 FAIL。
   「被删了」和「被**正确的**原因删了」是两件事。

退出码 0 = 全部对上；1 = 有出入（可直接进 CI）。

用法
----
    python scripts/reconcile_demo.py                    # 用默认路径
    python scripts/reconcile_demo.py --data X.jsonl --labels X_labels.jsonl
    make demo-check                                      # 等价的 make 目标
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dedup import run_stages          # noqa: E402

STAGE_ORDER = ("input", "format", "rules", "clip", "dedup")
STAGE_CN = {"format": "① 格式校验", "rules": "② 规则过滤",
            "clip": "③b 图文一致性", "dedup": "④ 去重", "keep": "保留"}


def _read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="演示数据 ground truth 对账")
    ap.add_argument("--data", default="data/fixtures/demo_synth.jsonl")
    ap.add_argument("--labels", default="data/fixtures/demo_synth_labels.jsonl")
    ap.add_argument("--verbose", action="store_true",
                    help="打印清洗流水线的每步统计")
    args = ap.parse_args()

    data_path, label_path = Path(args.data), Path(args.labels)
    for p in (data_path, label_path):
        if not p.exists():
            print(f"✗ 找不到 {p}")
            print("  先跑： python scripts/make_demo_data.py")
            return 2

    samples = _read_jsonl(data_path)
    labels = {r["id"]: r for r in _read_jsonl(label_path)}

    print("=" * 76)
    print("演示数据对账")
    print("=" * 76)
    print(f"数据 {data_path} （{len(samples):,} 条）")
    print(f"label {label_path} （{len(labels):,} 条）")
    print()

    st = run_stages(samples, verbose=True)

    # ---------- 断言 1：逐条 ----------
    rows_ok = 0
    per_item_bad: list[str] = []
    for sid, lb in labels.items():
        expect, actual = lb["expect"], st.stage_of(sid)
        if expect == "dedup_group":
            continue                       # 组级断言在下面单独做
        if expect == actual:
            rows_ok += 1
        else:
            per_item_bad.append(
                f"{lb['kind']:<26} id={sid}  期望={STAGE_CN.get(expect, expect)}"
                f"  实际={STAGE_CN.get(actual, actual)}"
                + (f"  原因={st.reasons.get(sid, '?')}" if actual != "keep" else ""))

    # ---------- 断言 2：组级（净剩 1 条）----------
    groups: dict[str, list[str]] = defaultdict(list)
    for sid, lb in labels.items():
        if lb.get("group"):
            groups[lb["group"]].append(sid)

    group_ok = 0
    group_bad: list[str] = []
    for g, members in sorted(groups.items()):
        survivors = [m for m in members if st.stage_of(m) == "keep"]
        dropped = [m for m in members if st.stage_of(m) != "keep"]
        if len(survivors) == 1 and len(dropped) == len(members) - 1:
            group_ok += 1
        else:
            detail = ", ".join(
                f"{m}:{STAGE_CN.get(st.stage_of(m), st.stage_of(m))}" for m in members)
            group_bad.append(
                f"{g}（{len(members)} 条）应净剩 1 条，实际剩 {len(survivors)} 条"
                f"  [{detail}]")

    # ---------- 汇总 ----------
    print()
    print("-" * 76)
    print("① 逐条断言（format / rules / dedup / keep）")
    print(f"   {rows_ok:,} / {rows_ok + len(per_item_bad):,} 条对上")
    for line in per_item_bad[:15]:
        print(f"   ✗ {line}")
    if len(per_item_bad) > 15:
        print(f"   … 另有 {len(per_item_bad) - 15} 条")

    print()
    print("② 组级断言（重复组应净剩 1 条）")
    print(f"   {group_ok:,} / {group_ok + len(group_bad):,} 组对上")
    for line in group_bad[:15]:
        print(f"   ✗ {line}")

    print()
    print("③ 关键语义：不该被删的有没有被删")
    must_keep_kinds = ("valid_same_img_diff_text", "valid_same_text_diff_img",
                       "near_dup_punctuation", "real_black_image")
    lost: list[str] = []
    for sid, lb in labels.items():
        if lb["kind"] in must_keep_kinds and st.stage_of(sid) != "keep":
            lost.append(f"{lb['kind']:<26} id={sid}  在 {STAGE_CN[st.stage_of(sid)]} "
                        f"被删（原因 {st.reasons.get(sid, '?')}）")
    n_must = sum(1 for lb in labels.values() if lb["kind"] in must_keep_kinds)
    print(f"   {n_must - len(lost):,} / {n_must:,} 条有效样本被正确保留")
    for line in lost[:15]:
        print(f"   ✗ {line}")

    print()
    print("阶段水量（每一步之后剩多少）")
    for s in STAGE_ORDER[1:]:
        print(f"   {STAGE_CN[s]:<12} → {len(st.ids_at(s)):>4} 条")

    n_bad = len(per_item_bad) + len(group_bad) + len(lost)
    print()
    print("=" * 76)
    if n_bad:
        print(f"✗ 对账未通过：{n_bad} 处不一致")
        print("  逐条/组级不一致说明清洗逻辑或 label 有一方错了，"
              "不要放过 —— 这类问题不会自己变好。")
        return 1
    print("✓ 对账通过：清洗流水线的行为与 ground truth 完全一致")
    print("  （含「同图不同文 / 同文不同图 / 全黑图 / 近似重复」四类有效样本均未被误删）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
