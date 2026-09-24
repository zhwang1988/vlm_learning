#!/usr/bin/env python3
"""
通用 VLM 榜单子集评测。

    python -m src.eval.run_benchmark --list
    python -m src.eval.run_benchmark --model Qwen/Qwen2.5-VL-3B-Instruct --suite ocrbench --n 100

**重要：这是子集评测，不是官方全量榜单成绩。**
报告里会显式标注 `子集 N/M`，不允许拿它去和榜单数字比 —— 那是学术诚信问题。
同理，本文件内置的 mini fixture **只用于验证流程跑通**，不能当分数。

依赖：需要 GPU（跑模型）。`--list` 不需要。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# 榜单注册表
# ---------------------------------------------------------------------------

SUITES: dict[str, dict] = {
    "ocrbench": {
        "desc": "OCR 文字识别：图中文字读取与理解",
        "metric": "归一化编辑距离 → 近似准确率",
        "official_n": 1000,
        "kind": "ocr",
    },
    "mmbench": {
        "desc": "综合视觉理解：多选题为主",
        "metric": "选项精确匹配",
        "official_n": 3000,
        "kind": "choice",
    },
    "pope": {
        "desc": "物体存在性幻觉探测：问图里有没有某物",
        "metric": "存在/不存在 二分类准确率",
        "official_n": 3000,
        "kind": "yesno",
    },
    "docvqa": {
        "desc": "文档问答：发票/表单/说明书",
        "metric": "归一化编辑距离",
        "official_n": 5000,
        "kind": "ocr",
    },
}

# 内置 mini fixture —— 只够验证「流程通不通」，绝不可当分数用
_FIXTURES: dict[str, list[dict]] = {
    "ocrbench": [
        {"id": "mini-1", "question": "图中的文字是什么？", "answers": ["SALE 50%"],
         "render": "SALE 50%"},
        {"id": "mini-2", "question": "读出图里的价格。", "answers": ["￥199"],
         "render": "￥199"},
        {"id": "mini-3", "question": "图里的型号是什么？", "answers": ["SKU-1001"],
         "render": "SKU-1001"},
    ],
    "mmbench": [
        {"id": "mini-1", "question": "图中是什么形状？A. 圆 B. 方 C. 三角", "answers": ["A"]},
        {"id": "mini-2", "question": "衣服颜色是？A. 白 B. 黑 C. 蓝", "answers": ["A"]},
    ],
    "pope": [
        {"id": "mini-1", "question": "图里有矩形吗？只回答有或没有。", "answers": ["有"]},
        {"id": "mini-2", "question": "图里有飞机吗？只回答有或没有。", "answers": ["没有"]},
    ],
    "docvqa": [
        {"id": "mini-1", "question": "发票金额是多少？", "answers": ["199.00"], "render": "199.00"},
    ],
}


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------

def load_suite(suite: str, n: int = 0, data_dir: str | Path = "data/benchmarks",
               seed: int = 42) -> tuple[list[dict], str]:
    """返回 (样本, 来源说明)。优先读本地缓存；没有就用内置 fixture。"""
    p = Path(data_dir) / f"{suite}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        rng = random.Random(seed)
        rng.shuffle(rows)
        return (rows[:n] if n else rows), f"本地缓存 {p}"
    rows = _FIXTURES.get(suite, [])
    return rows, "⚠️ 内置 mini fixture（只验证流程，不是分数）"


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return "".join(str(s).lower().split())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_one(answer: str, golds: list[str], kind: str) -> float:
    """0–1 的分数。一个样本多个参考答案时取最优。"""
    a = _norm(answer)
    best = 0.0
    for g in golds:
        g = _norm(g)
        if kind in ("choice", "yesno"):
            # 多选题/是非题：答案里含选项即可（模型常多写字）
            hit = 1.0 if (g and g in a) else 0.0
        else:
            d = edit_distance(a, g)
            hit = max(0.0, 1.0 - d / max(len(a), len(g), 1))
        best = max(best, hit)
    return best


SCORERS: dict[str, Callable[[str, list[str]], float]] = {
    k: (lambda a, g, _k=v["kind"]: score_one(a, g, _k)) for k, v in SUITES.items()
}


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

def _make_image(rec: dict):
    """有图就用图；fixture 用 render 字段画一张。"""
    from PIL import Image, ImageDraw
    imgs = rec.get("images") or []
    if imgs and Path(imgs[0]).exists():
        return Image.open(imgs[0]).convert("RGB")
    img = Image.new("RGB", (448, 224), "white")
    if rec.get("render"):
        ImageDraw.Draw(img).text((24, 96), rec["render"], fill=(20, 20, 20))
    return img


def run(model_id: str, suite: str, n: int = 0, data_dir: str = "data/benchmarks",
        max_pixels: int = 1280, out_dir: str = "reports") -> int:
    from .run_eval import LocalVLM          # 需要 torch

    spec = SUITES[suite]
    samples, source = load_suite(suite, n, data_dir)
    if not samples:
        print(f"✗ {suite} 没有可用样本")
        return 1

    print("=" * 78)
    print(f"榜单子集评测  suite={suite}  model={model_id}")
    print(f"来源: {source}")
    print(f"官方全量 {spec['official_n']} 条；本次跑 {len(samples)} 条（**子集**，不可与榜单数字直接比）")
    print("=" * 78)

    engine = LocalVLM(model_id, None, max_pixels)
    scorer = SCORERS[suite]
    scores, rows = [], []
    for i, s in enumerate(samples, 1):
        try:
            ans, ms = engine.generate([_make_image(s)], s["question"], "")
            sc = scorer(ans, s.get("answers", []))
        except Exception as e:
            ans, ms, sc = f"<失败 {type(e).__name__}>", 0, 0.0
        scores.append(sc)
        rows.append({"id": s.get("id"), "question": s["question"],
                     "answers": s.get("answers"), "prediction": ans,
                     "score": round(sc, 3), "latency_ms": ms})
        print(f"  [{i}/{len(samples)}] {sc:.2f}  {str(s['question'])[:40]}")

    acc = sum(scores) / len(scores)
    tag = f"{suite}_{Path(model_id).name}"
    out = Path(out_dir) / f"bench_{tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        f"# 榜单子集评测 · {suite}", "",
        f"- 模型：`{model_id}`",
        f"- 指标：{spec['metric']}",
        f"- 样本：**子集 {len(samples)} / 官方全量 {spec['official_n']}**"
        f"（不是官方成绩，不可直接对比榜单数字）",
        f"- 数据来源：{source}",
        f"- 得分：**{acc:.1%}**", "",
        "| # | 问题 | 参考答案 | 模型回答 | 得分 |", "|---|---|---|---|---|",
        *[f"| {i} | {str(r['question'])[:34].replace('|','／')} | "
          f"{str(r['answers'])[:22].replace('|','／')} | "
          f"{str(r['prediction'])[:40].replace('|','／')} | {r['score']:.2f} |"
          for i, r in enumerate(rows, 1)],
        "", "> 结论只能用来说「同一子集上 A 比 B 好多少」，不要用来宣称「达到 X 分」。", "",
    ]), encoding="utf-8")
    print(f"\n[score] {acc:.1%}")
    print(f"✓ 报告 → {out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="通用 VLM 榜单子集评测")
    ap.add_argument("--list", action="store_true", help="列出支持的榜（不需要 GPU）")
    ap.add_argument("--model", help="模型 ID 或本地路径")
    ap.add_argument("--suite", default="ocrbench", choices=list(SUITES))
    ap.add_argument("--n", type=int, default=100, help="子集大小（0 = 全用）")
    ap.add_argument("--data-dir", default="data/benchmarks")
    ap.add_argument("--max-pixels", type=int, default=1280)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    if args.list:
        print("支持的榜（子集评测，非官方成绩）：\n")
        for k, v in SUITES.items():
            local = Path(args.data_dir) / f"{k}.jsonl"
            mark = f"本地缓存 {local}" if local.exists() else "无本地缓存 → 用内置 mini fixture"
            print(f"  {k:10s} {v['desc']}")
            print(f"             指标: {v['metric']}   官方全量: {v['official_n']}   {mark}")
        print("\n下载官方数据放成 data/benchmarks/<suite>.jsonl 即可用真实子集。")
        return 0

    if not args.model:
        ap.error("--model 是必填（或者用 --list）")
    return run(args.model, args.suite, args.n, args.data_dir,
               args.max_pixels, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
