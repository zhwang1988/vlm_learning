"""
幻觉评测与缓解。

对应讲义 docs/08-evaluation.md「幻觉的三种类型」，对应计划 Day 22。

三种幻觉：
  存在性（existence）  —— 图里没有口袋，模型说有       ← 最严重也最容易测
  属性（attribute）    —— 图里是深蓝，模型说藏青
  关系（relation）     —— 说「左边的模特穿着」但图里没人

测法：POPE 式。问 10 个「图里有没有 X」，其中一半 X 不存在。
回答「有」的比例就是幻觉率。

四招缓解，这四种策略的对比实验就是 Day 22 的核心产出：
  ① Prompt 约束   ② 引用证据   ③ 拒答数据   ④ contrastive DPO
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 幻觉探测集构造
# ---------------------------------------------------------------------------

# 常见的「容易编造」的物体/属性词表
PLAUSIBLE_OBJECTS = [
    "口袋", "拉链", "纽扣", "腰带", "帽子", "围巾", "手套", "logo", "刺绣",
    "蕾丝", "荷叶边", "开衩", "肩垫", "内衬", "polo 领", "V 领", "立领",
    "拼接", "撞色", "印花", "格子", "条纹", "亮片",
]

PLAUSIBLE_ATTRIBUTES = {
    "颜色": ["米白", "藏青", "燕麦色", "雾霾蓝", "砖红", "豆绿", "藕粉"],
    "材质": ["羊毛", "羊绒", "真丝", "亚麻", "腈纶", "涤纶", "天丝"],
}


@dataclass
class ProbeQuestion:
    qid: str
    image: str
    question: str
    ground_truth: bool        # True = 图里真的有；False = 图里没有（探针）
    probe_type: str           # existence / attribute
    target: str


def build_probes(samples: list[dict], n_per_image: int = 10,
                 n_images: int = 30, seed: int = 42,
                 visible_objects_map: Optional[dict] = None) -> list[ProbeQuestion]:
    """构造 POPE 式探测问题。

    visible_objects_map: {image_path: [图里确实有的东西]}。
      有的话用真实标注；没有的话用启发式（随机一半设为「有」）。

    ⚠️ 生产环境必须用真实标注或检测器，否则测出来的「幻觉率」是假的。
       这里给出的启发式版本只用于跑通流程。
    """
    random.seed(seed)
    probes: list[ProbeQuestion] = []
    imgs = [s for s in samples if s.get("image_path") or s.get("images")]
    random.shuffle(imgs)
    imgs = imgs[:n_images]

    for s in imgs:
        img = s.get("image_path") or (s.get("images") or [""])[0]
        visible = (visible_objects_map or {}).get(img, [])

        n_pos = n_per_image // 2
        n_neg = n_per_image - n_pos

        # 正样本：图上确实有的
        for obj in (visible[:n_pos] if visible else
                    random.sample(PLAUSIBLE_OBJECTS, n_pos)):
            probes.append(ProbeQuestion(
                qid=f"{Path(img).stem}_pos_{obj}",
                image=img,
                question=f"这张图片里的这件衣服有{obj}吗？请只回答「有」或「没有」。",
                ground_truth=True, probe_type="existence", target=obj,
            ))

        # 负样本：图上不应有的（探针）
        candidates = [o for o in PLAUSIBLE_OBJECTS if o not in visible]
        for obj in random.sample(candidates, min(n_neg, len(candidates))):
            probes.append(ProbeQuestion(
                qid=f"{Path(img).stem}_neg_{obj}",
                image=img,
                question=f"这张图片里的这件衣服有{obj}吗？请只回答「有」或「没有」。",
                ground_truth=False, probe_type="existence", target=obj,
            ))

    return probes


# ---------------------------------------------------------------------------
# 幻觉率计算
# ---------------------------------------------------------------------------


YES = ["有", "是的", "存在", "可以看到有", "的确有"]
NO = ["没有", "无", "不存在", "未见", "看不到"]


def parse_yes_no(answer: str) -> Optional[bool]:
    """从回答里解析是否/否。返回 True/False/None（无法判断）。"""
    a = answer.strip()
    # 先看开头几个字，避免长回答里的干扰
    head = a[:20]
    for y in YES:
        if head.startswith(y):
            return True
    for n in NO:
        if head.startswith(n):
            return False
    # 兜底：全文找
    has_yes = any(y in a for y in YES)
    has_no = any(n in a for n in NO)
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def compute_hallucination_rate(probes: list[ProbeQuestion],
                               answers: list[str]) -> dict:
    """算幻觉率。

    三个关键指标：
      hallucination_rate  负样本上回答「有」的比例  ← 主指标，越低越好
      miss_rate           正样本上回答「没有」的比例（漏报）
      unparseable         无法解析的比例

    对外报告时**一定要带上 hallucination_rate 和 miss_rate 两个**，
    因为「一律说没有」可以把幻觉率压到 0，但 miss_rate 会爆。
    """
    assert len(probes) == len(answers)

    pos_total = pos_yes = 0
    neg_total = neg_yes = 0
    unparse = 0
    details = []

    for p, a in zip(probes, answers):
        yn = parse_yes_no(a)
        if yn is None:
            unparse += 1
            continue
        if p.ground_truth:
            pos_total += 1
            if yn:
                pos_yes += 1
        else:
            neg_total += 1
            if yn:
                neg_yes += 1
                details.append({"qid": p.qid, "question": p.question,
                                "answer": a[:80], "phantom": p.target})

    return {
        "hallucination_rate": neg_yes / max(neg_total, 1),
        "accuracy_on_positive": pos_yes / max(pos_total, 1),
        "miss_rate": 1 - pos_yes / max(pos_total, 1),
        "unparseable_rate": unparse / max(len(probes), 1),
        "n_probes": len(probes),
        "n_negative": neg_total,
        "n_positive": pos_total,
        "hallucination_examples": details[:15],
    }


# ---------------------------------------------------------------------------
# 四种缓解策略（Day 22 的对比实验）
# ---------------------------------------------------------------------------

MITIGATION_STRATEGIES = {
    "baseline": "",

    "prompt_constraint": (
        "\n\n【重要】只基于图片中**可见**的内容回答。"
        "如果图中看不到某个细节，必须回答「没有」或「图中看不到」，"
        "绝对不能根据常识或经验推测。"
    ),

    "evidence_first": (
        "\n\n【回答方式】请先描述你在图中**实际看到**的内容（一行），"
        "然后基于这个描述回答问题。不要描述你没看到的东西。"
    ),

    "uncertainty": (
        "\n\n【重要】如果图片不够清晰或角度看不到，"
        "请明确说明「图中看不到，无法确认」，不要猜测。"
        "不确定时说不知道比答错更好。"
    ),
}


def build_mitigation_prompt(question: str, strategy: str) -> str:
    return question + MITIGATION_STRATEGIES.get(strategy, "")


async def run_mitigation_experiment(
    probes: list[ProbeQuestion],
    infer_fn,                       # async (image_path, question) -> answer
    strategies: list[str] | None = None,
    limit: int = 200,
) -> dict:
    """对比不同缓解策略的效果。Day 22 的核心产出。

    infer_fn 由你传入（可以调本地模型或 API）。
    """
    strategies = strategies or list(MITIGATION_STRATEGIES.keys())
    probes = probes[:limit]
    results = {}

    for strat in strategies:
        print(f"  策略 {strat} ...")
        answers = await asyncio.gather(*[
            infer_fn(p.image, build_mitigation_prompt(p.question, strat))
            for p in probes
        ])
        res = compute_hallucination_rate(probes, list(answers))
        results[strat] = res
        print(f"    幻觉率 {res['hallucination_rate']:.1%}  "
              f"漏报率 {res['miss_rate']:.1%}")

    # 汇总表
    print("\n" + "=" * 74)
    print("幻觉缓解策略对比")
    print("=" * 74)
    print(f"{'策略':<22} {'幻觉率':>10} {'漏报率':>10} {'无法解析':>10}")
    print("-" * 74)
    base = results.get("baseline", {}).get("hallucination_rate", 0)
    for s, r in results.items():
        delta = r["hallucination_rate"] - base
        mark = "" if s == "baseline" else (f"  ({delta:+.1%})")
        print(f"{s:<22} {r['hallucination_rate']:>10.1%} "
              f"{r['miss_rate']:>10.1%} {r['unparseable_rate']:>10.1%}{mark}")

    return results


# ---------------------------------------------------------------------------
# 属性幻觉（颜色/材质）
# ---------------------------------------------------------------------------


def build_attribute_probes(samples: list[dict], n: int = 60,
                           seed: int = 42) -> list[ProbeQuestion]:
    """构造属性幻觉探测：问一个颜色/材质，看模型是否编造。

    做法：给一张已知是 A 色的图，问它「是 B 色吗」（B ≠ A）。
    正确回答应该是「不是」。回答「是」就是属性幻觉。
    """
    random.seed(seed)
    probes = []
    pool = [s for s in samples if s.get("image_path")]
    random.shuffle(pool)

    for s in pool[:n // 2]:
        img = s["image_path"]
        attr_kind = random.choice(list(PLAUSIBLE_ATTRIBUTES.keys()))
        values = PLAUSIBLE_ATTRIBUTES[attr_kind]
        a, b = random.sample(values, 2)
        probes.append(ProbeQuestion(
            qid=f"{Path(img).stem}_attr_{b}",
            image=img,
            question=f"这件衣服是{b}色的吗？",
            ground_truth=False, probe_type="attribute", target=b,
        ))
    return probes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="幻觉评测")
    ap.add_argument("--samples", default="data/processed/clean.jsonl")
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument("--out", default="data/eval/hallucination_probes.jsonl")
    args = ap.parse_args()

    print("=" * 74)
    print("幻觉探测集构造（POPE 式）")
    print("=" * 74)

    p = Path(args.samples)
    if not p.exists():
        print(f"\n✗ 找不到 {p}")
        print("  Day 22 之前你需要先跑完 Day 12 的数据构建。")
        print("\n  不过这个模块的逻辑可以先理解：")
        print("    - 一半问题问「图里有的东西」→ 测漏报率")
        print("    - 一半问题问「图里没有的东西」→ 测幻觉率")
        print("    - 只报告幻觉率是作弊（全说「没有」就能刷到 0）")
        print("    - 必须同时报告漏报率")
        raise SystemExit(0)

    samples = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    probes = build_probes(samples, n_images=args.n_images)
    probes += build_attribute_probes(samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for pr in probes:
            f.write(json.dumps(pr.__dict__, ensure_ascii=False) + "\n")

    n_pos = sum(1 for x in probes if x.ground_truth)
    print(f"\n✓ 构造 {len(probes)} 条探测（正样本 {n_pos} / 负样本 {len(probes) - n_pos}）")
    print(f"  → {out}")
    print()
    print("⚠️ 重要：这里的正/负样本是启发式生成的。")
    print("   要得到可信的幻觉率，必须用真实标注或检测器给出 visible_objects。")
    print("   否则你测的是「模型的回答和随机标注的一致性」，没有意义。")
    print()
    print("Day 22 的对比实验：")
    print("  strategies = baseline / prompt_constraint / evidence_first / uncertainty")
    print("  跑完后按「幻觉率 - 漏报率」权衡选出生产配置。")
