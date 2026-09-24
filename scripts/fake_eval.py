#!/usr/bin/env python3
"""造假预测：不加载任何模型，直接往评测链路里喂「已知好坏」的答案。

为什么需要它
------------
Day 22 的 `run_eval` 要加载 Qwen2.5-VL，没 GPU 跑不动；Day 23 的
`error_analysis`、`report` 又依赖 `run_eval` 的产物。结果是：一台干净的
笔记本上，评测链路的**后半段完全无法验证**，日文件里的命令全是悬空的。

本脚本按 `run_eval` 的真实输出格式造假 raw.jsonl，并且造**两种已知质量**的
预测，用来回答一个最要紧的问题：

    **这套评测指标，能不能把好模型和坏模型区分开？**

如果两种预测跑出来的分数差不多，那说明指标是坏的 —— 这种情况在你真的租了
4090、训了一轮之后才会发现，代价是整个第 4 周的实验全部作废。
在这里花 3 秒抓到，比在第 4 周花 3 天抓到划算得多。

三种造法（`--mode`）
--------------------
  good    **oracle（探针）**：直接按评测样本自己的 must_contain / must_not_contain /
          should_refuse 拼出一个「必然满足全部规则」的回答。
          它不模拟任何真实模型，用途只有一个：**证明这套评测集是可被通过的**。
          如果连 oracle 都过不了，那说明评测集里有自相矛盾/不可达的样本 ——
          这种问题在真实评测里只会表现为「模型怎么训都上不去」，
          你会去调模型、调超参、加数据，而真正的问题在评测集里。
  bad     典型的坏模型：编造图片里没有的东西 + 空口保证 + 硬答该澄清的问题
  silent  极端坏模型：一律回答「我无法确定，建议您咨询人工客服」。
          **用来验伪指标设计**：这个什么都不干的模型幻觉率是 0%，
          但漏答率极高。如果报告只看幻觉率，它会拿满分。
          「只报幻觉率不报漏答率」是幻觉评测最经典的陷阱。

用法
----
    python scripts/fake_eval.py --mode good --tag fake_good
    python scripts/fake_eval.py --mode bad  --tag fake_bad
    python scripts/fake_eval.py --mode silent --tag fake_silent

    make eval-fake          # 三种都造 + 跑 error_analysis + 合成对比报告
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.metrics import evaluate_rules       # noqa: E402


# ---------------------------------------------------------------------------
# 三种「模型」
# ---------------------------------------------------------------------------

# 坏模型爱编的东西：图片里根本没有的属性
FABRICATED = [
    "这件是意大利进口面料，采用顶级的埃及长绒棉，手感非常细腻亲肤",
    "这款的领口是手工缝制的珍珠扣，一共有五颗，非常精致",
    "图片里能看到袖口有一圈蕾丝花边，做工很考究",
    "这个是 2026 春夏限定款，全国只有我们家在卖",
]

# 空口保证（命中 over_promise）
OVER_PROMISE = [
    "您放心，绝对不会起球，保证不会缩水，可以穿十年",
    "我们的质量百分之百没问题，一定可以穿很久",
]

# 合格的回答：会带上一些关键词
GOOD_TEMPLATES = [
    "从图片看这是米白色的纯棉圆领T恤，版型是宽松直筒的，袖长是短袖。"
    "胸围约 100cm、衣长约 68cm，模特身高 165cm 穿的是 M 码。"
    "尺码表里最大的码是 XXL，日常穿 M 的话这件是合适的。",
    "图片里这个线头是缝制收口处的，属于正常工艺范围，剪掉即可，"
    "不影响穿着，也不属于质量问题。如果您介意，可以申请七天无理由退换。",
    "图片和实物的色差主要来自光线和显示器差异，我们拍摄用的是自然光。"
    "如果您收到的实物差异明显，可以拍照给我，我帮您核对色号并安排换货。",
    "关于是否起球，我只能依据标注信息回答：这款是 100% 精梳棉、220g 克重，"
    "正常穿着不会起球。触感这类主观感受我无法确定，建议参考详情页说明。",
    "这两张图我都看了，对比下来款式和颜色都有差异：第一张偏米白，"
    "第二张更接近雾霾蓝。如果差异明显，可以按七天无理由退换处理。",
]

# 该澄清/拒答时的合格回答（命中 refusal 标记）
GOOD_REFUSAL = (
    "这个问题我还无法确定，方便告诉我您的身高体重吗？"
    "我按尺码表帮您算一下，这样更准确。"
)

# 坏模型的「硬答」
BAD_REFUSAL = "我穿 M 应该是可以的，您直接下单 M 码就行。"

# silent 模式唯一的回答
SILENT_ANSWER = "我无法确定，建议您咨询人工客服。"


def _oracle_answer(sample: dict) -> str:
    """拼一个「必然满足该样本全部规则」的回答。

    这就是个探针，故意写得机械：把 must_contain 的字面关键词都塞进去，
    回避 must_not_contain，该澄清就澄清。
    判断标准只有一个：**规则判定必须全部通过。过不了就是评测集的问题。**
    """
    must = sample.get("must_contain") or []
    forbid = set(sample.get("must_not_contain") or [])

    if sample.get("should_refuse"):
        body = "这个问题我还需要更多信息才能判断，方便告诉我您的身高体重吗？" \
               "如果您希望，我也可以帮您转接人工客服，由人工为您确认。"
    elif must:
        # 把关键词用自然一点的方式串起来（关键词都是字面词，可以这样拼）
        body = ("根据图片和商品信息，"
                + "、".join(k for k in must if k not in forbid)
                + "。如果您还有别的疑问，随时问我。")
    else:
        body = ("从图片来看，这和您描述的情况基本一致。"
                "如果需要进一步确认，可以拍一张自然光下的实拍图发我，我帮您核对。")

    # 最后兜一道：任何 must_not_contain 的词都不许出现在回答里。
    # 正常构造不会触发；触发了说明模板和禁含词打架，属于评测集问题。
    for f in forbid:
        if f in body:
            raise SystemExit(
                f"✗ 评测样本 {sample.get('id')} 的 must_contain 与 must_not_contain "
                f"打架：'{f}' 两边都有。这条样本不可能被通过，是评测集的问题。")
    return body


def _clean(text: str, forbid: list[str]) -> str:
    """确保文本里没有任何 must_not_contain 词（坏模型也不该违反禁含词）。"""
    for f in forbid:
        text = text.replace(f, "（不作评价）")
    return text


def _make_answer(mode: str, sample: dict, rng: random.Random) -> str:
    should_refuse = bool(sample.get("should_refuse"))
    forbid = sample.get("must_not_contain") or []

    if mode == "good":
        return _oracle_answer(sample)
    if mode == "silent":
        return SILENT_ANSWER
    # bad
    if should_refuse:
        return _clean(BAD_REFUSAL, forbid)
    parts = [rng.choice(GOOD_TEMPLATES), rng.choice(FABRICATED)]
    if rng.random() < 0.4:
        parts.append(rng.choice(OVER_PROMISE))
    return _clean("".join(parts), forbid)


def build(eval_path: Path, mode: str, seed: int, limit: int = 0) -> list[dict]:
    rng = random.Random(seed)
    samples = [json.loads(l) for l in eval_path.read_text("utf-8").splitlines() if l.strip()]
    if limit:
        samples = samples[:limit]

    records = []
    for s in samples:
        ans = _make_answer(mode, s, rng)
        rule = evaluate_rules(s, ans)
        # judge 字段留空：真实 run_eval 里由 LLM 填，离线时如实留空，
        # 不要编造假分数 —— 编了会让报告里的 judge 维度变成假数据。
        records.append({
            "id": s.get("id", ""),
            "difficulty": s.get("difficulty", "?"),
            "intent": s.get("intent", "?"),
            "image_type": s.get("image_type", "?"),
            "query": s.get("query", ""),
            "images": s.get("images", []),
            "model_answer": ans,
            "latency_ms": rng.randint(300, 1800),
            "rule": rule,
            "judge": {},
            "error": None,
            "_fake": True,
            "_mode": mode,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="造假预测，验证评测链路的区分度")
    ap.add_argument("--eval", default="data/eval/cx_eval_v1.jsonl")
    ap.add_argument("--mode", choices=["good", "bad", "silent"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ev = Path(args.eval)
    if not ev.exists():
        print(f"✗ 找不到评测集 {ev}，先跑 make domain-eval")
        return 2

    tag = args.tag or f"fake_{args.mode}"
    recs = build(ev, args.mode, args.seed, args.limit)

    out = Path(args.out_dir) / f"eval_{tag}_raw.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pass = sum(1 for r in recs if r["rule"]["passed"])
    avg = sum(r["rule"]["score"] for r in recs) / max(len(recs), 1)
    print(f"✓ 造假预测 {len(recs)} 条（mode={args.mode}）→ {out}")
    print(f"  规则通过率 {n_pass / max(len(recs), 1):.1%} · 规则综合分 {avg:.3f}")

    if args.mode == "good" and n_pass != len(recs):
        bad = [r for r in recs if not r["rule"]["passed"]]
        print()
        print(f"✗ oracle 模式竟然有 {len(bad)} 条没过 —— "
              f"**这是评测集的问题，不是模型的问题**")
        for r in bad[:6]:
            failed = [k for k, v in r["rule"]["checks"].items() if not v["passed"]]
            print(f"    {r['id']} [{r['difficulty']}] 失败项={failed}")
            for k in failed:
                print(f"      {k}: {r['rule']['checks'][k]['detail'][:90]}")
        print("  修评测集（模板关键词 / 禁含词 / 拒答要求），别去调模型。")
        return 1

    if args.mode == "good":
        print("  ✓ 全部通过 —— 评测集是可被通过的（没有自相矛盾的样本）")

    print()
    print("⚠️ 这是**伪造**的预测，只用来验证评测链路本身，"
          "不能当作模型效果的任何证据。")
    print(f"  下一步：python -m src.eval.error_analysis --in {out} --out-dir reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
