"""
评测主流程：一条命令跑完全套，出 markdown 报告。

对应计划 Day 21 与 Day 24。

    python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct --tag base
    python -m src.eval.run_eval --model outputs/xxx --adapter outputs/xxx --tag sft

产出 reports/eval_<tag>.md：
  - 领域评测（分难度、分意图）
  - 幻觉指标
  - 通用榜单（防退化检查）
  - 失败模式 Top 5
  - 结论与下一步

**这份报告就是 Day 24 的交付物。**
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalRecord:
    sid: str
    difficulty: str
    intent: str
    image_type: str
    query: str
    images: list
    answer: str
    latency_ms: float
    rule: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# 推理封装
# ---------------------------------------------------------------------------


class LocalVLM:
    """本地模型推理（transformers）。评测时用贪心解码保证可复现。"""

    def __init__(self, model_id: str, adapter: Optional[str] = None,
                 max_pixels_ratio: int = 1280, dtype="bfloat16"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=max_pixels_ratio * 28 * 28,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=getattr(torch, dtype),
            device_map="auto", attn_implementation="sdpa",
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.model.config.use_cache = True      # ⚠️ 推理时必须开回来

    def generate(self, image_paths: list[str], query: str, system: str,
                 max_new_tokens: int = 256) -> tuple[str, float]:
        from PIL import Image
        t0 = time.perf_counter()

        content = []
        images = []
        for p in image_paths:
            try:
                im = Image.open(p).convert("RGB")
                images.append(im)
                content.append({"type": "image", "image": im})
            except Exception:
                continue
        content.append({"type": "text", "text": query})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images or None,
                                return_tensors="pt").to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return ans, (time.perf_counter() - t0) * 1000


class APIvLM:
    """走 API 的推理（用于 base 对比或没有本地模型时）。"""

    def __init__(self, model: Optional[str] = None):
        import os
        from openai import AsyncOpenAI
        self.model = model or os.getenv("SYNTH_MODEL", "qwen-vl-max")
        self.client = AsyncOpenAI(
            base_url=os.getenv("SYNTH_API_BASE"),
            api_key=os.getenv("SYNTH_API_KEY"),
        )

    def generate(self, image_paths, query, system, max_new_tokens=256):
        import base64
        t0 = time.perf_counter()
        content = []
        for p in image_paths:
            try:
                b64 = base64.b64encode(Path(p).read_bytes()).decode()
                suf = Path(p).suffix.lstrip(".").lower()
                mime = "image/jpeg" if suf in ("jpg", "jpeg") else f"image/{suf}"
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except Exception:
                continue
        content.append({"type": "text", "text": query})

        resp = asyncio.run(self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            temperature=0.0, max_tokens=max_new_tokens, timeout=90,
        ))
        return resp.choices[0].message.content, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# 主评测流程
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "你是一位专业的电商客服助手，服务于一家经营服饰鞋包的店铺。"
    "请基于用户提供的图片和文字，给出准确、有帮助、语气自然的回复。"
    "只依据图片中可见的内容和已知的商品信息回答；信息不足时请主动询问，"
    "不要编造商品参数。遇到超出你能力范围的请求（如索赔、投诉、法律问题），"
    "请安抚用户并说明将转接人工客服。"
)


def run_eval(eval_path: str | Path,
             model_id: str,
             adapter: Optional[str] = None,
             tag: str = "base",
             limit: int = 0,
             use_judge: bool = True,
             max_pixels: int = 1280,
             system: str = DEFAULT_SYSTEM) -> list[EvalRecord]:
    from .metrics import evaluate_rules

    samples = [json.loads(l) for l in
               Path(eval_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        samples = samples[:limit]

    print("=" * 78)
    print(f"评测: {tag}  |  模型 {model_id}" + (f" + LoRA {adapter}" if adapter else ""))
    print(f"评测集: {eval_path}  ({len(samples)} 条)")
    print(f"max_pixels: {max_pixels}*28*28")
    print("=" * 78)

    engine = LocalVLM(model_id, adapter, max_pixels)

    records: list[EvalRecord] = []
    for i, s in enumerate(samples):
        try:
            ans, ms = engine.generate(s.get("images", []), s["query"], system)
            rec = EvalRecord(
                sid=s["id"], difficulty=s["difficulty"], intent=s["intent"],
                image_type=s["image_type"], query=s["query"],
                images=s.get("images", []), answer=ans, latency_ms=ms,
                rule=evaluate_rules(s, ans),
            )
        except Exception as e:      # noqa: BLE001
            rec = EvalRecord(
                sid=s["id"], difficulty=s["difficulty"], intent=s["intent"],
                image_type=s["image_type"], query=s["query"],
                images=s.get("images", []), answer="", latency_ms=0,
                error=str(e)[:200],
            )
        records.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)} ...")

    # --- LLM judge ---
    if use_judge:
        try:
            from .judge import Judge
            jd = Judge()
            items = [{"query": r.query, "answer": r.answer or "（空）",
                      "reference": next((s.get("reference") for s in samples
                                         if s["id"] == r.sid), None)}
                     for r in records]
            print(f"\nLLM judge 打分中 ({len(items)} 条)...")
            scores = asyncio.run(jd.score_batch(items))
            for r, sc in zip(records, scores):
                r.judge = sc
        except Exception as e:      # noqa: BLE001
            print(f"⚠ judge 跳过: {e}")

    return records


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def build_report(records: list[EvalRecord], tag: str,
                 baseline: Optional[list[EvalRecord]] = None,
                 out_dir: str | Path = "reports") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def agg(recs: list[EvalRecord]) -> dict:
        ok = [r for r in recs if not r.error]
        n = max(len(ok), 1)
        return {
            "n": len(recs),
            "n_error": len(recs) - len(ok),
            "rule_pass": sum(1 for r in ok if r.rule.get("passed")) / n,
            "rule_score": sum(r.rule.get("score", 0) for r in ok) / n,
            "latency_p50": _pct([r.latency_ms for r in ok], 50),
            "latency_p95": _pct([r.latency_ms for r in ok], 95),
            "judge_acc": _avg([r.judge.get("accuracy") for r in ok if r.judge]),
            "judge_help": _avg([r.judge.get("helpfulness") for r in ok if r.judge]),
            "judge_tone": _avg([r.judge.get("tone") for r in ok if r.judge]),
        }

    cur = agg(records)
    base = agg(baseline) if baseline else None

    lines = [f"# Eval Report: {tag}", ""]
    lines.append(f"- 样本数: **{cur['n']}**" + (f"（{cur['n_error']} 条推理失败）"
                                                if cur["n_error"] else ""))
    lines.append(f"- 规则通过率: **{cur['rule_pass']:.1%}**")
    lines.append(f"- 规则综合分: **{cur['rule_score']:.3f}**")
    if cur["judge_acc"]:
        lines.append(f"- Judge: 事实 {cur['judge_acc']:.2f} / "
                     f"帮助性 {cur['judge_help']:.2f} / 语气 {cur['judge_tone']:.2f}")
    lines.append(f"- 延迟 P50 / P95: {cur['latency_p50']:.0f}ms / {cur['latency_p95']:.0f}ms")
    lines.append("")

    # --- 分难度 ---
    lines.append("## 分难度表现")
    lines.append("")
    lines.append("| 难度 | 样本 | 规则通过率 | 规则分 |" +
                 (" 对比基线 |" if base else ""))
    lines.append("|---|---:|---:|---:|" + ("---|" if base else ""))
    by_diff = defaultdict(list)
    by_diff_base = defaultdict(list)
    for r in records:
        by_diff[r.difficulty].append(r)
    if baseline:
        for r in baseline:
            by_diff_base[r.difficulty].append(r)

    for d in ("L1", "L2", "L3", "L4"):
        rs = by_diff.get(d, [])
        if not rs:
            continue
        a = agg(rs)
        row = f"| {d} | {a['n']} | {a['rule_pass']:.1%} | {a['rule_score']:.3f} |"
        if base:
            b_rs = by_diff_base.get(d, [])
            b = agg(b_rs) if b_rs else {"rule_pass": 0}
            delta = a["rule_pass"] - b["rule_pass"]
            row += f" {delta:+.1%} |"
        lines.append(row)
    lines.append("")

    # --- 分意图 ---
    lines.append("## 分意图表现")
    lines.append("")
    lines.append("| 意图 | 样本 | 规则通过率 |")
    lines.append("|---|---:|---:|")
    by_intent = defaultdict(list)
    for r in records:
        by_intent[r.intent].append(r)
    for k, rs in sorted(by_intent.items(), key=lambda x: -len(x[1])):
        a = agg(rs)
        lines.append(f"| {k} | {a['n']} | {a['rule_pass']:.1%} |")
    lines.append("")

    # --- judge 维度明细 ---
    if cur["judge_acc"]:
        lines.append("## Judge 维度明细（1-5 分）")
        lines.append("")
        lines.append("| 维度 | 分数 |")
        lines.append("|---|---:|")
        lines.append(f"| 事实准确性 | {cur['judge_acc']:.2f} |")
        lines.append(f"| 有用性 | {cur['judge_help']:.2f} |")
        lines.append(f"| 语气 | {cur['judge_tone']:.2f} |")
        lines.append("")

    # --- 失败模式 ---
    lines.append("## 失败模式统计")
    lines.append("")
    counter = Counter()
    examples = defaultdict(list)
    for r in records:
        if not r.rule:
            continue
        for k, v in r.rule.get("checks", {}).items():
            if not v["passed"]:
                counter[k] += 1
                if len(examples[k]) < 2:
                    examples[k].append(f"`{r.sid}` {r.query[:40]} → {r.answer[:60]}")
    if counter:
        lines.append("| 失败类型 | 次数 | 占比 |")
        lines.append("|---|---:|---:|")
        for k, v in counter.most_common():
            lines.append(f"| {k} | {v} | {v / max(len(records), 1):.1%} |")
        lines.append("")
        lines.append("### 典型样本")
        lines.append("")
        for k, exs in examples.items():
            lines.append(f"**{k}**")
            for e in exs:
                lines.append(f"- {e}")
            lines.append("")
    else:
        lines.append("（无规则失败）")
        lines.append("")

    # --- 指标表（对比基线）---
    if base:
        lines.append("## 对比基线")
        lines.append("")
        lines.append("| 指标 | 基线 | 当前 | Δ |")
        lines.append("|---|---:|---:|---:|")
        for k, label in [("rule_pass", "规则通过率"), ("rule_score", "规则综合分"),
                         ("judge_acc", "Judge 事实"), ("judge_help", "Judge 帮助性"),
                         ("judge_tone", "Judge 语气")]:
            if not base.get(k) and not cur.get(k):
                continue
            lines.append(f"| {label} | {base.get(k, 0):.3f} | {cur.get(k, 0):.3f} | "
                         f"{cur.get(k, 0) - base.get(k, 0):+.3f} |")
        lines.append("")

    # --- 结论模板 ---
    lines.append("## 结论与下一步")
    lines.append("")
    lines.append("（这一节要你自己填。看上面的数字，回答三个问题：）")
    lines.append("")
    lines.append("1. **哪一类提升最大？** 说明哪部分数据起了作用。")
    lines.append("2. **哪一类没改善甚至变差？** 这是下一轮数据的方向。")
    lines.append("3. **通用能力是否有退化？** 如果跑了通用榜单，对比一下。")
    lines.append("")
    lines.append("下一步建议的数据补充方向：")
    lines.append("")
    worst = sorted(by_diff.items(), key=lambda x: agg(x[1])["rule_pass"])[:2]
    for d, rs in worst:
        if d in ("L3", "L4"):
            lines.append(f"- 难度 {d} 表现最差（{agg(rs)['rule_pass']:.1%}）"
                         f"→ 需要补充这类场景的 SFT / 偏好数据")
    worst_intents = sorted(by_intent.items(),
                           key=lambda x: agg(x[1])["rule_pass"])[:3]
    for k, rs in worst_intents:
        if agg(rs)["rule_pass"] < 0.8:
            lines.append(f"- 意图 `{k}` 通过率仅 {agg(rs)['rule_pass']:.1%} → 需要补充样本")
    lines.append("")

    out = out_dir / f"eval_{tag}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 报告 → {out}")

    # 顺手存原始记录，方便 error_analysis
    raw = out_dir / f"eval_{tag}_raw.jsonl"
    with open(raw, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r.sid, "difficulty": r.difficulty, "intent": r.intent,
                "image_type": r.image_type, "query": r.query,
                "images": r.images, "model_answer": r.answer,
                "latency_ms": r.latency_ms, "rule": r.rule, "judge": r.judge,
                "error": r.error,
            }, ensure_ascii=False) + "\n")
    print(f"✓ 原始记录 → {raw}")
    print(f"  下一步：python -m src.eval.error_analysis --in {raw}")

    return out


def _pct(v: list[float], p: float) -> float:
    if not v:
        return 0.0
    v = sorted(v)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _avg(v: list) -> float:
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(description="评测主流程")
    ap.add_argument("--model", required=True, help="模型 ID 或本地路径")
    ap.add_argument("--adapter", help="LoRA adapter 路径")
    ap.add_argument("--eval", default="data/eval/cx_eval_v1.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=1280,
                    help="1200-2560，用第 4 周的曲线找拐点")
    ap.add_argument("--baseline", help="基线 raw.jsonl（用于对比报告）")
    args = ap.parse_args()

    records = run_eval(
        args.eval, args.model, args.adapter, args.tag,
        limit=args.limit, use_judge=not args.no_judge,
        max_pixels=args.max_pixels,
    )

    baseline = None
    if args.baseline:
        baseline = []
        for l in Path(args.baseline).read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            d = json.loads(l)
            baseline.append(EvalRecord(
                sid=d["id"], difficulty=d["difficulty"], intent=d["intent"],
                image_type=d["image_type"], query=d["query"], images=d.get("images", []),
                answer=d.get("model_answer", ""), latency_ms=d.get("latency_ms", 0),
                rule=d.get("rule", {}), judge=d.get("judge", {}),
            ))

    build_report(records, args.tag, baseline)


if __name__ == "__main__":
    main()
