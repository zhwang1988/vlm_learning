"""
LLM-as-a-Judge：主观维度打分 + 偏见校准。

对应讲义 docs/08-evaluation.md「LLM-as-judge 的四个偏见与对策」，对应计划 Day 21。

四个偏见及对策：
  位置偏见  → 交换顺序各评一次取平均
  长度偏见  → prompt 里明确说「长不等于好」
  自我偏好  → judge 用与 policy 不同的模型
  格式偏见  → 明确要求「忽略格式，只看内容」

**校准是必须的**：judge 与人工打分的一致性（Spearman）要 ≥ 0.7。
不校准的 judge 是自欺欺人。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Optional


JUDGE_SYSTEM = """你是电商客服质量评审员。

给定【用户问题】【图片信息】【客服回复】【参考答案】，从三个维度打分（1-5 分）：

1. **事实准确性**：回复是否与图片内容、商品信息一致？有无编造不存在的细节？
   5=完全准确  3=大部分准确但有轻微出入  1=明显编造

2. **有用性**：用户的问题是否被解决？该追问时是否主动追问了？
   5=完全解决或正确追问  3=部分解决  1=答非所问

3. **语气**：是否专业、有同理心、像真人客服？
   5=自然专业  3=略生硬  1=机械或冷漠

## 重要评分要求
- **忽略长度**：长回复不等于好回复。啰嗦的回复应该扣分。
- **忽略格式**：markdown、项目符号、emoji 不影响评分。
- **宁可严格**：不要因为「看起来还行」就给 4 分。

严格输出 JSON，不要任何其他文字：
{"accuracy": 1-5, "helpfulness": 1-5, "tone": 1-5, "reason": "简短理由（30字内）"}
"""


class Judge:
    """LLM judge 封装。支持 OpenAI 兼容接口。"""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        from openai import AsyncOpenAI

        self.model = model or os.getenv("JUDGE_MODEL", "qwen-max")
        self.base_url = base_url or os.getenv("JUDGE_API_BASE") or os.getenv("SYNTH_API_BASE")
        self.api_key = api_key or os.getenv("JUDGE_API_KEY") or os.getenv("SYNTH_API_KEY")
        if not self.api_key:
            raise RuntimeError("缺少 JUDGE_API_KEY。配置在 .env 里。")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def score_one(self, query: str, answer: str,
                        reference: Optional[str] = None,
                        image_desc: Optional[str] = None,
                        max_retries: int = 3) -> dict:
        user_msg = f"""【用户问题】
{query}

【图片信息】
{image_desc or "（无图片描述）"}

【客服回复】
{answer}

【参考答案】
{reference or "（无参考答案，请基于常识判断）"}
"""
        last = None
        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=60,
                )
                d = json.loads(resp.choices[0].message.content)
                for k in ("accuracy", "helpfulness", "tone"):
                    d[k] = max(1, min(5, int(d.get(k, 3))))
                return d
            except Exception as e:      # noqa: BLE001
                last = e
                await asyncio.sleep(2 ** attempt)
        return {"accuracy": 3, "helpfulness": 3, "tone": 3,
                "reason": f"judge 失败: {last}"}

    async def score_batch(self, items: list[dict], concurrency: int = 4) -> list[dict]:
        sem = asyncio.Semaphore(concurrency)

        async def bounded(it):
            async with sem:
                return await self.score_one(
                    it["query"], it["answer"],
                    it.get("reference"), it.get("image_desc"),
                )

        return await asyncio.gather(*[bounded(i) for i in items])


# ---------------------------------------------------------------------------
# 位置偏见校准
# ---------------------------------------------------------------------------


async def pairwise_with_calibration(judge: Judge, query: str, ans_a: str, ans_b: str,
                                    image_desc: Optional[str] = None,
                                    n_rounds: int = 2) -> dict:
    """成对比较，交换顺序各评一次，抵消位置偏见。

    返回 win_rate_a（A 胜出的比例，0-1）。
    """
    wins_a = 0
    total = 0
    for r in range(n_rounds):
        swap = (r % 2 == 1)
        first, second = (ans_b, ans_a) if swap else (ans_a, ans_b)

        prompt = f"""对比两个客服回复，判断哪个更好。

【用户问题】{query}
【图片信息】{image_desc or "（无）"}

【回复 1】
{first}

【回复 2】
{second}

只输出 JSON：{{"winner": 1 或 2 或 0（平局）, "reason": "20字内"}}
"""
        try:
            resp = await judge.client.chat.completions.create(
                model=judge.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=60,
            )
            d = json.loads(resp.choices[0].message.content)
            w = d.get("winner", 0)
            if w == 1:
                wins_a += (0 if swap else 1)
            elif w == 2:
                wins_a += (1 if swap else 0)
            total += 1
        except Exception:
            continue

    return {"win_rate_a": wins_a / max(total, 1), "n_rounds": total}


# ---------------------------------------------------------------------------
# 与人工打分的一致性校准
# ---------------------------------------------------------------------------


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """斯皮尔曼等级相关系数。judge vs 人工的一致性。"""
    n = len(x)
    if n < 3:
        return 0.0

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def calibrate(judge_scores: list[float], human_scores: list[float],
              threshold: float = 0.7) -> dict:
    """校准 judge。返回相关系数和结论。"""
    rho = spearman_correlation(judge_scores, human_scores)
    if rho >= 0.8:
        verdict = "✓ 一致性优秀，judge 可信"
    elif rho >= threshold:
        verdict = "✓ 一致性可接受，judge 可用于粗筛"
    elif rho >= 0.5:
        verdict = "⚠ 一致性偏低，建议改 judge prompt 或增加打分维度说明"
    else:
        verdict = "❌ 一致性太差，judge 结果不可信，必须重写 prompt"

    return {"spearman": rho, "n": len(judge_scores), "verdict": verdict}


# ---------------------------------------------------------------------------
# 校准辅助：生成人工标注模板
# ---------------------------------------------------------------------------


def make_calibration_sheet(eval_path: str | Path = "data/eval/cx_eval_v1.jsonl",
                           out_path: str | Path = "reports/calibration_sheet.md",
                           n: int = 50, seed: int = 42):
    """从评测集抽 50 条，生成人工打分表（带空位）。

    Day 21 的流程：
      1. 跑这个脚本生成表格
      2. 用 judge 给同样 50 条打分
      3. 你人工填分
      4. 跑 calibrate() 算一致性
    """
    random.seed(seed)
    samples = [json.loads(l) for l in
               Path(eval_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = random.sample(samples, min(n, len(samples)))

    lines = ["# Judge 校准表", "",
             "说明：下面是随机抽出的评测样本。",
             "请对每条【模型回复】从三个维度打分（1-5），填在表格里。",
             "打完后跑 `python -m src.eval.judge --calibrate` 计算与 judge 的一致性。", ""]

    for i, s in enumerate(picked):
        lines.append(f"## {i+1}. `{s['id']}` ({s['difficulty']})")
        lines.append("")
        lines.append(f"**用户问题**：{s['query']}")
        lines.append("")
        lines.append(f"**图片**：{', '.join(s.get('images', [])) or '（无）'}")
        lines.append("")
        if s.get("reference"):
            lines.append(f"**参考答案**：{s['reference']}")
        else:
            lines.append(f"**必须提到**：{s.get('must_contain', [])}")
            lines.append(f"**不能出现**：{s.get('must_not_contain', [])}")
            if s.get("should_refuse"):
                lines.append("**正确行为**：应主动追问或转人工")
        lines.append("")
        lines.append("**模型回复**：")
        lines.append("```")
        lines.append("（待填入 —— 跑完评测后自动填充）")
        lines.append("```")
        lines.append("")
        lines.append("**我的打分**：accuracy = ___ / helpfulness = ___ / tone = ___")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 校准表 → {out_path}（{len(picked)} 条）")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(description="LLM-as-judge")
    ap.add_argument("--make-sheet", action="store_true", help="生成人工校准表")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--calibrate", help="校准：传入 JSON，含 judge 和 human 两个数组")
    ap.add_argument("--score", help="对 JSONL 文件批量打分（需 query/answer 字段）")
    args = ap.parse_args()

    if args.make_sheet:
        make_calibration_sheet(n=args.n)
    elif args.calibrate:
        d = json.loads(Path(args.calibrate).read_text(encoding="utf-8"))
        res = calibrate(d["judge"], d["human"])
        print(f"Spearman ρ = {res['spearman']:.3f}  (n={res['n']})")
        print(res["verdict"])
    elif args.score:
        items = [json.loads(l) for l in
                 Path(args.score).read_text(encoding="utf-8").splitlines() if l.strip()]
        judge = Judge()
        results = asyncio.run(judge.score_batch(items))
        out = Path(args.score).with_suffix(".scored.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for it, r in zip(items, results):
                f.write(json.dumps({**it, "judge": r}, ensure_ascii=False) + "\n")
        acc = sum(r["accuracy"] for r in results) / len(results)
        help_ = sum(r["helpfulness"] for r in results) / len(results)
        tone = sum(r["tone"] for r in results) / len(results)
        print(f"✓ {out}")
        print(f"  平均: accuracy={acc:.2f}  helpfulness={help_:.2f}  tone={tone:.2f}")
    else:
        print("用法：")
        print("  python -m src.eval.judge --make-sheet        # 生成人工校准表")
        print("  python -m src.eval.judge --score results.jsonl")
        print("  python -m src.eval.judge --calibrate calib.json")


if __name__ == "__main__":
    main()
