"""
量化：AWQ / GPTQ + 混合精度（视觉塔保持 bf16）。

对应讲义 docs/09-inference.md「量化的四档」，对应计划 Day 29。

**核心知识点：VLM 量化时视觉塔比 LLM 敏感得多。**

原因：视觉特征里的高频信息（细节纹理、小字）在低精度下最先丢失。
表现恰好是客服场景最需要的三种能力退化：OCR 数字读错、颜色判断偏、
小瑕疵看不见。

解法：混合精度 —— 视觉塔保持 bf16，LLM 用 int4。
      代价：省得少一点（视觉塔约占 18%），但质量保住了。

用法：
    python -m src.serve.quantize --model outputs/xxx --method awq
    python -m src.serve.quantize --model Qwen/Qwen2.5-VL-3B-Instruct --method awq --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# 需要排除量化的模块（视觉塔 + 连接器）
# ---------------------------------------------------------------------------

# ⚠️ 这是本模块最重要的一个配置。
#    名单来自 Qwen2.5-VL 的模块命名，换模型时要重新确认。
VISION_EXCLUDE = [
    "visual",              # 视觉塔整体
    "vision_tower",
    "merger",              # 连接器（也在视觉侧，同样敏感）
]


def build_quant_config(method: str = "awq", exclude_vision: bool = True,
                       bits: int = 4) -> dict:
    """构造量化配置。

    exclude_vision=True → 混合精度（推荐）
    exclude_vision=False → 全量量化（省得多但质量掉）
    """
    cfg = {
        "method": method,
        "bits": bits,
        "group_size": 128,
        "desc_act": False,          # AWQ/GPTQ 的 act-order，开着小幅提质量但慢
    }
    if exclude_vision:
        cfg["modules_to_not_convert"] = VISION_EXCLUDE
    return cfg


# ---------------------------------------------------------------------------
# 显存与质量预估
# ---------------------------------------------------------------------------


QWEN25VL_PARAM_SPLIT = {
    # 视觉塔 + 连接器 约占 18%，LLM 占 82%
    "3B": {"vision": 0.66e9, "llm": 3.09e9},
    "7B": {"vision": 0.66e9, "llm": 7.63e9},
}


def estimate_savings(model_size: str = "3B", exclude_vision: bool = True) -> dict:
    """估算量化前后的显存。"""
    spec = QWEN25VL_PARAM_SPLIT.get(model_size, QWEN25VL_PARAM_SPLIT["3B"])

    # bf16 = 2 字节/参数
    bf16_gb = (spec["vision"] + spec["llm"]) * 2 / (1024 ** 3)

    if exclude_vision:
        # 视觉塔 bf16，LLM int4（0.5 字节 + 约 0.1 量化元数据）
        quant_gb = (spec["vision"] * 2 + spec["llm"] * 0.6) / (1024 ** 3)
    else:
        quant_gb = (spec["vision"] + spec["llm"]) * 0.6 / (1024 ** 3)

    return {
        "model_size": model_size,
        "exclude_vision": exclude_vision,
        "bf16_gb": round(bf16_gb, 2),
        "quantized_gb": round(quant_gb, 2),
        "saving_ratio": round(1 - quant_gb / bf16_gb, 3),
        "note": ("视觉塔保持 bf16 —— 保住 OCR/颜色/细节能力"
                 if exclude_vision else
                 "⚠️ 全量量化 —— 视觉能力会明显退化，不推荐用于客服场景"),
    }


def print_quality_warning():
    print()
    print("=" * 78)
    print("⚠️  VLM 量化的质量风险（务必读一遍）")
    print("=" * 78)
    print("""
视觉塔对量化比 LLM 敏感得多，原因是视觉特征里的高频信息
（纹理、小字、细微色差）在低精度下最先丢失。

退化表现（恰好是客服最需要的三种能力）：
  - OCR 数字读错       → 尺码表、吊牌、快递单读错
  - 颜色判断偏移       → 米白/纯白/燕麦色分不清
  - 小瑕疵看不见       → 线头、走线不齐判不出来

Day 29 的验收动作：
  量化前后各跑一遍领域评测集，**特别关注 OCR/颜色/瑕疵这三类**，
  以及 POPE 幻觉率。如果幻觉率上升超过 3pt，就不要用全量量化。

混合精度（exclude_vision=True）通常能把质量损失控制在 1pt 以内，
代价是节省比例从 70% 降到约 55%。
""")


# ---------------------------------------------------------------------------
# 执行量化
# ---------------------------------------------------------------------------


def quantize_awq(model_path: str, out_path: str, exclude_vision: bool = True,
                 calibration_samples: int = 128, dry_run: bool = False):
    """AWQ 量化。

    AWQ 的核心思路：**不是所有权重都同等重要，要保护对大激活值重要的权重通道。**
    它用少量校准数据统计激活分布，据此缩放权重，再量化。
    这比直接 round-to-nearest 好很多。
    """
    cfg = build_quant_config("awq", exclude_vision)
    out_path = Path(out_path)

    print(f"模型:      {model_path}")
    print(f"输出:      {out_path}")
    print(f"校准样本:  {calibration_samples}")
    print(f"排除模块:  {cfg.get('modules_to_not_convert', [])}")
    print()

    estimate = estimate_savings("3B" if "3B" in model_path or "0.5B" not in model_path
                                else "3B", exclude_vision)
    print(f"显存预估:  bf16 {estimate['bf16_gb']}GB → "
          f"量化后 {estimate['quantized_gb']}GB （省 {estimate['saving_ratio']:.0%}）")
    print(f"           {estimate['note']}")
    print()

    if dry_run:
        print("[dry-run] 不实际执行。")
        print()
        print("真实执行的命令（autoawq）：")
        print(f"  from awq import AutoAWQForCausalLM")
        print(f"  model = AutoAWQForCausalLM.from_pretrained('{model_path}')")
        print(f"  model.quantize(tokenizer, quant_config={json.dumps(cfg, indent=4)})")
        print(f"  model.save_quantized('{out_path}')")
        print()
        print("或者用 llm-compressor（更新的工具，支持混合精度更灵活）：")
        print(f"  llmcompressor -m {model_path} \\")
        print(f"    --scheme W4A16 \\")
        print(f"    --ignore {' '.join(VISION_EXCLUDE)} \\")
        print(f"    -o {out_path}")
        return None

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("✗ 未安装 autoawq：pip install autoawq")
        print("  或用 llmcompressor（推荐，对 VLM 支持更好）")
        return None

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    model.quantize(tokenizer, quant_config={
        "zero_point": True,
        "q_group_size": cfg["group_size"],
        "w_bit": cfg["bits"],
        "version": "GEMM",
        "modules_to_not_convert": cfg.get("modules_to_not_convert", []),
    })

    out_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    print(f"\n✓ 量化完成，耗时 {time.time() - t0:.0f}s → {out_path}")
    print()
    print("下一步（Day 29 的验收）:")
    print(f"  1. 起服务: bash src/serve/vllm_server.sh {out_path} cx-vlm awq")
    print(f"  2. 跑评测: python -m src.eval.run_eval --model {out_path} "
          f"--tag quantized --baseline reports/eval_sft_raw.jsonl")
    print(f"  3. 对比 OCR/颜色/瑕疵三类 + POPE 幻觉率")
    print(f"  4. 记录显存/首token延迟/吞吐（tok/s）")
    return out_path


# ---------------------------------------------------------------------------
# 基准测试
# ---------------------------------------------------------------------------


def benchmark(model_path: str, n_requests: int = 20, concurrency: int = 4):
    """测吞吐和延迟。量化前后各跑一次，出对比表。"""
    import asyncio
    import os

    from openai import AsyncOpenAI

    base = os.getenv("VLLM_BASE", "http://localhost:8000/v1")
    client = AsyncOpenAI(base_url=base, api_key="EMPTY")
    model_name = os.getenv("SERVED_MODEL", "cx-vlm")

    async def one(i):
        t0 = time.perf_counter()
        ttft = None
        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content":
                           f"请用一句话介绍商品编号 {i} 的退换货政策。"}],
                max_tokens=100, temperature=0.3, stream=True,
            )
            n = 0
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    n += 1
            total = (time.perf_counter() - t0) * 1000
            return {"ttft_ms": ttft or total, "total_ms": total, "tokens": n}
        except Exception as e:      # noqa: BLE001
            return {"error": str(e)[:80]}

    async def run():
        sem = asyncio.Semaphore(concurrency)

        async def bounded(i):
            async with sem:
                return await one(i)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded(i) for i in range(n_requests)])
        wall = time.perf_counter() - t0
        return results, wall

    print(f"基准测试: {n_requests} 请求, 并发 {concurrency}")
    results, wall = asyncio.run(run())

    ok = [r for r in results if "error" not in r]
    if not ok:
        print("全部失败。检查 vLLM 是否在运行。")
        return {}

    ttfts = sorted(r["ttft_ms"] for r in ok)
    toks = sum(r["tokens"] for r in ok)
    res = {
        "n_ok": len(ok),
        "ttft_p50_ms": ttfts[len(ttfts) // 2],
        "ttft_p95_ms": ttfts[int(len(ttfts) * 0.95)],
        "total_wall_s": round(wall, 2),
        "throughput_tok_per_s": round(toks / wall, 1),
    }
    print()
    print(f"  成功        {res['n_ok']}/{n_requests}")
    print(f"  首token P50 {res['ttft_p50_ms']:.0f}ms")
    print(f"  首token P95 {res['ttft_p95_ms']:.0f}ms")
    print(f"  总吞吐      {res['throughput_tok_per_s']:.1f} tok/s")
    print(f"  墙钟        {res['total_wall_s']}s")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="量化与基准测试")
    ap.add_argument("--model")
    ap.add_argument("--out", default="outputs/quantized")
    ap.add_argument("--method", default="awq", choices=["awq", "gptq"])
    ap.add_argument("--full-quant", action="store_true",
                    help="连视觉塔一起量化（不推荐）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="只打印显存预估")
    ap.add_argument("--benchmark", action="store_true", help="跑吞吐基准测试")
    ap.add_argument("--n", type=int, default=20, help="基准测试请求数")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    if args.estimate:
        for size in ("3B", "7B"):
            for ex in (True, False):
                e = estimate_savings(size, ex)
                print(f"{size} 排除视觉塔={ex}: "
                      f"{e['bf16_gb']}GB → {e['quantized_gb']}GB "
                      f"(省 {e['saving_ratio']:.0%})")
        print_quality_warning()
    elif args.benchmark:
        benchmark(args.model or "", args.n, args.concurrency)
    elif args.model:
        quantize_awq(args.model, args.out, not args.full_quant, dry_run=args.dry_run)
    else:
        ap.print_help()
        print()
        print("常见用法：")
        print("  python -m src.serve.quantize --estimate          # 先看省多少")
        print("  python -m src.serve.quantize --model outputs/xxx --dry-run")
        print("  python -m src.serve.quantize --model outputs/xxx")
        print("  python -m src.serve.quantize --benchmark --n 20 --concurrency 4")
