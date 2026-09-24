#!/usr/bin/env python3
"""
全链路压测 —— W8 Day 43。

    # 基本压测
    python scripts/loadtest.py --url http://localhost:8080/v1/chat --n 200 --concurrency 10

    # 按时长压（更接近真实流量）
    python scripts/loadtest.py --url ... --duration 60 --concurrency 8

    # 带图请求（图文的延迟和纯文本差 2–3 倍，必须分开压）
    python scripts/loadtest.py --url ... --n 50 --concurrency 4 --image data/raw_images/x.jpg

关键点：**报 P50/P95/P99，不报平均值**。
平均值会掩盖那 5% 的糟糕体验，而正是那 5% 决定用户会不会卸载你的 App。

纯标准库实现，不需要额外依赖。
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------

def build_payload(text: str, image: str | None) -> bytes:
    images = []
    if image:
        p = Path(image)
        if not p.exists():
            raise SystemExit(f"✗ 找不到图片 {image}")
        b64 = base64.b64encode(p.read_bytes()).decode()
        images.append("data:image/jpeg;base64," + b64)
    return json.dumps({
        "messages": [{"role": "user", "content": text}],
        "images": images,
    }, ensure_ascii=False).encode("utf-8")


def one_request(url: str, payload: bytes, timeout: float) -> dict:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return {"ok": 200 <= resp.status < 300, "status": resp.status,
                    "ms": (time.perf_counter() - t0) * 1000,
                    "bytes": len(body), "err": ""}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "ms": (time.perf_counter() - t0) * 1000,
                "bytes": 0, "err": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "status": 0, "ms": (time.perf_counter() - t0) * 1000,
                "bytes": 0, "err": type(e).__name__}


# ---------------------------------------------------------------------------
# 压测主体
# ---------------------------------------------------------------------------

def run(url: str, n: int, concurrency: int, timeout: float, payload: bytes,
        duration: float = 0.0, warmup: int = 3) -> dict:
    print(f"[loadtest] {url}")
    print(f"           n={n}  concurrency={concurrency}  timeout={timeout}s"
          + (f"  duration={duration}s" if duration else ""))

    for _ in range(warmup):                     # 预热：避免把冷启动算进去
        one_request(url, payload, timeout)

    results: list[dict] = []
    lock = threading.Lock()
    stop = time.perf_counter() + duration if duration else None

    def worker(i: int):
        if stop and time.perf_counter() >= stop:
            return
        r = one_request(url, payload, timeout)
        with lock:
            results.append(r)

    t0 = time.perf_counter()
    total = n if not duration else max(n, concurrency * 200)   # 按时长压时给个上限兜底
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(worker, range(total)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    lat = [r["ms"] for r in ok] or [0.0]
    fails = [r for r in results if not r["ok"]]
    err_kinds: dict[str, int] = {}
    for f in fails:
        err_kinds[f["err"]] = err_kinds.get(f["err"], 0) + 1

    def pct(p: float) -> float:
        s = sorted(lat)
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p / 100
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {
        "n_done": len(results), "n_ok": len(ok), "n_fail": len(fails),
        "wall_s": wall, "qps": len(results) / wall if wall else 0,
        "p50": pct(50), "p95": pct(95), "p99": pct(99),
        "mean": statistics.fmean(lat) if lat else 0,
        "errors": err_kinds,
    }


def render(stats: dict, concurrency: int) -> str:
    L = ["", f"  完成 {stats['n_done']}   成功 {stats['n_ok']}   失败 {stats['n_fail']}",
         f"  总耗时       : {stats['wall_s']:.1f} s",
         f"  QPS          : {stats['qps']:.1f}",
         f"  P50 延迟     : {stats['p50']/1000:.2f} s",
         f"  P95 延迟     : {stats['p95']/1000:.2f} s     ← 决定用户体验的是这个",
         f"  P99 延迟     : {stats['p99']/1000:.2f} s",
         f"  平均延迟     : {stats['mean']/1000:.2f} s     ← 参考值，别用它做决策"]
    if stats["errors"]:
        L.append("  错误分布     :")
        for k, v in sorted(stats["errors"].items(), key=lambda kv: -kv[1]):
            L.append(f"      {k}: {v}")
    if stats["p95"] > 5000:
        L.append("  ⚠️  P95 超过 5 秒 —— 用户体验已经很差了，先优化再谈扩容")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="全链路压测")
    ap.add_argument("--url", required=True, help="如 http://localhost:8080/v1/chat")
    ap.add_argument("--n", type=int, default=100, help="请求总数")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=0.0, help="秒；给了就按时长压")
    ap.add_argument("--text", default="这件米白针织衫 M 码有货吗？")
    ap.add_argument("--image", default=None, help="带图压测（图文延迟是纯文本的 2–3 倍）")
    ap.add_argument("--sweep", action="store_true",
                    help="并发扫描：1/2/4/8/16 各压一轮，找 P95 拐点")
    ap.add_argument("--json", default=None, help="把结果写到 JSON（供成本模型使用）")
    args = ap.parse_args()

    payload = build_payload(args.text, args.image)

    if args.sweep:
        rows = []
        print("=" * 70)
        print("并发扫描（找拐点：QPS 不再涨而 P95 开始飙的地方）")
        print("=" * 70)
        for c in (1, 2, 4, 8, 16):
            s = run(args.url, max(args.n, 20), c, args.timeout, payload)
            rows.append((c, s))
            print(f"  c={c:<3} QPS {s['qps']:6.1f}   P50 {s['p50']/1000:5.2f}s   "
                  f"P95 {s['p95']/1000:5.2f}s   失败 {s['n_fail']}")
        print()
        best = max(rows, key=lambda r: r[1]["qps"])
        print(f"  → 吞吐最好的是 c={best[0]}（QPS {best[1]['qps']:.1f}）")
        print("  → 拐点之后再加并发只会让 P95 变差，注意这个点")
    else:
        s = run(args.url, args.n, args.concurrency, args.timeout, payload, args.duration)
        print(render(s, args.concurrency))
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(s, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
            print(f"\n  ✓ 结果 → {args.json}（下一步交给成本模型）")

    print("\n  下一步：把 QPS / P95 填进 reports/cost_model.md 的成本模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
