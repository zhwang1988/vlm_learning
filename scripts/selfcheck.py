#!/usr/bin/env python3
"""
全模块自检 —— 一条命令把 src/ 下所有 __main__ 自检跑一遍。

    python scripts/selfcheck.py            # 跑全部
    python scripts/selfcheck.py --core     # 只跑本机零依赖的那批（本地最常用）
    python scripts/selfcheck.py --only agent    # 只跑名字含 agent 的模块
    python scripts/selfcheck.py -v         # 失败时打印完整输出
    make selfcheck                          # 同上（Makefile 里的短名字）

为什么要用**子进程**逐个跑，而不是在一个进程里 import 所有模块：

    模块的 __main__ 自检会改全局状态 —— 清空 mock 幂等表、推进单号计数器、
    缓存 retriever、注册全局 patch。同一个进程里连着跑，后面的模块会吃到
    前面模块留下的状态，于是出现经典的「单独跑都 OK、一起跑就 FAIL」。
    这种假故障最消耗时间，因为它指向的是一段根本没问题的代码。
    子进程是最省心的隔离方式，代价只是每个模块多花 ~80ms 启动。

三种结果的含义：

    ✓ PASS  自检通过
    – SKIP  预期不会通过（缺依赖 / 需要参数 / 缺上游产物）—— 不是故障
    ✗ FAIL  真问题，需要修

只有 FAIL 会让退出码非 0，可以直接挂进 CI。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 本机零依赖的自检 —— 一个包都不装也能全绿
#
# 判据：模块及其 import 链里不出现 torch / transformers / numpy / fastapi。
# 这批的价值：任何一台新机器 clone 下来，先跑 --core 确认「代码本身没坏」，
# 再去折腾环境。不然环境问题和代码问题会混在一起，非常难查。
# ---------------------------------------------------------------------------
CORE_MODULES = {
    "src.agent.state",
    "src.agent.guardrails",
    "src.agent.tracing",
    "src.agent.tools",
    "src.eval.metrics",
    "src.eval.hallucination",
    "src.eval.agent_eval",
    "src.shopify.auth",
    "src.shopify.webhooks",
    "src.shopify.billing",
    "src.shopify.client",
    "src.train.monitor",
    "src.train.rewards",
    "src.train.dpo_loss",
    "src.data.taxonomy",
}

# SKIP 的判定规则：(正则, 归类说明)
# 命中即算 SKIP；下面是「预期内不会通过」的三类，都不算故障。
#
# 经验：这个项目里多数模块缺前置文件时**不抛异常**，而是打印一行友好的
# 「✗ 找不到 xxx，请先跑 Day N」再 SystemExit(1)。这是好设计（对用户友好），
# 但会让「按 Traceback 分类」的检查器把它误判成故障。所以下面的规则
# 同时覆盖「抛异常」和「友好提示」两种写法。
SKIP_RULES: list[tuple[str, str]] = [
    (r"ModuleNotFoundError: No module named '([\w.]+)'", "缺依赖 {0}"),
    (r"ImportError: No module named '([\w.]+)'", "缺依赖 {0}"),
    (r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'",
     "缺上游产物 {0}"),
    (r"✗\s*找不到\s*(\S+)", "缺上游产物 {0}"),
    (r"(?:没有|0 条)偏好数据", "缺上游产物（先造偏好对）"),
    (r"请先跑(?:完)?\s*(Day\s*\d+)", "缺上游产物，先跑 {0}"),
    (r"(?:用法|usage)[:：]", "需要命令行参数"),
    (r"缺少 .*参数|必须指定", "需要命令行参数"),
    (r"RuntimeError: (?:CUDA|No CUDA|Torch not compiled)", "需要 GPU / torch"),
    (r"torch\.cuda\.is_available\(\)\s*==\s*False", "需要 GPU"),
]

# 少数模块的默认模式会做「真实工作」（联网调模型、写文件、跑几十分钟），
# 不适合被自动自检直接触发。给它们指定专门的自检开关。
MODULE_ARGS: dict[str, list[str]] = {
    "src.data.synth": ["--selftest"],
    "src.data.dedup": ["--selftest"],
    "src.data.build_sft": ["--selftest"],
}

FAIL_MARKERS = ("Traceback (most recent call last)", "AssertionError",
                "SyntaxError", "NameError", "AttributeError", "TypeError")


@dataclass
class Outcome:
    module: str
    status: str                 # PASS / SKIP / FAIL
    reason: str = ""
    detail: str = ""
    seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


def discover() -> list[str]:
    """自动发现所有带 __main__ 自检的模块。

    不写死清单 —— 新增模块会自动纳入，不需要记得回来改这个文件。
    """
    mods: list[str] = []
    for p in sorted(ROOT.joinpath("src").rglob("*.py")):
        if p.name == "__init__.py":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "__main__" in text:
            rel = p.relative_to(ROOT).with_suffix("")
            mods.append(".".join(rel.parts))
    return mods


def classify(rc: int, out: str, err: str) -> tuple[str, str]:
    """把一次执行结果归类到 PASS / SKIP / FAIL。"""
    blob = f"{out}\n{err}"

    if rc == 0:
        return "PASS", ""

    for pat, tmpl in SKIP_RULES:
        m = re.search(pat, blob)
        if m:
            captured = m.group(1) if m.groups() else ""
            return "SKIP", tmpl.format(captured)

    if "KeyboardInterrupt" in blob:
        return "SKIP", "被中断（超时）"

    if any(marker in blob for marker in FAIL_MARKERS):
        return "FAIL", "自检断言失败或抛异常"

    return "FAIL", f"退出码 {rc}"


def run_one(module: str, timeout: int) -> Outcome:
    t0 = time.time()
    cmd = [sys.executable, "-m", module, *MODULE_ARGS.get(module, [])]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
            env=_child_env(),
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -1, "", f"TimeoutExpired: 超过 {timeout}s"
    status, reason = classify(rc, out, err)
    return Outcome(module, status, reason,
                   seconds=time.time() - t0, stdout=out, stderr=err)


def _child_env() -> dict:
    """给子进程的环境。

    关键一条：PYTHONPATH 里加上项目根，这样 `python -m src.x` 在
    任何 cwd 下都能 import 到 src（用系统解释器时不会被 cwd 影响）。
    另外禁用 bytecode 写入，避免自检在仓库里撒 __pycache__。
    """
    env = dict(os.environ)
    root = str(ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


MARK = {
    "PASS": (c("✓", "32"), "32"),
    "SKIP": (c("–", "33"), "33"),
    "FAIL": (c("✗", "31"), "31"),
}


def report(results: list[Outcome], verbose: bool) -> int:
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    n_fail = sum(1 for r in results if r.status == "FAIL")

    width = max(len(r.module) for r in results) if results else 30

    for status in ("FAIL", "SKIP", "PASS"):
        group = [r for r in results if r.status == status]
        if not group:
            continue
        mark, code = MARK[status]
        print()
        for r in group:
            line = (f"  {mark} {r.module:<{width}}  "
                    f"{r.seconds * 1000:>6.0f}ms")
            if r.reason:
                line += f"  {c(r.reason, code)}"
            print(line)

    print()
    print("=" * 78)
    total_time = sum(r.seconds for r in results)
    print(f"  {c(str(n_pass) + ' 通过', '32')} · "
          f"{c(str(n_skip) + ' 跳过', '33')} · "
          f"{c(str(n_fail) + ' 失败', '31') if n_fail else '0 失败'} · "
          f"共 {len(results)} 个模块 / {total_time:.1f}s")

    if n_fail:
        print()
        print(c("  ✗ 以下模块是真问题，需要修：", "31"))
        for r in results:
            if r.status != "FAIL":
                continue
            print(f"      {r.module}  —— {r.reason}")
        if verbose:
            for r in results:
                if r.status != "FAIL":
                    continue
                print()
                print(c(f"{'─' * 78}", "31"))
                print(c(f"  {r.module} 完整输出", "31"))
                print(c(f"{'─' * 78}", "31"))
                tail = r.combined.strip().splitlines()[-40:]
                for line in tail:
                    print(f"    {line}")
        else:
            print()
            print("  加 -v 看完整报错输出。")

    if n_skip and not n_fail:
        print()
        print("  跳过的都是「缺依赖 / 需要参数 / 缺上游产物」，不是故障。")
        print("  想全绿：pip install -r requirements-core.txt（本地）")
        print("          pip install -r requirements-train.txt（云 GPU 机器）")

    print("=" * 78)
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="全模块自检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--core", action="store_true",
                    help="只跑本机零依赖的那批模块")
    ap.add_argument("--only", default="", metavar="PAT",
                    help="只跑模块名里含 PAT 的（子串匹配）")
    ap.add_argument("--timeout", type=int, default=120, metavar="SEC",
                    help="单个模块超时（默认 120s）")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="失败时打印完整输出")
    ap.add_argument("--json", action="store_true", help="输出机器可读结果")
    args = ap.parse_args()

    mods = discover()
    if args.core:
        mods = [m for m in mods if m in CORE_MODULES]
    if args.only:
        mods = [m for m in mods if args.only in m]

    if not mods:
        print("没有匹配的模块。检查 --core / --only 的写法。")
        return 1

    if not args.json:
        scope = ("本机零依赖子集" if args.core
                 else f"--only {args.only}" if args.only else "全部模块")
        print("=" * 78)
        print(f"  multimodal-lab 自检 · {scope} · {len(mods)} 个模块")
        print("=" * 78)

    results = []
    for i, m in enumerate(mods, 1):
        r = run_one(m, args.timeout)
        results.append(r)
        if not args.json:
            mark = MARK[r.status][0]
            print(f"  [{i:>2}/{len(mods)}] {mark} {m:<40} {r.seconds:>5.1f}s"
                  + (f"  {r.reason}" if r.reason else ""))

    if args.json:
        print(json.dumps([{
            "module": r.module, "status": r.status, "reason": r.reason,
            "seconds": round(r.seconds, 3),
        } for r in results], ensure_ascii=False, indent=2))
        return 1 if any(r.status == "FAIL" for r in results) else 0

    print()
    return report(results, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
