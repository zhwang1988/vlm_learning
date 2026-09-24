#!/usr/bin/env python3
"""
Agent 轨迹（trace）—— 每步落盘，可回放、可算钱、可 debug。

    python -m src.agent.tracing      # 自检

为什么要它：
  1. Agent 出错时，**失败的那一步**最值得看，而日志里往往只记了成功的
  2. 成本要按步算 —— 哪一步烧钱最多，优化才有靶子
  3. 出问题时能把整条轨迹贴给协作方，而不是「我这边跑不对」四个字

输出：JSONL，一行一步，方便 `grep` / `jq` / 灌进任何分析工具。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# 步类型
LLM = "llm"
TOOL = "tool"
GUARD = "guard"
RETRIEVE = "retrieve"


@dataclass
class Step:
    index: int
    kind: str                       # llm / tool / guard / retrieve
    name: str                       # 模型名 / 工具名 / 闸门名
    ms: float = 0.0
    ok: bool = True
    in_tokens: int = 0
    out_tokens: int = 0
    cost: float = 0.0               # 元
    detail: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class Trace:
    """一条会话的完整轨迹。"""

    def __init__(self, session_id: str, meta: Optional[dict] = None):
        self.session_id = session_id
        self.meta = meta or {}
        self.steps: list[Step] = []
        self.started_at = time.time()
        self._open: dict[int, float] = {}      # step index → 开始时间

    # ---- 记录 ----

    def begin(self, kind: str, name: str, **detail) -> int:
        """开始一步，返回 step index（用于 end）。"""
        idx = len(self.steps)
        self.steps.append(Step(index=idx, kind=kind, name=name, detail=detail))
        self._open[idx] = time.perf_counter()
        return idx

    def end(self, idx: int, ok: bool = True, in_tokens: int = 0,
            out_tokens: int = 0, cost: float = 0.0, **detail) -> Step:
        st = self.steps[idx]
        st.ms = (time.perf_counter() - self._open.pop(idx, time.perf_counter())) * 1000
        st.ok = ok
        st.in_tokens = in_tokens
        st.out_tokens = out_tokens
        st.cost = cost
        st.detail.update(detail)
        return st

    def add(self, kind: str, name: str, ms: float = 0.0, ok: bool = True,
            in_tokens: int = 0, out_tokens: int = 0, cost: float = 0.0,
            **detail) -> Step:
        """一次性写入一步（不需要计时的场景）。"""
        st = Step(index=len(self.steps), kind=kind, name=name, ms=ms, ok=ok,
                  in_tokens=in_tokens, out_tokens=out_tokens, cost=cost,
                  detail=detail)
        self.steps.append(st)
        return st

    def guard(self, reason: str, action: str) -> Step:
        """记录一次护栏裁决。

        ⚠️ `ok` 恒为 True，含义是「**护栏本身**正常工作」，不是「放行」。

           早期版本写成 `ok=(action == "continue")`，看上去很直觉，实际会出问题：
           abort / escalate 被算进 `failures()`，于是「失败步」列表里混满了
           「护栏成功拦住了一次重复调用」这种**好事**，真正的执行错误
           （工具报错、超时、模型输出无法解析）反而被淹掉。
           后果是排障时盯着失败步看半天，发现全是护栏在认真干活。

           两个维度必须分开：
             · 这一步本身执行成功了吗  → `ok`（guard 步永远是 True）
             · 护栏给出了什么裁决      → `detail["action"]`

           要筛被拦下的步用 `blocked()`，要看会话是否被中断用 `aborted()`。
        """
        return self.add(GUARD, action, ok=True, reason=reason, action=action)

    # ---- 统计 ----

    def total_ms(self) -> float:
        return sum(s.ms for s in self.steps)

    def total_tokens(self) -> tuple[int, int]:
        return (sum(s.in_tokens for s in self.steps),
                sum(s.out_tokens for s in self.steps))

    def total_cost(self) -> float:
        return sum(s.cost for s in self.steps)

    def failures(self) -> list[Step]:
        """真正出错的步（工具报错 / 超时 / 解析失败）。

        注意**不含**护栏拦截 —— 那是 `blocked()`。
        """
        return [s for s in self.steps if not s.ok]

    def blocked(self) -> list[Step]:
        """被护栏拦下的步（abort / escalate）。

        这和 failures() 是两回事：一条轨迹完全可以「零失败但被拦停」——
        护栏在重复调用第 2 次时就抓住并中止，全过程没有任何一步真出错。
        这种轨迹是**护栏生效的证据**，不该出现在故障报表里。
        """
        return [s for s in self.steps
                if s.kind == GUARD and s.detail.get("action") != "continue"]

    def aborted(self) -> bool:
        """这条轨迹是否被护栏中止过（abort 或 escalate）。

        用于统计「护栏触发率」—— 这个比率太高说明 prompt 或工具设计有问题，
        太低（长期为 0）说明护栏可能根本没接上。
        """
        return any(s.kind == GUARD
                   and s.detail.get("action") in ("abort", "escalate")
                   for s in self.steps)

    def slowest(self, n: int = 3) -> list[Step]:
        return sorted(self.steps, key=lambda s: -s.ms)[:n]

    def costliest(self, n: int = 3) -> list[Step]:
        return sorted(self.steps, key=lambda s: -s.cost)[:n]

    # ---- 落盘 / 回放 ----

    def to_jsonl(self, path: str | Path, include_header: bool = True) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            if include_header:
                f.write(json.dumps({
                    "type": "header", "session_id": self.session_id,
                    "meta": self.meta, "started_at": self.started_at,
                }, ensure_ascii=False) + "\n")
            for s in self.steps:
                f.write(json.dumps({"type": "step", **s.to_dict()},
                                   ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "type": "footer", "n_steps": len(self.steps),
                "total_ms": self.total_ms(), "total_cost": self.total_cost(),
                "in_tokens": self.total_tokens()[0], "out_tokens": self.total_tokens()[1],
            }, ensure_ascii=False) + "\n")
        return p

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Trace":
        header, steps = {}, []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("type") == "header":
                header = d
            elif d.get("type") == "step":
                d.pop("type", None)
                steps.append(Step(**d))
        t = cls(header.get("session_id", "unknown"), header.get("meta", {}))
        t.steps = steps
        t.started_at = header.get("started_at", time.time())
        return t

    def replay(self, verbose: bool = True) -> str:
        """把人能读的轨迹打出来。

        三种标记，别混：
          ✓  正常完成     ✗  真的出错     🛡️  护栏拦下（护栏在干活，不是故障）
        """
        L = [f"轨迹 {self.session_id}  共 {len(self.steps)} 步"]
        for s in self.steps:
            if s.kind == GUARD and s.detail.get("action") != "continue":
                mark = "🛡️"
            elif s.ok:
                mark = "✓"
            else:
                mark = "✗"
            L.append(f"  {s.index:>2} {mark} [{s.kind:8s}] {s.name:24s} "
                     f"{s.ms:7.0f}ms  {s.in_tokens}→{s.out_tokens} tok  ¥{s.cost:.5f}")
            if verbose and s.detail:
                brief = json.dumps(s.detail, ensure_ascii=False)[:110]
                L.append(f"        {brief}")
        ins, outs = self.total_tokens()
        L.append(f"  ── 合计 {self.total_ms():.0f}ms · {ins}→{outs} tokens · "
                 f"¥{self.total_cost():.5f}")
        if self.failures():
            L.append(f"  ⚠️ {len(self.failures())} 步失败 —— 这几步才是要看的地方")
        if self.blocked():
            L.append(f"  🛡️ {len(self.blocked())} 次护栏拦截"
                     f"（护栏正常工作，不是故障）")
        return "\n".join(L)

    def summary(self) -> str:
        ins, outs = self.total_tokens()
        tail = f"失败 {len(self.failures())}"
        if self.blocked():
            tail += f" · 拦截 {len(self.blocked())}"
        return (f"Trace({self.session_id}) {len(self.steps)} 步 · "
                f"{self.total_ms():.0f}ms · ¥{self.total_cost():.5f} · "
                f"{ins}→{outs} tokens · {tail}")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile

    print("=" * 72)
    print("Trace 自检")
    print("=" * 72)

    t = Trace("s_demo_1", meta={"shop": "demo.myshopify.com", "user": "u1"})
    t.add(LLM, "qwen25vl3b-cx-dpo", ms=1420, in_tokens=1840, out_tokens=64,
          cost=0.00028, intent="物流查询")
    t.add(RETRIEVE, "hybrid_search", ms=68, detail={"top_k": 5})
    i = t.begin(TOOL, "lookup_order", order_id="A1")
    t.end(i, ok=True, result={"status": "已发货", "eta": "2026-09-26"})
    t.guard("无异常", "continue")
    i = t.begin(TOOL, "lookup_order", order_id="A1")
    t.end(i, ok=False, error="重复调用")
    t.guard("重复动作 lookup_order(A1) 调了 2 次", "abort")
    t.add(LLM, "qwen25vl3b-cx-dpo", ms=980, in_tokens=2100, out_tokens=88,
          cost=0.00034, note="生成降级回复")

    print("\n[1] 回放")
    print(t.replay())

    print("\n[2] 统计")
    ins, outs = t.total_tokens()
    print(f"    步数 {len(t.steps)} · 总耗时 {t.total_ms():.0f}ms · "
          f"token {ins}→{outs} · 成本 ¥{t.total_cost():.5f}")
    print(f"    失败步 {len(t.failures())} 个 · "
          f"护栏拦截 {len(t.blocked())} 次 · aborted={t.aborted()}")
    print("    最慢的三步：")
    for s in t.slowest():
        print(f"      {s.name:22s} {s.ms:7.0f}ms")

    # ⭐ 这里是最容易写错的地方：护栏拦截**不算失败**
    #    这条轨迹里「重复调用」那一步真出错了（ok=False），
    #    而紧随其后的 guard abort 是护栏**成功**拦下来的，ok 必须是 True。
    #    早期版本用 ok=(action == "continue")，导致 failures() 里混进护栏记录。
    assert len(t.failures()) == 1, \
        f"应该恰好 1 个真失败步，实际 {len(t.failures())}"
    assert t.failures()[0].kind == TOOL, "失败的那步应该是工具调用"
    assert len(t.blocked()) == 1, \
        f"应该恰好 1 次护栏拦截，实际 {len(t.blocked())}"
    assert t.blocked()[0].ok is True, \
        "护栏步的 ok 必须为 True —— abort 是护栏在正常工作，不是故障"
    assert t.aborted() is True, "这条轨迹被护栏中止过"

    # 反例：没有任何护栏的轨迹，两个指标都该是 0 / False
    t_clean = Trace("s_clean")
    t_clean.add(LLM, "qwen25vl3b", ms=100)
    t_clean.add(TOOL, "lookup_order", ms=5)
    assert not t_clean.failures() and not t_clean.blocked()
    assert not t_clean.aborted()
    print("    ✓ 失败 / 拦截 / aborted 三者语义独立（这是排障不跑偏的关键）")

    print("\n[3] 落盘 + 读回")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trace.jsonl"
        t.to_jsonl(p)
        n_lines = len(p.read_text(encoding="utf-8").strip().splitlines())
        t2 = Trace.from_jsonl(p)
        print(f"    写出 {n_lines} 行（1 头 + {len(t.steps)} 步 + 1 尾）")
        print(f"    读回: {t2.summary()}")
        assert len(t2.steps) == len(t.steps)
        assert abs(t2.total_cost() - t.total_cost()) < 1e-9
        # 护栏裁决藏在 detail 里，必须能穿过 JSON 往返 —— 否则回放时
        # 分不清「工具报错」和「护栏拦截」，事后排障就瞎了
        assert len(t2.blocked()) == len(t.blocked())
        assert t2.aborted() == t.aborted()
        print("    ✓ 往返一致（可回放，且护栏语义不丢）")

    print("\n[4] 依赖检查（必须不依赖 torch）")
    import sys
    bad = [m for m in sys.modules if m.startswith(("torch", "transformers"))]
    print("    torch 相关模块:", bad or "无")
    assert not bad
    print("    ✓ 轨迹记录与模型推理解耦")

    print("\n" + "=" * 72)
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
