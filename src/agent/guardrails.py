#!/usr/bin/env python3
"""
Agent 护栏 —— 三道闸门，防止 Agent 失控烧钱或死循环。

    python -m src.agent.guardrails      # 自检

三道闸门（对应 docs/10-agent.md 第 6 节）：
  1. **步数上限**   —— 模型陷入循环时最直接的止损
  2. **重复动作检测** —— 连续两次「相同工具 + 相同参数」= 卡住了，立刻跳出
  3. **成本上限**   —— 累计 token / 金额超限就停，转人工

设计原则：**护栏判断不抛异常，只返回裁决**。
抛异常的话，异常会冒到模型层把对话搞乱；返回裁决则能被主循环优雅处理。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .state import SessionState, idempotency_key

# 裁决动作
CONTINUE = "continue"      # 继续
ABORT = "abort"            # 中断本轮，直接给降级回复
ESCALATE = "escalate"      # 中断并转人工


@dataclass
class GuardrailConfig:
    max_steps: int = 6                 # 单轮最多几步（模型↔工具往返次数）
    max_tool_calls: int = 4            # 单轮最多几次工具调用
    cost_limit_money: float = 0.05     # 单会话金额上限（元）
    cost_limit_tokens: int = 40000     # 单会话 token 上限
    repeat_threshold: int = 2          # 同一 (工具+参数) 出现几次算卡住
    max_session_turns: int = 40        # 会话轮次上限（防上下文爆炸）
    per_tool_timeout_s: float = 22.0   # 单个工具的超时预算
    wall_clock_limit_s: float = 45.0   # 单轮墙钟上限


@dataclass
class Verdict:
    action: str = CONTINUE
    reason: str = ""
    detail: dict = field(default_factory=dict)
    suggestion: str = ""

    @property
    def ok(self) -> bool:
        return self.action == CONTINUE

    def __str__(self) -> str:
        return f"[{self.action}] {self.reason}" + (f" → {self.suggestion}" if self.suggestion else "")


class Guardrails:
    """一次会话的护栏实例。主循环每一步都问它一次。"""

    def __init__(self, cfg: Optional[GuardrailConfig] = None):
        self.cfg = cfg or GuardrailConfig()
        self.steps = 0
        self.tool_calls_this_turn = 0
        self.turn_started_at = time.time()
        self.tripped: list[Verdict] = []          # 触发的历史（供 trace 用）

    # ---- 生命周期 ----

    def begin_turn(self) -> None:
        self.steps = 0
        self.tool_calls_this_turn = 0
        self.turn_started_at = time.time()

    def end_turn(self) -> None:
        pass

    # ---- 三道闸门 ----

    def before_step(self, state: SessionState) -> Verdict:
        """每步开始前调用。"""
        self.steps += 1

        if self.steps > self.cfg.max_steps:
            return self._trip(ABORT, f"步数超过上限 {self.cfg.max_steps}",
                              suggestion="可能陷入了循环。已中断，转人工。")

        if self.tool_calls_this_turn > self.cfg.max_tool_calls:
            return self._trip(ABORT,
                              f"工具调用次数超过上限 {self.cfg.max_tool_calls}",
                              suggestion="一次咨询里调太多次工具通常意味着没理解问题。")

        elapsed = time.time() - self.turn_started_at
        if elapsed > self.cfg.wall_clock_limit_s:
            return self._trip(ABORT, f"单轮耗时 {elapsed:.1f}s 超过 {self.cfg.wall_clock_limit_s}s",
                              suggestion="超时了，先给用户一个阶段性回复。")

        if state.cost_money > self.cfg.cost_limit_money:
            return self._trip(ESCALATE,
                              f"会话成本 ¥{state.cost_money:.4f} 超过 "
                              f"¥{self.cfg.cost_limit_money:.4f}",
                              suggestion="成本闸门触发，转人工。",
                              detail={"cost_money": state.cost_money})

        if state.cost_tokens > self.cfg.cost_limit_tokens:
            return self._trip(ESCALATE,
                              f"会话 token {state.cost_tokens} 超过 "
                              f"{self.cfg.cost_limit_tokens}",
                              suggestion="上下文太长了，先做摘要再继续。")

        if len(state.turns) > self.cfg.max_session_turns:
            return self._trip(ESCALATE,
                              f"会话轮次 {len(state.turns)} 超过 {self.cfg.max_session_turns}",
                              suggestion="太长的会话建议转人工，别让模型自己撑。")

        return Verdict()

    def before_tool(self, state: SessionState, tool: str, args: dict) -> Verdict:
        """调用工具前调用 —— 这里是重复动作检测的位置。"""
        self.tool_calls_this_turn += 1
        key = idempotency_key(state.session_id, tool, args)

        same = sum(1 for t, k in state.recent_actions(6) if k == key)
        if same >= self.cfg.repeat_threshold - 1:
            return self._trip(ABORT,
                              f"重复动作：{tool} 用完全相同的参数调了 {same + 1} 次",
                              suggestion="工具没给出新信息，继续调也是白调。跳出循环，"
                                         "把已知信息给用户并转人工。",
                              detail={"tool": tool, "idem_key": key, "count": same + 1})
        return Verdict()

    # ---- 内部 ----

    def _trip(self, action: str, reason: str, suggestion: str = "",
              detail: Optional[dict] = None) -> Verdict:
        v = Verdict(action=action, reason=reason, suggestion=suggestion,
                    detail=detail or {})
        self.tripped.append(v)
        return v

    def report(self) -> str:
        if not self.tripped:
            return "护栏：本次没有触发任何闸门"
        return "\n".join(f"  {v}" for v in self.tripped)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("=" * 72)
    print("护栏自检")
    print("=" * 72)

    # 1. 步数上限
    g = Guardrails(GuardrailConfig(max_steps=3))
    s = SessionState("s1")
    g.begin_turn()
    verdicts = [g.before_step(s) for _ in range(5)]
    print("\n[1] 步数上限 max_steps=3")
    for i, v in enumerate(verdicts, 1):
        print(f"    第 {i} 步: {v}")
    assert verdicts[2].ok and verdicts[3].action == ABORT
    print("    ✓ 第 4 步被拦下")

    # 2. 重复动作检测
    g = Guardrails(GuardrailConfig(repeat_threshold=2))
    s = SessionState("s1")
    g.begin_turn()
    args = {"order_id": "A1"}
    v1 = g.before_tool(s, "lookup_order", args)
    s.record_tool_call("lookup_order", args, ok=True)
    v2 = g.before_tool(s, "lookup_order", args)
    s.record_tool_call("lookup_order", args, ok=True)
    v3 = g.before_tool(s, "lookup_order", args)
    print("\n[2] 重复动作（同一工具 + 同一参数，threshold=2）")
    print("    第 1 次:", v1)
    print("    第 2 次:", v2)
    print("    第 3 次:", v3)
    assert v1.ok, "第 1 次不该被拦"
    assert v2.action == ABORT, "threshold=2 时，第 2 次相同调用就该被拦下"
    print("    ✓ 第 2 次即被拦下（threshold=2 的语义是「出现 2 次算卡住」）")

    # 3. 参数不同不该误杀
    g = Guardrails(GuardrailConfig(repeat_threshold=2))
    s = SessionState("s1")
    g.begin_turn()
    for oid in ("A1", "A2", "A3"):
        r = g.before_tool(s, "lookup_order", {"order_id": oid})
        s.record_tool_call("lookup_order", {"order_id": oid}, ok=True)
        assert r.ok, f"不同参数被误杀了：{oid}"
    print("\n[3] 同类工具不同参数")
    print("    ✓ 三次不同订单号的查询没有被误杀")

    # 4. 成本闸门
    g = Guardrails(GuardrailConfig(cost_limit_money=0.01))
    s = SessionState("s2")
    g.begin_turn()
    s.cost_money = 0.02
    v = g.before_step(s)
    print("\n[4] 成本闸门")
    print("   ", v)
    assert v.action == ESCALATE
    print("    ✓ 超限后转人工（不是静默继续烧钱）")

    print("\n" + "=" * 72)
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
