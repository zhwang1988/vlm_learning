"""
Agent 评测。

对应讲义 docs/10-agent.md「Agent 评测的不同之处」，对应计划 Day 36。

和模型评测最大的区别：**要看世界状态变了没有，不只看回复内容。**

指标：
  任务完成率 (Task Success Rate)   ← 主指标，目标 ≥ 65%
  工具选择准确率
  参数填充准确率
  平均交互轮数                      ← 越少越好
  不必要工具调用率                  ← 应尽量低
  P95 延迟 / 单次成本
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# 任务定义
# ---------------------------------------------------------------------------


@dataclass
class AgentTask:
    tid: str
    query: str
    images: list[str] = field(default_factory=list)
    intent: str = ""
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_slots: dict = field(default_factory=dict)
    # 完成判定：函数名 + 参数（由 task_verifier 实现）
    verifier: str = ""
    verifier_args: dict = field(default_factory=dict)
    max_steps: int = 6
    notes: str = ""


# 内置任务集（Day 36 直接用，也可以自己扩）
DEFAULT_TASKS = [
    AgentTask("t01", "我的快递到哪了？", intent="logistics",
              expected_tools=["lookup_order", "shipping_status"],
              forbidden_tools=["start_return"],
              verifier="answered_contains", verifier_args={"keys": ["物流", "快递", "运输"]}),
    AgentTask("t02", "这个能退吗？", images=["sample.jpg"], intent="return_refund",
              expected_tools=["lookup_order", "check_return_eligibility"],
              forbidden_tools=[], verifier="tool_called",
              verifier_args={"tool": "check_return_eligibility"}),
    AgentTask("t03", "M 码还有货吗？", images=["sample.jpg"], intent="size_fit",
              expected_tools=["check_stock"], verifier="tool_called",
              verifier_args={"tool": "check_stock"}),
    AgentTask("t04", "这张图里的线头算质量问题吗？", images=["sample.jpg"],
              intent="quality_issue",
              expected_tools=[], forbidden_tools=["lookup_order", "check_stock"],
              verifier="answered_contains", verifier_args={"keys": ["线头", "工艺", "正常"]}),
    AgentTask("t05", "我要退货", intent="return_refund",
              expected_tools=["lookup_order", "start_return"],
              verifier="tool_called", verifier_args={"tool": "start_return"}),
    AgentTask("t06", "你们这是欺诈！我要投诉", intent="return_refund",
              expected_tools=[], forbidden_tools=["start_return"],
              verifier="answered_contains",
              verifier_args={"keys": ["抱歉", "转接", "人工", "理解"]}),
    AgentTask("t07", "这件衣服是纯棉的吗？", images=["sample.jpg"], intent="material",
              expected_tools=[], verifier="answered_contains",
              verifier_args={"keys": ["成分", "材质", "标注", "棉"]}),
]


# ---------------------------------------------------------------------------
# 环境状态验证器（Agent 评测的核心）
# ---------------------------------------------------------------------------


class TaskVerifier:
    """验证任务是否真的完成。

    关键点：不只看回复，看**副作用**和**工具调用轨迹**。
    """

    def __init__(self, tool_executor_state: Optional[dict] = None):
        self.state = tool_executor_state or {}

    def verify(self, task: AgentTask, reply: str,
               tool_calls: list[dict]) -> tuple[bool, str]:
        kind = task.verifier
        args = task.verifier_args

        if kind == "tool_called":
            called = [c.get("action") for c in tool_calls]
            ok = args["tool"] in called
            return ok, (f"✓ 调用了 {args['tool']}" if ok
                        else f"✗ 未调用 {args['tool']}（实际: {called}）")

        if kind == "answered_contains":
            keys = args.get("keys", [])
            hit = [k for k in keys if k in reply]
            ok = len(hit) > 0
            return ok, (f"✓ 命中 {hit}" if ok else f"✗ 未提及 {keys}")

        if kind == "state_changed":
            key = args["key"]
            ok = bool(self.state.get(key))
            return ok, (f"✓ 状态 {key} 已变更" if ok
                        else f"✗ 状态 {key} 未变更（副作用缺失）")

        return False, f"未知的验证器 {kind}"


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------


@dataclass
class AgentMetrics:
    tid: str
    success: bool
    reason: str
    n_steps: int
    tools_called: list[str]
    tool_selection_ok: bool
    slot_filling_ok: bool
    unnecessary_calls: int
    latency_ms: float
    n_tokens_in: int = 0
    n_tokens_out: int = 0
    error: str = ""


async def run_agent_eval(agent_run_fn: Callable,
                         tasks: list[AgentTask] | None = None,
                         concurrency: int = 1) -> list[AgentMetrics]:
    """跑 Agent 评测。

    agent_run_fn: async (query, images) -> dict，必须返回
        {"reply": str, "tool_calls": [{"action":..., "action_input":...}],
         "n_steps": int, "tokens_in": int, "tokens_out": int,
         "state": dict}
    """
    tasks = tasks or DEFAULT_TASKS
    results: list[AgentMetrics] = []
    sem = asyncio.Semaphore(concurrency)

    async def one(task: AgentTask) -> AgentMetrics:
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await agent_run_fn(task.query, task.images)
                ms = (time.perf_counter() - t0) * 1000

                tool_calls = out.get("tool_calls", [])
                called = [c.get("action") for c in tool_calls]

                verifier = TaskVerifier(out.get("state", {}))
                ok, reason = verifier.verify(task, out.get("reply", ""), tool_calls)

                # 工具选择：期望的都调了，且没调禁用的
                sel_ok = (all(t in called for t in task.expected_tools)
                          and not any(t in called for t in task.forbidden_tools))

                # 参数填充
                slot_ok = True
                if task.expected_slots:
                    flat = {}
                    for c in tool_calls:
                        flat.update(c.get("action_input", {}) or {})
                    slot_ok = all(str(flat.get(k)) == str(v)
                                  for k, v in task.expected_slots.items())

                unnecessary = sum(1 for c in called
                                  if c not in task.expected_tools
                                  and c != "final_answer")

                return AgentMetrics(
                    tid=task.tid, success=ok, reason=reason,
                    n_steps=out.get("n_steps", 0), tools_called=called,
                    tool_selection_ok=sel_ok, slot_filling_ok=slot_ok,
                    unnecessary_calls=unnecessary, latency_ms=ms,
                    n_tokens_in=out.get("tokens_in", 0),
                    n_tokens_out=out.get("tokens_out", 0),
                )
            except Exception as e:      # noqa: BLE001
                return AgentMetrics(
                    tid=task.tid, success=False, reason=f"异常: {e}",
                    n_steps=0, tools_called=[], tool_selection_ok=False,
                    slot_filling_ok=False, unnecessary_calls=0,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=str(e)[:200],
                )

    results = await asyncio.gather(*[one(t) for t in tasks])
    return list(results)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def build_agent_report(metrics: list[AgentMetrics],
                       out_path: str | Path = "reports/agent_eval_v1.md",
                       cost_per_1k_tokens_in: float = 0.001,
                       cost_per_1k_tokens_out: float = 0.003) -> Path:
    n = len(metrics)
    succ = sum(1 for m in metrics if m.success)
    sel = sum(1 for m in metrics if m.tool_selection_ok)
    slot = sum(1 for m in metrics if m.slot_filling_ok)
    steps = [m.n_steps for m in metrics if not m.error]
    lat = sorted(m.latency_ms for m in metrics)
    unnec = sum(m.unnecessary_calls for m in metrics)
    tin = sum(m.n_tokens_in for m in metrics)
    tout = sum(m.n_tokens_out for m in metrics)
    cost = tin / 1000 * cost_per_1k_tokens_in + tout / 1000 * cost_per_1k_tokens_out

    def pct(v):
        return f"{v / max(n, 1):.1%}"

    lines = ["# Agent 评测报告 v1", ""]
    lines.append(f"- 任务数: **{n}**")
    lines.append(f"- **任务完成率: {pct(succ)}** （目标 ≥ 65%）")
    lines.append(f"- 工具选择准确率: {pct(sel)}")
    lines.append(f"- 参数填充准确率: {pct(slot)}")
    lines.append(f"- 平均交互轮数: {sum(steps) / max(len(steps), 1):.1f}")
    lines.append(f"- 不必要工具调用总数: {unnec}（平均 {unnec / max(n, 1):.2f} 次/任务）")
    lines.append(f"- 延迟 P50 / P95: "
                 f"{lat[len(lat) // 2] if lat else 0:.0f}ms / "
                 f"{lat[int(len(lat) * 0.95)] if lat else 0:.0f}ms")
    lines.append(f"- Token 消耗: {tin:,} in / {tout:,} out")
    lines.append(f"- 估算成本: ¥{cost:.4f}（{n} 个任务，约 ¥{cost / max(n, 1):.4f}/任务）")
    lines.append("")

    lines.append("## 逐任务结果")
    lines.append("")
    lines.append("| 任务 | 结果 | 轮数 | 调用工具 | 说明 |")
    lines.append("|---|---|---:|---|---|")
    for m in metrics:
        mark = "✅" if m.success else "❌"
        lines.append(f"| {m.tid} | {mark} | {m.n_steps} | "
                     f"{', '.join(m.tools_called) or '-'} | {m.reason[:60]} |")
    lines.append("")

    # 失败归因
    fails = [m for m in metrics if not m.success]
    if fails:
        lines.append("## 失败归因")
        lines.append("")
        cats = Counter()
        for m in fails:
            if m.error:
                cats["执行异常"] += 1
            elif not m.tool_selection_ok:
                cats["工具选择错"] += 1
            elif m.reason.startswith("✗ 未提及") or m.reason.startswith("✗ 未调用"):
                cats["能力不足"] += 1
            else:
                cats["其他"] += 1
        for k, v in cats.most_common():
            lines.append(f"- **{k}**: {v} 个")
        lines.append("")

    lines.append("## 优化方向")
    lines.append("")
    if succ / max(n, 1) < 0.65:
        lines.append("- ❌ 完成率未达标。优先排查：工具 schema 描述是否清晰、"
                     "system prompt 是否给了足够的工具使用说明、模型是否需要在"
                     "工具调用数据上做 SFT（见 PLAN.md 加餐方向）")
    else:
        lines.append("- ✓ 完成率达标")
    if unnec / max(n, 1) > 1:
        lines.append(f"- ⚠ 平均 {unnec / max(n, 1):.1f} 次不必要调用 → "
                     "在奖励函数里加 unnecessary_call 惩罚（已实现在 src/train/rewards.py）")
    if sum(steps) / max(len(steps), 1) > 4:
        lines.append("- ⚠ 平均轮数偏高 → 检查是否有重复调用同一工具（死循环）")
    lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Agent 评测报告 → {out_path}")
    print(f"  任务完成率 {pct(succ)}  |  工具选择 {pct(sel)}  |  "
          f"平均 {sum(steps) / max(len(steps), 1):.1f} 轮  |  ¥{cost:.4f}")
    return out_path


# ---------------------------------------------------------------------------
# 自检（用假 Agent 跑通流程）
# ---------------------------------------------------------------------------


async def _fake_agent(query: str, images: list[str]) -> dict:
    """一个按关键词规则行动的假 Agent，用来验证评测框架本身。"""
    q = query
    if "快递" in q:
        return {"reply": "您的快递正在运输中，预计明天到达。", "n_steps": 2,
                "tool_calls": [{"action": "lookup_order", "action_input": {}},
                               {"action": "shipping_status", "action_input": {}}],
                "tokens_in": 800, "tokens_out": 60, "state": {}}
    if "退" in q and "投诉" not in q:
        return {"reply": "我帮您查一下退货资格，已为您创建退货申请。", "n_steps": 3,
                "tool_calls": [{"action": "lookup_order", "action_input": {}},
                               {"action": "check_return_eligibility", "action_input": {}},
                               {"action": "start_return", "action_input": {}}],
                "tokens_in": 1200, "tokens_out": 90,
                "state": {"return_created": True}}
    if "码" in q and "货" in q:
        return {"reply": "M 码目前有货，可以直接下单。", "n_steps": 2,
                "tool_calls": [{"action": "check_stock", "action_input": {"size": "M"}}],
                "tokens_in": 700, "tokens_out": 40, "state": {}}
    if "线头" in q:
        return {"reply": "从图上看这是一处轻微的线头，属于正常工艺范围。", "n_steps": 1,
                "tool_calls": [], "tokens_in": 900, "tokens_out": 50, "state": {}}
    if "投诉" in q:
        return {"reply": "非常抱歉给您带来不好的体验，我理解您的心情，"
                         "这个情况我帮您转接人工客服处理。", "n_steps": 1,
                "tool_calls": [], "tokens_in": 600, "tokens_out": 60, "state": {}}
    return {"reply": "根据吊牌标注，这件是 95% 棉 5% 氨纶。", "n_steps": 1,
            "tool_calls": [], "tokens_in": 600, "tokens_out": 40, "state": {}}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Agent 评测")
    ap.add_argument("--self-test", action="store_true",
                    help="用假 Agent 跑通评测框架")
    ap.add_argument("--out", default="reports/agent_eval_v1.md")
    args = ap.parse_args()

    if args.self_test:
        print("=" * 74)
        print("Agent 评测框架自检（使用假 Agent）")
        print("=" * 74)
        print()
        metrics = asyncio.run(run_agent_eval(_fake_agent))
        build_agent_report(metrics, args.out)
        print()
        print("框架跑通。真实使用时传入你自己的 agent_run_fn：")
        print("  from src.agent.agent import CXAgent")
        print("  agent = CXAgent(...)")
        print("  metrics = asyncio.run(run_agent_eval(agent.arun))")
    else:
        print("用法：")
        print("  python -m src.eval.agent_eval --self-test   # 先验证框架")
        print("  # 然后在 Day 36 接入真实 Agent")
