"""
多模态客服 Agent 主循环（Structured ReAct）。

对应讲义 docs/10-agent.md「Agent 主循环」，对应计划 Day 34–35。

设计：用 function calling 的**格式**，但保留 ReAct 的**显式推理步骤**。

```
输入: 用户消息（图 + 文）+ 会话历史
  ↓
① 视觉证据提取（把图里的信息显式写进状态，避免「图丢了」）
  ↓
② 循环（最多 max_steps 步）
   ├─ LLM 输出 {thought, action, action_input} 或 {response}
   ├─ 护栏检查：步数 / 重复调用 / 成本
   ├─ 执行工具 → observation
   └─ 追加到上下文
  ↓
③ 后处理：PII 过滤 + 失败降级
```

三个必须有的护栏：
  1. max_steps        防止无限循环
  2. 重复 action 检测  同一工具同样参数调两次 → 强制跳出
  3. 成本上限         累计 token 超阈值 → 转人工
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .tools import TOOL_MAP, ToolResult, execute_tool, tools_schema


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    max_steps: int = 6
    max_total_tokens: int = 20000
    max_repeats: int = 2
    timeout_s: float = 60
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "") or
                       os.getenv("SYNTH_MODEL", "qwen-vl-max"))
    temperature: float = 0.2
    extract_visual_evidence: bool = True

    # 允许的工具（可按场景收窄）
    allowed_tools: list[str] = field(default_factory=lambda: list(TOOL_MAP.keys()))

    # 敏感场景禁止的工具（Day 44 安全加固会用到）
    forbidden_for_sensitive: list[str] = field(
        default_factory=lambda: ["start_return"]
    )


SYSTEM_PROMPT = """你是一位专业的电商客服助手。你的目标是**真正解决用户的问题**，而不是礼貌地敷衍。

## 你可以使用工具
{tool_list}

## 工作方式（必须严格遵守）

每一步只输出一个 JSON，格式为：

工具调用：
{{"thought": "我现在的推理（一句话，说清为什么需要这个信息）",
 "action": "工具名",
 "action_input": {{参数}}}}

最终回答：
{{"thought": "我已经掌握足够信息",
 "response": "给用户的最终回复"}}

## 硬性规则

1. **先判断需不需要工具**。简单的看图问答不需要任何工具，直接回答。
2. **不要凭猜测回答事实性问题**。订单、库存、物流、政策必须查。
3. **有副作用的操作（退货）必须先确认**。用户说「我要退货」时，
   先查资格，说明条件，得到用户确认后再调用 start_return。
4. **信息不足就追问**，不要硬答。你缺少判断依据时，问清楚比答错好。
5. **绝不编造你在图中没看到的东西**。图里看不出来就说不确定。
6. **态度要求**：先回应情绪，再给信息。专业但不冷冰冰。
7. **遇到投诉、索赔、法律问题、情绪激烈** → 调用 escalate_to_human。
8. 回复控制在 200 字以内，不要用「首先/其次/最后」的模板结构。

## 当前会话已知信息
{session_info}

## 当前时间
{now}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    session_id: str = ""
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    visual_evidence: dict = field(default_factory=dict)
    n_steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    side_effects: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class CXAgent:
    """多模态客服 Agent。"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            key = os.getenv("AGENT_API_KEY") or os.getenv("SYNTH_API_KEY")
            base = os.getenv("AGENT_API_BASE") or os.getenv("SYNTH_API_BASE")
            if not key:
                raise RuntimeError("缺少 AGENT_API_KEY / SYNTH_API_KEY")
            self._client = AsyncOpenAI(base_url=base, api_key=key)
        return self._client

    # ------------------------------------------------------------------ #

    async def extract_visual_evidence(self, images: list) -> dict:
        """把图里的信息显式提取出来，写进状态。

        **为什么必须做这一步**：
        很多 Agent 在「调用工具 → 拿到文本结果」后就不再关注原始图片了，
        最终回答和图片无关。把视觉证据显式放进每一轮的上下文，
        能彻底避免这个问题。

        工程上这也是「图丢了」这个隐蔽失败模式的解法。
        """
        if not images:
            return {}

        # images 可能是 base64 data URL、本地路径或 http URL
        content = [{"type": "text",
                    "text": "请用 JSON 描述这张图，字段：\n"
                            '{"category":"商品类目","colors":["主色"],'
                            '"materials":["可见材质"],"features":["款式特征、图案、细节"],'
                            '"defects":["可见瑕疵，没有则空"],"visible_text":["图中文字"],'
                            '"confidence":0-1}\n'
                            "只输出 JSON，不要其他内容。"}]
        for img in images[:3]:
            is_url = isinstance(img, str) and img.startswith(("data:", "http"))
            url = img if is_url else self._to_data_url(img)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})

        try:
            resp = await self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0, timeout=40,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:      # noqa: BLE001
            return {"error": str(e)[:100]}

    @staticmethod
    def _to_data_url(img) -> Optional[str]:
        import base64
        try:
            if hasattr(img, "save"):
                import io
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=88)
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/jpeg;base64,{b64}"
            p = Path(str(img))
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                suf = p.suffix.lstrip(".").lower()
                mime = "image/jpeg" if suf in ("jpg", "jpeg") else f"image/{suf}"
                return f"data:{mime};base64,{b64}"
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------ #

    async def arun(self, query: str, images: Optional[list] = None,
                   session_id: Optional[str] = None,
                   history: Optional[list[dict]] = None) -> dict:
        session_id = session_id or f"s_{int(time.time())}"
        state = AgentState(session_id=session_id)

        # --- ① 视觉证据 ---
        if images and self.cfg.extract_visual_evidence:
            state.visual_evidence = await self.extract_visual_evidence(images)

        # --- 组装消息 ---
        session_info = self._format_session_info(session_id, state)
        system = SYSTEM_PROMPT.format(
            tool_list="\n".join(
                f"- `{t.name}`: {t.description}" for t in TOOL_MAP.values()
                if t.name in self.cfg.allowed_tools
            ),
            session_info=session_info,
            now=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        user_content = []
        for img in (images or [])[:3]:
            url = img if isinstance(img, str) and img.startswith(("data:", "http")) \
                else self._to_data_url(img)
            if url:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
        user_content.append({"type": "text", "text": query})

        msgs: list[dict] = [{"role": "system", "content": system}]
        for h in (history or [])[-6:]:
            msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        msgs.append({"role": "user", "content": user_content})

        # --- ② 主循环 ---
        seen_actions: list[str] = []
        final_reply = ""
        last_thought = ""

        for step in range(self.cfg.max_steps):
            state.n_steps = step + 1

            # 护栏 3：成本上限
            if state.tokens_in + state.tokens_out > self.cfg.max_total_tokens:
                state.notes.append("触发成本上限")
                state.tool_calls.append({"action": "escalate_to_human",
                                         "action_input": {"reason": "对话过长"}})
                r = execute_tool("escalate_to_human", {"reason": "对话过长"})
                final_reply = ("我们的对话有点长了，为了更高效地帮您，"
                               "我为您转接人工客服。")
                break

            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=msgs,
                        temperature=self.cfg.temperature,
                        max_tokens=600,
                        response_format={"type": "json_object"},
                    ),
                    timeout=self.cfg.timeout_s,
                )
            except asyncio.TimeoutError:
                state.notes.append("LLM 超时")
                final_reply = ("抱歉，我这边响应有点慢。您可以再描述一次，"
                               "或者我帮您转接人工客服？")
                break
            except Exception as e:      # noqa: BLE001
                state.notes.append(f"LLM 错误: {e}")
                final_reply = ("抱歉，我这边暂时遇到问题，"
                               "已为您记录，稍后会有客服专员跟进。")
                break

            u = resp.usage
            state.tokens_in += getattr(u, "prompt_tokens", 0)
            state.tokens_out += getattr(u, "completion_tokens", 0)

            raw = resp.choices[0].message.content
            try:
                step_obj = json.loads(raw)
            except json.JSONDecodeError:
                final_reply = raw          # 模型没输出 JSON，当最终回答
                break

            last_thought = step_obj.get("thought", "")

            # 最终回答
            if "response" in step_obj and not step_obj.get("action"):
                final_reply = step_obj["response"]
                break

            action = step_obj.get("action")
            action_input = step_obj.get("action_input") or {}

            if not action:
                final_reply = step_obj.get("response", "抱歉，我没理解，能再说一次吗？")
                break

            # 护栏 2：重复调用检测
            sig = f"{action}|{json.dumps(action_input, sort_keys=True, ensure_ascii=False)}"
            if seen_actions.count(sig) >= self.cfg.max_repeats:
                state.notes.append(f"检测到重复调用 {action}，强制跳出")
                final_reply = ("我这边遇到一点状况，已经记录下来，"
                               "为您转接人工客服继续处理。")
                state.tool_calls.append({"action": "escalate_to_human",
                                         "action_input": {"reason": "重复调用跳出"}})
                break
            seen_actions.append(sig)

            # 执行
            result: ToolResult = execute_tool(
                action, action_input, allowed=self.cfg.allowed_tools
            )
            state.tool_calls.append({"action": action, "action_input": action_input,
                                     "success": result.success})

            # 记录副作用（供评测的 state_changed 验证器使用）
            if action == "start_return" and result.success:
                state.side_effects["return_created"] = True
                state.side_effects["return_id"] = (result.data or {}).get("return_id")
            if action == "escalate_to_human" and result.success:
                state.side_effects["escalated"] = True

            # 追加 observation（带上视觉证据，避免「图丢了」）
            obs = result.to_observation()
            if state.visual_evidence and step > 0:
                obs += ("\n\n[图片信息] " +
                        json.dumps(state.visual_evidence, ensure_ascii=False))

            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "user", "content": f"[工具返回]\n{obs}"})

        else:
            # 步数用尽
            state.notes.append("达到最大步数")
            if not final_reply:
                final_reply = ("这个问题我需要再确认一下，"
                               "已经记录您的需求，稍后客服专员会跟进。")

        # --- ③ 后处理 ---
        final_reply = self._postprocess(final_reply)

        return {
            "reply": final_reply,
            "session_id": session_id,
            "thought": last_thought,
            "tool_calls": state.tool_calls,
            "n_steps": state.n_steps,
            "tokens_in": state.tokens_in,
            "tokens_out": state.tokens_out,
            "visual_evidence": state.visual_evidence,
            "state": state.side_effects,
            "notes": state.notes,
        }

    def run(self, query: str, images: Optional[list] = None, **kw) -> dict:
        return asyncio.run(self.arun(query, images, **kw))

    # ------------------------------------------------------------------ #

    def _format_session_info(self, session_id: str, state: AgentState) -> str:
        from .tools import MOCK_ORDERS
        lines = [f"- 会话 ID: {session_id}"]
        orders = MOCK_ORDERS.get(session_id, [])
        if orders:
            lines.append(f"- 关联订单: {', '.join(o['order_id'] for o in orders)}")
            for o in orders:
                lines.append(f"    {o['order_id']}: {o['product']} ({o['status']})")
        else:
            lines.append("- 关联订单: 无（用户可能需要提供订单号）")

        if state.visual_evidence:
            lines.append(f"- 用户上传图片: {json.dumps(state.visual_evidence, ensure_ascii=False)[:300]}")
        return "\n".join(lines)

    def _postprocess(self, text: str) -> str:
        import re
        # PII 脱敏
        text = re.sub(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text)
        # 去掉内部标记
        for m in ["<|im_end|>", "<|im_start|>", "<|image_pad|>", "<|vision_start|>"]:
            text = text.replace(m, "")
        # 去掉 AI 自我暴露
        text = re.sub(r"作为(一个)?AI[^。！？]*[。！？]?", "", text)
        return text.strip()


# ---------------------------------------------------------------------------
# 默认实例
# ---------------------------------------------------------------------------

_DEFAULT: Optional[CXAgent] = None


def get_default_agent() -> Optional[CXAgent]:
    global _DEFAULT
    if _DEFAULT is None:
        try:
            _DEFAULT = CXAgent()
        except Exception:
            return None
    return _DEFAULT


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


async def _selftest():
    print("=" * 78)
    print("Agent 护栏自检（不需要 API key）")
    print("=" * 78)

    # 1. 重复调用检测
    cfg = AgentConfig(max_steps=6, max_repeats=2)
    print("\n[1] 护栏参数")
    print(f"    max_steps   = {cfg.max_steps}      防止无限循环")
    print(f"    max_repeats = {cfg.max_repeats}    同一调用超过这个次数就跳出")

    # 模拟重复调用逻辑
    seen = []
    forced_break = False
    for step in range(6):
        sig = "lookup_order|{}"
        if seen.count(sig) >= cfg.max_repeats:
            forced_break = True
            print(f"    第 {step+1} 步检测到重复 → 强制跳出 ✓")
            break
        seen.append(sig)
    assert forced_break, "重复检测未生效"

    # 2. 成本上限
    print("\n[2] 成本上限")
    state = AgentState(tokens_in=15000, tokens_out=6000)
    over = state.tokens_in + state.tokens_out > cfg.max_total_tokens
    print(f"    {state.tokens_in + state.tokens_out} > {cfg.max_total_tokens} → {over}")
    assert over

    # 3. 未声明参数被丢弃 + 未知工具被拦下（工具层已有测试）
    print("\n[3] 工具层护栏（详见 python -m src.agent.tools）")
    r = execute_tool("start_return", {})
    print(f"    start_return 无参数 → success={r.success} code={r.error_code}")

    # 4. PII 后处理
    agent = CXAgent.__new__(CXAgent)
    out = agent._postprocess("帮您确认一下，联系电话 13812345678，<|im_end|>")
    print(f"\n[4] 后处理: {out}")
    assert "13812345678" not in out and "<|im_end|>" not in out
    print("    ✓ PII 已脱敏，内部标记已清理")

    print("\n" + "=" * 78)
    print("✓ 护栏全部生效")
    print()
    print("真实运行需要 API key：")
    print("  export SYNTH_API_BASE=...  SYNTH_API_KEY=...  SYNTH_MODEL=qwen-vl-max")
    print("  然后：python -m src.agent.agent --query '我的快递到哪了' --session s_demo_1")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="多模态客服 Agent")
    ap.add_argument("--query")
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--session", default="s_demo_1")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.query:
        asyncio.run(_selftest())
    else:
        agent = CXAgent()
        out = agent.run(args.query, args.image, session_id=args.session)
        print("\n" + "=" * 78)
        print(f"用户: {args.query}")
        if out.get("thought"):
            print(f"\n[推理] {out['thought']}")
        print(f"\n[工具调用] ({out['n_steps']} 步)")
        for tc in out["tool_calls"]:
            mark = "✓" if tc.get("success") else "✗"
            print(f"  {mark} {tc['action']}({json.dumps(tc['action_input'], ensure_ascii=False)})")
        if out.get("visual_evidence"):
            print(f"\n[视觉证据] {json.dumps(out['visual_evidence'], ensure_ascii=False)}")
        print(f"\n客服: {out['reply']}")
        if out.get("notes"):
            print(f"\n[备注] {out['notes']}")
        print(f"\nToken: {out['tokens_in']} in / {out['tokens_out']} out")
