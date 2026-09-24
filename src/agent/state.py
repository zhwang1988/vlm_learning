#!/usr/bin/env python3
"""
会话状态 —— 从 agent.py 里抽出来的独立模块。

**为什么独立**：状态管理不该依赖 torch / transformers。
抽出去之后它可以被单独单元测试、可以序列化进 Redis、可以被多实例共享。

    python -m src.agent.state      # 自检

设计要点：
  1. `VisualEvidence`：解决 VLM Agent 的经典 bug —— **第二轮图片丢了**。
     做法是每隔几轮就把图里的关键信息抽成**文本证据**存下来，
     后续轮次不再依赖原图（原图可能已从上下文里被截掉）。
  2. 全部字段可 JSON 序列化（存 Redis / DB / 文件都行）。
  3. `IdempotencyKey` 是确定性的 —— 同样的会话 + 工具 + 参数永远得到同一个键。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 视觉证据
# ---------------------------------------------------------------------------

@dataclass
class VisualEvidence:
    """一条「图里有什么」的文本化证据。

    第一轮就把关键信息抽成文本，后面轮次就算看不到原图也能正常回答。
    """
    image_ref: str                       # 图片来源标识（路径 / URL / hash）
    description: str                     # 关键信息摘要（商品名、颜色、文字、缺陷…）
    turn_index: int = 0                  # 是在第几轮抽的
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_visual_evidence(image_ref: str, description: str,
                            turn_index: int = 0, confidence: float = 1.0
                            ) -> VisualEvidence:
    """工厂函数 —— 把一段描述包成证据对象。"""
    return VisualEvidence(image_ref=image_ref, description=description.strip(),
                          turn_index=turn_index, confidence=confidence)


# ---------------------------------------------------------------------------
# 幂等键
# ---------------------------------------------------------------------------

def idempotency_key(session_id: str, tool_name: str,
                    args: dict[str, Any]) -> str:
    """确定性幂等键。

    **不能用随机数 / 时间戳** —— 那样每次调用都是新键，幂等完全失效。
    必须是（会话 + 工具 + 参数）的确定性哈希。
    """
    payload = json.dumps([session_id, tool_name, _canonical(args)],
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _canonical(obj: Any) -> Any:
    """把嵌套结构规范化，保证 dict 顺序不影响哈希。"""
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str                            # user / assistant / tool
    content: str
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class SessionState:
    """一个会话的全部可持久化状态。"""

    def __init__(self, session_id: str, shop: str = ""):
        self.session_id = session_id
        self.shop = shop
        self.turns: list[Turn] = []
        self.slots: dict[str, Any] = {}          # 提取出的槽位：订单号 / 尺码 / 颜色…
        self.visual_evidence: list[VisualEvidence] = []
        self.tool_calls: list[dict] = []         # {tool, args, ok, ts}
        self.cost_tokens: int = 0
        self.cost_money: float = 0.0
        self.created_at = time.time()

    # ---- 对话 ----

    def add_turn(self, role: str, content: str, **meta) -> Turn:
        t = Turn(role=role, content=content, meta=meta)
        self.turns.append(t)
        return t

    def last_user_message(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return ""

    def context(self, n: int = 6) -> list[Turn]:
        """最近 n 轮 —— 别把全history塞进 prompt，会爆。"""
        return self.turns[-n:]

    # ---- 槽位 ----

    def set_slot(self, key: str, value: Any) -> None:
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)

    # ---- 视觉证据（关键）----

    def add_visual_evidence(self, image_ref: str, description: str,
                            confidence: float = 1.0) -> VisualEvidence:
        ev = extract_visual_evidence(image_ref, description,
                                     turn_index=len(self.turns),
                                     confidence=confidence)
        self.visual_evidence.append(ev)
        return ev

    def visual_context(self, max_chars: int = 400) -> str:
        """给模型看的「图片记忆」—— 这就是解决「图丢了」的东西。"""
        if not self.visual_evidence:
            return ""
        parts = [f"[图{i+1}] {e.description}" for i, e in enumerate(self.visual_evidence)]
        text = "；".join(parts)
        return text[:max_chars]

    # ---- 工具调用 ----

    def record_tool_call(self, tool: str, args: dict, ok: bool,
                         tokens: int = 0, money: float = 0.0) -> str:
        key = idempotency_key(self.session_id, tool, args)
        self.tool_calls.append({"tool": tool, "args": args, "ok": ok,
                                "key": key, "ts": time.time()})
        self.cost_tokens += tokens
        self.cost_money += money
        return key

    def recent_actions(self, n: int = 3) -> list[tuple[str, str]]:
        """最近 n 次的 (工具, 幂等键) —— 给护栏做重复检测用。"""
        return [(c["tool"], c["key"]) for c in self.tool_calls[-n:]]

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "shop": self.shop,
            "turns": [t.to_dict() for t in self.turns],
            "slots": self.slots,
            "visual_evidence": [e.to_dict() for e in self.visual_evidence],
            "tool_calls": self.tool_calls,
            "cost_tokens": self.cost_tokens,
            "cost_money": self.cost_money,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        s = cls(d.get("session_id", "unknown"), d.get("shop", ""))
        s.turns = [Turn(**t) for t in d.get("turns", [])]
        s.slots = d.get("slots", {})
        s.visual_evidence = [VisualEvidence(**e) for e in d.get("visual_evidence", [])]
        s.tool_calls = d.get("tool_calls", [])
        s.cost_tokens = d.get("cost_tokens", 0)
        s.cost_money = d.get("cost_money", 0.0)
        s.created_at = d.get("created_at", time.time())
        return s

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "SessionState":
        return cls.from_dict(json.loads(text))

    # ---- 展示 ----

    def summary(self) -> str:
        return (f"SessionState({self.session_id}) "
                f"轮次 {len(self.turns)} · 槽位 {self.slots or '{}'} · "
                f"视觉证据 {len(self.visual_evidence)} · "
                f"工具调用 {len(self.tool_calls)} · "
                f"成本 {self.cost_money:.4f} 元 / {self.cost_tokens} tokens")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("=" * 72)
    print("SessionState 自检")
    print("=" * 72)

    s = SessionState("s_test_1", shop="demo.myshopify.com")
    s.add_turn("user", "这件米白针织衫有货吗？", image=True)
    s.add_visual_evidence("img/white_knit.jpg",
                          "米白色针织衫，圆领，标签写着 95% 棉；有轻微起球")
    s.add_turn("assistant", "M/L 码有货，S 码预计 3 天补货。")
    s.set_slot("sku", "SKU-1001")
    s.set_slot("size", "M")

    # 第二轮：用户不再发图
    s.add_turn("user", "那这个多少钱？")
    print("\n[1] 第二轮「图丢了」问题")
    print("    用户问：", s.last_user_message())
    print("    视觉上下文：", s.visual_context())
    assert "针织衫" in s.visual_context(), "视觉证据应能在第二轮被取回"
    print("    ✓ 第二轮仍能拿到图里的信息")

    print("\n[2] 幂等键确定性")
    k1 = idempotency_key("s1", "start_return", {"order_id": "A1", "reason": "尺码"})
    k2 = idempotency_key("s1", "start_return", {"reason": "尺码", "order_id": "A1"})
    k3 = idempotency_key("s2", "start_return", {"order_id": "A1", "reason": "尺码"})
    print(f"    同会话同参数（键序不同）: {k1} == {k2} → {k1 == k2}")
    print(f"    异会话同参数            : {k1} != {k3} → {k1 != k3}")
    assert k1 == k2 and k1 != k3
    print("    ✓ 确定性正确（dict 顺序不影响结果）")

    print("\n[3] 序列化往返")
    s.record_tool_call("check_stock", {"sku": "SKU-1001"}, ok=True,
                       tokens=180, money=0.0002)
    s2 = SessionState.from_json(s.to_json())
    assert s2.to_json() == s.to_json()
    print("    ✓ JSON 往返一致（可存 Redis / DB）")

    print("\n[4] 摘要")
    print("   ", s2.summary())

    print("\n[5] 依赖检查（必须不依赖 torch）")
    import sys
    bad = [m for m in sys.modules if m.startswith(("torch", "transformers"))]
    print("    已加载的 torch 相关模块:", bad or "无")
    assert not bad, "state.py 不该引入 torch"
    print("    ✓ 状态管理与模型推理完全解耦")

    print("\n" + "=" * 72)
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
