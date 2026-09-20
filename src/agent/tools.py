"""
Agent 工具：schema 定义 + 实现 + 幂等 + 结构化错误。

对应讲义 docs/10-agent.md「工具设计原则」，对应计划 Day 31。

四条设计原则（都在代码里体现）：
  ① 任务级而非数据级 —— 一次调用拿到有用结果，不用调 5 次
  ② 参数给默认值和枚举 —— 减少模型填错
  ③ 有副作用的工具必须幂等 —— 否则用户点两次就出两张退货单
  ④ 错误返回可被模型理解 —— 带 suggestion，模型知道下一步干什么
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 统一返回结构
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error_code: Optional[str] = None
    message: str = ""
    suggestion: str = ""          # ⭐ 让模型知道下一步该干什么
    from_cache: bool = False      # 幂等命中

    def to_observation(self) -> str:
        """转成给模型看的文本。

        注意：不要把 JSON 原样塞进去（模型读起来费劲），
        整理成简洁的自然语言 + 关键字段。
        """
        if self.success:
            if isinstance(self.data, (dict, list)):
                return json.dumps(self.data, ensure_ascii=False, indent=2)
            return str(self.data)
        parts = [f"[失败] {self.message}"]
        if self.error_code:
            parts.append(f"错误码: {self.error_code}")
        if self.suggestion:
            parts.append(f"建议: {self.suggestion}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    needs_idempotency: bool = False       # 有副作用的工具设为 True

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Mock 数据层（Day 31–33 先用 mock，Day 40 换成 Shopify）
# ---------------------------------------------------------------------------

MOCK_ORDERS = {
    "s_demo_1": [
        {"order_id": "SO-2024-1001", "product": "圆领宽松针织毛衣 米白 M",
         "status": "shipped", "tracking": "SF1234567890",
         "carrier": "顺丰", "eta": "2026-09-22", "amount": 299.00,
         "shipped_at": "2026-09-18", "delivered_at": None},
    ],
    "s_demo_2": [
        {"order_id": "SO-2024-2043", "product": "高腰阔腿牛仔裤 深蓝 L",
         "status": "delivered", "tracking": "YT9876543210",
         "carrier": "圆通", "eta": "2026-09-15", "amount": 359.00,
         "shipped_at": "2026-09-11", "delivered_at": "2026-09-15"},
    ],
}

MOCK_STOCK = {
    ("SKU-MW", "M"): {"available": 12, "restock_date": None},
    ("SKU-MW", "L"): {"available": 0, "restock_date": "2026-09-28"},
    ("SKU-MW", "S"): {"available": 5, "restock_date": None},
    ("SKU-JN", "L"): {"available": 3, "restock_date": None},
}

RETURN_POLICY = {
    "window_days": 7,
    "conditions": [
        "商品未穿着、未洗涤、吊牌完整",
        "质量问题不受 7 天限制，凭照片可退",
        "特价商品仅支持换货",
    ],
    "quality_issue_requires_photo": True,
}

MOCK_RETURNS: dict[str, dict] = {}          # idempotency_key -> return record
MOCK_IDEMPOTENCY: dict[str, ToolResult] = {}


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def lookup_order(session_id: str = "", order_id: str = "") -> ToolResult:
    """查订单。任务级设计：一次拿到该会话/订单的全部信息。"""
    if order_id:
        for orders in MOCK_ORDERS.values():
            for o in orders:
                if o["order_id"] == order_id:
                    return ToolResult(True, o)
        return ToolResult(
            False, error_code="ORDER_NOT_FOUND",
            message=f"未找到订单 {order_id}",
            suggestion="请用户确认订单号，或让其提供下单时的手机号后四位",
        )

    orders = MOCK_ORDERS.get(session_id, [])
    if not orders:
        return ToolResult(
            False, error_code="NO_ORDER_IN_SESSION",
            message="当前会话没有关联任何订单",
            suggestion="请用户提供订单号，或确认是否在本店下单",
        )
    return ToolResult(True, orders if len(orders) > 1 else orders[0])


def shipping_status(session_id: str = "", order_id: str = "") -> ToolResult:
    r = lookup_order(session_id, order_id)
    if not r.success:
        return r
    o = r.data[0] if isinstance(r.data, list) else r.data

    status_map = {
        "pending": "还未发货",
        "shipped": f"运输中，承运商 {o['carrier']}，单号 {o['tracking']}",
        "delivered": f"已签收（{o.get('delivered_at')}）",
        "returning": "退货中",
    }
    return ToolResult(True, {
        "order_id": o["order_id"],
        "status": status_map.get(o["status"], o["status"]),
        "carrier": o["carrier"],
        "tracking": o["tracking"],
        "eta": o.get("eta"),
    })


def check_stock(sku_id: str = "", size: str = "", color: str = "") -> ToolResult:
    """查库存。size/color 留空表示查全部。"""
    sku = sku_id or "SKU-MW"
    if not size:
        all_sizes = {k[1]: v for k, v in MOCK_STOCK.items() if k[0] == sku}
        if not all_sizes:
            return ToolResult(
                False, error_code="SKU_NOT_FOUND",
                message=f"未找到商品 {sku}",
                suggestion="请先通过以图搜图或商品名确认 SKU",
            )
        return ToolResult(True, {
            "sku": sku,
            "sizes": {s: ("有货" if v["available"] > 0
                          else f"缺货，预计 {v['restock_date']} 补货")
                      for s, v in all_sizes.items()},
        })

    v = MOCK_STOCK.get((sku, size.upper()))
    if v is None:
        return ToolResult(
            False, error_code="SIZE_NOT_FOUND",
            message=f"{sku} 没有 {size} 这个尺码",
            suggestion=f"该商品可选尺码为 "
                       f"{sorted(k[1] for k in MOCK_STOCK if k[0] == sku)}",
        )
    if v["available"] <= 0:
        return ToolResult(True, {
            "sku": sku, "size": size, "available": 0,
            "restock_date": v["restock_date"],
            "note": f"该尺码缺货，预计 {v['restock_date']} 补货，可设置到货提醒",
        })
    return ToolResult(True, {"sku": sku, "size": size,
                             "available": v["available"], "note": "有货"})


def check_return_eligibility(session_id: str = "",
                             order_id: str = "") -> ToolResult:
    """查退货资格。一次调用返回能否退 + 原因（任务级设计）。"""
    r = lookup_order(session_id, order_id)
    if not r.success:
        return r
    o = r.data[0] if isinstance(r.data, list) else r.data

    from datetime import datetime, timedelta
    if o["status"] == "delivered" and o.get("delivered_at"):
        d = datetime.strptime(o["delivered_at"], "%Y-%m-%d")
        days = (datetime.now() - d).days
        if days <= RETURN_POLICY["window_days"]:
            return ToolResult(True, {
                "eligible": True,
                "days_since_delivery": days,
                "window_days": RETURN_POLICY["window_days"],
                "remaining_days": RETURN_POLICY["window_days"] - days,
                "conditions": RETURN_POLICY["conditions"],
                "note": "在 7 天无理由期内，可以退货",
            })
        return ToolResult(True, {
            "eligible": False,
            "days_since_delivery": days,
            "reason": f"已超过 {RETURN_POLICY['window_days']} 天无理由退货期",
            "exception": "如为质量问题，凭照片仍可申请，不受此限制",
            "note": "超期，但质量问题仍可特殊处理",
        })

    return ToolResult(True, {
        "eligible": False,
        "reason": f"订单状态为 {o['status']}，尚未签收",
        "note": "未签收的订单可以拒收或联系快递退回",
    })


def start_return(session_id: str = "", order_id: str = "",
                 reason: str = "", idempotency_key: str = "") -> ToolResult:
    """创建退货申请。**有副作用，必须幂等。**

    幂等实现：用 idempotency_key 做唯一键。
    没有 key 时自动生成（基于 session+order+reason 的 hash）。
    重复调用返回第一次的结果，不创建新单。
    """
    key = idempotency_key or hashlib.md5(
        f"{session_id}|{order_id}|{reason}".encode()
    ).hexdigest()[:16]

    if key in MOCK_IDEMPOTENCY:
        r = MOCK_IDEMPOTENCY[key]
        r_cached = ToolResult(r.success, r.data, r.error_code, r.message,
                              r.suggestion, from_cache=True)
        return r_cached

    elig = check_return_eligibility(session_id, order_id)
    if elig.success and isinstance(elig.data, dict) and not elig.data.get("eligible"):
        if "质量问题" not in reason and "质量" not in reason:
            res = ToolResult(
                False, error_code="NOT_ELIGIBLE",
                message=elig.data.get("reason", "不符合退货条件"),
                suggestion=elig.data.get("exception",
                                        "请用户说明具体问题，或转人工处理"),
            )
            MOCK_IDEMPOTENCY[key] = res
            return res

    rid = f"RET-{int(time.time()) % 100000:05d}"
    rec = {
        "return_id": rid,
        "order_id": order_id or "SO-2024-1001",
        "reason": reason,
        "status": "created",
        "next_step": "请在 24 小时内寄回，寄回后填写单号",
        "idempotency_key": key,
    }
    MOCK_RETURNS[key] = rec
    res = ToolResult(True, rec)
    MOCK_IDEMPOTENCY[key] = res
    return res


def product_qa(question: str = "", sku_id: str = "") -> ToolResult:
    """商品知识问答（走 RAG）。Day 32 接上 retriever 后替换实现。"""
    try:
        from .retriever import get_default_retriever
        ret = get_default_retriever()
        if ret is not None:
            hits = ret.search(question, top_k=3)
            if hits:
                return ToolResult(True, {"passages": hits})
    except Exception:
        pass

    return ToolResult(True, {
        "passages": [
            "材质：95% 棉 5% 氨纶，克重 280g",
            "洗护：可机洗，冷水，勿漂白，建议平铺晾干",
            "版型：宽松直筒，建议按平时尺码购买",
        ],
        "note": "（mock 数据，Day 32 后接入真实 RAG）",
    })


def escalate_to_human(reason: str = "", session_id: str = "") -> ToolResult:
    """转人工。永远成功（不能因为转人工失败而卡住）。"""
    return ToolResult(True, {
        "ticket_id": f"CS-{int(time.time()) % 100000:05d}",
        "queue": "普通咨询",
        "eta_minutes": 3,
        "note": "已创建工单，客服将在 3 分钟内接入，聊天记录已同步",
    })


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="lookup_order",
        description=(
            "查询订单信息。当用户询问订单状态、物流、价格、购买记录时使用。"
            "不提供 order_id 时会查当前会话关联的订单。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string",
                             "description": "订单号，如 SO-2024-1001。留空则查当前会话的订单"},
                "session_id": {"type": "string", "description": "会话 ID"},
            },
            "required": [],
        },
        fn=lookup_order,
    ),
    Tool(
        name="shipping_status",
        description="查询物流状态。当用户问「快递到哪了」「什么时候到」时使用。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": [],
        },
        fn=shipping_status,
    ),
    Tool(
        name="check_stock",
        description=(
            "查询商品库存。当用户询问某尺码/颜色是否有货、何时补货时使用。"
            "size 留空会返回全部尺码的库存情况。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "sku_id": {"type": "string", "description": "商品 SKU，可从图片检索结果获得"},
                "size": {"type": "string", "enum": ["XS", "S", "M", "L", "XL", "XXL"],
                         "description": "尺码，留空则查全部"},
                "color": {"type": "string", "description": "颜色，留空则查全部"},
            },
            "required": ["sku_id"],
        },
        fn=check_stock,
    ),
    Tool(
        name="check_return_eligibility",
        description=(
            "查询退货资格。当用户询问「能不能退」「还能退吗」时使用。"
            "一次调用即返回能否退、剩余天数、以及条件。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": [],
        },
        fn=check_return_eligibility,
    ),
    Tool(
        name="start_return",
        description=(
            "创建退货申请。**这是有副作用的操作**，只有在确认用户明确要退货、"
            "且已确认符合条件后才能调用。不要替用户做决定。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "session_id": {"type": "string"},
                "reason": {"type": "string", "description": "退货原因，用户的原话"},
                "idempotency_key": {
                    "type": "string",
                    "description": "幂等键。同一个退货请求重复调用时传相同的值，避免重复创建",
                },
            },
            "required": ["reason"],
        },
        fn=start_return,
        needs_idempotency=True,
    ),
    Tool(
        name="product_qa",
        description=(
            "查询商品知识库。当用户询问材质、洗护、尺码建议、政策条款等"
            "需要查资料的问题时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "要查询的问题"},
                "sku_id": {"type": "string"},
            },
            "required": ["question"],
        },
        fn=product_qa,
    ),
    Tool(
        name="escalate_to_human",
        description=(
            "转接人工客服。当遇到以下情况时使用：用户情绪激烈/投诉、"
            "涉及赔偿争议、问题超出你的处理范围、用户明确要求人工。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "转人工的原因"},
                "session_id": {"type": "string"},
            },
            "required": ["reason"],
        },
        fn=escalate_to_human,
    ),
]


TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------


def execute_tool(name: str, args: dict,
                 allowed: Optional[list[str]] = None) -> ToolResult:
    """执行工具。严格校验，绝不执行未知工具。

    参数白名单校验很重要：模型可能编造参数名，或者塞进来奇怪的键。
    直接 **args 展开会把 TypeError 泄给模型，产生混乱的 observation。
    """
    if allowed is not None and name not in allowed:
        return ToolResult(
            False, error_code="TOOL_NOT_ALLOWED",
            message=f"工具 {name} 在当前场景不可用",
            suggestion=f"可用工具: {allowed}",
        )

    tool = TOOL_MAP.get(name)
    if tool is None:
        return ToolResult(
            False, error_code="UNKNOWN_TOOL",
            message=f"不存在的工具: {name}",
            suggestion=f"可用工具: {list(TOOL_MAP.keys())}",
        )

    # 只保留 schema 里声明过的参数
    declared = set(tool.parameters.get("properties", {}).keys())
    clean = {k: v for k, v in (args or {}).items() if k in declared}
    dropped = set(args or {}) - declared
    if dropped:
        print(f"  ⚠️ 丢弃未声明参数: {dropped}")

    try:
        return tool.fn(**clean)
    except Exception as e:      # noqa: BLE001
        return ToolResult(
            False, error_code="TOOL_EXEC_ERROR",
            message=f"工具执行出错: {type(e).__name__}",
            suggestion="换一种方式获取信息，或转人工",
        )


def tools_schema(allowed: Optional[list[str]] = None) -> list[dict]:
    """给模型的工具定义列表。"""
    tools = TOOLS if allowed is None else [t for t in TOOLS if t.name in allowed]
    return [t.schema() for t in tools]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def run_tests():
    print("=" * 78)
    print("工具层自检")
    print("=" * 78)

    # 1. 正常查询
    r = lookup_order("s_demo_1")
    print(f"\n[1] 查订单: success={r.success}")
    assert r.success
    print(f"    {json.dumps(r.data, ensure_ascii=False)[:100]}")

    # 2. 结构化错误
    r = lookup_order(order_id="NOT-EXIST")
    print(f"\n[2] 查不存在的订单: success={r.success}")
    print(f"    error_code={r.error_code}")
    print(f"    suggestion={r.suggestion}")
    assert not r.success and r.suggestion, "错误必须带 suggestion"
    print("    ✓ 错误可被模型理解")

    # 3. 库存
    r = check_stock("SKU-MW", "L")
    print(f"\n[3] 查缺货尺码: available={r.data.get('available')}")
    print(f"    note={r.data.get('note')}")
    assert r.data["available"] == 0

    # 4. ⭐ 幂等（Day 33 的核心验收）
    print(f"\n[4] 幂等测试（连续调用 3 次相同的退货请求）")
    keys = set()
    for i in range(3):
        r = start_return("s_demo_2", "SO-2024-2043", "尺码不合适")
        keys.add(r.data["return_id"])
        print(f"    第 {i+1} 次: return_id={r.data['return_id']} "
              f"from_cache={r.from_cache}")
    assert len(keys) == 1, f"幂等失效！创建了 {len(keys)} 个退货单"
    print("    ✓ 三次调用只创建了一个退货单（幂等生效）")

    # 5. 资格校验
    res = start_return("s_demo_1", "SO-2024-1001", "不想要了")
    print(f"\n[5] 未签收订单申请退货: success={res.success}")
    print(f"    message={res.message}")
    print(f"    suggestion={res.suggestion}")

    # 6. 工具白名单
    r = execute_tool("start_return", {}, allowed=["lookup_order", "check_stock"])
    print(f"\n[6] 受限场景调用禁用工具: error_code={r.error_code}")
    assert r.error_code == "TOOL_NOT_ALLOWED"

    # 7. 未知工具（工具幻觉防护）
    r = execute_tool("send_email", {})
    print(f"[7] 调用不存在的工具: error_code={r.error_code}")
    assert r.error_code == "UNKNOWN_TOOL"
    print("    ✓ 工具幻觉被拦下")

    # 8. 未声明参数被丢弃
    r = execute_tool("check_stock", {"sku_id": "SKU-MW", "size": "M",
                                     "evil_param": "x"})
    print(f"[8] 未声明参数处理: success={r.success}")
    assert r.success

    # 9. schema
    print(f"\n[9] 工具数量: {len(TOOLS)}")
    for t in TOOLS:
        flag = " [需幂等]" if t.needs_idempotency else ""
        print(f"    {t.name:<26}{flag}")

    print("\n" + "=" * 78)
    print("✓ 全部通过")


if __name__ == "__main__":
    run_tests()
