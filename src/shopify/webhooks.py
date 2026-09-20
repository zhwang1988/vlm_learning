"""
Shopify Webhook 接收与分发。

对应讲义 docs/11-shopify.md「Webhook」，对应计划 Day 40。

两个必须处理的点：
  ① **HMAC 校验**：用 X-Shopify-Hmac-Sha256 头验证请求真的来自 Shopify。
     不校验等于把接口裸奔。
  ② **幂等 + 快速响应**：Shopify 要求 5 秒内返回 200，超时会重试。
     所以收到后立刻入队，用 X-Shopify-Webhook-Id 做幂等键异步处理。

三个 GDPR 强制 webhook：
  customers/data_request  —— 商家要导出一个顾客的数据
  customers/redact        —— 商家要删除某个顾客的数据
  shop/redact             —— 卸载 48 小时后，删除整个店铺的数据
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .auth import verify_webhook_hmac


# ---------------------------------------------------------------------------
# 幂等存储
# ---------------------------------------------------------------------------


class IdempotencyStore:
    """已处理的 webhook ID 记录。Shopify 重试时会带相同的 webhook id。"""

    def __init__(self, path: str | Path = "data/webhook_ids.json", max_memory: int = 5000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: deque[str] = deque(maxlen=max_memory)
        self.seen_set: set[str] = set()
        if self.path.exists():
            try:
                for k in json.loads(self.path.read_text(encoding="utf-8")):
                    self.seen.append(k)
                    self.seen_set.add(k)
            except Exception:
                pass
        self._dirty = 0

    def check_and_add(self, wid: str) -> bool:
        """返回 True 表示是新事件（应处理）；False 表示重复（跳过）。"""
        if not wid:
            return True          # 没有 id 就当作新事件处理
        if wid in self.seen_set:
            return False
        self.seen.append(wid)
        self.seen_set.add(wid)
        if len(self.seen) > self.seen.maxlen:
            self.seen_set.discard(self.seen[0])
        self._dirty += 1
        if self._dirty >= 20:
            self.flush()
        return True

    def flush(self):
        self.path.write_text(json.dumps(list(self.seen)), encoding="utf-8")
        self._dirty = 0


# ---------------------------------------------------------------------------
# 简单任务队列（生产建议换 Redis + worker）
# ---------------------------------------------------------------------------


class TaskQueue:
    """内存队列 + 后台 worker。够单机用。

    为什么需要队列：
      Shopify 要求 5 秒返回。而「商品更新 → 重新向量化」可能要几秒。
      所以必须「收到就返回 200，后台慢慢处理」。
    """

    def __init__(self, maxsize: int = 1000):
        self.q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.handlers: dict[str, Callable[[str, dict], Awaitable[None]]] = {}
        self._worker: Optional[asyncio.Task] = None

    def register(self, topic: str,
                 handler: Callable[[str, dict], Awaitable[None]]):
        self.handlers[topic] = handler

    async def enqueue(self, topic: str, shop: str, payload: dict):
        try:
            self.q.put_nowait((topic, shop, payload))
        except asyncio.QueueFull:
            print(f"⚠ 队列满，丢弃 {topic}")

    async def _run(self):
        while True:
            topic, shop, payload = await self.q.get()
            handler = self.handlers.get(topic)
            if handler is None:
                print(f"  （无 handler: {topic}）")
                self.q.task_done()
                continue
            for attempt in range(3):
                try:
                    await handler(shop, payload)
                    break
                except Exception as e:      # noqa: BLE001
                    print(f"  ✗ {topic} 处理失败 (第{attempt+1}次): {str(e)[:120]}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
            self.q.task_done()

    def start(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def stop(self):
        if self._worker:
            self._worker.cancel()


QUEUE = TaskQueue()
IDEMPOTENCY = IdempotencyStore()


# ---------------------------------------------------------------------------
# Handler 注册
# ---------------------------------------------------------------------------


async def handle_product_changed(shop: str, payload: dict):
    """商品创建/更新 → 更新向量索引（Day 40 的核心）。"""
    from .indexer import update_product_index
    await update_product_index(shop, payload)


async def handle_product_deleted(shop: str, payload: dict):
    from .indexer import delete_product_index
    delete_product_index(shop, str(payload.get("id", "")))


async def handle_order_created(shop: str, payload: dict):
    """订单创建 → 关联到会话（可选）。"""
    print(f"  [order] {shop} 新订单 {payload.get('name')}")
    # 实际实现：把 order 写进 DB，Agent 的 lookup_order 从此能查到真实订单


async def handle_refund(shop: str, payload: dict):
    print(f"  [refund] {shop} 退款 {payload.get('id')}")


async def handle_uninstalled(shop: str, payload: dict):
    """⭐ 必须：卸载时清理店铺数据。

    不清理的话：
      - 数据合规上过不了审核
      - 商家重新安装时会有脏数据
    """
    print(f"  [uninstall] 清理 {shop} 的数据")
    from .auth import TokenStore
    TokenStore().delete(shop)
    # 实际实现：删除该店铺的索引、会话、消息、订阅记录
    try:
        from .indexer import drop_shop_index
        drop_shop_index(shop)
    except Exception:
        pass


# --- GDPR 三个强制 webhook ---


async def handle_customers_data_request(shop: str, payload: dict):
    """顾客要求导出自己的数据。

    你需要在规定时间内把该顾客在你系统里的所有数据打包交给商家。
    客服场景里，数据 = 聊天记录 + 上传的图片 + 关联的订单号。
    """
    customer_id = payload.get("customer", {}).get("id")
    print(f"  [GDPR/data_request] {shop} 顾客 {customer_id} 请求导出数据")
    # 实际实现：查 DB，把该顾客的会话/消息导出成 JSON，通知商家


async def handle_customers_redact(shop: str, payload: dict):
    """顾客要求删除自己的数据。

    ⭐ 客服场景的关键：**用户上传的图片可能有脸，必须一起删。**
    """
    customer_id = payload.get("customer", {}).get("id")
    print(f"  [GDPR/redact] {shop} 删除顾客 {customer_id} 的数据")
    # 实际实现：删除会话、消息、上传的图片文件


async def handle_shop_redact(shop: str, payload: dict):
    """卸载 48 小时后，删除整个店铺的数据。"""
    print(f"  [GDPR/shop_redact] 彻底清理 {shop}")
    from .auth import TokenStore
    TokenStore().delete(shop)
    try:
        from .indexer import drop_shop_index
        drop_shop_index(shop)
    except Exception:
        pass


HANDLERS = {
    "products/create": handle_product_changed,
    "products/update": handle_product_changed,
    "products/delete": handle_product_deleted,
    "orders/create": handle_order_created,
    "orders/updated": handle_order_created,
    "refunds/create": handle_refund,
    "app/uninstalled": handle_uninstalled,
    "customers/data_request": handle_customers_data_request,
    "customers/redact": handle_customers_redact,
    "shop/redact": handle_shop_redact,
}


def register_all():
    for topic, h in HANDLERS.items():
        QUEUE.register(topic, h)


register_all()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


@dataclass
class WebhookResult:
    status: int
    message: str
    queued: bool = False
    duplicate: bool = False


async def process_webhook(topic: str, shop: str, body: bytes,
                          hmac_header: str,
                          webhook_id: str = "") -> WebhookResult:
    """处理一个 webhook。

    流程（顺序不能变）：
      ① 校验 HMAC（不通过直接 401）
      ② 校验幂等（重复直接返回 200，不重复处理）
      ③ 入队（立刻返回）
    """
    # ① HMAC —— 先校验，不通过就别往下走
    if not verify_webhook_hmac(body, hmac_header):
        return WebhookResult(401, "HMAC 校验失败")

    # ② 幂等
    if not IDEMPOTENCY.check_and_add(webhook_id):
        return WebhookResult(200, "重复事件，已跳过", duplicate=True)

    # ③ 入队
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return WebhookResult(400, "请求体不是合法 JSON")

    topic = topic.lower().replace("_", "/")
    await QUEUE.enqueue(topic, shop, payload)
    return WebhookResult(200, "已接收", queued=True)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def run_tests():
    import base64
    import hashlib
    import hmac as _hmac

    print("=" * 78)
    print("Webhook 处理自检")
    print("=" * 78)

    secret = "test_secret"
    body = json.dumps({"id": 123, "title": "测试商品"}).encode("utf-8")
    good = base64.b64encode(
        _hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ).decode()

    async def main():
        import os
        from . import auth
        auth.CFG.api_secret = secret

        # 1. HMAC 失败
        r = await process_webhook("products/update", "s.myshopify.com",
                                  body, "bad_hmac", "wid_1")
        print(f"\n[1] 错误 HMAC → status={r.status}  {r.message}")
        assert r.status == 401

        # 2. 正常
        r = await process_webhook("products/update", "s.myshopify.com",
                                  body, good, "wid_2")
        print(f"[2] 正确 HMAC → status={r.status}  queued={r.queued}")
        assert r.status == 200 and r.queued

        # 3. 幂等：重复同一个 webhook id
        r = await process_webhook("products/update", "s.myshopify.com",
                                  body, good, "wid_2")
        print(f"[3] 重复 webhook → status={r.status}  duplicate={r.duplicate}")
        assert r.duplicate, "幂等失效！"
        print("    ✓ 重复事件被跳过（Shopify 重试不会导致重复处理）")

        # 4. 新 id 正常
        r = await process_webhook("products/update", "s.myshopify.com",
                                  body, good, "wid_3")
        print(f"[4] 新 webhook → duplicate={r.duplicate}")
        assert not r.duplicate

        # 5. 队列处理
        QUEUE.start()
        await asyncio.sleep(0.3)
        await QUEUE.stop()
        print("[5] 队列 worker 已跑过（上面应有 [order]/[uninstall] 之类的日志）")

        # 6. 所有 topic 都有 handler
        print(f"\n[6] 注册的 topic ({len(HANDLERS)} 个):")
        for t in HANDLERS:
            gdpr = " ← GDPR 强制" if t.startswith("customers/") or t == "shop/redact" else ""
            print(f"    {t}{gdpr}")

    asyncio.run(main())

    print("\n" + "=" * 78)
    print("✓ 全部通过")
    print()
    print("关键设计（Day 40/44 的验收点）：")
    print("  ① HMAC 先校验，不通过直接 401")
    print("  ② 幂等键用 X-Shopify-Webhook-Id，Shopify 重试不会重复处理")
    print("  ③ 立刻入队返回 200（Shopify 要求 5 秒内响应）")
    print("  ④ app/uninstalled 必须清理数据（审核硬性要求）")
    print("  ⑤ 三个 GDPR webhook 必须实现并返回 200")


if __name__ == "__main__":
    run_tests()
