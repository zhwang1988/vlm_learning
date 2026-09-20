"""
Shopify Admin GraphQL 客户端。

对应讲义 docs/11-shopify.md「Admin API」，对应计划 Day 37。

三个必须处理好的点：
  ① 分页（默认最多 50/250 条，必须用 cursor 翻页）
  ② 限流（GraphQL 用「计算成本」，不是简单的请求数）
  ③ 重试（429 / 5xx 要指数退避）

**用 GraphQL 不用 REST**：
  REST 是漏桶（每秒 2 请求），GraphQL 是按查询复杂度算成本，
  更高效，也是官方推荐。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .auth import CFG, TokenStore, normalize_shop


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


@dataclass
class ShopifyClient:
    shop: str
    access_token: str
    api_version: str = field(default_factory=lambda: CFG.api_version)
    max_retries: int = 4
    min_available_cost: float = 100.0      # 低于这个就等等再发

    @property
    def endpoint(self) -> str:
        return f"https://{normalize_shop(self.shop)}/admin/api/{self.api_version}/graphql.json"

    @property
    def headers(self) -> dict:
        return {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------ #

    async def query(self, gql: str, variables: Optional[dict] = None,
                    retry_on_throttle: bool = True) -> dict:
        """执行一次 GraphQL 查询。带限流退避和重试。"""
        import httpx

        payload = {"query": gql, "variables": variables or {}}
        last_err = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(self.endpoint, headers=self.headers, json=payload)

                # 429：限流
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 2 ** attempt))
                    print(f"  ⚠ 限流，等待 {wait}s")
                    await asyncio.sleep(wait)
                    continue

                # 5xx：服务端错误，退避重试
                if r.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    last_err = f"HTTP {r.status_code}"
                    continue

                if r.status_code == 401:
                    raise PermissionError(f"Access token 失效或被撤销: {self.shop}")

                r.raise_for_status()
                data = r.json()

                # GraphQL 层的错误
                if "errors" in data:
                    errs = data["errors"]
                    # 限流错误也在这里返回（THROTTLED）
                    if any(e.get("extensions", {}).get("code") == "THROTTLED"
                           for e in errs) and retry_on_throttle:
                        wait = 2 ** attempt
                        print(f"  ⚠ GraphQL 限流，等待 {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(f"GraphQL 错误: {json.dumps(errs, ensure_ascii=False)[:400]}")

                # 主动检查剩余配额，提前 sleep
                cost = (data.get("extensions", {}) or {}).get("cost", {})
                status = cost.get("throttleStatus", {})
                avail = status.get("currentlyAvailable")
                if avail is not None and avail < self.min_available_cost:
                    restore = status.get("restoreRate", 50)
                    wait = max(1.0, (self.min_available_cost - avail) / max(restore, 1))
                    print(f"  ⚠ 配额不足({avail:.0f})，主动等待 {wait:.1f}s")
                    await asyncio.sleep(min(wait, 10))

                return data.get("data", {})

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_err = str(e)[:120]
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"查询失败（重试 {self.max_retries} 次）: {last_err}")

    async def paginate(self, gql: str, path: list[str],
                       variables: Optional[dict] = None,
                       page_size: int = 100,
                       max_pages: int = 100) -> AsyncIterator[dict]:
        """自动翻页。

        path 是结果里 nodes 的路径，如 ["products", "nodes"]。
        用法：
            async for p in client.paginate(GQL, ["products"]):
                ...
        """
        variables = dict(variables or {})
        variables["first"] = page_size
        cursor = None
        for _ in range(max_pages):
            if cursor:
                variables["after"] = cursor
            data = await self.query(gql, variables)

            node = data
            for k in path:
                node = (node or {}).get(k, {})
            items = node.get("nodes", []) if isinstance(node, dict) else []

            for it in items:
                yield it

            page_info = (node or {}).get("pageInfo", {}) if isinstance(node, dict) else {}
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")


# ---------------------------------------------------------------------------
# 常用查询
# ---------------------------------------------------------------------------


PRODUCTS_QUERY = """
query Products($first: Int!, $after: String) {
  products(first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      title
      description
      productType
      vendor
      tags
      status
      createdAt
      updatedAt
      images(first: 20) {
        nodes { id url altText width height }
      }
      variants(first: 100) {
        nodes {
          id
          title
          sku
          price
          compareAtPrice
          inventoryQuantity
          selectedOptions { name value }
        }
      }
    }
  }
}
"""

ORDERS_QUERY = """
query Orders($first: Int!, $after: String, $query: String) {
  orders(first: $first, after: $after, query: $query) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      name
      createdAt
      displayFinancialStatus
      displayFulfillmentStatus
      totalPriceSet { shopMoney { amount currencyCode } }
      customer { id email }
      lineItems(first: 20) {
        nodes {
          title
          quantity
          sku
          originalUnitPriceSet { shopMoney { amount currencyCode } }
        }
      }
      fulfillments {
        trackingInfo { number company url }
        status
        createdAt
      }
    }
  }
}
"""

SHOP_QUERY = """
query { shop { name myshopifyDomain email currencyCode primaryDomain { url } plan { displayName } } }
"""


# ---------------------------------------------------------------------------
# 高层封装
# ---------------------------------------------------------------------------


class ShopAPI:
    """面向业务的高层封装。Day 40 的索引同步会用它。"""

    def __init__(self, client: ShopifyClient):
        self.client = client

    async def shop_info(self) -> dict:
        d = await self.client.query(SHOP_QUERY)
        return d.get("shop", {})

    async def all_products(self, max_pages: int = 100) -> list[dict]:
        out = []
        async for p in self.client.paginate(PRODUCTS_QUERY, ["products"],
                                            max_pages=max_pages):
            out.append(p)
        return out

    async def orders(self, query_filter: str = "", max_pages: int = 20) -> list[dict]:
        """query_filter 语法参考 Shopify 搜索语法，如 "fulfillment_status:unshipped" """
        out = []
        vars_ = {"query": query_filter} if query_filter else {}
        async for o in self.client.paginate(ORDERS_QUERY, ["orders"],
                                            variables=vars_, max_pages=max_pages):
            out.append(o)
        return out

    async def find_order_by_name(self, name: str) -> Optional[dict]:
        """按订单号找订单。客服场景高频操作。"""
        orders = await self.orders(f'name:"{name}"', max_pages=1)
        return orders[0] if orders else None

    async def register_webhooks(self, callback_base: str) -> list[dict]:
        """注册 webhook。Day 40 用。

        topics 里前三个是 GDPR 强制要求的。
        """
        topics = [
            "PRODUCTS_CREATE", "PRODUCTS_UPDATE", "PRODUCTS_DELETE",
            "ORDERS_CREATE", "ORDERS_UPDATED", "REFUNDS_CREATE",
            "APP_UNINSTALLED",
            # GDPR 三个强制
            "CUSTOMERS_DATA_REQUEST", "CUSTOMERS_REDACT", "SHOP_REDACT",
        ]
        mutation = """
        mutation WebhookCreate($topic: WebhookSubscriptionTopic!, $url: String!) {
          webhookSubscriptionCreate(topic: $topic, webhookSubscription: {callbackUrl: $url, format: JSON}) {
            webhookSubscription { id topic }
            userErrors { field message }
          }
        }
        """
        out = []
        for t in topics:
            url = f"{callback_base.rstrip('/')}/webhooks/{t.lower()}"
            try:
                d = await self.client.query(mutation, {"topic": t, "url": url})
                sub = d.get("webhookSubscriptionCreate", {})
                if sub.get("userErrors"):
                    print(f"  ✗ {t}: {sub['userErrors']}")
                else:
                    print(f"  ✓ {t}")
                    out.append(sub.get("webhookSubscription", {}))
            except Exception as e:      # noqa: BLE001
                print(f"  ✗ {t}: {str(e)[:100]}")
        return out


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def get_client(shop: str, store: Optional[TokenStore] = None) -> ShopifyClient:
    store = store or TokenStore()
    token = store.get(shop)
    if not token:
        raise PermissionError(f"店铺 {shop} 未授权，需要重新安装")
    return ShopifyClient(shop=shop, access_token=token)


# ---------------------------------------------------------------------------
# Tariff：把 Shopify 的 GraphQL 结构转成内部结构
# ---------------------------------------------------------------------------


def to_internal_product(sp: dict) -> dict:
    """把 Shopify 的 product 对象转成我们内部用的结构。

    内部结构要和 src/data/taxonomy.py 的图像类型对齐，
    这样 RAG 索引和训练数据就在同一个语义空间里。
    """
    images = [n for n in (sp.get("images", {}) or {}).get("nodes", []) if n.get("url")]
    variants = (sp.get("variants", {}) or {}).get("nodes", [])

    return {
        "id": (sp.get("id") or "").split("/")[-1],       # gid://shopify/Product/123 -> 123
        "title": sp.get("title", ""),
        "description": sp.get("description", ""),
        "category": sp.get("productType", ""),
        "vendor": sp.get("vendor", ""),
        "tags": sp.get("tags", []),
        "image_urls": [i["url"] for i in images],
        "variants": [{
            "id": (v.get("id") or "").split("/")[-1],
            "title": v.get("title", ""),
            "sku": v.get("sku", ""),
            "price": v.get("price"),
            "inventory": v.get("inventoryQuantity"),
            "options": {o["name"]: o["value"] for o in v.get("selectedOptions", [])},
        } for v in variants],
        "updated_at": sp.get("updatedAt"),
    }


if __name__ == "__main__":
    print("=" * 78)
    print("Shopify Admin API 客户端")
    print("=" * 78)
    print("""
这个模块需要真实的店铺和 token 才能跑。先按下面的步骤准备（Day 37）：

1. 注册 Partner 账号: https://partners.shopify.com
2. 创建一个 Development Store（选「测试数据」，无限期免费）
3. 往店里塞 10-20 个商品，每个商品多张图  ← 这是你后训练的数据源
4. 创建一个 App（Public App），拿到 API key / secret
5. 把 .env 填好：
     SHOPIFY_API_KEY=xxx
     SHOPIFY_API_SECRET=xxx
     SHOPIFY_APP_URL=https://your-tunnel.ngrok.io

然后：
    python -m src.shopify.auth          # 先验证 HMAC 逻辑（不需要真店铺）
    python -m src.shopify.client        # 本文件

准备就绪后：
    from src.shopify.client import ShopAPI, get_client, to_internal_product
    api = ShopAPI(get_client("your-store.myshopify.com"))
    products = await api.all_products()
    internal = [to_internal_product(p) for p in products]

关键设计点（Day 37 的验收）：
  ✓ 用 GraphQL 不用 REST（限流按复杂度算，更高效）
  ✓ 自动翻页（paginate 是 async generator）
  ✓ 限流处理：读 throttleStatus.currentlyAvailable，低于阈值主动 sleep
     —— 不要用固定 sleep 硬编码
  ✓ 401 单独处理（token 失效 → 提示重新安装）
""")
    print("要点：")
    print("  - GraphQL 默认最多返回 50/250 条，必须用 cursor 翻页")
    print("  - 限流不是「每秒 N 请求」，而是「计算成本」")
    print("  - extensions.cost.throttleStatus 会告诉你剩余配额和恢复速率")
    print("  - 用配额而不是固定 sleep，效率高得多")
