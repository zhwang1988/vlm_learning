"""
shopify —— Shopify SaaS 产品化。

对应讲义：docs/11-shopify.md
对应计划：Week 7（Day 37–Day 45）

⚠️ Shopify 平台细节更新较快，字段名和审核条款请以 https://shopify.dev 官方文档为准。
   本模块实现的是**稳定的流程和易错点**。

文件导航：
  auth.py      Day 38  OAuth + HMAC 校验 + Session Token 验证 + Token 存储
  client.py    Day 37  Admin GraphQL 客户端（分页/限流退避/重试）
  webhooks.py  Day 40  Webhook 接收（HMAC + 幂等 + 队列）
  indexer.py   Day 40  商品向量索引同步（webhook 增量更新）
  billing.py   Day 41  订阅 + 按用量计费（幂等上报）
  models.py    Day 38  数据库 schema（PostgreSQL + pgvector）
  app.py       Day 38/42  FastAPI 主应用（全部路由）
  extensions/chat-widget/  Day 39  Theme App Extension（前台挂件）

一键流程：
    # Day 37 准备：注册 Partner → 创建 Development Store → 塞商品 → 建 App
    #          把 .env 填好

    # Day 38 验证认证逻辑（不需要真店铺）
    python -m src.shopify.auth

    # Day 38 启动服务
    python -m src.shopify.app --dev
    # 另开终端做 HTTPS 隧道（Shopify 要求 HTTPS）
    ngrok http 8080

    # Day 38 安装到测试店
    # 浏览器打开: http://localhost:8080/auth?shop=your-store.myshopify.com

    # Day 40 验证 webhook 逻辑
    python -m src.shopify.webhooks

    # Day 41 验证计费逻辑
    python -m src.shopify.billing

    # Day 45 审核清单
    # 见 docs/11-shopify.md 的「上线检查清单」

五个最容易踩的坑（都在代码注释里标了）：
  ① OAuth HMAC 校验：必须先剔除 hmac 参数，再按 key 排序拼接
     —— 直接拿原始 query string 算永远失败
  ② Webhook HMAC 是对 bytes 做、结果是 base64，和 OAuth 的 hex 不同
  ③ 所有 shop 参数都要过 validate_shop（防 SSRF，`xxx.myshopify.com.evil.com`）
  ④ Webhook 必须 5 秒内返回 200 → 入队后立刻返回
  ⑤ 计费上报必须幂等 → 用会话 ID 作 key，DB 加 UNIQUE 约束兜底
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.shopify import CFG      # 触发时才 import 对应模块
#     python -m src.shopify.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "CFG": ".auth",
    "HANDLERS": ".webhooks",
    "IDEMPOTENCY": ".webhooks",
    "LEDGER": ".billing",
    "ORDERS_QUERY": ".client",
    "PLANS": ".billing",
    "PRODUCTS_QUERY": ".client",
    "QUEUE": ".webhooks",
    "ShopAPI": ".client",
    "ShopifyClient": ".client",
    "TokenStore": ".auth",
    "build_install_url": ".auth",
    "check_entitlement": ".billing",
    "create_subscription": ".billing",
    "exchange_code_for_token": ".auth",
    "get_client": ".client",
    "normalize_shop": ".auth",
    "process_webhook": ".webhooks",
    "report_usage": ".billing",
    "to_internal_product": ".client",
    "unit_economics": ".billing",
    "validate_shop": ".auth",
    "verify_hmac": ".auth",
    "verify_session_token": ".auth",
    "verify_subscription": ".billing",
    "verify_webhook_hmac": ".auth",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value        # 缓存，下次不再走这里
    return value


def __dir__():
    return sorted(set(__all__) | set(_LAZY))


__all__ = [
    "CFG", "TokenStore", "build_install_url", "exchange_code_for_token",
    "normalize_shop", "validate_shop", "verify_hmac", "verify_session_token",
    "verify_webhook_hmac",
    "LEDGER", "PLANS", "check_entitlement", "create_subscription",
    "report_usage", "unit_economics", "verify_subscription",
    "ORDERS_QUERY", "PRODUCTS_QUERY", "ShopAPI", "ShopifyClient",
    "get_client", "to_internal_product",
    "HANDLERS", "IDEMPOTENCY", "QUEUE", "process_webhook",
]
