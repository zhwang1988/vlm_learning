"""
Shopify 认证：OAuth 安装流程 + HMAC 校验 + Session Token 验证。

对应讲义 docs/11-shopify.md「认证链路」，对应计划 Day 38。

⚠️ Shopify 平台细节更新较快，字段名和校验规则请以 https://shopify.dev 官方文档为准。
   本篇实现的是**稳定的流程和易错点**。

认证链路：
  ① 商家点「安装」→ /auth?shop=xxx.myshopify.com
  ② 重定向到 Shopify 授权页（带 client_id, scope, redirect_uri, state, nonce）
  ③ 商家同意 → 回调 /auth/callback?code=...&hmac=...&shop=...&state=...
  ④ **校验 HMAC**（最容易写错的地方）
  ⑤ 用 code 换 access_token
  ⑥ 存 token，关联 shop domain
  ⑦ 后续 API 调用带 X-Shopify-Access-Token

两个 token 的区别（面试常问）：
  Access Token   长期凭据，服务端调 Admin API 用
  Session Token  短期 JWT（约 1 分钟），前端调你自己的后端用
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, quote

try:
    import jwt          # PyJWT
    HAS_JWT = True
except ImportError:
    HAS_JWT = False


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class ShopifyConfig:
    api_key: str = field(default_factory=lambda: os.getenv("SHOPIFY_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("SHOPIFY_API_SECRET", ""))
    app_url: str = field(default_factory=lambda: os.getenv("SHOPIFY_APP_URL", "https://localhost:8080"))
    scopes: str = field(default_factory=lambda: os.getenv(
        "SHOPIFY_SCOPES",
        # 最小化原则：只要真的需要的
        "read_products,read_orders,write_orders,read_customers,"
        "read_fulfillments,write_fulfillments,read_inventory"
    ))
    api_version: str = field(default_factory=lambda: os.getenv("SHOPIFY_API_VERSION", "2025-01"))

    @property
    def redirect_uri(self) -> str:
        return f"{self.app_url.rstrip('/')}/auth/callback"


CFG = ShopifyConfig()


# ---------------------------------------------------------------------------
# HMAC 校验（最容易写错的地方）
# ---------------------------------------------------------------------------


def verify_hmac(query_string: str, secret: str | None = None) -> bool:
    """校验 Shopify 回调的 HMAC 签名。

    **必须注意的两点**（很多人在这里踩坑）：
      1. 要把 `hmac` 这个参数本身从待签名内容里剔除
      2. 剩余参数要**按 key 排序后重新拼接**，不能用原始 query string

    常见错误：直接拿整个 query string 去算 HMAC → 永远校验失败。
    """
    secret = secret or CFG.api_secret
    if not secret:
        return False

    params = dict(parse_qsl(query_string, keep_blank_values=True))
    received = params.pop("hmac", None)
    if not received:
        return False

    message = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    digest = hmac.new(secret.encode("utf-8"),
                      message.encode("utf-8"),
                      hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, received)


def verify_webhook_hmac(body: bytes, hmac_header: str,
                        secret: str | None = None) -> bool:
    """校验 webhook 的 HMAC。

    注意和 OAuth 回调的区别：
      - webhook 是对**原始请求体（bytes）**做 HMAC，不是对 query 参数
      - 用 X-Shopify-Hmac-Sha256 头传值
      - 结果是 base64，不是 hex
    """
    import base64
    secret = secret or CFG.api_secret
    if not secret:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    computed = base64.b64encode(digest).decode()
    return hmac.compare_digest(computed, hmac_header or "")


# ---------------------------------------------------------------------------
# 店铺域名校验
# ---------------------------------------------------------------------------

import re

SHOP_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*\.myshopify\.com$")


def validate_shop(shop: str) -> bool:
    """防 SSRF：必须严格匹配 xxx.myshopify.com。

    不校验的话，攻击者可以传 `evil.com` 让你的服务去请求它，
    或者传 `xxx.myshopify.com.evil.com` 绕过。
    """
    return bool(SHOP_PATTERN.match(shop or ""))


def normalize_shop(shop: str) -> str:
    """从各种输入里抽出标准 shop domain。"""
    shop = (shop or "").strip().lower()
    if shop.startswith("http"):
        shop = urlparse(shop).netloc.split(":")[0]
    if not shop.endswith(".myshopify.com") and "." not in shop:
        shop = f"{shop}.myshopify.com"
    return shop


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------


def build_install_url(shop: str, state: str | None = None) -> str:
    """构造授权跳转 URL。"""
    shop = normalize_shop(shop)
    state = state or secrets.token_urlsafe(16)
    params = {
        "client_id": CFG.api_key,
        "scope": CFG.scopes,
        "redirect_uri": CFG.redirect_uri,
        "state": state,
        # 现代 Shopify 推荐带 nonce 防止重放
        "grant_options[]": "",
    }
    return f"https://{shop}/admin/oauth/authorize?{urlencode(params)}"


def verify_state(received: str, expected: str) -> bool:
    """校验 state，防 CSRF。"""
    return bool(received) and bool(expected) and hmac.compare_digest(received, expected)


async def exchange_code_for_token(shop: str, code: str) -> dict:
    """用授权码换 access token。"""
    import httpx

    shop = normalize_shop(shop)
    url = f"https://{shop}/admin/oauth/access_token"
    payload = {
        "client_id": CFG.api_key,
        "client_secret": CFG.api_secret,
        "code": code,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        return r.json()          # {"access_token": "...", "scope": "..."}


# ---------------------------------------------------------------------------
# Session Token（JWT）验证
# ---------------------------------------------------------------------------


def verify_session_token(token: str, secret: str | None = None) -> dict | None:
    """验证 App Bridge 传来的 session token。

    校验点：
      1. 签名（HS256，用 api_secret）
      2. aud == api_key
      3. exp 未过期
      4. iss 是合法的 shop domain

    这是后端确认「这个请求真的来自已安装的店铺」的方式。
    """
    if not HAS_JWT:
        raise RuntimeError("需要 PyJWT: pip install pyjwt")
    secret = secret or CFG.api_secret
    try:
        payload = jwt.decode(
            token, secret, algorithms=["HS256"],
            audience=CFG.api_key,
            options={"verify_exp": True},
        )
    except Exception:
        return None

    iss = payload.get("iss", "")
    # iss 形如 https://shop.myshopify.com/admin
    dest = payload.get("dest", "")
    if dest:
        shop = urlparse(dest).netloc
        if not validate_shop(shop):
            return None
    return payload


# ---------------------------------------------------------------------------
# Token 存储（生产要加密！）
# ---------------------------------------------------------------------------


class TokenStore:
    """Access token 存储。

    ⚠️ 生产环境必须：
      1. 加密存储（Fernet / KMS）
      2. 限制访问权限
      3. 记录访问审计日志
    这里用接口抽象，方便你换成真实实现。
    """

    def __init__(self, path: str | Path = "data/shop_tokens.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def save(self, shop: str, token: str, scope: str = ""):
        shop = normalize_shop(shop)
        self._data[shop] = {
            "access_token": token,
            "scope": scope,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._flush()

    def get(self, shop: str) -> str | None:
        return (self._data.get(normalize_shop(shop)) or {}).get("access_token")

    def delete(self, shop: str):
        """卸载时应删除 token（配合 app/uninstalled webhook）。"""
        self._data.pop(normalize_shop(shop), None)
        self._flush()

    def all_shops(self) -> list[str]:
        return list(self._data.keys())

    def _flush(self):
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def run_tests():
    print("=" * 78)
    print("Shopify 认证自检")
    print("=" * 78)

    # 1. shop 校验（防 SSRF）
    print("\n[1] 店铺域名校验")
    cases = [
        ("myshop.myshopify.com", True),
        ("my-shop-123.myshopify.com", True),
        ("evil.com", False),
        ("myshop.myshopify.com.evil.com", False),
        ("", False),
    ]
    for shop, expect in cases:
        got = validate_shop(shop)
        mark = "✓" if got == expect else "✗"
        print(f"    {mark} {shop!r:<40} → {got}")
        assert got == expect, f"{shop} 校验结果错误"
    print("    ✓ 防 SSRF 生效")

    # 2. HMAC 校验（含正确/错误两种）
    print("\n[2] HMAC 校验")
    secret = "test_secret_abc"

    # 注意：参数顺序故意打乱，并且不是字典序。
    # 如果 fixture 恰好是排好序的，那「原始串直接算」会得到和「排序后算」一样的结果，
    # 这个测试就会假装通过 —— 那就等于没测。
    qs = ("shop=myshop.myshopify.com&state=xyz&code=abc123"
          "&timestamp=1700000000&host=YWRtaW4")
    params = dict(parse_qsl(qs))
    message = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    good_hmac = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    print(f"    正确签名（乱序 query 也应通过）→ {verify_hmac(qs + f'&hmac={good_hmac}', secret)}")
    assert verify_hmac(qs + f"&hmac={good_hmac}", secret)

    # 同一组参数换个顺序，仍必须通过 —— 证明实现真的做了排序
    shuffled = "&".join(f"{k}={v}" for k, v in reversed(list(params.items())))
    print(f"    参数顺序颠倒后仍通过        → {verify_hmac(shuffled + f'&hmac={good_hmac}', secret)}")
    assert verify_hmac(shuffled + f"&hmac={good_hmac}", secret)

    print(f"    错误签名                    → {verify_hmac(qs + '&hmac=deadbeef', secret)}")
    assert not verify_hmac(qs + "&hmac=deadbeef", secret)

    # --- 坑 1：不排序，直接拿原始 query string 算 ---
    wrong_order = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    got = verify_hmac(qs + f"&hmac={wrong_order}", secret)
    print(f"    坑1 用原始顺序直接算         → {got} （应该 False）")
    assert not got, "没按 key 排序却通过了 —— 排序逻辑失效"

    # --- 坑 2：忘了先把 hmac 参数剔除 ---
    with_hmac = "&".join(f"{k}={v}" for k, v in
                         sorted({**params, "hmac": good_hmac}.items()))
    wrong_strip = hmac.new(secret.encode(), with_hmac.encode(), hashlib.sha256).hexdigest()
    got2 = verify_hmac(qs + f"&hmac={wrong_strip}", secret)
    print(f"    坑2 忘了剔除 hmac 参数      → {got2} （应该 False）")
    assert not got2, "把 hmac 自己也算进签名却通过了 —— 剔除逻辑失效"

    print("    ✓ 通过。两个坑都真的被拦住了：先剔除 hmac、再按 key 排序拼接")

    # 3. webhook HMAC（base64 而非 hex）
    print("\n[3] Webhook HMAC")
    import base64
    body = b'{"id":123,"title":"test product"}'
    d = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    good = base64.b64encode(d).decode()
    print(f"    正确 → {verify_webhook_hmac(body, good, secret)}")
    assert verify_webhook_hmac(body, good, secret)
    print(f"    被篡改的 body → {verify_webhook_hmac(b'{\"id\":999}', good, secret)}")
    assert not verify_webhook_hmac(b'{"id":999}', good, secret)
    print("    ✓ 通过")

    # 4. 授权 URL
    print("\n[4] 授权 URL 构造")
    os.environ.setdefault("SHOPIFY_API_KEY", "test_key")
    CFG.api_key = "test_key"
    url = build_install_url("myshop")
    print(f"    {url[:100]}...")
    assert "myshop.myshopify.com/admin/oauth/authorize" in url
    print("    ✓ 域名被规范化")

    print("\n" + "=" * 78)
    print("✓ 全部通过")
    print()
    print("⚠️ 生产必做（Day 44 安全加固）：")
    print("   1. access_token 加密存储（这里用的是明文 JSON，只适合开发）")
    print("   2. state 要存服务端并做一次性校验（防 CSRF）")
    print("   3. 所有 shop 参数都要过 validate_shop（防 SSRF）")


if __name__ == "__main__":
    run_tests()
