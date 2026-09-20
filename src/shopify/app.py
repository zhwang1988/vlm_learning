"""
Shopify App 主应用（FastAPI）。

对应计划 Day 38（骨架）→ Day 42（部署）。

路由总览：
  GET  /                     → 后台界面入口（App Bridge）
  GET  /auth?shop=...        → OAuth 安装开始
  GET  /auth/callback        → OAuth 回调（校验 HMAC + 换 token）
  GET  /api/me               → 当前店铺信息（session token 校验）
  GET  /api/onboard          → 安装后初始化（拉商品 + 建索引）
  POST /webhooks/{topic}     → webhook 接收
  GET  /widget.js            → 前台挂件脚本（给 Theme Extension 用）
  POST /widget/chat          → 前台聊天接口
  GET  /billing/plans        → 套餐列表
  POST /billing/subscribe    → 创建订阅
  GET  /health               → 健康检查

启动：
    python -m src.shopify.app
    # 或
    uvicorn src.shopify.app:create_app --factory --port 8080
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import webhooks as wh
from .auth import (
    CFG,
    TokenStore,
    build_install_url,
    exchange_code_for_token,
    normalize_shop,
    validate_shop,
    verify_hmac,
    verify_session_token,
)

TOKENS = TokenStore()
_STATES: dict[str, float] = {}       # state -> 创建时间（生产用 Redis）

# 允许前台挂件跨域调用
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "https://cdn.shopify.com,https://extensions.shopifycdn.com"
).split(",")


def create_app() -> FastAPI:
    app = FastAPI(title="AI Customer Service for Shopify", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if os.getenv("DEV") == "1" else ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =================================================================== #
    # 基础
    # =================================================================== #

    @app.get("/health")
    async def health():
        return {"status": "ok", "ts": time.time(),
                "shops": len(TOKENS.all_shops())}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return ADMIN_HTML

    # =================================================================== #
    # OAuth（Day 38）
    # =================================================================== #

    @app.get("/auth")
    async def auth(shop: str = Query(...)):
        """安装入口。"""
        if not validate_shop(shop):
            raise HTTPException(400, "非法的 shop 参数")

        state = secrets.token_urlsafe(24)
        _STATES[state] = time.time()          # 生产用 Redis + TTL
        url = build_install_url(shop, state)
        return JSONResponse({"install_url": url, "state": state})

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        """OAuth 回调。三步：校验 HMAC → 校验 state → 换 token。"""
        qs = str(request.url.query)

        # ① HMAC —— 必须第一步做
        if not verify_hmac(qs):
            raise HTTPException(401, "HMAC 校验失败")

        params = dict(request.query_params)
        shop = normalize_shop(params.get("shop", ""))
        if not validate_shop(shop):
            raise HTTPException(400, "非法的 shop 参数")

        # ② state
        state = params.get("state", "")
        created = _STATES.pop(state, None)
        if created is None or time.time() - created > 600:
            raise HTTPException(400, "state 校验失败（过期或不存在）")

        # ③ 换 token
        code = params.get("code", "")
        if not code:
            raise HTTPException(400, "缺少 code")

        data = await exchange_code_for_token(shop, code)
        token = data.get("access_token")
        if not token:
            raise HTTPException(500, "换取 access token 失败")

        TOKENS.save(shop, token, data.get("scope", ""))
        print(f"[auth] ✓ {shop} 安装成功，scope={data.get('scope', '')[:80]}")

        # 重定向到后台（走 App Bridge）
        return HTMLResponse(f"""
        <!doctype html><html><head><meta charset="utf-8">
        <title>安装成功</title></head><body>
        <script src="https://cdn.shopify.com/shopifycloud/app-bridge.js"
          data-api-key="{CFG.api_key}"></script>
        <p>安装成功，正在跳转…</p>
        <script>
          const host = new URLSearchParams(location.search).get('host');
          if (window.shopify && host) {{
            const url = new URL(location.origin + '/');
            url.searchParams.set('host', host);
            url.searchParams.set('shop', '{shop}');
            open(url.toString(), '_top');
          }} else {{
            location.href = '/?shop={shop}';
          }}
        </script>
        </body></html>
        """)

    # =================================================================== #
    # 店铺 API（Day 38）
    # =================================================================== #

    def _require_shop(authorization: Optional[str], shop_query: Optional[str]) -> str:
        """校验请求来源。

        两种方式：
          - App Bridge 的 session token（Authorization: Bearer <jwt>）—— 后台界面用
          - shop 参数 + 已安装检查 —— 前台挂件用（前台没有 session token）
        """
        if authorization and authorization.startswith("Bearer "):
            payload = verify_session_token(authorization[7:])
            if not payload:
                raise HTTPException(401, "无效的 session token")
            dest = payload.get("dest", "")
            shop = normalize_shop(dest.split("//")[-1].split("/")[0]) if dest else ""
            if not validate_shop(shop):
                raise HTTPException(401, "session token 中的店铺非法")
            return shop

        if shop_query and validate_shop(shop_query):
            if not TOKENS.get(shop_query):
                raise HTTPException(403, "该店铺未安装本应用")
            return normalize_shop(shop_query)

        raise HTTPException(401, "缺少身份凭据")

    @app.get("/api/me")
    async def api_me(shop: Optional[str] = None,
                     authorization: Optional[str] = Header(None)):
        shop = _require_shop(authorization, shop)
        from .billing import check_entitlement, LEDGER
        from .indexer import index_stats

        st = LEDGER.get_sub(shop)
        return {
            "shop": shop,
            "installed": bool(TOKENS.get(shop)),
            "plan": st.plan,
            "entitlement": check_entitlement(shop),
            "index": index_stats(shop),
        }

    @app.post("/api/onboard")
    async def onboard(shop: str = Query(...),
                      authorization: Optional[str] = Header(None)):
        """安装后初始化：拉商品 + 建索引 + 注册 webhook。

        这一步可能耗时较长（几百个商品要下载图片 + 编码），
        生产环境应该放后台任务，前端轮询进度。
        """
        shop = _require_shop(authorization, shop)
        from .client import ShopAPI, get_client
        from .indexer import rebuild_index

        client = get_client(shop)
        api = ShopAPI(client)

        # 注册 webhook
        base = CFG.app_url.rstrip("/")
        try:
            await api.register_webhooks(base)
        except Exception as e:      # noqa: BLE001
            print(f"[onboard] webhook 注册部分失败: {str(e)[:120]}")

        # 建索引
        stats = await rebuild_index(shop, api)
        return {"ok": True, "index": stats}

    # =================================================================== #
    # Webhook（Day 40）
    # =================================================================== #

    @app.post("/webhooks/{topic:path}")
    async def webhook(topic: str, request: Request):
        """Webhook 接收。

        顺序不能变：① 校验 HMAC ② 幂等 ③ 入队后立刻返回 200
        Shopify 要求 5 秒内响应，超时会重试。
        """
        body = await request.body()
        hmac_header = request.headers.get("X-Shopify-Hmac-Sha256", "")
        webhook_id = request.headers.get("X-Shopify-Webhook-Id", "")
        shop = normalize_shop(request.headers.get("X-Shopify-Shop-Domain", ""))

        if not validate_shop(shop):
            raise HTTPException(400, "非法的 shop header")

        result = await wh.process_webhook(topic, shop, body, hmac_header, webhook_id)

        if result.status != 200:
            raise HTTPException(result.status, result.message)

        # ⚠️ 必须 200 且快速返回
        return PlainTextResponse("ok", status_code=200)

    # =================================================================== #
    # 前台挂件（Day 39）
    # =================================================================== #

    @app.get("/widget.js")
    async def widget_js(shop: str = Query(...)):
        """前台挂件脚本。Theme App Extension 通过 <script src> 引入这个。"""
        if not validate_shop(shop):
            raise HTTPException(400, "非法的 shop")
        script = WIDGET_JS.replace("__SHOP__", shop).replace(
            "__API__", CFG.app_url.rstrip("/"))
        return PlainTextResponse(script, media_type="application/javascript")

    @app.post("/widget/chat")
    async def widget_chat(request: Request):
        """前台聊天接口。走 Agent（若已启用）或直连模型。"""
        payload = await request.json()
        shop = normalize_shop(payload.get("shop", ""))
        if not validate_shop(shop):
            raise HTTPException(400, "非法的 shop")
        if not TOKENS.get(shop):
            raise HTTPException(403, "该店铺未安装本应用")

        # 权益检查
        from .billing import check_entitlement, report_usage
        ent = check_entitlement(shop)
        if not ent["allowed"]:
            return {"reply": f"抱歉，{ent['reason']}",
                    "blocked": True, "reason": ent["reason"]}

        message = (payload.get("message") or "").strip()
        images = payload.get("images") or []
        session_id = payload.get("session_id") or f"w_{secrets.token_hex(8)}"

        # 优先走 Agent
        try:
            from ..agent.agent import get_default_agent
            agent = get_default_agent()
            if agent is not None:
                out = await agent.arun(message, images, session_id=session_id)
                reply = out["reply"]
            else:
                reply = await _direct_model(message, images)
        except Exception as e:      # noqa: BLE001
            print(f"[widget] Agent 失败，降级: {str(e)[:150]}")
            reply = ("抱歉，我这边暂时遇到一点问题，"
                     "已经记录了您的问题，稍后会有客服专员跟进。")

        # 计费上报（幂等）
        try:
            await report_usage(_client_for(shop), 0.15,
                               description="AI 客服会话",
                               idempotency_key=session_id)
        except Exception:
            pass

        return {"reply": reply, "session_id": session_id}

    # =================================================================== #
    # 计费（Day 41）
    # =================================================================== #

    @app.get("/billing/plans")
    async def billing_plans():
        from .billing import PLANS
        return {"plans": PLANS}

    @app.post("/billing/subscribe")
    async def billing_subscribe(request: Request,
                                authorization: Optional[str] = Header(None)):
        shop_param = request.query_params.get("shop")
        shop = _require_shop(authorization, shop_param)
        payload = await request.json()
        plan = payload.get("plan", "standard")

        from .billing import create_subscription
        res = await create_subscription(_client_for(shop), plan)
        return res

    @app.post("/webhooks/billing/confirm")
    async def billing_confirm(shop: str = Query(...),
                              authorization: Optional[str] = Header(None)):
        """订阅回调后**必须**主动校验，不能只信回调参数。"""
        shop = _require_shop(authorization, shop)
        from .billing import verify_subscription
        st = await verify_subscription(_client_for(shop))
        return {"plan": st.plan, "status": st.status}

    return app


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _client_for(shop: str):
    from .client import get_client
    return get_client(shop)


async def _direct_model(message: str, images: list) -> str:
    """没有 Agent 时的降级路径：直连模型。"""
    import httpx
    base = os.getenv("VLLM_BASE", "http://localhost:8000/v1")
    content = [{"type": "image_url", "image_url": {"url": u}} for u in images[:3]]
    content.append({"type": "text", "text": message})
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{base}/chat/completions", json={
            "model": os.getenv("SERVED_MODEL", "cx-vlm"),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 400, "temperature": 0.3,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 内嵌前端资源
# ---------------------------------------------------------------------------

ADMIN_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 客服</title>
<script src="https://cdn.shopify.com/shopifycloud/app-bridge.js"
        data-api-key="__API_KEY__"></script>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      margin:0;padding:24px;background:#f6f6f7;color:#1a1a1a}
 .card{background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;
       box-shadow:0 1px 3px rgba(0,0,0,.08)}
 h1{font-size:20px;margin:0 0 16px}
 .row{display:flex;justify-content:space-between;padding:8px 0;
      border-bottom:1px solid #eee}
 .row:last-child{border:none}
 .k{color:#6b7177}.v{font-weight:600}
 .ok{color:#008060}.warn{color:#b98900}.bad{color:#d72c0d}
 button{background:#008060;color:#fff;border:none;padding:10px 18px;
        border-radius:8px;cursor:pointer;font-size:14px}
 button:hover{background:#006e52}
 button:disabled{background:#c9cccf;cursor:not-allowed}
 pre{background:#f6f6f7;padding:12px;border-radius:8px;overflow:auto;
     font-size:12px;max-height:300px}
</style></head>
<body>
<h1>AI 客服 · 控制台</h1>

<div class="card">
  <div class="row"><span class="k">店铺</span><span class="v" id="shop">—</span></div>
  <div class="row"><span class="k">当前套餐</span><span class="v" id="plan">—</span></div>
  <div class="row"><span class="k">本月已用会话</span><span class="v" id="used">—</span></div>
  <div class="row"><span class="k">商品索引</span><span class="v" id="index">—</span></div>
</div>

<div class="card">
  <h1 style="font-size:15px">初始化</h1>
  <p style="color:#6b7177;font-size:13px">
    拉取店铺商品、建立图片检索索引、注册 webhook。首次安装后必须执行一次。
  </p>
  <button id="btn-onboard" onclick="onboard()">开始初始化</button>
  <pre id="log" style="display:none"></pre>
</div>

<script>
const qs = new URLSearchParams(location.search);
const shop = qs.get('shop') || '';
document.getElementById('shop').textContent = shop || '未知';

async function authHeaders() {
  try {
    const token = await window.shopify.idToken();
    return {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'};
  } catch(e) { return {'Content-Type': 'application/json'}; }
}

async function loadMe() {
  try {
    const r = await fetch('/api/me?shop=' + encodeURIComponent(shop),
                          {headers: await authHeaders()});
    const d = await r.json();
    document.getElementById('plan').textContent = d.plan || '—';
    const e = d.entitlement || {};
    document.getElementById('used').textContent =
      (e.used ?? '—') + ' / ' + (e.included ?? '—');
    const ix = d.index || {};
    document.getElementById('index').textContent =
      (ix.n_products || 0) + ' 商品 / ' + (ix.n_vectors || 0) + ' 向量';
  } catch(e) { console.error(e); }
}

async function onboard() {
  const btn = document.getElementById('btn-onboard');
  const log = document.getElementById('log');
  btn.disabled = true; btn.textContent = '初始化中…（可能需要几分钟）';
  log.style.display = 'block'; log.textContent = '开始…';
  try {
    const r = await fetch('/api/onboard?shop=' + encodeURIComponent(shop),
                          {method:'POST', headers: await authHeaders()});
    const d = await r.json();
    log.textContent = JSON.stringify(d, null, 2);
    btn.textContent = '重新初始化';
    loadMe();
  } catch(e) {
    log.textContent = '失败: ' + e;
    btn.textContent = '重试';
  }
  btn.disabled = false;
}

loadMe();
</script>
</body></html>
""".replace("__API_KEY__", os.getenv("SHOPIFY_API_KEY", ""))


WIDGET_JS = r"""
/* AI 客服前台挂件。由 Theme App Extension 通过 <script src> 引入。
   嵌入方式见 extensions/chat-widget/blocks/chat_widget.liquid */
(function () {
  const SHOP = "__SHOP__";
  const API = "__API__";
  if (window.__cxWidgetLoaded) return;
  window.__cxWidgetLoaded = true;

  const SID_KEY = 'cx_session_id';
  let sessionId = localStorage.getItem(SID_KEY);
  if (!sessionId) {
    sessionId = 'w_' + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(SID_KEY, sessionId);
  }

  const css = `
  #cx-widget-root{position:fixed;right:20px;bottom:20px;z-index:2147483000;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #cx-launcher{width:56px;height:56px;border-radius:50%;background:#1a1a1a;
    color:#fff;border:none;cursor:pointer;font-size:24px;
    box-shadow:0 4px 16px rgba(0,0,0,.2)}
  #cx-panel{display:none;position:absolute;right:0;bottom:70px;width:360px;
    height:520px;background:#fff;border-radius:16px;overflow:hidden;
    box-shadow:0 8px 40px rgba(0,0,0,.18);display:none;flex-direction:column}
  #cx-panel.open{display:flex}
  #cx-head{padding:14px 16px;background:#1a1a1a;color:#fff;font-size:14px;font-weight:600}
  #cx-head small{display:block;font-weight:400;opacity:.7;font-size:11px;margin-top:2px}
  #cx-body{flex:1;overflow-y:auto;padding:12px;background:#f7f7f8}
  .cx-msg{margin-bottom:10px;display:flex}
  .cx-msg.u{justify-content:flex-end}
  .cx-bub{max-width:80%;padding:9px 13px;border-radius:14px;font-size:13px;
    line-height:1.55;word-break:break-word;white-space:pre-wrap}
  .cx-msg.a .cx-bub{background:#fff;border:1px solid #eee}
  .cx-msg.u .cx-bub{background:#1a1a1a;color:#fff}
  .cx-msg img{max-width:120px;border-radius:8px;margin-bottom:4px;display:block}
  #cx-foot{padding:10px;border-top:1px solid #eee;background:#fff}
  #cx-preview{display:flex;gap:6px;padding-bottom:6px;flex-wrap:wrap}
  #cx-preview img{width:44px;height:44px;object-fit:cover;border-radius:6px}
  #cx-row{display:flex;gap:6px;align-items:center}
  #cx-input{flex:1;border:1px solid #ddd;border-radius:20px;padding:9px 14px;
    font-size:13px;outline:none;font-family:inherit}
  #cx-input:focus{border-color:#1a1a1a}
  .cx-ic{width:34px;height:34px;border-radius:50%;border:1px solid #ddd;
    background:#fff;cursor:pointer;font-size:15px;line-height:1;flex:none}
  #cx-send{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
  #cx-send:disabled{background:#ccc;border-color:#ccc}
  .cx-hint{font-size:10px;color:#999;text-align:center;padding-top:6px}
  @media(max-width:480px){#cx-panel{width:calc(100vw - 24px);height:70vh}}
  `;
  const st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

  const root = document.createElement('div');
  root.id = 'cx-widget-root';
  root.innerHTML = `
    <div id="cx-panel">
      <div id="cx-head">在线客服<small>AI 助手 · 可发图片</small></div>
      <div id="cx-body"></div>
      <div id="cx-foot">
        <div id="cx-preview"></div>
        <div id="cx-row">
          <input type="file" id="cx-file" accept="image/*" multiple hidden>
          <button class="cx-ic" id="cx-img" title="发送图片">📎</button>
          <input id="cx-input" placeholder="描述您的问题…">
          <button class="cx-ic" id="cx-send" title="发送">➤</button>
        </div>
        <div class="cx-hint">AI 回复可能不准确，重要事项请联系人工客服</div>
      </div>
    </div>
    <button id="cx-launcher" aria-label="打开客服">💬</button>
  `;
  document.body.appendChild(root);

  const panel = root.querySelector('#cx-panel');
  const body = root.querySelector('#cx-body');
  const input = root.querySelector('#cx-input');
  const sendBtn = root.querySelector('#cx-send');
  const fileInput = root.querySelector('#cx-file');
  const preview = root.querySelector('#cx-preview');
  let pending = [];   // 待发送的 dataURL

  root.querySelector('#cx-launcher').onclick = () => panel.classList.toggle('open');

  function addMsg(role, text, imgs) {
    const d = document.createElement('div');
    d.className = 'cx-msg ' + (role === 'user' ? 'u' : 'a');
    const b = document.createElement('div');
    b.className = 'cx-bub';
    (imgs || []).forEach(u => {
      const im = document.createElement('img'); im.src = u; b.appendChild(im);
    });
    b.appendChild(document.createTextNode(text));
    d.appendChild(b); body.appendChild(d);
    body.scrollTop = body.scrollHeight;
    return b;
  }

  function renderPreview() {
    preview.innerHTML = '';
    pending.forEach((u, i) => {
      const w = document.createElement('div');
      w.style.position = 'relative';
      const im = document.createElement('img'); im.src = u;
      const x = document.createElement('span');
      x.textContent = '×';
      x.style.cssText = 'position:absolute;top:-4px;right:-4px;background:#000;' +
        'color:#fff;border-radius:50%;width:16px;height:16px;font-size:11px;' +
        'text-align:center;line-height:16px;cursor:pointer';
      x.onclick = () => { pending.splice(i, 1); renderPreview(); };
      w.appendChild(im); w.appendChild(x); preview.appendChild(w);
    });
  }

  root.querySelector('#cx-img').onclick = () => fileInput.click();

  fileInput.onchange = () => {
    Array.from(fileInput.files).slice(0, 3).forEach(f => {
      if (f.size > 4 * 1024 * 1024) { alert('图片请小于 4MB'); return; }
      const r = new FileReader();
      r.onload = e => { pending.push(e.target.result); renderPreview(); };
      r.readAsDataURL(f);
    });
    fileInput.value = '';
  };

  async function send() {
    const text = input.value.trim();
    if (!text && pending.length === 0) return;

    addMsg('user', text, pending.slice());
    const imgs = pending.slice();
    input.value = ''; pending = []; renderPreview();
    sendBtn.disabled = true;

    const bub = addMsg('assistant', '…');

    try {
      const r = await fetch(API + '/widget/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shop: SHOP, message: text || '（图片）',
                              images: imgs, session_id: sessionId})
      });
      const d = await r.json();
      bub.textContent = d.reply || '（无回复）';
    } catch (e) {
      bub.textContent = '网络异常，请稍后重试，或联系人工客服。';
    }
    sendBtn.disabled = false;
    body.scrollTop = body.scrollHeight;
  }

  sendBtn.onclick = send;
  input.onkeydown = e => { if (e.key === 'Enter') send(); };

  addMsg('assistant', '您好！有任何商品问题都可以问我，也可以直接发图片给我看。');
})();
"""


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Shopify App")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dev", action="store_true", help="放宽 CORS（本地开发用）")
    args = ap.parse_args()

    if args.dev:
        os.environ["DEV"] = "1"

    if not os.getenv("SHOPIFY_API_KEY"):
        print("⚠️  SHOPIFY_API_KEY 未设置。前台挂件和 OAuth 会不可用。")
        print("   把 .env 填好后再启动。")
        print()

    print(f"启动 http://{args.host}:{args.port}")
    print(f"  安装入口:  /auth?shop=your-store.myshopify.com")
    print(f"  后台界面:  /?shop=your-store.myshopify.com")
    print(f"  健康检查:  /health")
    print()
    print("本地开发需要 HTTPS 隧道（Shopify 要求）:")
    print("  ngrok http 8080")
    print("  然后设 SHOPIFY_APP_URL=https://xxx.ngrok.io")

    uvicorn.run(create_app(), host=args.host, port=args.port)
