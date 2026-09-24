# Day 38 · OAuth 与应用骨架

> 预计 3–4h ｜ 📓 `notebooks/day-38_shopify_oauth.ipynb` ｜ 💻 本地 · `src/shopify/app.py`、`src/shopify/auth.py`
> 前置：Day 37；已建好 Shopify App 拿到 API key / secret

## 今日目标（一句话）

把 OAuth 安装流程跑通：`/auth` → 用户授权 → 回调 → 换 access token → 存库；并说清 **HMAC 校验** 和 **session token** 的关系。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 3 节（OAuth 与鉴权）
- `src/shopify/auth.py` 的 `verify_hmac` —— **这份文件里埋了三个坑，注释都写了**

思考题（先自己想，答案在讲义或代码注释里）：
1. HMAC 校验时为什么必须先**剔除 `hmac` 参数**、再按 key 排序？
2. Access token 和 Session token 有什么区别？各自活多久？
3. 为什么必须校验 `shop` 参数的域名格式？（提示：SSRF）

## 二、写（100 min）

`src/shopify/auth.py` + `src/shopify/app.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `verify_hmac()` | **剔 hmac → 按 key 排序 → 拼接 → sha256**，顺序不能错 |
| `verify_webhook_hmac()` | webhook 的 HMAC 是 **base64** 不是 hex（另一个坑） |
| `validate_shop()` | 正则校验域名，防 SSRF —— **绝不能省** |
| `build_install_url()` | 生成授权链接（含 state 防 CSRF） |
| `exchange_code_for_token()` | 拿 code 换 access token |
| `verify_session_token()` | 校验 JWT（App Bridge 传来的），验签 + 验 exp + 验 aud |
| `TokenStore` | token 存库（**别存文件、别进 git**） |

## 三、跑（本地（无需 GPU））

```bash
# 自检：HMAC 正确/错误/两个坑 + SSRF 拦截
python -m src.shopify.auth
# 起 App（本地开发，放宽 CORS）
python -m src.shopify.app --dev --port 8080
# 走一次安装流程（需要真实测试店）
open http://localhost:8080/auth?shop=demo.myshopify.com
```

期望输出（节选）：
```
[1] HMAC 校验
    正确签名（乱序 query 也应通过）→ True
    参数顺序颠倒后仍通过        → True
    坑1 用原始顺序直接算        → False （应该 False）✓
    坑2 忘了剔除 hmac 参数      → False （应该 False）✓

[2] SSRF 防护
    demo.myshopify.com        → True
    evil.com                  → False ✓
    demo.myshopify.com.evil.io → False ✓

[3] 安装流程
    GET /auth?shop=demo.myshopify.com
    → 302 到 Shopify 授权页（带 state=xxx）
    → 回调 /auth/callback?code=...&hmac=...&state=xxx
    → 换到 access token，已入库（shops 表）
    ✓ 安装完成
```

## 四、验收清单

- [ ] `python -m src.shopify.auth` 自检全部通过，**两个 HMAC 坑都被拦住**
- [ ] 能在测试店完成一次完整安装（`/auth` → 授权 → 回前台正常）
- [ ] 能解释 HMAC 校验和 session token 的关系（谁在前、各自防什么）
- [ ] 知道 SSRF 是什么、`validate_shop` 拦的是哪种攻击

## 五、容易踩的坑

1. **HMAC 没剔除 `hmac` 参数** —— 最常见的错。带着 hmac 去算哈希，签名永远不对。
2. **忘了按 key 排序** —— query string 的参数顺序不保证，不排序就是碰运气。
3. **webhook 的 HMAC 用了 hex 解码** —— Shopify 的 webhook 签名是 **base64**，和 OAuth 回调的 hex 不是一套，混用直接失败。
4. **不校验 shop 域名** —— 攻击者传 `shop=evil.com`，你的服务器就拿着 token 去请求 evil.com 了。
5. access token 写进文件或日志 —— 这是能操作商家店铺的凭证，泄露等于店铺被接管。
6. state 参数没校验 —— CSRF：别人诱导店主点一个链接，就把他的店绑到你的 app 上。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
