# Day 42 · 部署上线

> 预计 3–4h ｜ 📓 `notebooks/day-42_deploy.ipynb` ｜ 💻 本地 · `Dockerfile`、`docker-compose.yml`、`src/shopify/app.py`
> 前置：Day 41

## 今日目标（一句话）

Docker 化并部署到有公网地址的地方；配好 HTTPS、环境变量、日志；验收是「从陌生店铺安装到前台能对话」全流程走通。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 8 节（部署）
- `docker-compose.yml` —— 看它怎么把 PostgreSQL + pgvector + 应用串起来
- `src/shopify/__init__.py` 里记录的「五个最容易踩的坑」

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么 webhook 必须走 HTTPS？（Shopify 会拒绝 http 回调）
2. 环境变量该怎么传给容器才不会被写进镜像层？
3. 数据库没做持久化卷会发生什么？

## 二、写（100 min）

部署（本日以配置和验证为主）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `Dockerfile` | python:3.11-slim，**不含 torch**（应用侧只调 vLLM） |
| `docker-compose.yml` | pgvector/pg16 + 应用；数据库有 healthcheck 与持久化卷 |
| 环境变量 | 用 `env_file: .env`，**绝不 `COPY .env`** 进镜像 |
| HTTPS | 平台自带证书；用 ngrok/cloudflared 也能本地演示 |
| 日志 | stdout 结构化输出，交给平台收集 |
| 健康检查 | `/health` 供平台探活 |

## 三、跑（本地（无需 GPU））

```bash
# 起数据库 + 应用
docker compose up -d
# 看容器状态
docker compose ps
# 探活
curl -s localhost:8080/health
# 看日志
docker compose logs -f app | head -30
# 收工（数据保留）
make down
```

期望输出（节选）：
```
NAME          STATUS
pgvector      Up (healthy)
app           Up (healthy)

$ curl localhost:8080/health
{"status":"ok","db":"ok","vllm":"ok"}

$ docker compose logs app
{"event":"startup","port":8080,"shop_count":1}
{"event":"webhook_registered","topics":["products/update","orders/create","refunds/create"]}

公网地址（平台给的）：
  https://your-app.onrender.com
  → Shopify 后台填这个作为 App URL 和 webhook 回调地址
```

## 四、验收清单

- [ ] 有一个**公网可访问**的地址，`/health` 返回 ok
- [ ] 从陌生店铺走完「安装 → 授权 → 前台能对话」全流程
- [ ] `.env` 没进镜像（用 `docker history` 能验）
- [ ] 数据库数据卷已挂载（重启容器数据还在）
- [ ] `progress/weekly-review.md` 的 W7 段已写；进度表 W7 六天 `[x]`，M7 打卡

## 五、容易踩的坑

1. **webhook 用 http 回调** —— Shopify 直接拒绝，你的索引永远不同步，而且不容易发现。
2. **`.env` 被 `COPY` 进镜像** —— token 全在镜像层里，推到公开 registry 就全泄露了。
3. 数据库没挂数据卷 —— 容器一重启，所有的 shops / subscriptions 全没了。
4. torch 打进了应用镜像 —— 镜像从 200 MB 变成 8 GB，部署慢十倍（应用侧只调 vLLM，不需要 torch）。
5. 平台冷启动导致 webhook 超时 —— 免费套餐会休眠，webhook 打过来要等几十秒才醒。要么用付费套餐，要么把 webhook 收口到一个常驻的轻量服务。
6. 忘了配 `SCOPES` / 环境变量 —— 装上去了但一调 API 就 403。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。

> **M7 达成**：有一个公网可访问的 Shopify App，从陌生店铺安装到前台对话全流程走通。
