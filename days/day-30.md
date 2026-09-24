# Day 30 · 端到端推理服务

> 预计 3–4h ｜ 📓 `notebooks/day-30_inference_service.ipynb` ｜ ☁️ 云 GPU · `src/serve/api.py`
> 前置：Day 29（vLLM 已能起来）

## 今日目标（一句话）

写一个 FastAPI 网关：收 base64/URL 图片 + 文本 → 转发 vLLM → 带超时、重试、限流、结构化日志；并发 10 请求不炸，且日志能追溯到单次耗时。

## 一、读（60 min）

材料：
- `docs/09-inference.md` 第 5–6 节（服务化与成本）
- `src/serve/api.py` 的 `validate_request` 与 `filter_output`

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么网关层要做图片校验？（提示：有人会传 20 MB 的图）
2. 超时该设多少？设置依据是什么？（提示：P95 延迟 + 余量）
3. 日志里为什么绝对不能记原图 base64？

## 二、写（100 min）

`src/serve/api.py`（已给实现，你要改限流与超时）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `validate_request()` | 图片数量 / 大小 / 格式校验 —— 第一道闸门 |
| `VLLMClient` | 转发到 vLLM，带超时与重试 |
| `filter_output()` | 输出过滤：PII 掩码（手机号 / 订单号） |
| 结构化日志 | request_id / 耗时 / token 数 / 是否降级 —— **可追溯单次请求** |
| `create_app()` | FastAPI 工厂；uvicorn 要用 `--factory` |

## 三、跑（在云 GPU 上）

```bash
# 起网关
python -m src.serve.api --port 8080
# 健康检查
curl -s localhost:8080/health
# 打通 /v1/chat
curl -s localhost:8080/v1/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"你好，这件有货吗"}],"images":[]}'
# 并发 10 压一把（loadtest 在 W8 完善）
python scripts/loadtest.py --url http://localhost:8080/v1/chat --n 10 --concurrency 10
```

期望输出（节选）：
```
INFO  starting gateway on 0.0.0.0:8080  vllm=http://localhost:8000
{"request_id":"a3f2","event":"request","n_images":0,"text_len":9}
{"request_id":"a3f2","event":"response","latency_ms":412,"out_tokens":58,"degraded":false}

$ curl -s localhost:8080/health
{"status":"ok","vllm":"ok","model":"qwen25vl3b-cx-dpo-v0"}

$ curl ... /v1/chat
{"reply":"您好，M/L 码有现货，S 码预计 3 天补货。","request_id":"a3f2"}
```

## 四、验收清单

- [ ] `curl` 能打通 `/health` 和 `/v1/chat`
- [ ] 并发 10 请求不崩、不超时，且日志能查到每一单的耗时
- [ ] 图片校验生效（传一张 20 MB 的图会被友好拒绝，不是崩）
- [ ] `progress/weekly-review.md` 的 W5 段已写；进度表 W5 六天 `[x]`，M5 打卡

## 五、容易踩的坑

1. **没有超时** —— 一个慢请求把连接池占满，后面所有请求堆积。必须设 connect/read 双超时。
2. 日志里记了 base64 原图 —— 日志文件瞬间膨胀，还涉嫌泄露用户数据。
3. uvicorn 起不来 —— 因为 `create_app` 是工厂函数，要加 `--factory`。
4. PII 只做了输出过滤没做输入过滤 —— 用户发的手机号照样进了日志。
5. 降级路径没测过 —— vLLM 挂了你才发现降级分支有语法错误。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。

> **M5 达成**：有一个能对外服务的推理 API + 一份量化对比表 + W5 周复盘。
