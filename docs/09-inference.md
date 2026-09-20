# 09 · 推理优化与服务化

> 对应 Day 29–Day 30

## 概念

### 从「能跑」到「能服务」差的六件事

```
训练脚本                              生产服务
─────────                            ─────────
单条输入                             并发请求
batch = 1                            continuous batching
无超时                               超时 + 重试 + 降级
OOM 就重启                           显存水位控制、请求排队
日志打屏幕上                          结构化日志 + 追踪 + 监控
精度优先  (bf16)                      成本优先  (int4/awq)
```

### vLLM 为什么快

核心是 **PagedAttention**。

```
传统 KV cache：为每个请求预留 max_seq_len 的连续显存
   → 实际用到 500 token，但预留了 4096 → 浪费 87%

PagedAttention：把 KV cache 切成固定大小的 block（如 16 token），
   按需分配，用页表映射（和操作系统虚拟内存一个思路）
   → 显存利用率从 ~20% 提升到 ~90%+
```

副产品是 **continuous batching**：不同请求的序列长度不同，传统方案要等最长的那个结束才能接新请求。有了分页，新请求可以立刻插入，**吞吐提升 5–20 倍**。

**实际数字参考**（Qwen2.5-VL-3B，A100 40G）：
- 单请求：首 token 延迟 ~150ms，生成 ~40 tok/s
- 8 并发：总吞吐 ~200 tok/s（不是 8×40=320，但远好于串行）

### 量化的四档

| 精度 | 权重占用 | 质量损失 | 适合 |
|---|---|---|---|
| bf16 | 100% | 无 | 训练、离线质量基线 |
| **FP8** | 50% | 极小（<0.5%） | H100/L40S，推荐 |
| **AWQ (int4)** | 25% | 小（1–2%） | A100/4090/消费卡，推荐 |
| GPTQ (int4) | 25% | 小（1–2%） | 同 AWQ，生态更老 |
| GGUF Q4_K_M | 25% | 中（3–5%） | CPU/端侧 |

**VLM 量化的特殊坑**：**视觉塔对量化比 LLM 敏感得多**。

原因：视觉特征里的高频信息（细节纹理）在低精度下最先丢失。表现是 OCR 数字读错、颜色判断偏、小瑕疵看不见——**恰好是客服场景最需要的三种能力**。

**实践做法**：**混合精度量化**——视觉塔保持 bf16，LLM 用 int4。AWQ 配置里可以通过 `modules_to_not_convert` 指定：

```python
# 量化时排除视觉塔
quant_config = {
    "modules_to_not_convert": ["visual", "vision_tower", "merger"],
}
```

代价是省得少一点（视觉塔占 675M/3675M ≈ 18%），但质量保住了。**这个取舍是值得的。**

### 延迟分解与优化点

一次图文请求的延迟构成：

```
网络传输    ~20ms   ← CDN、就近部署
图片下载    ~80ms   ← 预取、缓存、限制尺寸
图片预处理  ~30ms   ← 用 GPU 做、批处理
视觉编码    ~120ms  ← 视觉塔前向，token 越多越慢
Prefill    ~200ms  ← 和 visual token 数强相关
Decode      ~40ms/token × N  ← 主要成本
────────────────────
总计        1–3s（100 token 回复）
```

**优化优先级**（按 ROI 排序）：

1. **减少 visual token 数**：降 `max_pixels`。收益最大、风险最小（用第 4 周的评测曲线找拐点）
2. **减少输出长度**：客服回复 100 字够了，prompt 里明确限制
3. **流式输出**：用户感知延迟从 2s 降到 300ms（首 token 时间）
4. **提示词前缀缓存**：system prompt 固定 → vLLM 的 `enable_prefix_caching` 能省掉重复 prefill
5. **量化**：显存省 4×，允许更大 batch
6. **多副本**：最后才考虑，成本线性上升

### 单位经济

算清楚这个，你才能回答「这个产品能不能赚钱」（Day 43 的产出）。

```
成本项：
  GPU:  A100 40G 按量约 ¥8/小时 → ¥0.0022/秒
  API:  强模型调用（数据合成期用，生产不用）
  基础设施: 服务器 + 数据库 + 对象存储 ≈ ¥300/月

单次会话成本（假设 3 轮对话，每轮 1000 in + 150 out）：
  prefill:  3000 tok  → A100 上 ~0.15s
  decode:   450 tok   → ~11s 但并发 8 路 → 实际占用 ~1.4s
  GPU 成本 ≈ 1.55s × ¥0.0022 ≈ ¥0.0034

加 30% 冗余 → 单次会话 ≈ ¥0.005

定价参考：
  免费版：50 次/月
  标准版：¥99/月，1000 次会话（成本 ¥5，毛利 95%）
  但真实成本在「基础设施固定成本 + 人力」
  → 需要 500+ 付费商店才能打平
```

**这个数字不一定准，但做一遍的价值在于**：你会立刻意识到「模型效果」和「商业可行」是两件事，而后者约束了你的技术选择（比如不能无脑用 7B 全参）。

## 工程细节

### vLLM 部署多模态模型

```bash
# 起服务
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --served-model-name cx-vlm \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=3 \
  --enable-prefix-caching \
  --quantization awq \                 # 或留空用 bf16
  --trust-remote-code
```

**关键参数**：

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `--gpu-memory-utilization` | KV cache 占用比例 | 0.85–0.92，太高会 OOM |
| `--max-model-len` | 最大上下文 | **不要设成模型上限 128k**，会预留巨量 KV cache |
| `--max-num-seqs` | 最大并发 | 影响吞吐，缺省就好 |
| `--limit-mm-per-prompt` | 限制多模态输入数 | **必须设**，否则一个恶意请求塞 100 张图打爆显存 |
| `--enable-prefix-caching` | 前缀缓存 | 固定 system prompt 时收益大 |

### FastAPI 网关

`src/serve/api.py` 实现。要点：

```python
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    # 1. 输入校验（图片大小/数量/格式）
    validate_input(req)

    # 2. 超时保护
    try:
        async with asyncio.timeout(TIMEOUT):
            resp = await call_vllm(req)
    except TimeoutError:
        return fallback_response()      # 降级：返回预设话术 + 转人工

    # 3. 输出过滤
    resp = filter_output(resp)

    # 4. 结构化日志
    log.info("chat", extra={
        "session_id": req.session_id,
        "n_images": len(req.images),
        "visual_tokens": resp.usage.visual_tokens,
        "prefill_ms": resp.metrics.prefill_ms,
        "decode_ms": resp.metrics.decode_ms,
        "total_ms": elapsed,
    })
    return resp
```

**必须有降级路径**：模型服务挂了，客服系统不能挂。降级 = 返回「抱歉，我暂时无法处理，已为您转接人工」。

### 批处理与并发

```python
# 反例：串行处理，100 个请求 × 1.5s = 150s
for req in requests:
    await process(req)

# 正例：并发，受 vLLM 自身的 continuous batching 限制
sem = asyncio.Semaphore(MAX_CONCURRENCY)     # 如 16
async def bounded(req):
    async with sem:
        return await process(req)
results = await asyncio.gather(*[bounded(r) for r in requests])
```

`MAX_CONCURRENCY` 要通过压测找——超过某个值，吞吐不再上升但延迟线性恶化。

## 自检问题

1. PagedAttention 解决了什么问题？为什么它能同时提升吞吐和降低显存浪费？
2. VLM 量化时为什么视觉塔比 LLM 敏感？表现是什么？
3. 混合精度量化怎么做？代价是什么？
4. 一次图文请求的延迟由哪几部分组成？优化优先级怎么排？
5. `--max-model-len` 为什么不能直接设成模型的最大支持长度？
6. `--limit-mm-per-prompt` 不设会有什么风险？
7. 为什么生产服务必须有降级路径？

---

**上一篇**：[08-evaluation.md](08-evaluation.md) · **下一篇**：[10-agent.md](10-agent.md)
