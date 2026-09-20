"""
推理服务：vLLM 部署 + FastAPI 网关。

对应讲义 docs/09-inference.md，对应计划 Day 29–Day 30。

设计要点：
  - 网关只管协议转换、限流、降级、日志；真正推理交给 vLLM
  - **必须有降级路径**：模型服务挂了，客服系统不能挂
  - 结构化日志要能追溯到单次请求的每一段耗时

启动：
    # 1. 起 vLLM（另开终端）
    bash src/serve/vllm_server.sh Qwen/Qwen2.5-VL-3B-Instruct
    # 2. 起网关
    python -m src.serve.api
    # 3. 测试
    curl -X POST localhost:8080/v1/chat -H 'Content-Type: application/json' \
      -d '{"message": "你好", "images": []}'
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# 轻量级依赖，没有 fastapi 时也能 import 这个模块做单测
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    BaseModel = object
    Field = lambda **k: None       # noqa: E731


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ServeConfig:
    vllm_base: str = os.getenv("VLLM_BASE", "http://localhost:8000/v1")
    served_model: str = os.getenv("SERVED_MODEL", "cx-vlm")
    timeout_s: float = float(os.getenv("REQUEST_TIMEOUT", "30"))
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "16"))
    max_images: int = 3
    max_image_bytes: int = 8 * 1024 * 1024
    max_new_tokens: int = 512
    max_pixels: int = int(os.getenv("MAX_PIXELS_RATIO", "1280"))

    system_prompt: str = (
        "你是一位专业的电商客服助手。请基于用户提供的图片和文字，"
        "给出准确、有帮助、语气自然的回复。只依据图片中可见的内容和"
        "已知的商品信息回答；信息不足时请主动询问，不要编造。"
    )
    # 降级话术
    fallback_reply: str = (
        "抱歉，我这边暂时遇到一点问题，无法立即为您处理。"
        "已经为您记录了问题，稍后会有客服专员跟进，请稍等。"
    )


CFG = ServeConfig()


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


if HAS_FASTAPI:
    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=4000)
        images: list[str] = Field(default_factory=list,
                                  description="base64 data URL 或 http URL")
        session_id: Optional[str] = None
        history: list[dict] = Field(default_factory=list)
        stream: bool = False
        max_new_tokens: Optional[int] = None

    class ChatResponse(BaseModel):
        reply: str
        session_id: str
        degraded: bool = False
        usage: dict = Field(default_factory=dict)
        latency_ms: float = 0.0
else:
    class ChatRequest:      # type: ignore
        pass

    class ChatResponse:     # type: ignore
        pass


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def validate_request(req) -> Optional[str]:
    """返回错误信息，None 表示通过。

    校验的意义：防止一个恶意请求塞 100 张 4K 图把显存打爆。
    这就是 vLLM 的 --limit-mm-per-prompt 存在的理由。
    """
    if len(req.images) > CFG.max_images:
        return f"图片数量超限（{len(req.images)} > {CFG.max_images}）"

    total = 0
    for img in req.images:
        if img.startswith("data:"):
            try:
                b64 = img.split(",", 1)[1]
                size = len(b64) * 3 // 4
            except Exception:
                return "图片 data URL 格式错误"
        else:
            size = 0
        total += size
        if total > CFG.max_image_bytes:
            return f"图片总体积超限（>{CFG.max_image_bytes // 1024 // 1024}MB）"
    return None


# ---------------------------------------------------------------------------
# vLLM 调用
# ---------------------------------------------------------------------------


class VLLMClient:
    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_concurrency)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.cfg.vllm_base, api_key="EMPTY")
        return self._client

    async def chat(self, req: ChatRequest) -> tuple[str, dict]:
        content = []
        for url in req.images:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": req.message})

        messages = [{"role": "system", "content": self.cfg.system_prompt}]
        for h in req.history[-6:]:            # 只带最近 6 轮，控制上下文
            messages.append({"role": h.get("role", "user"),
                             "content": h.get("content", "")})
        messages.append({"role": "user", "content": content})

        async with self.sem:
            t0 = time.perf_counter()
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.cfg.served_model,
                    messages=messages,
                    max_tokens=req.max_new_tokens or self.cfg.max_new_tokens,
                    temperature=0.3,
                ),
                timeout=self.cfg.timeout_s,
            )
        elapsed = (time.perf_counter() - t0) * 1000

        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "vllm_ms": round(elapsed, 1),
        }
        return resp.choices[0].message.content, usage


VLLM = VLLMClient(CFG)


# ---------------------------------------------------------------------------
# 输出过滤
# ---------------------------------------------------------------------------

import re

PII_PATTERNS = [
    (r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2"),          # 手机号
    (r"(?<!\d)(\d{6})\d{8}(\d{4})(?!\d)", r"\1********\2"),          # 银行卡
]


def filter_output(text: str) -> str:
    """输出过滤：脱敏 PII。

    客服场景必须做这一步，因为日志和前端都会看到这段文本。
    """
    for pat, rep in PII_PATTERNS:
        text = re.sub(pat, rep, text)
    # 去掉内部标识泄露
    for marker in ["<|im_end|>", "<|im_start|>", "<|image_pad|>"]:
        text = text.replace(marker, "")
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    from ..train.rewards import extract_json as ej
    return ej(text)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------


def create_app():
    if not HAS_FASTAPI:
        raise RuntimeError("未安装 fastapi：pip install fastapi uvicorn")

    app = FastAPI(title="CX Multimodal Customer Service", version="0.1.0")

    @app.get("/health")
    async def health():
        """健康检查。生产环境要检查 vLLM 是否真的活着，不能只返回 ok。"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{CFG.vllm_base}/models")
            vllm_ok = r.status_code == 200
        except Exception:
            vllm_ok = False
        return {"gateway": "ok", "vllm": "ok" if vllm_ok else "unreachable",
                "degraded": not vllm_ok}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request):
        sid = req.session_id or str(uuid.uuid4())
        t_start = time.perf_counter()

        # 1. 校验
        err = validate_request(req)
        if err:
            raise HTTPException(status_code=400, detail=err)

        # 2. 推理（带超时和降级）
        degraded = False
        usage: dict = {}
        try:
            reply, usage = await VLLM.chat(req)
        except asyncio.TimeoutError:
            reply, degraded = CFG.fallback_reply, True
            usage = {"error": "timeout"}
        except Exception as e:      # noqa: BLE001
            reply, degraded = CFG.fallback_reply, True
            usage = {"error": str(e)[:200]}

        # 3. 过滤
        reply = filter_output(reply)

        total_ms = (time.perf_counter() - t_start) * 1000

        # 4. 结构化日志（能追溯到单次请求的每一段）
        log = {
            "event": "chat",
            "session_id": sid,
            "n_images": len(req.images),
            "msg_len": len(req.message),
            "reply_len": len(reply),
            "degraded": degraded,
            "total_ms": round(total_ms, 1),
            **usage,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        print(json.dumps(log, ensure_ascii=False))

        return ChatResponse(
            reply=reply, session_id=sid, degraded=degraded,
            usage=usage, latency_ms=round(total_ms, 1),
        )

    @app.post("/v1/chat/stream")
    async def chat_stream(req: ChatRequest):
        """流式输出。用户感知延迟从 2s 降到 ~300ms（首 token）。"""
        async def gen():
            sid = req.session_id or str(uuid.uuid4())
            yield f"data: {json.dumps({'type': 'start', 'session_id': sid})}\n\n"
            try:
                content = []
                for url in req.images:
                    content.append({"type": "image_url", "image_url": {"url": url}})
                content.append({"type": "text", "text": req.message})
                messages = [{"role": "system", "content": CFG.system_prompt},
                            {"role": "user", "content": content}]
                stream = await VLLM.client.chat.completions.create(
                    model=CFG.served_model, messages=messages,
                    max_tokens=CFG.max_new_tokens, temperature=0.3, stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield f"data: {json.dumps({'type': 'delta', 'text': delta}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:      # noqa: BLE001
                yield f"data: {json.dumps({'type': 'error', 'fallback': CFG.fallback_reply, 'error': str(e)[:120]}, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/agent")
    async def agent_chat(req: ChatRequest):
        """Agent 入口（Day 34 之后接入）。"""
        from ..agent.agent import get_default_agent
        agent = get_default_agent()
        if agent is None:
            raise HTTPException(status_code=503,
                                detail="Agent 未启用（需先完成 Day 31–34）")
        out = await agent.arun(req.message, req.images, session_id=req.session_id)
        return out

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="推理网关")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    print(f"网关启动在 http://{args.host}:{args.port}")
    print(f"  vLLM 后端: {CFG.vllm_base}")
    print(f"  并发上限: {CFG.max_concurrency}   超时: {CFG.timeout_s}s")
    print(f"  健康检查: GET /health")
    uvicorn.run(create_app(), host=args.host, port=args.port)
