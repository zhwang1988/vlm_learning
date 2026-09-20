"""
serve —— 推理服务化与优化。

对应讲义：docs/09-inference.md
对应计划：Week 5（Day 29–Day 30）

文件导航：
  vllm_server.sh   vLLM 启动脚本（含关键参数注释）
  api.py           FastAPI 网关（协议转换/限流/降级/结构化日志）
  quantize.py      AWQ/GPTQ 量化 + 混合精度 + 基准测试

启动顺序：
    # 1. 量化（可选，Day 29）
    python -m src.serve.quantize --model outputs/qwen25vl3b-cx-lora-v0

    # 2. 起 vLLM（后台）
    bash src/serve/vllm_server.sh outputs/qwen25vl3b-cx-lora-v0

    # 3. 起网关
    python -m src.serve.api

    # 4. 压测
    python -m src.serve.quantize --benchmark --n 20 --concurrency 4

三个必记的部署参数：
    --max-model-len       不要设成 128k 上限，会预留巨量 KV cache
    --limit-mm-per-prompt 必须设，否则一个请求塞 100 张图会打爆显存
    --gpu-memory-utilization  0.85–0.92，太高 OOM

生产必须有的两条：
    ① 降级路径（模型挂了返回预设话术 + 转人工）
    ② 结构化日志（能追溯到单次请求的每一段耗时）
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.serve import ChatRequest      # 触发时才 import 对应模块
#     python -m src.serve.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "ChatRequest": ".api",
    "ChatResponse": ".api",
    "ServeConfig": ".api",
    "filter_output": ".api",
    "validate_request": ".api",
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
    "ChatRequest",
    "ChatResponse",
    "ServeConfig",
    "filter_output",
    "validate_request",
]
