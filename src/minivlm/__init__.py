"""
minivlm —— 从零手搭的最小 VLM，用于理解架构。

对应讲义：docs/00-orientation.md ~ docs/04-qwen25vl.md
对应计划：Week 1（Day 1–Day 6）

模块导航：
    vision.py     ViT 从零实现 + SigLIP 生产封装 + visual token 估算
    connector.py  线性 / MLP / Perceiver 三种连接器
    processor.py  图像预处理 + visual token 数计算 + 与官方对拍
    model.py      MiniVLM 拼装（视觉塔 + 连接器 + LLM）
    generate.py   推理入口（官方 Qwen2.5-VL 和 Mini-VLM 两种）

快速开始：
    # 1. 看 ViT 和 token 估算
    python -m src.minivlm.vision

    # 2. 对比三种连接器
    python -m src.minivlm.connector

    # 3. 算 token 表
    python -m src.minivlm.processor --table

    # 4. ⭐ 与官方 processor 对拍（Day 4 的验收标准）
    python -m src.minivlm.processor --check

    # 5. 跑通拼装（需要下载权重，CPU 也能跑）
    python -m src.minivlm.model

    # 6. 真实推理
    python -m src.minivlm.generate --image assets/sample.jpg --compare-order
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.minivlm import ImageProcessConfig      # 触发时才 import 对应模块
#     python -m src.minivlm.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "ImageProcessConfig": ".processor",
    "LinearConnector": ".connector",
    "MLPConnector": ".connector",
    "PerceiverResampler": ".connector",
    "SiglipVisionWrapper": ".vision",
    "TinyViT": ".vision",
    "assign_bucket": ".processor",
    "build_2d_sincos_pos_embed": ".vision",
    "build_connector": ".connector",
    "compute_visual_tokens": ".processor",
    "estimate_tokens_for_batch": ".processor",
    "estimate_visual_tokens": ".vision",
    "smart_resize": ".processor",
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
    "LinearConnector",
    "MLPConnector",
    "PerceiverResampler",
    "build_connector",
    "ImageProcessConfig",
    "assign_bucket",
    "compute_visual_tokens",
    "estimate_tokens_for_batch",
    "smart_resize",
    "SiglipVisionWrapper",
    "TinyViT",
    "build_2d_sincos_pos_embed",
    "estimate_visual_tokens",
]
