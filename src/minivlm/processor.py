"""
图像预处理与 visual token 数计算。

对应讲义 docs/04-qwen25vl.md，对应计划 Day 4。

这是全项目最容易出错、也最值得反复验证的一个模块。
核心断言只有一个：

    **你算出的 visual token 数，必须和官方 processor 生成的 <|image_pad|> 个数完全一致。**

不一致的后果是训练时一个 shape mismatch 报错，而且报错信息不会告诉你
「是第 3 张图还是第 7 张图出了问题」。所以必须提前用脚本卡住。

用法：
    python -m src.minivlm.processor --check     # 与官方 processor 对拍
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


# =============================================================================
# 1. Visual token 计算（Day 4 的核心练习）
# =============================================================================


@dataclass
class ImageProcessConfig:
    """对齐 Qwen2.5-VL 的图像处理超参数。

    这些值来自官方 processor 的默认配置：
        patch_size          = 14
        merge_size          = 2
        min_pixels          = 56 * 56        (约 3136)
        max_pixels          = 1280 * 28 * 28 (约 1,003,520)
    """
    patch_size: int = 14
    merge_size: int = 2
    min_pixels: int = 56 * 56
    max_pixels: int = 1280 * 28 * 28

    @property
    def factor(self) -> int:
        """尺寸对齐因子。官方用 patch_size * merge_size = 28。"""
        return self.patch_size * self.merge_size


def smart_resize(
    height: int, width: int, cfg: ImageProcessConfig
) -> tuple[int, int]:
    """Qwen2.5-VL 的 smart_resize：保持比例，把总像素压到 [min, max] 区间内。

    和普通 resize 的三个区别：
      1. 保持宽高比（不 squash）
      2. 两个维度都必须是 factor(=28) 的整数倍
      3. 上下限是对总像素数（H*W）的限制，不是单边

    返回 (new_height, new_width)。
    """
    f = cfg.factor

    # 边界检查
    if height < f or width < f:
        raise ValueError(f"图像尺寸 {height}x{width} 太小，两边都必须 >= {f}")

    h_bar = max(f, round(height / f) * f)
    w_bar = max(f, round(width / f) * f)

    if h_bar * w_bar > cfg.max_pixels:
        # 图太大：等比缩小
        beta = math.sqrt((height * width) / cfg.max_pixels)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < cfg.min_pixels:
        # 图太小：等比放大（否则看到的细节太少）
        beta = math.sqrt(cfg.min_pixels / (height * width))
        h_bar = math.ceil(height * beta / f) * f
        w_bar = math.ceil(width * beta / f) * f

    return h_bar, w_bar


def compute_visual_tokens(
    height: int, width: int, cfg: ImageProcessConfig | None = None
) -> tuple[int, tuple[int, int]]:
    """算一张原图经过处理后会产生多少 visual token。

    步骤：
      1. smart_resize 得到对齐后的 (h_bar, w_bar)
      2. grid_h, grid_w = h_bar/patch, w_bar/patch
      3. 2x2 merging 后 token 数 = (grid_h/2) * (grid_w/2)

    返回 (token 数, (grid_h, grid_w))
    """
    cfg = cfg or ImageProcessConfig()
    h_bar, w_bar = smart_resize(height, width, cfg)

    grid_h = h_bar // cfg.patch_size
    grid_w = w_bar // cfg.patch_size

    n_tokens = (grid_h // cfg.merge_size) * (grid_w // cfg.merge_size)
    return n_tokens, (grid_h, grid_w)


# =============================================================================
# 2. 与官方 processor 对拍（Day 4 的验收标准）
# =============================================================================


def _make_test_image(h: int, w: int):
    """造一张指定尺寸的测试图（带一点纹理，避免纯色被优化掉）。"""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    # 加一点结构，避免被误判为无效图
    arr[h // 4: h // 2, w // 4: w // 2] = 200
    return Image.fromarray(arr)


def check_against_official(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """核心验收：自己算的 token 数 == 官方 processor 生成的 pad 数。

    这是 Day 4 代码任务的判定标准。跑通它，后面训练才不会出 shape 错误。
    """
    from transformers import AutoProcessor

    print("=" * 78)
    print("Visual Token 对拍：自己算 vs 官方 processor")
    print("=" * 78)

    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ 无法加载 processor: {e}")
        print("   请确认已 pip install transformers>=4.49 且有网络/HF token")
        return False

    cfg = ImageProcessConfig()
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    cases = [
        (224, 224), (448, 448), (512, 384), (800, 600),
        (1024, 1024), (1920, 1080), (1080, 1920), (300, 1200),
        (56, 56), (4000, 3000), (2000, 500),
    ]

    print(f"\n{'原图尺寸':>14} | {'自己算':>8} | {'官方':>8} | {'对齐后':>12} | 结果")
    print("-" * 78)

    all_ok = True
    for h, w in cases:
        img = _make_test_image(h, w)

        # 自己算
        try:
            mine, grid = compute_visual_tokens(h, w, cfg)
        except ValueError as e:
            print(f"{f'{h}x{w}':>14} | {'-':>8} | {'-':>8} | {'-':>12} | 跳过({e})")
            continue

        # 官方算
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "描述这张图"},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        official = (inputs["input_ids"] == pad_id).sum().item()

        h_bar, w_bar = smart_resize(h, w, cfg)
        ok = (mine == official)
        all_ok &= ok
        mark = "✓" if ok else "✗ 不一致"
        if not ok:
            print(f"\n  ⚠️  {h}x{w}: 自己算 {mine}, 官方 {official}")
            print(f"      差值 {official - mine}。常见原因：")
            print(f"      - merge_size 的取整方式（先 merge 再对齐 vs 先对齐再 merge）")
            print(f"      - grid 为奇数时的处理（官方可能 pad 到偶数）")
            print(f"      - min_pixels/max_pixels 与 processor 默认值不一致")

        print(f"{f'{h}x{w}':>14} | {mine:>8} | {official:>8} | "
              f"{f'{h_bar}x{w_bar}':>12} | {mark}")

    print("-" * 78)
    if all_ok:
        print("✓ 全部对齐！可以放心进入训练阶段。")
    else:
        print("✗ 存在不一致。回去检查 smart_resize 的取整逻辑，")
        print("  对照 transformers 里 Qwen2VLImageProcessor 的源码。")
        print("  （这个坑必须填，否则 Day 15 训练时一定报 shape 错误）")
    return all_ok


# =============================================================================
# 3. 数据构造用的辅助函数
# =============================================================================


def estimate_tokens_for_batch(sizes: list[tuple[int, int]],
                              cfg: ImageProcessConfig | None = None
                              ) -> dict:
    """给一批图片估算 token 分布。Day 7 的数据分析用。"""
    cfg = cfg or ImageProcessConfig()
    tokens = []
    for h, w in sizes:
        try:
            n, _ = compute_visual_tokens(h, w, cfg)
            tokens.append(n)
        except ValueError:
            continue

    if not tokens:
        return {}

    tokens_t = torch.tensor(tokens, dtype=torch.float32)
    return {
        "n_images": len(tokens),
        "total_tokens": int(tokens_t.sum()),
        "mean": float(tokens_t.mean()),
        "median": float(tokens_t.median()),
        "p90": float(tokens_t.quantile(0.9)),
        "p99": float(tokens_t.quantile(0.99)),
        "max": int(tokens_t.max()),
        "min": int(tokens_t.min()),
    }


def assign_bucket(height: int, width: int, n_buckets: int = 5,
                  cfg: ImageProcessConfig | None = None) -> int:
    """把图片按 token 数分桶。同一 batch 用同一桶的图，能省 20-40% 显存。

    原理：动态分辨率下同一个 batch 里 token 数差异巨大时，
          padding 会浪费大量显存。分桶让同 batch 内长度接近。
    """
    cfg = cfg or ImageProcessConfig()
    n, _ = compute_visual_tokens(height, width, cfg)
    # 假设单图上限约 1280 token
    bucket = min(int(n // (1280 / n_buckets)), n_buckets - 1)
    return max(0, bucket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="与官方 processor 对拍（需要下载模型配置）")
    parser.add_argument("--table", action="store_true", help="打印 token 估算表")
    args = parser.parse_args()

    if args.check:
        check_against_official()
    elif args.table:
        cfg = ImageProcessConfig()
        print(f"配置: patch={cfg.patch_size} merge={cfg.merge_size} "
              f"min_px={cfg.min_pixels:,} max_px={cfg.max_pixels:,}\n")
        print(f"{'原图尺寸':>14} | {'对齐后':>12} | {'grid':>10} | {'tokens':>8} | {'桶':>4}")
        print("-" * 64)
        for h, w in [(224, 224), (448, 448), (800, 600), (1024, 1024),
                     (1920, 1080), (1080, 1920), (300, 1200), (4000, 3000)]:
            n, grid = compute_visual_tokens(h, w, cfg)
            hb, wb = smart_resize(h, w, cfg)
            bk = assign_bucket(h, w, cfg=cfg)
            print(f"{f'{h}x{w}':>14} | {f'{hb}x{wb}':>12} | "
                  f"{f'{grid[0]}x{grid[1]}':>10} | {n:>8} | {bk:>4}")
        print()
        print("用法建议：")
        print("  - 训练时按「桶」组 batch，同桶内 token 数接近 → 省显存")
        print("  - 1920x1080 和 1080x1920 应落在不同桶（grid 形状不同，不能混）")
    else:
        # 默认：先跑不依赖外部模型的自检
        print("=" * 72)
        print("visual token 估算（无需下载模型）")
        print("=" * 72)
        cfg = ImageProcessConfig()
        print(f"\nmax_pixels = {cfg.max_pixels:,} (约 {cfg.max_pixels // (28*28)} 个 merge 单元)\n")
        print(f"{'原图尺寸':>14} | {'对齐后':>12} | {'tokens':>8}")
        print("-" * 44)
        for h, w in [(224, 224), (448, 448), (800, 600), (1024, 1024),
                     (1920, 1080), (1080, 1920), (300, 1200), (4000, 3000)]:
            n, _ = compute_visual_tokens(h, w, cfg)
            hb, wb = smart_resize(h, w, cfg)
            print(f"{f'{h}x{w}':>14} | {f'{hb}x{wb}':>12} | {n:>8}")
        print()
        print("下一步（Day 4 的验收）：")
        print("  python -m src.minivlm.processor --check")
        print("  必须全部 ✓ 才能进入训练阶段")
