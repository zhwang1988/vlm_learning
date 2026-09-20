"""
图像预处理工具。

对应讲义 docs/05-data-engineering.md 第 1–3 节，对应计划 Day 7。

电商图片比学术数据集脏得多。这个模块处理真实的五类脏数据：
  1. EXIF 旋转（手机拍的图躺着）
  2. 透明通道（PNG 转 RGB 变黑底）
  3. CMYK / 16bit（印刷来源）
  4. 超长图（详情页 1:20 比例）
  5. 拼接图（一张图拼了多个款式）

以及一个和训练直接相关的功能：pHash 去重。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageOps


# =============================================================================
# 1. 清洗与规范化
# =============================================================================


@dataclass
class ImageStats:
    """一张图处理前后的信息，用于出质量报告。"""
    path: str = ""
    original_size: tuple[int, int] = (0, 0)
    final_size: tuple[int, int] = (0, 0)
    original_mode: str = ""
    had_exif_rotation: bool = False
    had_alpha: bool = False
    was_cmyk: bool = False
    aspect_ratio: float = 1.0
    issues: list[str] = field(default_factory=list)


def load_and_normalize(
    src: str | Path | bytes | Image.Image,
    max_aspect_ratio: float = 5.0,
    force_white_bg: bool = True,
) -> tuple[Image.Image, ImageStats]:
    """把任意来源的图变成一张「干净」的 RGB 图。

    处理顺序很重要：
      EXIF 旋转 → 模式转换 → 透明合成 → 比例检查

    如果顺序错了（比如先转 RGB 再处理 EXIF），会得到错误的像素。
    """
    stats = ImageStats()

    # --- 读入 ---
    if isinstance(src, Image.Image):
        img = src
        stats.path = "<PIL>"
    elif isinstance(src, bytes):
        img = Image.open(io.BytesIO(src))
        stats.path = "<bytes>"
    else:
        img = Image.open(src)
        stats.path = str(src)

    img.load()
    stats.original_size = img.size
    stats.original_mode = img.mode

    # --- ① EXIF 旋转：必须最先做 ---
    # 手机拍的竖图，像素其实是横的，方向信息存在 EXIF 里。
    # 不处理的话模型看到的是躺着的图。
    try:
        before = img.size
        img = ImageOps.exif_transpose(img)
        if img.size != before:
            stats.had_exif_rotation = True
    except Exception:
        pass   # 没有 EXIF 或损坏，忽略

    # --- ② 模式转换 ---
    if img.mode == "CMYK":
        stats.was_cmyk = True
        img = img.convert("RGB")
    elif img.mode in ("RGBA", "LA", "PA"):
        stats.had_alpha = True
        if force_white_bg:
            # 关键：透明区域要合成到白底，不能直接 convert("RGB")
            # 直接 convert 会把透明区域变成黑色，白色商品图会变黑底
            bg = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.convert("RGBA").split()[-1]
            bg.paste(img.convert("RGBA"), mask=alpha)
            img = bg
        else:
            img = img.convert("RGB")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    # --- ③ 16bit 降位 ---
    if img.mode == "I;16" or (hasattr(img, "mode") and "16" in str(img.mode)):
        arr = np.array(img)
        arr = (arr / 256).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    # --- ④ 极端比例检查 ---
    w, h = img.size
    stats.aspect_ratio = max(w, h) / max(min(w, h), 1)
    if stats.aspect_ratio > max_aspect_ratio:
        stats.issues.append(f"extreme_aspect_{stats.aspect_ratio:.1f}")

    # --- ⑤ 极小图检查 ---
    if min(w, h) < 56:
        stats.issues.append(f"too_small_{w}x{h}")

    stats.final_size = img.size
    return img, stats


# =============================================================================
# 2. pHash 去重
# =============================================================================


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """感知哈希。返回一个 64 bit 整数。

    算法：
      1. 缩到 32x32 灰度（缩放本身就有抗噪作用）
      2. DCT 变换
      3. 取左上 hash_size x hash_size 低频块
      4. 每个系数和中位数比较，大于记 1，否则记 0

    为什么用 DCT 低频：低频保留了图像的结构信息，高频是噪声。
    这样「同一张图不同压缩质量」「裁掉一点边」都能被识别为重复，
    而「两张不同的白底商品图」不会被误判。
    """
    from scipy.fftpack import dct

    img = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)

    dct_out = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = dct_out[:hash_size, :hash_size]

    # 排除 DC 分量（左上角）再取中位数——DC 分量是整体亮度，对判断没帮助
    med = np.median(low.flatten()[1:])

    bits = (low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """两个 pHash 的汉明距离（不同 bit 的个数）。"""
    return bin(a ^ b).count("1")


def find_duplicate_groups(
    images: Iterable[Image.Image],
    threshold: int = 5,
    ids: Optional[list] = None,
) -> list[list[int]]:
    """找出重复图片的组。返回 [[idx, idx, ...], ...]。

    threshold=5 是经验值：
      - < 5：几乎肯定是同一张图（不同压缩/轻微裁剪）
      - 5-10：可能相似但不一定重复
      - > 10：基本是不同图
    """
    hashes = [phash(im) for im in images]
    ids = ids or list(range(len(hashes)))
    n = len(hashes)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) < threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return [g for g in groups.values() if len(g) > 1]


# =============================================================================
# 3. 拼接图检测（电商特有）
# =============================================================================


def detect_collage(img: Image.Image) -> tuple[bool, list[int]]:
    """粗检测「一张图拼了多个款式」的情况。

    方法：找图里明显的大块同色区域边界（简化版：用边缘密度的方差）。
    真正生产环境应该用目标检测，这里只做启发式提示。

    为什么重要：拼接图会让模型「只看一个」，导致训练信号噪声很大。
    这类样本要么裁开，要么在数据里打标记。
    """
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)

    # 行列方向的梯度
    gx = np.abs(np.diff(g, axis=1)).mean(axis=0)
    gy = np.abs(np.diff(g, axis=0)).mean(axis=1)

    # 找突出的「分割线」（梯度均值显著高于整体）
    def find_peaks(v, k=3.0):
        mu, sd = v.mean(), v.std()
        idx = np.where(v > mu + k * sd)[0]
        return idx.tolist()

    cols = find_peaks(gx)
    rows = find_peaks(gy)
    lines = sorted(cols + rows)

    return (len(lines) >= 2), lines


# =============================================================================
# 4. 批处理
# =============================================================================


def scan_directory(root: str | Path, exts=(".jpg", ".jpeg", ".png", ".webp", ".bmp")) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )


def file_md5(path: Path) -> str:
    """字节级去重用的哈希。比 pHash 便宜，作为第一道过滤。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys

    print("=" * 76)
    print("图像预处理自检")
    print("=" * 76)

    # --- 测试 1：透明通道处理 ---
    rgba = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    for x in range(100, 200):
        for y in range(50, 150):
            rgba.putpixel((x, y), (200, 30, 30, 255))

    wrong = rgba.convert("RGB")
    right, st = load_and_normalize(rgba)

    print("\n[测试 1] 透明通道处理")
    print(f"  原图 mode:          {rgba.mode}")
    print(f"  直接 convert('RGB') 的左上角像素: {wrong.getpixel((5, 5))}  ← 变黑了，错误")
    print(f"  正确处理的左上角像素:              {right.getpixel((5, 5))}  ← 白底，正确")
    print(f"  had_alpha 标记:     {st.had_alpha}")
    assert right.getpixel((5, 5)) == (255, 255, 255), "透明区域应该合成到白底"
    print("  ✓ 通过")

    # --- 测试 2：pHash 抗干扰能力 ---
    print("\n[测试 2] pHash 抗干扰能力")
    base = Image.new("RGB", (400, 400), (240, 240, 240))
    for x in range(50, 350):
        for y in range(100, 300):
            base.putpixel((x, y), (30, 60, 150))
    for x in range(200, 260):
        for y in range(150, 250):
            base.putpixel((x, y), (220, 200, 60))

    # 变体 A：JPEG 压缩
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=45)
    variant_a = Image.open(io.BytesIO(buf.getvalue()))

    # 变体 B：裁掉一点边
    variant_b = base.crop((8, 8, 392, 392))

    # 变体 C：完全不同的图
    different = Image.new("RGB", (400, 400), (10, 10, 10))
    for x in range(0, 400, 40):
        for y in range(400):
            different.putpixel((x, y), (250, 250, 250))

    h_base = phash(base)
    d_a = hamming_distance(h_base, phash(variant_a))
    d_b = hamming_distance(h_base, phash(variant_b))
    d_c = hamming_distance(h_base, phash(different))

    print(f"  vs JPEG 压缩(quality=45):  距离 {d_a:>2}  → {'判为重复' if d_a < 5 else '判为不同'}")
    print(f"  vs 裁掉 8px 边:            距离 {d_b:>2}  → {'判为重复' if d_b < 5 else '判为不同'}")
    print(f"  vs 完全不同的图:           距离 {d_c:>2}  → {'判为重复' if d_c < 5 else '判为不同'}")
    assert d_a < 5, "JPEG 压缩后应该仍被判为重复"
    assert d_b < 5, "轻微裁剪后应该仍被判为重复"
    assert d_c > 10, "不同的图不应该被判为重复"
    print("  ✓ 通过：抗压缩、抗裁剪，但不误判")

    # --- 测试 3：去重分组 ---
    print("\n[测试 3] 去重分组")
    imgs = [base, variant_a, variant_b, different]
    groups = find_duplicate_groups(imgs, threshold=5)
    print(f"  输入 4 张（3 张同源 + 1 张不同）")
    print(f"  识别出重复组: {groups}")
    assert len(groups) == 1 and len(groups[0]) == 3, "应该识别出 1 组、含 3 张"
    print("  ✓ 通过")

    # --- 测试 4：整体流程 ---
    print("\n[测试 4] 全流程（用真实文件）")
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        files = scan_directory(root)
        print(f"  扫描到 {len(files)} 张图")
        stats_list = []
        for f in files[:200]:
            try:
                img, st = load_and_normalize(f)
                stats_list.append(st)
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
        issues = {}
        for st in stats_list:
            for iss in st.issues:
                key = iss.split("_")[0]
                issues[key] = issues.get(key, 0) + 1
        print(f"  处理成功 {len(stats_list)} 张")
        print(f"  问题统计: {issues or '无'}")
        exif = sum(1 for s in stats_list if s.had_exif_rotation)
        alpha = sum(1 for s in stats_list if s.had_alpha)
        cmyk = sum(1 for s in stats_list if s.was_cmyk)
        print(f"  EXIF 旋转: {exif}  透明通道: {alpha}  CMYK: {cmyk}")
        print("  → 这几个数字就是你 Day 12 数据质量报告里的『脏数据统计』")
    else:
        print("  （传入一个图片目录可以跑真实数据：python -m src.data.image_utils /path/to/images）")

    print("\n" + "=" * 76)
    print("全部自检通过")
