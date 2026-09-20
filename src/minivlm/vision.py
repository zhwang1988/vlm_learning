"""
视觉编码器：从零手写的教学版 ViT + 生产可用的 SigLIP 封装。

对应讲义 docs/02-vision-encoder.md，对应计划 Day 2。

设计意图：
  TinyViT 是为了「理解」——每一行都写得直白，方便你打断点、打印 shape。
  SiglipVisionWrapper 是为了「使用」——直接加载预训练权重，拿 patch 特征喂给连接器。

两种都用一遍，你就同时有了「原理」和「工程」两个视角。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 第一部分：从零手写的 TinyViT（教学用）
# =============================================================================


class PatchEmbedding(nn.Module):
    """把图像切成 patch 并投影成向量。

    关键洞察：这个操作等价于一个 kernel_size == stride == patch_size 的卷积。
    用 Conv2d 实现比手写 unfold + Linear 更快，也更贴近主流实现。

        (B, 3, H, W) -> (B, num_patches, embed_dim)

    自检：H=448, W=448, patch=14 时，num_patches = (448//14)^2 = 1024
    """

    def __init__(self, img_size: int = 448, patch_size: int = 14, in_chans: int = 3,
                 embed_dim: int = 1152):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # 卷积等价于「拉平 patch + 线性投影」
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.proj(x)                       # (B, D, grid, grid)
        x = x.flatten(2)                       # (B, D, N)
        x = x.transpose(1, 2)                  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力。

    注意 D 必须能被 n_head 整除。head_dim = D / n_head。
    """

    def __init__(self, dim: int, n_head: int = 16, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % n_head == 0, f"dim={dim} 必须能被 n_head={n_head} 整除"
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, N, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 用 F.scaled_dot_product_attention 走 FlashAttention 内核（如果可用）
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )                                       # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class MLPBlock(nn.Module):
    """Transformer 里的 FFN。ViT 通常 ratio=4。"""

    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class ViTBlock(nn.Module):
    """pre-norm 结构的 Transformer block。

    pre-norm（LN 在子层之前）比 post-norm 更容易训深，是现代标配。
    """

    def __init__(self, dim: int, n_head: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_head, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))       # 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """二维正弦余弦位置编码。

    和 1D 版本的区别：把 embed_dim 劈成两半，一半编码行坐标、一半编码列坐标。
    这样 (3,5) 和 (3,6) 的编码接近，(3,5) 和 (8,5) 的编码也能体现上下关系。

    返回: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0, "二维位置编码要求 embed_dim 能被 4 整除"
    half = embed_dim // 2

    def sincos_1d(pos, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)   # (N, dim/2)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (N, dim)

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    emb_h = sincos_1d(grid[0].reshape(-1), half)     # (N, half)
    emb_w = sincos_1d(grid[1].reshape(-1), half)     # (N, half)
    return torch.cat([emb_h, emb_w], dim=1)          # (N, D)


class TinyViT(nn.Module):
    """一个能跑通、能读懂的最小 ViT。参数量远小于真实模型，但结构完全一致。

    真实 SigLIP-SO400M 是 27 层、1152 维、约 878M 参数；
    这里默认 12 层、384 维，方便在 CPU 上秒级跑完，看清每一步 shape。
    """

    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        n_head: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # 可学习的 [CLS] token（timm 的 SigLIP 用这个做全局特征）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_head, mlp_ratio, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 位置编码用固定的二维 sincos 初始化（不训练）
        pe = build_2d_sincos_pos_embed(self.patch_embed.proj.out_channels,
                                       self.patch_embed.grid_size)
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, return_cls: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)                                  # (B, N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)                                         # (B, N, D)
        if return_cls:
            return x.mean(dim=1)                                 # 简易池化
        return x


# =============================================================================
# 第二部分：生产用 SigLIP 封装
# =============================================================================


class SiglipVisionWrapper(nn.Module):
    """加载 HF 的 SigLIP 视觉塔，输出 patch 级特征给连接器使用。

    为什么用 SigLIP 而不是 CLIP：
      - sigmoid loss 不依赖全局 batch 归一化，训练更稳
      - 同等规模下细粒度略好
      - Qwen2.5-VL 的视觉塔就是 SigLIP 风格的

    注意：这里默认冻结全部参数（frozen=True）。后训练阶段解冻视觉塔
    在小数据集上极易造成灾难性遗忘，见 docs/01-architecture.md。
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384",
                 frozen: bool = True, dtype: torch.dtype = torch.bfloat16,
                 gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import SiglipVisionModel, SiglipVisionConfig

        # 关键：把 _attn_implementation 设成 sdpa 以启用 FlashAttention 路径
        cfg = SiglipVisionConfig.from_pretrained(model_name)
        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name, config=cfg, dtype=dtype,
            attn_implementation="sdpa",
        )
        self.hidden_size = self.vision_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size

        if gradient_checkpointing:
            self.vision_model.gradient_checkpointing_enable()

        if frozen:
            self.vision_model.requires_grad_(False)

    @property
    def dtype(self):
        return next(self.vision_model.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_model.parameters()).device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, H, W)，已按 SigLIP 的 mean/std 归一化
        返回: (B, N_patch, D_vision)  —— 已去掉 [CLS]
        """
        pixel_values = pixel_values.to(self.dtype)
        out = self.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        # last_hidden_state: (B, 1 + N_patch, D)，第 0 位是 CLS
        return out.last_hidden_state[:, 1:, :]


# =============================================================================
# 第三部分：工具函数
# =============================================================================


def count_patches(h: int, w: int, patch_size: int = 14) -> int:
    """给定高宽和 patch size，算 patch 数（不含 CLS）。"""
    return (h // patch_size) * (w // patch_size)


def estimate_visual_tokens(h: int, w: int, patch_size: int = 14,
                          merge_size: int = 2, max_pixels: Optional[int] = None
                          ) -> tuple[int, tuple[int, int]]:
    """估算一张图在 Qwen2.5-VL 里会消耗多少 visual token。

    这是 Day 4 的核心练习。简化版的逻辑（不含 min_pixels 放大部分）：
      1. 按 max_pixels 上限等比缩放
      2. 对齐到 patch_size 的整数倍
      3. 除以 merge_size 的平方

    返回 (token 数, (grid_h, grid_w))
    """
    h_bar = round(h / patch_size) * patch_size
    w_bar = round(w / patch_size) * patch_size

    if max_pixels is not None and h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(patch_size, math.floor(h / beta / patch_size)) * patch_size
        w_bar = max(patch_size, math.floor(w / beta / patch_size)) * patch_size

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size

    # 2x2 merging 要求网格是偶数
    grid_h = grid_h - grid_h % merge_size
    grid_w = grid_w - grid_w % merge_size

    n_tokens = (grid_h // merge_size) * (grid_w // merge_size)
    return n_tokens, (grid_h, grid_w)


if __name__ == "__main__":
    # 快速自检：跑通并打印 shape
    print("=" * 60)
    print("TinyViT 形状自检")
    print("=" * 60)

    model = TinyViT(img_size=448, patch_size=14, embed_dim=384, depth=4, n_head=6)
    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        out = model(x)

    expected_n = (448 // 14) ** 2
    print(f"输入:        {tuple(x.shape)}")
    print(f"输出:        {tuple(out.shape)}")
    print(f"期望 patch 数: {expected_n}  (448//14 = {448 // 14}, 平方)")
    assert out.shape[1] == expected_n, "patch 数不对！检查 PatchEmbedding"
    print("✓ patch 数正确")

    print()
    print("=" * 60)
    print("visual token 估算表")
    print("=" * 60)
    MAX_PIXELS = 1280 * 28 * 28
    print(f"max_pixels = {MAX_PIXELS}\n")
    print(f"{'原图尺寸':>16} | {'token数':>8} | {'grid':>12}")
    print("-" * 44)
    for h, w in [(224, 224), (448, 448), (800, 600), (1024, 1024),
                 (1920, 1080), (1080, 1920), (4000, 3000)]:
        n, grid = estimate_visual_tokens(h, w, max_pixels=MAX_PIXELS)
        print(f"{f'{h}x{w}':>16} | {n:>8} | {f'{grid[0]}x{grid[1]}':>12}")
    print()
    print("观察要点：")
    print("  - 小图 token 少（便宜），大图被缩到上限附近")
    print("  - 1920x1080 和 1080x1920 的 token 数接近，但 grid 形状不同")
    print("    → 说明比例被保留了，这正是 native dynamic resolution 的意义")
