"""
连接器：把视觉特征投影到语言模型的语义空间。

对应讲义 docs/03-connector.md，对应计划 Day 3。

三种实现，参数化可切换，方便你做对比实验：
  1. LinearConnector      —— 最原始，已淘汰
  2. MLPConnector         —— LLaVA / Qwen2.5-VL 方案，事实标准
  3. PerceiverResampler   —— BLIP-2 / Flamingo 方案，能压缩 token 数

核心结论（先记住，再验证）：
  参数量和效果不是正相关的，**输出 token 数才是效果的关键**。
  MLP 参数少但保真，Perceiver 参数多但会丢细节。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 线性投影（教学对照用）
# =============================================================================


class LinearConnector(nn.Module):
    """单层线性投影。MiniGPT-4 早期版本用的就是这个。

    问题：表达力不足，视觉和语言两个空间的对齐做不干净。
    你可以在 Day 3 亲自验证：同样的数据，它比 MLP 差一截。
    """

    def __init__(self, d_vision: int, d_llm: int):
        super().__init__()
        self.proj = nn.Linear(d_vision, d_llm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# 2. MLP 连接器（本项目默认）
# =============================================================================


class MLPConnector(nn.Module):
    """两层 MLP + GELU。LLaVA、Qwen2.5-VL、InternVL 都用这个思路。

    结构：Linear(d_v -> hidden) -> GELU -> Linear(hidden -> d_l)

    关于 hidden 维度的选择：
      LLaVA 用 d_v（视觉维度）作为 hidden，即「先不变维再加激活再降维」。
      也有实现用 d_l 或 4*d_v。差异不大，跟着你的基座源码走。

    重要：Qwen2.5-VL 官方实现里 **没有 LayerNorm**。
          你如果自作聪明加一层 LN，加载官方权重时 key 对不上。
          （Day 3 的检查任务：直接打印官方 state_dict 的 key 名对照。）
    """

    def __init__(self, d_vision: int, d_llm: int, hidden: int | None = None,
                 use_layernorm: bool = False):
        super().__init__()
        hidden = hidden or d_vision

        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(d_vision))
        layers += [
            nn.Linear(d_vision, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_llm),
        ]
        self.proj = nn.Sequential(*layers)
        self.d_vision = d_vision
        self.d_llm = d_llm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 3. Perceiver Resampler
# =============================================================================


class PerceiverAttention(nn.Module):
    """Perceiver 的注意力层。

    和普通自注意力的关键区别：
      - query 来自 **可学习的 latent**（长度固定 = n_query）
      - key / value 来自 **视觉特征**（长度可变 = N_patch）
      - 所以输出长度恒为 n_query，与输入图大小无关 ← 这就是压缩的来源
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # latents: (B, M, D)   x: (B, N, D)
        q = self.to_q(self.norm_q(latents))
        kv = self.to_kv(self.norm_kv(x))
        k, v = kv.chunk(2, dim=-1)

        B, M, _ = q.shape
        N = k.shape[1]
        q = q.reshape(B, M, self.heads, -1).transpose(1, 2)   # (B, H, M, Dh)
        k = k.reshape(B, N, self.heads, -1).transpose(1, 2)   # (B, H, N, Dh)
        v = v.reshape(B, N, self.heads, -1).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)          # (B, H, M, Dh)
        out = out.transpose(1, 2).reshape(B, M, -1)
        return self.to_out(out)


class PerceiverResamplerBlock(nn.Module):
    """一层 Perceiver block：cross-attn 抽信息 + FFN 加工。"""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = PerceiverAttention(dim, dim_head, heads)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, latents, x):
        latents = latents + self.attn(latents, x)
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class PerceiverResampler(nn.Module):
    """把任意长度的视觉特征压缩成 n_query 个 token。

    优点：输出长度可控 → 显存可控、延迟可控
    缺点：n_query 太小会丢细节。**对 OCR / 小字 / 瑕疵特写是致命的**

    实践建议：
      - 通用对话场景：n_query = 64 够用
      - 客服场景（要读尺码表、看瑕疵）：n_query >= 256，或者干脆用 MLP 不压缩
    """

    def __init__(self, d_vision: int, d_llm: int, n_query: int = 64,
                 n_layer: int = 2, n_head: int = 8):
        super().__init__()
        self.n_query = n_query
        self.proj_in = (nn.Linear(d_vision, d_llm) if d_vision != d_llm
                        else nn.Identity())
        self.latents = nn.Parameter(torch.randn(n_query, d_llm) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverResamplerBlock(d_llm, heads=n_head) for _ in range(n_layer)
        ])
        self.norm_out = nn.LayerNorm(d_llm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D_v)
        x = self.proj_in(x)                                     # (B, N, D_l)
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)   # (B, M, D_l)
        for layer in self.layers:
            latents = layer(latents, x)
        return self.norm_out(latents)                           # (B, M, D_l)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 4. 工厂 + 对比
# =============================================================================


def build_connector(kind: str, d_vision: int, d_llm: int, **kwargs) -> nn.Module:
    """按名字构建连接器。方便在配置里切换。"""
    kind = kind.lower()
    if kind == "linear":
        return LinearConnector(d_vision, d_llm)
    if kind == "mlp":
        return MLPConnector(d_vision, d_llm, **kwargs)
    if kind in ("perceiver", "resampler"):
        return PerceiverResampler(d_vision, d_llm, **kwargs)
    raise ValueError(f"未知的连接器类型: {kind}")


if __name__ == "__main__":
    torch.manual_seed(0)

    D_VISION = 1152       # SigLIP-SO400M 的隐藏维度
    D_LLM = 896           # Qwen2.5-0.5B 的隐藏维度

    print("=" * 78)
    print(f"连接器对比 (d_vision={D_VISION}, d_llm={D_LLM})")
    print("=" * 78)

    # 假设一张 448x448 的图 -> 1024 个 patch（未做 merging）
    N_PATCH = 1024
    x = torch.randn(2, N_PATCH, D_VISION)

    configs = [
        ("linear",            LinearConnector(D_VISION, D_LLM),                       1),
        ("mlp-2layer",        MLPConnector(D_VISION, D_LLM),                          2),
        ("mlp-ln",            MLPConnector(D_VISION, D_LLM, use_layernorm=True),      3),
        ("perceiver-64",      PerceiverResampler(D_VISION, D_LLM, n_query=64),        4),
        ("perceiver-256",     PerceiverResampler(D_VISION, D_LLM, n_query=256),       5),
    ]

    print(f"\n{'类型':<18} {'参数量':>12} {'输出tokens':>12} {'相对token':>10}")
    print("-" * 78)

    baseline_tokens = None
    for name, conn, _ in configs:
        conn.eval()
        with torch.no_grad():
            out = conn(x)
        n_tok = out.shape[1]
        if baseline_tokens is None:
            baseline_tokens = n_tok
        n_par = sum(p.numel() for p in conn.parameters())
        ratio = f"{n_tok / baseline_tokens:.0%}"
        print(f"{name:<18} {n_par:>12,} {n_tok:>12} {ratio:>10}")

    print()
    print("观察要点：")
    print("  1. MLP 的参数比 Perceiver 少一个数量级，但输出 token 多 16 倍")
    print("  2. Perceiver 的输出长度由 n_query 决定，与输入图大小无关")
    print("  3. 参数量 ≠ 效果。token 数（信息保真度）才是关键变量")
    print()
    print("对本项目的结论：")
    print("  客服要读尺码表、看瑕疵 → 不能压缩 → 用 MLP")
    print("  代价是 1024 个 visual token 进 LLM，显存和延迟都要算进去")
    print("  优化手段：降 max_pixels（第 4 周评测会给出拐点）")
