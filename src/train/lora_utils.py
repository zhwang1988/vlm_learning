"""
LoRA 原理实现与配置工具。

对应讲义 docs/06-sft-training.md 第 4 节，对应计划 Day 14。

第一部分是**手写的 LoRA 层**（20 行），用来真正理解 ΔW = BA 在做什么。
第二部分是 target_modules 的选取逻辑，这是 VLM 训 LoRA 最容易搞错的地方。
第三部分是参数量/显存估算，用来决定配置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# =============================================================================
# 1. 手写 LoRA 层（教学用，20 行）
# =============================================================================


class LoRALinear(nn.Module):
    """把 nn.Linear 包一层低秩旁路。

        h = W x + (alpha / r) * B(A x)

    两个关键设计：
      - **A 用随机初始化，B 初始化为全 0** → 训练开始时 ΔW = 0，
        模型行为与基座完全一致，不会一上来就破坏已学知识。这是 LoRA 稳的关键。
      - **forward 里 W 和 A/B 是并列的** → W 不参与梯度，只有 A/B 更新。

    参数：
        r       低秩维度。8-64。领域差距大就大一点，16 是安全默认。
        alpha   缩放因子。通常 = 2r。只调 alpha 不改 r ≈ 调学习率。
        dropout LoRA 层自己的 dropout，不是原层的。小数据(＜5k)用 0.1。
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        for p in self.base.parameters():
            p.requires_grad = False          # ← 冻结原权重

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))   # ← 零初始化！
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A 用 kaiming，和原论文一致
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """把 LoRA 权重合并回原层，供推理使用（无额外延迟）。"""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None,
                           dtype=self.base.weight.dtype,
                           device=self.base.weight.device)
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged

    @property
    def n_trainable(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


# =============================================================================
# 2. target_modules 选取（VLM 特有，最容易搞错）
# =============================================================================

# 保守：只加注意力投影
TARGET_ATTN_ONLY = ["q_proj", "v_proj"]

# 标准：注意力 + MLP（推荐）
TARGET_STANDARD = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 连接器模块名（Qwen2.5-VL 里是 merger.mlp）
# ⭐ VLM 的独门技巧：把 LoRA 也加在连接器上，
#    让模型同时调整「怎么看」和「怎么说」。
#    通常有额外收益，且几乎不增加参数。
TARGET_CONNECTOR = ["merger.mlp.0", "merger.mlp.2"]

# ❌ 绝对不要加在这里（除非有 50k+ 数据）
#    "visual.*" —— 视觉塔。小数据下解冻会破坏通用视觉能力，
#    表现是 OCR 变差、颜色判断偏、细粒度感知退化。
FORBIDDEN = ["visual"]


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    use_standard_targets: bool = True
    include_connector: bool = True       # ⭐ 推荐开启
    include_vision: bool = False         # ⚠️ 默认关闭
    learning_rate: float = 1e-4


def resolve_target_modules(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """根据模型实际的模块名，解析出要注入的 target_modules。

    **必须这样做**，而不是硬编码 ["q_proj", ...]。
    原因：不同模型的命名不一样，而且加了连接器之后名字更特殊。
    先扫一遍模型里所有 Linear 层的名字，再按规则匹配，最稳。
    """
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(name)

    targets = TARGET_STANDARD if cfg.use_standard_targets else TARGET_ATTN_ONLY

    # 匹配：最后一个点号后的名字在前缀集合里
    selected = set()
    for n in names:
        leaf = n.split(".")[-1]
        if leaf in targets:
            selected.add(n)

    if cfg.include_connector:
        # 连接器模块名可能是 merger.mlp.0 / merger.mlp.2
        for n in names:
            if "merger" in n or "connector" in n or "projector" in n:
                selected.add(n)

    if cfg.include_vision:
        for n in names:
            if any(k in n for k in ("visual", "vision_tower")):
                leaf = n.split(".")[-1]
                if leaf in targets:
                    selected.add(n)

    return sorted(selected)


def print_lora_plan(model: nn.Module, cfg: LoRAConfig):
    """打印 LoRA 注入计划，供人工确认。Day 14 的验收动作之一。"""
    targets = resolve_target_modules(model, cfg)
    print("=" * 78)
    print(f"LoRA 注入计划  (r={cfg.r}, alpha={cfg.alpha}, "
          f"scaling={cfg.alpha / cfg.r:.1f}, dropout={cfg.dropout})")
    print("=" * 78)

    by_group: dict[str, list[str]] = {}
    for t in targets:
        if "visual" in t or "vision_tower" in t:
            g = "视觉塔"
        elif "merger" in t or "connector" in t or "projector" in t:
            g = "连接器 ⭐"
        elif any(k in t for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            g = "注意力投影"
        else:
            g = "MLP"

        by_group.setdefault(g, []).append(t)

    total_trainable = 0
    for g, items in by_group.items():
        # 统计这一组的参数量
        n = 0
        for name in items:
            mod = dict(model.named_modules())[name]
            if isinstance(mod, nn.Linear):
                n += cfg.r * (mod.in_features + mod.out_features)
        total_trainable += n
        print(f"\n[{g}]  {len(items)} 个模块，可训练参数 {n:,}")
        for it in items[:6]:
            print(f"    {it}")
        if len(items) > 6:
            print(f"    ... 还有 {len(items) - 6} 个")

    total = sum(p.numel() for p in model.parameters())
    print()
    print("-" * 78)
    print(f"可训练参数  {total_trainable:>15,}")
    print(f"总参数      {total:>15,}")
    print(f"可训练比例  {total_trainable / total:>14.3%}")
    print("-" * 78)
    if cfg.include_vision:
        print("⚠️  已开启视觉塔 LoRA。小数据集上这会破坏通用视觉能力，")
        print("    建议只在数据 >50k 且确认 OCR/细粒度没退化时才用。")
    else:
        print("✓ 视觉塔保持冻结（推荐）")
    return targets


# =============================================================================
# 3. 参数量与显存估算（Day 13 的核心）
# =============================================================================


BYTES_PER_PARAM = {
    "full_bf16_adamw": 12,      # 权重2 + 梯度2 + AdamW状态8
    "full_bf16_sgd": 6,         # 权重2 + 梯度2 + SGD动量2
    "lora_bf16": 2,             # 只有 base 权重（可训练部分极少，忽略）
    "qlora_4bit": 0.5,          # 4bit 量化
}


def estimate_training_vram(
    n_params: float,
    mode: str = "lora_bf16",
    n_trainable_ratio: float = 0.005,
    seq_len: int = 4096,
    batch_size: int = 1,
    hidden_size: int = 2048,
    n_layers: int = 36,
    gradient_checkpointing: bool = True,
    n_images_per_sample: int = 1,
    visual_tokens: int = 1000,
) -> dict:
    """估算训练显存（GB）。

    这是 Day 13 的核心练习。公式不是精确值，但能帮你**判断量级**：
    是 8G、16G 还是 80G 的问题，决定你用不用得起这张卡。

    显存 = 静态部分（权重+梯度+优化器） + 动态部分（激活值）

    动态部分的估算（关键在 seq_len 和 hidden）：
      无 gc:  激活 ≈ batch × seq × hidden × layers × ~20  [字节]
      有 gc:  ≈ 1/3（前向重算，但还要存层边界）
    """
    static_gb = n_params * BYTES_PER_PARAM[mode] / (1024 ** 3)

    # LoRA：可训练部分自己的优化器状态（很小）
    if mode in ("lora_bf16", "qlora_4bit"):
        n_train = n_params * n_trainable_ratio
        static_gb += n_train * 10 / (1024 ** 3)   # bf16 m+v ≈ 8~10 字节

    # 动态：激活值
    # 序列长度 = 文本 + 视觉 token
    L = seq_len + visual_tokens * n_images_per_sample
    factor = 20 if not gradient_checkpointing else 7
    act_bytes = batch_size * L * hidden_size * n_layers * factor
    act_gb = act_bytes / (1024 ** 3)

    # CUDA 上下文 + 碎片 ≈ 1.5 GB
    overhead = 1.5

    return {
        "static_gb": round(static_gb, 2),
        "activation_gb": round(act_gb, 2),
        "overhead_gb": overhead,
        "total_gb": round(static_gb + act_gb + overhead, 2),
        "seq_len_effective": L,
        "gradient_checkpointing": gradient_checkpointing,
    }


QWEN25VL_SPECS = {
    "3B": {"n_params": 3.75e9, "hidden": 2048, "layers": 36},
    "7B": {"n_params": 8.29e9, "hidden": 3584, "layers": 28},
}


def print_vram_table():
    """打印显存需求对照表。Day 13 的交付物。"""
    print("=" * 92)
    print("Qwen2.5-VL 训练显存估算（GB）")
    print("=" * 92)
    print("假设：单图 1000 visual token，gradient_checkpointing 开启\n")

    header = f"{'模型':<6} {'方案':<14} {'seq':<6} {'batch':<6} {'静态':>8} {'激活':>8} {'总计':>8} {'16G卡':>8}"
    print(header)
    print("-" * 92)

    for model_name, spec in QWEN25VL_SPECS.items():
        for mode_label, mode, ratio in [
            ("全参 bf16", "full_bf16_adamw", 1.0),
            ("LoRA r=16", "lora_bf16", 0.005),
            ("QLoRA r=16", "qlora_4bit", 0.005),
        ]:
            for seq in (2048, 4096):
                r = estimate_training_vram(
                    n_params=spec["n_params"], mode=mode,
                    n_trainable_ratio=ratio, seq_len=seq,
                    batch_size=1, hidden_size=spec["hidden"],
                    n_layers=spec["layers"], gradient_checkpointing=True,
                )
                ok = "✓" if r["total_gb"] < 15 else ("⚠" if r["total_gb"] < 17 else "✗")
                print(f"{model_name:<6} {mode_label:<14} {seq:<6} {1:<6} "
                      f"{r['static_gb']:>8.1f} {r['activation_gb']:>8.1f} "
                      f"{r['total_gb']:>8.1f} {ok:>8}")
        print()

    print("-" * 92)
    print("结论（针对「无本地卡，走 Colab/云」的条件）：")
    print("  ✓ 3B + QLoRA  → Colab T4 (16G) 可跑")
    print("  ✓ 3B + LoRA   → Colab T4 勉强，A100 (40G) 轻松")
    print("  ✓ 7B + QLoRA  → Colab A100 (40G) 可跑")
    print("  ✗ 3B 全参     → 需要 80G")
    print()
    print("排序建议：从 3B QLoRA 开始跑通，再往上试。")
    print("不要一上来就追求全参 —— 数据质量的收益远大于参数量。")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="打印显存对照表")
    ap.add_argument("--demo", action="store_true", help="跑 LoRA 层和注入计划的演示")
    args = ap.parse_args()

    if args.table:
        print_vram_table()
    elif args.demo:
        print("=" * 78)
        print("手写 LoRA 层演示")
        print("=" * 78)
        base = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32)
        lora = LoRALinear(base, r=2, alpha=4, dropout=0.0)

        x = torch.randn(2, 8)
        with torch.no_grad():
            out_before = lora(x)
            out_base = base(x)
        print(f"\nbase 权重 shape:        {tuple(base.weight.shape)}")
        print(f"lora_A shape:           {tuple(lora.lora_A.shape)}   (r, d_in)")
        print(f"lora_B shape:           {tuple(lora.lora_B.shape)}   (d_out, r)")
        print(f"可训练参数:             {lora.n_trainable}  "
              f"(vs base {base.weight.numel()})")
        print(f"缩放系数 alpha/r:       {lora.scaling}")
        print()
        print(f"B 初始全 0，所以初始时 LoRA 输出 == base 输出:")
        print(f"  max|out - base_out| = {(out_before - out_base).abs().max().item():.2e}")
        assert torch.allclose(out_before, out_base, atol=1e-6), "初始状态必须与 base 一致！"
        print("  ✓ 通过（这是 LoRA 稳定的关键）")

        # 手动改 B，看输出变化
        with torch.no_grad():
            lora.lora_B.fill_(0.5)
        with torch.no_grad():
            out_after = lora(x)
        print(f"\n把 B 全设为 0.5 后:")
        print(f"  max|out_change| = {(out_after - out_before).abs().max().item():.4f}")
        print(f"  预期 = |scaling * 0.5 * sum(lora_A dim0)| 量级")
        print("  → ΔW = B@A 确实改变了输出，这就是可学习的地方")

        # merge 验证
        merged = lora.merge()
        with torch.no_grad():
            out_merged = merged(x)
        print(f"\nmerge 后 forward:")
        print(f"  max|merged - lora| = {(out_merged - out_after).abs().max().item():.2e}")
        assert torch.allclose(out_merged, out_after, atol=1e-4), "merge 应该等价"
        print("  ✓ merge 等价，推理时用它就没有额外延迟")
    else:
        print("用法：")
        print("  python -m src.train.lora_utils --table   # 显存对照表（Day 13）")
        print("  python -m src.train.lora_utils --demo    # LoRA 原理演示（Day 14）")
        print()
        print_vram_table()
