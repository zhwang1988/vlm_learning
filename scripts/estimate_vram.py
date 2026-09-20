"""
显存估算 —— 下单租卡之前先算一遍，比 OOM 之后再换卡便宜得多。

    python scripts/estimate_vram.py --model 3b --method qlora
    python scripts/estimate_vram.py --model 7b --method lora --seq 2048 --bs 2
    python scripts/estimate_vram.py --all                  # 全矩阵对比
    python scripts/estimate_vram.py --model 3b --method qlora --suggest
    python scripts/estimate_vram.py --model 3b --infer --kv-len 8192

原理（对应讲义 docs/06-sft-training.md 第二节）：

  总显存 ≈ 权重 + 梯度 + 优化器状态 + 激活值 + 视觉塔开销 + logits + 碎片

  关键系数（每参数字节数）：
    bf16 权重                  = 2
    梯度                       = 2
    AdamW 状态（m + v, fp32）   = 8
    AMP 的 fp32 master weight   = 4
    → 全参 bf16 + AdamW = 2+2+8+4 = 16 bytes/param
      这就是「全参 3B 要 56 GB」的来源，和模型大小成正比，和卡无关。

  多模态的特殊之处：
    视觉 token 极贵。1024x1024 图 = (1024/14)^2 = 5355 个 patch，
    经 2x2 merge 后 ≈ 1332 个 token 进入 LLM。
    一张图 ≈ 1300 字的 prompt，所以序列长度会被图片顶爆。
    → 调小 max_pixels 比调小 batch_size 更有效

  ⚠️ 解析估算天然偏乐观（通常低 20–50%），因为真实框架还有：
     输入 padding、dataloader 的 pinned buffer、flash-attn 的 workspace、
     HF Trainer 的 logits fp32 副本、以及「传了图但没关梯度」的视觉塔。
     脚本同时打印「实测经验区间」，以那个为准来选卡。
"""

from __future__ import annotations

import argparse
import math
import sys

# ---------------------------------------------------------------------------
# 模型档案
# ---------------------------------------------------------------------------

MODELS = {
    "0.5b": dict(
        name="Qwen2.5-VL-0.5B（教学用）",
        params_llm=0.49e9, params_vision=0.15e9,
        layers=24, hidden=896, n_heads=14, kv_heads=2, head_dim=64,
        intermediate=4864, vocab=151936,
        vis_layers=24, vis_hidden=896, vis_patch=14, vis_merge=2,
    ),
    "3b": dict(
        name="Qwen2.5-VL-3B-Instruct",
        params_llm=3.09e9, params_vision=0.44e9,
        layers=36, hidden=2048, n_heads=16, kv_heads=4, head_dim=128,
        intermediate=11008, vocab=152064,
        vis_layers=32, vis_hidden=1280, vis_patch=14, vis_merge=2,
    ),
    "7b": dict(
        name="Qwen2.5-VL-7B-Instruct",
        params_llm=7.07e9, params_vision=0.67e9,
        layers=28, hidden=3584, n_heads=28, kv_heads=4, head_dim=128,
        intermediate=18944, vocab=152064,
        vis_layers=32, vis_hidden=1280, vis_patch=14, vis_merge=2,
    ),
    "32b": dict(
        name="Qwen2.5-VL-32B-Instruct",
        params_llm=31.2e9, params_vision=0.67e9,
        layers=64, hidden=5120, n_heads=40, kv_heads=8, head_dim=128,
        intermediate=27648, vocab=152064,
        vis_layers=32, vis_hidden=1280, vis_patch=14, vis_merge=2,
    ),
}

GPUS = [
    ("RTX 3080Ti",   12, False),
    ("T4",           16, False),
    ("RTX 3090",     24, True),
    ("RTX 4090",     24, True),
    ("V100-32G",     32, False),
    ("RTX 5090",     32, True),
    ("L40S",         48, True),
    ("A800-80G",     80, True),
    ("H800-80G",     80, True),
    ("H20-96G",      96, True),
]

BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "8bit": 1, "4bit": 0.5}

# 实测经验区间（GB）—— 以这个为准。seq≈2048, batch=1, 梯度检查点开, 1 张 1024² 图
EMPIRICAL = {
    ("0.5b", "qlora"):  "4 – 6",
    ("0.5b", "lora"):   "6 – 9",
    ("0.5b", "full"):   "12 – 16",
    ("0.5b", "infer"):  "1.5 – 3",
    ("3b",   "qlora"):  "8 – 12",
    ("3b",   "lora"):   "16 – 20",
    ("3b",   "full"):   "48 – 60",
    ("3b",   "infer"):  "8 – 10",
    ("7b",   "qlora"):  "16 – 22",
    ("7b",   "lora"):   "30 – 38",
    ("7b",   "full"):   "100 – 130",
    ("7b",   "infer"):  "15 – 18",
    ("32b",  "qlora"):  "45 – 60",
    ("32b",  "lora"):   "80 – 100",
    ("32b",  "full"):   "420 – 520",
    ("32b",  "infer"):  "62 – 70",
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def visual_tokens(h, w, patch=14, merge=2):
    """一张图会产生多少个送进 LLM 的 token。"""
    ph, pw = math.ceil(h / patch), math.ceil(w / patch)
    merged = math.ceil(ph / merge) * math.ceil(pw / merge)
    return max(merged, 4)


def _lora_params(m: dict, r: int = 16) -> float:
    """LoRA 可训练参数量（只算常规 target + 连接器）。

    对每个被 target 的 Linear(w_out, w_in)，LoRA 加 r*(w_in) + r*(w_out) 个参数。
    """
    h, inter = m["hidden"], m["intermediate"]
    n_h, n_kv, hd = m["n_heads"], m["kv_heads"], m["head_dim"]
    per_layer = (
        r * (h + n_h * hd) +        # q_proj
        r * (h + n_kv * hd) +       # k_proj
        r * (h + n_kv * hd) +       # v_proj
        r * (n_h * hd + h) +        # o_proj
        r * (h + inter) +           # gate_proj
        r * (h + inter) +           # up_proj
        r * (inter + h)             # down_proj
    )
    total = per_layer * m["layers"]
    # 连接器 merger.mlp: Linear(vis_hidden*merge^2 -> hidden) + Linear(hidden -> hidden)
    conn_in = m["vis_hidden"] * m["vis_merge"] ** 2
    total += r * (conn_in + h) + r * (h + h)
    return total


# ---------------------------------------------------------------------------
# 核心估算
# ---------------------------------------------------------------------------

def estimate(
    model_key: str,
    method: str = "qlora",          # full | lora | qlora | infer
    seq_len: int = 2048,
    batch: int = 1,
    n_images: int = 1,
    image_size: int = 1024,
    grad_checkpointing: bool = True,
    optimizer: str = "adamw",
    train_vision: bool = False,
    zero_stage: int = 0,
    master_weights: bool = True,
    flash_attn: bool = True,
    lora_r: int = 16,
) -> dict:
    m = MODELS[model_key]
    p_llm, p_vis = m["params_llm"], m["params_vision"]
    L, H, HID = m["layers"], m["hidden"], m["hidden"]

    vis_tok = visual_tokens(image_size, image_size, m["vis_patch"], m["vis_merge"]) * n_images
    out = {
        "model": m["name"], "method": method, "seq_len": seq_len, "batch": batch,
        "n_images": n_images, "grad_checkpointing": grad_checkpointing,
        "zero_stage": zero_stage, "flash_attn": flash_attn,
        "visual_tokens": vis_tok,
    }

    # ============================== 推理 ==============================
    if method == "infer":
        wb = BYTES["bf16"]
        weights = (p_llm + p_vis) * wb
        # KV cache = 2(K,V) * layers * kv_heads * head_dim * seq * batch * bytes
        kv = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * seq_len * batch * wb
        # prefill 峰值激活（粗估）
        act = 12 * seq_len * HID * batch * wb
        cuda_ctx = 0.5 * 1024 ** 3
        # vLLM 的 PagedAttention 会有约 4% 的 KV 碎片
        total = (weights + kv * 1.04 + act + cuda_ctx)
        out.update(
            weights_gb=weights / 1024**3, kv_cache_gb=kv / 1024**3,
            activations_gb=act / 1024**3, cuda_context_gb=cuda_ctx / 1024**3,
            optimizer_gb=0.0, gradients_gb=0.0, lora_gb=0.0,
            total_gb=total / 1024**3,
            note="vLLM 用 PagedAttention + 前缀缓存，实测通常比这个保守上界低 20-30%",
        )
        return out

    # ============================== 训练 ==============================
    wb = BYTES["4bit"] if method == "qlora" else BYTES["bf16"]
    weights = (p_llm + p_vis) * wb
    # 重要：激活值永远是 bf16/fp16，和权重是否量化无关。
    # （QLoRA 只是权重存成 4bit，前向计算还是在 bf16 里做的）
    act_b = 2

    if method in ("lora", "qlora"):
        trainable = _lora_params(m, lora_r)
        frozen_but_gradless = True
    else:
        trainable = p_llm + p_vis
        frozen_but_gradless = False
    out["trainable_params_m"] = trainable / 1e6
    out["trainable_ratio"] = trainable / (p_llm + p_vis)

    # 梯度：只有可训练参数有
    gradients = trainable * 2

    # 优化器状态
    if optimizer == "adamw":
        opt = trainable * 8
    elif optimizer == "adafactor":
        opt = trainable * 4
    elif optimizer == "sgd":
        opt = trainable * 4
    else:
        opt = 0.0
    if master_weights and method == "full":
        opt += trainable * 4          # fp32 master weight（AMP 必须）
    if master_weights and method in ("lora", "qlora"):
        opt += trainable * 4          # LoRA 参数一般也存 fp32 副本

    # ---------------- 激活值 ----------------
    # 单层激活 = attention 矩阵 + 其余中间量
    # attention 矩阵 = n_heads * seq^2 * batch * bytes（flash-attn 下不需要物化）
    attn_mat = 0.0 if flash_attn else m["n_heads"] * seq_len ** 2 * batch * act_b
    per_layer_rest = seq_len * batch * HID * 10 * act_b   # qkv/o 输出 + MLP 中间 + 残差

    if grad_checkpointing:
        # 存每层输入（层边界），反向时重算一层
        stored = L * seq_len * batch * HID * act_b
        recompute_peak = attn_mat + per_layer_rest
        act = stored + recompute_peak
    else:
        act = L * (attn_mat + per_layer_rest)

    # 视觉塔：即使冻结，前向仍要激活；梯度检查点时只存层边界
    patch_n = math.ceil(image_size / m["vis_patch"]) ** 2 * n_images
    v_per_layer = 6 * patch_n * m["vis_hidden"] * act_b
    if train_vision or not grad_checkpointing:
        vis_act = m["vis_layers"] * v_per_layer
    else:
        vis_act = m["vis_layers"] * patch_n * m["vis_hidden"] * act_b + v_per_layer

    # logits：HF 在计算 loss 时按全词表算（fp32）
    logits = seq_len * batch * m["vocab"] * 4

    cuda_ctx = 0.5 * 1024 ** 3

    # ZeRO 分片
    if zero_stage >= 2:
        opt = opt / 2
    if zero_stage >= 3:
        weights = weights / 2
        gradients = gradients / 2

    # 碎片 + 临时缓冲
    frag = 1.15

    subtotal = weights + gradients + opt + act + vis_act + logits + cuda_ctx
    total = subtotal * frag

    out.update(
        weights_gb=weights / 1024**3,
        gradients_gb=gradients / 1024**3,
        optimizer_gb=opt / 1024**3,
        activations_gb=act / 1024**3,
        vision_activations_gb=vis_act / 1024**3,
        logits_gb=logits / 1024**3,
        cuda_context_gb=cuda_ctx / 1024**3,
        lora_gb=(trainable * 2 / 1024**3) if method in ("lora", "qlora") else 0.0,
        total_gb=total / 1024**3,
    )
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def bar(used, total, width=26):
    frac = max(0.0, min(used / total, 1.0)) if total else 0
    n = int(frac * width)
    col = "\033[32m" if frac < 0.75 else ("\033[33m" if frac < 0.95 else "\033[31m")
    reset = "\033[0m" if sys.stdout.isatty() else ""
    col = col if sys.stdout.isatty() else ""
    return f"[{col}{'#'*n}{'.'*(width-n)}{reset}] {frac*100:5.1f}%"


def print_breakdown(r: dict):
    print()
    print(f"  模型       : {r['model']}")
    print(f"  方法       : {r['method']}")
    print(f"  序列 / batch: {r['seq_len']} / {r['batch']}")
    if r["method"] != "infer":
        print(f"  梯度检查点  : {'开' if r['grad_checkpointing'] else '关'}"
              f"     flash-attn: {'开' if r.get('flash_attn') else '关'}"
              f"     ZeRO: {r['zero_stage']}")
    print(f"  图片       : {r['n_images']} 张 -> {r['visual_tokens']} 个视觉 token"
          f"（占序列 {r['visual_tokens']/max(r['seq_len'],1)*100:.0f}%）")
    if r["visual_tokens"] / max(r["seq_len"], 1) > 0.55:
        print("                  ^ 视觉 token 占比偏高。调小 max_pixels 比调小 batch 更有效")

    print("  " + "-" * 58)
    rows = [
        ("权重",             r["weights_gb"]),
        ("梯度",             r["gradients_gb"]),
        ("优化器状态",       r["optimizer_gb"]),
        ("激活值（LLM）",     r["activations_gb"]),
        ("激活值（视觉塔）",  r.get("vision_activations_gb", 0.0)),
        ("Logits",           r.get("logits_gb", 0.0)),
        ("KV cache",         r.get("kv_cache_gb", 0.0)),
        ("CUDA context",     r["cuda_context_gb"]),
    ]
    for name, gb in rows:
        if gb and gb > 0.001:
            print(f"    {name:20s} {gb:7.2f} GB")
    print("  " + "-" * 58)
    print(f"    {'解析估算合计':20s} {r['total_gb']:7.2f} GB   (已含 15% 碎片余量)")
    if "trainable_params_m" in r:
        print(f"    可训练参数            {r['trainable_params_m']:7.1f} M"
              f"  （占全模型 {r['trainable_ratio']*100:.2f}%）")
    if r.get("note"):
        print(f"    注: {r['note']}")


def empirical_note(r: dict, key: str):
    lo_hi = EMPIRICAL.get((key, r["method"]))
    if not lo_hi:
        return
    print()
    print(f"  ⚠️  实测经验区间 : {lo_hi} GB   （序列 ~2048, batch 1, 梯度检查点开, 1 张 1024² 图）")
    print("      解析估算天然偏乐观。选卡请按经验区间的上沿留余量，")
    print("      因为真实框架还有 padding、flash-attn workspace、dataloader buffer 等开销。")


def suggest(r: dict, key: str):
    est = r["total_gb"]
    lo_hi = EMPIRICAL.get((key, r["method"]))
    # 用经验区间上沿来选卡（经验区间已经包含了真实框架的开销，不再叠加系数）
    if lo_hi:
        need = float(lo_hi.split("–")[-1].strip())
        basis = "实测经验区间上沿"
    else:
        need = est * 1.5
        basis = "解析估算 x1.5"

    print()
    print(f"  该租什么卡？  （依据：{basis} = {need:.1f} GB）")
    print("  " + "-" * 72)

    comfortable, fits, not_recommended = [], [], []
    for name, mem, bf16 in GPUS:
        usable = mem * 0.92
        headroom = usable / need if need else 99
        if headroom < 1.0:
            continue
        entry = (name, mem, bf16, headroom)
        if not bf16 or headroom < 1.25:
            not_recommended.append(entry)
        elif headroom >= 1.8:
            comfortable.append(entry)
        else:
            fits.append(entry)

    if not (comfortable or fits or not_recommended):
        print(f"    需要 ≈ {need:.0f} GB，单卡都不够。")
        print("    选项：")
        print("      ① 换更省的方法：full -> lora -> qlora")
        print("      ② 缩短序列 / 调小 max_pixels（图片 token 少了，序列自然短）")
        print("      ③ 多卡 + ZeRO-3 / FSDP（显存按卡数摊，但通信开销上来了）")
        print("      ④ 关掉 train_vision —— 视觉塔解冻是显存杀手，通常也没必要")
        return

    if comfortable:
        print("    推荐（余量充足，能开 batch>1 或长序列）:")
        for name, mem, bf16, hr in comfortable[:4]:
            print(f"      {name:14s} {mem:3d} GB  " + bar(need, mem * 0.92)
                  + f"   余量 {hr:.1f}x")
    if fits:
        print("    可用（能跑，但余量不多，注意 batch 和 seq）:")
        for name, mem, bf16, hr in fits[:4]:
            print(f"      {name:14s} {mem:3d} GB  " + bar(need, mem * 0.92)
                  + f"   余量 {hr:.1f}x")
    if not_recommended:
        print("    不推荐（能塞进去，但会很难受）:")
        for name, mem, bf16, hr in not_recommended[:4]:
            why = "不支持 bf16" if not bf16 else "余量不足 25%"
            print(f"      {name:14s} {mem:3d} GB  ({why})")

    print("  " + "-" * 72)
    pick = (comfortable or fits)[0]
    print(f"    结论：租 {pick[0]}（{pick[1]} GB）")
    if not pick[2]:
        print("    ⚠️  这块卡不支持 bf16，配置里要写 bf16: false, fp16: true")
    print()
    print("    参考价（2026-09）：RTX 3090 ¥1.32/h · RTX 4090 ¥1.88/h ·")
    print("                       RTX 5090/32G ¥2.78/h · A800-80G ¥4.98/h")
    print("    详细选型见 docs/13-hardware-and-cost.md")


def print_matrix():
    print()
    print("  全矩阵  (解析估算 GB / 实测经验区间 GB)")
    print("  条件: batch=1, seq=2048, 1 张 1024² 图, 梯度检查点开, flash-attn 开")
    print("=" * 94)
    hdr = f"  {'模型':24s} {'全参bf16':>14s} {'LoRA':>14s} {'QLoRA':>14s} {'推理bf16':>14s}"
    print(hdr)
    print("-" * 94)
    for k in ["0.5b", "3b", "7b", "32b"]:
        cells = []
        for method in ["full", "lora", "qlora", "infer"]:
            r = estimate(k, method)
            emp = EMPIRICAL.get((k, method), "-")
            cells.append(f"{r['total_gb']:.0f} / {emp}")
        name = MODELS[k]["name"].replace("Qwen2.5-VL-", "").replace("-Instruct", "")[:24]
        print(f"  {name:24s} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s} {cells[3]:>14s}")
    print("=" * 94)
    print()
    print("  读法（以实测区间为准）:")
    print("    · 3B QLoRA  8–12 GB   -> 一张 24 GB 的 4090 很宽松，可以 batch 2–4")
    print("    · 3B LoRA   16–20 GB  -> 4090 可以跑，batch 保持 1，注意激活值")
    print("    · 3B 全参   48–60 GB  -> 必须 80 GB 卡")
    print("    · 7B QLoRA  16–22 GB  -> 单张 4090 可以，但 max_length 别开太大")
    print("    · 7B LoRA   30–38 GB  -> 单卡 4090 不够，要 A800-80G")
    print("    · 32B 任何方式单卡都不现实，本项目不涉及")
    print()
    print("  一句话结论：本项目的甜点区是 3B QLoRA + 24 GB 卡。")
    print("             预算允许就直接上 32 GB 的 5090，能少操很多心（¥2.78/h）。")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="显存估算器")
    ap.add_argument("--model", default="3b", choices=list(MODELS.keys()))
    ap.add_argument("--method", default="qlora", choices=["full", "lora", "qlora", "infer"])
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--images", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor", "sgd"])
    ap.add_argument("--train-vision", action="store_true", help="解冻视觉塔（显存杀手，通常没必要）")
    ap.add_argument("--zero", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-flash-attn", action="store_true", help="不装 flash-attn，attention 矩阵要物化")
    ap.add_argument("--no-master-weights", action="store_true")
    ap.add_argument("--kv-len", type=int, default=None, help="推理时的序列长度")
    ap.add_argument("--suggest", action="store_true", help="推荐租什么卡")
    ap.add_argument("--all", action="store_true", help="全矩阵对比")
    args = ap.parse_args()

    if args.all:
        print_matrix()
        return

    if args.method == "infer" and args.kv_len:
        args.seq = args.kv_len

    r = estimate(
        args.model, args.method,
        seq_len=args.seq, batch=args.bs, n_images=args.images,
        image_size=args.image_size, grad_checkpointing=not args.no_grad_ckpt,
        optimizer=args.optimizer, train_vision=args.train_vision,
        zero_stage=args.zero, flash_attn=not args.no_flash_attn,
        master_weights=not args.no_master_weights, lora_r=args.lora_r,
    )
    print_breakdown(r)
    empirical_note(r, args.model)
    if args.suggest or args.method == "infer":
        suggest(r, args.model)

    print()
    print("  调优顺序（显存不够时）:")
    print("    1) 调小 max_pixels   —— 图片 token 少了，序列自然短，最有效")
    print("    2) 调小 seq_len      —— 直接砍激活值和 logits")
    print("    3) 确认梯度检查点是开的（必须开）")
    print("    4) 从 lora 换到 qlora —— 显存砍半，质量损失有限")
    print("    5) 最后才考虑调小 batch（会拖慢训练）")
    print()
    print("  想提速时（余量充足）:")
    print("    batch -> seq_len -> 关梯度检查点（最后考虑，激活值会暴涨）")


if __name__ == "__main__":
    main()
