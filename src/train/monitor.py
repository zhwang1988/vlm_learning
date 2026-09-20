"""
训练日志分析与可视化。

对应讲义 docs/06-sft-training.md「三条曲线的正常形态」，对应计划 Day 16。

产出：loss / learning_rate / grad_norm 三联图 + 一段自动诊断。

**诊断比画图更有价值**——它会告诉你 loss 不降可能是哪个原因。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def parse_trainer_log(log_path: str | Path) -> dict[str, list]:
    """从 trainer 日志里抽取三条曲线。

    支持两种来源：
      1. trainer_state.json（HF Trainer 自动写的，最可靠）
      2. stdout 日志文本（正则抽，备用）
    """
    p = Path(log_path)

    # --- 优先读 trainer_state.json ---
    if p.is_dir():
        state_file = p / "trainer_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            hist = state.get("log_history", [])
            out: dict[str, list] = {
                "step": [], "loss": [], "eval_loss": [],
                "learning_rate": [], "grad_norm": [], "epoch": [],
            }
            for h in hist:
                if "loss" in h:
                    out["step"].append(h.get("step"))
                    out["loss"].append(h["loss"])
                    out["learning_rate"].append(h.get("learning_rate"))
                    out["grad_norm"].append(h.get("grad_norm"))
                    out["epoch"].append(h.get("epoch"))
                if "eval_loss" in h:
                    out["eval_loss"].append((h.get("step"), h["eval_loss"]))
            return out
        log_path = p / "trainer_log.txt"

    # --- 从文本日志正则抽取 ---
    out = {"step": [], "loss": [], "eval_loss": [],
           "learning_rate": [], "grad_norm": [], "epoch": []}
    if not Path(log_path).exists():
        return out

    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(
        r"\{[^{}]*'loss'[^{}]*\}"
    )
    for m in pat.finditer(text):
        try:
            d = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            continue
        if "loss" in d and "eval_loss" not in d:
            out["step"].append(d.get("step") or len(out["step"]))
            out["loss"].append(d["loss"])
            out["learning_rate"].append(d.get("learning_rate"))
            out["grad_norm"].append(d.get("grad_norm"))
            out["epoch"].append(d.get("epoch"))

    ev = re.compile(r"\{[^{}]*'eval_loss'[^{}]*\}")
    for m in ev.finditer(text):
        try:
            d = json.loads(m.group(0).replace("'", '"'))
            out["eval_loss"].append((d.get("step"), d["eval_loss"]))
        except Exception:
            continue
    return out


def diagnose(data: dict, max_length_hint: Optional[int] = None) -> list[str]:
    """根据曲线形态给出诊断建议。Day 16 的验收动作。"""
    msgs = []
    losses = [x for x in data.get("loss", []) if x is not None]
    grad_norms = [x for x in data.get("grad_norm", []) if x is not None]

    if not losses:
        return ["⚠ 没有解析到 loss 数据。检查日志路径是否正确。"]

    # --- loss 趋势 ---
    if len(losses) >= 10:
        head = sum(losses[:5]) / 5
        tail = sum(losses[-5:]) / 5
        drop = (head - tail) / head if head else 0

        if drop < 0.03:
            msgs.append(
                f"❌ **loss 几乎没降**（{head:.3f} → {tail:.3f}，降幅 {drop:.1%}）。"
                "按以下顺序排查：\n"
                "   1. label 是否全被 mask 了？跑 "
                "`python -m src.train.sft_peft --dry-run` 检查\n"
                "   2. 学习率是否太小？LoRA 需要 1e-4 量级，不是 1e-5\n"
                "   3. chat template 是否错配？对比 apply_chat_template 输出\n"
                "   4. LoRA 层是否真的可训练？看 print_trainable_parameters 的输出\n"
                "   5. 数据本身是否有问题？回看 cleaning_report.md"
            )
        elif drop > 0.7:
            msgs.append(
                f"⚠ loss 降得过快（{head:.3f} → {tail:.3f}，降幅 {drop:.1%}）。"
                "可能是数据重复率高、或者过拟合很快。检查 eval_loss 是否同步下降。"
            )
        else:
            msgs.append(f"✓ loss 正常下降（{head:.3f} → {tail:.3f}，降幅 {drop:.1%}）")

        # --- 尾部是否震荡 ---
        if len(losses) >= 20:
            import statistics
            tail_10 = losses[-10:]
            if statistics.stdev(tail_10) > 0.15 * (sum(tail_10) / 10):
                msgs.append(
                    "⚠ 尾部 loss 震荡较大。可能原因：学习率偏大、batch 太小、"
                    "数据噪声大。考虑降低 lr 或增大 gradient_accumulation。"
                )

    # --- eval loss 是否上升（过拟合）---
    ev = data.get("eval_loss", [])
    if len(ev) >= 2:
        vals = [v for _, v in ev]
        if vals[-1] > vals[-2] * 1.02:
            msgs.append(
                f"❌ **eval_loss 开始上升**（{vals[-2]:.4f} → {vals[-1]:.4f}）→ 过拟合。\n"
                "   处理：减少 epoch、增大 dropout、增加数据量、或用 "
                "load_best_model_at_end 回滚到最佳的 checkpoint。"
            )
        else:
            msgs.append(f"✓ eval_loss 稳定/下降（最新 {vals[-1]:.4f}）")

    # --- grad_norm ---
    if grad_norms:
        mx = max(grad_norms)
        avg = sum(grad_norms) / len(grad_norms)
        late = grad_norms[len(grad_norms) // 2:]
        late_avg = sum(late) / max(len(late), 1)

        if mx > 10:
            msgs.append(
                f"❌ **grad_norm 峰值 {mx:.2f}（>10）** → 梯度爆炸。\n"
                "   处理：降低学习率、增大 warmup_ratio、或调低 max_grad_norm。"
            )
        elif late_avg > avg * 1.5:
            msgs.append(
                f"⚠ grad_norm 后期上升（{avg:.2f} → {late_avg:.2f}）→ 训练不稳。"
                "考虑降 lr 或检查数据异常样本。"
            )
        else:
            msgs.append(f"✓ grad_norm 稳定（均值 {avg:.2f}，峰值 {mx:.2f}）")

    return msgs


def plot(data: dict, out_path: str | Path = "reports/training_curves.png",
         title: str = "Training Curves"):
    """画三联图。没有 matplotlib 时自动降级为文本输出。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("（未安装 matplotlib，跳过绘图）")
        return None

    steps = data.get("step", [])
    if not steps:
        print("（无数据可画）")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # --- loss ---
    ax = axes[0]
    ax.plot(steps, data["loss"], lw=1.6, label="train loss", color="#1f77b4")
    ev = data.get("eval_loss", [])
    if ev:
        ax.plot([s for s, _ in ev], [v for _, v in ev], "o-",
                lw=1.6, ms=4, label="eval loss", color="#d62728")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- learning rate ---
    ax = axes[1]
    lrs = [x for x in data.get("learning_rate", []) if x is not None]
    if lrs:
        ax.plot(steps[:len(lrs)], lrs, lw=1.6, color="#2ca02c")
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
        ax.set_title("Learning Rate (cosine 应降到接近 0)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax.text(0.5, 0.5, "no lr data", ha="center", va="center")
        ax.set_title("Learning Rate")
    ax.grid(alpha=0.3)

    # --- grad norm ---
    ax = axes[2]
    gns = [x for x in data.get("grad_norm", []) if x is not None]
    if gns:
        ax.plot(steps[:len(gns)], gns, lw=1.4, color="#ff7f0e")
        ax.axhline(10, ls="--", color="red", alpha=0.6, label="危险线 (10)")
        ax.set_xlabel("step")
        ax.set_ylabel("grad_norm")
        ax.set_title("Gradient Norm")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no grad_norm data", ha="center", va="center")
        ax.set_title("Gradient Norm")
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"✓ 三联图 → {out_path}")
    return out_path


def parse_and_plot(output_dir: str | Path, out_png: Optional[str | Path] = None):
    """读取 trainer_state.json，出图 + 诊断。训练脚本自动调用这个。"""
    output_dir = Path(output_dir)
    data = parse_trainer_log(output_dir)

    n = len(data.get("loss", []))
    if n == 0:
        print(f"（在 {output_dir} 没找到训练日志）")
        return data

    print(f"解析到 {n} 个训练步的日志")

    out_png = out_png or (output_dir / "training_curves.png")
    plot(data, out_png, title=str(output_dir.name))

    print()
    print("=" * 78)
    print("训练诊断")
    print("=" * 78)
    for m in diagnose(data):
        print(f"  {m}")
    print()

    return data


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="训练日志分析")
    ap.add_argument("output_dir", nargs="?", default="outputs/qwen25vl3b-cx-lora-v0")
    args = ap.parse_args()
    parse_and_plot(args.output_dir)
