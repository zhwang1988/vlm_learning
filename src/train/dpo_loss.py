"""
DPO：损失实现 + 偏好数据构造。

对应讲义 docs/07-alignment.md，对应计划 Day 25–Day 27。

两部分：
  Part A  手写 DPO loss（含数值验证）—— 理解闭式解
  Part B  多模态偏好数据构造 —— 三种类型的构造方法
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# torch 延迟导入
#
# 本文件有两半：Part A 算 loss（要 torch），Part B 造偏好数据（纯文本）。
# 造偏好数据是 W5 Day 26 的主力工作，它**不该要求你开着 GPU 机器**。
# 所以 torch 改成按需导入：只有 Part A 的路径会触发。
# ---------------------------------------------------------------------------

torch = None
F = None


def _ensure_torch():
    """按需导入 torch。Part B（造偏好数据）不会调用它。"""
    global torch, F
    if torch is None:
        try:
            import torch as _torch
            import torch.nn.functional as _F
        except ImportError as e:
            raise RuntimeError(
                "这条路径需要 PyTorch。造偏好数据（--from-badcases / --contrastive）"
                "不需要它；跑 --verify 请在装了 torch 的环境里执行。"
            ) from e
        torch, F = _torch, _F
    return torch


# =============================================================================
# Part A · DPO Loss
# =============================================================================


def get_batch_logps(
    logits: torch.Tensor,       # (B, L, V)
    labels: torch.Tensor,       # (B, L)  非监督位为 -100
    average_log_prob: bool = False,
) -> torch.Tensor:
    """算每个样本「被监督部分」的对数概率和。

    average_log_prob=True 时按 token 数归一化（SimPO 风格，抗长度偏见）。
    """
    _ensure_torch()
    labels = labels.clone()
    mask = labels != -100
    labels[~mask] = 0                      # 避免 gather 越界

    logp = F.log_softmax(logits, dim=-1)                     # (B, L, V)
    per_token = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # (B, L)
    per_token = per_token * mask                            # 屏蔽非监督位

    logps = per_token.sum(-1)                               # (B,)
    if average_log_prob:
        logps = logps / mask.sum(-1).clamp(min=1)
    return logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """DPO 损失。

    L = -log σ( β·(log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)) )

    直觉：
      log(π_θ/π_ref) 是「相对基座，策略对这个回答的偏好提升了多少」
      DPO 要求 chosen 的相对提升 > rejected 的相对提升
      除以 π_ref 是关键的锚——防止模型为了拉高 chosen 而破坏语言能力

    β 控制偏离 reference 的惩罚：大→保守，小→激进易崩。典型 0.1–0.5。
    """
    _ensure_torch()
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits_diff = chosen_rewards - rejected_rewards

    if label_smoothing > 0:
        # cDPO：对偏好标签做平滑，抗噪声标注
        losses = (-F.logsigmoid(logits_diff) * (1 - label_smoothing)
                  - F.logsigmoid(-logits_diff) * label_smoothing)
    else:
        losses = -F.logsigmoid(logits_diff)

    with torch.no_grad():
        acc = (chosen_rewards > rejected_rewards).float().mean()
        margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "loss": losses.mean().item(),
        "accuracy": acc.item(),                 # 应稳定上升趋近 1
        "rewards_chosen": chosen_rewards.mean().item(),
        "rewards_rejected": rejected_rewards.mean().item(),
        "reward_margin": margin.item(),         # 应稳步增大
    }
    return losses.mean(), metrics


def verify_dpo_loss():
    """用玩具数据验证 DPO loss 的数值行为。Day 25 的验收动作。"""
    print("=" * 78)
    print("DPO Loss 数值验证")
    print("=" * 78)

    _ensure_torch()
    torch.manual_seed(0)
    B = 4
    beta = 0.1

    # --- 场景 1：chosen 优于 rejected（策略已学会）---
    p_chosen = torch.tensor([-1.0, -1.2, -0.8, -1.1])
    p_rejected = torch.tensor([-2.0, -2.3, -1.9, -2.2])
    r_chosen = p_chosen.clone()          # reference == policy 初值
    r_rejected = p_rejected.clone()

    loss, m = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta)
    print(f"\n[场景 1] 策略已偏好 chosen")
    print(f"  loss          = {m['loss']:.4f}")
    print(f"  accuracy      = {m['accuracy']:.2f}   (期望 1.0)")
    print(f"  reward_margin = {m['reward_margin']:.4f}  (期望 > 0)")
    assert m["accuracy"] == 1.0, "chosen 更好时准确率必须为 1"
    print("  ✓ 通过")

    # --- 场景 2：两者无差别（策略没学到东西）---
    p_same = torch.tensor([-1.5] * B)
    loss2, m2 = dpo_loss(p_same, p_same, p_same, p_same, beta)
    print(f"\n[场景 2] chosen == rejected（无信息）")
    print(f"  loss          = {m2['loss']:.4f}   (期望 = ln2 ≈ 0.6931)")
    print(f"  reward_margin = {m2['reward_margin']:.4f}")
    assert abs(m2["loss"] - 0.6931) < 1e-3, "无差别时 loss 应为 ln2"
    print("  ✓ 通过（这正是 DPO 的数学性质）")

    # --- 场景 3：β 的影响 ---
    print(f"\n[场景 3] β 对 loss 的影响（chosen 优于 rejected 时）")
    print(f"  {'β':>6} {'loss':>10} {'margin':>10}")
    for b in (0.01, 0.05, 0.1, 0.5, 1.0):
        _, mm = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, b)
        print(f"  {b:>6} {mm['loss']:>10.4f} {mm['reward_margin']:>10.4f}")
    print("  → β 越大，margin 越大，loss 越小（梯度也可能更大，更激进）")

    # --- 场景 4：长度偏见 ---
    print(f"\n[场景 4] 长度归一化的影响")
    logits = torch.randn(2, 20, 50)
    labels_long = torch.full((2, 20), -100)
    labels_long[:, 5:] = torch.randint(0, 50, (2, 15))
    labels_short = torch.full((2, 20), -100)
    labels_short[:, 5:8] = torch.randint(0, 50, (2, 3))

    lp_long = get_batch_logps(logits, labels_long, average_log_prob=False)
    lp_short = get_batch_logps(logits, labels_short, average_log_prob=False)
    lp_long_avg = get_batch_logps(logits, labels_long, average_log_prob=True)
    lp_short_avg = get_batch_logps(logits, labels_short, average_log_prob=True)

    print(f"  不归一化: 长回答 {lp_long.mean():.2f}  短回答 {lp_short.mean():.2f}  "
          f"→ 长回答天然分数低（和是负数，越长越小）")
    print(f"  归一化后: 长回答 {lp_long_avg.mean():.2f}  短回答 {lp_short_avg.mean():.2f}")
    print("  → 不归一化会让 DPO 系统性偏好短回答（因为求和时长度影响大）")
    print("    但过度归一化也会偏向短答。实践中看评测结果决定。")

    print("\n✓ 全部验证通过")


# =============================================================================
# Part B · 多模态偏好数据构造
# =============================================================================

# 三种类型（见 docs/07-alignment.md）
#   type1_same_image   同图不同答 —— 修判断校准和语气。最容易构造，收益最大。
#   type2_contrastive  同答不同图 —— 修视觉 grounding。**治幻觉的利器**。
#   type3_format       同图同答不同格式 —— 修工具调用的稳定性。

TYPE1_TEMPLATE = """一条针对同一张图的客服对话，给出了两个回复版本。

【用户问题】
{query}

【图片内容】
{caption}

【版本 A（更好）】
{chosen}

【版本 B（更差）】
{rejected}

为什么 A 更好：{reason}

请再生成一组类似的对比数据，输出 JSON：
{{"query": "...", "chosen": "...", "rejected": "...", "reason": "..."}}

要求：
- 两个版本的差异必须「可学习」——即 A 的优势来自可辨识的依据，不是随机
- 常见差异维度：事实判断（是不是瑕疵）、语气（是否安抚）、是否主动澄清
- rejected 不能是明显的胡说，应该是有一定合理性但判断偏了/语气差了
"""

TYPE2_TEMPLATE = """给定一个用户问题，它可能配不同的图片。

【用户问题】
{query}

【图片 A】{caption_a}
【图片 B】{caption_b}

要求构造一对偏好数据：
- 正确配对的图 → 应该被偏好
- 错配的图 → 应该被拒绝（因为答案基于图 A 的内容，
  配图 B 时同一个答案就变成了幻觉）

这对数据的作用：让模型学会「必须看这张图才能这样回答」，
而不是靠语言先验瞎猜。这是治幻觉最有效的手段之一。

输出 JSON：
{{"query": "...", "chosen_caption": "...", "rejected_caption": "...", "answer": "..."}}
"""


def build_preference_from_badcases(
    bad_case_path: str | Path,
    out_path: str | Path = "data/processed/dpo_train.jsonl",
    image_root: Optional[str] = None,
) -> int:
    """从错误分析结果构造偏好数据（type1）。

    流程：
      1. 读 error_analysis.py 产出的 bad cases
      2. 每条 bad case 里，模型的错误回答 = rejected
      3. 用强模型生成正确回答 = chosen
      4. 输出成 DPO 训练格式

    这是最高性价比的偏好数据来源——因为 rejected 是模型真的会犯的错，
    而不是人造的。
    """
    bp = Path(bad_case_path)
    if not bp.exists():
        print(f"✗ 找不到 {bp}，请先跑 src/eval/error_analysis.py")
        return 0

    rows = [json.loads(l) for l in bp.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"读入 {len(rows)} 条 bad case")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            if not r.get("model_answer") or not r.get("reference"):
                skipped += 1
                continue
            record = {
                "id": r.get("id", f"dpo_{written}"),
                "images": r.get("images", []),
                "prompt": r.get("query", ""),
                "chosen": r["reference"],
                "rejected": r["model_answer"],
                "source": "bad_case",
                "error_type": r.get("error_type", "unknown"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if skipped:
        print(f"⚠️  跳过 {skipped} 条：缺 model_answer 或 reference 字段")
        print("    正确字段名是 query / model_answer / reference")
        print("    （由 src/eval/error_analysis.py 输出；别的格式需要先转换）")
    if written == 0 and rows:
        print("✗ 一对都没写出来 —— 先确认这个文件是 error_analysis.py 生成的 bad_cases.jsonl")
    print(f"✓ 写出 {written} 对偏好数据 → {out_path}")
    return written


def build_contrastive_pairs(
    samples: list[dict],
    out_path: str | Path = "data/processed/dpo_contrastive.jsonl",
    n_pairs: int = 2000,
    seed: int = 42,
) -> int:
    """构造 type2 对比数据：同答不同图。

    **几乎零成本的幻觉抑制手段。**

    做法：
      1. 拿一条正常样本 (图 A, 问题 Q, 答案 A)
      2. 随机找一张「不合适」的图 B（来自不同的 image_type 或不同商品）
      3. 构造偏好对：
         chosen   = (图A, Q) -> A      ← A 是基于图 A 写的，正确
         rejected = (图B, Q) -> A      ← A 和图 B 不匹配，变成幻觉

    模型要学的：同样的答案配错图 → 应该被拒绝。
    效果：模型学会「回答必须 grounded 在实际看到的那张图上」。
    """
    random.seed(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 按 image_type 分组，方便挑「不匹配」的图
    by_type: dict[str, list[dict]] = {}
    for s in samples:
        by_type.setdefault(s.get("image_type", "?"), []).append(s)

    types = list(by_type.keys())
    if len(types) < 2:
        print("⚠ 图像类型不足 2 种，无法构造对比数据")
        return 0

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            if written >= n_pairs:
                break
            imgs_a = s.get("images") or ([s.get("image_path")] if s.get("image_path") else [])
            if not imgs_a:
                continue
            try:
                query = next(c["text"] for c in s["messages"][0]["content"]
                             if c["type"] == "text")
                answer = next(c["text"] for c in s["messages"][1]["content"]
                              if c["type"] == "text")
            except Exception:
                continue

            # 挑一个不同 image_type 的图作为错配
            other_types = [t for t in types if t != s.get("image_type")]
            if not other_types:
                continue
            other = random.choice(by_type[random.choice(other_types)])
            imgs_b = other.get("images") or ([other.get("image_path")]
                                            if other.get("image_path") else [])
            if not imgs_b:
                continue

            f.write(json.dumps({
                "id": f"contrast_{written}",
                "query": query,
                "answer": answer,
                "chosen_images": imgs_a,
                "rejected_images": imgs_b,
                "source": "contrastive",
                "note": "同答不同图：让模型学会 grounded 在实际图上",
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"✓ 写出 {written} 对对比数据 → {out_path}")
    print("  用法：训练时 chosen 用 chosen_images，rejected 用 rejected_images，")
    print("        文本部分（query + answer）完全相同。")
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="验证 DPO loss")
    ap.add_argument("--from-badcases", help="从 bad case 构造偏好数据")
    ap.add_argument("--contrastive", help="从训练集构造对比数据（传入 clean.jsonl）")
    ap.add_argument("--n-pairs", type=int, default=2000)
    args = ap.parse_args()

    if args.verify:
        try:
            verify_dpo_loss()
        except RuntimeError as e:
            print(f"\n✗ {e}\n")
            print("  → --verify 是纯数学验证，要 torch，但不需要 GPU。")
            print("    在本地装 CPU 版即可：pip install torch --index-url "
                  "https://download.pytorch.org/whl/cpu")
            raise SystemExit(1)
    elif args.from_badcases:
        build_preference_from_badcases(args.from_badcases)
    elif args.contrastive:
        samples = [json.loads(l) for l in
                   Path(args.contrastive).read_text(encoding="utf-8").splitlines() if l.strip()]
        build_contrastive_pairs(samples, n_pairs=args.n_pairs)
    else:
        ap.print_help()
        print("\n常用：")
        print("  python -m src.train.dpo_loss --verify                  # 数值验证（要 torch）")
        print("  python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl"
              "   # 造偏好数据（本地可跑）")
        print("  python -m src.train.dpo_loss --contrastive data/processed/clean.jsonl"
              "        # 对比式偏好对")
