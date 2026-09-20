"""
train —— 后训练：SFT / DPO / GRPO。

对应讲义：docs/06-sft-training.md、docs/07-alignment.md
对应计划：Week 3（Day 13–18）、Week 5（Day 25–28）

两条路线，建议都走一遍：
  LLaMA-Factory  route：快、黑盒 → Day 15 建立基线
  原生 transformers route：慢、可读 → Day 17–18 理解细节

文件导航：
  lora_utils.py   LoRA 层手写实现 + target_modules 解析 + 显存估算
  sft_peft.py     原生路线：多模态 Dataset / Collator / Trainer
  dpo_loss.py     DPO loss 手写 + 验证 + 多模态偏好数据构造
  rewards.py      可验证奖励（GRPO 用）
  monitor.py      训练日志解析 + 三联图 + 自动诊断

一键流程：
    # Day 13 显存估算
    python -m src.train.lora_utils --table

    # Day 14 LoRA 原理演示
    python -m src.train.lora_utils --demo

    # Day 15 ⭐ 先 dry-run 验证数据
    python -m src.train.sft_peft --config configs/sft_lora_3b.yaml --dry-run

    # Day 15 正式训练（QLoRA，16G 卡可跑）
    python -m src.train.sft_peft --config configs/sft_lora_3b.yaml --qlora

    # Day 16 训练完成后自动出诊断
    python -m src.train.monitor outputs/qwen25vl3b-cx-lora-v0

    # Day 25 验证 DPO loss
    python -m src.train.dpo_loss --verify

    # Day 26 构造偏好数据
    python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl
    python -m src.train.dpo_loss --contrastive data/processed/clean.jsonl

    # Day 28 验证奖励函数
    python -m src.train.rewards
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.train import FORBIDDEN      # 触发时才 import 对应模块
#     python -m src.train.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "FORBIDDEN": ".lora_utils",
    "LoRAConfig": ".lora_utils",
    "LoRALinear": ".lora_utils",
    "MultimodalCollator": ".sft_peft",
    "MultimodalSFTDataset": ".sft_peft",
    "RewardContext": ".rewards",
    "RewardWeights": ".rewards",
    "TARGET_CONNECTOR": ".lora_utils",
    "TARGET_STANDARD": ".lora_utils",
    "TrainConfig": ".sft_peft",
    "build_contrastive_pairs": ".dpo_loss",
    "build_preference_from_badcases": ".dpo_loss",
    "compute_reward": ".rewards",
    "diagnose": ".monitor",
    "dpo_loss": ".dpo_loss",
    "estimate_training_vram": ".lora_utils",
    "get_batch_logps": ".dpo_loss",
    "parse_and_plot": ".monitor",
    "parse_trainer_log": ".monitor",
    "plot": ".monitor",
    "print_lora_plan": ".lora_utils",
    "print_vram_table": ".lora_utils",
    "resolve_target_modules": ".lora_utils",
    "train": ".sft_peft",
    "verify_dpo_loss": ".dpo_loss",
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
    "build_contrastive_pairs",
    "build_preference_from_badcases",
    "dpo_loss",
    "get_batch_logps",
    "verify_dpo_loss",
    "FORBIDDEN",
    "TARGET_CONNECTOR",
    "TARGET_STANDARD",
    "LoRAConfig",
    "LoRALinear",
    "estimate_training_vram",
    "print_lora_plan",
    "print_vram_table",
    "resolve_target_modules",
    "diagnose",
    "parse_and_plot",
    "parse_trainer_log",
    "plot",
    "RewardContext",
    "RewardWeights",
    "compute_reward",
    "MultimodalCollator",
    "MultimodalSFTDataset",
    "TrainConfig",
    "train",
]
