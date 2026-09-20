"""
data —— 客服图文数据工程。

对应讲义：docs/05-data-engineering.md
对应计划：Week 2（Day 7–Day 12）

流水线：
    taxonomy.py     Day 8  定义「意图 × 图像类型」矩阵，算目标样本量
    image_utils.py  Day 7  图像清洗（EXIF/alpha/CMYK/比例）+ pHash 去重
    synth.py        Day 9  按矩阵批量合成图文对话（persona/emotion 抗同质化）
    dedup.py        Day 10 规则过滤 + 图文一致性 + 去重 + 清洗报告
    build_sft.py    Day 11 转训练格式 + label mask + 泄漏检查
    report.py       Day 12 生成数据集卡片

一键跑通（Day 12）：
    python -m src.data.taxonomy                     # 看矩阵
    python -m src.data.taxonomy --export
    python -m src.data.synth --image-root data/images --sample
    python -m src.data.synth --image-root data/images --out data/raw/synth_v0.jsonl
    python -m src.data.dedup --in data/raw/synth_v0.jsonl
    python -m src.data.build_sft --in data/processed/clean.jsonl
    python -m src.data.build_sft --inspect data/processed/sft_train.jsonl
    python -m src.data.report
"""

# ---------------------------------------------------------------------------
# 惰性导出（PEP 562）
#
# 这里故意不写 `from .xxx import yyy`。因为本包下有些模块要 import torch，
# 有些不要。急切导入会让「只想跑 torch-free 模块」的人在本地直接撞
# ModuleNotFoundError —— 纯粹被连坐。
#
# 改成按需加载后：
#     from src.data import CleaningReport      # 触发时才 import 对应模块
#     python -m src.data.<torch-free 模块>   # 本地可跑
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    "CleaningReport": ".dedup",
    "DEFAULT_SYSTEM_PROMPT": ".build_sft",
    "DIFFICULTY_TIERS": ".taxonomy",
    "IMAGE_TYPES": ".taxonomy",
    "INTENTS": ".taxonomy",
    "ImageStats": ".image_utils",
    "TARGET_DISTRIBUTION": ".taxonomy",
    "build_dataset": ".build_sft",
    "build_generation_plan": ".taxonomy",
    "build_labeled_sample": ".build_sft",
    "check_leakage": ".build_sft",
    "clean_pipeline": ".dedup",
    "detect_collage": ".image_utils",
    "export": ".taxonomy",
    "find_duplicate_groups": ".image_utils",
    "generate_card": ".report",
    "hamming_distance": ".image_utils",
    "inspect_sample": ".build_sft",
    "load_and_normalize": ".image_utils",
    "phash": ".image_utils",
    "print_matrix": ".taxonomy",
    "rule_filter": ".dedup",
    "scan_directory": ".image_utils",
    "total_target": ".taxonomy",
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
    "DEFAULT_SYSTEM_PROMPT",
    "build_dataset",
    "build_labeled_sample",
    "check_leakage",
    "inspect_sample",
    "CleaningReport",
    "clean_pipeline",
    "rule_filter",
    "ImageStats",
    "detect_collage",
    "find_duplicate_groups",
    "hamming_distance",
    "load_and_normalize",
    "phash",
    "scan_directory",
    "generate_card",
    "DIFFICULTY_TIERS",
    "IMAGE_TYPES",
    "INTENTS",
    "TARGET_DISTRIBUTION",
    "build_generation_plan",
    "export",
    "print_matrix",
    "total_target",
]
