"""
客服场景数据 Taxonomy：意图 × 图像类型的二维矩阵。

对应讲义 docs/05-data-engineering.md 第 4 节，对应计划 Day 8。

Day 8 的产出就是这个矩阵 + 每个有效格子的真实例子。

为什么要先做分类：
  不做分类的后果是采完数据才发现「90% 是问尺码，但线上 30% 是质量投诉」。
  矩阵让你能一眼看出哪里缺样本，以及每个格子的目标数量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# 维度定义
# =============================================================================


@dataclass
class Intent:
    key: str
    name: str
    description: str
    example_queries: list[str] = field(default_factory=list)
    target_count: int = 0


@dataclass
class ImageType:
    key: str
    name: str
    description: str
    difficulty: int = 1          # 1-5，对模型的挑战程度


INTENTS: list[Intent] = [
    Intent("size_fit", "尺码合身", "用户想知道某尺码是否适合自己",
           ["我170/60穿M可以吗", "这个偏大还是偏小", "腰围多少"]),
    Intent("quality_issue", "质量/瑕疵", "用户对做工或收到的实物有疑问",
           ["这里是不是线头", "这个走线正常吗", "收到的和图片不太一样"]),
    Intent("color_mismatch", "色差", "实物颜色与图片/描述是否一致的判断",
           ["实物是米白还是纯白", "这个偏黄吗", "图片和实物颜色差多少"]),
    Intent("material", "材质成分", "面料、成分、手感相关问题",
           ["是纯棉吗", "会不会起球", "透气性怎么样"]),
    Intent("styling", "搭配建议", "怎么搭配、适合什么场合",
           ["配什么裤子好看", "适合夏天穿吗", "正式场合能穿吗"]),
    Intent("logistics", "物流", "快递、发货、时效相关问题",
           ["我的快递到哪了", "什么时候发货", "能不能改地址"]),
    Intent("return_refund", "退换货", "退货、换货、退款流程",
           ["这个能退吗", "怎么申请换货", "退款多久到账"]),
    Intent("presale_general", "售前其他", "库存、补货、优惠等",
           ["有别的颜色吗", "什么时候补货", "有优惠券吗"]),
]

IMAGE_TYPES: list[ImageType] = [
    ImageType("product_main", "商品主图", "白底或简洁背景的商品图", difficulty=1),
    ImageType("model_wearing", "模特上身图", "真人模特穿着展示", difficulty=2),
    ImageType("detail_closeup", "细节特写", "面料纹理、做工、五金等局部放大", difficulty=3),
    ImageType("defect_photo", "瑕疵实拍", "用户自己拍的瑕疵照片，光线构图较差", difficulty=5),
    ImageType("size_chart", "尺码表", "表格形式的尺寸对照表", difficulty=4),
    ImageType("user_compare", "实物对比", "用户收到的实物 vs 商品图，或多款对比", difficulty=5),
]


# =============================================================================
# 二维矩阵：哪些组合是有意义的
# =============================================================================

# 值 = 该格子的目标样本数；None = 该组合无意义
TARGET_DISTRIBUTION: dict[str, dict[str, int | None]] = {
    # intent -> {image_type: target_count}
    "size_fit": {
        "product_main": 300, "model_wearing": 500, "size_chart": 400,
        "detail_closeup": 150, "defect_photo": 0, "user_compare": 150,
    },
    "quality_issue": {
        "product_main": 100, "model_wearing": 60, "detail_closeup": 400,
        "defect_photo": 700,                      # ← 最难也最重要
        "size_chart": 0,   "user_compare": 300,
    },
    "color_mismatch": {
        "product_main": 250, "model_wearing": 300, "detail_closeup": 200,
        "defect_photo": 150, "size_chart": 0,     "user_compare": 400,
    },
    "material": {
        "product_main": 200, "model_wearing": 100, "detail_closeup": 450,
        "defect_photo": 100, "size_chart": 50,    "user_compare": 50,
    },
    "styling": {
        "product_main": 250, "model_wearing": 450, "detail_closeup": 50,
        "defect_photo": 0,   "size_chart": 0,     "user_compare": 50,
    },
    "logistics": {
        "product_main": 50,  "model_wearing": 0,   "detail_closeup": 0,
        "defect_photo": 0,   "size_chart": 0,     "user_compare": 250,   # 快递单截图
    },
    "return_refund": {
        "product_main": 60,  "model_wearing": 20,  "detail_closeup": 100,
        "defect_photo": 350, "size_chart": 40,    "user_compare": 200,
    },
    "presale_general": {
        "product_main": 300, "model_wearing": 150, "detail_closeup": 80,
        "defect_photo": 0,   "size_chart": 60,    "user_compare": 40,
    },
}

# 难度分级：Day 20 构造评测集时按这个分层抽样
DIFFICULTY_TIERS = {
    "L1": "单图单事实——图里直接能看到的信息",
    "L2": "单图推理——需要结合常识或政策判断",
    "L3": "多图/图+文冲突——需要对比或多信息源整合",
    "L4": "模糊/需澄清/应拒答——信息不足，正确行为是提问或转人工",
}


# =============================================================================
# 工具函数
# =============================================================================


def total_target() -> int:
    """矩阵里所有有效格子的目标样本总数。"""
    return sum(
        v for row in TARGET_DISTRIBUTION.values()
        for v in row.values() if v
    )


def spill_counts() -> list[tuple[str, str, int]]:
    """按目标数量降序排列的所有格子，用于生成数据的优先级排序。"""
    items = []
    for intent, row in TARGET_DISTRIBUTION.items():
        for img_type, n in row.items():
            if n:
                items.append((intent, img_type, n))
    return sorted(items, key=lambda x: -x[2])


def print_matrix():
    """打印可读的矩阵表。"""
    it_keys = [i.key for i in INTENTS]
    img_keys = [t.key for t in IMAGE_TYPES]

    w = 16
    print("=" * (14 + w * len(img_keys)))
    print("意图 × 图像类型 目标样本矩阵")
    print("=" * (14 + w * len(img_keys)))
    print(f"{'':<14}" + "".join(f"{k[:w-1]:>{w}}" for k in img_keys))
    print("-" * (14 + w * len(img_keys)))

    for ik in it_keys:
        row = TARGET_DISTRIBUTION[ik]
        cells = []
        for mk in img_keys:
            v = row.get(mk)
            cells.append(f"{'—':>{w}}" if not v else f"{v:>{w}}")
        print(f"{ik:<14}" + "".join(cells))

    print("-" * (14 + w * len(img_keys)))
    # 列合计
    totals = []
    for mk in img_keys:
        s = sum(TARGET_DISTRIBUTION[ik].get(mk, 0) or 0 for ik in it_keys)
        totals.append(f"{s:>{w}}")
    print(f"{'合计':<14}" + "".join(totals))
    print()
    print(f"目标总量: {total_target():,} 条")
    print()
    print("难度参考（1=易 5=难）:")
    for t in IMAGE_TYPES:
        bar = "█" * t.difficulty + "░" * (5 - t.difficulty)
        print(f"  {t.name:<10} {bar}  {t.description}")


def build_generation_plan() -> list[dict]:
    """生成合成数据的执行计划，按优先级排序。

    返回 [{intent, image_type, count, priority, difficulty}, ...]
    priority 越小越先做（难且量大的优先，因为风险最高）
    """
    from ..minivlm.processor import assign_bucket  # noqa: F401  (占位，实际不用)

    plan = []
    for i, (intent, img_type, count) in enumerate(spill_counts()):
        it = next(x for x in IMAGE_TYPES if x.key == img_type)
        int_t = next(x for x in INTENTS if x.key == intent)
        plan.append({
            "priority": i + 1,
            "intent": intent,
            "intent_name": int_t.name,
            "image_type": img_type,
            "image_type_name": it.name,
            "count": count,
            "difficulty": it.difficulty,
            "examples": int_t.example_queries,
        })
    return plan


def export(path: str | Path = "data/processed/taxonomy.json"):
    """导出矩阵，供 synth.py 使用。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "intents": [
            {"key": i.key, "name": i.name, "description": i.description,
             "examples": i.example_queries}
            for i in INTENTS
        ],
        "image_types": [
            {"key": t.key, "name": t.name, "description": t.description,
             "difficulty": t.difficulty}
            for t in IMAGE_TYPES
        ],
        "distribution": TARGET_DISTRIBUTION,
        "difficulty_tiers": DIFFICULTY_TIERS,
        "total_target": total_target(),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 已导出 {p}")
    return p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="打印生成计划")
    ap.add_argument("--export", action="store_true", help="导出 JSON")
    args = ap.parse_args()

    if args.plan:
        print("生成优先级（难 + 量大 优先）")
        print("=" * 84)
        print(f"{'#':>3} {'意图':<14} {'图像类型':<12} {'数量':>6} {'难度':>4}  示例")
        print("-" * 84)
        for p in build_generation_plan():
            ex = p["examples"][0] if p["examples"] else ""
            print(f"{p['priority']:>3} {p['intent_name']:<14} "
                  f"{p['image_type_name']:<12} {p['count']:>6} "
                  f"{'★' * p['difficulty']:<10} {ex}")
    elif args.export:
        export()
    else:
        print_matrix()
        print()
        print("下一步（Day 8 的任务）：")
        print("  1. 去真实电商平台把每个格子的真实问题抄 3-5 条，补进 taxonomy.py")
        print("  2. python -m src.data.taxonomy --export")
        print("  3. 然后才能进入 Day 9 的数据合成")
