"""
客服领域评测集构建。

对应讲义 docs/08-evaluation.md 第 3–4 节，对应计划 Day 20。

**这是整个 8 周最重要的资产。** 没有它，后面所有训练都是盲调。

四条构建原则：
  ① 来自真实分布，不是想出来的
  ② 四层难度（L1 单图单事实 → L4 应拒答）
  ③ 答案形态可比（可判定的走规则，不可判定的走 judge）
  ④ 防泄漏（与训练集做 pHash 交叉检查）
"""

from __future__ import annotations

import json
import random
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 评测样本结构
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    id: str
    query: str
    images: list[str]
    difficulty: str                  # L1 / L2 / L3 / L4
    intent: str
    image_type: str

    # 判定依据（三选一或组合）
    reference: Optional[str] = None          # 参考答案（judge 用）
    must_contain: list[str] = field(default_factory=list)   # 必须出现的关键要素
    must_not_contain: list[str] = field(default_factory=list)
    expected_intent: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_slots: dict = field(default_factory=dict)
    should_refuse: bool = False              # L4：正确行为是澄清/转人工
    visible_objects: list[str] = field(default_factory=list)  # 幻觉检测用
    phantom_objects: list[str] = field(default_factory=list)  # 图中不存在的东西

    notes: str = ""

    def to_json(self) -> str:
        from dataclasses import asdict
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# L1 单图单事实：图里直接能看到
# ---------------------------------------------------------------------------
#
# ⚠️ must_contain 里必须写**字面关键词**，不能写「类别描述」。
#    判分函数 `metrics.check_must_contain` 做的是**子串匹配**：
#        hits = [k for k in must_contain if k in answer]
#    写「颜色类关键词」的话，字面上永远不会出现在任何回答里 →
#    命中数恒为 0 → **这批样本对任何模型都判失败**。
#    后果是报告上「规则通过率」偏低，看起来像「模型不行」，
#    实际上是评测集自己坏了 —— 这类错误不会有异常、不会有告警。
#    修完之后 `_lint_templates()` 会拦住这种写法（见文件末的自检）。
#
#    语义：`must_contain` 是**任一命中即通过**（不是全部要命中），
#    所以要表达「长袖或短袖都可以」就写成 ["长袖", "短袖"]。

L1_TEMPLATES = {
    "product_main": [
        # 颜色：把店铺实际在售的颜色都列上，任一命中即可
        ("这件是什么颜色？",
         ["黑", "白", "米", "蓝", "灰", "驼", "红", "绿", "粉"], []),
        ("这个是什么款式？",
         ["圆领", "翻领", "高领", "直筒", "宽松", "紧身", "A 字", "高腰",
          "短款", "长款", "版型", "款式"], []),
        ("衣服上有图案吗？",
         ["有", "没有", "图案", "纯色", "条纹", "格子", "印花", "碎花"], []),
        ("这是长袖还是短袖？",
         ["长袖", "短袖", "中袖", "袖子"], []),
    ],
    "size_chart": [
        ("M 码的胸围是多少？",
         ["胸围", "cm", "厘米", "M"], []),
        ("L 码衣长多少厘米？",
         ["衣长", "cm", "厘米", "L"], []),
        ("尺码表里最大是哪个码？",
         ["XL", "XXL", "XXXL", "码"], []),
    ],
    "model_wearing": [
        ("模特穿的是哪个尺码？",
         ["尺码", "S", "M", "L", "XL", "码"], []),
        ("模特身高大概多少？",
         ["身高", "cm", "厘米", "模特"], []),
    ],
}

# ---------------------------------------------------------------------------
# L2 单图推理：需要结合常识或政策
# ---------------------------------------------------------------------------

L2_TEMPLATES = {
    "quality_issue": [
        ("图片里这个线头算质量问题吗？",
         ["正常", "工艺", "不影响", "轻微"], ["严重质量问题", "必须退货"]),
        ("这里的走线正常吗？",
         ["正常", "工艺范围", "常见"], ["严重缺陷"]),
    ],
    "color_mismatch": [
        ("实物和图片颜色差得多吗？",
         ["色差", "光线", "显示器"], ["完全一致", "绝对没有色差"]),
    ],
    "material": [
        ("从图片看这个面料会不会起球？",
         ["根据", "标注", "无法", "建议"], ["绝对不会", "保证不起球"]),
    ],
}

# ---------------------------------------------------------------------------
# L3 多图 / 图+文冲突
# ---------------------------------------------------------------------------

L3_TEMPLATES = [
    ("我收到的和图片上这个不太一样，你看是同一款吗？",
     ["对比", "差异", "款式", "颜色"], ["完全一样"]),
    ("这三张图哪个颜色更接近实物？",
     ["第一", "第二", "第三", "接近", "光线"], []),
]

# ---------------------------------------------------------------------------
# L4 应拒答 / 需澄清
# ---------------------------------------------------------------------------

L4_TEMPLATES = [
    ("这个好看吗？", True),
    ("我穿 S 行不行？", True),                # 没说身高体重
    ("能给我退款 500 块吗？", True),           # 索赔，超范围
    ("你们这是不是欺诈？我要投诉！", True),    # 情绪激烈 + 投诉
    ("这个和我在别家买的哪个好？", False),     # 其实可以答（给建议）
]


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------


def build_eval_set(
    source_samples: list[dict],
    n_target: int = 320,
    out_path: str | Path = "data/eval/cx_eval_v1.jsonl",
    version: str = "v1",
    seed: int = 42,
) -> dict:
    """从已有数据里构造评测集。

    切分比例（对应难度分层）：
      L1 30%  L2 35%  L3 25%  L4 10%

    注意：L4 的样本大部分需要**人工写**，因为它们要精心设计「信息不足」的场景。
    这里给的是半自动生成 + 模板，你必须人工复核一遍。
    """
    random.seed(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quotas = {"L1": int(n_target * 0.30), "L2": int(n_target * 0.35),
              "L3": int(n_target * 0.25), "L4": n_target - int(n_target * 0.90)}

    # 按 image_type 分组源数据（L1 用）
    by_type: dict[str, list[dict]] = {}
    for s in source_samples:
        by_type.setdefault(s.get("image_type", "?"), []).append(s)

    # ⚠️ L2 的模板是按 **intent** 分组的（「线头算质量问题吗」这类问题问的是
    #    **话题**，不是「图是什么类型」）。所以 L2 必须查 by_intent。
    #    早期版本给 L2 也用了 by_type，而 L2_TEMPLATES 的键是
    #    quality_issue / color_mismatch / material 这些 intent 名 ——
    #    它们在 image_type 里一个都不存在，`by_type.get(...)` 全返回空，
    #    于是 **L2 整整一层恒为 0 条**。
    #    报告上只表现为「L2: 0」一行，看上去像个正常的空桶，
    #    实际上意味着 35% 的评测样本从来没被构造出来。
    by_intent: dict[str, list[dict]] = {}
    for s in source_samples:
        by_intent.setdefault(s.get("intent", "?"), []).append(s)

    samples: list[EvalSample] = []
    sid = 0

    # --- L1 ---
    for img_type, tmpls in L1_TEMPLATES.items():
        pool = by_type.get(img_type, [])
        if not pool:
            continue
        per_t = max(1, quotas["L1"] // max(len(L1_TEMPLATES), 1) // max(len(tmpls), 1))
        for q, must, forbid in tmpls:
            for src in random.sample(pool, min(per_t, len(pool))):
                samples.append(EvalSample(
                    id=f"{version}_L1_{sid:04d}",
                    query=q,
                    images=_imgs_of(src),
                    difficulty="L1",
                    intent=src.get("intent", "?"),
                    image_type=img_type,
                    must_contain=must,
                    must_not_contain=forbid,
                    visible_objects=[src.get("image_type", "")],
                ))
                sid += 1

    # --- L2 ---
    for intent, tmpls in L2_TEMPLATES.items():
        pool = by_intent.get(intent, [])
        if not pool:
            print(f"  ⚠️ L2 模板 '{intent}' 在源数据里没有对应样本，"
                  f"该模板产出 0 条")
            continue
        per_t = max(1, quotas["L2"] // max(len(L2_TEMPLATES), 1) // max(len(tmpls), 1))
        for q, must, forbid in tmpls:
            for src in random.sample(pool, min(per_t, len(pool))):
                samples.append(EvalSample(
                    id=f"{version}_L2_{sid:04d}",
                    query=q,
                    images=_imgs_of(src),
                    difficulty="L2",
                    intent=intent,
                    image_type=src.get("image_type", "?"),
                    must_contain=must,
                    must_not_contain=forbid,
                ))
                sid += 1

    # --- L3 ---
    for q, must, forbid in L3_TEMPLATES:
        pool = [s for s in source_samples if len(_imgs_of(s)) >= 1]
        # L3 需要多图，尽量找有多图的样本
        multi = [s for s in source_samples if len(s.get("images", [])) >= 2]
        use = multi if multi else pool
        per_t = max(1, quotas["L3"] // max(len(L3_TEMPLATES), 1))
        for src in random.sample(use, min(per_t, len(use))):
            samples.append(EvalSample(
                id=f"{version}_L3_{sid:04d}",
                query=q,
                images=_imgs_of(src),
                difficulty="L3",
                intent=src.get("intent", "?"),
                image_type="user_compare",
                must_contain=must,
                must_not_contain=forbid,
            ))
            sid += 1

    # --- L4（模板 + 人工复核）---
    pool = source_samples
    for q, should_refuse in L4_TEMPLATES:
        for _ in range(max(1, quotas["L4"] // len(L4_TEMPLATES))):
            src = random.choice(pool)
            samples.append(EvalSample(
                id=f"{version}_L4_{sid:04d}",
                query=q,
                images=_imgs_of(src),
                difficulty="L4",
                intent="presale_general" if "好看" in q else "return_refund",
                image_type=src.get("image_type", "?"),
                should_refuse=should_refuse,
                must_contain=["方便告诉我", "需要确认", "转接", "无法确定",
                              "建议您提供"] if should_refuse else [],
                notes="⚠️ 需人工复核：确认这个问题的正确行为",
            ))
            sid += 1

    dist = Counter(s.difficulty for s in samples)

    # ⚠️ 每个难度层都必须有样本。
    #    L2 曾经恒为 0 条（模板按 intent 分组、却拿 image_type 去查池子），
    #    而报告上「L2: 0」看起来只是个正常的空桶 —— 35% 的评测样本
    #    就这样静默地没了。空层必须是**硬错误**，不能只是一行统计。
    empty_levels = [k for k in ("L1", "L2", "L3", "L4") if dist.get(k, 0) == 0]
    if empty_levels:
        raise SystemExit(
            f"✗ 这些难度层一条样本都没构造出来：{empty_levels}\n"
            f"  常见原因：模板的分组键（intent / image_type）和源数据的字段对不上，\n"
            f"  于是查到的候选池为空、整层被跳过。检查对应 *_TEMPLATES 的键。")

    # 落盘
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json() + "\n")

    stats = {
        "n_total": len(samples),
        "difficulty": dict(dist),
        "intent": dict(Counter(s.intent for s in samples)),
        "image_type": dict(Counter(s.image_type for s in samples)),
        "out_path": str(out_path),
    }

    print(f"✓ 写出 {len(samples)} 条评测样本 → {out_path}")
    print(f"\n难度分布:")
    for k in ("L1", "L2", "L3", "L4"):
        v = dist.get(k, 0)
        bar = "█" * int(40 * v / max(len(samples), 1))
        print(f"  {k}: {v:>4}  {bar}")

    print(f"\n⚠️  L4 样本必须人工复核！({dist.get('L4', 0)} 条)")
    print("   自动生成只能给出场景模板，正确行为的判定需要你自己确认。")

    return stats


def _imgs_of(src: dict) -> list[str]:
    imgs = src.get("images")
    if imgs:
        return imgs
    p = src.get("image_path")
    return [p] if p else []


# ---------------------------------------------------------------------------
# 模板自检
# ---------------------------------------------------------------------------

# 这些词出现在 must_contain 里就说明写的是「类别描述」而不是字面关键词。
# 判分是子串匹配，描述性文字永远命中不了 → 该条对任何模型都判失败。
_META_MARKERS = ("关键词", "类的词", "任意", "一个词", "相关词")


def _lint_templates(verbose: bool = True) -> list[str]:
    """静态检查所有模板，返回问题列表（空 = 通过）。

    能拦住的问题：
      1. `must_contain` 里写「类别描述」而不是字面关键词
         —— 子串匹配永远命不中，样本对任何模型都判失败
      2. `must_contain` 与 `must_not_contain` 有交集
         —— 同一个词既要求出现又禁止出现，该条永远不可能通过
      3. 空 query / query 过短
    """
    problems: list[str] = []

    def check(q: str, must: list[str], forbid: list[str], where: str) -> None:
        if not q or len(q) < 3:
            problems.append(f"{where}: query 过短或为空：{q!r}")
        for k in must:
            if any(m in k for m in _META_MARKERS):
                problems.append(
                    f"{where}: must_contain 里是**类别描述**而不是字面关键词：{k!r}"
                    f" —— 判分是子串匹配，这永远命不中")
            if not k.strip():
                problems.append(f"{where}: must_contain 里有空字符串")
        overlap = set(must) & set(forbid)
        if overlap:
            problems.append(
                f"{where}: must_contain 与 must_not_contain 有交集 {overlap}"
                f" —— 既要求出现又禁止出现，该条不可能通过")

    for t, tmpls in L1_TEMPLATES.items():
        for q, must, forbid in tmpls:
            check(q, must, forbid, f"L1[{t}]")
    for t, tmpls in L2_TEMPLATES.items():
        for q, must, forbid in tmpls:
            check(q, must, forbid, f"L2[{t}]")
    for i, (q, must, forbid) in enumerate(L3_TEMPLATES):
        check(q, must, forbid, f"L3[{i}]")
    for q, should_refuse in L4_TEMPLATES:
        if not q or len(q) < 3:
            problems.append(f"L4: query 过短或为空：{q!r}")

    if verbose and not problems:
        print("  ✓ 模板静态检查通过（关键词都是字面量、无自相矛盾）")
    return problems


def _selftest() -> int:
    """离线自检：模板静态检查 + 端到端构造（不依赖真实图片）。"""
    print("=" * 76)
    print("评测集构造自检（离线，不读写真实数据）")
    print("=" * 76)

    print("\n[1] 模板静态检查 _lint_templates()")
    probs = _lint_templates()
    for p in probs:
        print(f"    ✗ {p}")
    assert not probs, f"模板有问题：{probs}"
    print("    ✓ 无问题")

    print("\n[2] check_must_contain 的语义：类别描述永远命不中")
    from .metrics import check_must_contain
    r = check_must_contain(
        "从图片看这是米白色的纯棉圆领T恤，版型宽松。", ["颜色类关键词"])
    assert not r.passed, (
        "「颜色类关键词」不该被判通过 —— 如果通过了，说明判分做了语义匹配，"
        "那么 must_contain 的写法约定需要重新定义")
    r2 = check_must_contain(
        "从图片看这是米白色的纯棉圆领T恤，版型宽松。",
        ["黑", "米", "蓝", "圆领", "宽松"])
    assert r2.passed, "正常回答应该命中「米」和「圆领」"
    print(f"    类别描述 → passed={r.passed} （{r.detail[:40]}）")
    print(f"    字面关键词 → passed={r2.passed} （{r2.detail[:40]}）")
    print("    ✓ 修完的模板用字面关键词，能被正常回答命中")

    print("\n[3] 端到端构造：每个难度层都必须非空")
    # 造一份最小的源数据，覆盖 L1 需要的 image_type 和 L2 需要的 intent
    src = []
    for i, imt in enumerate(["product_main", "size_chart", "model_wearing",
                             "detail_closeup", "defect_photo", "user_compare"]):
        for k in range(3):
            src.append({
                "id": f"s{i}_{k}", "image_path": f"img_{imt}_{k}.jpg",
                "image_type": imt,
                "intent": ["quality_issue", "color_mismatch", "material",
                           "size_fit", "return_refund", "logistics"][i % 6],
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "path": f"img_{imt}_{k}.jpg"},
                        {"type": "text", "text": "这个问题够长了吧"}]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "这是一段足够长的示例回复内容，用于占位。"}]},
                ],
            })
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.jsonl"
        stats = build_eval_set(src, n_target=40, out_path=out, seed=1)
        dist = stats["difficulty"]
        print(f"    难度分布 {dist}")
        for k in ("L1", "L2", "L3", "L4"):
            assert dist.get(k, 0) > 0, (
                f"{k} 层 0 条 —— 这一层等于没构造。"
                f"检查 {k}_TEMPLATES 的分组键和源数据字段是否对得上")
        # 每一条的 must_contain 都不能是类别描述
        rows = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l]
        for r_ in rows:
            for k in r_.get("must_contain", []):
                assert not any(m in k for m in _META_MARKERS), \
                    f"{r_['id']} 的 must_contain 含类别描述：{k!r}"
        print(f"    ✓ 4 层都有样本，{len(rows)} 条的关键词全是字面量")

    print("\n" + "=" * 76)
    print("✓ 全部通过（离线）")
    return 0


# ---------------------------------------------------------------------------
# 评测集卡片
# ---------------------------------------------------------------------------

EVAL_CARD = """# Eval Set Card: {name}

## 规模与分布

| 难度 | 数量 | 占比 | 说明 |
|---|---:|---:|---|
{L1} | {n1} | {p1:.1%} | 单图单事实，图里直接可见 |
{L2} | {n2} | {p2:.1%} | 单图推理，需结合常识或政策 |
{L3} | {n3} | {p3:.1%} | 多图/图文冲突，需对比整合 |
{L4} | {n4} | {p4:.1%} | 信息不足，正确行为是澄清或转人工 |

总计 **{total}** 条

## 意图分布
{intent_table}

## 图像类型分布
{image_table}

## 判定方式

| 维度 | 方式 | 说明 |
|---|---|---|
| 意图分类 | 规则精确匹配 | 可判定 |
| 要素包含 | must_contain / must_not_contain | 可判定 |
| 拒答行为 | 规则 + 关键词 | 可判定 |
| 幻觉 | 检测 phantom_objects 是否被提及 | 可判定 |
| 事实性/帮助性/语气 | LLM-as-judge (1-5) | 需校准 |

## 泄漏检查

{eakage}

## 冻结声明

本评测集为 **{name}**，**一旦冻结不可修改**。
修改评测集等于换尺子，历史数据会失去可比性。
新增样本请开新版本（v2、v3...）。

## 已知局限

- L4 样本来自模板，场景多样性有限
- 参考答案只覆盖「必须提到的要素」，不穷举所有正确回答
- 未覆盖方言/错别字/极短输入等真实噪声
"""


def generate_eval_card(eval_path: str | Path = "data/eval/cx_eval_v1.jsonl",
                       out_path: str | Path = "data/eval/EVAL_CARD.md",
                       name: str = "cx_eval_v1",
                       train_path: str | Path = "data/processed/sft_train.jsonl"):
    p = Path(eval_path)
    samples = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    dist = Counter(s["difficulty"] for s in samples)
    total = len(samples)

    # 泄漏检查
    leak_md = "（未执行——建议运行 `python -m src.data.build_sft` 时开启 check_leak）"
    tp = Path(train_path)
    if tp.exists():
        try:
            train = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
            from ..data.build_sft import check_leakage
            lk = check_leakage(train, samples)
            if lk["n_leaks"] == 0:
                leak_md = f"✅ 无泄漏（检查了 {lk['n_eval_images']} 张评测图 vs {lk['n_train_images']} 张训练图）"
            else:
                leak_md = (f"❌ **发现 {lk['n_leaks']} 处泄漏**（{lk['leak_rate']:.1%}）\n\n"
                           + "\n".join(f"- `{e['eval_image']}` ~ `{e['train_image']}` (d={e['distance']})"
                                       for e in lk["examples"][:5])
                           + "\n\n**必须处理后才能用来评测**")
        except Exception as e:      # noqa: BLE001
            leak_md = f"（检查失败：{e}）"

    def fmt_counter(c: Counter):
        rows = []
        for k, v in sorted(c.items(), key=lambda x: -x[1]):
            rows.append(f"| {k} | {v} | {v / total:.1%} |")
        return "\n".join(rows) or "| - | - | - |"

    content = EVAL_CARD.format(
        name=name, total=total,
        L1="L1", L2="L2", L3="L3", L4="L4",
        n1=dist.get("L1", 0), n2=dist.get("L2", 0),
        n3=dist.get("L3", 0), n4=dist.get("L4", 0),
        p1=dist.get("L1", 0) / total, p2=dist.get("L2", 0) / total,
        p3=dist.get("L3", 0) / total, p4=dist.get("L4", 0) / total,
        intent_table=fmt_counter(Counter(s["intent"] for s in samples)),
        image_table=fmt_counter(Counter(s["image_type"] for s in samples)),
        eakage=leak_md,
    )
    # 修掉模板里的占位名（避免 format 冲突）
    content = content.replace("{L1} |", "| L1 |").replace("{L2} |", "| L2 |") \
                     .replace("{L3} |", "| L3 |").replace("{L4} |", "| L4 |")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"✓ 评测集卡片 → {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构建客服领域评测集")
    ap.add_argument("--source", default="data/processed/clean.jsonl")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", default="data/eval/cx_eval_v1.jsonl")
    ap.add_argument("--card", action="store_true", help="生成评测集卡片")
    ap.add_argument("--selftest", action="store_true",
                    help="离线自检：模板静态检查 + 四层都非空")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    if args.card:
        generate_eval_card(args.out)
    else:
        sp = Path(args.source)
        if not sp.exists():
            print(f"✗ 找不到 {sp}")
            print("  请先跑完 Day 12 的数据集构建流程。")
            raise SystemExit(1)
        src = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        build_eval_set(src, args.n, args.out)
        generate_eval_card(args.out)
