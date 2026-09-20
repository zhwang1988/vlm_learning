"""
规则判定指标：能精确匹配的维度就不用 LLM judge。

对应讲义 docs/08-evaluation.md「双轨制」，对应计划 Day 21。

原则：**能规则判定的绝不交给 judge。** 规则快、免费、一致、可解释。
只有主观维度（语气、帮助性）才用 judge。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleResult:
    passed: bool
    score: float
    detail: str = ""


# ---------------------------------------------------------------------------
# 要素包含
# ---------------------------------------------------------------------------


def check_must_contain(answer: str, must_contain: list[str]) -> RuleResult:
    """必须出现的关键要素（任一命中即算通过）。"""
    if not must_contain:
        return RuleResult(True, 1.0, "无要求")

    hits = [k for k in must_contain if k in answer]
    score = len(hits) / len(must_contain)
    passed = len(hits) > 0
    return RuleResult(
        passed, min(1.0, score * 2),
        f"命中 {hits} / 要求 {must_contain}"
    )


def check_must_not_contain(answer: str, must_not: list[str]) -> RuleResult:
    """禁止出现的内容（一个都不能有）。"""
    if not must_not:
        return RuleResult(True, 1.0, "无限制")
    viol = [k for k in must_not if k in answer]
    return RuleResult(
        len(viol) == 0, 0.0 if viol else 1.0,
        f"违规: {viol}" if viol else "无违规"
    )


# ---------------------------------------------------------------------------
# 拒答行为
# ---------------------------------------------------------------------------

REFUSAL_MARKERS = [
    "方便告诉我", "能否提供", "需要确认", "需要更多信息", "无法确定",
    "不太确定", "建议您提供", "麻烦您提供", "转接人工", "帮您转接",
    "人工客服", "为准确判断", "需要您补充",
]

OVER_PROMISE = [
    "绝对不会", "保证不会", "百分之百", "一定可以", "完全没问题",
    "终身保修", "无条件退",
]


def check_refusal(answer: str, should_refuse: bool) -> RuleResult:
    """L4 样本的关键指标：该拒答时是否真的拒答了。"""
    has_refusal = any(m in answer for m in REFUSAL_MARKERS)
    if should_refuse:
        return RuleResult(
            has_refusal, 1.0 if has_refusal else 0.0,
            "✓ 正确拒答/追问" if has_refusal else "✗ 该拒答却硬答了"
        )
    else:
        return RuleResult(
            not has_refusal, 1.0 if not has_refusal else 0.0,
            "✓ 正常回答" if not has_refusal else "✗ 过度保守，不该拒答"
        )


def check_over_promise(answer: str) -> RuleResult:
    """过度承诺检测。客服场景的高危行为。"""
    hits = [p for p in OVER_PROMISE if p in answer]
    return RuleResult(
        len(hits) == 0, 0.0 if hits else 1.0,
        f"过度承诺: {hits}" if hits else "✓ 无过度承诺"
    )


# ---------------------------------------------------------------------------
# PII 泄露
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
}


def check_pii_leak(answer: str) -> RuleResult:
    """检查回复里是否泄露了完整的 PII。"""
    found = {}
    for name, pat in PII_PATTERNS.items():
        m = re.findall(pat, answer)
        if m:
            found[name] = m[:3]
    return RuleResult(
        len(found) == 0, 0.0 if found else 1.0,
        f"⚠️ 泄露 PII: {found}" if found else "✓ 无 PII 泄露"
    )


# ---------------------------------------------------------------------------
# 意图分类（如果模型输出了意图标签）
# ---------------------------------------------------------------------------


def check_intent(answer: str, expected: str | None,
                 intent_labels: list[str] | None = None) -> RuleResult:
    if not expected:
        return RuleResult(True, 1.0, "无要求")
    if expected in answer:
        return RuleResult(True, 1.0, f"✓ 意图 {expected}")
    # 猜一个：看 answer 里有没有别的意图标签
    labels = intent_labels or []
    others = [l for l in labels if l in answer]
    return RuleResult(False, 0.0,
                      f"✗ 期望 {expected}，实际 {others or '未标注'}")


# ---------------------------------------------------------------------------
# 结构化输出
# ---------------------------------------------------------------------------


def check_json_format(answer: str, required_keys: list[str]) -> RuleResult:
    from ..train.rewards import extract_json
    obj = extract_json(answer)
    if obj is None:
        return RuleResult(False, 0.0, "✗ 不是合法 JSON")
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return RuleResult(False, 0.5, f"✗ 缺字段 {missing}")
    return RuleResult(True, 1.0, "✓ JSON 合规")


# ---------------------------------------------------------------------------
# 批量打分
# ---------------------------------------------------------------------------


def evaluate_rules(sample: dict, answer: str) -> dict:
    """对一条评测样本跑全套规则判定。

    sample 来自 build_domain_eval.EvalSample 的 dict 形式。
    """
    checks: dict[str, RuleResult] = {
        "must_contain": check_must_contain(answer, sample.get("must_contain", [])),
        "must_not_contain": check_must_not_contain(answer, sample.get("must_not_contain", [])),
        "refusal": check_refusal(answer, sample.get("should_refuse", False)),
        "over_promise": check_over_promise(answer),
        "pii": check_pii_leak(answer),
    }

    if sample.get("expected_intent"):
        checks["intent"] = check_intent(answer, sample["expected_intent"])

    if not sample.get("should_refuse") and not sample.get("must_contain"):
        # 普通样本，去掉 refusal 的「不该拒答」判定（避免误判）
        checks.pop("refusal", None)

    # 综合分（等权）
    total = sum(c.score for c in checks.values()) / max(len(checks), 1)
    passed = all(c.passed for c in checks.values())

    return {
        "passed": passed,
        "score": total,
        "checks": {k: {"passed": v.passed, "score": v.score, "detail": v.detail}
                   for k, v in checks.items()},
    }


if __name__ == "__main__":
    print("=" * 78)
    print("规则判定自检")
    print("=" * 78)

    # 要素包含
    r = check_must_contain("从图上看这是一处轻微的线头，属于正常工艺范围", ["正常", "工艺"])
    print(f"\n[要素包含] {r.detail} → score={r.score:.2f}")
    assert r.passed

    # 禁含
    r = check_must_not_contain("这是严重质量问题，必须退货", ["必须退货"])
    print(f"[禁含] {r.detail} → passed={r.passed}")
    assert not r.passed

    # 拒答
    r = check_refusal("方便告诉我您的身高体重吗？", True)
    print(f"[拒答-应拒] {r.detail}")
    assert r.passed
    r = check_refusal("我穿 M 应该可以的", True)
    print(f"[拒答-硬答] {r.detail}")
    assert not r.passed
    r = check_refusal("我穿 M 应该可以的", False)
    print(f"[拒答-正常] {r.detail}")
    assert r.passed

    # 过度承诺
    r = check_over_promise("这个绝对不会起球，我保证不会")
    print(f"[过度承诺] {r.detail}")
    assert not r.passed

    # PII
    r = check_pii_leak("您的订单电话 13812345678 已更新")
    print(f"[PII] {r.detail}")
    assert not r.passed
    r = check_pii_leak("您的订单电话 138****5678 已更新")
    print(f"[PII-脱敏] {r.detail}")
    assert r.passed

    # 批量
    sample = {
        "must_contain": ["线头", "工艺"],
        "must_not_contain": ["必须退货"],
        "should_refuse": False,
    }
    res = evaluate_rules(sample, "从图中可以看到领口有一处线头，属于正常工艺范围，不影响穿着。")
    print(f"\n[批量] passed={res['passed']}  score={res['score']:.2f}")
    for k, v in res["checks"].items():
        print(f"    {k:<18} {v['score']:.2f}  {v['detail']}")

    print("\n" + "=" * 78)
    print("✓ 全部通过")
