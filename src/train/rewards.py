"""
可验证奖励函数（GRPO 用）。

对应讲义 docs/07-alignment.md 第 5–6 节，对应计划 Day 28。

核心思路：**让「编造」在数学上真的亏。**

客服场景下，可验证奖励比人类偏好更划算，因为：
  - 格式对不对、工具选对没、JSON 能不能解析 —— 程序能判定
  - 人类标偏好很贵，规则判定免费且一致
  - 这些恰好是 Agent 场景最需要的稳定性

五个奖励函数，加一个组合器。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 奖励 1：格式合规
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """从文本里抽出第一个合法 JSON 对象。"""
    # 直接解析
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 从 ```json 代码块里抽
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 暴力找最外层花括号
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def reward_format(completion: str, required_keys: list[str] | None = None) -> float:
    """输出是否是合法 JSON 且包含必需字段。

    required_keys 默认 ["action"] 或 ["response"]（二者有其一即可）。
    """
    required_keys = required_keys or ["action"]
    obj = extract_json(completion)
    if obj is None:
        return 0.0
    if not isinstance(obj, dict):
        return 0.0

    # 支持两种形态：工具调用 或 最终回答
    has_action = "action" in obj
    has_response = "response" in obj or "answer" in obj
    if not (has_action or has_response):
        return 0.0

    for k in required_keys:
        if k == "action" and not has_action:
            continue
        if k not in obj:
            return 0.3           # 部分合规给部分分
    return 1.0


# ---------------------------------------------------------------------------
# 奖励 2：工具选择正确
# ---------------------------------------------------------------------------


def reward_tool_call(completion: str, expected_tool: str | None,
                     allowed_tools: list[str] | None = None) -> float:
    """工具选择是否正确。

    expected_tool=None 表示这题不该调用工具（应该直接回答）。
    """
    obj = extract_json(completion)
    if obj is None:
        return 0.0

    called = obj.get("action")

    if expected_tool is None:
        # 不该调用工具却调了 → 惩罚
        return 1.0 if called in (None, "", "final_answer", "respond") else 0.0

    if allowed_tools and called not in allowed_tools and called is not None:
        return 0.0               # 调用了不存在的工具 → 工具幻觉

    return 1.0 if called == expected_tool else 0.0


# ---------------------------------------------------------------------------
# 奖励 3：参数填充正确
# ---------------------------------------------------------------------------


def reward_slot_filling(completion: str, expected_slots: dict[str, Any]) -> float:
    """工具参数是否正确填入。按正确字段的比例给分。"""
    obj = extract_json(completion)
    if obj is None:
        return 0.0
    args = obj.get("action_input") or obj.get("arguments") or {}
    if not isinstance(args, dict):
        return 0.0
    if not expected_slots:
        return 1.0

    hit = sum(1 for k, v in expected_slots.items()
              if k in args and str(args[k]).strip() == str(v).strip())
    return hit / len(expected_slots)


# ---------------------------------------------------------------------------
# 奖励 4：视觉 grounding（治幻觉的核心）
# ---------------------------------------------------------------------------


def reward_grounding(completion: str, visible_objects: set[str]) -> float:
    """提到的视觉证据是否真实存在于图中。

    visible_objects 来自一个检测器/标注，是这张图里**确实有**的东西集合。
    """
    grounded, phantom = _extract_claims(completion, visible_objects)
    mentioned = grounded | phantom
    if not mentioned:
        return 0.5               # 没提任何视觉证据，中性
    return len(grounded) / len(mentioned)


def reward_hallucination_penalty(completion: str, visible_objects: set[str],
                                 penalty: float = 1.0) -> float:
    """提到图中不存在的东西 → 扣分。

    **这是治幻觉最直接的手段。**
    模型在 RL 里会发现：编造细节会稳定地降低总奖励，
    于是学会「不确定就说不确定」。
    """
    _, phantom = _extract_claims(completion, visible_objects)
    if not phantom:
        return 0.0               # 显式返回 0.0，避免 -1.0 * 0 印出 "-0.00"
    return -penalty * len(phantom)


# 可直接判定的视觉属性词表（颜色 / 材质 / 图案 / 瑕疵 / 版型）。
#
# ⚠️ 为什么刻意**不放**「袖子 / 领口 / 下摆 / 口袋」这类通用部位词：
#    针织衫天然有袖子，模型说一句「袖子偏长」是主观判断，不是编造实体。
#    把部位词也算成幻觉会大面积误伤，奖励信号变得又吵又假。
VISUAL_ATTR_WORDS: frozenset[str] = frozenset({
    # 颜色
    "黑色", "白色", "米白", "米白色", "灰色", "浅灰", "深灰",
    "红色", "酒红", "粉色", "藕粉", "橙色", "黄色", "绿色", "墨绿", "军绿",
    "蓝色", "浅蓝色", "深蓝色", "藏青", "紫色", "棕色", "卡其", "驼色",
    "杏色", "香槟",
    # 材质
    "纯棉", "棉质", "亚麻", "羊毛", "羊绒", "真丝", "丝绸", "涤纶",
    "牛仔", "皮革", "针织", "雪纺", "蕾丝", "摇粒绒", "灯芯绒",
    # 图案 / 工艺
    "条纹", "格子", "印花", "纯色", "波点", "刺绣", "镂空", "拼接", "扎染",
    # 瑕疵（客服场景最关心的一类）
    "破损", "破洞", "污渍", "线头", "起球", "褪色", "掉色", "开线",
    "变形", "色差", "勾丝", "抽丝", "瑕疵", "污点", "霉斑",
    # 版型 / 规格
    "圆领", "翻领", "高领", "一字领", "方领", "立领", "V领", "v领",
    "长袖", "短袖", "无袖", "七分袖", "泡泡袖",
    "修身", "宽松", "紧身", "直筒", "阔腿", "中长款", "加长", "九分",
})


def _extract_claims(text: str, visible_objects) -> tuple[set[str], set[str]]:
    """抽出回复里的视觉断言，分成 (有据的, 编造的)。

    判定规则：
      ① visible_objects 里已知的物体被原文提到 → 有据（并且把这段文本消费掉，
         避免它又被词表的子串重复判定一次）
      ② 词表里的属性词出现在原文 → 查 visible 里有没有对应物体：
         有 → 有据；没有 → 编造

    ⚠️ 为什么要「按词长降序 + 消费掉已匹配文本」：
       "浅蓝色" 和 "蓝色" 都在词表里。如果按集合顺序遍历，短词 "蓝色"
       可能先把 "浅蓝色" 里的 "蓝色" 匹配走，导致 visible={"深蓝色裤"}
       时把 "浅蓝色" 误判成有据。长词优先匹配可以先占位，短词随后就找不到了。

    ⚠️ 已知局限（真实工程必须补）：
       · 否定句会误判 —— 「这**不**是浅蓝色」会被当成提到了浅蓝色。
         生产环境要在抽取前做一层否定/反问句过滤。
       · 同义词没做归并 —— 「米白」和「米白色」、「破了个洞」和「破洞」。
         生产环境应该用商品属性词表 + 同义词归一，而不是硬编码集合。
       这里保持最小可用，是为了让奖励函数的逻辑一眼看得懂。
    """
    visible = {str(o).strip() for o in (visible_objects or set())
               if str(o).strip()}
    grounded: set[str] = set()
    phantom: set[str] = set()

    # ① 先在原文里找已知物体，命中就消费掉这段文本
    remaining = text
    for obj in sorted(visible, key=len, reverse=True):
        if obj in remaining:
            grounded.add(obj)
            remaining = remaining.replace(obj, "\x00")

    # ② 再用词表扫剩余文本（长词优先）
    for w in sorted(VISUAL_ATTR_WORDS, key=len, reverse=True):
        if w not in remaining:
            continue
        remaining = remaining.replace(w, "\x00")
        if any(w in obj or obj in w for obj in visible):
            grounded.add(w)
        else:
            phantom.add(w)

    return grounded, phantom


# ---------------------------------------------------------------------------
# 奖励 5：拒答正确性
# ---------------------------------------------------------------------------


REFUSAL_MARKERS = [
    "方便告诉我", "能否提供", "需要确认", "无法确定", "不太确定",
    "建议您提供", "转接人工", "帮您转接", "需要更多信息",
]


def reward_refusal(completion: str, should_refuse: bool) -> float:
    """应该拒答/澄清时是否真的拒答了。

    这是 L4 难度样本的关键指标。基座模型在这里几乎必错（会硬答）。
    """
    has_refusal = any(m in completion for m in REFUSAL_MARKERS)
    if should_refuse:
        return 1.0 if has_refusal else 0.0
    # 不该拒答却拒答 → 惩罚（过度保守也是问题）
    return -0.5 if has_refusal else 1.0


# ---------------------------------------------------------------------------
# 组合器
# ---------------------------------------------------------------------------


@dataclass
class RewardWeights:
    format: float = 1.0
    tool: float = 1.0
    slot: float = 0.5
    grounding: float = 0.8
    hallucination: float = 1.0
    refusal: float = 0.5
    length_penalty: float = 0.1
    unnecessary_call: float = 0.5

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RewardContext:
    """判一条 completion 好不好所需要的全部信息。"""
    expected_tool: str | None = None
    expected_slots: dict = field(default_factory=dict)
    allowed_tools: list[str] | None = None
    visible_objects: set = field(default_factory=set)
    should_refuse: bool = False
    target_length: int = 150


def compute_reward(completion: str, ctx: RewardContext,
                   w: RewardWeights | None = None) -> dict:
    """组合奖励。返回各项明细 + 总分（方便 debug 是哪个奖励在起作用）。"""
    w = w or RewardWeights()

    parts = {
        "format": reward_format(completion) * w.format,
        "tool": reward_tool_call(completion, ctx.expected_tool,
                                 ctx.allowed_tools) * w.tool,
        "slot": reward_slot_filling(completion, ctx.expected_slots) * w.slot,
        "grounding": reward_grounding(completion, ctx.visible_objects) * w.grounding,
        "hallucination": reward_hallucination_penalty(
            completion, ctx.visible_objects) * w.hallucination,
        "refusal": reward_refusal(completion, ctx.should_refuse) * w.refusal,
    }

    # 长度惩罚：偏离目标长度太多扣分
    n = len(completion)
    over = max(0, n - ctx.target_length * 2)
    parts["length"] = -w.length_penalty * (over / 100)

    total = sum(parts.values())
    return {"total": total, **parts}


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def run_tests():
    print("=" * 78)
    print("可验证奖励函数自检")
    print("=" * 78)

    # 格式
    assert reward_format('{"action": "lookup_order", "action_input": {}}') == 1.0
    assert reward_format("我帮你查一下订单，请稍等") == 0.0
    assert reward_format('```json\n{"action": "x"}\n```') == 1.0
    print("\n[1] 格式奖励 ✓")

    # 工具选择
    assert reward_tool_call('{"action": "lookup_order"}', "lookup_order") == 1.0
    assert reward_tool_call('{"action": "wrong_tool"}', "lookup_order",
                            allowed_tools=["lookup_order", "check_stock"]) == 0.0
    assert reward_tool_call('{"response": "这是米白色"}', None) == 1.0
    assert reward_tool_call('{"action": "lookup_order"}', None) == 0.0
    print("[2] 工具选择奖励 ✓")

    # 槽位
    s = reward_slot_filling(
        '{"action": "check_stock", "action_input": {"sku_id": "A123", "size": "M"}}',
        {"sku_id": "A123", "size": "M"},
    )
    assert s == 1.0
    print("[3] 槽位奖励 ✓")

    # grounding + 幻觉
    visible = {"浅蓝色上衣", "线头", "圆领"}
    good = "从图中可以看到这是一件浅蓝色上衣，领口处有一处线头，属于圆领款式。"
    bad = "图中显示这件衣服左侧口袋有明显破损，袖子也偏长。"

    g_good = reward_grounding(good, visible)
    g_bad = reward_grounding(bad, visible)
    h_good = reward_hallucination_penalty(good, visible)
    h_bad = reward_hallucination_penalty(bad, visible)

    print(f"\n[4] Grounding / 幻觉")
    print(f"  正常回复: grounding={g_good:.2f}  hallucination={h_good:.2f}")
    print(f"  幻觉回复: grounding={g_bad:.2f}  hallucination={h_bad:.2f}")
    assert h_bad < h_good, "幻觉回复的幻觉惩罚必须更重"
    assert g_good > g_bad, "有据回复的 grounding 必须更高"
    print("  ✓ 通过（幻觉被扣分，这是治幻觉的核心机制）")

    # 4b. ⭐ 对抗样本：奖励函数最容易被「薅」的三个地方
    #     RL 会主动去找奖励函数的漏洞，测试必须比模型先找到。
    print(f"\n[4b] 对抗样本（奖励函数必须扛得住）")

    # (a) 子串重叠：「蓝色」同时是「浅蓝色」和「深蓝色」的子串。
    #     图中只有深蓝裤，模型却说浅蓝上衣 —— 不能因为共享「蓝色」就放行。
    v = {"深蓝色裤"}
    g, p = _extract_claims("从图中可以看到这是浅蓝色上衣", v)
    print(f"    (a) visible={sorted(v)}，回复说「浅蓝色上衣」")
    print(f"        有据={sorted(g) or '∅'}  编造={sorted(p) or '∅'}")
    assert "浅蓝色" in p, "长词必须优先匹配，否则子串重叠会漏放幻觉"
    assert not g, "不能因为共享子串「蓝色」就把浅蓝判成有据"

    # (b) 只给建议、不给视觉证据 → 中性，不该奖也不该罚
    neutral = "这件衣服看起来不错，建议您选 M 码"
    g, p = _extract_claims(neutral, v)
    assert not p, "没提视觉属性就不该判幻觉 —— 否则模型会学会闭嘴"
    assert reward_grounding(neutral, v) == 0.5, "无证据应答应为中性 0.5"

    # (c) 罗列式幻觉：一句话堆 3 个不存在的瑕疵
    g, p = _extract_claims("图中有破洞、污渍，还起球了", {"浅蓝色上衣"})
    print(f"    (c) 罗列 3 个假瑕疵 → 编造={sorted(p)}")
    assert len(p) >= 3, "每个编造的属性都要独立扣分，否则堆一句只罚一次很划算"
    print("    ✓ 三个对抗样本都被正确处置")

    # 拒答
    assert reward_refusal("方便告诉我您的身高体重吗", True) == 1.0
    assert reward_refusal("我穿 M 应该可以的", True) == 0.0
    assert reward_refusal("我穿 M 应该可以的", False) == 1.0
    print("[5] 拒答奖励 ✓")

    # 组合
    ctx = RewardContext(
        expected_tool="lookup_order",
        expected_slots={"session_id": "s_1"},
        allowed_tools=["lookup_order", "check_stock"],
        visible_objects=visible,
        should_refuse=False,
    )
    good_c = ('{"thought": "需要查订单", "action": "lookup_order", '
              '"action_input": {"session_id": "s_1"}}')
    bad_c = "图中显示您的订单已经在路上了，预计明天到达。"

    r_good = compute_reward(good_c, ctx)
    r_bad = compute_reward(bad_c, ctx)
    print(f"\n[6] 组合奖励")
    print(f"  合规输出: total={r_good['total']:.3f}  {r_good}")
    print(f"  违规输出: total={r_bad['total']:.3f}  {r_bad}")
    assert r_good["total"] > r_bad["total"], "合规输出总分必须更高"
    print("  ✓ 通过")

    print("\n" + "=" * 78)
    print("全部通过。用法：把 compute_reward 接进 GRPO 的训练循环。")
    print("注意：奖励权重需要根据实际训练效果调，这里给的是初始值。")


if __name__ == "__main__":
    run_tests()
