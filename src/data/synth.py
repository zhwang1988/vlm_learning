"""
合成客服图文对话数据。

对应讲义 docs/05-data-engineering.md 第 5 节，对应计划 Day 9。

设计要点（都是血泪经验）：

1. **多样性种子**：persona + emotion + 商品属性 随机组合。
   没有它们，1000 条样本会长得一模一样，训出来的模型只会一种腔调。

2. **负面约束**：明确禁止「首先/其次/最后」这类模板结构。
   不写的话，AI 味会重到没法用。

3. **基于事实**：把结构化商品信息塞进 prompt，要求只能基于它回答。
   否则模型会编造「这件是 100% 桑蚕丝」。

4. **主动造拒答样本**：约 8% 的样本应该是「信息不足 → 主动澄清」或
   「无法处理 → 转人工」。缺了这类样本，模型上线后什么都敢答。

5. **可断点续跑**：合成 10000 条要跑很久，必须支持 resume。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 1. 多样性种子
# ---------------------------------------------------------------------------

PERSONAS = [
    ("焦虑型", "语气急，反复确认，容易担心买错"),
    ("理性型", "问得具体，关注参数和对比，少情绪词"),
    ("新手型", "不懂专业术语，用生活化描述，可能问得很基础"),
    ("挑剔型", "要求高，会拿其他家对比，对细节敏感"),
    ("沉默型", "话很少，一两个词，需要客服主动追问"),
    ("砍价型", "关心优惠和性价比，会问能不能便宜"),
    ("老客型", "熟客口吻，直接问具体问题，不爱寒暄"),
]

EMOTIONS = ["平静", "焦急", "不满", "失望", "期待", "困惑", "怀疑"]

# 商品池示例结构（Day 9 你要替换成真实商品数据）
PRODUCT_SEED_EXAMPLE = {
    "category": "女装/针织衫",
    "title": "圆领宽松针织毛衣 秋冬新款",
    "attributes": {
        "材质": "95% 棉 5% 氨纶",
        "克重": "280g",
        "颜色": "米白 / 藏青 / 燕麦色",
        "尺码": "S/M/L/XL，M 码肩宽 38cm 胸围 100cm 衣长 58cm",
        "版型": "宽松直筒",
        "洗护": "可机洗，冷水，勿漂白",
        "价格": "¥299",
        "产地": "浙江",
    },
}


# ---------------------------------------------------------------------------
# 2. 输出结构
# ---------------------------------------------------------------------------


@dataclass
class CXSample:
    """一条客服对话样本。"""
    id: str
    image_path: str
    image_type: str
    intent: str
    difficulty: int
    messages: list[dict]           # [{"role": "user"/"assistant", "content": [...]}]
    meta: dict = field(default_factory=dict)
    generator: str = ""
    created_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 3. Prompt 构造
# ---------------------------------------------------------------------------

# ⚠️ 这个模板会用 .format() 填充，所以模板里**每一处字面花括号都必须写成
#    {{ 和 }}**。JSON 示例里的花括号是最容易漏的地方 —— 漏了就报
#    `KeyError: '\n  "user_query"'` 这种看起来莫名其妙的错
#    （Python 把整个 JSON 块当成一个占位符名字了）。
#    改这个模板后务必跑一次：python -m src.data.synth --selftest
SYSTEM_PROMPT = """你是一个电商客服数据生成器。你的任务是产出一条真实、自然、可用的客服对话样本。

## 硬性要求

1. **用户提问必须口语化**
   - 用真实买家会说的话，不要书面语
   - 可以带情绪（焦虑、不满、犹豫、困惑）
   - 可以有轻微的打字习惯（少用标点、语气词）
   - 长度 5–60 字

2. **客服回复必须**
   - 先回应情绪，再给信息（不要一上来就报参数）
   - **只能基于【已知商品信息】回答**，信息里没有的绝不能说
   - 涉及判断时给出依据（"从图片看…" / "根据标注…"）
   - 信息不足时主动追问，而不是硬答
   - 长度 60–200 字
   - **禁止使用「首先/其次/再者/最后」「综上所述」「希望对您有帮助」这类模板结构**
   - 禁止过度承诺（"绝对不会有问题"）

3. **输出格式**：严格输出 JSON，不要任何解释文字
{{
  "user_query": "用户的提问",
  "assistant": "客服回复",
  "need_clarification": false,
  "escalate_to_human": false
}}

## 已知商品信息
{product_info}

## 图像信息
图像类型：{image_type_name}
图像描述：{image_caption}

## 本次生成要求
- 用户人设：{persona_name}（{persona_desc}）
- 用户情绪：{emotion}
- 咨询意图：{intent_name}——{intent_desc}
- 难度等级：{difficulty}（1=图里直接可见 5=需要复杂推理或应拒答）
{special_instruction}
"""

CLARIFY_INSTRUCTION = """
## ⚠️ 本条样本的特殊要求
这是一条「信息不足」样本。用户的问题**缺少必要信息**，正确的客服行为是
**主动追问**而不是猜测回答。例如用户说"我穿S行吗"但没说身高体重。
请让 assistant 的回复以追问为主，need_clarification 必须为 true。
"""

ESCALATE_INSTRUCTION = """
## ⚠️ 本条样本的特殊要求
这是一条「应转人工」样本。用户的需求超出 AI 客服能力范围
（如要求特殊赔偿、投诉、涉及法律问题、情绪激烈）。
assistant 的回复应该：先安抚，说明可以做什么，然后明确表示转人工处理。
escalate_to_human 必须为 true。
"""


def build_prompt(intent: dict, image_type: dict, product: dict,
                 persona: tuple[str, str], emotion: str, difficulty: int,
                 sample_kind: str = "normal") -> str:
    """拼装单条样本的生成 prompt。"""
    special = ""
    if sample_kind == "clarify":
        special = CLARIFY_INSTRUCTION
    elif sample_kind == "escalate":
        special = ESCALATE_INSTRUCTION

    return SYSTEM_PROMPT.format(
        product_info=json.dumps(product, ensure_ascii=False, indent=2),
        image_type_name=image_type["name"],
        image_caption=image_type["description"],
        persona_name=persona[0],
        persona_desc=persona[1],
        emotion=emotion,
        intent_name=intent["name"],
        intent_desc=intent["description"],
        difficulty=difficulty,
        special_instruction=special,
    )


# ---------------------------------------------------------------------------
# 4. 模型调用（多provider兼容）
# ---------------------------------------------------------------------------


class SynthClient:
    """统一的合成客户端。支持 OpenAI 兼容接口（含 DashScope、vLLM、Ollama）。

    配置通过环境变量：
        SYNTH_API_BASE   如 https://dashscope.aliyuncs.com/compatible-mode/v1
        SYNTH_API_KEY
        SYNTH_MODEL      如 qwen-vl-max / gpt-4o / Qwen/Qwen2.5-VL-72B-Instruct
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, max_concurrency: int = 4):
        from openai import AsyncOpenAI

        self.base_url = base_url or os.getenv("SYNTH_API_BASE")
        self.api_key = api_key or os.getenv("SYNTH_API_KEY")
        self.model = model or os.getenv("SYNTH_MODEL", "qwen-vl-max")
        self.sem = asyncio.Semaphore(max_concurrency)

        if not self.api_key:
            raise RuntimeError(
                "缺少 SYNTH_API_KEY。请在 .env 里配置，或用 --dry-run 只打印 prompt。"
            )
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       temperature: float = 0.9, max_retries: int = 3) -> dict:
        """调一次模型，返回解析后的 JSON。"""
        content: list[dict] = [{"type": "text", "text": prompt}]

        if image_path and Path(image_path).exists():
            import base64
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            suffix = Path(image_path).suffix.lstrip(".").lower()
            mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        last_err = None
        for attempt in range(max_retries):
            async with self.sem:
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content}],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        timeout=90,
                    )
                    raw = resp.choices[0].message.content
                    return json.loads(raw)
                except Exception as e:      # noqa: BLE001
                    last_err = e
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"生成失败（重试 {max_retries} 次）: {last_err}")


# ---------------------------------------------------------------------------
# 5. 质量校验
# ---------------------------------------------------------------------------

BANNED_PATTERNS = [
    "首先", "其次", "再者", "综上所述", "希望对您有帮助", "感谢您的咨询",
    "作为AI", "作为一个AI", "我是人工智能",
]


def validate_sample(obj: dict, require_image_facts: bool = True) -> tuple[bool, str]:
    """校验一条合成样本。返回 (是否合格, 原因)。"""
    for k in ("user_query", "assistant"):
        if k not in obj or not isinstance(obj[k], str) or not obj[k].strip():
            return False, f"缺少字段 {k}"

    uq, asst = obj["user_query"].strip(), obj["assistant"].strip()

    if not (4 <= len(uq) <= 120):
        return False, f"用户提问长度异常 ({len(uq)})"
    if not (25 <= len(asst) <= 600):
        return False, f"客服回复长度异常 ({len(asst)})"

    for p in BANNED_PATTERNS:
        if p in asst:
            return False, f"命中模板化用语: {p}"

    # 长度比过于悬殊通常说明有问题
    if len(asst) / max(len(uq), 1) > 12:
        return False, "回复/提问长度比异常"

    return True, "ok"


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------


class Synthesizer:
    def __init__(self, client: Optional[SynthClient], out_path: str | Path,
                 resume: bool = True):
        self.client = client
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.done_keys: set[str] = set()

        if resume and self.out_path.exists():
            with open(self.out_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.done_keys.add(json.loads(line)["id"])
                    except Exception:
                        continue
            print(f"↻ 断点续跑：已完成 {len(self.done_keys)} 条")

    @staticmethod
    def make_id(intent: str, img_type: str, image_path: str, salt: str) -> str:
        raw = f"{intent}|{img_type}|{image_path}|{salt}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    async def generate_one(self, intent: dict, image_type: dict, product: dict,
                           image_path: str, difficulty: int,
                           sample_kind: str = "normal",
                           salt: Optional[str] = None,
                           dry_run: bool = False) -> Optional[CXSample]:
        persona = random.choice(PERSONAS)
        emotion = random.choice(EMOTIONS)
        salt = salt or f"{time.time_ns()}-{random.random()}"

        sid = self.make_id(intent["key"], image_type["key"], image_path, salt)
        if sid in self.done_keys:
            return None

        prompt = build_prompt(intent, image_type, product, persona, emotion,
                              difficulty, sample_kind)

        if dry_run:
            print("=" * 78)
            print(prompt)
            print("=" * 78)
            return None

        obj = await self.client.generate(prompt, image_path=image_path)
        ok, reason = validate_sample(obj)
        if not ok:
            print(f"  ✗ 不合格 ({reason}): {obj.get('user_query', '')[:40]}")
            return None

        sample = CXSample(
            id=sid,
            image_path=image_path,
            image_type=image_type["key"],
            intent=intent["key"],
            difficulty=difficulty,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": obj["user_query"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": obj["assistant"]},
                ]},
            ],
            meta={
                "persona": persona[0],
                "emotion": emotion,
                "need_clarification": obj.get("need_clarification", False),
                "escalate_to_human": obj.get("escalate_to_human", False),
                "sample_kind": sample_kind,
            },
            generator=self.client.model if self.client else "dry-run",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        return sample

    @staticmethod
    def _resolve_intent(item: dict) -> dict:
        for k, n, d in _iter_intents():
            if k == item["intent"]:
                return {"key": k, "name": n, "description": d}
        raise KeyError(f"计划里的意图 {item['intent']} 不在 taxonomy.INTENTS 里")

    async def run(self, plan: list[dict], image_pool: dict[str, list[str]],
                  product_pool: list[dict], clarify_ratio: float = 0.06,
                  escalate_ratio: float = 0.04, dry_run: bool = False) -> int:
        """按计划批量生成。

        plan:          来自 taxonomy.build_generation_plan()
        image_pool:    {image_type_key: [图片路径, ...]}
        product_pool:  [商品dict, ...]

        返回值 = 成功写入条数（dry-run 恒为 0）。

        ⚠️ 调用方**必须**看返回值决定成败。早期版本在结尾无条件打
           「✓ 完成，共写入 N 条」，N=0 时也这么打 —— 于是脚本和 CI 都被
           静默骗过去，人也要翻半天日志才发现一条都没生成。
           「生成 0 条」和「生成成功」必须能区分开。
        """
        total = sum(p["count"] for p in plan)

        if dry_run:
            # dry-run 只验证「prompt 能不能拼出来」，不调模型、不写文件。
            # 它的价值：改完 SYSTEM_PROMPT 立刻知道花括号有没有写坏
            # （模板是用 .format() 填的，JSON 示例里的 {} 必须写成 {{}}）。
            print(f"dry-run：只拼 prompt，不调模型、不写文件。计划 {total:,} 条\n")
            for item in plan:
                paths = image_pool.get(item["image_type"], [])
                await self.generate_one(
                    self._resolve_intent(item),
                    {"key": item["image_type"], "name": item["image_type_name"],
                     "description": ""},
                    random.choice(product_pool),
                    paths[0] if paths else "__no_image__",
                    item["difficulty"], dry_run=True)
            print(f"\n✓ dry-run 结束：{len(plan)} 个格子的 prompt 都拼装成功，"
                  f"未调用模型")
            return 0

        print(f"计划生成 {total:,} 条样本"
              f"（含 {clarify_ratio:.0%} 澄清 + {escalate_ratio:.0%} 转人工）\n")

        written = failed = rejected = no_image = 0
        errors: list[str] = []

        with open(self.out_path, "a", encoding="utf-8") as f:
            for item in plan:
                intent = self._resolve_intent(item)
                image_type = {"key": item["image_type"],
                              "name": item["image_type_name"], "description": ""}
                paths = image_pool.get(item["image_type"], [])
                if not paths:
                    print(f"⚠ 跳过 {item['intent_name']}×{item['image_type_name']}：无可用图片")
                    no_image += 1
                    continue

                n = item["count"]
                n_clarify = int(n * clarify_ratio)
                n_escalate = int(n * escalate_ratio)
                kinds = ["normal"] * (n - n_clarify - n_escalate) + \
                        ["clarify"] * n_clarify + ["escalate"] * n_escalate
                random.shuffle(kinds)

                print(f"→ {item['intent_name']} × {item['image_type_name']} "
                      f"({n} 条, 难度 {item['difficulty']})")

                for i in range(n):
                    product = random.choice(product_pool)
                    image_path = random.choice(paths)
                    try:
                        sample = await self.generate_one(
                            intent, image_type, product, image_path,
                            item["difficulty"], kinds[i], dry_run=False,
                        )
                    except Exception as e:      # noqa: BLE001
                        # 记下来，别只 print 一句就 continue —— 全失败时
                        # 你需要知道错在哪，而不是只看到一个「完成」。
                        failed += 1
                        if len(errors) < 5:
                            errors.append(f"{type(e).__name__}: {e}")
                        continue

                    if sample:
                        f.write(sample.to_json() + "\n")
                        f.flush()
                        written += 1
                        if written % 20 == 0:
                            print(f"    ... 已写入 {written} 条")
                    else:
                        rejected += 1       # 生成出来了但没过质量门 / 续跑命中

        # ---- 汇总：把「成功」和「没成功」分清楚 ----
        print()
        if errors:
            print(f"  前 {len(errors)} 条错误：")
            for e in errors:
                print(f"      {e}")
            if failed > len(errors):
                print(f"      ... 另有 {failed - len(errors)} 条同类错误")

        parts = [f"写入 {written} 条"]
        if rejected:
            parts.append(f"未采用 {rejected} 条")
        if no_image:
            parts.append(f"无图片跳过 {no_image} 格")
        if failed:
            parts.append(f"失败 {failed} 条")
        line = " · ".join(parts)

        if failed == 0:
            print(f"✓ 完成：{line} → {self.out_path}")
        elif written == 0:
            print(f"✗ 一条都没成功：{line}")
            print("  先看上面的报错。若是 prompt 拼不出来，检查 SYSTEM_PROMPT 里"
                  " 的花括号有没有写成 {{}}。")
        else:
            ok_rate = written / max(written + failed, 1)
            print(f"⚠ 部分失败：{line}（成功率 {ok_rate:.0%}）→ {self.out_path}")
            print("  成功率明显偏低时不要直接拿这批数据去训 —— 先修 prompt。")

        return written


def _iter_intents():
    from .taxonomy import INTENTS
    for i in INTENTS:
        yield i.key, i.name, i.description


# ---------------------------------------------------------------------------
# 7. 自检（离线，不联网、不写文件）
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """离线自检。

    存在的直接原因：SYSTEM_PROMPT 的 JSON 示例用了**单个花括号**，而模板是
    用 .format() 填的 —— 于是每次生成都抛 `KeyError: '\\n  "user_query"'`，
    而外层把异常吞掉后照样打印「✓ 完成」。两个问题凑在一起，表现为
    「跑完说完成、文件里一条没有」，查起来很费劲。

    所以这个自检同时盯住两件事：
      ① 花括号转义 / 占位符替换（不需要联网就能测）
      ② 质量门判定是否符合预期
    """
    from .taxonomy import IMAGE_TYPES, INTENTS, build_generation_plan

    print("=" * 76)
    print("数据合成自检（离线，不联网、不写文件）")
    print("=" * 76)

    def _prompt(intent, image, kind="normal", product=None):
        return build_prompt(
            {"key": intent.key, "name": intent.name,
             "description": intent.description},
            {"key": image.key, "name": image.name,
             "description": image.description},
            product or PRODUCT_SEED_EXAMPLE, PERSONAS[0], EMOTIONS[0],
            image.difficulty, kind)

    # 1. ⭐ 花括号转义：所有组合都要能拼出来
    #    这是最直接的回归测试。模板里少写一个 } 就会在这里炸。
    print(f"\n[1] prompt 拼装（{len(INTENTS)}×{len(IMAGE_TYPES)} 全组合）")
    n = 0
    for it in INTENTS:
        for img in IMAGE_TYPES:
            p = _prompt(it, img)
            assert p and len(p) > 200, f"{it.key}×{img.key} 拼出的 prompt 太短"
            n += 1
    print(f"    {n} 种组合全部拼装成功")

    # 2. 占位符真的被替换了
    print("\n[2] 占位符替换")
    p = _prompt(INTENTS[0], IMAGE_TYPES[0])
    for token in ("{product_info}", "{persona_name}", "{emotion}",
                  "{intent_name}", "{difficulty}", "{image_type_name}"):
        assert token not in p, f"占位符 {token} 没被替换 —— .format() 漏了参数？"
    assert PRODUCT_SEED_EXAMPLE["title"] in p, "商品标题没进 prompt"
    assert PRODUCT_SEED_EXAMPLE["attributes"]["材质"] in p, \
        "商品属性没进 prompt —— 「只能基于已知商品信息回答」的前提就没了，模型必然编造"
    assert PERSONAS[0][0] in p, "人设没进 prompt"
    print("    ✓ 无残留占位符，商品事实已注入")

    # 3. JSON 示例的双花括号要还原成单花括号给模型看，
    #    否则模型会照着字面的 {{ 输出，解析必挂
    print("\n[3] 输出格式示例")
    assert '"user_query"' in p and '"assistant"' in p, \
        "JSON 输出示例没出现在 prompt 里"
    assert "{{" not in p and "}}" not in p, \
        "prompt 里残留了 {{ }} —— 转义多写了一层，模型会照抄双花括号"
    print("    ✓ JSON 示例以正确的单花括号形式给到模型")

    # 4. 三种样本类型都带上了各自的特殊要求
    print("\n[4] 三种样本类型")
    cases = (("normal", None), ("clarify", "主动追问"), ("escalate", "转人工"))
    for kind, marker in cases:
        pk = _prompt(INTENTS[0], IMAGE_TYPES[0], kind)
        if marker:
            assert marker in pk, f"{kind} 样本缺少特殊指令（找不到「{marker}」）"
        else:
            assert "本条样本的特殊要求" not in pk, "normal 样本不该带特殊指令"
        print(f"    {kind:<9} {'带特殊指令' if marker else '无特殊指令'}")
    print("    ✓ 澄清 / 转人工样本占比可控（见 run() 的 clarify_ratio）")

    # 5. 质量门
    print("\n[5] 质量门 validate_sample()")
    good = {
        "user_query": "我穿 M 码会不会太紧呀，平时穿 M 比较多",
        "assistant": "您好，理解您担心尺码。这款 M 码肩宽 38cm、胸围 100cm，"
                     "属于宽松直筒版型，平时穿 M 的话这件是合适的。"
                     "如果您偏好更修身的效果，可以考虑 S 码。",
    }
    ok, why = validate_sample(good)
    assert ok, f"正常样本应通过，实际被拒: {why}"

    # ⚠️ 断言必须指向**预期的拒绝原因**，不能只看「被拒了」。
    #    下面第三条一开始就是被长度检查顺手拦下的（24 字 < 25 下限），
    #    看着「测试通过」，其实根本没测到模板化用语的过滤逻辑。
    bads = [
        ({"user_query": "短", "assistant": "好的什么都好说"}, "提问过短", "长度"),
        ({"assistant": "您好，这款是纯棉的，上身很舒服，版型也宽松。" * 2},
         "缺 user_query", "缺少字段"),
        ({"user_query": "这个料子怎么样，会不会起球",
          "assistant": "首先，我们来看面料，这款是 95% 棉。其次，说下版型，"
                       "宽松直筒。再者，价格 299。综上所述，希望对您有帮助。"},
         "模板化用语", "模板化"),
    ]
    for bad, hint, expect_in in bads:
        ok2, why2 = validate_sample(bad)
        assert not ok2, f"「{hint}」应该被质量门拦下"
        assert expect_in in why2, f"「{hint}」被拦下的原因不对：{why2}"
        print(f"    {hint:<12} → 拦下（{why2}）")
    print("    ✓ 过短 / 缺字段 / 模板化用语都按预期原因拦下")

    # 6. 生成计划可用（synth 的输入）
    print("\n[6] 生成计划")
    plan = build_generation_plan()
    assert plan and all(p["count"] > 0 for p in plan)
    print(f"    {len(plan)} 个格子 · 合计 {sum(p['count'] for p in plan):,} 条")

    print("\n" + "=" * 76)
    print("✓ 全部通过（离线自检，未调用任何模型）")
    return 0


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def main():
    import argparse
    from .taxonomy import IMAGE_TYPES, INTENTS, build_generation_plan

    ap = argparse.ArgumentParser(description="合成客服图文数据")
    ap.add_argument("--image-root", help="商品图片根目录（按 image_type 分子目录）")
    ap.add_argument("--out", default="data/raw/synth_v0.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只生成前 N 条（调试用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt 不调模型")
    ap.add_argument("--sample", action="store_true", help="用内置示例商品跑一条")
    ap.add_argument("--selftest", action="store_true",
                    help="离线自检：prompt 拼装 + 质量门（不联网、不写文件）")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    plan = build_generation_plan()
    if args.limit:
        plan = plan[:args.limit]

    # 加载图片池
    image_pool: dict[str, list[str]] = {}
    if args.image_root:
        root = Path(args.image_root)
        for t in IMAGE_TYPES:
            d = root / t.key
            if d.exists():
                image_pool[t.key] = [str(p) for p in sorted(d.glob("*"))
                                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        print("图片池：")
        for k, v in image_pool.items():
            print(f"  {k}: {len(v)} 张")
        print()
    if not args.image_root:
        # 无图片也能跑（纯文本模式），用 placeholder
        for t in IMAGE_TYPES:
            image_pool[t.key] = ["__no_image__"]

    product_pool = [PRODUCT_SEED_EXAMPLE]

    client = None
    if not args.dry_run:
        try:
            client = SynthClient()
            print(f"合成模型: {client.model}")
        except RuntimeError as e:
            print(f"⚠ {e}\n  降级为 --dry-run 模式\n")
            args.dry_run = True

    synth = Synthesizer(client, args.out)

    if args.sample:
        plan = [plan[0]]
        plan[0]["count"] = 1

    written = asyncio.run(synth.run(plan, image_pool, product_pool,
                                    dry_run=args.dry_run))

    # ⭐ 非 dry-run 且一条都没成 → 非 0 退出码。
    #    「跑完了」和「跑成功了」是两件事，脚本和 CI 依赖这个区分。
    if not args.dry_run and written == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
