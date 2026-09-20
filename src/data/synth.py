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
{
  "user_query": "用户的提问",
  "assistant": "客服回复",
  "need_clarification": false,
  "escalate_to_human": false
}

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

    async def run(self, plan: list[dict], image_pool: dict[str, list[str]],
                  product_pool: list[dict], clarify_ratio: float = 0.06,
                  escalate_ratio: float = 0.04, dry_run: bool = False):
        """按计划批量生成。

        plan:          来自 taxonomy.build_generation_plan()
        image_pool:    {image_type_key: [图片路径, ...]}
        product_pool:  [商品dict, ...]
        """
        total = sum(p["count"] for p in plan)
        print(f"计划生成 {total:,} 条样本（含 {clarify_ratio:.0%} 澄清 + {escalate_ratio:.0%} 转人工）\n")

        written = 0
        with open(self.out_path, "a", encoding="utf-8") as f:
            for item in plan:
                intent = next(i for i in
                              [{"key": k, "name": n, "description": d}
                               for k, n, d in _iter_intents()] if i["key"] == item["intent"])
                image_type = {"key": item["image_type"], "name": item["image_type_name"],
                              "description": ""}
                paths = image_pool.get(item["image_type"], [])
                if not paths:
                    print(f"⚠ 跳过 {item['intent_name']}×{item['image_type_name']}：无可用图片")
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
                            item["difficulty"], kinds[i], dry_run=dry_run,
                        )
                    except Exception as e:      # noqa: BLE001
                        print(f"  ✗ 出错: {e}")
                        continue

                    if sample:
                        f.write(sample.to_json() + "\n")
                        f.flush()
                        written += 1
                        if written % 20 == 0:
                            print(f"    ... 已写入 {written} 条")

        print(f"\n✓ 完成，共写入 {written} 条 → {self.out_path}")
        return written


def _iter_intents():
    from .taxonomy import INTENTS
    for i in INTENTS:
        yield i.key, i.name, i.description


# ---------------------------------------------------------------------------
# 7. CLI
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
    args = ap.parse_args()

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

    asyncio.run(synth.run(plan, image_pool, product_pool, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
