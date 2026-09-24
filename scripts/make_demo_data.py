#!/usr/bin/env python3
"""生成一份**带 ground truth 清单**的演示数据，让 Day 9–12 / Day 20–24 在
没有 GPU、没有 API key 的机器上也能整条跑通。

为什么需要它
------------
Day 9 的数据合成要调大模型（要钱、要 key），Day 10–12 的清洗/去重/建 SFT
又要读上游产物。结果是：一台干净的笔记本上，第 9 天之后的所有命令都跑不动，
日文件里写的命令全是悬空的。这个脚本用**确定性随机**造一份等价数据，
把「上游缺产物」这个断点补上。

为什么必须带清单（`--manifest`）
--------------------------------
光给一份 jsonl 是不够的。清洗流水线跑完会打印「格式校验后 64 条 / 去重后 63 条」，
但你没法判断这个数字**对不对** —— 除非你知道我故意往里面掺了什么。

所以本脚本同时输出一份清单，写明注入了哪些脏数据：

    脏数据类型            注入条数    期望被哪一步拦下
    ------------------   --------   ------------------
    提问过短                    3   rule_filter（长度）
    含联系方式                  2   rule_filter（联系方式）
    AI 自我暴露                2   rule_filter（AI 自我暴露）
    结构异常                    2   格式校验
    缺 image_path              1   格式校验
    同图 + 同文（真重复）       8   去重
    同图 + 不同文（有效样本）   6   **必须保留**
    同文 + 不同图（有效样本）   4   **必须保留**
    近似重复（差语气词）         5   dedup 删 / dedup_simple 留

有了这张表，清洗报告就能**对账**，而不是只能看着数字点头。
最后两行是这张表里最重要的部分：它们定义的是「**不该被删的**」。
一个只会删数据的流水线看起来指标很漂亮，代价是把有效样本一起删了 ——
「删多了」和「删少了」都只是数字变化，只有 ground truth 能区分。

用法
----
    # 造数据 + 真图（推荐，会同时产出 data/fixtures/images/）
    python scripts/make_demo_data.py

    # 只造文本，图片一律用 __no_image__ 占位符（不依赖 Pillow）
    python scripts/make_demo_data.py --no-images

    # 指定规模与输出
    python scripts/make_demo_data.py --n-clean 200 --seed 7 \
        --out data/fixtures/demo_synth.jsonl

跑完之后：
    make data-clean     # 看清洗报告
    cat data/fixtures/DEMO_MANIFEST.md   # 和 ground truth 对账
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# 让脚本能直接 `python scripts/make_demo_data.py` 跑（和 selfcheck.py 一致）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 语料模板
# ---------------------------------------------------------------------------
# 贴真实客服语料的写法：口语、有语气词、句长不齐、偶尔带错别字。
# 刻意**不用**「问题 1 / 问题 2」这种占位文本 —— 那样去重逻辑会被过度简化，
# 测不出真实语料里的近重复（「我穿M码会不会太紧」vs「我穿M码会不会太紧啊」）。

PRODUCTS = [
    {"id": "p_tee", "title": "纯棉圆领短袖T恤",
     "attributes": {"材质": "100% 精梳棉", "克重": "220g", "版型": "宽松直筒",
                    "颜色": ["黑色", "米白", "雾霾蓝"], "袖长": "短袖"}},
    {"id": "p_denim", "title": "高腰直筒牛仔裤",
     "attributes": {"材质": "棉 98% 氨纶 2%", "版型": "直筒", "颜色": ["深蓝", "浅蓝"],
                    "腰型": "高腰", "弹力": "微弹"}},
    {"id": "p_dress", "title": "法式碎花连衣裙",
     "attributes": {"材质": "雪纺", "版型": "A 字", "颜色": ["雾霾蓝", "米白"],
                    "裙长": "中长款", "图案": "碎花"}},
    {"id": "p_coat", "title": "羊毛双面呢大衣",
     "attributes": {"材质": "羊毛 70% 聚酯纤维 30%", "版型": "H 型",
                    "颜色": ["驼色", "黑色"], "厚度": "加厚"}},
]

# intent_key → [(用户提问, 客服回复), ...]
# 键必须和 src/data/taxonomy.py 的 INTENTS 一一对应，否则生成计划会撞空。
DIALOGS: dict[str, list[tuple[str, str]]] = {
    "size_fit": [
        ("我165 50公斤穿M会不会太大",
         "您的身高体重对应我们尺码表的 M 码。这款是宽松直筒版型，M 码肩宽 38cm、"
         "胸围 100cm，按您的数据穿 M 是合适的，不会显大。"),
        ("我穿M码会不会太紧呀，平时穿M比较多",
         "平时穿 M 的话这件是合适的。它的胸围 100cm，比常规 M 码略宽松一点，"
         "不会紧。如果喜欢贴身的可以考虑 S。"),
        ("我穿M码会不会太紧",
         "这件 M 码胸围 100cm，属于宽松版型，正常情况下不会紧。"
         "如果您的胸围超过 96cm，建议选 L 更稳妥。"),
        ("这个偏大还是偏小啊",
         "整体偏正常。这款是宽松直筒版型，和常规码基本一致，不需要刻意大一码。"),
        ("腰围多少",
         "M 码平铺腰围 72cm，有微弹，拉伸后约 76cm。"),
        ("我175 70公斤要买什么码",
         "按您的身高体重建议选 L。L 码胸围 106cm、衣长 70cm，"
         "175 的身高穿 L 肩线会比较合适。"),
        ("尺码表和实物一致吗",
         "一致。我们尺码表是按平铺实测的，误差在 1-2cm 以内属于正常工艺范围。"),
        ("腿粗可以穿这个直筒裤吗",
         "可以的。直筒版型裤腿围 56cm，对大腿比较友好，不会箍腿。"),
    ],
    "quality_issue": [
        ("图里这个线头是正常的吗，需要退货吗",
         "从图片看是缝制收口处的线头，剪掉即可，不影响穿着，也不属于质量问题。"
         "如果您介意，可以在订单里申请七天无理由退换。"),
        ("这个走线正常吗",
         "图片里的走线密度是每 3cm 12 针，属于行业常规标准，是正常的。"),
        ("收到货颜色和图片差好多，色差有点大",
         "抱歉给您造成困扰。请拍一张自然光下的实拍图发我，我核对一下色号。"
         "如果确实存在明显色差，可以直接申请退换，运费我们承担。"),
        ("这个料子会不会起球，我皮肤比较敏感",
         "这款是 100% 精梳棉，220g 克重，正常穿着和洗涤不会起球。"
         "皮肤敏感的话建议首次穿着前先单独洗一遍。"),
        ("这里是不是破洞",
         "放大看这里是面料的纹理织法，不是破洞。如果您收到实物后仍觉得有问题，"
         "可以申请退货，我们承担运费。"),
        ("洗了一次就缩水了",
         "纯棉面料首次水洗有 2-3% 的自然缩率属于正常。建议冷水手洗、"
         "平铺阴干。如果缩水超过 5cm，可以申请售后。"),
    ],
    "color_mismatch": [
        ("这个颜色是米白吗",
         "是的，是米白色（偏暖调），不是纯白。详情页第 3 张图更接近实物。"),
        ("雾霾蓝是偏灰还是偏蓝",
         "偏灰调多一点，属于低饱和的蓝灰色，室内光下会更灰一些。"),
        ("黑色会不会掉色",
         "首次洗涤建议单独洗，深色面料首次下水有轻微浮色是正常现象，"
         "洗过两次就稳定了。"),
        ("驼色和黑色哪个显瘦",
         "从视觉效果看黑色更收敛一些。驼色偏温柔，如果您主要想显瘦可以选黑色。"),
    ],
    "material": [
        ("这个是纯棉的吗",
         "是的，100% 精梳棉，克重 220g，比普通棉更挺括不易变形。"),
        ("透气吗夏天穿会不会闷",
         "220g 纯棉中厚款，透气性不错。夏天室外长时间活动可能会有点热，"
         "空调房或春秋穿更合适。"),
        ("能不能机洗",
         "可以机洗，建议翻面装洗衣袋、冷水轻柔档，不要甩干太久。"),
        ("羊毛含量是多少",
         "羊毛 70%，聚酯纤维 30%。这个配比兼顾保暖和抗皱，"
         "比纯羊毛更好打理。"),
    ],
    "styling": [
        ("这个上衣配什么裤子好看",
         "宽松直筒版型配直筒牛仔裤或阔腿裤都比较协调，"
         "塞进腰头会更显腰线。"),
        ("适合上班穿吗",
         "可以的。这款版型干净，配西装裤或半裙都适合通勤，不会太休闲。"),
        ("个子矮能穿这个长裙吗",
         "中长款裙长 115cm。155 左右的身高穿会到脚踝上方，"
         "搭配有跟的鞋比例会更好。"),
    ],
    "logistics": [
        ("快递三天了还没动，能帮我查一下吗",
         "抱歉让您久等。我帮您查一下物流节点，请提供订单号，"
         "我会同步给快递催促。"),
        ("什么时候能发货",
         "现货商品在付款后 48 小时内发出，节假日顺延。发货后系统会推送物流单号，"
         "您可以在订单页实时查看。"),
        ("可以发顺丰吗",
         "可以。顺丰需要在下单时备注，部分地区会产生运费差价，"
         "具体以结算页为准。"),
        ("支持港澳台配送吗",
         "支持中国香港、中国澳门、中国台湾地区配送，"
         "时效一般 5-8 个工作日，运费在结算页显示。"),
    ],
    "return_refund": [
        ("我想退货，尺码不合适",
         "可以的。签收后 7 天内支持无理由退货，商品不影响二次销售即可。"
         "请在订单页申请，我会尽快审核。"),
        ("退货运费谁承担",
         "质量问题我们承担运费；七天无理由退换由买家承担运费，"
         "如果有运费险可以直接抵扣。"),
        ("换货要多久",
         "我们收到退货后 1-2 个工作日寄出新商品，"
         "全程大约 5-7 天，您可以在订单页看进度。"),
        ("拆了吊牌还能退吗",
         "拆掉吊牌会影响二次销售，这种情况无法走七天无理由。"
         "如果是质量问题，请拍照发我，我们单独处理。"),
    ],
    "presale_general": [
        ("这个有现货吗",
         "黑色和米白有现货，雾霾蓝目前需要 3-5 天备货。"),
        ("这个和详情页是同一件吗",
         "是的，就是详情页这款，颜色和版型都完全一致。我们所有在售商品的"
         "主图都是实物拍摄，没有使用效果图。"),
        ("能开发票吗",
         "可以开电子普通发票，下单时填写抬头，发货后 3 个工作日内推送。"),
    ],
}

# 图片类型 → 生成图片时的底色/纹理种子偏移
IMAGE_TYPE_STYLE = {
    "product_main": 0,
    "model_wearing": 1,
    "detail_closeup": 2,
    "defect_photo": 3,
    "size_chart": 4,
    "user_compare": 5,
}

ASSISTANT_MIN_LEN = 25      # 和 dedup.rule_filter 的下限对齐，别造出必被删的样本
ASSISTANT_MAX_LEN = 300


# ---------------------------------------------------------------------------
# 真图生成
# ---------------------------------------------------------------------------


def _make_image(path: Path, seed: int, style: int) -> int:
    """生成一张有唯一 pHash 的小图。返回 pHash（供调用方查重）。

    为什么要真图而不是全用 __no_image__：
      `dedup` 的图片指纹分支（pHash + 汉明距离）是 Day 10 的核心内容。
      全用占位符的话，这条分支**一次都不会被执行**，日文件里那些
      「同图不同文要保留」的结论就成了没有代码支撑的说法。
    """
    import numpy as np
    from PIL import Image

    from src.data.image_utils import phash

    rng = np.random.default_rng(seed * 1000 + style)
    h = w = 224
    arr = np.zeros((h, w, 3), dtype=np.uint8)

    # 水平渐变（给 DCT 一个稳定的低频结构 → pHash 可区分）
    ramp = np.linspace(20, 235, w).astype(np.uint8)
    arr[:, :, 0] = ramp[None, :]
    arr[:, :, 1] = np.roll(ramp, 17 * (style + 1))[None, :]
    arr[:, :, 2] = int(40 + 35 * style) % 256

    # 竖条纹，相位随 seed 变（制造高频差异）
    stripe_w = 2 + (seed % 5)
    cols = (np.arange(w) // stripe_w + seed) % 2 == 0
    arr[: h // 2, :, 1] = np.where(cols[None, :], 250, 30)
    arr[: h // 2, :, 2] = np.where(cols[None, :], 30, 250)

    # 一点噪声，避免不同 seed 恰好落进同一个 DCT 符号模式
    noise = rng.integers(0, 26, size=(h, w, 1), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, "RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return phash(img)


def _make_black_image(path: Path) -> int:
    """生成一张全黑图。它的 pHash **就是 0** —— 用来验证哨兵值语义。

    这张图是故意留的：`0` 既是「全黑图的合法 pHash」，又曾经被当成
    「没有图」的哨兵值（见 src/data/dedup.py 的 `_usable_hash`）。
    数据里放一张全黑图，就能顺着清洗报告确认这条路径没被走错。
    """
    from PIL import Image

    from src.data.image_utils import phash

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), (0, 0, 0))
    img.save(path)
    h = phash(img)
    assert h == 0, (
        f"全黑图的 pHash 应该是 0，实际 {h:#x} —— "
        f"如果 pHash 实现改了，本脚本关于哨兵值的假设需要重新核对")
    return h


# ---------------------------------------------------------------------------
# 样本构造
# ---------------------------------------------------------------------------


def _sample(sid: str, image_path: str, image_type: str, intent: str,
            difficulty: int, query: str, answer: str) -> dict:
    return {
        "id": sid,
        "image_path": image_path,
        "image_type": image_type,
        "intent": intent,
        "difficulty": difficulty,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": query},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "meta": {
            "persona": "买家",
            "emotion": "neutral",
            "need_clarification": False,
            "escalate_to_human": intent in ("return_refund", "quality_issue"),
            "sample_kind": "normal",
            "synthetic": True,
            "generator": "scripts/make_demo_data.py",
        },
    }


def _sid(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _validate_templates() -> None:
    """开工前一次性校验所有模板的提问/回复长度。

    ⚠️ 校验必须**一次性列出全部问题**，不能边生成边 assert。
       边生成边断言的后果是：改一条、跑一次、又撞下一条 —— 我在这个文件上
       实测撞了三次（24 字、20 字、再一条），每次都要重跑一遍。
       长度约束本身是死的，完全可以先全量扫一遍再动手。

    约束来源（必须和 src/data/dedup.py 的 rule_filter 默认值保持一致）：
      提问 4–120 字、回复 ≥ 25 字；不满足的样本会在 Day 10 第一步就被删掉，
      演示数据里出现这种样本，等于把「数据质量差」伪装成「流水线有效」。
    """
    from src.data.taxonomy import INTENTS      # 延迟导入：sys.path 要先生效

    intents_keys = {i.key for i in INTENTS}
    bad: list[str] = []
    for intent, pairs in DIALOGS.items():
        if intent not in intents_keys:
            bad.append(f"intent '{intent}' 不在 taxonomy 的 INTENTS 里，"
                       f"生成计划会撞空")
        for q, a in pairs:
            if not (4 <= len(q) <= 120):
                bad.append(f"[{intent}] 提问长度 {len(q)} 越界：{q[:30]}")
            if not (25 <= len(a) <= 300):
                bad.append(f"[{intent}] 回复长度 {len(a)} 越界：{a[:30]}")
    if bad:
        raise SystemExit("模板校验未通过：\n  " + "\n  ".join(bad))


def build(n_clean: int, seed: int, with_images: bool,
          img_root: Path) -> tuple[list[dict], dict]:
    _validate_templates()
    rng = random.Random(seed)
    intents = list(DIALOGS)
    img_types = list(IMAGE_TYPE_STYLE)

    # ---- 图片池：每个 (intent, image_type) 一张图，保证「不同图」真的不同 ----
    img_paths: dict[tuple[str, str], str] = {}
    if with_images:
        phashes: dict[int, str] = {}
        for i, (it, itype) in enumerate(
                (it, itype) for it in intents for itype in img_types):
            p = img_root / f"{it}__{itype}.png"
            h = _make_image(p, seed=seed + i, style=IMAGE_TYPE_STYLE[itype])
            if h in phashes:
                raise RuntimeError(
                    f"pHash 撞车：{p.name} 与 {phashes[h]} 都是 {h:#x}。"
                    f"pHash 相同的两张图会被去重当成同一张，演示数据就失真了。"
                    f"请调整 _make_image 的纹理参数。")
            phashes[h] = p.name
            img_paths[(it, itype)] = str(p)

    # ---- 干净样本：**枚举 (intent, image_type, dialog) 三元组**，天然唯一 ----
    # ⚠️ 这里必须保证 (image_path, 用户提问) 两两不同，否则「120 条干净样本」
    #    自己内部就藏着无控制的重复，去重阶段会多删一批 —— 而 ground truth
    #    里没记这些，对账时就会看到「删了 25 条，只解释了 8 条」，无法收尾。
    #    早先的写法是 pool[i % len(pool)] + rng.choice(...)，同一张图会被重复
    #    抽到同一句提问，正是这个毛病。
    triples = [(it, itype, di)
               for it in intents
               for itype in img_types
               for di in range(len(DIALOGS[it]))]
    if n_clean > len(triples):
        raise SystemExit(
            f"--n-clean {n_clean} 超过了模板能提供的唯一组合数 {len(triples)}。"
            f"要么减少条数，要么给 DIALOGS 加模板 —— 否则必然产生重复样本，"
            f"ground truth 就不再精确。")
    rng.shuffle(triples)          # 打乱组合顺序，避免分布上有规律

    clean: list[dict] = []
    labels: dict[str, dict] = {}          # id → {"kind", "expect"}
    for i, (intent, itype, di) in enumerate(triples[:n_clean]):
        q, a = DIALOGS[intent][di]
        img = img_paths.get((intent, itype), "__no_image__")
        sid = _sid("clean", intent, itype, di)
        clean.append(_sample(sid, img, itype, intent,
                             difficulty=min(5, 1 + (i % 5)), query=q, answer=a))
        labels[sid] = {"kind": "clean", "expect": "keep"}

    # 唯一性自检：干净样本两两不同（这是上面那段注释承诺的性质）
    assert len({(r["image_path"], _q(r)) for r in clean}) == len(clean), (
        "干净样本里出现了 (图, 文) 重复 —— ground truth 会不精确")

    injected = {
        "clean": len(clean),
        "dirty_too_short": 0,
        "dirty_contact": 0,
        "dirty_ai_self": 0,
        "dirty_bad_structure": 0,
        "dirty_no_image_path": 0,
        "dup_same_img_same_text": 0,
        "valid_same_img_diff_text": 0,
        "valid_same_text_diff_img": 0,
        "near_dup_punctuation": 0,
        "real_black_image": 0,
    }
    rows = list(clean)

    def _first(k: str) -> dict:
        return next(r for r in clean if r["intent"] == k)

    def _srcs_with_distinct_text(n: int, step: int, offset: int) -> list[dict]:
        """挑 n 条**提问互不相同**的干净样本。

        ⚠️ 不能只用 `clean[i*step % len]` 取下标 —— 那样取到的几条可能提问
           重复，于是派生出来的「同文不同图」注入样本彼此之间也会撞车，
           去重时被删掉一条，而 ground truth 说它该保留。按提问去重再取。
        """
        seen: set[str] = set()
        out: list[dict] = []
        for i in range(len(clean)):
            r = clean[(offset + i * step) % len(clean)]
            t = _q(r)
            if t in seen:
                continue
            seen.add(t)
            out.append(r)
            if len(out) == n:
                return out
        raise SystemExit(f"找不到 {n} 条提问互不相同的干净样本（只凑到 {len(out)} 条）")

    def _add(r: dict, sid: str, kind: str, expect: str,
             group: str | None = None) -> None:
        r["id"] = sid
        rows.append(r)
        lb = {"kind": kind, "expect": expect}
        if group:
            lb["group"] = group
        labels[sid] = lb
        injected[kind] += 1

    # ---- 脏数据：期望被 ①格式校验 / ②规则过滤 拦下 ----
    # 提问过短（< min_user_len=4）
    for i in range(3):
        r = json.loads(json.dumps(_first("size_fit")))
        r["messages"][0]["content"][1]["text"] = "太紧"
        _add(r, _sid("short", i), "dirty_too_short", "rules")

    # 含联系方式
    for i, contact in enumerate(["加我微信 mywx123 详聊", "QQ 88223344 私聊吧"]):
        r = json.loads(json.dumps(_first("return_refund")))
        r["messages"][0]["content"][1]["text"] = contact
        r["messages"][1]["content"][0]["text"] = (
            "好的，您加我微信 mywx123 吧，我们私下沟通退换的细节，"
            "这样处理起来更快一些，也能给您申请一点优惠。")
        _add(r, _sid("contact", i), "dirty_contact", "rules")

    # AI 自我暴露
    for i in range(2):
        r = json.loads(json.dumps(_first("material")))
        r["messages"][0]["content"][1]["text"] = "这个料子摸着舒服吗"
        r["messages"][1]["content"][0]["text"] = (
            "作为AI，我无法为您提供触感描述，建议您参考详情页的材质说明，"
            "或者到线下门店实际体验一下手感。")
        _add(r, _sid("aiself", i), "dirty_ai_self", "rules")

    # 结构异常（messages 只有一条）
    for i in range(2):
        r = {"image_path": "__no_image__", "intent": "styling",
             "image_type": "product_main", "difficulty": 2,
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": "这个配什么裤子好看"}]}]}
        _add(r, _sid("struct", i), "dirty_bad_structure", "format")

    # 缺 image_path
    r = json.loads(json.dumps(_first("logistics")))
    r.pop("image_path")
    _add(r, _sid("noimgpath", 0), "dirty_no_image_path", "format")

    # ---- 真重复：同图 + 同文 → 去重阶段应删 ----
    # ⚠️ 这里的 ground truth 只能用**组**表达，不能指定「哪条 id 应该被删」。
    #    去重的规则是「保留先出现的、删掉后出现的」，而样本会打乱顺序 ——
    #    所以副本排在原件前面时，活下来的是副本、被删的是原件。
    #    实测就撞到了这个：按 id 逐条比对时，1 条 dup 被判成「该删却留」，
    #    同时 1 条 clean 被判成「该留却删」，计数完全正确、结论完全错误。
    #    正确的断言是「组内净剩 1 条」—— 这是**集合级**性质，不是逐条性质。
    for i, src in enumerate(_srcs_with_distinct_text(8, step=13, offset=0)):
        g = f"dupgroup_{i}"
        r = json.loads(json.dumps(src))
        _add(r, _sid("dup", i), "dup_same_img_same_text", "dedup_group", group=g)
        # 把原件的 label 也并入同一组：它的命运不再由自己决定
        labels[src["id"]] = {"kind": "clean_in_dup_group",
                             "expect": "dedup_group", "group": g}

    # ---- ⭐ 有效样本：同图 + 不同文 → **必须保留** ----
    for i, src in enumerate(_srcs_with_distinct_text(6, step=7, offset=2)):
        r = json.loads(json.dumps(src))
        # 提问带上序号，保证这 6 条彼此之间也不同文
        r["messages"][0]["content"][1]["text"] = (
            f"另外还想问一下第 {i + 1} 个问题：这件有没有更浅一点的颜色可选呢")
        _add(r, _sid("sameimg", i), "valid_same_img_diff_text", "keep")

    # ---- ⭐ 有效样本：同文 + 不同图 → **必须保留** ----
    other = img_paths.get(("color_mismatch", "user_compare"), "__no_image__")
    used_other: set[str] = set()
    for i, src in enumerate(_srcs_with_distinct_text(4, step=11, offset=1)):
        if _q(src) in used_other or (other, _q(src)) in {
                (r.get("image_path"), _q(r)) for r in rows}:
            raise SystemExit("「同文不同图」注入样本与已有样本撞车，ground truth 会失真")
        used_other.add(_q(src))
        r = json.loads(json.dumps(src))
        r["image_path"] = other
        r["messages"][0]["content"][0]["path"] = other
        _add(r, _sid("sametext", i), "valid_same_text_diff_img", "keep")

    # ---- 近似重复：只差语气词 → dedup 删 / dedup_simple 留 ----
    for i, src in enumerate(_srcs_with_distinct_text(5, step=17, offset=3)):
        r = json.loads(json.dumps(src))
        r["messages"][0]["content"][1]["text"] = _q(src) + "啊"
        _add(r, _sid("near", i), "near_dup_punctuation", "keep")

    # ---- 一张真实的全黑图（pHash = 0）----
    if with_images:
        bp = img_root / "black_blank_upload.png"
        _make_black_image(bp)
        sid = _sid("black", 0)
        rows.append(_sample(
            sid, str(bp), "defect_photo", "quality_issue", 3,
            "收到的衣服是黑的，什么图案都看不到",
            "从图片看这张照片曝光有问题，看不出细节。麻烦您在自然光下重新拍一张，"
            "我帮您确认是面料问题还是拍摄问题。"))
        labels[sid] = {"kind": "real_black_image", "expect": "keep"}
        injected["real_black_image"] = 1

    # 打乱：让脏数据分散在文件各处，避免「刚好都在开头」这种假规律
    rng.shuffle(rows)
    return rows, injected, labels


def _q(r: dict) -> str:
    """取样本的用户提问文本（取不到返回空串）。"""
    try:
        return next(c["text"] for c in r["messages"][0]["content"]
                    if c["type"] == "text")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


MANIFEST_TMPL = """# 演示数据清单（ground truth）

> 由 `scripts/make_demo_data.py` 生成，**不要手改**。
> 这份清单的意义：让下游的清洗报告可以被**对账**，而不只是「看着数字点头」。

- 生成参数：`--n-clean {n_clean} --seed {seed}`，真图：{with_images}
- 数据文件：`{out}`
- **ID 级 label（对账用这个）**：`{labels_path}`
- 图片目录：`{img_root}`

> ⚠️ 光看**条数**是核对不了的。必须用 `{labels_path}` 里的 id 做集合运算 ——
> 「一共注入 8 条重复」这种描述，在运行时没法把那 8 条对上号。
> `make demo-check` 就是干这件事的。

## 总量

| 项目 | 条数 |
| --- | --- |
| 干净样本 | {clean} |
| 注入的脏数据 + 特种样本 | {injected_total} |
| **总计** | **{total}** |

## 第一张对账表：**期望被删掉的**

| 注入类型 | 条数 | 期望被哪一步拦下 | 备注 |
| --- | --- | --- | --- |
| 提问过短 | {dirty_too_short} | ② 规则过滤 | 长度 < 4 |
| 含联系方式 | {dirty_contact} | ② 规则过滤 | 命中联系方式正则 |
| AI 自我暴露 | {dirty_ai_self} | ② 规则过滤 | 出现「作为AI」 |
| 结构异常 | {dirty_bad_structure} | ① 格式校验 | messages 只有 1 条 |
| 缺 image_path | {dirty_no_image_path} | ① 格式校验 | 字段缺失 |
| 同图 + 同文 | {dup_same_img_same_text} | ④ 去重 | 真重复 |
| 近似重复（差语气词） | {near_dup_punctuation} | ④ 去重（`dedup`）/ 保留（`dedup_simple`） | 见下方说明 |

## 第二张对账表：**必须保留的**（这张更重要）

| 注入类型 | 条数 | 期望结果 | 为什么 |
| --- | --- | --- | --- |
| 同图 + 不同文 | {valid_same_img_diff_text} | **必须保留** | 一张商品图可以配很多不同问题，都是有效样本 |
| 同文 + 不同图 | {valid_same_text_diff_img} | **必须保留** | 同一个问题配不同商品图也是有效样本 |
| 全黑图（pHash = 0） | {real_black_image} | **必须保留**，且不能算作「无图」 | 0 是合法 pHash，不是缺失哨兵 |

**为什么第二张表更重要**：一个只会删数据的流水线，删得越多指标越漂亮。
「删多了」和「删少了」在报告上都只是数字变化 —— 只有 ground truth 能区分
「删掉的是垃圾」和「删掉的是有效样本」。这两条语义（同图不同文 / 同文不同图）
也是 Day 10 讲去重时最容易写错的地方。

## 怎么用

```bash
# 1. 生成（已生成，改参数时重跑）
python scripts/make_demo_data.py

# 2. ⭐ 一键对账：跑清洗 + 和 label 逐条比，PASS/FAIL 直接给结论
make demo-check

# 3. 看清洗报告本身（人读的那一份）
cat reports/cleaning_report.md

# 4. 单独验证「同图不同文必须保留」
python -m src.data.dedup --selftest      # [2]/[3]/[3b] 三组用例就是在测这个
```

## label 的 `expect` 取值

| 值 | 含义 |
| --- | --- |
| `keep` | 应当留到最终 clean.jsonl 里 |
| `format` | 应当被 ① 格式校验拦下 |
| `rules` | 应当被 ② 规则过滤拦下 |
| `dedup` | 应当被 ④ 去重删掉 |

`demo-check` 会逐个阶段比对：一条标了 `dedup` 的样本，如果在 ① 就被拦下了，
也算 **FAIL** —— 因为它没有测到它本该测的那条规则。
「被删了」和「被**正确的**原因删了」是两件事，测试断言必须指向预期原因。

## 已知的简化（别把它当真实数据用）

- **回复是模板**：只有 8 个 intent × 几条模板，真实语料的表达多样性远高于此。
  拿去训模型可以跑通流程，但**不能**用来判断效果。
- **图片是合成纹理**：pHash 可区分，但和真实商品图的 pHash 分布不同。
  别用这份数据去校准「汉明距离阈值取多少」—— 那个阈值要在真实图上调。
- **难度标签是均匀轮转的**，不是真实难度分布。
"""


def write_manifest(path: Path, *, out: str, img_root: str, n_clean: int,
                   seed: int, with_images: bool, injected: dict,
                   labels_path: str = "") -> None:
    injected_total = sum(v for k, v in injected.items() if k != "clean")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MANIFEST_TMPL.format(
        out=out, img_root=img_root, n_clean=n_clean, seed=seed,
        labels_path=labels_path,
        with_images="是" if with_images else "否（全部 __no_image__）",
        total=injected["clean"] + injected_total, injected_total=injected_total,
        **injected), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成带 ground truth 清单的演示数据（离线，不调模型）")
    ap.add_argument("--n-clean", type=int, default=120,
                    help="干净样本条数（默认 120）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--no-images", action="store_true",
                    help="不生成真图，全部用 __no_image__ 占位符")
    ap.add_argument("--out", default="data/fixtures/demo_synth.jsonl")
    ap.add_argument("--img-root", default="data/fixtures/images")
    ap.add_argument("--manifest", default="data/fixtures/DEMO_MANIFEST.md")
    args = ap.parse_args()

    with_images = not args.no_images
    rows, injected, labels = build(args.n_clean, args.seed, with_images,
                                   Path(args.img_root))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- ID 级 label 清单：对账靠它，不靠条数 ----
    # ⚠️ 只写「一共注入了 8 条重复」是不够的 —— 运行时无法把这 8 条对上号，
    #    于是「删了 25 条、其中 8 条是重复」这种核对只能靠人眼，迟早失守。
    #    写上 id 之后，对账就是两个集合的差集运算，可以自动化、可以进 CI。
    labels_path = out.with_name(out.stem + "_labels.jsonl")
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in rows:
            lb = labels.get(r["id"], {"kind": "unknown", "expect": "keep"})
            f.write(json.dumps({"id": r["id"], **lb}, ensure_ascii=False) + "\n")

    write_manifest(Path(args.manifest), out=args.out, img_root=args.img_root,
                   n_clean=args.n_clean, seed=args.seed, with_images=with_images,
                   injected=injected, labels_path=str(labels_path))

    total = len(rows)
    print(f"✓ 写入 {total:,} 条 → {out}")
    print(f"  其中干净样本 {injected['clean']:,} 条，"
          f"注入 {total - injected['clean']:,} 条特种样本：")
    for k, v in injected.items():
        if k != "clean" and v:
            print(f"    {k:<26} {v:>4}")
    if with_images:
        n_img = len(list(Path(args.img_root).glob("*.png")))
        print(f"✓ 生成 {n_img} 张真图 → {args.img_root}")
    print(f"✓ ground truth 清单 → {args.manifest}")
    print(f"✓ ID 级 label → {labels_path}")
    print()
    print("下一步： make demo-check   （自动对账，不用人眼比对）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
