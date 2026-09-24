"""
数据清洗与去重。

对应讲义 docs/05-data-engineering.md 第 6–7 节，对应计划 Day 10。

五个环节，每一步都出统计报告：
  ① 格式校验
  ② 规则过滤
  ③ 图文一致性（CLIP 相似度）
  ④ 去重（图片 pHash + 文本 embedding）
  ⑤ 质量打分（可选，用强模型）

**输出清洗报告是这个模块的一半价值。**
没有报告，你不知道数据问题出在哪；有了报告，你才知道该补什么。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 清洗报告
# ---------------------------------------------------------------------------


@dataclass
class CleaningReport:
    n_input: int = 0
    rejected: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_after_rules: int = 0
    n_dup_image: int = 0
    n_dup_text: int = 0
    n_dup_pair: int = 0
    n_final: int = 0
    intent_dist: dict = field(default_factory=dict)
    image_type_dist: dict = field(default_factory=dict)
    difficulty_dist: dict = field(default_factory=dict)
    length_stats: dict = field(default_factory=dict)
    rejection_examples: dict = field(default_factory=lambda: defaultdict(list))

    def reject(self, reason: str, sample: Optional[dict] = None):
        self.rejected[reason] += 1
        if sample is not None and len(self.rejection_examples[reason]) < 3:
            q = ""
            try:
                q = sample["messages"][0]["content"][-1]["text"][:60]
            except Exception:
                pass
            self.rejection_examples[reason].append(q)

    def to_markdown(self) -> str:
        lines = ["# 数据清洗报告", ""]
        lines.append(f"- 输入: **{self.n_input:,}** 条")
        lines.append(f"- 规则过滤后: **{self.n_after_rules:,}** 条")
        lines.append(f"- 去重后: **{self.n_final:,}** 条")
        total_loss = self.n_input - self.n_final
        if self.n_input:
            lines.append(f"- 总淘汰率: **{total_loss / self.n_input:.1%}**")
        lines.append("")

        lines.append("## 淘汰明细")
        lines.append("")
        lines.append("| 原因 | 数量 | 占比 | 示例 |")
        lines.append("|---|---:|---:|---|")
        for reason, cnt in sorted(self.rejected.items(), key=lambda x: -x[1]):
            pct = cnt / max(self.n_input, 1)
            ex = (self.rejection_examples.get(reason) or [""])[0].replace("|", "/")
            lines.append(f"| {reason} | {cnt:,} | {pct:.1%} | {ex} |")
        lines.append(f"| 图片重复 | {self.n_dup_image:,} | {self.n_dup_image / max(self.n_input, 1):.1%} | |")
        lines.append(f"| 文本重复 | {self.n_dup_text:,} | {self.n_dup_text / max(self.n_input, 1):.1%} | |")
        lines.append(f"| 图+文整体重复 | {self.n_dup_pair:,} | {self.n_dup_pair / max(self.n_input, 1):.1%} | |")
        lines.append("")

        if self.intent_dist:
            lines.append("## 意图分布（清洗后）")
            lines.append("")
            lines.append("| 意图 | 数量 | 占比 |")
            lines.append("|---|---:|---:|")
            for k, v in sorted(self.intent_dist.items(), key=lambda x: -x[1]):
                lines.append(f"| {k} | {v:,} | {v / max(self.n_final, 1):.1%} |")
            lines.append("")

        if self.image_type_dist:
            lines.append("## 图像类型分布（清洗后）")
            lines.append("")
            lines.append("| 类型 | 数量 | 占比 |")
            lines.append("|---|---:|---:|")
            for k, v in sorted(self.image_type_dist.items(), key=lambda x: -x[1]):
                lines.append(f"| {k} | {v:,} | {v / max(self.n_final, 1):.1%} |")
            lines.append("")

        if self.difficulty_dist:
            lines.append("## 难度分布")
            lines.append("")
            for k, v in sorted(self.difficulty_dist.items()):
                bar = "█" * int(30 * v / max(self.n_final, 1))
                lines.append(f"- 难度 {k}: {v:,} {bar}")
            lines.append("")

        if self.length_stats:
            lines.append("## 长度统计")
            lines.append("")
            lines.append("| 指标 | 用户提问 | 客服回复 |")
            lines.append("|---|---:|---:|")
            for k in ("mean", "median", "p90", "max"):
                lines.append(f"| {k} | {self.length_stats['user'][k]:.0f} | "
                             f"{self.length_stats['assistant'][k]:.0f} |")
            lines.append("")

        lines.append("## 下一步建议")
        lines.append("")
        lines.append("（这一节要你自己写：看过上面这些数字之后，你认为该补哪类数据？）")
        lines.append("")
        lines.append("提示：")
        lines.append("- 某个意图占比 <5% → 需要补")
        lines.append("- 淘汰率 >15% → 合成 prompt 需要改")
        lines.append("- 难度分布偏向 1-2 → 需要主动构造难例")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 规则过滤
# ---------------------------------------------------------------------------

CONTACT_PATTERNS = [
    r"\d{11}",                      # 手机号
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+",   # 邮箱
    r"(微信|QQ|vx|VX|wx)\s*[:：]?\s*\w+",
    r"https?://",
]
BANNED_WORDS = ["作为AI", "作为一个AI", "语言模型", "我无法访问", "我没有情感"]


def rule_filter(sample: dict, min_user_len: int = 4, max_user_len: int = 120,
                min_asst_len: int = 20, max_asst_len: int = 600,
                require_chinese: bool = True) -> tuple[bool, str]:
    """规则过滤。返回 (是否保留, 原因)。"""
    import re

    try:
        msgs = sample["messages"]
        user_text = next(c["text"] for c in msgs[0]["content"] if c["type"] == "text")
        asst_text = next(c["text"] for c in msgs[1]["content"] if c["type"] == "text")
    except Exception:
        return False, "结构异常"

    if not (min_user_len <= len(user_text) <= max_user_len):
        return False, f"用户提问长度越界({len(user_text)})"
    if not (min_asst_len <= len(asst_text) <= max_asst_len):
        return False, f"回复长度越界({len(asst_text)})"

    if require_chinese:
        n_cn = sum(1 for ch in user_text + asst_text if "\u4e00" <= ch <= "\u9fff")
        if n_cn / max(len(user_text) + len(asst_text), 1) < 0.3:
            return False, "中文占比过低"

    for pat in CONTACT_PATTERNS:
        if re.search(pat, user_text + asst_text):
            return False, "含联系方式"

    for w in BANNED_WORDS:
        if w in asst_text:
            return False, f"含AI自我暴露({w})"

    # 复读检测：回复里有没有大段重复
    if len(asst_text) > 40:
        half = len(asst_text) // 2
        if asst_text[:half] == asst_text[half:half * 2]:
            return False, "回复自我重复"

    return True, "ok"


# ---------------------------------------------------------------------------
# 图文一致性（CLIP）
# ---------------------------------------------------------------------------


class ClipConsistency:
    """用 CLIP 检查图和问题是否真的相关。

    阈值是经验值：
      > 0.25  基本相关
      0.15-0.25  模糊，可能是「图片只是背景」的情况
      < 0.15  大概率无关（比如快递单配了商品图的路径）

    注意：CLIP 相似度对「图 + 短问题」本身就偏低（因为问题不是图像描述），
    所以阈值不能设太高，否则会误杀大量正常样本。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    def score(self, image, text: str) -> float:
        import torch
        with torch.no_grad():
            inputs = self.processor(text=[text], images=[image],
                                    return_tensors="pt", padding=True,
                                    truncation=True, max_length=77).to(self.device)
            out = self.model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return float((img * txt).sum())


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------


# 文本「算同一个问题」的相似度阈值（字符 2-gram 下校准）。
#
# 校准依据（实测值见 _selftest 的 [1]）：
#     完全相同                       1.00
#     只多一个语气词「我穿M码会不会太紧」vs「…太紧啊」  0.89
#     完全无关两句                    0.00
# 取 0.85 —— 落在「差语气词」（0.89）和「无关」（0.00）之间。
#
# ⚠️ 不要把 embedding 场景常见的阈值（0.90~0.95）直接搬过来：
#    两者量纲不同。embedding 对「差一个语气词」几乎不敏感（仍 0.97+），
#    字符 n-gram 却会掉到 0.89。阈值必须跟着度量方式重新校准，
#    否则会出现「阈值看着很合理、实际一条都没判出来」的哑火。
DEFAULT_THRESHOLD_TEXT = 0.85


def _usable_hash(h) -> bool:
    """这个哈希能不能用来做「同一张图」的比对。

    哨兵值是 **`None`**，表示「没有可用的图」（占位符 `__no_image__`、文件不存在、
    读取失败）。`0` 是一个**合法的 pHash**，不属于哨兵值。

    ⚠️ 为什么这里不能用 `h != 0` 来判空 —— 实测过了，这不是理论风险：
           Image.new("RGB", (256,256), (0,0,0))  →  phash == 0x0000000000000000
       纯黑图的 pHash 就是 0。而「上传失败 / 空白图 / 全黑底图」在真实商品图里
       相当常见（黑衬衫拍在黑背景上、坏掉的图床返回全黑图）。如果拿 `h != 0`
       当判空条件，这些**真实存在**的图会被当成「没有图」，去重时直接跳过图比对，
       退化成纯文本去重 —— 静默地少删一批真正重复的样本，报告上看不出来。

       反过来的写法（`0` 当哨兵、拿 `0` 去和别人比）更危险：
       `hamming_distance(0, 0) == 0 < threshold` 恒成立，于是「全黑图」和
       「无图占位符」会被判成同一张图。

    一句话：**用 0 表示缺失，就永远失去了 0 这个合法值。** 缺失必须用 None。
    """
    return h is not None


def _same_text_literal(a: str, b: str) -> bool:
    """严格字面比较（只忽略首尾空白）。

    ⚠️ 这里曾经写的是 `a[:80] == b[:80]`，只比**前 80 个字符**，与本函数
       所服务的「字面完全相同」语义自相矛盾。后果是一个静默的假重复：
       两条长提问只要前 80 字相同、后面不同，就会被判成同一条而删掉一条。
       前 80 字相同在客服语料里并不罕见（「你好，我想问一下关于这个商品
       的尺码问题……」这类开场白），所以这不是纯理论风险。
       要近似匹配请用 `_text_similarity()`，不要用截断冒充「相同」。
    """
    return a.strip() == b.strip()


def _text_similarity(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度。中文短文本够用，且零依赖。

    为什么不用 embedding：
      · 客服问句只有十几个字，字符级就够 —— 「我穿M码会不会太紧」和
        「M码我穿会紧吗」的 2-gram 重合度已经很高，不需要语义模型
      · 零依赖 = 没有 GPU 的机器也能跑完整清洗流程
      · 可解释：能说清「为什么这两条被判重复」，embedding 说不清

    要换成真正的语义去重，把这个函数替换成向量余弦即可（接口不变）。
    """
    if not a or not b:
        return 0.0
    a, b = a.strip(), b.strip()
    if a == b:
        return 1.0
    if len(a) < n or len(b) < n:
        return 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    union = ga | gb
    return len(ga & gb) / len(union) if union else 0.0


def dedup(samples: list[dict], image_hashes: dict[str, Optional[int]],
          threshold_img: int = 5,
          threshold_text: float = DEFAULT_THRESHOLD_TEXT,
          same_image_diff_text_ok: bool = True) -> tuple[list[dict], dict]:
    """去重。返回 (保留的样本, 统计)。

    关键设计：**去重的是「图 + 文」这个整体，不是单独的图或文。**

    同一个商品图配不同的用户问题 → 是**有效样本**，要保留
    同一个问题配不同的商品图     → 也是**有效样本**，要保留
    同一张图 + 同一个问题        → 重复，删掉

    所以我们用组合键判断，而不是分别去重。

    ⚠️ 「组合键」的意思是：**必须是同一条已保留样本同时满足「同图」和「同文」**。
       这里踩过一个很隐蔽的坑 —— 分别判断再取交集：

           同图？（和任意一条比）  → True
           同文？（和任意一条比）  → True
           ⇒ 判为重复

       看起来等价，实际会误删。反例：
           已保留 (图A, 问题1) 和 (图B, 问题2)
           新来   (图A, 问题2)
       「图 A 出现过」且「问题 2 出现过」，但这个**组合是新的** ——
       它是有效样本，应该保留。分别判断会把它删掉。
       正确做法是逐条检查 (kh, kt) 是否同时匹配，见下面 _dup_of 的遍历。

    参数：
      threshold_img:            pHash 汉明距离小于它算「同一张图」
      threshold_text:           文本相似度达到它算「同一个问题」
                                默认值在 DEFAULT_THRESHOLD_TEXT（已按 2-gram 校准）
      same_image_diff_text_ok:  True（推荐）保留「同图不同文」

    ⚠️ 这个函数另外还踩过两处坑，都是「未完成实现伪装成可用接口」：
       ① 一段永远不会正确执行的代码
              for _, other_idx in [(x, y) for x, y in same_img_texts if str(y) == "0"]
          而 same_img_texts 里装的是**字符串**，把字符串解包成两个变量会直接
          ValueError —— 只要走到那条分支就崩。
       ② threshold_text 参数从头到尾没被用过，等于「传了不生效」。
       调用方以为自己在调一个近似去重器，实际拿到的却是别的东西。
       现在文本相似度由 _text_similarity 真实实现，threshold_text 真正生效。

    ⚠️ 复杂度：O(n²)（每条都和已保留的比）。1 万条以内可以接受，
      再大就需要分桶（按 phash 前缀分桶）+ 倒排索引。别直接喂 10 万条。
    """
    from .image_utils import hamming_distance

    stats: dict[str, int] = {
        "dup_pair": 0,              # 同图 + 同文（同一条）→ 删
        "dup_image": 0,             # 同图（仅当要求删同图时）→ 删
        "dup_text": 0,              # 同文不同图 → 保留，仅记录
        "same_image_diff_text": 0,  # 同图不同文 → 保留，仅记录
        "no_usable_hash": 0,        # 没有可用图片的条数（健康指标，见 _usable_hash）
    }

    texts: list[str] = []
    for s in samples:
        try:
            texts.append(next(c["text"] for c in s["messages"][0]["content"]
                              if c["type"] == "text"))
        except Exception:
            texts.append("")

    kept: list[dict] = []
    # (phash 或 None, 问题文本, 是否有可用图)。
    # 三个字段必须绑在一起遍历，不能拆成几个 list —— 拆开就会退化成
    # 「同图」和「同文」分别判断，导致组合键是新的却被误删（见函数文档）。
    kept_keys: list[tuple[Optional[int], str, bool]] = []

    for i, s in enumerate(samples):
        raw_h = image_hashes.get(s.get("image_path", ""))
        has_img = _usable_hash(raw_h)
        h = raw_h if has_img else None
        text = texts[i]
        if not has_img:
            stats["no_usable_hash"] += 1

        is_dup = False
        saw_same_image = False
        saw_same_text = False

        for kh, kt, k_has_img in kept_keys:
            txt_same = _text_similarity(text, kt) >= threshold_text
            if has_img and k_has_img:
                # 两边都有可用图 → 必须「同图 + 同文」才算重复
                img_same = hamming_distance(h, kh) < threshold_img
                if img_same and txt_same:
                    is_dup = True        # 同一条上「图+文」都撞了 → 真重复
                    break
                saw_same_image = saw_same_image or img_same
            else:
                # 有人没图 → 没法判图，退化成纯文本近似去重
                if txt_same:
                    is_dup = True
                    break
            saw_same_text = saw_same_text or txt_same

        if is_dup:
            stats["dup_pair"] += 1
            continue
        if saw_same_image and not same_image_diff_text_ok:
            stats["dup_image"] += 1
            continue
        if saw_same_image:
            stats["same_image_diff_text"] += 1
        elif saw_same_text:
            stats["dup_text"] += 1

        kept.append(s)
        kept_keys.append((h, text, has_img))

    return kept, stats


def dedup_simple(samples: list[dict], image_hashes: dict[str, Optional[int]],
                 threshold_img: int = 5) -> tuple[list[dict], dict]:
    """简化版去重，不依赖 embedding 模型。Day 10 先用这个跑通。

    语义（比 dedup 更保守，只做字面完全相同）：
      两边都有可用图片 → 必须「同图 + 同文」才算重复
      至少一边没有图   → 无法比对图，退化成纯文本去重（同文即重复）

    为什么无图时要退化成纯文本去重：连图都读不到，就没有任何依据去区分
    「同一个问题配了两张不同的图」和「同一条数据被抄了两遍」。
    这时按文本去重是唯一能做的事，也是最小的删除量。

    `image_hashes` 的约定：**值可能是 `None`**（表示这张图没读到）。
    ⚠️ 不要写成 `image_hashes.get(img, 0)`。用 `0` 当「没读到」的默认值，
       会和「全黑图的真实 pHash 恰好是 0」撞在一起 —— 详见 `_usable_hash`。
       默认值必须是 `None`，而 `None` 是不能省的：`.get(img)` 的隐式默认
       也是 `None`，但写出来才能让人一眼看懂这里允许缺失。
    """
    from .image_utils import hamming_distance

    stats = {"dup_pair": 0, "dup_text_only": 0, "no_usable_hash": 0}
    kept: list[dict] = []
    seen: list[tuple[Optional[int], str, bool]] = []   # (hash 或 None, 文本, 是否可用)

    for s in samples:
        img = s.get("image_path", "")
        raw_h = image_hashes.get(img, None)
        has_img = _usable_hash(raw_h)
        h = raw_h if has_img else None
        if not has_img:
            stats["no_usable_hash"] += 1

        try:
            text = next(c["text"] for c in s["messages"][0]["content"]
                        if c["type"] == "text")
        except Exception:
            text = ""

        is_dup = False
        dup_kind = ""
        for kh, ktext, k_has_img in seen:
            same_text = _same_text_literal(ktext, text)
            if has_img and k_has_img:
                # 两边都有图 → 必须同图 + 同文
                same_img = hamming_distance(h, kh) < threshold_img
                if same_img and same_text:
                    is_dup, dup_kind = True, "pair"
                    break
            else:
                # 有人没图 → 没法判图，退化成纯文本
                if same_text:
                    is_dup, dup_kind = True, "text"
                    break

        if is_dup:
            stats["dup_pair" if dup_kind == "pair" else "dup_text_only"] += 1
            continue

        seen.append((h, text, has_img))
        kept.append(s)

    return kept, stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


@dataclass
class CleanStages:
    """清洗流水线每一步之后的结果，**保留每一阶段的样本列表**。

    为什么要把中间结果留下来，而不是像早期版本那样在函数里一路 `kept2 = ...`
    覆盖掉：

      「删掉了多少条」是可以从计数器上看到的，但「**删对了吗**」看不到。
      想知道删对没删对，必须能回答「这一条是在哪一步被删的」——
      那就得留下每一阶段的成员。这也是 `scripts/reconcile_demo.py` 能
      自动对账的前提：它拿 ground truth 的 id 去逐阶段查，而不是靠条数猜。

    阶段命名和报告里的编号对齐：format / rules / clip / dedup。
    """
    raw: list[dict]
    after_format: list[dict]
    after_rules: list[dict]
    after_clip: list[dict]                 # 未启用 CLIP 时 == after_rules
    after_dedup: list[dict]
    deletes: dict[str, list[str]]           # 阶段名 → 被删样本的 id 列表
    reasons: dict[str, str]                 # id → 淘汰原因（人读）
    image_hashes: dict[str, Optional[int]]
    report: CleaningReport

    def stage_of(self, sid: str) -> str:
        """这条样本最后停在哪一步。'keep' 表示活到了最后。"""
        for st in ("format", "rules", "clip", "dedup"):
            if sid in self.deletes.get(st, ()):
                return st
        return "keep"

    def ids_at(self, stage: str) -> set[str]:
        m = {"input": self.raw, "format": self.after_format,
             "rules": self.after_rules, "clip": self.after_clip,
             "dedup": self.after_dedup}
        return {s.get("id", f"__idx_{i}") for i, s in enumerate(m[stage])}


def run_stages(samples: list[dict], use_clip: bool = False,
               min_clip_score: float = 0.15,
               verbose: bool = True) -> CleanStages:
    """跑完 ①格式 → ②规则 → ③图文一致性 → ④去重，返回每阶段结果。

    和 `clean_pipeline` 的区别：这里**不碰文件系统**（不读 jsonl、不写输出、
    不写报告）。纯净的「样本进、样本出」，所以可以在单元测试和演示数据对账里
    反复调用，也可以塞进 notebook 逐格看中间结果。
    """
    from .image_utils import load_and_normalize, phash

    rep = CleaningReport(n_input=len(samples))
    if verbose:
        print(f"输入 {len(samples):,} 条")
    deletes: dict[str, list[str]] = {k: [] for k in ("format", "rules", "clip", "dedup")}
    reasons: dict[str, str] = {}

    def _reject(stage: str, s: dict, reason: str) -> None:
        sid = s.get("id") or f"__noid_{len(reasons)}"
        deletes[stage].append(sid)
        reasons[sid] = reason
        rep.reject(reason.split("(")[0], s)

    # --- ① 格式校验 ---
    ok_samples = []
    for s in samples:
        if not isinstance(s.get("messages"), list) or len(s["messages"]) < 2:
            _reject("format", s, "结构异常")
            continue
        if not s.get("image_path"):
            _reject("format", s, "缺少图片路径")
            continue
        ok_samples.append(s)
    if verbose:
        print(f"① 格式校验后: {len(ok_samples):,}")

    # --- ② 规则过滤 ---
    kept2 = []
    for s in ok_samples:
        ok, reason = rule_filter(s)
        if ok:
            kept2.append(s)
        else:
            _reject("rules", s, reason)
    rep.n_after_rules = len(kept2)
    if verbose:
        print(f"② 规则过滤后: {len(kept2):,}")

    # --- ③ 图片哈希（用于去重 + 图文一致性）---
    # 约定：值 = 读到的 pHash；值 = None = 没有可用的图（占位符 / 文件缺失 / 读取失败）
    image_hashes: dict[str, Optional[int]] = {}
    valid_images = {}
    n_placeholder = n_unreadable = n_zero_hash = 0
    for s in kept2:
        p = s["image_path"]
        if p in image_hashes:
            continue
        if p == "__no_image__" or not Path(p).exists():
            image_hashes[p] = None
            n_placeholder += 1
            continue
        try:
            img, _ = load_and_normalize(p)
            h = phash(img)
            image_hashes[p] = h
            valid_images[p] = img
            # ⚠️ 全黑图的 pHash 就是 0（实测确认）。现在 0 是合法值、不会
            #    被当成缺失，但出现多个「0」时它们会互相判为同图 —— 值得报出来。
            if h == 0:
                n_zero_hash += 1
        except Exception:
            image_hashes[p] = None
            n_unreadable += 1

    # ⚠️ 这里必须把「真算出来的」和「占位/读不到的」分开报。
    #    早期版本打印 len(image_hashes)，把 __no_image__ 占位符也算成
    #    「一张图的指纹」，于是一张都没算的情况下打出「计算 1 张图的指纹」。
    #    和「分母塌成 0」是同一类问题：数字看起来正常，其实什么都没做。
    n_real = sum(1 for v in image_hashes.values() if v is not None)
    if verbose:
        print(f"③ 图片指纹：{n_real:,} 张真算 · {n_placeholder:,} 张占位（无图）"
              f" · {n_unreadable:,} 张读取失败")
        if n_unreadable:
            print(f"   ⚠️ 有 {n_unreadable:,} 张图文件存在但读不出来，"
                  f"这些样本将退化为纯文本去重")
        if n_zero_hash:
            print(f"   ⚠️ 有 {n_zero_hash:,} 张真实图片的 pHash 恰好为 0"
                  f"（通常是全黑图/空白图）。它们彼此会被判为同一张图。")

    # --- ③b 图文一致性（可选，慢）---
    kept_clip = kept2
    if use_clip:
        try:
            clip = ClipConsistency()
            kept3 = []
            for s in kept2:
                p = s["image_path"]
                img = valid_images.get(p)
                if img is None:
                    kept3.append(s)
                    continue
                try:
                    txt = next(c["text"] for c in s["messages"][0]["content"]
                               if c["type"] == "text")
                    sc = clip.score(img, txt)
                    if sc < min_clip_score:
                        _reject("clip", s, "图文不相关")
                        continue
                except Exception:
                    pass
                kept3.append(s)
            kept_clip = kept3
            if verbose:
                print(f"③b CLIP 过滤后: {len(kept_clip):,}")
        except ImportError:
            if verbose:
                print("   （跳过 CLIP：未安装 transformers/clip）")

    # --- ④ 去重 ---
    kept4, dstats = dedup_simple(kept_clip, image_hashes)
    rep.n_dup_pair = dstats["dup_pair"]
    rep.n_final = len(kept4)
    kept4_ids = {s.get("id") for s in kept4}
    for s in kept_clip:
        sid = s.get("id")
        if sid not in kept4_ids and sid not in deletes["dedup"]:
            # 可能是无 id 的样本，用位置兜底没意义；只记录有 id 的
            if sid:
                deletes["dedup"].append(sid)
                reasons[sid] = "图+文整体重复"
    if verbose:
        print(f"④ 去重后: {len(kept4):,}")

    return CleanStages(
        raw=samples, after_format=ok_samples, after_rules=kept2,
        after_clip=kept_clip, after_dedup=kept4,
        deletes=deletes, reasons=reasons,
        image_hashes=image_hashes, report=rep,
    )


def clean_pipeline(raw_path: str | Path,
                   out_path: str | Path = "data/processed/clean.jsonl",
                   image_root: Optional[str] = None,
                   use_clip: bool = False,
                   min_clip_score: float = 0.15,
                   report_path: str | Path = "reports/cleaning_report.md",
                   ) -> CleaningReport:
    """完整的清洗流水线：读 jsonl → 五步清洗 → 写 clean.jsonl + 报告。

    真正的清洗逻辑在 `run_stages()` 里；本函数只负责文件 I/O 和落盘。
    这样拆的理由：清洗逻辑要被单测、要被演示数据对账脚本反复调用，
    那些场景都不该碰磁盘。
    """
    raw_path = Path(raw_path)
    # ⚠️ 输入文件不存在时必须**当场报错**，不能当成「读到 0 条」往下走。
    #    实测踩过：`make data-clean` 在默认输入 `data/raw/synth_v0.jsonl`
    #    不存在时（还没跑 Day 9 的合成），会打印
    #        ✓ 输出 0 条 → data/processed/clean.jsonl
    #        ✓ 报告 → reports/cleaning_report.md
    #    然后**退出码 0**。于是一条 0 条的 clean.jsonl 覆盖掉了已有的数据，
    #    而 make 认为这一步成功了，继续往下跑 Day 11 —— 全链路在空数据上
    #    一路「成功」到训练。
    #    这和「0 泄漏」「幻觉率 0%」是同一个东西：**分母塌成 0 时，
    #    任何『成功』都成立。** 缺文件是故障，不是空数据集。
    if not raw_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件 {raw_path}。\n"
            f"  · 还没跑 Day 9 的数据合成？→ 先跑 `make data-synth`\n"
            f"  · 想在没有 API key 的机器上跑通？→ 用 `make demo-check`，"
            f"或者显式指定演示数据：\n"
            f"      make data-clean RAW=data/fixtures/demo_synth.jsonl")

    samples = []
    n_bad_json = 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    n_bad_json += 1
    if n_bad_json:
        print(f"⚠️ 有 {n_bad_json} 行不是合法 JSON，已跳过")

    if not samples:
        # 文件存在但一条有效样本都没有 —— 同样不能当成功
        raise ValueError(
            f"{raw_path} 存在但里面一条样本都没有（共 {n_bad_json} 行坏 JSON）。"
            f"别让它生成一份空的 clean.jsonl 覆盖掉已有数据。")

    st = run_stages(samples, use_clip=use_clip, min_clip_score=min_clip_score)
    rep = st.report
    kept4 = st.after_dedup

    # --- 统计 ---
    rep.intent_dist = dict(Counter(s.get("intent", "?") for s in kept4))
    rep.image_type_dist = dict(Counter(s.get("image_type", "?") for s in kept4))
    rep.difficulty_dist = dict(Counter(s.get("difficulty", 0) for s in kept4))

    ulens, alens = [], []
    for s in kept4:
        try:
            ulens.append(len(next(c["text"] for c in s["messages"][0]["content"]
                                  if c["type"] == "text")))
            alens.append(len(next(c["text"] for c in s["messages"][1]["content"]
                                  if c["type"] == "text")))
        except Exception:
            continue
    if ulens:
        rep.length_stats = {
            "user": _stat(ulens),
            "assistant": _stat(alens),
        }

    # --- 报告先落盘 ---
    # 报告是**诊断产物**，不是数据。即使清洗失败也要写出来 —— 一条都没留下的
    # 时候，恰恰是你最需要「淘汰明细」的时候。它也不会被任何下游当数据读，
    # 所以写它没有污染风险。数据文件（clean.jsonl）则相反，见下。
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rep.to_markdown(), encoding="utf-8")

    # --- 输出数据 ---
    # ⚠️ 清洗后一条不剩 = 上游或规则出了大问题，**不许静默落盘**。
    #    写下一份空的 clean.jsonl 会覆盖掉已有数据，而调用方（make / shell）
    #    看到的是「成功」。宁可炸掉，也不要让下游在空数据上继续跑。
    if not kept4:
        raise ValueError(
            f"清洗后一条样本都不剩（输入 {len(samples)} 条）。\n"
            f"  淘汰明细 → {report_path}（报告已写出，先看它）\n"
            f"  这种情况下**不会写出 {out_path}** —— 免得空文件覆盖掉已有数据。")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in kept4:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✓ 输出 {len(kept4):,} 条 → {out_path}")
    print(f"✓ 报告 → {report_path}")

    return rep


def _stat(v: list[int]) -> dict:
    a = np.array(v)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)), "max": int(a.max())}


def _selftest() -> int:
    """离线自检：不读外部文件、不写盘。

    重点覆盖 dedup() 的四条语义（这是整个模块最容易写错的地方）：
        同图 + 同文 → 删          同图 + 不同文 → 保留
        同文 + 不同图 → 保留       都不重复 → 保留

    另有两条语义陷阱（都真踩过，不是理论风险）：
        · 哨兵值必须是 None；0 是**合法** pHash（全黑图实测就是 0）
        · 「字面完全相同」必须比完整文本，不能截前 80 字符冒充

    以及 [6] 落地守卫：缺输入 / 空输入 / 清洗后为空，一律报错且不写数据文件。
    [6] 是本自检里唯一会碰磁盘的用例，用临时目录，跑完自动清理。
    """
    print("=" * 76)
    print("清洗与去重自检（离线，不读文件、不写盘）")
    print("=" * 76)

    def _sample(img: str, q: str,
                a: str = "您好，这款是宽松直筒版型，M 码肩宽 38cm、胸围 100cm，"
                         "平时穿 M 的话这件是合适的。") -> dict:
        return {
            "image_path": img,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "path": img},
                    {"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ],
        }

    # 1. 文本相似度（同时校准 DEFAULT_THRESHOLD_TEXT）
    print("\n[1] 文本相似度 _text_similarity()")
    same = _text_similarity("我穿M码会不会太紧", "我穿M码会不会太紧")
    near = _text_similarity("我穿M码会不会太紧", "我穿M码会不会太紧啊")
    far = _text_similarity("我穿M码会不会太紧", "这个颜色是米白吗")
    print(f"    完全相同                : {same:.2f}")
    print(f"    「…太紧」vs「…太紧啊」  : {near:.2f}")
    print(f"    两句不相关              : {far:.2f}")
    print(f"    阈值 DEFAULT_THRESHOLD_TEXT = {DEFAULT_THRESHOLD_TEXT}")
    assert same == 1.0
    # ⚠️ 断言绑阈值而不是写死数字：改了阈值这里会立刻发现不一致。
    #    写死 0.9 的话，阈值调到 0.92 就会出现「测试绿、功能哑火」。
    assert near >= DEFAULT_THRESHOLD_TEXT, (
        f"只差一个语气词的相似度 {near:.2f} 低于阈值 "
        f"{DEFAULT_THRESHOLD_TEXT} —— 这种几乎一样的问题会被判成不重复")
    assert far < DEFAULT_THRESHOLD_TEXT, (
        f"不相关两句的相似度 {far:.2f} 高于阈值，会误杀正常样本")
    assert _text_similarity("", "abc") == 0.0, "空串不该被判为相似"
    print("    ✓ 阈值落在「差语气词」和「不相关」之间（正确校准）")

    # 2. ⭐ 图 + 文整体去重（core 语义）
    print("\n[2] 图+文整体去重 dedup()")
    # a 与 c 的 pHash 只差 1 位（判为同一张图），b 差得远（不同图）
    H = {"a.jpg": 0b1111101000,
         "c.jpg": 0b1111101001,
         "b.jpg": 0b101010101010101010}
    samples = [
        _sample("a.jpg", "我穿M码会不会太紧"),    # 0 首条 → 留
        _sample("a.jpg", "我穿M码会不会太紧"),    # 1 同图 + 同文 → 删
        _sample("a.jpg", "这个颜色是米白吗"),      # 2 同图 + 不同文 → 留
        _sample("b.jpg", "我穿M码会不会太紧"),    # 3 同文 + 不同图 → 留
        _sample("c.jpg", "我穿M码会不会太紧啊"),  # 4 同图 + 近似文 → 删
        _sample("b.jpg", "这个颜色是米白吗"),      # 5 同文 + 不同图 → 留
    ]
    kept, st = dedup(samples, H)
    print(f"    输入 {len(samples)} 条 → 保留 {len(kept)} 条")
    print(f"    {st}")
    assert len(kept) == 4, f"应保留 4 条，实际 {len(kept)}"
    assert st["dup_pair"] == 2, f"同图+同文应删 2 条，实际 {st['dup_pair']}"
    assert st["same_image_diff_text"] >= 1, \
        "同图不同文必须保留 —— 一张商品图可以配很多不同的问题，这是有效样本"
    assert st["dup_text"] >= 1, \
        "同文不同图必须保留 —— 同一个问题配不同商品图也是有效样本"
    assert len(kept) == len(samples) - st["dup_pair"] - st["dup_image"]
    print("    ✓ 同图同文删 / 同图不同文留 / 同文不同图留（三条语义都对）")

    # 2b. 关掉「同图不同文保留」之后应该删更多
    kept_strict, st2 = dedup(samples, H, same_image_diff_text_ok=False)
    assert len(kept_strict) < len(kept), "关掉同图保留后应该删得更多"
    assert st2["dup_image"] >= 1
    print(f"    same_image_diff_text_ok=False → 保留 {len(kept_strict)} 条"
          f"（多删 {len(kept) - len(kept_strict)} 条）")

    # 3. dedup_simple（Day 10 主路径）—— 它是「精确字面」版，删得比 dedup 少
    print("\n[3] dedup_simple()（Day 10 先用这个跑通）")
    kept_s, st_s = dedup_simple(samples, H)
    print(f"    保留 {len(kept_s)} 条（dedup 保留 {len(kept)} 条）")
    assert st_s["dup_pair"] >= 1
    assert len(kept_s) >= len(kept), \
        "精确版不该比近似版删得更多 —— 两者差异只在「近似文本」这一条"
    print("    ✓ 两者差异符合设计：simple 只做字面完全相同，不做近似")

    # 3b. ⭐ 回归：哨兵值必须是 None，且 0 是**合法** pHash
    #     背景：3a 那轮端到端验证我怀疑过「无图样本的 h=0 被当成同图」，
    #     当时看到「65 条去重成 6 条」就下了定论。后来查明那是**误报** ——
    #     那份 fixture 里同一句话被复制了 11 遍，65 条只有 6 个唯一提问，
    #     去重成 6 条本来就是正确答案。误报不等于没问题，本组用例是重查
    #     之后真正站得住的结论：
    #       · 用 0 表示缺失，会把「纯黑图的真实 pHash = 0」也一起判成缺失
    #         （Image.new("RGB",(256,256),(0,0,0)) 的 pHash 实测就是 0）
    #       · 所以缺失一律用 None，0 完整让给合法哈希
    print("\n[3b] 哨兵值语义：None = 没有图，0 = 合法 pHash（全黑图）")
    noimg = [_sample("__no_image__", f"第 {i} 个完全不同的提问内容，长度也够")
             for i in range(10)]
    no_img_hashes: dict = {"__no_image__": None}      # ← 哨兵值 None
    kept_n, st_n = dedup_simple(noimg, no_img_hashes)
    print(f"    10 条各不相同的无图提问 → 保留 {len(kept_n)} 条 {st_n}")
    assert len(kept_n) == 10, (
        f"10 条各不相同的无图样本应全部保留，实际只剩 {len(kept_n)} 条")
    assert st_n["no_usable_hash"] == 10, "应统计出 10 条没有可用图"
    dup_noimg = noimg + [_sample("__no_image__", "第 3 个完全不同的提问内容，长度也够")]
    kept_d, _ = dedup_simple(dup_noimg, no_img_hashes)
    assert len(kept_d) == 10, f"只该删掉那 1 条真重复，实际 {len(kept_d)}"

    # ⭐ 全黑图（pHash=0）：是**有图**，不能被当成没有图
    black = {"black_a.png": 0, "black_b.png": 0}
    assert _usable_hash(0) is True, "0 是合法 pHash，不能当缺失"
    assert _usable_hash(None) is False, "None 才是缺失"
    black_samples = [
        _sample("black_a.png", "这件黑色上衣的领口是不是有点大"),   # 全黑图 1
        _sample("black_b.png", "这件黑色上衣的领口是不是有点大"),   # 全黑图 2 + 同文 → 删
        _sample("black_a.png", "这个袖长具体是多少厘米呢"),          # 全黑图 1 + 不同文 → 留
    ]
    kept_b, st_b = dedup_simple(black_samples, black)
    print(f"    3 条全黑图样本（hash 都是 0）→ 保留 {len(kept_b)} 条 {st_b}")
    assert st_b["no_usable_hash"] == 0, (
        "全黑图的 pHash 是 0，但它是**有图**的样本，不该被算进 no_usable_hash")
    assert len(kept_b) == 2, (
        f"全黑图应该按真实图片参与比对：同图同文删 1 条 → 保留 2 条，"
        f"实际 {len(kept_b)} 条")
    print("    ✓ None 表示缺失 / 0 表示全黑图，两者不再互相冒充")

    # 3c. ⭐ 回归：前 80 字符相同、后面不同的长提问不能被判成同一条
    #     原实现是 text[:80] == text[:80]，与「字面完全相同」的声明矛盾。
    print("\n[3c] 长提问前 80 字相同但后面不同")
    head = "你好，我想咨询一下关于这个商品的尺码问题，我平时穿的码数比较特殊，"
    head = head[:80] if len(head) >= 80 else head + "尺" * (80 - len(head))
    q_a = head + "所以想确认一下 M 码的肩宽具体是多少厘米。"
    q_b = head + "所以想问一下这件衣服能不能七天无理由退换货。"
    assert q_a[:80] == q_b[:80], "用例本身没构造出「前 80 字相同」"
    assert q_a != q_b
    two = [_sample("__no_image__", q_a), _sample("__no_image__", q_b)]
    kept_p, _ = dedup_simple(two, no_img_hashes)
    print(f"    前 80 字: {q_a[:36]}…（两条一致）")
    print(f"    后段不同 → 保留 {len(kept_p)} 条")
    assert len(kept_p) == 2, (
        f"两条提问后半段完全不同，应都保留，实际只剩 {len(kept_p)} 条 —— "
        f"疑似又按前 80 字符截断比较了")
    print("    ✓ 只比完整文本，不按长度截断")

    # 4. 规则过滤
    #    ⚠️ 断言要带**预期原因**。只断言「被过滤了」的话，一条本来想测
    #       「含联系方式」的用例，如果恰好因为别的原因（比如回复太短）
    #       被拦下，测试照样是绿的 —— 但它根本没测到联系方式的正则。
    print("\n[4] 规则过滤 rule_filter()")
    ok, why = rule_filter(_sample("a.jpg", "我穿M码会不会太紧呀，平时穿M比较多"))
    assert ok, f"正常样本应保留，实际: {why}"
    bads = [
        (_sample("a.jpg", "太紧"), "提问过短", "长度"),
        (_sample("a.jpg", "加我微信 mywx123 详聊",
                 "好的，您加我微信 mywx123 吧，我们私下沟通退换的细节，"
                 "这样处理起来更快一些。"),
         "含联系方式", "联系方式"),
        (_sample("a.jpg", "这个料子摸着舒服吗",
                 "作为AI，我无法为您提供触感描述，建议参考详情页的材质说明。"),
         "AI 自我暴露", "AI自我暴露"),
        ({"image_path": "a.jpg", "messages": [{"content": []}]},
         "结构异常", "结构异常"),
    ]
    for s, hint, expect_in in bads:
        ok2, why2 = rule_filter(s)
        assert not ok2, f"「{hint}」应被过滤"
        assert expect_in in why2, f"「{hint}」被过滤的原因不对：{why2}"
        print(f"    {hint:<12} → 过滤（{why2}）")
    print("    ✓ 结构 / 长度 / 联系方式 / AI 自我暴露 都按预期原因拦下")

    # 5. 报告
    print("\n[5] 清洗报告 CleaningReport")
    rep = CleaningReport(n_input=100)
    rep.reject("含联系方式", _sample("a.jpg", "加我微信 xyz"))
    rep.n_after_rules = 90
    rep.n_dup_pair = 3
    rep.n_final = 87
    md = rep.to_markdown()
    assert "# 数据清洗报告" in md
    assert "含联系方式" in md, "淘汰明细里应列出拒绝原因"
    assert "图+文整体重复" in md
    assert "13.0%" in md, f"淘汰率应为 13.0%，报告里没有：{md[:200]}"
    print(f"    报告 {len(md.splitlines())} 行 · 含淘汰明细 / 分布 / 长度统计")
    print("    ✓ 报告可生成且数字自洽")

    # 6. ⭐ 回归：缺输入 / 空输入 / 全被滤掉，都必须**报错**，不许静默成功
    #    实测踩过：`make data-clean` 在默认输入 data/raw/synth_v0.jsonl 不存在时
    #    打印
    #        ✓ 输出 0 条 → data/processed/clean.jsonl
    #        ✓ 报告 → reports/cleaning_report.md
    #    然后**退出码 0**。一条 0 条的 clean.jsonl 覆盖掉了已有数据，而 make
    #    认为这一步成功了，继续往下跑 Day 11 —— 全链路在空数据上一路「成功」
    #    直到训练出个没用的模型。
    #    这是本模块反复出现的同一个主题：**分母塌成 0 时，任何结论都成立。**
    #    「成功的空结果」和「真的成功」长得一模一样。缺文件是**故障**，
    #    不是「这个数据集恰好是空的」。
    print("\n[6] 守卫：不许写出空的 clean.jsonl")
    import contextlib
    import io
    import tempfile

    def _run_quiet(inp, out) -> Optional[BaseException]:
        """跑一遍 clean_pipeline，吞掉它打印的日志（自检输出才看得清）。
        返回异常对象或 None —— 不抛出，让用例自己去断言类型。"""
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                clean_pipeline(inp, out, report_path=Path(out).parent / "rep.md")
                return None
            except BaseException as e:        # noqa: BLE001 —— 要按类型断言
                return e

    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        out_p = td / "clean.jsonl"
        rep_p = td / "rep.md"

        # 6a. 输入文件压根不存在
        e = _run_quiet(td / "not_there.jsonl", out_p)
        assert isinstance(e, FileNotFoundError), \
            f"输入文件不存在时应抛 FileNotFoundError，实际 {type(e).__name__}: {e}"
        assert "demo-check" in str(e) or "data-synth" in str(e), \
            f"报错要告诉用户下一步怎么做，实际信息：{e}"
        assert not out_p.exists(), "输入缺失时绝不能写出输出文件"
        print("    6a 输入不存在        → FileNotFoundError ✓（未写输出）")

        # 6b. 文件存在但是空的
        (td / "empty.jsonl").write_text("", encoding="utf-8")
        e = _run_quiet(td / "empty.jsonl", out_p)
        assert isinstance(e, ValueError), \
            f"空输入应抛 ValueError，实际 {type(e).__name__}: {e}"
        assert not out_p.exists(), "空输入时绝不能写出输出文件"
        print("    6b 输入存在但是空文件 → ValueError ✓（未写输出）")

        # 6c. 全是坏 JSON —— 同样是「读到 0 条」，同样不能当成功
        (td / "bad.jsonl").write_text('{"id": 1,,}\nnot json at all\n', encoding="utf-8")
        e = _run_quiet(td / "bad.jsonl", out_p)
        assert isinstance(e, ValueError), \
            f"全是坏 JSON 应抛 ValueError，实际 {type(e).__name__}: {e}"
        assert not out_p.exists()
        print("    6c 全是坏 JSON        → ValueError ✓（坏行计数 + 未写输出）")

        # 6d. ⭐ 输入非空、格式也合法，但**全被规则滤掉**
        #     这条最容易被漏掉：前面两道守卫都拦不住它（输入确实有 5 条）。
        rows = [{"id": f"b{i}", "image_path": "__no_image__",
                 "intent": "x", "image_type": "y", "difficulty": 1,
                 "messages": [
                     {"role": "user", "content": [{"type": "text", "text": "太紧"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "嗯。"}]}]}
                for i in range(5)]
        (td / "allbad.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        e = _run_quiet(td / "allbad.jsonl", out_p)
        assert isinstance(e, ValueError), \
            f"清洗后 0 条应抛 ValueError（而不是写出空文件），实际 {type(e).__name__}: {e}"
        assert not out_p.exists(), "清洗后为空时必须**不写数据文件**，否则会覆盖已有数据"
        assert rep_p.exists(), \
            "清洗失败时报告**反而更要写** —— 一条都没留下的时候最需要淘汰明细"
        print("    6d 输入非空但全被滤掉 → ValueError ✓（报告写了 · 数据未写）")

        # 6e. 反例：只要留下一条就该正常写盘 —— 守卫不能误伤正常路径
        okrow = {"id": "ok1", "image_path": "__no_image__",
                 "intent": "x", "image_type": "y", "difficulty": 1,
                 "messages": [
                     {"role": "user",
                      "content": [{"type": "text",
                                   "text": "我穿M码会不会太紧呀，平时穿M比较多"}]},
                     {"role": "assistant",
                      "content": [{"type": "text",
                                   "text": "M 码胸围 96 厘米，比您平时的码数大半码，应该合适。"}]}]}
        (td / "ok.jsonl").write_text(json.dumps(okrow, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
        e = _run_quiet(td / "ok.jsonl", out_p)
        assert e is None, f"正常输入不该报错，实际 {type(e).__name__}: {e}"
        assert out_p.exists(), "正常输入必须写出输出文件"
        n_out = len([l for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()])
        assert n_out == 1, f"应写出 1 条，实际 {n_out} 条"
        print(f"    6e 正常输入（留 1 条）→ 正常写盘 {n_out} 条 ✓（守卫未误伤）")

    print("    ✓ 三条守卫都生效：缺文件 / 空文件 / 全滤掉 —— 一律报错且不落数据")

    print("\n" + "=" * 76)
    print("✓ 全部通过（未读写任何外部文件，临时文件已清理）")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="数据清洗与去重")
    ap.add_argument("--in", dest="inp", default="data/raw/synth_v0.jsonl")
    ap.add_argument("--out", default="data/processed/clean.jsonl")
    ap.add_argument("--clip", action="store_true", help="启用 CLIP 图文一致性过滤（慢）")
    ap.add_argument("--selftest", action="store_true",
                    help="离线自检：去重语义 + 规则过滤（不读文件、不写盘）")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    clean_pipeline(args.inp, args.out, use_clip=args.clip)
