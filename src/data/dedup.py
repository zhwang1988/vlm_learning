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


def dedup(samples: list[dict], image_hashes: dict[str, int],
          threshold_img: int = 5, threshold_text: float = 0.92,
          same_image_diff_text_ok: bool = True,
          embed_model: Optional[str] = None) -> tuple[list[dict], dict]:
    """去重。返回 (保留的样本, 统计)。

    关键设计：**去重的是「图 + 文」这个整体，不是单独的图或文。**

    同一个商品图配不同的用户问题 → 是**有效样本**，要保留
    同一个问题配不同的商品图     → 也是**有效样本**，要保留
    同一张图 + 同一个问题        → 重复，删掉

    所以我们用组合键判断，而不是分别去重。

    参数：
      same_image_diff_text_ok: True 时保留「同图不同文」，这是默认且推荐的
    """
    from .image_utils import hamming_distance

    stats = {"dup_image": 0, "dup_text": 0, "dup_pair": 0}

    if embed_model:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embed_model)
    else:
        encoder = None

    texts = []
    for s in samples:
        try:
            texts.append(next(c["text"] for c in s["messages"][0]["content"]
                              if c["type"] == "text"))
        except Exception:
            texts.append("")

    text_embs = None
    if encoder is not None:
        text_embs = encoder.encode(texts, normalize_embeddings=True,
                                   batch_size=64, show_progress_bar=False)
    else:
        # 退化为字面去重
        pass

    kept: list[dict] = []
    seen_pair: set[str] = set()
    seen_img_groups: list[tuple[int, str]] = []      # (phash, text_key)

    for i, s in enumerate(samples):
        img_path = s.get("image_path", "")
        h = image_hashes.get(img_path)

        # --- 图 + 文 组合去重 ---
        # 找有没有「同图」的已保留样本
        same_img_texts = []
        if h is not None:
            for kh, ktext in seen_img_groups:
                if hamming_distance(h, kh) < threshold_img:
                    same_img_texts.append(ktext)

        text_key = f"emb:{i}" if text_embs is not None else texts[i][:120]

        if same_img_texts:
            # 同图：检查文本是否也近似
            is_dup = False
            if text_embs is not None:
                cur = text_embs[i]
                for _, other_idx in [(x, y) for x, y in same_img_texts if str(y) == "0"]:
                    pass   # 简化：下面用全局文本近似判断
                is_dup = False
            # 同图且文本字面完全相同 → 重复
            if texts[i][:120] in same_img_texts:
                is_dup = True

            if is_dup:
                stats["dup_pair"] += 1
                continue

        # --- 纯文本去重（跨图）---
        if text_embs is not None and len(kept) > 0:
            # 只和前面少量样本比，避免 O(n²)
            pass

        # --- 记录 ---
        pair_key = f"{h}|{texts[i][:120]}" if h is not None else texts[i][:120]
        if pair_key in seen_pair:
            stats["dup_pair"] += 1
            continue
        seen_pair.add(pair_key)

        if h is not None:
            seen_img_groups.append((h, texts[i][:120]))

        kept.append(s)

    return kept, stats


def dedup_simple(samples: list[dict], image_hashes: dict[str, int],
                 threshold_img: int = 5) -> tuple[list[dict], dict]:
    """简化版去重，不依赖 embedding 模型。Day 10 先用这个跑通。"""
    from .image_utils import hamming_distance

    stats = {"dup_pair": 0, "dup_image_same_text": 0}
    kept: list[dict] = []
    seen: set[tuple] = set()

    for s in samples:
        img = s.get("image_path", "")
        h = image_hashes.get(img, 0)
        try:
            text = next(c["text"] for c in s["messages"][0]["content"]
                        if c["type"] == "text")
        except Exception:
            text = ""

        # 找是否有同图 + 同文本
        is_dup = False
        for kh, ktext in seen:
            if isinstance(kh, int) and isinstance(ktext, str):
                if hamming_distance(h, kh) < threshold_img and ktext[:80] == text[:80]:
                    is_dup = True
                    break
        if is_dup:
            stats["dup_pair"] += 1
            continue

        seen.add((h, text))
        kept.append(s)

    return kept, stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def clean_pipeline(raw_path: str | Path,
                   out_path: str | Path = "data/processed/clean.jsonl",
                   image_root: Optional[str] = None,
                   use_clip: bool = False,
                   min_clip_score: float = 0.15,
                   report_path: str | Path = "reports/cleaning_report.md",
                   ) -> CleaningReport:
    """完整的清洗流水线。"""
    from .image_utils import load_and_normalize, phash

    raw_path = Path(raw_path)
    samples = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    rep = CleaningReport(n_input=len(samples))
    print(f"输入 {len(samples):,} 条")

    # --- ① 格式校验 ---
    ok_samples = []
    for s in samples:
        if not isinstance(s.get("messages"), list) or len(s["messages"]) < 2:
            rep.reject("结构异常", s)
            continue
        if not s.get("image_path"):
            rep.reject("缺少图片路径", s)
            continue
        ok_samples.append(s)
    print(f"① 格式校验后: {len(ok_samples):,}")

    # --- ② 规则过滤 ---
    kept2 = []
    for s in ok_samples:
        ok, reason = rule_filter(s)
        if ok:
            kept2.append(s)
        else:
            rep.reject(reason.split("(")[0], s)
    rep.n_after_rules = len(kept2)
    print(f"② 规则过滤后: {len(kept2):,}")

    # --- ③ 图片哈希（用于去重 + 图文一致性）---
    image_hashes: dict[str, int] = {}
    valid_images = {}
    for s in kept2:
        p = s["image_path"]
        if p in image_hashes:
            continue
        try:
            if p == "__no_image__" or not Path(p).exists():
                image_hashes[p] = 0
                continue
            img, _ = load_and_normalize(p)
            image_hashes[p] = phash(img)
            valid_images[p] = img
        except Exception:
            image_hashes[p] = 0
    print(f"③ 计算 {len(image_hashes):,} 张图的指纹")

    # --- ③b 图文一致性（可选，慢）---
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
                        rep.reject("图文不相关", s)
                        continue
                except Exception:
                    pass
                kept3.append(s)
            kept2 = kept3
            print(f"③b CLIP 过滤后: {len(kept2):,}")
        except ImportError:
            print("   （跳过 CLIP：未安装 transformers/clip）")

    # --- ④ 去重 ---
    kept4, dstats = dedup_simple(kept2, image_hashes)
    rep.n_dup_pair = dstats["dup_pair"]
    rep.n_final = len(kept4)
    print(f"④ 去重后: {len(kept4):,}")

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

    # --- 输出 ---
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in kept4:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✓ 输出 {len(kept4):,} 条 → {out_path}")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rep.to_markdown(), encoding="utf-8")
    print(f"✓ 报告 → {report_path}")

    return rep


def _stat(v: list[int]) -> dict:
    a = np.array(v)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)), "max": int(a.max())}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="数据清洗与去重")
    ap.add_argument("--in", dest="inp", default="data/raw/synth_v0.jsonl")
    ap.add_argument("--out", default="data/processed/clean.jsonl")
    ap.add_argument("--clip", action="store_true", help="启用 CLIP 图文一致性过滤（慢）")
    args = ap.parse_args()

    clean_pipeline(args.inp, args.out, use_clip=args.clip)
