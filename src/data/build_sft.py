"""
把清洗后的数据转成训练格式（Qwen2.5-VL SFT）。

对应讲义 docs/05-data-engineering.md 第 8 节，对应计划 Day 11。

**这是最容易出错、也最值得反复验证的一个文件。**

三件事必须做对：
  ① 图像位置：<|vision_start|><|image_pad|>...<|vision_end|> 的 pad 数量
               必须等于该图实际产生的 visual token 数
  ② Label mask：只有 assistant 的回复部分参与 loss，其余全部 -100
  ③ 停止符：assistant 回复末尾的 <|im_end|> 必须参与 loss，
            否则模型不知道何时停下来

输出两个文件：
  sft_train.jsonl  LLaMA-Factory 格式（sharegpt 风格，含 images 字段）
  sft_eval.jsonl   （会做图片级泄漏检查）
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 1. 转为 LLaMA-Factory / sharegpt 格式
# ---------------------------------------------------------------------------


def to_llamafactory_format(sample: dict, system_prompt: str,
                           image_dir: Optional[str] = None) -> dict:
    """转成 LLaMA-Factory 的 multimodal sharegpt 格式。

    关键：`images` 列表的顺序必须和 content 里 <image> 出现的顺序一致。
    LLaMA-Factory 会用这个顺序去加载图片并生成视觉 token。
    """
    msgs = []
    images = []

    msgs.append({"role": "system", "content": system_prompt})

    for m in sample["messages"]:
        role = m["role"]
        parts = m["content"]

        text_parts = []
        for p in parts:
            if p["type"] == "image":
                path = p.get("path", "")
                if image_dir and path and not path.startswith("__"):
                    full = str(Path(image_dir) / Path(path).name)
                else:
                    full = path
                images.append(full)
                text_parts.append("<image>")
            elif p["type"] == "text":
                text_parts.append(p["text"])

        msgs.append({"role": role, "content": "\n".join(text_parts)})

    return {
        "messages": msgs,
        "images": images,
        "id": sample.get("id", ""),
        "intent": sample.get("intent", ""),
        "image_type": sample.get("image_type", ""),
        "difficulty": sample.get("difficulty", 0),
    }


# ---------------------------------------------------------------------------
# 2. 直接构造带 label mask 的训练样本（原生 transformers 路线）
# ---------------------------------------------------------------------------


def build_labeled_sample(sample: dict, processor, system_prompt: str,
                         max_length: int = 4096) -> Optional[dict]:
    """用官方 processor 生成 input_ids + labels，label mask 全部算好。

    这是原生 transformers 路线（src/train/sft_peft.py）用的格式。

    **核心逻辑**：
      1. 逐轮构造文本，用 processor.apply_chat_template 得到完整 input_ids
      2. 找到每个 assistant 片段的起止位置
      3. 只在这些位置保留 token id，其余全部置 -100
      4. 特别地：assistant 块末尾的 <|im_end|> 要保留（学停止）

    为什么要自己算 position 而不能直接用 template 的 return_assistant_tokens_mask：
      - 多模态下 template 的 mask 有时不包含 <|im_end|>
      - 显式算一遍你能看懂每个位置在干什么，出问题时能 debug
    """
    from PIL import Image

    images = []
    messages = [{"role": "system", "content": system_prompt}]

    for m in sample["messages"]:
        content = []
        for p in m["content"]:
            if p["type"] == "image":
                path = p.get("path", "")
                if path and not path.startswith("__") and Path(path).exists():
                    images.append(Image.open(path).convert("RGB"))
                    content.append({"type": "image", "image": path})
                else:
                    # 无图样本：退化成纯文本（用于混入通用文本数据）
                    pass
            else:
                content.append({"type": "text", "text": p["text"]})
        if content:
            messages.append({"role": m["role"], "content": content})

    # 完整文本（不带 generation prompt，因为我们要自己算 label）
    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    try:
        inputs = processor(text=[full_text], images=images if images else None,
                           return_tensors="pt")
    except Exception:
        return None

    input_ids = inputs["input_ids"][0]

    # --- 逐段定位 assistant 内容 ---
    labels = input_ids.clone()
    labels[:] = -100

    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids_list = input_ids.tolist()

    # 找到所有 <|im_start|>assistant 的位置
    token_assistant = processor.tokenizer.encode("assistant", add_special_tokens=False)

    spans = []
    i = 0
    while i < len(ids_list):
        if ids_list[i] == im_start_id:
            # 看后面紧跟的是不是 "assistant"
            j = i + 1
            if ids_list[j:j + len(token_assistant)] == token_assistant:
                start = j + len(token_assistant)
                # 跳过换行
                while start < len(ids_list) and ids_list[start] in (
                    processor.tokenizer.encode("\n", add_special_tokens=False) or [198]
                ):
                    start += 1
                # 找这个 assistant 块的 <|im_end|>
                end = start
                while end < len(ids_list) and ids_list[end] != im_end_id:
                    end += 1
                # 包含 <|im_end|>
                spans.append((start, min(end + 1, len(ids_list))))
                i = end
        i += 1

    for s, e in spans:
        labels[s:e] = input_ids[s:e]

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    n_supervised = int((labels != -100).sum())
    if n_supervised < 5:
        return None      # mask 失败，丢弃

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "n_supervised": n_supervised,
        "n_total": len(input_ids),
        "n_images": len(images),
    }


# ---------------------------------------------------------------------------
# 3. 图片级泄漏检查（Day 20 评测集构造必做）
# ---------------------------------------------------------------------------


def check_leakage(train_samples: list[dict], eval_samples: list[dict],
                  threshold: int = 8) -> dict:
    """检查评测集的图片有没有出现在训练集里。

    这是 VLM 评测最常见的作弊来源：图片重叠会让分数虚高，
    因为你测的是「模型记住了这张图」而不是「模型真的会看图」。

    threshold=8 比去重用的 5 更严格——宁可多报几个可疑的。
    """
    from ..data.image_utils import hamming_distance, load_and_normalize, phash

    def hashes(samples):
        out = {}
        for s in samples:
            p = s.get("image_path", "")
            if not p or p in out or p.startswith("__"):
                continue
            try:
                img, _ = load_and_normalize(p)
                out[p] = phash(img)
            except Exception:
                out[p] = 0
        return out

    tr = hashes(train_samples)
    ev = hashes(eval_samples)

    leaks = []
    for ep, eh in ev.items():
        if eh == 0:
            continue
        for tp, th in tr.items():
            if th == 0:
                continue
            d = hamming_distance(eh, th)
            if d < threshold:
                leaks.append({"eval_image": ep, "train_image": tp, "distance": d})

    return {
        "n_eval_images": len(ev),
        "n_train_images": len(tr),
        "n_leaks": len(leaks),
        "leak_rate": len(leaks) / max(len(ev), 1),
        "examples": leaks[:10],
    }


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "你是一位专业的电商客服助手，服务于一家经营服饰鞋包的店铺。"
    "请基于用户提供的图片和文字，给出准确、有帮助、语气自然的回复。"
    "只依据图片中可见的内容和已知的商品信息回答；信息不足时请主动询问，"
    "不要编造商品参数。遇到超出你能力范围的请求（如索赔、投诉、法律问题），"
    "请安抚用户并说明将转接人工客服。"
)


def build_dataset(clean_path: str | Path,
                  out_dir: str | Path = "data/processed",
                  eval_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                  seed: int = 42,
                  check_leak: bool = True) -> dict:
    random.seed(seed)

    clean_path = Path(clean_path)
    samples = []
    with open(clean_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"读入 {len(samples):,} 条")

    # 按 (intent, image_type) 分层切分，保证三个集合分布一致
    buckets: dict[tuple, list[dict]] = {}
    for s in samples:
        key = (s.get("intent", "?"), s.get("image_type", "?"))
        buckets.setdefault(key, []).append(s)

    train, val, test = [], [], []
    for key, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * test_ratio)) if n >= 5 else 0
        n_val = max(1, int(n * eval_ratio)) if n >= 5 else 0
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]

    print(f"分层切分: train {len(train):,} / val {len(val):,} / test {len(test):,}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 输出 LLaMA-Factory 格式 ---
    for name, data in [("sft_train", train), ("sft_eval", val), ("sft_test", test)]:
        rows = [to_llamafactory_format(s, system_prompt) for s in data]
        p = out_dir / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {p}  ({len(rows):,} 条)")

    # --- dataset_info.json（LLaMA-Factory 需要）---
    info = {
        "cx_vlm_sft_v0": {
            "file_name": "sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 统计 ---
    stats = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "intent_dist": dict(Counter(s.get("intent", "?") for s in train)),
        "image_type_dist": dict(Counter(s.get("image_type", "?") for s in train)),
        "difficulty_dist": dict(Counter(str(s.get("difficulty", 0)) for s in train)),
    }

    # --- 泄漏检查 ---
    if check_leak:
        print("\n图片级泄漏检查...")
        lk = check_leakage(train, val + test)
        stats["leakage"] = {k: v for k, v in lk.items() if k != "examples"}
        if lk["n_leaks"]:
            print(f"  ⚠️  发现 {lk['n_leaks']} 处泄漏（{lk['leak_rate']:.1%}）")
            for ex in lk["examples"][:5]:
                print(f"      {ex['eval_image']} ~ {ex['train_image']} (d={ex['distance']})")
            print("  → 必须处理！否则评测分数会虚高。")
        else:
            print("  ✓ 无泄漏")

    (out_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return stats


# ---------------------------------------------------------------------------
# 5. 抽检工具（Day 11 的验收标准）
# ---------------------------------------------------------------------------


def inspect_sample(jsonl_path: str | Path, idx: int = 0,
                   model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """decode 一条样本，肉眼检查 label mask 是否正确。

    Day 11 的验收动作。必须能看到：
      - prompt 部分的 label 是 -100
      - assistant 部分有真实 token id
      - assistant 末尾的 <|im_end|> 也被监督到
    """
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)

    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                sample = json.loads(line)
                break
        else:
            print(f"第 {idx} 条不存在")
            return

    print("=" * 78)
    print(f"样本 #{idx}  意图={sample.get('intent')}  图像类型={sample.get('image_type')}")
    print("=" * 78)
    print(f"图片数: {len(sample.get('images', []))}")
    for im in sample.get("images", []):
        print(f"  {im}")
    print()

    for m in sample["messages"]:
        role = m["role"]
        txt = m["content"]
        if role == "system":
            print(f"[SYSTEM] {txt[:200]}...")
        else:
            print(f"[{role.upper()}] {txt}")
        print()

    # 构造 labeled 版本
    from .build_sft import build_labeled_sample

    fake = {
        "messages": [
            {"role": "user",
             "content": [{"type": "image", "path": p} for p in sample.get("images", [])]
             + [{"type": "text",
                 "text": next(m["content"] for m in sample["messages"]
                              if m["role"] == "user")}]},
            {"role": "assistant",
             "content": [{"type": "text",
                          "text": next(m["content"] for m in sample["messages"]
                                       if m["role"] == "assistant")}]},
        ]
    }
    sys_p = next((m["content"] for m in sample["messages"] if m["role"] == "system"),
                 DEFAULT_SYSTEM_PROMPT)
    labeled = build_labeled_sample(fake, processor, sys_p)
    if labeled is None:
        print("✗ 该样本无法构造 label（可能是图片路径不存在）")
        return

    ids = labeled["input_ids"]
    labs = labeled["labels"]
    n_sup = labeled["n_supervised"]
    print("=" * 78)
    print(f"总长度 {labeled['n_total']}，被监督 {n_sup} 个 token "
          f"({n_sup / labeled['n_total']:.1%})")
    print("=" * 78)

    tok = processor.tokenizer
    print("\n被监督的片段（即模型要学的部分）:")
    print("-" * 78)
    print(tok.decode([int(x) for x in ids[labs != -100]]))
    print("-" * 78)
    print("\n前 30 个被 mask 的位置（应为 -100）:")
    masked = [i for i in range(min(30, len(labs))) if labs[i] == -100]
    print(f"  {masked}")
    print("\n✓ 检查要点：")
    print("  1. 上面『被监督的片段』应该只有客服回复的内容")
    print("  2. 结尾应该能看到 <|im_end|>（这是模型学会停下的关键）")
    print("  3. user 的问题和 system prompt 不应出现在被监督片段里")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构造 SFT 训练数据")
    ap.add_argument("--in", dest="inp", default="data/processed/clean.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--inspect", help="检查某个 jsonl 的样本")
    ap.add_argument("--idx", type=int, default=0)
    args = ap.parse_args()

    if args.inspect:
        inspect_sample(args.inspect, args.idx)
    else:
        stats = build_dataset(args.inp, args.out_dir)
        print("\n" + "=" * 60)
        print("数据集统计")
        print("=" * 60)
        print(json.dumps({k: v for k, v in stats.items()
                          if k not in ("intent_dist", "image_type_dist",
                                       "difficulty_dist")},
                         ensure_ascii=False, indent=2))
        print("\n意图分布:", stats["intent_dist"])
        print("图像类型分布:", stats["image_type_dist"])
        print("\n下一步（Day 11 验收）:")
        print(f"  python -m src.data.build_sft --inspect {args.out_dir}/sft_train.jsonl")
        print("  确认 label mask 正确后再进入 Day 13 的训练")
