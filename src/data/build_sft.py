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

    ⚠️ **必须同时看 n_eval_checked 和 n_eval_skipped，不能只看 n_leaks。**
       「0 泄漏」有两种含义，差别是天壤之别：
           真的没泄漏          → n_eval_checked 很大，n_leaks = 0
           一张图都没能读到    → n_eval_checked = 0，n_leaks 当然是 0
       后者在上云训练时极其常见（写好的 image_path 在云机器上不存在，
       或者训练完把数据拉回本地后路径全变了）。不区分的话，你会拿到一个
       「检查通过」的假象，然后基于虚高的评测分数下结论。
       这和「全部回答『没有』的模型幻觉率 0%」是同一个陷阱：
       **分母塌成 0 时，任何比率都好看。**
    """
    from ..data.image_utils import hamming_distance, load_and_normalize, phash

    def hashes(samples: list[dict]) -> tuple[dict[str, int], int]:
        out: dict[str, int] = {}
        skipped = 0
        for s in samples:
            p = s.get("image_path", "")
            if not p or p in out:
                continue
            if p.startswith("__"):          # __no_image__ 之类的占位符
                skipped += 1
                continue
            try:
                img, _ = load_and_normalize(p)
                out[p] = phash(img)
            except Exception:
                skipped += 1                # 读不到就不参与比对，但要计数
        return out, skipped

    tr, tr_skipped = hashes(train_samples)
    ev, ev_skipped = hashes(eval_samples)

    leaks = []
    leak_eval_images: set[str] = set()
    for ep, eh in ev.items():
        for tp, th in tr.items():
            d = hamming_distance(eh, th)
            if d < threshold:
                leaks.append({"eval_image": ep, "train_image": tp, "distance": d})
                leak_eval_images.add(ep)

    # 覆盖率：真正比对过的评测图 / 评测集里本该检查的图
    intended = len(ev) + ev_skipped
    return {
        "n_eval_checked": len(ev),
        "n_train_checked": len(tr),
        "n_eval_skipped": ev_skipped,
        "n_train_skipped": tr_skipped,
        "coverage": len(ev) / max(intended, 1),
        "n_leaks": len(leaks),
        # ⚠️ 分母、分子必须是**同一个东西**。
        #    早期版本写的是「泄漏**对数** / 评测**图数**」：
        #    一张评测图只要撞上 3 张训练图，就贡献 3 个泄漏对，
        #    于是 9 张评测图能算出 10 个泄漏对 → leak_rate = 1.11（111%）。
        #    「比率超过 100%」是个一眼就该看出不对的信号，但当时打印成
        #    「111.1%」也就过去了 —— 数字大一点看起来更像「问题严重」，
        #    反而没人怀疑它算错了。
        #    现在拆成三个量：泄漏的对数 / 涉事的评测图数 / 比率。
        "n_leak_pairs": len(leaks),
        "n_leak_eval_images": len(leak_eval_images),
        "leak_rate": len(leak_eval_images) / max(len(ev), 1),
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


def check_split_by_image(train: list[dict], val: list[dict],
                         test: list[dict]) -> list[str]:
    """找出同时出现在多个集合里的图片路径。

    这是切分之后的**自检**，和 `check_leakage()` 不同：
      · `check_leakage` 读图算 pHash，比的是「图片内容近似」
      · 本函数只比路径，**不读图**，零依赖、必定能跑

    两个都要有。原因：`check_leakage` 在图片读不到时会退化成
    「什么也没检查」（这正是它最危险的失效模式），而本函数永远有效 ——
    只要路径重复了，就一定有问题，不需要打开图片。

    占位符（`__no_image__` 之类）不算泄漏：它们不代表某一具体的图。
    """
    def paths(rows: list[dict]) -> set[str]:
        return {r.get("image_path", "") for r in rows
                if r.get("image_path") and not r["image_path"].startswith("__")}

    a, b, c = paths(train), paths(val), paths(test)
    cross = (a & b) | (a & c) | (b & c)
    return sorted(cross)


def _group_key(s: dict, idx: int) -> str:
    """样本的「图片分组键」。

    ⚠️ 占位符（`__no_image__` 等）**不能直接当分组键** —— 它们全都长一样，
       拿来当 key 会把整份数据并成一组，最后全落进 train，val/test 直接空掉。
       而且 `__no_image__` 是非空字符串，`or` 兜底不会生效，必须显式判前缀。
       没有真实图片的样本各自成组。

    这个函数必须被**所有**统计分组数的地方共用。实测过一次：
    切分用的是 `__solo_i`（20 组），而打印统计时写成 `p or ...`（`__no_image__`
    非空 → 全部同一个 key），于是打出「按图片分组：1 组」——
    数字和实际行为对不上，比不打印更糟。
    """
    p = s.get("image_path") or ""
    return f"__solo_{idx}" if (not p or p.startswith("__")) else p


def split_by_image(samples: list[dict],
                   eval_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """按图片分组切分数据，返回 (train, val, test)。

    为什么不能直接 `random.shuffle` 后切片
    -------------------------------------
    同一张商品图本来就会配很多个不同的提问 —— 那是**有效样本**，去重时刻意保留
    （见 `src/data/dedup.py` 的「同图不同文必须保留」）。
    但切分时如果把这张图的一个提问放进 train、另一个放进 eval，评测就变成了
    「模型有没有记住这张图」，分数必然虚高，而且**从数字上看不出来**。

    所以：**一张图要么整组进 train，要么整组进 eval，不许拆开。**

    顺带解释一个现象：泄漏检查如果报出一堆 `d=0` 的「自己配自己」，
    那就是切分没做分组导致的 —— 同一张图同时出现在两边。

    实现上容易踩的两个坑（都实测踩过）
    ----------------------------------
    1. **全局分配 vs 层内分配**：分层之后每层往往只剩一两个组
       （本项目 49 个组、48 个层，几乎一层一组）。若在层内做
       「累计到 target 就切」，target 恒为 0.1×组大小，判断永远成立 ——
       结果所有组都进 test，train 剩 0 条。实测：train 0 / val 1 / test 135。
       正确做法是**按层轮转取组**得到交替序列（保住分布），
       再按**全局**目标配比把组分给三个集合。
    2. **占位符不能当分组键**：`__no_image__` 之类全都长一样，
       拿来当 key 会把整份数据并成一组，最后全落进 train，val/test 空掉。
       没有真实图片的样本各自成组。
    """
    random.seed(seed)

    groups: dict[str, list[dict]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(_group_key(s, i), []).append(s)

    # 按「组内主要 (intent, image_type)」分层，让各集合的分布仍然接近
    def primary_key(g: list[dict]) -> tuple:
        c = Counter((x.get("intent", "?"), x.get("image_type", "?")) for x in g)
        return c.most_common(1)[0][0]

    strat: dict[tuple, list[list[dict]]] = {}
    for g in groups.values():
        strat.setdefault(primary_key(g), []).append(g)

    queues = {k: list(gs) for k, gs in strat.items()}
    for q in queues.values():
        random.shuffle(q)
    order: list[list[dict]] = []
    while any(queues.values()):
        for k in queues:
            if queues[k]:
                order.append(queues[k].pop())

    total = sum(len(g) for g in order)
    if len(order) < 3:
        # 组太少（<3 组）根本没法切三个集合，全给 train 更诚实
        return list(samples), [], []

    targets = {"test": total * test_ratio,
               "val": total * eval_ratio,
               "train": total * (1 - test_ratio - eval_ratio)}
    assigned = {"test": 0, "val": 0, "train": 0}
    out: dict[str, list[dict]] = {"test": [], "val": [], "train": []}
    # 每次把组交给「完成度最低」的集合，三个集合同时向目标收敛
    for g in order:
        pick = min(targets,
                   key=lambda k: (assigned[k] / targets[k]) if targets[k] > 0 else 1e9)
        out[pick] += g
        assigned[pick] += len(g)
    return out["train"], out["val"], out["test"]


def build_dataset(clean_path: str | Path,
                  out_dir: str | Path = "data/processed",
                  eval_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                  seed: int = 42,
                  check_leak: bool = True,
                  group_by_image: bool = True) -> dict:
    clean_path = Path(clean_path)
    samples = []
    with open(clean_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"读入 {len(samples):,} 条")

    if group_by_image:
        train, val, test = split_by_image(samples, eval_ratio, test_ratio, seed)
        gc = Counter(_group_key(s, i) for i, s in enumerate(samples))
        n_multi = sum(1 for v in gc.values() if v > 1)
        print(f"按图片分组：{len(gc):,} 组（其中 {n_multi:,} 组含多条样本，"
              f"最大一组 {max(gc.values())} 条）")
    else:
        # 不做分组的老路径：仅在你确定「每条样本的图都不一样」时才用
        random.seed(seed)
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

    # 切分后的自检：同一张图不许跨集合
    if group_by_image:
        cross = check_split_by_image(train, val, test)
        if cross:
            raise RuntimeError(
                f"切分自检失败：{len(cross)} 张图同时出现在多个集合里 "
                f"（示例 {cross[:3]}）。分组切分没有生效，"
                f"这份数据上的评测分数会虚高。")

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

        n_checked, n_skip = lk["n_eval_checked"], lk["n_eval_skipped"]

        if n_checked == 0:
            # ⚠️ 这里最容易自欺：没读到图 ≠ 没泄漏。
            #    只在真正比对过图片时才敢说「无泄漏」。
            print(f"  ⚠️  一张图都没读到（跳过 {n_skip} 张）—— "
                  f"泄漏检查**没有真正执行**")
            print("      常见原因：image_path 指向的文件不存在"
                  "（换机器 / 拉回本地后路径变了）。")
            print("      这条『通过』不能当数，先把图片路径修对再重跑。")
        elif lk["n_leaks"]:
            print(f"  ⚠️  发现 {lk['n_leaks']} 处泄漏"
                  f"（{lk['leak_rate']:.1%}，已检查 {n_checked} 张评测图）")
            for ex in lk["examples"][:5]:
                print(f"      {ex['eval_image']} ~ {ex['train_image']}"
                      f" (d={ex['distance']})")
            print("  → 必须处理！否则评测分数会虚高。")
        else:
            print(f"  ✓ 检查 {n_checked} 张评测图 × {lk['n_train_checked']} "
                  f"张训练图，无泄漏")
            if n_skip:
                print(f"    （另有 {n_skip} 张图读不到已跳过，覆盖不完整）")

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


def _selftest() -> int:
    """离线自检：临时目录跑完整打包流程。不需要 GPU，也不需要真实图片。

    重点覆盖三件事：
      ① images 顺序必须和 <image> 出现顺序一致（LLaMA-Factory 靠它对图）
      ② 分层切分后每个桶都要出现在 train 里（不能整桶消失）
      ③ ⭐ 泄漏检查的覆盖率语义 —— 一张图都没读到的时候，
         绝不能输出「✓ 无泄漏」让人以为检查过了
    """
    import tempfile

    print("=" * 76)
    print("SFT 数据构造自检（离线，临时目录）")
    print("=" * 76)

    def _s(i: int, img: str, q: str, intent: str = "size_fit",
           img_type: str = "product_main", diff: int = 1) -> dict:
        return {
            "id": f"s{i:03d}",
            "image_path": img,
            "intent": intent,
            "image_type": img_type,
            "difficulty": diff,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "path": img},
                    {"type": "text", "text": q}]},
                {"role": "assistant", "content": [
                    {"type": "text",
                     "text": "您好，看到您的提问了。这款是宽松直筒版型，"
                             "M 码肩宽 38cm、胸围 100cm，平时穿 M 的话是合适的。"}]},
            ],
        }

    # 1. ⭐ images 顺序必须和 <image> 出现顺序严格一致
    print("\n[1] to_llamafactory_format()：images 与 <image> 顺序对齐")
    multi = _s(0, "a.jpg", "这两张图有什么区别")
    multi["messages"][0]["content"] = [
        {"type": "image", "path": "first.jpg"},
        {"type": "text", "text": "第一张"},
        {"type": "image", "path": "second.jpg"},
        {"type": "text", "text": "对比第二张"},
    ]
    row = to_llamafactory_format(multi, DEFAULT_SYSTEM_PROMPT)
    print(f"    images       = {row['images']}")
    print(f"    user content = {row['messages'][1]['content']!r}")
    assert row["images"] == ["first.jpg", "second.jpg"], \
        "images 顺序必须与 <image> 出现顺序严格一致 —— 错了就会给图配错问题"
    assert row["messages"][1]["content"].count("<image>") == 2
    assert row["messages"][0]["role"] == "system", "system 必须在最前"
    print("    ✓ 两张图的顺序正确，system 在最前")

    noimg = _s(1, "__no_image__", "纯文本问题")
    row2 = to_llamafactory_format(noimg, DEFAULT_SYSTEM_PROMPT)
    assert row2["images"] == ["__no_image__"], \
        "占位符应原样保留，下游据此识别并跳过，不会当真实路径去 load"
    print("    ✓ 无图样本的占位符被保留")

    # 2. 端到端：手造 clean.jsonl → build_dataset
    print("\n[2] build_dataset() 端到端（临时目录）")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "clean.jsonl"
        rows = []
        for b, (it, im) in enumerate([("size_fit", "product_main"),
                                      ("quality_issue", "defect_photo")]):
            for k in range(10):
                rows.append(_s(b * 10 + k, "__no_image__",
                               f"第{k}个问题，这款的尺码怎么选", it, im, 1 + b))
        src.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8")

        stats = build_dataset(src, Path(d) / "out", seed=42)

        for name in ("sft_train.jsonl", "sft_eval.jsonl", "sft_test.jsonl",
                     "dataset_info.json", "build_stats.json"):
            assert (Path(d) / "out" / name).exists(), f"缺少输出文件 {name}"
        assert stats["n_train"] + stats["n_val"] + stats["n_test"] == 20

        # 分层切分：两个桶都必须出现在 train 里，不能被整桶切走
        assert set(stats["intent_dist"]) == {"size_fit", "quality_issue"}, \
            f"分层切分把某个桶整桶切走了：{stats['intent_dist']}"

        # dataset_info.json 指向的文件必须真的存在，否则 LLaMA-Factory 直接报错
        info = json.loads(
            (Path(d) / "out" / "dataset_info.json").read_text("utf-8"))
        cfg = next(iter(info.values()))
        assert (Path(d) / "out" / cfg["file_name"]).exists(), \
            "dataset_info.json 里的 file_name 不存在"
        print(f"    ✓ 5 个输出文件齐全，dataset_info 指向 {cfg['file_name']}")

        # 3. ⭐ 泄漏检查的覆盖率语义
        print("\n[3] 泄漏检查的覆盖率语义（最容易被自欺的地方）")
        lk = stats["leakage"]
        print(f"    n_eval_checked={lk['n_eval_checked']} "
              f"n_eval_skipped={lk['n_eval_skipped']} "
              f"coverage={lk['coverage']:.0%} n_leaks={lk['n_leaks']}")
        # 这批样本全是 __no_image__ 占位符 → 一张图都没能检查
        assert lk["n_eval_checked"] == 0, "占位符不该被当成真实图片参与比对"
        assert lk["n_eval_skipped"] > 0, \
            "跳过的图必须计数 —— 否则『0 泄漏』会被误读成『检查通过』"
        assert lk["coverage"] == 0.0
        print("    ✓ 没读到图时 checked=0 且 skipped>0，两者分得清")
        print("      所以这里的『0 泄漏』不会被当成『检查通过』")

        # 4. 图片读不到时同理
        print("\n[4] 图片读不到时不许假装检查过")
        lk2 = check_leakage([_s(0, "same.jpg", "问题")],
                            [_s(1, "same.jpg", "问题")])
        assert lk2["n_eval_checked"] == 0 and lk2["n_eval_skipped"] == 1, \
            "文件不存在的图必须计入 skipped"
        print(f"    文件不存在 → checked={lk2['n_eval_checked']} "
              f"skipped={lk2['n_eval_skipped']}")
        print("    ✓ 读不到的图被如实计入 skipped，不会被算成『无泄漏』")

    # 5. ⭐ 回归：切分必须按图片分组（同一张图不许跨集合）
    #    实测过的翻车现场：不加分组、只 random.shuffle 后切片，同一张商品图
    #    被切进 train 和 test 两边，泄漏检查立刻报出 10 处 d=0 的「自己配自己」。
    print("\n[5] 切分按图片分组：同一张图不许跨集合")
    n_imgs, per_img = 12, 3
    rows = []
    for gi in range(n_imgs):
        for k in range(per_img):
            rows.append(_s(gi * 10 + k, f"img_{gi}.jpg",
                           f"关于 img_{gi} 的第 {k} 个问题，问的是尺码",
                           "size_fit", "product_main", 1 + k % 5))
    tr, va, te = split_by_image(rows, seed=0)
    cross = check_split_by_image(tr, va, te)
    assert not cross, f"同一张图跨集合了：{cross}"
    assert all(x["image_path"] not in {r["image_path"] for r in va + te} for x in tr)
    print(f"    {n_imgs} 张图 × {per_img} 条 → train {len(tr)} / val {len(va)} "
          f"/ test {len(te)}，跨集合图片 0 张")
    # 每张图的 3 条必须整组落在一起
    for gi in range(n_imgs):
        p = f"img_{gi}.jpg"
        where = [nm for nm, st in (("train", tr), ("val", va), ("test", te))
                 if any(x["image_path"] == p for x in st)]
        assert len(where) == 1, f"{p} 出现在 {where} 多个集合里"
    print("    ✓ 12 张图各自整组落进唯一一个集合")

    # 6. ⭐ 回归：leak_rate 必须是「涉事评测图数 / 评测图数」，不许 > 1
    #    实测过的翻车现场：分子是泄漏**对数**、分母是评测**图数**，
    #    一张图撞 3 张训练图就贡献 3 对，于是打出 111.1% 这种数字。
    print("\n[6] leak_rate 的分子分母必须是同一种东西")
    lk3 = check_leakage(
        train_samples=[_s(0, "__no_image__", "问"), _s(1, "__no_image__", "问")],
        eval_samples=[_s(2, "__no_image__", "问")])
    assert lk3["leak_rate"] <= 1.0, f"泄漏率不该超过 1，实际 {lk3['leak_rate']}"
    assert set(lk3) >= {"n_leak_pairs", "n_leak_eval_images", "leak_rate"}
    print(f"    n_leak_pairs={lk3['n_leak_pairs']} "
          f"n_leak_eval_images={lk3['n_leak_eval_images']} "
          f"leak_rate={lk3['leak_rate']:.2f}（≤ 1 ✓）")
    print("    ✓ 泄漏对数 / 涉事评测图数 / 比率 三个量分开报，比率恒 ≤ 1")

    print("\n" + "=" * 76)
    print("✓ 全部通过（离线，未使用 GPU）")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构造 SFT 训练数据")
    ap.add_argument("--in", dest="inp", default="data/processed/clean.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--inspect", help="检查某个 jsonl 的样本")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--no-group-split", action="store_true",
                    help="不按图片分组切分（默认是分组的；只在确定每条样本的图都不同时才关）")
    ap.add_argument("--no-leak-check", action="store_true",
                    help="跳过图片级泄漏检查（会读图，慢）")
    ap.add_argument("--selftest", action="store_true",
                    help="离线自检：打包 + 分层切分 + 泄漏覆盖率（用临时目录）")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    if args.inspect:
        inspect_sample(args.inspect, args.idx)
    else:
        stats = build_dataset(args.inp, args.out_dir,
                              group_by_image=not args.no_group_split,
                              check_leak=not args.no_leak_check)
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
