"""
DPO 训练入口 —— Day 27–29 的主程序。

    # 1. 先看看偏好数据长什么样（不需要 torch，本地就能跑）
    python -m src.train.dpo --inspect --in data/processed/dpo_from_badcases.jsonl

    # 2. 数据格式体检 + 长度分布 + 显存预估（不加载模型）
    python -m src.train.dpo --dry-run --config configs/dpo_3b.yaml

    # 3. 正式训练
    python -m src.train.dpo --config configs/dpo_3b.yaml

──────────────────────────────────────────────────────────────────────────────
这个文件为什么这样设计（三个关键决定）
──────────────────────────────────────────────────────────────────────────────

① **policy 和 reference 用同一个模型的两个 adapter。**

   朴素做法是加载两份 3B 权重（policy + ref），24 GB 卡直接爆。
   这里用 PEFT 的多 adapter 能力：base 加载一次，
   挂两个 adapter，都从 SFT 的权重出发：

       base(Qwen2.5-VL-3B, 冻结)
         ├── adapter "policy"     ← is_trainable=True，在训练
         └── adapter "reference"  ← is_trainable=False，冻结

   然后 `model.set_adapter("reference")` 切过去算 ref logps。
   省下的是一整份模型的显存 —— 这是 DPO 能不能在 24 GB 卡上跑的分水岭。

② **reference 必须来自 SFT 权重，不能是原始基座。**

   DPO 的数学是 log(π_θ/π_ref)。ref 的语义是
   「生成这批偏好数据时的那个模型」。如果 ref 用了没 SFT 过的底座，
   分母就错了，训练会变成在纠正一个模型从没犯过的错。
   所以 sft_adapter 找不到时这里直接报错退出，不给你一个静默错误的实验。

③ **chosen / rejected 合成一个 batch，一次前向算完。**

   每个偏好对里 chosen 和 rejected 共用同一张图和同一个 prompt。
   把它们拼成 (2B, L) 的 batch（前 B 个是 chosen），
   一次前向就能同时拿到两边的 logps —— 前向次数从 4 次降到 2 次。

──────────────────────────────────────────────────────────────────────────────
盯这三个指标，不是 loss（docs/07-alignment.md）
──────────────────────────────────────────────────────────────────────────────
    accuracy        : 模型认为 chosen 更好的比例，应该稳步上升 → 1
    reward_margin   : chosen 和 rejected 的 reward 差，应该扩大
    loss            : 应该下降，但**它下降不代表模型变好**
  危险信号：accuracy 冲到 1.0 而评测同时变差 = 过拟合偏好数据。
            解法：减 epoch、加 beta、或检查偏好数据本身有没有错。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

POLICY = "policy"
REFERENCE = "reference"


# =============================================================================
# Part A · 偏好数据（纯标准库，不 import torch —— 本地也能跑）
# =============================================================================


@dataclass
class PreferencePair:
    """一条偏好对。

    chosen / rejected 是同一个 prompt（同一张图、同一个问题）下的两个回答。
    """

    id: str
    prompt: str
    chosen: str
    rejected: str
    images: list[str] = field(default_factory=list)
    source: str = "unknown"
    error_type: str = "unknown"


def load_pairs(paths: list[str | Path],
               require_images: bool = False,
               image_root: Optional[str] = None,
               verbose: bool = True) -> list[PreferencePair]:
    """读入并合并多个偏好数据文件。

    为什么要支持多个文件：
      type1（bad case，模型真犯过的错）和 type2（同答不同图，治幻觉）
      是两个独立流程产出的，训练时**应该混在一起**，而不是二选一。
      混合比例本身就是要调的超参。
    """
    pairs: list[PreferencePair] = []
    per_file: dict[str, int] = {}

    for p in paths:
        p = Path(p)
        if not p.exists():
            if verbose:
                print(f"  ⚠ 跳过不存在的文件：{p}")
            continue
        n0 = len(pairs)
        img_root = Path(image_root) if image_root else None

        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chosen = (r.get("chosen") or "").strip()
                rejected = (r.get("rejected") or "").strip()
                if not chosen or not rejected:
                    continue
                if chosen == rejected:
                    # 完全相同的两个回答构不成偏好信号，留着只会注入噪声
                    continue

                imgs = list(r.get("images") or ([r["image"]] if r.get("image") else []))
                if img_root:
                    imgs = [str(img_root / i) if not Path(i).is_absolute() else i
                            for i in imgs]
                if require_images and not imgs:
                    continue

                pairs.append(PreferencePair(
                    id=str(r.get("id", f"{p.stem}_{len(pairs)}")),
                    prompt=(r.get("prompt") or r.get("query") or "").strip(),
                    chosen=chosen,
                    rejected=rejected,
                    images=imgs,
                    source=r.get("source", p.stem),
                    error_type=r.get("error_type", "unknown"),
                ))

        per_file[str(p)] = len(pairs) - n0

    if verbose:
        print(f"  读入偏好对 {len(pairs)} 条：")
        for k, v in per_file.items():
            print(f"    {v:6d}  ← {k}")
        by_src: dict[str, int] = {}
        for pr in pairs:
            by_src[pr.source] = by_src.get(pr.source, 0) + 1
        print("  按来源：" + " · ".join(f"{k}({v})" for k, v in sorted(by_src.items())))
    return pairs


def validate_pairs(pairs: list[PreferencePair]) -> dict:
    """数据体检。这些是偏好数据最常见的四种脏法。"""
    issues: dict[str, int] = {
        "缺图片路径": 0,
        "图片文件不存在": 0,
        "prompt 为空": 0,
        "chosen 比 rejected 短很多": 0,   # ← 长度偏见会毁掉 DPO
        "rejected 明显像套话": 0,
    }
    boilerplate = ("很抱歉", "我无法", "作为AI", "作为一个AI", "不知道哦")

    for p in pairs:
        if not p.images:
            issues["缺图片路径"] += 1
        else:
            for i in p.images:
                if not Path(i).exists():
                    issues["图片文件不存在"] += 1
                    break
        if not p.prompt:
            issues["prompt 为空"] += 1
        # chosen 短于 rejected 的 1/3 → 模型可能学成「越短越好」
        if len(p.chosen) * 3 < len(p.rejected):
            issues["chosen 比 rejected 短很多"] += 1
        if any(b in p.rejected[:20] for b in boilerplate) and len(p.rejected) < 40:
            issues["rejected 明显像套话"] += 1

    return issues


def pair_stats(pairs: list[PreferencePair], words_per_sec: float = 6.0):
    """打印长度分布 + 预估训练时长。"""
    if not pairs:
        print("  ✗ 没有数据")
        return

    def pct(xs, q):
        if not xs:
            return 0
        xs = sorted(xs)
        return xs[min(int(len(xs) * q), len(xs) - 1)]

    cl = [len(p.chosen) for p in pairs]
    rl = [len(p.rejected) for p in pairs]
    pl = [len(p.prompt) for p in pairs]

    print()
    print("  字符长度分布")
    print("  " + "-" * 62)
    print(f"  {'':10s} {'中位':>8s} {'p90':>8s} {'p99':>8s} {'最大':>8s}")
    for name, xs in (("prompt", pl), ("chosen", cl), ("rejected", rl)):
        print(f"  {name:10s} {pct(xs, .5):>8d} {pct(xs, .9):>8d} "
              f"{pct(xs, .99):>8d} {max(xs):>8d}")
    print("  " + "-" * 62)
    print("  ⚠ prompt 里有 <image>，真实 token 数远大于字符数，"
          "务必用 --dry-run 看 token 长度")

    # 长度偏见检查
    longer_chosen = sum(1 for p in pairs if len(p.chosen) > len(p.rejected))
    ratio = longer_chosen / len(pairs)
    print(f"  chosen 更长的比例：{ratio:.1%}")
    if ratio > 0.75 or ratio < 0.25:
        print("  ⚠ 严重不均衡。模型会学到「长/短 = 好」，而不是内容更好。")
        print("    解法：① 人工检查偏好数据；② 训练时开 average_log_prob；")
        print("          ③ 直接丢掉长度差过大的对（下面的裁剪逻辑会做）")


# =============================================================================
# Part B · 配置
# =============================================================================


@dataclass
class DPOConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    # 从这里出发（必须存在！DPO 不能从没 SFT 过的模型开始）
    sft_adapter: str = "outputs/qwen25vl3b-cx-lora-v0"
    output_dir: str = "outputs/qwen25vl3b-cx-dpo-v0"

    # 可传多个：type1（bad case）+ type2（对比）= 效果最好
    preference_files: list[str] = field(default_factory=lambda: [
        "data/processed/dpo_from_badcases.jsonl",
        "data/processed/dpo_contrastive.jsonl",
    ])
    image_root: Optional[str] = None

    # 训练
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6          # ← 比 SFT 低一个量级，别调高
    num_train_epochs: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    optim: str = "adamw_torch"

    # DPO 超参
    beta: float = 0.1
    label_smoothing: float = 0.0
    average_log_prob: bool = False      # True → SimPO 风格，抗长度偏见
    # 长度差超过这个倍数的对直接丢掉（治长度偏见）
    max_len_ratio: float = 4.0

    # 量化
    use_qlora_for_policy: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 日志
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    run_name: str = "cx-dpo-v0"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOConfig":
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        # 兼容老配置里的单数写法 preference_file
        if "preference_file" in raw and "preference_files" not in raw:
            raw["preference_files"] = [raw.pop("preference_file")]
        return cls(**{k: v for k, v in raw.items() if k in known})


# =============================================================================
# Part C · 数据集与 Collator（需要 torch / processor）
# =============================================================================


class PreferenceDataset:
    """把 PreferencePair 变成 tokenized 的 (prompt, response) 对。

    关键点：chosen 和 rejected **共用同一个 prompt 前缀**，
    所以 prompt 的 token 数只算一次，两边复用。标签只在回答部分监督。
    """

    def __init__(self, pairs, processor, system_prompt: str,
                 max_length: int = 2048, max_len_ratio: float = 4.0,
                 verbose: bool = True):
        import torch  # noqa: F401  （仅确认可用）
        from PIL import Image

        self.processor = processor
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.items: list[dict] = []
        self.dropped = {"图片缺失": 0, "过长": 0, "长度差过大": 0, "mask失败": 0}

        tok = processor.tokenizer
        im_end = tok.convert_tokens_to_ids("<|im_end|>")

        for pr in pairs:
            imgs = []
            ok = True
            for p in pr.images[:1]:        # 客服场景绝大多数是单图；多图显存翻倍
                if not Path(p).exists():
                    ok = False
                    break
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    ok = False
                    break
            if not ok:
                self.dropped["图片缺失"] += 1
                continue

            # 长度偏见防护：长度差太离谱的对直接扔
            a, b = len(pr.chosen), len(pr.rejected)
            if min(a, b) > 0 and max(a, b) / min(a, b) > max_len_ratio:
                self.dropped["长度差过大"] += 1
                continue

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": (
                    [{"type": "image"}] if imgs else []
                ) + [{"type": "text", "text": pr.prompt}]},
            ]
            try:
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                self.dropped["mask失败"] += 1
                continue

            # prompt 的 token 长度（含图片展开后的 vision token）
            try:
                pe = processor(text=[prompt_text], images=imgs or None,
                               return_tensors="pt")
            except Exception:
                self.dropped["mask失败"] += 1
                continue
            plen = pe["input_ids"].shape[1]
            if plen >= max_length:
                self.dropped["过长"] += 1
                continue

            built = {}
            for role, text in (("chosen", pr.chosen), ("rejected", pr.rejected)):
                full = prompt_text + text + tok.decode([im_end])
                try:
                    enc = processor(text=[full], images=imgs or None,
                                    return_tensors="pt")
                except Exception:
                    built = {}
                    break
                ids = enc["input_ids"][0]
                if ids.shape[0] > max_length:
                    # 只截回答，不截 prompt（prompt 截了图就没了）
                    budget = max_length - plen
                    if budget < 8:
                        self.dropped["过长"] += 1
                        built = {}
                        break
                    cut = enc["input_ids"][:, :max_length]
                    enc["input_ids"] = cut
                    ids = cut[0]

                labels = ids.clone()
                labels[:plen] = -100
                if (labels != -100).sum() < 3:
                    self.dropped["mask失败"] += 1
                    built = {}
                    break

                d = {"input_ids": ids, "labels": labels,
                     "attention_mask": torch.ones_like(ids)}
                for k in ("pixel_values", "image_grid_thw"):
                    if k in enc:
                        d[k] = enc[k]
                built[role] = d

            if not built or "chosen" not in built or "rejected" not in built:
                continue
            built["pair_id"] = pr.id
            built["source"] = pr.source
            self.items.append(built)

        if verbose:
            print(f"  可用偏好对 {len(self.items)} 条")
            if any(self.dropped.values()):
                print("  丢弃：" + " · ".join(f"{k} {v}" for k, v in self.dropped.items()
                                             if v))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PairCollator:
    """把 N 个偏好对拼成 (2N, L) 的 batch：前 N 个是 chosen，后 N 个是 rejected。

    这样 policy 一次前向、reference 一次前向，
    就拿到了 DPO 公式需要的全部四个量。
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self._single = None

    def _collate(self, batch: list[dict]) -> dict:
        # 复用 SFT 的 collator，避免两份 padding 逻辑各自出 bug
        from .sft_peft import MultimodalCollator
        if self._single is None:
            self._single = MultimodalCollator(pad_token_id=self.pad_token_id)
        return self._single(batch)

    def __call__(self, items: list[dict]) -> dict:
        ch = self._collate([it["chosen"] for it in items])
        rj = self._collate([it["rejected"] for it in items])

        out = {
            "input_ids": torch.cat([ch["input_ids"], rj["input_ids"]], dim=0),
            "labels": torch.cat([ch["labels"], rj["labels"]], dim=0),
            "attention_mask": torch.cat([ch["attention_mask"], rj["attention_mask"]],
                                        dim=0),
            "n_pairs": len(items),
        }
        for k in ("pixel_values", "image_grid_thw"):
            if k in ch or k in rj:
                out[k] = torch.cat([ch.get(k, torch.empty(0)), rj.get(k, torch.empty(0))],
                                   dim=0)
        return out


# =============================================================================
# Part D · 训练
# =============================================================================


def _to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def compute_logps(model, batch: dict, adapter: str, need_grad: bool):
    """在指定 adapter 下算 logps。

    need_grad=False 时额外做两件事：
      ① torch.no_grad()    —— 参考模型绝对不能有梯度
      ② model.eval()       —— 关掉 LoRA dropout。参考值必须是确定的，
                              否则每步 ref 都在抖，log ratio 就成了噪声。
    """
    import torch
    from .dpo_loss import get_batch_logps

    model.set_adapter(adapter)
    fwd = {k: v for k, v in batch.items()
           if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    if need_grad:
        logits = model(**fwd).logits
        return get_batch_logps(logits, batch["labels"])

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(**fwd).logits
            return get_batch_logps(logits, batch["labels"])
    finally:
        model.train(was_training)


def build_model(cfg: DPOConfig):
    """组装 base + policy adapter + reference adapter。"""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    sft = Path(cfg.sft_adapter)
    if not sft.exists():
        raise FileNotFoundError(
            f"\n✗ 找不到 SFT adapter：{sft}\n"
            f"\n  DPO 必须从 SFT 之后的模型出发（见本文件顶部说明②）。\n"
            f"  先跑：make train        （或 make train-qloa）\n"
            f"  或者用 --sft-adapter 指向你实际的 SFT 输出目录。\n")

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    quant_cfg = None
    if cfg.use_qlora_for_policy:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        quantization_config=quant_cfg,
        torch_dtype=dtype if quant_cfg is None else None,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(cfg.model_name)

    if cfg.use_qlora_for_policy:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()
    base.config.use_cache = False

    # 两个 adapter 都从 SFT 权重出发 —— 初始时 policy == reference，
    # 这正是 DPO 的起点（log ratio 一开始为 0）
    model = PeftModel.from_pretrained(
        base, str(sft), adapter_name=POLICY, is_trainable=True)
    model.load_adapter(str(sft), adapter_name=REFERENCE, is_trainable=False)
    model.set_adapter(POLICY)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数 {n_tr / 1e6:.2f} M / 总计 {n_all / 1e6:.0f} M "
          f"（{n_tr / n_all:.3%}）")
    print(f"  adapter：[{POLICY}] 训练中，[{REFERENCE}] 冻结"
          f"（两者都来自 {sft}）")
    return model, processor


def train(cfg: DPOConfig, dry_run: bool = False, limit: int = 0,
          max_steps: int = 0, inspect_only: bool = False):
    # 注意：torch 只在真正要训练/建模型时才 import。
    # --inspect 走的是纯标准库路径，本地没装 torch 也能跑。
    print("=" * 70)
    print("  DPO 训练")
    print("=" * 70)

    pairs = load_pairs(cfg.preference_files, image_root=cfg.image_root)
    if not pairs:
        print("\n✗ 没有偏好数据。先跑：")
        print("    python -m src.train.dpo_loss --from-badcases reports/bad_cases.jsonl")
        print("    python -m src.train.dpo_loss --contrastive data/processed/clean.jsonl")
        return 1

    issues = validate_pairs(pairs)
    bad = {k: v for k, v in issues.items() if v}
    if bad:
        print("  ⚠ 数据体检：" + " · ".join(f"{k} {v}" for k, v in bad.items()))
    pair_stats(pairs)

    if inspect_only:
        print("\n  --inspect 模式：到这一步就够了，没加载模型。")
        return 0

    if limit:
        random.Random(cfg.seed).shuffle(pairs)
        pairs = pairs[:limit]
        print(f"  --limit：只用 {len(pairs)} 条")

    # ---- 先做数据格式体检，不加载模型（省时间也省钱）----
    if dry_run:
        print("\n  --dry-run：检查 tokenize 后的真实长度分布")
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(
            cfg.model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        ds = PreferenceDataset(pairs[: min(12, len(pairs))], proc,
                               system_prompt=SYSTEM_PROMPT,
                               max_length=cfg.max_length,
                               max_len_ratio=cfg.max_len_ratio)
        if not ds.items:
            print("  ✗ 一条都没通过，检查图片路径和 prompt 格式")
            return 1
        lens = [it["chosen"]["input_ids"].shape[0] for it in ds.items]
        print(f"  样本长度：min {min(lens)} / 中位 {sorted(lens)[len(lens) // 2]} "
              f"/ max {max(lens)}   （上限 {cfg.max_length}）")
        it = ds.items[0]
        sup = int((it["chosen"]["labels"] != -100).sum())
        print(f"  第 1 条：总 {it['chosen']['input_ids'].shape[0]} token，"
              f"其中被监督 {sup} token")
        print(f"  像素张量：{tuple(it['chosen']['pixel_values'].shape)}")
        print("\n  ✓ 格式没问题，可以去开训了（去掉 --dry-run）")
        print(f"    预计显存：见 python scripts/estimate_vram.py --model 3b --method qlora")
        return 0

    # ---- 正式训练 ----
    import torch
    from torch.utils.data import DataLoader

    model, processor = build_model(cfg)
    device = next(model.parameters()).device

    ds = PreferenceDataset(pairs, processor, system_prompt=SYSTEM_PROMPT,
                           max_length=cfg.max_length,
                           max_len_ratio=cfg.max_len_ratio)
    if not ds.items:
        print("✗ 数据全部不可用")
        return 1

    collator = PairCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)
    loader = DataLoader(ds, batch_size=cfg.per_device_train_batch_size,
                        shuffle=True, collate_fn=collator)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("✗ 没有可训练参数，检查 adapter 是否加载成功")
        return 1
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(loader) // cfg.gradient_accumulation_steps)
    total_steps = int(steps_per_epoch * cfg.num_train_epochs)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    warmup = int(total_steps * cfg.warmup_ratio)
    print(f"  优化步数 {total_steps}（warmup {warmup}），"
          f"等效 batch = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")

    try:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    except Exception:
        sched = None

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "dpo_metrics.jsonl"
    mf = open(metrics_path, "w", encoding="utf-8")

    history = []
    step = 0
    micro = 0
    t0 = time.time()
    stop = False

    print("\n  开始训练（盯 accuracy 和 reward_margin，不要只看 loss）")
    print("  " + "-" * 66)

    for epoch in range(math.ceil(cfg.num_train_epochs)):
        model.train()
        for batch in loader:
            batch = _to_device(batch, device)
            n = batch.pop("n_pairs")

            p_logps = compute_logps(model, batch, POLICY, need_grad=True)
            r_logps = compute_logps(model, batch, REFERENCE, need_grad=False)

            pc, pr_ = p_logps[:n], p_logps[n:]
            rc, rr = r_logps[:n], r_logps[n:]

            from .dpo_loss import dpo_loss
            loss, m = dpo_loss(pc, pr_, rc, rr,
                               beta=cfg.beta,
                               label_smoothing=cfg.label_smoothing)

            (loss / cfg.gradient_accumulation_steps).backward()
            micro += 1

            if micro % cfg.gradient_accumulation_steps == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                if sched:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                rec = {"step": step, "epoch": epoch, "loss": m["loss"],
                       "accuracy": m["accuracy"],
                       "reward_margin": m["reward_margin"],
                       "rewards_chosen": m["rewards_chosen"],
                       "rewards_rejected": m["rewards_rejected"],
                       "grad_norm": float(gnorm), "lr": opt.param_groups[0]["lr"],
                       "elapsed_s": round(time.time() - t0, 1)}
                history.append(rec)
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()

                if step % cfg.logging_steps == 0:
                    print(f"  step {step:>4d}/{total_steps}  loss {m['loss']:.4f}  "
                          f"acc {m['accuracy']:.3f}  margin {m['reward_margin']:+.4f}  "
                          f"grad {float(gnorm):.2f}  lr {opt.param_groups[0]['lr']:.2e}")

                if step % cfg.save_steps == 0:
                    ck = out_dir / f"checkpoint-{step}"
                    model.save_pretrained(ck, selected_adapters=[POLICY])
                    processor.save_pretrained(ck)
                    print(f"    ↳ 保存 {ck}")
                    _prune_checkpoints(out_dir, cfg.save_total_limit)

                if max_steps and step >= max_steps:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    mf.close()
    final = out_dir / "final"
    model.save_pretrained(final, selected_adapters=[POLICY])
    processor.save_pretrained(final)
    print(f"\n  ✓ 完成，{step} 步，用时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"    adapter → {final}")
    print(f"    指标   → {metrics_path}")

    _write_summary(out_dir, cfg, history)
    return 0


def _prune_checkpoints(out_dir: Path, keep: int):
    cks = sorted([p for p in out_dir.glob("checkpoint-*")
                  if p.is_dir()],
                 key=lambda p: int(p.name.split("-")[-1]))
    for p in cks[:-keep] if keep > 0 else []:
        import shutil
        shutil.rmtree(p, ignore_errors=True)


def _write_summary(out_dir: Path, cfg: DPOConfig, history: list[dict]):
    """写一份人看得懂的收尾报告 —— 判断这次 DPO 到底有没有用。"""
    lines = ["# DPO 训练小结", "",
             f"- 起点 adapter：`{cfg.sft_adapter}`",
             f"- beta = {cfg.beta}，lr = {cfg.learning_rate}，"
             f"epochs = {cfg.num_train_epochs}",
             f"- 等效 batch = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}",
             ""]
    if history:
        h0, h1 = history[0], history[-1]
        hh = history[len(history) // 2]
        lines += ["| | 第一步 | 中段 | 最后一步 |", "|---|---|---|---|",
                  f"| loss | {h0['loss']:.4f} | {hh['loss']:.4f} | {h1['loss']:.4f} |",
                  f"| accuracy | {h0['accuracy']:.3f} | {hh['accuracy']:.3f} | {h1['accuracy']:.3f} |",
                  f"| reward_margin | {h0['reward_margin']:+.4f} | {hh['reward_margin']:+.4f} | {h1['reward_margin']:+.4f} |",
                  ""]
        lines.append("## 怎么读这张表")
        lines.append("")
        if h1["accuracy"] > 0.98:
            lines.append("- ⚠️ accuracy 逼近 1.0。**这可能不是好消息** —— "
                         "通常意味着过拟合偏好数据。")
            lines.append("  必须去看评测有没有掉：`make eval`。"
                         "掉了就减 epoch 或把 beta 调大（0.1 → 0.3）。")
        elif h1["accuracy"] > h0["accuracy"]:
            lines.append("- ✓ accuracy 在上升，符合预期。")
        else:
            lines.append("- ⚠️ accuracy 没上升。常见原因：")
            lines.append("  ① 偏好数据噪声太大（chosen/rejected 本身有问题）"
                         "② lr 太小 ③ beta 太大把更新压住了")
        if h1["reward_margin"] <= h0["reward_margin"]:
            lines.append("- ⚠️ reward_margin 没有扩大，说明模型没学到偏好差异。")
    lines += ["", "## 下一步（必做）", "",
              "DPO 的唯一验收标准是**评测**，不是训练指标：",
              "```bash",
              "python -m src.eval.run_eval --model Qwen/Qwen2.5-VL-3B-Instruct \\",
              f"    --adapter {out_dir}/final --tag dpo --baseline reports/raw.jsonl",
              "```",
              "和 SFT 版本对比，看准确率是否提升、幻觉率是否下降。", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


SYSTEM_PROMPT = (
    "你是一名专业、耐心的电商客服助手。"
    "你会同时收到文字和商品图片，请基于图片和文字给出准确回答。"
    "看不清的地方不要猜，直接说明。"
)


# =============================================================================
# Part E · CLI
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="DPO 偏好对齐训练（Day 27–29）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用：
  python -m src.train.dpo --inspect --in data/processed/dpo_from_badcases.jsonl
  python -m src.train.dpo --dry-run --config configs/dpo_3b.yaml
  python -m src.train.dpo --config configs/dpo_3b.yaml

先跑 --inspect / --dry-run，再开训。理由：DPO 数据出问题时，
训练不会报错，只会静默地把模型训坏。
""")
    ap.add_argument("--config")
    ap.add_argument("--in", dest="inputs", action="append", default=[],
                    help="偏好数据 jsonl，可重复传多个（会合并）")
    ap.add_argument("--image-root")
    ap.add_argument("--sft-adapter")
    ap.add_argument("--output-dir")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--no-qlora", action="store_true", help="关闭 4-bit（要更多显存）")
    ap.add_argument("--inspect", action="store_true",
                    help="只看数据（不加载模型，本地可跑）")
    ap.add_argument("--dry-run", action="store_true",
                    help="数据格式体检 + 显存提示（要 processor，不要 GPU）")
    args = ap.parse_args()

    cfg = DPOConfig.from_yaml(args.config) if args.config else DPOConfig()

    for k in ("sft_adapter", "output_dir", "image_root"):
        v = getattr(args, k.replace("-", "_"), None)
        if v:
            setattr(cfg, k, v)
    if args.inputs:
        cfg.preference_files = args.inputs
    if args.beta is not None:
        cfg.beta = args.beta
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.max_length is not None:
        cfg.max_length = args.max_length
    if args.no_qlora:
        cfg.use_qlora_for_policy = False

    try:
        return train(cfg, dry_run=args.dry_run, limit=args.limit,
                     max_steps=args.max_steps, inspect_only=args.inspect)
    except FileNotFoundError as e:
        print(e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
