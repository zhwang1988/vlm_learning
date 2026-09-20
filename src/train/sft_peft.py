"""
多模态 SFT 训练（原生 transformers + peft 路线）。

对应讲义 docs/06-sft-training.md，对应计划 Day 15–Day 18。

为什么除了 LLaMA-Factory 还要有这个：
  LLaMA-Factory 快，但黑盒。Day 17–18 你要读一遍原生实现，才能理解：
    - 数据是怎么 collate 的（image_grid_thw 是什么）
    - label mask 是怎么生效的
    - gradient_checkpointing 和 use_cache 为什么冲突
  这个文件就是那份「能读懂的原生实现」。

用法：
    python -m src.train.sft_peft --config configs/sft_lora_3b.yaml
    python -m src.train.sft_peft --config configs/sft_lora_3b.yaml --dry-run
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =============================================================================
# 1. 数据集
# =============================================================================


class MultimodalSFTDataset(Dataset):
    """读 build_sft.py 产出的 sharegpt 格式，就地构造 input_ids + labels。

    为什么不用 tokenizer 的 return_assistant_tokens_mask：
      多模态下它对 <|im_end|> 的处理不一致，有些版本不包括停止符。
      自己算一遍更可控，出问题也能 debug。
    """

    def __init__(self, jsonl_path: str | Path, processor,
                 system_prompt: str, max_length: int = 4096):
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.rows: list[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        # 过滤掉图片不存在的样本
        before = len(self.rows)
        self.rows = [r for r in self.rows
                     if all(Path(p).exists() for p in r.get("images", []))
                     or not r.get("images")]
        if len(self.rows) < before:
            print(f"  ⚠ 过滤掉 {before - len(self.rows)} 条图片缺失的样本")

        self._tok = processor.tokenizer
        self._im_start = self._tok.convert_tokens_to_ids("<|im_start|>")
        self._im_end = self._tok.convert_tokens_to_ids("<|im_end|>")
        self._nl_ids = set(self._tok.encode("\n", add_special_tokens=False))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        from PIL import Image

        images = []
        messages = [{"role": "system", "content": self.system_prompt}]

        for m in row["messages"]:
            if m["role"] == "system":
                continue
            content = []
            text = m["content"]
            # 把 <image> 占位符转成 processor 认识的 content 结构
            if "<image>" in text:
                parts = text.split("<image>")
                for i, seg in enumerate(parts):
                    if i > 0:
                        # 这里需要一个真实图片，稍后统一加
                        content.append({"type": "image"})
                    if seg.strip():
                        content.append({"type": "text", "text": seg})
            else:
                content.append({"type": "text", "text": text})
            messages.append({"role": m["role"], "content": content})

        # 按顺序加载图片，替换占位
        img_iter = iter(row.get("images", []))
        for m in messages:
            if not isinstance(m["content"], list):
                continue
            new_c = []
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    p = next(img_iter, None)
                    if p and Path(p).exists():
                        im = Image.open(p).convert("RGB")
                        images.append(im)
                        new_c.append({"type": "image", "image": im})
                    # 图片不存在就跳过这个 token
                else:
                    new_c.append(c)
            m["content"] = new_c

        try:
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(
                text=[full_text],
                images=images if images else None,
                return_tensors="pt",
            )
        except Exception as e:      # noqa: BLE001
            return self.__getitem__((idx + 1) % len(self.rows))

        input_ids = inputs["input_ids"][0]
        labels = torch.full_like(input_ids, -100)

        # --- 定位所有 assistant 片段 ---
        ids = input_ids.tolist()
        t_assist = self._tok.encode("assistant", add_special_tokens=False)

        i = 0
        while i < len(ids):
            if ids[i] == self._im_start and ids[i + 1:i + 1 + len(t_assist)] == t_assist:
                start = i + 1 + len(t_assist)
                while start < len(ids) and ids[start] in self._nl_ids:
                    start += 1
                end = start
                while end < len(ids) and ids[end] != self._im_end:
                    end += 1
                end = min(end + 1, len(ids))     # ← 包含 <|im_end|>
                labels[start:end] = input_ids[start:end]
                i = end
                continue
            i += 1

        if (labels != -100).sum() < 5:
            # mask 失败（可能是模板不匹配），跳过
            return self.__getitem__((idx + 1) % len(self.rows))

        out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids))[0]
            if "attention_mask" in inputs else torch.ones_like(input_ids),
        }
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            out["image_grid_thw"] = inputs["image_grid_thw"]
        return out


# =============================================================================
# 2. Collator —— 多模态的坑几乎都在这里
# =============================================================================


@dataclass
class MultimodalCollator:
    """把变长样本拼成 batch。

    三个必须处理好的点：
      ① padding：序列长度不同 → pad 到 batch 内最长
      ② labels 的 pad：必须用 -100，不能用 0，否则会开始学 pad token
      ③ pixel_values / image_grid_thw：**所有样本的图要拼在一起**，
         因为 Qwen 用 grid_thw 来切分哪些 patch 属于哪张图
    """

    pad_token_id: int = 0

    def __call__(self, batch: list[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}

        max_len = max(b["input_ids"].size(0) for b in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            L = b["input_ids"].size(0)
            input_ids[i, :L] = b["input_ids"]
            labels[i, :L] = b["labels"]
            attn[i, :L] = b["attention_mask"]

        out = {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

        # 拼接所有图片
        pvs, grids = [], []
        for b in batch:
            if "pixel_values" in b:
                pv = b["pixel_values"]
                pvs.append(pv if pv.dim() == 2 else pv.reshape(-1, pv.size(-1)))
            if "image_grid_thw" in b:
                g = b["image_grid_thw"]
                grids.append(g if g.dim() == 2 else g.reshape(-1, g.size(-1)))

        if pvs:
            out["pixel_values"] = torch.cat(pvs, dim=0)
            out["image_grid_thw"] = torch.cat(grids, dim=0)

        return out


# =============================================================================
# 3. 训练入口
# =============================================================================


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    train_file: str = "data/processed/sft_train.jsonl"
    eval_file: str = "data/processed/sft_eval.jsonl"
    output_dir: str = "outputs/qwen25vl3b-cx-lora-v0"
    system_prompt_file: Optional[str] = None

    # LoRA
    use_lora: bool = True
    qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_include_connector: bool = True
    lora_include_vision: bool = False

    # 训练
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    num_train_epochs: float = 3.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    seed: int = 42

    # 日志与保存
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "tensorboard"
    run_name: str = "cx-sft-v0"


def train(cfg: TrainConfig, dry_run: bool = False):
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    from .lora_utils import LoRAConfig, print_lora_plan, resolve_target_modules

    print("=" * 80)
    print(f"多模态 SFT: {cfg.model_name}")
    print("=" * 80)

    # --- 加载模型 ---
    processor = AutoProcessor.from_pretrained(
        cfg.model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
    )

    model_kwargs = dict(attn_implementation="sdpa", dtype=torch.bfloat16)
    if cfg.qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs.pop("dtype")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name, **model_kwargs
    )

    # 视觉塔一律冻结
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
        print("✓ 视觉塔已冻结")

    # --- LoRA ---
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        lcfg = LoRAConfig(
            r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
            include_connector=cfg.lora_include_connector,
            include_vision=cfg.lora_include_vision,
            learning_rate=cfg.learning_rate,
        )
        targets = print_lora_plan(model, lcfg)

        if cfg.qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )

        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        # ⚠️ 关键：gradient checkpointing 和 use_cache 冲突，必须关
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # --- 数据 ---
    from .sft_peft import MultimodalCollator, MultimodalSFTDataset
    from src.data.build_sft import DEFAULT_SYSTEM_PROMPT

    sys_p = DEFAULT_SYSTEM_PROMPT
    if cfg.system_prompt_file and Path(cfg.system_prompt_file).exists():
        sys_p = Path(cfg.system_prompt_file).read_text(encoding="utf-8").strip()

    print(f"\n加载数据...")
    train_ds = MultimodalSFTDataset(cfg.train_file, processor, sys_p, cfg.max_length)
    eval_ds = None
    if Path(cfg.eval_file).exists():
        eval_ds = MultimodalSFTDataset(cfg.eval_file, processor, sys_p, cfg.max_length)
    print(f"  train: {len(train_ds):,}   eval: {len(eval_ds) if eval_ds else 0:,}")

    if dry_run:
        print("\n[dry-run] 检查第一条样本的 label mask:")
        s = train_ds[0]
        n_sup = int((s["labels"] != -100).sum())
        print(f"  序列长度 {s['input_ids'].size(0)}, 监督 token {n_sup} "
              f"({n_sup / s['input_ids'].size(0):.1%})")
        txt = processor.tokenizer.decode(
            [int(x) for x in s["input_ids"][s["labels"] != -100]]
        )
        print(f"\n  被监督内容:\n  {txt[:400]}")
        print("\n[dry-run] 未启动训练。确认上面内容正确后去掉 --dry-run。")
        return None

    collator = MultimodalCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.eval_steps if eval_ds else None,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
        remove_unused_columns=False,      # ⚠️ 多模态必须 False
        dataloader_num_workers=2,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print(f"\n开始训练...")
    print(f"  有效 batch = {cfg.per_device_train_batch_size} × "
          f"{cfg.gradient_accumulation_steps} = "
          f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"\n✓ 训练完成，模型保存至 {cfg.output_dir}")

    # --- 训练后检查 ---
    print("\n" + "=" * 80)
    print("训练日志分析")
    print("=" * 80)
    from .monitor import parse_and_plot
    parse_and_plot(cfg.output_dir)

    return trainer


# =============================================================================
# 4. CLI
# =============================================================================


def main():
    import argparse

    ap = argparse.ArgumentParser(description="多模态 LoRA/QLoRA SFT")
    ap.add_argument("--config", help="YAML 配置路径")
    ap.add_argument("--model", default=None)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只检查数据格式不训练（强烈建议第一次先用它）")
    args = ap.parse_args()

    cfg = TrainConfig()

    if args.config:
        import yaml
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if args.model:
        cfg.model_name = args.model
    if args.qlora:
        cfg.qlora = True
    if args.epochs:
        cfg.num_train_epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.output_dir:
        cfg.output_dir = args.output_dir

    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
