"""
数据集卡片生成。

对应计划 Day 12 的交付物。

「已知缺陷」这一节最有价值 —— 写不出来说明你没认真看过自己的数据。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


CARD_TEMPLATE = """# Dataset Card: {name}

## 概览

| 项 | 值 |
|---|---|
| 版本 | {version} |
| 训练/验证/测试 | {n_train:,} / {n_val:,} / {n_test:,} |
| 语言 | 中文 |
| 场景 | 电商客服（语言 + 图像） |
| 格式 | sharegpt (LLaMA-Factory multimodal) |
| 生成时间 | {created} |

## 意图分布（训练集）

| 意图 | 数量 | 占比 |
|---|---:|---:|
{intent_table}

## 图像类型分布（训练集）

| 类型 | 数量 | 占比 |
|---|---:|---:|
{image_type_table}

## 难度分布

{difficulty_bars}

## 数据来源

{source_section}

## 构造流程

```
原始图片 + 商品信息
      ↓ ① 图像清洗（EXIF/alpha/CMYK/比例检查）
      ↓ ② 合成对话（强模型 + persona/emotion 多样性种子）
      ↓ ③ 规则过滤（长度/联系方式/AI 自我暴露/复读）
      ↓ ④ 去重（pHash + 文本指纹）
      ↓ ⑤ 分层切分（按 intent × image_type 分层）
      ↓ ⑥ 图片级泄漏检查
   训练集 / 验证集 / 测试集
```

## 已知缺陷

{known_issues}

## 隐私与合规

{pii_section}

## 使用限制

- 仅用于学术研究与自建项目验证
- 合成样本可能存在事实错误，生产环境需人工抽检
- 不建议直接用于面向真实消费者的决策

---

*由 `src/data/report.py` 自动生成*
"""


def _pct_table(counter: Counter, total: int) -> str:
    rows = []
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        rows.append(f"| {k} | {v:,} | {v / max(total, 1):.1%} |")
    return "\n".join(rows) if rows else "| （无数据） | - | - |"


def _difficulty_bars(counter: Counter, total: int) -> str:
    lines = []
    for k in sorted(counter.keys(), key=lambda x: str(x)):
        v = counter[k]
        bar = "█" * int(30 * v / max(total, 1))
        lines.append(f"- **难度 {k}**: {v:,}  {bar}")
    return "\n".join(lines) if lines else "（无数据）"


def generate_card(data_dir: str | Path = "data/processed",
                  out_path: str | Path = "data/processed/DATASET_CARD.md",
                  name: str = "CX-VLM-SFT-v0",
                  version: str = "0.1.0",
                  source_note: str = "由强模型按 taxonomy 合成，商品池为手工整理示例",
                  pii_note: str = "不含真实用户数据；合成过程中已过滤联系方式",
                  known_issues: list[str] | None = None) -> Path:
    import time

    data_dir = Path(data_dir)

    def load(p):
        fp = data_dir / p
        if not fp.exists():
            return []
        return [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]

    train = load("sft_train.jsonl")
    val = load("sft_eval.jsonl")
    test = load("sft_test.jsonl")

    intent_c = Counter(s.get("intent", "?") for s in train)
    img_c = Counter(s.get("image_type", "?") for s in train)
    diff_c = Counter(str(s.get("difficulty", "?")) for s in train)

    # 自动发现缺陷
    auto_issues = []
    n = max(len(train), 1)
    for k, v in intent_c.items():
        if v / n < 0.05:
            auto_issues.append(f"**意图 `{k}` 样本偏少**（{v} 条，{v / n:.1%}）——线上占比可能更高，建议补足至 5% 以上")
    for k, v in img_c.items():
        if v / n < 0.05:
            auto_issues.append(f"**图像类型 `{k}` 样本偏少**（{v} 条，{v / n:.1%}）——该类图理解可能较弱")
    d12 = sum(diff_c.get(k, 0) for k in ("1", "2"))
    if d12 / n > 0.7:
        auto_issues.append(f"**难度分布偏低**——难度 1-2 占 {d12 / n:.1%}，模型缺少难例锻炼")

    stats_file = data_dir / "build_stats.json"
    if stats_file.exists():
        st = json.loads(stats_file.read_text(encoding="utf-8"))
        lk = st.get("leakage", {})
        if lk.get("n_leaks", 0) > 0:
            auto_issues.append(
                f"**⚠️ 图片级泄漏未处理**——发现 {lk['n_leaks']} 处（{lk.get('leak_rate', 0):.1%}），"
                f"评测分数会虚高，必须处理"
            )

    all_issues = (known_issues or []) + auto_issues
    if all_issues:
        issues_md = "\n".join(f"{i+1}. {x}" for i, x in enumerate(all_issues))
    else:
        issues_md = "（暂未发现明显缺陷——但请人工复核一遍，自动化检查只能发现分布问题）"

    content = CARD_TEMPLATE.format(
        name=name, version=version,
        n_train=len(train), n_val=len(val), n_test=len(test),
        created=time.strftime("%Y-%m-%d"),
        intent_table=_pct_table(intent_c, len(train)),
        image_type_table=_pct_table(img_c, len(train)),
        difficulty_bars=_difficulty_bars(diff_c, len(train)),
        source_section=source_note,
        known_issues=issues_md,
        pii_section=pii_note,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"✓ 数据集卡片 → {out_path}")
    if auto_issues:
        print(f"\n自动发现 {len(auto_issues)} 个潜在缺陷：")
        for i in auto_issues:
            print(f"  - {i.replace('**', '')}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--name", default="CX-VLM-SFT-v0")
    args = ap.parse_args()
    generate_card(args.data_dir, name=args.name)
