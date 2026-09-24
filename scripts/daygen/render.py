#!/usr/bin/env python3
"""
渲染器：把 w3.py ~ w8.py 里的 DAYS 渲染成 days/*.md + notebooks/*.ipynb。

    python scripts/gen_days.py            # 生成全部
    python scripts/gen_days.py --week 3   # 只生成第 3 周

同时负责：
  - 重建 PLAN.md 里的 📄/📓 索引行（幂等，可反复跑）
  - 重建 days/README.md 的总表

**内容是脚本生成的**：要改某一天，改 `scripts/daygen/w*.py` 再重跑，
不要手改 `days/*.md` 或 notebook（下次生成会被覆盖）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DAYS_DIR = ROOT / "days"
NB_DIR = ROOT / "notebooks"

# 在哪跑 —— (徽标, 第三节标题里的说法, notebook 里的提示)
WHERE = {
    "cloud": ("☁️ 云 GPU", "在云 GPU 上", "需要 GPU（云机器）"),
    "local": ("💻 本地", "本地（无需 GPU）", "本地可跑，不需要 GPU"),
    "api": ("💻 本地 + API", "本地 + 联网 API", "本地可跑，需要 .env 里的 API key"),
    "both": ("💻/☁️ 两可", "本地或云上都行", "本地能跑一部分，训练/推理要 GPU"),
}

WEEK_TITLE = {
    1: "W1 · VLM 架构解剖",
    3: "W3 · SFT 训练工程",
    4: "W4 · 评测体系",
    5: "W5 · 偏好对齐与推理优化",
    6: "W6 · 多模态 Agent",
    7: "W7 · Shopify SaaS 产品化",
    8: "W8 · 打磨与交付",
}


# --------------------------------------------------------------------------
# 名字
# --------------------------------------------------------------------------

def nb_name(d: dict) -> str:
    return f"day-{d['n']:02d}_{d['slug']}.ipynb"


def md_name(d: dict) -> str:
    return f"day-{d['n']:02d}.md"


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------

def render_md(d: dict) -> str:
    icon, runwhere, _ = WHERE[d["where"]]
    L: list[str] = []

    L.append(f"# Day {d['n']} · {d['title']}")
    L.append("")
    L.append(f"> 预计 3–4h ｜ 📓 `notebooks/{nb_name(d)}` ｜ {icon} · {d['files']}")
    L.append(f"> 前置：{d['prereq']}")
    L.append("")
    L.append("## 今日目标（一句话）")
    L.append("")
    L.append(d["goal"])
    L.append("")

    L.append("## 一、读（60 min）")
    L.append("")
    L.append("材料：")
    L += [f"- {x}" for x in d["read"]]
    L.append("")
    L.append("思考题（先自己想，答案在讲义或代码注释里）：")
    L += [f"{i}. {x}" for i, x in enumerate(d["think"], 1)]
    L.append("")

    L.append("## 二、写（100 min）")
    L.append("")
    L.append(d["write_title"])
    L.append("")
    L.append("| 函数 / 文件 | 你要做什么 |")
    L.append("|---|---|")
    L += [f"| {a} | {b} |" for a, b in d["write_rows"]]
    if d.get("write_note"):
        L.append("")
        L.append(d["write_note"])
    L.append("")

    L.append(f"## 三、跑（{runwhere}）")
    L.append("")
    L.append("```bash")
    for item in d["run"]:
        cmd, comment = item if isinstance(item, tuple) else (item, "")
        if comment:
            L.append(f"# {comment}")
        L.append(cmd)
    L.append("```")
    L.append("")
    if d.get("expect"):
        L.append("期望输出（节选）：")
        L.append("```")
        L += d["expect"].strip("\n").splitlines()
        L.append("```")
        L.append("")

    L.append("## 四、验收清单")
    L.append("")
    L += [f"- [ ] {x}" for x in d["accept"]]
    L.append("")

    L.append("## 五、容易踩的坑")
    L.append("")
    L += [f"{i}. {x}" for i, x in enumerate(d["pits"], 1)]
    L.append("")

    L.append("## 六、打卡")
    L.append("")
    L.append("复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。")
    if d.get("footer"):
        L.append("")
        L.append(d["footer"])
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# notebook
# --------------------------------------------------------------------------

def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


ENV_CLOUD = '''import sys, torch
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__)
print("cuda   :", torch.version.cuda, "| available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu    : {p.name}  {p.total_memory / 1024**3:.0f} GB")
    print("bf16   :", torch.cuda.is_bf16_supported())
else:
    print("⚠️  没有 GPU —— 这一天的训练/推理跑不了。先看 docs/13-hardware-and-cost.md 租机器")'''

ENV_LOCAL = '''import sys
print("python:", sys.version.split()[0])
for m in ("numpy", "PIL", "yaml", "pandas"):
    try:
        mod = __import__(m)
        print(f"  {m:7s} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"  {m:7s} ❌ 缺 → pip install {m}")
print("\\n→ 本机没 GPU 不影响今天：今天只用纯 Python / numpy")'''


def render_nb(d: dict) -> dict:
    icon, _, nbwhere = WHERE[d["where"]]
    cells: list[dict] = []

    cells.append(_md(
        f"# Day {d['n']} · {d['title']}\n\n"
        f"**配套讲义**: [`days/{md_name(d)}`](../days/{md_name(d)}) ｜ **{nbwhere}**\n\n"
        f"{d['goal']}\n\n"
        f"> 📌 本 notebook 由 `scripts/gen_days.py` 生成 —— **别手改**，\n"
        f"> 要改内容请改 `scripts/daygen/w{d.get('week', '?')}.py` 后重跑脚本。"
    ))

    cells.append(_md("## 0. 环境检查"))
    cells.append(_code(ENV_CLOUD if d["where"] == "cloud" else ENV_LOCAL))

    for kind, text in d.get("nb", []):
        cells.append(_md(text) if kind == "md" else _code(text))

    cells.append(_md(
        "## 验收清单\n\n"
        + "\n".join(f"- [ ] {x}" for x in d["accept"])
        + "\n\n**卡住了？** 回看 [`days/" + md_name(d) + "`](../days/" + md_name(d) + ") 第五节「容易踩的坑」。"
    ))

    nxt = d.get("next_day_hint")
    if nxt:
        cells.append(_md(f"> **明天**：{nxt}"))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------
# PLAN.md 索引（幂等）
# --------------------------------------------------------------------------

DAY_HEAD = re.compile(r"^### Day (\d+) —")
INDEX_LINE = re.compile(r"^> 📄 `days/day-(\d+)\.md`")


def patch_plan_index(all_days: list[dict]) -> int:
    plan = ROOT / "PLAN.md"
    if not plan.exists():
        return 0
    nb_of = {d["n"]: nb_name(d) for d in all_days}

    # 拆成 (head, body) 段：head 为 None 表示段头之前的内容
    chunks: list[tuple[str | None, list[str]]] = []
    head: str | None = None
    body: list[str] = []
    for ln in plan.read_text().splitlines():
        m = DAY_HEAD.match(ln)
        if m:
            chunks.append((head, body))
            head, body = ln, []
        else:
            m_idx = INDEX_LINE.match(ln)
            # 只丢掉「本次会重新插入」的索引行（幂等）；Day 5–12 的索引行保留
            if m_idx and int(m_idx.group(1)) in nb_of:
                continue
            body.append(ln)
    chunks.append((head, body))

    out: list[str] = []
    n_ins = 0
    for head, body in chunks:
        if head is None:
            out += body
            continue
        n = int(DAY_HEAD.match(head).group(1))
        # 去掉段体开头的空行，稍后统一补
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        out.append(head)
        out.append("")
        if n in nb_of:
            out.append(f"> 📄 `days/day-{n:02d}.md`　·　📓 `notebooks/{nb_of[n]}`")
            out.append("")
            n_ins += 1
        out += body
        out.append("")

    text = "\n".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if not text.endswith("\n"):
        text += "\n"
    plan.write_text(text)
    return n_ins


# --------------------------------------------------------------------------
# days/README.md 总表（自动重建）
# --------------------------------------------------------------------------

# Day 1–4 由 w0.py 生成，但它们的 md 名字要在这里显式写出来
# （下面 W1 表格的列布局和 W2–W8 不同，所以不从 all_days 直接推导）。
# Day 5–12 是早期手写的，生成器只负责列进总表。
FIXED_ROWS = [
    (1, "技术版图与问题定义", "[`day-01.md`](day-01.md)",
     "[`day-01_environment_and_first_inference.ipynb`]"
     "(../notebooks/day-01_environment_and_first_inference.ipynb)", "💻/☁️ 两可"),
    (2, "视觉编码器：ViT → SigLIP", "[`day-02.md`](day-02.md)",
     "[`day-02_vit_from_scratch.ipynb`](../notebooks/day-02_vit_from_scratch.ipynb)",
     "☁️ 云 GPU"),
    (3, "连接器：模态对齐那一层", "[`day-03.md`](day-03.md)",
     "[`day-03_connector_compare.ipynb`](../notebooks/day-03_connector_compare.ipynb)",
     "☁️ 云 GPU"),
    (4, "Qwen2.5-VL 架构精读", "[`day-04.md`](day-04.md)",
     "[`day-04_visual_token_budget.ipynb`]"
     "(../notebooks/day-04_visual_token_budget.ipynb)", "☁️ 云 GPU"),
    (5, "从零手搭 Mini-VLM", "[`day-05.md`](day-05.md)",
     "[`day-05_minivlm_assembly.ipynb`](../notebooks/day-05_minivlm_assembly.ipynb)", "☁️ 云 GPU"),
    (6, "复盘：一次完整的图文推理（M1）", "[`day-06.md`](day-06.md)",
     "[`day-06_full_inference_review.ipynb`](../notebooks/day-06_full_inference_review.ipynb)", "☁️ 云 GPU"),
    (7, "图像预处理全链路", "[`day-07.md`](day-07.md)",
     "[`day-07_image_preprocess.ipynb`](../notebooks/day-07_image_preprocess.ipynb)", "💻 本地"),
    (8, "客服数据 Taxonomy 设计", "[`day-08.md`](day-08.md)",
     "[`day-08_taxonomy_matrix.ipynb`](../notebooks/day-08_taxonomy_matrix.ipynb)", "💻 本地"),
    (9, "数据合成", "[`day-09.md`](day-09.md)",
     "[`day-09_data_synthesis.ipynb`](../notebooks/day-09_data_synthesis.ipynb)", "💻 本地 + API"),
    (10, "数据清洗与去重", "[`day-10.md`](day-10.md)",
     "[`day-10_cleaning_dedup.ipynb`](../notebooks/day-10_cleaning_dedup.ipynb)", "💻 本地"),
    (11, "数据打包与对话模板", "[`day-11.md`](day-11.md)",
     "[`day-11_build_sft.ipynb`](../notebooks/day-11_build_sft.ipynb)", "💻 本地"),
    (12, "数据集 v0 交付（M2）", "[`day-12.md`](day-12.md)",
     "[`day-12_dataset_card.ipynb`](../notebooks/day-12_dataset_card.ipynb)", "💻 本地"),
]


def render_days_readme(all_days: list[dict]) -> str:
    L: list[str] = []
    L.append("# days/ · 每日材料索引")
    L.append("")
    L.append("**一天一个 md，一天一个 notebook。** `PLAN.md` 只做总纲索引，正文都在这里。")
    L.append("")
    L.append("每个 `day-XX.md` 固定六节：**今日目标 → 读 → 写 → 跑 → 验收 → 坑 → 打卡**。")
    L.append("")
    L.append("> ⚠️ 这里是**生成物**。要改内容改 `scripts/daygen/w*.py`，然后：")
    L.append("> ```bash")
    L.append("> python scripts/gen_days.py           # 重新生成全部")
    L.append("> python scripts/gen_days.py --week 5  # 只重生第 5 周")
    L.append("> ```")
    L.append("")
    L.append("---")
    L.append("")

    def table(rows: list[tuple]) -> None:
        L.append("| Day | 主题 | 讲义 | notebook | 在哪跑 |")
        L.append("|---|---|---|---|---|")
        for n, title, md, nb, where in rows:
            L.append(f"| {n} | {title} | {md} | {nb} | {where} |")
        L.append("")

    L.append("## Week 1 · VLM 架构解剖")
    L.append("")
    table([r for r in FIXED_ROWS if r[0] <= 4])
    L.append("## Week 2 · 数据工程")
    L.append("")
    table([r for r in FIXED_ROWS if r[0] >= 5])

    for wk in sorted({d["week"] for d in all_days if d["n"] >= 13}):
        L.append(f"## {WEEK_TITLE.get(wk, f'Week {wk}')}")
        L.append("")
        table([(d["n"], d["title"], f"[`days/{md_name(d)}`]({md_name(d)})",
                f"[`notebooks/{nb_name(d)}`](../notebooks/{nb_name(d)})", WHERE[d["where"]][0])
               for d in all_days if d["n"] >= 13 and d["week"] == wk])

    L.append("---")
    L.append("")
    L.append("## 在哪跑？（省钱的关键）")
    L.append("")
    L.append("| 徽标 | 含义 |")
    L.append("|---|---|")
    L.append("| ☁️ 云 GPU | 必须租机器（训练 / 推理 / 服务）。什么时候开、开多久见 `docs/13-hardware-and-cost.md` |")
    L.append("| 💻 本地 | 你的 Mac 就能跑，不开 GPU，零成本 |")
    L.append("| 💻 本地 + API | 本地跑，但要 `.env` 里的 API key（合成 / 裁判模型） |")
    L.append("")
    L.append("**规律**：W2（数据）、W4（评测）、W7（Shopify）几乎全在本地；")
    L.append("只有 W3（训练）和 W5（对齐+推理）需要持续开卡。")
    L.append("")
    L.append("## 每天开工第一条命令")
    L.append("")
    L.append("```bash")
    L.append("make day N=25      # 打印当天：讲义路径 + notebook + 代码入口 + 命令")
    L.append("make list-days     # 列出全部日材料")
    L.append("```")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def write_all(days: list[dict], quiet: bool = False) -> dict:
    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    NB_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"md": 0, "nb": 0, "cells": 0}
    for d in days:
        p = DAYS_DIR / md_name(d)
        p.write_text(render_md(d), encoding="utf-8")
        stats["md"] += 1

        nbj = render_nb(d)
        (NB_DIR / nb_name(d)).write_text(
            json.dumps(nbj, ensure_ascii=False, indent=1), encoding="utf-8")
        stats["nb"] += 1
        stats["cells"] += len(nbj["cells"])
        if not quiet:
            print(f"  ✓ day-{d['n']:02d}  {d['title']:<24} {len(nbj['cells']):>2} cells  "
                  f"{WHERE[d['where']][0]}")
    return stats
