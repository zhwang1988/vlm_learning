#!/usr/bin/env python3
"""
生成 Day 5–12 的配套 notebook（一天一个）。

    python scripts/gen_notebooks.py

设计原则：
  - notebook 是「当天的操作台」：驱动 src/ 里对应模块跑通 + 可视化 + 自测题
  - 每格尽量独立可重跑；需要 GPU 的格子放显眼的 MD 提示
  - 重复运行本脚本会覆盖 notebook（内容以本脚本为准，别手改 notebook）
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
if not NB_DIR.exists():
    NB_DIR.mkdir()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ===========================================================================
# Day 5 · MiniVLM 拼装（云 GPU）
# ===========================================================================

day5 = [
    md("""# Day 5 · 从零手搭 Mini-VLM

**配套讲义**: `days/day-05.md` ｜ **需要 GPU**（torch + transformers）

今天的目标：把 SigLIP + Connector + Qwen2.5-0.5B 拼成 MiniVLM，
并**亲眼看到**视觉 embedding 如何替换 `<image>` 占位符。"""),

    md("## 0. 环境检查"),
    code("""import torch
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")"""),

    md("""## 1. 三块组件各自的输出形状

先分别跑一遍，记住每个环节的 shape —— 后面拼装出问题时按这个对照。"""),
    code("""import sys; sys.path.insert(0, "..")
import torch
from src.minivlm.vision import SiglipVisionWrapper
from src.minivlm.connector import build_connector

device = "cuda" if torch.cuda.is_available() else "cpu"

vision = SiglipVisionWrapper().to(device).eval()
conn = build_connector("mlp2",
                       vis_dim=1152, llm_dim=896).to(device).eval()

from PIL import Image, ImageDraw
img = Image.new("RGB", (448, 448), "white")
d = ImageDraw.Draw(img); d.ellipse([80, 80, 360, 360], fill=(230, 140, 60))

with torch.no_grad():
    vis = vision([img])            # [1, N_patches, 1152]
    proj = conn(vis)               # [1, N_patches, 896]
print("vision  :", tuple(vis.shape))
print("connector:", tuple(proj.shape))"""),

    md("""## 2. 拼装 MiniVLM 并观察 merge"""),
    code("""from src.minivlm.model import MiniVLM, MiniVLMConfig
from transformers import AutoTokenizer

cfg = MiniVLMConfig()
model = MiniVLM(cfg).to(device).eval()
tok = AutoTokenizer.from_pretrained(cfg.llm_name)

text = "<image>这张图里有什么？"
ids = tok(text, return_tensors="pt").input_ids.to(device)
n_img = (ids == model.image_token_id).sum().item()
print(f"文本 token: {ids.shape[1]}，其中占位符: {n_img}")

with torch.no_grad():
    out = model(input_ids=ids, images=[img])
print("merged 序列长度:", out.logits.shape[1])
print("期望值 =", ids.shape[1] - n_img + proj.shape[1], "（文本-占位符+视觉）")"""),

    md("""## 3. 反例实验：视觉 embedding 换成全零

形状没变、信息归零。如果 loss 还能算 —— 说明形状正确但「语义」才是关键。"""),
    code("""with torch.no_grad():
    vis_zero = torch.zeros_like(vision([img]))
    proj_zero = conn(vis_zero)
    out_zero = model(input_ids=ids, images=[img], visual_override=proj_zero) \\
        if hasattr(model, "visual_override") else model(input_ids=ids, images=[img])
print("正常输出 logits 均值:", out.logits.mean().item())
print("零视觉  logits 均值:", out_zero.logits.mean().item())
print("两个值不同 → 视觉信息真的参与了计算")"""),

    md("""## 4. 今日验收

- [ ] merged 长度 = 文本 - 占位符 + 视觉 token
- [ ] 手画数据流存 `assets/day5-flow.png`
- [ ] 回答：LLM 看到的到底是什么？

**卡住了？** 回看 `days/day-05.md` 第五节「容易踩的坑」（dtype / mask / 多占位符）。"""),
]

# ===========================================================================
# Day 6 · W1 收官复盘（云 GPU）
# ===========================================================================

day6 = [
    md("""# Day 6 · 复盘：一次完整的图文推理（M1 验收）

**配套讲义**: `days/day-06.md` ｜ **需要 GPU**

今天没有新知识。把官方 Qwen2.5-VL 和你的 MiniVLM 各跑一遍，对照差异。"""),

    md("## 1. 官方 Qwen2.5-VL 完整推理"),
    code("""import sys; sys.path.insert(0, "..")
from src.minivlm.generate import load_qwen, qwen_infer
from PIL import Image, ImageDraw

img = Image.new("RGB", (448, 448), "white")
d = ImageDraw.Draw(img)
d.ellipse([60, 100, 200, 240], fill=(230, 140, 60))
d.rectangle([250, 150, 390, 300], fill=(60, 110, 200))
d.text((70, 360), "MMLAB DAY6", fill=(20, 20, 20))

model, proc = load_qwen()
ans, n_vis = qwen_infer(model, proc, img, "图里有哪些形状？分别什么颜色？")
print("视觉 token:", n_vis); print("回答:", ans)"""),

    md("## 2. 图片顺序敏感性实验"),
    code("""from src.minivlm.generate import compare_order
compare_order(model, proc, img, "图里有哪些形状？分别什么颜色？")
# 观察 image-first 与 text-first 的输出差异；强模型下差异可能很小"""),

    md("## 3. MiniVLM 对比（0.5B 未训练基座）"),
    code("""from src.minivlm.generate import minivlm_infer
try:
    ans2 = minivlm_infer(img, "图里有哪些形状？")
    print("MiniVLM 回答:", ans2)
    print("→ 别期待质量。它没经过对齐，能出连贯中文就算成功。")
except Exception as e:
    print("MiniVLM 推理需要按 Day5 notebook 先验证 forward：", e)"""),

    md("## 4. 视觉 token 数对照（Day 4 遗留自检）"),
    code("""from src.minivlm.processor import compute_visual_tokens, check_against_official
for size in [(448, 448), (768, 432), (300, 3000)]:
    t = compute_visual_tokens(*size)
    print(f"{size[0]}x{size[1]}  →  {t} visual tokens")"""),

    md("""## 5. M1 验收清单

- [ ] 白板画数据流，拍照存 `assets/day6-whiteboard.png`
- [ ] 指出流水线最易出 bug 的一步 + 理由
- [ ] 官方 vs MiniVLM 差异的两个原因
- [ ] `progress/weekly-review.md` W1 段已写
- [ ] 进度表 W1 六天 `[x]`，M1 打卡"""),

    md("""> **下周预告**：W2 数据工程。Day 7 起全部可在本地跑，不开 GPU（省钱周）。
> 明早第一件事：打开 `days/day-07.md`。"""),
]

# ===========================================================================
# Day 7 · 图像预处理（本地）
# ===========================================================================

day7 = [
    md("""# Day 7 · 图像预处理全链路

**配套讲义**: `days/day-07.md` ｜ **本地可跑**（numpy + Pillow，无 GPU）

处理链：EXIF → 透明底 → resize → 归一化 → pHash。"""),

    md("## 1. 单图处理链可视化"),
    code("""import sys; sys.path.insert(0, "..")
from PIL import Image, ImageOps
from src.data.image_utils import load_and_normalize

# 造一张「脏图」：EXIF 旋转 + 透明通道
img = Image.new("RGBA", (640, 400), (0, 0, 0, 0))
from PIL import ImageDraw
d = ImageDraw.Draw(img); d.ellipse([100, 50, 500, 350], fill=(200, 90, 60, 255))
img.save("/tmp/_dirty.png", exif=b"") # 实际照片的 EXIF 旋转更常见

out = load_and_normalize("/tmp/_dirty.png")
print("mode:", out.mode, " size:", out.size)
out"""),

    md("## 2. pHash 鲁棒性：噪声 / 缩放 / JPEG 压缩"),
    code("""import io, random
from src.data.image_utils import phash, hamming_distance

base = load_and_normalize("/tmp/_dirty.png")
h0 = phash(base)

def variant(im, mode):
    if mode == "noise":
        from PIL import ImageFilter
        return im.filter(ImageFilter.GaussianBlur(1))
    if mode == "resize":
        return im.resize((im.width//2, im.height//2)).resize(im.size)
    if mode == "jpeg":
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=40); buf.seek(0)
        return Image.open(buf).convert(im.mode)

for mode in ["noise", "resize", "jpeg"]:
    h = phash(variant(base, mode))
    print(f"{mode:6s} 汉明距离 = {hamming_distance(h0, h):2d}   （≤8 视为重复）")"""),

    md("## 3. 重复图分组（union-find）"),
    code("""from src.data.image_utils import find_duplicate_groups

paths = ["/tmp/_dirty.png"] * 3 + ["/tmp/_other.png"]
Image.new("RGB", (300, 300), (30, 120, 200)).save("/tmp/_other.png")
groups = find_duplicate_groups(paths, threshold=8)
print("重复组:", groups)"""),

    md("""## 4. 作业：对你的商品图批量体检

把 5–200 张图放进 `data/raw_images/`，跑下面的格子。
输出尺寸 / 宽高比 / visual token 分布 —— 这是今天的主产出。"""),
    code("""from pathlib import Path
from src.minivlm.processor import compute_visual_tokens

folder = Path("../data/raw_images")
files = sorted(folder.glob("*")) if folder.exists() else []
if not files:
    print("把图片放进 data/raw_images/ 后重跑。现在用演示图代替。")
    files = ["/tmp/_dirty.png", "/tmp/_other.png"]

rows = []
for f in files:
    try:
        im = load_and_normalize(str(f))
        rows.append((f.name, im.size, round(im.width/im.height, 2),
                     compute_visual_tokens(im.width, im.height)))
    except Exception as e:
        rows.append((f.name, "ERROR", str(e)[:40], "-"))

print(f"{'文件':<24}{'尺寸':<14}{'比例':<7}{'visual tokens'}")
for r in rows:
    print(f"{str(r[0]):<24}{str(r[1]):<14}{str(r[2]):<7}{r[3]}")"""),

    md("""## 5. 验收
- [ ] 能讲清「为什么不直接 resize 到 448×448」
- [ ] 批量体检表已产出
- [ ] pHash 阈值 ≤8 的理由能自圆其说"""),
]

# ===========================================================================
# Day 8 · Taxonomy（本地）
# ===========================================================================

day8 = [
    md("""# Day 8 · 客服数据 Taxonomy 设计

**配套讲义**: `days/day-08.md` ｜ **本地可跑**

今天定整个项目的地基：8 意图 × 6 图像类型的数据矩阵。"""),

    md("## 1. 打印配比矩阵"),
    code("""import sys; sys.path.insert(0, "..")
from src.data.taxonomy import INTENTS, IMAGE_TYPES, TARGET_DISTRIBUTION, print_matrix
print_matrix()"""),

    md("## 2. 难度分层 L1–L4"),
    code("""from src.data.taxonomy import DIFFICULTY_TIERS
for tier, spec in DIFFICULTY_TIERS.items():
    print(f"{tier}: {spec['desc']}")
    print(f"   例: {spec['example']}\\n")"""),

    md("## 3. 生成合成任务清单"),
    code("""from src.data.taxonomy import build_generation_plan, export
plan = build_generation_plan()
print(f"合成任务总数: {len(plan)}")
for p in plan[:5]:
    print(" ", p)
export("../data/taxonomy_plan.json")
print("已导出 data/taxonomy_plan.json")"""),

    md("""## 4. 作业（今天的核心，30 分钟电商客服调研）

对矩阵里每个非零格子：
1. 去淘宝/京东客服页找一个**真实问法**记下来
2. 找不到真实例子的格子 → 把配量调 0（伪需求）
3. 修改 `src/data/taxonomy.py` 里的 `TARGET_DISTRIBUTION` 后重跑上面格子

**把抄来的问法直接贴到下面这个格子里**（Day 9 当合成种子）："""),
    code("""real_questions = {
    # "尺码咨询": ["175/80A 穿多大？", ...],   # ← 换成你抄的
}
print("待填写。参考格式：意图 → [问法1, 问法2, ...]")"""),

    md("""## 5. 验收
- [ ] 矩阵无孤岛（每行每列有量）
- [ ] 每个意图 ≥1 个真实问法
- [ ] 能回答：为什么「转人工」也要造数据"""),
]

# ===========================================================================
# Day 9 · 数据合成（本地 + API）
# ===========================================================================

day9 = [
    md("""# Day 9 · 数据合成

**配套讲义**: `days/day-09.md` ｜ **本地可跑，需要 `.env` 里的 API key**
⚠️ 今天花真金白银。流程：20 条试跑 → 抽检 ≥70% → 才放量 2k。"""),

    md("## 0. API key 就位检查"),
    code("""import sys; sys.path.insert(0, ".."); os_probe = True
from pathlib import Path
env = Path("../.env")
print("`.env` 存在:", env.exists())
if env.exists():
    keys = [l.split("=")[0] for l in env.read_text().splitlines()
            if "=" in l and not l.strip().startswith("#")]
    print("已定义的变量:", keys)
    print("SYNTH_API_KEY 有值:", any("SYNTH_API_KEY=" in l and l.split("=",1)[1].strip()
                                    for l in env.read_text().splitlines()))"""),

    md("## 1. 试跑 20 条（几分钱）"),
    code("""# 终端里跑更稳（notebook 里长任务不好中断）:
#   python -m src.data.synth --limit 20 --out data/synthetic/pilot.jsonl
# 此处直接调库演示：
from src.data.synth import Synthesizer
syn = Synthesizer(limit=20, out="../data/synthetic/pilot.jsonl")
try:
    syn.run()
except Exception as e:
    print("合成失败：", e)
    print("检查 .env 的 SYNTH_API_KEY / SYNTH_BASE_URL / SYNTH_MODEL")"""),

    md("## 2. 抽检样本"),
    code("""import json, random
from pathlib import Path
f = Path("../data/synthetic/pilot.jsonl")
if f.exists():
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    print(f"共 {len(rows)} 条\\n")
    for r in random.sample(rows, min(3, len(rows))):
        print("=" * 60)
        print("意图:", r.get("intent"), "| 情绪:", r.get("emotion"))
        print("问:", r.get("conversations", [{}])[0].get("value", "")[:80])
        print("答:", r.get("conversations", [{}])[-1].get("value", "")[:120])
else:
    print("先跑上面的试跑格子。")"""),

    md("## 3. 分布对照（合成 vs Day 8 计划）"),
    code("""from collections import Counter
from src.data.taxonomy import TARGET_DISTRIBUTION
if f.exists():
    c = Counter(r.get("intent", "?") for r in rows)
    total = sum(c.values())
    print(f"{'意图':<12}{'实际':>6}{'占比':>8}")
    for intent, cnt in c.most_common():
        print(f"{intent:<12}{cnt:>6}{cnt/total:>7.1%}")
    print("\\n→ 偏差最大的意图记进打卡，Day 10/11 补偿。")
else:
    print("试跑后再看。")"""),

    md("""## 4. 抽检合格率打分表（人工，5 分钟）

对 20 条逐条打分：合格 / 语气雷同 / 幻觉参数 / 答非所问。
合格率 < 70% → 改 `SYSTEM_PROMPT` / personas → 重跑。**不要带病放量。**

## 5. 验收
- [ ] 20 条试跑 ≥70% 合格
- [ ] 2k 条正式跑完（终端命令），断点文件在
- [ ] 记账：花了多少钱 / 多少条 / 单条成本"""),
]

# ===========================================================================
# Day 10 · 清洗去重（本地）
# ===========================================================================

day10 = [
    md("""# Day 10 · 数据清洗与去重

**配套讲义**: `days/day-10.md` ｜ **本地可跑**
W2 最重要的一天。每一条被删的数据，你都要能说出为什么。"""),

    md("## 1. 五段清洗流水线全量跑"),
    code("""import sys; sys.path.insert(0, "..")
from src.data.dedup import clean_pipeline

inp = "../data/synthetic/sft_2k.jsonl"     # 没有就先用 pilot.jsonl
out = "../data/clean/"
import os; os.makedirs(out, exist_ok=True)
try:
    report = clean_pipeline(inp, out)
    print(report.to_markdown())            # CleaningReport
except FileNotFoundError:
    print("找不到", inp, "—— 先跑 Day 9 的正式合成，或把 inp 改成 pilot 文件。")"""),

    md("## 2. 检查「同图不同问」被保留"),
    code("""from src.data.dedup import dedup_simple
# 构造：同一张图、两个不同问题 —— 正当样本，不能被图像去重误杀
same_img_diff_q = [
    {"id": "a1", "image": "imgA", "question": "这件多大码？", "answer": "M/L 有货。"},
    {"id": "a2", "image": "imgA", "question": "什么材质？",   "answer": "95% 棉。"},
    {"id": "a3", "image": "imgA", "question": "这件多大码？", "answer": "M/L 有货。"},  # 真·重复
]
kept, dropped = dedup_simple(same_img_diff_q)
print("保留:", [r["id"] for r in kept])      # 期望 a1, a2
print("淘汰:", [r["id"] for r in dropped])   # 期望 a3"""),

    md("## 3. 隔离区抽检（quarantine）"),
    code("""import json, random
from pathlib import Path
q = Path("../data/clean/quarantine.jsonl")
if q.exists():
    rows = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
    for r in random.sample(rows, min(5, len(rows))):
        print("-" * 50)
        print("淘汰原因:", r.get("drop_reason"))
        print("内容摘要:", str(r)[:120])
    print("\\n→ 误杀率感受一下。太高（>30%）就回去调阈值。")
else:
    print("流水线跑完才会有 quarantine.jsonl")"""),

    md("""## 4. 思考题自答（写在打卡里）
1. 字面去重漏什么？语义去重误杀什么？
2. 为什么按图去重而不是按样本？
3. 空回答样本删还是留？

## 5. 验收
- [ ] 报告每条淘汰有原因字段
- [ ] 语义去重误杀率 < 30%
- [ ] `data/clean/REPORT.md` 已生成"""),
]

# ===========================================================================
# Day 11 · 打包与 label mask（本地）
# ===========================================================================

day11 = [
    md("""# Day 11 · 数据打包与对话模板

**配套讲义**: `days/day-11.md` ｜ **本地可跑**
今天出错的样本，Day 14 会以「loss 不收敛」的形式报复你。"""),

    md("## 1. 先看 Qwen 的 chat template 长什么样"),
    code("""from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
msgs = [{"role": "system", "content": "你是客服"},
        {"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>多大码？"},
        {"role": "assistant", "content": "M/L 有现货。"}]
print(tok.apply_chat_template(msgs, tokenize=False))"""),

    md("## 2. 构造带 label mask 的训练样本"),
    code("""import sys; sys.path.insert(0, "..")
from src.data.build_sft import build_labeled_sample, inspect_sample
sample = {"system": "你是服装店客服", "image": "imgA",
          "user": "多大码有货？", "assistant": "您好，M/L 有现货。"}
enc = build_labeled_sample(sample)
inspect_sample(enc, tok)   # 逐 token 打印：哪些算 loss，哪些 -100"""),

    md("""## 3. 三种 mask 错误的反例（今天的考点）

逐个把下面的开关打开跑一遍，看 mask 错成什么样、**后果**是什么。"""),
    code("""WRONG_MODE = None   # 依次改成: "system_in_loss" / "miss_im_end" / "user_in_loss"
# WRONG_MODE = "system_in_loss"   # 模型学会抢答 system
# WRONG_MODE = "miss_im_end"      # 模型学不会停 → 推理时说个没完
# WRONG_MODE = "user_in_loss"     # 模型学会复述问题

if WRONG_MODE:
    enc_bad = build_labeled_sample(sample, wrong_mode=WRONG_MODE)
    inspect_sample(enc_bad, tok)
else:
    print("改 WRONG_MODE 依次观察三种错误。")"""),

    md("## 4. 泄漏检查（图像级）"),
    code("""from src.data.build_sft import check_leakage
train = [{"image": "imgA"}, {"image": "imgB"}]
eval_ = [{"image": "imgB"}]            # 故意泄漏
n = check_leakage(train, eval_, key=lambda s: s["image"])
print("泄漏数:", n, "（期望 1，被拦住才算检查有效）")"""),

    md("## 5. 全量打包"),
    code("""# 终端跑：
#   python -m src.data.build_sft --in data/clean/clean.jsonl --out data/processed/
#   python -m src.data.build_sft --inspect data/processed/sft_train.jsonl 0
print("打包命令见注释。跑完把 --inspect 的输出贴到打卡里。")"""),

    md("""## 6. 验收
- [ ] 抽 5 条肉眼检查：三种 mask 错误一个没有
- [ ] check_leakage = 0
- [ ] train/eval 意图分布偏差 < 3%"""),
]

# ===========================================================================
# Day 12 · 数据集交付（本地）
# ===========================================================================

day12 = [
    md("""# Day 12 · 数据集 v0 交付（M2 验收）

**配套讲义**: `days/day-12.md` ｜ **本地可跑**
今天交付一份别人能拿去训模型的数据集。"""),

    md("## 1. 数据集体检总表"),
    code("""import sys; sys.path.insert(0, "..")
import json
from collections import Counter
from pathlib import Path

p = Path("../data/processed/sft_train.jsonl")
if not p.exists():
    print("先跑完 Day 11 的打包。")
else:
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"train: {len(rows)} 条")
    n_img = sum(1 for r in rows if r.get("images"))
    n_multi = sum(1 for r in rows if r.get("images") and len(r["images"]) > 1)
    print(f"含图: {n_img}  多图: {n_multi}  纯文本: {len(rows)-n_img}")
    lens = [len(json.dumps(r, ensure_ascii=False)) for r in rows]
    print(f"样本字节长度 p50/p95: {sorted(lens)[len(lens)//2]}/{sorted(lens)[int(len(lens)*.95)]}")"""),

    md("## 2. 实际分布 vs Day 8 计划"),
    code("""from src.data.taxonomy import TARGET_DISTRIBUTION
if p.exists():
    c = Counter(r.get("intent", "?") for r in rows)
    total = sum(c.values())
    for intent, cnt in c.most_common():
        plan_n = sum(TARGET_DISTRIBUTION.get(intent, {}).values())
        plan_pct = plan_n / sum(sum(v.values()) for v in TARGET_DISTRIBUTION.values())
        print(f"{intent:<12} 实际 {cnt/total:>6.1%}   计划 {plan_pct:>6.1%}   Δ{cnt/total-plan_pct:+.1%}")"""),

    md("## 3. 生成 DATASET_CARD.md"),
    code("""from src.data.report import generate_card
card = generate_card(data_dir="../data/processed/")
print(card[:800])
print("……")
# 完整文件: data/processed/DATASET_CARD.md
# 「已知缺陷」至少写 3 条，诚实写。这是面试官最看重的部分。"""),

    md("## 4. 上传 HuggingFace（私有）"),
    code("""UPLOAD = False   # 确认无敏感信息后改 True
REPO = "your-name/cx-sft-v0"        # ← 改成你的
if UPLOAD:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, private=True, exist_ok=True, repo_type="dataset")
    api.upload_folder(folder_path="../data/processed/", repo_id=REPO, repo_type="dataset")
    print("上传完成:", f"https://huggingface.co/datasets/{REPO}")
else:
    print("上传前最后一遍 check_leakage + grep 敏感词！")"""),

    md("""## 5. M2 验收清单
- [ ] 卡片六字段齐全，已知缺陷 ≥3 条
- [ ] HF 私有仓库上传成功
- [ ] 周复盘写完，W2 六天 `[x]`，M2 `[x]`
- [ ] 一句话说清：这数据集适合训什么、不适合训什么"""),
]

NOTEBOOKS = {
    "day-05_minivlm_assembly.ipynb": day5,
    "day-06_full_inference_review.ipynb": day6,
    "day-07_image_preprocess.ipynb": day7,
    "day-08_taxonomy_matrix.ipynb": day8,
    "day-09_data_synthesis.ipynb": day9,
    "day-10_cleaning_dedup.ipynb": day10,
    "day-11_build_sft.ipynb": day11,
    "day-12_dataset_card.ipynb": day12,
}


def main():
    for name, cells in NOTEBOOKS.items():
        path = NB_DIR / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1))
        print(f"  ✓ {name}  ({len(cells)} cells)")
    print(f"\n共 {len(NOTEBOOKS)} 个 notebook → {NB_DIR}")


if __name__ == "__main__":
    main()