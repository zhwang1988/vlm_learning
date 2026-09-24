"""Week 1 · VLM 架构解剖（Day 1–4）

这一周的目标只有一个：把「多模态模型 = 视觉编码器 + 连接器 + 语言主干」
从**背下来**变成**能画出来、能讲出为什么**。

Day 1 在本地开工、云上做第一次推理（where="both"）；
Day 2–4 都是 `src/minivlm/` 里手写代码，需要 torch，所以在云上跑。

> Day 1–4 的内容原本内嵌在 `PLAN.md` 里。后来拆成独立日文件，
> 是为了兑现「一天一个 md」——不然 `progress-tracker.md` 的 Day 1–4
> 行指向的 `days/day-01.md` 根本不存在，点进去是 404。
"""
from __future__ import annotations


def D(**kw):
    return kw


DAYS = [
    # ======================================================================
    D(
        week=1, n=1, slug="environment_and_first_inference", title="技术版图与问题定义",
        where="both",
        files="`docs/00-orientation.md`、`scripts/download_model.py`、`notebooks/01_first_vlm_inference.ipynb`",
        prereq="无 —— 这是第 1 天。先把 `requirements-core.txt` 装好",
        goal="建立一张属于自己的「多模态技术树」：说清三条技术路线（对比学习式 / 融合式 / "
             "外挂式）各自把图塞进了哪一层，并在云上让 Qwen2.5-VL 真的描述一张商品图。",
        read=[
            "`docs/00-orientation.md` —— **先读这个**，它是 8 周的整体地图，读完你会知道每天在干嘛",
            "`docs/12-papers.md` 的「第一梯队」部分 + CLIP / Flamingo / BLIP-2 的**摘要页**。"
            "注意：今天只看摘要，**不要通读论文**（通读会耗掉一整天，而且 8 周里用不上细节）",
            "`docs/01-architecture.md` 第 1–2 节：三件套（视觉编码器 / 连接器 / LLM 主干）各自管什么",
        ],
        think=[
            "三条路线分别把图像信息塞进了哪一层？—— 对比学习塞进**向量空间**，"
            "融合式塞进**注意力层**，外挂式塞进**LLM 的输入序列**",
            "客服要「看图回答问题」，为什么「纯文本 LLM + OCR」这条路走不通？"
            "（提示：OCR 只给文字，丢掉了颜色/瑕疵/版型这些**非文字信息**）",
            "为什么 2024 年之后大家都收敛到「外挂式 + 强 LLM」这条路线？",
        ],
        write_title="环境体检 → 下载模型 → 第一次真实推理",
        write_rows=[
            ("`make check`", "本地环境体检。**torch/CUDA 标黄是正常的** —— 本地不训练，"
                             "只用来读文档改代码"),
            ("`scripts/download_model.py`", "已给实现，你只需要会用它。"
                                            "优先走 ModelScope（国内 2–5 分钟），带断点续传和完整性校验"),
            ("`notebooks/01_first_vlm_inference.ipynb`", "让 Qwen2.5-VL 描述一张商品图，"
                                                         "看它**实际**说了什么 —— 这是你第一次看到模型的真实水平"),
            ("`assets/tech-tree.png`", "手绘「多模态技术树」拍照存进来。手绘比复制别人的图有用得多"),
            ("`progress/daily-log.md`", "加三行打卡（今天最容易的一步，也是最容易忘的一步）"),
        ],
        write_note="> 今天的**真正产出不是代码，是心里有地图**。\n"
                   "> 代码你只是在「用」，亲手写的部分从 Day 2 才开始。别急着写模型。",
        run=[
            ("make check", "本地体检（torch 缺失只标黄，退出码 0）"),
            ("python scripts/download_model.py --status",
             "先看要下载多少、下到哪 —— 别直接开下，容易下到系统盘塞满"),
            ("python scripts/download_model.py --model 3b-instruct",
             "云上执行。约 6.2 GB，走 ModelScope 2–5 分钟"),
            ("make check-cloud", "云上体检。这里 torch/CUDA/显存缺失都是**硬伤**（退出码 1）"),
            ("jupyter lab notebooks/01_first_vlm_inference.ipynb", "跑通第一次推理"),
        ],
        expect="""
$ python scripts/download_model.py --status --root /root/autodl-tmp/models
模型存储根目录: /root/autodl-tmp/models
  Qwen2.5-VL-3B-Instruct      ✗ 未下载   预计 6.2 GB
  Qwen2.5-VL-7B-Instruct      ✗ 未下载   预计 14.5 GB
磁盘可用: 48.2 GB   ← 够下 3B，但 3B + 7B + checkpoint 会紧张

$ make check-cloud
[✓] python 3.11
[✓] torch 2.4.0+cu121
[✓] CUDA 可用, 设备: NVIDIA GeForce RTX 4090 (24 GB)
[✓] 依赖齐全
环境 OK（云训练模式）
""",
        accept=[
            "能**不看资料**说出三条技术路线的差异，以及为什么客服场景选外挂式",
            "云上能加载 Qwen2.5-VL-3B 并让它描述一张商品图（哪怕答案很一般）",
            "`assets/` 里有一张自己画的技术树（手绘拍照也算）",
            "知道「哪些事在本地做、哪些必须上云」—— 这一条决定了你 8 周花 ¥300 还是 ¥3000",
        ],
        pits=[
            "**从 HF 直接下载会非常慢甚至失败**。`download_model.py` 默认走 ModelScope，"
            "不要手动改成 `huggingface_hub.snapshot_download`。",
            "**不要把模型下到系统盘**。云机器系统盘通常只有 30–50 GB，下完模型就满了。"
            "统一用 `--root /root/autodl-tmp/models`（AutoDL 的数据盘）。",
            "第一次推理会发现模型答得很平庸 —— **这是正常的，也是这个项目存在的理由**。"
            "把它的原始回答抄进打卡，8 周后回来看对比。",
            "`make check` 在本地会标黄 torch —— **这不是错误**，别去花两小时装 torch。"
            "本地装 torch 只会得到一个跑不动的 CPU 版，毫无用处。",
        ],
        nb=[
            ("md", "## 0. 环境自检\n\n先确认自己在哪个环境跑（本地还是云上），两边的期望是不一样的。"),
            ("code", '''import platform, sys, shutil

print("Python :", sys.version.split()[0])
print("系统   :", platform.platform())

def has(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "已安装")
    except ImportError:
        return None

for pkg in ("torch", "transformers", "PIL", "numpy"):
    v = has(pkg)
    print(f"{pkg:14s} {'✗ 缺失' if v is None else v}")

try:
    import torch
    print("CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("设备    :", torch.cuda.get_device_name(0))
        print("显存    :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
except ImportError:
    print("\\n（本地没装 torch 是正常的 —— 本地只负责读文档改代码）")

print("\\n磁盘可用:", round(shutil.disk_usage(".").free / 1024**3, 1), "GB")'''),
            ("md", """## 1. 模型下载状态

**先看要下多少，再决定下不下。** 云机器系统盘经常只有 30–50 GB，
下完模型就满了 —— 这是新手最常踩的坑。"""),
            ("code", '''import subprocess, sys, os
root = os.environ.get("MODEL_ROOT", "/root/autodl-tmp/models")
r = subprocess.run([sys.executable, "../scripts/download_model.py",
                    "--status", "--root", root],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)'''),
            ("md", """## 2. 第一次真实推理

现在让 Qwen2.5-VL 描述一张商品图。

⚠️ 这一步需要**云 GPU + 已下载的模型权重**。本地跑会直接报错，那是预期的 ——
把这段代码留到云上跑。

**关键**：跑完把模型的原始回答抄进 `progress/daily-log.md`。8 周后你要拿它做对比。"""),
            ("code", '''# ⚠️ 需要云 GPU + 已下载权重。本地会报错，跳过即可。
import os
MODEL = os.environ.get("MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image

    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype="auto", device_map="auto").eval()

    # 有真实商品图就用真实的；没有就先用一张纯色图看流程通不通
    img_path = "../assets/sample_product.jpg"
    if not os.path.exists(img_path):
        img = Image.new("RGB", (448, 448), (240, 236, 228))
        print("（没找到示例商品图，用一张纯色图先验证流程）")
    else:
        img = Image.open(img_path).convert("RGB")
        print("用图:", img_path)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "请描述这张图片，然后告诉我它适合什么季节穿。"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)[0]
    print("\\n=== 模型回答 ===")
    print(answer)
    print("\\n→ 把这段原文抄进 progress/daily-log.md，8 周后再回来看")
except ImportError as e:
    print("本地缺少依赖（正常）:", e)
    print("这一步请在云 GPU 机器上跑")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    print("如果是找不到模型，先跑 Day 1 第三节的下载命令")'''),
            ("md", """## 3. 技术树（手绘）

不用打字，拿张纸画，拍照存 `assets/tech-tree.png`。

至少要画出来：

```
图像 → [?] → [?] → 语言模型 → 文本
        ①      ②          ③
      编码器   连接器     LLM
```

然后在三条路线上各标一个代表：CLIP / Flamingo / LLaVA。

画完问自己：**BLIP-2 的 Q-Former 属于①②③里的哪一环？它和 LLaVA 的 MLP 有什么本质区别？**"""),
            ("md", "## 4. 三行打卡\n\n复制下面这段，填好贴进 `progress/daily-log.md`。"),
            ("code", '''print("""今日打卡（填好贴进 progress/daily-log.md）
─────────────────────────────────────────
[学到] 三条技术路线的差别是 ______，我原来以为 ______
[产出] assets/tech-tree.png；模型原始回答：______
[卡住] ______（没卡就写「无」）
─────────────────────────────────────────""")'''),
        ],
        next_day_hint="`days/day-02.md` —— 手写 ViT：patch embedding + attention，打印每一层的 shape",
    ),

    # ======================================================================
    D(
        week=1, n=2, slug="vit_from_scratch", title="视觉编码器：ViT → SigLIP",
        where="cloud",
        files="`src/minivlm/vision.py`、`docs/02-vision-encoder.md`",
        prereq="Day 1 完成（模型已下载，云环境可用）",
        goal="亲手写出 ViT 的 patch embedding + attention block，跑通 `(B,3,448,448)` → "
             "`(B,N,D)`，并说清 N 是怎么算出来的、SigLIP 比 CLIP 改了什么。",
        read=[
            "`docs/02-vision-encoder.md` 前半 —— 重点是 patch 切分、位置编码、CLS token 的去留",
            "ViT 原文的 key idea（只看 figure 1 + method 前三段）",
            "CLIP 的 InfoNCE loss 推导 —— 搞懂「对比学习」到底在优化什么",
        ],
        think=[
            "`(3,448,448)` 的图，patch=14、无 CLS token，N 是多少？**先手算再跑代码**",
            "ViT 为什么必须有位置编码？如果去掉，模型会失去什么能力？",
            "CLIP 用 InfoNCE 对比 loss，SigLIP 换成了 sigmoid loss —— 换掉之后为什么"
            "小 batch 也能训得住？",
            "**本项目的关键一问**：Qwen2.5-VL 为什么最后选了 SigLIP 而不是 CLIP？",
        ],
        write_title="`src/minivlm/vision.py`（手写，这是本周最核心的代码）",
        write_rows=[
            ("`PatchEmbed`", "用 Conv2d 实现 patch 切分（kernel=stride=patch_size），"
                             "不要用 unfold —— 卷积版本更快也更短"),
            ("`MultiHeadSelfAttention`", "标准 MHA。注意 `scale = head_dim ** -0.5`，"
                                         "别写成 `sqrt(d_model)`，那是常见错误"),
            ("`ViTBlock`", "LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差。"
                           "**Pre-LN 的顺序不能改**，改了训练会崩"),
            ("`VisionEncoder`", "把 patch + 位置编码 + 若干 block 串起来，返回 `(B,N,D)`"),
            ("`shape_report()`", "逐层打印 shape —— 这个函数是给你调试用的，别删"),
            ("`load_siglip()`", "加载 timm 的 SigLIP 权重，与手写版对比输出形状"),
        ],
        write_note="> 手写 ViT 的价值不在「能跑」，而在**你会知道每个数字从哪来**。\n"
                   "> 后面读 Qwen2.5-VL 的 `2×2 patch merging` 时，你会立刻明白"
                   "「merge 把 N 除以 4、把 D 乘以 4」，因为它就是你今天写的这几行。",
        run=[
            ("python -m src.minivlm.vision", "自检：手写 ViT 前向 + 逐层 shape"),
            ("python -m src.minivlm.vision --compare-siglip",
             "和 timm 的 SigLIP 对比输出形状与参数量"),
        ],
        expect="""
$ python -m src.minivlm.vision
输入 (B,3,448,448)  patch=14  →  grid 32×32  →  N=1024
  patch_embed   (1, 1024, 1152)
  +pos_embed    (1, 1024, 1152)
  block ×27     (1, 1024, 1152)
  输出          (1, 1024, 1152)    参数量 0.30B

手算校验: ceil(448/14) ** 2 = 1024  ✓

$ python -m src.minivlm.vision --compare-siglip
手写版    (1, 1024, 1152)   0.30B
timm版    (1, 1024, 1152)   —     ✓ 形状一致
""",
        accept=[
            "能**手算**任意尺寸图片的 N（例如 1024×768、patch=14、merge=2 → 答案见讲义）",
            "`shape_report()` 打印的每一层形状都能解释清楚",
            "能说出 SigLIP 与 CLIP 的两点差异（loss 形式 / 是否用 softmax 归一化）",
            "能回答「为什么 Qwen2.5-VL 用 SigLIP 而不是 CLIP」",
        ],
        pits=[
            "**scale 写错** —— 是 `head_dim ** -0.5`，不是 `d_model ** -0.5`。"
            "这个错误不会报错，只是效果变差，最难查。",
            "**忘了把 patch 展平前的通道顺序搞对** —— Conv2d 输出的 `(B,C,H,W)` "
            "要 `flatten(2).transpose(1,2)` 才变成 `(B,N,C)`，顺序错了 shape 也对不上语义。",
            "**Pre-LN 写成 Post-LN** —— 现代 ViT 全用 Pre-LN。写错的话深层网络训不动。",
            "**timm 权重加载时 num_classes 对不上** —— 视觉编码器要的是特征不是分类头，"
            "记得 `num_classes=0`。",
        ],
        nb=[
            ("md", "## 1. 手算 N，再用代码验证\n\n**先算再跑** —— 直接跑代码你学不到东西。"),
            ("code", '''import math
for size, patch, merge in [(448, 14, 1), (448, 14, 2), (1024, 14, 2), (768, 14, 2)]:
    grid = math.ceil(size / patch)
    # merge=2 时，2×2 的相邻 patch 会被合并成一个 token → 数量除 4
    n = (grid // merge) ** 2
    print(f"{size}×{size}  patch={patch} merge={merge}  grid={grid}  N={n}  "
          f"（占 2048 序列的 {n/2048:.0%}）")
print("\\n→ 注意最后那列：1024² 的图会吃掉序列的一大半，这就是「视觉 token 经济」")'''),
            ("md", "## 2. 跑手写 ViT，逐层看 shape"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.minivlm.vision"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", """## 3. 动手改坏它，看会发生什么

学习最快的方式是**故意写错**，然后观察症状。把下面几处各改一次（改完记得改回来）："""),
            ("code", '''# 这段是"实验设计"，不是要你运行它
experiments = [
    ("把 scale 从 head_dim**-0.5 改成 d_model**-0.5",
     "能跑，不报错 —— 但注意力分布会被压平，效果变差。这就是它难查的原因"),
    ("把 Pre-LN 改成 Post-LN",
     "浅层看不出问题，堆到 27 层 loss 会震荡或直接 NaN"),
    ("去掉位置编码",
     "shape 完全一样，模型却失去空间概念 —— 打乱 patch 顺序结果不变"),
    ("把 patch 从 14 改成 16",
     "448/16=28 → N=784。视觉 token 少了，但和预训练权重不再匹配"),
]
for change, symptom in experiments:
    print(f"改：{change}")
    print(f"  → {symptom}\\n")
print("挑一个真的动手改一遍。看着代码在 shape 完全正确的情况下变差，是会记住的")'''),
            ("md", "## 4. 打卡"),
            ("code", '''print("""今日打卡
─────────────────────────────────────────
[学到] N 的算法是 ______；Pre-LN 和 Post-LN 的差别是 ______
[产出] src/minivlm/vision.py 通过自检
[卡住] ______
─────────────────────────────────────────""")'''),
        ],
        next_day_hint="`days/day-03.md` —— 连接器：为什么 LLaVA 用最笨的 MLP 反而赢了",
    ),

    # ======================================================================
    D(
        week=1, n=3, slug="connector_compare", title="连接器：模态对齐的关键那一层",
        where="cloud",
        files="`src/minivlm/connector.py`、`docs/03-connector.md`",
        prereq="Day 2 完成（手写 ViT 能跑通）",
        goal="实现三种连接器（单层 MLP / 两层 MLP+GELU / Perceiver Resampler），"
             "对比它们的参数量和输出 token 数，并回答「为什么 LLaVA 用最笨的 MLP 反而效果好」。",
        read=[
            "`docs/03-connector.md` —— MLP / Q-Former / Perceiver / Cross-Attn 四种方案的对比表",
            "LLaVA 论文的 architecture 部分（只看图 1 + 连接器那一段）",
            "BLIP-2 论文里 Q-Former 的设计动机（**重点看它为什么要固定数量的 query**）",
        ],
        think=[
            "连接器要解决的**根本矛盾**是什么？（提示：视觉 token 太多，语言模型序列装不下）",
            "Perceiver Resampler 用固定数量的 query，好处是什么？代价是什么？",
            "**关键一问**：LLaVA 用一层 MLP 就打赢了 Q-Former，这说明「对齐」这件事"
            "到底难在哪、不难在哪？",
            "如果连接器只做线性投影，为什么还需要非线性（GELU）？",
        ],
        write_title="`src/minivlm/connector.py`（三种连接器，参数化可切换）",
        write_rows=[
            ("`MLPConnector`", "单层线性投影 `D_vision → D_llm`。最笨，但是 LLaVA 的基线"),
            ("`TwoLayerMLPConnector`", "加一层 GELU 隐层。**LLaVA-1.5 用的就是它**，"
                                       "比单层明显好 —— 这个结论反直觉，值得记"),
            ("`PerceiverResampler`", "固定 K 个可学习 query + 交叉注意力。"
                                     "把 N 个视觉 token 压成 K 个，**这是 Flamingo 的方案**"),
            ("`count_params()`", "统计每种连接器的参数量 —— 你会发现它相对 LLM 小得可怜"),
            ("`compare_table()`", "输出的对比表：参数量 / 输出 token 数 / 是否有信息压缩"),
        ],
        write_note="> 连接器的参数量通常只占整个模型的 **0.1%–2%**，但它决定了"
                   "「视觉信息以什么形式进入语言模型」。\n"
                   "> 参数量小 ≠ 不重要。这是这一天的核心认知。",
        run=[
            ("python -m src.minivlm.connector", "自检 + 三种连接器对比表"),
            ("python -m src.minivlm.connector --n 1024 --k 64",
             "模拟真实规模：1024 个视觉 token 压到 64 个"),
        ],
        expect="""
$ python -m src.minivlm.connector
                    参数量        输入token   输出token   信息压缩
MLP                 21.2 M         1024        1024       无
2-layer MLP+GELU    22.3 M         1024        1024       无
Perceiver (K=64)    26.9 M         1024          64       16×

d_model: vision 1152 → llm 2048

结论：三者参数量同一量级（20–27 M），差的是**压缩能力**。
      LLaVA 选 2-layer MLP：不压缩，但靠更强的 LLM 吃下全部 token。
      Flamingo 选 Perceiver：必须压缩，因为要吃几十张图 + 视频帧。
""",
        accept=[
            "能画出三种连接器的结构图，并说清 query 的数量各自是多少",
            "能回答「LLaVA 为什么用 MLP 反而好」——**答案和 LLM 的强度有关**",
            "能说出 Perceiver 的 K 调大调小分别会怎样（K 太小丢信息，K 太大失去压缩意义）",
            "知道连接器参数量占比很小，但决定信息以什么形式进 LLM",
        ],
        pits=[
            "**以为连接器是「翻译官」** —— 它不是把视觉特征翻译成语言，"
            "它是把视觉特征**投影到 LLM 能读的维度**。真正「对齐语义」是 SFT 阶段做的事。",
            "**Perceiver 的 query 初始化** —— 全零初始化会让注意力均匀分布，"
            "导致训练初期几乎不更新。用 `randn * 0.02`。",
            "**忘了 mask** —— Cross-Attention 里视觉侧不需要 mask，"
            "但如果加了 causal mask 会丢掉一半视觉信息，且不会有任何报错。",
            "**参数量算错** —— bias 别漏。`nn.Linear(1152, 2048)` 是 "
            "`1152*2048 + 2048 = 2.36M`，不是 `1152*2048`。",
        ],
        nb=[
            ("md", "## 1. 三种连接器并排看"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.minivlm.connector"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)'''),
            ("md", """## 2. 手算 MLP 连接器的参数量

**算一遍你就不会忘了。** 视觉侧 D=1152，LLM 侧 D=2048。"""),
            ("code", '''D_v, D_l, D_h = 1152, 2048, 2048   # D_h 是 2-layer 的隐层宽

mlp1  = D_v * D_l + D_l
mlp2  = (D_v * D_h + D_h) + (D_h * D_l + D_l)
print(f"单层 MLP  : {mlp1/1e6:.2f} M")
print(f"两层 MLP  : {mlp2/1e6:.2f} M")

# 相对 3B 的 LLM 主干
print(f"\\n占 3B 模型的比例: {mlp2/3e9:.3%}")
print("→ 连接器确实很小，但它决定了视觉信息『以什么形式』进入 LLM")'''),
            ("md", """## 3. token 数 vs 压缩率

Perceiver 的 K 是**唯一的压缩旋钮**。把它拉大拉小，看列表怎么变。"""),
            ("code", '''N = 1024
print(f"{'K':>6} {'输出token':>10} {'压缩率':>8}  {'适用场景':<28}")
print("-" * 60)
for K, note in [(16, "丢信息过多，细节问题必错"),
                (64, "Flamingo 的默认，多图/视频"),
                (256, "较保守，接近不压缩"),
                (1024, "= 不压缩，那不如直接用 MLP")]:
    print(f"{K:>6} {K:>10} {N/K:>7.0f}×  {note:<28}")

print("\\n→ 注意 K=1024 那一行：既然不压缩，为什么要多一层注意力？"
      "\\n  这就是 LLaVA 选 MLP 的立场 —— 压缩交给 LLM 自己去做")'''),
            ("md", "## 4. 打卡"),
            ("code", '''print("""今日打卡
─────────────────────────────────────────
[学到] 连接器的作用是 ______；LLaVA 选 MLP 的原因是 ______
[产出] src/minivlm/connector.py 三种实现 + 对比表
[卡住] ______
─────────────────────────────────────────""")'''),
        ],
        next_day_hint="`days/day-04.md` —— 精读 Qwen2.5-VL：M-RoPE 和 2×2 patch merging",
    ),

    # ======================================================================
    D(
        week=1, n=4, slug="visual_token_budget", title="Qwen2.5-VL 架构精读",
        where="cloud",
        files="`src/minivlm/processor.py`、`docs/04-qwen25vl.md`",
        prereq="Day 2–3 完成（ViT 和连接器都手写过了）",
        goal="复现 Qwen2.5-VL 的「按原图比例切 patch、算 visual token 数」逻辑，"
             "**算出的数字必须等于官方 processor 的 `<|image_pad|>` 数量**。"
             "这一条通过，说明你真的搞懂了它的分辨率处理。",
        read=[
            "`docs/04-qwen25vl.md` —— 四个重点：native dynamic resolution、M-RoPE、"
            "window attention、视频 3D conv",
            "`src/minivlm/processor.py` 源码，特别是 `smart_resize()` 的**对齐取整**",
            "Qwen2.5-VL 技术报告里 M-RoPE 那一节（**只看位置编码怎么分解的图**）",
        ],
        think=[
            "`smart_resize` 为什么要求尺寸能被 `patch_size × merge_size` 整除？"
            "不能整除会发生什么？",
            "M-RoPE 把位置编码拆成 t/h/w 三个维度 —— 一张静态图片的 t 是多少？"
            "视频呢？",
            "window attention 和全局 attention 混合，是为了省什么？",
            "**动手前先猜**：1024×768 的图，Qwen2.5-VL 会产生多少个 visual token？",
        ],
        write_title="`src/minivlm/processor.py`（核心是对齐取整 + token 数计算）",
        write_rows=[
            ("`smart_resize()`", "把图片尺寸缩到能被 `patch*merge=28` 整除。"
                                 "**取整方向要用 round 不是 floor**，否则会丢一行/一列"),
            ("`assign_bucket()`", "把任意尺寸归到最近的 bucket —— 这是 Qwen2-VL 的老做法，"
                                  "2.5 已经改成动态分辨率了，这里留作对比"),
            ("`count_visual_tokens()`", "`merge` 之后的 token 数 = `(H/28) × (W/28)`"),
            ("`build_image_tokens()`", "生成 `<|vision_start|><|image_pad|>×N<|vision_end|>` 文本"),
            ("`check_against_official()`", "**Day 4 的验收函数**：和官方 processor 的 "
                                           "`<|image_pad|>` 数量逐个对齐"),
            ("`--check`", "CLI 入口，跑一组尺寸的完整校验并打印对照表"),
        ],
        write_note="> 视觉 token 数**算错一位**，整个训练都会出问题：\n"
                   "> `<|image_pad|>` 的数量必须等于实际注入的视觉 embedding 数，\n"
                   "> 否则 `model.py` 里替换 embedding 时形状对不上，直接崩。\n"
                   "> 所以这个函数值得你反复验证。",
        run=[
            ("python -m src.minivlm.processor --check",
             "**Day 4 验收**：一组尺寸下，我们算的 token 数 == 官方数字"),
            ("python -m src.minivlm.processor --table",
             "打印「图片尺寸 → visual token 数」对照表"),
        ],
        expect="""
$ python -m src.minivlm.processor --check
尺寸            缩放后        我们算N   官方N    一致
────────────────────────────────────────────────
 224×224      224×224         64       64      ✓
 448×448      448×448        256      256      ✓
1024×1024    1036×1036       1369     1369      ✓
1024×768     1036×756         999      999      ✓
 800×1200    812×1204        1044     1044      ✓

5/5 通过 —— 你算对了 Qwen2.5-VL 的分辨率处理

$ python -m src.minivlm.processor --table
   尺寸      缩放后      N      占 2048 序列
  448×448   448×448    256        12%
 1024×1024 1036×1036  1369        67%   ← 一张图吃掉三分之二
""",
        accept=[
            "**`make day4` 通过** —— 我们算的 token 数等于官方 `<|image_pad|>` 数",
            "能画图解释 M-RoPE 的 t/h/w 三维分别编码什么",
            "能说清「1024² 图占 67% 序列」对训练意味着什么（→ batch 只能很小）",
            "**W1 最难的一环啃下来了**：能白板画出完整 VLM 数据流并讲清每一环"
            "（M1 的正式确认在 Day 6 的复盘日）",
        ],
        pits=[
            "**取整方向错** —— `smart_resize` 必须 round 到最近的 28 倍数。"
            "用 floor 会少一行 patch，token 数差几十个，然后 `model.py` 里 embedding "
            "替换时形状不匹配 —— 而且报错信息完全指不到这里。",
            "**忘了 merge 是 2×2** —— `(H/14) × (W/14)` 是 patch 数，"
            "merge 之后要除以 4。少除这一步 token 数会差 4 倍。",
            "**以为 2.5 还和 2.0 一样用固定 bucket** —— 2.5 换成了 native dynamic。"
            "`assign_bucket` 保留下来只是给你对比用的，主路径不走它。",
            "**视频的 t 维度** —— 静态图 `t=0`，视频按帧数递增。别把静态图的 t 也当成 1 "
            "然后去算，位置 id 会整体偏移。",
        ],
        nb=[
            ("md", "## 1. 先猜，再算\n\n**1024×768 的图，会产生多少个 visual token？** 把猜测写下来。"),
            ("code", '''guesses = {"448×448": None, "1024×1024": None, "1024×768": None}
print("把你猜的数字填进上面的字典，然后往下跑对比。\\n")

import math
PATCH, MERGE = 14, 2
for size in [(448, 448), (1024, 1024), (1024, 768), (800, 1200)]:
    h, w = size
    # 对齐到 patch*merge = 28 的倍数（round，不是 floor）
    h2 = round(h / (PATCH * MERGE)) * (PATCH * MERGE)
    w2 = round(w / (PATCH * MERGE)) * (PATCH * MERGE)
    n = (h2 // PATCH // MERGE) * (w2 // PATCH // MERGE)
    print(f"{h}x{w}  缩放后 {h2}x{w2}  N={n:<6} 占2048序列 {n/2048:.0%}")'''),
            ("md", "## 2. 官方校验（需要下载 Qwen2.5-VL 的 processor）"),
            ("code", '''import subprocess, sys
r = subprocess.run([sys.executable, "-m", "src.minivlm.processor", "--check"],
                   capture_output=True, text=True, cwd="..")
print(r.stdout or r.stderr)
print("\\n→ 全部 ✓ 就说明你搞懂了；有 ✗ 就回去看 smart_resize 的取整")'''),
            ("md", """## 3. 视觉 token 经济账

这一步是**连接「今天」和「第 13 天的显存工程」**的关键。"""),
            ("code", '''print(f"{'图片尺寸':<12} {'visual token':>12} {'占2048':>8} {'单图激活(bf16,估)':>18}")
print("-" * 56)
for size in [(224,224), (448,448), (768,768), (1024,1024)]:
    h, w = size
    n = (h // 28) * (w // 28)
    # 粗略估算：每层激活 ~ n * d_model * 2 bytes（只算一层，实际要乘层数）
    act_mb = n * 2048 * 2 / 1024**2
    print(f"{h}×{w:<8} {n:>12} {n/2048:>7.0%} {act_mb:>15.1f} MB")

print("\\n→ 一张 1024² 的图 = 1369 个 token，比很多人默认的「256 个」多 5 倍。")
print("  这是 VLM 训练显存爆炸的头号原因，Day 13 会算这笔账。")'''),
            ("md", "## 4. M1 白板自检\n\n拿张纸画出：**图 → 缩放 → patch → ViT → merge → 连接器 → 拼进文本序列 → LLM → 回答**。\n\n每一环标注：张量形状 + 在哪一步 token 数从多少变成多少。\n\n画不出来就回 Day 2/3 重看 —— 这是 W1 唯一的硬骨头。\n\n> 今天先自己画一遍。**Day 6 会正式复盘并确认 M1** —— 那时候你要能不看任何资料讲出来。"),
            ("code", '''print("""今日打卡
─────────────────────────────────────────
[学到] M-RoPE 的 t/h/w 分别编码 ______；1024²图占序列 ______%
[产出] processor.py --check 全绿；白板数据流草图（拍照存 assets/）
[卡住] ______
─────────────────────────────────────────
W1 还差 Day 5（拼 Mini-VLM）+ Day 6（复盘确认 M1），明天继续""")'''),
        ],
        next_day_hint="`days/day-05.md` —— 从零手搭 Mini-VLM：把前四天的零件拼起来",
    ),
]
