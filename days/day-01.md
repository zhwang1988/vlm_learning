# Day 1 · 技术版图与问题定义

> 预计 3–4h ｜ 📓 `notebooks/day-01_environment_and_first_inference.ipynb` ｜ 💻/☁️ 两可 · `docs/00-orientation.md`、`scripts/download_model.py`、`notebooks/01_first_vlm_inference.ipynb`
> 前置：无 —— 这是第 1 天。先把 `requirements-core.txt` 装好

## 今日目标（一句话）

建立一张属于自己的「多模态技术树」：说清三条技术路线（对比学习式 / 融合式 / 外挂式）各自把图塞进了哪一层，并在云上让 Qwen2.5-VL 真的描述一张商品图。

## 一、读（60 min）

材料：
- `docs/00-orientation.md` —— **先读这个**，它是 8 周的整体地图，读完你会知道每天在干嘛
- `docs/12-papers.md` 的「第一梯队」部分 + CLIP / Flamingo / BLIP-2 的**摘要页**。注意：今天只看摘要，**不要通读论文**（通读会耗掉一整天，而且 8 周里用不上细节）
- `docs/01-architecture.md` 第 1–2 节：三件套（视觉编码器 / 连接器 / LLM 主干）各自管什么

思考题（先自己想，答案在讲义或代码注释里）：
1. 三条路线分别把图像信息塞进了哪一层？—— 对比学习塞进**向量空间**，融合式塞进**注意力层**，外挂式塞进**LLM 的输入序列**
2. 客服要「看图回答问题」，为什么「纯文本 LLM + OCR」这条路走不通？（提示：OCR 只给文字，丢掉了颜色/瑕疵/版型这些**非文字信息**）
3. 为什么 2024 年之后大家都收敛到「外挂式 + 强 LLM」这条路线？

## 二、写（100 min）

环境体检 → 下载模型 → 第一次真实推理

| 函数 / 文件 | 你要做什么 |
|---|---|
| `make check` | 本地环境体检。**torch/CUDA 标黄是正常的** —— 本地不训练，只用来读文档改代码 |
| `scripts/download_model.py` | 已给实现，你只需要会用它。优先走 ModelScope（国内 2–5 分钟），带断点续传和完整性校验 |
| `notebooks/01_first_vlm_inference.ipynb` | 让 Qwen2.5-VL 描述一张商品图，看它**实际**说了什么 —— 这是你第一次看到模型的真实水平 |
| `assets/tech-tree.png` | 手绘「多模态技术树」拍照存进来。手绘比复制别人的图有用得多 |
| `progress/daily-log.md` | 加三行打卡（今天最容易的一步，也是最容易忘的一步） |

> 今天的**真正产出不是代码，是心里有地图**。
> 代码你只是在「用」，亲手写的部分从 Day 2 才开始。别急着写模型。

## 三、跑（本地或云上都行）

```bash
# 本地体检（torch 缺失只标黄，退出码 0）
make check
# 先看要下载多少、下到哪 —— 别直接开下，容易下到系统盘塞满
python scripts/download_model.py --status
# 云上执行。约 6.2 GB，走 ModelScope 2–5 分钟
python scripts/download_model.py --model 3b-instruct
# 云上体检。这里 torch/CUDA/显存缺失都是**硬伤**（退出码 1）
make check-cloud
# 跑通第一次推理
jupyter lab notebooks/01_first_vlm_inference.ipynb
```

期望输出（节选）：
```
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
```

## 四、验收清单

- [ ] 能**不看资料**说出三条技术路线的差异，以及为什么客服场景选外挂式
- [ ] 云上能加载 Qwen2.5-VL-3B 并让它描述一张商品图（哪怕答案很一般）
- [ ] `assets/` 里有一张自己画的技术树（手绘拍照也算）
- [ ] 知道「哪些事在本地做、哪些必须上云」—— 这一条决定了你 8 周花 ¥300 还是 ¥3000

## 五、容易踩的坑

1. **从 HF 直接下载会非常慢甚至失败**。`download_model.py` 默认走 ModelScope，不要手动改成 `huggingface_hub.snapshot_download`。
2. **不要把模型下到系统盘**。云机器系统盘通常只有 30–50 GB，下完模型就满了。统一用 `--root /root/autodl-tmp/models`（AutoDL 的数据盘）。
3. 第一次推理会发现模型答得很平庸 —— **这是正常的，也是这个项目存在的理由**。把它的原始回答抄进打卡，8 周后回来看对比。
4. `make check` 在本地会标黄 torch —— **这不是错误**，别去花两小时装 torch。本地装 torch 只会得到一个跑不动的 CPU 版，毫无用处。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
