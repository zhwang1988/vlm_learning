# Day 7 · 图像预处理全链路（W2 开工）

> 预计 3–4h ｜ 📓 `notebooks/day-07_image_preprocess.ipynb` ｜ 💻 `src/data/image_utils.py`
> 本日起**全部可在本地 Mac 跑**（只依赖 numpy/Pillow），不用开 GPU 机。

## 今日目标（一句话）

搞清一张「脏图」进模型前的完整处理链：EXIF 纠正 → 透明底 → resize → 归一化 → pHash，
并理解**为什么 Qwen2.5-VL 不能粗暴 resize 到 448×448**。

## 一、读（60 min）

- `docs/05-data-engineering.md` 第 1–3 节（脏数据五类 + smart resize + 去重原理）
- `docs/02-vision-encoder.md` 回看「动态分辨率」一小节（和今天呼应）

思考题：
1. 一张 300×3000 的长图（裙子详情页），直接 resize 到 448×448 会发生什么？
2. 为什么 pHash 用 DCT 低频而不是像素均值？
3. CMYK 的商品图不转会怎样？（提示：颜色错 + 某些库直接抛异常）

## 二、写（90 min）

`src/data/image_utils.py`（已给实现，你的任务是**逐个函数跑通并改造一处**）：

| 函数 | 作用 | 你的动作 |
|---|---|---|
| `load_and_normalize` | EXIF→mode→alpha→比例 的**固定顺序** | 打乱顺序试一次，看哪个先坏 |
| `phash` | 64-bit 指纹（DCT 8×8） | 换一张图加高斯噪声，验证指纹不变 |
| `hamming_distance` | 汉明距离 | 手算一个例子对答案 |
| `find_duplicate_groups` | union-find 分组 | 造 3 张相同图验证分组 |
| `detect_collage` | 拼接图检测 | 上网找一张拼接图试试 |

## 三、跑（本地）

```bash
python -m src.data.image_utils       # 自检：透明底、pHash 鲁棒性测试
jupyter lab                          # notebooks/day-07_image_preprocess.ipynb
```

期望：自检输出两行 ✓（alpha→白底 转换正确；pHash 对噪声/缩放鲁棒）。
notebook 里对你自己找的 5 张电商图输出尺寸/比例/token 数表格。

## 四、验收清单

- [ ] 能解释「为什么不能简单粗暴 resize 到 448×448」（长宽比失真 + 小物体消失）
- [ ] 对 200 张商品图输出尺寸/宽高比/visual token 分布（notebook 里）
- [ ] pHash 汉明距离阈值：你能说出 `≤ 8` 视为重复的理由

## 五、容易踩的坑

- PIL 的 `Image.open` 是惰性的 —— EXIF 转正必须 `.transpose(...)` 或 `ImageOps.exif_transpose`
- RGBA 转 RGB 直接 `.convert("RGB")` 会把透明区变黑 —— 要先贴白底
- pHash 对**文字水印**敏感，电商图常有 —— 这是 Day 10 去重要处理的事

## 六、打卡

三行照旧。明天第一件事固定：打开 `docs/05` 第 4 节 + `src/data/taxonomy.py`。
