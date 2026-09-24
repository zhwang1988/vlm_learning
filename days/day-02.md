# Day 2 · 视觉编码器：ViT → SigLIP

> 预计 3–4h ｜ 📓 `notebooks/day-02_vit_from_scratch.ipynb` ｜ ☁️ 云 GPU · `src/minivlm/vision.py`、`docs/02-vision-encoder.md`
> 前置：Day 1 完成（模型已下载，云环境可用）

## 今日目标（一句话）

亲手写出 ViT 的 patch embedding + attention block，跑通 `(B,3,448,448)` → `(B,N,D)`，并说清 N 是怎么算出来的、SigLIP 比 CLIP 改了什么。

## 一、读（60 min）

材料：
- `docs/02-vision-encoder.md` 前半 —— 重点是 patch 切分、位置编码、CLS token 的去留
- ViT 原文的 key idea（只看 figure 1 + method 前三段）
- CLIP 的 InfoNCE loss 推导 —— 搞懂「对比学习」到底在优化什么

思考题（先自己想，答案在讲义或代码注释里）：
1. `(3,448,448)` 的图，patch=14、无 CLS token，N 是多少？**先手算再跑代码**
2. ViT 为什么必须有位置编码？如果去掉，模型会失去什么能力？
3. CLIP 用 InfoNCE 对比 loss，SigLIP 换成了 sigmoid loss —— 换掉之后为什么小 batch 也能训得住？
4. **本项目的关键一问**：Qwen2.5-VL 为什么最后选了 SigLIP 而不是 CLIP？

## 二、写（100 min）

`src/minivlm/vision.py`（手写，这是本周最核心的代码）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `PatchEmbed` | 用 Conv2d 实现 patch 切分（kernel=stride=patch_size），不要用 unfold —— 卷积版本更快也更短 |
| `MultiHeadSelfAttention` | 标准 MHA。注意 `scale = head_dim ** -0.5`，别写成 `sqrt(d_model)`，那是常见错误 |
| `ViTBlock` | LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差。**Pre-LN 的顺序不能改**，改了训练会崩 |
| `VisionEncoder` | 把 patch + 位置编码 + 若干 block 串起来，返回 `(B,N,D)` |
| `shape_report()` | 逐层打印 shape —— 这个函数是给你调试用的，别删 |
| `load_siglip()` | 加载 timm 的 SigLIP 权重，与手写版对比输出形状 |

> 手写 ViT 的价值不在「能跑」，而在**你会知道每个数字从哪来**。
> 后面读 Qwen2.5-VL 的 `2×2 patch merging` 时，你会立刻明白「merge 把 N 除以 4、把 D 乘以 4」，因为它就是你今天写的这几行。

## 三、跑（在云 GPU 上）

```bash
# 自检：手写 ViT 前向 + 逐层 shape
python -m src.minivlm.vision
# 和 timm 的 SigLIP 对比输出形状与参数量
python -m src.minivlm.vision --compare-siglip
```

期望输出（节选）：
```
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
```

## 四、验收清单

- [ ] 能**手算**任意尺寸图片的 N（例如 1024×768、patch=14、merge=2 → 答案见讲义）
- [ ] `shape_report()` 打印的每一层形状都能解释清楚
- [ ] 能说出 SigLIP 与 CLIP 的两点差异（loss 形式 / 是否用 softmax 归一化）
- [ ] 能回答「为什么 Qwen2.5-VL 用 SigLIP 而不是 CLIP」

## 五、容易踩的坑

1. **scale 写错** —— 是 `head_dim ** -0.5`，不是 `d_model ** -0.5`。这个错误不会报错，只是效果变差，最难查。
2. **忘了把 patch 展平前的通道顺序搞对** —— Conv2d 输出的 `(B,C,H,W)` 要 `flatten(2).transpose(1,2)` 才变成 `(B,N,C)`，顺序错了 shape 也对不上语义。
3. **Pre-LN 写成 Post-LN** —— 现代 ViT 全用 Pre-LN。写错的话深层网络训不动。
4. **timm 权重加载时 num_classes 对不上** —— 视觉编码器要的是特征不是分类头，记得 `num_classes=0`。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
