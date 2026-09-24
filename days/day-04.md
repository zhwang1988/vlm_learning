# Day 4 · Qwen2.5-VL 架构精读

> 预计 3–4h ｜ 📓 `notebooks/day-04_visual_token_budget.ipynb` ｜ ☁️ 云 GPU · `src/minivlm/processor.py`、`docs/04-qwen25vl.md`
> 前置：Day 2–3 完成（ViT 和连接器都手写过了）

## 今日目标（一句话）

复现 Qwen2.5-VL 的「按原图比例切 patch、算 visual token 数」逻辑，**算出的数字必须等于官方 processor 的 `<|image_pad|>` 数量**。这一条通过，说明你真的搞懂了它的分辨率处理。

## 一、读（60 min）

材料：
- `docs/04-qwen25vl.md` —— 四个重点：native dynamic resolution、M-RoPE、window attention、视频 3D conv
- `src/minivlm/processor.py` 源码，特别是 `smart_resize()` 的**对齐取整**
- Qwen2.5-VL 技术报告里 M-RoPE 那一节（**只看位置编码怎么分解的图**）

思考题（先自己想，答案在讲义或代码注释里）：
1. `smart_resize` 为什么要求尺寸能被 `patch_size × merge_size` 整除？不能整除会发生什么？
2. M-RoPE 把位置编码拆成 t/h/w 三个维度 —— 一张静态图片的 t 是多少？视频呢？
3. window attention 和全局 attention 混合，是为了省什么？
4. **动手前先猜**：1024×768 的图，Qwen2.5-VL 会产生多少个 visual token？

## 二、写（100 min）

`src/minivlm/processor.py`（核心是对齐取整 + token 数计算）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `smart_resize()` | 把图片尺寸缩到能被 `patch*merge=28` 整除。**取整方向要用 round 不是 floor**，否则会丢一行/一列 |
| `assign_bucket()` | 把任意尺寸归到最近的 bucket —— 这是 Qwen2-VL 的老做法，2.5 已经改成动态分辨率了，这里留作对比 |
| `count_visual_tokens()` | `merge` 之后的 token 数 = `(H/28) × (W/28)` |
| `build_image_tokens()` | 生成 `<|vision_start|><|image_pad|>×N<|vision_end|>` 文本 |
| `check_against_official()` | **Day 4 的验收函数**：和官方 processor 的 `<|image_pad|>` 数量逐个对齐 |
| `--check` | CLI 入口，跑一组尺寸的完整校验并打印对照表 |

> 视觉 token 数**算错一位**，整个训练都会出问题：
> `<|image_pad|>` 的数量必须等于实际注入的视觉 embedding 数，
> 否则 `model.py` 里替换 embedding 时形状对不上，直接崩。
> 所以这个函数值得你反复验证。

## 三、跑（在云 GPU 上）

```bash
# **Day 4 验收**：一组尺寸下，我们算的 token 数 == 官方数字
python -m src.minivlm.processor --check
# 打印「图片尺寸 → visual token 数」对照表
python -m src.minivlm.processor --table
```

期望输出（节选）：
```
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
```

## 四、验收清单

- [ ] **`make day4` 通过** —— 我们算的 token 数等于官方 `<|image_pad|>` 数
- [ ] 能画图解释 M-RoPE 的 t/h/w 三维分别编码什么
- [ ] 能说清「1024² 图占 67% 序列」对训练意味着什么（→ batch 只能很小）
- [ ] **W1 最难的一环啃下来了**：能白板画出完整 VLM 数据流并讲清每一环（M1 的正式确认在 Day 6 的复盘日）

## 五、容易踩的坑

1. **取整方向错** —— `smart_resize` 必须 round 到最近的 28 倍数。用 floor 会少一行 patch，token 数差几十个，然后 `model.py` 里 embedding 替换时形状不匹配 —— 而且报错信息完全指不到这里。
2. **忘了 merge 是 2×2** —— `(H/14) × (W/14)` 是 patch 数，merge 之后要除以 4。少除这一步 token 数会差 4 倍。
3. **以为 2.5 还和 2.0 一样用固定 bucket** —— 2.5 换成了 native dynamic。`assign_bucket` 保留下来只是给你对比用的，主路径不走它。
4. **视频的 t 维度** —— 静态图 `t=0`，视频按帧数递增。别把静态图的 t 也当成 1 然后去算，位置 id 会整体偏移。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
