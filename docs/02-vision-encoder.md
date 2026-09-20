# 02 · 视觉编码器：ViT → CLIP → SigLIP

> 对应 Day 2

## 概念

### ViT：把图像切成词

Transformer 只能吃序列，所以第一步是把图像「序列化」。

```
原图 224×224×3
   ↓ 切成 16×16 的 patch
14 × 14 = 196 个 patch，每个 patch 拉平成 16×16×3 = 768 维向量
   ↓ 线性投影（等价于一个 Conv2d, kernel=16, stride=16）
196 × 768
   ↓ 加可学习的 [CLS] token + 位置编码
197 × 768
   ↓ 12 层 Transformer Encoder（每层：LayerNorm → MHA → 残差 → LayerNorm → MLP → 残差）
   ↓ 取 [CLS] 位置输出（或全部 patch 输出）
```

**关键设计决策**：

1. **Patch size 决定 token 数**。224/16 = 196 个 token；224/14 = 256 个。patch 越小越细但越贵，是 O(n²)。
2. **位置编码是必需的**。ViT 的注意力对 patch 顺序不敏感（置换不变），不加位置编码，把图打乱重排输出不变。这显然不对。
3. **不需要卷积的归纳偏置**。ViT 去掉了 CNN 的平移不变性等先验，所以**需要更多数据**才能追上 ResNet。这是 ViT 论文的核心发现：小数据上 ViT 输，大数据上 ViT 赢。

> 对 VLM 的启示：视觉编码器通常是在几亿到几十亿图文对上预训练出来的，你不可能重训。所以**后训练阶段一律直接加载官方权重**。

### CLIP：用对比学习把图像和文本拉到一起

CLIP 的训练目标极简：

```
一个 batch 里有 N 个 (图, 文) 对
   ↓
图像塔 → N 个向量；文本塔 → N 个向量（各自 L2 归一化）
   ↓
算 N×N 的余弦相似度矩阵 S
   ↓
对角线是正样本（配对的图-文），其余是 N²-N 个负样本
   ↓
两个方向的交叉熵：行方向（图找文）+ 列方向（文找图）
   loss = (CE(S, labels) + CE(S.T, labels)) / 2
```

这就是 **InfoNCE / 对称对比损失**。训练完得到两个塔，可以做零样本分类（把类名写成 prompt "a photo of a {class}"，看哪个和图片最像）、检索。

**CLIP 对 VLM 的价值**：它的图像塔输出的特征已经**和语言语义对齐**了。这就是为什么把它接上 MLP 就能直接喂给 LLM——特征空间本来就离语言不远。

**CLIP 的局限**：它优化的是「整图 vs 整句」的相似度，**没有空间和细粒度信息**。CLIP 的特征里，一个像素级别的瑕疵基本消失了。所以纯 CLIP 塔的 VLM 在 OCR、小目标、细节判断上很弱。

### SigLIP：把 softmax 换掉

CLIP 的 InfoNCE 是全局归一化的——每个样本的 loss 依赖整个 batch 的所有负样本。这带来两个问题：batch 必须很大（CLIP 用了 32768），而且跨设备同步梯度很麻烦。

SigLIP 换成 **sigmoid 损失**：

```
CLIP:    loss_ij = -log( exp(S_ij) / Σ_k exp(S_ik) )     ← 全局 softmax
SigLIP:  loss_ij = -log( σ( S_ij · z_ij ) )              ← 逐对 sigmoid
                        其中 z_ij = +1 (正样本) / -1 (负样本)
```

好处：
- **不需要全局归一化** → batch 可以小，设备间不用全量通信
- 训练更稳、更容易扩展到更大规模
- 同等规模下效果更好（尤其小 batch）

**Qwen2.5-VL 用的是 SigLIP 风格的 ViT，但做了大改造**（见 `04-qwen25vl.md`）。

### 动态分辨率：为什么它改变了游戏规则

固定分辨率（如 336×336）的问题：

```
一张 1080×1920 的详情页长图
   ↓ resize 到 336×336
   → 压扁成 336×336，文字全部糊掉，完全读不了
```

现代 VLM 的解法有三种：

| 方案 | 做法 | 代表 | 代价 |
|---|---|---|---|
| **AnyRes 切图** | 保持比例切成多个 336 tile + 一个全局缩略图 | LLaVA-NeXT, InternVL | 实现复杂，tile 数不定 |
| **NaViT / 原生动态** | 不切图，直接按原图比例生成变长 patch 序列，用 sequence packing 组 batch | Qwen2-VL 起 | 变长序列，batch 效率低 |
| **固定 + 高分辨率** | 直接用 448 或 512 | 早期模型 | 极端比例还是不行 |

Qwen2.5-VL 走的是第二种，并且做了三个关键改进（下一节详述）。

## 工程细节

### 用 timm 加载 SigLIP，打印每一步的 shape

```python
import torch, timm

model = timm.create_model(
    "vit_so400m_patch14_siglip_384",
    pretrained=True,
    num_classes=0,          # 去掉分类头，只要特征
    img_size=384,
)
model.eval()

x = torch.randn(2, 3, 384, 384)
with torch.no_grad():
    out = model.forward_features(x)     # 不走 pool_head
print(out.shape)
# 期望: (2, 729, 1152)
#   729 = (384/14)² + 1，含 [CLS]
#   1152 = 隐藏维度
```

**验证清单**（Day 2 的代码任务）：

- `729 = 27×27 + 1`，`27 = 384 // 14` → 理解 patch 数和分辨率的关系
- 把 `img_size` 改成 448，token 数应该变成 `(448//14)² + 1 = 1025`
- 打印 `model.patch_embed` 的类型，确认是 `Conv2d(3, 1152, kernel_size=14, stride=14)`

### 几种图像塔的对比

| 编码器 | 参数 | 训练目标 | 适合场景 |
|---|---|---|---|
| CLIP ViT-L/14-336 | 304M | InfoNCE | 通用，学术基线 |
| SigLIP-SO400M/14-384 | 878M | Sigmoid | 通用 + 小 batch 友好 |
| EVA-CLIP / EVA-02 | 1B | 蒸馏 + 对比 | 高质量视觉特征 |
| InternViT-6B | 5.9B | 对比 + 自蒸馏 | 细粒度、OCR |
| Qwen2.5-VL ViT | ~675M | SigLIP 式 + 动态 | 中文、任意比例、视频 |

### 图像预处理：归一化那些事

```python
# SigLIP 的标准化参数（注意 Qwen2.5-VL 用的是 CLIP 的 mean/std）
mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)

# CLIP 的
mean = (0.48145466, 0.4578275, 0.40821073)
std  = (0.26862954, 0.26130258, 0.27577711)
```

**踩坑**：用错 mean/std，模型不会报错，但效果会诡异下降（尤其对颜色敏感的任务，比如「这是米白色还是纯白色」——这在服装客服里是高频问题）。**预处理参数必须和预训练时一致。**

### 商品图特有的预处理问题

电商图片比学术数据集脏得多，Day 7 会专门写代码处理这五类：

1. **EXIF 旋转**：手机拍的图带 orientation 标记，不处理会导致图躺着。`PIL.ImageOps.exif_transpose()`
2. **透明通道**：PNG 带 alpha，直接转 RGB 会把透明区域变黑。要先合成白底。
3. **CMYK / 16bit**：印刷来源的图。`.convert("RGB")` 统一。
4. **超长图**：详情页长图宽高比可能 1:20，直接送进去 token 数爆掉。要分段或降采样。
5. **水印和拼接图**：一张图上拼了 4 个颜色款式，模型容易只看一个。这类样本要在数据构造时特殊标注。

## 自检问题

1. ViT 把 448×448 的图切成 14×14 的 patch，得到多少个 token？
2. 为什么 ViT 必须加位置编码？不加会怎样？
3. 用一句话说清 CLIP 的对称对比损失在做什么。
4. SigLIP 相比 CLIP 的核心改动是什么？带来什么好处？
5. 一张 1080×1920 的长图，如果用 AnyRes 方案会怎么处理？用 Qwen2.5-VL 的原生方案又会怎么处理？
6. 为什么电商图片的预处理比学术数据集麻烦？举出至少三类问题。

---

**上一篇**：[01-architecture.md](01-architecture.md) · **下一篇**：[03-connector.md](03-connector.md)
