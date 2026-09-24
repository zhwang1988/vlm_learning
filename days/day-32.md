# Day 32 · 多模态 RAG

> 预计 3–4h ｜ 📓 `notebooks/day-32_multimodal_rag.ipynb` ｜ ☁️ 云 GPU · `src/agent/retriever.py`
> 前置：Day 31；一批商品图

## 今日目标（一句话）

搭商品图文双索引：**CLIP 向量做图搜**（用户上传图 → 找到对应商品）+ 文本向量做知识检索（退货政策、尺码表），并实现混合检索 + VLM 重排。

## 一、读（60 min）

材料：
- `docs/10-agent.md` 第 4 节（多模态 RAG 的三种架构）
- `src/agent/retriever.py` 的 `vlm_rerank()` —— 粗排 + 细排的两段式

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么「以图搜图」比「先让 VLM 描述再搜文本」更适合商品场景？
2. CLIP 粗排 100 条 → VLM 细排 top 5，这个两段式为什么比直接 VLM 排 100 条好？
3. 向量归一化漏了会怎样？（内积 ≠ 余弦相似度）

## 二、写（100 min）

`src/agent/retriever.py`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `ClipEncoder` | 图片 → 归一化向量（**归一化不能漏**） |
| `build_image_index()` | 把商品图灌进向量索引 |
| `build_text_index()` | 把知识文档灌进文本索引 |
| `search_by_image()` | 用户传图 → top-k 商品 |
| `hybrid_search()` | 图搜 + 文本搜融合（加权或 RRF） |
| `vlm_rerank()` | CLIP 粗排 top-50 → VLM 细排 top-5（精度主要来自这一步） |

## 三、跑（在云 GPU 上）

```bash
# 自检：造几个商品跑通全链路
python -m src.agent.retriever --demo
# 灌正式索引（没有数据文件就用 --demo 的输出）
python -m src.agent.retriever --products data/products.json --knowledge data/knowledge.json --out data/index
# 确认索引落盘
ls -la data/index/
```

期望输出（节选）：
```
[clip] 载入 CLIP 编码器（约 600 MB）
[build] 商品 100 件 → 图片索引 100 条，维度 512
[build] 知识 42 条 → 文本索引 42 条
[demo] 以图搜图 top-3：
   1. 米白色针织衫 (0.91)
   2. 奶白色针织开衫 (0.87)
   3. 米色圆领毛衣 (0.83)
[rerank] VLM 重排后 top-1 = 米白色针织衫（CLIP 排第 2）← 重排把名次纠正了
```

## 四、验收清单

- [ ] 图搜 top-3 准确率 ≥ 70%（在 100 张商品图上测，要人工核对）
- [ ] 混合检索能同时吃到图片信号和文本信号
- [ ] 能解释「以图搜图」比「先描述再搜文本」好在哪（**信息损失**角度）
- [ ] 知道归一化漏了会有什么后果（相似度排序全乱）

## 五、容易踩的坑

1. **只做文本检索** —— 用户传了图你却把图扔了，等于多模态白做。
2. **向量没归一化** —— 内积相似度会被向量模长主导，长文本/大图占便宜，排序全乱。
3. 没有重排 —— CLIP 的 top-1 经常不准，加一层 VLM 重排能显著提升。
4. 索引没持久化 —— 每次请求都重新编码 100 张图，延迟爆炸。
5. 商品下架了索引没删 —— 用户搜到一个买不到的商品（Day 40 的 webhook 要解决这个）。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
