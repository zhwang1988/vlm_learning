"""
多模态 RAG：以图搜图 + 文本知识检索 + 混合重排。

对应讲义 docs/10-agent.md「多模态 RAG」，对应计划 Day 32。

三个索引协同：
  索引 1 · 商品图库    CLIP image embedding → 以图搜图（用户上传图 → 定位商品）
  索引 2 · 知识库      文本 embedding       → 政策/FAQ/尺码表检索
  索引 3 · 商品属性    结构化过滤          → 类目/价格/库存

**CLIP 的局限**：它匹配的是「整体语义」，对「同一款不同颜色」这种细粒度
差异不敏感。所以粗排交给 CLIP，**精排必须靠 VLM**。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 向量索引（简易实现，生产建议换 faiss / pgvector）
# ---------------------------------------------------------------------------


@dataclass
class IndexEntry:
    id: str
    text: str = ""
    image_path: str = ""
    payload: dict = field(default_factory=dict)


class VectorIndex:
    """最小的向量索引。数据量 <10 万时 numpy 暴力检索足够（毫秒级）。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.entries: list[IndexEntry] = []
        self.vectors: Optional[np.ndarray] = None

    def add(self, entry: IndexEntry, vector: np.ndarray):
        v = vector / (np.linalg.norm(vector) + 1e-9)
        self.entries.append(entry)
        if self.vectors is None:
            self.vectors = v.reshape(1, -1).astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, v.reshape(1, -1)])

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[IndexEntry, float]]:
        if self.vectors is None or len(self.entries) == 0:
            return []
        q = query / (np.linalg.norm(query) + 1e-9)
        sims = self.vectors @ q.astype(np.float32)
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.entries[i], float(sims[i])) for i in idx]

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"entries": self.entries, "vectors": self.vectors,
                         "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls(d["dim"])
        idx.entries = d["entries"]
        idx.vectors = d["vectors"]
        return idx

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------


class ClipEncoder:
    """CLIP 双塔编码器。文本和图像映射到同一空间，所以能做跨模态检索。

    用 CLIP 的原因：它是唯一开箱即用的、图文共享空间的模型。
    （SigLIP 也可以，把类名换掉即可。）
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim

    def encode_images(self, images: list) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.processor(images=images, return_tensors="pt",
                                 padding=True).to(self.device)
            feats = self.model.get_image_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.processor(text=texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=77).to(self.device)
            feats = self.model.get_text_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


class TextEncoder:
    """纯文本检索用的 encoder（bge / m3e 等中文模型效果更好）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)


# ---------------------------------------------------------------------------
# 主检索器
# ---------------------------------------------------------------------------


class MultimodalRetriever:
    """商品图库 + 知识库双索引检索器。"""

    def __init__(self, clip: Optional[ClipEncoder] = None,
                 text_encoder: Optional[TextEncoder] = None):
        self.clip = clip or ClipEncoder()
        self.text_encoder = text_encoder
        self.image_index = VectorIndex(self.clip.dim)
        self.text_index: Optional[VectorIndex] = None

    # --- 构建 ---

    def build_image_index(self, products: list[dict]):
        """products: [{"id":..., "title":..., "image_path":..., "attrs": {...}}, ...]"""
        from PIL import Image

        imgs, entries = [], []
        for p in products:
            path = p.get("image_path") or p.get("image")
            if not path or not Path(path).exists():
                continue
            try:
                imgs.append(Image.open(path).convert("RGB"))
                entries.append(IndexEntry(
                    id=p.get("id", str(len(entries))),
                    text=p.get("title", ""),
                    image_path=path,
                    payload={k: v for k, v in p.items()
                             if k not in ("image_path", "image")},
                ))
            except Exception:
                continue

        if not imgs:
            print("⚠ 没有可用图片，图索引为空")
            return

        vecs = self.clip.encode_images(imgs)
        for e, v in zip(entries, vecs):
            self.image_index.add(e, v)
        print(f"✓ 图索引: {len(self.image_index)} 个商品")

    def build_text_index(self, docs: list[dict]):
        """docs: [{"id":..., "text":..., "source":...}, ...]"""
        if self.text_encoder is None:
            self.text_encoder = TextEncoder()
        if self.text_index is None:
            self.text_index = VectorIndex(self.text_encoder.dim)

        texts = [d["text"] for d in docs]
        vecs = self.text_encoder.encode(texts)
        for d, v in zip(docs, vecs):
            self.text_index.add(IndexEntry(
                id=d.get("id", ""), text=d["text"], payload=d), v)
        print(f"✓ 文本索引: {len(self.text_index)} 条知识")

    # --- 检索 ---

    def search_by_image(self, image, top_k: int = 10,
                        category_filter: Optional[str] = None
                        ) -> list[dict]:
        """以图搜图。"""
        vec = self.clip.encode_images([image])[0]
        raw = self.image_index.search(vec, top_k * 3 if category_filter else top_k)

        results = []
        for entry, score in raw:
            if category_filter and entry.payload.get("category") != category_filter:
                continue
            results.append({
                "product_id": entry.id,
                "title": entry.text,
                "image_path": entry.image_path,
                "score": round(score, 4),
                "attrs": entry.payload,
            })
            if len(results) >= top_k:
                break
        return results

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.3) -> list[str]:
        """文本知识检索。返回知识片段列表。"""
        if self.text_index is None or len(self.text_index) == 0:
            return []
        if self.text_encoder is None:
            return []
        vec = self.text_encoder.encode([query])[0]
        raw = self.text_index.search(vec, top_k)
        return [e.text for e, s in raw if s >= min_score]

    def hybrid_search(self, query: str, image=None, top_k: int = 3) -> list[dict]:
        """混合检索：图 + 文都用上。

        客服场景的典型用法：
          用户发了商品图 + 问「这个有 M 码吗」
          → 以图搜图定位商品
          → 文本检索查该商品的尺码信息
          → 结构化过滤查库存
        """
        out = []
        if image is not None:
            hits = self.search_by_image(image, top_k=top_k)
            for h in hits:
                out.append({"type": "product_image", **h})
        if query:
            for t in self.search(query, top_k=top_k):
                out.append({"type": "knowledge", "text": t})
        return out

    # --- 持久化 ---

    def save(self, dir_path: str | Path):
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.image_index.save(d / "image_index.pkl")
        if self.text_index is not None:
            self.text_index.save(d / "text_index.pkl")
        print(f"✓ 索引已保存到 {d}")

    @classmethod
    def load(cls, dir_path: str | Path) -> "MultimodalRetriever":
        d = Path(dir_path)
        ret = cls.__new__(cls)
        ret.clip = ClipEncoder()
        ret.text_encoder = None
        ret.image_index = VectorIndex.load(d / "image_index.pkl")
        if (d / "text_index.pkl").exists():
            ret.text_index = VectorIndex.load(d / "text_index.pkl")
            ret.text_encoder = TextEncoder()
        else:
            ret.text_index = None
        return ret


# ---------------------------------------------------------------------------
# VLM 精排（解决 CLIP 细粒度不足的问题）
# ---------------------------------------------------------------------------


async def vlm_rerank(candidates: list[dict], query: str,
                     image, top_k: int = 3) -> list[dict]:
    """用 VLM 对 CLIP 粗排结果精排。

    为什么必须精排：同一款不同颜色在 CLIP 空间里几乎重合，
    粗排分不出。VLM 看得到颜色、图案、细节，能分。

    成本考虑：只对 top-10 精排，不是全库。所以延迟可控。
    """
    import base64
    import os

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=os.getenv("SYNTH_API_BASE"),
                         api_key=os.getenv("SYNTH_API_KEY"))
    model = os.getenv("SYNTH_MODEL", "qwen-vl-max")

    # 把用户图和候选图拼成一次多图请求
    content = [{"type": "text",
                "text": f"用户在找以下哪个商品？用户的问题：「{query}」\n"
                        f"第一张图是用户上传的，后面依次是候选项。\n"
                        f"只输出最匹配的候选序号（1-{len(candidates)}），"
                        f"不匹配就输出 0。"}]

    def to_url(p):
        b = base64.b64encode(Path(p).read_bytes()).decode()
        suf = Path(p).suffix.lstrip(".").lower()
        mime = "image/jpeg" if suf in ("jpg", "jpeg") else f"image/{suf}"
        return f"data:{mime};base64,{b}"

    try:
        content.insert(0, {"type": "image_url", "image_url": {"url": to_url(_tmp_save(image))}})
    except Exception:
        pass

    for c in candidates:
        try:
            content.append({"type": "image_url",
                            "image_url": {"url": to_url(c["image_path"])}})
        except Exception:
            content.append({"type": "text", "text": c.get("title", "")})

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0, timeout=60,
        )
        raw = resp.choices[0].message.content.strip()
        import re
        m = re.search(r"\d+", raw)
        pick = int(m.group()) if m else 0
        if 1 <= pick <= len(candidates):
            best = candidates[pick - 1]
            rest = [c for i, c in enumerate(candidates) if i != pick - 1]
            return [best] + rest[:top_k - 1]
    except Exception as e:      # noqa: BLE001
        print(f"  ⚠ VLM 精排失败: {e}")

    return candidates[:top_k]


def _tmp_save(image) -> str:
    """把 PIL 图临时存盘（API 需要路径）。"""
    import tempfile
    p = Path(tempfile.gettempdir()) / f"rerank_q_{abs(hash(image.tobytes())) % 10**8}.jpg"
    if not p.exists():
        image.convert("RGB").save(p, quality=90)
    return str(p)


# ---------------------------------------------------------------------------
# 默认实例（供 tools.py 调用）
# ---------------------------------------------------------------------------

_DEFAULT: Optional[MultimodalRetriever] = None


def get_default_retriever() -> Optional[MultimodalRetriever]:
    global _DEFAULT
    if _DEFAULT is None:
        p = Path("data/index")
        if (p / "image_index.pkl").exists():
            try:
                _DEFAULT = MultimodalRetriever.load(p)
            except Exception:
                _DEFAULT = None
    return _DEFAULT


def set_default_retriever(r: MultimodalRetriever):
    global _DEFAULT
    _DEFAULT = r


# ---------------------------------------------------------------------------
# CLI：构建索引
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="多模态 RAG 索引")
    ap.add_argument("--products", help="商品 JSON：id/title/image_path/category")
    ap.add_argument("--knowledge", help="知识 JSON：id/text")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--demo", action="store_true", help="跑自检")
    args = ap.parse_args()

    if args.demo:
        print("=" * 76)
        print("多模态 RAG 自检")
        print("=" * 76)
        enc = ClipEncoder()
        print(f"\nCLIP 维度: {enc.dim}")

        # 造两张差别很大的图
        from PIL import Image, ImageDraw
        red = Image.new("RGB", (300, 300), (200, 40, 40))
        blue = Image.new("RGB", (300, 300), (40, 60, 200))
        d = ImageDraw.Draw(red)
        d.rectangle([100, 100, 200, 200], fill=(255, 255, 255))
        d2 = ImageDraw.Draw(blue)
        d2.ellipse([80, 80, 220, 220], fill=(255, 255, 0))

        idx = VectorIndex(enc.dim)
        idx.add(IndexEntry("red", "一件红色上衣"), enc.encode_images([red])[0])
        idx.add(IndexEntry("blue", "一件蓝色上衣"), enc.encode_images([blue])[0])

        q = enc.encode_images([red])[0]
        hits = idx.search(q, 2)
        print("\n以图搜图（查询=红图）:")
        for e, s in hits:
            print(f"  {e.id:<8} {e.text:<16} 相似度 {s:.4f}")

        # 跨模态：文搜图
        tq = enc.encode_texts(["一条蓝色的衣服"])[0]
        hits2 = idx.search(tq, 2)
        print("\n以文搜图（查询='一条蓝色的衣服'）:")
        for e, s in hits2:
            print(f"  {e.id:<8} {e.text:<16} 相似度 {s:.4f}")

        print("\n⚠️ 注意 CLIP 的局限：")
        print("   它在同款不同色上的区分度很低（相似度可能都在 0.95+）")
        print("   → 所以生产环境必须加 VLM 精排（见 vlm_rerank）")
        print("\nDay 32 的验收标准：")
        print("   在 100 张商品图上测以图搜图 top-3 准确率 ≥ 70%")
    else:
        ret = MultimodalRetriever()
        if args.products:
            products = json.loads(Path(args.products).read_text(encoding="utf-8"))
            ret.build_image_index(products)
        if args.knowledge:
            docs = json.loads(Path(args.knowledge).read_text(encoding="utf-8"))
            ret.build_text_index(docs)
        if args.products or args.knowledge:
            ret.save(args.out)
        else:
            ap.print_help()
