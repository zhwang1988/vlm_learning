"""
商品向量索引同步。

对应讲义 docs/11-shopify.md「Webhook」，对应计划 Day 40。

流程：
  商品变更 → webhook → 下载商品图 → CLIP 编码 → 写入索引 → Agent 能搜到

验收标准：改一个商品标题，30 秒内 RAG 索引同步。
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Optional

from .client import to_internal_product


INDEX_DIR = Path("data/shop_index")


def _shop_dir(shop: str) -> Path:
    d = INDEX_DIR / shop.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(shop: str) -> dict:
    p = _shop_dir(shop) / "products.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(shop: str, data: dict):
    p = _shop_dir(shop) / "products.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 下载商品图（离线，避免每次搜索都请求 Shopify CDN）
# ---------------------------------------------------------------------------


async def download_images(urls: list[str], out_dir: Path,
                          max_per_product: int = 5) -> list[str]:
    """下载商品图到本地。

    为什么要本地化：
      - CLIP 编码需要图片文件
      - 每次搜索都从 CDN 拉图太慢
      - 商品下架后 CDN 链接会失效，本地留一份更稳
    """
    import httpx

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        for url in urls[:max_per_product]:
            name = url.split("?")[0].rsplit("/", 1)[-1]
            if not name or "." not in name:
                name = f"{abs(hash(url)) % 10**10}.jpg"
            p = out_dir / name
            if p.exists():
                saved.append(str(p))
                continue
            try:
                r = await c.get(url)
                r.raise_for_status()
                p.write_bytes(r.content)
                saved.append(str(p))
            except Exception as e:      # noqa: BLE001
                print(f"    ✗ 下载失败 {name}: {str(e)[:80]}")
    return saved


# ---------------------------------------------------------------------------
# 索引同步
# ---------------------------------------------------------------------------


async def rebuild_index(shop: str, api, force: bool = False) -> dict:
    """全量重建索引。首次安装时调用。"""
    t0 = time.time()
    print(f"[index] 开始重建 {shop} 的索引...")

    products = await api.all_products()
    print(f"[index] 拉取到 {len(products)} 个商品")

    out_dir = _shop_dir(shop) / "images"
    records = []

    for i, sp in enumerate(products):
        internal = to_internal_product(sp)
        saved = await download_images(internal["image_urls"], out_dir)
        internal["local_images"] = saved
        internal["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        records.append(internal)
        if (i + 1) % 20 == 0:
            print(f"[index]   {i+1}/{len(products)}")

    _save(shop, {r["id"]: r for r in records})

    # 编码
    try:
        from ..agent.retriever import ClipEncoder, IndexEntry, VectorIndex
        from PIL import Image

        enc = ClipEncoder()
        idx = VectorIndex(enc.dim)
        for r in records:
            for img in r.get("local_images", []):
                try:
                    im = Image.open(img).convert("RGB")
                    v = enc.encode_images([im])[0]
                    idx.add(IndexEntry(
                        id=r["id"],
                        text=r["title"],
                        image_path=img,
                        payload={"title": r["title"], "category": r["category"],
                                 "variants": r["variants"], "tags": r["tags"]},
                    ), v)
                except Exception:
                    continue
        idx.save(_shop_dir(shop) / "image_index.pkl")
        n_vec = len(idx)
    except Exception as e:      # noqa: BLE001
        print(f"[index] ⚠ 编码失败（索引数据已保存，稍后重试编码）: {str(e)[:120]}")
        n_vec = 0

    elapsed = time.time() - t0
    print(f"[index] ✓ 完成: {len(records)} 商品 / {n_vec} 向量，耗时 {elapsed:.1f}s")
    return {"n_products": len(records), "n_vectors": n_vec, "elapsed_s": round(elapsed, 1)}


async def update_product_index(shop: str, payload: dict):
    """增量更新单个商品（webhook 触发）。

    验收：改一个商品标题，30 秒内生效。
    """
    t0 = time.time()
    pid = str(payload.get("id", "")).split("/")[-1]
    if not pid:
        return

    internal = to_internal_product(payload)
    out_dir = _shop_dir(shop) / "images"
    internal["local_images"] = await download_images(internal["image_urls"], out_dir)
    internal["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    data = _load(shop)
    data[pid] = internal
    _save(shop, data)

    # 增量更新向量：这里简化成重建（商品量小时够用）
    # 生产环境应该用支持 upsert 的向量库（pgvector / Qdrant）
    await _reencode(shop)

    print(f"[index] ✓ 商品 {pid} 已更新（{time.time() - t0:.1f}s）")


async def _reencode(shop: str):
    try:
        from ..agent.retriever import ClipEncoder, IndexEntry, VectorIndex
        from PIL import Image

        data = _load(shop)
        enc = ClipEncoder()
        idx = VectorIndex(enc.dim)
        for r in data.values():
            for img in r.get("local_images", []):
                try:
                    im = Image.open(img).convert("RGB")
                    idx.add(IndexEntry(id=r["id"], text=r["title"], image_path=img,
                                       payload={"title": r["title"],
                                                "category": r["category"],
                                                "variants": r["variants"]}), 
                            enc.encode_images([im])[0])
                except Exception:
                    continue
        idx.save(_shop_dir(shop) / "image_index.pkl")
        # 通知 retriever 重新加载
        from ..agent import retriever as R
        R._DEFAULT = None
    except Exception as e:      # noqa: BLE001
        print(f"[index] 编码失败: {str(e)[:100]}")


def delete_product_index(shop: str, product_id: str):
    pid = str(product_id).split("/")[-1]
    data = _load(shop)
    if pid in data:
        del data[pid]
        _save(shop, data)
        print(f"[index] ✓ 商品 {pid} 已从索引移除")


def drop_shop_index(shop: str):
    """卸载时清理。GDPR 要求。"""
    import shutil
    d = _shop_dir(shop)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"[index] ✓ 已清理 {shop} 的索引")


def get_shop_products(shop: str) -> list[dict]:
    return list(_load(shop).values())


def index_stats(shop: str) -> dict:
    data = _load(shop)
    d = _shop_dir(shop)
    idx = d / "image_index.pkl"
    n_vec = 0
    if idx.exists():
        try:
            with open(idx, "rb") as f:
                n_vec = len(pickle.load(f)["entries"])
        except Exception:
            pass
    return {
        "shop": shop,
        "n_products": len(data),
        "n_vectors": n_vec,
        "has_index": idx.exists(),
        "images_dir": str(d / "images"),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="商品索引同步")
    ap.add_argument("--shop", help="店铺域名")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not args.shop:
        print("用法：")
        print("  python -m src.shopify.indexer --shop your-store.myshopify.com --rebuild")
        print("  python -m src.shopify.indexer --shop your-store.myshopify.com --stats")
        print()
        print("设计要点（Day 40）：")
        print("  ① 商品图必须本地化 —— CDN 链接会失效，且每次拉太慢")
        print("  ② webhook 增量更新，不是定时全量轮询")
        print("  ③ 卸载时清理（GDPR 要求）")
        print("  ④ 验收标准：改一个商品标题，30 秒内 RAG 索引同步")
        raise SystemExit(0)

    if args.stats:
        print(json.dumps(index_stats(args.shop), ensure_ascii=False, indent=2))
    elif args.rebuild:
        import asyncio
        from .client import ShopAPI, get_client
        api = ShopAPI(get_client(args.shop))
        asyncio.run(rebuild_index(args.shop, api))
