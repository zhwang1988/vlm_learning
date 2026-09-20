#!/usr/bin/env python3
"""
下载 Qwen2.5-VL 模型权重。

支持两个后端（自动选最快的）:
  - ModelScope（国内推荐，速度通常 50–100 MB/s）
  - HuggingFace（用 hf-mirror.com 镜像，避开网络问题）

用法:
  # 最常用：下 3B Instruct（够 W3/W5 用，约 6.2 GB）
  python scripts/download_model.py --model 3b-instruct

  # 同时下 7B Instruct 备用（约 14 GB）
  python scripts/download_model.py --model 7b-instruct

  # 一次性下多个
  python scripts/download_model.py --model 3b-instruct 7b-instruct

  # 换后端
  python scripts/download_model.py --model 3b-instruct --backend modelscope
  python scripts/download_model.py --model 3b-instruct --backend huggingface

  # 验完整性（只看校验和，不重下）
  python scripts/download_model.py --verify --model 3b-instruct

  # 看磁盘用了多少
  python scripts/download_model.py --status

放在数据盘：默认 /root/autodl-tmp/models/，可通过 --root 改。

设计取舍：
  - 默认放数据盘（autodl-tmp），不是系统盘
  - allow_patterns 显式列出 safetensors/json/txt/model/py，**不下载 .bin**（已弃用且占空间）
  - chat_template.json 单独确保拿到（Qwen2.5-VL 的对话模板在这文件里）
  - 下载完会算总大小和文件清单，方便你判断要不要删 7B 省空间
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 支持的模型清单（id 别打错）
# ---------------------------------------------------------------------------

MODELS = {
    # —— 主训练目标 ——
    "3b-instruct": {
        "ms": "qwen/Qwen2.5-VL-3B-Instruct",
        "hf": "Qwen/Qwen2.5-VL-3B-Instruct",
        "size_gb": 6.2,
        "when": "W3 SFT / W5 DPO 的主训练目标。客服场景 3B 够用。",
    },
    "7b-instruct": {
        "ms": "qwen/Qwen2.5-VL-7B-Instruct",
        "hf": "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_gb": 14.5,
        "when": "对比实验用。W5 做对齐可以顺便训一版 7B 看效果差距。",
    },
    # —— 基座版（无 Instruct 对齐），用于对比评测 ——
    "3b-base": {
        "ms": "qwen/Qwen2.5-VL-3B",
        "hf": "Qwen/Qwen2.5-VL-3B",
        "size_gb": 6.2,
        "when": "W4 评测时作为 baseline 对比，证明 SFT/DPO 真的有提升。",
    },
    "7b-base": {
        "ms": "qwen/Qwen2.5-VL-7B",
        "hf": "Qwen/Qwen2.5-VL-7B",
        "size_gb": 14.5,
        "when": "7B 基线。如果你只跑 3B，这个可以不下载。",
    },
}

# 要下载的文件类型。.bin 已弃用；.gguf 是量化版（我们用 AWQ，不用这个）；
# *.tiktoken 是 tokenizer；*.py 是 modeling/processing 代码（Qwen2.5-VL 必须本地有 modeling 文件）。
PATTERNS = [
    "*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken",
    "*.py", "*.md",
    "merges.txt", "vocab.json", "tokenizer*",
]

# 必须存在的文件（缺失就说明下载不完整）
REQUIRED = [
    "config.json",
    "preprocessor_config.json",
    "chat_template.json",
]


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def default_root() -> Path:
    """数据盘第一挂的优先；都没有就放 ~/.cache/models"""
    for cand in ("/root/autodl-tmp", "/data", "/mnt/data"):
        if Path(cand).is_dir():
            return Path(cand) / "models"
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home.parent / "models"


def resolve_root(arg_root: Path | None) -> Path:
    """
    决定实际缓存目录的最终位置。
    规则：
      1. 显式 --root 且父目录可创建 → 用 --root
      2. 显式 --root 但父目录建不出（Mac 上常见）→ 警告，降级到 ~/.cache/models
      3. 没显式 → 走 default_root()
    """
    if arg_root is not None:
        parent = arg_root.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            arg_root.mkdir(parents=True, exist_ok=True)
            return arg_root
        except (PermissionError, OSError) as e:
            print(f"  ⚠ --root={arg_root} 不可写 ({e.__class__.__name__})")
            print(f"     自动降级到 {Path.home() / '.cache' / 'models'}")
    fallback = Path.home() / ".cache" / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def ms_cache_dir() -> Path:
    return Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))


# ---------------------------------------------------------------------------
# 后端：ModelScope
# ---------------------------------------------------------------------------

def download_modelscope(repo_id: str, root: Path) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("    ModelScope 未装，装一下…")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "modelscope"])
        from modelscope import snapshot_download

    print(f"    源: ModelScope  repo={repo_id}")
    print(f"    到: {root}")
    # ModelScope 默认放 ~/.cache/modelscope，我们显式 cache_dir 到数据盘
    target = root / "modelscope" / repo_id.split("/")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id,
        cache_dir=str(root / "modelscope"),
        allow_patterns=PATTERNS,
    )
    return Path(path)


# ---------------------------------------------------------------------------
# 后端：HuggingFace（用 hf-mirror 镜像避开网络）
# ---------------------------------------------------------------------------

def download_huggingface(repo_id: str, root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    # 镜像：hf-mirror.com 是 huggingface.co 的国内镜像
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = endpoint
    print(f"    源: HuggingFace (via {endpoint})  repo={repo_id}")
    print(f"    到: {root / 'huggingface'}")
    target = root / "huggingface" / repo_id.split("/")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id,
        cache_dir=str(root / "huggingface"),
        allow_patterns=PATTERNS,
        etag_timeout=30,
    )
    return Path(path)


# ---------------------------------------------------------------------------
# 工具：找已下载的目录
# ---------------------------------------------------------------------------

def find_local(root: Path, key: str) -> Path | None:
    """在本机缓存里找 model key 对应的目录。"""
    info = MODELS[key]
    # ModelScope 默认命名是 "{org}__{name}"
    ms_name = info["ms"].replace("/", "__")
    for base in (root / "modelscope", ms_cache_dir()):
        cand = base / ms_name
        if cand.exists():
            return cand
    # HuggingFace
    hf_name = info["hf"].split("/")[-1]
    for base in (root / "huggingface",
                 Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))):
        cand = base / hub_layout(hf_name)
        if cand.exists():
            return cand
    return None


def hub_layout(name: str) -> Path:
    """HF 的 snapshot_download 默认放在 models--{org}--{name}/snapshots/{hash}/"""
    return Path(f"models--{name.split('--')[0]}--{'--'.join(name.split('--')[1:])}").parent if False else Path(name)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def verify(path: Path, key: str) -> tuple[bool, list[str]]:
    """检查必需文件 + 估算大小。"""
    missing = [r for r in REQUIRED if not (path / r).exists()]
    if missing:
        return False, [f"缺文件: {m}" for m in missing]

    # 算 safetensors 大小
    safes = list(path.glob("*.safetensors"))
    total = sum(s.stat().st_size for s in safes) / 1024 ** 3
    expected = MODELS[key]["size_gb"]
    # 容忍 ±15%（不同精度版本略不同）
    if not (expected * 0.85 <= total <= expected * 1.15):
        return False, [f"大小异常: 实测 {total:.2f} GB, 期望 ~{expected} GB"]
    return True, [f"safetensors: {len(safes)} 个, 共 {total:.2f} GB"]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def cmd_status(root: Path):
    print(f"缓存根目录: {root}")
    print(f"磁盘可用  : {shutil.disk_usage(root).free / 1024**3:.1f} GB")
    print()
    print(f"{'模型':<14} {'本地路径':<60} {'大小':>8} {'状态'}")
    print("-" * 96)
    for key in MODELS:
        path = find_local(root, key)
        if path is None:
            print(f"{key:<14} {'(未下载)':<60} {'-':>8}  ❌")
            continue
        ok, _ = verify(path, key)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3
        print(f"{key:<14} {str(path)[-58:]:<60} {size:6.2f} GB  {'✓' if ok else '⚠ 不完整'}")


def cmd_verify(root: Path, keys: Iterable[str]):
    any_bad = False
    for key in keys:
        path = find_local(root, key)
        if path is None:
            print(f"  {key:<14} ❌ 本地不存在")
            any_bad = True
            continue
        ok, msgs = verify(path, key)
        flag = "✓" if ok else "⚠"
        print(f"  {key:<14} {flag} {path}")
        for m in msgs:
            print(f"      {m}")
        if not ok:
            any_bad = True
    return 0 if not any_bad else 1


def cmd_download(root: Path, keys: list[str], backend: str, force: bool):
    free = shutil.disk_usage(root).free / 1024**3
    need = sum(MODELS[k]["size_gb"] for k in keys)
    if need > free:
        print(f"❌ 磁盘空间不够：要 {need:.1f} GB，可用 {free:.1f} GB")
        print(f"   目标: {root}")
        print(f"   解决：删旧模型 / 扩数据盘 / 选更小的 key")
        return 1

    # 自动选后端
    if backend == "auto":
        # 国内基本都通 ModelScope；HF 在某些云上反而被 block
        try:
            import modelscope  # noqa: F401
            backend = "modelscope"
        except ImportError:
            backend = "huggingface"
        print(f"  自动选后端: {backend}")

    print("=" * 70)
    print(f"  下载 Qwen2.5-VL 模型到 {root}")
    print(f"  后端: {backend}    磁盘: {free:.1f} GB 可用, 需 {need:.1f} GB")
    print("=" * 70)

    rc = 0
    for key in keys:
        info = MODELS[key]
        print(f"\n[{key}] {info['hf']}")
        print(f"  用途: {info['when']}")

        if not force:
            existing = find_local(root, key)
            if existing:
                ok, msgs = verify(existing, key)
                if ok:
                    print(f"  ✓ 已存在且完整，跳过: {existing}")
                    continue

        repo_id = info["ms" if backend == "modelscope" else "hf"]
        t0 = time.time()
        try:
            if backend == "modelscope":
                path = download_modelscope(repo_id, root)
            else:
                path = download_huggingface(repo_id, root)
        except Exception as e:
            print(f"  ❌ 失败: {type(e).__name__}: {e}")
            print(f"   建议：换后端重试  python scripts/download_model.py --model {key} --backend huggingface")
            rc = 1
            continue

        elapsed = time.time() - t0
        ok, msgs = verify(path, key)
        flag = "✓" if ok else "⚠"
        print(f"  {flag} 完成: {path}")
        print(f"    耗时 {elapsed:.0f}s")
        for m in msgs:
            print(f"    {m}")
        if not ok:
            rc = 1

    print()
    print("=" * 70)
    if rc == 0:
        print(f"  全部下载完成。下一步:")
        print(f"    jupyter lab   # 打开 notebooks/01_first_vlm_inference.ipynb")
    else:
        print(f"  有失败项，见上面 ⚠ 标记。可重跑本脚本（断点续传）。")
    print("=" * 70)
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="下载 Qwen2.5-VL 模型权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --model 3b-instruct
  %(prog)s --model 3b-instruct 7b-instruct
  %(prog)s --verify --model 3b-instruct
  %(prog)s --status
  %(prog)s --model 3b-instruct --backend huggingface
        """)
    ap.add_argument("--model", nargs="+", choices=list(MODELS.keys()),
                    help="要下载的模型 key（可多个）")
    ap.add_argument("--backend", choices=["auto", "modelscope", "huggingface"],
                    default="auto", help="下载后端（默认 auto，会先试 ModelScope）")
    ap.add_argument("--root", type=Path, default=None,
                    help="缓存根目录（默认数据盘 /root/autodl-tmp/models/）")
    ap.add_argument("--force", action="store_true", help="即使存在也重下")
    ap.add_argument("--verify", action="store_true", help="只校验，不下载")
    ap.add_argument("--status", action="store_true", help="列出已下载模型")
    args = ap.parse_args()

    root = resolve_root(args.root)

    if args.status:
        cmd_status(root)
        return 0
    if args.verify:
        keys = args.model or list(MODELS.keys())
        return cmd_verify(root, keys)
    if args.model:
        return cmd_download(root, args.model, args.backend, args.force)

    # 没指定参数：给个引导
    ap.print_help()
    print()
    print("=" * 70)
    print("  你没指定 --model。最常见的 3 种场景:")
    print("=" * 70)
    print()
    print("  场景 A: 我就要训一个能用的客服模型")
    print("    python scripts/download_model.py --model 3b-instruct")
    print()
    print("  场景 B: 我想把 3B 和 7B 都训一遍对比")
    print("    python scripts/download_model.py --model 3b-instruct 7b-instruct")
    print()
    print("  场景 C: 我已经有模型了，想知道本地状态")
    print("    python scripts/download_model.py --status")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)