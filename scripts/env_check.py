"""
环境体检 —— 本地开发机和云 GPU 机器都用同一个脚本。

    python scripts/env_check.py                  # 自动判断环境
    python scripts/env_check.py --mode local     # 强制按本地开发机检查
    python scripts/env_check.py --mode cloud     # 强制按云训练机检查
    python scripts/env_check.py --json           # 机器可读
    python scripts/env_check.py --quiet          # 只输出一行结论

为什么一个脚本要分模式：

  本地开发机的工作是「读文档 + 改代码 + 跑纯 Python 自检」，
  没有 NVIDIA 卡是正常的，torch 也可以先不装。
  云 GPU 机器的工作是「训练 + 推理」，torch / CUDA / 显存一样不能少。

  所以同一个「没装 torch」，在本地只是提示（warn），在云上是硬伤（fail）。
  脚本自动识别：有 CUDA → 云训练机；否则 → 本地开发机。

它检查 6 组东西：
  1. Python / 平台          —— 版本够不够
  2. GPU / CUDA             —— 有没有卡、卡能不能真用、支不支持 bf16
  3. 关键依赖版本            —— transformers 够不够新（Qwen2.5-VL 的硬要求）
  4. 磁盘 / 内存             —— 空间够不够放模型和 checkpoint
  5. 镜像 / 网络             —— HF_ENDPOINT 配了没（不配下载会卡死）
  6. 项目自身                —— 目录结构、数据文件、代码能不能 import

退出码：0 = 没有必须修复项；1 = 有（可直接用于 CI / 启动脚本的门禁）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 运行环境判定
#
# 这个脚本同时跑在两种机器上，但两种机器的「及格线」完全不同：
#   本地开发机  —— 读文档、改代码、跑纯 Python 自检；没有 CUDA 是正常的
#   云 GPU 机器 —— 训练/推理；torch、CUDA、显存一样不能少
# 不区分的话，本地会输出一堆红色 fail，看着像环境坏了，其实什么都没坏。
# ---------------------------------------------------------------------------

MODE_LOCAL = "local"
MODE_CLOUD = "cloud"


def detect_mode() -> str:
    """有 NVIDIA GPU 就是云训练机，否则当本地开发机。"""
    try:
        import torch
        return MODE_CLOUD if torch.cuda.is_available() else MODE_LOCAL
    except ImportError:
        pass
    if platform.system() == "Linux" and shutil.which("nvidia-smi"):
        return MODE_CLOUD
    return MODE_LOCAL


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    status: str          # ok | warn | fail | info
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    mode: str = MODE_LOCAL

    def add(self, name, status, detail, fix=""):
        self.checks.append(Check(name, status, detail, fix))
        return self

    @property
    def n_fail(self):
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def n_warn(self):
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def is_local(self):
        return self.mode == MODE_LOCAL

    def add_soft(self, name, detail, fix=""):
        """本地模式下降级为 warn，云上算 fail。

        用途：torch/CUDA/显存这类只在训练机上才必须的东西。
        """
        self.add(name, "warn" if self.is_local else "fail", detail, fix)
        return self


ICON = {"ok": "[ok]", "warn": "[!]", "fail": "[X]", "info": "[i]"}


# ---------------------------------------------------------------------------
# 1. Python / 平台
# ---------------------------------------------------------------------------

def check_python(rep: Report):
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 9):
        rep.add("Python 版本", "fail", f"{ver} 太旧",
                "升级到 3.10+。云镜像一般自带 3.10/3.11。")
    elif v < (3, 10):
        rep.add("Python 版本", "warn", f"{ver}（建议 3.10+）",
                "多数依赖仍可用，但部分新版本 wheel 可能装不上。")
    else:
        rep.add("Python 版本", "ok", ver)

    rep.add("解释器路径", "info", sys.executable)
    rep.add("操作系统", "info", f"{platform.system()} {platform.release()} ({platform.machine()})")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        rep.add("虚拟环境", "ok", sys.prefix)
    else:
        rep.add("虚拟环境", "warn", "当前不在虚拟环境里",
                "建议激活项目的 venv，避免污染系统 Python：source <venv>/bin/activate")


# ---------------------------------------------------------------------------
# 2. GPU / CUDA
# ---------------------------------------------------------------------------

def check_gpu(rep: Report):
    try:
        import torch
    except ImportError:
        rep.add_soft("PyTorch", "未安装",
                     "本地可以先不装（不影响读文档/改代码）；"
                     "云上必须装 → pip install -r requirements-core.txt，选预装 PyTorch 的镜像可省这步")
        return

    rep.add("PyTorch", "ok", torch.__version__)
    rep.add("CUDA（torch 编译版本）", "info", str(torch.version.cuda))

    if not torch.cuda.is_available():
        rep.add_soft("GPU 可用性", "torch.cuda.is_available() == False",
                     "① 本地开发机没有 NVIDIA 卡，这是正常的，训练放到云上做；"
                     "② 云上如果也是 False：可能装了 CPU 版 torch → 重装含 cu 的版本，"
                     "或改用预装 PyTorch 的镜像；"
                     "③ 也可能你开的是『无卡模式』，此时 GPU 本来就不该可见")
        return

    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    rep.add("GPU", "ok", f"{name} x{n}  ({total:.1f} GB, sm_{cap[0]}{cap[1]})")

    # 显存档次 → 能干什么
    if total >= 70:
        tier = "7B LoRA bf16 / 3B 全参 / 多实验并行"
    elif total >= 44:
        tier = "7B LoRA bf16（压短序列）/ 3B 全参（紧）"
    elif total >= 30:
        tier = "7B QLoRA 宽松 / 3B LoRA bf16"
    elif total >= 22:
        tier = "3B QLoRA 宽松 / 7B QLoRA 压短序列 / vLLM 推理 3B"
    elif total >= 14:
        tier = "3B QLoRA 需 max_length<=1024"
    else:
        tier = "只够推理小模型，训练基本不够"
    rep.add("可支撑的训练规模", "info", tier)

    # bf16
    try:
        bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = cap[0] >= 8
    if bf16_ok:
        rep.add("bf16 支持", "ok", "配置里用 bf16: true")
    else:
        rep.add("bf16 支持", "warn", f"sm_{cap[0]}{cap[1]} 不支持 bf16（T4/V100 常见）",
                "config 里改成 bf16: false, fp16: true。"
                "用 bf16 会报 'not supported' 或直接 NaN。")

    # 真跑一次，防止「看得见卡但用不了」
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        _ = a @ b
        torch.cuda.synchronize()
        rep.add("GPU 实算测试", "ok", "矩阵乘法通过")
    except Exception as e:
        rep.add("GPU 实算测试", "fail", f"{type(e).__name__}: {e}",
                "卡不可用。重建实例，并选预装 PyTorch 的镜像。")

    # 当前占用
    try:
        used = torch.cuda.memory_allocated() / 1024 ** 3
        rep.add("当前已分配显存", "info", f"{used:.2f} GB")
    except Exception:
        pass


def check_nvidia_smi(rep: Report):
    if shutil.which("nvidia-smi") is None:
        rep.add("nvidia-smi", "info", "不存在（无卡模式下属正常）")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            rep.add("nvidia-smi", "ok", out.stdout.strip().splitlines()[0])
    except Exception as e:
        rep.add("nvidia-smi", "warn", f"调用失败: {e}")


# ---------------------------------------------------------------------------
# 3. 依赖版本
# ---------------------------------------------------------------------------

# (import 名, 最低版本, 为什么需要)
DEPS = [
    ("transformers",  "4.49.0", "Qwen2.5-VL 的硬要求，低于此版本连 processor 都构建不了"),
    ("peft",          "0.13.0", "LoRA 训练"),
    ("accelerate",    "0.34.0", "device_map / 混合精度"),
    ("torch",         "2.2.0",  "Qwen2.5-VL 的 patch 处理和 attention 需要"),
    ("datasets",      "2.21.0", "数据加载"),
    ("PIL",           "10.0.0", "图像处理（import 名是 PIL）"),
    ("numpy",         "1.26.0", "数值"),
    ("yaml",          None,     "读 configs/*.yaml（import 名是 yaml）"),
    ("safetensors",   None,     "权重加载"),
]

OPTIONAL = [
    ("bitsandbytes", "QLoRA 4bit 量化需要；CPU 上装不了，Mac 上装不了"),
    ("qwen_vl_utils", "Qwen2.5-VL 官方图像预处理工具，强烈建议"),
    ("trl",          "DPO/GRPO 训练"),
    ("scipy",        "评测里的 Spearman 相关系数"),
    ("matplotlib",   "训练曲线绘图"),
    ("open_clip",    "多模态检索（CLIP 编码器）"),
    ("gradio",       "Agent demo 界面"),
    ("fastapi",      "部署 API"),
    ("psycopg",      "Shopify App 的 PostgreSQL 连接"),
]


def _ver_tuple(s: str):
    import re
    nums = re.findall(r"\d+", str(s).split("+")[0].split(".dev")[0])
    return tuple(int(x) for x in nums[:3]) if nums else (0,)


def check_deps(rep: Report):
    for mod, minv, why in DEPS:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            rep.add_soft(f"依赖 {mod}", "未安装", f"pip install {mod}  # {why}")
            continue
        v = getattr(m, "__version__", "?")
        if minv and _ver_tuple(v) < _ver_tuple(minv):
            rep.add_soft(f"依赖 {mod}", f"{v} < {minv}",
                         f"pip install -U '{mod}>={minv}'  # {why}")
        else:
            rep.add(f"依赖 {mod}", "ok", v)

    # 可选依赖单独一组，缺了不算失败
    missing_opt = []
    for mod, why in OPTIONAL:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing_opt.append((mod, why))
    if missing_opt:
        rep.add("可选依赖缺失", "info",
                f"{len(missing_opt)} 个未装: " + ", ".join(m for m, _ in missing_opt),
                "按当前阶段需要再装，例如：" + missing_opt[0][1])


# ---------------------------------------------------------------------------
# 4. 磁盘 / 内存
# ---------------------------------------------------------------------------

def check_resources(rep: Report):
    # 内存
    try:
        if Path("/proc/meminfo").exists():
            txt = Path("/proc/meminfo").read_text()
            total_kb = int([l for l in txt.splitlines() if l.startswith("MemTotal")][0].split()[1])
            gb = total_kb / 1024 ** 2
            if gb < 15:
                rep.add("系统内存", "warn", f"{gb:.1f} GB",
                        "数据预处理（尤其是图像解码）会很慢。建议 >= 30 GB")
            else:
                rep.add("系统内存", "ok", f"{gb:.1f} GB")
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True)
            gb = int(out.stdout.strip()) / 1024 ** 3
            rep.add("系统内存", "ok", f"{gb:.1f} GB")
    except Exception:
        pass

    # 候选目录的剩余空间
    candidates = []
    env_root = os.getenv("MMLAB_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates += [Path("/root/autodl-tmp"), REPO_ROOT, Path.home()]

    seen = set()
    for p in candidates:
        try:
            if not p.exists() or str(p) in seen:
                continue
            seen.add(str(p))
            free = shutil.disk_usage(p).free / 1024 ** 3
            status = "ok"
            fix = ""
            if free < 15:
                status, fix = "fail", "空间严重不足。换大一点的数据盘，或清理旧 checkpoint。"
            elif free < 40:
                status, fix = "warn", "模型 7 GB + 数据集 + checkpoint 很容易超。建议 >= 60 GB 可用。"
            rep.add(f"磁盘剩余 {p}", status, f"{free:.1f} GB", fix)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 5. 镜像 / 网络
# ---------------------------------------------------------------------------

def check_network(rep: Report):
    ep = os.getenv("HF_ENDPOINT")
    if ep:
        rep.add("HF_ENDPOINT", "ok", ep)
    else:
        rep.add("HF_ENDPOINT", "warn", "未设置",
                "在国内不配镜像下载 HF 模型会极慢甚至超时。"
                "export HF_ENDPOINT=https://hf-mirror.com")

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        rep.add("HF_HOME", "ok", hf_home)
        p = Path(hf_home)
        if p.exists():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3
            rep.add("模型缓存大小", "info", f"{size:.2f} GB @ {hf_home}")
        else:
            rep.add("HF_HOME", "warn", f"{hf_home} 不存在",
                    "缓存会写到系统盘。建议 export HF_HOME=<数据盘>/hf-cache")
    else:
        default = Path.home() / ".cache" / "huggingface"
        rep.add("HF_HOME", "warn", f"未设置，默认 {default}",
                "云上把默认缓存放系统盘，实例删除就没了。建议 export HF_HOME=<数据盘>/hf-cache")

    # 连通性（快速，超时 5s）
    try:
        import urllib.request
        url = (ep or "https://huggingface.co") + "/"
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        rep.add("镜像连通性", "ok", f"{url} 可达")
    except Exception as e:
        rep.add("镜像连通性", "warn", f"{type(e).__name__}",
                "下载模型前先确认网络。也可以用 ModelScope 下模型再本地加载。")

    # 代理
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if os.getenv(k):
            rep.add(f"代理 {k}", "info", os.getenv(k))


# ---------------------------------------------------------------------------
# 6. 项目自身
# ---------------------------------------------------------------------------

EXPECTED = [
    "PLAN.md",
    "README.md",
    "docs/04-qwen25vl.md",
    "docs/06-sft-training.md",
    "src/minivlm/processor.py",
    "src/data/build_sft.py",
    "src/train/sft_peft.py",
    "src/eval/run_eval.py",
    "src/serve/api.py",
    "src/agent/agent.py",
    "src/shopify/app.py",
    "configs/sft_lora_3b.yaml",
]


def check_project(rep: Report):
    missing = [p for p in EXPECTED if not (REPO_ROOT / p).exists()]
    if missing:
        rep.add("项目文件完整性", "fail", f"缺 {len(missing)} 个: {missing[:4]}",
                "确认你在仓库根目录跑，并且 clone 完整。")
    else:
        rep.add("项目文件完整性", "ok", f"{len(EXPECTED)} 个关键文件都在")

    # 代码能不能 import（这才是真正的检验）
    sys.path.insert(0, str(REPO_ROOT))
    import_ok = 0
    for mod in ["src.minivlm.processor", "src.train.lora_utils", "src.eval.metrics"]:
        try:
            importlib.import_module(mod)
            rep.add(f"import {mod}", "ok", "")
            import_ok += 1
        except Exception as e:
            rep.add_soft(f"import {mod}", f"{type(e).__name__}: {e}",
                         "可能是缺依赖，或 PYTHONPATH 不对。在仓库根目录跑这个脚本。")
    if import_ok:
        rep.add_soft("代码可导入性", f"{import_ok}/3 个模块导入成功",
                     "至少一个模块能跑，说明仓库结构没问题；剩下的装完依赖就好")

    # 数据
    for rel in ["data/processed/sft_train.jsonl", "data/processed/sft_eval.jsonl"]:
        p = REPO_ROOT / rel
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            rep.add(f"数据 {rel}", "ok", f"{n} 条")
        else:
            rep.add(f"数据 {rel}", "info", "还不存在",
                    "Day 12 之前这是正常的。到时候用 src/data/synth.py 合成。")

    # 产物
    for rel in ["outputs", "reports"]:
        p = REPO_ROOT / rel
        if p.exists():
            n = len(list(p.glob("*")))
            rep.add(f"目录 {rel}/", "ok", f"{n} 项")


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def render(rep: Report) -> str:
    lines = []
    lines.append("=" * 74)
    if rep.is_local:
        lines.append("  环境体检报告  multimodal-lab      运行环境：本地开发机（无 NVIDIA GPU）")
    else:
        lines.append("  环境体检报告  multimodal-lab      运行环境：云 GPU 训练机")
    lines.append("=" * 74)
    if rep.is_local:
        lines.append("  [i]   本地定位：读文档、改代码、跑纯 Python 自检。")
        lines.append("  [i]   这里的 torch / CUDA / 显存 标黄是预期结果，不代表环境坏了；")
        lines.append("  [i]   训练与推理在云上做 —— 云上跑同一个脚本时这些项会自动变成硬性检查。")
        lines.append("-" * 74)
    for c in rep.checks:
        lines.append(f"{ICON.get(c.status, '[?]'):5s} {c.name:26s} {c.detail}")

    fixes = [c for c in rep.checks if c.fix and c.status in ("fail", "warn")]
    if fixes:
        lines.append("")
        lines.append("-" * 74)
        lines.append("  需要处理" + ("（含本地可选、云端必做）" if rep.is_local else ""))
        lines.append("-" * 74)
        for c in fixes:
            lines.append(f"  {c.name}: {c.fix}")

    lines.append("")
    lines.append("-" * 74)
    if rep.n_fail:
        lines.append(f"  结论：{rep.n_fail} 项必须修复，{rep.n_warn} 项建议处理")
        lines.append("        先修上面『需要处理』里标 fail 的，看不懂就把这段发给协作方。")
    elif rep.is_local:
        lines.append(f"  结论：本地开发机可用。{rep.n_warn} 项标黄的都是「云上才需要」的，")
        lines.append("        在本地可以忽略；要开训时把这个脚本在云机器上再跑一遍。")
    elif rep.n_warn:
        lines.append(f"  结论：可以直接开始。{rep.n_warn} 项建议但不阻塞。")
    else:
        lines.append("  结论：环境完全就绪，开干。")
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="multimodal-lab 环境体检（本地开发机 / 云 GPU 机器 都跑同一个脚本）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--quiet", action="store_true", help="只输出结论")
    ap.add_argument("--mode", choices=[MODE_LOCAL, MODE_CLOUD], default=None,
                    help="强制指定环境类型（默认自动判断：有 CUDA 就算云训练机）")
    args = ap.parse_args()

    rep = Report()
    rep.mode = args.mode or detect_mode()
    check_python(rep)
    check_nvidia_smi(rep)
    check_gpu(rep)
    check_deps(rep)
    check_resources(rep)
    check_network(rep)
    check_project(rep)

    if args.json:
        print(json.dumps({
            "mode": rep.mode,
            "checks": [asdict(c) for c in rep.checks],
            "n_fail": rep.n_fail,
            "n_warn": rep.n_warn,
        }, ensure_ascii=False, indent=2))
        return 0 if rep.n_fail == 0 else 1

    if args.quiet:
        print(f"mode={rep.mode} fail={rep.n_fail} warn={rep.n_warn}")
        return 0 if rep.n_fail == 0 else 1

    print(render(rep))
    return 0 if rep.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
