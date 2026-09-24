# =============================================================================
# Dockerfile —— 应用侧镜像（Shopify App / Agent API）
#
# ⚠️ 这个镜像不含 torch / 训练栈，所以很小（~200 MB）。
#    模型推理走远端：要么云上的 vLLM 服务，要么商用 API。
#    训练镜像不存在 —— 训练在云 GPU 上直接跑（平台镜像 + cloud_bootstrap.sh），
#    容器化训练是套娃（宿主机已有 CUDA driver + 平台镜像已含 PyTorch）。
#
# 构建：  docker build -t mmlab-app .
# 运行：  docker compose up -d  （会用 compose 里的配置）
# =============================================================================

FROM python:3.11-slim

# libpq 是 psycopg 需要的系统库；curl 留给健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 层缓存：requirements 不变就不重装）
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# 再拷代码（代码改动不会触发依赖重装）
COPY src/ ./src/
COPY scripts/ ./scripts/

# 非 root 运行（Shopify 应用商店审核也看这个）
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8001

# 健康检查打到容器内的 /health（src/serve/api.py 定义了）
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

# uvicorn 起 FastAPI 应用。--factory 表示 create_app() 返回 app 实例
# （不是模块级变量），这样 import 时不会真的执行任何东西，惰性安全。
CMD ["uvicorn", "src.serve.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]