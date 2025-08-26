# Qwen3-4B-Thinking 模型服务 Docker 镜像
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# 安装 uv
RUN pip install uv

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY configs/ ./configs/

# 使用 uv 安装依赖
RUN uv sync --frozen

# 复制模型服务启动脚本
COPY scripts/start_service.sh ./
RUN chmod +x start_service.sh

# 创建模型和日志目录
RUN mkdir -p /app/models /app/logs

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["./start_service.sh"]