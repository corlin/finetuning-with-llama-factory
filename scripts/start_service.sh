#!/bin/bash

# Qwen3-4B-Thinking 模型服务启动脚本

set -e

echo "=== Qwen3-4B-Thinking 模型服务启动 ==="

# 检查环境变量
MODEL_PATH=${MODEL_PATH:-"/app/models/qwen3-4b-thinking"}
QUANTIZATION_FORMAT=${QUANTIZATION_FORMAT:-"int8"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
MAX_WORKERS=${MAX_WORKERS:-1}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}

echo "配置信息:"
echo "  模型路径: $MODEL_PATH"
echo "  量化格式: $QUANTIZATION_FORMAT"
echo "  日志级别: $LOG_LEVEL"
echo "  工作进程: $MAX_WORKERS"
echo "  监听地址: $HOST:$PORT"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU进行推理"
fi

# 检查模型文件
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo "请确保模型文件已正确挂载到容器中"
fi

# 创建日志目录
mkdir -p /app/logs

# 设置Python路径
export PYTHONPATH="/app:$PYTHONPATH"

# 启动服务
echo "启动模型服务..."

if [ "$MAX_WORKERS" -eq 1 ]; then
    # 单进程模式
    exec uv run python -m uvicorn src.model_service:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --loop uvloop
else
    # 多进程模式
    exec uv run python -m gunicorn src.model_service:app \
        -w "$MAX_WORKERS" \
        -k uvicorn.workers.UvicornWorker \
        --bind "$HOST:$PORT" \
        --log-level "$LOG_LEVEL" \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --timeout 300 \
        --keep-alive 2 \
        --max-requests 1000 \
        --max-requests-jitter 100
fi