# Qwen3-4B-Thinking 模型服务部署指南

## 概述

本文档提供了 Qwen3-4B-Thinking 模型服务的完整部署指南，包括环境要求、配置说明、API使用示例和运维指南。

## 环境要求

### 硬件要求

**最低配置：**
- CPU: 8核心以上
- 内存: 16GB RAM
- 存储: 50GB 可用空间
- GPU: 可选，推荐 NVIDIA GPU (8GB+ VRAM)

**推荐配置：**
- CPU: 16核心以上
- 内存: 32GB RAM
- 存储: 100GB SSD
- GPU: NVIDIA RTX 4090 / A100 (24GB+ VRAM)

### 软件要求

- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / Windows 10+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: 2.0+ (GPU部署)
- **Python**: 3.12+ (本地部署)
- **CUDA**: 12.1+ (GPU支持)

## 部署方式

### 1. Docker 部署 (推荐)

#### 1.1 准备模型文件

```bash
# 创建模型目录
mkdir -p ./models/qwen3-4b-thinking

# 下载或复制模型文件到该目录
# 模型文件应包含：
# - config.json
# - pytorch_model.bin 或 model.safetensors
# - tokenizer.json
# - tokenizer_config.json
# - vocab.txt
```

#### 1.2 配置环境变量

创建 `.env` 文件：

```bash
# 模型配置
MODEL_PATH=/app/models/qwen3-4b-thinking
QUANTIZATION_FORMAT=int8
LOG_LEVEL=INFO

# 服务配置
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=1

# GPU配置
CUDA_VISIBLE_DEVICES=0

# 推理配置
DEFAULT_MAX_LENGTH=512
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
```

#### 1.3 启动服务

```bash
# 基础服务启动
docker-compose up -d qwen-model-service

# 包含监控的完整启动
docker-compose --profile monitoring up -d

# 包含缓存的启动
docker-compose --profile cache up -d

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f qwen-model-service
```

#### 1.4 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 模型信息
curl http://localhost:8000/model_info

# 简单测试
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "什么是AES加密算法？",
    "max_length": 200,
    "temperature": 0.7
  }'
```

### 2. 本地部署

#### 2.1 环境准备

```bash
# 安装 uv
pip install uv

# 克隆项目
git clone <repository-url>
cd qwen-thinking-service

# 安装依赖
uv sync
```

#### 2.2 配置模型

```bash
# 设置模型路径
export MODEL_PATH="./models/qwen3-4b-thinking"
export QUANTIZATION_FORMAT="int8"
```

#### 2.3 启动服务

```bash
# 开发模式
uv run python -m uvicorn src.model_service:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uv run python -m gunicorn src.model_service:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API 使用指南

### 1. 文本生成 API

**端点**: `POST /generate`

**请求示例**:
```json
{
  "prompt": "请解释RSA加密算法的工作原理",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

**响应示例**:
```json
{
  "generated_text": "RSA加密算法是一种非对称加密算法...",
  "prompt": "请解释RSA加密算法的工作原理",
  "generation_time": 2.34,
  "token_count": 156,
  "model_info": {
    "model_type": "Qwen3-4B-Thinking",
    "quantization": "int8"
  }
}
```

### 2. 深度思考推理 API

**端点**: `POST /thinking`

**请求示例**:
```json
{
  "question": "如何设计一个安全的密码存储系统？",
  "context": "需要考虑哈希算法、盐值和密码策略",
  "thinking_depth": 3,
  "include_reasoning": true
}
```

**响应示例**:
```json
{
  "question": "如何设计一个安全的密码存储系统？",
  "thinking_process": "让我分析密码存储的安全要求...",
  "final_answer": "设计安全的密码存储系统需要以下几个关键组件...",
  "reasoning_steps": [
    "分析密码存储的安全威胁",
    "选择合适的哈希算法",
    "设计盐值生成策略"
  ],
  "confidence_score": 0.85,
  "processing_time": 3.21
}
```

### 3. 批量生成 API

**端点**: `POST /batch_generate`

**请求示例**:
```json
{
  "prompts": [
    "什么是对称加密？",
    "什么是非对称加密？",
    "什么是哈希函数？"
  ],
  "max_length": 200,
  "temperature": 0.7
}
```

### 4. 健康检查 API

**端点**: `GET /health`

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T10:30:00Z",
  "model_loaded": true,
  "gpu_available": true,
  "memory_usage": {
    "total_gb": 32.0,
    "used_gb": 12.5,
    "available_gb": 19.5,
    "percent": 39.1,
    "gpu_used_gb": 6.2
  },
  "uptime_seconds": 3600
}
```

### 5. 服务统计 API

**端点**: `GET /stats`

**响应示例**:
```json
{
  "total_requests": 1250,
  "successful_requests": 1200,
  "failed_requests": 50,
  "average_response_time": 2.34,
  "model_memory_usage": 6.2,
  "gpu_utilization": 75.5,
  "uptime_hours": 24.5
}
```

## 配置说明

### 1. 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_PATH` | `/app/models/qwen3-4b-thinking` | 模型文件路径 |
| `QUANTIZATION_FORMAT` | `int8` | 量化格式 (int8/int4/gptq) |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `MAX_WORKERS` | `1` | 工作进程数 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见GPU设备 |

### 2. 模型配置

```yaml
# configs/model_config.yaml
model:
  name: "Qwen3-4B-Thinking"
  path: "/app/models/qwen3-4b-thinking"
  quantization:
    format: "int8"
    enable: true
  
inference:
  max_length: 2048
  temperature: 0.7
  top_p: 0.9
  batch_size: 4
  
memory:
  enable_optimization: true
  gradient_checkpointing: true
  cpu_offload: false
```

### 3. 服务配置

```yaml
# configs/service_config.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  timeout: 300
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/app/logs/service.log"
  
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
```

## 监控和日志

### 1. 日志配置

日志文件位置：
- 服务日志: `/app/logs/service.log`
- 访问日志: `/app/logs/access.log`
- 错误日志: `/app/logs/error.log`

### 2. 监控指标

使用 Prometheus + Grafana 进行监控：

```bash
# 启动监控服务
docker-compose --profile monitoring up -d

# 访问 Grafana
http://localhost:3000 (admin/admin)

# 访问 Prometheus
http://localhost:9090
```

**关键监控指标**:
- 请求数量和成功率
- 响应时间分布
- GPU/CPU使用率
- 内存使用情况
- 模型推理性能

### 3. 告警配置

```yaml
# monitoring/alerts.yml
groups:
  - name: qwen-service
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
```

## 性能优化

### 1. GPU优化

```bash
# 设置GPU内存增长
export CUDA_MEMORY_FRACTION=0.8

# 启用混合精度
export ENABLE_MIXED_PRECISION=true

# 优化CUDA内核
export CUDA_LAUNCH_BLOCKING=0
```

### 2. 批处理优化

```python
# 批量推理配置
BATCH_CONFIG = {
    "max_batch_size": 8,
    "batch_timeout": 100,  # ms
    "enable_dynamic_batching": True
}
```

### 3. 缓存优化

```bash
# 启用Redis缓存
docker-compose --profile cache up -d

# 配置缓存策略
CACHE_CONFIG = {
    "enable": True,
    "ttl": 3600,  # 1小时
    "max_size": "1GB"
}
```

## 安全配置

### 1. API安全

```python
# 添加API密钥认证
API_KEYS = {
    "production": "your-secure-api-key",
    "development": "dev-api-key"
}

# 启用HTTPS
SSL_CONFIG = {
    "cert_file": "/app/certs/cert.pem",
    "key_file": "/app/certs/key.pem"
}
```

### 2. 网络安全

```yaml
# docker-compose.yml 网络配置
networks:
  qwen-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 3. 容器安全

```dockerfile
# 使用非root用户
RUN useradd -m -u 1000 qwen
USER qwen

# 限制容器权限
--security-opt no-new-privileges:true
--read-only
--tmpfs /tmp
```

## 故障排除

### 1. 常见问题

**问题**: 模型加载失败
```bash
# 检查模型文件
ls -la /app/models/qwen3-4b-thinking/

# 检查权限
chmod -R 755 /app/models/

# 检查磁盘空间
df -h
```

**问题**: GPU内存不足
```bash
# 检查GPU状态
nvidia-smi

# 减少批次大小
export BATCH_SIZE=1

# 启用CPU卸载
export ENABLE_CPU_OFFLOAD=true
```

**问题**: 服务响应慢
```bash
# 检查系统资源
htop

# 检查网络延迟
ping localhost

# 优化推理参数
export MAX_LENGTH=256
export TEMPERATURE=0.5
```

### 2. 调试命令

```bash
# 查看容器日志
docker logs qwen-thinking-service

# 进入容器调试
docker exec -it qwen-thinking-service bash

# 检查服务状态
curl -v http://localhost:8000/health

# 性能分析
docker stats qwen-thinking-service
```

### 3. 日志分析

```bash
# 错误日志分析
grep "ERROR" /app/logs/service.log | tail -20

# 性能日志分析
grep "response_time" /app/logs/access.log | awk '{print $NF}' | sort -n

# 内存使用分析
grep "memory" /app/logs/service.log | tail -10
```

## 扩展部署

### 1. 负载均衡

```yaml
# nginx.conf
upstream qwen_backend {
    server qwen-service-1:8000;
    server qwen-service-2:8000;
    server qwen-service-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://qwen_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. 多实例部署

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  qwen-model-service:
    deploy:
      replicas: 3
    environment:
      - CUDA_VISIBLE_DEVICES=${GPU_ID}
```

### 3. Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-thinking-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qwen-thinking-service
  template:
    metadata:
      labels:
        app: qwen-thinking-service
    spec:
      containers:
      - name: qwen-service
        image: qwen-thinking-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
```

## 维护指南

### 1. 定期维护

```bash
# 每日检查
./scripts/daily_check.sh

# 每周维护
./scripts/weekly_maintenance.sh

# 月度报告
./scripts/monthly_report.sh
```

### 2. 备份策略

```bash
# 模型备份
tar -czf models_backup_$(date +%Y%m%d).tar.gz ./models/

# 配置备份
tar -czf configs_backup_$(date +%Y%m%d).tar.gz ./configs/

# 日志归档
tar -czf logs_archive_$(date +%Y%m%d).tar.gz ./logs/
```

### 3. 更新流程

```bash
# 1. 备份当前版本
docker tag qwen-thinking-service:latest qwen-thinking-service:backup

# 2. 构建新版本
docker build -t qwen-thinking-service:latest .

# 3. 滚动更新
docker-compose up -d --no-deps qwen-model-service

# 4. 验证更新
curl http://localhost:8000/health
```

## 联系支持

如有问题，请联系：
- 技术支持: support@example.com
- 文档反馈: docs@example.com
- GitHub Issues: https://github.com/your-repo/issues