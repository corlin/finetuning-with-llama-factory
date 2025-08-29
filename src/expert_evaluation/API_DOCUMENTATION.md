# 专家评估系统 API 文档

## 概述

专家评估系统提供基于FastAPI的RESTful API接口，支持异步评估任务处理、批量评估、任务管理等功能。API设计遵循REST原则，提供完整的错误处理和状态管理。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API版本**: v1.0.0
- **文档地址**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc地址**: `http://localhost:8000/redoc`

## 认证

当前版本不需要认证，后续版本将支持API密钥或JWT认证。

## 通用响应格式

### 成功响应
```json
{
  "status": "success",
  "data": {...},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 错误响应
```json
{
  "error": "错误类型",
  "detail": "详细错误信息",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## API 端点

### 1. 健康检查

#### GET /health

检查服务健康状态和系统信息。

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "model_loaded": true,
  "active_tasks": 2,
  "system_info": {
    "cpu_percent": 45.2,
    "memory_percent": 68.5,
    "disk_percent": 23.1
  }
}
```

### 2. 服务统计

#### GET /stats

获取服务运行统计信息。

**响应示例:**
```json
{
  "total_requests": 1250,
  "successful_requests": 1180,
  "failed_requests": 70,
  "active_tasks": 3,
  "average_response_time": 2.45,
  "uptime": 86400.0
}
```

### 3. 模型管理

#### POST /load_model

加载评估模型。

**请求体:**
```json
{
  "model_path": "/path/to/model",
  "config": {
    "quantization": "int8",
    "device": "cuda"
  }
}
```

**响应示例:**
```json
{
  "status": "success",
  "message": "模型 /path/to/model 加载成功"
}
```

#### GET /model_info

获取当前加载模型的信息。

**响应示例:**
```json
{
  "model_name": "Qwen3-4B-Thinking",
  "model_path": "/path/to/model",
  "parameters": "4B",
  "quantization": "int8",
  "device": "cuda:0",
  "memory_usage": "3.2GB"
}
```

### 4. 模型评估

#### POST /evaluate

执行模型评估，支持同步和异步模式。

**请求体:**
```json
{
  "qa_items": [
    {
      "question_id": "q001",
      "question": "什么是AES加密？",
      "context": "密码学基础",
      "reference_answer": "AES是高级加密标准...",
      "model_answer": "AES（Advanced Encryption Standard）...",
      "domain_tags": ["密码学", "对称加密"],
      "difficulty_level": "intermediate",
      "expected_concepts": ["对称加密", "密钥长度"]
    }
  ],
  "config": {
    "evaluation_dimensions": ["semantic_similarity", "domain_accuracy"],
    "industry_weights": {
      "semantic_similarity": 0.6,
      "domain_accuracy": 0.4
    }
  },
  "async_mode": false
}
```

**同步响应示例:**
```json
{
  "overall_score": 0.85,
  "dimension_scores": {
    "语义相似性": 0.88,
    "领域准确性": 0.82,
    "响应相关性": 0.85
  },
  "industry_metrics": {
    "domain_relevance": 0.87,
    "practical_applicability": 0.83,
    "innovation_level": 0.75,
    "completeness": 0.90
  },
  "detailed_feedback": {
    "strengths": "答案准确性高，概念理解正确",
    "weaknesses": "可以增加更多实际应用示例",
    "suggestions": "建议补充AES的具体应用场景"
  },
  "improvement_suggestions": [
    "增加具体的代码示例",
    "补充安全性分析",
    "添加与其他算法的比较"
  ],
  "confidence_intervals": {
    "overall_score": [0.82, 0.88],
    "semantic_similarity": [0.85, 0.91]
  },
  "statistical_significance": {
    "overall_score": 0.95,
    "dimension_comparison": 0.89
  }
}
```

**异步响应示例:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "评估任务已提交，请使用 /task/{task_id} 查询进度"
}
```

### 5. 批量评估

#### POST /batch_evaluate

执行批量模型评估。

**请求体:**
```json
{
  "datasets": [
    [
      {
        "question_id": "dataset1_q001",
        "question": "...",
        "reference_answer": "...",
        "model_answer": "..."
      }
    ],
    [
      {
        "question_id": "dataset2_q001", 
        "question": "...",
        "reference_answer": "...",
        "model_answer": "..."
      }
    ]
  ],
  "config": {...},
  "async_mode": true
}
```

**响应示例:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "pending",
  "message": "批量评估任务已提交，请使用 /task/{task_id} 查询进度"
}
```

### 6. 任务管理

#### GET /task/{task_id}

获取任务状态和结果。

**响应示例:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:05:30Z",
  "result": {
    "overall_score": 0.85,
    "dimension_scores": {...},
    "industry_metrics": {...}
  },
  "error": null
}
```

#### DELETE /task/{task_id}

取消正在运行的任务。

**响应示例:**
```json
{
  "message": "任务 550e8400-e29b-41d4-a716-446655440000 已取消"
}
```

#### GET /tasks

列出所有任务。

**响应示例:**
```json
{
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "progress": 1.0,
      "start_time": "2024-01-01T12:00:00Z",
      "end_time": "2024-01-01T12:05:30Z"
    }
  ],
  "total_count": 10,
  "active_count": 2
}
```

### 7. 报告生成

#### POST /generate_report/{task_id}

生成评估报告。

**查询参数:**
- `format`: 报告格式 (json, html, pdf)

**响应示例 (JSON格式):**
```json
{
  "report_id": "report_001",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "format": "json",
  "generated_at": "2024-01-01T12:10:00Z",
  "summary": {
    "total_questions": 50,
    "average_score": 0.85,
    "top_performing_dimensions": ["语义相似性", "领域准确性"],
    "improvement_areas": ["创新性", "实用价值"]
  },
  "detailed_analysis": {...},
  "recommendations": [...]
}
```

## 数据模型

### QA评估项 (QAItemRequest)

```json
{
  "question_id": "string (必需)",
  "question": "string (必需)",
  "context": "string (可选)",
  "reference_answer": "string (必需)",
  "model_answer": "string (必需)",
  "domain_tags": ["string"],
  "difficulty_level": "beginner|intermediate|advanced|expert",
  "expected_concepts": ["string"]
}
```

### 评估配置 (Config)

```json
{
  "evaluation_dimensions": [
    "semantic_similarity",
    "domain_accuracy", 
    "response_relevance",
    "factual_correctness",
    "completeness",
    "innovation",
    "practical_value",
    "logical_consistency"
  ],
  "industry_weights": {
    "semantic_similarity": 0.25,
    "domain_accuracy": 0.25,
    "response_relevance": 0.20,
    "factual_correctness": 0.15,
    "completeness": 0.15
  },
  "threshold_settings": {
    "min_score": 0.6,
    "confidence_level": 0.95
  },
  "enable_detailed_analysis": true,
  "comparison_baseline": "string (可选)"
}
```

## 错误代码

| 状态码 | 错误类型 | 描述 |
|--------|----------|------|
| 400 | Bad Request | 请求参数错误或数据格式错误 |
| 404 | Not Found | 资源不存在（如任务ID不存在） |
| 422 | Unprocessable Entity | 评估处理错误 |
| 500 | Internal Server Error | 服务器内部错误 |
| 503 | Service Unavailable | 服务不可用（如模型未加载） |

## 使用示例

### Python客户端示例

```python
import requests

# 基础配置
BASE_URL = "http://localhost:8000"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. 同步评估
qa_data = {
    "qa_items": [
        {
            "question_id": "q001",
            "question": "什么是RSA算法？",
            "reference_answer": "RSA是一种公钥加密算法...",
            "model_answer": "RSA算法是基于大数分解的公钥密码系统...",
            "domain_tags": ["密码学", "公钥加密"],
            "difficulty_level": "intermediate"
        }
    ],
    "async_mode": False
}

response = requests.post(f"{BASE_URL}/evaluate", json=qa_data)
result = response.json()
print(f"评估得分: {result['overall_score']}")

# 3. 异步评估
qa_data["async_mode"] = True
response = requests.post(f"{BASE_URL}/evaluate", json=qa_data)
task_id = response.json()["task_id"]

# 轮询任务状态
import time
while True:
    response = requests.get(f"{BASE_URL}/task/{task_id}")
    status = response.json()
    
    if status["status"] == "completed":
        print("评估完成!")
        print(f"结果: {status['result']}")
        break
    elif status["status"] == "failed":
        print(f"评估失败: {status['error']}")
        break
    
    print(f"进度: {status['progress']:.1%}")
    time.sleep(2)
```

### cURL示例

```bash
# 健康检查
curl -X GET "http://localhost:8000/health"

# 同步评估
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "qa_items": [
      {
        "question_id": "q001",
        "question": "什么是AES加密？",
        "reference_answer": "AES是高级加密标准...",
        "model_answer": "AES算法是一种对称加密算法...",
        "domain_tags": ["密码学"],
        "difficulty_level": "intermediate"
      }
    ],
    "async_mode": false
  }'

# 获取任务状态
curl -X GET "http://localhost:8000/task/{task_id}"
```

## 性能考虑

### 请求限制
- 单次评估最大QA项数: 1000
- 批量评估最大数据集数: 10
- 并发任务限制: 50

### 超时设置
- 同步评估超时: 300秒
- 异步任务超时: 3600秒
- 连接超时: 30秒

### 缓存策略
- 模型推理结果缓存: 1小时
- 评估配置缓存: 30分钟
- 任务状态缓存: 24小时

## 部署说明

### 开发环境启动
```bash
# 安装依赖
uv sync

# 启动API服务器
uv run python -m src.expert_evaluation.api

# 或使用uvicorn直接启动
uv run uvicorn src.expert_evaluation.api:app --host 0.0.0.0 --port 8000 --reload
```

### 生产环境部署
```bash
# 使用Gunicorn + Uvicorn
gunicorn src.expert_evaluation.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 使用Docker
docker build -t expert-evaluation-api .
docker run -p 8000:8000 expert-evaluation-api
```

### 环境变量配置
```bash
# API配置
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# 模型配置
export MODEL_PATH=/path/to/model
export MODEL_DEVICE=cuda
export MODEL_QUANTIZATION=int8

# 日志配置
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/expert-evaluation.log
```

## 监控和日志

### 健康检查端点
- `/health`: 基础健康检查
- `/stats`: 详细统计信息

### 日志格式
```
2024-01-01 12:00:00 [INFO] 评估任务开始: task_id=xxx, qa_count=50
2024-01-01 12:05:30 [INFO] 评估任务完成: task_id=xxx, score=0.85, duration=330s
2024-01-01 12:10:15 [ERROR] 评估失败: task_id=xxx, error=模型加载失败
```

### 监控指标
- 请求QPS
- 平均响应时间
- 错误率
- 活跃任务数
- 内存使用率
- GPU利用率

## 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口是否被占用
   - 验证Python环境和依赖
   - 查看启动日志

2. **模型加载失败**
   - 检查模型路径是否正确
   - 验证模型格式兼容性
   - 确认GPU内存充足

3. **评估任务超时**
   - 减少QA项数量
   - 使用异步模式
   - 检查系统资源

4. **内存不足**
   - 调整批处理大小
   - 启用模型量化
   - 增加系统内存

### 联系支持
- 技术文档: [链接]
- 问题反馈: [邮箱]
- 社区讨论: [论坛链接]