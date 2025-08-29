# 专家评估系统配置参考指南

## 概述

本文档提供专家评估系统的完整配置参考，包括配置文件格式、参数说明、最佳实践和常见配置示例。

## 配置文件结构

### 主配置文件 (config.json)

```json
{
  "model": {
    "model_path": "/path/to/model",
    "device": "cuda",
    "quantization": "int8",
    "max_length": 2048,
    "batch_size": 4
  },
  "evaluation": {
    "dimensions": [...],
    "weights": {...},
    "thresholds": {...},
    "algorithms": {...}
  },
  "performance": {
    "max_workers": 4,
    "timeout": 300,
    "memory_limit": "8GB",
    "cache_size": "1GB"
  },
  "output": {
    "format": "json",
    "detailed": true,
    "save_intermediate": false,
    "compression": true
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/expert_evaluation.log",
    "max_size": "100MB",
    "backup_count": 5
  }
}
```

## 详细配置说明

### 1. 模型配置 (model)

#### model_path
- **类型**: string
- **必需**: 是
- **描述**: 模型文件路径或HuggingFace模型名称
- **示例**: 
  ```json
  "model_path": "/models/qwen3-4b-thinking"
  "model_path": "Qwen/Qwen3-4B-Thinking-2507"
  ```

#### device
- **类型**: string
- **默认值**: "auto"
- **可选值**: "cpu", "cuda", "cuda:0", "cuda:1", "auto"
- **描述**: 模型运行设备
- **示例**:
  ```json
  "device": "cuda:0"  // 使用第一块GPU
  "device": "cpu"     // 使用CPU
  "device": "auto"    // 自动选择最佳设备
  ```

#### quantization
- **类型**: string
- **默认值**: null
- **可选值**: null, "int8", "int4", "fp16"
- **描述**: 模型量化方式
- **示例**:
  ```json
  "quantization": "int8"  // 8位整数量化
  "quantization": null    // 不使用量化
  ```

#### max_length
- **类型**: integer
- **默认值**: 2048
- **范围**: 128-8192
- **描述**: 模型最大输入长度
- **示例**:
  ```json
  "max_length": 4096  // 支持更长的输入
  ```

#### batch_size
- **类型**: integer
- **默认值**: 1
- **范围**: 1-32
- **描述**: 批处理大小
- **注意**: 需要根据GPU内存调整
- **示例**:
  ```json
  "batch_size": 4  // 同时处理4个样本
  ```

### 2. 评估配置 (evaluation)

#### dimensions
- **类型**: array of strings
- **默认值**: 所有维度
- **可选值**: 
  - "semantic_similarity" (语义相似性)
  - "domain_accuracy" (领域准确性)
  - "response_relevance" (响应相关性)
  - "factual_correctness" (事实正确性)
  - "completeness" (完整性)
  - "innovation" (创新性)
  - "practical_value" (实用价值)
  - "logical_consistency" (逻辑一致性)
- **示例**:
  ```json
  "dimensions": [
    "semantic_similarity",
    "domain_accuracy",
    "response_relevance"
  ]
  ```

#### weights
- **类型**: object
- **描述**: 各评估维度的权重配置
- **约束**: 所有权重之和应为1.0
- **示例**:
  ```json
  "weights": {
    "semantic_similarity": 0.25,
    "domain_accuracy": 0.25,
    "response_relevance": 0.20,
    "factual_correctness": 0.15,
    "completeness": 0.15
  }
  ```

#### thresholds
- **类型**: object
- **描述**: 评估阈值设置
- **示例**:
  ```json
  "thresholds": {
    "min_score": 0.6,           // 最低可接受分数
    "confidence_level": 0.95,   // 置信水平
    "significance_level": 0.05, // 显著性水平
    "outlier_threshold": 2.0    // 异常值检测阈值
  }
  ```

#### algorithms
- **类型**: object
- **描述**: 各维度使用的算法配置
- **示例**:
  ```json
  "algorithms": {
    "semantic_similarity": {
      "method": "cosine",
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "threshold": 0.7
    },
    "domain_accuracy": {
      "method": "bert_score",
      "model": "bert-base-chinese",
      "use_idf": true
    },
    "innovation": {
      "method": "novelty_detection",
      "baseline_size": 100,
      "diversity_weight": 0.3
    }
  }
  ```

### 3. 性能配置 (performance)

#### max_workers
- **类型**: integer
- **默认值**: CPU核心数
- **范围**: 1-32
- **描述**: 最大并发工作线程数
- **建议**: 
  - CPU密集型任务: CPU核心数
  - I/O密集型任务: CPU核心数 × 2
- **示例**:
  ```json
  "max_workers": 8
  ```

#### timeout
- **类型**: integer
- **默认值**: 300
- **单位**: 秒
- **描述**: 单个评估任务超时时间
- **示例**:
  ```json
  "timeout": 600  // 10分钟超时
  ```

#### memory_limit
- **类型**: string
- **默认值**: "8GB"
- **格式**: 数字 + 单位 (MB, GB, TB)
- **描述**: 内存使用限制
- **示例**:
  ```json
  "memory_limit": "16GB"
  ```

#### cache_size
- **类型**: string
- **默认值**: "1GB"
- **描述**: 缓存大小限制
- **示例**:
  ```json
  "cache_size": "2GB"
  ```

### 4. 输出配置 (output)

#### format
- **类型**: string
- **默认值**: "json"
- **可选值**: "json", "html", "csv", "xlsx"
- **描述**: 输出格式
- **示例**:
  ```json
  "format": "html"  // 生成HTML报告
  ```

#### detailed
- **类型**: boolean
- **默认值**: true
- **描述**: 是否包含详细分析
- **示例**:
  ```json
  "detailed": false  // 只输出基本结果
  ```

#### save_intermediate
- **类型**: boolean
- **默认值**: false
- **描述**: 是否保存中间结果
- **示例**:
  ```json
  "save_intermediate": true  // 保存中间计算结果
  ```

#### compression
- **类型**: boolean
- **默认值**: false
- **描述**: 是否压缩输出文件
- **示例**:
  ```json
  "compression": true  // 压缩大文件
  ```

### 5. 日志配置 (logging)

#### level
- **类型**: string
- **默认值**: "INFO"
- **可选值**: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
- **描述**: 日志级别
- **示例**:
  ```json
  "level": "DEBUG"  // 输出详细调试信息
  ```

#### file
- **类型**: string
- **默认值**: null (控制台输出)
- **描述**: 日志文件路径
- **示例**:
  ```json
  "file": "/var/log/expert_evaluation.log"
  ```

#### max_size
- **类型**: string
- **默认值**: "100MB"
- **描述**: 单个日志文件最大大小
- **示例**:
  ```json
  "max_size": "50MB"
  ```

#### backup_count
- **类型**: integer
- **默认值**: 5
- **描述**: 保留的日志文件备份数量
- **示例**:
  ```json
  "backup_count": 10
  ```

## 环境变量配置

系统支持通过环境变量覆盖配置文件设置：

```bash
# 模型配置
export EXPERT_EVAL_MODEL_PATH="/path/to/model"
export EXPERT_EVAL_DEVICE="cuda:0"
export EXPERT_EVAL_QUANTIZATION="int8"

# 性能配置
export EXPERT_EVAL_MAX_WORKERS="8"
export EXPERT_EVAL_TIMEOUT="600"
export EXPERT_EVAL_MEMORY_LIMIT="16GB"

# 日志配置
export EXPERT_EVAL_LOG_LEVEL="DEBUG"
export EXPERT_EVAL_LOG_FILE="/var/log/expert_evaluation.log"

# API配置
export EXPERT_EVAL_API_HOST="0.0.0.0"
export EXPERT_EVAL_API_PORT="8000"
export EXPERT_EVAL_API_WORKERS="4"
```

## 配置模板

### 1. 开发环境配置

```json
{
  "model": {
    "model_path": "./models/qwen3-4b-thinking",
    "device": "cpu",
    "quantization": null,
    "max_length": 1024,
    "batch_size": 1
  },
  "evaluation": {
    "dimensions": [
      "semantic_similarity",
      "domain_accuracy"
    ],
    "weights": {
      "semantic_similarity": 0.6,
      "domain_accuracy": 0.4
    },
    "thresholds": {
      "min_score": 0.5,
      "confidence_level": 0.90
    }
  },
  "performance": {
    "max_workers": 2,
    "timeout": 120,
    "memory_limit": "4GB",
    "cache_size": "512MB"
  },
  "output": {
    "format": "json",
    "detailed": true,
    "save_intermediate": true
  },
  "logging": {
    "level": "DEBUG",
    "file": null
  }
}
```

### 2. 生产环境配置

```json
{
  "model": {
    "model_path": "/models/qwen3-4b-thinking",
    "device": "cuda:0",
    "quantization": "int8",
    "max_length": 2048,
    "batch_size": 8
  },
  "evaluation": {
    "dimensions": [
      "semantic_similarity",
      "domain_accuracy",
      "response_relevance",
      "factual_correctness",
      "completeness",
      "innovation",
      "practical_value",
      "logical_consistency"
    ],
    "weights": {
      "semantic_similarity": 0.20,
      "domain_accuracy": 0.20,
      "response_relevance": 0.15,
      "factual_correctness": 0.15,
      "completeness": 0.10,
      "innovation": 0.10,
      "practical_value": 0.05,
      "logical_consistency": 0.05
    },
    "thresholds": {
      "min_score": 0.7,
      "confidence_level": 0.95,
      "significance_level": 0.05
    }
  },
  "performance": {
    "max_workers": 16,
    "timeout": 600,
    "memory_limit": "32GB",
    "cache_size": "4GB"
  },
  "output": {
    "format": "json",
    "detailed": true,
    "save_intermediate": false,
    "compression": true
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/expert_evaluation.log",
    "max_size": "100MB",
    "backup_count": 10
  }
}
```

### 3. 高性能配置

```json
{
  "model": {
    "model_path": "/models/qwen3-4b-thinking",
    "device": "cuda",
    "quantization": "int4",
    "max_length": 4096,
    "batch_size": 16
  },
  "evaluation": {
    "dimensions": [
      "semantic_similarity",
      "domain_accuracy",
      "response_relevance"
    ],
    "weights": {
      "semantic_similarity": 0.4,
      "domain_accuracy": 0.4,
      "response_relevance": 0.2
    },
    "algorithms": {
      "semantic_similarity": {
        "method": "fast_cosine",
        "cache_embeddings": true
      }
    }
  },
  "performance": {
    "max_workers": 32,
    "timeout": 1200,
    "memory_limit": "64GB",
    "cache_size": "8GB"
  },
  "output": {
    "format": "json",
    "detailed": false,
    "compression": true
  }
}
```

## 最佳实践

### 1. 性能优化配置

#### GPU配置优化
```json
{
  "model": {
    "device": "cuda:0",
    "quantization": "int8",     // 减少显存使用
    "batch_size": 8             // 根据GPU内存调整
  },
  "performance": {
    "max_workers": 4,           // 避免GPU资源竞争
    "cache_size": "2GB"         // 充分利用缓存
  }
}
```

#### CPU配置优化
```json
{
  "model": {
    "device": "cpu",
    "batch_size": 1             // CPU模式使用小批次
  },
  "performance": {
    "max_workers": 16,          // 充分利用CPU核心
    "timeout": 300
  }
}
```

### 2. 内存优化配置

#### 大数据集处理
```json
{
  "performance": {
    "memory_limit": "16GB",
    "cache_size": "1GB"         // 限制缓存大小
  },
  "output": {
    "save_intermediate": false, // 不保存中间结果
    "compression": true         // 压缩输出
  }
}
```

#### 流式处理配置
```json
{
  "evaluation": {
    "batch_processing": true,
    "batch_size": 100           // 分批处理
  },
  "performance": {
    "streaming": true,
    "buffer_size": 1000
  }
}
```

### 3. 准确性优化配置

#### 高精度评估
```json
{
  "evaluation": {
    "thresholds": {
      "confidence_level": 0.99,
      "significance_level": 0.01
    },
    "algorithms": {
      "semantic_similarity": {
        "method": "bert_score",
        "model": "bert-large-chinese"
      }
    }
  }
}
```

#### 快速评估
```json
{
  "evaluation": {
    "dimensions": [
      "semantic_similarity",
      "domain_accuracy"
    ],
    "algorithms": {
      "semantic_similarity": {
        "method": "fast_cosine"
      }
    }
  }
}
```

### 4. 安全配置

#### 生产环境安全
```json
{
  "security": {
    "encrypt_results": true,
    "anonymize_data": true,
    "audit_logging": true
  },
  "logging": {
    "level": "WARNING",         // 减少敏感信息记录
    "sanitize_logs": true
  }
}
```

## 配置验证

### 配置文件验证工具

```bash
# 验证配置文件格式
uv run python -m src.expert_evaluation.cli validate-config config.json

# 测试配置文件
uv run python -m src.expert_evaluation.cli test-config config.json
```

### 配置检查清单

1. **必需配置检查:**
   - [ ] model_path 已设置且文件存在
   - [ ] 评估维度权重之和为1.0
   - [ ] 设备配置与硬件匹配
   - [ ] 内存限制合理设置

2. **性能配置检查:**
   - [ ] max_workers 不超过CPU核心数的2倍
   - [ ] batch_size 适合GPU内存
   - [ ] timeout 设置合理
   - [ ] 缓存大小适中

3. **安全配置检查:**
   - [ ] 日志级别适当
   - [ ] 敏感数据处理配置
   - [ ] 访问权限设置
   - [ ] 审计日志启用

## 故障排除

### 常见配置错误

1. **模型加载失败**
   ```json
   // 错误配置
   {
     "model": {
       "model_path": "/nonexistent/path",
       "device": "cuda:99"
     }
   }
   
   // 正确配置
   {
     "model": {
       "model_path": "/models/qwen3-4b-thinking",
       "device": "auto"
     }
   }
   ```

2. **内存不足**
   ```json
   // 问题配置
   {
     "model": {
       "batch_size": 32
     },
     "performance": {
       "max_workers": 16
     }
   }
   
   // 优化配置
   {
     "model": {
       "batch_size": 4,
       "quantization": "int8"
     },
     "performance": {
       "max_workers": 4,
       "memory_limit": "8GB"
     }
   }
   ```

3. **权重配置错误**
   ```json
   // 错误配置 (权重之和不为1.0)
   {
     "evaluation": {
       "weights": {
         "semantic_similarity": 0.5,
         "domain_accuracy": 0.6
       }
     }
   }
   
   // 正确配置
   {
     "evaluation": {
       "weights": {
         "semantic_similarity": 0.5,
         "domain_accuracy": 0.5
       }
     }
   }
   ```

### 配置调试技巧

1. **使用配置验证工具**
   ```bash
   uv run python -c "
   from src.expert_evaluation.config import ExpertEvaluationConfig
   config = ExpertEvaluationConfig.from_file('config.json')
   print('配置验证成功')
   "
   ```

2. **启用详细日志**
   ```json
   {
     "logging": {
       "level": "DEBUG",
       "file": "debug.log"
     }
   }
   ```

3. **分步测试配置**
   ```bash
   # 测试模型加载
   uv run python -c "
   from src.expert_evaluation.engine import ExpertEvaluationEngine
   engine = ExpertEvaluationEngine('config.json')
   print('引擎初始化成功')
   "
   ```

## 配置迁移

### 版本升级配置迁移

```python
# 配置迁移脚本示例
def migrate_config_v1_to_v2(old_config: dict) -> dict:
    """从v1.0配置迁移到v2.0配置"""
    new_config = {
        "model": {
            "model_path": old_config.get("model_path"),
            "device": old_config.get("device", "auto"),
            "quantization": old_config.get("quantization")
        },
        "evaluation": {
            "dimensions": old_config.get("dimensions", []),
            "weights": old_config.get("weights", {}),
            "thresholds": old_config.get("thresholds", {})
        }
    }
    return new_config
```

### 环境间配置同步

```bash
# 开发环境配置导出
uv run python -m src.expert_evaluation.cli export-config --env dev

# 生产环境配置导入
uv run python -m src.expert_evaluation.cli import-config --env prod --file dev_config.json
```

## 总结

本配置参考指南提供了专家评估系统的完整配置说明，包括各种环境下的最佳实践配置。正确的配置是系统高效运行的基础，建议根据实际需求和硬件环境选择合适的配置模板，并根据使用情况进行调优。