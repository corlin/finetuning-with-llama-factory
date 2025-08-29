# 专家评估系统常见问题解答 (FAQ)

## 概述

本文档收集了用户在使用专家评估系统过程中最常遇到的问题和解答，帮助用户快速找到解决方案。

## 安装和配置

### Q1: 如何安装专家评估系统？

**A:** 专家评估系统是现有项目的一个模块，安装步骤如下：

```bash
# 1. 确保在项目根目录
cd /path/to/project

# 2. 安装依赖
uv sync

# 3. 验证安装
uv run python -c "from src.expert_evaluation import *; print('安装成功')"
```

### Q2: 系统支持哪些Python版本？

**A:** 专家评估系统要求Python 3.12或更高版本。可以通过以下命令检查：

```bash
python --version
# 或
uv run python --version
```

### Q3: 如何配置GPU支持？

**A:** GPU配置步骤：

1. **检查GPU可用性:**
```bash
nvidia-smi
uv run python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

2. **配置文件设置:**
```json
{
  "model": {
    "device": "cuda",
    "quantization": "int8"
  }
}
```

3. **如果GPU内存不足，使用量化:**
```json
{
  "model": {
    "device": "cuda",
    "quantization": "int4",
    "batch_size": 1
  }
}
```

### Q4: 配置文件应该放在哪里？

**A:** 配置文件可以放在以下位置：

- 项目根目录: `config.json`
- 配置目录: `configs/expert_evaluation.json`
- 用户指定路径: 通过 `-c` 参数指定

创建默认配置文件：
```bash
uv run python -m src.expert_evaluation.cli init-config
```

## 模型和数据

### Q5: 支持哪些模型格式？

**A:** 系统支持以下模型格式：

- **HuggingFace格式**: 本地路径或模型名称
- **PyTorch格式**: .pt, .pth文件
- **ONNX格式**: .onnx文件 (实验性支持)

示例配置：
```json
{
  "model": {
    "model_path": "Qwen/Qwen3-4B-Thinking-2507",  // HuggingFace Hub
    "model_path": "/path/to/local/model",          // 本地路径
    "model_path": "./models/qwen3-4b-thinking"     // 相对路径
  }
}
```

### Q6: QA数据格式要求是什么？

**A:** QA数据应为JSON格式的数组，每个元素包含以下字段：

```json
[
  {
    "question_id": "q001",           // 必需: 问题唯一标识
    "question": "什么是AES加密？",    // 必需: 问题内容
    "context": "密码学基础",         // 可选: 问题上下文
    "reference_answer": "AES是...",  // 必需: 参考答案
    "model_answer": "AES算法是...",  // 必需: 模型答案
    "domain_tags": ["密码学"],       // 可选: 领域标签
    "difficulty_level": "intermediate", // 可选: 难度级别
    "expected_concepts": ["对称加密"]   // 可选: 期望概念
  }
]
```

验证数据格式：
```bash
uv run python -m src.expert_evaluation.cli validate-data your_data.json
```

### Q7: 如何处理大数据集？

**A:** 处理大数据集的建议：

1. **分批处理:**
```json
{
  "performance": {
    "batch_size": 100,
    "max_workers": 4
  }
}
```

2. **使用流式处理:**
```bash
# 分割大文件
split -l 1000 large_data.json small_data_

# 分别处理
for file in small_data_*; do
    uv run python -m src.expert_evaluation.cli evaluate "$file"
done
```

3. **启用压缩:**
```json
{
  "output": {
    "compression": true
  }
}
```

## 评估和指标

### Q8: 系统提供哪些评估维度？

**A:** 系统提供8个评估维度：

1. **语义相似性** (semantic_similarity): 答案与参考答案的语义相似程度
2. **领域准确性** (domain_accuracy): 专业领域知识的准确性
3. **响应相关性** (response_relevance): 答案与问题的相关程度
4. **事实正确性** (factual_correctness): 事实信息的正确性
5. **完整性** (completeness): 答案的完整程度
6. **创新性** (innovation): 答案的创新和独特性
7. **实用价值** (practical_value): 实际应用价值
8. **逻辑一致性** (logical_consistency): 逻辑推理的一致性

### Q9: 如何自定义评估权重？

**A:** 在配置文件中设置权重（总和必须为1.0）：

```json
{
  "evaluation": {
    "weights": {
      "semantic_similarity": 0.30,
      "domain_accuracy": 0.25,
      "response_relevance": 0.20,
      "factual_correctness": 0.15,
      "completeness": 0.10
    }
  }
}
```

### Q10: 评估结果如何解读？

**A:** 评估结果包含多个层次：

1. **总体得分** (overall_score): 0-1之间，越高越好
2. **维度得分** (dimension_scores): 各维度的具体得分
3. **行业指标** (industry_metrics): 行业特定的评估指标
4. **置信区间** (confidence_intervals): 结果的可信度范围
5. **改进建议** (improvement_suggestions): 具体的改进建议

示例解读：
```json
{
  "overall_score": 0.85,        // 总体表现良好
  "dimension_scores": {
    "semantic_similarity": 0.88, // 语义理解优秀
    "domain_accuracy": 0.75      // 专业知识有待提升
  },
  "confidence_intervals": {
    "overall_score": [0.82, 0.88] // 95%置信区间
  }
}
```

## 性能和优化

### Q11: 评估速度太慢怎么办？

**A:** 性能优化建议：

1. **使用GPU加速:**
```json
{
  "model": {
    "device": "cuda",
    "quantization": "int8"
  }
}
```

2. **增加并行度:**
```json
{
  "performance": {
    "max_workers": 8,
    "batch_size": 4
  }
}
```

3. **减少评估维度:**
```json
{
  "evaluation": {
    "dimensions": [
      "semantic_similarity",
      "domain_accuracy"
    ]
  }
}
```

4. **启用缓存:**
```json
{
  "performance": {
    "cache_size": "2GB"
  }
}
```

### Q12: 内存使用过多怎么解决？

**A:** 内存优化方案：

1. **限制内存使用:**
```json
{
  "performance": {
    "memory_limit": "8GB"
  }
}
```

2. **使用模型量化:**
```json
{
  "model": {
    "quantization": "int8"  // 或 "int4"
  }
}
```

3. **减少批处理大小:**
```json
{
  "model": {
    "batch_size": 1
  }
}
```

4. **清理GPU缓存:**
```bash
uv run python -c "import torch; torch.cuda.empty_cache()"
```

### Q13: 如何监控系统性能？

**A:** 性能监控方法：

1. **使用内置监控:**
```bash
# 检查API健康状态
curl http://localhost:8000/health

# 查看系统统计
curl http://localhost:8000/stats
```

2. **系统资源监控:**
```bash
# CPU和内存
htop

# GPU使用率
nvidia-smi -l 1

# 磁盘I/O
iotop
```

3. **日志监控:**
```bash
# 实时查看日志
tail -f /var/log/expert_evaluation.log

# 分析日志
grep "ERROR" /var/log/expert_evaluation.log
```

## API和集成

### Q14: 如何启动API服务？

**A:** API服务启动方法：

1. **开发模式:**
```bash
uv run python -m src.expert_evaluation.api
# 或
uv run uvicorn src.expert_evaluation.api:app --reload
```

2. **生产模式:**
```bash
uv run uvicorn src.expert_evaluation.api:app --host 0.0.0.0 --port 8000 --workers 4
```

3. **使用Docker:**
```bash
docker build -t expert-evaluation .
docker run -p 8000:8000 expert-evaluation
```

### Q15: API接口如何使用？

**A:** 主要API接口：

1. **健康检查:**
```bash
curl -X GET "http://localhost:8000/health"
```

2. **同步评估:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "qa_items": [...],
    "async_mode": false
  }'
```

3. **异步评估:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "qa_items": [...],
    "async_mode": true
  }'
```

4. **查询任务状态:**
```bash
curl -X GET "http://localhost:8000/task/{task_id}"
```

### Q16: 如何集成到现有系统？

**A:** 集成方案：

1. **Python SDK集成:**
```python
from src.expert_evaluation.engine import ExpertEvaluationEngine

engine = ExpertEvaluationEngine()
result = engine.evaluate_model(qa_data)
```

2. **REST API集成:**
```python
import requests

response = requests.post(
    "http://localhost:8000/evaluate",
    json={"qa_items": qa_data, "async_mode": False}
)
result = response.json()
```

3. **命令行集成:**
```bash
# 在脚本中调用
uv run python -m src.expert_evaluation.cli evaluate data.json -o results.json
```

## 故障排除

### Q17: 模块导入失败怎么办？

**A:** 常见解决方案：

1. **检查Python路径:**
```bash
uv run python -c "import sys; print(sys.path)"
```

2. **重新安装依赖:**
```bash
uv sync --reinstall
```

3. **检查项目结构:**
```bash
ls -la src/expert_evaluation/
```

### Q18: GPU内存不足错误如何解决？

**A:** GPU内存优化：

1. **减少批处理大小:**
```json
{"model": {"batch_size": 1}}
```

2. **启用量化:**
```json
{"model": {"quantization": "int8"}}
```

3. **清理GPU缓存:**
```bash
uv run python -c "import torch; torch.cuda.empty_cache()"
```

4. **使用CPU模式:**
```json
{"model": {"device": "cpu"}}
```

### Q19: 配置文件格式错误怎么办？

**A:** 配置文件修复：

1. **验证JSON格式:**
```bash
cat config.json | python -m json.tool
```

2. **重新生成配置:**
```bash
uv run python -m src.expert_evaluation.cli init-config
```

3. **使用配置验证工具:**
```bash
uv run python -m src.expert_evaluation.cli validate-config config.json
```

### Q20: 评估结果不准确怎么办？

**A:** 结果准确性优化：

1. **检查数据质量:**
```bash
uv run python -m src.expert_evaluation.cli validate-data data.json
```

2. **调整评估参数:**
```json
{
  "evaluation": {
    "thresholds": {
      "confidence_level": 0.99
    }
  }
}
```

3. **使用更精确的算法:**
```json
{
  "evaluation": {
    "algorithms": {
      "semantic_similarity": {
        "method": "bert_score",
        "model": "bert-large-chinese"
      }
    }
  }
}
```

## 最佳实践

### Q21: 生产环境部署建议？

**A:** 生产环境最佳实践：

1. **资源配置:**
- CPU: 8核以上
- 内存: 16GB以上
- GPU: 8GB显存以上 (可选)
- 磁盘: SSD，100GB以上

2. **安全配置:**
```json
{
  "security": {
    "encrypt_results": true,
    "anonymize_data": true,
    "audit_logging": true
  }
}
```

3. **监控告警:**
- 设置资源使用告警
- 配置服务健康检查
- 启用详细日志记录

4. **备份策略:**
- 定期备份配置文件
- 备份重要评估结果
- 建立灾难恢复计划

### Q22: 如何提高评估质量？

**A:** 评估质量优化：

1. **数据质量:**
- 确保参考答案准确性
- 提供充分的上下文信息
- 标注领域标签和难度级别

2. **配置优化:**
- 根据应用场景调整权重
- 选择合适的评估维度
- 设置合理的阈值

3. **模型选择:**
- 使用领域相关的预训练模型
- 考虑模型大小和性能平衡
- 定期更新模型版本

### Q23: 如何进行A/B测试？

**A:** A/B测试方案：

1. **准备测试数据:**
```bash
# 创建测试集
uv run python -c "
import json
import random

with open('full_data.json', 'r') as f:
    data = json.load(f)

random.shuffle(data)
test_size = len(data) // 2

with open('test_a.json', 'w') as f:
    json.dump(data[:test_size], f)

with open('test_b.json', 'w') as f:
    json.dump(data[test_size:], f)
"
```

2. **配置不同的评估策略:**
```json
// config_a.json
{
  "evaluation": {
    "weights": {
      "semantic_similarity": 0.5,
      "domain_accuracy": 0.5
    }
  }
}

// config_b.json
{
  "evaluation": {
    "weights": {
      "semantic_similarity": 0.3,
      "domain_accuracy": 0.4,
      "innovation": 0.3
    }
  }
}
```

3. **运行对比测试:**
```bash
# 测试A
uv run python -m src.expert_evaluation.cli -c config_a.json evaluate test_a.json -o results_a.json

# 测试B
uv run python -m src.expert_evaluation.cli -c config_b.json evaluate test_b.json -o results_b.json

# 比较结果
uv run python -c "
import json

with open('results_a.json', 'r') as f:
    results_a = json.load(f)

with open('results_b.json', 'r') as f:
    results_b = json.load(f)

print(f'配置A平均得分: {results_a[\"overall_score\"]:.3f}')
print(f'配置B平均得分: {results_b[\"overall_score\"]:.3f}')
"
```

## 社区和支持

### Q24: 如何获取技术支持？

**A:** 技术支持渠道：

1. **文档资源:**
- 架构设计文档
- 配置参考指南
- 故障排除指南
- API文档

2. **问题报告:**
- 使用GitHub Issues
- 提供详细的环境信息
- 包含完整的错误日志
- 描述重现步骤

3. **社区讨论:**
- 参与技术论坛
- 加入用户群组
- 分享使用经验
- 贡献改进建议

### Q25: 如何贡献代码？

**A:** 代码贡献流程：

1. **Fork项目:**
```bash
git clone https://github.com/your-username/project.git
cd project
```

2. **创建功能分支:**
```bash
git checkout -b feature/new-evaluation-metric
```

3. **开发和测试:**
```bash
# 安装开发依赖
uv sync --dev

# 运行测试
uv run python -m pytest tests/ -v

# 代码格式化
uv run black src/
uv run isort src/
```

4. **提交Pull Request:**
- 描述变更内容
- 包含测试用例
- 更新相关文档
- 通过CI检查

## 总结

本FAQ涵盖了专家评估系统使用过程中的常见问题和解答。如果您的问题没有在此列出，请：

1. 查阅详细的技术文档
2. 搜索已有的问题报告
3. 在社区论坛提问
4. 联系技术支持团队

我们持续更新FAQ内容，欢迎用户反馈常见问题和建议。