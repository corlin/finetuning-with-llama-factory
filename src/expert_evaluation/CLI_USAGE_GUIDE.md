# 专家评估系统 CLI 使用指南

## 概述

专家评估系统命令行工具提供了完整的CLI接口，支持配置文件处理、进度显示、交互功能等。

## 安装和设置

```bash
# 确保已安装依赖
uv sync

# 验证CLI工具可用
uv run python -m src.expert_evaluation.cli --help
```

## 基本用法

### 1. 显示帮助信息

```bash
# 显示主帮助
uv run python -m src.expert_evaluation.cli --help

# 显示特定命令帮助
uv run python -m src.expert_evaluation.cli evaluate --help
```

### 2. 初始化配置文件

```bash
# 创建默认配置文件
uv run python -m src.expert_evaluation.cli init-config

# 指定输出路径
uv run python -m src.expert_evaluation.cli init-config --output my_config.json
```

### 3. 验证数据格式

```bash
# 验证QA数据文件
uv run python -m src.expert_evaluation.cli validate-data data.json
```

### 4. 运行评估

```bash
# 基本评估
uv run python -m src.expert_evaluation.cli evaluate data.json

# 使用自定义配置
uv run python -m src.expert_evaluation.cli -c config.json evaluate data.json

# 保存结果到文件
uv run python -m src.expert_evaluation.cli evaluate data.json -o results.json

# 显示详细结果
uv run python -m src.expert_evaluation.cli evaluate data.json --detailed

# 不显示进度条
uv run python -m src.expert_evaluation.cli evaluate data.json --no-progress
```

## 命令详解

### init-config

初始化配置文件，创建包含默认设置的JSON配置文件。

**语法:**
```bash
uv run python -m src.expert_evaluation.cli init-config [OPTIONS]
```

**选项:**
- `--output, -o`: 配置文件输出路径 (默认: config.json)

**示例:**
```bash
uv run python -m src.expert_evaluation.cli init-config -o my_config.json
```

### validate-data

验证QA数据文件的格式和内容，显示数据统计信息。

**语法:**
```bash
uv run python -m src.expert_evaluation.cli validate-data DATA_PATH
```

**参数:**
- `DATA_PATH`: QA数据文件路径 (必需)

**示例:**
```bash
uv run python -m src.expert_evaluation.cli validate-data sample_data.json
```

### evaluate

运行模型评估，支持多种输出格式和选项。

**语法:**
```bash
uv run python -m src.expert_evaluation.cli evaluate DATA_PATH [OPTIONS]
```

**参数:**
- `DATA_PATH`: QA数据文件路径 (必需)

**选项:**
- `--output, -o`: 输出文件路径
- `--format, -f`: 输出格式 (json, html, csv)
- `--detailed, -d`: 显示详细结果
- `--no-progress`: 不显示进度条

**示例:**
```bash
# 基本评估
uv run python -m src.expert_evaluation.cli evaluate data.json

# 保存为HTML格式
uv run python -m src.expert_evaluation.cli evaluate data.json -o report.html -f html

# 详细模式
uv run python -m src.expert_evaluation.cli evaluate data.json --detailed
```

## 配置文件格式

配置文件使用JSON格式，包含以下字段：

```json
{
  "model_path": "",
  "evaluation_dimensions": [
    "语义相似性",
    "领域准确性",
    "响应相关性",
    "事实正确性",
    "完整性",
    "创新性",
    "实用价值",
    "逻辑一致性"
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
  "output_format": "json"
}
```

## QA数据格式

QA数据文件应为JSON格式的数组，每个元素包含以下字段：

```json
[
  {
    "question_id": "unique_id",
    "question": "问题内容",
    "context": "问题上下文 (可选)",
    "reference_answer": "参考答案",
    "model_answer": "模型答案",
    "domain_tags": ["标签1", "标签2"],
    "difficulty_level": "beginner|intermediate|advanced|expert",
    "expected_concepts": ["概念1", "概念2"]
  }
]
```

## 输出格式

### JSON格式
```json
{
  "overall_score": 0.85,
  "dimension_scores": {
    "语义相似性": 0.88,
    "领域准确性": 0.82
  },
  "industry_metrics": {
    "domain_relevance": 0.87,
    "practical_applicability": 0.83
  },
  "detailed_feedback": {
    "strengths": "答案准确性高",
    "suggestions": "可以增加更多示例"
  },
  "improvement_suggestions": [
    "增加具体示例",
    "补充技术细节"
  ],
  "confidence_intervals": {
    "overall_score": [0.82, 0.88]
  },
  "statistical_significance": {
    "overall_score": 0.95
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## 常见用法示例

### 完整评估流程

```bash
# 1. 初始化配置
uv run python -m src.expert_evaluation.cli init-config -o eval_config.json

# 2. 验证数据
uv run python -m src.expert_evaluation.cli validate-data qa_data.json

# 3. 运行评估
uv run python -m src.expert_evaluation.cli -c eval_config.json evaluate qa_data.json -o results.json --detailed

# 4. 查看结果
cat results.json
```

### 批量处理

```bash
# 处理多个数据文件
for file in data/*.json; do
    echo "Processing $file..."
    uv run python -m src.expert_evaluation.cli evaluate "$file" -o "results/$(basename "$file" .json)_results.json"
done
```

## 故障排除

### 常见错误

1. **配置文件错误**
   ```
   错误: 配置文件加载失败
   解决: 检查JSON格式，使用 init-config 重新生成
   ```

2. **数据格式错误**
   ```
   错误: 问题内容不能为空
   解决: 检查QA数据文件，确保必需字段存在
   ```

3. **模型加载失败**
   ```
   错误: 引擎初始化失败
   解决: 检查模型路径，确保模型文件存在
   ```

### 调试选项

```bash
# 启用详细输出
uv run python -m src.expert_evaluation.cli -v evaluate data.json

# 检查配置
uv run python -m src.expert_evaluation.cli -c config.json validate-data data.json
```

## 性能优化

### 大数据集处理

```bash
# 关闭进度条以提高性能
uv run python -m src.expert_evaluation.cli evaluate large_data.json --no-progress

# 使用简化输出格式
uv run python -m src.expert_evaluation.cli evaluate data.json -f csv
```

### 内存优化

- 处理大文件时，考虑分批处理
- 使用 `--no-progress` 选项减少内存使用
- 定期清理输出文件

## 集成和自动化

### 脚本集成

```bash
#!/bin/bash
# 自动化评估脚本

CONFIG_FILE="production_config.json"
DATA_DIR="./evaluation_data"
OUTPUT_DIR="./results"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 处理所有数据文件
for data_file in "$DATA_DIR"/*.json; do
    filename=$(basename "$data_file" .json)
    output_file="$OUTPUT_DIR/${filename}_results.json"
    
    echo "评估 $filename..."
    uv run python -m src.expert_evaluation.cli \
        -c "$CONFIG_FILE" \
        evaluate "$data_file" \
        -o "$output_file" \
        --no-progress
    
    if [ $? -eq 0 ]; then
        echo "✓ $filename 评估完成"
    else
        echo "✗ $filename 评估失败"
    fi
done

echo "所有评估任务完成"
```

### CI/CD 集成

```yaml
# GitHub Actions 示例
name: Model Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        pip install uv
        uv sync
    - name: Run evaluation
      run: |
        uv run python -m src.expert_evaluation.cli \
          evaluate test_data.json \
          -o evaluation_results.json
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: evaluation-results
        path: evaluation_results.json
```