# Qwen3-4B-Thinking 密码学微调工具用户手册

## 概述

Qwen3-4B-Thinking 密码学微调工具是一个专门为中文密码学领域设计的模型微调系统。它基于原生PyTorch框架，支持深度思考(thinking)数据格式，提供完整的端到端训练流水线。

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU，推荐16GB+ VRAM
- **内存**: 16GB+ 系统内存
- **存储**: 50GB+ 可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.12+
- **CUDA**: 12.9+ (如果使用GPU)
- **依赖管理**: uv (推荐) 或 pip

## 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd qwen3-4b-thinking-finetuning
```

### 2. 安装依赖
使用uv (推荐):
```bash
uv sync
```

或使用pip:
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
uv run python src/cli_tools_simple.py list-gpus
```

## 快速开始

### 1. 生成配置文件
```bash
uv run python src/cli_tools_simple.py init-config --output config.yaml
```

### 2. 准备训练数据
创建JSON格式的训练数据文件，支持以下格式：

```json
[
  {
    "instruction": "什么是AES加密算法？",
    "input": "",
    "output": "AES是高级加密标准...",
    "thinking": "<thinking>用户询问AES的基本概念...</thinking>",
    "crypto_terms": ["AES", "对称加密"],
    "difficulty": 1
  }
]
```

### 3. 验证配置和数据
```bash
# 验证配置文件
uv run python src/cli_tools_simple.py validate-config config.yaml

# 检查训练数据
uv run python src/cli_tools_simple.py inspect-data data.json --sample 3
```

### 4. 开始训练
```bash
# 干运行（仅验证）
uv run python src/cli_tools_simple.py train data.json config.yaml --dry-run

# 实际训练
uv run python src/cli_tools_simple.py train data.json config.yaml --output-dir ./output
```

### 5. 监控训练状态
```bash
uv run python src/cli_tools_simple.py status ./output
```

## CLI命令详解

### init-config
生成配置文件模板

```bash
uv run python src/cli_tools_simple.py init-config [OPTIONS]
```

**选项:**
- `--output, -o`: 输出文件路径 (默认: config.yaml)

**示例:**
```bash
uv run python src/cli_tools_simple.py init-config --output my_config.yaml
```

### validate-config
验证配置文件

```bash
uv run python src/cli_tools_simple.py validate-config CONFIG_FILE
```

**参数:**
- `CONFIG_FILE`: 配置文件路径

**示例:**
```bash
uv run python src/cli_tools_simple.py validate-config config.yaml
```

### list-gpus
列出可用的GPU设备

```bash
uv run python src/cli_tools_simple.py list-gpus
```

### inspect-data
检查训练数据

```bash
uv run python src/cli_tools_simple.py inspect-data DATA_FILE [OPTIONS]
```

**参数:**
- `DATA_FILE`: 数据文件路径

**选项:**
- `--format`: 数据格式 (json, jsonl)
- `--sample`: 显示样本数量

**示例:**
```bash
uv run python src/cli_tools_simple.py inspect-data data.json --sample 5
```

### train
开始训练

```bash
uv run python src/cli_tools_simple.py train DATA_FILE CONFIG_FILE [OPTIONS]
```

**参数:**
- `DATA_FILE`: 训练数据文件路径
- `CONFIG_FILE`: 配置文件路径

**选项:**
- `--output-dir, -o`: 输出目录 (默认: ./training_output)
- `--pipeline-id`: 流水线ID
- `--resume`: 从检查点恢复
- `--dry-run`: 仅验证配置，不执行训练

**示例:**
```bash
# 基本训练
uv run python src/cli_tools_simple.py train data.json config.yaml

# 指定输出目录
uv run python src/cli_tools_simple.py train data.json config.yaml --output-dir ./my_output

# 从检查点恢复
uv run python src/cli_tools_simple.py train data.json config.yaml --resume checkpoint_id
```

### status
查看训练状态

```bash
uv run python src/cli_tools_simple.py status PIPELINE_DIR
```

**参数:**
- `PIPELINE_DIR`: 流水线输出目录

**示例:**
```bash
uv run python src/cli_tools_simple.py status ./training_output
```

## 配置文件详解

配置文件采用YAML格式，包含以下主要部分：

### model 配置
```yaml
model:
  model_name: "Qwen/Qwen3-4B-Thinking-2507"
  model_revision: "main"
  trust_remote_code: true
```

### training 配置
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: true
  save_strategy: "steps"
  save_steps: 500
  eval_strategy: "steps"
  eval_steps: 500
  logging_steps: 10
  max_seq_length: 2048
  seed: 42
```

### lora 配置
```yaml
lora:
  rank: 8
  alpha: 16
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  optimization_strategy: "auto"
```

### data 配置
```yaml
data:
  train_split_ratio: 0.9
  eval_split_ratio: 0.1
  max_samples: null
  shuffle_data: true
  data_format: "alpaca"
```

### parallel 配置
```yaml
parallel:
  strategy: "data_parallel"
  world_size: 1
  enable_distributed: false
```

### system 配置
```yaml
system:
  output_dir: "./output"
  logging_dir: "./logs"
  cache_dir: "./cache"
  log_level: "INFO"
```

## 数据格式说明

### 基本格式
```json
{
  "instruction": "问题或指令",
  "input": "可选的输入上下文",
  "output": "期望的输出回答",
  "thinking": "<thinking>思考过程</thinking>",
  "crypto_terms": ["相关的密码学术语"],
  "difficulty": 1
}
```

### 字段说明
- `instruction`: 必需，问题或指令文本
- `input`: 可选，额外的输入上下文
- `output`: 必需，期望的模型输出
- `thinking`: 可选，深度思考过程，使用`<thinking>`标签包围
- `crypto_terms`: 可选，相关的密码学术语列表
- `difficulty`: 可选，难度级别 (1-4)

### 难度级别
- 1: BEGINNER (初学者)
- 2: INTERMEDIATE (中级)
- 3: ADVANCED (高级)
- 4: EXPERT (专家)

### 支持的数据格式
- **JSON**: 单个文件包含所有样本的数组
- **JSONL**: 每行一个JSON对象

## 训练流水线

训练流水线包含以下阶段：

1. **初始化**: 验证配置和环境
2. **数据准备**: 转换数据格式，创建训练/验证集
3. **配置生成**: 生成训练配置文件
4. **环境设置**: 设置分布式训练环境
5. **训练执行**: 执行实际的模型训练
6. **评估**: 评估训练结果
7. **完成**: 生成最终报告

### 检查点机制
- 自动在每个阶段创建检查点
- 支持从任意检查点恢复训练
- 检查点包含完整的状态信息

### 进度监控
- 实时显示训练进度
- 各阶段状态跟踪
- GPU利用率监控
- 内存使用监控

## 多GPU支持

### 自动检测
系统会自动检测可用的GPU并推荐最优配置：

```bash
uv run python src/cli_tools_simple.py list-gpus
```

### 配置多GPU训练
在配置文件中设置：

```yaml
parallel:
  strategy: "data_parallel"
  world_size: 2  # GPU数量
  enable_distributed: true
```

### 支持的并行策略
- **data_parallel**: 数据并行
- **model_parallel**: 模型并行
- **pipeline_parallel**: 流水线并行
- **hybrid_parallel**: 混合并行
- **auto**: 自动选择

## 内存优化

### 自动优化
系统会根据GPU内存自动调整：
- 批次大小
- 梯度累积步数
- LoRA参数
- 混合精度设置

### 手动优化选项
```yaml
training:
  gradient_checkpointing: true  # 启用梯度检查点
  fp16: true                    # 使用FP16混合精度
  per_device_train_batch_size: 1  # 减小批次大小
  gradient_accumulation_steps: 8   # 增加梯度累积

lora:
  rank: 4      # 减小LoRA rank
  alpha: 8     # 相应调整alpha
```

## 最佳实践

### 数据准备
1. **数据质量**: 确保训练数据质量高，格式正确
2. **数据量**: 建议至少1000条高质量样本
3. **数据平衡**: 保持不同难度级别的数据平衡
4. **Thinking数据**: 充分利用thinking标签提供推理过程

### 训练配置
1. **学习率**: 从2e-4开始，根据效果调整
2. **批次大小**: 根据GPU内存调整，保持总批次大小合理
3. **训练轮次**: 通常3-5轮足够，避免过拟合
4. **保存策略**: 定期保存检查点，便于恢复

### 性能优化
1. **混合精度**: 启用bf16或fp16
2. **梯度检查点**: 在内存不足时启用
3. **LoRA参数**: 根据任务复杂度调整rank
4. **多GPU**: 充分利用多GPU资源

## 常见问题

### Q: 如何选择合适的LoRA参数？
A: 
- 简单任务: rank=4-8, alpha=8-16
- 复杂任务: rank=16-32, alpha=32-64
- 内存受限: 使用较小的rank值

### Q: 训练过程中出现OOM错误怎么办？
A:
1. 减小批次大小
2. 启用梯度检查点
3. 使用混合精度训练
4. 减小LoRA rank
5. 减小序列长度

### Q: 如何提高训练效果？
A:
1. 提高数据质量
2. 增加训练数据量
3. 调整学习率
4. 使用合适的LoRA参数
5. 充分利用thinking数据

### Q: 支持哪些操作系统？
A: 支持Windows、Linux和macOS，推荐使用Linux进行大规模训练。

### Q: 如何在多台机器上进行分布式训练？
A: 当前版本主要支持单机多GPU，多机分布式训练需要额外配置。

## 技术支持

如果遇到问题，请：

1. 查看日志文件获取详细错误信息
2. 检查系统环境报告
3. 验证配置文件格式
4. 确认数据格式正确
5. 查看故障排除指南

## 更新日志

### v1.0.0
- 初始版本发布
- 支持Qwen3-4B-Thinking模型
- 完整的CLI工具集
- 跨平台GPU检测
- 自动化训练流水线
- 深度思考数据支持