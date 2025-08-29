# Qwen3-4B-Thinking 中文密码学微调系统

专门针对Qwen3-4B-Thinking模型的中文密码学领域微调系统，提供完整的训练、评估和部署解决方案。

## 🎯 核心特性

### 模型与数据
- 🚀 **专业模型支持**: 针对Qwen/Qwen3-4B-Thinking-2507模型深度优化
- 🧠 **思考链推理**: 完整支持`<thinking>`标签的CoT推理数据格式
- 🇨🇳 **中文NLP优化**: 专业中文文本处理、繁简转换、密码学术语识别
- 📚 **密码学专业**: 内置密码学术语库、专业QA数据处理

### 训练与优化
- 💾 **内存高效**: LoRA微调、4bit/8bit量化、混合精度训练、梯度检查点
- 🔄 **多GPU并行**: 数据并行、模型并行、流水线并行、自动策略推荐
- ⚡ **性能优化**: 智能批次调整、内存管理、OOM预防、NUMA优化
- 🎛️ **自适应配置**: 硬件检测、参数自动调优、并行策略推荐

### 监控与评估
- 📊 **实时监控**: TensorBoard集成、训练指标跟踪、GPU利用率监控
- 🔍 **专家评估**: 多维度模型评估、中文能力验证、密码学专业评估
- 📈 **性能分析**: 训练效率分析、资源使用统计、异常检测

### 部署与服务
- 📦 **模型导出**: 多种量化格式(FP16/INT8/INT4)、安全导出、格式转换
- 🌐 **服务化部署**: REST API、模型服务、Docker容器化
- 🔧 **工具链**: CLI工具、批处理脚本、自动化流水线

## 📋 系统要求

### 基础环境
- **Python**: 3.12+ (必需)
- **CUDA**: 12.9+ (GPU训练推荐)
- **包管理器**: uv (推荐) 或 pip

### 硬件配置
| 配置级别 | GPU内存 | 系统内存 | 推荐用途 |
|----------|---------|----------|----------|
| 最小配置 | 8GB | 16GB | 基础训练、量化推理 |
| 推荐配置 | 16GB+ | 32GB+ | 完整训练、多GPU并行 |
| 高性能配置 | 24GB+ | 64GB+ | 大规模训练、专业部署 |

### 支持的GPU
- NVIDIA RTX 30/40/50系列
- NVIDIA Tesla/Quadro系列
- 支持CUDA计算能力6.0+

## 🚀 快速开始

### 1. 环境安装

#### 方式一：使用uv (推荐)
```bash
# 安装uv包管理器
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目并安装依赖
git clone <repository-url>
cd qwen3-4b-thinking-finetuning
uv sync --extra dev
```

#### 方式二：使用pip
```bash
git clone <repository-url>
cd qwen3-4b-thinking-finetuning
pip install -e .
pip install -r requirements.txt
```

### 2. 环境验证

```bash
# 完整环境设置
uv run python setup.py

# 检查系统环境和GPU状态
uv run python scripts/check_environment.py

# 验证核心功能
uv run python test_basic_setup.py
```

### 3. 数据准备

#### 数据目录结构
```
data/
├── raw/                    # 原始markdown文件 (*.md)
├── train/                  # 训练数据 (*.json)
├── eval/                   # 验证数据 (*.json)
├── test/                   # 测试数据 (*.json)
└── processed/              # 处理后的数据
```

#### 数据格式示例
```json
{
  "instruction": "请解释AES加密算法的工作原理",
  "input": "",
  "output": "<thinking>需要从AES的基本概念开始...</thinking>AES是一种对称加密算法..."
}
```

### 4. 快速演示

```bash
# 运行快速演示 (推荐新手)
uv run python run_demo.py

# 运行完整演示
uv run python demo_final.py

# 运行简化训练演示
uv run python demo_simple_finetuning.py
```

### 5. 开始训练

#### 单GPU训练
```bash
# 使用默认配置
uv run python scripts/train.py

# 使用自定义配置
uv run python scripts/train.py --config configs/custom_config.yaml
```

#### 多GPU训练
```bash
# 2个GPU数据并行
uv run torchrun --nproc_per_node=2 scripts/train.py

# 分布式训练
uv run python run_training_true_distributed.py
```

#### 高级训练选项
```bash
# 优化的单GPU训练
uv run python run_training_optimized_single.py

# 内存优化训练
uv run python run_training_minimal.py

# 性能基准测试
uv run python run_performance_optimization_validation.py
```

## 📁 项目结构

```
qwen3-4b-thinking-finetuning/
├── src/                           # 🔧 核心功能模块
│   ├── training_pipeline.py      # 训练流水线管理
│   ├── distributed_training_engine.py # 分布式训练引擎
│   ├── model_exporter.py         # 模型导出和量化
│   ├── expert_evaluation/         # 专家评估系统
│   ├── chinese_nlp_processor.py  # 中文NLP处理
│   ├── crypto_term_processor.py  # 密码学术语处理
│   ├── thinking_generator.py     # 思考链数据生成
│   ├── performance_optimizer.py  # 性能优化器
│   ├── memory_manager.py         # 内存管理
│   ├── gpu_utils.py              # GPU检测和管理
│   ├── parallel_strategy_recommender.py # 并行策略推荐
│   └── ...                       # 其他核心模块
├── scripts/                       # 🚀 执行脚本
│   ├── train.py                  # 主训练脚本
│   ├── check_environment.py      # 环境检查
│   ├── deploy_service.py         # 部署服务
│   └── validate_service.py       # 服务验证
├── examples/                      # 📚 示例和演示
│   ├── expert_evaluation_demo.py # 专家评估演示
│   ├── chinese_nlp_demo.py       # 中文NLP演示
│   ├── crypto_term_demo.py       # 密码学术语演示
│   ├── thinking_generator_demo.py # 思考链生成演示
│   ├── model_export_deployment_demo.py # 模型导出部署演示
│   └── ...                       # 其他演示程序
├── configs/                       # ⚙️ 配置文件
│   ├── config.yaml               # 主配置文件
│   └── cryptography_evaluation.yaml # 密码学评估配置
├── data/                          # 📊 数据目录
│   ├── raw/                      # 原始markdown文件
│   ├── train/                    # 训练数据
│   ├── eval/                     # 验证数据
│   ├── test/                     # 测试数据
│   └── processed/                # 处理后的数据
├── tests/                         # 🧪 测试套件
│   ├── integration/              # 集成测试
│   ├── expert_evaluation/        # 专家评估测试
│   └── ...                       # 单元测试
├── docs/                          # 📖 文档
│   ├── deployment_guide.md       # 部署指南
│   ├── expert_evaluation_architecture.md # 专家评估架构
│   ├── troubleshooting_guide.md  # 故障排除指南
│   └── ...                       # 其他文档
├── monitoring/                    # 📈 监控配置
│   ├── prometheus.yml            # Prometheus配置
│   ├── grafana/                  # Grafana仪表板
│   └── alerts.yml                # 告警规则
├── output/                        # 📤 训练输出
├── logs/                          # 📝 日志文件
├── cache/                         # 💾 缓存目录
├── models/                        # 🤖 模型文件
└── checkpoints/                   # 💾 训练检查点
```

## ⚙️ 配置说明

### 模型配置 (`model`)
```yaml
model:
  model_name: "Qwen/Qwen3-4B-Thinking-2507"  # 模型名称
  load_in_4bit: false          # 4bit量化加载
  load_in_8bit: false          # 8bit量化加载
  torch_dtype: "auto"          # 数据类型 (auto/float16/bfloat16)
  device_map: "auto"           # 设备映射策略
  max_seq_length: 2048         # 最大序列长度
  trust_remote_code: true      # 信任远程代码
```

### LoRA微调配置 (`lora`)
```yaml
lora:
  r: 16                        # LoRA rank (8-64)
  lora_alpha: 32              # LoRA缩放参数
  lora_dropout: 0.1           # Dropout率
  target_modules:             # 目标模块
    - "q_proj"
    - "k_proj" 
    - "v_proj"
    - "o_proj"
  use_rslora: false           # 使用RSLoRA
  use_dora: false             # 使用DoRA
```

### 训练配置 (`training`)
```yaml
training:
  num_train_epochs: 3                    # 训练轮数
  per_device_train_batch_size: 1         # 每设备批次大小
  gradient_accumulation_steps: 4         # 梯度累积步数
  learning_rate: 2e-4                    # 学习率
  lr_scheduler_type: "cosine"            # 学习率调度器
  warmup_ratio: 0.1                      # 预热比例
  weight_decay: 0.01                     # 权重衰减
  gradient_checkpointing: true           # 梯度检查点
  bf16: true                             # BF16混合精度
  dataloader_num_workers: 4              # 数据加载器工作进程
```

### 多GPU并行配置 (`multigpu`)
```yaml
multigpu:
  enable_distributed: false              # 启用分布式训练
  world_size: 1                         # 总进程数
  backend: "nccl"                       # 通信后端 (nccl/gloo)
  data_parallel: true                   # 数据并行
  model_parallel: false                 # 模型并行
  pipeline_parallel: false              # 流水线并行
  zero_stage: 2                         # ZeRO优化阶段
```

### 中文处理配置 (`chinese`)
```yaml
chinese:
  tokenizer_name: "Qwen/Qwen3-4B-Thinking-2507"
  add_special_tokens: true              # 添加特殊token
  preserve_thinking_structure: true     # 保留thinking结构
  thinking_start_token: "<thinking>"    # thinking开始token
  thinking_end_token: "</thinking>"     # thinking结束token
  enable_traditional_conversion: true   # 启用繁简转换
  normalize_punctuation: true           # 标准化标点符号
```

### 数据处理配置 (`data`)
```yaml
data:
  train_data_path: "./data/train"       # 训练数据路径
  eval_data_path: "./data/eval"         # 验证数据路径
  test_data_path: "./data/test"         # 测试数据路径
  data_format: "json"                   # 数据格式
  preserve_thinking_tags: true          # 保留thinking标签
  enable_chinese_preprocessing: true    # 启用中文预处理
  preserve_crypto_terms: true           # 保留密码学术语
```

## 📊 数据格式与处理

### 支持的数据格式

#### 1. 标准QA格式
```json
{
  "instruction": "请解释AES加密算法的工作原理",
  "input": "",
  "output": "AES（高级加密标准）是一种对称加密算法，采用分组密码体制..."
}
```

#### 2. 思考链格式 (CoT)
```json
{
  "instruction": "分析RSA算法的安全性",
  "input": "",
  "output": "<thinking>首先需要考虑RSA算法的数学基础：1. 大整数分解困难性 2. 欧拉函数性质 3. 模运算特性...</thinking>RSA算法的安全性主要基于大整数分解的数学难题..."
}
```

#### 3. 多轮对话格式
```json
{
  "conversations": [
    {"from": "human", "value": "什么是对称加密？"},
    {"from": "gpt", "value": "<thinking>用户询问对称加密的基本概念...</thinking>对称加密是指加密和解密使用相同密钥的加密方式..."},
    {"from": "human", "value": "能举个例子吗？"},
    {"from": "gpt", "value": "当然可以。AES就是一个典型的对称加密算法..."}
  ]
}
```

### 数据处理功能

#### 中文文本处理
- **繁简转换**: 自动识别并转换繁体中文
- **标点标准化**: 统一中英文标点符号
- **分词优化**: 针对密码学术语的专业分词
- **编码处理**: UTF-8编码确保和emoji处理

#### 密码学专业处理
- **术语识别**: 内置密码学术语库，保护专业词汇
- **概念标注**: 自动标注密码学概念和算法名称
- **难度分级**: 根据内容复杂度自动分级
- **质量验证**: 检查QA对的完整性和准确性

#### 思考链处理
- **结构验证**: 检查`<thinking>`标签的完整性
- **内容分析**: 分析思考过程的逻辑性
- **长度优化**: 自动调整思考链长度
- **格式标准化**: 统一思考链格式

## 🚀 性能优化指南

### GPU内存优化策略

| GPU内存 | 量化设置 | 批次大小 | 梯度累积 | 序列长度 | 预期性能 |
|---------|----------|----------|----------|----------|----------|
| 8GB | 4bit量化 | 1 | 8 | 1024 | 基础训练 |
| 12GB | 8bit量化 | 1 | 4 | 1536 | 标准训练 |
| 16GB | FP16 | 2 | 4 | 2048 | 高效训练 |
| 24GB+ | BF16 | 4 | 2 | 2048 | 最佳性能 |

### 自动优化配置

#### 使用性能优化器
```python
from src.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()
config = optimizer.optimize_for_hardware()
print(f"推荐配置: {config}")
```

#### 并行策略推荐
```python
from src.parallel_strategy_recommender import ParallelStrategyRecommender

recommender = ParallelStrategyRecommender()
strategy = recommender.recommend_strategy(
    num_gpus=2,
    gpu_memory_gb=16,
    model_size="4B"
)
```

### 内存管理

#### OOM预防
```bash
# 启用OOM管理器
uv run python src/oom_manager.py --monitor

# 内存使用分析
uv run python src/memory_manager.py --analyze
```

#### NUMA优化
```bash
# NUMA检测和优化
uv run python test_numa_detection.py

# 查看NUMA优化报告
cat numa_solution_report.md
```

### 训练加速技巧

#### 1. 梯度检查点
```yaml
training:
  gradient_checkpointing: true  # 减少内存使用
  dataloader_num_workers: 4     # 并行数据加载
  dataloader_pin_memory: true   # 固定内存
```

#### 2. 混合精度训练
```yaml
training:
  bf16: true                    # BF16精度 (推荐)
  fp16: false                   # FP16精度 (备选)
```

#### 3. 优化器设置
```yaml
training:
  optim: "adamw_torch"          # 优化器选择
  adam_beta1: 0.9               # Adam参数
  adam_beta2: 0.999
  weight_decay: 0.01            # 权重衰减
```

## 📊 监控与评估系统

### 实时训练监控

#### TensorBoard集成
```bash
# 启动TensorBoard
tensorboard --logdir logs/ --port 6006

# 查看训练指标
# - 训练/验证损失
# - 学习率变化
# - GPU利用率
# - 内存使用情况
```

#### 训练监控器
```python
from src.training_monitor import TrainingMonitor

monitor = TrainingMonitor()
monitor.start_monitoring()
# 自动记录训练指标、GPU状态、内存使用
```

### 专家评估系统

#### 基础评估
```bash
# 运行基础评估
uv run python examples/expert_evaluation_basic_usage.py

# 高级评估场景
uv run python examples/expert_evaluation_advanced_scenarios.py

# 性能基准测试
uv run python examples/expert_evaluation_performance_benchmark.py
```

#### 中文能力验证
```bash
# 中文NLP能力测试
uv run python examples/chinese_capability_validation_demo.py

# 中文指标计算
uv run python examples/chinese_nlp_demo.py
```

#### 密码学专业评估
```bash
# 密码学术语测试
uv run python examples/crypto_term_demo.py

# 高级密码学评估
uv run python examples/advanced_crypto_demo.py
```

### 评估指标

#### 通用指标
- **困惑度 (Perplexity)**: 模型预测能力
- **BLEU分数**: 文本生成质量
- **ROUGE分数**: 摘要质量评估
- **准确率**: 分类任务准确性

#### 中文专项指标
- **中文BLEU**: 针对中文优化的BLEU
- **字符级准确率**: 中文字符预测准确性
- **词汇覆盖率**: 中文词汇识别能力
- **语法正确性**: 中文语法结构评估

#### 密码学专项指标
- **术语准确率**: 密码学术语使用准确性
- **概念理解度**: 密码学概念掌握程度
- **推理逻辑性**: 密码学推理过程评估
- **专业深度**: 回答的专业程度评分

### 监控工具

#### 系统监控
```bash
# GPU状态监控
nvidia-smi -l 1

# 系统资源监控
uv run python src/system_config.py --monitor

# 分布式指标收集
uv run python src/distributed_metrics_collector.py
```

#### 异常检测
```bash
# 训练异常检测
uv run python src/anomaly_detector.py

# 性能异常分析
uv run python tests/test_performance_benchmarks.py
```

## 📦 模型导出与部署

### 模型导出功能

#### 支持的导出格式
```bash
# FP16导出 (推荐)
uv run python src/model_exporter.py --format fp16 --input checkpoints/best_model

# INT8量化导出
uv run python src/model_exporter.py --format int8 --input checkpoints/best_model

# INT4量化导出 (最小体积)
uv run python src/model_exporter.py --format int4 --input checkpoints/best_model

# 安全导出 (包含验证)
uv run python src/model_exporter.py --format safe --input checkpoints/best_model
```

#### 导出演示
```bash
# 完整导出演示
uv run python examples/model_export_deployment_demo.py

# 实用导出演示
uv run python practical_model_export_demo.py
```

### 模型服务化

#### 启动模型服务
```bash
# 启动REST API服务
uv run python start_model_service.py --model path/to/exported/model

# 使用Docker部署
docker-compose up -d

# 验证服务状态
uv run python scripts/validate_service.py
```

#### API使用示例
```python
import requests

# 基础推理
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "请解释AES加密算法",
    "max_length": 200,
    "temperature": 0.7
})

# 思考链推理
response = requests.post("http://localhost:8000/thinking", json={
    "prompt": "分析RSA算法的安全性",
    "enable_thinking": True
})
```

### 部署配置

#### Docker部署
```dockerfile
# 查看 Dockerfile 了解容器配置
# 支持GPU加速和多种部署模式
```

#### 监控配置
```bash
# Prometheus监控
cat monitoring/prometheus.yml

# Grafana仪表板
ls monitoring/grafana/

# 告警规则
cat monitoring/alerts.yml
```

## 🛠️ 开发工具与工作流

### uv包管理器

#### 基础命令
```bash
# 同步依赖
uv sync --extra dev

# 安装新包
uv add <package>

# 移除包
uv remove <package>

# 查看依赖树
uv tree

# 运行脚本
uv run python <script.py>

# 运行测试
uv run pytest
```

#### 开发工作流
```bash
# 1. 环境设置
uv sync --extra dev
uv run python setup_with_uv.py

# 2. 代码质量检查
uv run black src/          # 代码格式化
uv run isort src/          # 导入排序
uv run flake8 src/         # 代码检查

# 3. 测试执行
uv run pytest tests/       # 单元测试
uv run python run_integration_tests.py  # 集成测试

# 4. 性能验证
uv run python run_performance_optimization_validation.py
```

### CLI工具

#### 简化CLI工具
```bash
# 使用简化CLI
uv run python src/cli_tools_simple.py --help

# 完整CLI功能
uv run python src/cli_tools.py --help
```

#### 批处理脚本
```bash
# 数据处理批处理
uv run python convert_all_enhanced_data.py

# 训练数据验证
uv run python validate_training_data.py

# 最终数据质量测试
uv run python final_data_quality_test.py
```

## 🔧 故障排除指南

### 环境问题

#### uv包管理器问题
```bash
# 重新同步依赖
uv sync --extra dev --reinstall

# 清理缓存
uv cache clean

# 检查uv版本
uv --version

# 验证Python环境
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
```

#### CUDA环境问题
```bash
# 检查CUDA安装
nvidia-smi
nvcc --version

# 验证PyTorch CUDA支持
uv run python -c "import torch; print(torch.cuda.is_available())"

# GPU检测测试
uv run python test_gpu_detection.py
```

### 训练问题

#### 内存不足 (OOM)
```bash
# 启用OOM管理器
uv run python src/oom_manager.py --prevent

# 内存优化建议
uv run python src/memory_manager.py --optimize

# 检查内存使用
uv run python test_memory_usage.py
```

**解决方案**:
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing: true`
- 使用量化: `load_in_4bit: true`

#### 训练速度慢
```bash
# 性能分析
uv run python src/performance_optimizer.py --analyze

# 并行策略优化
uv run python src/parallel_strategy_recommender.py --recommend
```

**优化建议**:
- 启用混合精度: `bf16: true`
- 增加 `dataloader_num_workers`
- 使用多GPU并行训练
- 优化数据预处理

#### 分布式训练问题
```bash
# 分布式训练测试
uv run python test_distributed_training.py

# 检查网络配置
uv run python src/distributed_training_engine.py --check
```

### 数据问题

#### 中文编码问题
```bash
# 中文处理测试
uv run python examples/chinese_nlp_demo.py

# 编码验证
uv run python test_enhanced_data.py
```

**解决方案**:
- 确保文件使用UTF-8编码
- 检查 `chinese.normalize_punctuation: true`
- 验证tokenizer配置

#### 数据格式问题
```bash
# 数据验证
uv run python validate_training_data.py

# 格式检查
uv run python test_json_serialization_fix.py
```

### 模型问题

#### 模型加载失败
```bash
# 模型测试
uv run python test_qwen3_4b_thinking.py

# 网络连接测试
uv run python test_api_import.py
```

**解决方案**:
- 检查网络连接和代理设置
- 验证模型名称: `Qwen/Qwen3-4B-Thinking-2507`
- 清理缓存: `rm -rf cache/`
- 设置 `local_files_only: false`

#### 量化问题
```bash
# 量化测试
uv run python fixed_quantization_final.py

# 查看量化修复报告
cat QUANTIZATION_FIX_SUMMARY.md
```

### 系统问题

#### NUMA优化
```bash
# NUMA检测
uv run python test_numa_detection.py

# 查看优化建议
cat numa_solution_report.md
```

#### 依赖冲突
```bash
# 依赖检查
uv run python test_basic_setup.py

# 环境验证
uv run python comprehensive_validation.py
```

### 调试工具

#### 日志分析
```bash
# 查看设置日志
cat logs/setup.log

# 训练日志
tail -f logs/train.log

# 错误日志
grep -i error logs/*.log
```

#### 测试套件
```bash
# 基础功能测试
uv run python test_minimal_api.py

# 集成测试
uv run python run_integration_tests.py

# 性能测试
uv run python run_performance_optimization_validation.py
```

### 获取帮助

#### 自动诊断
```bash
# 环境检查
uv run python scripts/check_environment.py

# 系统验证
uv run python comprehensive_validation_sync.py

# 问题报告生成
uv run python generate_issue_report.py  # (如果存在)
```

#### 文档资源
- 📖 [故障排除指南](docs/troubleshooting_guide.md)
- 📖 [专家评估故障排除](docs/expert_evaluation_troubleshooting.md)
- 📖 [部署指南](docs/deployment_guide.md)
- 📖 [用户手册](docs/user_manual.md)

## 📚 示例与演示

### 快速演示程序
```bash
# 🚀 快速入门演示
uv run python run_demo.py

# 🎯 完整功能演示  
uv run python demo_final.py

# 🔧 简化训练演示
uv run python demo_simple_finetuning.py

# 📊 综合集成测试
uv run python comprehensive_integration_test.py
```

### 专业功能演示
```bash
# 🧠 专家评估系统
uv run python examples/expert_evaluation_demo.py

# 🇨🇳 中文NLP处理
uv run python examples/chinese_nlp_demo.py

# 🔐 密码学术语处理
uv run python examples/crypto_term_demo.py

# 💭 思考链生成
uv run python examples/thinking_generator_demo.py

# 📦 模型导出部署
uv run python examples/model_export_deployment_demo.py
```

### 高级功能演示
```bash
# ⚡ 分布式训练
uv run python examples/distributed_training_demo.py

# 📈 性能基准测试
uv run python examples/expert_evaluation_performance_benchmark.py

# 🎛️ LoRA配置优化
uv run python examples/lora_config_demo.py

# 📊 数据集分割
uv run python examples/dataset_splitter_demo.py
```

## 🏆 项目亮点

### 技术创新
- ✨ **思考链推理**: 完整支持CoT推理，提升模型逻辑能力
- 🧠 **专家评估**: 多维度智能评估系统，确保模型质量
- 🇨🇳 **中文优化**: 深度中文NLP处理，专业术语保护
- ⚡ **性能优化**: 智能硬件检测，自动配置优化策略

### 工程实践
- 🔧 **模块化设计**: 高度模块化，易于扩展和维护
- 📊 **完整监控**: 实时训练监控，异常检测和预警
- 🚀 **自动化流水线**: 端到端自动化，从数据到部署
- 🛡️ **稳定可靠**: 完善的错误处理和恢复机制

### 专业领域
- 🔐 **密码学专业**: 专门针对密码学领域优化
- 📚 **教育友好**: 丰富的示例和文档，易于学习
- 🌐 **生产就绪**: 支持容器化部署和服务化
- 📈 **可扩展性**: 支持大规模分布式训练

## 📖 相关文档

### 核心文档
- 📋 [API使用指南](API_USAGE_GUIDE.md)
- 🤖 [模型使用指南](MODEL_USAGE_GUIDE.md)
- 🚀 [演示程序指南](DEMO_README.md)
- 🎯 [最终演示指南](FINAL_DEMO_README.md)

### 专业文档
- 🏗️ [专家评估架构](docs/expert_evaluation_architecture.md)
- ⚙️ [专家评估配置](docs/expert_evaluation_configuration.md)
- 🚀 [部署指南](docs/deployment_guide.md)
- 📋 [操作指南](docs/operations_guide.md)

### 技术文档
- 💭 [思考生成器实现](docs/thinking_generator_implementation.md)
- 🔐 [密码学术语增强](docs/crypto_term_enhancement_summary.md)
- 🔧 [故障排除指南](docs/troubleshooting_guide.md)
- 👤 [用户手册](docs/user_manual.md)

## 🤝 贡献指南

### 参与贡献
1. Fork 项目仓库
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 提交Pull Request

### 开发规范
- 遵循PEP 8代码规范
- 添加适当的测试用例
- 更新相关文档
- 确保所有测试通过

### 问题反馈
- 🐛 [报告Bug](../../issues/new?template=bug_report.md)
- 💡 [功能建议](../../issues/new?template=feature_request.md)
- ❓ [使用问题](../../discussions)

## 📄 许可证

本项目基于 [MIT许可证](LICENSE) 开源。

## 🏷️ 版本历史

### v0.2.0 (当前版本)
- ✅ 完整的专家评估系统
- ✅ 中文NLP深度优化
- ✅ 密码学专业术语处理
- ✅ 思考链数据生成
- ✅ 模型导出和部署
- ✅ 分布式训练支持
- ✅ 性能监控和优化
- ✅ 完整的文档体系

### v0.1.0 (初始版本)
- ✅ 基础Qwen3-4B-Thinking支持
- ✅ LoRA微调功能
- ✅ GPU检测和优化
- ✅ 基础训练流水线

---

<div align="center">

**🎉 感谢使用 Qwen3-4B-Thinking 中文密码学微调系统！**

*如果这个项目对您有帮助，请考虑给我们一个 ⭐*

</div>