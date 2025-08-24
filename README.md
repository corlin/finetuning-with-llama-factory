# LLaMA Factory Finetuning for Qwen3-4B-Thinking

基于LLaMA Factory框架的Qwen3-4B-Thinking模型微调系统，专门针对中文密码学领域优化。

## 功能特性

- 🚀 **专业模型支持**: 针对Qwen/Qwen3-4B-Thinking-2507模型优化
- 🧠 **深度思考数据**: 支持`<thinking>`标签的推理数据格式
- 🇨🇳 **中文优化**: 专门针对中文文本和密码学术语处理
- 💾 **内存高效**: LoRA微调、混合精度训练、梯度检查点
- 🔄 **多GPU支持**: 数据并行、模型并行、流水线并行
- 📊 **智能监控**: 实时训练监控和专家评估系统
- 📦 **模型导出**: 支持多种量化格式导出

## 系统要求

- Python 3.12+
- CUDA 12.9+ (推荐)
- GPU内存: 最小8GB，推荐16GB+
- 系统内存: 最小16GB
- uv包管理器

## 快速开始

### 1. 安装uv包管理器

首先确保已安装uv包管理器：

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用pip安装
pip install uv
```

### 2. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd llama-factory-finetuning

# 使用uv安装依赖
uv sync --extra dev

# 运行环境设置
uv run python src/environment_setup.py
```

### 3. 检查环境

```bash
# 检查系统环境和GPU状态
uv run python scripts/check_environment.py
```

### 4. 准备数据

将训练数据放置在以下目录：
- `data/raw/` - 原始markdown文件
- `data/train/` - 训练数据
- `data/eval/` - 验证数据
- `data/test/` - 测试数据

### 4. 配置调整

编辑 `configs/config.yaml` 文件，根据需要调整配置：

```yaml
model:
  model_name: "Qwen/Qwen3-4B-Thinking-2507"
  load_in_4bit: true  # 4bit量化以节省内存

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_train_epochs: 3

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### 6. 开始训练

```bash
# 单GPU训练
uv run python scripts/train.py

# 多GPU训练
uv run torchrun --nproc_per_node=2 scripts/train.py
```

## 项目结构

```
llama-factory-finetuning/
├── src/                    # 源代码
│   ├── gpu_utils.py       # GPU检测和管理
│   ├── model_config.py    # 模型配置管理
│   ├── config_manager.py  # 配置管理系统
│   └── environment_setup.py # 环境设置
├── scripts/               # 执行脚本
│   ├── train.py          # 训练脚本
│   └── check_environment.py # 环境检查
├── configs/               # 配置文件
│   └── config.yaml       # 主配置文件
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── train/            # 训练数据
│   ├── eval/             # 验证数据
│   └── test/             # 测试数据
├── output/                # 训练输出
├── logs/                  # 日志文件
├── cache/                 # 缓存目录
└── models/                # 模型文件
```

## 配置说明

### 模型配置

- `model_name`: Qwen3-4B-Thinking模型名称
- `load_in_4bit/8bit`: 量化加载以节省内存
- `torch_dtype`: 数据类型（auto/float16/bfloat16）
- `device_map`: 设备映射策略

### LoRA配置

- `r`: LoRA rank，控制适配器大小
- `lora_alpha`: LoRA缩放参数
- `target_modules`: 目标模块列表
- `lora_dropout`: Dropout率

### 训练配置

- `per_device_train_batch_size`: 每设备批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率
- `num_train_epochs`: 训练轮数

### 多GPU配置

- `enable_distributed`: 启用分布式训练
- `world_size`: 总进程数
- `backend`: 通信后端（nccl/gloo）

## 数据格式

### 标准格式

```json
{
  "instruction": "请解释AES加密算法的工作原理",
  "input": "",
  "output": "AES（高级加密标准）是一种对称加密算法..."
}
```

### 思考格式

```json
{
  "instruction": "分析RSA算法的安全性",
  "input": "",
  "output": "<thinking>首先需要考虑RSA算法的数学基础...</thinking>RSA算法的安全性主要基于..."
}
```

## GPU内存优化

### 8GB GPU
- 启用4bit量化: `load_in_4bit: true`
- 批次大小: `per_device_train_batch_size: 1`
- 梯度累积: `gradient_accumulation_steps: 8`

### 16GB GPU
- 启用8bit量化: `load_in_8bit: true`
- 批次大小: `per_device_train_batch_size: 2`
- 梯度累积: `gradient_accumulation_steps: 4`

### 24GB+ GPU
- 标准精度训练
- 批次大小: `per_device_train_batch_size: 4`
- 梯度累积: `gradient_accumulation_steps: 2`

## 监控和评估

训练过程中可以通过以下方式监控：

1. **TensorBoard**: `tensorboard --logdir logs/`
2. **日志文件**: 查看 `logs/` 目录下的日志
3. **GPU监控**: 使用 `nvidia-smi` 或项目内置监控

## uv包管理器使用

本项目使用uv作为包管理器，提供更快的依赖解析和安装。

### 常用uv命令

```bash
# 同步依赖
uv sync

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

# 查看已安装包
uv pip list

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 开发工作流

```bash
# 1. 克隆项目
git clone <repository-url>
cd llama-factory-finetuning

# 2. 使用uv设置环境
python setup_with_uv.py

# 3. 测试环境
uv run python test_uv_setup.py

# 4. 开发和测试
uv run python scripts/check_environment.py
uv run python scripts/train.py
```

## 故障排除

### 常见问题

1. **uv相关问题**
   - 重新同步依赖: `uv sync --extra dev`
   - 清理缓存: `uv cache clean`
   - 检查uv版本: `uv --version`

2. **CUDA内存不足**
   - 减小批次大小
   - 启用梯度检查点
   - 使用量化加载

3. **模型加载失败**
   - 检查网络连接
   - 验证模型名称
   - 检查缓存目录权限

4. **中文编码问题**
   - 确保文件使用UTF-8编码
   - 检查tokenizer配置

5. **依赖安装问题**
   - 使用uv重新安装: `uv sync --reinstall`
   - 检查Python版本: `uv run python --version`
   - 查看详细错误: `uv run python -c "import <module>"`

### 获取帮助

- 查看日志文件: `logs/setup.log`
- 运行环境检查: `uv run python scripts/check_environment.py`
- 测试uv环境: `uv run python test_uv_setup.py`
- 查看GPU状态: `nvidia-smi`

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 更新日志

### v0.1.0
- 初始版本发布
- 支持Qwen3-4B-Thinking模型
- 实现基础训练功能
- 添加GPU检测和优化