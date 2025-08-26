# 直接使用已实现模块进行微调

本目录包含了不依赖LlamaFactory，直接使用已实现模块进行微调的脚本。

**目标模型**: `Qwen/Qwen3-4B-Thinking-2507` - 专门支持深度思考推理的4B参数模型

## 文件说明

### 核心脚本

1. **`direct_finetuning_with_existing_modules.py`** - 主要的微调脚本
   - 直接使用PyTorch和已实现的模块进行微调
   - 支持LoRA微调、混合精度训练、梯度检查点等
   - 集成了内存管理、GPU检测、中文处理等功能

2. **`test_direct_finetuning.py`** - 完整功能测试脚本
   - 测试模型加载、数据处理、推理等功能
   - 包含模型下载和推理测试

3. **`quick_test_existing_modules.py`** - 快速模块测试脚本
   - 只测试已实现模块的导入和基本功能
   - 不需要下载大模型，运行速度快

4. **`run_direct_finetuning_test.py`** - 运行脚本
   - 提供交互式选择界面
   - 使用uv环境管理器运行测试和训练

5. **`test_qwen3_4b_thinking.py`** - Qwen3-4B-Thinking专用测试
   - 测试模型可用性和内存需求
   - 验证thinking格式处理
   - 优化配置测试

## 使用方法

### 1. 快速环境检查

首先运行快速测试，检查已实现模块是否正常工作：

```bash
uv run python quick_test_existing_modules.py
```

### 2. Qwen3-4B-Thinking专用测试

针对目标模型的专门测试：

```bash
uv run python test_qwen3_4b_thinking.py
```

这个脚本会测试：
- 模型可用性和下载
- Thinking数据格式处理
- GPU内存需求评估
- 优化配置验证

这个脚本会测试：
- 模块导入
- 数据模型创建
- GPU检测
- 中文处理
- 密码学术语处理
- 数据加载
- PyTorch环境

### 3. 完整功能测试

如果基础测试通过，可以运行完整测试：

```bash
uv run python test_direct_finetuning.py
```

这个脚本会测试：
- 模型加载（会下载小模型）
- LoRA配置
- 简单推理
- 所有已实现模块的集成

### 4. 开始微调

测试通过后，可以开始微调：

```bash
uv run python direct_finetuning_with_existing_modules.py
```

或者使用交互式脚本：

```bash
uv run python run_direct_finetuning_test.py
```

## Qwen3-4B-Thinking模型特殊要求

### 硬件要求
- **GPU内存**: 至少12GB VRAM（推荐16GB+）
- **系统内存**: 至少16GB RAM
- **存储空间**: 至少20GB可用空间（模型+检查点）

### 优化配置
- 使用fp16混合精度训练
- 启用梯度检查点节省内存
- 较小的批次大小（1-2）
- 增加梯度累积步数（8+）

### Thinking数据格式
模型专门支持`<thinking>`标签格式：
```
<thinking>
这里是模型的思考过程...
分析问题、推理步骤、得出结论
</thinking>

这里是最终回答...
```

## 配置说明

### DirectTrainingConfig 配置项

```python
@dataclass
class DirectTrainingConfig:
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"  # 模型名称
    data_path: str = "final_demo_output/data/crypto_qa_dataset_train.json"  # 数据路径
    output_dir: str = "direct_training_output"  # 输出目录
    max_seq_length: int = 1024  # 最大序列长度
    batch_size: int = 2  # 批次大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    learning_rate: float = 2e-4  # 学习率
    num_epochs: int = 1  # 训练轮数
    
    # LoRA配置
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.1  # LoRA dropout
    
    # 优化选项
    use_gradient_checkpointing: bool = True  # 梯度检查点
    use_fp16: bool = True  # 混合精度
```

## 已集成的功能模块

### 1. GPU检测和管理
- `GPUDetector`: 自动检测GPU配置
- `MemoryManager`: 动态内存管理和OOM预防

### 2. 数据处理
- `ChineseNLPProcessor`: 中文文本预处理
- `CryptoTermProcessor`: 密码学术语识别和处理
- `DatasetSplitter`: 智能数据集分割

### 3. 训练监控
- `TrainingMonitor`: 训练过程监控
- 实时损失跟踪
- GPU利用率监控

### 4. 并行训练支持
- `ParallelStrategyRecommender`: 并行策略推荐
- 多GPU配置优化

## 数据格式

训练数据应为JSON格式，每条记录包含：

```json
{
  "instruction": "问题或指令",
  "input": "额外输入（可选）",
  "output": "期望输出，可包含<thinking>标签",
  "system": "系统提示（可选）",
  "difficulty": 1
}
```

## 输出结果

训练完成后，会在输出目录生成：

- `checkpoint-{step}/`: 训练检查点
- `final_model/`: 最终微调模型
- `training.log`: 训练日志

## 故障排除

### 常见问题

1. **模块导入失败**
   ```bash
   # 确保src目录在Python路径中
   export PYTHONPATH=src:$PYTHONPATH
   ```

2. **GPU内存不足**
   - 减少`batch_size`
   - 启用`use_gradient_checkpointing`
   - 减少`max_seq_length`

3. **模型下载失败**
   - 检查网络连接
   - 使用镜像源：`export HF_ENDPOINT=https://hf-mirror.com`

4. **数据文件不存在**
   - 确保数据文件路径正确
   - 检查`final_demo_output/data/`目录

### 调试模式

启用详细日志：

```bash
export PYTHONPATH=src
export CUDA_LAUNCH_BLOCKING=1
uv run python direct_finetuning_with_existing_modules.py
```

## 性能优化建议

1. **内存优化**
   - 使用混合精度训练（fp16/bf16）
   - 启用梯度检查点
   - 适当设置批次大小和梯度累积

2. **速度优化**
   - 使用多GPU并行训练
   - 优化数据加载（pin_memory=True）
   - 合理设置学习率调度

3. **质量优化**
   - 使用合适的LoRA参数
   - 监控训练指标
   - 定期保存检查点

## 下一步计划

1. 添加分布式训练支持
2. 集成模型量化功能
3. 添加更多评估指标
4. 支持更多模型架构