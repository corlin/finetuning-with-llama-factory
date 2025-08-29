# 综合微调演示程序

基于当前已完成的功能，使用 `data/raw` 数据进行模型微调的完整演示程序。

## 功能特性

- ✅ 自动加载和处理 `data/raw` 中的所有数据文件
- ✅ 智能数据预处理和格式转换
- ✅ 自动配置 GPU 并行策略和 LoRA 参数
- ✅ 集成原生PyTorch进行模型微调
- ✅ 实时训练监控和进度跟踪
- ✅ 完整的训练流水线管理

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
# 使用 uv 安装依赖
uv sync

# 或者使用 pip
pip install -r requirements.txt
```

### 2. 快速演示

运行快速演示程序，验证环境和数据处理功能：

```bash
uv run python run_demo.py
```

这个脚本会：
- 检查环境配置
- 检测GPU硬件
- 处理示例数据
- 生成处理报告

### 3. 完整训练演示

运行完整的训练流水线：

```bash
# 使用默认配置
uv run python demo_comprehensive_finetuning.py

# 指定数据目录和输出目录
uv run python demo_comprehensive_finetuning.py --data-dir data/raw --output-dir my_output

# 启用详细输出
uv run python demo_comprehensive_finetuning.py --verbose
```

## 程序结构

### 主要文件

- `demo_comprehensive_finetuning.py` - 完整的微调演示程序
- `run_demo.py` - 快速演示和环境验证脚本
- `demo_config.yaml` - 演示配置文件
- `DEMO_README.md` - 本说明文件

### 输出目录结构

```
demo_output/
├── data/                    # 处理后的训练数据
│   ├── demo_dataset_train.json
│   ├── demo_dataset_val.json
│   └── dataset_info.json
├── configs/                 # 生成的配置文件
│   ├── training_config_*.yaml
│   └── dataset_info.json
├── pipeline/                # 训练流水线输出
│   ├── checkpoints/
│   ├── logs/
│   └── pipeline_report.json
├── model_output/            # 模型训练输出
└── training_report_*.json   # 训练报告
```

## 数据处理流程

1. **数据加载**: 自动扫描 `data/raw` 目录中的所有 `.md` 文件
2. **内容解析**: 解析 QA 对，提取问题、答案和 thinking 过程
3. **格式转换**: 转换为标准训练数据格式
4. **数据验证**: 验证数据格式和完整性
5. **数据分割**: 按比例分割训练集和验证集

## 训练配置

### 自动配置功能

程序会自动检测和配置：

- **GPU环境**: 自动检测可用GPU数量和内存
- **并行策略**: 根据GPU数量选择合适的并行策略
- **LoRA参数**: 根据GPU内存自动优化LoRA配置
- **批次大小**: 根据硬件资源调整批次大小

### 手动配置

可以通过修改 `demo_config.yaml` 来自定义配置：

```yaml
# 模型配置
model:
  name: "Qwen/Qwen3-4B-Thinking-2507"
  
# 训练配置
training:
  num_epochs: 2
  batch_size: 1
  learning_rate: 2e-4
  
# LoRA配置
lora:
  rank: 8
  alpha: 16
```

## 监控和日志

### 训练监控

- 实时进度跟踪
- 阶段状态更新
- 性能指标监控
- 错误检测和报告

### 日志文件

- `demo_log_*.log` - 详细执行日志
- `pipeline_report.json` - 流水线执行报告
- `training_report_*.json` - 训练总结报告

## 故障排除

### 常见问题

1. **数据文件未找到**
   ```
   错误: 在 data/raw 中未找到markdown文件
   解决: 确保 data/raw 目录存在且包含 .md 文件
   ```

2. **GPU检测失败**
   ```
   警告: 未检测到GPU，将使用CPU训练
   解决: 检查CUDA安装和GPU驱动
   ```

3. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 减小batch_size或启用gradient_checkpointing
   ```

### 调试模式

启用详细输出进行调试：

```bash
uv run python demo_comprehensive_finetuning.py --verbose
```

## 扩展功能

### 自定义数据处理

可以继承 `ComprehensiveFinetuningDemo` 类来自定义数据处理逻辑：

```python
class CustomDemo(ComprehensiveFinetuningDemo):
    def parse_qa_content(self, content, source_file):
        # 自定义解析逻辑
        return custom_examples
```

### 自定义训练配置

可以重写配置方法来使用自定义参数：

```python
def custom_configure_training(self):
    # 自定义配置逻辑
    return custom_configs
```

## 性能优化建议

1. **GPU利用率优化**
   - 使用多GPU并行训练
   - 启用混合精度训练
   - 调整批次大小

2. **内存优化**
   - 启用梯度检查点
   - 使用LoRA微调
   - 调整序列长度

3. **训练效率**
   - 使用余弦学习率调度
   - 启用梯度累积
   - 合理设置保存间隔

## 技术支持

如果遇到问题，请检查：

1. 日志文件中的详细错误信息
2. GPU和CUDA环境配置
3. 数据文件格式和完整性
4. 依赖包版本兼容性

## 更新日志

- v1.0.0 - 初始版本，支持基本的微调流水线
- 支持自动数据处理和格式转换
- 集成原生PyTorch训练框架
- 提供完整的监控和报告功能