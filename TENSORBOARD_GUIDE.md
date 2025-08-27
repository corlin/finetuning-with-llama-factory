# TensorBoard 训练可视化指南

## 概述

本项目已集成 TensorBoard 支持，可以实时监控 Qwen3-4B-Thinking 模型的训练过程，包括损失曲线、学习率变化、GPU 使用情况、内存监控等多种指标。

## 功能特性

### 🎯 核心训练指标
- **训练损失 (Training/Loss)**: 实时监控模型训练损失变化
- **学习率 (Training/Learning_Rate)**: 跟踪学习率调度变化
- **梯度范数 (Training/Gradient_Norm)**: 监控梯度裁剪和梯度爆炸
- **训练轮次 (Training/Epoch)**: 当前训练进度

### 💾 内存和资源监控
- **GPU 内存使用 (Memory/GPU_Allocated_MB)**: 实时 GPU 内存占用
- **GPU 内存利用率 (Memory/GPU_Utilization_Percent)**: GPU 内存使用百分比
- **内存压力等级 (Memory/Pressure_Level)**: 内存压力预警

### 🖥️ 多GPU监控
- **GPU 利用率 (GPU_X/Utilization_Percent)**: 每个 GPU 的计算利用率
- **GPU 内存使用 (GPU_X/Memory_Usage_Percent)**: 每个 GPU 的内存使用率
- **GPU 温度 (GPU_X/Temperature_C)**: GPU 温度监控

### 📊 数据集统计
- **样本统计**: 总样本数、平均文本长度、Thinking 样本比例
- **质量指标**: 中文文本质量、密码学术语复杂度
- **难度分布**: 训练样本难度级别分布
- **术语分布**: 密码学专业术语分布统计

### 🎯 收敛监控
- **收敛评分 (Monitoring/Convergence_Score)**: 模型收敛程度评估
- **损失趋势 (Monitoring/Loss_Trend)**: 损失变化趋势分析

### ⚙️ 训练配置
- **批次大小 (Config/Batch_Size)**: 当前批次大小
- **序列长度 (Config/Sequence_Length)**: 输入序列长度
- **梯度累积步数 (Config/Gradient_Accumulation_Steps)**: 梯度累积配置

## 使用方法

### 1. 启动训练
```bash
# 使用 uv 启动训练（自动启用 TensorBoard 日志）
uv run python direct_finetuning_with_existing_modules.py
```

### 2. 启动 TensorBoard
训练开始后，在新的终端窗口中启动 TensorBoard：

```bash
# 启动 TensorBoard（指向训练输出目录）
tensorboard --logdir=qwen3_4b_thinking_output/tensorboard_logs

# 或者指定端口
tensorboard --logdir=qwen3_4b_thinking_output/tensorboard_logs --port=6006
```

### 3. 查看训练曲线
在浏览器中打开：
```
http://localhost:6006
```

## TensorBoard 界面说明

### 📈 SCALARS 标签页
- **Training**: 核心训练指标（损失、学习率、梯度范数）
- **Memory**: 内存使用监控
- **GPU_X**: 各GPU的详细监控
- **Dataset**: 数据集统计信息
- **Monitoring**: 收敛和趋势分析
- **Config**: 训练配置参数
- **Epoch**: 每轮训练统计
- **Final_XXX**: 训练完成后的最终统计

### 📝 TEXT 标签页
- **Dataset/Summary**: 数据集统计摘要
- **Training/Epoch_Summary**: 每轮训练摘要
- **Training/Final_Summary**: 最终训练摘要
- **Config/Training_Settings**: 训练配置详情

### 📊 HISTOGRAMS 标签页
- **Model/Weights_Distribution**: 模型权重分布（如果启用）

## 日志文件结构

```
qwen3_4b_thinking_output/
├── tensorboard_logs/
│   └── qwen3_4b_thinking_YYYYMMDD_HHMMSS/
│       ├── events.out.tfevents.xxx
│       └── ...
├── training_logs/
├── checkpoints/
└── training_statistics.json
```

## 实用技巧

### 🔍 监控要点
1. **损失曲线**: 应该呈现下降趋势，如果震荡过大可能需要调整学习率
2. **GPU 利用率**: 应该保持在 80% 以上，过低说明存在瓶颈
3. **内存使用**: 监控是否接近 OOM，及时调整批次大小
4. **收敛评分**: 逐渐增加表示模型正在收敛

### 📊 性能优化
- 如果 GPU 利用率低，考虑增加批次大小或减少数据加载时间
- 如果内存压力高，减少批次大小或启用梯度检查点
- 如果损失不下降，检查学习率设置和梯度范数

### 🚨 异常检测
- **梯度爆炸**: 梯度范数突然增大
- **内存泄漏**: 内存使用持续增长
- **训练停滞**: 损失长时间不变化
- **GPU 过热**: 温度持续超过 85°C

## 测试 TensorBoard 功能

运行测试脚本验证 TensorBoard 集成：

```bash
# 测试 TensorBoard 基础功能
uv run test_tensorboard_integration.py

# 查看测试结果
tensorboard --logdir=test_tensorboard_logs
```

## 故障排除

### 常见问题

1. **TensorBoard 无法启动**
   ```bash
   # 检查是否安装了 tensorboard
   pip install tensorboard
   
   # 或使用 uv 安装
   uv add tensorboard
   ```

2. **浏览器无法访问**
   - 检查防火墙设置
   - 尝试使用不同端口：`tensorboard --logdir=... --port=6007`
   - 使用 `--host=0.0.0.0` 允许外部访问

3. **日志文件过大**
   - 定期清理旧的日志文件
   - 调整日志记录频率（修改 `logging_steps` 参数）

4. **数据不更新**
   - 刷新浏览器页面
   - 检查训练是否正在运行
   - 确认日志目录路径正确

## 高级配置

### 自定义日志频率
在 `DirectTrainingConfig` 中调整：
```python
logging_steps: int = 5  # 每5步记录一次日志
save_steps: int = 50    # 每50步保存一次检查点
```

### 添加自定义指标
在训练循环中添加：
```python
if self.tensorboard_writer:
    self.tensorboard_writer.add_scalar('Custom/My_Metric', value, step)
```

## 总结

TensorBoard 集成为 Qwen3-4B-Thinking 模型训练提供了全面的可视化监控能力，帮助您：

- 🎯 实时监控训练进度和性能
- 🔍 及早发现和解决训练问题  
- 📊 分析数据集质量和分布
- ⚡ 优化训练配置和资源使用
- 📈 跟踪模型收敛和性能指标

通过 TensorBoard，您可以更好地理解和优化训练过程，提高模型训练的效率和效果。