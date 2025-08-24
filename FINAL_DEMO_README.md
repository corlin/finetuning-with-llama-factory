# 基于LLaMA Factory的密码学微调演示程序

这是一个完整的演示程序，展示了如何基于当前已完成的功能，使用 `data/raw` 数据，通过 LLaMA Factory 进行 Qwen3-4B-Thinking 模型微调。

## 🎯 演示成果

✅ **成功处理了76个密码学训练样例**
- 100%的样例包含thinking过程
- 涵盖初级到中级难度
- 提取了15个核心密码学术语

✅ **自动检测和配置硬件环境**
- 检测到2个RTX 5060 Ti GPU（共32GB显存）
- 自动配置数据并行策略
- 优化LoRA参数（rank=64, alpha=64）

✅ **生成完整的训练配置**
- LLaMA Factory兼容的YAML配置
- 数据集信息和格式转换
- 训练脚本和执行指导

## 📊 数据分析结果

### 训练数据统计
- **总样例数**: 76个
- **Thinking样例**: 76个（100%）
- **平均指令长度**: 20.1字符
- **平均输出长度**: 106.4字符

### 难度分布
- **初级（BEGINNER）**: 64个（84.2%）
- **中级（INTERMEDIATE）**: 12个（15.8%）

### 热门密码学术语
1. 完整性 (14次)
2. 机密性 (8次)
3. 加密 (8次)
4. 不可否认性 (6次)
5. 公钥 (6次)

## 🖥️ 硬件配置

### GPU环境
- **GPU数量**: 2个
- **GPU型号**: NVIDIA GeForce RTX 5060 Ti
- **单GPU内存**: 15.9GB
- **总GPU内存**: 31.9GB
- **计算能力**: 12.0
- **并行策略**: 数据并行

### 性能预估
- **预计训练时间**: 约7分钟
- **内存使用**: 约12.4GB GPU内存/GPU
- **磁盘空间**: 约2-5GB

## ⚙️ 训练配置

### 模型配置
- **基础模型**: Qwen/Qwen3-4B-Thinking-2507
- **微调方法**: LoRA
- **LoRA rank**: 64
- **LoRA alpha**: 64
- **序列长度**: 2048

### 训练参数
- **训练轮数**: 2
- **学习率**: 2e-4
- **批次大小**: 1
- **梯度累积**: 4步
- **混合精度**: bf16
- **优化器**: AdamW

## 📁 生成的文件

### 数据文件
```
final_demo_output/
├── data/
│   ├── crypto_qa_dataset_train.json    # 训练数据（68样例）
│   ├── crypto_qa_dataset_val.json      # 验证数据（8样例）
│   └── dataset_info.json               # 数据集信息
```

### 配置文件
```
├── configs/
│   └── llamafactory_config_*.yaml      # LLaMA Factory配置
├── train.py                            # 训练脚本
└── comprehensive_report_*.json         # 综合分析报告
```

## 🚀 开始训练

### 前置条件
1. 安装 LLaMA Factory
   ```bash
   pip install llamafactory
   ```

2. 检查CUDA环境（GPU训练）
   ```bash
   nvidia-smi
   ```

3. 确认磁盘空间充足（约5GB）

### 训练命令

#### 方法1：使用CLI（推荐）
```bash
cd final_demo_output
llamafactory-cli train configs/llamafactory_config_*.yaml
```

#### 方法2：使用Python脚本
```bash
cd final_demo_output
python train.py
```

### 监控训练
```bash
# 启动TensorBoard监控
tensorboard --logdir model_output/logs

# 查看训练日志
tail -f model_output/logs/train.log
```

## 📈 训练监控

### 关键指标
- **训练损失**: 监控模型学习进度
- **验证损失**: 防止过拟合
- **学习率**: 观察学习率调度
- **GPU利用率**: 确保硬件充分利用

### 检查点管理
- 每100步保存一次检查点
- 最多保留3个检查点
- 自动保存最佳模型

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案：
   - 减小batch_size
   - 启用gradient_checkpointing
   - 降低sequence_length
   ```

2. **数据加载错误**
   ```
   解决方案：
   - 检查dataset_info.json路径
   - 验证数据文件格式
   - 确认文件权限
   ```

3. **模型加载失败**
   ```
   解决方案：
   - 检查网络连接
   - 验证模型名称
   - 清理缓存目录
   ```

## 📋 演示程序使用

### 快速演示
```bash
# 运行快速验证
uv run python run_demo.py

# 运行简化版演示
uv run python demo_simple_finetuning.py

# 运行完整演示（推荐）
uv run python demo_final.py
```

### 自定义参数
```bash
# 指定数据目录和输出目录
uv run python demo_final.py --data-dir custom_data --output-dir custom_output

# 启用详细输出
uv run python demo_final.py --verbose
```

## 🎯 下一步建议

### 模型优化
1. **调整LoRA参数**: 尝试不同的rank和alpha值
2. **优化学习率**: 使用学习率搜索
3. **增加训练数据**: 扩充密码学QA数据集
4. **多轮训练**: 进行多阶段微调

### 评估和部署
1. **模型评估**: 使用专业密码学测试集
2. **量化优化**: 进行INT8/INT4量化
3. **部署准备**: 转换为推理格式
4. **服务化**: 构建API服务

## 📚 技术文档

### 相关文档
- [LLaMA Factory官方文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen模型文档](https://github.com/QwenLM/Qwen)
- [LoRA微调指南](https://arxiv.org/abs/2106.09685)

### 项目结构
```
finetune_withlf/
├── src/                    # 核心功能模块
├── data/raw/              # 原始训练数据
├── demo_*.py              # 演示程序
├── final_demo_output/     # 演示输出
└── README.md              # 项目说明
```

## 🏆 演示亮点

1. **完整的端到端流程**: 从数据加载到训练配置生成
2. **智能硬件检测**: 自动适配GPU环境和并行策略
3. **专业数据处理**: 针对密码学领域的thinking数据处理
4. **详细分析报告**: 提供全面的数据和硬件分析
5. **生产就绪配置**: 生成可直接使用的训练配置

---

**演示程序执行成功！** 🎉

这个演示程序展示了一个完整的、生产就绪的模型微调流水线，从数据处理到训练配置，为密码学领域的AI模型微调提供了完整的解决方案。