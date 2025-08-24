# 故障排除指南

## 常见问题和解决方案

### 安装和环境问题

#### 1. CUDA不可用
**症状**: 运行`list-gpus`命令显示"未检测到GPU设备"

**可能原因**:
- CUDA驱动未安装或版本不兼容
- PyTorch未正确安装CUDA版本
- 环境变量配置错误

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装PyTorch CUDA版本
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 2. 依赖包冲突
**症状**: 导入模块时出现版本冲突错误

**解决方案**:
```bash
# 使用uv创建干净环境
uv venv --python 3.12
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

uv sync
```

#### 3. 权限问题 (Linux/macOS)
**症状**: 无法创建文件或目录

**解决方案**:
```bash
# 确保有写权限
chmod +x src/cli_tools_simple.py
sudo chown -R $USER:$USER ./
```

### 配置文件问题

#### 1. 配置验证失败
**症状**: `validate-config`命令报告配置错误

**常见错误和修复**:

```yaml
# 错误: 学习率必须是正数
training:
  learning_rate: 0  # 错误
  learning_rate: 2e-4  # 正确

# 错误: LoRA rank必须是正整数
lora:
  rank: 0  # 错误
  rank: 8  # 正确

# 错误: 数据分割比例总和应为1.0
data:
  train_split_ratio: 0.8
  eval_split_ratio: 0.3  # 错误，总和>1
  eval_split_ratio: 0.2  # 正确
```

#### 2. 模型路径错误
**症状**: 无法加载模型

**解决方案**:
```yaml
model:
  model_name: "Qwen/Qwen3-4B-Thinking-2507"  # 使用正确的模型名称
  trust_remote_code: true  # 必须设置为true
```

### 数据格式问题

#### 1. JSON格式错误
**症状**: `inspect-data`命令解析失败

**常见错误**:
```json
// 错误: 缺少逗号
{
  "instruction": "问题"
  "output": "回答"
}

// 正确
{
  "instruction": "问题",
  "output": "回答"
}
```

**验证JSON格式**:
```bash
# 使用jq验证JSON格式
cat data.json | jq .

# 或使用Python验证
python -c "import json; json.load(open('data.json'))"
```

#### 2. 编码问题
**症状**: 中文字符显示乱码

**解决方案**:
```bash
# 确保文件使用UTF-8编码
file -bi data.json

# 转换编码
iconv -f GBK -t UTF-8 data.json > data_utf8.json
```

#### 3. Thinking标签格式错误
**症状**: Thinking数据解析失败

**错误示例**:
```json
{
  "thinking": "<thinking>思考过程"  // 缺少结束标签
}
```

**正确格式**:
```json
{
  "thinking": "<thinking>思考过程</thinking>"
}
```

### 训练过程问题

#### 1. 内存不足 (OOM)
**症状**: 训练过程中出现CUDA OOM错误

**解决方案**:
```yaml
# 方案1: 减小批次大小
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

# 方案2: 启用内存优化
training:
  gradient_checkpointing: true
  fp16: true  # 或 bf16: true

# 方案3: 减小LoRA参数
lora:
  rank: 4
  alpha: 8

# 方案4: 减小序列长度
training:
  max_seq_length: 1024
```

#### 2. 训练速度慢
**症状**: 训练进度缓慢

**优化方案**:
```yaml
# 启用混合精度
training:
  bf16: true

# 优化数据加载
training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true

# 使用多GPU
parallel:
  strategy: "data_parallel"
  world_size: 2
  enable_distributed: true
```

#### 3. 损失不收敛
**症状**: 训练损失不下降或震荡

**解决方案**:
```yaml
# 调整学习率
training:
  learning_rate: 1e-4  # 降低学习率
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1

# 调整LoRA参数
lora:
  rank: 16  # 增加表达能力
  alpha: 32
  dropout: 0.05  # 减少dropout

# 检查数据质量
# 确保训练数据质量高，格式正确
```

#### 4. 梯度爆炸
**症状**: 损失突然变为NaN或无穷大

**解决方案**:
```yaml
training:
  max_grad_norm: 0.5  # 降低梯度裁剪阈值
  learning_rate: 5e-5  # 降低学习率
```

### 多GPU问题

#### 1. 分布式训练初始化失败
**症状**: 多GPU训练无法启动

**检查步骤**:
```bash
# 检查GPU可见性
nvidia-smi

# 检查NCCL
python -c "import torch; print(torch.distributed.is_nccl_available())"

# 检查端口占用
netstat -an | grep 29500
```

**解决方案**:
```yaml
parallel:
  master_port: 29501  # 更换端口
  backend: "gloo"     # 尝试不同后端
```

#### 2. GPU内存不均衡
**症状**: 某些GPU内存使用率高，其他GPU空闲

**解决方案**:
```yaml
parallel:
  strategy: "data_parallel"  # 确保使用数据并行
  enable_dynamic_load_balancing: true
```

### 性能问题

#### 1. GPU利用率低
**症状**: GPU利用率不足50%

**优化方案**:
```yaml
# 增加批次大小
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4

# 优化数据加载
training:
  dataloader_num_workers: 8
  preprocessing_num_workers: 8
```

#### 2. 磁盘I/O瓶颈
**症状**: 训练过程中频繁等待数据加载

**解决方案**:
- 使用SSD存储训练数据
- 增加数据加载进程数
- 预处理数据到内存

### 平台特定问题

#### Windows问题

**1. 路径分隔符问题**
```python
# 错误
path = "data/train.json"

# 正确
path = os.path.join("data", "train.json")
# 或
path = Path("data") / "train.json"
```

**2. 长路径问题**
```bash
# 启用长路径支持
git config --system core.longpaths true
```

#### Linux问题

**1. 共享内存不足**
```bash
# 增加共享内存
sudo mount -o remount,size=8G /dev/shm
```

**2. 文件描述符限制**
```bash
# 增加文件描述符限制
ulimit -n 65536
```

#### macOS问题

**1. MPS后端问题**
```python
# 禁用MPS后端
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### 调试技巧

#### 1. 启用详细日志
```bash
# 启用详细输出
uv run python src/cli_tools_simple.py --verbose train data.json config.yaml

# 设置日志级别
export PYTHONPATH=src
export LOG_LEVEL=DEBUG
```

#### 2. 使用干运行模式
```bash
# 验证配置而不实际训练
uv run python src/cli_tools_simple.py train data.json config.yaml --dry-run
```

#### 3. 检查系统状态
```bash
# 生成系统报告
uv run python test_gpu_detection.py > system_report.txt
```

#### 4. 监控资源使用
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控内存使用
htop

# 监控磁盘I/O
iotop
```

### 错误代码参考

| 错误代码 | 含义 | 解决方案 |
|---------|------|----------|
| CUDA_ERROR_OUT_OF_MEMORY | GPU内存不足 | 减小批次大小，启用内存优化 |
| CUDA_ERROR_INVALID_DEVICE | GPU设备无效 | 检查GPU可见性和驱动 |
| RuntimeError: NCCL error | 分布式通信错误 | 检查网络和NCCL配置 |
| FileNotFoundError | 文件未找到 | 检查文件路径和权限 |
| JSONDecodeError | JSON格式错误 | 验证JSON格式 |
| ValidationError | 配置验证失败 | 检查配置文件格式 |

### 获取帮助

#### 1. 查看日志
```bash
# 训练日志
tail -f logs/training.log

# 系统日志
journalctl -u nvidia-persistenced
```

#### 2. 收集诊断信息
```bash
# 生成诊断报告
uv run python -c "
from gpu_utils import GPUDetector
detector = GPUDetector()
print(detector.generate_system_report())
" > diagnostic_report.txt
```

#### 3. 社区支持
- 检查GitHub Issues
- 查看文档和FAQ
- 提交详细的错误报告

### 预防措施

#### 1. 定期备份
```bash
# 备份配置和数据
cp config.yaml config.yaml.bak
cp -r data data.bak
```

#### 2. 版本控制
```bash
# 使用git跟踪配置变更
git add config.yaml
git commit -m "Update training config"
```

#### 3. 环境隔离
```bash
# 使用虚拟环境
uv venv training_env
source training_env/bin/activate
```

#### 4. 资源监控
- 设置GPU温度告警
- 监控磁盘空间
- 定期检查内存使用

通过遵循这些故障排除指南，大多数常见问题都可以得到解决。如果问题仍然存在，请收集详细的错误信息和系统状态，以便进一步诊断。