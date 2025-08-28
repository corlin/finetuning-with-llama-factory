# Qwen3-4B-Thinking 检查点合并和量化导出演示

本目录包含三个演示程序，展示如何将微调检查点合并到基础模型并导出多种格式的量化模型。

## 演示程序概览

### 1. `demo_checkpoint_merge_and_export.py` - 完整功能演示
**功能最全面的演示程序**

- ✅ 完整的LoRA检查点合并流程
- ✅ 多格式量化导出（INT8, INT4, GPTQ, FP16）
- ✅ 中文处理能力验证
- ✅ 自动生成部署包和文档
- ✅ 性能基准测试
- ⚠️ 需要完整的依赖环境

**适用场景**: 生产环境部署，需要完整功能

### 2. `simple_checkpoint_merge_demo.py` - 简化演示
**最容易运行的演示程序**

- ✅ 检查点结构分析
- ✅ 模拟合并过程（无需复杂依赖）
- ✅ 生成多精度版本目录结构
- ✅ 详细的使用文档和部署指南
- ✅ 快速测试脚本
- 🔧 适合学习和理解流程

**适用场景**: 学习演示，环境受限时使用

### 3. `practical_model_export_demo.py` - 实用导出
**平衡功能和实用性**

- ✅ 使用现有自研模块
- ✅ 智能降级处理（依赖不可用时）
- ✅ 实际量化导出
- ✅ 中文能力验证
- ✅ 生成使用示例
- 🎯 推荐日常使用

**适用场景**: 日常开发，实际模型导出

## 快速开始

### 环境要求

**基础要求**:
```bash
pip install torch>=2.0.0
```

**完整功能**:
```bash
pip install torch>=2.0.0 transformers>=4.35.0 peft>=0.6.0 bitsandbytes>=0.41.0 accelerate>=0.20.0
```

### 运行演示

#### 方法1: 推荐使用实用导出演示
```bash
python practical_model_export_demo.py
```

#### 方法2: 简化演示（无需复杂依赖）
```bash
python simple_checkpoint_merge_demo.py
```

#### 方法3: 完整功能演示
```bash
python demo_checkpoint_merge_and_export.py
```

### 自定义参数

所有演示程序都支持命令行参数：

```bash
# 指定检查点路径
python practical_model_export_demo.py --checkpoint_path "your/checkpoint/path"

# 指定基础模型
python practical_model_export_demo.py --base_model "Qwen/Qwen3-4B-Thinking-2507"

# 指定输出目录
python practical_model_export_demo.py --output_dir "custom_output"
```

## 检查点要求

### LoRA检查点结构
```
qwen3_4b_thinking_output/final_model/
├── adapter_config.json      # LoRA配置
├── adapter_model.safetensors # LoRA权重
├── tokenizer.json           # 分词器
├── tokenizer_config.json    # 分词器配置
├── special_tokens_map.json  # 特殊token映射
└── ...                      # 其他配置文件
```

### 完整模型检查点结构
```
checkpoint_directory/
├── pytorch_model.bin        # 模型权重
├── config.json             # 模型配置
├── tokenizer.json          # 分词器
└── ...                     # 其他文件
```

## 输出结果

### 目录结构
```
output_directory/
├── merged_model/           # 合并后的完整模型
├── fp16/                  # FP16精度版本
├── int8/                  # INT8量化版本
├── int4/                  # INT4量化版本
├── export_report.json     # 导出报告
├── deployment_guide.md    # 部署指南
├── usage_examples.py      # 使用示例
├── requirements.txt       # 依赖列表
└── test_model.py         # 快速测试脚本
```

### 量化格式说明

| 格式 | 内存使用 | 推理速度 | 精度保持 | 适用场景 |
|------|----------|----------|----------|----------|
| FP16 | ~8GB | 基准 | 100% | 高精度要求 |
| INT8 | ~4GB | 1.5-2x | ~98% | 平衡性能 |
| INT4 | ~2GB | 2-3x | ~95% | 资源受限 |

## 使用导出的模型

### 基础加载
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载FP16模型
model = AutoModelForCausalLM.from_pretrained(
    "output_directory/fp16",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "output_directory/fp16",
    trust_remote_code=True
)
```

### 量化模型加载
```python
from transformers import BitsAndBytesConfig

# INT8量化
int8_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "output_directory/int8",
    quantization_config=int8_config,
    device_map="auto",
    trust_remote_code=True
)

# INT4量化
int4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "output_directory/int4",
    quantization_config=int4_config,
    device_map="auto",
    trust_remote_code=True
)
```

### 推理示例
```python
# 基础对话
prompt = "什么是AES加密算法？"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# 深度思考推理
thinking_prompt = "<thinking>我需要分析这个密码学问题</thinking>请解释RSA算法的工作原理。"
inputs = tokenizer(thinking_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=300, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 故障排除

### 常见问题

#### 1. 依赖库缺失
```bash
# 错误: ModuleNotFoundError: No module named 'transformers'
pip install transformers peft torch bitsandbytes accelerate
```

#### 2. CUDA内存不足
```bash
# 解决方案1: 使用更小的量化格式
python practical_model_export_demo.py --formats int4

# 解决方案2: 使用CPU
export CUDA_VISIBLE_DEVICES=""
python practical_model_export_demo.py
```

#### 3. 检查点路径错误
```bash
# 检查路径是否存在
ls -la qwen3_4b_thinking_output/final_model/

# 使用绝对路径
python practical_model_export_demo.py --checkpoint_path "/absolute/path/to/checkpoint"
```

#### 4. 权限问题
```bash
# 确保输出目录有写权限
chmod 755 output_directory
```

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 性能优化

#### 内存优化
- 使用更激进的量化（INT4而非INT8）
- 启用CPU offloading
- 减小batch_size

#### 速度优化
- 使用GPU推理
- 启用混合精度
- 考虑TensorRT优化

## 技术细节

### LoRA合并原理
1. 加载基础模型权重
2. 加载LoRA适配器权重
3. 将LoRA权重合并到基础模型
4. 保存合并后的完整模型

### 量化技术
- **INT8**: 使用BitsAndBytes库进行8位量化
- **INT4**: 使用NF4量化算法
- **GPTQ**: 基于梯度的后训练量化
- **动态量化**: PyTorch原生动态量化

### 中文能力验证
- 中文字符编码准确性测试
- 密码学专业术语保持测试
- 思考结构完整性验证
- 语义连贯性评估

## 扩展功能

### 自定义量化配置
```python
from src.model_exporter import QuantizationConfig, QuantizationFormat, QuantizationBackend

custom_config = QuantizationConfig(
    format=QuantizationFormat.INT4,
    backend=QuantizationBackend.BITSANDBYTES,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 批量处理
```python
# 批量导出多个检查点
checkpoints = [
    "checkpoint1/path",
    "checkpoint2/path", 
    "checkpoint3/path"
]

for i, checkpoint in enumerate(checkpoints):
    output_dir = f"batch_export_{i}"
    # 运行导出...
```

### API服务部署
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    # 使用导出的模型进行推理
    pass
```

## 贡献指南

### 添加新的量化格式
1. 在`QuantizationFormat`枚举中添加新格式
2. 在`ModelQuantizer`中实现量化逻辑
3. 添加相应的配置类
4. 更新文档和测试

### 改进中文验证
1. 扩展测试用例集
2. 添加新的评估指标
3. 优化验证算法
4. 增加专业领域测试

## 许可证

本演示程序遵循项目主许可证。

## 支持

如有问题，请：
1. 查看生成的`export_report.json`
2. 检查各目录下的README文件
3. 运行`test_model.py`进行诊断
4. 提交Issue并附上错误日志

---

**最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: v1.0.0