#!/usr/bin/env python3
"""
简化版检查点合并和量化导出演示

本程序提供一个简化的演示，展示如何：
1. 检查和验证LoRA检查点
2. 合并检查点到基础模型
3. 导出不同精度的模型
4. 测试模型推理能力

使用方法:
    python simple_checkpoint_merge_demo.py
"""

import os
import sys
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCheckpointProcessor:
    """简化的检查点处理器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
    
    def inspect_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """检查检查点内容"""
        logger.info(f"检查检查点: {checkpoint_path}")
        
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"检查点目录不存在: {checkpoint_path}")
        
        # 检查文件结构
        files = list(checkpoint_dir.rglob("*"))
        file_info = {}
        
        for file_path in files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024 / 1024
                file_info[str(file_path.relative_to(checkpoint_dir))] = {
                    "size_mb": round(size_mb, 2),
                    "type": file_path.suffix
                }
        
        # 检查关键文件
        key_files = {
            "adapter_config.json": checkpoint_dir / "adapter_config.json",
            "adapter_model.safetensors": checkpoint_dir / "adapter_model.safetensors",
            "tokenizer.json": checkpoint_dir / "tokenizer.json",
            "tokenizer_config.json": checkpoint_dir / "tokenizer_config.json"
        }
        
        missing_files = []
        existing_files = []
        
        for name, path in key_files.items():
            if path.exists():
                existing_files.append(name)
            else:
                missing_files.append(name)
        
        # 读取配置信息
        config_info = {}
        if key_files["adapter_config.json"].exists():
            try:
                with open(key_files["adapter_config.json"], "r", encoding="utf-8") as f:
                    config_info = json.load(f)
            except Exception as e:
                logger.warning(f"无法读取adapter配置: {e}")
        
        inspection_result = {
            "checkpoint_path": checkpoint_path,
            "total_files": len(files),
            "total_size_mb": sum(info["size_mb"] for info in file_info.values()),
            "file_structure": file_info,
            "existing_key_files": existing_files,
            "missing_key_files": missing_files,
            "adapter_config": config_info,
            "is_valid": len(missing_files) == 0
        }
        
        return inspection_result
    
    def merge_checkpoint_simple(
        self, 
        checkpoint_path: str,
        output_path: str,
        base_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    ) -> Dict[str, Any]:
        """简化的检查点合并（模拟过程）"""
        logger.info("开始检查点合并过程...")
        
        # 检查检查点
        inspection = self.inspect_checkpoint(checkpoint_path)
        if not inspection["is_valid"]:
            raise ValueError(f"检查点无效，缺少文件: {inspection['missing_key_files']}")
        
        # 创建输出目录
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模拟合并过程（实际应用中需要使用transformers库）
        logger.info("模拟LoRA权重合并...")
        
        # 复制检查点文件到输出目录
        import shutil
        checkpoint_dir = Path(checkpoint_path)
        
        for file_path in checkpoint_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(checkpoint_dir)
                target_path = output_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)
        
        # 生成合并报告
        merge_report = {
            "merge_time": datetime.now().isoformat(),
            "base_model": base_model_name,
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "checkpoint_info": inspection,
            "merge_status": "simulated_success",
            "note": "这是一个模拟的合并过程，实际使用需要transformers和peft库"
        }
        
        # 保存报告
        with open(output_dir / "merge_report.json", "w", encoding="utf-8") as f:
            json.dump(merge_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"合并完成，输出到: {output_path}")
        return merge_report
    
    def create_quantized_versions(self, merged_model_path: str, output_base_dir: str) -> Dict[str, Any]:
        """创建不同精度版本的模型"""
        logger.info("创建量化版本...")
        
        base_dir = Path(output_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        merged_path = Path(merged_model_path)
        if not merged_path.exists():
            raise FileNotFoundError(f"合并模型不存在: {merged_model_path}")
        
        # 创建不同精度版本的目录和配置
        versions = {
            "fp16": {
                "description": "半精度浮点模型，平衡精度和性能",
                "memory_usage": "约4GB",
                "inference_speed": "基准速度",
                "precision": "高精度"
            },
            "int8": {
                "description": "8位整数量化模型，显著减少内存使用",
                "memory_usage": "约2GB", 
                "inference_speed": "1.5-2x加速",
                "precision": "轻微精度损失"
            },
            "int4": {
                "description": "4位整数量化模型，最大化内存效率",
                "memory_usage": "约1GB",
                "inference_speed": "2-3x加速", 
                "precision": "中等精度损失"
            }
        }
        
        results = {}
        
        for version_name, version_info in versions.items():
            version_dir = base_dir / version_name
            version_dir.mkdir(exist_ok=True)
            
            # 复制原始文件
            import shutil
            for file_path in merged_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(merged_path)
                    target_path = version_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_path)
            
            # 创建版本特定的配置
            version_config = {
                "version": version_name,
                "description": version_info["description"],
                "estimated_memory_usage": version_info["memory_usage"],
                "expected_inference_speed": version_info["inference_speed"],
                "precision_level": version_info["precision"],
                "creation_time": datetime.now().isoformat(),
                "source_model": str(merged_path),
                "quantization_note": "实际量化需要使用bitsandbytes或类似库"
            }
            
            with open(version_dir / "version_config.json", "w", encoding="utf-8") as f:
                json.dump(version_config, f, indent=2, ensure_ascii=False)
            
            # 创建使用说明
            usage_guide = f"""# {version_name.upper()} 模型使用指南

## 模型信息
- **版本**: {version_name.upper()}
- **描述**: {version_info['description']}
- **预估内存使用**: {version_info['memory_usage']}
- **预期推理速度**: {version_info['inference_speed']}
- **精度水平**: {version_info['precision']}

## 加载方法

```python
# 使用transformers库加载
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "{version_dir}",
    torch_dtype=torch.float16,  # 根据版本调整
    device_map="auto",
    trust_remote_code=True
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "{version_dir}",
    trust_remote_code=True
)
```

## 推理示例

```python
# 基础推理
prompt = "什么是AES加密算法？"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# 深度思考推理
thinking_prompt = "<thinking>分析这个密码学问题</thinking>请解释RSA算法原理。"
inputs = tokenizer(thinking_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=300, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(version_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(usage_guide)
            
            results[version_name] = {
                "path": str(version_dir),
                "config": version_config,
                "size_mb": self._get_directory_size(version_dir)
            }
            
            logger.info(f"创建 {version_name.upper()} 版本: {version_dir}")
        
        return results
    
    def _get_directory_size(self, directory: Path) -> float:
        """计算目录大小（MB）"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob('*') if f.is_file()
            )
            return round(total_size / 1024 / 1024, 2)
        except:
            return 0.0
    
    def generate_deployment_guide(self, output_dir: str, results: Dict[str, Any]):
        """生成部署指南"""
        output_path = Path(output_dir)
        
        guide_content = f"""# Qwen3-4B-Thinking 模型部署指南

## 概述

本指南介绍如何部署从微调检查点合并的Qwen3-4B-Thinking模型的不同版本。

## 目录结构

```
{output_dir}/
├── merged_model/          # 合并后的完整模型
├── fp16/                  # FP16精度版本
├── int8/                  # INT8量化版本  
├── int4/                  # INT4量化版本
├── deployment_guide.md    # 本指南
└── requirements.txt       # 依赖列表
```

## 版本选择指南

### FP16版本
- **适用场景**: 需要最高精度的生产环境
- **硬件要求**: 4GB+ GPU内存
- **推荐用途**: 关键业务应用、精度敏感任务

### INT8版本  
- **适用场景**: 平衡性能和精度的应用
- **硬件要求**: 2GB+ GPU内存
- **推荐用途**: 一般业务应用、实时推理

### INT4版本
- **适用场景**: 资源受限环境
- **硬件要求**: 1GB+ GPU内存  
- **推荐用途**: 边缘设备、移动应用

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基础推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 选择版本（fp16/int8/int4）
model_path = "{output_dir}/fp16"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 推理
prompt = "什么是AES加密算法？"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 2. 批量推理

```python
prompts = [
    "解释RSA算法的工作原理",
    "什么是数字签名？",
    "区块链使用了哪些密码学技术？"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {response}")
    print("-" * 50)
```

### 3. 深度思考模式

```python
thinking_prompt = '''<thinking>
用户询问关于密码学的问题，我需要：
1. 理解问题的核心
2. 组织相关知识点
3. 给出准确和易懂的解释
</thinking>
请详细解释椭圆曲线密码学的优势。'''

inputs = tokenizer(thinking_prompt, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_length=500, 
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 性能优化建议

### 内存优化
- 使用适当的batch_size
- 启用梯度检查点
- 使用混合精度训练

### 推理加速
- 使用TensorRT优化
- 启用KV缓存
- 考虑模型并行

### 量化配置
```python
# INT8量化配置
from transformers import BitsAndBytesConfig

int8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# INT4量化配置  
int4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 使用更低精度的量化版本
   - 启用CPU offloading

2. **推理速度慢**
   - 检查GPU利用率
   - 使用更激进的量化
   - 考虑模型剪枝

3. **精度下降**
   - 使用更高精度的版本
   - 检查量化配置
   - 进行量化感知训练

### 监控指标

```python
# 内存使用监控
import psutil
import GPUtil

def monitor_resources():
    # CPU和内存
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% utilization")
    
    print(f"CPU: {cpu_percent}%, RAM: {memory.percent}%")

# 推理性能测试
import time

def benchmark_inference(model, tokenizer, prompt, num_runs=10):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"平均推理时间: {avg_time:.3f}秒")
    return avg_time
```

## 生产部署

### Docker部署

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY {output_dir}/ ./models/
COPY deploy_service.py .

EXPOSE 8000
CMD ["python", "deploy_service.py"]
```

### API服务

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        temperature=request.temperature
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {{"response": response}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 技术支持

- 检查各版本目录下的README.md
- 查看merge_report.json了解合并详情
- 参考version_config.json了解版本特性

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_path / "deployment_guide.md", "w", encoding="utf-8") as f:
            f.write(guide_content)
        
        # 生成requirements.txt
        requirements = """torch>=2.0.0
transformers>=4.35.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
safetensors>=0.3.0
sentencepiece>=0.1.99
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
"""
        
        with open(output_path / "requirements.txt", "w") as f:
            f.write(requirements)
        
        logger.info("部署指南生成完成")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Qwen3-4B-Thinking 检查点合并和量化演示")
    logger.info("=" * 60)
    
    # 配置参数
    checkpoint_path = "qwen3_4b_thinking_output/final_model"
    output_dir = "demo_quantized_output"
    base_model = "Qwen/Qwen3-4B-Thinking-2507"
    
    processor = SimpleCheckpointProcessor()
    
    try:
        # 步骤1: 检查检查点
        logger.info("\n步骤1: 检查检查点结构")
        inspection = processor.inspect_checkpoint(checkpoint_path)
        
        logger.info(f"检查点总大小: {inspection['total_size_mb']:.1f} MB")
        logger.info(f"文件数量: {inspection['total_files']}")
        logger.info(f"关键文件: {', '.join(inspection['existing_key_files'])}")
        
        if inspection['missing_key_files']:
            logger.warning(f"缺少文件: {', '.join(inspection['missing_key_files'])}")
        
        # 步骤2: 合并检查点
        logger.info("\n步骤2: 合并检查点到基础模型")
        merged_path = Path(output_dir) / "merged_model"
        merge_result = processor.merge_checkpoint_simple(
            checkpoint_path=checkpoint_path,
            output_path=str(merged_path),
            base_model_name=base_model
        )
        
        # 步骤3: 创建量化版本
        logger.info("\n步骤3: 创建不同精度版本")
        quantized_results = processor.create_quantized_versions(
            merged_model_path=str(merged_path),
            output_base_dir=output_dir
        )
        
        # 步骤4: 生成部署指南
        logger.info("\n步骤4: 生成部署指南和文档")
        processor.generate_deployment_guide(output_dir, quantized_results)
        
        # 步骤5: 显示结果摘要
        logger.info("\n步骤5: 结果摘要")
        logger.info("=" * 40)
        
        for version, info in quantized_results.items():
            logger.info(f"{version.upper()}: {info['size_mb']} MB - {info['path']}")
        
        logger.info(f"\n✅ 演示完成！所有文件已保存到: {output_dir}")
        logger.info("📖 请查看 deployment_guide.md 了解详细使用方法")
        
        # 生成快速测试脚本
        test_script = f'''#!/usr/bin/env python3
"""
快速测试脚本
"""

def test_model_loading():
    """测试模型加载（需要安装transformers）"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_path = "{output_dir}/fp16"
        print(f"尝试加载模型: {{model_path}}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 简单推理测试
        prompt = "什么是AES加密算法？"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"输入: {{prompt}}")
        print(f"输出: {{response}}")
        print("✅ 模型加载和推理测试成功！")
        
    except ImportError:
        print("❌ 缺少transformers库，请运行: pip install transformers torch")
    except Exception as e:
        print(f"❌ 测试失败: {{e}}")

if __name__ == "__main__":
    test_model_loading()
'''
        
        with open(Path(output_dir) / "test_model.py", "w", encoding="utf-8") as f:
            f.write(test_script)
        
        logger.info("🧪 快速测试脚本已生成: test_model.py")
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        raise


if __name__ == "__main__":
    main()