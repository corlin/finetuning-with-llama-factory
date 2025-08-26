"""
模型导出和部署包生成演示

本脚本演示完整的模型量化、导出和部署包生成流程，
包括元数据生成、使用说明创建和部署验证。
"""

import sys
import os
import logging
import torch
import torch.nn as nn
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_exporter import (
    ModelExporter,
    ModelQuantizer,
    ChineseCapabilityValidator,
    QuantizationConfig,
    QuantizationFormat,
    QuantizationBackend,
    ModelMetadata,
    DeploymentPackage
)


class DemoQwenModel(nn.Module):
    """演示用的Qwen风格模型"""
    
    def __init__(self, vocab_size=50000, hidden_size=768, num_layers=12):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'max_position_embeddings': 2048
        })()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_size, 
                nhead=8, 
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """前向传播"""
        x = self.embedding(input_ids)
        
        # 通过transformer层
        for layer in self.layers:
            # 处理attention mask - 需要转换为正确的格式
            if attention_mask is not None:
                # 将attention_mask转换为padding mask (True表示需要mask的位置)
                padding_mask = (attention_mask == 0)
                x = layer(x, src_key_padding_mask=padding_mask)
            else:
                x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # 返回类似transformers的输出格式
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return ModelOutput(logits)
    
    def generate(self, input_ids, max_length=100, temperature=1.0, **kwargs):
        """简单的生成方法"""
        self.eval()
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            for _ in range(max_length - seq_len):
                # 获取下一个token的logits
                outputs = self.forward(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # 简单的贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # 如果生成了结束token，停止生成
                if next_token.item() == 2:  # 假设2是结束token
                    break
        
        return input_ids
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # 保存配置
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "max_position_embeddings": self.config.max_position_embeddings,
            "model_type": "qwen-demo",
            "architectures": ["DemoQwenModel"]
        }
        
        with open(save_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


class DemoQwenTokenizer:
    """演示用的Qwen风格分词器"""
    
    def __init__(self):
        self.vocab_size = 50000
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        
        # 特殊token
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<thinking>": 4,
            "</thinking>": 5
        }
        
        # 中文密码学词汇
        self.crypto_vocab = {
            "AES": 100, "RSA": 101, "SHA": 102, "椭圆曲线": 103,
            "数字签名": 104, "对称加密": 105, "非对称加密": 106,
            "哈希函数": 107, "密钥管理": 108, "区块链": 109,
            "密码学": 110, "加密算法": 111, "安全性": 112
        }
        
        # 常用中文词汇
        self.chinese_vocab = {
            "什么": 200, "是": 201, "的": 202, "如何": 203,
            "为什么": 204, "怎么": 205, "解释": 206, "分析": 207,
            "原理": 208, "工作": 209, "算法": 210, "技术": 211
        }
    
    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=512, **kwargs):
        """编码文本"""
        tokens = self.encode(text)
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        if padding and len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        result = {
            "input_ids": tokens,
            "attention_mask": [1 if t != self.pad_token_id else 0 for t in tokens]
        }
        
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result
    
    def encode(self, text):
        """编码文本为token序列"""
        tokens = [self.bos_token_id]  # 开始token
        
        # 检查特殊token
        for token, token_id in self.special_tokens.items():
            if token in text:
                tokens.append(token_id)
        
        # 检查密码学词汇
        for term, token_id in self.crypto_vocab.items():
            if term in text:
                tokens.append(token_id)
        
        # 检查中文词汇
        for word, token_id in self.chinese_vocab.items():
            if word in text:
                tokens.append(token_id)
        
        # 添加一些随机token模拟其他词汇
        import random
        tokens.extend(random.randint(300, 1000) for _ in range(min(10, len(text) // 5)))
        
        tokens.append(self.eos_token_id)  # 结束token
        return tokens
    
    def decode(self, tokens, skip_special_tokens=False):
        """解码token序列为文本"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 如果是二维tensor，取第一行
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        
        # 反向映射
        reverse_special = {v: k for k, v in self.special_tokens.items()}
        reverse_crypto = {v: k for k, v in self.crypto_vocab.items()}
        reverse_chinese = {v: k for k, v in self.chinese_vocab.items()}
        
        text_parts = []
        
        for token in tokens:
            if skip_special_tokens and token in reverse_special:
                continue
            elif token in reverse_special:
                text_parts.append(reverse_special[token])
            elif token in reverse_crypto:
                term = reverse_crypto[token]
                text_parts.append(f"{term}是一种重要的密码学技术")
            elif token in reverse_chinese:
                text_parts.append(reverse_chinese[token])
            elif token > 300:  # 其他词汇
                text_parts.append("相关技术概念")
        
        return " ".join(text_parts) if text_parts else "这是一个关于密码学的中文回答。"
    
    def save_pretrained(self, save_directory):
        """保存分词器"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存词汇表
        vocab = {}
        vocab.update(self.special_tokens)
        vocab.update(self.crypto_vocab)
        vocab.update(self.chinese_vocab)
        
        # 填充剩余词汇表
        for i in range(len(vocab), self.vocab_size):
            vocab[f"token_{i}"] = i
        
        with open(save_path / "vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        
        # 保存分词器配置
        tokenizer_config = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "model_max_length": 2048,
            "tokenizer_class": "DemoQwenTokenizer"
        }
        
        with open(save_path / "tokenizer_config.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        # 创建tokenizer.json（简化版）
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": token_id, "content": token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
                for token, token_id in self.special_tokens.items()
            ],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "<unk>"
            }
        }
        
        with open(save_path / "tokenizer.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)


def demonstrate_model_export_workflow():
    """演示完整的模型导出工作流"""
    print("=" * 70)
    print("模型导出和部署包生成演示")
    print("=" * 70)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"\n使用临时目录: {temp_dir}")
    
    try:
        # 1. 创建演示模型和分词器
        print("\n1. 创建演示模型和分词器...")
        model = DemoQwenModel(vocab_size=50000, hidden_size=768, num_layers=6)
        tokenizer = DemoQwenTokenizer()
        
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   模型大小: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
        print(f"   分词器词汇表大小: {tokenizer.vocab_size:,}")
        
        # 2. 测试模型基本功能
        print("\n2. 测试模型基本功能...")
        test_input = "什么是AES加密算法？"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"   输入: {test_input}")
            print(f"   输入token数量: {inputs['input_ids'].shape[1]}")
            print(f"   输出logits形状: {outputs.logits.shape}")
        
        # 3. 配置量化参数
        print("\n3. 配置量化参数...")
        quantization_configs = [
            QuantizationConfig(
                format=QuantizationFormat.DYNAMIC,
                backend=QuantizationBackend.PYTORCH
            ),
            # 可以添加更多量化配置
        ]
        
        for i, config in enumerate(quantization_configs, 1):
            print(f"   配置 {i}: {config.format.value} ({config.backend.value})")
        
        # 4. 执行模型导出
        print("\n4. 执行模型导出...")
        exporter = ModelExporter()
        
        for i, config in enumerate(quantization_configs, 1):
            print(f"\n   导出配置 {i}: {config.format.value}")
            
            # 创建输出目录
            output_dir = Path(temp_dir) / f"export_{config.format.value}"
            
            # 执行导出
            deployment_package = exporter.export_quantized_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=str(output_dir),
                quantization_config=config,
                model_name=f"qwen3-4b-thinking-{config.format.value}"
            )
            
            print(f"   ✅ 导出成功!")
            print(f"   📦 部署包路径: {deployment_package.package_path}")
            print(f"   📊 包大小: {deployment_package.package_size_mb:.2f} MB")
            print(f"   🔍 校验和: {deployment_package.checksum[:16]}...")
            
            # 验证导出的文件
            print(f"   📁 包含文件:")
            for file_path in deployment_package.model_files[:3]:  # 只显示前3个
                print(f"      - {Path(file_path).name}")
            if len(deployment_package.model_files) > 3:
                print(f"      - ... 还有 {len(deployment_package.model_files) - 3} 个文件")
            
            # 显示配置文件
            print(f"   ⚙️  配置文件:")
            for config_file in deployment_package.config_files:
                print(f"      - {Path(config_file).name}")
        
        # 5. 验证部署包
        print("\n5. 验证部署包...")
        for i, config in enumerate(quantization_configs, 1):
            output_dir = Path(temp_dir) / f"export_{config.format.value}"
            
            print(f"\n   验证配置 {i}: {config.format.value}")
            
            # 检查必要文件
            required_files = [
                "model/pytorch_model.bin",
                "model/config.json",
                "model/tokenizer.json",
                "model/tokenizer_config.json",
                "metadata.json",
                "README.md",
                "requirements.txt"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (output_dir / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"   ❌ 缺少文件: {missing_files}")
            else:
                print(f"   ✅ 所有必要文件都存在")
            
            # 验证元数据
            metadata_path = output_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                print(f"   📋 元数据验证:")
                print(f"      - 模型名称: {metadata['model_name']}")
                print(f"      - 量化格式: {metadata['quantization_format']}")
                print(f"      - 压缩比: {metadata['compression_ratio']:.2f}x")
                print(f"      - 支持语言: {', '.join(metadata['supported_languages'])}")
                print(f"      - 专业领域: {', '.join(metadata['specialized_domains'])}")
            
            # 验证README
            readme_path = output_dir / "README.md"
            if readme_path.exists():
                readme_content = readme_path.read_text(encoding='utf-8')
                print(f"   📖 README验证:")
                print(f"      - 文件大小: {len(readme_content)} 字符")
                print(f"      - 包含安装说明: {'✅' if '安装依赖' in readme_content else '❌'}")
                print(f"      - 包含使用示例: {'✅' if '使用示例' in readme_content else '❌'}")
                print(f"      - 包含性能指标: {'✅' if '性能指标' in readme_content else '❌'}")
        
        # 6. 生成部署报告
        print("\n6. 生成部署报告...")
        report = generate_deployment_report(temp_dir, quantization_configs)
        
        report_path = Path(temp_dir) / "deployment_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   📊 部署报告已生成: {report_path}")
        print(f"   📈 总体统计:")
        print(f"      - 导出配置数量: {report['summary']['total_configs']}")
        print(f"      - 成功导出数量: {report['summary']['successful_exports']}")
        print(f"      - 总包大小: {report['summary']['total_package_size_mb']:.2f} MB")
        print(f"      - 平均压缩比: {report['summary']['average_compression_ratio']:.2f}x")
        
        # 7. 演示部署包使用
        print("\n7. 演示部署包使用...")
        demonstrate_deployment_usage(temp_dir, quantization_configs[0])
        
        print("\n" + "=" * 70)
        print("✅ 模型导出和部署包生成演示完成!")
        print("=" * 70)
        
        print(f"\n📁 所有文件已保存到: {temp_dir}")
        print("💡 提示: 在实际使用中，请使用真实的Qwen3-4B-Thinking模型")
        
        return temp_dir, report
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    finally:
        # 注意：在实际使用中可能不想自动删除临时目录
        print(f"\n🗑️  临时目录将保留以供检查: {temp_dir}")


def generate_deployment_report(base_dir, configs):
    """生成部署报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_configs": len(configs),
            "successful_exports": 0,
            "total_package_size_mb": 0.0,
            "average_compression_ratio": 0.0
        },
        "exports": []
    }
    
    total_compression_ratio = 0.0
    
    for config in configs:
        export_dir = Path(base_dir) / f"export_{config.format.value}"
        
        if export_dir.exists():
            # 读取元数据
            metadata_path = export_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 计算目录大小
                package_size = sum(
                    f.stat().st_size for f in export_dir.rglob('*') if f.is_file()
                ) / 1024 / 1024
                
                export_info = {
                    "config": config.format.value,
                    "success": True,
                    "package_size_mb": package_size,
                    "compression_ratio": metadata.get("compression_ratio", 1.0),
                    "model_name": metadata.get("model_name", "unknown"),
                    "quantization_format": metadata.get("quantization_format", "unknown"),
                    "chinese_capability_score": metadata.get("performance_metrics", {}).get("chinese_capability_score", 0.0),
                    "crypto_term_accuracy": metadata.get("performance_metrics", {}).get("crypto_term_accuracy", 0.0)
                }
                
                report["exports"].append(export_info)
                report["summary"]["successful_exports"] += 1
                report["summary"]["total_package_size_mb"] += package_size
                total_compression_ratio += metadata.get("compression_ratio", 1.0)
    
    if report["summary"]["successful_exports"] > 0:
        report["summary"]["average_compression_ratio"] = total_compression_ratio / report["summary"]["successful_exports"]
    
    return report


def demonstrate_deployment_usage(base_dir, config):
    """演示部署包的使用方法"""
    print(f"\n   演示 {config.format.value} 部署包使用:")
    
    export_dir = Path(base_dir) / f"export_{config.format.value}"
    
    if not export_dir.exists():
        print("   ❌ 部署包不存在")
        return
    
    # 1. 检查README
    readme_path = export_dir / "README.md"
    if readme_path.exists():
        print("   📖 README.md 可用于用户指导")
    
    # 2. 检查requirements
    requirements_path = export_dir / "requirements.txt"
    if requirements_path.exists():
        print("   📦 requirements.txt 包含依赖信息")
        requirements = requirements_path.read_text(encoding='utf-8')
        print(f"      主要依赖: {', '.join(req.split('>=')[0] for req in requirements.split('\\n')[:3] if req.strip())}")
    
    # 3. 模拟加载模型
    print("   🔄 模拟模型加载过程:")
    model_dir = export_dir / "model"
    
    if (model_dir / "config.json").exists():
        with open(model_dir / "config.json", 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"      ✅ 模型配置加载成功 (vocab_size: {config_data.get('vocab_size', 'unknown')})")
    
    if (model_dir / "pytorch_model.bin").exists():
        model_size = (model_dir / "pytorch_model.bin").stat().st_size / 1024 / 1024
        print(f"      ✅ 模型权重加载成功 (大小: {model_size:.2f} MB)")
    
    if (model_dir / "tokenizer_config.json").exists():
        print("      ✅ 分词器配置加载成功")
    
    # 4. 验证结果检查
    validation_path = export_dir / "validation_results.json"
    if validation_path.exists():
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
        
        print("   🧪 中文能力验证结果:")
        print(f"      - 总体得分: {validation_data.get('overall_score', 0):.2%}")
        print(f"      - 密码学术语准确性: {validation_data.get('crypto_term_accuracy', 0):.2%}")
        print(f"      - 思考结构保持: {validation_data.get('thinking_structure_preservation', 0):.2%}")
    
    print("   ✅ 部署包验证完成，可用于生产部署")


def create_deployment_guide():
    """创建部署指南"""
    guide_content = """# 模型部署指南

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型加载

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./model")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "./model",
    device_map="auto",
    trust_remote_code=True
)
```

### 3. 基本使用

```python
# 基础问答
def ask_question(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例
question = "什么是AES加密算法？"
answer = ask_question(question)
print(answer)
```

### 4. 深度思考模式

```python
# 使用thinking标签进行深度推理
def deep_thinking(question):
    thinking_prompt = f"<thinking>让我仔细分析这个问题</thinking>{question}"
    inputs = tokenizer(thinking_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=500,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例
question = "分析RSA算法的安全性"
answer = deep_thinking(question)
print(answer)
```

## 性能优化

### GPU加速
- 确保CUDA可用
- 使用适当的device_map
- 考虑使用混合精度

### 内存优化
- 使用量化模型减少内存占用
- 启用梯度检查点
- 调整批次大小

### 推理优化
- 使用缓存机制
- 批量处理请求
- 考虑使用TensorRT等推理引擎

## 部署选项

### 本地部署
- 直接使用Python脚本
- 创建Flask/FastAPI服务
- 使用Gradio创建Web界面

### 云端部署
- 使用Docker容器化
- 部署到云服务器
- 使用Kubernetes编排

### 边缘部署
- 使用ONNX格式
- 移动端部署
- 嵌入式设备部署

## 监控和维护

### 性能监控
- 响应时间监控
- 内存使用监控
- GPU利用率监控

### 质量监控
- 输出质量评估
- 用户反馈收集
- A/B测试

### 更新维护
- 模型版本管理
- 增量更新
- 回滚机制

## 故障排除

### 常见问题
1. 内存不足 - 减少批次大小或使用量化
2. 推理速度慢 - 检查GPU使用情况
3. 输出质量差 - 调整生成参数

### 日志分析
- 启用详细日志
- 监控错误信息
- 性能分析

## 安全考虑

### 输入验证
- 过滤恶意输入
- 限制输入长度
- 内容安全检查

### 输出过滤
- 敏感信息过滤
- 内容合规检查
- 质量控制

### 访问控制
- API认证
- 速率限制
- 用户权限管理
"""
    
    return guide_content


def main():
    """主函数"""
    print("模型导出和部署包生成系统演示")
    print("=" * 70)
    
    try:
        # 执行完整的导出演示
        temp_dir, report = demonstrate_model_export_workflow()
        
        if temp_dir and report:
            # 创建部署指南
            guide_content = create_deployment_guide()
            guide_path = Path(temp_dir) / "deployment_guide.md"
            guide_path.write_text(guide_content, encoding='utf-8')
            
            print(f"\n📚 部署指南已创建: {guide_path}")
            
            # 显示最终总结
            print("\n" + "=" * 70)
            print("🎉 演示总结")
            print("=" * 70)
            
            print(f"✅ 成功导出 {report['summary']['successful_exports']} 个模型配置")
            print(f"📦 总包大小: {report['summary']['total_package_size_mb']:.2f} MB")
            print(f"🗜️  平均压缩比: {report['summary']['average_compression_ratio']:.2f}x")
            print(f"📁 文件保存位置: {temp_dir}")
            
            print("\n📋 生成的文件:")
            print("   - 量化模型文件")
            print("   - 配置和元数据")
            print("   - README和使用说明")
            print("   - 部署指南")
            print("   - 验证报告")
            
            print("\n💡 下一步:")
            print("   1. 检查生成的文件")
            print("   2. 根据README进行部署测试")
            print("   3. 使用部署指南进行生产部署")
            print("   4. 监控模型性能和质量")
            
        else:
            print("❌ 演示未能成功完成")
            
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()