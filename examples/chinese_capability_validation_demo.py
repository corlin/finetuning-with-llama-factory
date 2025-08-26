"""
中文处理能力验证演示

本脚本演示如何使用ChineseCapabilityValidator验证量化模型的中文处理能力，
特别是密码学专业术语的准确性和思考结构的保持。
"""

import sys
import os
import logging
import torch
import torch.nn as nn
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_exporter import (
    ChineseCapabilityValidator,
    ModelQuantizer,
    QuantizationConfig,
    QuantizationFormat,
    QuantizationBackend
)


class DemoModel(nn.Module):
    """演示用的简单模型"""
    
    def __init__(self, vocab_size=50000, hidden_size=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, 2048, batch_first=True),
            num_layers=6
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        
        # 返回类似transformers的输出格式
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return ModelOutput(logits)
    
    def generate(self, input_ids, max_length=100, **kwargs):
        """简单的生成方法"""
        batch_size, seq_len = input_ids.shape
        
        # 生成一些随机token作为演示
        new_tokens = max_length - seq_len
        if new_tokens > 0:
            generated = torch.randint(1, 1000, (batch_size, new_tokens))
            return torch.cat([input_ids, generated], dim=1)
        
        return input_ids


class DemoTokenizer:
    """演示用的简单分词器"""
    
    def __init__(self):
        self.vocab_size = 50000
        self.pad_token_id = 0
        
        # 中文密码学术语映射
        self.crypto_terms = {
            "AES": 100,
            "RSA": 101,
            "SHA-256": 102,
            "椭圆曲线": 103,
            "数字签名": 104,
            "对称加密": 105,
            "非对称加密": 106,
            "哈希函数": 107,
            "密钥管理": 108,
            "区块链": 109
        }
        
        # 思考标签
        self.thinking_tokens = {
            "<thinking>": 200,
            "</thinking>": 201
        }
    
    def __call__(self, text, **kwargs):
        """编码文本"""
        tokens = self.encode(text)
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1] * len(tokens)])
        }
    
    def encode(self, text):
        """简单的编码实现"""
        tokens = []
        
        # 检查密码学术语
        for term, token_id in self.crypto_terms.items():
            if term in text:
                tokens.append(token_id)
        
        # 检查思考标签
        for tag, token_id in self.thinking_tokens.items():
            if tag in text:
                tokens.append(token_id)
        
        # 添加一些基础token
        tokens.extend([1, 2, 3, 4, 5])
        
        return tokens
    
    def decode(self, tokens, skip_special_tokens=False):
        """简单的解码实现"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 根据token生成相应的中文文本
        text_parts = []
        
        for token in tokens:
            if token in [100, 101, 102]:  # AES, RSA, SHA-256
                text_parts.append("这是一个关于密码学算法的专业回答。")
            elif token in [103, 104, 105, 106]:  # 其他密码学术语
                text_parts.append("涉及密码学专业概念的详细解释。")
            elif token == 200:  # <thinking>
                text_parts.append("<thinking>")
            elif token == 201:  # </thinking>
                text_parts.append("</thinking>")
            elif token > 500:  # 高token值表示中文内容
                text_parts.append("包含中文密码学术语的专业回答，如AES加密算法、RSA非对称加密等。")
        
        if not text_parts:
            text_parts.append("这是一个中文回答，包含密码学相关内容。")
        
        return " ".join(text_parts)


def demonstrate_chinese_capability_validation():
    """演示中文处理能力验证"""
    print("=" * 60)
    print("中文处理能力验证演示")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建演示模型和分词器
    print("\n1. 创建演示模型和分词器...")
    model = DemoModel()
    tokenizer = DemoTokenizer()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"分词器词汇表大小: {tokenizer.vocab_size:,}")
    
    # 创建验证器
    print("\n2. 创建中文能力验证器...")
    validator = ChineseCapabilityValidator()
    
    # 执行中文能力验证
    print("\n3. 执行中文处理能力验证...")
    validation_results = validator.validate_chinese_capability(model, tokenizer)
    
    # 显示验证结果
    print("\n4. 验证结果:")
    print("-" * 40)
    print(f"总体得分: {validation_results['overall_score']:.2%}")
    print(f"中文编码准确性: {validation_results['chinese_encoding_accuracy']:.2%}")
    print(f"密码学术语准确性: {validation_results['crypto_term_accuracy']:.2%}")
    print(f"思考结构保持: {validation_results['thinking_structure_preservation']:.2%}")
    print(f"语义连贯性: {validation_results['semantic_coherence']:.2%}")
    
    # 显示详细测试结果
    print("\n5. 详细测试结果:")
    print("-" * 40)
    for i, result in enumerate(validation_results['test_results'], 1):
        print(f"\n测试用例 {i}:")
        print(f"  输入: {result['input'][:50]}...")
        print(f"  响应: {result['response'][:100]}...")
        print(f"  得分: {result['score']:.2%}")
        print(f"  类别: {result['category']}")
        print(f"  成功: {'是' if result['success'] else '否'}")
    
    return validation_results


def demonstrate_quantization_comparison():
    """演示量化前后的中文能力对比"""
    print("\n" + "=" * 60)
    print("量化前后中文能力对比演示")
    print("=" * 60)
    
    # 创建原始模型
    print("\n1. 创建原始模型...")
    original_model = DemoModel()
    tokenizer = DemoTokenizer()
    
    # 创建量化器和验证器
    quantizer = ModelQuantizer()
    validator = ChineseCapabilityValidator()
    
    # 量化模型
    print("\n2. 量化模型...")
    config = QuantizationConfig(
        format=QuantizationFormat.DYNAMIC,
        backend=QuantizationBackend.PYTORCH
    )
    
    quantized_model, quant_result = quantizer.quantize_model(
        original_model, tokenizer, config
    )
    
    print(f"量化结果: {'成功' if quant_result.success else '失败'}")
    if quant_result.success:
        print(f"压缩比: {quant_result.compression_ratio:.2f}x")
        print(f"内存减少: {quant_result.memory_reduction:.2%}")
        print(f"推理加速: {quant_result.inference_speedup:.2f}x")
    
    # 验证原始模型的中文能力
    print("\n3. 验证原始模型中文能力...")
    original_results = validator.validate_chinese_capability(original_model, tokenizer)
    
    # 验证量化模型的中文能力
    print("\n4. 验证量化模型中文能力...")
    quantized_results = validator.validate_chinese_capability(quantized_model, tokenizer)
    
    # 对比结果
    print("\n5. 中文能力对比:")
    print("-" * 50)
    print(f"{'指标':<20} {'原始模型':<12} {'量化模型':<12} {'差异':<10}")
    print("-" * 50)
    
    metrics = [
        ("总体得分", "overall_score"),
        ("中文编码准确性", "chinese_encoding_accuracy"),
        ("密码学术语准确性", "crypto_term_accuracy"),
        ("思考结构保持", "thinking_structure_preservation"),
        ("语义连贯性", "semantic_coherence")
    ]
    
    for name, key in metrics:
        original_val = original_results[key]
        quantized_val = quantized_results[key]
        diff = quantized_val - original_val
        
        print(f"{name:<20} {original_val:<12.2%} {quantized_val:<12.2%} {diff:+.2%}")
    
    # 分析结果
    print("\n6. 分析结果:")
    print("-" * 40)
    
    overall_diff = quantized_results['overall_score'] - original_results['overall_score']
    if abs(overall_diff) < 0.05:
        print("✅ 量化对中文处理能力影响很小，在可接受范围内")
    elif overall_diff > 0:
        print("🎉 量化后中文处理能力有所提升")
    else:
        print("⚠️  量化对中文处理能力有一定影响，需要进一步优化")
    
    # 专项分析
    crypto_diff = quantized_results['crypto_term_accuracy'] - original_results['crypto_term_accuracy']
    if crypto_diff >= 0:
        print("✅ 密码学术语处理能力保持良好")
    else:
        print("⚠️  密码学术语处理能力有所下降")
    
    thinking_diff = quantized_results['thinking_structure_preservation'] - original_results['thinking_structure_preservation']
    if thinking_diff >= 0:
        print("✅ 思考结构保持能力良好")
    else:
        print("⚠️  思考结构保持能力有所下降")
    
    return original_results, quantized_results


def demonstrate_custom_test_cases():
    """演示自定义测试用例"""
    print("\n" + "=" * 60)
    print("自定义中文密码学测试用例演示")
    print("=" * 60)
    
    # 创建模型和验证器
    model = DemoModel()
    tokenizer = DemoTokenizer()
    validator = ChineseCapabilityValidator()
    
    # 定义自定义测试用例
    custom_test_cases = [
        {
            "input": "请解释AES-256加密算法的工作原理。",
            "expected_keywords": ["对称加密", "分组密码", "256位密钥"],
            "category": "算法原理"
        },
        {
            "input": "<thinking>我需要分析这个密码学问题的安全性</thinking>RSA-2048的安全强度如何？",
            "expected_keywords": ["非对称加密", "2048位", "安全强度"],
            "category": "安全分析"
        },
        {
            "input": "区块链中的哈希函数有什么作用？",
            "expected_keywords": ["SHA-256", "工作量证明", "数据完整性"],
            "category": "应用场景"
        },
        {
            "input": "椭圆曲线密码学相比RSA有什么优势？",
            "expected_keywords": ["密钥长度", "计算效率", "安全性"],
            "category": "技术对比"
        }
    ]
    
    print(f"\n1. 使用 {len(custom_test_cases)} 个自定义测试用例...")
    
    # 执行验证
    results = validator.validate_chinese_capability(model, tokenizer, custom_test_cases)
    
    print("\n2. 自定义测试结果:")
    print("-" * 50)
    
    for i, result in enumerate(results['test_results'], 1):
        print(f"\n测试用例 {i}: {result['category']}")
        print(f"  问题: {result['input']}")
        print(f"  回答: {result['response']}")
        print(f"  得分: {result['score']:.2%}")
        print(f"  状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
    
    print(f"\n3. 总体评估:")
    print(f"  平均得分: {results['overall_score']:.2%}")
    print(f"  成功率: {sum(1 for r in results['test_results'] if r['success']) / len(results['test_results']):.2%}")
    
    return results


def main():
    """主函数"""
    print("中文处理能力验证系统演示")
    print("=" * 60)
    
    try:
        # 基础中文能力验证
        basic_results = demonstrate_chinese_capability_validation()
        
        # 量化前后对比
        original_results, quantized_results = demonstrate_quantization_comparison()
        
        # 自定义测试用例
        custom_results = demonstrate_custom_test_cases()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        print("\n总结:")
        print(f"- 基础验证得分: {basic_results['overall_score']:.2%}")
        print(f"- 量化后得分: {quantized_results['overall_score']:.2%}")
        print(f"- 自定义测试得分: {custom_results['overall_score']:.2%}")
        
        print("\n建议:")
        print("1. 在实际使用中，建议使用真实的Qwen3-4B-Thinking模型")
        print("2. 可以根据具体需求定制测试用例")
        print("3. 量化前后的对比有助于选择合适的量化策略")
        print("4. 定期验证模型的中文处理能力，确保质量稳定")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()