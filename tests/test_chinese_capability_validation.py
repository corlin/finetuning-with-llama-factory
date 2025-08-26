"""
中文处理能力验证专项测试

测试量化模型的中文处理能力，特别是密码学专业术语的准确性验证。
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.model_exporter import (
    ChineseCapabilityValidator,
    ModelQuantizer,
    QuantizationConfig,
    QuantizationFormat,
    QuantizationBackend
)


class CryptoAwareModel(nn.Module):
    """密码学感知的演示模型"""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50000, 768)
        self.linear = nn.Linear(768, 50000)
        
        # 密码学术语映射
        self.crypto_responses = {
            "AES": "AES是高级加密标准，是一种对称加密算法。",
            "RSA": "RSA是一种非对称加密算法，基于大数分解难题。",
            "SHA-256": "SHA-256是一种密码学哈希函数，输出256位摘要。",
            "椭圆曲线": "椭圆曲线密码学提供了高安全性和较短的密钥长度。",
            "数字签名": "数字签名用于验证数据完整性和身份认证。"
        }
    
    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        logits = self.linear(x.mean(dim=1))
        
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)
    
    def generate(self, input_ids, max_length=100, **kwargs):
        """生成包含密码学术语的响应"""
        batch_size, seq_len = input_ids.shape
        
        # 简单的生成逻辑：根据输入返回相应的密码学响应
        generated_tokens = []
        
        # 模拟生成过程
        for _ in range(max_length - seq_len):
            # 生成一些随机token
            generated_tokens.append(torch.randint(1, 1000, (batch_size, 1)))
        
        if generated_tokens:
            generated = torch.cat(generated_tokens, dim=1)
            return torch.cat([input_ids, generated], dim=1)
        
        return input_ids


class CryptoAwareTokenizer:
    """密码学感知的分词器"""
    
    def __init__(self):
        self.vocab_size = 50000
        self.pad_token_id = 0
        
        # 密码学术语词典
        self.crypto_vocab = {
            "AES": 1001,
            "RSA": 1002,
            "SHA-256": 1003,
            "椭圆曲线": 1004,
            "数字签名": 1005,
            "对称加密": 1006,
            "非对称加密": 1007,
            "哈希函数": 1008,
            "密钥管理": 1009,
            "区块链": 1010,
            "密码学": 1011,
            "加密算法": 1012,
            "安全性": 1013,
            "密钥": 1014,
            "算法": 1015
        }
        
        # 思考标签
        self.thinking_vocab = {
            "<thinking>": 2001,
            "</thinking>": 2002
        }
        
        # 中文常用词
        self.chinese_vocab = {
            "什么": 3001,
            "是": 3002,
            "的": 3003,
            "如何": 3004,
            "为什么": 3005,
            "怎么": 3006,
            "解释": 3007,
            "分析": 3008,
            "原理": 3009,
            "工作": 3010
        }
    
    def __call__(self, text, **kwargs):
        tokens = self.encode(text)
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1] * len(tokens)])
        }
    
    def encode(self, text):
        """编码文本，优先识别密码学术语"""
        tokens = []
        
        # 检查密码学术语
        for term, token_id in self.crypto_vocab.items():
            if term in text:
                tokens.append(token_id)
        
        # 检查思考标签
        for tag, token_id in self.thinking_vocab.items():
            if tag in text:
                tokens.append(token_id)
        
        # 检查中文词汇
        for word, token_id in self.chinese_vocab.items():
            if word in text:
                tokens.append(token_id)
        
        # 如果没有匹配到特殊词汇，添加基础token
        if not tokens:
            tokens = [1, 2, 3, 4, 5]
        
        return tokens
    
    def decode(self, tokens, skip_special_tokens=False):
        """解码token为中文文本"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 反向映射
        reverse_crypto = {v: k for k, v in self.crypto_vocab.items()}
        reverse_thinking = {v: k for k, v in self.thinking_vocab.items()}
        reverse_chinese = {v: k for k, v in self.chinese_vocab.items()}
        
        text_parts = []
        
        for token in tokens:
            if token in reverse_crypto:
                term = reverse_crypto[token]
                if term == "AES":
                    text_parts.append("AES是一种对称加密算法，广泛应用于数据保护。")
                elif term == "RSA":
                    text_parts.append("RSA是一种非对称加密算法，基于大数分解的数学难题。")
                elif term == "SHA-256":
                    text_parts.append("SHA-256是一种密码学哈希函数，能够生成256位的摘要。")
                elif term == "椭圆曲线":
                    text_parts.append("椭圆曲线密码学提供了高安全性和较短的密钥长度。")
                elif term == "数字签名":
                    text_parts.append("数字签名技术确保了数据的完整性和不可否认性。")
                else:
                    text_parts.append(f"关于{term}的专业解释。")
            elif token in reverse_thinking:
                text_parts.append(reverse_thinking[token])
            elif token in reverse_chinese:
                text_parts.append(reverse_chinese[token])
            elif token > 1000:  # 高token值表示生成的内容
                text_parts.append("这是一个专业的密码学回答。")
        
        if not text_parts:
            text_parts.append("这是一个中文回答。")
        
        return " ".join(text_parts)


class TestChineseCapabilityValidation:
    """中文处理能力验证测试"""
    
    @pytest.fixture
    def validator(self):
        """创建验证器"""
        return ChineseCapabilityValidator()
    
    @pytest.fixture
    def crypto_model(self):
        """创建密码学感知模型"""
        return CryptoAwareModel()
    
    @pytest.fixture
    def crypto_tokenizer(self):
        """创建密码学感知分词器"""
        return CryptoAwareTokenizer()
    
    def test_crypto_terms_recognition(self, validator, crypto_model, crypto_tokenizer):
        """测试密码学术语识别"""
        accuracy = validator._test_crypto_terms(crypto_model, crypto_tokenizer)
        
        # 使用密码学感知的分词器应该有更高的准确性
        assert accuracy > 0.5
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_chinese_encoding_with_crypto_terms(self, validator, crypto_model, crypto_tokenizer):
        """测试包含密码学术语的中文编码"""
        accuracy = validator._test_chinese_encoding(crypto_model, crypto_tokenizer)
        
        # 密码学感知的分词器应该能更好地处理专业术语
        assert accuracy >= 0.0  # 至少不应该出错
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_thinking_structure_with_crypto_content(self, validator, crypto_model, crypto_tokenizer):
        """测试包含密码学内容的思考结构保持"""
        score = validator._test_thinking_structure(crypto_model, crypto_tokenizer)
        
        # 应该能正确保持thinking标签
        assert score >= 0.5
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_crypto_specific_test_cases(self, validator, crypto_model, crypto_tokenizer):
        """测试密码学专项测试用例"""
        crypto_test_cases = [
            {
                "input": "AES加密算法的密钥长度有哪些？",
                "expected_keywords": ["128位", "192位", "256位", "对称加密"],
                "category": "算法参数"
            },
            {
                "input": "RSA算法为什么被认为是安全的？",
                "expected_keywords": ["大数分解", "数学难题", "非对称加密"],
                "category": "安全性分析"
            },
            {
                "input": "<thinking>分析椭圆曲线密码学的优势</thinking>ECC相比RSA有什么优点？",
                "expected_keywords": ["密钥长度", "计算效率", "安全性"],
                "category": "技术对比"
            }
        ]
        
        results = validator.validate_chinese_capability(
            crypto_model, crypto_tokenizer, crypto_test_cases
        )
        
        assert "overall_score" in results
        assert "test_results" in results
        assert len(results["test_results"]) == len(crypto_test_cases)
        
        # 密码学专项测试应该有较好的表现
        assert results["overall_score"] > 0.3
        
        # 检查每个测试用例的结果
        for result in results["test_results"]:
            assert "input" in result
            assert "response" in result
            assert "score" in result
            assert "category" in result
            assert "success" in result
            assert result["success"] is True
    
    def test_quantization_impact_on_crypto_terms(self, validator, crypto_model, crypto_tokenizer):
        """测试量化对密码学术语处理的影响"""
        # 创建量化器
        quantizer = ModelQuantizer()
        
        # 量化配置
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        # 量化模型
        quantized_model, quant_result = quantizer.quantize_model(
            crypto_model, crypto_tokenizer, config
        )
        
        assert quant_result.success is True
        
        # 验证原始模型的密码学术语处理能力
        original_accuracy = validator._test_crypto_terms(crypto_model, crypto_tokenizer)
        
        # 验证量化模型的密码学术语处理能力
        quantized_accuracy = validator._test_crypto_terms(quantized_model, crypto_tokenizer)
        
        # 量化后的准确性不应该显著下降
        accuracy_diff = abs(quantized_accuracy - original_accuracy)
        assert accuracy_diff < 0.2  # 允许20%的差异
    
    def test_comprehensive_chinese_crypto_validation(self, validator, crypto_model, crypto_tokenizer):
        """测试综合中文密码学能力验证"""
        # 执行完整的验证
        results = validator.validate_chinese_capability(crypto_model, crypto_tokenizer)
        
        # 检查所有必要的指标
        required_metrics = [
            "overall_score",
            "chinese_encoding_accuracy",
            "crypto_term_accuracy",
            "thinking_structure_preservation",
            "semantic_coherence"
        ]
        
        for metric in required_metrics:
            assert metric in results
            assert isinstance(results[metric], float)
            assert 0.0 <= results[metric] <= 1.0
        
        # 密码学感知模型应该在密码学术语方面表现更好
        assert results["crypto_term_accuracy"] > 0.5
        
        # 思考结构应该保持良好
        assert results["thinking_structure_preservation"] >= 0.5
        
        # 总体得分应该合理
        assert results["overall_score"] > 0.3
    
    def test_performance_comparison_report(self, validator, crypto_model, crypto_tokenizer):
        """测试性能对比报告生成"""
        # 创建量化器
        quantizer = ModelQuantizer()
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        # 量化模型
        quantized_model, _ = quantizer.quantize_model(crypto_model, crypto_tokenizer, config)
        
        # 验证原始模型和量化模型
        original_results = validator.validate_chinese_capability(crypto_model, crypto_tokenizer)
        quantized_results = validator.validate_chinese_capability(quantized_model, crypto_tokenizer)
        
        # 生成对比报告
        comparison_report = {
            "original_model": original_results,
            "quantized_model": quantized_results,
            "performance_diff": {
                metric: quantized_results[metric] - original_results[metric]
                for metric in ["overall_score", "chinese_encoding_accuracy", 
                              "crypto_term_accuracy", "thinking_structure_preservation"]
            }
        }
        
        # 验证报告结构
        assert "original_model" in comparison_report
        assert "quantized_model" in comparison_report
        assert "performance_diff" in comparison_report
        
        # 检查性能差异
        for metric, diff in comparison_report["performance_diff"].items():
            assert isinstance(diff, float)
            # 性能差异应该在合理范围内
            assert abs(diff) < 0.3
    
    def test_custom_crypto_vocabulary_validation(self, validator):
        """测试自定义密码学词汇验证"""
        # 创建包含更多密码学术语的分词器
        extended_tokenizer = CryptoAwareTokenizer()
        extended_tokenizer.crypto_vocab.update({
            "ECDSA": 1020,
            "HMAC": 1021,
            "PBKDF2": 1022,
            "Diffie-Hellman": 1023,
            "量子密码学": 1024,
            "同态加密": 1025,
            "零知识证明": 1026
        })
        
        model = CryptoAwareModel()
        
        # 测试扩展词汇的识别
        accuracy = validator._test_crypto_terms(model, extended_tokenizer)
        
        # 扩展词汇应该提高识别准确性
        assert accuracy >= 0.0
        assert isinstance(accuracy, float)
    
    def test_multilingual_crypto_terms(self, validator, crypto_model):
        """测试多语言密码学术语处理"""
        # 创建支持中英文混合的分词器
        multilingual_tokenizer = CryptoAwareTokenizer()
        multilingual_tokenizer.crypto_vocab.update({
            "Advanced Encryption Standard": 1030,
            "Public Key Infrastructure": 1031,
            "Certificate Authority": 1032,
            "密钥交换": 1033,
            "身份认证": 1034,
            "访问控制": 1035
        })
        
        # 测试中英文混合的密码学术语
        test_cases = [
            {
                "input": "AES (Advanced Encryption Standard) 是什么？",
                "expected_keywords": ["对称加密", "高级加密标准"],
                "category": "中英混合"
            },
            {
                "input": "PKI (Public Key Infrastructure) 的作用是什么？",
                "expected_keywords": ["公钥基础设施", "数字证书"],
                "category": "缩写解释"
            }
        ]
        
        results = validator.validate_chinese_capability(
            crypto_model, multilingual_tokenizer, test_cases
        )
        
        assert results["overall_score"] >= 0.0
        assert len(results["test_results"]) == len(test_cases)


if __name__ == "__main__":
    pytest.main([__file__])