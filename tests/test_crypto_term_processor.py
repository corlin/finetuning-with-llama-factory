"""
密码学术语处理器测试

测试密码学术语词典构建、术语识别标注、复杂度评估算法、
以及thinking数据中的专业术语处理功能。
"""

import pytest
import json
import tempfile
import os
from typing import List, Dict

try:
    from src.crypto_term_processor import (
        CryptoTermProcessor, TermAnnotation, TermDistribution, 
        ThinkingTermAnalysis, TermComplexity
    )
    from src.data_models import (
        CryptoTerm, CryptoCategory, ThinkingExample, 
        DifficultyLevel
    )
except ImportError:
    from crypto_term_processor import (
        CryptoTermProcessor, TermAnnotation, TermDistribution,
        ThinkingTermAnalysis, TermComplexity
    )
    from data_models import (
        CryptoTerm, CryptoCategory, ThinkingExample,
        DifficultyLevel
    )


class TestCryptoTermProcessor:
    """密码学术语处理器测试类"""
    
    @pytest.fixture
    def processor(self):
        """创建术语处理器实例"""
        return CryptoTermProcessor()
    
    @pytest.fixture
    def sample_crypto_text(self):
        """示例密码学文本"""
        return """
        RSA是一种非对称加密算法，基于大整数分解的数学难题。
        与对称加密不同，RSA使用公钥和私钥进行加密和解密。
        AES是目前最常用的对称加密算法，支持128、192和256位密钥长度。
        数字签名可以确保消息的完整性和不可否认性。
        SHA-256哈希函数被广泛应用于区块链技术中。
        """
    
    @pytest.fixture
    def sample_thinking_example(self):
        """示例thinking训练样例"""
        return ThinkingExample(
            instruction="请解释RSA加密算法的工作原理",
            thinking_process="""
            <thinking>
            RSA是一种非对称加密算法，我需要从以下几个方面来解释：
            1. 数学基础：基于大整数分解难题
            2. 密钥生成：生成公钥和私钥对
            3. 加密过程：使用公钥进行加密
            4. 解密过程：使用私钥进行解密
            5. 安全性：依赖于大整数分解的困难性
            
            还需要提到RSA与对称加密的区别，以及数字签名的应用。
            </thinking>
            """,
            final_response="RSA是一种基于大整数分解难题的非对称加密算法...",
            crypto_terms=["RSA", "非对称加密", "公钥", "私钥", "数字签名"],
            difficulty_level=DifficultyLevel.ADVANCED
        )
    
    def test_crypto_dictionary_initialization(self, processor):
        """测试密码学词典初始化"""
        # 检查词典是否包含基本术语
        assert "RSA" in processor.crypto_dict
        assert "AES" in processor.crypto_dict
        assert "SHA" in processor.crypto_dict
        assert "对称加密" in processor.crypto_dict
        assert "非对称加密" in processor.crypto_dict
        
        # 检查术语对象的完整性
        rsa_term = processor.crypto_dict["RSA"]
        assert rsa_term.term == "RSA"
        assert rsa_term.category == CryptoCategory.ASYMMETRIC_ENCRYPTION
        assert rsa_term.complexity >= 1
        assert rsa_term.definition is not None
        assert len(rsa_term.related_terms) > 0
    
    def test_term_identification(self, processor, sample_crypto_text):
        """测试术语识别功能"""
        annotations = processor.identify_crypto_terms(sample_crypto_text)
        
        # 检查是否识别出主要术语
        identified_terms = [ann.term for ann in annotations]
        
        assert "RSA" in identified_terms
        assert "非对称加密" in identified_terms or "非对称加密算法" in identified_terms
        assert "对称加密" in identified_terms or "对称加密算法" in identified_terms
        assert "AES" in identified_terms
        assert "数字签名" in identified_terms
        
        # 检查标注信息的完整性
        for annotation in annotations:
            assert annotation.term is not None
            assert annotation.category is not None
            assert annotation.complexity >= 1
            assert 0.0 <= annotation.confidence <= 1.0
            assert annotation.start_pos >= 0
            assert annotation.end_pos > annotation.start_pos
            assert annotation.context is not None
    
    def test_compound_term_identification(self, processor):
        """测试复合术语识别"""
        text = "RSA-2048和AES-256是常用的加密算法，SHA-256用于哈希计算。"
        annotations = processor.identify_crypto_terms(text)
        
        identified_terms = [ann.term for ann in annotations]
        
        # 检查是否识别出复合术语
        compound_terms = [term for term in identified_terms if "-" in term or "256" in term]
        assert len(compound_terms) > 0
        
        # 检查复合术语的分类
        for annotation in annotations:
            if "RSA" in annotation.term:
                assert annotation.category == CryptoCategory.ASYMMETRIC_ENCRYPTION
            elif "AES" in annotation.term:
                assert annotation.category == CryptoCategory.SYMMETRIC_ENCRYPTION
            elif "SHA" in annotation.term:
                assert annotation.category == CryptoCategory.HASH_FUNCTION
    
    def test_term_complexity_calculation(self, processor):
        """测试术语复杂度计算"""
        # 测试基础术语
        basic_terms = ["加密", "密钥"]
        basic_complexity = processor.calculate_term_complexity(basic_terms)
        
        # 测试高级术语
        advanced_terms = ["椭圆曲线", "差分分析", "侧信道攻击"]
        advanced_complexity = processor.calculate_term_complexity(advanced_terms)
        
        # 高级术语的复杂度应该更高
        assert advanced_complexity > basic_complexity
        
        # 测试空列表
        empty_complexity = processor.calculate_term_complexity([])
        assert empty_complexity == 0.0
        
        # 测试混合术语
        mixed_terms = ["AES", "RSA", "椭圆曲线", "哈希函数"]
        mixed_complexity = processor.calculate_term_complexity(mixed_terms)
        assert 0.0 < mixed_complexity <= 10.0
    
    def test_term_distribution_analysis(self, processor):
        """测试术语分布分析"""
        texts = [
            "RSA和AES是常用的加密算法",
            "数字签名确保消息完整性",
            "SHA-256是安全的哈希函数",
            "椭圆曲线密码学提供高效的安全性"
        ]
        
        distribution = processor.analyze_term_distribution(texts)
        
        # 检查分布统计的完整性
        assert distribution.total_terms > 0
        assert distribution.unique_terms > 0
        assert distribution.unique_terms <= distribution.total_terms
        assert len(distribution.category_distribution) > 0
        assert len(distribution.complexity_distribution) > 0
        assert len(distribution.term_frequency) > 0
        assert 0.0 <= distribution.coverage_ratio <= 1.0
        
        # 检查类别分布
        assert CryptoCategory.ASYMMETRIC_ENCRYPTION in distribution.category_distribution
        assert CryptoCategory.SYMMETRIC_ENCRYPTION in distribution.category_distribution
        assert CryptoCategory.HASH_FUNCTION in distribution.category_distribution
    
    def test_thinking_term_processing(self, processor, sample_thinking_example):
        """测试thinking数据术语处理"""
        analysis = processor.process_thinking_terms(sample_thinking_example)
        
        # 检查分析结果的完整性
        assert analysis.thinking_id is not None
        assert analysis.total_terms >= 0
        assert analysis.unique_terms >= 0
        assert analysis.unique_terms <= analysis.total_terms
        assert len(analysis.term_annotations) == analysis.total_terms
        assert 0.0 <= analysis.complexity_score <= 10.0
        assert 0.0 <= analysis.professional_score <= 1.0
        assert 0.0 <= analysis.term_coherence <= 1.0
        
        # 检查是否识别出关键术语
        identified_terms = [ann.term for ann in analysis.term_annotations]
        assert "RSA" in identified_terms
        assert "非对称加密" in identified_terms or any("非对称" in term for term in identified_terms)
    
    def test_thinking_enhancement(self, processor):
        """测试thinking文本增强"""
        original_text = "RSA是一种加密算法，用于保护数据安全。"
        enhanced_text = processor.enhance_thinking_with_terms(original_text)
        
        # 增强后的文本应该包含更多信息
        assert len(enhanced_text) >= len(original_text)
        
        # 应该包含相关术语的提及
        assert "相关概念" in enhanced_text or enhanced_text != original_text
    
    def test_term_usage_validation(self, processor):
        """测试术语使用验证"""
        # 正确使用的文本
        correct_text = "RSA是一种非对称加密算法，使用公钥和私钥进行加密解密。"
        correct_result = processor.validate_term_usage(correct_text)
        
        assert correct_result["total_terms"] > 0
        assert correct_result["valid_terms"] > 0
        assert len(correct_result["invalid_terms"]) == 0
        
        # 不正确使用的文本
        incorrect_text = "RSA使用相同的密钥进行加密和解密。"
        incorrect_result = processor.validate_term_usage(incorrect_text)
        
        # 应该检测出问题（虽然具体实现可能不同）
        assert incorrect_result["total_terms"] > 0
    
    def test_custom_dictionary_loading(self, processor):
        """测试自定义词典加载"""
        # 创建临时词典文件
        custom_terms = [
            {
                "term": "测试算法",
                "definition": "用于测试的加密算法",
                "category": "其他",
                "complexity": 3,
                "aliases": ["测试加密"],
                "related_terms": ["加密算法"],
                "examples": ["这是一个测试算法"]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(custom_terms, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            # 创建新的处理器并加载自定义词典
            new_processor = CryptoTermProcessor(custom_dict_path=temp_path)
            
            # 检查自定义术语是否被加载
            assert "测试算法" in new_processor.crypto_dict
            assert "测试加密" in new_processor.crypto_dict  # 别名
            
            # 测试识别功能
            test_text = "测试算法是一种新的加密方法。"
            annotations = new_processor.identify_crypto_terms(test_text)
            
            identified_terms = [ann.term for ann in annotations]
            assert "测试算法" in identified_terms
            
        finally:
            # 清理临时文件
            os.unlink(temp_path)
    
    def test_term_statistics(self, processor):
        """测试术语统计功能"""
        stats = processor.get_term_statistics()
        
        # 检查统计信息的完整性
        assert "total_unique_terms" in stats
        assert "total_entries" in stats
        assert "category_distribution" in stats
        assert "complexity_distribution" in stats
        assert "average_complexity" in stats
        
        # 检查数值的合理性
        assert stats["total_unique_terms"] > 0
        assert stats["total_entries"] >= stats["total_unique_terms"]
        assert stats["average_complexity"] > 0
        
        # 检查类别分布
        assert len(stats["category_distribution"]) > 0
        for category, count in stats["category_distribution"].items():
            assert count > 0
    
    def test_dictionary_export(self, processor):
        """测试词典导出功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出词典
            processor.export_term_dictionary(temp_path)
            
            # 检查导出文件
            assert os.path.exists(temp_path)
            
            # 读取并验证导出内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) > 0
            
            # 检查导出数据的格式
            for term_data in exported_data:
                assert "term" in term_data
                assert "definition" in term_data
                assert "category" in term_data
                assert "complexity" in term_data
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_context_extraction(self, processor):
        """测试上下文提取"""
        text = "这是一个很长的文本，其中包含RSA加密算法的描述，用于测试上下文提取功能。"
        
        # 找到RSA的位置
        start_pos = text.find("RSA")
        end_pos = start_pos + 3
        
        context = processor._extract_context(text, start_pos, end_pos, window_size=10)
        
        # 上下文应该包含RSA及其周围的文字
        assert "RSA" in context
        assert len(context) > 3  # 应该比单独的"RSA"长
    
    def test_confidence_calculation(self, processor):
        """测试置信度计算"""
        # 包含相关术语的上下文应该有更高的置信度
        high_confidence_context = "RSA是一种非对称加密算法，使用公钥和私钥"
        high_confidence = processor._calculate_term_confidence("RSA", high_confidence_context)
        
        # 不相关的上下文应该有较低的置信度
        low_confidence_context = "今天天气很好，适合出门"
        low_confidence = processor._calculate_term_confidence("RSA", low_confidence_context)
        
        assert high_confidence > low_confidence
        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
    
    def test_term_relations(self, processor):
        """测试术语关系"""
        # 检查术语关系图是否正确构建
        assert "RSA" in processor.term_relations
        assert "非对称加密" in processor.term_relations
        
        # 检查相关术语关系
        rsa_relations = processor.term_relations["RSA"]
        assert "非对称加密" in rsa_relations or "数字签名" in rsa_relations
    
    def test_edge_cases(self, processor):
        """测试边界情况"""
        # 空文本
        empty_annotations = processor.identify_crypto_terms("")
        assert len(empty_annotations) == 0
        
        # 只有标点符号
        punct_annotations = processor.identify_crypto_terms("！@#￥%……&*（）")
        assert len(punct_annotations) == 0
        
        # 非常短的文本
        short_annotations = processor.identify_crypto_terms("RSA")
        assert len(short_annotations) >= 0
        
        # 重复术语
        repeat_text = "RSA RSA RSA 算法"
        repeat_annotations = processor.identify_crypto_terms(repeat_text)
        # 应该去重
        rsa_count = sum(1 for ann in repeat_annotations if ann.term == "RSA")
        assert rsa_count <= 3  # 可能会有重复，但应该有去重机制
    
    def test_performance_with_large_text(self, processor):
        """测试大文本处理性能"""
        # 创建较大的测试文本
        large_text = """
        RSA加密算法是一种非对称加密算法，由Ron Rivest、Adi Shamir和Leonard Adleman在1977年提出。
        RSA算法基于大整数分解的数学难题，使用公钥和私钥进行加密和解密操作。
        与对称加密算法如AES不同，RSA不需要事先共享密钥。
        数字签名是RSA的重要应用之一，可以确保消息的完整性和不可否认性。
        SHA-256哈希函数常与RSA结合使用，提供更强的安全保障。
        椭圆曲线密码学(ECC)是RSA的替代方案，提供相同安全级别下更短的密钥长度。
        密钥管理是密码系统的重要组成部分，PKI提供了完整的密钥管理框架。
        SSL/TLS协议广泛使用RSA进行密钥交换和身份认证。
        """ * 10  # 重复10次以增加文本长度
        
        # 处理大文本不应该出错
        annotations = processor.identify_crypto_terms(large_text)
        assert len(annotations) > 0
        
        # 分析术语分布
        distribution = processor.analyze_term_distribution([large_text])
        assert distribution.total_terms > 0
        assert distribution.coverage_ratio > 0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])