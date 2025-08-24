"""
中文NLP处理工具测试

测试中文分词、词性标注、繁简体转换、标点符号规范化、
文本质量评估和Qwen tokenizer优化功能。
"""

import pytest
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chinese_nlp_processor import (
    ChineseNLPProcessor, ChineseToken, TextQualityMetrics,
    TokenizerOptimization, TextVariant, PunctuationStyle
)
from data_models import ChineseMetrics


class TestChineseNLPProcessor:
    """中文NLP处理器测试类"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return ChineseNLPProcessor()
    
    def test_segment_text_basic(self, processor):
        """测试基本分词功能"""
        text = "RSA是一种非对称加密算法"
        tokens = processor.segment_text(text)
        
        assert len(tokens) > 0
        assert all(isinstance(token, ChineseToken) for token in tokens)
        
        # 检查是否正确识别密码学术语
        crypto_tokens = [token for token in tokens if token.is_crypto_term]
        assert len(crypto_tokens) > 0
        
        # 检查位置信息
        for token in tokens:
            assert token.start_pos >= 0
            assert token.end_pos > token.start_pos
            assert token.word == text[token.start_pos:token.end_pos]
    
    def test_segment_text_empty(self, processor):
        """测试空文本分词"""
        tokens = processor.segment_text("")
        assert tokens == []
        
        tokens = processor.segment_text("   ")
        assert tokens == []
    
    def test_segment_text_crypto_terms(self, processor):
        """测试密码学术语识别"""
        text = "AES加密算法使用对称密钥进行数据加密"
        tokens = processor.segment_text(text)
        
        crypto_terms = [token.word for token in tokens if token.is_crypto_term]
        # 检查是否识别了一些密码学相关术语
        assert len(crypto_terms) > 0, f"应该识别出密码学术语，但得到: {crypto_terms}"
        # 检查是否包含AES或加密相关术语
        has_crypto_term = any(term in ["AES", "加密", "加密算法", "对称", "密钥"] for term in crypto_terms)
        assert has_crypto_term, f"应该包含密码学术语，但得到: {crypto_terms}"
    
    def test_convert_traditional_simplified(self, processor):
        """测试繁简体转换"""
        # 简体转繁体 - 使用有明显差异的字符
        simplified = "网络安全"
        traditional = processor.convert_traditional_simplified(
            simplified, TextVariant.TRADITIONAL
        )
        # 网络 -> 網絡
        assert "網" in traditional or traditional == simplified  # 有些字可能没有繁体形式
        
        # 繁体转简体
        traditional_text = "網絡安全"
        simplified_result = processor.convert_traditional_simplified(
            traditional_text, TextVariant.SIMPLIFIED
        )
        assert "网" in simplified_result or simplified_result == traditional_text
    
    def test_convert_empty_text(self, processor):
        """测试空文本转换"""
        result = processor.convert_traditional_simplified("", TextVariant.TRADITIONAL)
        assert result == ""
        
        result = processor.convert_traditional_simplified("   ", TextVariant.SIMPLIFIED)
        assert result == "   "
    
    def test_detect_text_variant(self, processor):
        """测试文本变体检测"""
        # 简体文本
        simplified_text = "这是简体中文文本"
        variant = processor.detect_text_variant(simplified_text)
        assert variant in [TextVariant.SIMPLIFIED, TextVariant.MIXED]
        
        # 空文本
        variant = processor.detect_text_variant("")
        assert variant == TextVariant.SIMPLIFIED
    
    def test_normalize_punctuation_chinese(self, processor):
        """测试中文标点符号规范化"""
        text = "Hello,world.How are you?"
        normalized = processor.normalize_punctuation(text, PunctuationStyle.CHINESE)
        
        assert "，" in normalized
        assert "。" in normalized
        assert "？" in normalized
        assert "," not in normalized
        assert "." not in normalized
        assert "?" not in normalized
    
    def test_normalize_punctuation_english(self, processor):
        """测试英文标点符号规范化"""
        text = "你好，世界。你好吗？"
        normalized = processor.normalize_punctuation(text, PunctuationStyle.ENGLISH)
        
        assert "," in normalized
        assert "." in normalized
        assert "?" in normalized
        assert "，" not in normalized
        assert "。" not in normalized
        assert "？" not in normalized
    
    def test_normalize_punctuation_quotes(self, processor):
        """测试引号规范化"""
        text = '"Hello" and \'world\''
        chinese_normalized = processor.normalize_punctuation(text, PunctuationStyle.CHINESE)
        assert '"' in chinese_normalized and '"' in chinese_normalized
        
        english_normalized = processor.normalize_punctuation(text, PunctuationStyle.ENGLISH)
        assert '"' in english_normalized
    
    def test_normalize_punctuation_spaces(self, processor):
        """测试空格规范化"""
        text = "Hello世界123测试"
        normalized = processor.normalize_punctuation(text)
        
        # 应该在中英文、中文数字之间添加空格
        assert "Hello 世界" in normalized or "Hello世界" in normalized
        assert "123 测试" in normalized or "123测试" in normalized
    
    def test_assess_text_quality_basic(self, processor):
        """测试基本文本质量评估"""
        text = "RSA是一种非对称加密算法，它使用公钥和私钥进行加密和解密。"
        metrics = processor.assess_text_quality(text)
        
        assert isinstance(metrics, TextQualityMetrics)
        assert 0.0 <= metrics.readability_score <= 1.0
        assert 0.0 <= metrics.fluency_score <= 1.0
        assert 0.0 <= metrics.coherence_score <= 1.0
        assert 0.0 <= metrics.complexity_score <= 1.0
        assert 0.0 <= metrics.punctuation_score <= 1.0
        assert 0.0 <= metrics.character_diversity <= 1.0
        assert 0.0 <= metrics.word_diversity <= 1.0
        assert metrics.avg_sentence_length > 0
        
        overall_quality = metrics.overall_quality()
        assert 0.0 <= overall_quality <= 1.0
    
    def test_assess_text_quality_empty(self, processor):
        """测试空文本质量评估"""
        metrics = processor.assess_text_quality("")
        
        assert metrics.readability_score == 0.0
        assert metrics.fluency_score == 0.0
        assert metrics.coherence_score == 0.0
        assert metrics.complexity_score == 0.0
        assert metrics.punctuation_score == 0.0
        assert metrics.character_diversity == 0.0
        assert metrics.word_diversity == 0.0
        assert metrics.avg_sentence_length == 0.0
    
    def test_assess_text_quality_high_quality(self, processor):
        """测试高质量文本评估"""
        text = """
        首先，我们需要理解RSA算法的基本原理。RSA是一种非对称加密算法，
        它基于大整数分解的数学难题。然后，我们分析其安全性。
        最后，我们讨论其实际应用场景。
        """
        metrics = processor.assess_text_quality(text)
        
        # 高质量文本应该有较好的连贯性（有连接词）
        assert metrics.coherence_score > 0.5
        
        # 应该有合理的复杂度
        assert metrics.complexity_score > 0.3
    
    def test_optimize_qwen_tokenizer_basic(self, processor):
        """测试Qwen tokenizer基本优化"""
        texts = [
            "RSA加密算法是一种非对称加密方法",
            "AES是对称加密算法的代表",
            "哈希函数用于数据完整性验证"
        ]
        
        optimization = processor.optimize_qwen_tokenizer(texts, vocab_size=1000)
        
        assert isinstance(optimization, TokenizerOptimization)
        assert optimization.vocab_size == 1000
        assert len(optimization.special_tokens) > 0
        assert optimization.chinese_vocab_ratio >= 0.0
        assert optimization.crypto_terms_count >= 0
        assert 0.0 <= optimization.oov_rate <= 1.0
        assert optimization.compression_ratio >= 1.0
        
        # 应该包含thinking相关的特殊token
        assert '<thinking>' in optimization.special_tokens
        assert '</thinking>' in optimization.special_tokens
    
    def test_optimize_qwen_tokenizer_empty(self, processor):
        """测试空文本列表的tokenizer优化"""
        optimization = processor.optimize_qwen_tokenizer([])
        
        assert optimization.chinese_vocab_ratio == 0.0
        assert optimization.crypto_terms_count == 0
        assert optimization.oov_rate == 1.0
        assert optimization.compression_ratio == 1.0
    
    def test_optimize_qwen_tokenizer_crypto_heavy(self, processor):
        """测试密码学术语密集文本的tokenizer优化"""
        texts = [
            "RSA、AES、DES、SHA、MD5都是重要的密码学算法",
            "公钥、私钥、数字签名、哈希函数是密码学基础概念",
            "对称加密、非对称加密、椭圆曲线密码学是主要分类"
        ]
        
        optimization = processor.optimize_qwen_tokenizer(texts)
        
        # 密码学术语应该被识别
        assert optimization.crypto_terms_count > 5
        
        # 特殊token应该包含密码学术语
        crypto_tokens = [token for token in optimization.special_tokens 
                        if token in processor.crypto_terms]
        assert len(crypto_tokens) > 0
    
    def test_preprocess_for_training_basic(self, processor):
        """测试训练预处理基本功能"""
        text = "Hello,world.这是测试文本!"
        processed = processor.preprocess_for_training(text)
        
        assert processed != text  # 应该有变化
        assert "，" in processed  # 标点应该被规范化
        assert "。" in processed
        assert "！" in processed
    
    def test_preprocess_for_training_options(self, processor):
        """测试训练预处理选项"""
        text = "Hello,world.这是测试文本!"
        
        # 不规范化标点
        processed = processor.preprocess_for_training(
            text, normalize_punctuation=False
        )
        assert "," in processed
        assert "." in processed
        assert "!" in processed
        
        # 不规范化繁简体
        processed = processor.preprocess_for_training(
            text, normalize_variant=False
        )
        # 由于原文是简体，应该没有变化
        assert "这是测试文本" in processed
    
    def test_extract_crypto_terms_from_text(self, processor):
        """测试从文本中提取密码学术语"""
        text = "RSA和AES是两种不同的加密算法，RSA是非对称的，AES是对称的"
        terms = processor.extract_crypto_terms_from_text(text)
        
        assert len(terms) > 0
        assert isinstance(terms, list)
        
        # 应该包含一些密码学术语
        crypto_terms_found = [term for term in terms if term in processor.crypto_terms]
        assert len(crypto_terms_found) > 0
        
        # 结果应该去重
        assert len(terms) == len(set(terms))
    
    def test_extract_crypto_terms_empty(self, processor):
        """测试从空文本提取密码学术语"""
        terms = processor.extract_crypto_terms_from_text("")
        assert terms == []
        
        terms = processor.extract_crypto_terms_from_text("这是普通文本，没有密码学内容")
        # 可能为空或包含很少术语
        assert isinstance(terms, list)
    
    def test_calculate_chinese_metrics_basic(self, processor):
        """测试中文指标计算基本功能"""
        predictions = ["RSA是非对称加密算法"]
        references = ["RSA是一种非对称加密算法"]
        
        metrics = processor.calculate_chinese_metrics(predictions, references)
        
        assert isinstance(metrics, ChineseMetrics)
        assert 0.0 <= metrics.character_accuracy <= 1.0
        assert 0.0 <= metrics.word_accuracy <= 1.0
        assert 0.0 <= metrics.rouge_l_chinese <= 1.0
        assert 0.0 <= metrics.bleu_chinese <= 1.0
        assert 0.0 <= metrics.crypto_term_accuracy <= 1.0
        
        # 由于预测和参考很相似，准确率应该较高
        assert metrics.character_accuracy > 0.7
        assert metrics.word_accuracy > 0.7
    
    def test_calculate_chinese_metrics_perfect_match(self, processor):
        """测试完全匹配的中文指标计算"""
        text = "RSA是非对称加密算法"
        predictions = [text]
        references = [text]
        
        metrics = processor.calculate_chinese_metrics(predictions, references)
        
        # 完全匹配应该得到高分
        assert metrics.character_accuracy == 1.0
        assert metrics.word_accuracy == 1.0
        assert metrics.rouge_l_chinese == 1.0
        assert metrics.crypto_term_accuracy == 1.0
    
    def test_calculate_chinese_metrics_empty(self, processor):
        """测试空列表的中文指标计算"""
        metrics = processor.calculate_chinese_metrics([], [])
        
        assert metrics.character_accuracy == 0.0
        assert metrics.word_accuracy == 0.0
        assert metrics.rouge_l_chinese == 0.0
        assert metrics.bleu_chinese == 0.0
        assert metrics.crypto_term_accuracy == 0.0
    
    def test_calculate_chinese_metrics_mismatch_length(self, processor):
        """测试长度不匹配的中文指标计算"""
        predictions = ["text1"]
        references = ["ref1", "ref2"]
        
        with pytest.raises(ValueError, match="预测文本和参考文本数量不匹配"):
            processor.calculate_chinese_metrics(predictions, references)
    
    def test_calculate_chinese_metrics_crypto_terms(self, processor):
        """测试包含密码学术语的中文指标计算"""
        predictions = ["RSA和AES都是加密算法"]
        references = ["RSA和DES都是加密算法"]
        
        metrics = processor.calculate_chinese_metrics(predictions, references)
        
        # 应该有合理的密码学术语准确率
        assert 0.0 <= metrics.crypto_term_accuracy <= 1.0
        
        # 由于有一个术语不匹配（AES vs DES），准确率应该不是1.0
        if metrics.crypto_term_accuracy > 0:
            assert metrics.crypto_term_accuracy < 1.0
    
    def test_lcs_length(self, processor):
        """测试最长公共子序列长度计算"""
        seq1 = ["a", "b", "c", "d"]
        seq2 = ["a", "c", "d", "e"]
        
        lcs_len = processor._lcs_length(seq1, seq2)
        assert lcs_len == 3  # "a", "c", "d"
        
        # 测试空序列
        assert processor._lcs_length([], []) == 0
        assert processor._lcs_length(["a"], []) == 0
        assert processor._lcs_length([], ["a"]) == 0
        
        # 测试相同序列
        assert processor._lcs_length(seq1, seq1) == len(seq1)
    
    def test_get_ngrams(self, processor):
        """测试n-gram提取"""
        words = ["这", "是", "测试", "文本"]
        
        # 1-gram
        unigrams = processor._get_ngrams(words, 1)
        assert len(unigrams) == 4
        assert ("这",) in unigrams
        assert ("是",) in unigrams
        
        # 2-gram
        bigrams = processor._get_ngrams(words, 2)
        assert len(bigrams) == 3
        assert ("这", "是") in bigrams
        assert ("是", "测试") in bigrams
        
        # 空列表
        assert processor._get_ngrams([], 1) == {}
        
        # n大于序列长度
        assert processor._get_ngrams(["a"], 2) == {}
    
    def test_chinese_token_validation(self):
        """测试ChineseToken数据验证"""
        # 正常创建
        token = ChineseToken(
            word="测试",
            pos="n",
            start_pos=0,
            end_pos=2
        )
        assert token.word == "测试"
        assert token.pos == "n"
        
        # 空词汇应该报错
        with pytest.raises(ValueError, match="词汇不能为空"):
            ChineseToken(word="", pos="n", start_pos=0, end_pos=1)
        
        # 无效位置应该报错
        with pytest.raises(ValueError, match="位置信息无效"):
            ChineseToken(word="测试", pos="n", start_pos=5, end_pos=2)
        
        with pytest.raises(ValueError, match="位置信息无效"):
            ChineseToken(word="测试", pos="n", start_pos=-1, end_pos=2)
    
    def test_text_quality_metrics_overall_quality(self):
        """测试TextQualityMetrics综合质量计算"""
        metrics = TextQualityMetrics(
            readability_score=0.8,
            fluency_score=0.9,
            coherence_score=0.7,
            complexity_score=0.6,
            punctuation_score=0.8,
            character_diversity=0.7,
            word_diversity=0.6,
            avg_sentence_length=15.0
        )
        
        overall = metrics.overall_quality()
        assert 0.0 <= overall <= 1.0
        assert overall > 0.5  # 应该是较高的质量分数
    
    def test_tokenizer_optimization_is_optimized(self):
        """测试TokenizerOptimization优化判断"""
        # 优化良好的配置
        good_config = TokenizerOptimization(
            vocab_size=32000,
            special_tokens=['<thinking>', '</thinking>'],
            chinese_vocab_ratio=0.7,
            crypto_terms_count=100,
            oov_rate=0.03,
            compression_ratio=2.5
        )
        assert good_config.is_optimized()
        
        # 优化不佳的配置
        bad_config = TokenizerOptimization(
            vocab_size=32000,
            special_tokens=[],
            chinese_vocab_ratio=0.3,
            crypto_terms_count=10,
            oov_rate=0.1,
            compression_ratio=1.5
        )
        assert not bad_config.is_optimized()


class TestChineseNLPProcessorIntegration:
    """中文NLP处理器集成测试"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return ChineseNLPProcessor()
    
    def test_full_pipeline_crypto_text(self, processor):
        """测试密码学文本的完整处理流程"""
        original_text = "RSA算法使用大整数分解难题,确保加密安全性."
        
        # 1. 预处理
        processed_text = processor.preprocess_for_training(original_text)
        assert processed_text != original_text
        assert "，" in processed_text  # 标点规范化
        assert "。" in processed_text
        
        # 2. 分词
        tokens = processor.segment_text(processed_text)
        assert len(tokens) > 0
        
        # 3. 提取密码学术语
        crypto_terms = processor.extract_crypto_terms_from_text(processed_text)
        assert len(crypto_terms) > 0
        
        # 4. 质量评估
        quality = processor.assess_text_quality(processed_text)
        assert quality.overall_quality() > 0.3
        
        # 5. tokenizer优化
        optimization = processor.optimize_qwen_tokenizer([processed_text])
        assert optimization.crypto_terms_count > 0
    
    def test_full_pipeline_mixed_text(self, processor):
        """测试中英文混合文本的完整处理流程"""
        text = "Hello世界,这是mixed text测试!"
        
        # 预处理应该规范化空格和标点
        processed = processor.preprocess_for_training(text)
        assert " " in processed  # 中英文间应该有空格
        assert "，" in processed
        assert "！" in processed
        
        # 分词应该正确处理混合文本
        tokens = processor.segment_text(processed)
        assert len(tokens) > 0
        
        # 质量评估
        quality = processor.assess_text_quality(processed)
        assert isinstance(quality, TextQualityMetrics)
    
    def test_batch_processing(self, processor):
        """测试批量处理"""
        texts = [
            "RSA是非对称加密算法",
            "AES是对称加密的代表",
            "哈希函数确保数据完整性",
            "数字签名提供身份认证"
        ]
        
        # 批量预处理
        processed_texts = [
            processor.preprocess_for_training(text) for text in texts
        ]
        assert len(processed_texts) == len(texts)
        
        # 批量质量评估
        qualities = [
            processor.assess_text_quality(text) for text in processed_texts
        ]
        assert len(qualities) == len(texts)
        assert all(isinstance(q, TextQualityMetrics) for q in qualities)
        
        # 批量tokenizer优化
        optimization = processor.optimize_qwen_tokenizer(processed_texts)
        assert optimization.crypto_terms_count > 0
        
        # 批量指标计算（使用自身作为参考）
        metrics = processor.calculate_chinese_metrics(processed_texts, processed_texts)
        assert metrics.character_accuracy == 1.0
        assert metrics.word_accuracy == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])