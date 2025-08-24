"""
中文特定指标计算模块测试

测试ChineseMetricsCalculator类的中文字符级和词级准确率计算、密码学术语学习进度跟踪、
thinking数据质量评估指标等功能。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.chinese_metrics_calculator import (
    ChineseMetricsCalculator, ChineseCharacterMetrics, ChineseWordMetrics,
    ChineseSemanticMetrics, ThinkingQualityMetrics, CryptoTermLearningProgress
)
from src.data_models import ChineseMetrics, CryptoTerm, CryptoCategory


class TestChineseCharacterMetrics:
    """测试中文字符级指标类"""
    
    def test_character_metrics_initialization(self):
        """测试字符指标初始化"""
        metrics = ChineseCharacterMetrics()
        
        assert metrics.total_characters == 0
        assert metrics.correct_characters == 0
        assert metrics.character_accuracy == 0.0
        assert metrics.hanzi_count == 0
        assert metrics.substitution_errors == 0
    
    def test_calculate_accuracy(self):
        """测试准确率计算"""
        metrics = ChineseCharacterMetrics()
        metrics.total_characters = 100
        metrics.correct_characters = 85
        
        metrics.calculate_accuracy()
        
        assert metrics.character_accuracy == 0.85
    
    def test_calculate_accuracy_zero_total(self):
        """测试零总字符数的准确率计算"""
        metrics = ChineseCharacterMetrics()
        metrics.total_characters = 0
        metrics.correct_characters = 0
        
        metrics.calculate_accuracy()
        
        assert metrics.character_accuracy == 0.0
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        metrics = ChineseCharacterMetrics()
        metrics.total_characters = 50
        metrics.correct_characters = 40
        metrics.hanzi_count = 35
        metrics.substitution_errors = 5
        metrics.calculate_accuracy()
        
        data = metrics.to_dict()
        
        assert data["total_characters"] == 50
        assert data["correct_characters"] == 40
        assert data["character_accuracy"] == 0.8
        assert data["hanzi_count"] == 35
        assert data["substitution_errors"] == 5


class TestChineseWordMetrics:
    """测试中文词级指标类"""
    
    def test_word_metrics_initialization(self):
        """测试词指标初始化"""
        metrics = ChineseWordMetrics()
        
        assert metrics.total_words == 0
        assert metrics.correct_words == 0
        assert metrics.word_accuracy == 0.0
        assert metrics.crypto_terms_count == 0
    
    def test_calculate_accuracy(self):
        """测试词准确率计算"""
        metrics = ChineseWordMetrics()
        metrics.total_words = 20
        metrics.correct_words = 16
        
        metrics.calculate_accuracy()
        
        assert metrics.word_accuracy == 0.8
    
    def test_crypto_term_accuracy_property(self):
        """测试密码学术语准确率属性"""
        metrics = ChineseWordMetrics()
        metrics.crypto_terms_count = 10
        metrics.crypto_terms_correct = 8
        
        assert metrics.crypto_term_accuracy == 0.8
    
    def test_crypto_term_accuracy_zero_count(self):
        """测试零术语数量的准确率"""
        metrics = ChineseWordMetrics()
        metrics.crypto_terms_count = 0
        metrics.crypto_terms_correct = 0
        
        assert metrics.crypto_term_accuracy == 0.0
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        metrics = ChineseWordMetrics()
        metrics.total_words = 30
        metrics.correct_words = 25
        metrics.crypto_terms_count = 5
        metrics.crypto_terms_correct = 4
        metrics.calculate_accuracy()
        
        data = metrics.to_dict()
        
        assert data["total_words"] == 30
        assert data["word_accuracy"] == 25/30
        assert data["crypto_term_accuracy"] == 0.8


class TestChineseSemanticMetrics:
    """测试中文语义指标类"""
    
    def test_semantic_metrics_initialization(self):
        """测试语义指标初始化"""
        metrics = ChineseSemanticMetrics()
        
        assert metrics.rouge_l_score == 0.0
        assert metrics.bleu_score == 0.0
        assert metrics.fluency_score == 0.0
        assert metrics.terminology_usage == 0.0
    
    def test_calculate_overall_score(self):
        """测试综合评分计算"""
        metrics = ChineseSemanticMetrics()
        metrics.rouge_l_score = 0.8
        metrics.bleu_score = 0.7
        metrics.semantic_similarity = 0.9
        metrics.fluency_score = 0.85
        metrics.coherence_score = 0.75
        metrics.terminology_usage = 0.8
        
        overall_score = metrics.calculate_overall_score()
        
        # 验证加权平均计算
        expected = (0.8 * 0.25 + 0.7 * 0.25 + 0.9 * 0.20 + 
                   0.85 * 0.10 + 0.75 * 0.10 + 0.8 * 0.10)
        assert abs(overall_score - expected) < 1e-6
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        metrics = ChineseSemanticMetrics()
        metrics.rouge_l_score = 0.8
        metrics.bleu_score = 0.7
        metrics.fluency_score = 0.85
        
        data = metrics.to_dict()
        
        assert data["rouge_l_score"] == 0.8
        assert data["bleu_score"] == 0.7
        assert data["fluency_score"] == 0.85
        assert "overall_score" in data


class TestThinkingQualityMetrics:
    """测试思考质量指标类"""
    
    def test_thinking_quality_metrics_initialization(self):
        """测试思考质量指标初始化"""
        metrics = ThinkingQualityMetrics()
        
        assert metrics.thinking_completeness == 0.0
        assert metrics.logical_consistency == 0.0
        assert metrics.thinking_steps_count == 0
        assert metrics.crypto_reasoning_accuracy == 0.0
    
    def test_calculate_overall_quality(self):
        """测试综合质量评分计算"""
        metrics = ThinkingQualityMetrics()
        metrics.thinking_completeness = 0.9
        metrics.logical_consistency = 0.8
        metrics.reasoning_depth = 0.85
        metrics.step_clarity = 0.75
        metrics.crypto_reasoning_accuracy = 0.8
        
        overall_quality = metrics.calculate_overall_quality()
        
        # 验证加权平均计算
        expected = (0.9 * 0.25 + 0.8 * 0.25 + 0.85 * 0.20 + 
                   0.75 * 0.15 + 0.8 * 0.15)
        assert abs(overall_quality - expected) < 1e-6
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        metrics = ThinkingQualityMetrics()
        metrics.thinking_completeness = 0.9
        metrics.thinking_steps_count = 5
        metrics.crypto_reasoning_accuracy = 0.8
        
        data = metrics.to_dict()
        
        assert data["thinking_completeness"] == 0.9
        assert data["thinking_steps_count"] == 5
        assert data["crypto_reasoning_accuracy"] == 0.8
        assert "overall_quality" in data


class TestCryptoTermLearningProgress:
    """测试密码学术语学习进度类"""
    
    def test_crypto_progress_initialization(self):
        """测试学习进度初始化"""
        progress = CryptoTermLearningProgress()
        
        assert progress.total_terms_encountered == 0
        assert progress.correctly_used_terms == 0
        assert progress.improvement_rate == 0.0
        assert len(progress.category_progress) == 0
    
    def test_overall_accuracy_property(self):
        """测试总体准确率属性"""
        progress = CryptoTermLearningProgress()
        progress.total_terms_encountered = 20
        progress.correctly_used_terms = 16
        
        assert progress.overall_accuracy == 0.8
    
    def test_overall_accuracy_zero_total(self):
        """测试零总术语数的准确率"""
        progress = CryptoTermLearningProgress()
        progress.total_terms_encountered = 0
        progress.correctly_used_terms = 0
        
        assert progress.overall_accuracy == 0.0
    
    def test_mastery_level_property(self):
        """测试掌握水平属性"""
        progress = CryptoTermLearningProgress()
        
        # 测试不同准确率对应的掌握水平
        test_cases = [
            (0.95, "专家"),
            (0.85, "高级"),
            (0.75, "中级"),
            (0.65, "初级"),
            (0.5, "入门")
        ]
        
        for accuracy, expected_level in test_cases:
            progress.total_terms_encountered = 100
            progress.correctly_used_terms = int(accuracy * 100)
            assert progress.mastery_level == expected_level
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        progress = CryptoTermLearningProgress()
        progress.total_terms_encountered = 50
        progress.correctly_used_terms = 40
        progress.category_progress = {"对称加密": 0.8, "哈希函数": 0.9}
        
        data = progress.to_dict()
        
        assert data["total_terms_encountered"] == 50
        assert data["correctly_used_terms"] == 40
        assert data["overall_accuracy"] == 0.8
        assert data["mastery_level"] == "高级"
        assert data["category_progress"]["对称加密"] == 0.8


class TestChineseMetricsCalculator:
    """测试中文指标计算器类"""
    
    def setup_method(self):
        """测试前设置"""
        self.calculator = ChineseMetricsCalculator()
    
    @patch('src.chinese_metrics_calculator.ChineseNLPProcessor')
    @patch('src.chinese_metrics_calculator.CryptoTermProcessor')
    def test_calculator_initialization(self, mock_crypto_processor, mock_nlp_processor):
        """测试计算器初始化"""
        calculator = ChineseMetricsCalculator()
        
        assert calculator.chinese_nlp is not None
        assert calculator.crypto_processor is not None
        assert len(calculator.historical_metrics) == 0
        assert len(calculator.crypto_progress_history) == 0
    
    def test_calculate_edit_distance_operations(self):
        """测试编辑距离操作计算"""
        seq1 = ['a', 'b', 'c']
        seq2 = ['a', 'x', 'c']
        
        operations = self.calculator._calculate_edit_distance_operations(seq1, seq2)
        
        # 应该有一个替换操作
        assert len(operations) == 1
        assert operations[0][0] == 'substitute'
    
    def test_longest_common_subsequence(self):
        """测试最长公共子序列计算"""
        seq1 = ['a', 'b', 'c', 'd']
        seq2 = ['a', 'c', 'd', 'e']
        
        lcs_length = self.calculator._longest_common_subsequence(seq1, seq2)
        
        assert lcs_length == 3  # 'a', 'c', 'd'
    
    def test_get_ngrams(self):
        """测试n-gram提取"""
        words = ['这', '是', '一个', '测试']
        
        # 测试2-gram
        bigrams = self.calculator._get_ngrams(words, 2)
        
        assert len(bigrams) == 3
        assert ('这', '是') in bigrams
        assert ('是', '一个') in bigrams
        assert ('一个', '测试') in bigrams
    
    @patch.object(ChineseMetricsCalculator, '_calculate_edit_distance_operations')
    def test_calculate_character_metrics(self, mock_edit_ops):
        """测试字符级指标计算"""
        # 模拟编辑距离操作
        mock_edit_ops.return_value = [('substitute', 1, 1), ('insert', -1, 2)]
        
        predicted = "这是测试文本"
        reference = "这是测试文档"
        
        metrics = self.calculator.calculate_character_metrics(predicted, reference)
        
        assert isinstance(metrics, ChineseCharacterMetrics)
        assert metrics.total_characters == len(reference)
        assert metrics.substitution_errors == 1
        assert metrics.insertion_errors == 1
        assert metrics.hanzi_count > 0  # 应该检测到汉字
    
    def test_calculate_word_metrics(self):
        """测试词级指标计算"""
        # 模拟分词结果
        self.calculator.chinese_nlp.segment_text = Mock(side_effect=[
            ['这', '是', '测试', '文本'],  # predicted
            ['这', '是', '测试', '文档']   # reference
        ])
        
        # 模拟词性标注
        self.calculator.chinese_nlp.pos_tag = Mock(return_value=[
            ('这', 'r'), ('是', 'v'), ('测试', 'n'), ('文档', 'n')
        ])
        
        # 模拟密码学术语提取
        mock_crypto_term = Mock()
        mock_crypto_term.term = "AES"
        self.calculator.crypto_processor.extract_crypto_terms = Mock(return_value=[mock_crypto_term])
        
        predicted = "这是测试文本"
        reference = "这是测试文档"
        
        metrics = self.calculator.calculate_word_metrics(predicted, reference)
        
        assert isinstance(metrics, ChineseWordMetrics)
        assert metrics.total_words == 4
        assert metrics.correct_words == 3  # '这', '是', '测试' 匹配
        assert metrics.noun_count == 2  # '测试', '文档'
        assert metrics.verb_count == 1  # '是'
    
    @patch.object(ChineseMetricsCalculator, '_calculate_chinese_rouge_l')
    @patch.object(ChineseMetricsCalculator, '_calculate_chinese_bleu')
    @patch.object(ChineseMetricsCalculator, '_calculate_semantic_similarity')
    def test_calculate_semantic_metrics(self, mock_similarity, mock_bleu, mock_rouge):
        """测试语义指标计算"""
        # 模拟各种指标计算结果
        mock_rouge.return_value = 0.8
        mock_bleu.return_value = 0.7
        mock_similarity.return_value = 0.85
        
        # 模拟其他评估方法
        self.calculator._evaluate_fluency = Mock(return_value=0.9)
        self.calculator._evaluate_coherence = Mock(return_value=0.8)
        self.calculator._evaluate_naturalness = Mock(return_value=0.85)
        self.calculator._evaluate_terminology_usage = Mock(return_value=0.75)
        self.calculator._evaluate_technical_depth = Mock(return_value=0.7)
        
        predicted = "这是一个测试文本"
        reference = "这是一个参考文本"
        
        metrics = self.calculator.calculate_semantic_metrics(predicted, reference)
        
        assert isinstance(metrics, ChineseSemanticMetrics)
        assert metrics.rouge_l_score == 0.8
        assert metrics.bleu_score == 0.7
        assert metrics.semantic_similarity == 0.85
        assert metrics.fluency_score == 0.9
        assert metrics.terminology_usage == 0.75
    
    def test_calculate_chinese_rouge_l(self):
        """测试中文ROUGE-L计算"""
        # 模拟分词结果
        self.calculator.chinese_nlp.segment_text = Mock(side_effect=[
            ['这', '是', '测试', '文本'],  # predicted
            ['这', '是', '参考', '文本']   # reference
        ])
        
        predicted = "这是测试文本"
        reference = "这是参考文本"
        
        rouge_score = self.calculator._calculate_chinese_rouge_l(predicted, reference)
        
        assert 0.0 <= rouge_score <= 1.0
        assert rouge_score > 0  # 应该有一些重叠
    
    def test_calculate_chinese_bleu(self):
        """测试中文BLEU计算"""
        # 模拟分词结果
        self.calculator.chinese_nlp.segment_text = Mock(side_effect=[
            ['这', '是', '测试', '文本'],  # predicted
            ['这', '是', '参考', '文本']   # reference
        ])
        
        predicted = "这是测试文本"
        reference = "这是参考文本"
        
        bleu_score = self.calculator._calculate_chinese_bleu(predicted, reference)
        
        assert 0.0 <= bleu_score <= 1.0
    
    def test_calculate_semantic_similarity(self):
        """测试语义相似度计算"""
        # 模拟分词结果
        self.calculator.chinese_nlp.segment_text = Mock(side_effect=[
            ['这', '是', '测试'],     # text1
            ['这', '是', '参考']     # text2
        ])
        
        text1 = "这是测试"
        text2 = "这是参考"
        
        similarity = self.calculator._calculate_semantic_similarity(text1, text2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # 应该有一些相似性
    
    def test_evaluate_fluency(self):
        """测试流畅性评估"""
        # 测试正常文本
        normal_text = "这是一个正常的句子。它有适当的长度和标点符号。"
        fluency_score = self.calculator._evaluate_fluency(normal_text)
        assert 0.0 <= fluency_score <= 1.0
        
        # 测试空文本
        empty_text = ""
        fluency_score_empty = self.calculator._evaluate_fluency(empty_text)
        assert fluency_score_empty == 0.0
    
    def test_evaluate_coherence(self):
        """测试连贯性评估"""
        # 测试有连接词的文本
        coherent_text = "首先，我们需要分析问题。然后，我们可以找到解决方案。因此，这个方法是有效的。"
        coherence_score = self.calculator._evaluate_coherence(coherent_text)
        assert 0.0 <= coherence_score <= 1.0
        
        # 测试空文本
        empty_text = ""
        coherence_score_empty = self.calculator._evaluate_coherence(empty_text)
        assert coherence_score_empty == 0.0
    
    def test_evaluate_naturalness(self):
        """测试自然性评估"""
        # 模拟分词结果
        self.calculator.chinese_nlp.segment_text = Mock(return_value=[
            '这', '是', '一个', '自然', '的', '文本', '示例'
        ])
        
        text = "这是一个自然的文本示例"
        naturalness_score = self.calculator._evaluate_naturalness(text)
        
        assert 0.0 <= naturalness_score <= 1.0
    
    @patch.object(ChineseMetricsCalculator, '_parse_thinking_structure')
    def test_calculate_thinking_quality_metrics(self, mock_parse):
        """测试思考质量指标计算"""
        # 模拟思考结构解析结果
        mock_structure = {
            'steps': ['步骤1：分析问题', '步骤2：寻找解决方案', '步骤3：验证结果'],
            'reasoning_chain': ['分析', '推理', '验证'],
            'conclusion': '因此，这个方法是正确的'
        }
        mock_parse.return_value = mock_structure
        
        # 模拟各种评估方法
        self.calculator._evaluate_thinking_completeness = Mock(return_value=0.9)
        self.calculator._evaluate_logical_consistency = Mock(return_value=0.8)
        self.calculator._evaluate_reasoning_depth = Mock(return_value=0.85)
        self.calculator._evaluate_step_clarity = Mock(return_value=0.75)
        self.calculator._evaluate_conclusion_quality = Mock(return_value=0.8)
        self.calculator._evaluate_crypto_reasoning_accuracy = Mock(return_value=0.7)
        self.calculator._evaluate_technical_terminology_usage = Mock(return_value=0.75)
        
        thinking_text = "步骤1：分析问题。步骤2：寻找解决方案。步骤3：验证结果。"
        
        metrics = self.calculator.calculate_thinking_quality_metrics(thinking_text)
        
        assert isinstance(metrics, ThinkingQualityMetrics)
        assert metrics.thinking_steps_count == 3
        assert metrics.reasoning_chain_length == 3
        assert metrics.thinking_completeness == 0.9
        assert metrics.logical_consistency == 0.8
    
    def test_track_crypto_term_learning_progress(self):
        """测试密码学术语学习进度跟踪"""
        # 模拟密码学术语
        mock_term1 = Mock()
        mock_term1.term = "AES"
        mock_term1.complexity = 5
        mock_term1.category = CryptoCategory.SYMMETRIC_ENCRYPTION
        
        mock_term2 = Mock()
        mock_term2.term = "RSA"
        mock_term2.complexity = 7
        mock_term2.category = CryptoCategory.ASYMMETRIC_ENCRYPTION
        
        # 模拟术语提取结果 - 需要为每次调用提供返回值
        extract_calls = [
            [mock_term1],  # predicted_texts[0]
            [mock_term1, mock_term2],  # reference_texts[0]
            [mock_term2],  # predicted_texts[1]
            [mock_term2],   # reference_texts[1]
            # 为category统计添加额外的调用
            [mock_term1],  # predicted_texts[0] 再次调用
            [mock_term1, mock_term2],  # reference_texts[0] 再次调用
            [mock_term2],  # predicted_texts[1] 再次调用
            [mock_term2]   # reference_texts[1] 再次调用
        ]
        self.calculator.crypto_processor.extract_crypto_terms = Mock(side_effect=extract_calls)
        
        # 模拟术语信息获取
        self.calculator.crypto_processor.get_term_info = Mock(side_effect=[mock_term1, mock_term2])
        
        predicted_texts = ["使用AES加密", "使用RSA加密"]
        reference_texts = ["使用AES和RSA加密", "使用RSA加密"]
        
        progress = self.calculator.track_crypto_term_learning_progress(
            predicted_texts, reference_texts
        )
        
        assert isinstance(progress, CryptoTermLearningProgress)
        assert progress.total_terms_encountered == 2  # AES, RSA
        assert progress.correctly_used_terms == 2     # 两个都正确使用
        assert progress.intermediate_terms_mastered == 1  # AES (complexity=5)
        assert progress.advanced_terms_mastered == 1     # RSA (complexity=7)
    
    def test_create_comprehensive_chinese_metrics(self):
        """测试创建综合中文指标"""
        # 模拟各种计算方法
        mock_char_metrics = ChineseCharacterMetrics()
        mock_char_metrics.character_accuracy = 0.9
        
        mock_word_metrics = ChineseWordMetrics()
        mock_word_metrics.word_accuracy = 0.85
        mock_word_metrics.crypto_terms_count = 5
        mock_word_metrics.crypto_terms_correct = 4
        mock_word_metrics.calculate_accuracy()
        
        mock_semantic_metrics = ChineseSemanticMetrics()
        mock_semantic_metrics.rouge_l_score = 0.8
        mock_semantic_metrics.bleu_score = 0.75
        mock_semantic_metrics.semantic_similarity = 0.85
        mock_semantic_metrics.fluency_score = 0.9
        mock_semantic_metrics.coherence_score = 0.8
        
        self.calculator.calculate_character_metrics = Mock(return_value=mock_char_metrics)
        self.calculator.calculate_word_metrics = Mock(return_value=mock_word_metrics)
        self.calculator.calculate_semantic_metrics = Mock(return_value=mock_semantic_metrics)
        
        predicted = "这是预测文本"
        reference = "这是参考文本"
        
        metrics = self.calculator.create_comprehensive_chinese_metrics(predicted, reference)
        
        assert isinstance(metrics, ChineseMetrics)
        assert metrics.character_accuracy == 0.9
        assert metrics.word_accuracy == mock_word_metrics.word_accuracy
        assert metrics.rouge_l_chinese == 0.8
        assert metrics.bleu_chinese == 0.75
        assert metrics.crypto_term_accuracy == 0.8  # 4/5
        assert len(self.calculator.historical_metrics) == 1
    
    def test_get_historical_trend(self):
        """测试获取历史趋势"""
        # 添加一些历史数据
        for i in range(5):
            metrics = ChineseMetrics(
                character_accuracy=0.8 + i * 0.02,
                word_accuracy=0.75 + i * 0.03,
                rouge_l_chinese=0.7 + i * 0.04,
                bleu_chinese=0.65 + i * 0.05,
                crypto_term_accuracy=0.6 + i * 0.06
            )
            self.calculator.historical_metrics.append(metrics)
        
        # 测试获取字符准确率趋势
        char_trend = self.calculator.get_historical_trend("character_accuracy", 3)
        assert len(char_trend) == 3
        assert char_trend[-1] == 0.88  # 最新的值
        
        # 测试获取词准确率趋势
        word_trend = self.calculator.get_historical_trend("word_accuracy", 5)
        assert len(word_trend) == 5
        assert word_trend[0] == 0.75   # 最早的值
        assert word_trend[-1] == 0.87  # 最新的值
        
        # 测试不存在的指标
        invalid_trend = self.calculator.get_historical_trend("invalid_metric")
        assert len(invalid_trend) == 0
    
    def test_export_metrics_report(self):
        """测试导出指标报告"""
        # 测试没有历史数据的情况
        empty_report = self.calculator.export_metrics_report()
        assert "error" in empty_report
        
        # 添加历史数据
        metrics = ChineseMetrics(
            character_accuracy=0.9,
            word_accuracy=0.85,
            rouge_l_chinese=0.8,
            bleu_chinese=0.75,
            crypto_term_accuracy=0.8
        )
        self.calculator.historical_metrics.append(metrics)
        
        # 添加密码学学习进度
        progress = CryptoTermLearningProgress()
        progress.total_terms_encountered = 10
        progress.correctly_used_terms = 8
        self.calculator.crypto_progress_history.append(progress)
        
        report = self.calculator.export_metrics_report()
        
        assert "summary" in report
        assert "trends" in report
        assert "crypto_learning_progress" in report
        
        # 检查摘要信息
        summary = report["summary"]
        assert summary["total_evaluations"] == 1
        assert summary["latest_character_accuracy"] == 0.9
        assert summary["latest_word_accuracy"] == 0.85
        
        # 检查趋势信息
        trends = report["trends"]
        assert "character_accuracy" in trends
        assert "word_accuracy" in trends
        
        # 检查密码学学习进度
        crypto_progress = report["crypto_learning_progress"]
        assert crypto_progress["total_terms_encountered"] == 10
        assert crypto_progress["correctly_used_terms"] == 8


class TestPrivateMethods:
    """测试私有方法"""
    
    def setup_method(self):
        """测试前设置"""
        self.calculator = ChineseMetricsCalculator()
    
    def test_parse_thinking_structure(self):
        """测试思考结构解析"""
        thinking_text = """
        1. 首先分析问题的本质
        2. 然后寻找可能的解决方案
        3. 最后验证解决方案的有效性
        因此，这个方法是可行的。
        """
        
        structure = self.calculator._parse_thinking_structure(thinking_text)
        
        assert "steps" in structure
        assert "reasoning_chain" in structure
        assert "conclusion" in structure
        assert len(structure["steps"]) >= 3
    
    def test_evaluate_thinking_completeness(self):
        """测试思考完整性评估"""
        # 测试完整的思考结构
        complete_structure = {
            "steps": [
                "步骤1：分析问题",
                "步骤2：寻找解决方案", 
                "步骤3：实施方案",
                "步骤4：验证结果",
                "步骤5：总结经验"
            ]
        }
        
        completeness = self.calculator._evaluate_thinking_completeness(complete_structure)
        assert 0.0 <= completeness <= 1.0
        assert completeness > 0.6  # 5步应该得到较高分数
        
        # 测试不完整的思考结构
        incomplete_structure = {"steps": ["只有一个步骤"]}
        incomplete_completeness = self.calculator._evaluate_thinking_completeness(incomplete_structure)
        assert incomplete_completeness < completeness
    
    def test_evaluate_logical_consistency(self):
        """测试逻辑一致性评估"""
        # 测试有逻辑连接的步骤
        consistent_structure = {
            "steps": [
                "首先，我们需要分析问题",
                "因为问题很复杂，所以我们需要分步解决",
                "因此，我们采用这个方案"
            ]
        }
        
        consistency = self.calculator._evaluate_logical_consistency(consistent_structure)
        assert 0.0 <= consistency <= 1.0
        assert consistency > 0.5  # 应该有较好的一致性
        
        # 测试单步情况
        single_step_structure = {"steps": ["只有一个步骤"]}
        single_consistency = self.calculator._evaluate_logical_consistency(single_step_structure)
        assert single_consistency == 1.0  # 单步认为一致
    
    def test_get_term_context(self):
        """测试获取术语上下文"""
        text = "在密码学中，AES是一种对称加密算法，广泛应用于数据保护。"
        term = "AES"
        
        context = self.calculator._get_term_context(text, term)
        
        assert term in context
        assert len(context) <= len(text)
        assert "密码学" in context or "对称加密" in context  # 应该包含相关上下文
    
    def test_evaluate_term_context_accuracy(self):
        """测试术语上下文准确性评估"""
        # 创建模拟术语
        term = CryptoTerm(
            term="AES",
            definition="高级加密标准",
            category=CryptoCategory.SYMMETRIC_ENCRYPTION,
            complexity=5,
            related_terms=["对称加密", "加密算法"]
        )
        
        # 测试包含相关术语的上下文
        good_context = "AES是一种对称加密算法"
        good_accuracy = self.calculator._evaluate_term_context_accuracy(term, good_context)
        assert 0.0 <= good_accuracy <= 1.0
        assert good_accuracy > 0.5
        
        # 测试不包含相关术语的上下文
        poor_context = "AES是一个缩写"
        poor_accuracy = self.calculator._evaluate_term_context_accuracy(term, poor_context)
        assert poor_accuracy <= good_accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])