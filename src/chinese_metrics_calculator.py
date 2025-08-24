"""
中文特定指标计算模块

本模块实现了中文文本的专门指标计算功能，包括中文字符级和词级准确率计算、
密码学术语学习进度跟踪、thinking数据质量评估指标等功能。
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict
import json
import logging
import numpy as np

from .data_models import ChineseMetrics, CryptoTerm, ThinkingStructure
from .chinese_nlp_processor import ChineseNLPProcessor
from .crypto_term_processor import CryptoTermProcessor


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChineseCharacterMetrics:
    """中文字符级指标"""
    total_characters: int = 0
    correct_characters: int = 0
    character_accuracy: float = 0.0
    
    # 字符类型统计
    hanzi_count: int = 0
    punctuation_count: int = 0
    number_count: int = 0
    english_count: int = 0
    
    # 错误分析
    substitution_errors: int = 0  # 替换错误
    insertion_errors: int = 0     # 插入错误
    deletion_errors: int = 0      # 删除错误
    
    def calculate_accuracy(self):
        """计算字符准确率"""
        if self.total_characters > 0:
            self.character_accuracy = self.correct_characters / self.total_characters
        else:
            self.character_accuracy = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_characters": self.total_characters,
            "correct_characters": self.correct_characters,
            "character_accuracy": self.character_accuracy,
            "hanzi_count": self.hanzi_count,
            "punctuation_count": self.punctuation_count,
            "number_count": self.number_count,
            "english_count": self.english_count,
            "substitution_errors": self.substitution_errors,
            "insertion_errors": self.insertion_errors,
            "deletion_errors": self.deletion_errors
        }


@dataclass
class ChineseWordMetrics:
    """中文词级指标"""
    total_words: int = 0
    correct_words: int = 0
    word_accuracy: float = 0.0
    
    # 词长度统计
    single_char_words: int = 0    # 单字词
    two_char_words: int = 0       # 双字词
    multi_char_words: int = 0     # 多字词
    
    # 词性统计
    noun_count: int = 0           # 名词
    verb_count: int = 0           # 动词
    adjective_count: int = 0      # 形容词
    adverb_count: int = 0         # 副词
    
    # 专业术语统计
    crypto_terms_count: int = 0   # 密码学术语数量
    crypto_terms_correct: int = 0 # 正确的密码学术语数量
    
    def calculate_accuracy(self):
        """计算词准确率"""
        if self.total_words > 0:
            self.word_accuracy = self.correct_words / self.total_words
        else:
            self.word_accuracy = 0.0
    
    @property
    def crypto_term_accuracy(self) -> float:
        """密码学术语准确率"""
        if self.crypto_terms_count > 0:
            return self.crypto_terms_correct / self.crypto_terms_count
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_words": self.total_words,
            "correct_words": self.correct_words,
            "word_accuracy": self.word_accuracy,
            "single_char_words": self.single_char_words,
            "two_char_words": self.two_char_words,
            "multi_char_words": self.multi_char_words,
            "noun_count": self.noun_count,
            "verb_count": self.verb_count,
            "adjective_count": self.adjective_count,
            "adverb_count": self.adverb_count,
            "crypto_terms_count": self.crypto_terms_count,
            "crypto_terms_correct": self.crypto_terms_correct,
            "crypto_term_accuracy": self.crypto_term_accuracy
        }


@dataclass
class ChineseSemanticMetrics:
    """中文语义指标"""
    rouge_l_score: float = 0.0
    bleu_score: float = 0.0
    semantic_similarity: float = 0.0
    
    # 语言质量指标
    fluency_score: float = 0.0      # 流畅性
    coherence_score: float = 0.0    # 连贯性
    naturalness_score: float = 0.0  # 自然性
    
    # 专业性指标
    terminology_usage: float = 0.0   # 术语使用准确性
    technical_depth: float = 0.0     # 技术深度
    
    def calculate_overall_score(self) -> float:
        """计算综合语义评分"""
        weights = {
            'rouge_l': 0.25,
            'bleu': 0.25,
            'semantic_similarity': 0.20,
            'fluency': 0.10,
            'coherence': 0.10,
            'terminology_usage': 0.10
        }
        
        return (
            self.rouge_l_score * weights['rouge_l'] +
            self.bleu_score * weights['bleu'] +
            self.semantic_similarity * weights['semantic_similarity'] +
            self.fluency_score * weights['fluency'] +
            self.coherence_score * weights['coherence'] +
            self.terminology_usage * weights['terminology_usage']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "rouge_l_score": self.rouge_l_score,
            "bleu_score": self.bleu_score,
            "semantic_similarity": self.semantic_similarity,
            "fluency_score": self.fluency_score,
            "coherence_score": self.coherence_score,
            "naturalness_score": self.naturalness_score,
            "terminology_usage": self.terminology_usage,
            "technical_depth": self.technical_depth,
            "overall_score": self.calculate_overall_score()
        }


@dataclass
class ThinkingQualityMetrics:
    """思考质量指标"""
    thinking_completeness: float = 0.0    # 思考完整性
    logical_consistency: float = 0.0      # 逻辑一致性
    reasoning_depth: float = 0.0          # 推理深度
    step_clarity: float = 0.0             # 步骤清晰度
    
    # 思考结构指标
    thinking_steps_count: int = 0         # 思考步骤数量
    reasoning_chain_length: int = 0       # 推理链长度
    conclusion_quality: float = 0.0       # 结论质量
    
    # 专业性指标
    crypto_reasoning_accuracy: float = 0.0  # 密码学推理准确性
    technical_terminology_usage: float = 0.0  # 技术术语使用
    
    def calculate_overall_quality(self) -> float:
        """计算综合思考质量评分"""
        weights = {
            'completeness': 0.25,
            'consistency': 0.25,
            'depth': 0.20,
            'clarity': 0.15,
            'crypto_accuracy': 0.15
        }
        
        return (
            self.thinking_completeness * weights['completeness'] +
            self.logical_consistency * weights['consistency'] +
            self.reasoning_depth * weights['depth'] +
            self.step_clarity * weights['clarity'] +
            self.crypto_reasoning_accuracy * weights['crypto_accuracy']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "thinking_completeness": self.thinking_completeness,
            "logical_consistency": self.logical_consistency,
            "reasoning_depth": self.reasoning_depth,
            "step_clarity": self.step_clarity,
            "thinking_steps_count": self.thinking_steps_count,
            "reasoning_chain_length": self.reasoning_chain_length,
            "conclusion_quality": self.conclusion_quality,
            "crypto_reasoning_accuracy": self.crypto_reasoning_accuracy,
            "technical_terminology_usage": self.technical_terminology_usage,
            "overall_quality": self.calculate_overall_quality()
        }


@dataclass
class CryptoTermLearningProgress:
    """密码学术语学习进度"""
    total_terms_encountered: int = 0      # 遇到的术语总数
    correctly_used_terms: int = 0         # 正确使用的术语数
    incorrectly_used_terms: int = 0       # 错误使用的术语数
    
    # 按难度级别统计
    beginner_terms_mastered: int = 0      # 掌握的初级术语
    intermediate_terms_mastered: int = 0  # 掌握的中级术语
    advanced_terms_mastered: int = 0      # 掌握的高级术语
    expert_terms_mastered: int = 0        # 掌握的专家级术语
    
    # 按类别统计
    category_progress: Dict[str, float] = field(default_factory=dict)
    
    # 学习趋势
    improvement_rate: float = 0.0         # 改进率
    consistency_score: float = 0.0        # 一致性评分
    
    @property
    def overall_accuracy(self) -> float:
        """总体准确率"""
        if self.total_terms_encountered > 0:
            return self.correctly_used_terms / self.total_terms_encountered
        return 0.0
    
    @property
    def mastery_level(self) -> str:
        """掌握水平"""
        accuracy = self.overall_accuracy
        if accuracy >= 0.9:
            return "专家"
        elif accuracy >= 0.8:
            return "高级"
        elif accuracy >= 0.7:
            return "中级"
        elif accuracy >= 0.6:
            return "初级"
        else:
            return "入门"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_terms_encountered": self.total_terms_encountered,
            "correctly_used_terms": self.correctly_used_terms,
            "incorrectly_used_terms": self.incorrectly_used_terms,
            "overall_accuracy": self.overall_accuracy,
            "mastery_level": self.mastery_level,
            "beginner_terms_mastered": self.beginner_terms_mastered,
            "intermediate_terms_mastered": self.intermediate_terms_mastered,
            "advanced_terms_mastered": self.advanced_terms_mastered,
            "expert_terms_mastered": self.expert_terms_mastered,
            "category_progress": self.category_progress,
            "improvement_rate": self.improvement_rate,
            "consistency_score": self.consistency_score
        }


class ChineseMetricsCalculator:
    """中文指标计算器"""
    
    def __init__(self):
        """初始化中文指标计算器"""
        self.chinese_nlp = ChineseNLPProcessor()
        self.crypto_processor = CryptoTermProcessor()
        
        # 中文字符分类正则表达式
        self.hanzi_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.punctuation_pattern = re.compile(r'[，。！？；：""''（）【】《》、]')
        self.number_pattern = re.compile(r'[0-9０-９]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        
        # 历史数据用于趋势分析
        self.historical_metrics: List[ChineseMetrics] = []
        self.crypto_progress_history: List[CryptoTermLearningProgress] = []
        
        logger.info("中文指标计算器初始化完成")
    
    def calculate_character_metrics(self, predicted_text: str, reference_text: str) -> ChineseCharacterMetrics:
        """计算字符级指标"""
        metrics = ChineseCharacterMetrics()
        
        # 字符对齐和比较
        pred_chars = list(predicted_text)
        ref_chars = list(reference_text)
        
        metrics.total_characters = len(ref_chars)
        
        # 使用编辑距离算法计算字符级错误
        edit_operations = self._calculate_edit_distance_operations(pred_chars, ref_chars)
        
        # 统计错误类型
        for op_type, _, _ in edit_operations:
            if op_type == 'substitute':
                metrics.substitution_errors += 1
            elif op_type == 'insert':
                metrics.insertion_errors += 1
            elif op_type == 'delete':
                metrics.deletion_errors += 1
        
        # 计算正确字符数
        total_errors = metrics.substitution_errors + metrics.insertion_errors + metrics.deletion_errors
        metrics.correct_characters = max(0, metrics.total_characters - total_errors)
        
        # 统计字符类型
        for char in ref_chars:
            if self.hanzi_pattern.match(char):
                metrics.hanzi_count += 1
            elif self.punctuation_pattern.match(char):
                metrics.punctuation_count += 1
            elif self.number_pattern.match(char):
                metrics.number_count += 1
            elif self.english_pattern.match(char):
                metrics.english_count += 1
        
        # 计算准确率
        metrics.calculate_accuracy()
        
        return metrics
    
    def calculate_word_metrics(self, predicted_text: str, reference_text: str) -> ChineseWordMetrics:
        """计算词级指标"""
        metrics = ChineseWordMetrics()
        
        # 分词
        pred_words = self.chinese_nlp.segment_text(predicted_text)
        ref_words = self.chinese_nlp.segment_text(reference_text)
        
        metrics.total_words = len(ref_words)
        
        # 计算词级准确率（使用集合交集）
        pred_word_set = set(pred_words)
        ref_word_set = set(ref_words)
        correct_words = pred_word_set.intersection(ref_word_set)
        metrics.correct_words = len(correct_words)
        
        # 统计词长度
        for word in ref_words:
            word_len = len(word)
            if word_len == 1:
                metrics.single_char_words += 1
            elif word_len == 2:
                metrics.two_char_words += 1
            else:
                metrics.multi_char_words += 1
        
        # 词性标注和统计
        pos_tags = self.chinese_nlp.pos_tag(ref_words)
        for word, pos in pos_tags:
            if pos.startswith('n'):  # 名词
                metrics.noun_count += 1
            elif pos.startswith('v'):  # 动词
                metrics.verb_count += 1
            elif pos.startswith('a'):  # 形容词
                metrics.adjective_count += 1
            elif pos.startswith('d'):  # 副词
                metrics.adverb_count += 1
        
        # 密码学术语统计
        ref_crypto_terms = self.crypto_processor.extract_crypto_terms(reference_text)
        pred_crypto_terms = self.crypto_processor.extract_crypto_terms(predicted_text)
        
        metrics.crypto_terms_count = len(ref_crypto_terms)
        
        # 计算正确的密码学术语
        ref_term_set = set(term.term for term in ref_crypto_terms)
        pred_term_set = set(term.term for term in pred_crypto_terms)
        correct_crypto_terms = ref_term_set.intersection(pred_term_set)
        metrics.crypto_terms_correct = len(correct_crypto_terms)
        
        # 计算准确率
        metrics.calculate_accuracy()
        
        return metrics
    
    def calculate_semantic_metrics(self, predicted_text: str, reference_text: str) -> ChineseSemanticMetrics:
        """计算语义指标"""
        metrics = ChineseSemanticMetrics()
        
        # ROUGE-L计算（针对中文优化）
        metrics.rouge_l_score = self._calculate_chinese_rouge_l(predicted_text, reference_text)
        
        # BLEU计算（针对中文优化）
        metrics.bleu_score = self._calculate_chinese_bleu(predicted_text, reference_text)
        
        # 语义相似度计算
        metrics.semantic_similarity = self._calculate_semantic_similarity(predicted_text, reference_text)
        
        # 语言质量评估
        metrics.fluency_score = self._evaluate_fluency(predicted_text)
        metrics.coherence_score = self._evaluate_coherence(predicted_text)
        metrics.naturalness_score = self._evaluate_naturalness(predicted_text)
        
        # 专业性评估
        metrics.terminology_usage = self._evaluate_terminology_usage(predicted_text, reference_text)
        metrics.technical_depth = self._evaluate_technical_depth(predicted_text)
        
        return metrics
    
    def calculate_thinking_quality_metrics(self, thinking_text: str, 
                                         reference_thinking: Optional[str] = None) -> ThinkingQualityMetrics:
        """计算思考质量指标"""
        metrics = ThinkingQualityMetrics()
        
        # 解析思考结构
        thinking_structure = self._parse_thinking_structure(thinking_text)
        
        # 基础结构指标
        metrics.thinking_steps_count = len(thinking_structure.get('steps', []))
        metrics.reasoning_chain_length = len(thinking_structure.get('reasoning_chain', []))
        
        # 完整性评估
        metrics.thinking_completeness = self._evaluate_thinking_completeness(thinking_structure)
        
        # 逻辑一致性评估
        metrics.logical_consistency = self._evaluate_logical_consistency(thinking_structure)
        
        # 推理深度评估
        metrics.reasoning_depth = self._evaluate_reasoning_depth(thinking_structure)
        
        # 步骤清晰度评估
        metrics.step_clarity = self._evaluate_step_clarity(thinking_structure)
        
        # 结论质量评估
        metrics.conclusion_quality = self._evaluate_conclusion_quality(thinking_structure)
        
        # 密码学推理准确性评估
        metrics.crypto_reasoning_accuracy = self._evaluate_crypto_reasoning_accuracy(thinking_text)
        
        # 技术术语使用评估
        metrics.technical_terminology_usage = self._evaluate_technical_terminology_usage(thinking_text)
        
        return metrics
    
    def track_crypto_term_learning_progress(self, 
                                          predicted_texts: List[str],
                                          reference_texts: List[str]) -> CryptoTermLearningProgress:
        """跟踪密码学术语学习进度"""
        progress = CryptoTermLearningProgress()
        
        all_encountered_terms = set()
        correctly_used_terms = set()
        
        for pred_text, ref_text in zip(predicted_texts, reference_texts):
            # 提取术语
            pred_terms = self.crypto_processor.extract_crypto_terms(pred_text)
            ref_terms = self.crypto_processor.extract_crypto_terms(ref_text)
            
            # 统计遇到的术语
            ref_term_names = set(term.term for term in ref_terms)
            pred_term_names = set(term.term for term in pred_terms)
            
            all_encountered_terms.update(ref_term_names)
            correctly_used_terms.update(ref_term_names.intersection(pred_term_names))
        
        progress.total_terms_encountered = len(all_encountered_terms)
        progress.correctly_used_terms = len(correctly_used_terms)
        progress.incorrectly_used_terms = progress.total_terms_encountered - progress.correctly_used_terms
        
        # 按难度级别统计
        for term_name in correctly_used_terms:
            term_info = self.crypto_processor.get_term_info(term_name)
            if term_info:
                complexity = term_info.complexity
                if complexity <= 3:
                    progress.beginner_terms_mastered += 1
                elif complexity <= 5:
                    progress.intermediate_terms_mastered += 1
                elif complexity <= 7:
                    progress.advanced_terms_mastered += 1
                else:
                    progress.expert_terms_mastered += 1
        
        # 按类别统计进度
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred_text, ref_text in zip(predicted_texts, reference_texts):
            ref_terms = self.crypto_processor.extract_crypto_terms(ref_text)
            pred_terms = self.crypto_processor.extract_crypto_terms(pred_text)
            
            pred_term_names = set(term.term for term in pred_terms)
            
            for term in ref_terms:
                category = term.category.value
                category_stats[category]['total'] += 1
                if term.term in pred_term_names:
                    category_stats[category]['correct'] += 1
        
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                progress.category_progress[category] = stats['correct'] / stats['total']
        
        # 计算改进率（需要历史数据）
        if self.crypto_progress_history:
            last_progress = self.crypto_progress_history[-1]
            if last_progress.total_terms_encountered > 0:
                last_accuracy = last_progress.overall_accuracy
                current_accuracy = progress.overall_accuracy
                progress.improvement_rate = current_accuracy - last_accuracy
        
        # 计算一致性评分（基于最近几次的表现）
        recent_accuracies = [p.overall_accuracy for p in self.crypto_progress_history[-5:]]
        recent_accuracies.append(progress.overall_accuracy)
        if len(recent_accuracies) > 1:
            progress.consistency_score = 1.0 - np.std(recent_accuracies)
        
        # 保存历史数据
        self.crypto_progress_history.append(progress)
        
        return progress
    
    def create_comprehensive_chinese_metrics(self,
                                           predicted_text: str,
                                           reference_text: str,
                                           thinking_text: Optional[str] = None) -> ChineseMetrics:
        """创建综合中文指标"""
        
        # 计算各项指标
        char_metrics = self.calculate_character_metrics(predicted_text, reference_text)
        word_metrics = self.calculate_word_metrics(predicted_text, reference_text)
        semantic_metrics = self.calculate_semantic_metrics(predicted_text, reference_text)
        
        # 创建ChineseMetrics对象
        chinese_metrics = ChineseMetrics(
            character_accuracy=char_metrics.character_accuracy,
            word_accuracy=word_metrics.word_accuracy,
            rouge_l_chinese=semantic_metrics.rouge_l_score,
            bleu_chinese=semantic_metrics.bleu_score,
            crypto_term_accuracy=word_metrics.crypto_term_accuracy,
            semantic_similarity=semantic_metrics.semantic_similarity,
            fluency_score=semantic_metrics.fluency_score,
            coherence_score=semantic_metrics.coherence_score
        )
        
        # 保存历史数据
        self.historical_metrics.append(chinese_metrics)
        
        return chinese_metrics
    
    # 私有辅助方法
    def _calculate_edit_distance_operations(self, seq1: List[str], seq2: List[str]) -> List[Tuple[str, int, int]]:
        """计算编辑距离操作序列"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # 删除
                        dp[i][j-1],    # 插入
                        dp[i-1][j-1]   # 替换
                    )
        
        # 回溯操作序列
        operations = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq1[i-1] == seq2[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(('substitute', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append(('delete', i-1, -1))
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                operations.append(('insert', -1, j-1))
                j -= 1
        
        return operations[::-1]
    
    def _calculate_chinese_rouge_l(self, predicted: str, reference: str) -> float:
        """计算中文ROUGE-L分数"""
        # 分词
        pred_words = self.chinese_nlp.segment_text(predicted)
        ref_words = self.chinese_nlp.segment_text(reference)
        
        if not pred_words or not ref_words:
            return 0.0
        
        # 计算最长公共子序列
        lcs_length = self._longest_common_subsequence(pred_words, ref_words)
        
        # 计算ROUGE-L
        if len(ref_words) == 0:
            return 0.0
        
        recall = lcs_length / len(ref_words)
        precision = lcs_length / len(pred_words) if len(pred_words) > 0 else 0.0
        
        if recall + precision == 0:
            return 0.0
        
        f1_score = 2 * recall * precision / (recall + precision)
        return f1_score
    
    def _calculate_chinese_bleu(self, predicted: str, reference: str) -> float:
        """计算中文BLEU分数"""
        pred_words = self.chinese_nlp.segment_text(predicted)
        ref_words = self.chinese_nlp.segment_text(reference)
        
        if not pred_words or not ref_words:
            return 0.0
        
        # 计算1-gram到4-gram的精确度
        bleu_scores = []
        for n in range(1, 5):
            pred_ngrams = self._get_ngrams(pred_words, n)
            ref_ngrams = self._get_ngrams(ref_words, n)
            
            if not pred_ngrams:
                bleu_scores.append(0.0)
                continue
            
            # 计算n-gram精确度
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            bleu_scores.append(precision)
        
        # 计算几何平均
        if any(score == 0 for score in bleu_scores):
            return 0.0
        
        geometric_mean = math.exp(sum(math.log(score) for score in bleu_scores) / len(bleu_scores))
        
        # 简化的长度惩罚
        brevity_penalty = min(1.0, len(pred_words) / len(ref_words))
        
        return brevity_penalty * geometric_mean
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度（简化实现）"""
        # 这里使用简化的基于词汇重叠的相似度计算
        words1 = set(self.chinese_nlp.segment_text(text1))
        words2 = set(self.chinese_nlp.segment_text(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _evaluate_fluency(self, text: str) -> float:
        """评估流畅性"""
        # 简化的流畅性评估：基于句子长度分布和标点符号使用
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 计算句子长度方差（适中的方差表示流畅性好）
        lengths = [len(s) for s in sentences]
        mean_length = np.mean(lengths)
        length_variance = np.var(lengths)
        
        # 理想句子长度在10-30字之间
        length_score = 1.0 - abs(mean_length - 20) / 20
        length_score = max(0.0, min(1.0, length_score))
        
        # 方差评分（适中的方差得分更高）
        variance_score = 1.0 / (1.0 + length_variance / 100)
        
        return 0.7 * length_score + 0.3 * variance_score
    
    def _evaluate_coherence(self, text: str) -> float:
        """评估连贯性"""
        # 简化的连贯性评估：基于连接词和代词的使用
        coherence_indicators = [
            '因此', '所以', '但是', '然而', '而且', '另外', '首先', '其次', '最后',
            '这', '那', '它', '他们', '我们', '这样', '那样'
        ]
        
        indicator_count = 0
        for indicator in coherence_indicators:
            indicator_count += text.count(indicator)
        
        # 基于文本长度标准化
        text_length = len(text)
        if text_length == 0:
            return 0.0
        
        coherence_ratio = indicator_count / (text_length / 100)  # 每100字的连贯性指标数量
        
        # 适中的比例得分最高
        if coherence_ratio < 1:
            return coherence_ratio
        elif coherence_ratio < 3:
            return 1.0
        else:
            return max(0.0, 1.0 - (coherence_ratio - 3) / 5)
    
    def _evaluate_naturalness(self, text: str) -> float:
        """评估自然性"""
        # 简化的自然性评估：基于常用词汇和表达模式
        # 这里使用一个简化的实现
        words = self.chinese_nlp.segment_text(text)
        
        if not words:
            return 0.0
        
        # 计算词汇多样性
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # 适中的多样性表示自然性好
        if diversity < 0.3:
            return diversity / 0.3
        elif diversity < 0.7:
            return 1.0
        else:
            return max(0.0, 1.0 - (diversity - 0.7) / 0.3)
    
    def _evaluate_terminology_usage(self, predicted: str, reference: str) -> float:
        """评估术语使用准确性"""
        pred_terms = self.crypto_processor.extract_crypto_terms(predicted)
        ref_terms = self.crypto_processor.extract_crypto_terms(reference)
        
        if not ref_terms:
            return 1.0  # 如果参考文本没有术语，则认为使用正确
        
        pred_term_names = set(term.term for term in pred_terms)
        ref_term_names = set(term.term for term in ref_terms)
        
        correct_terms = pred_term_names.intersection(ref_term_names)
        
        return len(correct_terms) / len(ref_term_names)
    
    def _evaluate_technical_depth(self, text: str) -> float:
        """评估技术深度"""
        crypto_terms = self.crypto_processor.extract_crypto_terms(text)
        
        if not crypto_terms:
            return 0.0
        
        # 基于术语复杂度计算技术深度
        complexity_scores = [term.complexity for term in crypto_terms]
        avg_complexity = np.mean(complexity_scores)
        
        # 标准化到0-1范围
        return min(1.0, avg_complexity / 10.0)
    
    def _parse_thinking_structure(self, thinking_text: str) -> Dict[str, Any]:
        """解析思考结构"""
        # 简化的思考结构解析
        structure = {
            'steps': [],
            'reasoning_chain': [],
            'conclusion': ''
        }
        
        # 按行分割并识别步骤
        lines = thinking_text.split('\n')
        current_step = ''
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 识别步骤标记
            if re.match(r'^\d+[.、]', line) or line.startswith('步骤') or line.startswith('首先') or line.startswith('然后') or line.startswith('最后'):
                if current_step:
                    structure['steps'].append(current_step)
                current_step = line
            else:
                current_step += ' ' + line if current_step else line
        
        if current_step:
            structure['steps'].append(current_step)
        
        # 简化的推理链识别
        structure['reasoning_chain'] = structure['steps']  # 简化实现
        
        # 识别结论
        if structure['steps']:
            last_step = structure['steps'][-1]
            if '因此' in last_step or '所以' in last_step or '综上' in last_step:
                structure['conclusion'] = last_step
        
        return structure
    
    def _evaluate_thinking_completeness(self, thinking_structure: Dict[str, Any]) -> float:
        """评估思考完整性"""
        steps = thinking_structure.get('steps', [])
        
        if not steps:
            return 0.0
        
        # 基于步骤数量和内容长度评估完整性
        step_count_score = min(1.0, len(steps) / 5)  # 5步为满分
        
        total_length = sum(len(step) for step in steps)
        length_score = min(1.0, total_length / 500)  # 500字为满分
        
        return 0.6 * step_count_score + 0.4 * length_score
    
    def _evaluate_logical_consistency(self, thinking_structure: Dict[str, Any]) -> float:
        """评估逻辑一致性"""
        steps = thinking_structure.get('steps', [])
        
        if len(steps) < 2:
            return 1.0  # 单步或无步骤认为一致
        
        # 简化的一致性检查：查找逻辑连接词
        consistency_score = 0.0
        connection_words = ['因为', '所以', '因此', '由于', '既然', '那么']
        
        for i in range(1, len(steps)):
            step = steps[i]
            has_connection = any(word in step for word in connection_words)
            if has_connection:
                consistency_score += 1.0
        
        return consistency_score / (len(steps) - 1) if len(steps) > 1 else 1.0
    
    def _evaluate_reasoning_depth(self, thinking_structure: Dict[str, Any]) -> float:
        """评估推理深度"""
        steps = thinking_structure.get('steps', [])
        
        if not steps:
            return 0.0
        
        # 基于推理层次和复杂度评估深度
        depth_indicators = ['分析', '推导', '证明', '假设', '验证', '比较', '评估']
        
        depth_score = 0.0
        for step in steps:
            step_depth = sum(1 for indicator in depth_indicators if indicator in step)
            depth_score += min(1.0, step_depth / 3)  # 每步最多1分
        
        return depth_score / len(steps)
    
    def _evaluate_step_clarity(self, thinking_structure: Dict[str, Any]) -> float:
        """评估步骤清晰度"""
        steps = thinking_structure.get('steps', [])
        
        if not steps:
            return 0.0
        
        clarity_scores = []
        for step in steps:
            # 基于句子结构和长度评估清晰度
            sentences = re.split(r'[。！？]', step)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                clarity_scores.append(0.0)
                continue
            
            # 句子长度适中得分高
            avg_sentence_length = np.mean([len(s) for s in sentences])
            length_score = 1.0 - abs(avg_sentence_length - 20) / 20
            length_score = max(0.0, min(1.0, length_score))
            
            clarity_scores.append(length_score)
        
        return np.mean(clarity_scores) if clarity_scores else 0.0
    
    def _evaluate_conclusion_quality(self, thinking_structure: Dict[str, Any]) -> float:
        """评估结论质量"""
        conclusion = thinking_structure.get('conclusion', '')
        
        if not conclusion:
            return 0.0
        
        # 基于结论的完整性和逻辑性评估
        conclusion_indicators = ['综上', '因此', '所以', '总结', '结论']
        has_indicator = any(indicator in conclusion for indicator in conclusion_indicators)
        
        # 长度适中
        length_score = min(1.0, len(conclusion) / 100)  # 100字为满分
        
        # 是否有明确的结论性表述
        indicator_score = 1.0 if has_indicator else 0.5
        
        return 0.6 * indicator_score + 0.4 * length_score
    
    def _evaluate_crypto_reasoning_accuracy(self, thinking_text: str) -> float:
        """评估密码学推理准确性"""
        # 提取密码学术语
        crypto_terms = self.crypto_processor.extract_crypto_terms(thinking_text)
        
        if not crypto_terms:
            return 0.0
        
        # 简化的准确性评估：基于术语使用的上下文合理性
        accuracy_score = 0.0
        
        for term in crypto_terms:
            # 检查术语周围的上下文是否合理
            term_context = self._get_term_context(thinking_text, term.term)
            context_score = self._evaluate_term_context_accuracy(term, term_context)
            accuracy_score += context_score
        
        return accuracy_score / len(crypto_terms)
    
    def _evaluate_technical_terminology_usage(self, thinking_text: str) -> float:
        """评估技术术语使用"""
        crypto_terms = self.crypto_processor.extract_crypto_terms(thinking_text)
        
        if not crypto_terms:
            return 0.0
        
        # 基于术语密度和多样性评估
        text_length = len(thinking_text)
        term_density = len(crypto_terms) / (text_length / 100)  # 每100字的术语数量
        
        # 术语类别多样性
        categories = set(term.category for term in crypto_terms)
        category_diversity = len(categories) / len(self.crypto_processor.get_all_categories())
        
        # 综合评分
        density_score = min(1.0, term_density / 2)  # 每100字2个术语为满分
        diversity_score = category_diversity
        
        return 0.6 * density_score + 0.4 * diversity_score
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _get_ngrams(self, words: List[str], n: int) -> Counter:
        """获取n-gram计数"""
        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _get_term_context(self, text: str, term: str) -> str:
        """获取术语的上下文"""
        # 简化实现：获取术语前后各20个字符
        term_pos = text.find(term)
        if term_pos == -1:
            return ""
        
        start = max(0, term_pos - 20)
        end = min(len(text), term_pos + len(term) + 20)
        
        return text[start:end]
    
    def _evaluate_term_context_accuracy(self, term: CryptoTerm, context: str) -> float:
        """评估术语上下文准确性"""
        # 简化实现：检查上下文中是否包含相关术语
        related_terms = term.related_terms
        
        if not related_terms:
            return 0.8  # 默认评分
        
        related_count = sum(1 for related in related_terms if related in context)
        
        return min(1.0, 0.5 + related_count / len(related_terms) * 0.5)
    
    # 公共接口方法
    def get_historical_trend(self, metric_name: str, window_size: int = 10) -> List[float]:
        """获取指标的历史趋势"""
        if not self.historical_metrics:
            return []
        
        recent_metrics = self.historical_metrics[-window_size:]
        
        if metric_name == "character_accuracy":
            return [m.character_accuracy for m in recent_metrics]
        elif metric_name == "word_accuracy":
            return [m.word_accuracy for m in recent_metrics]
        elif metric_name == "rouge_l_chinese":
            return [m.rouge_l_chinese for m in recent_metrics]
        elif metric_name == "bleu_chinese":
            return [m.bleu_chinese for m in recent_metrics]
        elif metric_name == "crypto_term_accuracy":
            return [m.crypto_term_accuracy for m in recent_metrics]
        elif metric_name == "overall_score":
            return [m.overall_score() for m in recent_metrics]
        else:
            return []
    
    def export_metrics_report(self) -> Dict[str, Any]:
        """导出指标报告"""
        if not self.historical_metrics:
            return {"error": "没有历史指标数据"}
        
        latest_metrics = self.historical_metrics[-1]
        
        report = {
            "summary": {
                "total_evaluations": len(self.historical_metrics),
                "latest_overall_score": latest_metrics.overall_score(),
                "latest_character_accuracy": latest_metrics.character_accuracy,
                "latest_word_accuracy": latest_metrics.word_accuracy,
                "latest_crypto_term_accuracy": latest_metrics.crypto_term_accuracy
            },
            "trends": {
                "character_accuracy": self.get_historical_trend("character_accuracy"),
                "word_accuracy": self.get_historical_trend("word_accuracy"),
                "rouge_l_chinese": self.get_historical_trend("rouge_l_chinese"),
                "bleu_chinese": self.get_historical_trend("bleu_chinese"),
                "crypto_term_accuracy": self.get_historical_trend("crypto_term_accuracy"),
                "overall_score": self.get_historical_trend("overall_score")
            },
            "crypto_learning_progress": []
        }
        
        # 添加密码学学习进度
        if self.crypto_progress_history:
            latest_progress = self.crypto_progress_history[-1]
            report["crypto_learning_progress"] = latest_progress.to_dict()
        
        return report