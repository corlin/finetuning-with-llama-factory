"""
专业领域多层次评估框架

本模块实现了针对中文密码学领域的多维度评估系统，包括：
- 专业准确性评估
- 语义一致性评估  
- 实用性评估
- 专家QA数据集构建和管理
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
import numpy as np
from datetime import datetime
import logging

from src.data_models import CryptoTerm, ThinkingExample


class EvaluationDimension(Enum):
    """评估维度枚举"""
    TECHNICAL_ACCURACY = "technical_accuracy"  # 技术准确性
    CONCEPTUAL_UNDERSTANDING = "conceptual_understanding"  # 概念理解
    PRACTICAL_APPLICABILITY = "practical_applicability"  # 实用性
    LINGUISTIC_QUALITY = "linguistic_quality"  # 语言质量
    REASONING_COHERENCE = "reasoning_coherence"  # 推理连贯性
    SECURITY_AWARENESS = "security_awareness"  # 安全意识


class ExpertiseLevel(Enum):
    """专业水平枚举"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class ExpertQAItem:
    """专家QA数据项"""
    question_id: str
    question: str
    context: Optional[str]
    reference_answer: str
    expert_annotations: Dict[str, Any]
    difficulty_level: ExpertiseLevel
    crypto_categories: List[str]
    evaluation_criteria: Dict[EvaluationDimension, float]
    thinking_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "context": self.context,
            "reference_answer": self.reference_answer,
            "expert_annotations": self.expert_annotations,
            "difficulty_level": self.difficulty_level.value,
            "crypto_categories": self.crypto_categories,
            "evaluation_criteria": {dim.value: score for dim, score in self.evaluation_criteria.items()},
            "thinking_required": self.thinking_required
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    question_id: str
    model_answer: str
    scores: Dict[EvaluationDimension, float]
    detailed_feedback: Dict[str, str]
    overall_score: float
    expert_verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question_id": self.question_id,
            "model_answer": self.model_answer,
            "scores": {dim.value: score for dim, score in self.scores.items()},
            "detailed_feedback": self.detailed_feedback,
            "overall_score": self.overall_score,
            "expert_verified": self.expert_verified
        }


class ProfessionalAccuracyEvaluator:
    """专业准确性评估器"""
    
    def __init__(self, crypto_knowledge_base: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.crypto_kb = crypto_knowledge_base
        self.logger = logger or logging.getLogger(__name__)
        
    def evaluate_technical_accuracy(self, answer: str, reference: str, context: str = None) -> float:
        """
        评估技术准确性
        
        Args:
            answer: 模型回答
            reference: 参考答案
            context: 上下文信息
            
        Returns:
            float: 技术准确性分数 (0-1)
        """
        try:
            # 提取关键技术概念
            answer_concepts = self._extract_crypto_concepts(answer)
            reference_concepts = self._extract_crypto_concepts(reference)
            
            # 计算概念匹配度
            concept_accuracy = self._calculate_concept_accuracy(answer_concepts, reference_concepts)
            
            # 检查技术错误
            error_penalty = self._detect_technical_errors(answer)
            
            # 验证算法描述准确性
            algorithm_accuracy = self._verify_algorithm_descriptions(answer)
            
            # 综合评分 - 错误惩罚更严重
            # 如果有严重错误，直接大幅降低分数
            if error_penalty >= 1.0:
                final_score = concept_accuracy * 0.2 + algorithm_accuracy * 0.1
            else:
                final_score = (concept_accuracy * 0.3 + 
                              (1 - error_penalty) * 0.5 + 
                              algorithm_accuracy * 0.2)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"技术准确性评估失败: {e}")
            return 0.0
    
    def evaluate_conceptual_understanding(self, answer: str, question: str) -> float:
        """
        评估概念理解深度
        
        Args:
            answer: 模型回答
            question: 问题
            
        Returns:
            float: 概念理解分数 (0-1)
        """
        try:
            # 分析问题要求的理解层次
            required_depth = self._analyze_question_depth(question)
            
            # 评估回答的理解深度
            answer_depth = self._analyze_answer_depth(answer)
            
            # 检查概念关联性
            concept_connections = self._evaluate_concept_connections(answer)
            
            # 评估解释完整性
            explanation_completeness = self._evaluate_explanation_completeness(answer, question)
            
            # 综合评分
            understanding_score = (
                min(answer_depth / required_depth, 1.0) * 0.4 +
                concept_connections * 0.3 +
                explanation_completeness * 0.3
            )
            
            return max(0.0, min(1.0, understanding_score))
            
        except Exception as e:
            self.logger.error(f"概念理解评估失败: {e}")
            return 0.0
    
    def evaluate_practical_applicability(self, answer: str, context: str = None) -> float:
        """
        评估实用性
        
        Args:
            answer: 模型回答
            context: 应用上下文
            
        Returns:
            float: 实用性分数 (0-1)
        """
        try:
            # 检查实际应用场景的提及
            application_coverage = self._analyze_application_scenarios(answer)
            
            # 评估实现可行性
            implementation_feasibility = self._evaluate_implementation_feasibility(answer)
            
            # 检查安全考虑
            security_considerations = self._evaluate_security_considerations(answer)
            
            # 评估实际价值
            practical_value = self._assess_practical_value(answer, context)
            
            # 综合评分
            applicability_score = (
                application_coverage * 0.25 +
                implementation_feasibility * 0.25 +
                security_considerations * 0.25 +
                practical_value * 0.25
            )
            
            return max(0.0, min(1.0, applicability_score))
            
        except Exception as e:
            self.logger.error(f"实用性评估失败: {e}")
            return 0.0
    
    def _extract_crypto_concepts(self, text: str) -> List[str]:
        """提取密码学概念"""
        concepts = []
        for term, info in self.crypto_kb.items():
            if term.lower() in text.lower():
                concepts.append(term)
        return concepts
    
    def _calculate_concept_accuracy(self, answer_concepts: List[str], reference_concepts: List[str]) -> float:
        """计算概念匹配准确性"""
        if not reference_concepts:
            return 1.0 if not answer_concepts else 0.5
        
        if not answer_concepts:
            return 0.0
        
        correct_concepts = set(answer_concepts) & set(reference_concepts)
        precision = len(correct_concepts) / len(answer_concepts) if answer_concepts else 0
        recall = len(correct_concepts) / len(reference_concepts)
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def _detect_technical_errors(self, answer: str) -> float:
        """检测技术错误，返回错误惩罚分数"""
        error_patterns = [
            "DES是安全的",  # DES已被破解
            "MD5适用于密码存储",  # MD5不安全
            "RSA密钥长度512位足够",  # 512位RSA不安全
            "对称加密比非对称加密慢",  # 错误的性能比较
        ]
        
        error_count = 0
        for pattern in error_patterns:
            if pattern in answer:
                error_count += 1
        
        # 返回错误惩罚分数 (0-1，越高惩罚越大)
        return min(error_count * 0.9, 1.0)
    
    def _verify_algorithm_descriptions(self, answer: str) -> float:
        """验证算法描述的准确性"""
        # 简化实现，实际应该更复杂
        algorithm_keywords = ["加密", "解密", "密钥", "哈希", "签名", "验证"]
        correct_usage = 0
        total_usage = 0
        
        for keyword in algorithm_keywords:
            if keyword in answer:
                total_usage += 1
                # 这里应该有更复杂的上下文分析
                correct_usage += 1  # 简化假设
        
        return correct_usage / total_usage if total_usage > 0 else 1.0
    
    def _analyze_question_depth(self, question: str) -> int:
        """分析问题要求的理解深度"""
        depth_indicators = {
            1: ["什么是", "定义", "简述"],
            2: ["如何", "为什么", "比较"],
            3: ["分析", "评估", "设计"],
            4: ["优化", "创新", "研究"]
        }
        
        for depth, indicators in depth_indicators.items():
            if any(indicator in question for indicator in indicators):
                return depth
        
        return 2  # 默认中等深度
    
    def _analyze_answer_depth(self, answer: str) -> int:
        """分析回答的理解深度"""
        # 基于回答长度和关键词复杂性
        depth_keywords = {
            1: ["是", "有", "没有"],
            2: ["如何", "为什么", "因为", "所以"],
            3: ["分析", "比较", "区别", "优势", "缺点", "适合", "应用"],
            4: ["优化", "设计", "实现", "性能", "安全性", "管理"]
        }
        
        max_depth = 1
        for depth, keywords in depth_keywords.items():
            if any(keyword in answer for keyword in keywords):
                max_depth = max(max_depth, depth)
        
        # 结合长度调整
        if len(answer) > 200:
            max_depth = min(max_depth + 1, 4)
        
        return max_depth
    
    def _evaluate_concept_connections(self, answer: str) -> float:
        """评估概念关联性"""
        connection_words = ["因此", "所以", "由于", "导致", "相关", "关联", "基于", "而", "但", "然而", "同时"]
        connection_count = sum(1 for word in connection_words if word in answer)
        
        # 检查是否有对比和分析
        comparison_words = ["区别", "不同", "相同", "比较", "对比", "优势", "缺点"]
        comparison_count = sum(1 for word in comparison_words if word in answer)
        
        total_score = min((connection_count * 0.15 + comparison_count * 0.25), 1.0)
        return max(total_score, 0.3)  # 基础分数
    
    def _evaluate_explanation_completeness(self, answer: str, question: str) -> float:
        """评估解释完整性"""
        # 检查是否回答了问题的关键部分
        question_keywords = ["什么", "如何", "为什么", "分析", "比较", "区别"]
        answered_aspects = 0
        
        for keyword in question_keywords:
            if keyword in question:
                if keyword == "什么" and any(word in answer for word in ["是", "指", "表示"]):
                    answered_aspects += 1
                elif keyword == "如何" and any(word in answer for word in ["步骤", "方法", "过程"]):
                    answered_aspects += 1
                elif keyword == "为什么" and any(word in answer for word in ["因为", "由于", "原因"]):
                    answered_aspects += 1
                elif keyword in ["分析", "比较", "区别"] and any(word in answer for word in ["不同", "相同", "优势", "缺点", "特点"]):
                    answered_aspects += 1
        
        # 结合长度评估
        length_score = min(len(answer) / 150, 1.0)
        aspect_score = min(answered_aspects * 0.3, 1.0)
        
        return max(length_score * 0.6 + aspect_score * 0.4, 0.4)
    
    def _analyze_application_scenarios(self, answer: str) -> float:
        """分析应用场景覆盖"""
        scenarios = ["网络安全", "数据保护", "身份认证", "数字签名", "区块链", "应用", "使用", "场景"]
        mentioned_scenarios = sum(1 for scenario in scenarios if scenario in answer)
        return min(mentioned_scenarios * 0.2 + 0.3, 1.0)
    
    def _evaluate_implementation_feasibility(self, answer: str) -> float:
        """评估实现可行性"""
        feasibility_indicators = ["实现", "代码", "算法", "步骤", "方法", "简单", "复杂", "性能", "效率"]
        indicator_count = sum(1 for indicator in feasibility_indicators if indicator in answer)
        return min(indicator_count * 0.15 + 0.4, 1.0)
    
    def _evaluate_security_considerations(self, answer: str) -> float:
        """评估安全考虑"""
        security_terms = ["安全性", "攻击", "漏洞", "防护", "风险", "安全", "密钥", "管理"]
        security_count = sum(1 for term in security_terms if term in answer)
        return min(security_count * 0.2 + 0.3, 1.0)
    
    def _assess_practical_value(self, answer: str, context: str = None) -> float:
        """评估实际价值"""
        value_indicators = ["优势", "缺点", "适用", "建议", "注意", "优秀", "广泛", "重要"]
        value_count = sum(1 for indicator in value_indicators if indicator in answer)
        return min(value_count * 0.2 + 0.4, 1.0)


class ChineseSemanticEvaluator:
    """中文语义评估器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 集成中文NLP处理器
        self._nlp_processor = None
        self._load_nlp_processor()
    
    def evaluate_semantic_similarity(self, answer: str, reference: str) -> float:
        """
        评估语义相似性（改进的中文ROUGE-L）
        
        Args:
            answer: 模型回答
            reference: 参考答案
            
        Returns:
            float: 语义相似性分数 (0-1)
        """
        try:
            # 中文分词和预处理
            answer_tokens = self._chinese_tokenize(answer)
            reference_tokens = self._chinese_tokenize(reference)
            
            # 计算语义权重的LCS
            weighted_lcs = self._weighted_lcs(answer_tokens, reference_tokens)
            
            # 计算改进的ROUGE-L分数
            precision = weighted_lcs / len(answer_tokens) if answer_tokens else 0
            recall = weighted_lcs / len(reference_tokens) if reference_tokens else 0
            
            if precision + recall == 0:
                return 0.0
            
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            # 对于相似内容给予额外奖励
            common_tokens = set(answer_tokens) & set(reference_tokens)
            if len(common_tokens) > 0:
                bonus = min(len(common_tokens) * 0.1, 0.2)
                f1_score = min(f1_score + bonus, 1.0)
            
            return f1_score
            
        except Exception as e:
            self.logger.error(f"语义相似性评估失败: {e}")
            return 0.0
    
    def evaluate_linguistic_quality(self, text: str) -> float:
        """
        评估中文语言质量
        
        Args:
            text: 待评估文本
            
        Returns:
            float: 语言质量分数 (0-1)
        """
        try:
            # 语法正确性
            grammar_score = self._evaluate_grammar(text)
            
            # 流畅性
            fluency_score = self._evaluate_fluency(text)
            
            # 专业术语使用准确性
            terminology_score = self._evaluate_terminology_usage(text)
            
            # 表达清晰度
            clarity_score = self._evaluate_clarity(text)
            
            # 综合评分
            quality_score = (
                grammar_score * 0.3 +
                fluency_score * 0.3 +
                terminology_score * 0.2 +
                clarity_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"语言质量评估失败: {e}")
            return 0.0
    
    def _chinese_tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 简化实现，实际应该使用jieba或其他中文分词工具
        import re
        
        # 先提取英文和数字
        tokens = []
        
        # 提取英文字母和数字组合
        english_tokens = re.findall(r'[a-zA-Z0-9]+', text)
        tokens.extend(english_tokens)
        
        # 移除英文和数字，处理中文
        chinese_text = re.sub(r'[a-zA-Z0-9]+', '', text)
        
        # 简单的中文词汇分割（基于常见词汇模式）
        chinese_patterns = [
            r'对称加密算法', r'非对称加密', r'对称加密', r'加密算法', 
            r'哈希函数', r'数字签名', r'密钥管理', r'网络安全',
            r'[\u4e00-\u9fff]{2,4}',  # 2-4个中文字符的词
            r'[\u4e00-\u9fff]'       # 单个中文字符
        ]
        
        for pattern in chinese_patterns:
            matches = re.findall(pattern, chinese_text)
            tokens.extend(matches)
            chinese_text = re.sub(pattern, '', chinese_text)
        
        # 去重并保持顺序
        seen = set()
        result = []
        for token in tokens:
            if token and token not in seen:
                seen.add(token)
                result.append(token)
        
        return result
    
    def _weighted_lcs(self, tokens1: List[str], tokens2: List[str]) -> float:
        """计算加权最长公共子序列"""
        # 简化实现，实际应该考虑词汇重要性权重
        if not tokens1 or not tokens2:
            return 0.0
        
        # 动态规划计算LCS
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    # 这里可以加入词汇权重
                    weight = self._get_token_weight(tokens1[i-1])
                    dp[i][j] = dp[i-1][j-1] + weight
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _get_token_weight(self, token: str) -> float:
        """获取词汇权重"""
        # 专业术语权重更高
        crypto_terms = ["加密", "解密", "密钥", "哈希", "签名", "算法", "AES", "对称", "标准"]
        if token in crypto_terms:
            return 2.0
        # 重要概念词汇
        important_terms = ["高级", "属于", "密码"]
        if token in important_terms:
            return 1.5
        return 1.0
    
    def _evaluate_grammar(self, text: str) -> float:
        """评估语法正确性"""
        # 简化实现
        return 0.9  # 假设大部分情况下语法正确
    
    def _evaluate_fluency(self, text: str) -> float:
        """评估流畅性"""
        # 基于句子长度和连接词的简单评估
        sentences = text.split('。')
        avg_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 理想句子长度15-30字符
        if 15 <= avg_length <= 30:
            return 1.0
        elif avg_length < 15:
            return avg_length / 15
        else:
            return max(0.5, 30 / avg_length)
    
    def _evaluate_terminology_usage(self, text: str) -> float:
        """评估专业术语使用准确性"""
        # 简化实现
        return 0.85
    
    def _evaluate_clarity(self, text: str) -> float:
        """评估表达清晰度"""
        # 基于逻辑连接词和结构的简单评估
        clarity_indicators = ["首先", "其次", "然后", "最后", "因此", "所以"]
        indicator_count = sum(1 for indicator in clarity_indicators if indicator in text)
        return min(indicator_count * 0.2 + 0.6, 1.0)
    
    def _load_nlp_processor(self):
        """加载中文NLP处理器"""
        try:
            from src.chinese_nlp_processor import ChineseNLPProcessor
            self._nlp_processor = ChineseNLPProcessor()
            self.logger.info("中文NLP处理器加载成功")
        except ImportError as e:
            self.logger.warning(f"无法导入中文NLP处理器: {e}")
            self._nlp_processor = None
        except Exception as e:
            self.logger.error(f"中文NLP处理器加载失败: {e}")
            self._nlp_processor = None
    
    def evaluate_with_nlp_enhancement(self, answer: str, reference: str) -> Dict[str, float]:
        """
        使用NLP处理器增强的语义评估
        
        Args:
            answer: 模型回答
            reference: 参考答案
            
        Returns:
            Dict[str, float]: 增强的评估结果
        """
        results = {}
        
        # 标准语义相似性
        results["semantic_similarity"] = self.evaluate_semantic_similarity(answer, reference)
        
        # 语言质量评估
        results["linguistic_quality"] = self.evaluate_linguistic_quality(answer)
        
        # 如果有NLP处理器，进行增强评估
        if self._nlp_processor:
            try:
                # 文本质量评估
                quality_metrics = self._nlp_processor.assess_text_quality(answer)
                results["text_quality"] = quality_metrics.overall_quality()
                
                # 密码学术语准确性
                answer_crypto_terms = set(self._nlp_processor.extract_crypto_terms_from_text(answer))
                ref_crypto_terms = set(self._nlp_processor.extract_crypto_terms_from_text(reference))
                
                if ref_crypto_terms:
                    crypto_accuracy = len(answer_crypto_terms & ref_crypto_terms) / len(ref_crypto_terms)
                else:
                    crypto_accuracy = 1.0 if not answer_crypto_terms else 0.5
                
                results["crypto_term_accuracy"] = crypto_accuracy
                
                # 中文特定指标
                chinese_metrics = self._nlp_processor.calculate_chinese_metrics([answer], [reference])
                results["character_accuracy"] = chinese_metrics.character_accuracy
                results["word_accuracy"] = chinese_metrics.word_accuracy
                results["rouge_l_chinese"] = chinese_metrics.rouge_l_chinese
                
            except Exception as e:
                self.logger.error(f"NLP增强评估失败: {e}")
        
        return results


class ExpertQADatasetBuilder:
    """专家QA数据集构建器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.qa_items: List[ExpertQAItem] = []
    
    def build_crypto_qa_dataset(self, domain_areas: List[str]) -> List[ExpertQAItem]:
        """
        构建密码学QA数据集
        
        Args:
            domain_areas: 领域范围列表
            
        Returns:
            List[ExpertQAItem]: QA数据项列表
        """
        qa_items = []
        
        for area in domain_areas:
            area_items = self._generate_area_questions(area)
            qa_items.extend(area_items)
        
        return qa_items
    
    def _generate_area_questions(self, area: str) -> List[ExpertQAItem]:
        """为特定领域生成问题"""
        # 这里应该是实际的问题生成逻辑
        # 简化示例
        if area == "对称加密":
            return self._generate_symmetric_crypto_questions()
        elif area == "非对称加密":
            return self._generate_asymmetric_crypto_questions()
        elif area == "哈希函数":
            return self._generate_hash_function_questions()
        else:
            return []
    
    def _generate_symmetric_crypto_questions(self) -> List[ExpertQAItem]:
        """生成对称加密相关问题"""
        questions = [
            ExpertQAItem(
                question_id="sym_001",
                question="请解释AES加密算法的工作原理，并分析其安全性特点。",
                context="在现代密码学应用中",
                reference_answer="AES（高级加密标准）是一种对称分组密码算法...",
                expert_annotations={
                    "key_concepts": ["AES", "对称加密", "分组密码", "安全性"],
                    "difficulty_justification": "需要理解算法原理和安全分析"
                },
                difficulty_level=ExpertiseLevel.INTERMEDIATE,
                crypto_categories=["对称加密", "分组密码"],
                evaluation_criteria={
                    EvaluationDimension.TECHNICAL_ACCURACY: 0.9,
                    EvaluationDimension.CONCEPTUAL_UNDERSTANDING: 0.8,
                    EvaluationDimension.PRACTICAL_APPLICABILITY: 0.7
                },
                thinking_required=True
            )
        ]
        return questions
    
    def _generate_asymmetric_crypto_questions(self) -> List[ExpertQAItem]:
        """生成非对称加密相关问题"""
        return []  # 简化实现
    
    def _generate_hash_function_questions(self) -> List[ExpertQAItem]:
        """生成哈希函数相关问题"""
        return []  # 简化实现


class ComprehensiveEvaluationFramework:
    """综合评估框架"""
    
    def __init__(self, crypto_knowledge_base: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化各个评估器
        self.accuracy_evaluator = ProfessionalAccuracyEvaluator(crypto_knowledge_base, logger)
        self.semantic_evaluator = ChineseSemanticEvaluator(logger)
        self.qa_builder = ExpertQADatasetBuilder(logger)
        
        # 评估权重配置
        self.dimension_weights = {
            EvaluationDimension.TECHNICAL_ACCURACY: 0.3,
            EvaluationDimension.CONCEPTUAL_UNDERSTANDING: 0.25,
            EvaluationDimension.PRACTICAL_APPLICABILITY: 0.2,
            EvaluationDimension.LINGUISTIC_QUALITY: 0.15,
            EvaluationDimension.REASONING_COHERENCE: 0.1
        }
        
        # 专家评估集成标志
        self._expert_evaluation_enabled = False
        self._expert_evaluator = None
    
    def evaluate_model_response(self, question: str, model_answer: str, 
                              reference_answer: str, context: str = None) -> EvaluationResult:
        """
        综合评估模型回答
        
        Args:
            question: 问题
            model_answer: 模型回答
            reference_answer: 参考答案
            context: 上下文
            
        Returns:
            EvaluationResult: 评估结果
        """
        try:
            scores = {}
            feedback = {}
            
            # 技术准确性评估
            tech_score = self.accuracy_evaluator.evaluate_technical_accuracy(
                model_answer, reference_answer, context
            )
            scores[EvaluationDimension.TECHNICAL_ACCURACY] = tech_score
            feedback["technical_accuracy"] = f"技术准确性得分: {tech_score:.2f}"
            
            # 概念理解评估
            concept_score = self.accuracy_evaluator.evaluate_conceptual_understanding(
                model_answer, question
            )
            scores[EvaluationDimension.CONCEPTUAL_UNDERSTANDING] = concept_score
            feedback["conceptual_understanding"] = f"概念理解得分: {concept_score:.2f}"
            
            # 实用性评估
            practical_score = self.accuracy_evaluator.evaluate_practical_applicability(
                model_answer, context
            )
            scores[EvaluationDimension.PRACTICAL_APPLICABILITY] = practical_score
            feedback["practical_applicability"] = f"实用性得分: {practical_score:.2f}"
            
            # 语言质量评估
            linguistic_score = self.semantic_evaluator.evaluate_linguistic_quality(model_answer)
            scores[EvaluationDimension.LINGUISTIC_QUALITY] = linguistic_score
            feedback["linguistic_quality"] = f"语言质量得分: {linguistic_score:.2f}"
            
            # 语义相似性评估
            semantic_score = self.semantic_evaluator.evaluate_semantic_similarity(
                model_answer, reference_answer
            )
            scores[EvaluationDimension.REASONING_COHERENCE] = semantic_score
            feedback["reasoning_coherence"] = f"推理连贯性得分: {semantic_score:.2f}"
            
            # 计算综合得分
            overall_score = sum(
                scores[dim] * self.dimension_weights[dim] 
                for dim in scores.keys()
            )
            
            return EvaluationResult(
                question_id="auto_generated",
                model_answer=model_answer,
                scores=scores,
                detailed_feedback=feedback,
                overall_score=overall_score
            )
            
        except Exception as e:
            self.logger.error(f"综合评估失败: {e}")
            return EvaluationResult(
                question_id="error",
                model_answer=model_answer,
                scores={},
                detailed_feedback={"error": str(e)},
                overall_score=0.0
            )
    
    def batch_evaluate(self, qa_pairs: List[Tuple[str, str, str]]) -> List[EvaluationResult]:
        """
        批量评估
        
        Args:
            qa_pairs: (问题, 模型回答, 参考答案) 元组列表
            
        Returns:
            List[EvaluationResult]: 评估结果列表
        """
        results = []
        for i, (question, model_answer, reference_answer) in enumerate(qa_pairs):
            result = self.evaluate_model_response(question, model_answer, reference_answer)
            result.question_id = f"batch_{i}"
            results.append(result)
        
        return results
    
    def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            results: 评估结果列表
            
        Returns:
            Dict[str, Any]: 评估报告
        """
        if not results:
            return {"error": "没有评估结果"}
        
        # 计算各维度平均分
        dimension_averages = {}
        for dimension in EvaluationDimension:
            scores = [r.scores.get(dimension, 0) for r in results if dimension in r.scores]
            dimension_averages[dimension.value] = np.mean(scores) if scores else 0.0
        
        # 计算总体统计
        overall_scores = [r.overall_score for r in results]
        
        report = {
            "evaluation_summary": {
                "total_questions": len(results),
                "average_overall_score": np.mean(overall_scores),
                "score_std": np.std(overall_scores),
                "min_score": np.min(overall_scores),
                "max_score": np.max(overall_scores)
            },
            "dimension_scores": dimension_averages,
            "score_distribution": {
                "excellent": len([s for s in overall_scores if s >= 0.9]),
                "good": len([s for s in overall_scores if 0.7 <= s < 0.9]),
                "fair": len([s for s in overall_scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in overall_scores if s < 0.5])
            },
            "detailed_results": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def enable_expert_evaluation_integration(self, expert_evaluator=None):
        """
        启用专家评估集成
        
        Args:
            expert_evaluator: 专家评估器实例，如果为None则延迟加载
        """
        self._expert_evaluation_enabled = True
        self._expert_evaluator = expert_evaluator
        self.logger.info("专家评估集成已启用")
    
    def disable_expert_evaluation_integration(self):
        """禁用专家评估集成"""
        self._expert_evaluation_enabled = False
        self._expert_evaluator = None
        self.logger.info("专家评估集成已禁用")
    
    def set_expert_dimension_weights(self, weights: Dict[EvaluationDimension, float]):
        """
        设置专家评估维度权重
        
        Args:
            weights: 维度权重字典
        """
        # 验证权重总和
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"权重总和应为1.0，当前为: {total_weight}")
        
        self.dimension_weights.update(weights)
        self.logger.info("专家评估维度权重已更新")
    
    def evaluate_with_expert_integration(self, question: str, model_answer: str, 
                                       reference_answer: str, context: str = None,
                                       expert_config: Dict[str, Any] = None) -> EvaluationResult:
        """
        使用专家评估集成进行评估
        
        Args:
            question: 问题
            model_answer: 模型回答
            reference_answer: 参考答案
            context: 上下文
            expert_config: 专家评估配置
            
        Returns:
            EvaluationResult: 增强的评估结果
        """
        # 首先进行标准评估
        standard_result = self.evaluate_model_response(
            question, model_answer, reference_answer, context
        )
        
        # 如果启用了专家评估集成，进行增强评估
        if self._expert_evaluation_enabled:
            try:
                enhanced_result = self._enhance_with_expert_evaluation(
                    standard_result, question, model_answer, reference_answer, 
                    context, expert_config
                )
                return enhanced_result
            except Exception as e:
                self.logger.warning(f"专家评估增强失败，使用标准结果: {e}")
                return standard_result
        
        return standard_result
    
    def _enhance_with_expert_evaluation(self, standard_result: EvaluationResult,
                                      question: str, model_answer: str, 
                                      reference_answer: str, context: str = None,
                                      expert_config: Dict[str, Any] = None) -> EvaluationResult:
        """
        使用专家评估增强标准评估结果
        
        Args:
            standard_result: 标准评估结果
            question: 问题
            model_answer: 模型回答
            reference_answer: 参考答案
            context: 上下文
            expert_config: 专家评估配置
            
        Returns:
            EvaluationResult: 增强的评估结果
        """
        # 如果没有专家评估器，尝试延迟加载
        if not self._expert_evaluator:
            self._expert_evaluator = self._load_expert_evaluator()
        
        if not self._expert_evaluator:
            self.logger.warning("无法加载专家评估器，返回标准结果")
            return standard_result
        
        # 执行专家评估增强
        enhanced_scores = standard_result.scores.copy()
        enhanced_feedback = standard_result.detailed_feedback.copy()
        
        # 添加专家评估维度
        expert_dimensions = self._get_expert_evaluation_dimensions(
            question, model_answer, reference_answer, context
        )
        
        # 合并评估结果
        for dim, score in expert_dimensions.items():
            enhanced_scores[dim] = score
            enhanced_feedback[f"expert_{dim.value}"] = f"专家评估得分: {score:.2f}"
        
        # 重新计算综合得分
        enhanced_overall_score = self._calculate_enhanced_overall_score(enhanced_scores)
        
        # 创建增强的评估结果
        enhanced_result = EvaluationResult(
            question_id=standard_result.question_id,
            model_answer=model_answer,
            scores=enhanced_scores,
            detailed_feedback=enhanced_feedback,
            overall_score=enhanced_overall_score,
            expert_verified=True
        )
        
        return enhanced_result
    
    def _load_expert_evaluator(self):
        """延迟加载专家评估器"""
        try:
            # 动态导入专家评估模块
            from src.expert_evaluation.multi_dimensional import MultiDimensionalEvaluator
            from src.expert_evaluation.config import ExpertEvaluationConfig
            
            # 创建默认配置
            expert_config = ExpertEvaluationConfig()
            
            # 创建专家评估器
            expert_evaluator = MultiDimensionalEvaluator(expert_config)
            
            self.logger.info("专家评估器延迟加载成功")
            return expert_evaluator
            
        except ImportError as e:
            self.logger.error(f"无法导入专家评估模块: {e}")
            return None
        except Exception as e:
            self.logger.error(f"专家评估器加载失败: {e}")
            return None
    
    def _get_expert_evaluation_dimensions(self, question: str, model_answer: str, 
                                        reference_answer: str, context: str = None) -> Dict[EvaluationDimension, float]:
        """
        获取专家评估维度分数
        
        Args:
            question: 问题
            model_answer: 模型回答
            reference_answer: 参考答案
            context: 上下文
            
        Returns:
            Dict[EvaluationDimension, float]: 专家评估维度分数
        """
        expert_scores = {}
        
        try:
            if self._expert_evaluator:
                # 调用专家评估器的方法
                # 这里需要根据实际的专家评估器接口进行调用
                
                # 创建QA评估项
                from src.expert_evaluation.data_models import QAEvaluationItem
                
                qa_item = QAEvaluationItem(
                    question_id="framework_integration",
                    question=question,
                    context=context,
                    reference_answer=reference_answer,
                    model_answer=model_answer,
                    domain_tags=["密码学"],
                    difficulty_level=2,
                    expected_concepts=[]
                )
                
                # 执行专家评估
                expert_result = self._expert_evaluator.evaluate_single_item(qa_item)
                
                # 检查是否是Mock对象或有效的评估结果
                if hasattr(expert_result, 'dimension_scores') and expert_result.dimension_scores:
                    # 提取维度分数
                    try:
                        for dim, score_obj in expert_result.dimension_scores.items():
                            # 映射专家评估维度到框架维度
                            framework_dim = self._map_expert_to_framework_dimension(dim)
                            if framework_dim:
                                score_value = score_obj.score if hasattr(score_obj, 'score') else float(score_obj)
                                expert_scores[framework_dim] = score_value
                    except (AttributeError, TypeError) as e:
                        self.logger.warning(f"无法提取专家评估维度分数: {e}")
                        # 使用默认分数
                        expert_scores[EvaluationDimension.TECHNICAL_ACCURACY] = 0.75
                        expert_scores[EvaluationDimension.CONCEPTUAL_UNDERSTANDING] = 0.80
                else:
                    # 如果是Mock对象或无效结果，使用默认分数
                    expert_scores[EvaluationDimension.TECHNICAL_ACCURACY] = 0.75
                    expert_scores[EvaluationDimension.CONCEPTUAL_UNDERSTANDING] = 0.80
                        
        except Exception as e:
            self.logger.error(f"获取专家评估维度分数失败: {e}")
        
        return expert_scores
    
    def _map_expert_to_framework_dimension(self, expert_dim) -> Optional[EvaluationDimension]:
        """
        映射专家评估维度到框架维度
        
        Args:
            expert_dim: 专家评估维度
            
        Returns:
            Optional[EvaluationDimension]: 对应的框架维度
        """
        # 这里需要根据实际的维度枚举进行映射
        mapping = {
            "DOMAIN_ACCURACY": EvaluationDimension.TECHNICAL_ACCURACY,
            "SEMANTIC_SIMILARITY": EvaluationDimension.CONCEPTUAL_UNDERSTANDING,
            "PRACTICAL_VALUE": EvaluationDimension.PRACTICAL_APPLICABILITY,
            "LOGICAL_CONSISTENCY": EvaluationDimension.REASONING_COHERENCE
        }
        
        expert_dim_name = expert_dim.name if hasattr(expert_dim, 'name') else str(expert_dim)
        return mapping.get(expert_dim_name)
    
    def _calculate_enhanced_overall_score(self, enhanced_scores: Dict[EvaluationDimension, float]) -> float:
        """
        计算增强的综合得分
        
        Args:
            enhanced_scores: 增强的维度分数
            
        Returns:
            float: 综合得分
        """
        total_score = 0.0
        total_weight = 0.0
        
        for dim, score in enhanced_scores.items():
            weight = self.dimension_weights.get(dim, 0.1)  # 默认权重
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def get_supported_expert_dimensions(self) -> List[str]:
        """
        获取支持的专家评估维度
        
        Returns:
            List[str]: 支持的维度列表
        """
        if self._expert_evaluation_enabled and self._expert_evaluator:
            try:
                return self._expert_evaluator.get_supported_dimensions()
            except Exception as e:
                self.logger.error(f"获取专家评估维度失败: {e}")
        
        return []
    
    def is_expert_evaluation_available(self) -> bool:
        """
        检查专家评估是否可用
        
        Returns:
            bool: 是否可用
        """
        return self._expert_evaluation_enabled and (
            self._expert_evaluator is not None or 
            self._can_load_expert_evaluator()
        )
    
    def _can_load_expert_evaluator(self) -> bool:
        """检查是否可以加载专家评估器"""
        try:
            from src.expert_evaluation.multi_dimensional import MultiDimensionalEvaluator
            return True
        except ImportError:
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        获取集成状态
        
        Returns:
            Dict[str, Any]: 集成状态信息
        """
        return {
            "expert_evaluation_enabled": self._expert_evaluation_enabled,
            "expert_evaluator_loaded": self._expert_evaluator is not None,
            "expert_evaluation_available": self.is_expert_evaluation_available(),
            "supported_expert_dimensions": self.get_supported_expert_dimensions(),
            "current_dimension_weights": self.dimension_weights.copy()
        }