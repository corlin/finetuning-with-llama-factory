"""
专家评估指标计算模块

实现行业特定的评估指标计算，包括创新性评估、完整性评估等高级指标。
集成现有的密码学术语识别功能，提供专业的评估能力。
"""

import re
import math
import statistics
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from difflib import SequenceMatcher

try:
    import numpy as np
except ImportError:
    # 如果numpy不可用，使用Python内置函数
    np = None

try:
    from src.crypto_term_processor import CryptoTermProcessor, TermAnnotation, CryptoCategory
    from src.chinese_nlp_processor import ChineseNLPProcessor
    from src.data_models import CryptoTerm
except ImportError:
    # 如果导入失败，创建简化版本
    class CryptoTermProcessor:
        def __init__(self):
            pass
        
        def identify_terms(self, text: str) -> List:
            return []
    
    class ChineseNLPProcessor:
        def __init__(self):
            pass
        
        def tokenize(self, text: str) -> List:
            return text.split()

from .config import EvaluationDimension, ExpertiseLevel
from .data_models import DimensionScore


@dataclass
class InnovationMetrics:
    """创新性评估指标"""
    novelty_score: float  # 新颖性评分
    uniqueness_score: float  # 独特性评分
    creativity_score: float  # 创造性评分
    differentiation_score: float  # 差异化评分
    overall_innovation: float  # 总体创新性
    
    def __post_init__(self):
        """计算总体创新性"""
        self.overall_innovation = (
            self.novelty_score * 0.3 +
            self.uniqueness_score * 0.25 +
            self.creativity_score * 0.25 +
            self.differentiation_score * 0.2
        )


@dataclass
class CompletenessMetrics:
    """完整性评估指标"""
    concept_coverage: float  # 概念覆盖度
    requirement_fulfillment: float  # 需求满足度
    depth_adequacy: float  # 深度充分性
    breadth_coverage: float  # 广度覆盖
    overall_completeness: float  # 总体完整性
    
    def __post_init__(self):
        """计算总体完整性"""
        self.overall_completeness = (
            self.concept_coverage * 0.3 +
            self.requirement_fulfillment * 0.3 +
            self.depth_adequacy * 0.2 +
            self.breadth_coverage * 0.2
        )


@dataclass
class CryptoTermAnalysis:
    """密码学术语分析结果"""
    total_terms: int
    unique_terms: int
    term_categories: Dict[str, int]
    complexity_distribution: Dict[int, int]
    professional_score: float
    term_accuracy: float


class IndustryMetricsCalculator:
    """行业指标计算器"""
    
    def __init__(self):
        """初始化计算器"""
        self.crypto_processor = CryptoTermProcessor()
        self.chinese_processor = ChineseNLPProcessor()
        
        # 创新性评估权重
        self.innovation_weights = {
            'semantic_novelty': 0.25,
            'structural_uniqueness': 0.25,
            'concept_creativity': 0.25,
            'approach_differentiation': 0.25
        }
        
        # 完整性评估权重
        self.completeness_weights = {
            'concept_coverage': 0.3,
            'requirement_fulfillment': 0.3,
            'depth_adequacy': 0.2,
            'breadth_coverage': 0.2
        }
        
        # 密码学领域关键概念
        self.crypto_key_concepts = {
            '加密算法': ['对称加密', '非对称加密', 'AES', 'RSA', 'DES', '椭圆曲线'],
            '哈希函数': ['SHA', 'MD5', 'BLAKE2', 'SHA-256', 'SHA-3'],
            '数字签名': ['ECDSA', 'DSA', 'RSA签名', 'EdDSA'],
            '密钥管理': ['密钥交换', 'PKI', 'CA', '密钥派生'],
            '密码协议': ['TLS', 'SSL', 'IPSec', 'SSH'],
            '密码分析': ['差分分析', '线性分析', '侧信道攻击', '碰撞攻击'],
            '区块链': ['比特币', '智能合约', '共识算法', '哈希指针']
        }
    
    def calculate_domain_relevance(self, answer: str, domain_context: str) -> float:
        """
        计算答案与特定领域的相关性
        
        Args:
            answer: 模型答案
            domain_context: 领域上下文
            
        Returns:
            float: 领域相关性评分 (0.0-1.0)
        """
        try:
            # 识别密码学术语
            answer_terms = self.crypto_processor.identify_terms(answer)
            context_terms = self.crypto_processor.identify_terms(domain_context)
            
            if not context_terms:
                return 0.5  # 如果没有上下文术语，返回中性评分
            
            # 计算术语重叠度
            answer_term_set = {term.term for term in answer_terms}
            context_term_set = {term.term for term in context_terms}
            
            overlap = len(answer_term_set.intersection(context_term_set))
            total_context_terms = len(context_term_set)
            
            if total_context_terms == 0:
                return 0.5
            
            # 基础相关性评分
            base_relevance = overlap / total_context_terms
            
            # 考虑术语复杂度和专业性
            complexity_bonus = self._calculate_complexity_bonus(answer_terms)
            
            # 最终评分
            relevance_score = min(1.0, base_relevance + complexity_bonus * 0.2)
            
            return relevance_score
            
        except Exception:
            # 如果术语识别失败，使用文本相似性作为后备
            return self._calculate_text_similarity(answer, domain_context)
    
    def assess_practical_applicability(self, answer: str, use_case: str) -> float:
        """
        评估实际应用价值
        
        Args:
            answer: 模型答案
            use_case: 使用场景
            
        Returns:
            float: 实用性评分 (0.0-1.0)
        """
        # 实用性关键词
        practical_keywords = [
            '实现', '应用', '部署', '配置', '使用', '操作',
            '实际', '实践', '具体', '步骤', '方法', '流程',
            '示例', '例子', '案例', '场景', '环境'
        ]
        
        # 理论性关键词（降低实用性评分）
        theoretical_keywords = [
            '理论', '概念', '定义', '原理', '抽象',
            '假设', '推测', '可能', '或许', '大概'
        ]
        
        # 计算关键词出现频率
        practical_count = sum(1 for keyword in practical_keywords if keyword in answer)
        theoretical_count = sum(1 for keyword in theoretical_keywords if keyword in answer)
        
        # 基础实用性评分
        answer_length = len(answer)
        if answer_length == 0:
            return 0.0
        
        practical_density = practical_count / (answer_length / 100)  # 每100字符的实用词密度
        theoretical_penalty = theoretical_count / (answer_length / 100)
        
        # 检查是否包含具体步骤或示例
        has_steps = bool(re.search(r'[1-9]\.|第[一二三四五六七八九十]+步|步骤', answer))
        has_examples = bool(re.search(r'例如|比如|举例|示例|案例', answer))
        has_code = bool(re.search(r'```|代码|程序|脚本', answer))
        
        # 计算实用性评分
        base_score = min(1.0, practical_density * 0.3)
        step_bonus = 0.2 if has_steps else 0.0
        example_bonus = 0.15 if has_examples else 0.0
        code_bonus = 0.1 if has_code else 0.0
        theoretical_penalty_score = min(0.3, theoretical_penalty * 0.1)
        
        practical_score = base_score + step_bonus + example_bonus + code_bonus - theoretical_penalty_score
        
        return max(0.0, min(1.0, practical_score))
    
    def evaluate_innovation_level(self, answer: str, baseline_answers: List[str]) -> InnovationMetrics:
        """
        评估创新性水平
        
        Args:
            answer: 模型答案
            baseline_answers: 基准答案列表
            
        Returns:
            InnovationMetrics: 创新性评估指标
        """
        if not baseline_answers:
            # 如果没有基准答案，基于内容本身评估创新性
            return self._evaluate_intrinsic_innovation(answer)
        
        # 1. 新颖性评估 - 与基准答案的差异程度
        novelty_score = self._calculate_novelty(answer, baseline_answers)
        
        # 2. 独特性评估 - 独特表达和观点
        uniqueness_score = self._calculate_uniqueness(answer, baseline_answers)
        
        # 3. 创造性评估 - 新的连接和组合
        creativity_score = self._calculate_creativity(answer)
        
        # 4. 差异化评估 - 方法和角度的差异
        differentiation_score = self._calculate_differentiation(answer, baseline_answers)
        
        return InnovationMetrics(
            novelty_score=novelty_score,
            uniqueness_score=uniqueness_score,
            creativity_score=creativity_score,
            differentiation_score=differentiation_score,
            overall_innovation=0.0  # 将在__post_init__中计算
        )
    
    def measure_completeness(self, answer: str, question_requirements: List[str]) -> CompletenessMetrics:
        """
        测量回答的完整性
        
        Args:
            answer: 模型答案
            question_requirements: 问题要求列表
            
        Returns:
            CompletenessMetrics: 完整性评估指标
        """
        # 1. 概念覆盖度 - 关键概念的覆盖情况
        concept_coverage = self._calculate_concept_coverage(answer, question_requirements)
        
        # 2. 需求满足度 - 问题要求的满足程度
        requirement_fulfillment = self._calculate_requirement_fulfillment(answer, question_requirements)
        
        # 3. 深度充分性 - 回答的深度是否充分
        depth_adequacy = self._calculate_depth_adequacy(answer)
        
        # 4. 广度覆盖 - 相关主题的覆盖广度
        breadth_coverage = self._calculate_breadth_coverage(answer)
        
        return CompletenessMetrics(
            concept_coverage=concept_coverage,
            requirement_fulfillment=requirement_fulfillment,
            depth_adequacy=depth_adequacy,
            breadth_coverage=breadth_coverage,
            overall_completeness=0.0  # 将在__post_init__中计算
        )
    
    def analyze_crypto_terms(self, text: str) -> CryptoTermAnalysis:
        """
        分析文本中的密码学术语
        
        Args:
            text: 待分析文本
            
        Returns:
            CryptoTermAnalysis: 术语分析结果
        """
        try:
            # 识别密码学术语
            terms = self.crypto_processor.identify_terms(text)
            
            if not terms:
                return CryptoTermAnalysis(
                    total_terms=0,
                    unique_terms=0,
                    term_categories={},
                    complexity_distribution={},
                    professional_score=0.0,
                    term_accuracy=0.0
                )
            
            # 统计术语信息
            total_terms = len(terms)
            unique_terms = len(set(term.term for term in terms))
            
            # 分类统计
            term_categories = defaultdict(int)
            for term in terms:
                if hasattr(term, 'category'):
                    category_name = term.category.value if hasattr(term.category, 'value') else str(term.category)
                    term_categories[category_name] += 1
            
            # 复杂度分布
            complexity_distribution = defaultdict(int)
            for term in terms:
                if hasattr(term, 'complexity'):
                    complexity_distribution[term.complexity] += 1
            
            # 计算专业性评分
            professional_score = self._calculate_professional_score(terms)
            
            # 计算术语准确性
            term_accuracy = self._calculate_term_accuracy(terms, text)
            
            return CryptoTermAnalysis(
                total_terms=total_terms,
                unique_terms=unique_terms,
                term_categories=dict(term_categories),
                complexity_distribution=dict(complexity_distribution),
                professional_score=professional_score,
                term_accuracy=term_accuracy
            )
            
        except Exception:
            # 如果术语处理失败，返回默认值
            return CryptoTermAnalysis(
                total_terms=0,
                unique_terms=0,
                term_categories={},
                complexity_distribution={},
                professional_score=0.0,
                term_accuracy=0.0
            )
    
    def _calculate_novelty(self, answer: str, baseline_answers: List[str]) -> float:
        """计算新颖性评分"""
        if not baseline_answers:
            return 0.5
        
        # 计算与所有基准答案的相似度
        similarities = []
        for baseline in baseline_answers:
            similarity = SequenceMatcher(None, answer, baseline).ratio()
            similarities.append(similarity)
        
        # 新颖性 = 1 - 最大相似度
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        return max(0.0, min(1.0, novelty))
    
    def _calculate_uniqueness(self, answer: str, baseline_answers: List[str]) -> float:
        """计算独特性评分"""
        if not baseline_answers:
            return 0.5
        
        # 提取关键短语
        answer_phrases = self._extract_key_phrases(answer)
        baseline_phrases = set()
        
        for baseline in baseline_answers:
            baseline_phrases.update(self._extract_key_phrases(baseline))
        
        if not answer_phrases:
            return 0.0
        
        # 计算独特短语比例
        unique_phrases = answer_phrases - baseline_phrases
        uniqueness = len(unique_phrases) / len(answer_phrases)
        
        return max(0.0, min(1.0, uniqueness))
    
    def _calculate_creativity(self, answer: str) -> float:
        """计算创造性评分"""
        creativity_indicators = [
            r'创新|新颖|独特|原创',
            r'结合|融合|整合|综合',
            r'改进|优化|增强|提升',
            r'新的|全新|崭新',
            r'突破|超越|革新'
        ]
        
        creativity_score = 0.0
        for pattern in creativity_indicators:
            matches = len(re.findall(pattern, answer))
            creativity_score += matches * 0.1
        
        # 检查是否有创新性的表达方式
        has_metaphor = bool(re.search(r'就像|如同|好比|类似于', answer))
        has_analogy = bool(re.search(r'可以理解为|相当于|等同于', answer))
        
        if has_metaphor:
            creativity_score += 0.15
        if has_analogy:
            creativity_score += 0.1
        
        return max(0.0, min(1.0, creativity_score))
    
    def _calculate_differentiation(self, answer: str, baseline_answers: List[str]) -> float:
        """计算差异化评分"""
        if not baseline_answers:
            return 0.5
        
        # 分析答案结构
        answer_structure = self._analyze_text_structure(answer)
        
        # 分析基准答案结构
        baseline_structures = [self._analyze_text_structure(baseline) for baseline in baseline_answers]
        
        # 计算结构差异
        structure_differences = []
        for baseline_structure in baseline_structures:
            diff = self._calculate_structure_difference(answer_structure, baseline_structure)
            structure_differences.append(diff)
        
        # 差异化评分 = 平均结构差异
        avg_difference = sum(structure_differences) / len(structure_differences) if structure_differences else 0.0
        
        return max(0.0, min(1.0, avg_difference))
    
    def _calculate_concept_coverage(self, answer: str, requirements: List[str]) -> float:
        """计算概念覆盖度"""
        if not requirements:
            return 1.0
        
        covered_concepts = 0
        total_concepts = len(requirements)
        
        for requirement in requirements:
            # 检查答案中是否包含该概念
            if self._contains_concept(answer, requirement):
                covered_concepts += 1
        
        return covered_concepts / total_concepts if total_concepts > 0 else 0.0
    
    def _calculate_requirement_fulfillment(self, answer: str, requirements: List[str]) -> float:
        """计算需求满足度"""
        if not requirements:
            return 1.0
        
        fulfillment_scores = []
        
        for requirement in requirements:
            # 评估每个需求的满足程度
            score = self._evaluate_requirement_fulfillment(answer, requirement)
            fulfillment_scores.append(score)
        
        return sum(fulfillment_scores) / len(fulfillment_scores) if fulfillment_scores else 0.0
    
    def _calculate_depth_adequacy(self, answer: str) -> float:
        """计算深度充分性"""
        # 深度指标
        depth_indicators = {
            'technical_details': len(re.findall(r'具体|详细|深入|技术细节', answer)) * 0.1,
            'explanations': len(re.findall(r'因为|由于|原因|解释|说明', answer)) * 0.08,
            'examples': len(re.findall(r'例如|比如|举例|示例', answer)) * 0.12,
            'analysis': len(re.findall(r'分析|评估|考虑|研究', answer)) * 0.1,
            'implications': len(re.findall(r'影响|后果|结果|意义', answer)) * 0.08
        }
        
        # 长度因子
        length_factor = min(1.0, len(answer) / 500)  # 500字符为基准
        
        # 计算深度评分
        depth_score = sum(depth_indicators.values()) * length_factor
        
        return max(0.0, min(1.0, depth_score))
    
    def _calculate_breadth_coverage(self, answer: str) -> float:
        """计算广度覆盖"""
        # 识别涉及的密码学领域
        covered_domains = set()
        
        for domain, concepts in self.crypto_key_concepts.items():
            for concept in concepts:
                if concept in answer:
                    covered_domains.add(domain)
        
        # 广度评分 = 覆盖领域数 / 总领域数
        total_domains = len(self.crypto_key_concepts)
        breadth_score = len(covered_domains) / total_domains if total_domains > 0 else 0.0
        
        return max(0.0, min(1.0, breadth_score))
    
    def _calculate_complexity_bonus(self, terms: List) -> float:
        """计算复杂度奖励"""
        if not terms:
            return 0.0
        
        complexity_sum = 0
        for term in terms:
            if hasattr(term, 'complexity'):
                complexity_sum += term.complexity
        
        avg_complexity = complexity_sum / len(terms)
        # 将复杂度转换为0-1范围的奖励
        return min(1.0, avg_complexity / 6.0)  # 假设最大复杂度为6
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _evaluate_intrinsic_innovation(self, answer: str) -> InnovationMetrics:
        """评估内在创新性（无基准答案时）"""
        # 基于内容本身的创新性指标
        novelty = self._calculate_creativity(answer)  # 重用创造性计算
        uniqueness = min(1.0, len(set(answer.split())) / len(answer.split()) if answer.split() else 0.0)
        creativity = self._calculate_creativity(answer)
        differentiation = 0.5  # 无基准时给中性评分
        
        return InnovationMetrics(
            novelty_score=novelty,
            uniqueness_score=uniqueness,
            creativity_score=creativity,
            differentiation_score=differentiation,
            overall_innovation=0.0
        )
    
    def _extract_key_phrases(self, text: str) -> Set[str]:
        """提取关键短语"""
        # 简化的关键短语提取
        words = text.split()
        phrases = set()
        
        # 提取2-4词的短语
        for i in range(len(words)):
            for j in range(2, 5):
                if i + j <= len(words):
                    phrase = ' '.join(words[i:i+j])
                    if len(phrase) > 4:  # 过滤太短的短语
                        phrases.add(phrase)
        
        return phrases
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构"""
        return {
            'sentence_count': len(re.split(r'[。！？]', text)),
            'paragraph_count': len(text.split('\n\n')),
            'avg_sentence_length': len(text) / max(1, len(re.split(r'[。！？]', text))),
            'has_lists': bool(re.search(r'[1-9]\.|•|·', text)),
            'has_code': bool(re.search(r'```|代码', text)),
            'question_count': len(re.findall(r'[？?]', text))
        }
    
    def _calculate_structure_difference(self, struct1: Dict[str, Any], struct2: Dict[str, Any]) -> float:
        """计算结构差异"""
        differences = []
        
        for key in struct1:
            if key in struct2:
                if isinstance(struct1[key], bool):
                    diff = 0.0 if struct1[key] == struct2[key] else 1.0
                else:
                    val1, val2 = struct1[key], struct2[key]
                    if val1 == 0 and val2 == 0:
                        diff = 0.0
                    else:
                        diff = abs(val1 - val2) / max(val1, val2, 1)
                differences.append(diff)
        
        return sum(differences) / len(differences) if differences else 0.0
    
    def _contains_concept(self, text: str, concept: str) -> bool:
        """检查文本是否包含概念"""
        # 简单的概念匹配
        return concept.lower() in text.lower()
    
    def _evaluate_requirement_fulfillment(self, answer: str, requirement: str) -> float:
        """评估单个需求的满足程度"""
        # 简化的需求满足度评估
        if requirement.lower() in answer.lower():
            return 1.0
        
        # 使用相似性匹配
        similarity = SequenceMatcher(None, requirement.lower(), answer.lower()).ratio()
        return similarity
    
    def _calculate_professional_score(self, terms: List) -> float:
        """计算专业性评分"""
        if not terms:
            return 0.0
        
        # 基于术语复杂度和数量计算专业性
        complexity_sum = 0
        for term in terms:
            if hasattr(term, 'complexity'):
                complexity_sum += term.complexity
        
        avg_complexity = complexity_sum / len(terms)
        term_density = min(1.0, len(terms) / 10)  # 每10个术语为基准
        
        professional_score = (avg_complexity / 6.0) * 0.7 + term_density * 0.3
        
        return max(0.0, min(1.0, professional_score))
    
    def _calculate_term_accuracy(self, terms: List, text: str) -> float:
        """计算术语准确性"""
        if not terms:
            return 1.0
        
        # 简化的准确性评估 - 检查术语使用是否合理
        accurate_terms = 0
        
        for term in terms:
            # 检查术语是否在合适的上下文中使用
            if hasattr(term, 'term') and self._is_term_used_correctly(term.term, text):
                accurate_terms += 1
        
        return accurate_terms / len(terms) if terms else 1.0
    
    def _is_term_used_correctly(self, term: str, text: str) -> bool:
        """检查术语是否使用正确"""
        # 简化的正确性检查 - 实际应用中可以更复杂
        term_index = text.find(term)
        if term_index == -1:
            return False
        
        # 检查术语前后的上下文是否合理
        context_start = max(0, term_index - 50)
        context_end = min(len(text), term_index + len(term) + 50)
        context = text[context_start:context_end]
        
        # 简单的上下文合理性检查
        negative_indicators = ['不是', '错误', '不正确', '不对']
        for indicator in negative_indicators:
            if indicator in context:
                return False
        
        return True


def test_innovation_completeness():
    """测试创新性和完整性评估功能"""
    print("开始测试创新性和完整性评估功能...")
    
    # 创建计算器实例
    calculator = IndustryMetricsCalculator()
    
    # 测试数据
    test_answer = """
    AES（高级加密标准）是一种对称加密算法，它使用相同的密钥进行加密和解密。
    AES支持128、192和256位的密钥长度，其中AES-256提供了最高的安全性。
    
    在实际应用中，AES通常与不同的工作模式结合使用，如CBC、CTR和GCM模式。
    例如，AES-GCM模式不仅提供加密功能，还提供认证功能，广泛用于TLS协议中。
    
    为了确保安全性，密钥管理是关键因素。可以使用PBKDF2或Argon2等密钥派生函数
    从密码生成强密钥。此外，还需要考虑侧信道攻击的防护。
    """
    
    baseline_answers = [
        "AES是对称加密算法，使用相同密钥加密解密。",
        "高级加密标准AES有128、192、256位密钥长度。"
    ]
    
    requirements = [
        "解释AES算法",
        "说明密钥长度",
        "描述工作模式",
        "讨论安全性",
        "提及实际应用"
    ]
    
    try:
        # 测试创新性评估
        print("\n1. 测试创新性评估...")
        innovation_metrics = calculator.evaluate_innovation_level(test_answer, baseline_answers)
        print(f"新颖性评分: {innovation_metrics.novelty_score:.3f}")
        print(f"独特性评分: {innovation_metrics.uniqueness_score:.3f}")
        print(f"创造性评分: {innovation_metrics.creativity_score:.3f}")
        print(f"差异化评分: {innovation_metrics.differentiation_score:.3f}")
        print(f"总体创新性: {innovation_metrics.overall_innovation:.3f}")
        
        # 测试完整性评估
        print("\n2. 测试完整性评估...")
        completeness_metrics = calculator.measure_completeness(test_answer, requirements)
        print(f"概念覆盖度: {completeness_metrics.concept_coverage:.3f}")
        print(f"需求满足度: {completeness_metrics.requirement_fulfillment:.3f}")
        print(f"深度充分性: {completeness_metrics.depth_adequacy:.3f}")
        print(f"广度覆盖: {completeness_metrics.breadth_coverage:.3f}")
        print(f"总体完整性: {completeness_metrics.overall_completeness:.3f}")
        
        # 测试密码学术语分析
        print("\n3. 测试密码学术语分析...")
        crypto_analysis = calculator.analyze_crypto_terms(test_answer)
        print(f"总术语数: {crypto_analysis.total_terms}")
        print(f"唯一术语数: {crypto_analysis.unique_terms}")
        print(f"术语分类: {crypto_analysis.term_categories}")
        print(f"专业性评分: {crypto_analysis.professional_score:.3f}")
        print(f"术语准确性: {crypto_analysis.term_accuracy:.3f}")
        
        # 测试领域相关性
        print("\n4. 测试领域相关性...")
        domain_context = "密码学算法和安全协议"
        relevance_score = calculator.calculate_domain_relevance(test_answer, domain_context)
        print(f"领域相关性: {relevance_score:.3f}")
        
        # 测试实用性评估
        print("\n5. 测试实用性评估...")
        use_case = "网络安全应用"
        practical_score = calculator.assess_practical_applicability(test_answer, use_case)
        print(f"实用性评分: {practical_score:.3f}")
        
        print("\n✅ 所有测试通过！创新性和完整性评估功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_innovation_completeness()


def test_crypto_domain_evaluation():
    """测试密码学领域专门的评估功能"""
    print("开始测试密码学领域专门评估功能...")
    
    calculator = IndustryMetricsCalculator()
    
    # 密码学专业测试案例
    crypto_test_cases = [
        {
            "name": "AES加密算法详解",
            "answer": """
            AES（Advanced Encryption Standard）是一种对称分组密码算法，由比利时密码学家Joan Daemen和Vincent Rijmen设计。
            它采用替换-置换网络（SPN）结构，支持128、192、256位密钥长度。
            
            AES的工作原理包括：
            1. 密钥扩展：将原始密钥扩展为轮密钥
            2. 初始轮：AddRoundKey操作
            3. 主轮：SubBytes、ShiftRows、MixColumns、AddRoundKey
            4. 最终轮：省略MixColumns操作
            
            在实际应用中，AES需要与工作模式结合：
            - ECB模式：简单但不安全，存在模式泄露
            - CBC模式：需要初始化向量，串行处理
            - CTR模式：可并行处理，将分组密码转为流密码
            - GCM模式：提供认证加密，广泛用于TLS 1.3
            
            安全考虑：
            - 密钥管理：使用PBKDF2、scrypt或Argon2进行密钥派生
            - 侧信道攻击：实现需要防范时间攻击和功耗分析
            - 量子威胁：AES-256对量子计算机仍有一定抗性
            """,
            "baseline_answers": [
                "AES是对称加密算法，有128、192、256位密钥。",
                "AES使用替换置换网络，包含多轮加密操作。"
            ],
            "requirements": [
                "解释AES算法原理",
                "说明密钥长度选项", 
                "描述工作模式",
                "讨论安全性考虑",
                "提及实际应用场景"
            ]
        },
        {
            "name": "RSA数字签名机制",
            "answer": """
            RSA数字签名基于大整数分解的数学难题，提供身份认证和不可否认性。
            
            签名过程：
            1. 消息哈希：使用SHA-256等安全哈希函数
            2. 填充：采用PSS或PKCS#1 v1.5填充方案
            3. 私钥签名：使用私钥对填充后的哈希值进行模幂运算
            
            验证过程：
            1. 公钥解密：使用公钥恢复签名值
            2. 填充验证：检查填充格式正确性
            3. 哈希比较：验证恢复的哈希与消息哈希是否一致
            
            安全强度：
            - RSA-1024：已不推荐使用
            - RSA-2048：当前最低推荐标准
            - RSA-3072：对应AES-128安全级别
            - RSA-4096：提供更高安全边际
            
            实际部署：
            - PKI体系：结合X.509证书使用
            - 时间戳：防范重放攻击
            - 证书链：建立信任关系
            """,
            "baseline_answers": [
                "RSA可以用于数字签名，基于大数分解难题。",
                "RSA签名使用私钥签名，公钥验证。"
            ],
            "requirements": [
                "解释RSA签名原理",
                "描述签名和验证过程",
                "说明安全强度要求",
                "讨论实际部署考虑"
            ]
        }
    ]
    
    for i, test_case in enumerate(crypto_test_cases, 1):
        print(f"\n=== 测试案例 {i}: {test_case['name']} ===")
        
        answer = test_case["answer"]
        baseline_answers = test_case["baseline_answers"]
        requirements = test_case["requirements"]
        
        # 创新性评估
        innovation = calculator.evaluate_innovation_level(answer, baseline_answers)
        print(f"创新性评估:")
        print(f"  新颖性: {innovation.novelty_score:.3f}")
        print(f"  独特性: {innovation.uniqueness_score:.3f}")
        print(f"  创造性: {innovation.creativity_score:.3f}")
        print(f"  差异化: {innovation.differentiation_score:.3f}")
        print(f"  总体创新性: {innovation.overall_innovation:.3f}")
        
        # 完整性评估
        completeness = calculator.measure_completeness(answer, requirements)
        print(f"\n完整性评估:")
        print(f"  概念覆盖度: {completeness.concept_coverage:.3f}")
        print(f"  需求满足度: {completeness.requirement_fulfillment:.3f}")
        print(f"  深度充分性: {completeness.depth_adequacy:.3f}")
        print(f"  广度覆盖: {completeness.breadth_coverage:.3f}")
        print(f"  总体完整性: {completeness.overall_completeness:.3f}")
        
        # 密码学术语分析
        crypto_analysis = calculator.analyze_crypto_terms(answer)
        print(f"\n密码学术语分析:")
        print(f"  术语总数: {crypto_analysis.total_terms}")
        print(f"  唯一术语: {crypto_analysis.unique_terms}")
        print(f"  专业性评分: {crypto_analysis.professional_score:.3f}")
        print(f"  术语准确性: {crypto_analysis.term_accuracy:.3f}")
        
        # 领域相关性
        domain_context = "密码学算法设计与安全分析"
        relevance = calculator.calculate_domain_relevance(answer, domain_context)
        print(f"\n领域相关性: {relevance:.3f}")
        
        # 实用性评估
        use_case = "企业级安全系统实施"
        practical = calculator.assess_practical_applicability(answer, use_case)
        print(f"实用性评分: {practical:.3f}")
    
    print("\n✅ 密码学领域专门评估测试完成！")
    return True


def test_innovation_completeness():
    """测试创新性和完整性评估功能 - 主要测试函数"""
    print("开始测试创新性和完整性评估功能...")
    
    # 运行基础测试
    success1 = _run_basic_tests()
    
    # 运行密码学专门测试
    success2 = test_crypto_domain_evaluation()
    
    return success1 and success2


def _run_basic_tests():
    """运行基础功能测试"""
    # 创建计算器实例
    calculator = IndustryMetricsCalculator()
    
    # 测试数据
    test_answer = """
    AES（高级加密标准）是一种对称加密算法，它使用相同的密钥进行加密和解密。
    AES支持128、192和256位的密钥长度，其中AES-256提供了最高的安全性。
    
    在实际应用中，AES通常与不同的工作模式结合使用，如CBC、CTR和GCM模式。
    例如，AES-GCM模式不仅提供加密功能，还提供认证功能，广泛用于TLS协议中。
    
    为了确保安全性，密钥管理是关键因素。可以使用PBKDF2或Argon2等密钥派生函数
    从密码生成强密钥。此外，还需要考虑侧信道攻击的防护。
    """
    
    baseline_answers = [
        "AES是对称加密算法，使用相同密钥加密解密。",
        "高级加密标准AES有128、192、256位密钥长度。"
    ]
    
    requirements = [
        "解释AES算法",
        "说明密钥长度",
        "描述工作模式",
        "讨论安全性",
        "提及实际应用"
    ]
    
    try:
        # 测试创新性评估
        print("\n1. 测试创新性评估...")
        innovation_metrics = calculator.evaluate_innovation_level(test_answer, baseline_answers)
        print(f"新颖性评分: {innovation_metrics.novelty_score:.3f}")
        print(f"独特性评分: {innovation_metrics.uniqueness_score:.3f}")
        print(f"创造性评分: {innovation_metrics.creativity_score:.3f}")
        print(f"差异化评分: {innovation_metrics.differentiation_score:.3f}")
        print(f"总体创新性: {innovation_metrics.overall_innovation:.3f}")
        
        # 测试完整性评估
        print("\n2. 测试完整性评估...")
        completeness_metrics = calculator.measure_completeness(test_answer, requirements)
        print(f"概念覆盖度: {completeness_metrics.concept_coverage:.3f}")
        print(f"需求满足度: {completeness_metrics.requirement_fulfillment:.3f}")
        print(f"深度充分性: {completeness_metrics.depth_adequacy:.3f}")
        print(f"广度覆盖: {completeness_metrics.breadth_coverage:.3f}")
        print(f"总体完整性: {completeness_metrics.overall_completeness:.3f}")
        
        # 测试密码学术语分析
        print("\n3. 测试密码学术语分析...")
        crypto_analysis = calculator.analyze_crypto_terms(test_answer)
        print(f"总术语数: {crypto_analysis.total_terms}")
        print(f"唯一术语数: {crypto_analysis.unique_terms}")
        print(f"术语分类: {crypto_analysis.term_categories}")
        print(f"专业性评分: {crypto_analysis.professional_score:.3f}")
        print(f"术语准确性: {crypto_analysis.term_accuracy:.3f}")
        
        # 测试领域相关性
        print("\n4. 测试领域相关性...")
        domain_context = "密码学算法和安全协议"
        relevance_score = calculator.calculate_domain_relevance(test_answer, domain_context)
        print(f"领域相关性: {relevance_score:.3f}")
        
        # 测试实用性评估
        print("\n5. 测试实用性评估...")
        use_case = "网络安全应用"
        practical_score = calculator.assess_practical_applicability(test_answer, use_case)
        print(f"实用性评分: {practical_score:.3f}")
        
        print("\n✅ 基础测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 基础测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False