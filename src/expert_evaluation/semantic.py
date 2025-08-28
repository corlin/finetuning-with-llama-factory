"""
高级语义评估模块

该模块实现了超越传统文本相似性的高级语义评估功能，包括：
- 语义深度计算
- 逻辑一致性评估
- 上下文理解能力评估
- 概念覆盖度评估
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
import math

# 导入现有模块
from src.evaluation_framework import ChineseSemanticEvaluator
from src.data_models import CryptoTerm
from src.crypto_term_processor import CryptoTermProcessor


@dataclass
class ConceptCoverageResult:
    """概念覆盖度评估结果"""
    total_concepts: int
    covered_concepts: int
    coverage_ratio: float
    missing_concepts: List[str]
    concept_weights: Dict[str, float]
    weighted_coverage: float
    quality_adjusted_coverage: float = 0.0
    coverage_details: Dict = None
    
    def __post_init__(self):
        if self.coverage_details is None:
            self.coverage_details = {}


@dataclass
class SemanticDepthResult:
    """语义深度评估结果"""
    surface_similarity: float
    semantic_depth: float
    conceptual_alignment: float
    logical_coherence: float
    overall_depth: float


class ConceptWeightConfig:
    """概念权重配置系统"""
    
    def __init__(self):
        # 基础权重配置
        self.base_weights = {
            "核心概念": 3.0,      # 加密、解密、密钥等核心概念
            "算法名称": 2.5,      # AES、RSA、SHA等具体算法
            "技术特性": 2.0,      # 对称、非对称、哈希等技术特性
            "应用场景": 1.5,      # 网络安全、数据保护等应用
            "一般术语": 1.0       # 其他相关术语
        }
        
        # 详细的概念分类和权重
        self.concept_categories = {
            # 核心密码学概念
            "核心概念": {
                "keywords": ["加密", "解密", "密钥", "算法", "哈希", "签名", "认证", "完整性"],
                "weight": 3.0,
                "boost_factor": 1.2  # 在特定上下文中的权重提升
            },
            
            # 具体算法名称
            "算法名称": {
                "keywords": ["AES", "RSA", "SHA", "DES", "3DES", "MD5", "ECDSA", "ECDH"],
                "patterns": [r"[A-Z]{2,5}-?\d*", r"SHA-?\d+", r"AES-?\d+", r"RSA-?\d+"],
                "weight": 2.5,
                "boost_factor": 1.3
            },
            
            # 技术特性和属性
            "技术特性": {
                "keywords": ["对称", "非对称", "公钥", "私钥", "分组", "流密码", "椭圆曲线"],
                "weight": 2.0,
                "boost_factor": 1.1
            },
            
            # 应用场景和用途
            "应用场景": {
                "keywords": ["网络安全", "数据保护", "身份认证", "数字证书", "SSL", "TLS"],
                "weight": 1.5,
                "boost_factor": 1.0
            },
            
            # 攻击和威胁
            "安全威胁": {
                "keywords": ["攻击", "破解", "碰撞", "暴力破解", "中间人攻击", "重放攻击"],
                "weight": 1.8,
                "boost_factor": 1.1
            },
            
            # 标准和协议
            "标准协议": {
                "keywords": ["标准", "协议", "RFC", "NIST", "FIPS", "ISO"],
                "weight": 1.7,
                "boost_factor": 1.0
            }
        }
    
    def get_concept_weight(self, concept: str, context: str = "") -> float:
        """获取概念权重"""
        base_weight = 1.0
        boost_factor = 1.0
        
        for category, config in self.concept_categories.items():
            # 关键词匹配
            if any(keyword in concept for keyword in config["keywords"]):
                base_weight = max(base_weight, config["weight"])
                boost_factor = config["boost_factor"]
                break
            
            # 模式匹配
            if "patterns" in config:
                for pattern in config["patterns"]:
                    if re.search(pattern, concept):
                        base_weight = max(base_weight, config["weight"])
                        boost_factor = config["boost_factor"]
                        break
        
        # 上下文相关的权重调整
        if context:
            context_boost = self._calculate_context_boost(concept, context)
            boost_factor *= context_boost
        
        return base_weight * boost_factor
    
    def _calculate_context_boost(self, concept: str, context: str) -> float:
        """计算上下文相关的权重提升"""
        boost = 1.0
        
        # 如果概念在上下文中被强调（如定义、解释）
        emphasis_patterns = [
            f"{concept}是", f"{concept}指", f"{concept}定义", 
            f"所谓{concept}", f"{concept}的作用", f"{concept}的特点"
        ]
        
        for pattern in emphasis_patterns:
            if pattern in context:
                boost += 0.2
                break
        
        # 如果概念与其他重要概念共现
        important_cooccurrence = ["安全", "重要", "关键", "核心", "主要"]
        for term in important_cooccurrence:
            if term in context and concept in context:
                boost += 0.1
        
        return min(boost, 1.5)  # 最大提升50%


class AdvancedSemanticEvaluator(ChineseSemanticEvaluator):
    """
    高级语义评估器
    
    继承ChineseSemanticEvaluator，提供更深层次的语义分析能力：
    - 语义深度计算，超越表面文本相似性
    - 逻辑一致性评估算法
    - 上下文理解能力评估
    - 概念覆盖度评估
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.crypto_processor = CryptoTermProcessor()
        self.weight_config = ConceptWeightConfig()
        
        # 逻辑连接词和结构指示词
        self.logical_connectors = {
            "因果关系": ["因为", "所以", "因此", "由于", "导致", "造成", "引起"],
            "转折关系": ["但是", "然而", "不过", "虽然", "尽管", "相反"],
            "递进关系": ["而且", "并且", "此外", "另外", "更重要的是", "进一步"],
            "条件关系": ["如果", "假如", "只要", "除非", "当", "一旦"],
            "对比关系": ["相比", "对比", "与此相反", "另一方面", "相对于"]
        }
        
        # 概念匹配算法配置
        self.matching_config = {
            "exact_match_weight": 1.0,      # 精确匹配权重
            "partial_match_weight": 0.7,    # 部分匹配权重
            "synonym_match_weight": 0.8,    # 同义词匹配权重
            "fuzzy_match_threshold": 0.6    # 模糊匹配阈值
        }
    
    def calculate_semantic_depth(self, answer: str, reference: str) -> SemanticDepthResult:
        """
        计算语义深度，超越表面文本相似性
        
        Args:
            answer: 模型回答
            reference: 参考答案
            
        Returns:
            SemanticDepthResult: 语义深度评估结果
        """
        try:
            # 1. 表面相似性（继承自父类）
            surface_similarity = self.evaluate_semantic_similarity(answer, reference)
            
            # 2. 概念层面的对齐度
            conceptual_alignment = self._evaluate_conceptual_alignment(answer, reference)
            
            # 3. 逻辑连贯性
            logical_coherence = self._evaluate_logical_coherence(answer)
            
            # 4. 语义深度综合计算
            semantic_depth = self._calculate_depth_score(
                surface_similarity, conceptual_alignment, logical_coherence
            )
            
            # 5. 整体语义深度
            overall_depth = (
                surface_similarity * 0.3 +
                conceptual_alignment * 0.4 +
                logical_coherence * 0.3
            )
            
            return SemanticDepthResult(
                surface_similarity=surface_similarity,
                semantic_depth=semantic_depth,
                conceptual_alignment=conceptual_alignment,
                logical_coherence=logical_coherence,
                overall_depth=overall_depth
            )
            
        except Exception as e:
            self.logger.error(f"语义深度计算失败: {e}")
            return SemanticDepthResult(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def assess_logical_consistency(self, answer: str) -> float:
        """
        评估逻辑一致性
        
        Args:
            answer: 待评估的回答
            
        Returns:
            float: 逻辑一致性分数 (0-1)
        """
        try:
            # 1. 逻辑结构分析
            structure_score = self._analyze_logical_structure(answer)
            
            # 2. 论证连贯性
            coherence_score = self._evaluate_argument_coherence(answer)
            
            # 3. 矛盾检测
            contradiction_penalty = self._detect_contradictions(answer)
            
            # 4. 综合逻辑一致性分数
            consistency_score = (
                structure_score * 0.4 +
                coherence_score * 0.4 -
                contradiction_penalty * 0.2
            )
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            self.logger.error(f"逻辑一致性评估失败: {e}")
            return 0.0
    
    def evaluate_contextual_understanding(self, answer: str, context: str) -> float:
        """
        评估上下文理解能力
        
        Args:
            answer: 模型回答
            context: 上下文信息
            
        Returns:
            float: 上下文理解分数 (0-1)
        """
        try:
            if not context:
                return 0.5  # 无上下文时给予中等分数
            
            # 1. 上下文关键信息提取
            context_keywords = self._extract_context_keywords(context)
            answer_keywords = self._extract_context_keywords(answer)
            
            # 2. 关键信息覆盖度
            coverage_score = self._calculate_keyword_coverage(context_keywords, answer_keywords)
            
            # 3. 上下文相关性
            relevance_score = self._evaluate_context_relevance(answer, context)
            
            # 4. 上下文适应性
            adaptation_score = self._evaluate_context_adaptation(answer, context)
            
            # 5. 综合上下文理解分数
            understanding_score = (
                coverage_score * 0.4 +
                relevance_score * 0.3 +
                adaptation_score * 0.3
            )
            
            return max(0.0, min(1.0, understanding_score))
            
        except Exception as e:
            self.logger.error(f"上下文理解评估失败: {e}")
            return 0.0
    
    def measure_concept_coverage(self, answer: str, key_concepts: List[str], context: str = "") -> ConceptCoverageResult:
        """
        测量关键概念覆盖度
        
        Args:
            answer: 模型回答
            key_concepts: 关键概念列表
            context: 上下文信息（可选）
            
        Returns:
            ConceptCoverageResult: 概念覆盖度评估结果
        """
        try:
            # 1. 提取回答中的概念
            answer_concepts = self._extract_concepts_from_text(answer)
            
            # 2. 计算概念权重（考虑上下文）
            full_context = f"{context} {answer}" if context else answer
            concept_weights = self._calculate_concept_weights(key_concepts, full_context)
            
            # 3. 详细的概念匹配分析
            covered_concepts = []
            missing_concepts = []
            coverage_details = {}
            
            for concept in key_concepts:
                match_result = self._analyze_concept_match(concept, answer_concepts, answer)
                if match_result["is_covered"]:
                    covered_concepts.append(concept)
                    coverage_details[concept] = match_result
                else:
                    missing_concepts.append(concept)
            
            # 4. 基础覆盖率
            coverage_ratio = len(covered_concepts) / len(key_concepts) if key_concepts else 0.0
            
            # 5. 加权覆盖率
            total_weight = sum(concept_weights.values())
            covered_weight = sum(concept_weights.get(concept, 1.0) for concept in covered_concepts)
            weighted_coverage = covered_weight / total_weight if total_weight > 0 else 0.0
            
            # 6. 质量调整的覆盖率（考虑匹配质量）
            quality_adjusted_coverage = self._calculate_quality_adjusted_coverage(
                covered_concepts, coverage_details, concept_weights
            )
            
            result = ConceptCoverageResult(
                total_concepts=len(key_concepts),
                covered_concepts=len(covered_concepts),
                coverage_ratio=coverage_ratio,
                missing_concepts=missing_concepts,
                concept_weights=concept_weights,
                weighted_coverage=weighted_coverage
            )
            
            # 添加质量调整覆盖率到结果中
            result.quality_adjusted_coverage = quality_adjusted_coverage
            result.coverage_details = coverage_details
            
            return result
            
        except Exception as e:
            self.logger.error(f"概念覆盖度评估失败: {e}")
            return ConceptCoverageResult(0, 0, 0.0, [], {}, 0.0)
    
    def _analyze_concept_match(self, concept: str, answer_concepts: List[str], full_answer: str) -> Dict:
        """分析概念匹配的详细情况"""
        match_result = {
            "is_covered": False,
            "match_type": None,
            "match_confidence": 0.0,
            "matched_concept": None,
            "context_relevance": 0.0
        }
        
        # 1. 精确匹配
        if concept in answer_concepts:
            match_result.update({
                "is_covered": True,
                "match_type": "exact",
                "match_confidence": 1.0,
                "matched_concept": concept
            })
        
        # 2. 部分匹配
        elif any(concept in ac or ac in concept for ac in answer_concepts):
            for ac in answer_concepts:
                if concept in ac or ac in concept:
                    match_result.update({
                        "is_covered": True,
                        "match_type": "partial",
                        "match_confidence": 0.7,
                        "matched_concept": ac
                    })
                    break
        
        # 3. 同义词匹配
        elif self._check_synonym_match(concept, answer_concepts):
            match_result.update({
                "is_covered": True,
                "match_type": "synonym",
                "match_confidence": 0.8,
                "matched_concept": "synonym_match"
            })
        
        # 4. 模糊匹配
        elif self._check_fuzzy_match(concept, answer_concepts):
            best_match = None
            best_similarity = 0.0
            for ac in answer_concepts:
                similarity = self._calculate_string_similarity(concept, ac)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = ac
            
            if best_similarity >= self.matching_config["fuzzy_match_threshold"]:
                match_result.update({
                    "is_covered": True,
                    "match_type": "fuzzy",
                    "match_confidence": best_similarity,
                    "matched_concept": best_match
                })
        
        # 5. 计算上下文相关性
        if match_result["is_covered"]:
            match_result["context_relevance"] = self._calculate_concept_context_relevance(
                concept, full_answer
            )
        
        return match_result
    
    def _calculate_quality_adjusted_coverage(self, covered_concepts: List[str], 
                                           coverage_details: Dict, 
                                           concept_weights: Dict[str, float]) -> float:
        """计算质量调整的覆盖率"""
        if not covered_concepts:
            return 0.0
        
        total_quality_weight = 0.0
        total_weight = sum(concept_weights.values())
        
        for concept in covered_concepts:
            base_weight = concept_weights.get(concept, 1.0)
            match_quality = coverage_details.get(concept, {}).get("match_confidence", 1.0)
            context_relevance = coverage_details.get(concept, {}).get("context_relevance", 1.0)
            
            # 综合质量分数
            quality_score = (match_quality * 0.7 + context_relevance * 0.3)
            quality_weight = base_weight * quality_score
            total_quality_weight += quality_weight
        
        return total_quality_weight / total_weight if total_weight > 0 else 0.0
    
    def _calculate_concept_context_relevance(self, concept: str, text: str) -> float:
        """计算概念在文本中的上下文相关性"""
        # 检查概念是否在有意义的上下文中出现
        relevance_indicators = [
            f"{concept}是", f"{concept}指", f"{concept}用于", 
            f"使用{concept}", f"采用{concept}", f"基于{concept}",
            f"{concept}的特点", f"{concept}的优势", f"{concept}的应用"
        ]
        
        relevance_score = 0.5  # 基础分数
        
        for indicator in relevance_indicators:
            if indicator in text:
                relevance_score += 0.1
        
        # 检查概念是否与其他重要概念共现
        important_cooccurrence = ["安全", "加密", "算法", "密钥", "保护"]
        for term in important_cooccurrence:
            if term in text and concept in text:
                relevance_score += 0.05
        
        return min(1.0, relevance_score)
    
    def _evaluate_conceptual_alignment(self, answer: str, reference: str) -> float:
        """评估概念层面的对齐度"""
        # 提取两个文本的核心概念
        answer_concepts = self._extract_concepts_from_text(answer)
        reference_concepts = self._extract_concepts_from_text(reference)
        
        if not reference_concepts:
            return 0.5
        
        # 计算概念交集
        common_concepts = set(answer_concepts) & set(reference_concepts)
        
        # 基于概念重要性的加权计算
        alignment_score = 0.0
        total_weight = 0.0
        
        for concept in reference_concepts:
            weight = self._get_concept_weight(concept)
            total_weight += weight
            
            if concept in common_concepts:
                alignment_score += weight
        
        return alignment_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_logical_coherence(self, text: str) -> float:
        """评估逻辑连贯性"""
        # 检查逻辑连接词的使用
        connector_score = self._evaluate_logical_connectors(text)
        
        # 检查句子间的逻辑关系
        sentence_flow_score = self._evaluate_sentence_flow(text)
        
        # 检查论证结构
        argument_structure_score = self._evaluate_argument_structure(text)
        
        return (connector_score + sentence_flow_score + argument_structure_score) / 3.0
    
    def _calculate_depth_score(self, surface: float, conceptual: float, logical: float) -> float:
        """计算语义深度综合分数"""
        # 深度分数应该反映超越表面相似性的理解程度
        depth_bonus = 0.0
        
        # 如果概念对齐度高于表面相似性，给予深度奖励
        if conceptual > surface:
            depth_bonus += (conceptual - surface) * 0.5
        
        # 如果逻辑连贯性高，给予额外奖励
        if logical > 0.7:
            depth_bonus += (logical - 0.7) * 0.3
        
        base_depth = (conceptual + logical) / 2.0
        return min(1.0, base_depth + depth_bonus)
    
    def _analyze_logical_structure(self, text: str) -> float:
        """分析逻辑结构"""
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        structure_indicators = 0
        total_sentences = len(sentences)
        
        for sentence in sentences:
            # 检查是否包含逻辑指示词
            for category, connectors in self.logical_connectors.items():
                if any(connector in sentence for connector in connectors):
                    structure_indicators += 1
                    break
        
        return min(1.0, structure_indicators / total_sentences + 0.3)
    
    def _evaluate_argument_coherence(self, text: str) -> float:
        """评估论证连贯性"""
        # 简化实现：检查关键词的一致性和递进关系
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if len(sentences) < 2:
            return 0.6
        
        # 检查主题一致性
        topic_consistency = self._check_topic_consistency(sentences)
        
        # 检查论证递进
        argument_progression = self._check_argument_progression(sentences)
        
        return (topic_consistency + argument_progression) / 2.0
    
    def _detect_contradictions(self, text: str) -> float:
        """检测矛盾"""
        # 简化实现：检查明显的矛盾表述
        contradiction_patterns = [
            (r'不是.*是', 0.3),
            (r'没有.*有', 0.3),
            (r'不能.*能', 0.3),
            (r'不会.*会', 0.3)
        ]
        
        contradiction_score = 0.0
        for pattern, penalty in contradiction_patterns:
            if re.search(pattern, text):
                contradiction_score += penalty
        
        return min(1.0, contradiction_score)
    
    def _extract_context_keywords(self, text: str) -> List[str]:
        """提取上下文关键词"""
        # 使用现有的分词方法
        tokens = self._chinese_tokenize(text)
        
        # 过滤重要关键词
        keywords = []
        for token in tokens:
            if len(token) > 1 and self._is_important_keyword(token):
                keywords.append(token)
        
        return keywords
    
    def _calculate_keyword_coverage(self, context_keywords: List[str], answer_keywords: List[str]) -> float:
        """计算关键词覆盖度"""
        if not context_keywords:
            return 1.0
        
        covered = set(context_keywords) & set(answer_keywords)
        return len(covered) / len(context_keywords)
    
    def _evaluate_context_relevance(self, answer: str, context: str) -> float:
        """评估上下文相关性"""
        # 基于共同概念和主题的相关性评估
        answer_concepts = self._extract_concepts_from_text(answer)
        context_concepts = self._extract_concepts_from_text(context)
        
        if not context_concepts:
            return 0.5
        
        common_concepts = set(answer_concepts) & set(context_concepts)
        return len(common_concepts) / len(context_concepts)
    
    def _evaluate_context_adaptation(self, answer: str, context: str) -> float:
        """评估上下文适应性"""
        # 检查回答是否适应了上下文的语境和要求
        # 简化实现
        return 0.8  # 假设大部分情况下适应性良好
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """从文本中提取概念"""
        concepts = []
        
        # 使用密码学术语处理器
        try:
            crypto_annotations = self.crypto_processor.identify_crypto_terms(text)
            concepts.extend([annotation.term for annotation in crypto_annotations])
        except Exception as e:
            self.logger.warning(f"密码学术语提取失败: {e}")
        
        # 添加通用概念提取
        concept_patterns = [
            r'[A-Z]{2,}',  # 大写缩写
            r'[\u4e00-\u9fff]{2,4}算法',  # 中文算法名
            r'[\u4e00-\u9fff]{2,4}加密',  # 中文加密相关
            r'[\u4e00-\u9fff]{2,4}安全',  # 中文安全相关
            r'[\u4e00-\u9fff]{2,4}函数',  # 中文函数相关
            r'[\u4e00-\u9fff]{2,4}协议',  # 中文协议相关
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        return list(set(concepts))  # 去重
    
    def _calculate_concept_weights(self, concepts: List[str], context: str = "") -> Dict[str, float]:
        """计算概念权重"""
        weights = {}
        
        for concept in concepts:
            weight = self.weight_config.get_concept_weight(concept, context)
            weights[concept] = weight
        
        return weights
    
    def _get_concept_weight(self, concept: str) -> float:
        """获取单个概念的权重（保持向后兼容）"""
        return self.weight_config.get_concept_weight(concept)
    
    def _is_concept_covered(self, concept: str, answer_concepts: List[str]) -> bool:
        """
        检查概念是否被覆盖
        使用多种匹配策略：精确匹配、部分匹配、同义词匹配、模糊匹配
        """
        # 1. 精确匹配
        if concept in answer_concepts:
            return True
        
        # 2. 部分匹配（包含关系）
        for answer_concept in answer_concepts:
            if concept in answer_concept or answer_concept in concept:
                return True
        
        # 3. 同义词匹配
        if self._check_synonym_match(concept, answer_concepts):
            return True
        
        # 4. 模糊匹配（基于编辑距离）
        if self._check_fuzzy_match(concept, answer_concepts):
            return True
        
        return False
    
    def _check_synonym_match(self, concept: str, answer_concepts: List[str]) -> bool:
        """检查同义词匹配"""
        # 定义同义词映射
        synonym_map = {
            "加密": ["密码", "编码", "加密算法"],
            "解密": ["解码", "解密算法", "破译"],
            "密钥": ["秘钥", "钥匙", "key"],
            "哈希": ["散列", "摘要", "hash"],
            "签名": ["数字签名", "电子签名"],
            "对称加密": ["对称密码", "私钥加密"],
            "非对称加密": ["公钥加密", "公开密钥加密"],
            "AES": ["高级加密标准", "Rijndael"],
            "RSA": ["RSA算法"],
            "SHA": ["安全哈希算法"],
            "DES": ["数据加密标准"],
            "MD5": ["消息摘要5"]
        }
        
        # 检查概念的同义词是否在答案中
        synonyms = synonym_map.get(concept, [])
        for synonym in synonyms:
            if synonym in answer_concepts:
                return True
        
        # 反向检查：答案概念是否是目标概念的同义词
        for answer_concept in answer_concepts:
            if answer_concept in synonym_map and concept in synonym_map[answer_concept]:
                return True
        
        return False
    
    def _check_fuzzy_match(self, concept: str, answer_concepts: List[str]) -> bool:
        """检查模糊匹配"""
        threshold = self.matching_config["fuzzy_match_threshold"]
        
        for answer_concept in answer_concepts:
            similarity = self._calculate_string_similarity(concept, answer_concept)
            if similarity >= threshold:
                return True
        
        return False
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度（基于编辑距离）"""
        if not str1 or not str2:
            return 0.0
        
        # 简化的编辑距离计算
        len1, len2 = len(str1), len(str2)
        if len1 == 0:
            return 0.0
        if len2 == 0:
            return 0.0
        
        # 动态规划计算编辑距离
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_distance = dp[len1][len2]
        max_len = max(len1, len2)
        similarity = 1.0 - (edit_distance / max_len)
        
        return max(0.0, similarity)
    
    def _is_important_keyword(self, token: str) -> bool:
        """判断是否为重要关键词"""
        # 过滤停用词和无意义词汇
        stop_words = ["的", "是", "在", "有", "和", "与", "或", "但", "这", "那"]
        return token not in stop_words and len(token) > 1
    
    def _evaluate_logical_connectors(self, text: str) -> float:
        """评估逻辑连接词的使用"""
        connector_count = 0
        total_connectors = 0
        
        for category, connectors in self.logical_connectors.items():
            for connector in connectors:
                total_connectors += 1
                if connector in text:
                    connector_count += 1
        
        return min(1.0, connector_count / 5.0 + 0.3)  # 基础分0.3，最多5个连接词
    
    def _evaluate_sentence_flow(self, text: str) -> float:
        """评估句子流畅性"""
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if len(sentences) < 2:
            return 0.7
        
        # 检查句子长度的合理性
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        length_score = 1.0 if 10 <= avg_length <= 50 else max(0.5, min(avg_length/50, 50/avg_length))
        
        return length_score
    
    def _evaluate_argument_structure(self, text: str) -> float:
        """评估论证结构"""
        # 检查是否有明确的论证结构
        structure_indicators = ["首先", "其次", "最后", "总之", "综上", "因此"]
        indicator_count = sum(1 for indicator in structure_indicators if indicator in text)
        
        return min(1.0, indicator_count * 0.2 + 0.4)
    
    def _check_topic_consistency(self, sentences: List[str]) -> float:
        """检查主题一致性"""
        # 简化实现：检查关键词在句子间的重复出现
        all_keywords = []
        for sentence in sentences:
            keywords = self._extract_context_keywords(sentence)
            all_keywords.extend(keywords)
        
        if not all_keywords:
            return 0.5
        
        keyword_counts = Counter(all_keywords)
        repeated_keywords = sum(1 for count in keyword_counts.values() if count > 1)
        
        return min(1.0, repeated_keywords / len(sentences) + 0.3)
    
    def _check_argument_progression(self, sentences: List[str]) -> float:
        """检查论证递进"""
        # 检查是否有递进关系的表述
        progression_indicators = ["进一步", "更重要", "另外", "此外", "而且"]
        
        progression_count = 0
        for sentence in sentences:
            if any(indicator in sentence for indicator in progression_indicators):
                progression_count += 1
        
        return min(1.0, progression_count / len(sentences) + 0.4)


def test_concept_coverage():
    """测试概念覆盖算法"""
    try:
        evaluator = AdvancedSemanticEvaluator()
        
        # 测试数据
        test_answer = "AES是一种对称加密算法，使用相同的密钥进行加密和解密。它支持128、192和256位密钥长度。RSA是非对称加密的代表。"
        test_concepts = ["AES", "对称加密", "密钥", "加密", "解密", "位长度", "RSA", "非对称加密"]
        test_context = "在现代密码学中，我们需要了解各种加密算法的特点和应用。"
        
        # 执行测试
        result = evaluator.measure_concept_coverage(test_answer, test_concepts, test_context)
        
        print("=== 概念覆盖度测试结果 ===")
        print(f"总概念数: {result.total_concepts}")
        print(f"覆盖概念数: {result.covered_concepts}")
        print(f"基础覆盖率: {result.coverage_ratio:.2f}")
        print(f"加权覆盖率: {result.weighted_coverage:.2f}")
        print(f"质量调整覆盖率: {result.quality_adjusted_coverage:.2f}")
        print(f"缺失概念: {result.missing_concepts}")
        
        # 显示概念权重
        print("\n=== 概念权重分析 ===")
        for concept, weight in result.concept_weights.items():
            print(f"{concept}: {weight:.2f}")
        
        # 显示匹配详情
        print("\n=== 概念匹配详情 ===")
        for concept, details in result.coverage_details.items():
            print(f"{concept}:")
            print(f"  匹配类型: {details['match_type']}")
            print(f"  匹配置信度: {details['match_confidence']:.2f}")
            print(f"  上下文相关性: {details['context_relevance']:.2f}")
            if details['matched_concept'] != concept:
                print(f"  匹配到的概念: {details['matched_concept']}")
        
        # 测试语义深度
        reference_answer = "AES（高级加密标准）是目前最广泛使用的对称加密算法，采用相同密钥进行数据加密和解密操作。"
        depth_result = evaluator.calculate_semantic_depth(test_answer, reference_answer)
        
        print("\n=== 语义深度测试结果 ===")
        print(f"表面相似性: {depth_result.surface_similarity:.2f}")
        print(f"概念对齐度: {depth_result.conceptual_alignment:.2f}")
        print(f"逻辑连贯性: {depth_result.logical_coherence:.2f}")
        print(f"语义深度: {depth_result.semantic_depth:.2f}")
        print(f"整体深度: {depth_result.overall_depth:.2f}")
        
        # 测试逻辑一致性
        logic_test_text = "AES是对称加密算法，因此它使用相同的密钥进行加密和解密。然而，RSA是非对称加密，所以它使用不同的密钥对。"
        logic_score = evaluator.assess_logical_consistency(logic_test_text)
        print(f"\n=== 逻辑一致性测试 ===")
        print(f"逻辑一致性分数: {logic_score:.2f}")
        
        # 测试上下文理解
        context_test = "在网络安全应用中"
        context_answer = "SSL/TLS协议使用RSA进行密钥交换，然后使用AES进行数据加密，这样结合了两种算法的优势。"
        context_score = evaluator.evaluate_contextual_understanding(context_answer, context_test)
        print(f"\n=== 上下文理解测试 ===")
        print(f"上下文理解分数: {context_score:.2f}")
        
        print("\n✅ 概念覆盖算法测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 概念覆盖算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_concept_coverage()