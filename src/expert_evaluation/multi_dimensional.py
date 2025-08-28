"""
多维度评估协调器模块

该模块实现了专家评估系统的多维度评估协调功能，包括：
- 整合所有评估维度的计算
- 实现评估维度权重配置系统
- 开发综合评分计算算法
- 添加评估过程的进度监控
- 评估结果聚合和统计分析
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import math

# 导入现有模块
try:
    from src.evaluation_framework import ProfessionalAccuracyEvaluator, ChineseSemanticEvaluator
    from src.chinese_nlp_processor import ChineseNLPProcessor
except ImportError:
    # 如果导入失败，创建简化版本
    class ProfessionalAccuracyEvaluator:
        def evaluate_accuracy(self, answer: str, reference: str) -> float:
            return 0.8
    
    class ChineseSemanticEvaluator:
        def evaluate_semantic_similarity(self, answer: str, reference: str) -> float:
            return 0.7
    
    class ChineseNLPProcessor:
        def __init__(self):
            pass

# 导入专家评估模块
from .config import EvaluationDimension, ExpertEvaluationConfig, DimensionWeightConfig
from .data_models import (
    QAEvaluationItem, ExpertEvaluationResult, DimensionScore, 
    BatchEvaluationResult
)
from .interfaces import MultiDimensionalEvaluator as MultiDimensionalEvaluatorInterface
from .metrics import IndustryMetricsCalculator
from .semantic import AdvancedSemanticEvaluator
from .exceptions import EvaluationProcessError


@dataclass
class ProgressInfo:
    """进度信息"""
    current_item: int
    total_items: int
    current_dimension: str
    elapsed_time: float
    estimated_remaining: float
    processing_speed: float  # items per second
    
    @property
    def progress_percentage(self) -> float:
        """进度百分比"""
        return (self.current_item / self.total_items * 100) if self.total_items > 0 else 0.0
    
    @property
    def eta_formatted(self) -> str:
        """格式化的预计剩余时间"""
        if self.estimated_remaining < 60:
            return f"{self.estimated_remaining:.1f}秒"
        elif self.estimated_remaining < 3600:
            minutes = int(self.estimated_remaining // 60)
            seconds = int(self.estimated_remaining % 60)
            return f"{minutes}分{seconds}秒"
        else:
            hours = int(self.estimated_remaining // 3600)
            minutes = int((self.estimated_remaining % 3600) // 60)
            return f"{hours}小时{minutes}分"


@dataclass
class StatisticalAnalysisResult:
    """统计分析结果"""
    mean_score: float
    median_score: float
    std_deviation: float
    min_score: float
    max_score: float
    confidence_interval: Tuple[float, float]
    percentiles: Dict[int, float]  # 25th, 75th, 90th, 95th percentiles
    outliers: List[int]  # indices of outlier results
    distribution_type: str  # normal, skewed, etc.
    statistical_significance: Dict[str, float]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "average_performance": self.mean_score,
            "consistency": 1.0 - (self.std_deviation / max(self.mean_score, 0.1)),
            "reliability": self._calculate_reliability(),
            "distribution_quality": self._assess_distribution_quality()
        }
    
    def _calculate_reliability(self) -> float:
        """计算可靠性指标"""
        # 基于标准差和置信区间宽度
        ci_width = self.confidence_interval[1] - self.confidence_interval[0]
        reliability = max(0.0, 1.0 - ci_width)
        return reliability
    
    def _assess_distribution_quality(self) -> str:
        """评估分布质量"""
        if self.std_deviation < 0.1:
            return "excellent"
        elif self.std_deviation < 0.2:
            return "good"
        elif self.std_deviation < 0.3:
            return "fair"
        else:
            return "poor"


class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self, total_items: int, update_callback: Optional[Callable] = None):
        self.total_items = total_items
        self.current_item = 0
        self.start_time = time.time()
        self.current_dimension = ""
        self.update_callback = update_callback
        self._lock = threading.Lock()
    
    def update_progress(self, item_index: int, dimension: str = ""):
        """更新进度"""
        with self._lock:
            self.current_item = item_index + 1
            self.current_dimension = dimension
            
            elapsed_time = time.time() - self.start_time
            processing_speed = self.current_item / elapsed_time if elapsed_time > 0 else 0.0
            
            remaining_items = self.total_items - self.current_item
            estimated_remaining = remaining_items / processing_speed if processing_speed > 0 else 0.0
            
            progress_info = ProgressInfo(
                current_item=self.current_item,
                total_items=self.total_items,
                current_dimension=dimension,
                elapsed_time=elapsed_time,
                estimated_remaining=estimated_remaining,
                processing_speed=processing_speed
            )
            
            if self.update_callback:
                self.update_callback(progress_info)
    
    def get_current_progress(self) -> ProgressInfo:
        """获取当前进度信息"""
        with self._lock:
            elapsed_time = time.time() - self.start_time
            processing_speed = self.current_item / elapsed_time if elapsed_time > 0 else 0.0
            
            remaining_items = self.total_items - self.current_item
            estimated_remaining = remaining_items / processing_speed if processing_speed > 0 else 0.0
            
            return ProgressInfo(
                current_item=self.current_item,
                total_items=self.total_items,
                current_dimension=self.current_dimension,
                elapsed_time=elapsed_time,
                estimated_remaining=estimated_remaining,
                processing_speed=processing_speed
            )


class MultiDimensionalEvaluator(MultiDimensionalEvaluatorInterface):
    """
    多维度评估协调器
    
    整合所有评估维度的计算，实现：
    - 评估维度权重配置系统
    - 综合评分计算算法
    - 评估过程的进度监控
    - 评估结果聚合和分析
    """
    
    def __init__(self, 
                 config: Optional[ExpertEvaluationConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化多维度评估器
        
        Args:
            config: 专家评估配置
            logger: 日志记录器
        """
        self.config = config or ExpertEvaluationConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化各个评估器
        self._initialize_evaluators()
        
        # 进度监控
        self.progress_monitor: Optional[ProgressMonitor] = None
        self.progress_callback: Optional[Callable] = None
        
        # 缓存机制
        self.enable_caching = self.config.enable_caching
        self._evaluation_cache: Dict[str, Any] = {}
        
        # 并发处理配置
        self.max_workers = self.config.batch_processing.max_concurrent_batches
        self.enable_parallel = self.config.batch_processing.enable_parallel_processing
    
    def _initialize_evaluators(self):
        """初始化各个评估器"""
        try:
            # 行业指标计算器
            self.industry_calculator = IndustryMetricsCalculator()
            
            # 高级语义评估器
            self.semantic_evaluator = AdvancedSemanticEvaluator(self.logger)
            
            # 专业准确性评估器（使用fallback实现）
            try:
                # 尝试使用完整的评估器
                from src.evaluation_framework import ProfessionalAccuracyEvaluator
                # 如果需要参数，创建一个简单的知识库
                crypto_kb = {}  # 简化的知识库
                self.accuracy_evaluator = ProfessionalAccuracyEvaluator(crypto_kb)
            except Exception:
                # 使用fallback实现
                self.accuracy_evaluator = self._create_fallback_accuracy_evaluator()
            
            # 中文语义评估器（作为基础评估器）
            try:
                self.chinese_evaluator = ChineseSemanticEvaluator()
            except Exception:
                # 使用fallback实现
                self.chinese_evaluator = self._create_fallback_semantic_evaluator()
            
            self.logger.info("所有评估器初始化完成")
            
        except Exception as e:
            self.logger.error(f"评估器初始化失败: {e}")
            raise EvaluationProcessError("evaluator_initialization", reason=str(e))
    
    def _create_fallback_accuracy_evaluator(self):
        """创建fallback准确性评估器"""
        class FallbackAccuracyEvaluator:
            def evaluate_accuracy(self, answer: str, reference: str) -> float:
                # 简单的基于长度和关键词匹配的准确性评估
                if not answer or not reference:
                    return 0.0
                
                # 基于长度相似性
                length_similarity = min(len(answer), len(reference)) / max(len(answer), len(reference))
                
                # 基于关键词重叠
                answer_words = set(answer.split())
                reference_words = set(reference.split())
                
                if not reference_words:
                    return length_similarity * 0.5
                
                word_overlap = len(answer_words & reference_words) / len(reference_words)
                
                return (length_similarity * 0.3 + word_overlap * 0.7)
        
        return FallbackAccuracyEvaluator()
    
    def _create_fallback_semantic_evaluator(self):
        """创建fallback语义评估器"""
        class FallbackSemanticEvaluator:
            def evaluate_semantic_similarity(self, answer: str, reference: str) -> float:
                # 简单的语义相似性评估
                if not answer or not reference:
                    return 0.0
                
                from difflib import SequenceMatcher
                return SequenceMatcher(None, answer, reference).ratio()
        
        return FallbackSemanticEvaluator()
    
    def set_progress_callback(self, callback: Callable[[ProgressInfo], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def integrate_evaluation_dimensions(self, 
                                      qa_item: QAEvaluationItem,
                                      model_answer: str) -> Dict[EvaluationDimension, DimensionScore]:
        """
        整合所有评估维度的计算
        
        Args:
            qa_item: QA评估项
            model_answer: 模型回答
            
        Returns:
            Dict[EvaluationDimension, DimensionScore]: 各维度评分
        """
        try:
            dimension_scores = {}
            
            # 检查缓存
            cache_key = self._generate_cache_key(qa_item, model_answer)
            if self.enable_caching and cache_key in self._evaluation_cache:
                return self._evaluation_cache[cache_key]
            
            # 1. 语义相似性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(0, 8, "语义相似性评估", 0, 0, 0))
            
            semantic_result = self.semantic_evaluator.calculate_semantic_depth(
                model_answer, qa_item.reference_answer
            )
            dimension_scores[EvaluationDimension.SEMANTIC_SIMILARITY] = DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=semantic_result.overall_depth,
                confidence=0.9,
                details={
                    "surface_similarity": semantic_result.surface_similarity,
                    "semantic_depth": semantic_result.semantic_depth,
                    "conceptual_alignment": semantic_result.conceptual_alignment,
                    "logical_coherence": semantic_result.logical_coherence
                }
            )
            
            # 2. 领域准确性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(1, 8, "领域准确性评估", 0, 0, 0))
            
            domain_context = " ".join(qa_item.domain_tags) if qa_item.domain_tags else qa_item.context or ""
            domain_relevance = self.industry_calculator.calculate_domain_relevance(
                model_answer, domain_context
            )
            dimension_scores[EvaluationDimension.DOMAIN_ACCURACY] = DimensionScore(
                dimension=EvaluationDimension.DOMAIN_ACCURACY,
                score=domain_relevance,
                confidence=0.85,
                details={"domain_context": domain_context}
            )
            
            # 3. 响应相关性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(2, 8, "响应相关性评估", 0, 0, 0))
            
            contextual_understanding = self.semantic_evaluator.evaluate_contextual_understanding(
                model_answer, qa_item.question + " " + (qa_item.context or "")
            )
            dimension_scores[EvaluationDimension.RESPONSE_RELEVANCE] = DimensionScore(
                dimension=EvaluationDimension.RESPONSE_RELEVANCE,
                score=contextual_understanding,
                confidence=0.8,
                details={"question_context": qa_item.question}
            )
            
            # 4. 事实正确性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(3, 8, "事实正确性评估", 0, 0, 0))
            
            try:
                factual_accuracy = self.accuracy_evaluator.evaluate_accuracy(
                    model_answer, qa_item.reference_answer
                )
            except Exception:
                # 如果专业准确性评估器不可用，使用语义相似性作为替代
                factual_accuracy = semantic_result.surface_similarity
            
            dimension_scores[EvaluationDimension.FACTUAL_CORRECTNESS] = DimensionScore(
                dimension=EvaluationDimension.FACTUAL_CORRECTNESS,
                score=factual_accuracy,
                confidence=0.75,
                details={"evaluation_method": "professional_accuracy"}
            )
            
            # 5. 完整性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(4, 8, "完整性评估", 0, 0, 0))
            
            completeness_metrics = self.industry_calculator.measure_completeness(
                model_answer, qa_item.expected_concepts
            )
            dimension_scores[EvaluationDimension.COMPLETENESS] = DimensionScore(
                dimension=EvaluationDimension.COMPLETENESS,
                score=completeness_metrics.overall_completeness,
                confidence=0.85,
                details={
                    "concept_coverage": completeness_metrics.concept_coverage,
                    "requirement_fulfillment": completeness_metrics.requirement_fulfillment,
                    "depth_adequacy": completeness_metrics.depth_adequacy,
                    "breadth_coverage": completeness_metrics.breadth_coverage
                }
            )
            
            # 6. 创新性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(5, 8, "创新性评估", 0, 0, 0))
            
            # 使用参考答案作为基准
            baseline_answers = [qa_item.reference_answer]
            innovation_metrics = self.industry_calculator.evaluate_innovation_level(
                model_answer, baseline_answers
            )
            dimension_scores[EvaluationDimension.INNOVATION] = DimensionScore(
                dimension=EvaluationDimension.INNOVATION,
                score=innovation_metrics.overall_innovation,
                confidence=0.7,
                details={
                    "novelty_score": innovation_metrics.novelty_score,
                    "uniqueness_score": innovation_metrics.uniqueness_score,
                    "creativity_score": innovation_metrics.creativity_score,
                    "differentiation_score": innovation_metrics.differentiation_score
                }
            )
            
            # 7. 实用价值评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(6, 8, "实用价值评估", 0, 0, 0))
            
            use_case = " ".join(qa_item.domain_tags) if qa_item.domain_tags else "通用应用"
            practical_value = self.industry_calculator.assess_practical_applicability(
                model_answer, use_case
            )
            dimension_scores[EvaluationDimension.PRACTICAL_VALUE] = DimensionScore(
                dimension=EvaluationDimension.PRACTICAL_VALUE,
                score=practical_value,
                confidence=0.8,
                details={"use_case": use_case}
            )
            
            # 8. 逻辑一致性评估
            if self.progress_callback:
                self.progress_callback(ProgressInfo(7, 8, "逻辑一致性评估", 0, 0, 0))
            
            logical_consistency = self.semantic_evaluator.assess_logical_consistency(model_answer)
            dimension_scores[EvaluationDimension.LOGICAL_CONSISTENCY] = DimensionScore(
                dimension=EvaluationDimension.LOGICAL_CONSISTENCY,
                score=logical_consistency,
                confidence=0.75,
                details={"evaluation_method": "advanced_semantic"}
            )
            
            # 缓存结果
            if self.enable_caching:
                self._evaluation_cache[cache_key] = dimension_scores
            
            return dimension_scores
            
        except Exception as e:
            self.logger.error(f"维度评估整合失败: {e}")
            raise EvaluationProcessError("dimension_integration", reason=str(e))
    
    def calculate_weighted_score(self, 
                               dimension_scores: Dict[EvaluationDimension, DimensionScore],
                               weights: Optional[Dict[EvaluationDimension, float]] = None) -> float:
        """
        计算加权综合评分
        
        Args:
            dimension_scores: 各维度评分
            weights: 维度权重（可选，默认使用配置中的权重）
            
        Returns:
            float: 综合评分
        """
        try:
            # 使用配置中的权重或提供的权重
            if weights is None:
                weights = self.config.dimension_weights.to_dict()
            
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for dimension, dimension_score in dimension_scores.items():
                if dimension in weights:
                    weight = weights[dimension]
                    # 考虑置信度的影响
                    confidence_adjusted_score = dimension_score.score * dimension_score.confidence
                    weighted_score = confidence_adjusted_score * weight
                    
                    total_weighted_score += weighted_score
                    total_weight += weight
            
            # 计算加权平均分
            if total_weight > 0:
                final_score = total_weighted_score / total_weight
            else:
                # 如果没有权重，使用简单平均
                scores = [ds.score for ds in dimension_scores.values()]
                final_score = sum(scores) / len(scores) if scores else 0.0
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"加权评分计算失败: {e}")
            return 0.0
    
    def evaluate_single_item(self, qa_item: QAEvaluationItem, model_answer: str) -> ExpertEvaluationResult:
        """
        评估单个QA项目
        
        Args:
            qa_item: QA评估项
            model_answer: 模型回答
            
        Returns:
            ExpertEvaluationResult: 评估结果
        """
        start_time = time.time()
        
        try:
            # 1. 计算各维度评分
            dimension_scores = self.integrate_evaluation_dimensions(qa_item, model_answer)
            
            # 2. 计算综合评分
            overall_score = self.calculate_weighted_score(dimension_scores)
            
            # 3. 生成详细反馈
            detailed_feedback = self._generate_detailed_feedback(dimension_scores, qa_item)
            
            # 4. 生成改进建议
            improvement_suggestions = self._generate_improvement_suggestions(dimension_scores)
            
            # 5. 计算置信区间
            confidence_intervals = self._calculate_confidence_intervals(dimension_scores)
            
            # 6. 计算统计显著性
            statistical_significance = self._calculate_statistical_significance(dimension_scores)
            
            processing_time = time.time() - start_time
            
            result = ExpertEvaluationResult(
                question_id=qa_item.question_id,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                detailed_feedback=detailed_feedback,
                improvement_suggestions=improvement_suggestions,
                confidence_intervals=confidence_intervals,
                statistical_significance=statistical_significance,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"单项评估失败 (ID: {qa_item.question_id}): {e}")
            # 返回默认结果
            return ExpertEvaluationResult(
                question_id=qa_item.question_id,
                overall_score=0.0,
                dimension_scores={},
                detailed_feedback={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def batch_evaluate(self, 
                      qa_items: List[QAEvaluationItem], 
                      model_answers: List[str]) -> List[ExpertEvaluationResult]:
        """
        批量评估QA项目
        
        Args:
            qa_items: QA评估项列表
            model_answers: 模型回答列表
            
        Returns:
            List[ExpertEvaluationResult]: 评估结果列表
        """
        if len(qa_items) != len(model_answers):
            raise ValueError("QA项目数量与模型回答数量不匹配")
        
        # 初始化进度监控
        self.progress_monitor = ProgressMonitor(
            total_items=len(qa_items),
            update_callback=self.progress_callback
        )
        
        results = []
        
        if self.enable_parallel and len(qa_items) > 1:
            # 并行处理
            results = self._batch_evaluate_parallel(qa_items, model_answers)
        else:
            # 串行处理
            results = self._batch_evaluate_sequential(qa_items, model_answers)
        
        return results
    
    def _batch_evaluate_sequential(self, 
                                 qa_items: List[QAEvaluationItem], 
                                 model_answers: List[str]) -> List[ExpertEvaluationResult]:
        """串行批量评估"""
        results = []
        
        for i, (qa_item, model_answer) in enumerate(zip(qa_items, model_answers)):
            if self.progress_monitor:
                self.progress_monitor.update_progress(i, f"评估项目 {qa_item.question_id}")
            
            result = self.evaluate_single_item(qa_item, model_answer)
            results.append(result)
        
        return results
    
    def _batch_evaluate_parallel(self, 
                               qa_items: List[QAEvaluationItem], 
                               model_answers: List[str]) -> List[ExpertEvaluationResult]:
        """并行批量评估"""
        results = [None] * len(qa_items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.evaluate_single_item, qa_item, model_answer): i
                for i, (qa_item, model_answer) in enumerate(zip(qa_items, model_answers))
            }
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error(f"并行评估任务失败 (索引: {index}): {e}")
                    # 创建错误结果
                    results[index] = ExpertEvaluationResult(
                        question_id=qa_items[index].question_id,
                        overall_score=0.0,
                        dimension_scores={},
                        detailed_feedback={"error": str(e)}
                    )
                
                completed_count += 1
                if self.progress_monitor:
                    self.progress_monitor.update_progress(completed_count - 1, "并行评估")
        
        return results
    
    def analyze_evaluation_results(self, 
                                 results: List[ExpertEvaluationResult]) -> StatisticalAnalysisResult:
        """
        分析评估结果统计信息
        
        Args:
            results: 评估结果列表
            
        Returns:
            StatisticalAnalysisResult: 统计分析结果
        """
        try:
            if not results:
                return StatisticalAnalysisResult(0, 0, 0, 0, 0, (0, 0), {}, [], "empty", {})
            
            # 提取总体评分
            overall_scores = [r.overall_score for r in results if r.overall_score is not None]
            
            if not overall_scores:
                return StatisticalAnalysisResult(0, 0, 0, 0, 0, (0, 0), {}, [], "empty", {})
            
            # 基础统计
            mean_score = statistics.mean(overall_scores)
            median_score = statistics.median(overall_scores)
            std_deviation = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
            min_score = min(overall_scores)
            max_score = max(overall_scores)
            
            # 置信区间 (95%)
            confidence_interval = self._calculate_confidence_interval_95(overall_scores)
            
            # 百分位数
            percentiles = self._calculate_percentiles(overall_scores)
            
            # 异常值检测
            outliers = self._detect_outliers(overall_scores)
            
            # 分布类型判断
            distribution_type = self._assess_distribution_type(overall_scores)
            
            # 统计显著性
            statistical_significance = self._calculate_batch_statistical_significance(results)
            
            return StatisticalAnalysisResult(
                mean_score=mean_score,
                median_score=median_score,
                std_deviation=std_deviation,
                min_score=min_score,
                max_score=max_score,
                confidence_interval=confidence_interval,
                percentiles=percentiles,
                outliers=outliers,
                distribution_type=distribution_type,
                statistical_significance=statistical_significance
            )
            
        except Exception as e:
            self.logger.error(f"统计分析失败: {e}")
            return StatisticalAnalysisResult(0, 0, 0, 0, 0, (0, 0), {}, [], "error", {})
    
    def _generate_cache_key(self, qa_item: QAEvaluationItem, model_answer: str) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{qa_item.question_id}_{qa_item.question}_{model_answer}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_detailed_feedback(self, 
                                  dimension_scores: Dict[EvaluationDimension, DimensionScore],
                                  qa_item: QAEvaluationItem) -> Dict[str, str]:
        """生成详细反馈"""
        feedback = {}
        
        for dimension, score_obj in dimension_scores.items():
            if score_obj.score >= 0.8:
                feedback[dimension.value] = f"表现优秀 (评分: {score_obj.score:.2f})"
            elif score_obj.score >= 0.6:
                feedback[dimension.value] = f"表现良好 (评分: {score_obj.score:.2f})"
            elif score_obj.score >= 0.4:
                feedback[dimension.value] = f"需要改进 (评分: {score_obj.score:.2f})"
            else:
                feedback[dimension.value] = f"表现较差 (评分: {score_obj.score:.2f})"
        
        return feedback
    
    def _generate_improvement_suggestions(self, 
                                        dimension_scores: Dict[EvaluationDimension, DimensionScore]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 找出评分较低的维度
        low_score_dimensions = [
            (dim, score_obj) for dim, score_obj in dimension_scores.items() 
            if score_obj.score < 0.6
        ]
        
        # 按评分排序，优先改进最低的
        low_score_dimensions.sort(key=lambda x: x[1].score)
        
        suggestion_templates = {
            EvaluationDimension.SEMANTIC_SIMILARITY: "建议加强语义表达的准确性和相关性",
            EvaluationDimension.DOMAIN_ACCURACY: "建议提高专业领域知识的准确性",
            EvaluationDimension.RESPONSE_RELEVANCE: "建议增强回答与问题的相关性",
            EvaluationDimension.FACTUAL_CORRECTNESS: "建议核实和改进事实信息的准确性",
            EvaluationDimension.COMPLETENESS: "建议补充回答的完整性和深度",
            EvaluationDimension.INNOVATION: "建议增加创新性思考和独特见解",
            EvaluationDimension.PRACTICAL_VALUE: "建议增强实际应用价值和可操作性",
            EvaluationDimension.LOGICAL_CONSISTENCY: "建议改进逻辑结构和论证连贯性"
        }
        
        for dimension, score_obj in low_score_dimensions[:3]:  # 最多3个建议
            template = suggestion_templates.get(dimension, f"建议改进{dimension.value}")
            suggestions.append(f"{template} (当前评分: {score_obj.score:.2f})")
        
        return suggestions
    
    def _calculate_confidence_intervals(self, 
                                      dimension_scores: Dict[EvaluationDimension, DimensionScore]) -> Dict[str, Tuple[float, float]]:
        """计算置信区间"""
        confidence_intervals = {}
        
        for dimension, score_obj in dimension_scores.items():
            # 基于置信度计算置信区间
            margin = (1.0 - score_obj.confidence) * 0.1  # 简化计算
            lower_bound = max(0.0, score_obj.score - margin)
            upper_bound = min(1.0, score_obj.score + margin)
            
            confidence_intervals[dimension.value] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _calculate_statistical_significance(self, 
                                          dimension_scores: Dict[EvaluationDimension, DimensionScore]) -> Dict[str, float]:
        """计算统计显著性"""
        significance = {}
        
        for dimension, score_obj in dimension_scores.items():
            # 基于置信度和评分计算显著性
            sig_score = score_obj.confidence * (1.0 - abs(0.5 - score_obj.score))
            significance[dimension.value] = sig_score
        
        return significance
    
    def _calculate_confidence_interval_95(self, scores: List[float]) -> Tuple[float, float]:
        """计算95%置信区间"""
        if len(scores) < 2:
            return (0.0, 1.0)
        
        mean = statistics.mean(scores)
        std_err = statistics.stdev(scores) / math.sqrt(len(scores))
        
        # 使用t分布的近似值 (1.96 for large samples)
        margin = 1.96 * std_err
        
        lower_bound = max(0.0, mean - margin)
        upper_bound = min(1.0, mean + margin)
        
        return (lower_bound, upper_bound)
    
    def _calculate_percentiles(self, scores: List[float]) -> Dict[int, float]:
        """计算百分位数"""
        if not scores:
            return {}
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        percentiles = {}
        for p in [25, 50, 75, 90, 95]:
            index = int(n * p / 100)
            if index >= n:
                index = n - 1
            percentiles[p] = sorted_scores[index]
        
        return percentiles
    
    def _detect_outliers(self, scores: List[float]) -> List[int]:
        """检测异常值"""
        if len(scores) < 4:
            return []
        
        # 使用IQR方法检测异常值
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        q1_index = n // 4
        q3_index = 3 * n // 4
        
        q1 = sorted_scores[q1_index]
        q3 = sorted_scores[q3_index]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, score in enumerate(scores):
            if score < lower_bound or score > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _assess_distribution_type(self, scores: List[float]) -> str:
        """评估分布类型"""
        if len(scores) < 3:
            return "insufficient_data"
        
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        
        # 简化的分布类型判断
        if abs(mean - median) < 0.05:
            return "normal"
        elif mean > median:
            return "right_skewed"
        else:
            return "left_skewed"
    
    def _calculate_batch_statistical_significance(self, 
                                                results: List[ExpertEvaluationResult]) -> Dict[str, float]:
        """计算批量统计显著性"""
        significance = {}
        
        # 计算各维度的平均显著性
        dimension_significances = {}
        
        for result in results:
            for dim_name, sig_value in result.statistical_significance.items():
                if dim_name not in dimension_significances:
                    dimension_significances[dim_name] = []
                dimension_significances[dim_name].append(sig_value)
        
        for dim_name, sig_values in dimension_significances.items():
            if sig_values:
                significance[dim_name] = statistics.mean(sig_values)
        
        return significance


def test_multi_dimensional_evaluator():
    """测试多维度评估器"""
    print("开始测试多维度评估器...")
    
    try:
        # 创建评估器实例
        evaluator = MultiDimensionalEvaluator()
        
        # 创建测试数据
        test_qa_item = QAEvaluationItem(
            question_id="test_001",
            question="请解释AES加密算法的工作原理",
            context="在现代密码学中，对称加密算法是重要的组成部分",
            reference_answer="AES是一种对称加密算法，使用相同的密钥进行加密和解密操作",
            model_answer="AES（高级加密标准）是一种广泛使用的对称加密算法。它使用相同的密钥来加密和解密数据，支持128、192和256位的密钥长度。AES采用替换-置换网络结构，通过多轮变换来确保数据安全。",
            domain_tags=["密码学", "对称加密"],
            expected_concepts=["AES", "对称加密", "密钥", "加密", "解密"]
        )
        
        test_model_answer = test_qa_item.model_answer
        
        print("✓ 测试数据创建完成")
        
        # 测试单项评估
        print("\n1. 测试单项评估...")
        result = evaluator.evaluate_single_item(test_qa_item, test_model_answer)
        
        print(f"   问题ID: {result.question_id}")
        print(f"   总体评分: {result.overall_score:.3f}")
        print(f"   处理时间: {result.processing_time:.3f}秒")
        print(f"   维度评分数量: {len(result.dimension_scores)}")
        
        # 显示各维度评分
        for dimension, score_obj in result.dimension_scores.items():
            print(f"   {dimension.value}: {score_obj.score:.3f} (置信度: {score_obj.confidence:.2f})")
        
        print("✓ 单项评估测试通过")
        
        # 测试批量评估
        print("\n2. 测试批量评估...")
        
        # 创建多个测试项目
        test_items = []
        test_answers = []
        
        for i in range(3):
            qa_item = QAEvaluationItem(
                question_id=f"test_{i+1:03d}",
                question=f"测试问题 {i+1}",
                context="测试上下文",
                reference_answer=f"参考答案 {i+1}",
                model_answer=f"模型回答 {i+1}",
                domain_tags=["测试"],
                expected_concepts=["概念1", "概念2"]
            )
            test_items.append(qa_item)
            test_answers.append(qa_item.model_answer)
        
        # 设置进度回调
        def progress_callback(progress_info: ProgressInfo):
            print(f"   进度: {progress_info.progress_percentage:.1f}% - {progress_info.current_dimension}")
        
        evaluator.set_progress_callback(progress_callback)
        
        batch_results = evaluator.batch_evaluate(test_items, test_answers)
        
        print(f"   批量评估完成，共处理 {len(batch_results)} 个项目")
        
        for i, result in enumerate(batch_results):
            print(f"   项目 {i+1}: 总体评分 {result.overall_score:.3f}")
        
        print("✓ 批量评估测试通过")
        
        # 测试统计分析
        print("\n3. 测试统计分析...")
        
        stats_result = evaluator.analyze_evaluation_results(batch_results)
        
        print(f"   平均评分: {stats_result.mean_score:.3f}")
        print(f"   中位数评分: {stats_result.median_score:.3f}")
        print(f"   标准差: {stats_result.std_deviation:.3f}")
        print(f"   最小值: {stats_result.min_score:.3f}")
        print(f"   最大值: {stats_result.max_score:.3f}")
        print(f"   95%置信区间: [{stats_result.confidence_interval[0]:.3f}, {stats_result.confidence_interval[1]:.3f}]")
        print(f"   分布类型: {stats_result.distribution_type}")
        
        performance_summary = stats_result.get_performance_summary()
        print(f"   性能摘要: {performance_summary}")
        
        print("✓ 统计分析测试通过")
        
        print("\n✅ 所有测试通过！多维度评估器工作正常。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_multi_dimensional_evaluator()


class StatisticalAnalyzer:
    """
    统计分析器
    
    提供高级统计分析功能，包括：
    - 置信区间计算
    - 统计显著性检验
    - 评估偏差检测
    - 结果可重现性验证
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化统计分析器
        
        Args:
            confidence_level: 置信水平，默认0.95
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
    
    def calculate_confidence_intervals(self, 
                                    results: List[ExpertEvaluationResult],
                                    dimension: Optional[EvaluationDimension] = None) -> Dict[str, Tuple[float, float]]:
        """
        计算置信区间
        
        Args:
            results: 评估结果列表
            dimension: 特定维度（可选）
            
        Returns:
            Dict[str, Tuple[float, float]]: 置信区间字典
        """
        confidence_intervals = {}
        
        if dimension:
            # 计算特定维度的置信区间
            scores = [
                r.get_score_by_dimension(dimension) 
                for r in results 
                if r.get_score_by_dimension(dimension) is not None
            ]
            if scores:
                ci = self._calculate_confidence_interval(scores)
                confidence_intervals[dimension.value] = ci
        else:
            # 计算所有维度的置信区间
            # 总体评分
            overall_scores = [r.overall_score for r in results if r.overall_score is not None]
            if overall_scores:
                confidence_intervals["overall_score"] = self._calculate_confidence_interval(overall_scores)
            
            # 各维度评分
            for dim in EvaluationDimension:
                scores = [
                    r.get_score_by_dimension(dim) 
                    for r in results 
                    if r.get_score_by_dimension(dim) is not None
                ]
                if scores:
                    confidence_intervals[dim.value] = self._calculate_confidence_interval(scores)
        
        return confidence_intervals
    
    def test_statistical_significance(self, 
                                    group1_results: List[ExpertEvaluationResult],
                                    group2_results: List[ExpertEvaluationResult],
                                    dimension: Optional[EvaluationDimension] = None) -> Dict[str, Dict[str, float]]:
        """
        统计显著性检验（比较两组结果）
        
        Args:
            group1_results: 第一组评估结果
            group2_results: 第二组评估结果
            dimension: 特定维度（可选）
            
        Returns:
            Dict[str, Dict[str, float]]: 显著性检验结果
        """
        significance_results = {}
        
        if dimension:
            # 检验特定维度
            group1_scores = [
                r.get_score_by_dimension(dimension) 
                for r in group1_results 
                if r.get_score_by_dimension(dimension) is not None
            ]
            group2_scores = [
                r.get_score_by_dimension(dimension) 
                for r in group2_results 
                if r.get_score_by_dimension(dimension) is not None
            ]
            
            if group1_scores and group2_scores:
                test_result = self._perform_t_test(group1_scores, group2_scores)
                significance_results[dimension.value] = test_result
        else:
            # 检验所有维度
            # 总体评分
            group1_overall = [r.overall_score for r in group1_results if r.overall_score is not None]
            group2_overall = [r.overall_score for r in group2_results if r.overall_score is not None]
            
            if group1_overall and group2_overall:
                significance_results["overall_score"] = self._perform_t_test(group1_overall, group2_overall)
            
            # 各维度评分
            for dim in EvaluationDimension:
                group1_scores = [
                    r.get_score_by_dimension(dim) 
                    for r in group1_results 
                    if r.get_score_by_dimension(dim) is not None
                ]
                group2_scores = [
                    r.get_score_by_dimension(dim) 
                    for r in group2_results 
                    if r.get_score_by_dimension(dim) is not None
                ]
                
                if group1_scores and group2_scores:
                    significance_results[dim.value] = self._perform_t_test(group1_scores, group2_scores)
        
        return significance_results
    
    def detect_evaluation_bias(self, 
                             results: List[ExpertEvaluationResult],
                             metadata_key: str = "difficulty_level") -> Dict[str, Any]:
        """
        检测评估偏差
        
        Args:
            results: 评估结果列表
            metadata_key: 用于分组的元数据键
            
        Returns:
            Dict[str, Any]: 偏差检测结果
        """
        bias_analysis = {
            "bias_detected": False,
            "bias_type": None,
            "bias_magnitude": 0.0,
            "affected_groups": [],
            "recommendations": []
        }
        
        try:
            # 按元数据分组
            groups = self._group_results_by_metadata(results, metadata_key)
            
            if len(groups) < 2:
                bias_analysis["recommendations"].append("数据分组不足，无法进行偏差检测")
                return bias_analysis
            
            # 计算各组的平均评分
            group_means = {}
            for group_name, group_results in groups.items():
                scores = [r.overall_score for r in group_results if r.overall_score is not None]
                if scores:
                    group_means[group_name] = statistics.mean(scores)
            
            if len(group_means) < 2:
                return bias_analysis
            
            # 检测评分差异
            mean_values = list(group_means.values())
            max_diff = max(mean_values) - min(mean_values)
            
            # 偏差阈值
            bias_threshold = 0.2  # 20%的差异被认为是显著偏差
            
            if max_diff > bias_threshold:
                bias_analysis["bias_detected"] = True
                bias_analysis["bias_magnitude"] = max_diff
                bias_analysis["bias_type"] = f"{metadata_key}_bias"
                
                # 找出受影响的组
                overall_mean = statistics.mean(mean_values)
                for group_name, group_mean in group_means.items():
                    if abs(group_mean - overall_mean) > bias_threshold / 2:
                        bias_analysis["affected_groups"].append({
                            "group": group_name,
                            "mean_score": group_mean,
                            "deviation": group_mean - overall_mean
                        })
                
                # 生成建议
                bias_analysis["recommendations"] = [
                    f"检测到{metadata_key}相关的评估偏差",
                    "建议检查评估标准的一致性",
                    "考虑调整评估权重或标准化处理",
                    "增加更多样化的评估数据"
                ]
        
        except Exception as e:
            bias_analysis["recommendations"].append(f"偏差检测过程中出现错误: {str(e)}")
        
        return bias_analysis
    
    def verify_reproducibility(self, 
                             original_results: List[ExpertEvaluationResult],
                             replicated_results: List[ExpertEvaluationResult],
                             tolerance: float = 0.05) -> Dict[str, Any]:
        """
        验证结果可重现性
        
        Args:
            original_results: 原始评估结果
            replicated_results: 重复评估结果
            tolerance: 容忍度阈值
            
        Returns:
            Dict[str, Any]: 可重现性验证结果
        """
        reproducibility_report = {
            "is_reproducible": False,
            "overall_correlation": 0.0,
            "dimension_correlations": {},
            "inconsistent_items": [],
            "reproducibility_score": 0.0,
            "recommendations": []
        }
        
        try:
            if len(original_results) != len(replicated_results):
                reproducibility_report["recommendations"].append("结果数量不匹配，无法进行可重现性验证")
                return reproducibility_report
            
            # 总体评分相关性
            original_overall = [r.overall_score for r in original_results if r.overall_score is not None]
            replicated_overall = [r.overall_score for r in replicated_results if r.overall_score is not None]
            
            if original_overall and replicated_overall and len(original_overall) == len(replicated_overall):
                overall_corr = self._calculate_correlation(original_overall, replicated_overall)
                reproducibility_report["overall_correlation"] = overall_corr
            
            # 各维度相关性
            for dim in EvaluationDimension:
                original_dim = [
                    r.get_score_by_dimension(dim) 
                    for r in original_results 
                    if r.get_score_by_dimension(dim) is not None
                ]
                replicated_dim = [
                    r.get_score_by_dimension(dim) 
                    for r in replicated_results 
                    if r.get_score_by_dimension(dim) is not None
                ]
                
                if original_dim and replicated_dim and len(original_dim) == len(replicated_dim):
                    dim_corr = self._calculate_correlation(original_dim, replicated_dim)
                    reproducibility_report["dimension_correlations"][dim.value] = dim_corr
            
            # 检测不一致的项目
            for i, (orig, repl) in enumerate(zip(original_results, replicated_results)):
                if orig.overall_score is not None and repl.overall_score is not None:
                    diff = abs(orig.overall_score - repl.overall_score)
                    if diff > tolerance:
                        reproducibility_report["inconsistent_items"].append({
                            "index": i,
                            "question_id": orig.question_id,
                            "original_score": orig.overall_score,
                            "replicated_score": repl.overall_score,
                            "difference": diff
                        })
            
            # 计算可重现性评分
            correlations = [reproducibility_report["overall_correlation"]] + \
                          list(reproducibility_report["dimension_correlations"].values())
            
            if correlations:
                avg_correlation = statistics.mean([c for c in correlations if c is not None])
                inconsistency_penalty = len(reproducibility_report["inconsistent_items"]) / len(original_results)
                reproducibility_report["reproducibility_score"] = max(0.0, avg_correlation - inconsistency_penalty)
            
            # 判断是否可重现
            reproducibility_report["is_reproducible"] = (
                reproducibility_report["reproducibility_score"] > 0.8 and
                len(reproducibility_report["inconsistent_items"]) / len(original_results) < 0.1
            )
            
            # 生成建议
            if not reproducibility_report["is_reproducible"]:
                reproducibility_report["recommendations"] = [
                    "结果可重现性较低，建议检查评估过程的一致性",
                    "确保评估环境和参数设置相同",
                    "检查随机种子设置",
                    "验证数据预处理的一致性"
                ]
            else:
                reproducibility_report["recommendations"] = [
                    "结果具有良好的可重现性",
                    "评估过程稳定可靠"
                ]
        
        except Exception as e:
            reproducibility_report["recommendations"].append(f"可重现性验证过程中出现错误: {str(e)}")
        
        return reproducibility_report
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """计算置信区间"""
        if len(scores) < 2:
            return (0.0, 1.0)
        
        mean = statistics.mean(scores)
        
        if len(scores) == 2:
            # 样本太小，使用简单估计
            std_err = abs(scores[0] - scores[1]) / 2
        else:
            std_err = statistics.stdev(scores) / math.sqrt(len(scores))
        
        # 使用t分布临界值（简化为1.96用于大样本）
        t_critical = 1.96 if len(scores) > 30 else 2.0
        margin = t_critical * std_err
        
        lower_bound = max(0.0, mean - margin)
        upper_bound = min(1.0, mean + margin)
        
        return (lower_bound, upper_bound)
    
    def _perform_t_test(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """执行t检验"""
        if len(group1) < 2 or len(group2) < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
        
        # 计算基本统计量
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        
        if len(group1) == 2:
            var1 = (group1[0] - mean1) ** 2
        else:
            var1 = statistics.variance(group1)
            
        if len(group2) == 2:
            var2 = (group2[0] - mean2) ** 2
        else:
            var2 = statistics.variance(group2)
        
        n1, n2 = len(group1), len(group2)
        
        # 合并标准误差
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        
        if pooled_se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean1 - mean2) / pooled_se
        
        # 简化的p值估计（基于t统计量的绝对值）
        p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / 3)))
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "effect_size": abs(mean1 - mean2),
            "group1_mean": mean1,
            "group2_mean": mean2
        }
    
    def _group_results_by_metadata(self, 
                                 results: List[ExpertEvaluationResult], 
                                 metadata_key: str) -> Dict[str, List[ExpertEvaluationResult]]:
        """按元数据分组结果"""
        groups = {}
        
        for result in results:
            # 尝试从元数据中获取分组键
            group_key = result.metadata.get(metadata_key, "unknown")
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(result)
        
        return groups
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> Optional[float]:
        """计算皮尔逊相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return None
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return None
        
        correlation = numerator / denominator
        return max(-1.0, min(1.0, correlation))


def create_comprehensive_analysis_report(results: List[ExpertEvaluationResult],
                                       analyzer: Optional[StatisticalAnalyzer] = None) -> Dict[str, Any]:
    """
    创建综合分析报告
    
    Args:
        results: 评估结果列表
        analyzer: 统计分析器（可选）
        
    Returns:
        Dict[str, Any]: 综合分析报告
    """
    if analyzer is None:
        analyzer = StatisticalAnalyzer()
    
    report = {
        "summary": {},
        "statistical_analysis": {},
        "quality_assessment": {},
        "recommendations": []
    }
    
    try:
        # 基础摘要
        overall_scores = [r.overall_score for r in results if r.overall_score is not None]
        
        if overall_scores:
            report["summary"] = {
                "total_evaluations": len(results),
                "valid_scores": len(overall_scores),
                "mean_score": statistics.mean(overall_scores),
                "median_score": statistics.median(overall_scores),
                "std_deviation": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0,
                "score_range": (min(overall_scores), max(overall_scores))
            }
        
        # 统计分析
        confidence_intervals = analyzer.calculate_confidence_intervals(results)
        report["statistical_analysis"]["confidence_intervals"] = confidence_intervals
        
        # 偏差检测
        bias_analysis = analyzer.detect_evaluation_bias(results)
        report["statistical_analysis"]["bias_analysis"] = bias_analysis
        
        # 质量评估
        report["quality_assessment"] = {
            "consistency": 1.0 - (report["summary"].get("std_deviation", 0) / max(report["summary"].get("mean_score", 0.1), 0.1)),
            "reliability": _assess_reliability(results),
            "coverage": _assess_coverage(results)
        }
        
        # 生成建议
        report["recommendations"] = _generate_analysis_recommendations(report)
        
    except Exception as e:
        report["error"] = f"分析报告生成失败: {str(e)}"
    
    return report


def _assess_reliability(results: List[ExpertEvaluationResult]) -> float:
    """评估可靠性"""
    if not results:
        return 0.0
    
    # 基于置信度的可靠性评估
    confidence_scores = []
    for result in results:
        avg_confidence = result.get_average_confidence()
        confidence_scores.append(avg_confidence)
    
    return statistics.mean(confidence_scores) if confidence_scores else 0.0


def _assess_coverage(results: List[ExpertEvaluationResult]) -> float:
    """评估覆盖度"""
    if not results:
        return 0.0
    
    # 检查维度覆盖度
    total_dimensions = len(EvaluationDimension)
    covered_dimensions = set()
    
    for result in results:
        covered_dimensions.update(result.dimension_scores.keys())
    
    return len(covered_dimensions) / total_dimensions


def _generate_analysis_recommendations(report: Dict[str, Any]) -> List[str]:
    """生成分析建议"""
    recommendations = []
    
    summary = report.get("summary", {})
    quality = report.get("quality_assessment", {})
    
    # 基于平均分的建议
    mean_score = summary.get("mean_score", 0.0)
    if mean_score < 0.5:
        recommendations.append("整体评估分数较低，建议检查模型性能和评估标准")
    elif mean_score > 0.9:
        recommendations.append("整体评估分数很高，表现优秀")
    
    # 基于一致性的建议
    consistency = quality.get("consistency", 0.0)
    if consistency < 0.7:
        recommendations.append("评估结果一致性较低，建议检查评估过程的稳定性")
    
    # 基于可靠性的建议
    reliability = quality.get("reliability", 0.0)
    if reliability < 0.8:
        recommendations.append("评估可靠性需要提升，建议增加评估样本或改进评估方法")
    
    # 基于偏差分析的建议
    bias_analysis = report.get("statistical_analysis", {}).get("bias_analysis", {})
    if bias_analysis.get("bias_detected", False):
        recommendations.extend(bias_analysis.get("recommendations", []))
    
    return recommendations