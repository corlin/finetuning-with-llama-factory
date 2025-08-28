"""
专家评估系统核心接口和抽象类

定义了专家评估系统的核心接口，确保各组件之间的一致性和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from .data_models import (
    QAEvaluationItem, 
    ExpertEvaluationResult, 
    BatchEvaluationResult,
    EvaluationReport,
    ValidationResult,
    EvaluationDataset
)
from .config import ExpertEvaluationConfig, EvaluationDimension


class ExpertEvaluationEngine(ABC):
    """专家评估引擎抽象基类"""
    
    @abstractmethod
    def __init__(self, config: ExpertEvaluationConfig):
        """
        初始化评估引擎
        
        Args:
            config: 专家评估配置
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        加载已合并的最终模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 加载是否成功
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, qa_data: List[QAEvaluationItem]) -> ExpertEvaluationResult:
        """
        执行单个QA数据集的评估
        
        Args:
            qa_data: QA评估数据列表
            
        Returns:
            ExpertEvaluationResult: 评估结果
        """
        pass
    
    @abstractmethod
    def batch_evaluate(self, qa_datasets: List[EvaluationDataset]) -> BatchEvaluationResult:
        """
        批量评估多个数据集
        
        Args:
            qa_datasets: 评估数据集列表
            
        Returns:
            BatchEvaluationResult: 批量评估结果
        """
        pass
    
    @abstractmethod
    def generate_report(self, results: ExpertEvaluationResult) -> EvaluationReport:
        """
        生成详细的评估报告
        
        Args:
            results: 评估结果
            
        Returns:
            EvaluationReport: 评估报告
        """
        pass


class IndustryMetricsCalculator(ABC):
    """行业指标计算器抽象基类"""
    
    @abstractmethod
    def calculate_domain_relevance(self, answer: str, domain_context: str) -> float:
        """
        计算答案与特定领域的相关性
        
        Args:
            answer: 模型回答
            domain_context: 领域上下文
            
        Returns:
            float: 领域相关性分数 (0-1)
        """
        pass
    
    @abstractmethod
    def assess_practical_applicability(self, answer: str, use_case: str) -> float:
        """
        评估实际应用价值
        
        Args:
            answer: 模型回答
            use_case: 使用场景
            
        Returns:
            float: 实用性分数 (0-1)
        """
        pass
    
    @abstractmethod
    def evaluate_innovation_level(self, answer: str, baseline_answers: List[str]) -> float:
        """
        评估创新性和独特性
        
        Args:
            answer: 模型回答
            baseline_answers: 基准答案列表
            
        Returns:
            float: 创新性分数 (0-1)
        """
        pass
    
    @abstractmethod
    def measure_completeness(self, answer: str, question_requirements: List[str]) -> float:
        """
        测量回答的完整性
        
        Args:
            answer: 模型回答
            question_requirements: 问题要求列表
            
        Returns:
            float: 完整性分数 (0-1)
        """
        pass


class AdvancedSemanticEvaluator(ABC):
    """高级语义评估器抽象基类"""
    
    @abstractmethod
    def calculate_semantic_depth(self, answer: str, reference: str) -> float:
        """
        计算语义深度，超越表面相似性
        
        Args:
            answer: 模型回答
            reference: 参考答案
            
        Returns:
            float: 语义深度分数 (0-1)
        """
        pass
    
    @abstractmethod
    def assess_logical_consistency(self, answer: str) -> float:
        """
        评估逻辑一致性
        
        Args:
            answer: 模型回答
            
        Returns:
            float: 逻辑一致性分数 (0-1)
        """
        pass
    
    @abstractmethod
    def evaluate_contextual_understanding(self, answer: str, context: str) -> float:
        """
        评估上下文理解能力
        
        Args:
            answer: 模型回答
            context: 上下文信息
            
        Returns:
            float: 上下文理解分数 (0-1)
        """
        pass
    
    @abstractmethod
    def measure_concept_coverage(self, answer: str, key_concepts: List[str]) -> float:
        """
        测量关键概念覆盖度
        
        Args:
            answer: 模型回答
            key_concepts: 关键概念列表
            
        Returns:
            float: 概念覆盖度分数 (0-1)
        """
        pass


class EvaluationDataManager(ABC):
    """评估数据管理器抽象基类"""
    
    @abstractmethod
    def load_qa_data(self, data_path: str) -> List[QAEvaluationItem]:
        """
        加载QA格式的测试数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            List[QAEvaluationItem]: QA评估数据列表
        """
        pass
    
    @abstractmethod
    def validate_data_format(self, qa_data: List[QAEvaluationItem]) -> ValidationResult:
        """
        验证数据格式的正确性
        
        Args:
            qa_data: QA评估数据列表
            
        Returns:
            ValidationResult: 验证结果
        """
        pass
    
    @abstractmethod
    def prepare_evaluation_dataset(self, raw_data: List[Dict[str, Any]]) -> EvaluationDataset:
        """
        准备评估数据集
        
        Args:
            raw_data: 原始数据列表
            
        Returns:
            EvaluationDataset: 评估数据集
        """
        pass
    
    @abstractmethod
    def export_results(self, results: ExpertEvaluationResult, format: str) -> str:
        """
        导出评估结果到不同格式
        
        Args:
            results: 评估结果
            format: 导出格式 (json, csv, xlsx)
            
        Returns:
            str: 导出文件路径
        """
        pass


class MultiDimensionalEvaluator(ABC):
    """多维度评估协调器抽象基类"""
    
    @abstractmethod
    def integrate_evaluation_dimensions(self, 
                                      qa_item: QAEvaluationItem,
                                      model_answer: str) -> Dict[EvaluationDimension, float]:
        """
        整合所有评估维度的计算
        
        Args:
            qa_item: QA评估项
            model_answer: 模型回答
            
        Returns:
            Dict[EvaluationDimension, float]: 各维度评分
        """
        pass
    
    @abstractmethod
    def calculate_weighted_score(self, 
                               dimension_scores: Dict[EvaluationDimension, float],
                               weights: Dict[EvaluationDimension, float]) -> float:
        """
        计算加权综合评分
        
        Args:
            dimension_scores: 各维度评分
            weights: 维度权重
            
        Returns:
            float: 综合评分
        """
        pass
    
    @abstractmethod
    def analyze_evaluation_results(self, 
                                 results: List[ExpertEvaluationResult]) -> Dict[str, Any]:
        """
        分析评估结果统计信息
        
        Args:
            results: 评估结果列表
            
        Returns:
            Dict[str, Any]: 统计分析结果
        """
        pass


class EvaluationReportGenerator(ABC):
    """评估报告生成器抽象基类"""
    
    @abstractmethod
    def generate_detailed_report(self, 
                               results: ExpertEvaluationResult,
                               format: str = "json") -> EvaluationReport:
        """
        生成详细评估报告
        
        Args:
            results: 评估结果
            format: 报告格式 (json, html, pdf)
            
        Returns:
            EvaluationReport: 评估报告
        """
        pass
    
    @abstractmethod
    def create_visualization_charts(self, 
                                  results: ExpertEvaluationResult) -> Dict[str, Any]:
        """
        创建可视化图表
        
        Args:
            results: 评估结果
            
        Returns:
            Dict[str, Any]: 图表数据
        """
        pass
    
    @abstractmethod
    def generate_improvement_suggestions(self, 
                                       results: ExpertEvaluationResult) -> List[str]:
        """
        生成改进建议
        
        Args:
            results: 评估结果
            
        Returns:
            List[str]: 改进建议列表
        """
        pass
    
    @abstractmethod
    def export_report(self, 
                     report: EvaluationReport, 
                     output_path: str,
                     format: str = "json") -> bool:
        """
        导出报告到文件
        
        Args:
            report: 评估报告
            output_path: 输出路径
            format: 导出格式
            
        Returns:
            bool: 导出是否成功
        """
        pass


class BatchProcessingInterface(ABC):
    """批量处理接口"""
    
    @abstractmethod
    def process_batch(self, 
                     batch_data: List[Any], 
                     batch_size: int = 32) -> List[Any]:
        """
        批量处理数据
        
        Args:
            batch_data: 批量数据
            batch_size: 批次大小
            
        Returns:
            List[Any]: 处理结果
        """
        pass
    
    @abstractmethod
    def optimize_batch_size(self, 
                          data_size: int, 
                          memory_limit: Optional[int] = None) -> int:
        """
        优化批次大小
        
        Args:
            data_size: 数据大小
            memory_limit: 内存限制
            
        Returns:
            int: 优化后的批次大小
        """
        pass


class PerformanceMonitorInterface(ABC):
    """性能监控接口"""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """开始性能监控"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        停止性能监控
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况
        
        Returns:
            Dict[str, float]: 内存使用统计
        """
        pass
    
    @abstractmethod
    def get_processing_speed(self) -> float:
        """
        获取处理速度
        
        Returns:
            float: 处理速度 (items/second)
        """
        pass


class ErrorHandlingInterface(ABC):
    """错误处理接口"""
    
    @abstractmethod
    def handle_evaluation_error(self, 
                              error: Exception, 
                              context: Dict[str, Any]) -> Optional[Any]:
        """
        处理评估错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            
        Returns:
            Optional[Any]: 恢复结果或None
        """
        pass
    
    @abstractmethod
    def implement_fallback_strategy(self, 
                                  failed_component: str,
                                  input_data: Any) -> Any:
        """
        实现降级策略
        
        Args:
            failed_component: 失败的组件名称
            input_data: 输入数据
            
        Returns:
            Any: 降级处理结果
        """
        pass
    
    @abstractmethod
    def log_error_details(self, 
                         error: Exception, 
                         context: Dict[str, Any]) -> None:
        """
        记录错误详情
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        pass