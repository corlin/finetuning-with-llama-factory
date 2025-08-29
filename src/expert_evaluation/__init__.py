"""
专家模型评估模块

本模块实现了全面的专家级行业化评估系统，用于评估训练后已合并的最终模型。
提供比传统BLEU、ROUGE更适合行业场景的多维度评估能力。

主要组件:
- ExpertEvaluationEngine: 专家评估引擎
- IndustryMetricsCalculator: 行业指标计算器
- AdvancedSemanticEvaluator: 高级语义评估器
- EvaluationDataManager: 评估数据管理器
- MultiDimensionalEvaluator: 多维度评估协调器
- EvaluationReportGenerator: 评估报告生成器
"""

from .interfaces import (
    ExpertEvaluationEngine as ExpertEvaluationEngineInterface,
    IndustryMetricsCalculator as IndustryMetricsCalculatorInterface,
    AdvancedSemanticEvaluator as AdvancedSemanticEvaluatorInterface,
    EvaluationDataManager as EvaluationDataManagerInterface,
    MultiDimensionalEvaluator as MultiDimensionalEvaluatorInterface,
    EvaluationReportGenerator as EvaluationReportGeneratorInterface
)

# Import concrete implementations
from .engine import ExpertEvaluationEngine
from .report_generator import EvaluationReportGenerator
from .improvement_advisor import ImprovementAdvisor

from .config import (
    ExpertEvaluationConfig,
    EvaluationDimension,
    ExpertiseLevel
)

from .data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    BatchEvaluationResult,
    EvaluationReport,
    ValidationResult,
    EvaluationDataset
)

from .exceptions import (
    ExpertEvaluationError,
    ModelLoadError,
    EvaluationProcessError,
    DataFormatError,
    ConfigurationError,
    ResourceError,
    ValidationError,
    MetricCalculationError,
    ReportGenerationError,
    BatchProcessingError,
    CacheError,
    TimeoutError,
    ImprovementAnalysisError
)

__version__ = "1.0.0"
__author__ = "Expert Evaluation Team"

__all__ = [
    # 核心接口
    "ExpertEvaluationEngineInterface",
    "IndustryMetricsCalculatorInterface", 
    "AdvancedSemanticEvaluatorInterface",
    "EvaluationDataManagerInterface",
    "MultiDimensionalEvaluatorInterface",
    "EvaluationReportGeneratorInterface",
    
    # 具体实现
    "ExpertEvaluationEngine",
    "EvaluationReportGenerator",
    "ImprovementAdvisor",
    
    # 配置类
    "ExpertEvaluationConfig",
    "EvaluationDimension",
    "ExpertiseLevel",
    
    # 数据模型
    "QAEvaluationItem",
    "ExpertEvaluationResult",
    "BatchEvaluationResult", 
    "EvaluationReport",
    "ValidationResult",
    "EvaluationDataset",
    
    # 异常类
    "ExpertEvaluationError",
    "ModelLoadError",
    "EvaluationProcessError",
    "DataFormatError",
    "ConfigurationError",
    "ResourceError",
    "ValidationError",
    "MetricCalculationError",
    "ReportGenerationError",
    "BatchProcessingError",
    "CacheError",
    "TimeoutError",
    "ImprovementAnalysisError"
]