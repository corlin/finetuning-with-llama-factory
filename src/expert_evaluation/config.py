"""
专家评估配置数据模型

定义了专家评估系统的配置类和枚举类型，支持灵活的配置管理。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path


class EvaluationDimension(Enum):
    """评估维度枚举"""
    SEMANTIC_SIMILARITY = "语义相似性"
    DOMAIN_ACCURACY = "领域准确性"
    RESPONSE_RELEVANCE = "响应相关性"
    FACTUAL_CORRECTNESS = "事实正确性"
    COMPLETENESS = "完整性"
    INNOVATION = "创新性"
    PRACTICAL_VALUE = "实用价值"
    LOGICAL_CONSISTENCY = "逻辑一致性"
    CONCEPT_COVERAGE = "概念覆盖度"
    TECHNICAL_DEPTH = "技术深度"


class ExpertiseLevel(Enum):
    """专业水平枚举"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class EvaluationMode(Enum):
    """评估模式枚举"""
    COMPREHENSIVE = "comprehensive"  # 全面评估
    QUICK = "quick"                 # 快速评估
    DETAILED = "detailed"           # 详细评估
    COMPARATIVE = "comparative"     # 对比评估


class OutputFormat(Enum):
    """输出格式枚举"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"


@dataclass
class DimensionWeightConfig:
    """评估维度权重配置"""
    semantic_similarity: float = 0.15
    domain_accuracy: float = 0.20
    response_relevance: float = 0.15
    factual_correctness: float = 0.15
    completeness: float = 0.10
    innovation: float = 0.08
    practical_value: float = 0.10
    logical_consistency: float = 0.07
    
    def __post_init__(self):
        """验证权重总和"""
        total_weight = sum([
            self.semantic_similarity,
            self.domain_accuracy,
            self.response_relevance,
            self.factual_correctness,
            self.completeness,
            self.innovation,
            self.practical_value,
            self.logical_consistency
        ])
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"权重总和应为1.0，当前为: {total_weight}")
    
    def to_dict(self) -> Dict[EvaluationDimension, float]:
        """转换为字典格式"""
        return {
            EvaluationDimension.SEMANTIC_SIMILARITY: self.semantic_similarity,
            EvaluationDimension.DOMAIN_ACCURACY: self.domain_accuracy,
            EvaluationDimension.RESPONSE_RELEVANCE: self.response_relevance,
            EvaluationDimension.FACTUAL_CORRECTNESS: self.factual_correctness,
            EvaluationDimension.COMPLETENESS: self.completeness,
            EvaluationDimension.INNOVATION: self.innovation,
            EvaluationDimension.PRACTICAL_VALUE: self.practical_value,
            EvaluationDimension.LOGICAL_CONSISTENCY: self.logical_consistency
        }


@dataclass
class ThresholdSettings:
    """阈值设置"""
    excellent_threshold: float = 0.9
    good_threshold: float = 0.7
    fair_threshold: float = 0.5
    poor_threshold: float = 0.3
    
    # 统计显著性阈值
    significance_level: float = 0.05
    confidence_level: float = 0.95
    
    # 性能阈值
    max_processing_time_per_item: float = 30.0  # 秒
    max_memory_usage_mb: float = 8192.0
    
    def __post_init__(self):
        """验证阈值设置"""
        thresholds = [
            self.excellent_threshold,
            self.good_threshold,
            self.fair_threshold,
            self.poor_threshold
        ]
        
        # 检查阈值递减顺序
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= thresholds[i + 1]:
                raise ValueError("阈值应按递减顺序设置")
        
        # 检查范围
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("阈值应在0.0-1.0范围内")


@dataclass
class BatchProcessingConfig:
    """批量处理配置"""
    # 基础批处理配置
    batch_size: int = 4  # 添加batch_size参数以兼容测试
    initial_batch_size: int = 4
    min_batch_size: int = 1
    max_batch_size: int = 32
    max_workers: int = 4
    
    # 处理策略
    processing_strategy: str = "adaptive"  # sequential, parallel_thread, parallel_process, adaptive
    
    # 超时和重试配置
    batch_timeout: int = 300  # 秒
    max_retries: int = 3
    
    # 内存和性能配置
    enable_memory_monitoring: bool = True
    enable_auto_adjustment: bool = True
    memory_threshold: float = 0.85
    
    # 进度监控
    enable_progress_tracking: bool = True
    progress_update_interval: int = 10  # 每处理多少项更新一次进度


@dataclass
class ReportConfig:
    """报告配置"""
    include_detailed_analysis: bool = True
    include_visualization: bool = True
    include_improvement_suggestions: bool = True
    include_statistical_analysis: bool = True
    
    # 输出格式
    default_format: OutputFormat = OutputFormat.JSON
    supported_formats: List[OutputFormat] = field(default_factory=lambda: [
        OutputFormat.JSON, OutputFormat.HTML, OutputFormat.PDF
    ])
    
    # 可视化配置
    chart_types: List[str] = field(default_factory=lambda: [
        "radar", "bar", "line", "heatmap"
    ])
    chart_resolution: str = "high"  # low, medium, high
    
    # 报告内容
    max_suggestions_count: int = 10
    include_raw_scores: bool = True
    include_confidence_intervals: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str = ""
    model_type: str = "auto"  # auto, transformers, pytorch
    device: str = "auto"  # auto, cpu, cuda, cuda:0
    
    # 加载配置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    
    # 推理配置
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    
    # 缓存配置
    use_cache: bool = True
    cache_dir: Optional[str] = None


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    qa_data_path: str = ""
    reference_data_path: Optional[str] = None
    output_dir: str = "./expert_evaluation_output"
    
    # 数据格式
    input_format: str = "json"  # json, jsonl, csv
    encoding: str = "utf-8"
    
    # 数据验证
    validate_input_data: bool = True
    skip_invalid_items: bool = True
    max_question_length: int = 2048
    max_answer_length: int = 4096
    
    # 数据预处理
    normalize_text: bool = True
    remove_extra_whitespace: bool = True
    handle_special_characters: bool = True


@dataclass
class ExpertEvaluationConfig:
    """专家评估主配置类"""
    
    # 基础配置
    evaluation_mode: EvaluationMode = EvaluationMode.COMPREHENSIVE
    enable_detailed_analysis: bool = True
    comparison_baseline: Optional[str] = None
    
    # 兼容性参数（直接传递给子配置）
    batch_size: Optional[int] = None
    
    # 子配置
    dimension_weights: DimensionWeightConfig = field(default_factory=DimensionWeightConfig)
    threshold_settings: ThresholdSettings = field(default_factory=ThresholdSettings)
    batch_processing: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    report_config: ReportConfig = field(default_factory=ReportConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # 高级配置
    enable_caching: bool = True
    cache_evaluation_results: bool = True
    enable_error_recovery: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_performance_logging: bool = True
    
    # 实验配置
    random_seed: int = 42
    reproducible_results: bool = True
    
    def __post_init__(self):
        """配置后处理和验证"""
        # 处理兼容性参数
        if self.batch_size is not None:
            self.batch_processing.batch_size = self.batch_size
            self.batch_processing.initial_batch_size = self.batch_size
        
        # 创建输出目录
        output_path = Path(self.data_config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 验证模型路径
        if self.model_config.model_path and not Path(self.model_config.model_path).exists():
            if not self.model_config.model_path.startswith(("huggingface://", "hf://", "/")):
                raise ValueError(f"模型路径不存在: {self.model_config.model_path}")
    
    def validate_config(self) -> Dict[str, bool]:
        """验证配置有效性"""
        validation_results = {
            "model_config_valid": True,
            "data_config_valid": True,
            "weights_valid": True,
            "thresholds_valid": True,
            "paths_exist": True
        }
        
        # 验证模型配置
        if not self.model_config.model_path:
            validation_results["model_config_valid"] = False
        
        # 验证数据配置
        if not self.data_config.qa_data_path:
            validation_results["data_config_valid"] = False
        
        # 验证权重配置
        try:
            self.dimension_weights.to_dict()
        except ValueError:
            validation_results["weights_valid"] = False
        
        # 验证阈值配置
        try:
            # 阈值验证在__post_init__中进行
            pass
        except ValueError:
            validation_results["thresholds_valid"] = False
        
        # 验证路径
        if self.data_config.qa_data_path and not Path(self.data_config.qa_data_path).exists():
            validation_results["paths_exist"] = False
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "evaluation_mode": self.evaluation_mode.value,
            "enable_detailed_analysis": self.enable_detailed_analysis,
            "comparison_baseline": self.comparison_baseline,
            "dimension_weights": self.dimension_weights.__dict__,
            "threshold_settings": self.threshold_settings.__dict__,
            "batch_processing": self.batch_processing.__dict__,
            "report_config": {
                **self.report_config.__dict__,
                "default_format": self.report_config.default_format.value,
                "supported_formats": [f.value for f in self.report_config.supported_formats]
            },
            "model_config": self.model_config.__dict__,
            "data_config": self.data_config.__dict__,
            "enable_caching": self.enable_caching,
            "cache_evaluation_results": self.cache_evaluation_results,
            "enable_error_recovery": self.enable_error_recovery,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_performance_logging": self.enable_performance_logging,
            "random_seed": self.random_seed,
            "reproducible_results": self.reproducible_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertEvaluationConfig':
        """从字典创建配置实例"""
        # 处理枚举类型
        if "evaluation_mode" in data:
            data["evaluation_mode"] = EvaluationMode(data["evaluation_mode"])
        
        # 处理嵌套配置
        if "dimension_weights" in data:
            data["dimension_weights"] = DimensionWeightConfig(**data["dimension_weights"])
        
        if "threshold_settings" in data:
            data["threshold_settings"] = ThresholdSettings(**data["threshold_settings"])
        
        if "batch_processing" in data:
            data["batch_processing"] = BatchProcessingConfig(**data["batch_processing"])
        
        if "report_config" in data:
            report_data = data["report_config"].copy()
            if "default_format" in report_data:
                report_data["default_format"] = OutputFormat(report_data["default_format"])
            if "supported_formats" in report_data:
                report_data["supported_formats"] = [
                    OutputFormat(f) for f in report_data["supported_formats"]
                ]
            data["report_config"] = ReportConfig(**report_data)
        
        if "model_config" in data:
            data["model_config"] = ModelConfig(**data["model_config"])
        
        if "data_config" in data:
            data["data_config"] = DataConfig(**data["data_config"])
        
        return cls(**data)
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        import json
        
        config_dict = self.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ExpertEvaluationConfig':
        """从文件加载配置"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def create_optimized_config(self, 
                              hardware_info: Optional[Dict[str, Any]] = None) -> 'ExpertEvaluationConfig':
        """
        根据硬件信息创建优化配置
        
        Args:
            hardware_info: 硬件信息字典
            
        Returns:
            ExpertEvaluationConfig: 优化后的配置
        """
        optimized_config = ExpertEvaluationConfig(**self.__dict__)
        
        if hardware_info:
            # 根据GPU内存调整批次大小
            gpu_memory = hardware_info.get("gpu_memory_mb", 0)
            if gpu_memory > 0:
                if gpu_memory < 8192:  # 8GB
                    optimized_config.batch_processing.batch_size = 8
                    optimized_config.model_config.load_in_4bit = True
                elif gpu_memory < 16384:  # 16GB
                    optimized_config.batch_processing.batch_size = 16
                    optimized_config.model_config.load_in_8bit = True
                else:
                    optimized_config.batch_processing.batch_size = 32
            
            # 根据CPU核心数调整并发
            cpu_cores = hardware_info.get("cpu_cores", 1)
            optimized_config.batch_processing.max_concurrent_batches = min(cpu_cores, 8)
        
        return optimized_config