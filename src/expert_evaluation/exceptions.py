"""
专家评估系统异常类定义

定义了专家评估系统使用的各种异常类，提供详细的错误信息和恢复建议。
"""

from typing import Optional, Dict, Any, List


class ExpertEvaluationError(Exception):
    """专家评估系统基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ModelLoadError(ExpertEvaluationError):
    """模型加载失败异常"""
    
    def __init__(self, message: str, model_path: Optional[str] = None, 
                 suggestions: Optional[List[str]] = None):
        # 兼容旧的调用方式
        if model_path and not message.startswith("模型加载失败"):
            # 如果第一个参数是model_path，第二个是reason
            reason = message
            message = f"模型加载失败: {model_path} - {reason}"
            self.model_path = model_path
            self.reason = reason
        else:
            # 新的调用方式，第一个参数直接是message
            self.model_path = model_path or "unknown"
            self.reason = message
        
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.suggestions = suggestions or [
            "检查模型路径是否正确",
            "确认模型文件完整性",
            "验证模型格式兼容性",
            "检查系统内存是否充足"
        ]
    
    def get_recovery_suggestions(self) -> List[str]:
        """获取恢复建议"""
        return self.suggestions


class EvaluationProcessError(ExpertEvaluationError):
    """评估过程异常"""
    
    def __init__(self, stage: str, question_id: Optional[str] = None, 
                 reason: str = "", original_error: Optional[Exception] = None):
        message = f"评估过程失败 - 阶段: {stage}"
        if question_id:
            message += f", 问题ID: {question_id}"
        if reason:
            message += f" - {reason}"
        
        super().__init__(message, "EVALUATION_PROCESS_ERROR")
        self.stage = stage
        self.question_id = question_id
        self.reason = reason
        self.original_error = original_error
    
    def get_fallback_strategy(self) -> str:
        """获取降级策略"""
        fallback_strategies = {
            "model_inference": "使用缓存结果或简化评估",
            "metric_calculation": "使用基础指标替代",
            "semantic_analysis": "跳过高级语义分析",
            "report_generation": "生成简化报告"
        }
        return fallback_strategies.get(self.stage, "跳过当前步骤并继续")


class DataFormatError(ExpertEvaluationError):
    """数据格式错误异常"""
    
    def __init__(self, data_path: str, format_issues: List[str], 
                 line_number: Optional[int] = None):
        message = f"数据格式错误: {data_path}"
        if line_number:
            message += f" (行号: {line_number})"
        
        super().__init__(message, "DATA_FORMAT_ERROR")
        self.data_path = data_path
        self.format_issues = format_issues
        self.line_number = line_number
    
    def get_format_requirements(self) -> Dict[str, str]:
        """获取格式要求"""
        return {
            "required_fields": "question_id, question, reference_answer, model_answer",
            "optional_fields": "context, domain_tags, difficulty_level, expected_concepts",
            "encoding": "UTF-8",
            "format": "JSON或JSONL"
        }


class ConfigurationError(ExpertEvaluationError):
    """配置错误异常"""
    
    def __init__(self, config_section: str, invalid_params: List[str], 
                 validation_errors: Optional[Dict[str, str]] = None):
        message = f"配置错误 - 节: {config_section}, 无效参数: {', '.join(invalid_params)}"
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_section = config_section
        self.invalid_params = invalid_params
        self.validation_errors = validation_errors or {}
    
    def get_valid_ranges(self) -> Dict[str, str]:
        """获取有效参数范围"""
        return {
            "batch_size": "1-128",
            "learning_rate": "1e-6 - 1e-2",
            "threshold_values": "0.0 - 1.0",
            "dimension_weights": "总和必须为1.0"
        }


class ResourceError(ExpertEvaluationError):
    """资源不足异常"""
    
    def __init__(self, resource_type: str, required: str, available: str,
                 optimization_suggestions: Optional[List[str]] = None):
        message = f"资源不足 - {resource_type}: 需要 {required}, 可用 {available}"
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.required = required
        self.available = available
        self.optimization_suggestions = optimization_suggestions or []
    
    def get_optimization_strategies(self) -> List[str]:
        """获取优化策略"""
        strategies = {
            "memory": [
                "减少批次大小",
                "启用梯度检查点",
                "使用模型量化",
                "清理缓存数据"
            ],
            "gpu_memory": [
                "使用混合精度训练",
                "减少序列长度",
                "启用CPU卸载",
                "使用模型并行"
            ],
            "disk_space": [
                "清理临时文件",
                "压缩输出数据",
                "使用流式处理",
                "删除旧的检查点"
            ]
        }
        return strategies.get(self.resource_type, self.optimization_suggestions)


class ValidationError(ExpertEvaluationError):
    """数据验证异常"""
    
    def __init__(self, validation_type: str, failed_items: List[str],
                 error_details: Optional[Dict[str, List[str]]] = None):
        message = f"数据验证失败 - 类型: {validation_type}, 失败项目数: {len(failed_items)}"
        super().__init__(message, "VALIDATION_ERROR")
        self.validation_type = validation_type
        self.failed_items = failed_items
        self.error_details = error_details or {}
    
    def get_validation_rules(self) -> Dict[str, str]:
        """获取验证规则"""
        return {
            "question_length": "不超过2048字符",
            "answer_length": "不超过4096字符",
            "required_fields": "必须包含所有必需字段",
            "data_types": "字段类型必须正确",
            "encoding": "必须使用UTF-8编码"
        }


class MetricCalculationError(ExpertEvaluationError):
    """指标计算异常"""
    
    def __init__(self, metric_name: str, calculation_stage: str,
                 input_data: Optional[Dict[str, Any]] = None,
                 fallback_available: bool = True):
        message = f"指标计算失败 - 指标: {metric_name}, 阶段: {calculation_stage}"
        super().__init__(message, "METRIC_CALCULATION_ERROR")
        self.metric_name = metric_name
        self.calculation_stage = calculation_stage
        self.input_data = input_data
        self.fallback_available = fallback_available
    
    def get_fallback_metrics(self) -> List[str]:
        """获取可用的降级指标"""
        fallback_map = {
            "semantic_similarity": ["bleu_score", "rouge_score"],
            "domain_accuracy": ["keyword_matching", "concept_overlap"],
            "logical_consistency": ["sentence_coherence", "basic_validation"],
            "innovation_level": ["uniqueness_score", "diversity_measure"]
        }
        return fallback_map.get(self.metric_name, ["basic_score"])


class ReportGenerationError(ExpertEvaluationError):
    """报告生成异常"""
    
    def __init__(self, report_type: str, generation_stage: str,
                 output_format: str, partial_data_available: bool = False):
        message = f"报告生成失败 - 类型: {report_type}, 阶段: {generation_stage}, 格式: {output_format}"
        super().__init__(message, "REPORT_GENERATION_ERROR")
        self.report_type = report_type
        self.generation_stage = generation_stage
        self.output_format = output_format
        self.partial_data_available = partial_data_available
    
    def get_alternative_formats(self) -> List[str]:
        """获取可选格式"""
        return ["json", "txt", "csv"] if self.partial_data_available else ["txt"]


class BatchProcessingError(ExpertEvaluationError):
    """批量处理异常"""
    
    def __init__(self, batch_id: str, failed_items: List[str],
                 successful_items: List[str], error_summary: Dict[str, int]):
        message = f"批量处理部分失败 - 批次: {batch_id}, 失败: {len(failed_items)}, 成功: {len(successful_items)}"
        super().__init__(message, "BATCH_PROCESSING_ERROR")
        self.batch_id = batch_id
        self.failed_items = failed_items
        self.successful_items = successful_items
        self.error_summary = error_summary
    
    def get_retry_strategy(self) -> Dict[str, Any]:
        """获取重试策略"""
        return {
            "retry_failed_items": True,
            "reduce_batch_size": True,
            "increase_timeout": True,
            "use_fallback_methods": True,
            "max_retries": 3
        }


class CacheError(ExpertEvaluationError):
    """缓存异常"""
    
    def __init__(self, cache_operation: str, cache_key: str, reason: str):
        message = f"缓存操作失败 - 操作: {cache_operation}, 键: {cache_key} - {reason}"
        super().__init__(message, "CACHE_ERROR")
        self.cache_operation = cache_operation
        self.cache_key = cache_key
        self.reason = reason
    
    def should_continue_without_cache(self) -> bool:
        """是否应该在没有缓存的情况下继续"""
        return self.cache_operation in ["read", "write"]  # 读写失败可以继续，删除失败需要处理


class TimeoutError(ExpertEvaluationError):
    """超时异常"""
    
    def __init__(self, operation: str, timeout_seconds: float, 
                 elapsed_seconds: float, partial_results: Optional[Any] = None):
        message = f"操作超时 - {operation}: 超时限制 {timeout_seconds}s, 已用时 {elapsed_seconds}s"
        super().__init__(message, "TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.partial_results = partial_results
    
    def has_partial_results(self) -> bool:
        """是否有部分结果可用"""
        return self.partial_results is not None


# 异常处理工具函数
def handle_evaluation_exception(exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一异常处理函数
    
    Args:
        exception: 异常对象
        context: 异常上下文
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    error_info = {
        "handled": False,
        "recovery_action": None,
        "fallback_result": None,
        "should_retry": False,
        "error_details": {}
    }
    
    if isinstance(exception, ExpertEvaluationError):
        error_info["handled"] = True
        error_info["error_details"] = exception.to_dict()
        
        # 根据异常类型确定处理策略
        if isinstance(exception, ModelLoadError):
            error_info["recovery_action"] = "try_alternative_model"
            error_info["should_retry"] = True
            
        elif isinstance(exception, EvaluationProcessError):
            error_info["recovery_action"] = exception.get_fallback_strategy()
            error_info["should_retry"] = False
            
        elif isinstance(exception, ResourceError):
            error_info["recovery_action"] = "optimize_resource_usage"
            error_info["should_retry"] = True
            
        elif isinstance(exception, BatchProcessingError):
            error_info["recovery_action"] = "retry_failed_items"
            error_info["should_retry"] = True
            error_info["fallback_result"] = {
                "successful_items": exception.successful_items,
                "failed_items": exception.failed_items
            }
            
        elif isinstance(exception, TimeoutError):
            error_info["recovery_action"] = "use_partial_results"
            error_info["should_retry"] = False
            error_info["fallback_result"] = exception.partial_results
    
    return error_info


def create_error_recovery_plan(errors: List[Exception]) -> Dict[str, Any]:
    """
    创建错误恢复计划
    
    Args:
        errors: 错误列表
        
    Returns:
        Dict[str, Any]: 恢复计划
    """
    recovery_plan = {
        "total_errors": len(errors),
        "error_categories": {},
        "recovery_steps": [],
        "estimated_success_rate": 0.0
    }
    
    # 分类错误
    for error in errors:
        error_type = type(error).__name__
        recovery_plan["error_categories"][error_type] = \
            recovery_plan["error_categories"].get(error_type, 0) + 1
    
    # 生成恢复步骤
    if ModelLoadError.__name__ in recovery_plan["error_categories"]:
        recovery_plan["recovery_steps"].append({
            "step": "检查和修复模型加载问题",
            "priority": "high",
            "estimated_time": "5-10分钟"
        })
    
    if ResourceError.__name__ in recovery_plan["error_categories"]:
        recovery_plan["recovery_steps"].append({
            "step": "优化资源使用配置",
            "priority": "medium",
            "estimated_time": "2-5分钟"
        })
    
    if DataFormatError.__name__ in recovery_plan["error_categories"]:
        recovery_plan["recovery_steps"].append({
            "step": "修复数据格式问题",
            "priority": "high",
            "estimated_time": "10-30分钟"
        })
    
    # 估算成功率
    recoverable_errors = [
        ModelLoadError.__name__,
        ResourceError.__name__,
        BatchProcessingError.__name__,
        CacheError.__name__
    ]
    
    recoverable_count = sum(
        recovery_plan["error_categories"].get(error_type, 0)
        for error_type in recoverable_errors
    )
    
    recovery_plan["estimated_success_rate"] = min(
        recoverable_count / len(errors) * 0.8, 0.95
    ) if errors else 0.0
    
    return recovery_plan

class PerformanceError(ExpertEvaluationError):
    """性能相关异常"""
    
    def __init__(self, performance_issue: str, current_metrics: Optional[Dict[str, Any]] = None,
                 optimization_suggestions: Optional[List[str]] = None):
        message = f"性能问题 - {performance_issue}"
        super().__init__(message, "PERFORMANCE_ERROR")
        self.performance_issue = performance_issue
        self.current_metrics = current_metrics or {}
        self.optimization_suggestions = optimization_suggestions or []
    
    def get_performance_recommendations(self) -> List[str]:
        """获取性能优化建议"""
        default_recommendations = [
            "检查系统资源使用情况",
            "优化批处理大小",
            "启用性能监控",
            "考虑使用缓存机制",
            "检查并发处理配置"
        ]
        
        return self.optimization_suggestions + default_recommendations


class ImprovementAnalysisError(ExpertEvaluationError):
    """改进分析异常"""
    
    def __init__(self, analysis_stage: str, reason: str, 
                 partial_suggestions: Optional[List[str]] = None):
        message = f"改进分析失败 - 阶段: {analysis_stage} - {reason}"
        super().__init__(message, "IMPROVEMENT_ANALYSIS_ERROR")
        self.analysis_stage = analysis_stage
        self.reason = reason
        self.partial_suggestions = partial_suggestions or []
    
    def get_fallback_suggestions(self) -> List[str]:
        """获取降级建议"""
        fallback_suggestions = [
            "建议进行更详细的人工评估以获得具体改进方向",
            "考虑增加训练数据的多样性和质量",
            "检查模型架构是否适合当前任务",
            "评估当前评估指标的适用性",
            "寻求领域专家的意见和建议"
        ]
        
        return self.partial_suggestions + fallback_suggestions
    
    def get_diagnostic_steps(self) -> List[str]:
        """获取诊断步骤"""
        return [
            "检查评估数据的质量和完整性",
            "验证评估指标的计算逻辑",
            "分析评估结果的分布特征",
            "对比历史评估结果趋势",
            "检查系统配置和参数设置"
        ]