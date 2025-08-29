"""
异常处理单元测试

测试专家评估系统的异常类和错误处理机制，确保异常处理的正确性和完整性。
"""

import pytest
from typing import Dict, List, Any, Optional

from src.expert_evaluation.exceptions import (
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
    PerformanceError,
    ImprovementAnalysisError,
    handle_evaluation_exception,
    create_error_recovery_plan
)


class TestExpertEvaluationError:
    """ExpertEvaluationError基础异常测试"""
    
    def test_basic_error_creation(self):
        """测试基础异常创建"""
        error = ExpertEvaluationError(
            message="测试错误信息",
            error_code="TEST_ERROR",
            context={"component": "test", "operation": "test_op"}
        )
        
        assert str(error) == "[TEST_ERROR] 测试错误信息"
        assert error.message == "测试错误信息"
        assert error.error_code == "TEST_ERROR"
        assert error.context["component"] == "test"
    
    def test_error_without_code(self):
        """测试无错误代码的异常"""
        error = ExpertEvaluationError("简单错误信息")
        
        assert str(error) == "简单错误信息"
        assert error.error_code is None
        assert error.context == {}
    
    def test_error_to_dict(self):
        """测试异常转换为字典"""
        error = ExpertEvaluationError(
            message="测试错误",
            error_code="TEST_001",
            context={"data": "test_data"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "ExpertEvaluationError"
        assert error_dict["message"] == "测试错误"
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["context"]["data"] == "test_data"


class TestModelLoadError:
    """ModelLoadError测试类"""
    
    def test_model_load_error_creation(self):
        """测试模型加载错误创建"""
        error = ModelLoadError(
            model_path="/path/to/model",
            reason="文件不存在",
            suggestions=["检查路径", "验证权限"]
        )
        
        assert "模型加载失败" in str(error)
        assert "/path/to/model" in str(error)
        assert "文件不存在" in str(error)
        assert error.model_path == "/path/to/model"
        assert error.reason == "文件不存在"
        assert len(error.suggestions) == 2
    
    def test_default_suggestions(self):
        """测试默认建议"""
        error = ModelLoadError(
            model_path="/test/model",
            reason="加载失败"
        )
        
        suggestions = error.get_recovery_suggestions()
        
        assert len(suggestions) >= 4
        assert any("检查模型路径" in s for s in suggestions)
        assert any("确认模型文件完整性" in s for s in suggestions)
        assert any("验证模型格式兼容性" in s for s in suggestions)
        assert any("检查系统内存" in s for s in suggestions)
    
    def test_custom_suggestions(self):
        """测试自定义建议"""
        custom_suggestions = ["重新下载模型", "检查网络连接"]
        error = ModelLoadError(
            model_path="/test/model",
            reason="下载失败",
            suggestions=custom_suggestions
        )
        
        suggestions = error.get_recovery_suggestions()
        
        assert "重新下载模型" in suggestions
        assert "检查网络连接" in suggestions


class TestEvaluationProcessError:
    """EvaluationProcessError测试类"""
    
    def test_evaluation_process_error_creation(self):
        """测试评估过程错误创建"""
        error = EvaluationProcessError(
            stage="model_inference",
            question_id="q_001",
            reason="内存不足",
            original_error=RuntimeError("CUDA out of memory")
        )
        
        assert "评估过程失败" in str(error)
        assert "model_inference" in str(error)
        assert "q_001" in str(error)
        assert "内存不足" in str(error)
        assert error.stage == "model_inference"
        assert error.question_id == "q_001"
        assert isinstance(error.original_error, RuntimeError)
    
    def test_fallback_strategy(self):
        """测试降级策略"""
        error = EvaluationProcessError(stage="model_inference")
        strategy = error.get_fallback_strategy()
        assert strategy == "使用缓存结果或简化评估"
        
        error = EvaluationProcessError(stage="metric_calculation")
        strategy = error.get_fallback_strategy()
        assert strategy == "使用基础指标替代"
        
        error = EvaluationProcessError(stage="unknown_stage")
        strategy = error.get_fallback_strategy()
        assert strategy == "跳过当前步骤并继续"
    
    def test_minimal_error_info(self):
        """测试最小错误信息"""
        error = EvaluationProcessError(stage="test_stage")
        
        assert "test_stage" in str(error)
        assert error.question_id is None
        assert error.reason == ""
        assert error.original_error is None


class TestDataFormatError:
    """DataFormatError测试类"""
    
    def test_data_format_error_creation(self):
        """测试数据格式错误创建"""
        format_issues = ["缺少必需字段", "数据类型错误", "编码问题"]
        error = DataFormatError(
            data_path="/data/test.json",
            format_issues=format_issues,
            line_number=42
        )
        
        assert "数据格式错误" in str(error)
        assert "/data/test.json" in str(error)
        assert "行号: 42" in str(error)
        assert error.data_path == "/data/test.json"
        assert error.format_issues == format_issues
        assert error.line_number == 42
    
    def test_format_requirements(self):
        """测试格式要求"""
        error = DataFormatError(
            data_path="/test.json",
            format_issues=["测试问题"]
        )
        
        requirements = error.get_format_requirements()
        
        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "encoding" in requirements
        assert "format" in requirements
        
        assert "question_id" in requirements["required_fields"]
        assert "UTF-8" in requirements["encoding"]
    
    def test_without_line_number(self):
        """测试无行号的错误"""
        error = DataFormatError(
            data_path="/test.json",
            format_issues=["格式问题"]
        )
        
        assert "行号" not in str(error)
        assert error.line_number is None


class TestConfigurationError:
    """ConfigurationError测试类"""
    
    def test_configuration_error_creation(self):
        """测试配置错误创建"""
        invalid_params = ["batch_size", "learning_rate"]
        validation_errors = {
            "batch_size": "必须为正整数",
            "learning_rate": "必须在有效范围内"
        }
        
        error = ConfigurationError(
            config_section="model_config",
            invalid_params=invalid_params,
            validation_errors=validation_errors
        )
        
        assert "配置错误" in str(error)
        assert "model_config" in str(error)
        assert "batch_size" in str(error)
        assert "learning_rate" in str(error)
        assert error.config_section == "model_config"
        assert error.invalid_params == invalid_params
        assert error.validation_errors == validation_errors
    
    def test_valid_ranges(self):
        """测试有效参数范围"""
        error = ConfigurationError(
            config_section="test_section",
            invalid_params=["test_param"]
        )
        
        ranges = error.get_valid_ranges()
        
        assert "batch_size" in ranges
        assert "learning_rate" in ranges
        assert "threshold_values" in ranges
        assert "dimension_weights" in ranges
        
        assert "1-128" in ranges["batch_size"]
        assert "0.0 - 1.0" in ranges["threshold_values"]


class TestResourceError:
    """ResourceError测试类"""
    
    def test_resource_error_creation(self):
        """测试资源错误创建"""
        optimization_suggestions = ["减少批次大小", "启用模型量化"]
        error = ResourceError(
            resource_type="memory",
            required="16GB",
            available="8GB",
            optimization_suggestions=optimization_suggestions
        )
        
        assert "资源不足" in str(error)
        assert "memory" in str(error)
        assert "16GB" in str(error)
        assert "8GB" in str(error)
        assert error.resource_type == "memory"
        assert error.required == "16GB"
        assert error.available == "8GB"
        assert error.optimization_suggestions == optimization_suggestions
    
    def test_optimization_strategies(self):
        """测试优化策略"""
        # 内存优化策略
        error = ResourceError("memory", "16GB", "8GB")
        strategies = error.get_optimization_strategies()
        
        assert any("减少批次大小" in s for s in strategies)
        assert any("启用梯度检查点" in s for s in strategies)
        
        # GPU内存优化策略
        error = ResourceError("gpu_memory", "12GB", "6GB")
        strategies = error.get_optimization_strategies()
        
        assert any("混合精度训练" in s for s in strategies)
        assert any("CPU卸载" in s for s in strategies)
        
        # 磁盘空间优化策略
        error = ResourceError("disk_space", "100GB", "50GB")
        strategies = error.get_optimization_strategies()
        
        assert any("清理临时文件" in s for s in strategies)
        assert any("压缩输出数据" in s for s in strategies)
    
    def test_unknown_resource_type(self):
        """测试未知资源类型"""
        custom_suggestions = ["自定义建议1", "自定义建议2"]
        error = ResourceError(
            resource_type="unknown_resource",
            required="100",
            available="50",
            optimization_suggestions=custom_suggestions
        )
        
        strategies = error.get_optimization_strategies()
        
        assert "自定义建议1" in strategies
        assert "自定义建议2" in strategies


class TestValidationError:
    """ValidationError测试类"""
    
    def test_validation_error_creation(self):
        """测试验证错误创建"""
        failed_items = ["item_1", "item_2", "item_3"]
        error_details = {
            "item_1": ["字段缺失", "类型错误"],
            "item_2": ["长度超限"],
            "item_3": ["格式错误", "编码问题"]
        }
        
        error = ValidationError(
            validation_type="data_integrity",
            failed_items=failed_items,
            error_details=error_details
        )
        
        assert "数据验证失败" in str(error)
        assert "data_integrity" in str(error)
        assert "失败项目数: 3" in str(error)
        assert error.validation_type == "data_integrity"
        assert error.failed_items == failed_items
        assert error.error_details == error_details
    
    def test_validation_rules(self):
        """测试验证规则"""
        error = ValidationError(
            validation_type="format_check",
            failed_items=["test_item"]
        )
        
        rules = error.get_validation_rules()
        
        assert "question_length" in rules
        assert "answer_length" in rules
        assert "required_fields" in rules
        assert "data_types" in rules
        assert "encoding" in rules
        
        assert "2048字符" in rules["question_length"]
        assert "UTF-8" in rules["encoding"]


class TestMetricCalculationError:
    """MetricCalculationError测试类"""
    
    def test_metric_calculation_error_creation(self):
        """测试指标计算错误创建"""
        input_data = {"answer": "测试答案", "reference": "参考答案"}
        error = MetricCalculationError(
            metric_name="semantic_similarity",
            calculation_stage="embedding_generation",
            input_data=input_data,
            fallback_available=True
        )
        
        assert "指标计算失败" in str(error)
        assert "semantic_similarity" in str(error)
        assert "embedding_generation" in str(error)
        assert error.metric_name == "semantic_similarity"
        assert error.calculation_stage == "embedding_generation"
        assert error.input_data == input_data
        assert error.fallback_available is True
    
    def test_fallback_metrics(self):
        """测试降级指标"""
        error = MetricCalculationError(
            metric_name="semantic_similarity",
            calculation_stage="test"
        )
        fallback = error.get_fallback_metrics()
        
        assert "bleu_score" in fallback
        assert "rouge_score" in fallback
        
        error = MetricCalculationError(
            metric_name="domain_accuracy",
            calculation_stage="test"
        )
        fallback = error.get_fallback_metrics()
        
        assert "keyword_matching" in fallback
        assert "concept_overlap" in fallback
        
        error = MetricCalculationError(
            metric_name="unknown_metric",
            calculation_stage="test"
        )
        fallback = error.get_fallback_metrics()
        
        assert "basic_score" in fallback


class TestBatchProcessingError:
    """BatchProcessingError测试类"""
    
    def test_batch_processing_error_creation(self):
        """测试批量处理错误创建"""
        failed_items = ["item_1", "item_3", "item_5"]
        successful_items = ["item_2", "item_4", "item_6", "item_7"]
        error_summary = {
            "timeout_errors": 2,
            "memory_errors": 1,
            "format_errors": 0
        }
        
        error = BatchProcessingError(
            batch_id="batch_001",
            failed_items=failed_items,
            successful_items=successful_items,
            error_summary=error_summary
        )
        
        assert "批量处理部分失败" in str(error)
        assert "batch_001" in str(error)
        assert "失败: 3" in str(error)
        assert "成功: 4" in str(error)
        assert error.batch_id == "batch_001"
        assert error.failed_items == failed_items
        assert error.successful_items == successful_items
        assert error.error_summary == error_summary
    
    def test_retry_strategy(self):
        """测试重试策略"""
        error = BatchProcessingError(
            batch_id="test_batch",
            failed_items=["item_1"],
            successful_items=["item_2"],
            error_summary={}
        )
        
        strategy = error.get_retry_strategy()
        
        assert strategy["retry_failed_items"] is True
        assert strategy["reduce_batch_size"] is True
        assert strategy["increase_timeout"] is True
        assert strategy["use_fallback_methods"] is True
        assert strategy["max_retries"] == 3


class TestTimeoutError:
    """TimeoutError测试类"""
    
    def test_timeout_error_creation(self):
        """测试超时错误创建"""
        partial_results = {"completed": 5, "total": 10}
        error = TimeoutError(
            operation="batch_evaluation",
            timeout_seconds=300.0,
            elapsed_seconds=450.0,
            partial_results=partial_results
        )
        
        assert "操作超时" in str(error)
        assert "batch_evaluation" in str(error)
        assert "300.0s" in str(error)
        assert "450.0s" in str(error)
        assert error.operation == "batch_evaluation"
        assert error.timeout_seconds == 300.0
        assert error.elapsed_seconds == 450.0
        assert error.partial_results == partial_results
    
    def test_has_partial_results(self):
        """测试是否有部分结果"""
        # 有部分结果
        error = TimeoutError(
            operation="test_op",
            timeout_seconds=100.0,
            elapsed_seconds=150.0,
            partial_results={"data": "test"}
        )
        assert error.has_partial_results() is True
        
        # 无部分结果
        error = TimeoutError(
            operation="test_op",
            timeout_seconds=100.0,
            elapsed_seconds=150.0,
            partial_results=None
        )
        assert error.has_partial_results() is False


class TestPerformanceError:
    """PerformanceError测试类"""
    
    def test_performance_error_creation(self):
        """测试性能错误创建"""
        current_metrics = {
            "processing_speed": 0.5,
            "memory_usage": 0.95,
            "cpu_usage": 0.8
        }
        optimization_suggestions = ["优化算法", "增加缓存"]
        
        error = PerformanceError(
            performance_issue="处理速度过慢",
            current_metrics=current_metrics,
            optimization_suggestions=optimization_suggestions
        )
        
        assert "性能问题" in str(error)
        assert "处理速度过慢" in str(error)
        assert error.performance_issue == "处理速度过慢"
        assert error.current_metrics == current_metrics
        assert error.optimization_suggestions == optimization_suggestions
    
    def test_performance_recommendations(self):
        """测试性能优化建议"""
        error = PerformanceError(
            performance_issue="内存使用过高",
            optimization_suggestions=["减少缓存大小"]
        )
        
        recommendations = error.get_performance_recommendations()
        
        assert "减少缓存大小" in recommendations
        assert any("检查系统资源" in r for r in recommendations)
        assert any("优化批处理大小" in r for r in recommendations)


class TestImprovementAnalysisError:
    """ImprovementAnalysisError测试类"""
    
    def test_improvement_analysis_error_creation(self):
        """测试改进分析错误创建"""
        partial_suggestions = ["建议1", "建议2"]
        error = ImprovementAnalysisError(
            analysis_stage="suggestion_generation",
            reason="数据不足",
            partial_suggestions=partial_suggestions
        )
        
        assert "改进分析失败" in str(error)
        assert "suggestion_generation" in str(error)
        assert "数据不足" in str(error)
        assert error.analysis_stage == "suggestion_generation"
        assert error.reason == "数据不足"
        assert error.partial_suggestions == partial_suggestions
    
    def test_fallback_suggestions(self):
        """测试降级建议"""
        error = ImprovementAnalysisError(
            analysis_stage="test_stage",
            reason="测试原因",
            partial_suggestions=["部分建议"]
        )
        
        suggestions = error.get_fallback_suggestions()
        
        assert "部分建议" in suggestions
        assert any("人工评估" in s for s in suggestions)
        assert any("训练数据" in s for s in suggestions)
        assert any("模型架构" in s for s in suggestions)
    
    def test_diagnostic_steps(self):
        """测试诊断步骤"""
        error = ImprovementAnalysisError(
            analysis_stage="diagnosis",
            reason="分析失败"
        )
        
        steps = error.get_diagnostic_steps()
        
        assert any("评估数据的质量" in s for s in steps)
        assert any("评估指标的计算" in s for s in steps)
        assert any("评估结果的分布" in s for s in steps)
        assert any("历史评估结果" in s for s in steps)
        assert any("系统配置" in s for s in steps)


class TestExceptionHandling:
    """异常处理工具函数测试"""
    
    def test_handle_evaluation_exception(self):
        """测试异常处理函数"""
        # 测试ModelLoadError处理
        model_error = ModelLoadError("/test/model", "文件不存在")
        context = {"operation": "model_loading"}
        
        result = handle_evaluation_exception(model_error, context)
        
        assert result["handled"] is True
        assert result["recovery_action"] == "try_alternative_model"
        assert result["should_retry"] is True
        assert "error_details" in result
        
        # 测试EvaluationProcessError处理
        process_error = EvaluationProcessError("model_inference")
        result = handle_evaluation_exception(process_error, context)
        
        assert result["handled"] is True
        assert result["recovery_action"] == "使用缓存结果或简化评估"
        assert result["should_retry"] is False
        
        # 测试BatchProcessingError处理
        batch_error = BatchProcessingError(
            "batch_001", ["item_1"], ["item_2"], {}
        )
        result = handle_evaluation_exception(batch_error, context)
        
        assert result["handled"] is True
        assert result["recovery_action"] == "retry_failed_items"
        assert result["should_retry"] is True
        assert "fallback_result" in result
        
        # 测试TimeoutError处理
        timeout_error = TimeoutError(
            "test_op", 100.0, 150.0, {"partial": "data"}
        )
        result = handle_evaluation_exception(timeout_error, context)
        
        assert result["handled"] is True
        assert result["recovery_action"] == "use_partial_results"
        assert result["should_retry"] is False
        assert result["fallback_result"] == {"partial": "data"}
        
        # 测试非专家评估异常
        generic_error = ValueError("通用错误")
        result = handle_evaluation_exception(generic_error, context)
        
        assert result["handled"] is False
        assert result["recovery_action"] is None
    
    def test_create_error_recovery_plan(self):
        """测试错误恢复计划创建"""
        errors = [
            ModelLoadError("/test/model", "加载失败"),
            ResourceError("memory", "16GB", "8GB"),
            DataFormatError("/test/data", ["格式错误"]),
            CacheError("read", "test_key", "缓存失败")
        ]
        
        recovery_plan = create_error_recovery_plan(errors)
        
        assert recovery_plan["total_errors"] == 4
        assert "error_categories" in recovery_plan
        assert "recovery_steps" in recovery_plan
        assert "estimated_success_rate" in recovery_plan
        
        # 验证错误分类
        categories = recovery_plan["error_categories"]
        assert "ModelLoadError" in categories
        assert "ResourceError" in categories
        assert "DataFormatError" in categories
        assert "CacheError" in categories
        
        # 验证恢复步骤
        steps = recovery_plan["recovery_steps"]
        assert len(steps) > 0
        
        step_descriptions = [step["step"] for step in steps]
        assert any("模型加载问题" in desc for desc in step_descriptions)
        assert any("资源使用配置" in desc for desc in step_descriptions)
        assert any("数据格式问题" in desc for desc in step_descriptions)
        
        # 验证成功率估算
        success_rate = recovery_plan["estimated_success_rate"]
        assert 0.0 <= success_rate <= 1.0
    
    def test_empty_error_list(self):
        """测试空错误列表"""
        recovery_plan = create_error_recovery_plan([])
        
        assert recovery_plan["total_errors"] == 0
        assert recovery_plan["error_categories"] == {}
        assert recovery_plan["recovery_steps"] == []
        assert recovery_plan["estimated_success_rate"] == 0.0
    
    def test_single_error_recovery_plan(self):
        """测试单个错误的恢复计划"""
        errors = [ModelLoadError("/test/model", "测试错误")]
        recovery_plan = create_error_recovery_plan(errors)
        
        assert recovery_plan["total_errors"] == 1
        assert len(recovery_plan["error_categories"]) == 1
        assert "ModelLoadError" in recovery_plan["error_categories"]
        assert recovery_plan["error_categories"]["ModelLoadError"] == 1


# 异常继承测试
class TestExceptionInheritance:
    """异常继承关系测试"""
    
    def test_exception_hierarchy(self):
        """测试异常继承层次"""
        # 所有专家评估异常都应继承自ExpertEvaluationError
        model_error = ModelLoadError("/test", "测试")
        assert isinstance(model_error, ExpertEvaluationError)
        
        process_error = EvaluationProcessError("test_stage")
        assert isinstance(process_error, ExpertEvaluationError)
        
        data_error = DataFormatError("/test", ["错误"])
        assert isinstance(data_error, ExpertEvaluationError)
        
        config_error = ConfigurationError("test_section", ["param"])
        assert isinstance(config_error, ExpertEvaluationError)
        
        # 所有异常都应继承自Exception
        assert isinstance(model_error, Exception)
        assert isinstance(process_error, Exception)
        assert isinstance(data_error, Exception)
        assert isinstance(config_error, Exception)
    
    def test_exception_catching(self):
        """测试异常捕获"""
        # 测试捕获特定异常
        try:
            raise ModelLoadError("/test", "测试错误")
        except ModelLoadError as e:
            assert e.model_path == "/test"
        except Exception:
            pytest.fail("应该捕获ModelLoadError")
        
        # 测试捕获基类异常
        try:
            raise DataFormatError("/test", ["错误"])
        except ExpertEvaluationError as e:
            assert isinstance(e, DataFormatError)
        except Exception:
            pytest.fail("应该捕获ExpertEvaluationError")
        
        # 测试捕获所有异常
        try:
            raise ValidationError("test", ["item"])
        except Exception as e:
            assert isinstance(e, ValidationError)
            assert isinstance(e, ExpertEvaluationError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])