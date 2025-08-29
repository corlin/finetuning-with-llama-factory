"""
接口模块单元测试

测试专家评估系统的抽象接口定义，确保接口设计的正确性。
"""

import pytest
from abc import ABC
from typing import List, Dict, Any

from src.expert_evaluation.interfaces import (
    ExpertEvaluationEngine,
    IndustryMetricsCalculator,
    AdvancedSemanticEvaluator,
    EvaluationDataManager,
    MultiDimensionalEvaluator,
    EvaluationReportGenerator,
    BatchProcessingInterface,
    PerformanceMonitorInterface,
    ErrorHandlingInterface
)
from src.expert_evaluation.config import ExpertEvaluationConfig, EvaluationDimension
from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    BatchEvaluationResult,
    EvaluationReport,
    ValidationResult,
    EvaluationDataset
)


class TestAbstractInterfaces:
    """抽象接口测试"""
    
    def test_expert_evaluation_engine_interface(self):
        """测试专家评估引擎接口"""
        # 验证接口是抽象类
        assert issubclass(ExpertEvaluationEngine, ABC)
        
        # 验证抽象方法存在
        abstract_methods = ExpertEvaluationEngine.__abstractmethods__
        expected_methods = {
            '__init__',
            'load_model',
            'evaluate_model',
            'batch_evaluate',
            'generate_report'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_industry_metrics_calculator_interface(self):
        """测试行业指标计算器接口"""
        assert issubclass(IndustryMetricsCalculator, ABC)
        
        abstract_methods = IndustryMetricsCalculator.__abstractmethods__
        expected_methods = {
            'calculate_domain_relevance',
            'assess_practical_applicability',
            'evaluate_innovation_level',
            'measure_completeness'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_advanced_semantic_evaluator_interface(self):
        """测试高级语义评估器接口"""
        assert issubclass(AdvancedSemanticEvaluator, ABC)
        
        abstract_methods = AdvancedSemanticEvaluator.__abstractmethods__
        expected_methods = {
            'calculate_semantic_depth',
            'assess_logical_consistency',
            'evaluate_contextual_understanding',
            'measure_concept_coverage'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_evaluation_data_manager_interface(self):
        """测试评估数据管理器接口"""
        assert issubclass(EvaluationDataManager, ABC)
        
        abstract_methods = EvaluationDataManager.__abstractmethods__
        expected_methods = {
            'load_qa_data',
            'validate_data_format',
            'prepare_evaluation_dataset',
            'export_results'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_multi_dimensional_evaluator_interface(self):
        """测试多维度评估协调器接口"""
        assert issubclass(MultiDimensionalEvaluator, ABC)
        
        abstract_methods = MultiDimensionalEvaluator.__abstractmethods__
        expected_methods = {
            'integrate_evaluation_dimensions',
            'calculate_weighted_score',
            'analyze_evaluation_results'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_evaluation_report_generator_interface(self):
        """测试评估报告生成器接口"""
        assert issubclass(EvaluationReportGenerator, ABC)
        
        abstract_methods = EvaluationReportGenerator.__abstractmethods__
        expected_methods = {
            'generate_detailed_report',
            'create_visualization_charts',
            'generate_improvement_suggestions',
            'export_report'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_batch_processing_interface(self):
        """测试批量处理接口"""
        assert issubclass(BatchProcessingInterface, ABC)
        
        abstract_methods = BatchProcessingInterface.__abstractmethods__
        expected_methods = {
            'process_batch',
            'optimize_batch_size'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_performance_monitor_interface(self):
        """测试性能监控接口"""
        assert issubclass(PerformanceMonitorInterface, ABC)
        
        abstract_methods = PerformanceMonitorInterface.__abstractmethods__
        expected_methods = {
            'start_monitoring',
            'stop_monitoring',
            'get_memory_usage',
            'get_processing_speed'
        }
        
        assert expected_methods.issubset(abstract_methods)
    
    def test_error_handling_interface(self):
        """测试错误处理接口"""
        assert issubclass(ErrorHandlingInterface, ABC)
        
        abstract_methods = ErrorHandlingInterface.__abstractmethods__
        expected_methods = {
            'handle_evaluation_error',
            'implement_fallback_strategy',
            'log_error_details'
        }
        
        assert expected_methods.issubset(abstract_methods)


class TestInterfaceMethodSignatures:
    """接口方法签名测试"""
    
    def test_expert_evaluation_engine_signatures(self):
        """测试专家评估引擎方法签名"""
        # 获取方法签名
        import inspect
        
        # 测试load_model方法签名
        load_model_sig = inspect.signature(ExpertEvaluationEngine.load_model)
        params = list(load_model_sig.parameters.keys())
        assert 'self' in params
        assert 'model_path' in params
        assert load_model_sig.return_annotation == bool
        
        # 测试evaluate_model方法签名
        evaluate_model_sig = inspect.signature(ExpertEvaluationEngine.evaluate_model)
        params = list(evaluate_model_sig.parameters.keys())
        assert 'self' in params
        assert 'qa_data' in params
        assert evaluate_model_sig.return_annotation == ExpertEvaluationResult
    
    def test_industry_metrics_calculator_signatures(self):
        """测试行业指标计算器方法签名"""
        import inspect
        
        # 测试calculate_domain_relevance方法签名
        method_sig = inspect.signature(IndustryMetricsCalculator.calculate_domain_relevance)
        params = list(method_sig.parameters.keys())
        assert 'self' in params
        assert 'answer' in params
        assert 'domain_context' in params
        assert method_sig.return_annotation == float
        
        # 测试assess_practical_applicability方法签名
        method_sig = inspect.signature(IndustryMetricsCalculator.assess_practical_applicability)
        params = list(method_sig.parameters.keys())
        assert 'self' in params
        assert 'answer' in params
        assert 'use_case' in params
        assert method_sig.return_annotation == float
    
    def test_advanced_semantic_evaluator_signatures(self):
        """测试高级语义评估器方法签名"""
        import inspect
        
        # 测试calculate_semantic_depth方法签名
        method_sig = inspect.signature(AdvancedSemanticEvaluator.calculate_semantic_depth)
        params = list(method_sig.parameters.keys())
        assert 'self' in params
        assert 'answer' in params
        assert 'reference' in params
        assert method_sig.return_annotation == float
        
        # 测试assess_logical_consistency方法签名
        method_sig = inspect.signature(AdvancedSemanticEvaluator.assess_logical_consistency)
        params = list(method_sig.parameters.keys())
        assert 'self' in params
        assert 'answer' in params
        assert method_sig.return_annotation == float


class TestInterfaceDocumentation:
    """接口文档测试"""
    
    def test_interface_docstrings(self):
        """测试接口文档字符串"""
        # 验证主要接口都有文档字符串
        assert ExpertEvaluationEngine.__doc__ is not None
        assert "专家评估引擎抽象基类" in ExpertEvaluationEngine.__doc__
        
        assert IndustryMetricsCalculator.__doc__ is not None
        assert "行业指标计算器抽象基类" in IndustryMetricsCalculator.__doc__
        
        assert AdvancedSemanticEvaluator.__doc__ is not None
        assert "高级语义评估器抽象基类" in AdvancedSemanticEvaluator.__doc__
    
    def test_method_docstrings(self):
        """测试方法文档字符串"""
        # 验证关键方法都有文档字符串
        assert ExpertEvaluationEngine.load_model.__doc__ is not None
        assert "加载已合并的最终模型" in ExpertEvaluationEngine.load_model.__doc__
        
        assert ExpertEvaluationEngine.evaluate_model.__doc__ is not None
        assert "执行单个QA数据集的评估" in ExpertEvaluationEngine.evaluate_model.__doc__
        
        assert IndustryMetricsCalculator.calculate_domain_relevance.__doc__ is not None
        assert "计算答案与特定领域的相关性" in IndustryMetricsCalculator.calculate_domain_relevance.__doc__


class MockExpertEvaluationEngine(ExpertEvaluationEngine):
    """用于测试的模拟专家评估引擎实现"""
    
    def __init__(self, config: ExpertEvaluationConfig):
        self.config = config
        self.is_loaded = False
    
    def load_model(self, model_path: str) -> bool:
        self.is_loaded = True
        return True
    
    def evaluate_model(self, qa_data: List[QAEvaluationItem]) -> ExpertEvaluationResult:
        from src.expert_evaluation.data_models import DimensionScore
        
        dimension_scores = {
            EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=0.8,
                confidence=0.9
            )
        }
        
        return ExpertEvaluationResult(
            question_id=qa_data[0].question_id if qa_data else "test",
            overall_score=0.8,
            dimension_scores=dimension_scores
        )
    
    def batch_evaluate(self, qa_datasets: List[EvaluationDataset]) -> BatchEvaluationResult:
        individual_results = []
        for dataset in qa_datasets:
            result = self.evaluate_model(dataset.qa_items)
            individual_results.append(result)
        
        return BatchEvaluationResult(
            batch_id="test_batch",
            individual_results=individual_results
        )
    
    def generate_report(self, results: ExpertEvaluationResult) -> EvaluationReport:
        return EvaluationReport(
            report_id="test_report",
            title="测试报告",
            summary="测试摘要",
            evaluation_results=results
        )


class TestConcreteImplementation:
    """具体实现测试"""
    
    def test_mock_implementation_instantiation(self):
        """测试模拟实现实例化"""
        config = ExpertEvaluationConfig()
        engine = MockExpertEvaluationEngine(config)
        
        assert engine.config == config
        assert engine.is_loaded is False
    
    def test_mock_implementation_methods(self):
        """测试模拟实现方法"""
        config = ExpertEvaluationConfig()
        engine = MockExpertEvaluationEngine(config)
        
        # 测试模型加载
        result = engine.load_model("/test/model")
        assert result is True
        assert engine.is_loaded is True
        
        # 测试评估
        qa_item = QAEvaluationItem(
            question_id="test_001",
            question="测试问题",
            context="测试上下文",
            reference_answer="参考答案",
            model_answer="模型答案"
        )
        
        evaluation_result = engine.evaluate_model([qa_item])
        assert isinstance(evaluation_result, ExpertEvaluationResult)
        assert evaluation_result.question_id == "test_001"
        assert evaluation_result.overall_score == 0.8
        
        # 测试报告生成
        report = engine.generate_report(evaluation_result)
        assert isinstance(report, EvaluationReport)
        assert report.report_id == "test_report"
        assert report.title == "测试报告"
    
    def test_interface_compliance(self):
        """测试接口合规性"""
        config = ExpertEvaluationConfig()
        engine = MockExpertEvaluationEngine(config)
        
        # 验证实现了所有必需的方法
        required_methods = [
            'load_model',
            'evaluate_model', 
            'batch_evaluate',
            'generate_report'
        ]
        
        for method_name in required_methods:
            assert hasattr(engine, method_name)
            assert callable(getattr(engine, method_name))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])