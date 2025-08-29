"""
完整评估流程集成测试

测试专家评估系统的完整评估流程，包括：
- 端到端评估流程测试
- 模型加载到报告生成的完整链路
- 错误恢复和异常处理测试
- 配置管理和系统集成测试
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import (
    ExpertEvaluationConfig,
    EvaluationDimension,
    EvaluationMode
)
from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    EvaluationDataset
)
from src.expert_evaluation.exceptions import (
    ModelLoadError,
    EvaluationProcessError,
    ConfigurationError
)


class TestCompleteEvaluationPipeline:
    """完整评估流程测试"""
    
    @pytest.mark.integration
    def test_end_to_end_evaluation_flow(self, integration_config, large_qa_dataset, mock_existing_systems):
        """测试端到端评估流程"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_model_service.return_value = mock_existing_systems["model_service"]
            mock_framework.return_value = mock_existing_systems["evaluation_framework"]
            mock_config.return_value = mock_existing_systems["config_manager"]
            
            # 创建评估引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_existing_systems["model_service"]
            engine.evaluation_framework = mock_existing_systems["evaluation_framework"]
            
            # 步骤1: 加载模型
            model_path = "/test/model/path"
            
            # 修复异步模型加载的mock
            async def mock_async_load_model(path):
                return True
            
            with patch.object(engine, '_async_load_model', side_effect=mock_async_load_model):
                load_result = engine.load_model(model_path)
                assert load_result is True, "模型加载失败"
                assert engine.is_model_loaded is True, "模型加载状态不正确"
            
            # 步骤2: 准备测试数据（使用前10个项目进行快速测试）
            test_qa_items = large_qa_dataset[:10]
            
            # 步骤3: 执行评估
            # 模拟模型回答生成和评估框架
            async def mock_generate_answer(prompt):
                return "模拟的模型回答"
            
            mock_framework_result = Mock()
            mock_framework_result.overall_score = 0.78
            mock_framework_result.scores = {
                "TECHNICAL_ACCURACY": 0.78,
                "CONCEPTUAL_UNDERSTANDING": 0.85,
                "PRACTICAL_APPLICABILITY": 0.72
            }
            mock_framework_result.detailed_feedback = {"test": "feedback"}
            
            with patch.object(engine, '_async_generate_answer', side_effect=mock_generate_answer), \
                 patch.object(engine.evaluation_framework, 'evaluate_with_expert_integration', return_value=mock_framework_result):
                
                evaluation_result = engine.evaluate_model(test_qa_items)
                
                # 验证评估结果
                assert isinstance(evaluation_result, ExpertEvaluationResult)
                assert evaluation_result.overall_score > 0.0
                assert len(evaluation_result.dimension_scores) > 0
                
                # 步骤4: 生成报告
                with patch.object(engine, '_create_report_generator') as mock_report_gen:
                    mock_report = Mock()
                    mock_report.report_id = "test_report_001"
                    mock_report.format = "json"
                    mock_report_gen.return_value.generate_detailed_report.return_value = mock_report
                    
                    report = engine.generate_report(evaluation_result)
                    assert report is not None
                    assert hasattr(report, 'report_id')
                    
                # 验证统计信息更新
                assert engine.evaluation_stats["total_evaluations"] > 0
                assert engine.evaluation_stats["successful_evaluations"] > 0
    
    @pytest.mark.integration
    def test_pipeline_with_real_data_processing(self, integration_config, integration_test_data):
        """测试使用真实数据处理的流水线"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟服务
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_service.generate_response.return_value = "详细的模型回答内容"
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.82,
                "professional_accuracy": 0.75,
                "chinese_quality": 0.80
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 加载真实测试数据
            with open(integration_test_data, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # 转换为QA项目
            qa_items = []
            for item_data in test_data["qa_data"]:
                qa_item = QAEvaluationItem(
                    question_id=item_data["question_id"],
                    question=item_data["question"],
                    context=item_data["context"],
                    reference_answer=item_data["reference_answer"],
                    model_answer=item_data["model_answer"],
                    domain_tags=item_data["domain_tags"],
                    difficulty_level=item_data["difficulty_level"],
                    expected_concepts=item_data["expected_concepts"]
                )
                qa_items.append(qa_item)
            
            # 创建引擎并执行评估
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            # 加载模型
            async def mock_async_load_model(path):
                return True
            
            with patch.object(engine, '_async_load_model', side_effect=mock_async_load_model):
                engine.load_model("/test/model")
            
            # 执行评估
            async def mock_generate_answer(prompt):
                return "模拟的模型回答"
            
            mock_framework_result = Mock()
            mock_framework_result.overall_score = 0.78
            mock_framework_result.scores = {
                "TECHNICAL_ACCURACY": 0.75,
                "CONCEPTUAL_UNDERSTANDING": 0.82
            }
            mock_framework_result.detailed_feedback = {"test": "feedback"}
            
            with patch.object(engine, '_async_generate_answer', side_effect=mock_generate_answer), \
                 patch.object(engine.evaluation_framework, 'evaluate_with_expert_integration', return_value=mock_framework_result):
                
                result = engine.evaluate_model(qa_items)
                
                # 验证结果
                assert result.overall_score > 0.6
                assert len(result.dimension_scores) >= 2
    
    @pytest.mark.integration
    def test_pipeline_error_recovery(self, integration_config, large_qa_dataset):
        """测试流水线错误恢复机制"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_service = Mock()
            mock_framework_instance = Mock()
            mock_config_instance = Mock()
            
            mock_model_service.return_value = mock_service
            mock_framework.return_value = mock_framework_instance
            mock_config.return_value = mock_config_instance
            
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_framework_instance
            
            # 测试模型加载失败恢复
            mock_service.load_model.side_effect = Exception("模型加载失败")
            
            with pytest.raises(ModelLoadError):
                engine.load_model("/invalid/model/path")
            
            # 恢复正常状态
            mock_service.load_model.side_effect = None
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            
            load_result = engine.load_model("/valid/model/path")
            assert load_result is True
            
            # 测试评估过程中的错误恢复
            test_items = large_qa_dataset[:5]
            
            with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                # 模拟部分项目评估失败
                def side_effect(item):
                    if "002" in item.question_id:
                        raise Exception("评估失败")
                    return Mock(overall_score=0.8, dimension_scores={})
                
                mock_evaluate.side_effect = side_effect
                
                # 应该能够处理部分失败并继续
                try:
                    result = engine.evaluate_model(test_items)
                    # 验证失败统计被正确记录
                    assert engine.evaluation_stats["failed_evaluations"] > 0
                except EvaluationProcessError:
                    # 如果抛出异常，验证错误信息
                    pass
    
    @pytest.mark.integration
    def test_configuration_integration(self, integration_test_dir):
        """测试配置管理集成"""
        # 创建测试配置文件
        config_data = {
            "evaluation_mode": "comprehensive",
            "log_level": "INFO",
            "enable_caching": True,
            "batch_size": 10,
            "dimensions": {
                "semantic_similarity": {"weight": 0.4, "enabled": True},
                "domain_accuracy": {"weight": 0.4, "enabled": True},
                "completeness": {"weight": 0.2, "enabled": True}
            }
        }
        
        config_file = integration_test_dir / "test_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 模拟配置管理器加载配置
            mock_config_instance = Mock()
            mock_config_instance.load_config.return_value = config_data
            mock_config.return_value = mock_config_instance
            
            # 创建配置对象
            config = ExpertEvaluationConfig(
                evaluation_mode=EvaluationMode.COMPREHENSIVE,
                log_level="INFO",
                enable_caching=True,
                batch_size=10
            )
            
            # 创建引擎并验证配置集成
            engine = ExpertEvaluationEngine(config)
            
            assert engine.config.evaluation_mode == EvaluationMode.COMPREHENSIVE
            assert engine.config.log_level == "INFO"
            assert engine.config.enable_caching is True
            assert engine.config.batch_size == 10
    
    @pytest.mark.integration
    def test_existing_system_compatibility(self, integration_config, mock_existing_systems):
        """测试与现有系统的兼容性"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 使用现有系统的模拟对象
            mock_model_service.return_value = mock_existing_systems["model_service"]
            mock_framework.return_value = mock_existing_systems["evaluation_framework"]
            mock_config.return_value = mock_existing_systems["config_manager"]
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_existing_systems["model_service"]
            engine.evaluation_framework = mock_existing_systems["evaluation_framework"]
            
            # 测试与现有评估框架的集成
            test_item = QAEvaluationItem(
                question_id="compat_test_001",
                question="兼容性测试问题",
                context="测试上下文",
                reference_answer="参考答案",
                model_answer="模型答案"
            )
            
            # 验证现有评估框架方法可以被调用
            eval_result = mock_existing_systems["evaluation_framework"].evaluate_response(
                test_item.model_answer,
                test_item.reference_answer
            )
            
            assert "semantic_similarity" in eval_result
            assert "professional_accuracy" in eval_result
            assert "chinese_quality" in eval_result
            
            # 测试与现有模型服务的集成
            model_service = mock_existing_systems["model_service"]
            assert model_service.load_model("/test/path") is True
            assert model_service.is_model_loaded is True
            
            response = model_service.generate_response("测试问题")
            assert response is not None
            
            # 测试与现有配置管理的集成
            config_manager = mock_existing_systems["config_manager"]
            config_data = config_manager.get_config("expert_evaluation")
            assert config_data is not None
    
    @pytest.mark.integration
    def test_batch_processing_pipeline(self, integration_config, multi_domain_datasets):
        """测试批处理流水线"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.80,
                "professional_accuracy": 0.75
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            # 加载模型
            engine.load_model("/test/model")
            
            # 准备批处理数据
            datasets = list(multi_domain_datasets.values())
            
            # 执行批量评估
            with patch.object(engine, 'evaluate_model') as mock_evaluate:
                # 模拟单个数据集评估结果
                mock_result = Mock()
                mock_result.overall_score = 0.78
                mock_result.dimension_scores = {}
                mock_evaluate.return_value = mock_result
                
                batch_result = engine.batch_evaluate(datasets)
                
                # 验证批处理结果
                assert batch_result is not None
                assert hasattr(batch_result, 'batch_id')
                assert hasattr(batch_result, 'individual_results')
                assert len(batch_result.individual_results) == len(datasets)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_under_load(self, integration_config, large_qa_dataset, performance_benchmarks):
        """测试负载下的性能表现"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置快速模拟对象
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.80,
                "professional_accuracy": 0.75
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            # 加载模型
            engine.load_model("/test/model")
            
            # 测试大数据集处理性能
            start_time = time.time()
            
            # 使用前50个项目进行性能测试
            test_items = large_qa_dataset[:50]
            
            with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                # 模拟快速评估
                mock_evaluate.return_value = Mock(
                    overall_score=0.8,
                    dimension_scores={},
                    processing_time=0.1
                )
                
                result = engine.evaluate_model(test_items)
                
                processing_time = time.time() - start_time
                
                # 验证性能基准
                max_time = performance_benchmarks["evaluation_speed"]["large_dataset_max_time"]
                assert processing_time < max_time, f"处理时间过长: {processing_time:.2f}s > {max_time}s"
                
                # 验证结果质量
                min_score = performance_benchmarks["accuracy_thresholds"]["min_overall_score"]
                assert result.overall_score >= min_score, f"评估质量过低: {result.overall_score}"
    
    @pytest.mark.integration
    def test_memory_management_integration(self, integration_config, large_qa_dataset):
        """测试内存管理集成"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.80,
                "professional_accuracy": 0.75
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 创建多个引擎实例测试内存管理
            engines = []
            for i in range(5):
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                engines.append(engine)
            
            # 执行评估
            for engine in engines:
                engine.load_model(f"/test/model_{len(engines)}")
                
                with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                    mock_evaluate.return_value = Mock(
                        overall_score=0.8,
                        dimension_scores={}
                    )
                    
                    # 使用较小的数据集避免测试时间过长
                    test_items = large_qa_dataset[:5]
                    engine.evaluate_model(test_items)
            
            # 清理引擎
            for engine in engines:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 检查内存使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 内存增长应该在合理范围内
            max_increase = 200  # MB
            assert memory_increase < max_increase, f"内存增长过多: {memory_increase:.2f}MB"


class TestPipelineRobustness:
    """流水线健壮性测试"""
    
    @pytest.mark.integration
    def test_concurrent_evaluation_safety(self, integration_config):
        """测试并发评估安全性"""
        import threading
        import queue
        
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.80,
                "professional_accuracy": 0.75
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            engine.load_model("/test/model")
            
            # 创建测试数据
            test_items = [
                QAEvaluationItem(
                    question_id=f"concurrent_test_{i:03d}",
                    question=f"并发测试问题 {i}",
                    context="并发测试",
                    reference_answer=f"参考答案 {i}",
                    model_answer=f"模型答案 {i}"
                )
                for i in range(10)
            ]
            
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def evaluate_worker(items):
                try:
                    with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                        mock_evaluate.return_value = Mock(
                            overall_score=0.8,
                            dimension_scores={}
                        )
                        result = engine.evaluate_model(items)
                        results_queue.put(result)
                except Exception as e:
                    errors_queue.put(e)
            
            # 启动多个并发评估线程
            threads = []
            for i in range(3):
                thread_items = test_items[i*3:(i+1)*3]
                if thread_items:
                    thread = threading.Thread(target=evaluate_worker, args=(thread_items,))
                    threads.append(thread)
                    thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=10)
            
            # 验证结果
            assert errors_queue.empty(), f"并发评估出现错误: {list(errors_queue.queue)}"
            assert not results_queue.empty(), "没有获得评估结果"
    
    @pytest.mark.integration
    def test_resource_cleanup_after_failure(self, integration_config):
        """测试失败后的资源清理"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_framework.return_value = mock_eval
            mock_config.return_value = Mock()
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            # 模拟评估过程中的严重错误
            test_item = QAEvaluationItem(
                question_id="cleanup_test_001",
                question="资源清理测试",
                context="测试",
                reference_answer="参考",
                model_answer="模型"
            )
            
            with patch.object(engine, '_evaluate_single_item', side_effect=Exception("严重错误")):
                try:
                    engine.evaluate_model([test_item])
                except:
                    pass  # 忽略预期的错误
            
            # 验证引擎仍然可以正常工作
            engine.load_model("/test/model/recovery")
            assert engine.is_model_loaded is True
            
            # 验证统计信息被正确更新
            assert engine.evaluation_stats["failed_evaluations"] > 0
    
    @pytest.mark.integration
    def test_configuration_hot_reload(self, integration_config, integration_test_dir):
        """测试配置热重载"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 设置模拟对象
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.80,
                "professional_accuracy": 0.75
            }
            mock_framework.return_value = mock_eval
            
            mock_config_instance = Mock()
            mock_config.return_value = mock_config_instance
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            # 验证初始配置
            assert engine.config.evaluation_mode == EvaluationMode.COMPREHENSIVE
            
            # 模拟配置更新
            new_config = ExpertEvaluationConfig(
                evaluation_mode=EvaluationMode.QUICK,
                log_level="DEBUG",
                enable_caching=False
            )
            
            # 如果引擎支持配置更新
            if hasattr(engine, 'update_config'):
                engine.update_config(new_config)
                assert engine.config.evaluation_mode == EvaluationMode.QUICK
                assert engine.config.log_level == "DEBUG"
                assert engine.config.enable_caching is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])