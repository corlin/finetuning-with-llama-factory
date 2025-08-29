"""
基础集成测试

验证专家评估系统的基本集成功能。
"""

import pytest
from unittest.mock import Mock, patch
from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import ExpertEvaluationConfig, EvaluationMode
from src.expert_evaluation.data_models import QAEvaluationItem, ExpertiseLevel
from .conftest import setup_engine_mocks, mock_engine_evaluation


class TestBasicIntegration:
    """基础集成测试"""
    
    @pytest.mark.integration
    def test_engine_initialization(self, integration_config):
        """测试引擎初始化"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 验证基本属性
            assert engine.config == integration_config
            assert engine.is_model_loaded is False
            assert engine.evaluation_stats["total_evaluations"] == 0
            assert engine.logger is not None
    
    @pytest.mark.integration
    def test_model_loading(self, integration_config):
        """测试模型加载"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 模拟异步加载
            async def mock_async_load_model(path):
                return True
            
            with patch.object(engine, '_async_load_model', side_effect=mock_async_load_model):
                result = engine.load_model("/test/model")
                
                assert result is True
                assert engine.is_model_loaded is True
    
    @pytest.mark.integration
    def test_basic_evaluation(self, integration_config):
        """测试基本评估功能"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 创建测试数据
            qa_item = QAEvaluationItem(
                question_id="basic_test_001",
                question="什么是RSA加密算法？",
                context="密码学基础",
                reference_answer="RSA是一种非对称加密算法",
                model_answer="RSA算法是公钥加密算法",
                domain_tags=["密码学"],
                difficulty_level=ExpertiseLevel.INTERMEDIATE
            )
            
            # 执行评估
            result = mock_engine_evaluation(engine, [qa_item])
            
            # 验证结果
            assert result is not None
            assert hasattr(result, 'overall_score')
            assert result.overall_score > 0.0
            assert hasattr(result, 'dimension_scores')
    
    @pytest.mark.integration
    def test_batch_evaluation(self, integration_config):
        """测试批量评估"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 创建测试数据集
            from src.expert_evaluation.data_models import EvaluationDataset
            
            qa_items = []
            for i in range(3):
                item = QAEvaluationItem(
                    question_id=f"batch_test_{i:03d}",
                    question=f"测试问题 {i}",
                    context="测试上下文",
                    reference_answer=f"参考答案 {i}",
                    model_answer=f"模型答案 {i}",
                    domain_tags=["测试"],
                    difficulty_level=ExpertiseLevel.BEGINNER
                )
                qa_items.append(item)
            
            dataset = EvaluationDataset(
                dataset_id="test_dataset",
                name="测试数据集",
                description="用于测试的数据集",
                qa_items=qa_items
            )
            
            # 设置mock
            mocks = setup_engine_mocks(engine)
            
            with patch.object(engine, '_async_load_model', side_effect=mocks['async_load_model']), \
                 patch.object(engine, '_async_generate_answer', side_effect=mocks['generate_answer']), \
                 patch.object(engine.evaluation_framework, 'evaluate_with_expert_integration', return_value=mocks['framework_result']):
                
                # 加载模型
                engine.load_model("/test/model")
                
                # 执行批量评估
                batch_result = engine.batch_evaluate([dataset])
                
                # 验证结果
                assert batch_result is not None
                assert hasattr(batch_result, 'batch_id')
                assert hasattr(batch_result, 'individual_results')
                assert len(batch_result.individual_results) == 1
    
    @pytest.mark.integration
    def test_report_generation(self, integration_config):
        """测试报告生成"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 创建测试数据
            qa_item = QAEvaluationItem(
                question_id="report_test_001",
                question="测试报告生成",
                context="测试",
                reference_answer="参考答案",
                model_answer="模型答案"
            )
            
            # 执行评估
            evaluation_result = mock_engine_evaluation(engine, [qa_item])
            
            # 生成报告
            report = engine.generate_report(evaluation_result)
            
            # 验证报告
            assert report is not None
            assert hasattr(report, 'report_id')
            assert hasattr(report, 'title')
            assert hasattr(report, 'summary')
    
    @pytest.mark.integration
    def test_error_handling(self, integration_config):
        """测试错误处理"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 测试未加载模型时的评估
            qa_item = QAEvaluationItem(
                question_id="error_test_001",
                question="错误测试",
                context="测试",
                reference_answer="参考答案",
                model_answer="模型答案"
            )
            
            from src.expert_evaluation.exceptions import EvaluationProcessError
            
            with pytest.raises(EvaluationProcessError, match="模型未加载"):
                engine.evaluate_model([qa_item])
            
            # 测试空数据评估
            mocks = setup_engine_mocks(engine)
            
            with patch.object(engine, '_async_load_model', side_effect=mocks['async_load_model']):
                engine.load_model("/test/model")
            
            from src.expert_evaluation.exceptions import DataFormatError
            
            with pytest.raises(DataFormatError, match="QA数据不能为空"):
                engine.evaluate_model([])
    
    @pytest.mark.integration
    def test_statistics_tracking(self, integration_config):
        """测试统计信息跟踪"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 初始统计
            stats = engine.get_evaluation_stats()
            assert stats["total_evaluations"] == 0
            assert stats["successful_evaluations"] == 0
            
            # 执行评估
            qa_item = QAEvaluationItem(
                question_id="stats_test_001",
                question="统计测试",
                context="测试",
                reference_answer="参考答案",
                model_answer="模型答案"
            )
            
            mock_engine_evaluation(engine, [qa_item])
            
            # 检查统计更新
            updated_stats = engine.get_evaluation_stats()
            assert updated_stats["total_evaluations"] > 0
            assert updated_stats["successful_evaluations"] > 0
    
    @pytest.mark.integration
    def test_engine_ready_status(self, integration_config):
        """测试引擎就绪状态"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 初始状态
            assert engine.is_ready() is False
            
            # 加载模型后
            mocks = setup_engine_mocks(engine)
            
            with patch.object(engine, '_async_load_model', side_effect=mocks['async_load_model']):
                engine.load_model("/test/model")
            
            assert engine.is_ready() is True
    
    @pytest.mark.integration
    def test_model_info(self, integration_config):
        """测试模型信息获取"""
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(integration_config)
            
            # 未加载模型时
            info = engine.get_model_info()
            assert info["status"] == "model_not_loaded"
            
            # 加载模型后
            mocks = setup_engine_mocks(engine)
            
            with patch.object(engine, '_async_load_model', side_effect=mocks['async_load_model']):
                engine.load_model("/test/model")
            
            info = engine.get_model_info()
            assert info["status"] == "loaded"
            assert "model_path" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])