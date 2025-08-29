"""
多模型比较测试

测试专家评估系统的多模型比较功能，包括：
- 多个模型的并行评估
- 模型性能对比分析
- 相对评估和排名
- 统计显著性测试
"""

import pytest
import time
import statistics
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import (
    ExpertEvaluationConfig,
    EvaluationDimension,
    EvaluationMode
)
from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    BatchEvaluationResult,
    EvaluationDataset,
    DimensionScore
)


class TestMultiModelComparison:
    """多模型比较测试"""
    
    @pytest.mark.integration
    def test_parallel_model_evaluation(self, integration_config, multi_domain_datasets, mock_model_comparison_data):
        """测试并行模型评估"""
        models = mock_model_comparison_data
        evaluation_results = {}
        
        # 为每个模型创建评估引擎
        for model_id, model_info in models.items():
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置模型特定的模拟对象
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_service.model_path = model_info["path"]
                mock_service.get_model_info.return_value = {
                    "name": model_info["name"],
                    "path": model_info["path"]
                }
                mock_model_service.return_value = mock_service
                
                # 模拟不同模型的性能差异
                expected_perf = model_info["expected_performance"]
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": expected_perf["semantic_similarity"],
                    "professional_accuracy": expected_perf["domain_accuracy"],
                    "chinese_quality": 0.80
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎并评估
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                # 加载模型
                engine.load_model(model_info["path"])
                
                # 使用密码学数据集进行评估
                crypto_dataset = multi_domain_datasets["cryptography"]
                test_items = crypto_dataset.qa_items[:10]  # 使用前10个项目
                
                with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                    mock_eval_instance = Mock()
                    mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                        EvaluationDimension.SEMANTIC_SIMILARITY: expected_perf["semantic_similarity"],
                        EvaluationDimension.DOMAIN_ACCURACY: expected_perf["domain_accuracy"],
                        EvaluationDimension.COMPLETENESS: expected_perf["completeness"]
                    }
                    mock_eval_instance.calculate_weighted_score.return_value = sum(expected_perf.values()) / len(expected_perf)
                    mock_evaluator.return_value = mock_eval_instance
                    
                    result = engine.evaluate_model(test_items)
                    evaluation_results[model_id] = {
                        "result": result,
                        "model_info": model_info,
                        "engine": engine
                    }
        
        # 验证所有模型都被成功评估
        assert len(evaluation_results) == len(models)
        
        # 验证评估结果的差异性
        scores = [res["result"].overall_score for res in evaluation_results.values()]
        assert len(set(scores)) > 1, "模型评估结果应该有差异"
        
        # 验证最佳模型
        best_model = max(evaluation_results.items(), key=lambda x: x[1]["result"].overall_score)
        assert best_model[0] in models, "最佳模型应该在测试模型列表中"
    
    @pytest.mark.integration
    def test_model_performance_comparison(self, integration_config, large_qa_dataset, mock_model_comparison_data):
        """测试模型性能对比分析"""
        models = mock_model_comparison_data
        comparison_results = {}
        
        # 创建标准化的测试数据
        test_items = large_qa_dataset[:20]  # 使用20个项目进行比较
        
        for model_id, model_info in models.items():
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置模拟对象
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                expected_perf = model_info["expected_performance"]
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": expected_perf["semantic_similarity"],
                    "professional_accuracy": expected_perf["domain_accuracy"]
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                engine.load_model(model_info["path"])
                
                # 记录评估时间
                start_time = time.time()
                
                with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                    mock_eval_instance = Mock()
                    
                    # 为每个维度创建详细的评分
                    dimension_scores = {}
                    for dim in EvaluationDimension:
                        if dim.value in ["语义相似性", "领域准确性", "完整性"]:
                            base_score = expected_perf.get("semantic_similarity", 0.75)
                            # 添加一些随机变化
                            variation = (hash(f"{model_id}_{dim.value}") % 100) / 1000  # -0.05 to 0.05
                            score = max(0.0, min(1.0, base_score + variation))
                            
                            dimension_scores[dim] = DimensionScore(
                                dimension=dim,
                                score=score,
                                confidence=0.85 + (hash(f"{model_id}_{dim.value}") % 20) / 100,
                                details={"model": model_id, "method": "mock_evaluation"}
                            )
                    
                    mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                        dim: score.score for dim, score in dimension_scores.items()
                    }
                    mock_eval_instance.calculate_weighted_score.return_value = sum(
                        score.score for score in dimension_scores.values()
                    ) / len(dimension_scores)
                    mock_evaluator.return_value = mock_eval_instance
                    
                    result = engine.evaluate_model(test_items)
                    result.dimension_scores = dimension_scores  # 设置详细的维度评分
                
                evaluation_time = time.time() - start_time
                
                comparison_results[model_id] = {
                    "result": result,
                    "evaluation_time": evaluation_time,
                    "model_info": model_info,
                    "throughput": len(test_items) / evaluation_time if evaluation_time > 0 else 0
                }
        
        # 分析比较结果
        analysis = self._analyze_model_comparison(comparison_results)
        
        # 验证分析结果
        assert "ranking" in analysis
        assert "dimension_comparison" in analysis
        assert "performance_metrics" in analysis
        
        # 验证排名的合理性
        ranking = analysis["ranking"]
        assert len(ranking) == len(models)
        
        # 验证维度比较
        dimension_comparison = analysis["dimension_comparison"]
        for dim in EvaluationDimension:
            if dim in dimension_comparison:
                dim_scores = dimension_comparison[dim]
                assert len(dim_scores) == len(models)
    
    @pytest.mark.integration
    def test_statistical_significance_analysis(self, integration_config, large_qa_dataset, mock_model_comparison_data):
        """测试统计显著性分析"""
        models = list(mock_model_comparison_data.keys())[:2]  # 使用两个模型进行比较
        model_results = {}
        
        # 为每个模型生成多次评估结果以进行统计分析
        for model_id in models:
            model_info = mock_model_comparison_data[model_id]
            multiple_results = []
            
            for run_id in range(5):  # 每个模型运行5次
                with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                     patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                     patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                    
                    # 设置模拟对象
                    mock_service = Mock()
                    mock_service.load_model.return_value = True
                    mock_service.is_model_loaded = True
                    mock_model_service.return_value = mock_service
                    
                    expected_perf = model_info["expected_performance"]
                    mock_eval = Mock()
                    mock_eval.evaluate_response.return_value = {
                        "semantic_similarity": expected_perf["semantic_similarity"],
                        "professional_accuracy": expected_perf["domain_accuracy"]
                    }
                    mock_framework.return_value = mock_eval
                    
                    mock_config.return_value = Mock()
                    
                    # 创建引擎
                    engine = ExpertEvaluationEngine(integration_config)
                    engine.model_service = mock_service
                    engine.evaluation_framework = mock_eval
                    
                    engine.load_model(model_info["path"])
                    
                    # 使用相同的测试数据
                    test_items = large_qa_dataset[:10]
                    
                    with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                        mock_eval_instance = Mock()
                        
                        # 添加一些随机变化来模拟真实的评估变异
                        base_score = expected_perf["semantic_similarity"]
                        variation = (hash(f"{model_id}_{run_id}") % 100 - 50) / 1000  # -0.05 to 0.05
                        final_score = max(0.0, min(1.0, base_score + variation))
                        
                        mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                            EvaluationDimension.SEMANTIC_SIMILARITY: final_score,
                            EvaluationDimension.DOMAIN_ACCURACY: expected_perf["domain_accuracy"] + variation
                        }
                        mock_eval_instance.calculate_weighted_score.return_value = final_score
                        mock_evaluator.return_value = mock_eval_instance
                        
                        result = engine.evaluate_model(test_items)
                        multiple_results.append(result)
            
            model_results[model_id] = multiple_results
        
        # 进行统计显著性分析
        statistical_analysis = self._perform_statistical_analysis(model_results)
        
        # 验证统计分析结果
        assert "mean_scores" in statistical_analysis
        assert "standard_deviations" in statistical_analysis
        assert "confidence_intervals" in statistical_analysis
        
        # 验证置信区间
        for model_id in models:
            assert model_id in statistical_analysis["confidence_intervals"]
            ci = statistical_analysis["confidence_intervals"][model_id]
            assert len(ci) == 2  # 下界和上界
            assert ci[0] <= ci[1]  # 下界应该小于等于上界
    
    @pytest.mark.integration
    def test_cross_domain_model_comparison(self, integration_config, multi_domain_datasets, mock_model_comparison_data):
        """测试跨领域模型比较"""
        models = mock_model_comparison_data
        domain_results = {}
        
        # 对每个领域和每个模型进行评估
        for domain_name, dataset in multi_domain_datasets.items():
            domain_results[domain_name] = {}
            
            for model_id, model_info in models.items():
                with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                     patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                     patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                    
                    # 设置模拟对象
                    mock_service = Mock()
                    mock_service.load_model.return_value = True
                    mock_service.is_model_loaded = True
                    mock_model_service.return_value = mock_service
                    
                    # 根据领域调整性能
                    expected_perf = model_info["expected_performance"].copy()
                    if domain_name == "cryptography":
                        # 密码学领域可能有不同的性能表现
                        expected_perf["domain_accuracy"] *= 1.1  # 提升领域准确性
                    elif domain_name == "security":
                        # 网络安全领域
                        expected_perf["semantic_similarity"] *= 0.95  # 略微降低语义相似性
                    
                    mock_eval = Mock()
                    mock_eval.evaluate_response.return_value = {
                        "semantic_similarity": expected_perf["semantic_similarity"],
                        "professional_accuracy": expected_perf["domain_accuracy"]
                    }
                    mock_framework.return_value = mock_eval
                    
                    mock_config.return_value = Mock()
                    
                    # 创建引擎
                    engine = ExpertEvaluationEngine(integration_config)
                    engine.model_service = mock_service
                    engine.evaluation_framework = mock_eval
                    
                    engine.load_model(model_info["path"])
                    
                    # 使用领域特定的测试数据
                    test_items = dataset.qa_items[:8]  # 使用8个项目
                    
                    with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                        mock_eval_instance = Mock()
                        mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                            EvaluationDimension.SEMANTIC_SIMILARITY: expected_perf["semantic_similarity"],
                            EvaluationDimension.DOMAIN_ACCURACY: expected_perf["domain_accuracy"]
                        }
                        mock_eval_instance.calculate_weighted_score.return_value = sum(expected_perf.values()) / len(expected_perf)
                        mock_evaluator.return_value = mock_eval_instance
                        
                        result = engine.evaluate_model(test_items)
                        domain_results[domain_name][model_id] = result
        
        # 分析跨领域性能
        cross_domain_analysis = self._analyze_cross_domain_performance(domain_results)
        
        # 验证跨领域分析结果
        assert "domain_rankings" in cross_domain_analysis
        assert "model_consistency" in cross_domain_analysis
        assert "domain_specialization" in cross_domain_analysis
        
        # 验证每个领域都有排名
        for domain_name in multi_domain_datasets.keys():
            assert domain_name in cross_domain_analysis["domain_rankings"]
            ranking = cross_domain_analysis["domain_rankings"][domain_name]
            assert len(ranking) == len(models)
    
    @pytest.mark.integration
    def test_model_ensemble_evaluation(self, integration_config, large_qa_dataset, mock_model_comparison_data):
        """测试模型集成评估"""
        models = mock_model_comparison_data
        individual_results = {}
        
        # 首先获取每个模型的单独评估结果
        test_items = large_qa_dataset[:15]
        
        for model_id, model_info in models.items():
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置模拟对象
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                expected_perf = model_info["expected_performance"]
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": expected_perf["semantic_similarity"],
                    "professional_accuracy": expected_perf["domain_accuracy"]
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                engine.load_model(model_info["path"])
                
                with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                    mock_eval_instance = Mock()
                    mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                        EvaluationDimension.SEMANTIC_SIMILARITY: expected_perf["semantic_similarity"],
                        EvaluationDimension.DOMAIN_ACCURACY: expected_perf["domain_accuracy"]
                    }
                    mock_eval_instance.calculate_weighted_score.return_value = sum(expected_perf.values()) / len(expected_perf)
                    mock_evaluator.return_value = mock_eval_instance
                    
                    result = engine.evaluate_model(test_items)
                    individual_results[model_id] = result
        
        # 模拟集成评估
        ensemble_result = self._simulate_ensemble_evaluation(individual_results, test_items)
        
        # 验证集成结果
        assert ensemble_result is not None
        assert hasattr(ensemble_result, 'overall_score')
        
        # 集成结果应该不差于最佳单个模型
        best_individual_score = max(result.overall_score for result in individual_results.values())
        assert ensemble_result.overall_score >= best_individual_score * 0.95  # 允许5%的误差
    
    def _analyze_model_comparison(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析模型比较结果"""
        # 按总体评分排序
        ranking = sorted(
            comparison_results.items(),
            key=lambda x: x[1]["result"].overall_score,
            reverse=True
        )
        
        # 维度比较
        dimension_comparison = {}
        for model_id, data in comparison_results.items():
            result = data["result"]
            for dim, score_obj in result.dimension_scores.items():
                if dim not in dimension_comparison:
                    dimension_comparison[dim] = {}
                dimension_comparison[dim][model_id] = score_obj.score
        
        # 性能指标
        performance_metrics = {
            "average_score": statistics.mean([data["result"].overall_score for data in comparison_results.values()]),
            "score_variance": statistics.variance([data["result"].overall_score for data in comparison_results.values()]) if len(comparison_results) > 1 else 0,
            "average_evaluation_time": statistics.mean([data["evaluation_time"] for data in comparison_results.values()]),
            "throughput_comparison": {model_id: data["throughput"] for model_id, data in comparison_results.items()}
        }
        
        return {
            "ranking": [(model_id, data["result"].overall_score) for model_id, data in ranking],
            "dimension_comparison": dimension_comparison,
            "performance_metrics": performance_metrics
        }
    
    def _perform_statistical_analysis(self, model_results: Dict[str, List]) -> Dict[str, Any]:
        """执行统计显著性分析"""
        analysis = {
            "mean_scores": {},
            "standard_deviations": {},
            "confidence_intervals": {}
        }
        
        for model_id, results in model_results.items():
            scores = [result.overall_score for result in results]
            
            mean_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
            
            # 计算95%置信区间
            if len(scores) > 1:
                import math
                margin_error = 1.96 * std_dev / math.sqrt(len(scores))  # 假设正态分布
                ci_lower = mean_score - margin_error
                ci_upper = mean_score + margin_error
            else:
                ci_lower = ci_upper = mean_score
            
            analysis["mean_scores"][model_id] = mean_score
            analysis["standard_deviations"][model_id] = std_dev
            analysis["confidence_intervals"][model_id] = (ci_lower, ci_upper)
        
        return analysis
    
    def _analyze_cross_domain_performance(self, domain_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析跨领域性能"""
        analysis = {
            "domain_rankings": {},
            "model_consistency": {},
            "domain_specialization": {}
        }
        
        # 每个领域的排名
        for domain_name, model_results in domain_results.items():
            ranking = sorted(
                model_results.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            analysis["domain_rankings"][domain_name] = [(model_id, result.overall_score) for model_id, result in ranking]
        
        # 模型一致性分析
        for model_id in next(iter(domain_results.values())).keys():
            scores = [domain_results[domain][model_id].overall_score for domain in domain_results.keys()]
            consistency = 1.0 - (statistics.stdev(scores) if len(scores) > 1 else 0)  # 标准差越小，一致性越高
            analysis["model_consistency"][model_id] = consistency
        
        # 领域专业化分析
        for domain_name in domain_results.keys():
            domain_scores = [result.overall_score for result in domain_results[domain_name].values()]
            analysis["domain_specialization"][domain_name] = {
                "average_score": statistics.mean(domain_scores),
                "score_range": max(domain_scores) - min(domain_scores)
            }
        
        return analysis
    
    def _simulate_ensemble_evaluation(self, individual_results: Dict[str, Any], test_items: List) -> Any:
        """模拟集成评估"""
        # 简单的集成策略：平均分数
        ensemble_score = statistics.mean([result.overall_score for result in individual_results.values()])
        
        # 创建模拟的集成结果
        from unittest.mock import Mock
        ensemble_result = Mock()
        ensemble_result.overall_score = ensemble_score
        ensemble_result.dimension_scores = {}
        ensemble_result.processing_time = sum([getattr(result, 'processing_time', 1.0) for result in individual_results.values()])
        
        return ensemble_result


class TestModelComparisonRobustness:
    """模型比较健壮性测试"""
    
    @pytest.mark.integration
    def test_comparison_with_failed_models(self, integration_config, large_qa_dataset, mock_model_comparison_data):
        """测试包含失败模型的比较"""
        models = mock_model_comparison_data
        comparison_results = {}
        
        for i, (model_id, model_info) in enumerate(models.items()):
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置模拟对象
                mock_service = Mock()
                
                # 模拟第二个模型加载失败
                if i == 1:
                    mock_service.load_model.side_effect = Exception("模型加载失败")
                else:
                    mock_service.load_model.return_value = True
                    mock_service.is_model_loaded = True
                
                mock_model_service.return_value = mock_service
                
                expected_perf = model_info["expected_performance"]
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": expected_perf["semantic_similarity"],
                    "professional_accuracy": expected_perf["domain_accuracy"]
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                try:
                    engine.load_model(model_info["path"])
                    
                    test_items = large_qa_dataset[:5]
                    
                    with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                        mock_eval_instance = Mock()
                        mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                            EvaluationDimension.SEMANTIC_SIMILARITY: expected_perf["semantic_similarity"]
                        }
                        mock_eval_instance.calculate_weighted_score.return_value = expected_perf["semantic_similarity"]
                        mock_evaluator.return_value = mock_eval_instance
                        
                        result = engine.evaluate_model(test_items)
                        comparison_results[model_id] = {"result": result, "status": "success"}
                        
                except Exception as e:
                    comparison_results[model_id] = {"error": str(e), "status": "failed"}
        
        # 验证比较结果包含成功和失败的模型
        successful_models = [k for k, v in comparison_results.items() if v["status"] == "success"]
        failed_models = [k for k, v in comparison_results.items() if v["status"] == "failed"]
        
        assert len(successful_models) > 0, "应该有成功评估的模型"
        assert len(failed_models) > 0, "应该有失败的模型"
        
        # 验证可以基于成功的模型进行比较
        if len(successful_models) > 1:
            scores = [comparison_results[model_id]["result"].overall_score for model_id in successful_models]
            assert all(isinstance(score, (int, float)) for score in scores), "所有成功模型都应该有有效评分"
    
    @pytest.mark.integration
    def test_comparison_consistency(self, integration_config, large_qa_dataset, mock_model_comparison_data):
        """测试比较一致性"""
        models = list(mock_model_comparison_data.keys())[:2]  # 使用两个模型
        
        # 进行多次比较
        comparison_rounds = []
        
        for round_id in range(3):
            round_results = {}
            
            for model_id in models:
                model_info = mock_model_comparison_data[model_id]
                
                with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                     patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                     patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                    
                    # 设置模拟对象
                    mock_service = Mock()
                    mock_service.load_model.return_value = True
                    mock_service.is_model_loaded = True
                    mock_model_service.return_value = mock_service
                    
                    expected_perf = model_info["expected_performance"]
                    # 添加轻微的随机变化
                    variation = (hash(f"{model_id}_{round_id}") % 20 - 10) / 1000  # -0.01 to 0.01
                    
                    mock_eval = Mock()
                    mock_eval.evaluate_response.return_value = {
                        "semantic_similarity": expected_perf["semantic_similarity"] + variation,
                        "professional_accuracy": expected_perf["domain_accuracy"] + variation
                    }
                    mock_framework.return_value = mock_eval
                    
                    mock_config.return_value = Mock()
                    
                    # 创建引擎
                    engine = ExpertEvaluationEngine(integration_config)
                    engine.model_service = mock_service
                    engine.evaluation_framework = mock_eval
                    
                    engine.load_model(model_info["path"])
                    
                    test_items = large_qa_dataset[:10]
                    
                    with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                        mock_eval_instance = Mock()
                        final_score = expected_perf["semantic_similarity"] + variation
                        mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                            EvaluationDimension.SEMANTIC_SIMILARITY: final_score
                        }
                        mock_eval_instance.calculate_weighted_score.return_value = final_score
                        mock_evaluator.return_value = mock_eval_instance
                        
                        result = engine.evaluate_model(test_items)
                        round_results[model_id] = result.overall_score
            
            comparison_rounds.append(round_results)
        
        # 分析一致性
        for model_id in models:
            scores = [round_results[model_id] for round_results in comparison_rounds]
            score_variance = statistics.variance(scores) if len(scores) > 1 else 0
            
            # 方差应该很小，表示一致性好
            assert score_variance < 0.01, f"模型 {model_id} 的评估一致性不够好，方差: {score_variance}"
        
        # 验证相对排名的一致性
        rankings = []
        for round_results in comparison_rounds:
            ranking = sorted(round_results.items(), key=lambda x: x[1], reverse=True)
            rankings.append([model_id for model_id, _ in ranking])
        
        # 检查排名是否一致
        first_ranking = rankings[0]
        for ranking in rankings[1:]:
            assert ranking == first_ranking, f"排名不一致: {first_ranking} vs {ranking}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])