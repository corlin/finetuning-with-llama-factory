"""
统计分析测试模块

测试专家评估系统的统计分析功能，包括：
- 置信区间计算
- 统计显著性检验
- 评估偏差检测
- 结果可重现性验证
"""

import pytest
import statistics
from datetime import datetime
from typing import List

from src.expert_evaluation.config import EvaluationDimension, ExpertiseLevel
from src.expert_evaluation.data_models import (
    QAEvaluationItem, ExpertEvaluationResult, DimensionScore
)
from src.expert_evaluation.multi_dimensional import (
    StatisticalAnalyzer, create_comprehensive_analysis_report
)


class TestStatisticalAnalyzer:
    """统计分析器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.analyzer = StatisticalAnalyzer(confidence_level=0.95)
        self.test_results = self._create_test_results()
    
    def _create_test_results(self) -> List[ExpertEvaluationResult]:
        """创建测试用的评估结果"""
        results = []
        
        # 创建不同评分的测试结果
        test_scores = [0.8, 0.75, 0.85, 0.7, 0.9, 0.65, 0.88, 0.72, 0.83, 0.77]
        
        for i, overall_score in enumerate(test_scores):
            # 创建维度评分
            dimension_scores = {}
            for j, dimension in enumerate(EvaluationDimension):
                # 为每个维度创建略有不同的评分
                dim_score = overall_score + (j - 4) * 0.02  # 轻微变化
                dim_score = max(0.0, min(1.0, dim_score))  # 确保在有效范围内
                
                dimension_scores[dimension] = DimensionScore(
                    dimension=dimension,
                    score=dim_score,
                    confidence=0.85 + i * 0.01  # 递增的置信度
                )
            
            result = ExpertEvaluationResult(
                question_id=f"test_{i+1:03d}",
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                processing_time=1.0 + i * 0.1,
                timestamp=datetime.now(),
                metadata={"difficulty_level": "intermediate" if i % 2 == 0 else "advanced"}
            )
            
            results.append(result)
        
        return results
    
    def test_confidence_interval_calculation(self):
        """测试置信区间计算"""
        # 测试总体评分的置信区间
        confidence_intervals = self.analyzer.calculate_confidence_intervals(self.test_results)
        
        assert "overall_score" in confidence_intervals
        lower, upper = confidence_intervals["overall_score"]
        
        # 检查置信区间的合理性
        assert 0.0 <= lower <= upper <= 1.0
        assert lower < upper  # 下界应小于上界
        
        # 检查置信区间包含均值
        overall_scores = [r.overall_score for r in self.test_results]
        mean_score = statistics.mean(overall_scores)
        assert lower <= mean_score <= upper
        
        print(f"总体评分置信区间: [{lower:.3f}, {upper:.3f}]")
        print(f"平均分: {mean_score:.3f}")
    
    def test_dimension_confidence_intervals(self):
        """测试各维度置信区间计算"""
        # 测试特定维度的置信区间
        dimension = EvaluationDimension.SEMANTIC_SIMILARITY
        confidence_intervals = self.analyzer.calculate_confidence_intervals(
            self.test_results, dimension=dimension
        )
        
        assert dimension.value in confidence_intervals
        lower, upper = confidence_intervals[dimension.value]
        
        assert 0.0 <= lower <= upper <= 1.0
        
        print(f"{dimension.value}置信区间: [{lower:.3f}, {upper:.3f}]")
    
    def test_statistical_significance_testing(self):
        """测试统计显著性检验"""
        # 将结果分为两组
        mid_point = len(self.test_results) // 2
        group1 = self.test_results[:mid_point]
        group2 = self.test_results[mid_point:]
        
        # 执行显著性检验
        significance_results = self.analyzer.test_statistical_significance(group1, group2)
        
        assert "overall_score" in significance_results
        
        test_result = significance_results["overall_score"]
        assert "t_statistic" in test_result
        assert "p_value" in test_result
        assert "significant" in test_result
        
        # 检查p值范围
        assert 0.0 <= test_result["p_value"] <= 1.0
        
        print(f"t统计量: {test_result['t_statistic']:.3f}")
        print(f"p值: {test_result['p_value']:.3f}")
        print(f"显著性: {test_result['significant']}")
    
    def test_bias_detection(self):
        """测试评估偏差检测"""
        bias_analysis = self.analyzer.detect_evaluation_bias(
            self.test_results, metadata_key="difficulty_level"
        )
        
        assert "bias_detected" in bias_analysis
        assert "bias_type" in bias_analysis
        assert "bias_magnitude" in bias_analysis
        assert "affected_groups" in bias_analysis
        assert "recommendations" in bias_analysis
        
        print(f"偏差检测结果: {bias_analysis['bias_detected']}")
        print(f"偏差类型: {bias_analysis['bias_type']}")
        print(f"偏差幅度: {bias_analysis['bias_magnitude']:.3f}")
        
        if bias_analysis["affected_groups"]:
            print("受影响的组:")
            for group in bias_analysis["affected_groups"]:
                print(f"  {group['group']}: {group['mean_score']:.3f} (偏差: {group['deviation']:.3f})")
    
    def test_reproducibility_verification(self):
        """测试可重现性验证"""
        # 创建"重复"结果（添加小的随机变化）
        replicated_results = []
        
        for original in self.test_results:
            # 添加小的噪声来模拟重复实验
            noise = 0.02  # 2%的噪声
            
            replicated_dimension_scores = {}
            for dim, score_obj in original.dimension_scores.items():
                new_score = score_obj.score + (hash(original.question_id + dim.value) % 100 - 50) / 100 * noise
                new_score = max(0.0, min(1.0, new_score))
                
                replicated_dimension_scores[dim] = DimensionScore(
                    dimension=dim,
                    score=new_score,
                    confidence=score_obj.confidence
                )
            
            replicated_overall = original.overall_score + (hash(original.question_id) % 100 - 50) / 100 * noise
            replicated_overall = max(0.0, min(1.0, replicated_overall))
            
            replicated_result = ExpertEvaluationResult(
                question_id=original.question_id,
                overall_score=replicated_overall,
                dimension_scores=replicated_dimension_scores,
                processing_time=original.processing_time,
                timestamp=datetime.now(),
                metadata=original.metadata
            )
            
            replicated_results.append(replicated_result)
        
        # 验证可重现性
        reproducibility_report = self.analyzer.verify_reproducibility(
            self.test_results, replicated_results, tolerance=0.05
        )
        
        assert "is_reproducible" in reproducibility_report
        assert "overall_correlation" in reproducibility_report
        assert "dimension_correlations" in reproducibility_report
        assert "inconsistent_items" in reproducibility_report
        assert "reproducibility_score" in reproducibility_report
        
        print(f"可重现性: {reproducibility_report['is_reproducible']}")
        print(f"总体相关性: {reproducibility_report['overall_correlation']:.3f}")
        print(f"可重现性评分: {reproducibility_report['reproducibility_score']:.3f}")
        print(f"不一致项目数: {len(reproducibility_report['inconsistent_items'])}")
        
        # 检查相关性范围
        if reproducibility_report['overall_correlation'] is not None:
            assert -1.0 <= reproducibility_report['overall_correlation'] <= 1.0
    
    def test_comprehensive_analysis_report(self):
        """测试综合分析报告"""
        report = create_comprehensive_analysis_report(self.test_results, self.analyzer)
        
        # 检查报告结构
        assert "summary" in report
        assert "statistical_analysis" in report
        assert "quality_assessment" in report
        assert "recommendations" in report
        
        # 检查摘要部分
        summary = report["summary"]
        assert "total_evaluations" in summary
        assert "mean_score" in summary
        assert "std_deviation" in summary
        
        # 检查统计分析部分
        stats = report["statistical_analysis"]
        assert "confidence_intervals" in stats
        assert "bias_analysis" in stats
        
        # 检查质量评估部分
        quality = report["quality_assessment"]
        assert "consistency" in quality
        assert "reliability" in quality
        assert "coverage" in quality
        
        # 检查建议部分
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        
        print("=== 综合分析报告 ===")
        print(f"总评估数: {summary['total_evaluations']}")
        print(f"平均分: {summary['mean_score']:.3f}")
        print(f"标准差: {summary['std_deviation']:.3f}")
        print(f"一致性: {quality['consistency']:.3f}")
        print(f"可靠性: {quality['reliability']:.3f}")
        print(f"覆盖度: {quality['coverage']:.3f}")
        print(f"建议数量: {len(recommendations)}")
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空结果列表
        empty_results = []
        confidence_intervals = self.analyzer.calculate_confidence_intervals(empty_results)
        assert confidence_intervals == {}
        
        # 测试单个结果
        single_result = [self.test_results[0]]
        confidence_intervals = self.analyzer.calculate_confidence_intervals(single_result)
        # 应该能处理单个结果而不崩溃
        
        # 测试所有评分相同的情况
        identical_results = []
        for i in range(5):
            result = ExpertEvaluationResult(
                question_id=f"identical_{i}",
                overall_score=0.8,
                dimension_scores={},
                processing_time=1.0,
                timestamp=datetime.now()
            )
            identical_results.append(result)
        
        confidence_intervals = self.analyzer.calculate_confidence_intervals(identical_results)
        if "overall_score" in confidence_intervals:
            lower, upper = confidence_intervals["overall_score"]
            # 对于相同的评分，置信区间应该很窄
            assert upper - lower < 0.1
        
        print("边界情况测试通过")


def test_statistical_analysis_integration():
    """集成测试：统计分析功能"""
    print("\n开始统计分析集成测试...")
    
    # 创建测试实例
    test_instance = TestStatisticalAnalyzer()
    test_instance.setup_method()
    
    try:
        # 运行所有测试
        test_instance.test_confidence_interval_calculation()
        print("✓ 置信区间计算测试通过")
        
        test_instance.test_dimension_confidence_intervals()
        print("✓ 维度置信区间测试通过")
        
        test_instance.test_statistical_significance_testing()
        print("✓ 统计显著性检验测试通过")
        
        test_instance.test_bias_detection()
        print("✓ 偏差检测测试通过")
        
        test_instance.test_reproducibility_verification()
        print("✓ 可重现性验证测试通过")
        
        test_instance.test_comprehensive_analysis_report()
        print("✓ 综合分析报告测试通过")
        
        test_instance.test_edge_cases()
        print("✓ 边界情况测试通过")
        
        print("\n✅ 所有统计分析测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 统计分析测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行集成测试
    test_statistical_analysis_integration()