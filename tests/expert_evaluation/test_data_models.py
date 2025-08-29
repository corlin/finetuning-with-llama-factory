"""
数据模型单元测试

测试专家评估系统的核心数据模型，包括QAEvaluationItem、ExpertEvaluationResult等。
"""

import pytest
import json
from datetime import datetime
from typing import Dict, List

from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    DimensionScore,
    ExpertEvaluationResult,
    BatchEvaluationResult,
    ValidationResult,
    EvaluationDataset,
    EvaluationReport
)
from src.expert_evaluation.config import EvaluationDimension, ExpertiseLevel
from src.expert_evaluation.exceptions import ValidationError


class TestQAEvaluationItem:
    """QAEvaluationItem测试类"""
    
    def test_valid_qa_item_creation(self):
        """测试有效QA项目创建"""
        item = QAEvaluationItem(
            question_id="test_001",
            question="什么是RSA加密算法？",
            context="密码学基础知识",
            reference_answer="RSA是一种非对称加密算法...",
            model_answer="RSA算法是基于大数分解难题的公钥加密算法...",
            domain_tags=["密码学", "加密算法"],
            difficulty_level=ExpertiseLevel.INTERMEDIATE,
            expected_concepts=["非对称加密", "公钥", "私钥"]
        )
        
        assert item.question_id == "test_001"
        assert item.question == "什么是RSA加密算法？"
        assert item.difficulty_level == ExpertiseLevel.INTERMEDIATE
        assert "密码学" in item.domain_tags
        assert "非对称加密" in item.expected_concepts
    
    def test_invalid_qa_item_creation(self):
        """测试无效QA项目创建"""
        # 空问题ID
        with pytest.raises(ValueError, match="问题ID不能为空"):
            QAEvaluationItem(
                question_id="",
                question="测试问题",
                context=None,
                reference_answer="参考答案",
                model_answer="模型答案"
            )
        
        # 空问题内容
        with pytest.raises(ValueError, match="问题内容不能为空"):
            QAEvaluationItem(
                question_id="test_001",
                question="",
                context=None,
                reference_answer="参考答案",
                model_answer="模型答案"
            )
        
        # 空参考答案
        with pytest.raises(ValueError, match="参考答案不能为空"):
            QAEvaluationItem(
                question_id="test_001",
                question="测试问题",
                context=None,
                reference_answer="",
                model_answer="模型答案"
            )
    
    def test_qa_item_to_dict(self):
        """测试QA项目转换为字典"""
        item = QAEvaluationItem(
            question_id="test_001",
            question="测试问题",
            context="测试上下文",
            reference_answer="参考答案",
            model_answer="模型答案",
            domain_tags=["测试"],
            difficulty_level=ExpertiseLevel.BEGINNER
        )
        
        item_dict = item.to_dict()
        
        assert item_dict["question_id"] == "test_001"
        assert item_dict["question"] == "测试问题"
        assert item_dict["difficulty_level"] == ExpertiseLevel.BEGINNER.value
        assert isinstance(item_dict["domain_tags"], list)
    
    def test_qa_item_from_dict(self):
        """测试从字典创建QA项目"""
        data = {
            "question_id": "test_001",
            "question": "测试问题",
            "context": "测试上下文",
            "reference_answer": "参考答案",
            "model_answer": "模型答案",
            "domain_tags": ["测试"],
            "difficulty_level": ExpertiseLevel.BEGINNER.value,
            "expected_concepts": ["概念1", "概念2"]
        }
        
        item = QAEvaluationItem.from_dict(data)
        
        assert item.question_id == "test_001"
        assert item.difficulty_level == ExpertiseLevel.BEGINNER
        assert len(item.expected_concepts) == 2


class TestDimensionScore:
    """DimensionScore测试类"""
    
    def test_valid_dimension_score_creation(self):
        """测试有效维度评分创建"""
        score = DimensionScore(
            dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
            score=0.85,
            confidence=0.92,
            details={"method": "cosine_similarity"},
            sub_scores={"lexical": 0.8, "semantic": 0.9}
        )
        
        assert score.dimension == EvaluationDimension.SEMANTIC_SIMILARITY
        assert score.score == 0.85
        assert score.confidence == 0.92
        assert score.details["method"] == "cosine_similarity"
    
    def test_invalid_score_range(self):
        """测试无效评分范围"""
        # 评分超出范围
        with pytest.raises(ValueError, match="评分必须在0.0-1.0范围内"):
            DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=1.5,
                confidence=0.9
            )
        
        # 置信度超出范围
        with pytest.raises(ValueError, match="置信度必须在0.0-1.0范围内"):
            DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=0.8,
                confidence=1.2
            )
    
    def test_dimension_score_to_dict(self):
        """测试维度评分转换为字典"""
        score = DimensionScore(
            dimension=EvaluationDimension.DOMAIN_ACCURACY,
            score=0.75,
            confidence=0.88
        )
        
        score_dict = score.to_dict()
        
        assert score_dict["dimension"] == EvaluationDimension.DOMAIN_ACCURACY.value
        assert score_dict["score"] == 0.75
        assert score_dict["confidence"] == 0.88


class TestExpertEvaluationResult:
    """ExpertEvaluationResult测试类"""
    
    def create_sample_result(self) -> ExpertEvaluationResult:
        """创建示例评估结果"""
        dimension_scores = {
            EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=0.85,
                confidence=0.9
            ),
            EvaluationDimension.DOMAIN_ACCURACY: DimensionScore(
                dimension=EvaluationDimension.DOMAIN_ACCURACY,
                score=0.78,
                confidence=0.85
            )
        }
        
        return ExpertEvaluationResult(
            question_id="test_001",
            overall_score=0.82,
            dimension_scores=dimension_scores,
            detailed_feedback={"strength": "语义理解准确", "weakness": "技术深度不足"},
            improvement_suggestions=["增加技术细节", "提高专业术语使用"],
            processing_time=2.5
        )
    
    def test_valid_evaluation_result_creation(self):
        """测试有效评估结果创建"""
        result = self.create_sample_result()
        
        assert result.question_id == "test_001"
        assert result.overall_score == 0.82
        assert len(result.dimension_scores) == 2
        assert len(result.improvement_suggestions) == 2
    
    def test_invalid_overall_score(self):
        """测试无效总体评分"""
        with pytest.raises(ValueError, match="总体评分必须在0.0-1.0范围内"):
            ExpertEvaluationResult(
                question_id="test_001",
                overall_score=1.5,
                dimension_scores={}
            )
    
    def test_get_score_by_dimension(self):
        """测试按维度获取评分"""
        result = self.create_sample_result()
        
        semantic_score = result.get_score_by_dimension(EvaluationDimension.SEMANTIC_SIMILARITY)
        assert semantic_score == 0.85
        
        missing_score = result.get_score_by_dimension(EvaluationDimension.INNOVATION)
        assert missing_score is None
    
    def test_get_average_confidence(self):
        """测试获取平均置信度"""
        result = self.create_sample_result()
        
        avg_confidence = result.get_average_confidence()
        expected = (0.9 + 0.85) / 2
        assert abs(avg_confidence - expected) < 0.01
    
    def test_get_performance_category(self):
        """测试获取性能类别"""
        # 优秀
        result = ExpertEvaluationResult(
            question_id="test_001",
            overall_score=0.95,
            dimension_scores={}
        )
        assert result.get_performance_category() == "优秀"
        
        # 良好
        result.overall_score = 0.75
        assert result.get_performance_category() == "良好"
        
        # 一般
        result.overall_score = 0.55
        assert result.get_performance_category() == "一般"
        
        # 需要改进
        result.overall_score = 0.35
        assert result.get_performance_category() == "需要改进"
    
    def test_evaluation_result_to_dict(self):
        """测试评估结果转换为字典"""
        result = self.create_sample_result()
        result_dict = result.to_dict()
        
        assert result_dict["question_id"] == "test_001"
        assert result_dict["overall_score"] == 0.82
        assert "performance_category" in result_dict
        assert "average_confidence" in result_dict
        assert isinstance(result_dict["dimension_scores"], dict)


class TestBatchEvaluationResult:
    """BatchEvaluationResult测试类"""
    
    def create_sample_batch_result(self) -> BatchEvaluationResult:
        """创建示例批量评估结果"""
        individual_results = []
        
        for i in range(5):
            dimension_scores = {
                EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
                    dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                    score=0.8 + i * 0.02,
                    confidence=0.9
                )
            }
            
            result = ExpertEvaluationResult(
                question_id=f"test_{i:03d}",
                overall_score=0.75 + i * 0.05,
                dimension_scores=dimension_scores
            )
            individual_results.append(result)
        
        return BatchEvaluationResult(
            batch_id="batch_001",
            individual_results=individual_results,
            total_processing_time=12.5
        )
    
    def test_batch_result_creation(self):
        """测试批量结果创建"""
        batch_result = self.create_sample_batch_result()
        
        assert batch_result.batch_id == "batch_001"
        assert len(batch_result.individual_results) == 5
        assert batch_result.total_processing_time == 12.5
        assert "total_items" in batch_result.summary_statistics
    
    def test_summary_statistics_calculation(self):
        """测试汇总统计计算"""
        batch_result = self.create_sample_batch_result()
        stats = batch_result.summary_statistics
        
        assert stats["total_items"] == 5
        assert "average_score" in stats
        assert "min_score" in stats
        assert "max_score" in stats
        assert "dimension_statistics" in stats
        assert "performance_distribution" in stats
    
    def test_get_top_performers(self):
        """测试获取最佳表现者"""
        batch_result = self.create_sample_batch_result()
        top_performers = batch_result.get_top_performers(n=3)
        
        assert len(top_performers) == 3
        # 验证按评分降序排列
        for i in range(len(top_performers) - 1):
            assert top_performers[i].overall_score >= top_performers[i + 1].overall_score
    
    def test_get_bottom_performers(self):
        """测试获取最差表现者"""
        batch_result = self.create_sample_batch_result()
        bottom_performers = batch_result.get_bottom_performers(n=2)
        
        assert len(bottom_performers) == 2
        # 验证按评分升序排列
        for i in range(len(bottom_performers) - 1):
            assert bottom_performers[i].overall_score <= bottom_performers[i + 1].overall_score


class TestValidationResult:
    """ValidationResult测试类"""
    
    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = ValidationResult(
            is_valid=True,
            valid_items_count=10,
            invalid_items_count=2
        )
        
        assert result.is_valid is True
        assert result.valid_items_count == 10
        assert result.invalid_items_count == 2
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """测试添加错误"""
        result = ValidationResult(is_valid=True)
        result.add_error("测试错误信息")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "测试错误信息"
    
    def test_add_warning(self):
        """测试添加警告"""
        result = ValidationResult(is_valid=True)
        result.add_warning("测试警告信息")
        
        assert result.is_valid is True  # 警告不影响有效性
        assert len(result.warnings) == 1
        assert result.warnings[0] == "测试警告信息"
    
    def test_validation_result_to_dict(self):
        """测试验证结果转换为字典"""
        result = ValidationResult(
            is_valid=False,
            valid_items_count=8,
            invalid_items_count=2
        )
        result.add_error("格式错误")
        result.add_warning("数据质量警告")
        
        result_dict = result.to_dict()
        
        assert result_dict["is_valid"] is False
        assert result_dict["valid_items_count"] == 8
        assert result_dict["invalid_items_count"] == 2
        assert len(result_dict["errors"]) == 1
        assert len(result_dict["warnings"]) == 1


class TestEvaluationDataset:
    """EvaluationDataset测试类"""
    
    def create_sample_dataset(self) -> EvaluationDataset:
        """创建示例数据集"""
        qa_items = []
        
        for i in range(10):
            item = QAEvaluationItem(
                question_id=f"q_{i:03d}",
                question=f"测试问题 {i}",
                context=f"测试上下文 {i}",
                reference_answer=f"参考答案 {i}",
                model_answer=f"模型答案 {i}",
                domain_tags=["测试", f"类别{i % 3}"],
                difficulty_level=list(ExpertiseLevel)[i % 4]
            )
            qa_items.append(item)
        
        return EvaluationDataset(
            dataset_id="test_dataset_001",
            name="测试数据集",
            description="用于单元测试的数据集",
            qa_items=qa_items
        )
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = self.create_sample_dataset()
        
        assert dataset.dataset_id == "test_dataset_001"
        assert dataset.name == "测试数据集"
        assert len(dataset.qa_items) == 10
        assert "total_items" in dataset.metadata
        assert dataset.metadata["total_items"] == 10
    
    def test_invalid_dataset_creation(self):
        """测试无效数据集创建"""
        # 空数据集ID
        with pytest.raises(ValueError, match="数据集ID不能为空"):
            EvaluationDataset(
                dataset_id="",
                name="测试数据集",
                description="测试描述",
                qa_items=[]
            )
        
        # 空数据集名称
        with pytest.raises(ValueError, match="数据集名称不能为空"):
            EvaluationDataset(
                dataset_id="test_001",
                name="",
                description="测试描述",
                qa_items=[]
            )
    
    def test_get_items_by_difficulty(self):
        """测试按难度获取项目"""
        dataset = self.create_sample_dataset()
        
        beginner_items = dataset.get_items_by_difficulty(ExpertiseLevel.BEGINNER)
        # 应该有3个BEGINNER级别的项目 (索引0, 4, 8)
        assert len(beginner_items) == 3
        
        for item in beginner_items:
            assert item.difficulty_level == ExpertiseLevel.BEGINNER
    
    def test_get_items_by_domain(self):
        """测试按领域获取项目"""
        dataset = self.create_sample_dataset()
        
        test_items = dataset.get_items_by_domain("测试")
        assert len(test_items) == 10  # 所有项目都有"测试"标签
        
        category_0_items = dataset.get_items_by_domain("类别0")
        # 应该有4个"类别0"的项目 (索引0, 3, 6, 9)
        assert len(category_0_items) == 4
    
    def test_split_dataset(self):
        """测试数据集分割"""
        dataset = self.create_sample_dataset()
        
        train_ds, val_ds, test_ds = dataset.split_dataset(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # 验证分割比例
        total_items = len(dataset.qa_items)
        assert len(train_ds.qa_items) == int(total_items * 0.6)
        assert len(val_ds.qa_items) == int(total_items * 0.2)
        assert len(test_ds.qa_items) == total_items - len(train_ds.qa_items) - len(val_ds.qa_items)
        
        # 验证数据集ID和名称
        assert train_ds.dataset_id == "test_dataset_001_train"
        assert "训练集" in train_ds.name
    
    def test_invalid_split_ratios(self):
        """测试无效分割比例"""
        dataset = self.create_sample_dataset()
        
        with pytest.raises(ValueError, match="分割比例总和应为1.0"):
            dataset.split_dataset(
                train_ratio=0.6,
                val_ratio=0.3,
                test_ratio=0.2  # 总和为1.1
            )
    
    def test_dataset_to_dict(self):
        """测试数据集转换为字典"""
        dataset = self.create_sample_dataset()
        dataset_dict = dataset.to_dict()
        
        assert dataset_dict["dataset_id"] == "test_dataset_001"
        assert dataset_dict["name"] == "测试数据集"
        assert len(dataset_dict["qa_items"]) == 10
        assert "metadata" in dataset_dict
        assert "created_at" in dataset_dict


class TestEvaluationReport:
    """EvaluationReport测试类"""
    
    def create_sample_report(self) -> EvaluationReport:
        """创建示例评估报告"""
        # 创建评估结果
        dimension_scores = {
            EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=0.85,
                confidence=0.9
            ),
            EvaluationDimension.DOMAIN_ACCURACY: DimensionScore(
                dimension=EvaluationDimension.DOMAIN_ACCURACY,
                score=0.78,
                confidence=0.85
            )
        }
        
        evaluation_result = ExpertEvaluationResult(
            question_id="test_001",
            overall_score=0.82,
            dimension_scores=dimension_scores
        )
        
        return EvaluationReport(
            report_id="report_001",
            title="测试评估报告",
            summary="这是一个测试评估报告的摘要",
            evaluation_results=evaluation_result
        )
    
    def test_report_creation(self):
        """测试报告创建"""
        report = self.create_sample_report()
        
        assert report.report_id == "report_001"
        assert report.title == "测试评估报告"
        assert isinstance(report.evaluation_results, ExpertEvaluationResult)
        assert report.format_type == "json"
    
    def test_invalid_report_creation(self):
        """测试无效报告创建"""
        # 空报告ID
        with pytest.raises(ValueError, match="报告ID不能为空"):
            EvaluationReport(
                report_id="",
                title="测试报告",
                summary="测试摘要",
                evaluation_results=None
            )
        
        # 空报告标题
        with pytest.raises(ValueError, match="报告标题不能为空"):
            EvaluationReport(
                report_id="report_001",
                title="",
                summary="测试摘要",
                evaluation_results=None
            )
    
    def test_add_chart(self):
        """测试添加图表"""
        report = self.create_sample_report()
        
        chart_data = {
            "type": "bar",
            "data": [0.8, 0.7, 0.9],
            "labels": ["维度1", "维度2", "维度3"]
        }
        
        report.add_chart("performance_chart", chart_data)
        
        assert "performance_chart" in report.charts_data
        assert report.charts_data["performance_chart"]["type"] == "bar"
    
    def test_add_recommendation(self):
        """测试添加建议"""
        report = self.create_sample_report()
        
        report.add_recommendation("提高语义理解能力")
        report.add_recommendation("增强技术深度")
        report.add_recommendation("")  # 空建议应该被忽略
        
        assert len(report.recommendations) == 2
        assert "提高语义理解能力" in report.recommendations
        assert "增强技术深度" in report.recommendations
    
    def test_get_executive_summary(self):
        """测试获取执行摘要"""
        report = self.create_sample_report()
        report.add_recommendation("测试建议1")
        report.add_recommendation("测试建议2")
        
        summary = report.get_executive_summary()
        
        assert "overall_score" in summary
        assert "performance_category" in summary
        assert "key_strengths" in summary
        assert "improvement_areas" in summary
        assert summary["recommendation_count"] == 2
    
    def test_report_to_dict(self):
        """测试报告转换为字典"""
        report = self.create_sample_report()
        report_dict = report.to_dict()
        
        assert report_dict["report_id"] == "report_001"
        assert report_dict["title"] == "测试评估报告"
        assert "evaluation_results" in report_dict
        assert "executive_summary" in report_dict
        assert "generated_at" in report_dict
    
    def test_export_to_json(self, tmp_path):
        """测试导出为JSON"""
        report = self.create_sample_report()
        
        json_file = tmp_path / "test_report.json"
        success = report.export_to_json(str(json_file))
        
        assert success is True
        assert json_file.exists()
        
        # 验证JSON内容
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["report_id"] == "report_001"
            assert data["title"] == "测试评估报告"


# 性能测试
class TestDataModelsPerformance:
    """数据模型性能测试"""
    
    def test_large_dataset_creation_performance(self):
        """测试大数据集创建性能"""
        import time
        
        start_time = time.time()
        
        # 创建大数据集
        qa_items = []
        for i in range(1000):
            item = QAEvaluationItem(
                question_id=f"perf_test_{i:06d}",
                question=f"性能测试问题 {i} " * 10,  # 较长的问题
                context=f"性能测试上下文 {i} " * 5,
                reference_answer=f"性能测试参考答案 {i} " * 20,
                model_answer=f"性能测试模型答案 {i} " * 20,
                domain_tags=[f"标签{j}" for j in range(5)],
                difficulty_level=list(ExpertiseLevel)[i % 4]
            )
            qa_items.append(item)
        
        dataset = EvaluationDataset(
            dataset_id="perf_test_dataset",
            name="性能测试数据集",
            description="用于性能测试的大数据集",
            qa_items=qa_items
        )
        
        creation_time = time.time() - start_time
        
        # 验证创建成功
        assert len(dataset.qa_items) == 1000
        assert "total_items" in dataset.metadata
        
        # 性能要求：1000个项目的数据集创建应在5秒内完成
        assert creation_time < 5.0, f"数据集创建耗时过长: {creation_time:.2f}秒"
    
    def test_batch_result_statistics_performance(self):
        """测试批量结果统计计算性能"""
        import time
        
        # 创建大量评估结果
        individual_results = []
        for i in range(500):
            dimension_scores = {}
            for dim in EvaluationDimension:
                dimension_scores[dim] = DimensionScore(
                    dimension=dim,
                    score=0.5 + (i % 50) * 0.01,
                    confidence=0.8 + (i % 20) * 0.01
                )
            
            result = ExpertEvaluationResult(
                question_id=f"perf_test_{i:06d}",
                overall_score=0.6 + (i % 40) * 0.01,
                dimension_scores=dimension_scores
            )
            individual_results.append(result)
        
        start_time = time.time()
        
        batch_result = BatchEvaluationResult(
            batch_id="perf_test_batch",
            individual_results=individual_results
        )
        
        calculation_time = time.time() - start_time
        
        # 验证统计计算正确
        assert len(batch_result.individual_results) == 500
        assert "total_items" in batch_result.summary_statistics
        assert batch_result.summary_statistics["total_items"] == 500
        
        # 性能要求：500个结果的统计计算应在2秒内完成
        assert calculation_time < 2.0, f"统计计算耗时过长: {calculation_time:.2f}秒"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])