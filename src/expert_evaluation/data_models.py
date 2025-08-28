"""
专家评估数据模型

定义了专家评估系统使用的核心数据结构，包括QA评估项、评估结果、报告等。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from enum import Enum
import json
from .config import EvaluationDimension, ExpertiseLevel


@dataclass
class QAEvaluationItem:
    """QA评估项数据结构"""
    question_id: str
    question: str
    context: Optional[str]
    reference_answer: str
    model_answer: str
    domain_tags: List[str] = field(default_factory=list)
    difficulty_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    expected_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据验证"""
        if not self.question_id.strip():
            raise ValueError("问题ID不能为空")
        if not self.question.strip():
            raise ValueError("问题内容不能为空")
        if not self.reference_answer.strip():
            raise ValueError("参考答案不能为空")
        if not self.model_answer.strip():
            raise ValueError("模型答案不能为空")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "context": self.context,
            "reference_answer": self.reference_answer,
            "model_answer": self.model_answer,
            "domain_tags": self.domain_tags,
            "difficulty_level": self.difficulty_level.value,
            "expected_concepts": self.expected_concepts,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAEvaluationItem':
        """从字典创建实例"""
        data_copy = data.copy()
        if "difficulty_level" in data_copy:
            data_copy["difficulty_level"] = ExpertiseLevel(data["difficulty_level"])
        return cls(**data_copy)


@dataclass
class DimensionScore:
    """维度评分详情"""
    dimension: EvaluationDimension
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据验证"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("评分必须在0.0-1.0范围内")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("置信度必须在0.0-1.0范围内")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "confidence": self.confidence,
            "details": self.details,
            "sub_scores": self.sub_scores
        }


@dataclass
class ExpertEvaluationResult:
    """专家评估结果"""
    question_id: str
    overall_score: float
    dimension_scores: Dict[EvaluationDimension, DimensionScore]
    detailed_feedback: Dict[str, str] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据验证"""
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("总体评分必须在0.0-1.0范围内")
        if not self.question_id.strip():
            raise ValueError("问题ID不能为空")
    
    def get_score_by_dimension(self, dimension: EvaluationDimension) -> Optional[float]:
        """获取指定维度的评分"""
        dimension_score = self.dimension_scores.get(dimension)
        return dimension_score.score if dimension_score else None
    
    def get_average_confidence(self) -> float:
        """获取平均置信度"""
        if not self.dimension_scores:
            return 0.0
        
        confidences = [ds.confidence for ds in self.dimension_scores.values()]
        return sum(confidences) / len(confidences)
    
    def get_performance_category(self) -> str:
        """获取性能类别"""
        if self.overall_score >= 0.9:
            return "优秀"
        elif self.overall_score >= 0.7:
            return "良好"
        elif self.overall_score >= 0.5:
            return "一般"
        else:
            return "需要改进"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question_id": self.question_id,
            "overall_score": self.overall_score,
            "dimension_scores": {
                dim.value: score.to_dict() 
                for dim, score in self.dimension_scores.items()
            },
            "detailed_feedback": self.detailed_feedback,
            "improvement_suggestions": self.improvement_suggestions,
            "confidence_intervals": {
                k: list(v) for k, v in self.confidence_intervals.items()
            },
            "statistical_significance": self.statistical_significance,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "performance_category": self.get_performance_category(),
            "average_confidence": self.get_average_confidence()
        }


@dataclass
class BatchEvaluationResult:
    """批量评估结果"""
    batch_id: str
    individual_results: List[ExpertEvaluationResult]
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """计算汇总统计"""
        if self.individual_results:
            self._calculate_summary_statistics()
    
    def _calculate_summary_statistics(self):
        """计算汇总统计信息"""
        if not self.individual_results:
            return
        
        # 基础统计
        overall_scores = [r.overall_score for r in self.individual_results]
        self.summary_statistics.update({
            "total_items": len(self.individual_results),
            "average_score": sum(overall_scores) / len(overall_scores),
            "min_score": min(overall_scores),
            "max_score": max(overall_scores),
            "score_std": self._calculate_std(overall_scores)
        })
        
        # 维度统计
        dimension_stats = {}
        for dimension in EvaluationDimension:
            scores = [
                r.get_score_by_dimension(dimension) 
                for r in self.individual_results 
                if r.get_score_by_dimension(dimension) is not None
            ]
            if scores:
                dimension_stats[dimension.value] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": self._calculate_std(scores)
                }
        
        self.summary_statistics["dimension_statistics"] = dimension_stats
        
        # 性能分布
        categories = {}
        for result in self.individual_results:
            category = result.get_performance_category()
            categories[category] = categories.get(category, 0) + 1
        
        self.summary_statistics["performance_distribution"] = categories
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_top_performers(self, n: int = 5) -> List[ExpertEvaluationResult]:
        """获取表现最好的N个结果"""
        return sorted(
            self.individual_results, 
            key=lambda x: x.overall_score, 
            reverse=True
        )[:n]
    
    def get_bottom_performers(self, n: int = 5) -> List[ExpertEvaluationResult]:
        """获取表现最差的N个结果"""
        return sorted(
            self.individual_results, 
            key=lambda x: x.overall_score
        )[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "batch_id": self.batch_id,
            "individual_results": [r.to_dict() for r in self.individual_results],
            "summary_statistics": self.summary_statistics,
            "total_processing_time": self.total_processing_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    valid_items_count: int = 0
    invalid_items_count: int = 0
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error_message: str):
        """添加错误信息"""
        self.errors.append(error_message)
        self.is_valid = False
    
    def add_warning(self, warning_message: str):
        """添加警告信息"""
        self.warnings.append(warning_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "valid_items_count": self.valid_items_count,
            "invalid_items_count": self.invalid_items_count,
            "validation_details": self.validation_details
        }


@dataclass
class EvaluationDataset:
    """评估数据集"""
    dataset_id: str
    name: str
    description: str
    qa_items: List[QAEvaluationItem]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据验证和统计"""
        if not self.dataset_id.strip():
            raise ValueError("数据集ID不能为空")
        if not self.name.strip():
            raise ValueError("数据集名称不能为空")
        
        # 计算统计信息
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """计算数据集统计信息"""
        if not self.qa_items:
            return
        
        # 难度分布
        difficulty_dist = {}
        for item in self.qa_items:
            level = item.difficulty_level.name
            difficulty_dist[level] = difficulty_dist.get(level, 0) + 1
        
        # 领域标签分布
        domain_dist = {}
        for item in self.qa_items:
            for tag in item.domain_tags:
                domain_dist[tag] = domain_dist.get(tag, 0) + 1
        
        # 长度统计
        question_lengths = [len(item.question) for item in self.qa_items]
        answer_lengths = [len(item.reference_answer) for item in self.qa_items]
        
        self.metadata.update({
            "total_items": len(self.qa_items),
            "difficulty_distribution": difficulty_dist,
            "domain_distribution": domain_dist,
            "average_question_length": sum(question_lengths) / len(question_lengths),
            "average_answer_length": sum(answer_lengths) / len(answer_lengths)
        })
    
    def get_items_by_difficulty(self, level: ExpertiseLevel) -> List[QAEvaluationItem]:
        """根据难度级别获取项目"""
        return [item for item in self.qa_items if item.difficulty_level == level]
    
    def get_items_by_domain(self, domain_tag: str) -> List[QAEvaluationItem]:
        """根据领域标签获取项目"""
        return [item for item in self.qa_items if domain_tag in item.domain_tags]
    
    def split_dataset(self, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> Tuple['EvaluationDataset', 'EvaluationDataset', 'EvaluationDataset']:
        """分割数据集"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("分割比例总和应为1.0")
        
        total_items = len(self.qa_items)
        train_size = int(total_items * train_ratio)
        val_size = int(total_items * val_ratio)
        
        # 随机打乱
        import random
        shuffled_items = self.qa_items.copy()
        random.shuffle(shuffled_items)
        
        # 分割
        train_items = shuffled_items[:train_size]
        val_items = shuffled_items[train_size:train_size + val_size]
        test_items = shuffled_items[train_size + val_size:]
        
        # 创建子数据集
        train_dataset = EvaluationDataset(
            dataset_id=f"{self.dataset_id}_train",
            name=f"{self.name} - 训练集",
            description=f"{self.description} (训练集)",
            qa_items=train_items
        )
        
        val_dataset = EvaluationDataset(
            dataset_id=f"{self.dataset_id}_val",
            name=f"{self.name} - 验证集",
            description=f"{self.description} (验证集)",
            qa_items=val_items
        )
        
        test_dataset = EvaluationDataset(
            dataset_id=f"{self.dataset_id}_test",
            name=f"{self.name} - 测试集",
            description=f"{self.description} (测试集)",
            qa_items=test_items
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "qa_items": [item.to_dict() for item in self.qa_items],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class EvaluationReport:
    """评估报告"""
    report_id: str
    title: str
    summary: str
    evaluation_results: Union[ExpertEvaluationResult, BatchEvaluationResult]
    charts_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    format_type: str = "json"
    
    def __post_init__(self):
        """报告后处理"""
        if not self.report_id.strip():
            raise ValueError("报告ID不能为空")
        if not self.title.strip():
            raise ValueError("报告标题不能为空")
    
    def add_chart(self, chart_name: str, chart_data: Dict[str, Any]):
        """添加图表数据"""
        self.charts_data[chart_name] = chart_data
    
    def add_recommendation(self, recommendation: str):
        """添加建议"""
        if recommendation.strip():
            self.recommendations.append(recommendation)
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if isinstance(self.evaluation_results, ExpertEvaluationResult):
            return {
                "overall_score": self.evaluation_results.overall_score,
                "performance_category": self.evaluation_results.get_performance_category(),
                "key_strengths": self._extract_strengths(),
                "improvement_areas": self._extract_improvement_areas(),
                "recommendation_count": len(self.recommendations)
            }
        elif isinstance(self.evaluation_results, BatchEvaluationResult):
            stats = self.evaluation_results.summary_statistics
            return {
                "total_items": stats.get("total_items", 0),
                "average_score": stats.get("average_score", 0.0),
                "performance_distribution": stats.get("performance_distribution", {}),
                "recommendation_count": len(self.recommendations)
            }
        else:
            return {}
    
    def _extract_strengths(self) -> List[str]:
        """提取优势点"""
        strengths = []
        if isinstance(self.evaluation_results, ExpertEvaluationResult):
            for dim, score_obj in self.evaluation_results.dimension_scores.items():
                if score_obj.score >= 0.8:
                    strengths.append(f"{dim.value}: {score_obj.score:.2f}")
        return strengths[:3]  # 最多3个
    
    def _extract_improvement_areas(self) -> List[str]:
        """提取改进领域"""
        areas = []
        if isinstance(self.evaluation_results, ExpertEvaluationResult):
            for dim, score_obj in self.evaluation_results.dimension_scores.items():
                if score_obj.score < 0.6:
                    areas.append(f"{dim.value}: {score_obj.score:.2f}")
        return areas[:3]  # 最多3个
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "summary": self.summary,
            "evaluation_results": self.evaluation_results.to_dict(),
            "charts_data": self.charts_data,
            "recommendations": self.recommendations,
            "detailed_analysis": self.detailed_analysis,
            "generated_at": self.generated_at.isoformat(),
            "format_type": self.format_type,
            "executive_summary": self.get_executive_summary()
        }
    
    def export_to_json(self, file_path: str) -> bool:
        """导出为JSON格式"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False