"""
改进建议生成器

基于评估结果的改进建议算法，实现问题诊断和解决方案推荐，
创建最佳实践和优化建议库，添加个性化建议生成功能。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .data_models import ExpertEvaluationResult, BatchEvaluationResult, DimensionScore
from .config import EvaluationDimension, ExpertiseLevel
from .exceptions import ImprovementAnalysisError


class ImprovementPriority(Enum):
    """改进优先级"""
    CRITICAL = "关键"
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class ImprovementCategory(Enum):
    """改进类别"""
    DATA_QUALITY = "数据质量"
    MODEL_ARCHITECTURE = "模型架构"
    TRAINING_STRATEGY = "训练策略"
    EVALUATION_METHOD = "评估方法"
    DEPLOYMENT_OPTIMIZATION = "部署优化"
    DOMAIN_SPECIFIC = "领域特定"


@dataclass
class ImprovementSuggestion:
    """改进建议数据结构"""
    id: str
    title: str
    description: str
    category: ImprovementCategory
    priority: ImprovementPriority
    affected_dimensions: List[EvaluationDimension]
    implementation_steps: List[str]
    expected_impact: str
    effort_level: str  # 低、中、高
    time_estimate: str  # 预估时间
    prerequisites: List[str]
    success_metrics: List[str]
    related_suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "affected_dimensions": [dim.value for dim in self.affected_dimensions],
            "implementation_steps": self.implementation_steps,
            "expected_impact": self.expected_impact,
            "effort_level": self.effort_level,
            "time_estimate": self.time_estimate,
            "prerequisites": self.prerequisites,
            "success_metrics": self.success_metrics,
            "related_suggestions": self.related_suggestions
        }


class ImprovementAdvisor:
    """改进建议生成器"""
    
    def __init__(self):
        """初始化改进建议生成器"""
        self.logger = logging.getLogger(__name__)
        
        # 加载建议模板和规则
        self.suggestion_templates = self._load_suggestion_templates()
        self.diagnostic_rules = self._load_diagnostic_rules()
        self.best_practices = self._load_best_practices()
        
        # 建议计数器
        self._suggestion_counter = 0
    
    def generate_comprehensive_suggestions(self, 
                                         results: ExpertEvaluationResult,
                                         context: Optional[Dict[str, Any]] = None) -> List[ImprovementSuggestion]:
        """
        生成全面的改进建议
        
        Args:
            results: 评估结果
            context: 额外上下文信息
            
        Returns:
            List[ImprovementSuggestion]: 改进建议列表
        """
        try:
            suggestions = []
            context = context or {}
            
            # 1. 诊断问题
            issues = self._diagnose_issues(results)
            
            # 2. 为每个问题生成建议
            for issue in issues:
                issue_suggestions = self._generate_suggestions_for_issue(issue, results, context)
                suggestions.extend(issue_suggestions)
            
            # 3. 生成最佳实践建议
            best_practice_suggestions = self._generate_best_practice_suggestions(results, context)
            suggestions.extend(best_practice_suggestions)
            
            # 4. 个性化建议
            personalized_suggestions = self._generate_personalized_suggestions(results, context)
            suggestions.extend(personalized_suggestions)
            
            # 5. 去重和排序
            suggestions = self._deduplicate_and_prioritize(suggestions)
            
            self.logger.info(f"生成了 {len(suggestions)} 条改进建议")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"改进建议生成失败: {str(e)}")
            raise ImprovementAnalysisError(f"建议生成失败: {str(e)}")
    
    def generate_batch_suggestions(self, 
                                 results: BatchEvaluationResult,
                                 context: Optional[Dict[str, Any]] = None) -> List[ImprovementSuggestion]:
        """
        为批量评估结果生成改进建议
        
        Args:
            results: 批量评估结果
            context: 额外上下文信息
            
        Returns:
            List[ImprovementSuggestion]: 改进建议列表
        """
        try:
            suggestions = []
            context = context or {}
            
            # 1. 分析批量结果模式
            patterns = self._analyze_batch_patterns(results)
            
            # 2. 为每个模式生成建议
            for pattern in patterns:
                pattern_suggestions = self._generate_suggestions_for_pattern(pattern, results, context)
                suggestions.extend(pattern_suggestions)
            
            # 3. 系统级建议
            system_suggestions = self._generate_system_level_suggestions(results, context)
            suggestions.extend(system_suggestions)
            
            # 4. 去重和排序
            suggestions = self._deduplicate_and_prioritize(suggestions)
            
            self.logger.info(f"为批量结果生成了 {len(suggestions)} 条改进建议")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"批量改进建议生成失败: {str(e)}")
            raise ImprovementAnalysisError(f"批量建议生成失败: {str(e)}")
    
    def _diagnose_issues(self, results: ExpertEvaluationResult) -> List[Dict[str, Any]]:
        """诊断评估结果中的问题"""
        issues = []
        
        # 1. 总体性能问题
        if results.overall_score < 0.5:
            issues.append({
                "type": "overall_performance",
                "severity": "critical",
                "description": "模型整体性能严重不足",
                "affected_dimensions": list(results.dimension_scores.keys()),
                "score": results.overall_score
            })
        elif results.overall_score < 0.7:
            issues.append({
                "type": "overall_performance",
                "severity": "high",
                "description": "模型整体性能需要改进",
                "affected_dimensions": list(results.dimension_scores.keys()),
                "score": results.overall_score
            })
        
        # 2. 维度特定问题
        for dim, score_obj in results.dimension_scores.items():
            if score_obj.score < 0.6:
                issues.append({
                    "type": "dimension_performance",
                    "severity": "high" if score_obj.score < 0.4 else "medium",
                    "description": f"{dim.value}维度表现不佳",
                    "affected_dimensions": [dim],
                    "score": score_obj.score,
                    "confidence": score_obj.confidence
                })
        
        # 3. 置信度问题
        avg_confidence = results.get_average_confidence()
        if avg_confidence < 0.6:
            issues.append({
                "type": "confidence",
                "severity": "medium",
                "description": "评估置信度较低",
                "affected_dimensions": list(results.dimension_scores.keys()),
                "confidence": avg_confidence
            })
        
        # 4. 一致性问题
        scores = [score_obj.score for score_obj in results.dimension_scores.values()]
        if len(scores) > 1:
            score_std = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
            if score_std > 0.3:
                issues.append({
                    "type": "consistency",
                    "severity": "medium",
                    "description": "各维度表现不一致",
                    "affected_dimensions": list(results.dimension_scores.keys()),
                    "std_deviation": score_std
                })
        
        return issues
    
    def _generate_suggestions_for_issue(self, 
                                      issue: Dict[str, Any], 
                                      results: ExpertEvaluationResult,
                                      context: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """为特定问题生成建议"""
        suggestions = []
        issue_type = issue["type"]
        severity = issue["severity"]
        
        # 根据问题类型选择模板
        templates = self.suggestion_templates.get(issue_type, [])
        
        for template in templates:
            if self._is_template_applicable(template, issue, results, context):
                suggestion = self._create_suggestion_from_template(template, issue, results, context)
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_best_practice_suggestions(self, 
                                          results: ExpertEvaluationResult,
                                          context: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """生成最佳实践建议"""
        suggestions = []
        
        # 基于当前性能水平推荐最佳实践
        performance_level = self._get_performance_level(results.overall_score)
        
        applicable_practices = self.best_practices.get(performance_level, [])
        
        for practice in applicable_practices:
            if self._is_practice_applicable(practice, results, context):
                suggestion = self._create_suggestion_from_practice(practice, results, context)
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_personalized_suggestions(self, 
                                         results: ExpertEvaluationResult,
                                         context: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """生成个性化建议"""
        suggestions = []
        
        # 基于用户偏好和历史数据的个性化建议
        user_preferences = context.get("user_preferences", {})
        historical_data = context.get("historical_data", {})
        
        # 1. 基于用户关注的维度
        priority_dimensions = user_preferences.get("priority_dimensions", [])
        for dim_name in priority_dimensions:
            try:
                dim = EvaluationDimension(dim_name)
                if dim in results.dimension_scores:
                    score_obj = results.dimension_scores[dim]
                    if score_obj.score < 0.8:  # 有改进空间
                        suggestion = self._create_personalized_dimension_suggestion(dim, score_obj, context)
                        suggestions.append(suggestion)
            except ValueError:
                continue
        
        # 2. 基于历史改进效果
        if historical_data:
            effective_strategies = historical_data.get("effective_strategies", [])
            for strategy in effective_strategies:
                if self._is_strategy_applicable(strategy, results):
                    suggestion = self._create_suggestion_from_strategy(strategy, results, context)
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _analyze_batch_patterns(self, results: BatchEvaluationResult) -> List[Dict[str, Any]]:
        """分析批量结果中的模式"""
        patterns = []
        stats = results.summary_statistics
        
        # 1. 分数分布模式
        score_std = stats.get("score_std", 0.0)
        if score_std > 0.3:
            patterns.append({
                "type": "high_variance",
                "description": "评分分布差异较大",
                "severity": "medium",
                "metrics": {"std_deviation": score_std}
            })
        
        # 2. 维度表现模式
        dimension_stats = stats.get("dimension_statistics", {})
        weak_dimensions = []
        strong_dimensions = []
        
        for dim_name, dim_stats in dimension_stats.items():
            avg_score = dim_stats.get("average", 0.0)
            if avg_score < 0.6:
                weak_dimensions.append(dim_name)
            elif avg_score > 0.8:
                strong_dimensions.append(dim_name)
        
        if weak_dimensions:
            patterns.append({
                "type": "weak_dimensions",
                "description": f"多个维度表现较弱: {', '.join(weak_dimensions)}",
                "severity": "high",
                "affected_dimensions": weak_dimensions
            })
        
        # 3. 性能分布模式
        performance_dist = stats.get("performance_distribution", {})
        need_improvement_ratio = performance_dist.get("需要改进", 0) / stats.get("total_items", 1)
        
        if need_improvement_ratio > 0.3:
            patterns.append({
                "type": "high_failure_rate",
                "description": "大量项目需要改进",
                "severity": "critical",
                "metrics": {"failure_rate": need_improvement_ratio}
            })
        
        return patterns
    
    def _generate_suggestions_for_pattern(self, 
                                        pattern: Dict[str, Any], 
                                        results: BatchEvaluationResult,
                                        context: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """为特定模式生成建议"""
        suggestions = []
        pattern_type = pattern["type"]
        
        if pattern_type == "high_variance":
            suggestions.append(self._create_variance_reduction_suggestion(pattern, results, context))
        elif pattern_type == "weak_dimensions":
            suggestions.extend(self._create_dimension_improvement_suggestions(pattern, results, context))
        elif pattern_type == "high_failure_rate":
            suggestions.append(self._create_systematic_improvement_suggestion(pattern, results, context))
        
        return suggestions
    
    def _generate_system_level_suggestions(self, 
                                         results: BatchEvaluationResult,
                                         context: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """生成系统级改进建议"""
        suggestions = []
        
        # 基于整体统计生成系统级建议
        stats = results.summary_statistics
        avg_score = stats.get("average_score", 0.0)
        
        if avg_score < 0.7:
            suggestions.append(ImprovementSuggestion(
                id=self._generate_suggestion_id(),
                title="系统性能优化",
                description="整体系统性能需要全面优化",
                category=ImprovementCategory.TRAINING_STRATEGY,
                priority=ImprovementPriority.HIGH,
                affected_dimensions=list(EvaluationDimension),
                implementation_steps=[
                    "重新评估训练数据质量",
                    "调整模型架构参数",
                    "优化训练超参数",
                    "增加数据增强策略",
                    "实施渐进式训练"
                ],
                expected_impact="预期提升整体性能15-25%",
                effort_level="高",
                time_estimate="4-6周",
                prerequisites=["充足的计算资源", "高质量训练数据"],
                success_metrics=["整体评分提升至0.75以上", "各维度均衡发展"],
                related_suggestions=[]
            ))
        
        return suggestions    

    def _create_suggestion_from_template(self, 
                                       template: Dict[str, Any], 
                                       issue: Dict[str, Any],
                                       results: ExpertEvaluationResult,
                                       context: Dict[str, Any]) -> ImprovementSuggestion:
        """从模板创建建议"""
        return ImprovementSuggestion(
            id=self._generate_suggestion_id(),
            title=template["title"].format(**issue),
            description=template["description"].format(**issue),
            category=ImprovementCategory(template["category"]),
            priority=ImprovementPriority(template["priority"]),
            affected_dimensions=[EvaluationDimension(dim) for dim in template["affected_dimensions"]],
            implementation_steps=template["implementation_steps"],
            expected_impact=template["expected_impact"],
            effort_level=template["effort_level"],
            time_estimate=template["time_estimate"],
            prerequisites=template["prerequisites"],
            success_metrics=template["success_metrics"],
            related_suggestions=[]
        )
    
    def _create_suggestion_from_practice(self, 
                                       practice: Dict[str, Any],
                                       results: ExpertEvaluationResult,
                                       context: Dict[str, Any]) -> ImprovementSuggestion:
        """从最佳实践创建建议"""
        return ImprovementSuggestion(
            id=self._generate_suggestion_id(),
            title=practice["title"],
            description=practice["description"],
            category=ImprovementCategory(practice["category"]),
            priority=ImprovementPriority(practice["priority"]),
            affected_dimensions=[EvaluationDimension(dim) for dim in practice["affected_dimensions"]],
            implementation_steps=practice["implementation_steps"],
            expected_impact=practice["expected_impact"],
            effort_level=practice["effort_level"],
            time_estimate=practice["time_estimate"],
            prerequisites=practice["prerequisites"],
            success_metrics=practice["success_metrics"],
            related_suggestions=[]
        )
    
    def _create_personalized_dimension_suggestion(self, 
                                                dimension: EvaluationDimension,
                                                score_obj: DimensionScore,
                                                context: Dict[str, Any]) -> ImprovementSuggestion:
        """创建个性化维度建议"""
        dimension_templates = {
            EvaluationDimension.SEMANTIC_SIMILARITY: {
                "title": "提升语义相似性表现",
                "description": f"当前{dimension.value}评分为{score_obj.score:.2f}，建议通过语义增强训练提升",
                "steps": [
                    "增加同义词和近义词训练数据",
                    "使用预训练语义模型进行知识蒸馏",
                    "实施对比学习训练策略",
                    "优化词向量表示"
                ]
            },
            EvaluationDimension.DOMAIN_ACCURACY: {
                "title": "增强领域准确性",
                "description": f"当前{dimension.value}评分为{score_obj.score:.2f}，建议加强专业领域训练",
                "steps": [
                    "收集更多领域专业数据",
                    "构建领域知识图谱",
                    "实施领域适应性训练",
                    "增加专家标注数据"
                ]
            },
            EvaluationDimension.FACTUAL_CORRECTNESS: {
                "title": "提高事实正确性",
                "description": f"当前{dimension.value}评分为{score_obj.score:.2f}，建议加强事实验证能力",
                "steps": [
                    "集成外部知识库",
                    "实施事实检查机制",
                    "增加可信数据源",
                    "训练事实验证模型"
                ]
            }
        }
        
        template = dimension_templates.get(dimension, {
            "title": f"改进{dimension.value}",
            "description": f"当前{dimension.value}评分为{score_obj.score:.2f}，需要针对性改进",
            "steps": ["分析具体问题", "制定改进策略", "实施优化方案", "验证改进效果"]
        })
        
        return ImprovementSuggestion(
            id=self._generate_suggestion_id(),
            title=template["title"],
            description=template["description"],
            category=ImprovementCategory.DOMAIN_SPECIFIC,
            priority=ImprovementPriority.HIGH if score_obj.score < 0.5 else ImprovementPriority.MEDIUM,
            affected_dimensions=[dimension],
            implementation_steps=template["steps"],
            expected_impact=f"预期提升{dimension.value}评分10-20%",
            effort_level="中" if score_obj.score > 0.4 else "高",
            time_estimate="2-4周",
            prerequisites=["相关训练数据", "领域专家支持"],
            success_metrics=[f"{dimension.value}评分提升至0.75以上"],
            related_suggestions=[]
        )
    
    def _create_variance_reduction_suggestion(self, 
                                            pattern: Dict[str, Any],
                                            results: BatchEvaluationResult,
                                            context: Dict[str, Any]) -> ImprovementSuggestion:
        """创建方差减少建议"""
        return ImprovementSuggestion(
            id=self._generate_suggestion_id(),
            title="减少评估结果方差",
            description="评估结果分布差异较大，建议提升模型一致性",
            category=ImprovementCategory.TRAINING_STRATEGY,
            priority=ImprovementPriority.MEDIUM,
            affected_dimensions=list(EvaluationDimension),
            implementation_steps=[
                "分析高方差原因",
                "标准化训练数据格式",
                "实施一致性正则化",
                "增加模型集成策略",
                "优化训练稳定性"
            ],
            expected_impact="预期减少结果方差30-50%",
            effort_level="中",
            time_estimate="3-4周",
            prerequisites=["稳定的训练环境", "一致的数据预处理"],
            success_metrics=["评分标准差降至0.2以下", "结果可重现性提升"],
            related_suggestions=[]
        )
    
    def _create_dimension_improvement_suggestions(self, 
                                                pattern: Dict[str, Any],
                                                results: BatchEvaluationResult,
                                                context: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """创建维度改进建议"""
        suggestions = []
        weak_dimensions = pattern.get("affected_dimensions", [])
        
        for dim_name in weak_dimensions:
            try:
                dimension = EvaluationDimension(dim_name)
                suggestion = ImprovementSuggestion(
                    id=self._generate_suggestion_id(),
                    title=f"专项改进{dim_name}",
                    description=f"{dim_name}维度在批量评估中表现较弱，需要专项优化",
                    category=ImprovementCategory.DOMAIN_SPECIFIC,
                    priority=ImprovementPriority.HIGH,
                    affected_dimensions=[dimension],
                    implementation_steps=[
                        f"深入分析{dim_name}失败案例",
                        f"收集{dim_name}相关训练数据",
                        f"设计{dim_name}专项训练任务",
                        f"实施{dim_name}增强训练",
                        f"验证{dim_name}改进效果"
                    ],
                    expected_impact=f"预期{dim_name}维度提升20-30%",
                    effort_level="高",
                    time_estimate="4-6周",
                    prerequisites=[f"{dim_name}领域专业知识", "充足的训练数据"],
                    success_metrics=[f"{dim_name}平均分提升至0.7以上"],
                    related_suggestions=[]
                )
                suggestions.append(suggestion)
            except ValueError:
                continue
        
        return suggestions
    
    def _create_systematic_improvement_suggestion(self, 
                                                pattern: Dict[str, Any],
                                                results: BatchEvaluationResult,
                                                context: Dict[str, Any]) -> ImprovementSuggestion:
        """创建系统性改进建议"""
        failure_rate = pattern.get("metrics", {}).get("failure_rate", 0.0)
        
        return ImprovementSuggestion(
            id=self._generate_suggestion_id(),
            title="系统性质量改进",
            description=f"当前有{failure_rate:.1%}的项目需要改进，建议实施系统性质量提升",
            category=ImprovementCategory.TRAINING_STRATEGY,
            priority=ImprovementPriority.CRITICAL,
            affected_dimensions=list(EvaluationDimension),
            implementation_steps=[
                "全面审查训练流程",
                "重新设计数据收集策略",
                "实施多阶段质量控制",
                "建立持续改进机制",
                "加强模型验证流程"
            ],
            expected_impact="预期整体质量提升40-60%",
            effort_level="高",
            time_estimate="6-8周",
            prerequisites=["管理层支持", "充足资源投入", "跨团队协作"],
            success_metrics=["失败率降至15%以下", "整体平均分提升至0.75以上"],
            related_suggestions=[]
        )
    
    def _deduplicate_and_prioritize(self, suggestions: List[ImprovementSuggestion]) -> List[ImprovementSuggestion]:
        """去重和优先级排序"""
        # 简单的去重逻辑（基于标题）
        seen_titles = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.title not in seen_titles:
                seen_titles.add(suggestion.title)
                unique_suggestions.append(suggestion)
        
        # 按优先级排序
        priority_order = {
            ImprovementPriority.CRITICAL: 4,
            ImprovementPriority.HIGH: 3,
            ImprovementPriority.MEDIUM: 2,
            ImprovementPriority.LOW: 1
        }
        
        unique_suggestions.sort(key=lambda x: priority_order[x.priority], reverse=True)
        
        return unique_suggestions[:15]  # 最多返回15条建议
    
    def _generate_suggestion_id(self) -> str:
        """生成建议ID"""
        self._suggestion_counter += 1
        return f"suggestion_{self._suggestion_counter:04d}"
    
    def _get_performance_level(self, score: float) -> str:
        """获取性能水平"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "average"
        else:
            return "poor"
    
    def _is_template_applicable(self, 
                              template: Dict[str, Any], 
                              issue: Dict[str, Any],
                              results: ExpertEvaluationResult,
                              context: Dict[str, Any]) -> bool:
        """检查模板是否适用"""
        # 简化的适用性检查
        template_severity = template.get("min_severity", "low")
        issue_severity = issue.get("severity", "low")
        
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        return severity_levels.get(issue_severity, 1) >= severity_levels.get(template_severity, 1)
    
    def _is_practice_applicable(self, 
                              practice: Dict[str, Any],
                              results: ExpertEvaluationResult,
                              context: Dict[str, Any]) -> bool:
        """检查最佳实践是否适用"""
        # 检查性能水平匹配
        required_level = practice.get("performance_level", "any")
        current_level = self._get_performance_level(results.overall_score)
        
        if required_level != "any" and required_level != current_level:
            return False
        
        # 检查维度要求
        required_dimensions = practice.get("required_dimensions", [])
        available_dimensions = list(results.dimension_scores.keys())
        
        for req_dim_name in required_dimensions:
            try:
                req_dim = EvaluationDimension(req_dim_name)
                if req_dim not in available_dimensions:
                    return False
            except ValueError:
                return False
        
        return True
    
    def _is_strategy_applicable(self, 
                              strategy: Dict[str, Any],
                              results: ExpertEvaluationResult) -> bool:
        """检查策略是否适用"""
        # 检查策略的适用条件
        min_score = strategy.get("min_score", 0.0)
        max_score = strategy.get("max_score", 1.0)
        
        return min_score <= results.overall_score <= max_score
    
    def _create_suggestion_from_strategy(self, 
                                       strategy: Dict[str, Any],
                                       results: ExpertEvaluationResult,
                                       context: Dict[str, Any]) -> ImprovementSuggestion:
        """从策略创建建议"""
        return ImprovementSuggestion(
            id=self._generate_suggestion_id(),
            title=strategy["title"],
            description=strategy["description"],
            category=ImprovementCategory(strategy["category"]),
            priority=ImprovementPriority(strategy["priority"]),
            affected_dimensions=[EvaluationDimension(dim) for dim in strategy["affected_dimensions"]],
            implementation_steps=strategy["implementation_steps"],
            expected_impact=strategy["expected_impact"],
            effort_level=strategy["effort_level"],
            time_estimate=strategy["time_estimate"],
            prerequisites=strategy["prerequisites"],
            success_metrics=strategy["success_metrics"],
            related_suggestions=[]
        )
    
    def _load_suggestion_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载建议模板"""
        return {
            "overall_performance": [
                {
                    "title": "全面性能优化",
                    "description": "模型整体性能{description}，建议进行全面优化",
                    "category": "训练策略",
                    "priority": "关键",
                    "min_severity": "critical",
                    "affected_dimensions": ["语义相似性", "领域准确性", "响应相关性"],
                    "implementation_steps": [
                        "重新评估训练数据质量",
                        "调整模型架构",
                        "优化训练超参数",
                        "增加数据增强",
                        "实施多阶段训练"
                    ],
                    "expected_impact": "预期整体性能提升25-40%",
                    "effort_level": "高",
                    "time_estimate": "6-8周",
                    "prerequisites": ["充足计算资源", "高质量数据"],
                    "success_metrics": ["整体评分提升至0.75以上"]
                }
            ],
            "dimension_performance": [
                {
                    "title": "维度专项改进",
                    "description": "{description}，需要针对性优化",
                    "category": "领域特定",
                    "priority": "高",
                    "min_severity": "medium",
                    "affected_dimensions": [],  # 动态填充
                    "implementation_steps": [
                        "分析维度失败原因",
                        "收集相关训练数据",
                        "设计专项训练任务",
                        "实施增强训练",
                        "验证改进效果"
                    ],
                    "expected_impact": "预期该维度提升15-25%",
                    "effort_level": "中",
                    "time_estimate": "3-4周",
                    "prerequisites": ["领域专业知识"],
                    "success_metrics": ["目标维度评分提升至0.7以上"]
                }
            ],
            "confidence": [
                {
                    "title": "提升评估置信度",
                    "description": "评估置信度较低，建议增强数据质量",
                    "category": "数据质量",
                    "priority": "中",
                    "min_severity": "medium",
                    "affected_dimensions": ["语义相似性", "事实正确性"],
                    "implementation_steps": [
                        "增加评估数据量",
                        "提高数据标注质量",
                        "实施交叉验证",
                        "优化评估方法"
                    ],
                    "expected_impact": "预期置信度提升20-30%",
                    "effort_level": "中",
                    "time_estimate": "2-3周",
                    "prerequisites": ["高质量标注数据"],
                    "success_metrics": ["平均置信度提升至0.8以上"]
                }
            ]
        }
    
    def _load_diagnostic_rules(self) -> Dict[str, Any]:
        """加载诊断规则"""
        return {
            "score_thresholds": {
                "critical": 0.3,
                "high": 0.5,
                "medium": 0.7,
                "low": 0.9
            },
            "confidence_thresholds": {
                "low": 0.6,
                "medium": 0.8,
                "high": 0.9
            },
            "consistency_thresholds": {
                "high_variance": 0.3,
                "medium_variance": 0.2,
                "low_variance": 0.1
            }
        }
    
    def _load_best_practices(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载最佳实践"""
        return {
            "poor": [
                {
                    "title": "基础质量保证",
                    "description": "建立基础的质量保证流程",
                    "category": "评估方法",
                    "priority": "高",
                    "performance_level": "poor",
                    "affected_dimensions": ["语义相似性", "事实正确性"],
                    "implementation_steps": [
                        "建立数据质量检查流程",
                        "实施基础模型验证",
                        "设置性能基准线",
                        "建立持续监控机制"
                    ],
                    "expected_impact": "建立稳定的质量基线",
                    "effort_level": "中",
                    "time_estimate": "2-3周",
                    "prerequisites": ["基础设施支持"],
                    "success_metrics": ["建立完整的质量流程"]
                }
            ],
            "average": [
                {
                    "title": "性能优化策略",
                    "description": "实施系统性的性能优化",
                    "category": "训练策略",
                    "priority": "中",
                    "performance_level": "average",
                    "affected_dimensions": ["领域准确性", "逻辑一致性"],
                    "implementation_steps": [
                        "优化训练数据分布",
                        "调整模型超参数",
                        "实施渐进式训练",
                        "增加模型集成"
                    ],
                    "expected_impact": "预期性能提升15-20%",
                    "effort_level": "中",
                    "time_estimate": "4-5周",
                    "prerequisites": ["优化经验"],
                    "success_metrics": ["整体性能突破0.75"]
                }
            ],
            "good": [
                {
                    "title": "精细化调优",
                    "description": "进行精细化的模型调优",
                    "category": "模型架构",
                    "priority": "中",
                    "performance_level": "good",
                    "affected_dimensions": ["创新性", "实用价值"],
                    "implementation_steps": [
                        "分析性能瓶颈",
                        "实施精细调参",
                        "优化推理效率",
                        "增强模型鲁棒性"
                    ],
                    "expected_impact": "预期性能提升5-10%",
                    "effort_level": "低",
                    "time_estimate": "2-3周",
                    "prerequisites": ["深度调优经验"],
                    "success_metrics": ["达到优秀性能水平"]
                }
            ]
        }


def test_improvement_suggestions():
    """测试改进建议生成功能"""
    try:
        from .config import EvaluationDimension, ExpertiseLevel
        from .data_models import ExpertEvaluationResult, DimensionScore
        
        # 创建测试数据
        dimension_scores = {
            EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                score=0.4,  # 低分
                confidence=0.7
            ),
            EvaluationDimension.DOMAIN_ACCURACY: DimensionScore(
                dimension=EvaluationDimension.DOMAIN_ACCURACY,
                score=0.8,  # 高分
                confidence=0.9
            ),
            EvaluationDimension.FACTUAL_CORRECTNESS: DimensionScore(
                dimension=EvaluationDimension.FACTUAL_CORRECTNESS,
                score=0.6,  # 中等分
                confidence=0.5  # 低置信度
            )
        }
        
        test_result = ExpertEvaluationResult(
            question_id="test_improvement_001",
            overall_score=0.6,
            dimension_scores=dimension_scores
        )
        
        # 测试改进建议生成器
        advisor = ImprovementAdvisor()
        
        # 生成改进建议
        suggestions = advisor.generate_comprehensive_suggestions(test_result)
        
        print(f"生成了 {len(suggestions)} 条改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion.title}")
            print(f"   类别: {suggestion.category.value}")
            print(f"   优先级: {suggestion.priority.value}")
            print(f"   描述: {suggestion.description}")
            print(f"   预期影响: {suggestion.expected_impact}")
            print(f"   实施步骤: {len(suggestion.implementation_steps)} 步")
        
        # 验证建议质量
        assert len(suggestions) > 0, "应该生成至少一条建议"
        assert any(s.priority == ImprovementPriority.HIGH for s in suggestions), "应该有高优先级建议"
        
        print("\n✓ 改进建议生成功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 改进建议生成功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_improvement_suggestions()