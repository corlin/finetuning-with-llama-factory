"""
专家评估报告生成器

实现详细评估报告的生成功能，支持多种输出格式（JSON、HTML、PDF），
包含评估结果的可视化图表生成和改进建议生成功能。
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import base64
from io import BytesIO

from .interfaces import EvaluationReportGenerator as ReportGeneratorInterface
from .data_models import (
    ExpertEvaluationResult, 
    BatchEvaluationResult, 
    EvaluationReport,
    DimensionScore
)
from .config import EvaluationDimension, ExpertEvaluationConfig
from .exceptions import ReportGenerationError
from .improvement_advisor import ImprovementAdvisor


class EvaluationReportGenerator(ReportGeneratorInterface):
    """评估报告生成器实现类"""
    
    def __init__(self, config: Optional[ExpertEvaluationConfig] = None):
        """
        初始化报告生成器
        
        Args:
            config: 专家评估配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 支持的输出格式
        self.supported_formats = ["json", "html", "pdf"]
        
        # 图表配置
        self.chart_config = {
            "width": 800,
            "height": 600,
            "colors": {
                "primary": "#2E86AB",
                "secondary": "#A23B72", 
                "success": "#F18F01",
                "warning": "#C73E1D",
                "info": "#6C757D"
            }
        }
        
        # 改进建议模板库
        self.improvement_templates = self._load_improvement_templates()
        
        # 改进建议生成器
        self.improvement_advisor = ImprovementAdvisor()
    
    def generate_detailed_report(self, 
                               results: Union[ExpertEvaluationResult, BatchEvaluationResult],
                               format: str = "json") -> EvaluationReport:
        """
        生成详细评估报告
        
        Args:
            results: 评估结果
            format: 报告格式 (json, html, pdf)
            
        Returns:
            EvaluationReport: 评估报告
        """
        try:
            if format not in self.supported_formats:
                raise ReportGenerationError(f"不支持的报告格式: {format}")
            
            # 生成报告ID和标题
            report_id = self._generate_report_id(results)
            title = self._generate_report_title(results)
            summary = self._generate_report_summary(results)
            
            # 创建基础报告对象
            report = EvaluationReport(
                report_id=report_id,
                title=title,
                summary=summary,
                evaluation_results=results,
                format_type=format
            )
            
            # 生成图表数据
            charts_data = self.create_visualization_charts(results)
            for chart_name, chart_data in charts_data.items():
                report.add_chart(chart_name, chart_data)
            
            # 生成改进建议
            suggestions = self.generate_improvement_suggestions(results)
            for suggestion in suggestions:
                if isinstance(suggestion, str):
                    report.add_recommendation(suggestion)
                else:
                    # 如果是ImprovementSuggestion对象，转换为字符串
                    report.add_recommendation(f"{suggestion.title}: {suggestion.description}")
            
            # 生成详细分析
            report.detailed_analysis = self._generate_detailed_analysis(results)
            
            self.logger.info(f"成功生成{format}格式的评估报告: {report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"生成评估报告失败: {str(e)}")
            raise ReportGenerationError(f"报告生成失败: {str(e)}")
    
    def create_visualization_charts(self, 
                                  results: Union[ExpertEvaluationResult, BatchEvaluationResult]) -> Dict[str, Any]:
        """
        创建可视化图表
        
        Args:
            results: 评估结果
            
        Returns:
            Dict[str, Any]: 图表数据
        """
        charts = {}
        
        try:
            if isinstance(results, ExpertEvaluationResult):
                charts.update(self._create_single_result_charts(results))
            elif isinstance(results, BatchEvaluationResult):
                charts.update(self._create_batch_result_charts(results))
            
            self.logger.info(f"成功生成 {len(charts)} 个图表")
            return charts
            
        except Exception as e:
            self.logger.error(f"图表生成失败: {str(e)}")
            return {}
    
    def generate_improvement_suggestions(self, 
                                       results: Union[ExpertEvaluationResult, BatchEvaluationResult]) -> List[str]:
        """
        生成改进建议
        
        Args:
            results: 评估结果
            
        Returns:
            List[str]: 改进建议列表
        """
        suggestions = []
        
        try:
            # 使用改进建议生成器生成详细建议
            if isinstance(results, ExpertEvaluationResult):
                detailed_suggestions = self.improvement_advisor.generate_comprehensive_suggestions(results)
                # 转换为简单字符串格式
                for suggestion in detailed_suggestions:
                    suggestions.append(f"[{suggestion.priority.value}] {suggestion.title}: {suggestion.description}")
                
                # 添加传统建议作为补充
                traditional_suggestions = self._generate_single_result_suggestions(results)
                suggestions.extend(traditional_suggestions)
                
            elif isinstance(results, BatchEvaluationResult):
                detailed_suggestions = self.improvement_advisor.generate_batch_suggestions(results)
                # 转换为简单字符串格式
                for suggestion in detailed_suggestions:
                    suggestions.append(f"[{suggestion.priority.value}] {suggestion.title}: {suggestion.description}")
                
                # 添加传统建议作为补充
                traditional_suggestions = self._generate_batch_result_suggestions(results)
                suggestions.extend(traditional_suggestions)
            
            # 去重并排序
            suggestions = list(set(suggestions))
            suggestions.sort(key=lambda x: self._get_suggestion_priority(x), reverse=True)
            
            self.logger.info(f"生成了 {len(suggestions)} 条改进建议")
            return suggestions[:15]  # 最多返回15条建议
            
        except Exception as e:
            self.logger.error(f"改进建议生成失败: {str(e)}")
            return ["建议进行更详细的人工评估以获得具体改进方向"]
    
    def export_report(self, 
                     report: EvaluationReport, 
                     output_path: str,
                     format: str = "json") -> bool:
        """
        导出报告到文件
        
        Args:
            report: 评估报告
            output_path: 输出路径
            format: 导出格式
            
        Returns:
            bool: 导出是否成功
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                return self._export_json_report(report, output_path)
            elif format == "html":
                return self._export_html_report(report, output_path)
            elif format == "pdf":
                return self._export_pdf_report(report, output_path)
            else:
                raise ReportGenerationError(f"不支持的导出格式: {format}")
                
        except Exception as e:
            self.logger.error(f"报告导出失败: {str(e)}")
            return False
    
    def _generate_report_id(self, results: Union[ExpertEvaluationResult, BatchEvaluationResult]) -> str:
        """生成报告ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(results, ExpertEvaluationResult):
            return f"expert_eval_{results.question_id}_{timestamp}"
        elif isinstance(results, BatchEvaluationResult):
            return f"batch_eval_{results.batch_id}_{timestamp}"
        else:
            return f"eval_report_{timestamp}"
    
    def _generate_report_title(self, results: Union[ExpertEvaluationResult, BatchEvaluationResult]) -> str:
        """生成报告标题"""
        if isinstance(results, ExpertEvaluationResult):
            category = results.get_performance_category()
            return f"专家评估报告 - {category} (问题ID: {results.question_id})"
        elif isinstance(results, BatchEvaluationResult):
            total_items = len(results.individual_results)
            avg_score = results.summary_statistics.get("average_score", 0.0)
            return f"批量评估报告 - {total_items}项评估 (平均分: {avg_score:.2f})"
        else:
            return "专家评估报告"
    
    def _generate_report_summary(self, results: Union[ExpertEvaluationResult, BatchEvaluationResult]) -> str:
        """生成报告摘要"""
        if isinstance(results, ExpertEvaluationResult):
            return (f"本报告针对问题ID {results.question_id} 进行了专家级评估，"
                   f"总体评分为 {results.overall_score:.2f}，"
                   f"性能类别为 {results.get_performance_category()}。"
                   f"评估涵盖了 {len(results.dimension_scores)} 个维度，"
                   f"平均置信度为 {results.get_average_confidence():.2f}。")
        elif isinstance(results, BatchEvaluationResult):
            stats = results.summary_statistics
            total_items = stats.get("total_items", 0)
            avg_score = stats.get("average_score", 0.0)
            return (f"本报告对 {total_items} 个评估项目进行了批量专家评估，"
                   f"平均评分为 {avg_score:.2f}，"
                   f"评分范围为 {stats.get('min_score', 0.0):.2f} - {stats.get('max_score', 0.0):.2f}。"
                   f"评估总耗时 {results.total_processing_time:.2f} 秒。")
        else:
            return "专家评估系统生成的详细评估报告"    

    def _create_single_result_charts(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """为单个评估结果创建图表"""
        charts = {}
        
        # 1. 维度评分雷达图
        charts["dimension_radar"] = self._create_dimension_radar_chart(result)
        
        # 2. 维度评分柱状图
        charts["dimension_bar"] = self._create_dimension_bar_chart(result)
        
        # 3. 置信度分布图
        charts["confidence_distribution"] = self._create_confidence_chart(result)
        
        # 4. 性能类别饼图
        charts["performance_pie"] = self._create_performance_pie_chart(result)
        
        return charts
    
    def _create_batch_result_charts(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """为批量评估结果创建图表"""
        charts = {}
        
        # 1. 评分分布直方图
        charts["score_distribution"] = self._create_score_distribution_chart(results)
        
        # 2. 维度平均分对比图
        charts["dimension_comparison"] = self._create_dimension_comparison_chart(results)
        
        # 3. 性能类别分布图
        charts["performance_distribution"] = self._create_performance_distribution_chart(results)
        
        # 4. 处理时间趋势图
        charts["processing_time_trend"] = self._create_processing_time_chart(results)
        
        return charts
    
    def _create_dimension_radar_chart(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """创建维度评分雷达图"""
        dimensions = []
        scores = []
        
        for dim, score_obj in result.dimension_scores.items():
            dimensions.append(dim.value)
            scores.append(score_obj.score)
        
        return {
            "type": "radar",
            "title": "各维度评分雷达图",
            "data": {
                "labels": dimensions,
                "datasets": [{
                    "label": "评分",
                    "data": scores,
                    "backgroundColor": "rgba(46, 134, 171, 0.2)",
                    "borderColor": self.chart_config["colors"]["primary"],
                    "pointBackgroundColor": self.chart_config["colors"]["primary"]
                }]
            },
            "options": {
                "scale": {
                    "ticks": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "stepSize": 0.2
                    }
                }
            }
        }
    
    def _create_dimension_bar_chart(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """创建维度评分柱状图"""
        dimensions = []
        scores = []
        confidences = []
        
        for dim, score_obj in result.dimension_scores.items():
            dimensions.append(dim.value)
            scores.append(score_obj.score)
            confidences.append(score_obj.confidence)
        
        return {
            "type": "bar",
            "title": "各维度评分对比",
            "data": {
                "labels": dimensions,
                "datasets": [
                    {
                        "label": "评分",
                        "data": scores,
                        "backgroundColor": self.chart_config["colors"]["primary"],
                        "yAxisID": "y"
                    },
                    {
                        "label": "置信度",
                        "data": confidences,
                        "backgroundColor": self.chart_config["colors"]["secondary"],
                        "yAxisID": "y1"
                    }
                ]
            },
            "options": {
                "scales": {
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "max": 1.0
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "max": 1.0
                    }
                }
            }
        }
    
    def _create_confidence_chart(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """创建置信度分布图"""
        dimensions = []
        confidences = []
        
        for dim, score_obj in result.dimension_scores.items():
            dimensions.append(dim.value)
            confidences.append(score_obj.confidence)
        
        return {
            "type": "line",
            "title": "各维度置信度分布",
            "data": {
                "labels": dimensions,
                "datasets": [{
                    "label": "置信度",
                    "data": confidences,
                    "borderColor": self.chart_config["colors"]["success"],
                    "backgroundColor": "rgba(241, 143, 1, 0.1)",
                    "fill": True
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0
                    }
                }
            }
        }
    
    def _create_performance_pie_chart(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """创建性能类别饼图"""
        category = result.get_performance_category()
        
        # 根据评分计算各类别的占比（模拟）
        score = result.overall_score
        
        if score >= 0.9:
            data = [100, 0, 0, 0]
            labels = ["优秀", "良好", "一般", "需要改进"]
        elif score >= 0.7:
            data = [0, 100, 0, 0]
            labels = ["优秀", "良好", "一般", "需要改进"]
        elif score >= 0.5:
            data = [0, 0, 100, 0]
            labels = ["优秀", "良好", "一般", "需要改进"]
        else:
            data = [0, 0, 0, 100]
            labels = ["优秀", "良好", "一般", "需要改进"]
        
        return {
            "type": "pie",
            "title": f"性能类别: {category}",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        self.chart_config["colors"]["success"],
                        self.chart_config["colors"]["primary"],
                        self.chart_config["colors"]["warning"],
                        self.chart_config["colors"]["warning"]
                    ]
                }]
            }
        }
    
    def _create_score_distribution_chart(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """创建评分分布直方图"""
        scores = [r.overall_score for r in results.individual_results]
        
        # 创建分数区间
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_counts = [0] * (len(bins) - 1)
        bin_labels = []
        
        for i in range(len(bins) - 1):
            bin_labels.append(f"{bins[i]:.1f}-{bins[i+1]:.1f}")
            bin_counts[i] = sum(1 for score in scores if bins[i] <= score < bins[i+1])
        
        # 处理最后一个区间（包含1.0）
        bin_counts[-1] += sum(1 for score in scores if score == 1.0)
        
        return {
            "type": "bar",
            "title": "评分分布直方图",
            "data": {
                "labels": bin_labels,
                "datasets": [{
                    "label": "项目数量",
                    "data": bin_counts,
                    "backgroundColor": self.chart_config["colors"]["primary"]
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        }
    
    def _create_dimension_comparison_chart(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """创建维度平均分对比图"""
        dimension_stats = results.summary_statistics.get("dimension_statistics", {})
        
        dimensions = list(dimension_stats.keys())
        averages = [stats["average"] for stats in dimension_stats.values()]
        
        return {
            "type": "bar",
            "title": "各维度平均分对比",
            "data": {
                "labels": dimensions,
                "datasets": [{
                    "label": "平均分",
                    "data": averages,
                    "backgroundColor": self.chart_config["colors"]["secondary"]
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0
                    }
                }
            }
        }
    
    def _create_performance_distribution_chart(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """创建性能类别分布图"""
        distribution = results.summary_statistics.get("performance_distribution", {})
        
        labels = list(distribution.keys())
        data = list(distribution.values())
        
        return {
            "type": "doughnut",
            "title": "性能类别分布",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        self.chart_config["colors"]["success"],
                        self.chart_config["colors"]["primary"],
                        self.chart_config["colors"]["warning"],
                        self.chart_config["colors"]["warning"]
                    ]
                }]
            }
        }
    
    def _create_processing_time_chart(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """创建处理时间趋势图"""
        processing_times = [r.processing_time for r in results.individual_results]
        question_ids = [r.question_id for r in results.individual_results]
        
        return {
            "type": "line",
            "title": "处理时间趋势",
            "data": {
                "labels": question_ids[:20],  # 最多显示20个点
                "datasets": [{
                    "label": "处理时间(秒)",
                    "data": processing_times[:20],
                    "borderColor": self.chart_config["colors"]["info"],
                    "backgroundColor": "rgba(108, 117, 125, 0.1)",
                    "fill": True
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        }
    
    def _generate_single_result_suggestions(self, result: ExpertEvaluationResult) -> List[str]:
        """为单个评估结果生成改进建议"""
        suggestions = []
        
        # 基于总体评分的建议
        if result.overall_score < 0.5:
            suggestions.append("模型整体表现需要显著改进，建议进行全面的模型优化")
        elif result.overall_score < 0.7:
            suggestions.append("模型表现一般，建议针对薄弱维度进行专项训练")
        elif result.overall_score < 0.9:
            suggestions.append("模型表现良好，可通过微调进一步提升性能")
        
        # 基于维度评分的建议
        for dim, score_obj in result.dimension_scores.items():
            if score_obj.score < 0.6:
                template = self.improvement_templates.get(dim.name, {})
                if template:
                    suggestions.append(template.get("low_score", f"需要改进{dim.value}方面的表现"))
            elif score_obj.confidence < 0.7:
                suggestions.append(f"{dim.value}维度的评估置信度较低，建议增加相关训练数据")
        
        # 基于置信度的建议
        avg_confidence = result.get_average_confidence()
        if avg_confidence < 0.6:
            suggestions.append("整体评估置信度较低，建议使用更多样化的评估数据")
        
        return suggestions
    
    def _generate_batch_result_suggestions(self, results: BatchEvaluationResult) -> List[str]:
        """为批量评估结果生成改进建议"""
        suggestions = []
        stats = results.summary_statistics
        
        # 基于平均分的建议
        avg_score = stats.get("average_score", 0.0)
        if avg_score < 0.6:
            suggestions.append("批量评估平均分较低，建议全面检查模型训练质量")
        elif avg_score < 0.8:
            suggestions.append("批量评估结果中等，建议针对低分项目进行专项改进")
        
        # 基于分数分布的建议
        score_std = stats.get("score_std", 0.0)
        if score_std > 0.3:
            suggestions.append("评分分布差异较大，建议检查数据质量和模型一致性")
        
        # 基于性能分布的建议
        performance_dist = stats.get("performance_distribution", {})
        need_improvement_count = performance_dist.get("需要改进", 0)
        total_items = stats.get("total_items", 1)
        
        if need_improvement_count / total_items > 0.3:
            suggestions.append("超过30%的项目需要改进，建议重新评估训练策略")
        
        # 基于维度统计的建议
        dimension_stats = stats.get("dimension_statistics", {})
        for dim_name, dim_stats in dimension_stats.items():
            if dim_stats["average"] < 0.6:
                suggestions.append(f"{dim_name}维度整体表现较差，建议加强相关能力训练")
        
        return suggestions 
   
    def _get_suggestion_priority(self, suggestion: str) -> int:
        """获取建议的优先级"""
        # 根据关键词确定优先级
        high_priority_keywords = ["显著改进", "全面", "重新评估", "需要改进"]
        medium_priority_keywords = ["建议", "可通过", "针对"]
        
        suggestion_lower = suggestion.lower()
        
        for keyword in high_priority_keywords:
            if keyword in suggestion_lower:
                return 3
        
        for keyword in medium_priority_keywords:
            if keyword in suggestion_lower:
                return 2
        
        return 1
    
    def _generate_detailed_analysis(self, 
                                  results: Union[ExpertEvaluationResult, BatchEvaluationResult]) -> Dict[str, Any]:
        """生成详细分析"""
        analysis = {}
        
        if isinstance(results, ExpertEvaluationResult):
            analysis.update(self._analyze_single_result(results))
        elif isinstance(results, BatchEvaluationResult):
            analysis.update(self._analyze_batch_results(results))
        
        return analysis
    
    def _analyze_single_result(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """分析单个评估结果"""
        analysis = {
            "performance_analysis": {
                "overall_score": result.overall_score,
                "performance_category": result.get_performance_category(),
                "average_confidence": result.get_average_confidence()
            },
            "dimension_analysis": {},
            "strengths": [],
            "weaknesses": [],
            "risk_assessment": {}
        }
        
        # 维度分析
        for dim, score_obj in result.dimension_scores.items():
            analysis["dimension_analysis"][dim.value] = {
                "score": score_obj.score,
                "confidence": score_obj.confidence,
                "sub_scores": score_obj.sub_scores,
                "details": score_obj.details
            }
            
            # 识别优势和劣势
            if score_obj.score >= 0.8:
                analysis["strengths"].append({
                    "dimension": dim.value,
                    "score": score_obj.score,
                    "description": f"{dim.value}表现优秀"
                })
            elif score_obj.score < 0.6:
                analysis["weaknesses"].append({
                    "dimension": dim.value,
                    "score": score_obj.score,
                    "description": f"{dim.value}需要改进"
                })
        
        # 风险评估
        analysis["risk_assessment"] = self._assess_risks(result)
        
        return analysis
    
    def _analyze_batch_results(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """分析批量评估结果"""
        stats = results.summary_statistics
        
        analysis = {
            "batch_summary": {
                "total_items": stats.get("total_items", 0),
                "average_score": stats.get("average_score", 0.0),
                "score_range": {
                    "min": stats.get("min_score", 0.0),
                    "max": stats.get("max_score", 0.0)
                },
                "score_std": stats.get("score_std", 0.0)
            },
            "dimension_analysis": stats.get("dimension_statistics", {}),
            "performance_distribution": stats.get("performance_distribution", {}),
            "quality_indicators": {},
            "trends": {}
        }
        
        # 质量指标
        analysis["quality_indicators"] = self._calculate_quality_indicators(results)
        
        # 趋势分析
        analysis["trends"] = self._analyze_trends(results)
        
        return analysis
    
    def _assess_risks(self, result: ExpertEvaluationResult) -> Dict[str, Any]:
        """评估风险"""
        risks = {
            "overall_risk_level": "低",
            "specific_risks": [],
            "mitigation_suggestions": []
        }
        
        # 基于总体评分评估风险
        if result.overall_score < 0.5:
            risks["overall_risk_level"] = "高"
            risks["specific_risks"].append("模型整体性能不达标")
            risks["mitigation_suggestions"].append("建议暂停部署，进行全面优化")
        elif result.overall_score < 0.7:
            risks["overall_risk_level"] = "中"
            risks["specific_risks"].append("模型性能存在改进空间")
            risks["mitigation_suggestions"].append("建议在监控下小范围部署")
        
        # 基于置信度评估风险
        avg_confidence = result.get_average_confidence()
        if avg_confidence < 0.6:
            risks["specific_risks"].append("评估置信度较低")
            risks["mitigation_suggestions"].append("建议增加评估数据量")
        
        # 基于维度评分评估风险
        critical_dimensions = [EvaluationDimension.FACTUAL_CORRECTNESS, EvaluationDimension.LOGICAL_CONSISTENCY]
        for dim in critical_dimensions:
            if dim in result.dimension_scores:
                score = result.dimension_scores[dim].score
                if score < 0.7:
                    risks["specific_risks"].append(f"{dim.value}维度风险较高")
                    risks["mitigation_suggestions"].append(f"重点改进{dim.value}")
        
        return risks
    
    def _calculate_quality_indicators(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """计算质量指标"""
        stats = results.summary_statistics
        total_items = stats.get("total_items", 0)
        
        if total_items == 0:
            return {}
        
        # 计算各种质量指标
        performance_dist = stats.get("performance_distribution", {})
        
        quality_indicators = {
            "excellence_rate": performance_dist.get("优秀", 0) / total_items,
            "good_rate": performance_dist.get("良好", 0) / total_items,
            "improvement_needed_rate": performance_dist.get("需要改进", 0) / total_items,
            "consistency_score": 1.0 - min(stats.get("score_std", 0.0) / 0.5, 1.0),  # 标准化一致性分数
            "reliability_score": self._calculate_reliability_score(results)
        }
        
        return quality_indicators
    
    def _calculate_reliability_score(self, results: BatchEvaluationResult) -> float:
        """计算可靠性分数"""
        # 基于置信度和一致性计算可靠性
        confidences = []
        for result in results.individual_results:
            confidences.append(result.get_average_confidence())
        
        if not confidences:
            return 0.0
        
        avg_confidence = sum(confidences) / len(confidences)
        confidence_std = (sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)) ** 0.5
        
        # 可靠性 = 平均置信度 * (1 - 置信度标准差)
        reliability = avg_confidence * (1.0 - min(confidence_std, 1.0))
        return max(0.0, min(1.0, reliability))
    
    def _analyze_trends(self, results: BatchEvaluationResult) -> Dict[str, Any]:
        """分析趋势"""
        trends = {
            "score_trend": "稳定",
            "processing_time_trend": "稳定",
            "quality_trend": "稳定"
        }
        
        # 简化的趋势分析（实际应用中可以更复杂）
        scores = [r.overall_score for r in results.individual_results]
        processing_times = [r.processing_time for r in results.individual_results]
        
        if len(scores) > 1:
            # 评分趋势
            if scores[-1] > scores[0]:
                trends["score_trend"] = "上升"
            elif scores[-1] < scores[0]:
                trends["score_trend"] = "下降"
            
            # 处理时间趋势
            if processing_times[-1] > processing_times[0] * 1.2:
                trends["processing_time_trend"] = "增长"
            elif processing_times[-1] < processing_times[0] * 0.8:
                trends["processing_time_trend"] = "减少"
        
        return trends
    
    def _export_json_report(self, report: EvaluationReport, output_path: Path) -> bool:
        """导出JSON格式报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"JSON报告导出失败: {str(e)}")
            return False
    
    def _export_html_report(self, report: EvaluationReport, output_path: Path) -> bool:
        """导出HTML格式报告"""
        try:
            html_content = self._generate_html_content(report)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"HTML报告导出失败: {str(e)}")
            return False
    
    def _export_pdf_report(self, report: EvaluationReport, output_path: Path) -> bool:
        """导出PDF格式报告"""
        try:
            # 这里可以集成PDF生成库，如reportlab或weasyprint
            # 为了简化，先生成HTML然后转换为PDF的占位符实现
            
            self.logger.warning("PDF导出功能需要额外的依赖库，当前使用HTML格式替代")
            
            # 生成HTML版本作为替代
            html_path = output_path.with_suffix('.html')
            return self._export_html_report(report, html_path)
            
        except Exception as e:
            self.logger.error(f"PDF报告导出失败: {str(e)}")
            return False
    
    def _generate_html_content(self, report: EvaluationReport) -> str:
        """生成HTML报告内容"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #2E86AB; color: white; padding: 20px; border-radius: 8px; }}
        .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .section {{ margin: 20px 0; }}
        .chart-container {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .recommendation {{ background: #e7f3ff; padding: 10px; margin: 5px 0; border-left: 4px solid #2E86AB; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .score-high {{ color: #28a745; font-weight: bold; }}
        .score-medium {{ color: #ffc107; font-weight: bold; }}
        .score-low {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>报告ID: {report_id}</p>
        <p>生成时间: {generated_at}</p>
    </div>
    
    <div class="summary">
        <h2>执行摘要</h2>
        <p>{summary}</p>
        {executive_summary}
    </div>
    
    <div class="section">
        <h2>详细分析</h2>
        {detailed_analysis}
    </div>
    
    <div class="section">
        <h2>可视化图表</h2>
        {charts_section}
    </div>
    
    <div class="section">
        <h2>改进建议</h2>
        {recommendations_section}
    </div>
    
    <div class="section">
        <h2>附录</h2>
        <p>本报告由专家评估系统自动生成，基于多维度评估指标体系。</p>
        <p>如需更详细的分析，请参考JSON格式的完整报告数据。</p>
    </div>
</body>
</html>
        """
        
        # 格式化各个部分
        executive_summary = self._format_executive_summary_html(report)
        detailed_analysis = self._format_detailed_analysis_html(report)
        charts_section = self._format_charts_section_html(report)
        recommendations_section = self._format_recommendations_html(report)
        
        return html_template.format(
            title=report.title,
            report_id=report.report_id,
            generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            summary=report.summary,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            charts_section=charts_section,
            recommendations_section=recommendations_section
        )
    
    def _format_executive_summary_html(self, report: EvaluationReport) -> str:
        """格式化执行摘要HTML"""
        exec_summary = report.get_executive_summary()
        
        if isinstance(report.evaluation_results, ExpertEvaluationResult):
            return f"""
            <table>
                <tr><th>总体评分</th><td class="score-{self._get_score_class(exec_summary.get('overall_score', 0))}">{exec_summary.get('overall_score', 0):.2f}</td></tr>
                <tr><th>性能类别</th><td>{exec_summary.get('performance_category', 'N/A')}</td></tr>
                <tr><th>建议数量</th><td>{exec_summary.get('recommendation_count', 0)}</td></tr>
            </table>
            """
        else:
            return f"""
            <table>
                <tr><th>评估项目总数</th><td>{exec_summary.get('total_items', 0)}</td></tr>
                <tr><th>平均评分</th><td class="score-{self._get_score_class(exec_summary.get('average_score', 0))}">{exec_summary.get('average_score', 0):.2f}</td></tr>
                <tr><th>建议数量</th><td>{exec_summary.get('recommendation_count', 0)}</td></tr>
            </table>
            """
    
    def _format_detailed_analysis_html(self, report: EvaluationReport) -> str:
        """格式化详细分析HTML"""
        analysis = report.detailed_analysis
        
        if not analysis:
            return "<p>暂无详细分析数据</p>"
        
        html_parts = []
        
        # 性能分析
        if "performance_analysis" in analysis:
            perf = analysis["performance_analysis"]
            html_parts.append(f"""
            <h3>性能分析</h3>
            <table>
                <tr><th>总体评分</th><td class="score-{self._get_score_class(perf.get('overall_score', 0))}">{perf.get('overall_score', 0):.2f}</td></tr>
                <tr><th>性能类别</th><td>{perf.get('performance_category', 'N/A')}</td></tr>
                <tr><th>平均置信度</th><td>{perf.get('average_confidence', 0):.2f}</td></tr>
            </table>
            """)
        
        # 维度分析
        if "dimension_analysis" in analysis:
            dim_analysis = analysis["dimension_analysis"]
            html_parts.append("<h3>维度分析</h3>")
            html_parts.append("<table><tr><th>维度</th><th>评分</th><th>置信度</th></tr>")
            
            for dim_name, dim_data in dim_analysis.items():
                score = dim_data.get("score", 0)
                confidence = dim_data.get("confidence", 0)
                html_parts.append(f"""
                <tr>
                    <td>{dim_name}</td>
                    <td class="score-{self._get_score_class(score)}">{score:.2f}</td>
                    <td>{confidence:.2f}</td>
                </tr>
                """)
            
            html_parts.append("</table>")
        
        return "".join(html_parts)
    
    def _format_charts_section_html(self, report: EvaluationReport) -> str:
        """格式化图表部分HTML"""
        if not report.charts_data:
            return "<p>暂无图表数据</p>"
        
        html_parts = []
        for chart_name, chart_data in report.charts_data.items():
            html_parts.append(f"""
            <div class="chart-container">
                <h3>{chart_data.get('title', chart_name)}</h3>
                <p>图表类型: {chart_data.get('type', 'unknown')}</p>
                <p><em>注: 完整的交互式图表请参考JSON报告数据</em></p>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _format_recommendations_html(self, report: EvaluationReport) -> str:
        """格式化建议HTML"""
        if not report.recommendations:
            return "<p>暂无改进建议</p>"
        
        html_parts = []
        for i, recommendation in enumerate(report.recommendations, 1):
            html_parts.append(f"""
            <div class="recommendation">
                <strong>建议 {i}:</strong> {recommendation}
            </div>
            """)
        
        return "".join(html_parts)
    
    def _get_score_class(self, score: float) -> str:
        """获取评分对应的CSS类"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _load_improvement_templates(self) -> Dict[str, Dict[str, str]]:
        """加载改进建议模板"""
        return {
            "SEMANTIC_SIMILARITY": {
                "low_score": "语义相似性较低，建议增加语义理解训练数据",
                "medium_score": "语义相似性中等，可通过同义词扩展改进",
                "high_score": "语义相似性表现良好"
            },
            "DOMAIN_ACCURACY": {
                "low_score": "领域准确性不足，建议增加专业领域训练数据",
                "medium_score": "领域准确性一般，建议加强专业术语训练",
                "high_score": "领域准确性表现优秀"
            },
            "RESPONSE_RELEVANCE": {
                "low_score": "回答相关性较差，建议改进问题理解能力",
                "medium_score": "回答相关性中等，建议优化上下文理解",
                "high_score": "回答相关性表现出色"
            },
            "FACTUAL_CORRECTNESS": {
                "low_score": "事实正确性存在问题，建议加强知识库训练",
                "medium_score": "事实正确性一般，建议增加事实验证机制",
                "high_score": "事实正确性表现可靠"
            },
            "COMPLETENESS": {
                "low_score": "回答完整性不足，建议训练更全面的回答生成",
                "medium_score": "回答完整性中等，建议优化信息覆盖度",
                "high_score": "回答完整性表现全面"
            },
            "INNOVATION": {
                "low_score": "创新性较低，建议增加创造性思维训练",
                "medium_score": "创新性一般，建议鼓励多样化回答",
                "high_score": "创新性表现突出"
            },
            "PRACTICAL_VALUE": {
                "low_score": "实用价值有限，建议增加实际应用场景训练",
                "medium_score": "实用价值中等，建议优化实践指导能力",
                "high_score": "实用价值表现优异"
            },
            "LOGICAL_CONSISTENCY": {
                "low_score": "逻辑一致性存在问题，建议加强逻辑推理训练",
                "medium_score": "逻辑一致性一般，建议优化推理链条",
                "high_score": "逻辑一致性表现严密"
            }
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
                score=0.5,
                confidence=0.8
            ),
            EvaluationDimension.DOMAIN_ACCURACY: DimensionScore(
                dimension=EvaluationDimension.DOMAIN_ACCURACY,
                score=0.9,
                confidence=0.9
            )
        }
        
        test_result = ExpertEvaluationResult(
            question_id="test_001",
            overall_score=0.7,
            dimension_scores=dimension_scores
        )
        
        # 测试报告生成器
        generator = EvaluationReportGenerator()
        
        # 测试改进建议生成
        suggestions = generator.generate_improvement_suggestions(test_result)
        print(f"生成了 {len(suggestions)} 条改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        # 测试报告生成
        report = generator.generate_detailed_report(test_result)
        print(f"\n成功生成报告: {report.report_id}")
        print(f"报告标题: {report.title}")
        print(f"图表数量: {len(report.charts_data)}")
        print(f"建议数量: {len(report.recommendations)}")
        
        print("\n✓ 改进建议生成功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 改进建议生成功能测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    test_improvement_suggestions()