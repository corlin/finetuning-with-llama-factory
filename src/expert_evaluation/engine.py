"""
专家评估引擎核心实现

本模块实现了ExpertEvaluationEngine核心类，提供专家级行业化评估功能。
集成现有的ModelService和评估框架，提供统一的评估接口。
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json

from ..model_service import ModelService
from ..evaluation_framework import ComprehensiveEvaluationFramework
from ..config_manager import ConfigManager
from .interfaces import ExpertEvaluationEngine as ExpertEvaluationEngineInterface
from .config import ExpertEvaluationConfig, EvaluationDimension
from .data_models import (
    QAEvaluationItem, 
    ExpertEvaluationResult, 
    BatchEvaluationResult,
    EvaluationReport,
    ValidationResult,
    EvaluationDataset,
    DimensionScore
)
from .exceptions import (
    ModelLoadError,
    EvaluationProcessError,
    DataFormatError,
    ConfigurationError
)


class ExpertEvaluationEngine(ExpertEvaluationEngineInterface):
    """专家评估引擎核心实现类"""
    
    def __init__(self, config: ExpertEvaluationConfig):
        """
        初始化评估引擎
        
        Args:
            config: 专家评估配置
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 核心组件
        self.model_service: Optional[ModelService] = None
        self.evaluation_framework: Optional[ComprehensiveEvaluationFramework] = None
        self.config_manager: Optional[ConfigManager] = None
        
        # 状态管理
        self.is_model_loaded = False
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_processing_time": 0.0,
            "last_evaluation_time": None
        }
        
        # 初始化组件
        self._initialize_components()
        
        self.logger.info("专家评估引擎初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{__name__}.ExpertEvaluationEngine")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # 文件处理器（如果配置了日志文件）
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
                file_handler.setFormatter(console_formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_components(self):
        """初始化核心组件"""
        try:
            # 初始化配置管理器
            self.config_manager = ConfigManager()
            
            # 初始化模型服务
            self.model_service = ModelService()
            
            # 初始化评估框架
            # 这里需要密码学知识库，暂时使用空字典
            crypto_kb = self._load_crypto_knowledge_base()
            self.evaluation_framework = ComprehensiveEvaluationFramework(
                crypto_knowledge_base=crypto_kb,
                logger=self.logger
            )
            
            self.logger.info("核心组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise ConfigurationError(f"组件初始化失败: {e}")
    
    def _load_crypto_knowledge_base(self) -> Dict[str, Any]:
        """加载密码学知识库"""
        # 简化实现，实际应该从配置文件或数据库加载
        return {
            "AES": {"type": "对称加密", "security": "高", "key_sizes": [128, 192, 256]},
            "RSA": {"type": "非对称加密", "security": "高", "key_sizes": [2048, 3072, 4096]},
            "SHA-256": {"type": "哈希函数", "security": "高", "output_size": 256},
            "DES": {"type": "对称加密", "security": "低", "deprecated": True},
            "MD5": {"type": "哈希函数", "security": "低", "deprecated": True}
        }
    
    def load_model(self, model_path: str) -> bool:
        """
        加载已合并的最终模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            self.logger.info(f"开始加载模型: {model_path}")
            
            # 验证模型路径
            if not model_path:
                raise ModelLoadError("模型路径不能为空")
            
            # 更新配置中的模型路径
            self.config.model_config.model_path = model_path
            
            # 使用ModelService加载模型
            success = asyncio.run(self._async_load_model(model_path))
            
            if success:
                self.is_model_loaded = True
                self.logger.info("模型加载成功")
                return True
            else:
                self.is_model_loaded = False
                self.logger.error("模型加载失败")
                return False
                
        except Exception as e:
            self.logger.error(f"模型加载异常: {e}")
            self.is_model_loaded = False
            raise ModelLoadError(f"模型加载失败: {e}")
    
    async def _async_load_model(self, model_path: str) -> bool:
        """异步加载模型"""
        try:
            # 根据配置确定量化格式
            quantization_format = "int8"
            if self.config.model_config.load_in_4bit:
                quantization_format = "int4"
            elif self.config.model_config.load_in_8bit:
                quantization_format = "int8"
            else:
                quantization_format = "standard"
            
            # 调用ModelService的加载方法
            await self.model_service.load_model(model_path, quantization_format)
            return True
            
        except Exception as e:
            self.logger.error(f"异步模型加载失败: {e}")
            return False
    
    def evaluate_model(self, qa_data: List[QAEvaluationItem]) -> ExpertEvaluationResult:
        """
        执行单个QA数据集的评估
        
        Args:
            qa_data: QA评估数据列表
            
        Returns:
            ExpertEvaluationResult: 评估结果
        """
        if not self.is_model_loaded:
            raise EvaluationProcessError("模型未加载，无法进行评估")
        
        if not qa_data:
            raise DataFormatError("QA数据不能为空")
        
        start_time = time.time()
        
        try:
            self.logger.info(f"开始评估 {len(qa_data)} 个QA项目")
            
            # 验证QA数据格式
            validation_result = self._validate_qa_data(qa_data)
            if not validation_result.is_valid:
                raise DataFormatError(f"QA数据验证失败: {validation_result.errors}")
            
            # 执行评估
            evaluation_results = []
            
            for qa_item in qa_data:
                try:
                    # 生成模型回答
                    model_answer = self._generate_model_answer(qa_item)
                    
                    # 使用评估框架进行评估
                    framework_result = self.evaluation_framework.evaluate_model_response(
                        question=qa_item.question,
                        model_answer=model_answer,
                        reference_answer=qa_item.reference_answer,
                        context=qa_item.context
                    )
                    
                    # 转换为专家评估结果格式
                    expert_result = self._convert_to_expert_result(qa_item, framework_result)
                    evaluation_results.append(expert_result)
                    
                except Exception as e:
                    self.logger.error(f"评估QA项目 {qa_item.question_id} 失败: {e}")
                    # 创建失败结果
                    failed_result = self._create_failed_result(qa_item, str(e))
                    evaluation_results.append(failed_result)
            
            # 计算综合结果
            if evaluation_results:
                overall_result = self._aggregate_evaluation_results(evaluation_results)
            else:
                raise EvaluationProcessError("没有成功的评估结果")
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self._update_evaluation_stats(processing_time, success=True)
            
            self.logger.info(f"评估完成，总体得分: {overall_result.overall_score:.3f}")
            
            return overall_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_evaluation_stats(processing_time, success=False)
            self.logger.error(f"评估过程失败: {e}")
            raise EvaluationProcessError(f"评估过程失败: {e}")
    
    def batch_evaluate(self, qa_datasets: List[EvaluationDataset]) -> BatchEvaluationResult:
        """
        批量评估多个数据集
        
        Args:
            qa_datasets: 评估数据集列表
            
        Returns:
            BatchEvaluationResult: 批量评估结果
        """
        if not self.is_model_loaded:
            raise EvaluationProcessError("模型未加载，无法进行批量评估")
        
        if not qa_datasets:
            raise DataFormatError("数据集列表不能为空")
        
        start_time = datetime.now()
        batch_id = f"batch_{int(time.time())}"
        
        try:
            self.logger.info(f"开始批量评估 {len(qa_datasets)} 个数据集")
            
            individual_results = []
            
            for i, dataset in enumerate(qa_datasets):
                try:
                    self.logger.info(f"评估数据集 {i+1}/{len(qa_datasets)}: {dataset.name}")
                    
                    # 评估单个数据集
                    dataset_result = self.evaluate_model(dataset.qa_items)
                    individual_results.append(dataset_result)
                    
                except Exception as e:
                    self.logger.error(f"数据集 {dataset.name} 评估失败: {e}")
                    # 创建失败结果
                    failed_result = ExpertEvaluationResult(
                        question_id=f"dataset_{dataset.dataset_id}",
                        overall_score=0.0,
                        dimension_scores={},
                        detailed_feedback={"error": str(e)},
                        processing_time=0.0
                    )
                    individual_results.append(failed_result)
            
            # 创建批量结果
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds()
            
            batch_result = BatchEvaluationResult(
                batch_id=batch_id,
                individual_results=individual_results,
                total_processing_time=total_processing_time,
                start_time=start_time,
                end_time=end_time
            )
            
            self.logger.info(f"批量评估完成，处理了 {len(individual_results)} 个数据集")
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"批量评估失败: {e}")
            raise EvaluationProcessError(f"批量评估失败: {e}")
    
    def generate_report(self, results: ExpertEvaluationResult) -> EvaluationReport:
        """
        生成详细的评估报告
        
        Args:
            results: 评估结果
            
        Returns:
            EvaluationReport: 评估报告
        """
        try:
            report_id = f"report_{int(time.time())}"
            
            # 生成报告标题和摘要
            title = f"专家评估报告 - {results.question_id}"
            summary = self._generate_report_summary(results)
            
            # 生成改进建议
            recommendations = self._generate_improvement_recommendations(results)
            
            # 生成详细分析
            detailed_analysis = self._generate_detailed_analysis(results)
            
            # 创建报告
            report = EvaluationReport(
                report_id=report_id,
                title=title,
                summary=summary,
                evaluation_results=results,
                recommendations=recommendations,
                detailed_analysis=detailed_analysis,
                format_type=self.config.report_config.default_format.value
            )
            
            self.logger.info(f"评估报告生成完成: {report_id}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            raise EvaluationProcessError(f"报告生成失败: {e}")
    
    def _validate_qa_data(self, qa_data: List[QAEvaluationItem]) -> ValidationResult:
        """验证QA数据格式"""
        validation_result = ValidationResult(is_valid=True)
        
        for i, qa_item in enumerate(qa_data):
            try:
                # 验证必填字段
                if not qa_item.question_id.strip():
                    validation_result.add_error(f"项目 {i}: 问题ID不能为空")
                
                if not qa_item.question.strip():
                    validation_result.add_error(f"项目 {i}: 问题内容不能为空")
                
                if not qa_item.reference_answer.strip():
                    validation_result.add_error(f"项目 {i}: 参考答案不能为空")
                
                # 验证长度限制
                if len(qa_item.question) > self.config.data_config.max_question_length:
                    validation_result.add_warning(f"项目 {i}: 问题长度超过限制")
                
                if len(qa_item.reference_answer) > self.config.data_config.max_answer_length:
                    validation_result.add_warning(f"项目 {i}: 答案长度超过限制")
                
                if validation_result.is_valid:
                    validation_result.valid_items_count += 1
                else:
                    validation_result.invalid_items_count += 1
                    
            except Exception as e:
                validation_result.add_error(f"项目 {i}: 验证异常 - {e}")
                validation_result.invalid_items_count += 1
        
        return validation_result
    
    def _generate_model_answer(self, qa_item: QAEvaluationItem) -> str:
        """生成模型回答"""
        try:
            # 构建提示
            prompt = self._build_evaluation_prompt(qa_item)
            
            # 调用模型服务生成回答
            response = asyncio.run(self._async_generate_answer(prompt))
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成模型回答失败: {e}")
            raise EvaluationProcessError(f"生成模型回答失败: {e}")
    
    def _build_evaluation_prompt(self, qa_item: QAEvaluationItem) -> str:
        """构建评估提示"""
        prompt = f"问题：{qa_item.question}\n\n"
        
        if qa_item.context:
            prompt += f"上下文：{qa_item.context}\n\n"
        
        prompt += "请提供详细的回答："
        
        return prompt
    
    async def _async_generate_answer(self, prompt: str) -> str:
        """异步生成回答"""
        from ..model_service import GenerateRequest
        
        # 创建生成请求
        request = GenerateRequest(
            prompt=prompt,
            max_length=self.config.model_config.max_new_tokens,
            temperature=self.config.model_config.temperature,
            top_p=self.config.model_config.top_p,
            do_sample=self.config.model_config.do_sample
        )
        
        # 调用模型服务
        response = await self.model_service.generate_text(request)
        
        return response.generated_text
    
    def _convert_to_expert_result(self, qa_item: QAEvaluationItem, framework_result) -> ExpertEvaluationResult:
        """将评估框架结果转换为专家评估结果"""
        # 转换维度评分
        dimension_scores = {}
        for dim, score in framework_result.scores.items():
            # 映射评估框架的维度到专家评估维度
            expert_dim = self._map_evaluation_dimension(dim)
            if expert_dim:
                dimension_scores[expert_dim] = DimensionScore(
                    dimension=expert_dim,
                    score=score,
                    confidence=0.8,  # 默认置信度
                    details={"source": "evaluation_framework"}
                )
        
        # 创建专家评估结果
        expert_result = ExpertEvaluationResult(
            question_id=qa_item.question_id,
            overall_score=framework_result.overall_score,
            dimension_scores=dimension_scores,
            detailed_feedback=framework_result.detailed_feedback,
            improvement_suggestions=[],  # 后续生成
            processing_time=0.0
        )
        
        return expert_result
    
    def _map_evaluation_dimension(self, framework_dim) -> Optional[EvaluationDimension]:
        """映射评估框架维度到专家评估维度"""
        # 这里需要根据实际的维度枚举进行映射
        mapping = {
            "TECHNICAL_ACCURACY": EvaluationDimension.DOMAIN_ACCURACY,
            "CONCEPTUAL_UNDERSTANDING": EvaluationDimension.SEMANTIC_SIMILARITY,
            "PRACTICAL_APPLICABILITY": EvaluationDimension.PRACTICAL_VALUE,
            "LINGUISTIC_QUALITY": EvaluationDimension.LOGICAL_CONSISTENCY,
            "REASONING_COHERENCE": EvaluationDimension.LOGICAL_CONSISTENCY
        }
        
        return mapping.get(framework_dim.name if hasattr(framework_dim, 'name') else str(framework_dim))
    
    def _create_failed_result(self, qa_item: QAEvaluationItem, error_message: str) -> ExpertEvaluationResult:
        """创建失败的评估结果"""
        return ExpertEvaluationResult(
            question_id=qa_item.question_id,
            overall_score=0.0,
            dimension_scores={},
            detailed_feedback={"error": error_message},
            improvement_suggestions=["评估过程出现错误，请检查输入数据和模型状态"],
            processing_time=0.0
        )
    
    def _aggregate_evaluation_results(self, results: List[ExpertEvaluationResult]) -> ExpertEvaluationResult:
        """聚合多个评估结果"""
        if not results:
            raise EvaluationProcessError("没有评估结果可以聚合")
        
        # 计算平均分数
        total_score = sum(r.overall_score for r in results)
        avg_score = total_score / len(results)
        
        # 聚合维度分数
        aggregated_dimensions = {}
        all_dimensions = set()
        for result in results:
            all_dimensions.update(result.dimension_scores.keys())
        
        for dim in all_dimensions:
            scores = [r.dimension_scores[dim].score for r in results if dim in r.dimension_scores]
            if scores:
                avg_dim_score = sum(scores) / len(scores)
                aggregated_dimensions[dim] = DimensionScore(
                    dimension=dim,
                    score=avg_dim_score,
                    confidence=0.8,
                    details={"aggregated_from": len(scores), "total_items": len(results)}
                )
        
        # 聚合反馈
        aggregated_feedback = {
            "total_items": len(results),
            "successful_items": len([r for r in results if r.overall_score > 0]),
            "average_score": avg_score,
            "score_range": f"{min(r.overall_score for r in results):.3f} - {max(r.overall_score for r in results):.3f}"
        }
        
        # 聚合改进建议
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.improvement_suggestions)
        
        # 去重并限制数量
        unique_suggestions = list(set(all_suggestions))[:10]
        
        return ExpertEvaluationResult(
            question_id="aggregated_result",
            overall_score=avg_score,
            dimension_scores=aggregated_dimensions,
            detailed_feedback=aggregated_feedback,
            improvement_suggestions=unique_suggestions,
            processing_time=sum(r.processing_time for r in results)
        )
    
    def _update_evaluation_stats(self, processing_time: float, success: bool):
        """更新评估统计信息"""
        self.evaluation_stats["total_evaluations"] += 1
        
        if success:
            self.evaluation_stats["successful_evaluations"] += 1
        else:
            self.evaluation_stats["failed_evaluations"] += 1
        
        # 更新平均处理时间
        total_time = (self.evaluation_stats["average_processing_time"] * 
                     (self.evaluation_stats["total_evaluations"] - 1) + processing_time)
        self.evaluation_stats["average_processing_time"] = total_time / self.evaluation_stats["total_evaluations"]
        
        self.evaluation_stats["last_evaluation_time"] = datetime.now().isoformat()
    
    def _generate_report_summary(self, results: ExpertEvaluationResult) -> str:
        """生成报告摘要"""
        performance_category = results.get_performance_category()
        avg_confidence = results.get_average_confidence()
        
        summary = f"""
        评估摘要：
        - 总体得分: {results.overall_score:.3f}
        - 性能类别: {performance_category}
        - 平均置信度: {avg_confidence:.3f}
        - 评估维度数: {len(results.dimension_scores)}
        - 处理时间: {results.processing_time:.2f}秒
        """
        
        return summary.strip()
    
    def _generate_improvement_recommendations(self, results: ExpertEvaluationResult) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于维度分数生成建议
        for dim, score_obj in results.dimension_scores.items():
            if score_obj.score < 0.6:
                recommendations.append(f"需要改进{dim.value}方面的表现，当前得分: {score_obj.score:.2f}")
        
        # 基于总体分数生成建议
        if results.overall_score < 0.5:
            recommendations.append("总体表现需要显著改进，建议重新训练或调整模型参数")
        elif results.overall_score < 0.7:
            recommendations.append("表现一般，建议针对薄弱环节进行专项优化")
        
        # 添加通用建议
        if not recommendations:
            recommendations.append("表现良好，可以考虑在更复杂的场景下进行测试")
        
        return recommendations[:5]  # 限制建议数量
    
    def _generate_detailed_analysis(self, results: ExpertEvaluationResult) -> Dict[str, Any]:
        """生成详细分析"""
        analysis = {
            "score_analysis": {
                "overall_score": results.overall_score,
                "performance_category": results.get_performance_category(),
                "dimension_breakdown": {
                    dim.value: {
                        "score": score_obj.score,
                        "confidence": score_obj.confidence,
                        "details": score_obj.details
                    }
                    for dim, score_obj in results.dimension_scores.items()
                }
            },
            "strengths": [],
            "weaknesses": [],
            "technical_details": {
                "processing_time": results.processing_time,
                "timestamp": results.timestamp.isoformat(),
                "confidence_intervals": results.confidence_intervals,
                "statistical_significance": results.statistical_significance
            }
        }
        
        # 识别优势和劣势
        for dim, score_obj in results.dimension_scores.items():
            if score_obj.score >= 0.8:
                analysis["strengths"].append(f"{dim.value}: {score_obj.score:.2f}")
            elif score_obj.score < 0.6:
                analysis["weaknesses"].append(f"{dim.value}: {score_obj.score:.2f}")
        
        return analysis
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        return self.evaluation_stats.copy()
    
    def is_ready(self) -> bool:
        """检查引擎是否准备就绪"""
        return (self.is_model_loaded and 
                self.model_service is not None and 
                self.evaluation_framework is not None)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_model_loaded or not self.model_service:
            return {"status": "model_not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.config.model_config.model_path,
            "model_type": self.config.model_config.model_type,
            "device": self.config.model_config.device,
            "load_in_8bit": self.config.model_config.load_in_8bit,
            "load_in_4bit": self.config.model_config.load_in_4bit
        }