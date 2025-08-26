"""
异常检测和报告生成模块

本模块实现了训练异常检测算法（梯度爆炸、收敛失败等）、综合训练报告生成功能、
可视化训练曲线和指标图表、训练建议和优化提示生成等功能。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from pathlib import Path

from parallel_config import DistributedTrainingMetrics
from src.data_models import ChineseMetrics
from chinese_metrics_calculator import CryptoTermLearningProgress


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """异常事件数据结构"""
    timestamp: datetime
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_metrics: List[str]
    suggested_actions: List[str]
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "affected_metrics": self.affected_metrics,
            "suggested_actions": self.suggested_actions,
            "context_data": self.context_data
        }


@dataclass
class TrainingReport:
    """训练报告数据结构"""
    report_id: str
    generation_time: datetime = field(default_factory=datetime.now)
    
    # 基本信息
    training_duration: timedelta = field(default_factory=lambda: timedelta(0))
    total_epochs: int = 0
    total_steps: int = 0
    
    # 性能摘要
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_train_loss: float = float('inf')
    best_val_loss: float = float('inf')
    convergence_achieved: bool = False
    
    # 中文指标摘要
    chinese_metrics_summary: Dict[str, float] = field(default_factory=dict)
    crypto_learning_progress: Dict[str, Any] = field(default_factory=dict)
    
    # GPU和系统性能
    gpu_utilization_summary: Dict[str, float] = field(default_factory=dict)
    memory_efficiency_summary: Dict[str, float] = field(default_factory=dict)
    communication_efficiency: float = 0.0
    
    # 异常和问题
    anomalies_detected: List[AnomalyEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 建议和优化
    optimization_recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # 可视化文件路径
    visualization_files: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "report_id": self.report_id,
            "generation_time": self.generation_time.isoformat(),
            "training_duration": str(self.training_duration),
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_train_loss": self.best_train_loss,
            "best_val_loss": self.best_val_loss,
            "convergence_achieved": self.convergence_achieved,
            "chinese_metrics_summary": self.chinese_metrics_summary,
            "crypto_learning_progress": self.crypto_learning_progress,
            "gpu_utilization_summary": self.gpu_utilization_summary,
            "memory_efficiency_summary": self.memory_efficiency_summary,
            "communication_efficiency": self.communication_efficiency,
            "anomalies_detected": [anomaly.to_dict() for anomaly in self.anomalies_detected],
            "warnings": self.warnings,
            "optimization_recommendations": self.optimization_recommendations,
            "next_steps": self.next_steps,
            "visualization_files": self.visualization_files
        }


class AnomalyDetector:
    """训练异常检测器"""
    
    def __init__(self, 
                 gradient_explosion_threshold: float = 10.0,
                 loss_spike_threshold: float = 2.0,
                 convergence_patience: int = 100,
                 memory_threshold: float = 95.0,
                 temperature_threshold: float = 85.0):
        """
        初始化异常检测器
        
        Args:
            gradient_explosion_threshold: 梯度爆炸阈值
            loss_spike_threshold: 损失突增阈值（倍数）
            convergence_patience: 收敛耐心值（步数）
            memory_threshold: 内存使用阈值（百分比）
            temperature_threshold: 温度阈值（摄氏度）
        """
        self.gradient_explosion_threshold = gradient_explosion_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.convergence_patience = convergence_patience
        self.memory_threshold = memory_threshold
        self.temperature_threshold = temperature_threshold
        
        # 历史数据用于异常检测
        self.loss_history: deque = deque(maxlen=1000)
        self.gradient_history: deque = deque(maxlen=1000)
        self.gpu_metrics_history: deque = deque(maxlen=1000)
        
        # 异常事件历史
        self.anomaly_history: List[AnomalyEvent] = []
        
        logger.info("异常检测器初始化完成")
    
    def detect_anomalies(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测训练异常"""
        anomalies = []
        
        # 更新历史数据
        self.loss_history.append((metrics.global_step, metrics.train_loss))
        self.gradient_history.append((metrics.global_step, metrics.gradient_norm))
        self.gpu_metrics_history.append((metrics.global_step, metrics.gpu_metrics))
        
        # 1. 梯度爆炸检测
        gradient_anomalies = self._detect_gradient_explosion(metrics)
        anomalies.extend(gradient_anomalies)
        
        # 2. 损失异常检测
        loss_anomalies = self._detect_loss_anomalies(metrics)
        anomalies.extend(loss_anomalies)
        
        # 3. 收敛失败检测
        convergence_anomalies = self._detect_convergence_failure(metrics)
        anomalies.extend(convergence_anomalies)
        
        # 4. GPU异常检测
        gpu_anomalies = self._detect_gpu_anomalies(metrics)
        anomalies.extend(gpu_anomalies)
        
        # 5. 内存异常检测
        memory_anomalies = self._detect_memory_anomalies(metrics)
        anomalies.extend(memory_anomalies)
        
        # 6. 性能异常检测
        performance_anomalies = self._detect_performance_anomalies(metrics)
        anomalies.extend(performance_anomalies)
        
        # 保存异常历史
        self.anomaly_history.extend(anomalies)
        
        return anomalies
    
    def _detect_gradient_explosion(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测梯度爆炸"""
        anomalies = []
        
        if metrics.gradient_norm > self.gradient_explosion_threshold:
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type="gradient_explosion",
                severity="critical",
                description=f"检测到梯度爆炸：梯度范数 {metrics.gradient_norm:.4f} 超过阈值 {self.gradient_explosion_threshold}",
                affected_metrics=["gradient_norm", "train_loss"],
                suggested_actions=[
                    "降低学习率",
                    "启用梯度裁剪",
                    "检查数据预处理",
                    "减少批次大小",
                    "检查模型架构"
                ],
                context_data={
                    "gradient_norm": metrics.gradient_norm,
                    "threshold": self.gradient_explosion_threshold,
                    "step": metrics.global_step
                }
            )
            anomalies.append(anomaly)
        
        # 检测梯度范数的突然增长
        if len(self.gradient_history) >= 10:
            recent_gradients = [grad for _, grad in list(self.gradient_history)[-10:]]
            if len(recent_gradients) >= 2:
                current_grad = recent_gradients[-1]
                avg_prev_grad = np.mean(recent_gradients[:-1])
                
                if avg_prev_grad > 0 and current_grad > avg_prev_grad * 5:
                    anomaly = AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type="gradient_spike",
                        severity="high",
                        description=f"检测到梯度突增：当前梯度 {current_grad:.4f} 是之前平均值的 {current_grad/avg_prev_grad:.2f} 倍",
                        affected_metrics=["gradient_norm"],
                        suggested_actions=[
                            "检查最近的数据批次",
                            "考虑启用梯度裁剪",
                            "降低学习率"
                        ],
                        context_data={
                            "current_gradient": current_grad,
                            "average_previous": avg_prev_grad,
                            "ratio": current_grad / avg_prev_grad
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_loss_anomalies(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测损失异常"""
        anomalies = []
        
        # 检测NaN或无穷大损失
        if np.isnan(metrics.train_loss) or np.isinf(metrics.train_loss):
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type="invalid_loss",
                severity="critical",
                description=f"检测到无效损失值：{metrics.train_loss}",
                affected_metrics=["train_loss"],
                suggested_actions=[
                    "检查数据预处理",
                    "降低学习率",
                    "检查模型权重初始化",
                    "启用梯度裁剪",
                    "检查损失函数实现"
                ],
                context_data={
                    "train_loss": str(metrics.train_loss),
                    "step": metrics.global_step
                }
            )
            anomalies.append(anomaly)
        
        # 检测损失突增
        if len(self.loss_history) >= 10:
            recent_losses = [loss for _, loss in list(self.loss_history)[-10:]]
            if len(recent_losses) >= 2 and not np.isnan(metrics.train_loss) and not np.isinf(metrics.train_loss):
                current_loss = recent_losses[-1]
                avg_prev_loss = np.mean(recent_losses[:-1])
                
                if avg_prev_loss > 0 and current_loss > avg_prev_loss * self.loss_spike_threshold:
                    anomaly = AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type="loss_spike",
                        severity="high",
                        description=f"检测到损失突增：当前损失 {current_loss:.4f} 是之前平均值的 {current_loss/avg_prev_loss:.2f} 倍",
                        affected_metrics=["train_loss"],
                        suggested_actions=[
                            "检查当前批次数据",
                            "降低学习率",
                            "检查是否有数据污染",
                            "考虑回滚到之前的检查点"
                        ],
                        context_data={
                            "current_loss": current_loss,
                            "average_previous": avg_prev_loss,
                            "spike_ratio": current_loss / avg_prev_loss
                        }
                    )
                    anomalies.append(anomaly)
        
        # 检测验证损失与训练损失的差异
        if metrics.val_loss > 0 and metrics.train_loss > 0:
            loss_gap = metrics.val_loss - metrics.train_loss
            loss_ratio = metrics.val_loss / metrics.train_loss
            
            if loss_ratio > 2.0:  # 验证损失是训练损失的2倍以上
                anomaly = AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type="overfitting",
                    severity="medium",
                    description=f"检测到可能的过拟合：验证损失 {metrics.val_loss:.4f} 远高于训练损失 {metrics.train_loss:.4f}",
                    affected_metrics=["train_loss", "val_loss"],
                    suggested_actions=[
                        "增加正则化",
                        "减少模型复杂度",
                        "增加训练数据",
                        "使用Dropout",
                        "早停训练"
                    ],
                    context_data={
                        "train_loss": metrics.train_loss,
                        "val_loss": metrics.val_loss,
                        "loss_ratio": loss_ratio,
                        "loss_gap": loss_gap
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_convergence_failure(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测收敛失败"""
        anomalies = []
        
        # 检查是否长时间没有改善
        if len(self.loss_history) >= self.convergence_patience:
            recent_losses = [loss for _, loss in list(self.loss_history)[-self.convergence_patience:]]
            
            if len(recent_losses) >= self.convergence_patience:
                # 计算损失的改善程度
                early_avg = np.mean(recent_losses[:self.convergence_patience//4])
                late_avg = np.mean(recent_losses[-self.convergence_patience//4:])
                
                improvement_ratio = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
                
                if improvement_ratio < 0.01:  # 改善小于1%
                    anomaly = AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type="convergence_stagnation",
                        severity="medium",
                        description=f"检测到收敛停滞：最近 {self.convergence_patience} 步损失改善不足 1%",
                        affected_metrics=["train_loss", "convergence_score"],
                        suggested_actions=[
                            "调整学习率",
                            "使用学习率调度器",
                            "检查数据质量",
                            "尝试不同的优化器",
                            "增加模型复杂度"
                        ],
                        context_data={
                            "improvement_ratio": improvement_ratio,
                            "patience_steps": self.convergence_patience,
                            "early_avg_loss": early_avg,
                            "late_avg_loss": late_avg
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_gpu_anomalies(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测GPU异常"""
        anomalies = []
        
        for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
            # 检测GPU温度异常
            temperature = gpu_metrics.get("temperature", 0)
            if temperature > self.temperature_threshold:
                anomaly = AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type="gpu_overheating",
                    severity="high",
                    description=f"GPU {gpu_id} 温度过高：{temperature:.1f}°C 超过阈值 {self.temperature_threshold}°C",
                    affected_metrics=[f"gpu_{gpu_id}_temperature"],
                    suggested_actions=[
                        "检查GPU散热",
                        "降低训练强度",
                        "检查环境温度",
                        "清理GPU风扇",
                        "考虑降低GPU功耗限制"
                    ],
                    context_data={
                        "gpu_id": gpu_id,
                        "temperature": temperature,
                        "threshold": self.temperature_threshold
                    }
                )
                anomalies.append(anomaly)
            
            # 检测GPU利用率异常低
            utilization = gpu_metrics.get("utilization", 0)
            if utilization < 30:  # 利用率低于30%
                anomaly = AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type="low_gpu_utilization",
                    severity="low",
                    description=f"GPU {gpu_id} 利用率过低：{utilization:.1f}%",
                    affected_metrics=[f"gpu_{gpu_id}_utilization"],
                    suggested_actions=[
                        "增加批次大小",
                        "检查数据加载瓶颈",
                        "优化数据预处理",
                        "检查I/O性能",
                        "考虑使用混合精度训练"
                    ],
                    context_data={
                        "gpu_id": gpu_id,
                        "utilization": utilization
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_memory_anomalies(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测内存异常"""
        anomalies = []
        
        for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
            memory_usage = gpu_metrics.get("memory_usage_percent", 0)
            
            if memory_usage > self.memory_threshold:
                anomaly = AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type="high_memory_usage",
                    severity="high",
                    description=f"GPU {gpu_id} 内存使用率过高：{memory_usage:.1f}% 超过阈值 {self.memory_threshold}%",
                    affected_metrics=[f"gpu_{gpu_id}_memory_usage"],
                    suggested_actions=[
                        "减少批次大小",
                        "启用梯度检查点",
                        "使用混合精度训练",
                        "清理不必要的缓存",
                        "考虑模型并行"
                    ],
                    context_data={
                        "gpu_id": gpu_id,
                        "memory_usage": memory_usage,
                        "threshold": self.memory_threshold
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_performance_anomalies(self, metrics: DistributedTrainingMetrics) -> List[AnomalyEvent]:
        """检测性能异常"""
        anomalies = []
        
        # 检测吞吐量异常低
        if metrics.throughput_tokens_per_second > 0 and metrics.throughput_tokens_per_second < 100:
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type="low_throughput",
                severity="medium",
                description=f"训练吞吐量过低：{metrics.throughput_tokens_per_second:.1f} tokens/s",
                affected_metrics=["throughput_tokens_per_second"],
                suggested_actions=[
                    "增加批次大小",
                    "优化数据加载",
                    "使用混合精度训练",
                    "检查I/O瓶颈",
                    "优化模型架构"
                ],
                context_data={
                    "throughput": metrics.throughput_tokens_per_second
                }
            )
            anomalies.append(anomaly)
        
        # 检测负载不均衡
        if metrics.load_balance_score < 0.7:
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type="load_imbalance",
                severity="medium",
                description=f"检测到负载不均衡：负载均衡评分 {metrics.load_balance_score:.3f}",
                affected_metrics=["load_balance_score"],
                suggested_actions=[
                    "调整数据分片策略",
                    "优化并行配置",
                    "检查GPU性能差异",
                    "平衡工作负载分配"
                ],
                context_data={
                    "load_balance_score": metrics.load_balance_score
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def get_anomaly_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取异常摘要"""
        if time_window:
            cutoff_time = datetime.now() - time_window
            recent_anomalies = [a for a in self.anomaly_history if a.timestamp >= cutoff_time]
        else:
            recent_anomalies = self.anomaly_history
        
        # 按类型统计
        anomaly_counts = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for anomaly in recent_anomalies:
            anomaly_counts[anomaly.anomaly_type] = anomaly_counts.get(anomaly.anomaly_type, 0) + 1
            severity_counts[anomaly.severity] += 1
        
        return {
            "total_anomalies": len(recent_anomalies),
            "anomaly_types": anomaly_counts,
            "severity_distribution": severity_counts,
            "most_common_anomaly": max(anomaly_counts.items(), key=lambda x: x[1])[0] if anomaly_counts else None,
            "critical_anomalies": [a.to_dict() for a in recent_anomalies if a.severity == "critical"]
        }


class TrainingReportGenerator:
    """训练报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"训练报告生成器初始化完成，输出目录：{self.output_dir}")
    
    def generate_comprehensive_report(self,
                                    metrics_history: List[DistributedTrainingMetrics],
                                    chinese_metrics_history: List[ChineseMetrics],
                                    crypto_progress: Optional[CryptoTermLearningProgress],
                                    anomalies: List[AnomalyEvent],
                                    training_start_time: datetime,
                                    training_end_time: datetime) -> TrainingReport:
        """生成综合训练报告"""
        
        report_id = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = TrainingReport(report_id=report_id)
        
        # 基本信息
        report.training_duration = training_end_time - training_start_time
        if metrics_history:
            report.total_epochs = metrics_history[-1].epoch
            report.total_steps = metrics_history[-1].global_step
            report.final_train_loss = metrics_history[-1].train_loss
            report.final_val_loss = metrics_history[-1].val_loss
            
            # 最佳损失
            train_losses = [m.train_loss for m in metrics_history if not np.isnan(m.train_loss)]
            val_losses = [m.val_loss for m in metrics_history if not np.isnan(m.val_loss) and m.val_loss > 0]
            
            if train_losses:
                report.best_train_loss = min(train_losses)
            if val_losses:
                report.best_val_loss = min(val_losses)
        
        # 中文指标摘要
        if chinese_metrics_history:
            latest_chinese = chinese_metrics_history[-1]
            report.chinese_metrics_summary = {
                "character_accuracy": latest_chinese.character_accuracy,
                "word_accuracy": latest_chinese.word_accuracy,
                "rouge_l_chinese": latest_chinese.rouge_l_chinese,
                "bleu_chinese": latest_chinese.bleu_chinese,
                "crypto_term_accuracy": latest_chinese.crypto_term_accuracy,
                "overall_score": latest_chinese.overall_score()
            }
        
        # 密码学学习进度
        if crypto_progress:
            report.crypto_learning_progress = crypto_progress.to_dict()
        
        # GPU性能摘要
        if metrics_history:
            self._calculate_gpu_summary(report, metrics_history)
        
        # 异常和警告
        report.anomalies_detected = anomalies
        report.warnings = self._generate_warnings(metrics_history, chinese_metrics_history)
        
        # 优化建议
        report.optimization_recommendations = self._generate_optimization_recommendations(
            metrics_history, chinese_metrics_history, anomalies
        )
        
        # 下一步建议
        report.next_steps = self._generate_next_steps(report)
        
        # 生成可视化
        report.visualization_files = self._generate_visualizations(
            report_id, metrics_history, chinese_metrics_history
        )
        
        # 保存报告
        self._save_report(report)
        
        logger.info(f"综合训练报告生成完成：{report_id}")
        return report
    
    def _calculate_gpu_summary(self, report: TrainingReport, metrics_history: List[DistributedTrainingMetrics]):
        """计算GPU性能摘要"""
        if not metrics_history:
            return
        
        # 收集所有GPU指标
        all_gpu_metrics = {}
        communication_efficiencies = []
        
        for metrics in metrics_history:
            for gpu_id, gpu_data in metrics.gpu_metrics.items():
                if gpu_id not in all_gpu_metrics:
                    all_gpu_metrics[gpu_id] = {
                        "utilizations": [],
                        "memory_usages": [],
                        "temperatures": []
                    }
                
                all_gpu_metrics[gpu_id]["utilizations"].append(gpu_data.get("utilization", 0))
                all_gpu_metrics[gpu_id]["memory_usages"].append(gpu_data.get("memory_usage_percent", 0))
                all_gpu_metrics[gpu_id]["temperatures"].append(gpu_data.get("temperature", 0))
            
            if hasattr(metrics.communication_metrics, 'communication_efficiency'):
                communication_efficiencies.append(metrics.communication_metrics.communication_efficiency)
        
        # 计算平均值
        for gpu_id, data in all_gpu_metrics.items():
            report.gpu_utilization_summary[f"gpu_{gpu_id}_avg_utilization"] = np.mean(data["utilizations"])
            report.memory_efficiency_summary[f"gpu_{gpu_id}_avg_memory_usage"] = np.mean(data["memory_usages"])
            report.gpu_utilization_summary[f"gpu_{gpu_id}_avg_temperature"] = np.mean(data["temperatures"])
        
        if communication_efficiencies:
            report.communication_efficiency = np.mean(communication_efficiencies)
    
    def _generate_warnings(self, 
                          metrics_history: List[DistributedTrainingMetrics],
                          chinese_metrics_history: List[ChineseMetrics]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        if not metrics_history:
            warnings.append("没有训练指标历史数据")
            return warnings
        
        # 检查训练稳定性
        if len(metrics_history) >= 10:
            recent_losses = [m.train_loss for m in metrics_history[-10:] if not np.isnan(m.train_loss)]
            if recent_losses and np.std(recent_losses) > np.mean(recent_losses) * 0.5:
                warnings.append("训练损失波动较大，可能存在稳定性问题")
        
        # 检查中文指标趋势
        if chinese_metrics_history and len(chinese_metrics_history) >= 5:
            recent_scores = [m.overall_score() for m in chinese_metrics_history[-5:]]
            if len(recent_scores) >= 2:
                trend = recent_scores[-1] - recent_scores[0]
                if trend < -0.05:
                    warnings.append("中文指标评分呈下降趋势，需要关注")
        
        # 检查GPU利用率
        if metrics_history:
            latest_metrics = metrics_history[-1]
            low_util_gpus = []
            for gpu_id, gpu_data in latest_metrics.gpu_metrics.items():
                if gpu_data.get("utilization", 0) < 50:
                    low_util_gpus.append(gpu_id)
            
            if low_util_gpus:
                warnings.append(f"GPU {low_util_gpus} 利用率较低，可能存在性能瓶颈")
        
        return warnings
    
    def _generate_optimization_recommendations(self,
                                             metrics_history: List[DistributedTrainingMetrics],
                                             chinese_metrics_history: List[ChineseMetrics],
                                             anomalies: List[AnomalyEvent]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于异常生成建议
        anomaly_types = set(a.anomaly_type for a in anomalies)
        
        if "gradient_explosion" in anomaly_types:
            recommendations.append("建议启用梯度裁剪并降低学习率以防止梯度爆炸")
        
        if "low_gpu_utilization" in anomaly_types:
            recommendations.append("建议增加批次大小或优化数据加载以提高GPU利用率")
        
        if "high_memory_usage" in anomaly_types:
            recommendations.append("建议启用梯度检查点或使用混合精度训练以降低内存使用")
        
        if "overfitting" in anomaly_types:
            recommendations.append("建议增加正则化或使用早停以防止过拟合")
        
        # 基于性能指标生成建议
        if metrics_history:
            latest_metrics = metrics_history[-1]
            
            if latest_metrics.load_balance_score < 0.8:
                recommendations.append("建议优化数据分片策略以改善负载均衡")
            
            if latest_metrics.throughput_tokens_per_second < 500:
                recommendations.append("建议优化数据预处理和I/O性能以提高训练吞吐量")
        
        # 基于中文指标生成建议
        if chinese_metrics_history:
            latest_chinese = chinese_metrics_history[-1]
            
            if latest_chinese.crypto_term_accuracy < 0.8:
                recommendations.append("建议增加密码学术语相关的训练数据以提高专业术语准确性")
            
            if latest_chinese.character_accuracy < 0.9:
                recommendations.append("建议优化中文分词和字符处理以提高字符级准确率")
        
        return recommendations
    
    def _generate_next_steps(self, report: TrainingReport) -> List[str]:
        """生成下一步建议"""
        next_steps = []
        
        # 基于收敛状态
        if not report.convergence_achieved:
            next_steps.append("继续训练直到收敛，监控损失变化")
        else:
            next_steps.append("考虑进行模型评估和测试")
        
        # 基于异常数量
        critical_anomalies = [a for a in report.anomalies_detected if a.severity == "critical"]
        if critical_anomalies:
            next_steps.append("优先解决关键异常问题后再继续训练")
        
        # 基于中文指标
        if report.chinese_metrics_summary:
            overall_score = report.chinese_metrics_summary.get("overall_score", 0)
            if overall_score < 0.8:
                next_steps.append("针对中文处理能力进行专项优化")
        
        # 基于GPU性能
        avg_utilization = np.mean([v for k, v in report.gpu_utilization_summary.items() 
                                  if "avg_utilization" in k])
        if avg_utilization < 70:
            next_steps.append("优化GPU利用率以提高训练效率")
        
        # 通用建议
        next_steps.extend([
            "定期保存模型检查点",
            "进行模型量化和导出准备",
            "准备部署和测试环境"
        ])
        
        return next_steps
    
    def _generate_visualizations(self,
                               report_id: str,
                               metrics_history: List[DistributedTrainingMetrics],
                               chinese_metrics_history: List[ChineseMetrics]) -> Dict[str, str]:
        """生成可视化图表"""
        viz_files = {}
        
        try:
            # 1. 损失曲线
            if metrics_history:
                loss_file = self._plot_loss_curves(report_id, metrics_history)
                viz_files["loss_curves"] = str(loss_file)
            
            # 2. GPU利用率图表
            if metrics_history:
                gpu_file = self._plot_gpu_metrics(report_id, metrics_history)
                viz_files["gpu_metrics"] = str(gpu_file)
            
            # 3. 中文指标趋势
            if chinese_metrics_history:
                chinese_file = self._plot_chinese_metrics(report_id, chinese_metrics_history)
                viz_files["chinese_metrics"] = str(chinese_file)
            
            # 4. 性能摘要仪表板
            if metrics_history:
                dashboard_file = self._plot_performance_dashboard(report_id, metrics_history)
                viz_files["performance_dashboard"] = str(dashboard_file)
                
        except Exception as e:
            logger.error(f"生成可视化图表时出错: {e}")
        
        return viz_files
    
    def _plot_loss_curves(self, report_id: str, metrics_history: List[DistributedTrainingMetrics]) -> Path:
        """绘制损失曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        steps = [m.global_step for m in metrics_history]
        train_losses = [m.train_loss for m in metrics_history if not np.isnan(m.train_loss)]
        val_losses = [m.val_loss for m in metrics_history if not np.isnan(m.val_loss) and m.val_loss > 0]
        
        # 训练损失
        if train_losses:
            train_steps = steps[:len(train_losses)]
            ax1.plot(train_steps, train_losses, label='训练损失', color='blue', linewidth=2)
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('损失值')
            ax1.set_title('训练损失曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 验证损失
        if val_losses:
            val_steps = [steps[i] for i, m in enumerate(metrics_history) 
                        if not np.isnan(m.val_loss) and m.val_loss > 0]
            ax2.plot(val_steps, val_losses, label='验证损失', color='red', linewidth=2)
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('损失值')
            ax2.set_title('验证损失曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = self.output_dir / f"{report_id}_loss_curves.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    def _plot_gpu_metrics(self, report_id: str, metrics_history: List[DistributedTrainingMetrics]) -> Path:
        """绘制GPU指标图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        steps = [m.global_step for m in metrics_history]
        
        # 收集GPU数据
        gpu_data = {}
        for metrics in metrics_history:
            for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
                if gpu_id not in gpu_data:
                    gpu_data[gpu_id] = {
                        "utilization": [],
                        "memory_usage": [],
                        "temperature": [],
                        "power_usage": []
                    }
                gpu_data[gpu_id]["utilization"].append(gpu_metrics.get("utilization", 0))
                gpu_data[gpu_id]["memory_usage"].append(gpu_metrics.get("memory_usage_percent", 0))
                gpu_data[gpu_id]["temperature"].append(gpu_metrics.get("temperature", 0))
                gpu_data[gpu_id]["power_usage"].append(gpu_metrics.get("power_usage", 0))
        
        # GPU利用率
        for gpu_id, data in gpu_data.items():
            ax1.plot(steps, data["utilization"], label=f'GPU {gpu_id}', linewidth=2)
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('利用率 (%)')
        ax1.set_title('GPU利用率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 内存使用
        for gpu_id, data in gpu_data.items():
            ax2.plot(steps, data["memory_usage"], label=f'GPU {gpu_id}', linewidth=2)
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('内存使用率 (%)')
        ax2.set_title('GPU内存使用率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 温度
        for gpu_id, data in gpu_data.items():
            if any(t > 0 for t in data["temperature"]):
                ax3.plot(steps, data["temperature"], label=f'GPU {gpu_id}', linewidth=2)
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('温度 (°C)')
        ax3.set_title('GPU温度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 功耗
        for gpu_id, data in gpu_data.items():
            if any(p > 0 for p in data["power_usage"]):
                ax4.plot(steps, data["power_usage"], label=f'GPU {gpu_id}', linewidth=2)
        ax4.set_xlabel('训练步数')
        ax4.set_ylabel('功耗 (W)')
        ax4.set_title('GPU功耗')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = self.output_dir / f"{report_id}_gpu_metrics.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    def _plot_chinese_metrics(self, report_id: str, chinese_metrics_history: List[ChineseMetrics]) -> Path:
        """绘制中文指标趋势"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        steps = list(range(len(chinese_metrics_history)))
        
        # 准确率指标
        char_acc = [m.character_accuracy for m in chinese_metrics_history]
        word_acc = [m.word_accuracy for m in chinese_metrics_history]
        crypto_acc = [m.crypto_term_accuracy for m in chinese_metrics_history]
        
        ax1.plot(steps, char_acc, label='字符准确率', linewidth=2)
        ax1.plot(steps, word_acc, label='词准确率', linewidth=2)
        ax1.plot(steps, crypto_acc, label='密码学术语准确率', linewidth=2)
        ax1.set_xlabel('评估轮次')
        ax1.set_ylabel('准确率')
        ax1.set_title('准确率指标趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 语义指标
        rouge_scores = [m.rouge_l_chinese for m in chinese_metrics_history]
        bleu_scores = [m.bleu_chinese for m in chinese_metrics_history]
        
        ax2.plot(steps, rouge_scores, label='ROUGE-L', linewidth=2)
        ax2.plot(steps, bleu_scores, label='BLEU', linewidth=2)
        ax2.set_xlabel('评估轮次')
        ax2.set_ylabel('评分')
        ax2.set_title('语义相似度指标')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 语言质量指标
        fluency_scores = [m.fluency_score for m in chinese_metrics_history]
        coherence_scores = [m.coherence_score for m in chinese_metrics_history]
        
        ax3.plot(steps, fluency_scores, label='流畅性', linewidth=2)
        ax3.plot(steps, coherence_scores, label='连贯性', linewidth=2)
        ax3.set_xlabel('评估轮次')
        ax3.set_ylabel('评分')
        ax3.set_title('语言质量指标')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 综合评分
        overall_scores = [m.overall_score() for m in chinese_metrics_history]
        ax4.plot(steps, overall_scores, label='综合评分', linewidth=3, color='red')
        ax4.set_xlabel('评估轮次')
        ax4.set_ylabel('综合评分')
        ax4.set_title('中文指标综合评分')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = self.output_dir / f"{report_id}_chinese_metrics.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    def _plot_performance_dashboard(self, report_id: str, metrics_history: List[DistributedTrainingMetrics]) -> Path:
        """绘制性能仪表板"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        steps = [m.global_step for m in metrics_history]
        
        # 吞吐量
        throughput = [m.throughput_tokens_per_second for m in metrics_history if m.throughput_tokens_per_second > 0]
        throughput_steps = [steps[i] for i, m in enumerate(metrics_history) if m.throughput_tokens_per_second > 0]
        
        if throughput:
            ax1.plot(throughput_steps, throughput, linewidth=2, color='green')
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('吞吐量 (tokens/s)')
            ax1.set_title('训练吞吐量')
            ax1.grid(True, alpha=0.3)
        
        # 负载均衡评分
        load_balance = [m.load_balance_score for m in metrics_history if m.load_balance_score > 0]
        balance_steps = [steps[i] for i, m in enumerate(metrics_history) if m.load_balance_score > 0]
        
        if load_balance:
            ax2.plot(balance_steps, load_balance, linewidth=2, color='orange')
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('负载均衡评分')
            ax2.set_title('负载均衡评分')
            ax2.grid(True, alpha=0.3)
        
        # 内存效率
        memory_eff = [m.memory_efficiency for m in metrics_history if m.memory_efficiency > 0]
        memory_steps = [steps[i] for i, m in enumerate(metrics_history) if m.memory_efficiency > 0]
        
        if memory_eff:
            ax3.plot(memory_steps, memory_eff, linewidth=2, color='purple')
            ax3.set_xlabel('训练步数')
            ax3.set_ylabel('内存效率')
            ax3.set_title('内存效率')
            ax3.grid(True, alpha=0.3)
        
        # 收敛评分
        convergence = [m.convergence_score for m in metrics_history if m.convergence_score > 0]
        conv_steps = [steps[i] for i, m in enumerate(metrics_history) if m.convergence_score > 0]
        
        if convergence:
            ax4.plot(conv_steps, convergence, linewidth=2, color='red')
            ax4.set_xlabel('训练步数')
            ax4.set_ylabel('收敛评分')
            ax4.set_title('收敛评分')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = self.output_dir / f"{report_id}_performance_dashboard.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    def _save_report(self, report: TrainingReport):
        """保存报告到文件"""
        # 保存JSON格式
        json_file = self.output_dir / f"{report.report_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 生成HTML报告
        html_file = self.output_dir / f"{report.report_id}.html"
        self._generate_html_report(report, html_file)
        
        logger.info(f"报告已保存: JSON={json_file}, HTML={html_file}")
    
    def _generate_html_report(self, report: TrainingReport, html_file: Path):
        """生成HTML格式报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>训练报告 - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9e9e9; border-radius: 3px; }}
                .anomaly {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 5px 0; }}
                .recommendation {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; padding: 10px; margin: 5px 0; }}
                .warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; margin: 5px 0; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>训练报告</h1>
                <p><strong>报告ID:</strong> {report.report_id}</p>
                <p><strong>生成时间:</strong> {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>训练时长:</strong> {report.training_duration}</p>
            </div>
            
            <div class="section">
                <h2>训练摘要</h2>
                <div class="metric"><strong>总轮次:</strong> {report.total_epochs}</div>
                <div class="metric"><strong>总步数:</strong> {report.total_steps}</div>
                <div class="metric"><strong>最终训练损失:</strong> {report.final_train_loss:.4f}</div>
                <div class="metric"><strong>最终验证损失:</strong> {report.final_val_loss:.4f}</div>
                <div class="metric"><strong>最佳训练损失:</strong> {report.best_train_loss:.4f}</div>
                <div class="metric"><strong>最佳验证损失:</strong> {report.best_val_loss:.4f}</div>
                <div class="metric"><strong>收敛状态:</strong> {'已收敛' if report.convergence_achieved else '未收敛'}</div>
            </div>
        """
        
        # 中文指标部分
        if report.chinese_metrics_summary:
            html_content += """
            <div class="section">
                <h2>中文指标摘要</h2>
                <table>
                    <tr><th>指标</th><th>数值</th></tr>
            """
            for metric, value in report.chinese_metrics_summary.items():
                html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
            html_content += "</table></div>"
        
        # 异常检测部分
        if report.anomalies_detected:
            html_content += "<div class='section'><h2>检测到的异常</h2>"
            for anomaly in report.anomalies_detected:
                html_content += f"""
                <div class="anomaly">
                    <strong>{anomaly.anomaly_type}</strong> ({anomaly.severity})
                    <p>{anomaly.description}</p>
                    <p><strong>建议操作:</strong> {', '.join(anomaly.suggested_actions)}</p>
                </div>
                """
            html_content += "</div>"
        
        # 警告部分
        if report.warnings:
            html_content += "<div class='section'><h2>警告</h2>"
            for warning in report.warnings:
                html_content += f"<div class='warning'>{warning}</div>"
            html_content += "</div>"
        
        # 优化建议部分
        if report.optimization_recommendations:
            html_content += "<div class='section'><h2>优化建议</h2>"
            for rec in report.optimization_recommendations:
                html_content += f"<div class='recommendation'>{rec}</div>"
            html_content += "</div>"
        
        # 可视化图表部分
        if report.visualization_files:
            html_content += "<div class='section'><h2>可视化图表</h2>"
            for chart_name, file_path in report.visualization_files.items():
                if Path(file_path).exists():
                    html_content += f"<h3>{chart_name}</h3><img src='{Path(file_path).name}' alt='{chart_name}'>"
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


# 便捷函数
def create_anomaly_detector(**kwargs) -> AnomalyDetector:
    """创建异常检测器的便捷函数"""
    return AnomalyDetector(**kwargs)


def create_report_generator(output_dir: str = "reports") -> TrainingReportGenerator:
    """创建报告生成器的便捷函数"""
    return TrainingReportGenerator(output_dir)