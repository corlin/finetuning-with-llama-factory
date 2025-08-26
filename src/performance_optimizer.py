#!/usr/bin/env python3
"""
性能优化和调优模块

本模块实现了训练性能的分析、优化和调优功能，包括：
- 训练性能瓶颈分析和内存使用分析
- 数据加载和预处理性能优化
- 多GPU通信和负载均衡调优
- 自动超参数调优建议
- 性能优化验证和测试
"""

import os
import sys
import time
import logging
import statistics
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
from collections import deque, defaultdict
from enum import Enum

import torch
import torch.distributed as dist
import psutil

from src.training_monitor import TrainingMonitor, GPUMonitoringMetrics
from src.memory_manager import MemoryManager, MemorySnapshot, MemoryPressureLevel
from src.distributed_training_engine import MultiGPUProcessManager
from src.parallel_strategy_recommender import ParallelStrategyRecommender, StrategyRecommendation
from src.gpu_utils import GPUDetector, GPUTopology
from src.parallel_config import DistributedTrainingMetrics, CommunicationMetrics


class BottleneckType(Enum):
    """性能瓶颈类型"""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    GPU_COMPUTE_BOUND = "gpu_compute_bound"
    GPU_MEMORY_BOUND = "gpu_memory_bound"
    IO_BOUND = "io_bound"
    COMMUNICATION_BOUND = "communication_bound"
    LOAD_IMBALANCE = "load_imbalance"


class OptimizationStrategy(Enum):
    """优化策略类型"""
    BATCH_SIZE_TUNING = "batch_size_tuning"
    LEARNING_RATE_TUNING = "learning_rate_tuning"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    DATA_LOADING_OPTIMIZATION = "data_loading_optimization"
    COMMUNICATION_OPTIMIZATION = "communication_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PARALLEL_STRATEGY_OPTIMIZATION = "parallel_strategy_optimization"


@dataclass
class PerformanceBottleneck:
    """性能瓶颈分析结果"""
    bottleneck_type: BottleneckType
    severity: float  # 0-1, 1为最严重
    description: str
    affected_components: List[str]
    metrics: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity,
            "description": self.description,
            "affected_components": self.affected_components,
            "metrics": self.metrics,
            "recommendations": self.recommendations
        }


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    strategy: OptimizationStrategy
    priority: int  # 1-10, 10为最高优先级
    description: str
    expected_improvement: float  # 预期性能提升百分比
    implementation_difficulty: int  # 1-5, 5为最难实现
    parameters: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "priority": self.priority,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "implementation_difficulty": self.implementation_difficulty,
            "parameters": self.parameters,
            "side_effects": self.side_effects
        }


@dataclass
class HyperparameterSuggestion:
    """超参数调优建议"""
    parameter_name: str
    current_value: Any
    suggested_value: Any
    reasoning: str
    confidence: float  # 0-1
    expected_impact: str  # "positive", "negative", "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "expected_impact": self.expected_impact
        }


class PerformanceBottleneckAnalyzer:
    """性能瓶颈分析器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_detector = GPUDetector()
        
        # 瓶颈检测阈值
        self.thresholds = {
            "cpu_utilization_high": 90.0,
            "memory_utilization_high": 85.0,
            "gpu_utilization_low": 60.0,
            "gpu_memory_high": 90.0,
            "communication_overhead_high": 30.0,  # 通信时间占比
            "load_imbalance_threshold": 20.0,  # GPU间负载差异百分比
            "io_wait_high": 20.0  # IO等待时间占比
        }
    
    def analyze_training_bottlenecks(self, 
                                   training_metrics: List[DistributedTrainingMetrics],
                                   memory_snapshots: Dict[int, List[MemorySnapshot]],
                                   system_metrics: Optional[Dict[str, Any]] = None) -> List[PerformanceBottleneck]:
        """分析训练性能瓶颈"""
        bottlenecks = []
        
        if not training_metrics:
            self.logger.warning("没有训练指标数据，无法分析瓶颈")
            return bottlenecks
        
        # 分析GPU计算瓶颈
        gpu_bottlenecks = self._analyze_gpu_bottlenecks(training_metrics, memory_snapshots)
        bottlenecks.extend(gpu_bottlenecks)
        
        # 分析内存瓶颈
        memory_bottlenecks = self._analyze_memory_bottlenecks(memory_snapshots)
        bottlenecks.extend(memory_bottlenecks)
        
        # 分析通信瓶颈
        communication_bottlenecks = self._analyze_communication_bottlenecks(training_metrics)
        bottlenecks.extend(communication_bottlenecks)
        
        # 分析负载均衡
        load_balance_bottlenecks = self._analyze_load_balance(training_metrics)
        bottlenecks.extend(load_balance_bottlenecks)
        
        # 分析系统资源瓶颈
        if system_metrics:
            system_bottlenecks = self._analyze_system_bottlenecks(system_metrics)
            bottlenecks.extend(system_bottlenecks)
        
        # 按严重程度排序
        bottlenecks.sort(key=lambda x: x.severity, reverse=True)
        
        return bottlenecks
    
    def _analyze_gpu_bottlenecks(self, 
                               training_metrics: List[DistributedTrainingMetrics],
                               memory_snapshots: Dict[int, List[MemorySnapshot]]) -> List[PerformanceBottleneck]:
        """分析GPU相关瓶颈"""
        bottlenecks = []
        
        # 分析GPU利用率
        gpu_utilizations = defaultdict(list)
        for metric in training_metrics:
            for gpu_id, gpu_metric in metric.gpu_metrics.items():
                utilization = gpu_metric.get("utilization", 0)
                gpu_utilizations[gpu_id].append(utilization)
        
        for gpu_id, utilizations in gpu_utilizations.items():
            if utilizations:
                avg_utilization = statistics.mean(utilizations)
                if avg_utilization < self.thresholds["gpu_utilization_low"]:
                    bottleneck = PerformanceBottleneck(
                        bottleneck_type=BottleneckType.GPU_COMPUTE_BOUND,
                        severity=1.0 - (avg_utilization / 100.0),
                        description=f"GPU {gpu_id} 利用率过低: {avg_utilization:.1f}%",
                        affected_components=[f"gpu_{gpu_id}"],
                        metrics={"avg_utilization": avg_utilization},
                        recommendations=[
                            "增加批次大小以提高GPU利用率",
                            "检查数据加载是否成为瓶颈",
                            "考虑使用混合精度训练",
                            "优化模型计算图"
                        ]
                    )
                    bottlenecks.append(bottleneck)
        
        # 分析GPU内存使用
        for gpu_id, snapshots in memory_snapshots.items():
            if snapshots:
                memory_utilizations = [s.utilization_rate for s in snapshots]
                avg_memory_util = statistics.mean(memory_utilizations)
                max_memory_util = max(memory_utilizations)
                
                if max_memory_util > self.thresholds["gpu_memory_high"] / 100.0:
                    bottleneck = PerformanceBottleneck(
                        bottleneck_type=BottleneckType.GPU_MEMORY_BOUND,
                        severity=max_memory_util,
                        description=f"GPU {gpu_id} 内存使用率过高: {max_memory_util*100:.1f}%",
                        affected_components=[f"gpu_{gpu_id}"],
                        metrics={
                            "avg_memory_utilization": avg_memory_util,
                            "max_memory_utilization": max_memory_util
                        },
                        recommendations=[
                            "减小批次大小",
                            "启用梯度检查点",
                            "使用混合精度训练",
                            "启用CPU卸载"
                        ]
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_memory_bottlenecks(self, 
                                  memory_snapshots: Dict[int, List[MemorySnapshot]]) -> List[PerformanceBottleneck]:
        """分析内存瓶颈"""
        bottlenecks = []
        
        for gpu_id, snapshots in memory_snapshots.items():
            if not snapshots:
                continue
            
            # 分析内存压力
            critical_pressure_count = sum(
                1 for s in snapshots 
                if s.pressure_level == MemoryPressureLevel.CRITICAL
            )
            
            if critical_pressure_count > len(snapshots) * 0.1:  # 超过10%的时间处于临界状态
                severity = critical_pressure_count / len(snapshots)
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.MEMORY_BOUND,
                    severity=severity,
                    description=f"GPU {gpu_id} 频繁出现内存压力",
                    affected_components=[f"gpu_{gpu_id}"],
                    metrics={
                        "critical_pressure_ratio": severity,
                        "total_snapshots": len(snapshots)
                    },
                    recommendations=[
                        "启用动态批次大小调整",
                        "使用梯度累积",
                        "启用内存优化策略",
                        "考虑模型并行"
                    ]
                )
                bottlenecks.append(bottleneck)
            
            # 分析内存碎片
            memory_efficiencies = []
            for snapshot in snapshots:
                if snapshot.total_memory > 0:
                    efficiency = snapshot.allocated_memory / snapshot.total_memory
                    memory_efficiencies.append(efficiency)
            
            if memory_efficiencies:
                avg_efficiency = statistics.mean(memory_efficiencies)
                if avg_efficiency < 0.7:  # 内存效率低于70%
                    bottleneck = PerformanceBottleneck(
                        bottleneck_type=BottleneckType.MEMORY_BOUND,
                        severity=1.0 - avg_efficiency,
                        description=f"GPU {gpu_id} 内存使用效率低: {avg_efficiency*100:.1f}%",
                        affected_components=[f"gpu_{gpu_id}"],
                        metrics={"memory_efficiency": avg_efficiency},
                        recommendations=[
                            "清理CUDA缓存",
                            "优化内存分配策略",
                            "使用内存池",
                            "减少中间变量"
                        ]
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_communication_bottlenecks(self, 
                                         training_metrics: List[DistributedTrainingMetrics]) -> List[PerformanceBottleneck]:
        """分析通信瓶颈"""
        bottlenecks = []
        
        if len(training_metrics) < 2:
            return bottlenecks
        
        # 分析通信开销
        communication_times = []
        total_times = []
        
        for metric in training_metrics:
            if hasattr(metric, 'communication_metrics') and metric.communication_metrics:
                comm_time = metric.communication_metrics.total_communication_time
                if comm_time > 0:
                    # 估算总时间（简化）
                    total_time = comm_time / 0.1  # 假设通信占10%
                    
                    communication_times.append(comm_time)
                    total_times.append(total_time)
        
        if communication_times and total_times:
            comm_ratios = [
                comm_time / total_time for comm_time, total_time in zip(communication_times, total_times)
                if total_time > 0
            ]
            
            if comm_ratios:
                avg_comm_ratio = statistics.mean(comm_ratios)
            else:
                return bottlenecks
            
            if avg_comm_ratio > self.thresholds["communication_overhead_high"] / 100.0:
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.COMMUNICATION_BOUND,
                    severity=avg_comm_ratio,
                    description=f"通信开销过高: {avg_comm_ratio*100:.1f}%",
                    affected_components=["distributed_training"],
                    metrics={"communication_overhead_ratio": avg_comm_ratio},
                    recommendations=[
                        "启用梯度压缩",
                        "优化通信拓扑",
                        "使用更高带宽的互联",
                        "调整并行策略",
                        "启用通信与计算重叠"
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_load_balance(self, 
                            training_metrics: List[DistributedTrainingMetrics]) -> List[PerformanceBottleneck]:
        """分析负载均衡"""
        bottlenecks = []
        
        # 分析GPU间负载差异
        for metric in training_metrics:
            if len(metric.gpu_metrics) > 1:
                utilizations = [
                    gpu_metric.get("utilization", 0) 
                    for gpu_metric in metric.gpu_metrics.values()
                ]
                
                if utilizations:
                    max_util = max(utilizations)
                    min_util = min(utilizations)
                    
                    if max_util > 0:
                        load_imbalance = (max_util - min_util) / max_util * 100
                        
                        if load_imbalance > self.thresholds["load_imbalance_threshold"]:
                            bottleneck = PerformanceBottleneck(
                                bottleneck_type=BottleneckType.LOAD_IMBALANCE,
                                severity=load_imbalance / 100.0,
                                description=f"GPU负载不均衡: 最大差异{load_imbalance:.1f}%",
                                affected_components=["distributed_training"],
                                metrics={
                                    "load_imbalance_percent": load_imbalance,
                                    "max_utilization": max_util,
                                    "min_utilization": min_util
                                },
                                recommendations=[
                                    "检查数据分布是否均匀",
                                    "优化数据加载策略",
                                    "调整并行策略",
                                    "检查GPU硬件差异"
                                ]
                            )
                            bottlenecks.append(bottleneck)
                            break  # 只报告一次负载不均衡
        
        return bottlenecks
    
    def _analyze_system_bottlenecks(self, system_metrics: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """分析系统资源瓶颈"""
        bottlenecks = []
        
        # 分析CPU使用率
        cpu_utilization = system_metrics.get("cpu_utilization", 0)
        if cpu_utilization > self.thresholds["cpu_utilization_high"]:
            bottleneck = PerformanceBottleneck(
                bottleneck_type=BottleneckType.CPU_BOUND,
                severity=cpu_utilization / 100.0,
                description=f"CPU使用率过高: {cpu_utilization:.1f}%",
                affected_components=["system"],
                metrics={"cpu_utilization": cpu_utilization},
                recommendations=[
                    "优化数据预处理",
                    "增加数据加载进程数",
                    "使用更快的CPU",
                    "减少CPU密集型操作"
                ]
            )
            bottlenecks.append(bottleneck)
        
        # 分析内存使用率
        memory_utilization = system_metrics.get("memory_utilization", 0)
        if memory_utilization > self.thresholds["memory_utilization_high"]:
            bottleneck = PerformanceBottleneck(
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                severity=memory_utilization / 100.0,
                description=f"系统内存使用率过高: {memory_utilization:.1f}%",
                affected_components=["system"],
                metrics={"memory_utilization": memory_utilization},
                recommendations=[
                    "增加系统内存",
                    "优化内存使用",
                    "启用内存映射",
                    "减少缓存大小"
                ]
            )
            bottlenecks.append(bottleneck)
        
        # 分析IO等待
        io_wait = system_metrics.get("io_wait", 0)
        if io_wait > self.thresholds["io_wait_high"]:
            bottleneck = PerformanceBottleneck(
                bottleneck_type=BottleneckType.IO_BOUND,
                severity=io_wait / 100.0,
                description=f"IO等待时间过高: {io_wait:.1f}%",
                affected_components=["storage"],
                metrics={"io_wait": io_wait},
                recommendations=[
                    "使用更快的存储设备",
                    "增加数据预加载",
                    "优化数据格式",
                    "使用数据缓存"
                ]
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks


class DataLoadingOptimizer:
    """数据加载优化器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def analyze_data_loading_performance(self, 
                                       data_loading_times: List[float],
                                       batch_processing_times: List[float]) -> Dict[str, Any]:
        """分析数据加载性能"""
        analysis = {}
        
        if data_loading_times:
            analysis["data_loading"] = {
                "avg_time": statistics.mean(data_loading_times),
                "max_time": max(data_loading_times),
                "min_time": min(data_loading_times),
                "std_time": statistics.stdev(data_loading_times) if len(data_loading_times) > 1 else 0
            }
        
        if batch_processing_times:
            analysis["batch_processing"] = {
                "avg_time": statistics.mean(batch_processing_times),
                "max_time": max(batch_processing_times),
                "min_time": min(batch_processing_times),
                "std_time": statistics.stdev(batch_processing_times) if len(batch_processing_times) > 1 else 0
            }
        
        # 计算数据加载占比
        if data_loading_times and batch_processing_times:
            total_data_time = sum(data_loading_times)
            total_batch_time = sum(batch_processing_times)
            total_time = total_data_time + total_batch_time
            
            if total_time > 0:
                analysis["data_loading_ratio"] = total_data_time / total_time
                analysis["batch_processing_ratio"] = total_batch_time / total_time
        
        return analysis
    
    def generate_data_loading_recommendations(self, 
                                            performance_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """生成数据加载优化建议"""
        recommendations = []
        
        data_loading_ratio = performance_analysis.get("data_loading_ratio", 0)
        
        # 如果数据加载时间占比过高
        if data_loading_ratio > 0.3:  # 超过30%
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.DATA_LOADING_OPTIMIZATION,
                priority=8,
                description="数据加载时间占比过高，需要优化数据加载流程",
                expected_improvement=20.0,
                implementation_difficulty=3,
                parameters={
                    "num_workers": "建议增加到CPU核心数",
                    "pin_memory": True,
                    "prefetch_factor": 2
                },
                side_effects=["增加内存使用", "可能增加CPU使用率"]
            ))
        
        # 检查数据加载时间变异性
        data_loading_stats = performance_analysis.get("data_loading", {})
        if data_loading_stats:
            avg_time = data_loading_stats.get("avg_time", 0)
            std_time = data_loading_stats.get("std_time", 0)
            
            if avg_time > 0 and std_time / avg_time > 0.5:  # 变异系数大于0.5
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.DATA_LOADING_OPTIMIZATION,
                    priority=6,
                    description="数据加载时间不稳定，建议优化数据预处理",
                    expected_improvement=15.0,
                    implementation_difficulty=4,
                    parameters={
                        "persistent_workers": True,
                        "cache_preprocessed_data": True
                    },
                    side_effects=["增加内存使用", "增加存储空间需求"]
                ))
        
        return recommendations


class CommunicationOptimizer:
    """通信优化器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_communication_patterns(self, 
                                     communication_metrics: List[CommunicationMetrics]) -> Dict[str, Any]:
        """分析通信模式"""
        analysis = {}
        
        if not communication_metrics:
            return analysis
        
        # 分析AllReduce性能
        allreduce_times = [m.allreduce_time for m in communication_metrics if m.allreduce_time > 0]
        if allreduce_times:
            analysis["allreduce"] = {
                "avg_time": statistics.mean(allreduce_times),
                "max_time": max(allreduce_times),
                "total_time": sum(allreduce_times)
            }
        
        # 分析通信量
        communication_volumes = [m.communication_volume for m in communication_metrics if m.communication_volume > 0]
        if communication_volumes:
            analysis["communication_volume"] = {
                "avg_volume": statistics.mean(communication_volumes),
                "total_volume": sum(communication_volumes)
            }
        
        # 计算通信效率
        if allreduce_times and communication_volumes:
            total_time = sum(allreduce_times)
            total_volume = sum(communication_volumes)
            if total_time > 0:
                analysis["communication_bandwidth"] = total_volume / total_time  # MB/s
        
        return analysis
    
    def generate_communication_recommendations(self, 
                                             communication_analysis: Dict[str, Any],
                                             gpu_topology: GPUTopology) -> List[OptimizationRecommendation]:
        """生成通信优化建议"""
        recommendations = []
        
        # 检查通信带宽
        bandwidth = communication_analysis.get("communication_bandwidth", 0)
        if bandwidth > 0 and bandwidth < 1000:  # 低于1GB/s
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.COMMUNICATION_OPTIMIZATION,
                priority=7,
                description=f"通信带宽较低({bandwidth:.1f} MB/s)，建议优化通信策略",
                expected_improvement=25.0,
                implementation_difficulty=3,
                parameters={
                    "compression": "启用梯度压缩",
                    "bucket_size": "调整通信桶大小",
                    "overlap_computation": "启用计算通信重叠"
                }
            ))
        
        # 检查GPU拓扑
        if gpu_topology.num_gpus > 1:
            has_nvlink = any(
                interconnect.interconnect_type.value == "nvlink" 
                for interconnect in gpu_topology.interconnects
            )
            
            if not has_nvlink:
                recommendations.append(OptimizationRecommendation(
                    strategy=OptimizationStrategy.COMMUNICATION_OPTIMIZATION,
                    priority=5,
                    description="缺少高速互联，建议优化通信模式",
                    expected_improvement=15.0,
                    implementation_difficulty=2,
                    parameters={
                        "reduce_communication_frequency": True,
                        "use_hierarchical_allreduce": True
                    }
                ))
        
        return recommendations


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_training_dynamics(self, 
                                training_metrics: List[DistributedTrainingMetrics]) -> Dict[str, Any]:
        """分析训练动态"""
        analysis = {}
        
        if not training_metrics:
            return analysis
        
        # 分析损失趋势
        train_losses = [m.train_loss for m in training_metrics if m.train_loss > 0]
        if len(train_losses) > 1:
            # 计算损失变化趋势
            loss_changes = [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
            analysis["loss_trend"] = {
                "avg_change": statistics.mean(loss_changes),
                "decreasing_ratio": sum(1 for change in loss_changes if change < 0) / len(loss_changes)
            }
        
        # 分析学习率
        learning_rates = [m.learning_rate for m in training_metrics if m.learning_rate > 0]
        if learning_rates:
            analysis["learning_rate"] = {
                "current": learning_rates[-1],
                "avg": statistics.mean(learning_rates)
            }
        
        # 分析收敛性
        convergence_scores = [m.convergence_score for m in training_metrics if m.convergence_score > 0]
        if convergence_scores:
            analysis["convergence"] = {
                "current_score": convergence_scores[-1],
                "avg_score": statistics.mean(convergence_scores),
                "improving": len(convergence_scores) > 1 and convergence_scores[-1] > convergence_scores[-2]
            }
        
        return analysis
    
    def generate_hyperparameter_suggestions(self, 
                                          training_analysis: Dict[str, Any],
                                          current_config: Dict[str, Any]) -> List[HyperparameterSuggestion]:
        """生成超参数调优建议"""
        suggestions = []
        
        # 学习率调优
        loss_trend = training_analysis.get("loss_trend", {})
        if loss_trend:
            decreasing_ratio = loss_trend.get("decreasing_ratio", 0)
            avg_change = loss_trend.get("avg_change", 0)
            
            current_lr = current_config.get("learning_rate", 2e-4)
            
            if decreasing_ratio < 0.3:  # 损失下降比例低
                if avg_change > 0:  # 损失在上升
                    suggestions.append(HyperparameterSuggestion(
                        parameter_name="learning_rate",
                        current_value=current_lr,
                        suggested_value=current_lr * 0.5,
                        reasoning="损失上升，建议降低学习率",
                        confidence=0.8,
                        expected_impact="positive"
                    ))
                else:  # 损失下降缓慢
                    suggestions.append(HyperparameterSuggestion(
                        parameter_name="learning_rate",
                        current_value=current_lr,
                        suggested_value=current_lr * 1.2,
                        reasoning="损失下降缓慢，可以尝试提高学习率",
                        confidence=0.6,
                        expected_impact="positive"
                    ))
        
        # 批次大小调优
        convergence_info = training_analysis.get("convergence", {})
        if convergence_info:
            current_score = convergence_info.get("current_score", 0)
            improving = convergence_info.get("improving", False)
            
            current_batch_size = current_config.get("batch_size", 4)
            
            if current_score < 0.5 and not improving:
                suggestions.append(HyperparameterSuggestion(
                    parameter_name="batch_size",
                    current_value=current_batch_size,
                    suggested_value=max(1, current_batch_size // 2),
                    reasoning="收敛性差，建议减小批次大小提高梯度质量",
                    confidence=0.7,
                    expected_impact="positive"
                ))
        
        return suggestions


class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(self, output_dir: str = "performance_optimization_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.bottleneck_analyzer = PerformanceBottleneckAnalyzer(self.logger)
        self.data_loading_optimizer = DataLoadingOptimizer(self.logger)
        self.communication_optimizer = CommunicationOptimizer(self.logger)
        self.hyperparameter_tuner = HyperparameterTuner(self.logger)
        self.strategy_recommender = ParallelStrategyRecommender()
        
        self.logger.info(f"性能优化器初始化完成，输出目录: {self.output_dir}")
    
    def analyze_and_optimize(self, 
                           training_metrics: List[DistributedTrainingMetrics],
                           memory_snapshots: Dict[int, List[MemorySnapshot]],
                           current_config: Dict[str, Any],
                           system_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """分析性能并生成优化建议"""
        self.logger.info("开始性能分析和优化...")
        
        optimization_report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "bottlenecks": [],
            "optimization_recommendations": [],
            "hyperparameter_suggestions": [],
            "parallel_strategy_recommendation": None,
            "summary": {}
        }
        
        # 1. 瓶颈分析
        bottlenecks = self.bottleneck_analyzer.analyze_training_bottlenecks(
            training_metrics, memory_snapshots, system_metrics
        )
        optimization_report["bottlenecks"] = [b.to_dict() for b in bottlenecks]
        
        # 2. 数据加载优化分析
        if system_metrics and "data_loading_times" in system_metrics:
            data_loading_analysis = self.data_loading_optimizer.analyze_data_loading_performance(
                system_metrics["data_loading_times"],
                system_metrics.get("batch_processing_times", [])
            )
            
            data_loading_recommendations = self.data_loading_optimizer.generate_data_loading_recommendations(
                data_loading_analysis
            )
            optimization_report["optimization_recommendations"].extend([r.to_dict() for r in data_loading_recommendations])
        
        # 3. 通信优化分析
        if training_metrics:
            communication_metrics = [m.communication_metrics for m in training_metrics if m.communication_metrics]
            if communication_metrics:
                communication_analysis = self.communication_optimizer.analyze_communication_patterns(
                    communication_metrics
                )
                
                gpu_detector = GPUDetector()
                gpu_topology = gpu_detector.detect_gpu_topology()
                
                communication_recommendations = self.communication_optimizer.generate_communication_recommendations(
                    communication_analysis, gpu_topology
                )
                optimization_report["optimization_recommendations"].extend([r.to_dict() for r in communication_recommendations])
        
        # 4. 超参数调优建议
        training_analysis = self.hyperparameter_tuner.analyze_training_dynamics(training_metrics)
        hyperparameter_suggestions = self.hyperparameter_tuner.generate_hyperparameter_suggestions(
            training_analysis, current_config
        )
        optimization_report["hyperparameter_suggestions"] = [s.to_dict() for s in hyperparameter_suggestions]
        
        # 5. 并行策略优化
        try:
            strategy_recommendation = self.strategy_recommender.recommend_strategy(
                batch_size=current_config.get("batch_size", 4),
                sequence_length=current_config.get("sequence_length", 2048),
                enable_lora=current_config.get("enable_lora", True),
                lora_rank=current_config.get("lora_rank", 64)
            )
            optimization_report["parallel_strategy_recommendation"] = {
                "strategy": strategy_recommendation.strategy.value,
                "confidence": strategy_recommendation.confidence,
                "reasoning": strategy_recommendation.reasoning,
                "expected_performance": strategy_recommendation.expected_performance
            }
        except Exception as e:
            self.logger.error(f"并行策略推荐失败: {e}")
        
        # 6. 生成优化摘要
        optimization_report["summary"] = self._generate_optimization_summary(
            bottlenecks, optimization_report["optimization_recommendations"], hyperparameter_suggestions
        )
        
        # 保存报告
        self._save_optimization_report(optimization_report)
        
        self.logger.info("性能分析和优化完成")
        return optimization_report
    
    def _generate_optimization_summary(self, 
                                     bottlenecks: List[PerformanceBottleneck],
                                     recommendations: List[Dict[str, Any]],
                                     hyperparameter_suggestions: List[HyperparameterSuggestion]) -> Dict[str, Any]:
        """生成优化摘要"""
        summary = {
            "total_bottlenecks": len(bottlenecks),
            "critical_bottlenecks": len([b for b in bottlenecks if b.severity > 0.8]),
            "total_recommendations": len(recommendations),
            "high_priority_recommendations": len([r for r in recommendations if r.get("priority", 0) >= 8]),
            "hyperparameter_suggestions_count": len(hyperparameter_suggestions),
            "expected_total_improvement": 0.0,
            "top_bottleneck_types": [],
            "top_optimization_strategies": []
        }
        
        # 计算预期总体改进
        expected_improvements = [r.get("expected_improvement", 0) for r in recommendations]
        if expected_improvements:
            # 使用平方根合并改进（避免过度乐观）
            summary["expected_total_improvement"] = np.sqrt(sum(imp**2 for imp in expected_improvements))
        
        # 统计主要瓶颈类型
        bottleneck_types = [b.bottleneck_type.value for b in bottlenecks]
        type_counts = {}
        for bt in bottleneck_types:
            type_counts[bt] = type_counts.get(bt, 0) + 1
        
        summary["top_bottleneck_types"] = sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        
        # 统计主要优化策略
        strategy_counts = {}
        for rec in recommendations:
            strategy = rec.get("strategy", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        summary["top_optimization_strategies"] = sorted(
            strategy_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        
        return summary
    
    def _save_optimization_report(self, report: Dict[str, Any]):
        """保存优化报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"optimization_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 同时保存最新报告
            latest_report_file = self.output_dir / "latest_optimization_report.json"
            with open(latest_report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"优化报告已保存到: {report_file}")
            
        except Exception as e:
            self.logger.error(f"保存优化报告失败: {e}")
    
    def apply_optimization_recommendations(self, 
                                         recommendations: List[Dict[str, Any]],
                                         current_config: Dict[str, Any]) -> Dict[str, Any]:
        """应用优化建议"""
        optimized_config = current_config.copy()
        applied_optimizations = []
        
        # 按优先级排序
        sorted_recommendations = sorted(
            recommendations, 
            key=lambda x: x.get("priority", 0), 
            reverse=True
        )
        
        for rec in sorted_recommendations:
            strategy = rec.get("strategy", "")
            parameters = rec.get("parameters", {})
            
            try:
                if strategy == "batch_size_tuning":
                    if "batch_size" in parameters:
                        optimized_config["batch_size"] = parameters["batch_size"]
                        applied_optimizations.append(f"调整批次大小为 {parameters['batch_size']}")
                
                elif strategy == "learning_rate_tuning":
                    if "learning_rate" in parameters:
                        optimized_config["learning_rate"] = parameters["learning_rate"]
                        applied_optimizations.append(f"调整学习率为 {parameters['learning_rate']}")
                
                elif strategy == "gradient_accumulation":
                    if "gradient_accumulation_steps" in parameters:
                        optimized_config["gradient_accumulation_steps"] = parameters["gradient_accumulation_steps"]
                        applied_optimizations.append(f"设置梯度累积步数为 {parameters['gradient_accumulation_steps']}")
                
                elif strategy == "mixed_precision":
                    optimized_config["enable_mixed_precision"] = True
                    applied_optimizations.append("启用混合精度训练")
                
                elif strategy == "gradient_checkpointing":
                    optimized_config["enable_gradient_checkpointing"] = True
                    applied_optimizations.append("启用梯度检查点")
                
                elif strategy == "data_loading_optimization":
                    for param, value in parameters.items():
                        optimized_config[param] = value
                        applied_optimizations.append(f"数据加载优化: {param} = {value}")
                
            except Exception as e:
                self.logger.error(f"应用优化建议失败 {strategy}: {e}")
        
        return {
            "optimized_config": optimized_config,
            "applied_optimizations": applied_optimizations
        }


def main():
    """主函数，用于测试性能优化功能"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    print("=== 性能优化器测试 ===")
    
    # 模拟训练指标
    training_metrics = []
    for i in range(10):
        metric = DistributedTrainingMetrics(
            epoch=1,
            global_step=i,
            train_loss=2.0 - i * 0.1,
            val_loss=2.1 - i * 0.1,
            learning_rate=2e-4,
            gpu_metrics={
                0: {"utilization": 50 + i * 2, "memory_usage_percent": 70 + i}
            },
            communication_metrics=CommunicationMetrics(),
            throughput_tokens_per_second=100 + i * 5,
            convergence_score=0.3 + i * 0.05
        )
        training_metrics.append(metric)
    
    # 模拟内存快照
    memory_snapshots = {0: []}
    for i in range(10):
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=10000 + i * 200,
            cached_memory=2000,
            free_memory=6384 - i * 200,
            utilization_rate=(10000 + i * 200) / 16384,
            pressure_level=MemoryPressureLevel.MODERATE,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=8192,
            process_memory_percent=25.0
        )
        memory_snapshots[0].append(snapshot)
    
    # 当前配置
    current_config = {
        "batch_size": 4,
        "learning_rate": 2e-4,
        "sequence_length": 2048,
        "enable_lora": True,
        "lora_rank": 64
    }
    
    # 运行优化分析
    optimization_report = optimizer.analyze_and_optimize(
        training_metrics, memory_snapshots, current_config
    )
    
    print(f"\n发现 {optimization_report['summary']['total_bottlenecks']} 个性能瓶颈")
    print(f"生成 {optimization_report['summary']['total_recommendations']} 个优化建议")
    print(f"预期总体性能提升: {optimization_report['summary']['expected_total_improvement']:.1f}%")
    
    if optimization_report["bottlenecks"]:
        print("\n主要性能瓶颈:")
        for bottleneck in optimization_report["bottlenecks"][:3]:
            print(f"  - {bottleneck['description']} (严重程度: {bottleneck['severity']:.2f})")
    
    if optimization_report["optimization_recommendations"]:
        print("\n高优先级优化建议:")
        high_priority_recs = [r for r in optimization_report["optimization_recommendations"] if r.get("priority", 0) >= 7]
        for rec in high_priority_recs[:3]:
            print(f"  - {rec['description']} (预期改进: {rec['expected_improvement']:.1f}%)")


if __name__ == "__main__":
    main()