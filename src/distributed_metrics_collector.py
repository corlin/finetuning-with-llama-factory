"""
分布式训练指标收集模块

本模块实现了分布式训练环境下的指标收集功能，包括跨GPU通信开销分析和统计、
负载均衡评估和优化建议生成、通信效率和带宽利用率监控等功能。
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import logging
import numpy as np
import torch
import torch.distributed as dist

from .parallel_config import DistributedTrainingMetrics, CommunicationMetrics, GPUInfo


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CommunicationEvent:
    """通信事件数据结构"""
    event_type: str  # allreduce, broadcast, p2p, barrier
    start_time: float
    end_time: float
    data_size: int  # bytes
    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None
    group_size: Optional[int] = None
    
    @property
    def duration(self) -> float:
        """通信持续时间（秒）"""
        return self.end_time - self.start_time
    
    @property
    def bandwidth_mbps(self) -> float:
        """带宽利用率（MB/s）"""
        if self.duration <= 0:
            return 0.0
        return (self.data_size / (1024 * 1024)) / self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "event_type": self.event_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "data_size": self.data_size,
            "bandwidth_mbps": self.bandwidth_mbps,
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "group_size": self.group_size
        }


@dataclass
class LoadBalanceMetrics:
    """负载均衡指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # GPU利用率统计
    gpu_utilizations: Dict[int, float] = field(default_factory=dict)
    mean_utilization: float = 0.0
    utilization_variance: float = 0.0
    utilization_std: float = 0.0
    
    # 内存使用统计
    memory_usages: Dict[int, float] = field(default_factory=dict)  # 百分比
    mean_memory_usage: float = 0.0
    memory_variance: float = 0.0
    memory_std: float = 0.0
    
    # 工作负载统计
    workload_distribution: Dict[int, float] = field(default_factory=dict)  # 每个GPU的工作量
    workload_imbalance_score: float = 0.0  # 0-1，0表示完全均衡
    
    # 通信负载统计
    communication_loads: Dict[int, float] = field(default_factory=dict)  # 每个GPU的通信负载
    communication_imbalance: float = 0.0
    
    def calculate_balance_scores(self):
        """计算负载均衡评分"""
        # 计算GPU利用率统计
        if self.gpu_utilizations:
            utils = list(self.gpu_utilizations.values())
            self.mean_utilization = np.mean(utils)
            self.utilization_variance = np.var(utils)
            self.utilization_std = np.std(utils)
        
        # 计算内存使用统计
        if self.memory_usages:
            mem_utils = list(self.memory_usages.values())
            self.mean_memory_usage = np.mean(mem_utils)
            self.memory_variance = np.var(mem_utils)
            self.memory_std = np.std(mem_utils)
        
        # 计算工作负载不均衡评分
        if self.workload_distribution:
            workloads = list(self.workload_distribution.values())
            if len(workloads) > 1:
                workload_std = np.std(workloads)
                workload_mean = np.mean(workloads)
                # 标准化不均衡评分
                self.workload_imbalance_score = workload_std / workload_mean if workload_mean > 0 else 0.0
        
        # 计算通信负载不均衡
        if self.communication_loads:
            comm_loads = list(self.communication_loads.values())
            if len(comm_loads) > 1:
                comm_std = np.std(comm_loads)
                comm_mean = np.mean(comm_loads)
                self.communication_imbalance = comm_std / comm_mean if comm_mean > 0 else 0.0
    
    @property
    def overall_balance_score(self) -> float:
        """综合负载均衡评分（0-1，1表示完全均衡）"""
        # 综合考虑各种不均衡指标
        util_balance = 1.0 - min(1.0, self.utilization_std / 100.0)  # GPU利用率均衡性
        mem_balance = 1.0 - min(1.0, self.memory_std / 100.0)        # 内存使用均衡性
        workload_balance = 1.0 - min(1.0, self.workload_imbalance_score)  # 工作负载均衡性
        comm_balance = 1.0 - min(1.0, self.communication_imbalance)  # 通信负载均衡性
        
        # 加权平均
        return 0.3 * util_balance + 0.3 * mem_balance + 0.25 * workload_balance + 0.15 * comm_balance
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_utilizations": self.gpu_utilizations,
            "mean_utilization": self.mean_utilization,
            "utilization_variance": self.utilization_variance,
            "utilization_std": self.utilization_std,
            "memory_usages": self.memory_usages,
            "mean_memory_usage": self.mean_memory_usage,
            "memory_variance": self.memory_variance,
            "memory_std": self.memory_std,
            "workload_distribution": self.workload_distribution,
            "workload_imbalance_score": self.workload_imbalance_score,
            "communication_loads": self.communication_loads,
            "communication_imbalance": self.communication_imbalance,
            "overall_balance_score": self.overall_balance_score
        }


@dataclass
class CommunicationEfficiencyMetrics:
    """通信效率指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 基础通信统计
    total_communication_time: float = 0.0  # 总通信时间（秒）
    total_computation_time: float = 0.0    # 总计算时间（秒）
    communication_computation_ratio: float = 0.0  # 通信/计算时间比
    
    # 不同类型通信的统计
    allreduce_stats: Dict[str, float] = field(default_factory=dict)  # 平均时间、总时间、次数等
    broadcast_stats: Dict[str, float] = field(default_factory=dict)
    p2p_stats: Dict[str, float] = field(default_factory=dict)
    
    # 带宽利用率
    peak_bandwidth_mbps: float = 0.0       # 峰值带宽
    average_bandwidth_mbps: float = 0.0    # 平均带宽
    bandwidth_utilization: float = 0.0     # 带宽利用率（相对于理论峰值）
    
    # 通信模式分析
    communication_patterns: Dict[str, int] = field(default_factory=dict)  # 通信模式统计
    hotspot_ranks: List[int] = field(default_factory=list)  # 通信热点rank
    
    # 效率评分
    communication_efficiency: float = 0.0  # 通信效率评分（0-1）
    
    def calculate_efficiency_metrics(self, events: List[CommunicationEvent]):
        """根据通信事件计算效率指标"""
        if not events:
            return
        
        # 按类型分组统计
        allreduce_events = [e for e in events if e.event_type == "allreduce"]
        broadcast_events = [e for e in events if e.event_type == "broadcast"]
        p2p_events = [e for e in events if e.event_type == "p2p"]
        
        # AllReduce统计
        if allreduce_events:
            durations = [e.duration for e in allreduce_events]
            bandwidths = [e.bandwidth_mbps for e in allreduce_events]
            self.allreduce_stats = {
                "count": len(allreduce_events),
                "total_time": sum(durations),
                "avg_time": np.mean(durations),
                "avg_bandwidth": np.mean(bandwidths),
                "max_bandwidth": max(bandwidths)
            }
        
        # Broadcast统计
        if broadcast_events:
            durations = [e.duration for e in broadcast_events]
            bandwidths = [e.bandwidth_mbps for e in broadcast_events]
            self.broadcast_stats = {
                "count": len(broadcast_events),
                "total_time": sum(durations),
                "avg_time": np.mean(durations),
                "avg_bandwidth": np.mean(bandwidths),
                "max_bandwidth": max(bandwidths)
            }
        
        # P2P统计
        if p2p_events:
            durations = [e.duration for e in p2p_events]
            bandwidths = [e.bandwidth_mbps for e in p2p_events]
            self.p2p_stats = {
                "count": len(p2p_events),
                "total_time": sum(durations),
                "avg_time": np.mean(durations),
                "avg_bandwidth": np.mean(bandwidths),
                "max_bandwidth": max(bandwidths)
            }
        
        # 总体统计
        self.total_communication_time = sum(e.duration for e in events)
        all_bandwidths = [e.bandwidth_mbps for e in events if e.bandwidth_mbps > 0]
        if all_bandwidths:
            self.peak_bandwidth_mbps = max(all_bandwidths)
            self.average_bandwidth_mbps = np.mean(all_bandwidths)
        
        # 通信模式分析
        self.communication_patterns = {
            "allreduce": len(allreduce_events),
            "broadcast": len(broadcast_events),
            "p2p": len(p2p_events)
        }
        
        # 计算通信效率评分
        self._calculate_efficiency_score()
    
    def _calculate_efficiency_score(self):
        """计算通信效率评分"""
        # 基于多个因素计算效率评分
        
        # 1. 通信/计算时间比（越小越好）
        ratio_score = 1.0 / (1.0 + self.communication_computation_ratio) if self.communication_computation_ratio > 0 else 1.0
        
        # 2. 带宽利用率（越高越好，但有上限）
        bandwidth_score = min(1.0, self.bandwidth_utilization / 0.8)  # 80%利用率为满分
        
        # 3. 通信模式效率（AllReduce通常比P2P更高效）
        total_comm = sum(self.communication_patterns.values())
        if total_comm > 0:
            allreduce_ratio = self.communication_patterns.get("allreduce", 0) / total_comm
            pattern_score = 0.5 + 0.5 * allreduce_ratio  # AllReduce比例越高评分越高
        else:
            pattern_score = 0.5
        
        # 综合评分
        self.communication_efficiency = 0.4 * ratio_score + 0.4 * bandwidth_score + 0.2 * pattern_score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_communication_time": self.total_communication_time,
            "total_computation_time": self.total_computation_time,
            "communication_computation_ratio": self.communication_computation_ratio,
            "allreduce_stats": self.allreduce_stats,
            "broadcast_stats": self.broadcast_stats,
            "p2p_stats": self.p2p_stats,
            "peak_bandwidth_mbps": self.peak_bandwidth_mbps,
            "average_bandwidth_mbps": self.average_bandwidth_mbps,
            "bandwidth_utilization": self.bandwidth_utilization,
            "communication_patterns": self.communication_patterns,
            "hotspot_ranks": self.hotspot_ranks,
            "communication_efficiency": self.communication_efficiency
        }


class CommunicationProfiler:
    """通信性能分析器"""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.active_events: Dict[str, CommunicationEvent] = {}
        self.profiling_enabled = False
        
        # 理论带宽配置（可根据硬件调整）
        self.theoretical_bandwidth_gbps = {
            "nvlink": 300.0,    # NVLink 3.0
            "infiniband": 100.0, # InfiniBand HDR
            "ethernet": 10.0     # 10GbE
        }
    
    def enable_profiling(self):
        """启用通信性能分析"""
        self.profiling_enabled = True
        logger.info("通信性能分析已启用")
    
    def disable_profiling(self):
        """禁用通信性能分析"""
        self.profiling_enabled = False
        logger.info("通信性能分析已禁用")
    
    def start_communication_event(self, event_type: str, data_size: int, 
                                 src_rank: Optional[int] = None, 
                                 dst_rank: Optional[int] = None,
                                 group_size: Optional[int] = None) -> str:
        """开始记录通信事件"""
        if not self.profiling_enabled:
            return ""
        
        event_id = f"{event_type}_{time.time()}_{id(self)}"
        event = CommunicationEvent(
            event_type=event_type,
            start_time=time.time(),
            end_time=0.0,
            data_size=data_size,
            src_rank=src_rank,
            dst_rank=dst_rank,
            group_size=group_size
        )
        
        self.active_events[event_id] = event
        return event_id
    
    def end_communication_event(self, event_id: str):
        """结束记录通信事件"""
        if not self.profiling_enabled or event_id not in self.active_events:
            return
        
        event = self.active_events.pop(event_id)
        event.end_time = time.time()
        self.events.append(event)
    
    def get_recent_events(self, count: int = 100) -> List[CommunicationEvent]:
        """获取最近的通信事件"""
        recent_events = list(self.events)
        return recent_events[-count:] if count > 0 else recent_events
    
    def get_events_by_type(self, event_type: str, count: int = 100) -> List[CommunicationEvent]:
        """按类型获取通信事件"""
        filtered_events = [e for e in self.events if e.event_type == event_type]
        return filtered_events[-count:] if count > 0 else filtered_events
    
    def clear_events(self):
        """清空事件历史"""
        self.events.clear()
        self.active_events.clear()


class DistributedMetricsCollector:
    """分布式训练指标收集器"""
    
    def __init__(self, 
                 world_size: int,
                 rank: int,
                 gpu_ids: List[int],
                 collection_interval: float = 1.0):
        """
        初始化分布式指标收集器
        
        Args:
            world_size: 总进程数
            rank: 当前进程rank
            gpu_ids: GPU ID列表
            collection_interval: 收集间隔（秒）
        """
        self.world_size = world_size
        self.rank = rank
        self.gpu_ids = gpu_ids
        self.collection_interval = collection_interval
        
        # 通信性能分析器
        self.comm_profiler = CommunicationProfiler()
        
        # 指标历史
        self.load_balance_history: deque = deque(maxlen=1000)
        self.communication_efficiency_history: deque = deque(maxlen=1000)
        
        # 收集状态
        self.collecting = False
        self.collection_thread = None
        
        # 工作负载跟踪
        self.workload_tracker: Dict[int, float] = defaultdict(float)
        self.computation_start_times: Dict[int, float] = {}
        
        logger.info(f"分布式指标收集器初始化完成: world_size={world_size}, rank={rank}")
    
    def start_collection(self):
        """开始指标收集"""
        if self.collecting:
            logger.warning("指标收集已在运行中")
            return
        
        self.collecting = True
        self.comm_profiler.enable_profiling()
        
        # 启动收集线程
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info("分布式指标收集已启动")
    
    def stop_collection(self):
        """停止指标收集"""
        if not self.collecting:
            return
        
        self.collecting = False
        self.comm_profiler.disable_profiling()
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        logger.info("分布式指标收集已停止")
    
    def _collection_loop(self):
        """指标收集循环"""
        while self.collecting:
            try:
                # 收集负载均衡指标
                load_balance_metrics = self._collect_load_balance_metrics()
                if load_balance_metrics:
                    self.load_balance_history.append(load_balance_metrics)
                
                # 收集通信效率指标
                comm_efficiency_metrics = self._collect_communication_efficiency_metrics()
                if comm_efficiency_metrics:
                    self.communication_efficiency_history.append(comm_efficiency_metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"指标收集出错: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_load_balance_metrics(self) -> Optional[LoadBalanceMetrics]:
        """收集负载均衡指标"""
        try:
            metrics = LoadBalanceMetrics()
            
            # 收集GPU利用率（这里需要与GPU监控器集成）
            # 简化实现：使用模拟数据
            for gpu_id in self.gpu_ids:
                # 实际实现中应该从GPU监控器获取真实数据
                metrics.gpu_utilizations[gpu_id] = np.random.uniform(70, 95)  # 模拟数据
                metrics.memory_usages[gpu_id] = np.random.uniform(60, 90)     # 模拟数据
                metrics.workload_distribution[gpu_id] = self.workload_tracker.get(gpu_id, 0.0)
                metrics.communication_loads[gpu_id] = np.random.uniform(10, 30)  # 模拟数据
            
            # 计算均衡评分
            metrics.calculate_balance_scores()
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集负载均衡指标失败: {e}")
            return None
    
    def _collect_communication_efficiency_metrics(self) -> Optional[CommunicationEfficiencyMetrics]:
        """收集通信效率指标"""
        try:
            metrics = CommunicationEfficiencyMetrics()
            
            # 获取最近的通信事件
            recent_events = self.comm_profiler.get_recent_events(100)
            
            if recent_events:
                # 计算效率指标
                metrics.calculate_efficiency_metrics(recent_events)
                
                # 设置理论带宽（简化实现）
                metrics.bandwidth_utilization = min(1.0, metrics.average_bandwidth_mbps / 1000.0)  # 假设1GB/s理论带宽
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集通信效率指标失败: {e}")
            return None
    
    def record_computation_start(self, gpu_id: int):
        """记录计算开始时间"""
        self.computation_start_times[gpu_id] = time.time()
    
    def record_computation_end(self, gpu_id: int, samples_processed: int = 1):
        """记录计算结束时间并更新工作负载"""
        if gpu_id in self.computation_start_times:
            duration = time.time() - self.computation_start_times[gpu_id]
            self.workload_tracker[gpu_id] += duration * samples_processed
            del self.computation_start_times[gpu_id]
    
    def record_allreduce(self, tensor_size: int) -> str:
        """记录AllReduce操作"""
        return self.comm_profiler.start_communication_event(
            "allreduce", 
            tensor_size, 
            group_size=self.world_size
        )
    
    def record_broadcast(self, tensor_size: int, src_rank: int) -> str:
        """记录Broadcast操作"""
        return self.comm_profiler.start_communication_event(
            "broadcast", 
            tensor_size, 
            src_rank=src_rank,
            group_size=self.world_size
        )
    
    def record_p2p_send(self, tensor_size: int, dst_rank: int) -> str:
        """记录P2P发送操作"""
        return self.comm_profiler.start_communication_event(
            "p2p", 
            tensor_size, 
            src_rank=self.rank,
            dst_rank=dst_rank
        )
    
    def finish_communication(self, event_id: str):
        """完成通信事件记录"""
        self.comm_profiler.end_communication_event(event_id)
    
    def get_current_load_balance_metrics(self) -> Optional[LoadBalanceMetrics]:
        """获取当前负载均衡指标"""
        if not self.load_balance_history:
            return None
        return self.load_balance_history[-1]
    
    def get_current_communication_efficiency(self) -> Optional[CommunicationEfficiencyMetrics]:
        """获取当前通信效率指标"""
        if not self.communication_efficiency_history:
            return None
        return self.communication_efficiency_history[-1]
    
    def get_load_balance_history(self, count: int = 100) -> List[LoadBalanceMetrics]:
        """获取负载均衡历史"""
        history = list(self.load_balance_history)
        return history[-count:] if count > 0 else history
    
    def get_communication_efficiency_history(self, count: int = 100) -> List[CommunicationEfficiencyMetrics]:
        """获取通信效率历史"""
        history = list(self.communication_efficiency_history)
        return history[-count:] if count > 0 else history
    
    def generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于负载均衡指标生成建议
        current_balance = self.get_current_load_balance_metrics()
        if current_balance:
            if current_balance.overall_balance_score < 0.7:
                recommendations.append("检测到负载不均衡，建议调整数据分片策略或GPU分配")
            
            if current_balance.utilization_std > 20:
                recommendations.append("GPU利用率差异较大，建议检查数据加载和预处理流程")
            
            if current_balance.memory_std > 15:
                recommendations.append("GPU内存使用不均衡，建议优化内存分配策略")
        
        # 基于通信效率指标生成建议
        current_comm = self.get_current_communication_efficiency()
        if current_comm:
            if current_comm.communication_efficiency < 0.6:
                recommendations.append("通信效率较低，建议优化通信模式或增加计算/通信重叠")
            
            if current_comm.communication_computation_ratio > 0.3:
                recommendations.append("通信开销过高，建议增加批次大小或减少通信频率")
            
            if current_comm.bandwidth_utilization < 0.5:
                recommendations.append("带宽利用率较低，建议检查网络配置或增加数据传输量")
        
        return recommendations
    
    def create_distributed_training_metrics(self, 
                                          epoch: int,
                                          global_step: int,
                                          train_loss: float,
                                          val_loss: float,
                                          learning_rate: float,
                                          gpu_metrics: Dict[int, Dict[str, float]]) -> DistributedTrainingMetrics:
        """创建分布式训练指标"""
        
        # 获取当前通信指标
        current_comm = self.get_current_communication_efficiency()
        comm_metrics = CommunicationMetrics()
        
        if current_comm:
            comm_metrics.total_communication_time = current_comm.total_communication_time
            comm_metrics.allreduce_time = current_comm.allreduce_stats.get("total_time", 0.0)
            comm_metrics.broadcast_time = current_comm.broadcast_stats.get("total_time", 0.0)
            comm_metrics.p2p_time = current_comm.p2p_stats.get("total_time", 0.0)
            comm_metrics.communication_volume = sum(
                event.data_size for event in self.comm_profiler.get_recent_events(10)
            ) / (1024 * 1024)  # 转换为MB
            comm_metrics.bandwidth_utilization = current_comm.bandwidth_utilization * 100
        
        # 获取当前负载均衡指标
        current_balance = self.get_current_load_balance_metrics()
        load_balance_score = current_balance.overall_balance_score if current_balance else 0.0
        
        # 创建分布式训练指标
        metrics = DistributedTrainingMetrics(
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            gpu_metrics=gpu_metrics,
            communication_metrics=comm_metrics,
            load_balance_score=load_balance_score
        )
        
        return metrics
    
    def export_metrics_summary(self) -> Dict[str, Any]:
        """导出指标摘要"""
        summary = {
            "collection_info": {
                "world_size": self.world_size,
                "rank": self.rank,
                "gpu_ids": self.gpu_ids,
                "collection_interval": self.collection_interval
            },
            "load_balance_summary": {},
            "communication_efficiency_summary": {},
            "optimization_recommendations": self.generate_optimization_recommendations()
        }
        
        # 负载均衡摘要
        if self.load_balance_history:
            recent_balance = list(self.load_balance_history)[-10:]  # 最近10个
            balance_scores = [b.overall_balance_score for b in recent_balance]
            summary["load_balance_summary"] = {
                "avg_balance_score": np.mean(balance_scores),
                "min_balance_score": min(balance_scores),
                "max_balance_score": max(balance_scores),
                "current_balance_score": balance_scores[-1] if balance_scores else 0.0
            }
        
        # 通信效率摘要
        if self.communication_efficiency_history:
            recent_comm = list(self.communication_efficiency_history)[-10:]  # 最近10个
            efficiency_scores = [c.communication_efficiency for c in recent_comm]
            summary["communication_efficiency_summary"] = {
                "avg_efficiency": np.mean(efficiency_scores),
                "min_efficiency": min(efficiency_scores),
                "max_efficiency": max(efficiency_scores),
                "current_efficiency": efficiency_scores[-1] if efficiency_scores else 0.0
            }
        
        return summary


# 上下文管理器，用于自动记录通信事件
class CommunicationContext:
    """通信上下文管理器"""
    
    def __init__(self, collector: DistributedMetricsCollector, 
                 event_type: str, tensor_size: int, **kwargs):
        self.collector = collector
        self.event_type = event_type
        self.tensor_size = tensor_size
        self.kwargs = kwargs
        self.event_id = None
    
    def __enter__(self):
        if self.event_type == "allreduce":
            self.event_id = self.collector.record_allreduce(self.tensor_size)
        elif self.event_type == "broadcast":
            self.event_id = self.collector.record_broadcast(
                self.tensor_size, self.kwargs.get("src_rank", 0)
            )
        elif self.event_type == "p2p":
            self.event_id = self.collector.record_p2p_send(
                self.tensor_size, self.kwargs.get("dst_rank", 0)
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.event_id:
            self.collector.finish_communication(self.event_id)


# 便捷函数
def create_distributed_metrics_collector(world_size: int, rank: int, gpu_ids: List[int]) -> DistributedMetricsCollector:
    """创建分布式指标收集器的便捷函数"""
    return DistributedMetricsCollector(world_size, rank, gpu_ids)


def with_communication_profiling(collector: DistributedMetricsCollector, 
                                event_type: str, tensor_size: int, **kwargs):
    """通信性能分析装饰器"""
    return CommunicationContext(collector, event_type, tensor_size, **kwargs)