"""
多GPU并行配置模型

本模块实现了多GPU并行训练的配置管理，包括GPU拓扑检测、并行策略配置、
分布式训练指标监控等功能。支持数据并行、模型并行、流水线并行等多种策略。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
import torch
from datetime import datetime


class ParallelStrategy(Enum):
    """并行策略枚举"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    AUTO = "auto"


class CommunicationBackend(Enum):
    """通信后端枚举"""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


class ZeroStage(Enum):
    """ZeRO优化阶段"""
    DISABLED = 0
    OPTIMIZER_STATE = 1
    OPTIMIZER_GRADIENT = 2
    OPTIMIZER_GRADIENT_PARAMETER = 3


@dataclass
class GPUInfo:
    """GPU信息数据结构"""
    gpu_id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    compute_capability: Tuple[int, int]
    temperature: float = 0.0
    power_usage: float = 0.0
    utilization: float = 0.0
    
    def __post_init__(self):
        """数据验证"""
        if self.gpu_id < 0:
            raise ValueError("GPU ID必须为非负整数")
        if self.memory_total <= 0:
            raise ValueError("GPU总内存必须大于0")
        if not 0 <= self.utilization <= 100:
            raise ValueError("GPU利用率必须在0-100之间")
    
    @property
    def memory_used(self) -> int:
        """计算已使用内存"""
        return self.memory_total - self.memory_free
    
    @property
    def memory_usage_percent(self) -> float:
        """计算内存使用百分比"""
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "memory_total": self.memory_total,
            "memory_free": self.memory_free,
            "memory_used": self.memory_used,
            "memory_usage_percent": self.memory_usage_percent,
            "compute_capability": self.compute_capability,
            "temperature": self.temperature,
            "power_usage": self.power_usage,
            "utilization": self.utilization
        }


@dataclass
class GPUTopology:
    """GPU拓扑结构"""
    num_gpus: int
    gpu_info: Dict[int, GPUInfo]
    interconnect_bandwidth: Dict[Tuple[int, int], float]  # GB/s
    numa_topology: Dict[int, int]  # GPU ID -> NUMA节点
    pcie_topology: Dict[int, str] = field(default_factory=dict)  # GPU ID -> PCIe路径
    nvlink_connections: Dict[int, List[int]] = field(default_factory=dict)  # GPU ID -> 连接的GPU列表
    
    def __post_init__(self):
        """数据验证和后处理"""
        if self.num_gpus <= 0:
            raise ValueError("GPU数量必须大于0")
        
        if len(self.gpu_info) != self.num_gpus:
            raise ValueError(f"GPU信息数量({len(self.gpu_info)})与GPU数量({self.num_gpus})不匹配")
        
        # 验证GPU ID的连续性
        expected_ids = set(range(self.num_gpus))
        actual_ids = set(self.gpu_info.keys())
        if expected_ids != actual_ids:
            raise ValueError(f"GPU ID不连续，期望{expected_ids}，实际{actual_ids}")
    
    def get_total_memory(self) -> int:
        """获取总GPU内存"""
        return sum(gpu.memory_total for gpu in self.gpu_info.values())
    
    def get_available_memory(self) -> int:
        """获取可用GPU内存"""
        return sum(gpu.memory_free for gpu in self.gpu_info.values())
    
    def get_memory_balanced_gpus(self) -> List[int]:
        """获取内存使用相对均衡的GPU列表"""
        usage_rates = [(gpu_id, gpu.memory_usage_percent) 
                      for gpu_id, gpu in self.gpu_info.items()]
        usage_rates.sort(key=lambda x: x[1])
        return [gpu_id for gpu_id, _ in usage_rates]
    
    def has_nvlink(self, gpu1: int, gpu2: int) -> bool:
        """检查两个GPU之间是否有NVLink连接"""
        return (gpu1 in self.nvlink_connections and 
                gpu2 in self.nvlink_connections[gpu1])
    
    def get_optimal_gpu_pairs(self) -> List[Tuple[int, int]]:
        """获取最优的GPU配对（基于NVLink连接和带宽）"""
        pairs = []
        used_gpus = set()
        
        # 优先选择有NVLink连接的GPU对
        for gpu1, connected_gpus in self.nvlink_connections.items():
            if gpu1 in used_gpus:
                continue
            for gpu2 in connected_gpus:
                if gpu2 not in used_gpus:
                    pairs.append((gpu1, gpu2))
                    used_gpus.update([gpu1, gpu2])
                    break
        
        # 为剩余GPU配对
        remaining_gpus = [gpu_id for gpu_id in range(self.num_gpus) 
                         if gpu_id not in used_gpus]
        for i in range(0, len(remaining_gpus) - 1, 2):
            pairs.append((remaining_gpus[i], remaining_gpus[i + 1]))
        
        return pairs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "num_gpus": self.num_gpus,
            "gpu_info": {str(k): v.to_dict() for k, v in self.gpu_info.items()},
            "interconnect_bandwidth": {f"{k[0]}-{k[1]}": v for k, v in self.interconnect_bandwidth.items()},
            "numa_topology": self.numa_topology,
            "pcie_topology": self.pcie_topology,
            "nvlink_connections": self.nvlink_connections,
            "total_memory": self.get_total_memory(),
            "available_memory": self.get_available_memory()
        }


@dataclass
class ParallelConfig:
    """并行训练配置"""
    strategy: ParallelStrategy
    data_parallel_enabled: bool = True
    model_parallel_enabled: bool = False
    pipeline_parallel_enabled: bool = False
    
    # 并行度配置
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # 通信配置
    communication_backend: CommunicationBackend = CommunicationBackend.NCCL
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # 内存优化配置
    enable_zero_optimization: bool = True
    zero_stage: ZeroStage = ZeroStage.OPTIMIZER_GRADIENT
    enable_cpu_offload: bool = False
    enable_gradient_checkpointing: bool = True
    
    # 训练配置
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 高级配置
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"  # fp16, bf16
    enable_activation_checkpointing: bool = False
    
    def __post_init__(self):
        """数据验证和配置优化"""
        if self.data_parallel_size <= 0:
            raise ValueError("数据并行大小必须大于0")
        if self.tensor_parallel_size <= 0:
            raise ValueError("张量并行大小必须大于0")
        if self.pipeline_parallel_size <= 0:
            raise ValueError("流水线并行大小必须大于0")
        
        # 验证端口范围
        if not 1024 <= self.master_port <= 65535:
            raise ValueError("主节点端口必须在1024-65535范围内")
        
        # 自动调整配置
        self._auto_adjust_config()
    
    def _auto_adjust_config(self):
        """自动调整配置"""
        total_gpus = self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
        
        # 如果只有一个GPU，禁用所有并行策略
        if total_gpus == 1:
            self.model_parallel_enabled = False
            self.pipeline_parallel_enabled = False
            self.enable_zero_optimization = False
        
        # 根据并行策略调整具体配置
        if self.strategy == ParallelStrategy.DATA_PARALLEL:
            self.model_parallel_enabled = False
            self.pipeline_parallel_enabled = False
            self.tensor_parallel_size = 1
            self.pipeline_parallel_size = 1
        elif self.strategy == ParallelStrategy.MODEL_PARALLEL:
            self.data_parallel_enabled = True  # 通常与数据并行结合
            self.model_parallel_enabled = True
            self.pipeline_parallel_enabled = False
            self.pipeline_parallel_size = 1
        elif self.strategy == ParallelStrategy.PIPELINE_PARALLEL:
            self.data_parallel_enabled = True
            self.model_parallel_enabled = False
            self.pipeline_parallel_enabled = True
            self.tensor_parallel_size = 1
    
    @property
    def world_size(self) -> int:
        """计算总的进程数"""
        return self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
    
    @property
    def is_distributed(self) -> bool:
        """检查是否为分布式训练"""
        return self.world_size > 1
    
    def get_process_group_config(self) -> Dict[str, Any]:
        """获取进程组配置"""
        return {
            "backend": self.communication_backend.value,
            "init_method": f"tcp://{self.master_addr}:{self.master_port}",
            "world_size": self.world_size,
            "timeout": 1800  # 30分钟超时
        }
    
    def validate_gpu_topology(self, topology: GPUTopology) -> bool:
        """验证配置与GPU拓扑的兼容性"""
        if self.world_size > topology.num_gpus:
            return False
        
        # 检查内存需求
        min_memory_per_gpu = 4000  # 4GB最小内存需求
        for gpu in topology.gpu_info.values():
            if gpu.memory_free < min_memory_per_gpu:
                return False
        
        return True
    
    def optimize_for_topology(self, topology: GPUTopology) -> 'ParallelConfig':
        """根据GPU拓扑优化配置"""
        optimized = ParallelConfig(
            strategy=self.strategy,
            data_parallel_enabled=self.data_parallel_enabled,
            model_parallel_enabled=self.model_parallel_enabled,
            pipeline_parallel_enabled=self.pipeline_parallel_enabled,
            communication_backend=self.communication_backend,
            master_addr=self.master_addr,
            master_port=self.master_port
        )
        
        # 根据GPU数量调整并行度
        if topology.num_gpus >= 8:
            optimized.data_parallel_size = min(8, topology.num_gpus)
            optimized.tensor_parallel_size = max(1, topology.num_gpus // 8)
        elif topology.num_gpus >= 4:
            optimized.data_parallel_size = topology.num_gpus
            optimized.tensor_parallel_size = 1
        else:
            optimized.data_parallel_size = topology.num_gpus
            optimized.tensor_parallel_size = 1
        
        # 根据内存情况调整ZeRO配置
        total_memory = topology.get_total_memory()
        if total_memory < 32000:  # 小于32GB
            optimized.zero_stage = ZeroStage.OPTIMIZER_GRADIENT_PARAMETER
            optimized.enable_cpu_offload = True
        elif total_memory < 64000:  # 小于64GB
            optimized.zero_stage = ZeroStage.OPTIMIZER_GRADIENT
        else:
            optimized.zero_stage = ZeroStage.OPTIMIZER_STATE
        
        return optimized
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "strategy": self.strategy.value,
            "data_parallel_enabled": self.data_parallel_enabled,
            "model_parallel_enabled": self.model_parallel_enabled,
            "pipeline_parallel_enabled": self.pipeline_parallel_enabled,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "world_size": self.world_size,
            "communication_backend": self.communication_backend.value,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "enable_zero_optimization": self.enable_zero_optimization,
            "zero_stage": self.zero_stage.value,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "enable_mixed_precision": self.enable_mixed_precision,
            "mixed_precision_dtype": self.mixed_precision_dtype,
            "enable_activation_checkpointing": self.enable_activation_checkpointing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParallelConfig':
        """从字典创建实例"""
        data_copy = data.copy()
        data_copy["strategy"] = ParallelStrategy(data["strategy"])
        data_copy["communication_backend"] = CommunicationBackend(data["communication_backend"])
        data_copy["zero_stage"] = ZeroStage(data["zero_stage"])
        
        # 移除计算属性
        data_copy.pop("world_size", None)
        
        return cls(**data_copy)


@dataclass
class CommunicationMetrics:
    """通信指标"""
    total_communication_time: float = 0.0  # 总通信时间(秒)
    allreduce_time: float = 0.0            # AllReduce时间(秒)
    broadcast_time: float = 0.0            # 广播时间(秒)
    p2p_time: float = 0.0                  # 点对点通信时间(秒)
    
    communication_volume: float = 0.0       # 通信数据量(MB)
    bandwidth_utilization: float = 0.0      # 带宽利用率(%)
    
    def __post_init__(self):
        """数据验证"""
        if self.total_communication_time < 0:
            raise ValueError("通信时间不能为负数")
        if not 0 <= self.bandwidth_utilization <= 100:
            raise ValueError("带宽利用率必须在0-100之间")
    
    @property
    def communication_efficiency(self) -> float:
        """计算通信效率"""
        if self.total_communication_time == 0:
            return 1.0
        
        useful_time = self.allreduce_time + self.broadcast_time
        return useful_time / self.total_communication_time if self.total_communication_time > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_communication_time": self.total_communication_time,
            "allreduce_time": self.allreduce_time,
            "broadcast_time": self.broadcast_time,
            "p2p_time": self.p2p_time,
            "communication_volume": self.communication_volume,
            "bandwidth_utilization": self.bandwidth_utilization,
            "communication_efficiency": self.communication_efficiency
        }


@dataclass
class DistributedTrainingMetrics:
    """分布式训练指标"""
    epoch: int
    global_step: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 训练指标
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    
    # GPU指标
    gpu_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # 通信指标
    communication_metrics: CommunicationMetrics = field(default_factory=CommunicationMetrics)
    
    # 性能指标
    throughput_samples_per_second: float = 0.0
    throughput_tokens_per_second: float = 0.0
    memory_efficiency: float = 0.0
    
    # 负载均衡指标
    load_balance_score: float = 0.0
    gpu_utilization_variance: float = 0.0
    
    # 收敛指标
    convergence_score: float = 0.0
    gradient_norm: float = 0.0
    
    def __post_init__(self):
        """数据验证"""
        if self.epoch < 0:
            raise ValueError("epoch必须为非负整数")
        if self.global_step < 0:
            raise ValueError("global_step必须为非负整数")
    
    def calculate_load_balance_score(self) -> float:
        """计算负载均衡评分"""
        if not self.gpu_metrics:
            return 0.0
        
        utilizations = [metrics.get("utilization", 0.0) for metrics in self.gpu_metrics.values()]
        if not utilizations:
            return 0.0
        
        mean_util = sum(utilizations) / len(utilizations)
        variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
        
        self.gpu_utilization_variance = variance
        # 负载均衡评分：方差越小，评分越高
        self.load_balance_score = max(0.0, 1.0 - variance / 100.0)
        
        return self.load_balance_score
    
    def calculate_memory_efficiency(self) -> float:
        """计算内存效率"""
        if not self.gpu_metrics:
            return 0.0
        
        memory_usages = [metrics.get("memory_usage_percent", 0.0) for metrics in self.gpu_metrics.values()]
        if not memory_usages:
            return 0.0
        
        # 内存效率：使用率适中（60-90%）时效率最高
        avg_usage = sum(memory_usages) / len(memory_usages)
        if 60 <= avg_usage <= 90:
            self.memory_efficiency = 1.0 - abs(avg_usage - 75) / 15
        elif avg_usage < 60:
            self.memory_efficiency = avg_usage / 60
        else:  # avg_usage > 90
            self.memory_efficiency = max(0.0, (100 - avg_usage) / 10)
        
        return self.memory_efficiency
    
    def update_gpu_metrics(self, gpu_id: int, metrics: Dict[str, float]):
        """更新GPU指标"""
        self.gpu_metrics[gpu_id] = metrics
        
        # 重新计算负载均衡和内存效率
        self.calculate_load_balance_score()
        self.calculate_memory_efficiency()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "load_balance_score": self.load_balance_score,
            "memory_efficiency": self.memory_efficiency,
            "communication_efficiency": self.communication_metrics.communication_efficiency,
            "convergence_score": self.convergence_score
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "timestamp": self.timestamp.isoformat(),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "gpu_metrics": self.gpu_metrics,
            "communication_metrics": self.communication_metrics.to_dict(),
            "throughput_samples_per_second": self.throughput_samples_per_second,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "memory_efficiency": self.memory_efficiency,
            "load_balance_score": self.load_balance_score,
            "gpu_utilization_variance": self.gpu_utilization_variance,
            "convergence_score": self.convergence_score,
            "gradient_norm": self.gradient_norm
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTrainingMetrics':
        """从字典创建实例"""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # 处理通信指标，移除计算属性
        comm_data = data["communication_metrics"].copy()
        comm_data.pop("communication_efficiency", None)  # 移除计算属性
        data_copy["communication_metrics"] = CommunicationMetrics(**comm_data)
        
        return cls(**data_copy)


# 配置验证和优化工具
class ParallelConfigValidator:
    """并行配置验证器"""
    
    @staticmethod
    def validate_config(config: ParallelConfig, topology: GPUTopology) -> Dict[str, Any]:
        """验证并行配置"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # 检查GPU数量
        if config.world_size > topology.num_gpus:
            result["valid"] = False
            result["errors"].append(f"配置需要{config.world_size}个GPU，但只有{topology.num_gpus}个可用")
        
        # 检查内存需求
        min_memory_per_gpu = 4000  # 4GB
        for gpu_id, gpu in topology.gpu_info.items():
            if gpu.memory_free < min_memory_per_gpu:
                result["warnings"].append(f"GPU {gpu_id}可用内存({gpu.memory_free}MB)可能不足")
        
        # 检查通信后端兼容性
        if config.communication_backend == CommunicationBackend.NCCL and topology.num_gpus == 1:
            result["warnings"].append("单GPU环境建议使用GLOO后端")
        
        # 性能优化建议
        if topology.num_gpus >= 4 and config.data_parallel_size == 1:
            result["recommendations"].append("建议启用数据并行以提高GPU利用率")
        
        if topology.get_total_memory() > 64000 and config.zero_stage == ZeroStage.OPTIMIZER_GRADIENT_PARAMETER:
            result["recommendations"].append("内存充足，可以降低ZeRO优化级别以提高性能")
        
        return result
    
    @staticmethod
    def optimize_config(config: ParallelConfig, topology: GPUTopology) -> ParallelConfig:
        """优化并行配置"""
        return config.optimize_for_topology(topology)
    
    @staticmethod
    def estimate_memory_usage(config: ParallelConfig, model_size_gb: float) -> Dict[str, float]:
        """估算内存使用"""
        # 简化的内存估算
        base_memory = model_size_gb * 1024  # 转换为MB
        
        # 根据并行策略调整
        if config.model_parallel_enabled:
            base_memory /= config.tensor_parallel_size
        
        # 根据ZeRO优化调整
        if config.enable_zero_optimization:
            if config.zero_stage == ZeroStage.OPTIMIZER_STATE:
                optimizer_memory = base_memory * 0.5 / config.data_parallel_size
            elif config.zero_stage == ZeroStage.OPTIMIZER_GRADIENT:
                optimizer_memory = base_memory * 0.75 / config.data_parallel_size
            else:  # ZeroStage.OPTIMIZER_GRADIENT_PARAMETER
                optimizer_memory = base_memory * 1.0 / config.data_parallel_size
        else:
            optimizer_memory = base_memory * 2.0  # 优化器状态通常是模型的2倍
        
        # 激活内存
        activation_memory = base_memory * 0.3
        if config.enable_gradient_checkpointing:
            activation_memory *= 0.5
        
        total_per_gpu = base_memory + optimizer_memory + activation_memory
        
        return {
            "model_memory_mb": base_memory,
            "optimizer_memory_mb": optimizer_memory,
            "activation_memory_mb": activation_memory,
            "total_per_gpu_mb": total_per_gpu,
            "total_all_gpus_mb": total_per_gpu * config.world_size
        }