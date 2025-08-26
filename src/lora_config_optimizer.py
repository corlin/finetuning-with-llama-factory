"""
LoRA参数动态配置模块

本模块实现基于GPU内存的LoRA参数动态计算和多GPU环境下的LoRA配置优化。
支持根据硬件资源自动调整LoRA参数，确保内存效率和训练性能的平衡。
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

try:
    from parallel_config import GPUTopology, GPUInfo, ParallelConfig
    from data_models import ChineseMetrics
except ImportError:
    from parallel_config import GPUTopology, GPUInfo, ParallelConfig
    from data_models import ChineseMetrics


class LoRAOptimizationStrategy(Enum):
    """LoRA优化策略"""
    MEMORY_EFFICIENT = "memory_efficient"  # 内存优先
    PERFORMANCE_BALANCED = "performance_balanced"  # 性能平衡
    QUALITY_FOCUSED = "quality_focused"  # 质量优先
    AUTO = "auto"  # 自动选择


class LoRATargetModule(Enum):
    """LoRA目标模块类型"""
    ATTENTION_ONLY = "attention_only"  # 仅注意力层
    ATTENTION_MLP = "attention_mlp"    # 注意力层+MLP
    ALL_LINEAR = "all_linear"          # 所有线性层
    CUSTOM = "custom"                  # 自定义


@dataclass
class LoRAMemoryProfile:
    """LoRA内存使用配置文件"""
    rank: int
    alpha: int
    target_modules: List[str]
    
    # 内存估算
    lora_params_mb: float = 0.0
    optimizer_memory_mb: float = 0.0
    gradient_memory_mb: float = 0.0
    total_memory_mb: float = 0.0
    
    # 性能指标
    training_efficiency: float = 0.0
    convergence_score: float = 0.0
    memory_efficiency: float = 0.0
    
    def __post_init__(self):
        """数据验证"""
        if self.rank <= 0:
            raise ValueError("LoRA rank必须大于0")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha必须大于0")
        if not self.target_modules:
            raise ValueError("目标模块列表不能为空")
    
    @property
    def memory_overhead_ratio(self) -> float:
        """计算内存开销比例"""
        if self.total_memory_mb == 0:
            return 0.0
        return self.lora_params_mb / self.total_memory_mb
    
    @property
    def parameter_efficiency(self) -> float:
        """计算参数效率（rank/alpha比值）"""
        return self.rank / self.alpha if self.alpha > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "lora_params_mb": self.lora_params_mb,
            "optimizer_memory_mb": self.optimizer_memory_mb,
            "gradient_memory_mb": self.gradient_memory_mb,
            "total_memory_mb": self.total_memory_mb,
            "training_efficiency": self.training_efficiency,
            "convergence_score": self.convergence_score,
            "memory_efficiency": self.memory_efficiency,
            "memory_overhead_ratio": self.memory_overhead_ratio,
            "parameter_efficiency": self.parameter_efficiency
        }


@dataclass
class MultiGPULoRAConfig:
    """多GPU LoRA配置"""
    global_config: LoRAMemoryProfile
    per_gpu_configs: Dict[int, LoRAMemoryProfile] = field(default_factory=dict)
    
    # 分布式配置
    enable_gradient_synchronization: bool = True
    sync_frequency: int = 1  # 梯度同步频率
    enable_parameter_sharding: bool = False
    
    # 负载均衡
    enable_dynamic_load_balancing: bool = True
    memory_balance_threshold: float = 0.1  # 10%内存差异阈值
    
    # 通信优化
    enable_gradient_compression: bool = False
    compression_ratio: float = 0.5
    
    def __post_init__(self):
        """数据验证和初始化"""
        if not self.per_gpu_configs:
            # 如果没有指定per-GPU配置，使用全局配置
            self.per_gpu_configs = {0: self.global_config}
    
    @property
    def total_lora_parameters(self) -> int:
        """计算总LoRA参数数量"""
        total = 0
        for config in self.per_gpu_configs.values():
            # 简化计算：rank * 2 * hidden_size * num_target_modules
            total += config.rank * 2 * 4096 * len(config.target_modules)
        return total
    
    @property
    def average_memory_usage(self) -> float:
        """计算平均内存使用"""
        if not self.per_gpu_configs:
            return 0.0
        
        total_memory = sum(config.total_memory_mb for config in self.per_gpu_configs.values())
        return total_memory / len(self.per_gpu_configs)
    
    def get_memory_balance_score(self) -> float:
        """计算内存平衡评分"""
        if len(self.per_gpu_configs) <= 1:
            return 1.0
        
        memory_usages = [config.total_memory_mb for config in self.per_gpu_configs.values()]
        avg_memory = sum(memory_usages) / len(memory_usages)
        
        if avg_memory == 0:
            return 1.0
        
        variance = sum((usage - avg_memory) ** 2 for usage in memory_usages) / len(memory_usages)
        coefficient_of_variation = math.sqrt(variance) / avg_memory
        
        # 变异系数越小，平衡性越好
        return max(0.0, 1.0 - coefficient_of_variation)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "global_config": self.global_config.to_dict(),
            "per_gpu_configs": {str(k): v.to_dict() for k, v in self.per_gpu_configs.items()},
            "enable_gradient_synchronization": self.enable_gradient_synchronization,
            "sync_frequency": self.sync_frequency,
            "enable_parameter_sharding": self.enable_parameter_sharding,
            "enable_dynamic_load_balancing": self.enable_dynamic_load_balancing,
            "memory_balance_threshold": self.memory_balance_threshold,
            "enable_gradient_compression": self.enable_gradient_compression,
            "compression_ratio": self.compression_ratio,
            "total_lora_parameters": self.total_lora_parameters,
            "average_memory_usage": self.average_memory_usage,
            "memory_balance_score": self.get_memory_balance_score()
        }


class LoRAConfigOptimizer:
    """LoRA配置优化器"""
    
    def __init__(self, 
                 model_size_gb: float = 4.0,  # Qwen3-4B模型大小
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_attention_heads: int = 32):
        """
        初始化LoRA配置优化器
        
        Args:
            model_size_gb: 模型大小(GB)
            hidden_size: 隐藏层大小
            num_layers: 层数
            num_attention_heads: 注意力头数
        """
        self.model_size_gb = model_size_gb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.logger = logging.getLogger(__name__)
        
        # 预定义的目标模块配置
        self.target_module_configs = {
            LoRATargetModule.ATTENTION_ONLY: [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ],
            LoRATargetModule.ATTENTION_MLP: [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            LoRATargetModule.ALL_LINEAR: [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"
            ]
        }
    
    def calculate_lora_memory_usage(self, 
                                  rank: int, 
                                  alpha: int,
                                  target_modules: List[str],
                                  batch_size: int = 4,
                                  sequence_length: int = 2048) -> LoRAMemoryProfile:
        """
        计算LoRA内存使用情况
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha
            target_modules: 目标模块列表
            batch_size: 批次大小
            sequence_length: 序列长度
            
        Returns:
            LoRAMemoryProfile: 内存使用配置文件
        """
        # 计算LoRA参数数量
        lora_params = self._calculate_lora_parameters(rank, target_modules)
        lora_params_mb = lora_params * 4 / (1024 ** 2)  # FP32字节转MB
        
        # 优化器内存（Adam需要2倍参数内存）
        optimizer_memory_mb = lora_params_mb * 2
        
        # 梯度内存
        gradient_memory_mb = lora_params_mb
        
        # 激活值内存（简化估算）
        activation_memory_mb = (batch_size * sequence_length * self.hidden_size * 
                              len(target_modules) * 4) / (1024 ** 2)
        
        # 总内存
        total_memory_mb = (lora_params_mb + optimizer_memory_mb + 
                          gradient_memory_mb + activation_memory_mb)
        
        profile = LoRAMemoryProfile(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            lora_params_mb=lora_params_mb,
            optimizer_memory_mb=optimizer_memory_mb,
            gradient_memory_mb=gradient_memory_mb,
            total_memory_mb=total_memory_mb
        )
        
        # 计算效率指标
        profile.memory_efficiency = self._calculate_memory_efficiency(profile)
        profile.training_efficiency = self._calculate_training_efficiency(profile)
        profile.convergence_score = self._estimate_convergence_score(rank, alpha)
        
        return profile
    
    def _calculate_lora_parameters(self, rank: int, target_modules: List[str]) -> int:
        """计算LoRA参数数量"""
        total_params = 0
        
        for module in target_modules:
            if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                # 注意力层：hidden_size x hidden_size
                total_params += 2 * rank * self.hidden_size
            elif module in ["gate_proj", "up_proj"]:
                # MLP上投影：hidden_size x (4 * hidden_size)
                total_params += 2 * rank * self.hidden_size * 4
            elif module == "down_proj":
                # MLP下投影：(4 * hidden_size) x hidden_size
                total_params += 2 * rank * self.hidden_size * 4
            elif module == "embed_tokens":
                # 词嵌入层
                vocab_size = 151936  # Qwen3词汇表大小
                total_params += 2 * rank * vocab_size
            elif module == "lm_head":
                # 语言模型头
                vocab_size = 151936
                total_params += 2 * rank * vocab_size
        
        return total_params * self.num_layers
    
    def _calculate_memory_efficiency(self, profile: LoRAMemoryProfile) -> float:
        """计算内存效率"""
        # 基于参数数量与内存使用的比值
        if profile.total_memory_mb == 0:
            return 0.0
        
        # 理想情况下，内存使用应该主要由LoRA参数决定
        ideal_ratio = 0.4  # 40%用于LoRA参数是比较理想的
        actual_ratio = profile.lora_params_mb / profile.total_memory_mb
        
        # 越接近理想比例，效率越高
        efficiency = 1.0 - abs(actual_ratio - ideal_ratio) / ideal_ratio
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_training_efficiency(self, profile: LoRAMemoryProfile) -> float:
        """计算训练效率"""
        # 基于rank和alpha的比值以及参数数量
        param_efficiency = profile.parameter_efficiency
        
        # rank在8-64之间通常效果较好
        rank_score = 1.0
        if profile.rank < 8:
            rank_score = profile.rank / 8
        elif profile.rank > 64:
            rank_score = 64 / profile.rank
        
        # alpha通常是rank的2倍效果较好
        alpha_score = 1.0
        ideal_alpha = profile.rank * 2
        if profile.alpha != ideal_alpha:
            alpha_score = min(profile.alpha, ideal_alpha) / max(profile.alpha, ideal_alpha)
        
        return (param_efficiency * 0.3 + rank_score * 0.4 + alpha_score * 0.3)
    
    def _estimate_convergence_score(self, rank: int, alpha: int) -> float:
        """估算收敛性评分"""
        # 基于经验的收敛性估算
        # rank越高，表达能力越强，但可能过拟合
        # alpha控制学习率缩放
        
        rank_score = 1.0
        if rank < 16:
            rank_score = rank / 16  # rank太小可能表达能力不足
        elif rank > 128:
            rank_score = 128 / rank  # rank太大可能过拟合
        
        alpha_ratio = alpha / rank if rank > 0 else 1.0
        alpha_score = 1.0
        if alpha_ratio < 1.0:
            alpha_score = alpha_ratio  # alpha太小可能学习不充分
        elif alpha_ratio > 4.0:
            alpha_score = 4.0 / alpha_ratio  # alpha太大可能不稳定
        
        return (rank_score + alpha_score) / 2
    
    def optimize_for_single_gpu(self, 
                               available_memory_mb: int,
                               strategy: LoRAOptimizationStrategy = LoRAOptimizationStrategy.AUTO,
                               target_module_type: LoRATargetModule = LoRATargetModule.ATTENTION_MLP,
                               batch_size: int = 4,
                               sequence_length: int = 2048) -> LoRAMemoryProfile:
        """
        为单GPU优化LoRA配置
        
        Args:
            available_memory_mb: 可用内存(MB)
            strategy: 优化策略
            target_module_type: 目标模块类型
            batch_size: 批次大小
            sequence_length: 序列长度
            
        Returns:
            LoRAMemoryProfile: 优化后的配置
        """
        self.logger.info(f"为单GPU优化LoRA配置，可用内存: {available_memory_mb}MB")
        
        # 获取目标模块
        target_modules = self.target_module_configs.get(
            target_module_type, 
            self.target_module_configs[LoRATargetModule.ATTENTION_MLP]
        )
        
        # 根据策略确定搜索范围
        if strategy == LoRAOptimizationStrategy.MEMORY_EFFICIENT:
            rank_range = range(4, 33, 4)  # 4, 8, 12, ..., 32
            alpha_multipliers = [1, 2]
        elif strategy == LoRAOptimizationStrategy.QUALITY_FOCUSED:
            rank_range = range(32, 129, 8)  # 32, 40, 48, ..., 128
            alpha_multipliers = [1, 2, 4]
        elif strategy == LoRAOptimizationStrategy.PERFORMANCE_BALANCED:
            rank_range = range(8, 65, 8)  # 8, 16, 24, ..., 64
            alpha_multipliers = [1, 2, 3]
        else:  # AUTO
            rank_range = range(4, 65, 4)  # 4, 8, 12, ..., 64
            alpha_multipliers = [1, 2, 4]
        
        best_config = None
        best_score = -1.0
        
        # 搜索最优配置
        for rank in rank_range:
            for alpha_mult in alpha_multipliers:
                alpha = rank * alpha_mult
                
                # 计算内存使用
                profile = self.calculate_lora_memory_usage(
                    rank, alpha, target_modules, batch_size, sequence_length
                )
                
                # 检查内存约束
                if profile.total_memory_mb > available_memory_mb * 0.9:  # 留10%余量
                    continue
                
                # 计算综合评分
                score = self._calculate_optimization_score(profile, strategy)
                
                if score > best_score:
                    best_score = score
                    best_config = profile
        
        if best_config is None:
            # 如果没有找到合适配置，使用最小配置
            self.logger.warning("未找到合适的LoRA配置，使用最小配置")
            best_config = self.calculate_lora_memory_usage(
                4, 8, target_modules[:2], batch_size, sequence_length  # 最小配置
            )
        
        self.logger.info(f"选择LoRA配置: rank={best_config.rank}, alpha={best_config.alpha}, "
                        f"内存使用={best_config.total_memory_mb:.1f}MB")
        
        return best_config
    
    def optimize_for_multi_gpu(self, 
                              topology: GPUTopology,
                              parallel_config: ParallelConfig,
                              strategy: LoRAOptimizationStrategy = LoRAOptimizationStrategy.AUTO,
                              target_module_type: LoRATargetModule = LoRATargetModule.ATTENTION_MLP,
                              batch_size: int = 4,
                              sequence_length: int = 2048) -> MultiGPULoRAConfig:
        """
        为多GPU环境优化LoRA配置
        
        Args:
            topology: GPU拓扑结构
            parallel_config: 并行配置
            strategy: 优化策略
            target_module_type: 目标模块类型
            batch_size: 批次大小
            sequence_length: 序列长度
            
        Returns:
            MultiGPULoRAConfig: 多GPU LoRA配置
        """
        self.logger.info(f"为{topology.num_gpus}个GPU优化LoRA配置")
        
        # 分析GPU内存分布
        gpu_memories = [gpu.memory_free for gpu in topology.gpu_info.values()]
        min_memory = min(gpu_memories)
        max_memory = max(gpu_memories)
        avg_memory = sum(gpu_memories) / len(gpu_memories)
        
        self.logger.info(f"GPU内存分布: 最小={min_memory}MB, 最大={max_memory}MB, 平均={avg_memory:.1f}MB")
        
        # 根据并行策略调整配置
        per_gpu_batch_size = batch_size // parallel_config.data_parallel_size
        
        # 为每个GPU生成配置
        per_gpu_configs = {}
        
        if max_memory - min_memory > min_memory * 0.2:  # 内存差异超过20%
            # 内存不均衡，为每个GPU单独优化
            self.logger.info("检测到GPU内存不均衡，为每个GPU单独优化配置")
            
            for gpu_id, gpu_info in topology.gpu_info.items():
                config = self.optimize_for_single_gpu(
                    gpu_info.memory_free,
                    strategy,
                    target_module_type,
                    per_gpu_batch_size,
                    sequence_length
                )
                per_gpu_configs[gpu_id] = config
        else:
            # 内存相对均衡，使用统一配置
            self.logger.info("GPU内存相对均衡，使用统一配置")
            
            base_config = self.optimize_for_single_gpu(
                min_memory,  # 使用最小内存作为约束
                strategy,
                target_module_type,
                per_gpu_batch_size,
                sequence_length
            )
            
            for gpu_id in topology.gpu_info.keys():
                per_gpu_configs[gpu_id] = base_config
        
        # 创建全局配置（使用第一个GPU的配置作为基准）
        global_config = per_gpu_configs[0]
        
        # 创建多GPU配置
        multi_gpu_config = MultiGPULoRAConfig(
            global_config=global_config,
            per_gpu_configs=per_gpu_configs
        )
        
        # 根据拓扑优化通信设置
        self._optimize_communication_settings(multi_gpu_config, topology, parallel_config)
        
        self.logger.info(f"多GPU LoRA配置完成，平均内存使用: {multi_gpu_config.average_memory_usage:.1f}MB")
        
        return multi_gpu_config
    
    def _calculate_optimization_score(self, 
                                    profile: LoRAMemoryProfile, 
                                    strategy: LoRAOptimizationStrategy) -> float:
        """计算优化评分"""
        if strategy == LoRAOptimizationStrategy.MEMORY_EFFICIENT:
            # 内存效率优先
            return (profile.memory_efficiency * 0.6 + 
                   profile.training_efficiency * 0.3 + 
                   profile.convergence_score * 0.1)
        elif strategy == LoRAOptimizationStrategy.QUALITY_FOCUSED:
            # 质量优先
            return (profile.convergence_score * 0.6 + 
                   profile.training_efficiency * 0.3 + 
                   profile.memory_efficiency * 0.1)
        elif strategy == LoRAOptimizationStrategy.PERFORMANCE_BALANCED:
            # 平衡策略
            return (profile.training_efficiency * 0.4 + 
                   profile.memory_efficiency * 0.3 + 
                   profile.convergence_score * 0.3)
        else:  # AUTO
            # 自动策略：综合考虑
            return (profile.training_efficiency * 0.35 + 
                   profile.memory_efficiency * 0.35 + 
                   profile.convergence_score * 0.3)
    
    def _optimize_communication_settings(self, 
                                       config: MultiGPULoRAConfig,
                                       topology: GPUTopology,
                                       parallel_config: ParallelConfig):
        """优化通信设置"""
        # 根据GPU数量和拓扑调整通信设置
        if topology.num_gpus <= 2:
            # 少量GPU，使用简单设置
            config.enable_gradient_compression = False
            config.sync_frequency = 1
        elif topology.num_gpus <= 8:
            # 中等数量GPU
            config.enable_gradient_compression = True
            config.compression_ratio = 0.5
            config.sync_frequency = 1
        else:
            # 大量GPU，需要更多优化
            config.enable_gradient_compression = True
            config.compression_ratio = 0.3
            config.sync_frequency = 2  # 降低同步频率
        
        # 根据内存使用情况调整参数分片
        avg_memory_usage = config.average_memory_usage
        if avg_memory_usage > 8000:  # 8GB以上启用参数分片
            config.enable_parameter_sharding = True
        
        # 根据内存平衡情况调整负载均衡
        balance_score = config.get_memory_balance_score()
        if balance_score < 0.8:  # 平衡性较差时启用动态负载均衡
            config.enable_dynamic_load_balancing = True
            config.memory_balance_threshold = 0.05  # 5%阈值
    
    def validate_lora_config(self, 
                           config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                           topology: GPUTopology) -> Dict[str, Any]:
        """
        验证LoRA配置
        
        Args:
            config: LoRA配置
            topology: GPU拓扑
            
        Returns:
            Dict: 验证结果
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        if isinstance(config, LoRAMemoryProfile):
            # 单GPU配置验证
            self._validate_single_gpu_config(config, topology, result)
        else:
            # 多GPU配置验证
            self._validate_multi_gpu_config(config, topology, result)
        
        return result
    
    def _validate_single_gpu_config(self, 
                                  config: LoRAMemoryProfile,
                                  topology: GPUTopology,
                                  result: Dict[str, Any]):
        """验证单GPU配置"""
        if topology.num_gpus == 0:
            result["valid"] = False
            result["errors"].append("没有可用的GPU")
            return
        
        gpu_info = list(topology.gpu_info.values())[0]
        
        # 检查内存约束
        if config.total_memory_mb > gpu_info.memory_free * 0.95:
            result["valid"] = False
            result["errors"].append(f"内存需求({config.total_memory_mb:.1f}MB)超过可用内存({gpu_info.memory_free}MB)")
        
        # 检查参数合理性
        if config.rank < 4:
            result["warnings"].append("LoRA rank较小，可能影响模型表达能力")
        elif config.rank > 128:
            result["warnings"].append("LoRA rank较大，可能导致过拟合")
        
        if config.alpha < config.rank:
            result["warnings"].append("LoRA alpha小于rank，可能影响学习效果")
        
        # 性能建议
        if config.memory_efficiency < 0.6:
            result["recommendations"].append("考虑调整rank和alpha以提高内存效率")
        
        if config.training_efficiency < 0.7:
            result["recommendations"].append("考虑调整目标模块以提高训练效率")
    
    def _validate_multi_gpu_config(self, 
                                 config: MultiGPULoRAConfig,
                                 topology: GPUTopology,
                                 result: Dict[str, Any]):
        """验证多GPU配置"""
        # 检查GPU数量匹配
        if len(config.per_gpu_configs) != topology.num_gpus:
            result["valid"] = False
            result["errors"].append(f"配置GPU数量({len(config.per_gpu_configs)})与实际GPU数量({topology.num_gpus})不匹配")
        
        # 检查每个GPU的内存约束
        for gpu_id, gpu_config in config.per_gpu_configs.items():
            if gpu_id not in topology.gpu_info:
                result["errors"].append(f"GPU {gpu_id}不存在")
                continue
            
            gpu_info = topology.gpu_info[gpu_id]
            if gpu_config.total_memory_mb > gpu_info.memory_free * 0.95:
                result["errors"].append(f"GPU {gpu_id}内存需求超限")
        
        # 检查负载均衡
        balance_score = config.get_memory_balance_score()
        if balance_score < 0.7:
            result["warnings"].append(f"GPU负载不均衡(评分: {balance_score:.2f})")
        
        # 通信优化建议
        if topology.num_gpus > 4 and not config.enable_gradient_compression:
            result["recommendations"].append("建议启用梯度压缩以减少通信开销")
        
        if config.average_memory_usage > 10000 and not config.enable_parameter_sharding:
            result["recommendations"].append("建议启用参数分片以减少内存使用")
    
    def generate_config_report(self, 
                             config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                             topology: GPUTopology) -> Dict[str, Any]:
        """
        生成配置报告
        
        Args:
            config: LoRA配置
            topology: GPU拓扑
            
        Returns:
            Dict: 配置报告
        """
        report = {
            "timestamp": "2024-01-01T00:00:00",  # 实际应用中使用datetime.now()
            "config_type": "single_gpu" if isinstance(config, LoRAMemoryProfile) else "multi_gpu",
            "gpu_topology": topology.to_dict(),
            "config_details": config.to_dict(),
            "validation_result": self.validate_lora_config(config, topology),
            "performance_estimates": {},
            "recommendations": []
        }
        
        # 性能估算
        if isinstance(config, LoRAMemoryProfile):
            report["performance_estimates"] = {
                "memory_efficiency": config.memory_efficiency,
                "training_efficiency": config.training_efficiency,
                "convergence_score": config.convergence_score,
                "parameter_count": self._calculate_lora_parameters(config.rank, config.target_modules)
            }
        else:
            report["performance_estimates"] = {
                "average_memory_efficiency": sum(c.memory_efficiency for c in config.per_gpu_configs.values()) / len(config.per_gpu_configs),
                "memory_balance_score": config.get_memory_balance_score(),
                "total_parameters": config.total_lora_parameters,
                "communication_overhead_estimate": self._estimate_communication_overhead(config, topology)
            }
        
        return report
    
    def _estimate_communication_overhead(self, 
                                       config: MultiGPULoRAConfig,
                                       topology: GPUTopology) -> float:
        """估算通信开销"""
        # 简化的通信开销估算
        base_overhead = 0.1  # 10%基础开销
        
        # 根据GPU数量调整
        gpu_factor = min(topology.num_gpus / 8, 2.0)  # 最多2倍开销
        
        # 根据参数数量调整
        param_factor = min(config.total_lora_parameters / 1000000, 1.5)  # 最多1.5倍开销
        
        # 根据压缩设置调整
        compression_factor = config.compression_ratio if config.enable_gradient_compression else 1.0
        
        overhead = base_overhead * gpu_factor * param_factor * compression_factor
        return min(overhead, 0.5)  # 最多50%开销