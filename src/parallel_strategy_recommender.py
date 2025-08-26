#!/usr/bin/env python3
"""
并行策略自动推荐模块
基于硬件配置自动推荐最优的并行训练策略
支持数据并行、模型并行、流水线并行配置
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from gpu_utils import GPUDetector, GPUTopology, InterconnectType


class ParallelStrategy(Enum):
    """并行策略枚举"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    SINGLE_GPU = "single_gpu"


@dataclass
class SimpleParallelConfig:
    """简化的并行配置"""
    data_parallel: bool = False
    model_parallel: bool = False
    pipeline_parallel: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_zero_optimization: bool = False
    gradient_accumulation_steps: int = 1


@dataclass
class StrategyRecommendation:
    """策略推荐结果"""
    strategy: ParallelStrategy
    config: SimpleParallelConfig
    confidence: float  # 0-1之间的置信度
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    expected_performance: Dict[str, float] = field(default_factory=dict)
    
    def add_reasoning(self, reason: str):
        """添加推荐理由"""
        self.reasoning.append(reason)
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


@dataclass
class ModelRequirements:
    """模型需求配置"""
    model_name: str = "Qwen3-4B-Thinking"
    model_size_gb: float = 8.0  # 模型大小（GB）
    min_memory_per_gpu: int = 8192  # 最小GPU内存需求（MB）
    recommended_memory_per_gpu: int = 16384  # 推荐GPU内存（MB）
    max_sequence_length: int = 2048
    vocab_size: int = 151936
    hidden_size: int = 3584
    num_layers: int = 32
    num_attention_heads: int = 28
    supports_gradient_checkpointing: bool = True
    supports_mixed_precision: bool = True


class ParallelStrategyRecommender:
    """并行策略推荐器"""
    
    def __init__(self, model_requirements: Optional[ModelRequirements] = None):
        self.logger = logging.getLogger(__name__)
        self.gpu_detector = GPUDetector()
        self.model_requirements = model_requirements or ModelRequirements()
        
    def recommend_strategy(self, 
                          batch_size: int = 4,
                          sequence_length: int = 2048,
                          enable_lora: bool = True,
                          lora_rank: int = 64) -> StrategyRecommendation:
        """
        推荐最优的并行训练策略
        
        Args:
            batch_size: 批次大小
            sequence_length: 序列长度
            enable_lora: 是否启用LoRA
            lora_rank: LoRA rank值
            
        Returns:
            StrategyRecommendation: 推荐结果
        """
        # 获取GPU拓扑信息
        topology = self.gpu_detector.detect_gpu_topology()
        
        # 估算内存需求
        memory_requirements = self._estimate_memory_requirements(
            batch_size, sequence_length, enable_lora, lora_rank
        )
        
        # 分析硬件能力
        hardware_analysis = self._analyze_hardware_capabilities(topology)
        
        # 生成推荐策略
        recommendation = self._generate_recommendation(
            topology, memory_requirements, hardware_analysis,
            batch_size, sequence_length, enable_lora, lora_rank
        )
        
        return recommendation
    
    def _estimate_memory_requirements(self, 
                                    batch_size: int,
                                    sequence_length: int,
                                    enable_lora: bool,
                                    lora_rank: int) -> Dict[str, float]:
        """估算内存需求"""
        requirements = {}
        
        # 基础模型内存（参数 + 优化器状态）
        model_params_gb = self.model_requirements.model_size_gb
        optimizer_states_gb = model_params_gb * 2  # Adam优化器需要2倍参数内存
        
        if enable_lora:
            # LoRA参数内存
            lora_params_gb = (lora_rank * self.model_requirements.hidden_size * 
                            self.model_requirements.num_layers * 2) / (1024**3)
            model_params_gb += lora_params_gb
            optimizer_states_gb = lora_params_gb * 2  # 只优化LoRA参数
        
        # 激活值内存（与batch_size和sequence_length相关）
        activation_memory_gb = (batch_size * sequence_length * 
                              self.model_requirements.hidden_size * 
                              self.model_requirements.num_layers * 4) / (1024**3)
        
        # 梯度内存
        gradient_memory_gb = model_params_gb
        
        # 总内存需求
        total_memory_gb = (model_params_gb + optimizer_states_gb + 
                          activation_memory_gb + gradient_memory_gb)
        
        requirements.update({
            "model_params_gb": model_params_gb,
            "optimizer_states_gb": optimizer_states_gb,
            "activation_memory_gb": activation_memory_gb,
            "gradient_memory_gb": gradient_memory_gb,
            "total_memory_gb": total_memory_gb,
            "per_gpu_memory_mb": total_memory_gb * 1024  # 单GPU情况
        })
        
        return requirements
    
    def _analyze_hardware_capabilities(self, topology: GPUTopology) -> Dict[str, Any]:
        """分析硬件能力"""
        analysis = {
            "num_gpus": topology.num_gpus,
            "total_memory_mb": 0,
            "min_memory_mb": float('inf'),
            "max_memory_mb": 0,
            "avg_memory_mb": 0,
            "has_nvlink": False,
            "has_high_bandwidth": False,
            "numa_aware": False,
            "interconnect_types": set(),
            "bandwidth_matrix": topology.bandwidth_matrix,
            "topology_score": 0.0
        }
        
        if topology.num_gpus == 0:
            return analysis
        
        # 分析GPU内存
        memory_values = []
        for gpu_info in topology.gpu_info.values():
            memory_mb = gpu_info.total_memory
            memory_values.append(memory_mb)
            analysis["total_memory_mb"] += memory_mb
            analysis["min_memory_mb"] = min(analysis["min_memory_mb"], memory_mb)
            analysis["max_memory_mb"] = max(analysis["max_memory_mb"], memory_mb)
        
        if memory_values:
            analysis["avg_memory_mb"] = sum(memory_values) / len(memory_values)
        
        # 分析互联能力
        for interconnect in topology.interconnects:
            analysis["interconnect_types"].add(interconnect.interconnect_type)
            
            if interconnect.interconnect_type == InterconnectType.NVLINK:
                analysis["has_nvlink"] = True
            
            if interconnect.bandwidth_gbps >= 25.0:  # 高带宽阈值
                analysis["has_high_bandwidth"] = True
        
        # 分析NUMA拓扑
        if topology.numa_topology:
            analysis["numa_aware"] = True
        
        # 计算拓扑评分
        analysis["topology_score"] = self._calculate_topology_score(topology, analysis)
        
        return analysis
    
    def _calculate_topology_score(self, topology: GPUTopology, analysis: Dict[str, Any]) -> float:
        """计算拓扑评分（0-100）"""
        score = 0.0
        
        # GPU数量评分（最多40分）
        if topology.num_gpus == 1:
            score += 20
        elif topology.num_gpus == 2:
            score += 30
        elif topology.num_gpus <= 4:
            score += 35
        elif topology.num_gpus <= 8:
            score += 40
        else:
            score += 35  # 太多GPU可能带来通信开销
        
        # 内存评分（最多30分）
        if analysis["min_memory_mb"] >= self.model_requirements.recommended_memory_per_gpu:
            score += 30
        elif analysis["min_memory_mb"] >= self.model_requirements.min_memory_per_gpu:
            score += 20
        else:
            score += 10
        
        # 互联评分（最多20分）
        if analysis["has_nvlink"]:
            score += 20
        elif analysis["has_high_bandwidth"]:
            score += 15
        else:
            score += 5
        
        # NUMA评分（最多10分）
        if analysis["numa_aware"]:
            score += 10
        else:
            score += 5
        
        return min(score, 100.0)
    
    def _generate_recommendation(self,
                               topology: GPUTopology,
                               memory_requirements: Dict[str, float],
                               hardware_analysis: Dict[str, Any],
                               batch_size: int,
                               sequence_length: int,
                               enable_lora: bool,
                               lora_rank: int) -> StrategyRecommendation:
        """生成推荐策略"""
        
        num_gpus = topology.num_gpus
        total_memory_needed_mb = memory_requirements["per_gpu_memory_mb"]
        
        # 单GPU情况
        if num_gpus <= 1:
            return self._recommend_single_gpu(
                topology, memory_requirements, hardware_analysis,
                batch_size, sequence_length, enable_lora, lora_rank
            )
        
        # 多GPU情况
        return self._recommend_multi_gpu(
            topology, memory_requirements, hardware_analysis,
            batch_size, sequence_length, enable_lora, lora_rank
        )
    
    def _recommend_single_gpu(self,
                            topology: GPUTopology,
                            memory_requirements: Dict[str, float],
                            hardware_analysis: Dict[str, Any],
                            batch_size: int,
                            sequence_length: int,
                            enable_lora: bool,
                            lora_rank: int) -> StrategyRecommendation:
        """推荐单GPU策略"""
        
        config = SimpleParallelConfig(
            data_parallel=False,
            model_parallel=False,
            pipeline_parallel=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_zero_optimization=False,
            gradient_accumulation_steps=1
        )
        
        recommendation = StrategyRecommendation(
            strategy=ParallelStrategy.SINGLE_GPU,
            config=config,
            confidence=0.9
        )
        
        # 检查内存是否足够
        if len(topology.gpu_info) > 0:
            gpu_info = list(topology.gpu_info.values())[0]
            available_memory_mb = gpu_info.total_memory
            needed_memory_mb = memory_requirements["per_gpu_memory_mb"]
            
            if available_memory_mb >= needed_memory_mb:
                recommendation.add_reasoning(f"单GPU内存({available_memory_mb}MB)足够运行模型")
                recommendation.confidence = 0.95
            else:
                recommendation.add_warning(
                    f"GPU内存可能不足：需要{needed_memory_mb:.0f}MB，可用{available_memory_mb}MB"
                )
                recommendation.add_reasoning("建议启用梯度检查点和混合精度训练")
                recommendation.confidence = 0.7
                
                # 调整配置以节省内存
                config.gradient_accumulation_steps = max(2, int(needed_memory_mb / available_memory_mb))
        
        if enable_lora:
            recommendation.add_reasoning("启用LoRA可以显著减少内存使用")
        
        recommendation.expected_performance = {
            "memory_efficiency": 0.8 if enable_lora else 0.6,
            "training_speed": 1.0,
            "scalability": 0.3
        }
        
        return recommendation
    
    def _recommend_multi_gpu(self,
                           topology: GPUTopology,
                           memory_requirements: Dict[str, float],
                           hardware_analysis: Dict[str, Any],
                           batch_size: int,
                           sequence_length: int,
                           enable_lora: bool,
                           lora_rank: int) -> StrategyRecommendation:
        """推荐多GPU策略"""
        
        num_gpus = topology.num_gpus
        min_memory_mb = hardware_analysis["min_memory_mb"]
        needed_memory_mb = memory_requirements["per_gpu_memory_mb"]
        has_nvlink = hardware_analysis["has_nvlink"]
        
        # 判断是否可以使用数据并行
        can_use_data_parallel = min_memory_mb >= needed_memory_mb
        
        if can_use_data_parallel:
            # 推荐数据并行
            config = SimpleParallelConfig(
                data_parallel=True,
                model_parallel=False,
                pipeline_parallel=False,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=num_gpus,
                enable_zero_optimization=True,
                gradient_accumulation_steps=1
            )
            
            recommendation = StrategyRecommendation(
                strategy=ParallelStrategy.DATA_PARALLEL,
                config=config,
                confidence=0.9
            )
            
            recommendation.add_reasoning(f"每个GPU内存({min_memory_mb}MB)足够独立运行模型")
            recommendation.add_reasoning(f"数据并行可以有效利用{num_gpus}个GPU")
            
            if has_nvlink:
                recommendation.add_reasoning("检测到NVLink，数据并行通信效率高")
                recommendation.confidence = 0.95
            
            recommendation.expected_performance = {
                "memory_efficiency": 0.8,
                "training_speed": min(num_gpus * 0.85, 8.0),  # 考虑通信开销
                "scalability": 0.9
            }
            
        else:
            # 推荐模型并行或混合并行
            if num_gpus >= 4 and has_nvlink:
                # 推荐混合并行
                tensor_parallel_size = min(4, num_gpus)
                data_parallel_size = num_gpus // tensor_parallel_size
                
                config = SimpleParallelConfig(
                    data_parallel=data_parallel_size > 1,
                    model_parallel=True,
                    pipeline_parallel=False,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=1,
                    data_parallel_size=data_parallel_size,
                    enable_zero_optimization=True,
                    gradient_accumulation_steps=2
                )
                
                recommendation = StrategyRecommendation(
                    strategy=ParallelStrategy.HYBRID_PARALLEL,
                    config=config,
                    confidence=0.8
                )
                
                recommendation.add_reasoning("GPU内存不足以支持纯数据并行")
                recommendation.add_reasoning(f"使用{tensor_parallel_size}路张量并行分割模型")
                if data_parallel_size > 1:
                    recommendation.add_reasoning(f"结合{data_parallel_size}路数据并行")
                
            else:
                # 推荐张量并行
                config = SimpleParallelConfig(
                    data_parallel=False,
                    model_parallel=True,
                    pipeline_parallel=False,
                    tensor_parallel_size=min(num_gpus, 4),
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    enable_zero_optimization=True,
                    gradient_accumulation_steps=2
                )
                
                recommendation = StrategyRecommendation(
                    strategy=ParallelStrategy.MODEL_PARALLEL,
                    config=config,
                    confidence=0.75
                )
                
                recommendation.add_reasoning("GPU内存不足，需要模型并行分割模型")
                recommendation.add_reasoning(f"使用{config.tensor_parallel_size}路张量并行")
            
            if not has_nvlink:
                recommendation.add_warning("缺少NVLink，模型并行通信开销较大")
                recommendation.confidence *= 0.9
            
            recommendation.expected_performance = {
                "memory_efficiency": 0.9,
                "training_speed": num_gpus * 0.6,  # 模型并行通信开销较大
                "scalability": 0.7
            }
        
        return recommendation
    
    def get_optimization_suggestions(self, 
                                   recommendation: StrategyRecommendation,
                                   current_batch_size: int = 4) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        # 基于策略的建议
        if recommendation.strategy == ParallelStrategy.SINGLE_GPU:
            suggestions.extend([
                "启用梯度检查点以减少内存使用",
                "使用混合精度训练(FP16/BF16)",
                "考虑使用LoRA微调减少参数量",
                "适当减小批次大小如果遇到OOM"
            ])
        
        elif recommendation.strategy == ParallelStrategy.DATA_PARALLEL:
            suggestions.extend([
                "启用ZeRO优化器状态分片",
                "使用梯度压缩减少通信开销",
                "确保数据加载不成为瓶颈",
                "监控GPU利用率确保负载均衡"
            ])
        
        elif recommendation.strategy in [ParallelStrategy.MODEL_PARALLEL, ParallelStrategy.HYBRID_PARALLEL]:
            suggestions.extend([
                "优化模型分割策略减少通信",
                "使用流水线并行提高GPU利用率",
                "启用激活值检查点节省内存",
                "监控跨GPU通信带宽使用"
            ])
        
        # 基于警告的建议
        if recommendation.warnings:
            suggestions.append("注意解决以上警告以获得最佳性能")
        
        # 基于置信度的建议
        if recommendation.confidence < 0.8:
            suggestions.append("建议进行小规模测试验证配置效果")
        
        return suggestions
    
    def compare_strategies(self, 
                         strategies: List[StrategyRecommendation]) -> StrategyRecommendation:
        """比较多个策略并选择最佳的"""
        if not strategies:
            raise ValueError("策略列表不能为空")
        
        if len(strategies) == 1:
            return strategies[0]
        
        # 计算综合评分
        best_strategy = None
        best_score = -1
        
        for strategy in strategies:
            # 综合评分 = 置信度 * 0.4 + 训练速度 * 0.3 + 内存效率 * 0.2 + 可扩展性 * 0.1
            performance = strategy.expected_performance
            score = (strategy.confidence * 0.4 + 
                    performance.get("training_speed", 0) / 10 * 0.3 +
                    performance.get("memory_efficiency", 0) * 0.2 +
                    performance.get("scalability", 0) * 0.1)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    def generate_config_file(self, 
                           recommendation: StrategyRecommendation,
                           output_path: str = "parallel_config.yaml") -> str:
        """生成配置文件"""
        import yaml
        
        config_dict = {
            "parallel_strategy": {
                "strategy_type": recommendation.strategy.value,
                "confidence": recommendation.confidence,
                "data_parallel": recommendation.config.data_parallel,
                "model_parallel": recommendation.config.model_parallel,
                "pipeline_parallel": recommendation.config.pipeline_parallel,
                "tensor_parallel_size": recommendation.config.tensor_parallel_size,
                "pipeline_parallel_size": recommendation.config.pipeline_parallel_size,
                "data_parallel_size": recommendation.config.data_parallel_size,
                "enable_zero_optimization": recommendation.config.enable_zero_optimization,
                "gradient_accumulation_steps": recommendation.config.gradient_accumulation_steps
            },
            "reasoning": recommendation.reasoning,
            "warnings": recommendation.warnings,
            "expected_performance": recommendation.expected_performance,
            "optimization_suggestions": self.get_optimization_suggestions(recommendation)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        return output_path


def main():
    """主函数，用于测试并行策略推荐功能"""
    logging.basicConfig(level=logging.INFO)
    
    recommender = ParallelStrategyRecommender()
    
    print("=== 并行策略推荐测试 ===")
    
    # 测试不同场景
    scenarios = [
        {"batch_size": 4, "sequence_length": 2048, "enable_lora": True, "lora_rank": 64},
        {"batch_size": 8, "sequence_length": 2048, "enable_lora": False, "lora_rank": 0},
        {"batch_size": 2, "sequence_length": 4096, "enable_lora": True, "lora_rank": 128},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n场景 {i}: {scenario}")
        recommendation = recommender.recommend_strategy(**scenario)
        
        print(f"推荐策略: {recommendation.strategy.value}")
        print(f"置信度: {recommendation.confidence:.2f}")
        print(f"配置: {recommendation.config}")
        
        if recommendation.reasoning:
            print("推荐理由:")
            for reason in recommendation.reasoning:
                print(f"  - {reason}")
        
        if recommendation.warnings:
            print("警告:")
            for warning in recommendation.warnings:
                print(f"  ⚠️ {warning}")
        
        print(f"预期性能: {recommendation.expected_performance}")
        
        # 获取优化建议
        suggestions = recommender.get_optimization_suggestions(recommendation)
        if suggestions:
            print("优化建议:")
            for suggestion in suggestions:
                print(f"  💡 {suggestion}")


if __name__ == "__main__":
    main()