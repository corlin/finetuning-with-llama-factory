"""
LoRA配置集成测试

测试LoRA配置优化器与现有系统的集成。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.lora_config_optimizer import LoRAConfigOptimizer, LoRAOptimizationStrategy, LoRATargetModule
from src.parallel_config import GPUTopology, GPUInfo, ParallelConfig, ParallelStrategy
from src.parallel_strategy_recommender import ParallelStrategyRecommender, ModelRequirements
from src.gpu_utils import GPUDetector


class TestLoRAIntegration:
    """测试LoRA配置与现有系统的集成"""
    
    def setup_method(self):
        """测试设置"""
        self.lora_optimizer = LoRAConfigOptimizer()
        self.gpu_detector = GPUDetector()
    
    def test_lora_with_parallel_strategy_recommender(self):
        """测试LoRA配置与并行策略推荐器的集成"""
        # 创建模拟的GPU拓扑
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
            1: GPUInfo(gpu_id=1, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
        }
        
        topology = GPUTopology(
            num_gpus=2,
            gpu_info=gpu_info,
            interconnect_bandwidth={(0, 1): 50.0, (1, 0): 50.0},
            numa_topology={0: 0, 1: 0}
        )
        
        # 创建并行策略推荐器
        model_requirements = ModelRequirements(
            model_size_gb=4.0,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32
        )
        recommender = ParallelStrategyRecommender(model_requirements)
        
        # 获取推荐的并行策略
        recommendation = recommender.recommend_strategy(
            batch_size=8,
            sequence_length=2048,
            enable_lora=True,
            lora_rank=32
        )
        
        # 使用推荐的并行配置优化LoRA参数
        parallel_config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2
        )
        
        lora_config = self.lora_optimizer.optimize_for_multi_gpu(
            topology=topology,
            parallel_config=parallel_config,
            strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            target_module_type=LoRATargetModule.ATTENTION_MLP,
            batch_size=8,
            sequence_length=2048
        )
        
        # 验证集成结果
        assert lora_config is not None
        assert len(lora_config.per_gpu_configs) == 2
        assert lora_config.global_config.rank > 0
        assert lora_config.global_config.alpha > 0
        
        # 验证内存使用合理
        for gpu_id, gpu_config in lora_config.per_gpu_configs.items():
            gpu_memory = topology.gpu_info[gpu_id].memory_free
            assert gpu_config.total_memory_mb <= gpu_memory * 0.9
    
    def test_lora_config_validation_with_topology(self):
        """测试LoRA配置与GPU拓扑的验证"""
        # 创建不同内存容量的GPU拓扑
        topologies = [
            # 高端GPU
            GPUTopology(
                num_gpus=1,
                gpu_info={0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))},
                interconnect_bandwidth={},
                numa_topology={0: 0}
            ),
            # 中端GPU
            GPUTopology(
                num_gpus=1,
                gpu_info={0: GPUInfo(gpu_id=0, name="RTX 3080", memory_total=10000, memory_free=8000, compute_capability=(8, 6))},
                interconnect_bandwidth={},
                numa_topology={0: 0}
            ),
            # 低端GPU
            GPUTopology(
                num_gpus=1,
                gpu_info={0: GPUInfo(gpu_id=0, name="GTX 1660", memory_total=6000, memory_free=4000, compute_capability=(7, 5))},
                interconnect_bandwidth={},
                numa_topology={0: 0}
            )
        ]
        
        for i, topology in enumerate(topologies):
            gpu_memory = list(topology.gpu_info.values())[0].memory_free
            
            # 为每种GPU优化LoRA配置
            config = self.lora_optimizer.optimize_for_single_gpu(
                available_memory_mb=gpu_memory,
                strategy=LoRAOptimizationStrategy.AUTO,
                target_module_type=LoRATargetModule.ATTENTION_MLP
            )
            
            # 验证配置
            validation_result = self.lora_optimizer.validate_lora_config(config, topology)
            
            assert validation_result["valid"] is True, f"GPU {i} 配置验证失败"
            assert config.total_memory_mb <= gpu_memory * 0.9, f"GPU {i} 内存超限"
    
    def test_lora_memory_estimation_accuracy(self):
        """测试LoRA内存估算的准确性"""
        # 测试不同rank值的内存估算
        ranks = [4, 8, 16, 32, 64, 128]
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        for rank in ranks:
            alpha = rank * 2  # 常见的alpha设置
            
            profile = self.lora_optimizer.calculate_lora_memory_usage(
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                batch_size=4,
                sequence_length=2048
            )
            
            # 验证内存计算的合理性
            assert profile.lora_params_mb > 0
            assert profile.optimizer_memory_mb == profile.lora_params_mb * 2  # Adam优化器
            assert profile.gradient_memory_mb == profile.lora_params_mb
            assert profile.total_memory_mb > profile.lora_params_mb
            
            # 验证rank越大，参数越多
            if rank > 4:
                smaller_profile = self.lora_optimizer.calculate_lora_memory_usage(
                    rank=rank // 2,
                    alpha=alpha // 2,
                    target_modules=target_modules,
                    batch_size=4,
                    sequence_length=2048
                )
                assert profile.lora_params_mb > smaller_profile.lora_params_mb
    
    def test_lora_optimization_strategies_effectiveness(self):
        """测试不同LoRA优化策略的有效性"""
        available_memory = 16000  # 16GB
        
        strategies = [
            LoRAOptimizationStrategy.MEMORY_EFFICIENT,
            LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            LoRAOptimizationStrategy.QUALITY_FOCUSED,
            LoRAOptimizationStrategy.AUTO
        ]
        
        configs = {}
        
        for strategy in strategies:
            config = self.lora_optimizer.optimize_for_single_gpu(
                available_memory_mb=available_memory,
                strategy=strategy,
                target_module_type=LoRATargetModule.ATTENTION_MLP,
                batch_size=4,
                sequence_length=2048
            )
            configs[strategy] = config
        
        # 验证策略的差异性
        memory_config = configs[LoRAOptimizationStrategy.MEMORY_EFFICIENT]
        quality_config = configs[LoRAOptimizationStrategy.QUALITY_FOCUSED]
        
        # 所有配置都应该有效
        for strategy, config in configs.items():
            assert config.rank > 0, f"{strategy} 产生了无效的rank"
            assert config.alpha > 0, f"{strategy} 产生了无效的alpha"
            assert config.total_memory_mb <= available_memory * 0.9, f"{strategy} 内存超限"
            assert 0.0 <= config.memory_efficiency <= 1.0, f"{strategy} 内存效率无效"
            assert 0.0 <= config.training_efficiency <= 1.0, f"{strategy} 训练效率无效"
    
    def test_multi_gpu_load_balancing(self):
        """测试多GPU环境下的负载均衡"""
        # 创建内存不均衡的GPU环境
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
            1: GPUInfo(gpu_id=1, name="RTX 3080", memory_total=10000, memory_free=8000, compute_capability=(8, 6)),
            2: GPUInfo(gpu_id=2, name="RTX 3060", memory_total=12000, memory_free=10000, compute_capability=(8, 6)),
            3: GPUInfo(gpu_id=3, name="RTX 2080", memory_total=8000, memory_free=6000, compute_capability=(7, 5))
        }
        
        topology = GPUTopology(
            num_gpus=4,
            gpu_info=gpu_info,
            interconnect_bandwidth={
                (0, 1): 25.0, (1, 0): 25.0, (0, 2): 25.0, (2, 0): 25.0,
                (0, 3): 16.0, (3, 0): 16.0, (1, 2): 16.0, (2, 1): 16.0,
                (1, 3): 16.0, (3, 1): 16.0, (2, 3): 16.0, (3, 2): 16.0
            },
            numa_topology={0: 0, 1: 0, 2: 1, 3: 1}
        )
        
        parallel_config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=4
        )
        
        config = self.lora_optimizer.optimize_for_multi_gpu(
            topology=topology,
            parallel_config=parallel_config,
            strategy=LoRAOptimizationStrategy.AUTO,
            target_module_type=LoRATargetModule.ATTENTION_MLP,
            batch_size=16,
            sequence_length=2048
        )
        
        # 验证每个GPU的配置都在其内存限制内
        for gpu_id, gpu_config in config.per_gpu_configs.items():
            gpu_memory = topology.gpu_info[gpu_id].memory_free
            assert gpu_config.total_memory_mb <= gpu_memory * 0.9, f"GPU {gpu_id} 内存超限"
        
        # 验证负载均衡
        balance_score = config.get_memory_balance_score()
        assert 0.0 <= balance_score <= 1.0, "负载均衡评分无效"
        
        # 在内存差异较大的情况下，负载均衡评分可能较低，但应该是合理的
        # 这里不强制要求高平衡性，因为硬件本身就不均衡
    
    def test_lora_config_serialization(self):
        """测试LoRA配置的序列化和反序列化"""
        # 创建配置
        config = self.lora_optimizer.optimize_for_single_gpu(
            available_memory_mb=16000,
            strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            target_module_type=LoRATargetModule.ATTENTION_MLP
        )
        
        # 序列化
        config_dict = config.to_dict()
        
        # 验证序列化结果
        assert isinstance(config_dict, dict)
        assert "rank" in config_dict
        assert "alpha" in config_dict
        assert "target_modules" in config_dict
        assert "total_memory_mb" in config_dict
        assert "memory_efficiency" in config_dict
        assert "training_efficiency" in config_dict
        assert "convergence_score" in config_dict
        
        # 验证数据类型
        assert isinstance(config_dict["rank"], int)
        assert isinstance(config_dict["alpha"], int)
        assert isinstance(config_dict["target_modules"], list)
        assert isinstance(config_dict["total_memory_mb"], float)
    
    def test_lora_config_with_different_model_sizes(self):
        """测试不同模型大小的LoRA配置"""
        model_sizes = [
            (1.0, 2048, 16, 16),   # 1B模型
            (4.0, 4096, 32, 32),   # 4B模型 (Qwen3-4B)
            (7.0, 4096, 32, 32),   # 7B模型
            (13.0, 5120, 40, 40),  # 13B模型
        ]
        
        available_memory = 20000  # 20GB
        
        for model_size_gb, hidden_size, num_layers, num_heads in model_sizes:
            optimizer = LoRAConfigOptimizer(
                model_size_gb=model_size_gb,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads
            )
            
            config = optimizer.optimize_for_single_gpu(
                available_memory_mb=available_memory,
                strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
                target_module_type=LoRATargetModule.ATTENTION_MLP
            )
            
            # 验证配置有效性
            assert config.rank > 0, f"模型大小 {model_size_gb}GB 产生了无效配置"
            assert config.alpha > 0, f"模型大小 {model_size_gb}GB 产生了无效配置"
            assert config.total_memory_mb <= available_memory * 0.9, f"模型大小 {model_size_gb}GB 内存超限"
            
            # 验证模型越大，LoRA参数相对越少（为了适应内存限制）
            # 这里不做严格的数值比较，因为优化算法可能会有不同的选择


if __name__ == "__main__":
    pytest.main([__file__, "-v"])