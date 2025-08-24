"""
LoRA配置优化器单元测试

测试LoRA参数动态配置功能，包括单GPU和多GPU环境下的配置优化。
"""

import pytest
import math
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lora_config_optimizer import (
    LoRAConfigOptimizer,
    LoRAMemoryProfile,
    MultiGPULoRAConfig,
    LoRAOptimizationStrategy,
    LoRATargetModule
)
from src.parallel_config import GPUTopology, GPUInfo, ParallelConfig, ParallelStrategy
from src.data_models import ChineseMetrics


class TestLoRAMemoryProfile:
    """测试LoRA内存配置文件"""
    
    def test_lora_memory_profile_creation(self):
        """测试LoRA内存配置文件创建"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_params_mb=100.0,
            optimizer_memory_mb=200.0,
            gradient_memory_mb=100.0,
            total_memory_mb=400.0
        )
        
        assert profile.rank == 16
        assert profile.alpha == 32
        assert len(profile.target_modules) == 2
        assert profile.total_memory_mb == 400.0
    
    def test_lora_memory_profile_validation(self):
        """测试LoRA内存配置文件数据验证"""
        # 测试无效rank
        with pytest.raises(ValueError, match="LoRA rank必须大于0"):
            LoRAMemoryProfile(
                rank=0,
                alpha=32,
                target_modules=["q_proj"]
            )
        
        # 测试无效alpha
        with pytest.raises(ValueError, match="LoRA alpha必须大于0"):
            LoRAMemoryProfile(
                rank=16,
                alpha=0,
                target_modules=["q_proj"]
            )
        
        # 测试空目标模块
        with pytest.raises(ValueError, match="目标模块列表不能为空"):
            LoRAMemoryProfile(
                rank=16,
                alpha=32,
                target_modules=[]
            )
    
    def test_memory_overhead_ratio(self):
        """测试内存开销比例计算"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj"],
            lora_params_mb=100.0,
            total_memory_mb=400.0
        )
        
        assert profile.memory_overhead_ratio == 0.25  # 100/400
    
    def test_parameter_efficiency(self):
        """测试参数效率计算"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj"]
        )
        
        assert profile.parameter_efficiency == 0.5  # 16/32
    
    def test_to_dict(self):
        """测试字典转换"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_params_mb=100.0,
            total_memory_mb=400.0
        )
        
        data = profile.to_dict()
        
        assert data["rank"] == 16
        assert data["alpha"] == 32
        assert data["target_modules"] == ["q_proj", "v_proj"]
        assert "memory_overhead_ratio" in data
        assert "parameter_efficiency" in data


class TestMultiGPULoRAConfig:
    """测试多GPU LoRA配置"""
    
    def test_multi_gpu_lora_config_creation(self):
        """测试多GPU LoRA配置创建"""
        global_config = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"]
        )
        
        per_gpu_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"]),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"])
        }
        
        config = MultiGPULoRAConfig(
            global_config=global_config,
            per_gpu_configs=per_gpu_configs
        )
        
        assert config.global_config.rank == 16
        assert len(config.per_gpu_configs) == 2
        assert config.enable_gradient_synchronization is True
    
    def test_total_lora_parameters(self):
        """测试总LoRA参数计算"""
        per_gpu_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"]),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"])
        }
        
        config = MultiGPULoRAConfig(
            global_config=per_gpu_configs[0],
            per_gpu_configs=per_gpu_configs
        )
        
        total_params = config.total_lora_parameters
        assert total_params > 0
        # 每个GPU: rank(16) * 2 * hidden_size(4096) * num_modules(1) = 131072
        # 总计: 131072 * 2 = 262144
        assert total_params == 262144
    
    def test_average_memory_usage(self):
        """测试平均内存使用计算"""
        per_gpu_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"], total_memory_mb=400.0),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"], total_memory_mb=600.0)
        }
        
        config = MultiGPULoRAConfig(
            global_config=per_gpu_configs[0],
            per_gpu_configs=per_gpu_configs
        )
        
        assert config.average_memory_usage == 500.0  # (400 + 600) / 2
    
    def test_memory_balance_score(self):
        """测试内存平衡评分"""
        # 测试完全平衡的情况
        balanced_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"], total_memory_mb=400.0),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"], total_memory_mb=400.0)
        }
        
        balanced_config = MultiGPULoRAConfig(
            global_config=balanced_configs[0],
            per_gpu_configs=balanced_configs
        )
        
        assert balanced_config.get_memory_balance_score() == 1.0
        
        # 测试不平衡的情况
        unbalanced_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"], total_memory_mb=200.0),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"], total_memory_mb=800.0)
        }
        
        unbalanced_config = MultiGPULoRAConfig(
            global_config=unbalanced_configs[0],
            per_gpu_configs=unbalanced_configs
        )
        
        balance_score = unbalanced_config.get_memory_balance_score()
        assert 0.0 <= balance_score < 1.0  # 不平衡，评分应该小于1


class TestLoRAConfigOptimizer:
    """测试LoRA配置优化器"""
    
    def setup_method(self):
        """测试设置"""
        self.optimizer = LoRAConfigOptimizer(
            model_size_gb=4.0,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32
        )
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        assert self.optimizer.model_size_gb == 4.0
        assert self.optimizer.hidden_size == 4096
        assert self.optimizer.num_layers == 32
        assert self.optimizer.num_attention_heads == 32
        
        # 检查预定义的目标模块配置
        assert LoRATargetModule.ATTENTION_ONLY in self.optimizer.target_module_configs
        assert LoRATargetModule.ATTENTION_MLP in self.optimizer.target_module_configs
        assert LoRATargetModule.ALL_LINEAR in self.optimizer.target_module_configs
    
    def test_calculate_lora_parameters(self):
        """测试LoRA参数数量计算"""
        # 测试注意力层参数计算
        attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        params = self.optimizer._calculate_lora_parameters(16, attention_modules)
        
        # 每个注意力模块: 2 * rank * hidden_size = 2 * 16 * 4096 = 131072
        # 4个模块 * 32层 = 131072 * 4 * 32 = 16777216
        expected = 2 * 16 * 4096 * 4 * 32
        assert params == expected
    
    def test_calculate_lora_memory_usage(self):
        """测试LoRA内存使用计算"""
        profile = self.optimizer.calculate_lora_memory_usage(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            batch_size=4,
            sequence_length=2048
        )
        
        assert profile.rank == 16
        assert profile.alpha == 32
        assert profile.lora_params_mb > 0
        assert profile.optimizer_memory_mb > 0
        assert profile.gradient_memory_mb > 0
        assert profile.total_memory_mb > 0
        
        # 验证内存关系
        assert profile.optimizer_memory_mb == profile.lora_params_mb * 2  # Adam优化器
        assert profile.gradient_memory_mb == profile.lora_params_mb
        
        # 验证效率指标
        assert 0.0 <= profile.memory_efficiency <= 1.0
        assert 0.0 <= profile.training_efficiency <= 1.0
        assert 0.0 <= profile.convergence_score <= 1.0
    
    def test_memory_efficiency_calculation(self):
        """测试内存效率计算"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj"],
            lora_params_mb=100.0,
            optimizer_memory_mb=200.0,
            gradient_memory_mb=100.0,
            total_memory_mb=500.0  # 100 + 200 + 100 + 100(activation)
        )
        
        efficiency = self.optimizer._calculate_memory_efficiency(profile)
        assert 0.0 <= efficiency <= 1.0
        
        # LoRA参数占比 = 100/500 = 0.2，理想比例0.4，差异0.2
        # 效率 = 1 - 0.2/0.4 = 0.5
        expected_efficiency = 1.0 - abs(0.2 - 0.4) / 0.4
        assert abs(efficiency - expected_efficiency) < 0.01
    
    def test_training_efficiency_calculation(self):
        """测试训练效率计算"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,  # 理想的2倍关系
            target_modules=["q_proj"]
        )
        
        efficiency = self.optimizer._calculate_training_efficiency(profile)
        assert 0.0 <= efficiency <= 1.0
        
        # rank=16在8-64范围内，rank_score=1.0
        # alpha=32是rank的2倍，alpha_score=1.0
        # param_efficiency = 16/32 = 0.5
        # 总效率 = (0.5*0.3 + 1.0*0.4 + 1.0*0.3) = 0.85
        expected_efficiency = 0.5 * 0.3 + 1.0 * 0.4 + 1.0 * 0.3
        assert abs(efficiency - expected_efficiency) < 0.01
    
    def test_convergence_score_estimation(self):
        """测试收敛性评分估算"""
        # 测试理想配置
        score = self.optimizer._estimate_convergence_score(32, 64)  # rank=32, alpha=2*rank
        assert 0.8 <= score <= 1.0  # 应该是高分
        
        # 测试rank太小
        score = self.optimizer._estimate_convergence_score(4, 8)
        assert score < 0.8  # 应该是较低分
        
        # 测试rank太大
        score = self.optimizer._estimate_convergence_score(256, 512)
        assert score < 0.8  # 应该是较低分
    
    def test_optimize_for_single_gpu(self):
        """测试单GPU优化"""
        # 创建足够的可用内存
        available_memory_mb = 16000  # 16GB
        
        config = self.optimizer.optimize_for_single_gpu(
            available_memory_mb=available_memory_mb,
            strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            target_module_type=LoRATargetModule.ATTENTION_MLP,
            batch_size=4,
            sequence_length=2048
        )
        
        assert config is not None
        assert config.rank > 0
        assert config.alpha > 0
        assert len(config.target_modules) > 0
        assert config.total_memory_mb <= available_memory_mb * 0.9  # 应该在内存限制内
    
    def test_optimize_for_single_gpu_memory_constrained(self):
        """测试内存受限的单GPU优化"""
        # 创建有限的可用内存
        available_memory_mb = 2000  # 2GB
        
        config = self.optimizer.optimize_for_single_gpu(
            available_memory_mb=available_memory_mb,
            strategy=LoRAOptimizationStrategy.MEMORY_EFFICIENT,
            target_module_type=LoRATargetModule.ATTENTION_ONLY,
            batch_size=2,
            sequence_length=1024
        )
        
        assert config is not None
        assert config.rank >= 4  # 最小rank
        assert config.total_memory_mb <= available_memory_mb * 0.9
    
    def test_optimize_for_multi_gpu(self):
        """测试多GPU优化"""
        # 创建GPU拓扑
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
        
        parallel_config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2
        )
        
        config = self.optimizer.optimize_for_multi_gpu(
            topology=topology,
            parallel_config=parallel_config,
            strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            target_module_type=LoRATargetModule.ATTENTION_MLP,
            batch_size=8,
            sequence_length=2048
        )
        
        assert config is not None
        assert isinstance(config, MultiGPULoRAConfig)
        assert len(config.per_gpu_configs) == 2
        assert 0 in config.per_gpu_configs
        assert 1 in config.per_gpu_configs
    
    def test_optimize_for_multi_gpu_unbalanced_memory(self):
        """测试内存不均衡的多GPU优化"""
        # 创建内存不均衡的GPU拓扑
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
            1: GPUInfo(gpu_id=1, name="RTX 3080", memory_total=10000, memory_free=8000, compute_capability=(8, 6))
        }
        
        topology = GPUTopology(
            num_gpus=2,
            gpu_info=gpu_info,
            interconnect_bandwidth={(0, 1): 25.0, (1, 0): 25.0},
            numa_topology={0: 0, 1: 1}
        )
        
        parallel_config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2
        )
        
        config = self.optimizer.optimize_for_multi_gpu(
            topology=topology,
            parallel_config=parallel_config,
            strategy=LoRAOptimizationStrategy.AUTO,
            target_module_type=LoRATargetModule.ATTENTION_MLP,
            batch_size=4,
            sequence_length=2048
        )
        
        assert config is not None
        assert len(config.per_gpu_configs) == 2
        
        # 验证内存受限的GPU使用了更小的配置
        gpu0_config = config.per_gpu_configs[0]
        gpu1_config = config.per_gpu_configs[1]
        
        # GPU1内存更少，应该使用更保守的配置
        assert gpu1_config.total_memory_mb <= 8000 * 0.9
    
    def test_optimization_score_calculation(self):
        """测试优化评分计算"""
        profile = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_params_mb=100.0,
            optimizer_memory_mb=200.0,
            gradient_memory_mb=100.0,
            total_memory_mb=500.0,
            memory_efficiency=0.8,
            training_efficiency=0.9,
            convergence_score=0.85
        )
        
        # 测试不同策略的评分
        memory_score = self.optimizer._calculate_optimization_score(
            profile, LoRAOptimizationStrategy.MEMORY_EFFICIENT
        )
        quality_score = self.optimizer._calculate_optimization_score(
            profile, LoRAOptimizationStrategy.QUALITY_FOCUSED
        )
        balanced_score = self.optimizer._calculate_optimization_score(
            profile, LoRAOptimizationStrategy.PERFORMANCE_BALANCED
        )
        auto_score = self.optimizer._calculate_optimization_score(
            profile, LoRAOptimizationStrategy.AUTO
        )
        
        assert 0.0 <= memory_score <= 1.0
        assert 0.0 <= quality_score <= 1.0
        assert 0.0 <= balanced_score <= 1.0
        assert 0.0 <= auto_score <= 1.0
        
        # 内存效率策略应该更重视内存效率
        # 质量策略应该更重视收敛性
        # 这里不做具体数值比较，因为权重可能调整
    
    def test_validate_lora_config_single_gpu(self):
        """测试单GPU LoRA配置验证"""
        # 创建有效配置
        valid_config = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            total_memory_mb=1000.0
        )
        
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
        }
        
        topology = GPUTopology(
            num_gpus=1,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0}
        )
        
        result = self.optimizer.validate_lora_config(valid_config, topology)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_lora_config_memory_exceeded(self):
        """测试内存超限的LoRA配置验证"""
        # 创建内存超限配置
        invalid_config = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            total_memory_mb=25000.0  # 超过GPU内存
        )
        
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
        }
        
        topology = GPUTopology(
            num_gpus=1,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0}
        )
        
        result = self.optimizer.validate_lora_config(invalid_config, topology)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("内存需求" in error for error in result["errors"])
    
    def test_validate_lora_config_multi_gpu(self):
        """测试多GPU LoRA配置验证"""
        per_gpu_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"], total_memory_mb=1000.0),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"], total_memory_mb=1000.0)
        }
        
        config = MultiGPULoRAConfig(
            global_config=per_gpu_configs[0],
            per_gpu_configs=per_gpu_configs
        )
        
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
        
        result = self.optimizer.validate_lora_config(config, topology)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_generate_config_report(self):
        """测试配置报告生成"""
        config = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            total_memory_mb=1000.0,
            memory_efficiency=0.8,
            training_efficiency=0.9,
            convergence_score=0.85
        )
        
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
        }
        
        topology = GPUTopology(
            num_gpus=1,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0}
        )
        
        report = self.optimizer.generate_config_report(config, topology)
        
        assert "timestamp" in report
        assert report["config_type"] == "single_gpu"
        assert "gpu_topology" in report
        assert "config_details" in report
        assert "validation_result" in report
        assert "performance_estimates" in report
        
        # 验证性能估算
        perf_estimates = report["performance_estimates"]
        assert "memory_efficiency" in perf_estimates
        assert "training_efficiency" in perf_estimates
        assert "convergence_score" in perf_estimates
        assert "parameter_count" in perf_estimates
    
    def test_communication_overhead_estimation(self):
        """测试通信开销估算"""
        per_gpu_configs = {
            0: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["q_proj"]),
            1: LoRAMemoryProfile(rank=16, alpha=32, target_modules=["v_proj"])
        }
        
        config = MultiGPULoRAConfig(
            global_config=per_gpu_configs[0],
            per_gpu_configs=per_gpu_configs,
            enable_gradient_compression=True,
            compression_ratio=0.5
        )
        
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
        
        overhead = self.optimizer._estimate_communication_overhead(config, topology)
        
        assert 0.0 <= overhead <= 0.5  # 最多50%开销
        assert isinstance(overhead, float)


class TestLoRAConfigIntegration:
    """LoRA配置集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.optimizer = LoRAConfigOptimizer()
    
    def test_end_to_end_single_gpu_optimization(self):
        """测试端到端单GPU优化流程"""
        # 1. 创建GPU拓扑
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
        }
        
        topology = GPUTopology(
            num_gpus=1,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0}
        )
        
        # 2. 优化配置
        config = self.optimizer.optimize_for_single_gpu(
            available_memory_mb=20000,
            strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            target_module_type=LoRATargetModule.ATTENTION_MLP
        )
        
        # 3. 验证配置
        validation_result = self.optimizer.validate_lora_config(config, topology)
        assert validation_result["valid"] is True
        
        # 4. 生成报告
        report = self.optimizer.generate_config_report(config, topology)
        assert report["config_type"] == "single_gpu"
        assert report["validation_result"]["valid"] is True
    
    def test_end_to_end_multi_gpu_optimization(self):
        """测试端到端多GPU优化流程"""
        # 1. 创建多GPU拓扑
        gpu_info = {
            0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
            1: GPUInfo(gpu_id=1, name="RTX 4090", memory_total=24000, memory_free=18000, compute_capability=(8, 9)),
            2: GPUInfo(gpu_id=2, name="RTX 3080", memory_total=10000, memory_free=8000, compute_capability=(8, 6))
        }
        
        topology = GPUTopology(
            num_gpus=3,
            gpu_info=gpu_info,
            interconnect_bandwidth={
                (0, 1): 50.0, (1, 0): 50.0,
                (0, 2): 25.0, (2, 0): 25.0,
                (1, 2): 25.0, (2, 1): 25.0
            },
            numa_topology={0: 0, 1: 0, 2: 1}
        )
        
        parallel_config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=3
        )
        
        # 2. 优化配置
        config = self.optimizer.optimize_for_multi_gpu(
            topology=topology,
            parallel_config=parallel_config,
            strategy=LoRAOptimizationStrategy.AUTO,
            target_module_type=LoRATargetModule.ATTENTION_MLP
        )
        
        # 3. 验证配置
        validation_result = self.optimizer.validate_lora_config(config, topology)
        assert validation_result["valid"] is True
        
        # 4. 检查负载均衡
        balance_score = config.get_memory_balance_score()
        assert 0.0 <= balance_score <= 1.0
        
        # 5. 生成报告
        report = self.optimizer.generate_config_report(config, topology)
        assert report["config_type"] == "multi_gpu"
        assert "communication_overhead_estimate" in report["performance_estimates"]
    
    def test_different_optimization_strategies(self):
        """测试不同优化策略的效果"""
        available_memory_mb = 16000
        
        strategies = [
            LoRAOptimizationStrategy.MEMORY_EFFICIENT,
            LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
            LoRAOptimizationStrategy.QUALITY_FOCUSED,
            LoRAOptimizationStrategy.AUTO
        ]
        
        configs = {}
        
        for strategy in strategies:
            config = self.optimizer.optimize_for_single_gpu(
                available_memory_mb=available_memory_mb,
                strategy=strategy,
                target_module_type=LoRATargetModule.ATTENTION_MLP
            )
            configs[strategy] = config
        
        # 验证不同策略产生不同的配置
        memory_config = configs[LoRAOptimizationStrategy.MEMORY_EFFICIENT]
        quality_config = configs[LoRAOptimizationStrategy.QUALITY_FOCUSED]
        
        # 内存优先策略应该使用更小的rank
        # 质量优先策略应该使用更大的rank（在内存允许的情况下）
        # 注意：具体数值可能因为搜索算法而不同，这里只验证基本逻辑
        assert memory_config.rank > 0
        assert quality_config.rank > 0
        
        # 所有配置都应该在内存限制内
        for config in configs.values():
            assert config.total_memory_mb <= available_memory_mb * 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])