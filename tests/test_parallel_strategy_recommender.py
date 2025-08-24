#!/usr/bin/env python3
"""
并行策略推荐器测试模块
测试并行策略自动推荐功能
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.parallel_strategy_recommender import (
    ParallelStrategyRecommender, 
    StrategyRecommendation,
    ParallelStrategy,
    ModelRequirements,
    SimpleParallelConfig
)
from src.gpu_utils import GPUTopology, GPUInfo, GPUInterconnect, InterconnectType


class TestParallelStrategyRecommender:
    """并行策略推荐器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.recommender = ParallelStrategyRecommender()
    
    def test_init(self):
        """测试初始化"""
        assert self.recommender is not None
        assert self.recommender.model_requirements is not None
        assert self.recommender.gpu_detector is not None
    
    def test_custom_model_requirements(self):
        """测试自定义模型需求"""
        custom_requirements = ModelRequirements(
            model_name="Custom-Model",
            model_size_gb=16.0,
            min_memory_per_gpu=16384
        )
        
        recommender = ParallelStrategyRecommender(custom_requirements)
        assert recommender.model_requirements.model_name == "Custom-Model"
        assert recommender.model_requirements.model_size_gb == 16.0
    
    def test_estimate_memory_requirements(self):
        """测试内存需求估算"""
        memory_req = self.recommender._estimate_memory_requirements(
            batch_size=4,
            sequence_length=2048,
            enable_lora=True,
            lora_rank=64
        )
        
        assert "model_params_gb" in memory_req
        assert "optimizer_states_gb" in memory_req
        assert "activation_memory_gb" in memory_req
        assert "gradient_memory_gb" in memory_req
        assert "total_memory_gb" in memory_req
        assert "per_gpu_memory_mb" in memory_req
        
        # 验证LoRA减少了优化器状态内存
        assert memory_req["optimizer_states_gb"] < memory_req["model_params_gb"]
    
    def test_estimate_memory_requirements_without_lora(self):
        """测试不使用LoRA的内存需求估算"""
        memory_req = self.recommender._estimate_memory_requirements(
            batch_size=4,
            sequence_length=2048,
            enable_lora=False,
            lora_rank=0
        )
        
        # 不使用LoRA时，优化器状态内存应该是模型参数的2倍
        assert memory_req["optimizer_states_gb"] == memory_req["model_params_gb"] * 2
    
    def test_analyze_hardware_capabilities_single_gpu(self):
        """测试单GPU硬件能力分析"""
        # 创建单GPU拓扑
        gpu_info = GPUInfo(
            gpu_id=0,
            name="RTX 4090",
            total_memory=24576,
            free_memory=20000,
            used_memory=4576,
            utilization=10.0
        )
        
        topology = GPUTopology(
            num_gpus=1,
            gpu_info={0: gpu_info},
            interconnects=[],
            numa_topology={},
            bandwidth_matrix={},
            topology_type="Single"
        )
        
        analysis = self.recommender._analyze_hardware_capabilities(topology)
        
        assert analysis["num_gpus"] == 1
        assert analysis["total_memory_mb"] == 24576
        assert analysis["min_memory_mb"] == 24576
        assert analysis["max_memory_mb"] == 24576
        assert analysis["avg_memory_mb"] == 24576
        assert not analysis["has_nvlink"]
        assert not analysis["has_high_bandwidth"]
        assert not analysis["numa_aware"]
    
    def test_analyze_hardware_capabilities_multi_gpu_nvlink(self):
        """测试多GPU NVLink硬件能力分析"""
        # 创建多GPU拓扑
        gpu_info_0 = GPUInfo(gpu_id=0, name="RTX 4090", total_memory=24576, 
                           free_memory=20000, used_memory=4576, utilization=10.0)
        gpu_info_1 = GPUInfo(gpu_id=1, name="RTX 4090", total_memory=24576,
                           free_memory=20000, used_memory=4576, utilization=10.0)
        
        interconnect = GPUInterconnect(
            gpu_a=0, gpu_b=1,
            interconnect_type=InterconnectType.NVLINK,
            bandwidth_gbps=50.0
        )
        
        topology = GPUTopology(
            num_gpus=2,
            gpu_info={0: gpu_info_0, 1: gpu_info_1},
            interconnects=[interconnect],
            numa_topology={0: 0, 1: 1},
            bandwidth_matrix={(0, 1): 50.0},
            topology_type="NVLink"
        )
        
        analysis = self.recommender._analyze_hardware_capabilities(topology)
        
        assert analysis["num_gpus"] == 2
        assert analysis["total_memory_mb"] == 49152
        assert analysis["has_nvlink"]
        assert analysis["has_high_bandwidth"]
        assert analysis["numa_aware"]
        assert InterconnectType.NVLINK in analysis["interconnect_types"]
    
    def test_calculate_topology_score(self):
        """测试拓扑评分计算"""
        # 创建高质量拓扑
        gpu_info = GPUInfo(gpu_id=0, name="RTX 4090", total_memory=24576,
                         free_memory=20000, used_memory=4576, utilization=10.0)
        
        topology = GPUTopology(
            num_gpus=2,
            gpu_info={0: gpu_info, 1: gpu_info},
            interconnects=[],
            numa_topology={0: 0, 1: 1},
            bandwidth_matrix={},
            topology_type="PCIe"
        )
        
        analysis = {
            "has_nvlink": True,
            "has_high_bandwidth": True,
            "numa_aware": True,
            "min_memory_mb": 24576
        }
        
        score = self.recommender._calculate_topology_score(topology, analysis)
        
        assert 0 <= score <= 100
        assert score > 80  # 高质量配置应该得到高分
    
    @patch('src.parallel_strategy_recommender.GPUDetector')
    def test_recommend_strategy_single_gpu(self, mock_gpu_detector):
        """测试单GPU策略推荐"""
        # Mock GPU检测器
        mock_detector = Mock()
        mock_gpu_detector.return_value = mock_detector
        
        # 创建单GPU拓扑
        gpu_info = GPUInfo(gpu_id=0, name="RTX 4090", total_memory=24576,
                         free_memory=20000, used_memory=4576, utilization=10.0)
        
        topology = GPUTopology(
            num_gpus=1,
            gpu_info={0: gpu_info},
            interconnects=[],
            numa_topology={},
            bandwidth_matrix={},
            topology_type="Single"
        )
        
        mock_detector.detect_gpu_topology.return_value = topology
        
        # 重新创建推荐器以使用mock
        recommender = ParallelStrategyRecommender()
        recommender.gpu_detector = mock_detector
        
        recommendation = recommender.recommend_strategy(
            batch_size=4,
            sequence_length=2048,
            enable_lora=True,
            lora_rank=64
        )
        
        assert recommendation.strategy == ParallelStrategy.SINGLE_GPU
        assert not recommendation.config.data_parallel
        assert not recommendation.config.model_parallel
        assert recommendation.config.tensor_parallel_size == 1
        assert recommendation.config.data_parallel_size == 1
        assert recommendation.confidence > 0.7
    
    @patch('src.parallel_strategy_recommender.GPUDetector')
    def test_recommend_strategy_data_parallel(self, mock_gpu_detector):
        """测试数据并行策略推荐"""
        # Mock GPU检测器
        mock_detector = Mock()
        mock_gpu_detector.return_value = mock_detector
        
        # 创建多GPU拓扑（内存充足）
        gpu_info = GPUInfo(gpu_id=0, name="RTX 4090", total_memory=24576,
                         free_memory=20000, used_memory=4576, utilization=10.0)
        
        topology = GPUTopology(
            num_gpus=2,
            gpu_info={0: gpu_info, 1: gpu_info},
            interconnects=[],
            numa_topology={},
            bandwidth_matrix={},
            topology_type="PCIe"
        )
        
        mock_detector.detect_gpu_topology.return_value = topology
        
        # 重新创建推荐器以使用mock
        recommender = ParallelStrategyRecommender()
        recommender.gpu_detector = mock_detector
        
        recommendation = recommender.recommend_strategy(
            batch_size=4,
            sequence_length=2048,
            enable_lora=True,
            lora_rank=64
        )
        
        assert recommendation.strategy == ParallelStrategy.DATA_PARALLEL
        assert recommendation.config.data_parallel
        assert not recommendation.config.model_parallel
        assert recommendation.config.data_parallel_size == 2
        assert recommendation.config.tensor_parallel_size == 1
        assert recommendation.confidence > 0.8
    
    @patch('src.parallel_strategy_recommender.GPUDetector')
    def test_recommend_strategy_model_parallel(self, mock_gpu_detector):
        """测试模型并行策略推荐"""
        # Mock GPU检测器
        mock_detector = Mock()
        mock_gpu_detector.return_value = mock_detector
        
        # 创建多GPU拓扑（内存不足）
        gpu_info = GPUInfo(gpu_id=0, name="RTX 3060", total_memory=8192,
                         free_memory=6000, used_memory=2192, utilization=10.0)
        
        topology = GPUTopology(
            num_gpus=2,
            gpu_info={0: gpu_info, 1: gpu_info},
            interconnects=[],
            numa_topology={},
            bandwidth_matrix={},
            topology_type="PCIe"
        )
        
        mock_detector.detect_gpu_topology.return_value = topology
        
        # 重新创建推荐器以使用mock
        recommender = ParallelStrategyRecommender()
        recommender.gpu_detector = mock_detector
        
        recommendation = recommender.recommend_strategy(
            batch_size=4,
            sequence_length=2048,
            enable_lora=False,  # 不使用LoRA，增加内存需求
            lora_rank=0
        )
        
        assert recommendation.strategy == ParallelStrategy.MODEL_PARALLEL
        assert not recommendation.config.data_parallel
        assert recommendation.config.model_parallel
        assert recommendation.config.tensor_parallel_size > 1
        assert recommendation.config.data_parallel_size == 1
    
    def test_get_optimization_suggestions(self):
        """测试优化建议获取"""
        # 创建单GPU推荐
        config = SimpleParallelConfig(
            data_parallel=False,
            model_parallel=False,
            pipeline_parallel=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        
        recommendation = StrategyRecommendation(
            strategy=ParallelStrategy.SINGLE_GPU,
            config=config,
            confidence=0.9
        )
        
        suggestions = self.recommender.get_optimization_suggestions(recommendation)
        
        assert len(suggestions) > 0
        assert any("梯度检查点" in s for s in suggestions)
        assert any("混合精度" in s for s in suggestions)
        assert any("LoRA" in s for s in suggestions)
    
    def test_compare_strategies(self):
        """测试策略比较"""
        # 创建两个策略
        config1 = SimpleParallelConfig(data_parallel=False, model_parallel=False)
        strategy1 = StrategyRecommendation(
            strategy=ParallelStrategy.SINGLE_GPU,
            config=config1,
            confidence=0.8
        )
        strategy1.expected_performance = {
            "training_speed": 1.0,
            "memory_efficiency": 0.7,
            "scalability": 0.3
        }
        
        config2 = SimpleParallelConfig(data_parallel=True, model_parallel=False)
        strategy2 = StrategyRecommendation(
            strategy=ParallelStrategy.DATA_PARALLEL,
            config=config2,
            confidence=0.9
        )
        strategy2.expected_performance = {
            "training_speed": 1.8,
            "memory_efficiency": 0.8,
            "scalability": 0.9
        }
        
        best = self.recommender.compare_strategies([strategy1, strategy2])
        
        # 数据并行策略应该更好
        assert best.strategy == ParallelStrategy.DATA_PARALLEL
    
    def test_compare_strategies_empty_list(self):
        """测试空策略列表比较"""
        with pytest.raises(ValueError):
            self.recommender.compare_strategies([])
    
    def test_compare_strategies_single_strategy(self):
        """测试单个策略比较"""
        config = SimpleParallelConfig(data_parallel=False, model_parallel=False)
        strategy = StrategyRecommendation(
            strategy=ParallelStrategy.SINGLE_GPU,
            config=config,
            confidence=0.8
        )
        
        result = self.recommender.compare_strategies([strategy])
        assert result == strategy
    
    @patch('builtins.open', create=True)
    @patch('yaml.dump')
    def test_generate_config_file(self, mock_yaml_dump, mock_open):
        """测试配置文件生成"""
        # 创建推荐
        config = SimpleParallelConfig(
            data_parallel=True,
            model_parallel=False,
            data_parallel_size=2
        )
        
        recommendation = StrategyRecommendation(
            strategy=ParallelStrategy.DATA_PARALLEL,
            config=config,
            confidence=0.9
        )
        recommendation.add_reasoning("测试理由")
        recommendation.add_warning("测试警告")
        recommendation.expected_performance = {"training_speed": 1.8}
        
        # Mock文件操作
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        output_path = self.recommender.generate_config_file(recommendation, "test_config.yaml")
        
        assert output_path == "test_config.yaml"
        mock_open.assert_called_once_with("test_config.yaml", 'w', encoding='utf-8')
        mock_yaml_dump.assert_called_once()
        
        # 验证传递给yaml.dump的数据结构
        call_args = mock_yaml_dump.call_args[0][0]
        assert "parallel_strategy" in call_args
        assert "reasoning" in call_args
        assert "warnings" in call_args
        assert "expected_performance" in call_args
        assert "optimization_suggestions" in call_args


class TestStrategyRecommendation:
    """策略推荐结果测试类"""
    
    def test_init(self):
        """测试初始化"""
        config = ParallelConfig(data_parallel=True)
        recommendation = StrategyRecommendation(
            strategy=ParallelStrategy.DATA_PARALLEL,
            config=config,
            confidence=0.9
        )
        
        assert recommendation.strategy == ParallelStrategy.DATA_PARALLEL
        assert recommendation.config == config
        assert recommendation.confidence == 0.9
        assert recommendation.reasoning == []
        assert recommendation.warnings == []
        assert recommendation.expected_performance == {}
    
    def test_add_reasoning(self):
        """测试添加推荐理由"""
        config = ParallelConfig(data_parallel=True)
        recommendation = StrategyRecommendation(
            strategy=ParallelStrategy.DATA_PARALLEL,
            config=config,
            confidence=0.9
        )
        
        recommendation.add_reasoning("测试理由1")
        recommendation.add_reasoning("测试理由2")
        
        assert len(recommendation.reasoning) == 2
        assert "测试理由1" in recommendation.reasoning
        assert "测试理由2" in recommendation.reasoning
    
    def test_add_warning(self):
        """测试添加警告"""
        config = ParallelConfig(data_parallel=True)
        recommendation = StrategyRecommendation(
            strategy=ParallelStrategy.DATA_PARALLEL,
            config=config,
            confidence=0.9
        )
        
        recommendation.add_warning("测试警告1")
        recommendation.add_warning("测试警告2")
        
        assert len(recommendation.warnings) == 2
        assert "测试警告1" in recommendation.warnings
        assert "测试警告2" in recommendation.warnings


class TestModelRequirements:
    """模型需求测试类"""
    
    def test_default_values(self):
        """测试默认值"""
        requirements = ModelRequirements()
        
        assert requirements.model_name == "Qwen3-4B-Thinking"
        assert requirements.model_size_gb == 8.0
        assert requirements.min_memory_per_gpu == 8192
        assert requirements.recommended_memory_per_gpu == 16384
        assert requirements.supports_gradient_checkpointing
        assert requirements.supports_mixed_precision
    
    def test_custom_values(self):
        """测试自定义值"""
        requirements = ModelRequirements(
            model_name="Custom-Model",
            model_size_gb=16.0,
            min_memory_per_gpu=16384,
            recommended_memory_per_gpu=32768,
            supports_gradient_checkpointing=False
        )
        
        assert requirements.model_name == "Custom-Model"
        assert requirements.model_size_gb == 16.0
        assert requirements.min_memory_per_gpu == 16384
        assert requirements.recommended_memory_per_gpu == 32768
        assert not requirements.supports_gradient_checkpointing


if __name__ == "__main__":
    pytest.main([__file__])