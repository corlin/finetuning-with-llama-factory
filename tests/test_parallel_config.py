"""
多GPU并行配置模型单元测试

测试GPU拓扑检测、并行策略配置、分布式训练指标等功能。
"""

import pytest
from datetime import datetime
from src.parallel_config import (
    GPUInfo, GPUTopology, ParallelConfig, DistributedTrainingMetrics,
    CommunicationMetrics, ParallelStrategy, CommunicationBackend, ZeroStage,
    ParallelConfigValidator
)


class TestGPUInfo:
    """GPU信息测试"""
    
    def test_gpu_info_creation(self):
        """测试GPU信息创建"""
        gpu = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24000,
            memory_free=20000,
            compute_capability=(8, 9),
            temperature=65.0,
            power_usage=350.0,
            utilization=80.0
        )
        
        assert gpu.gpu_id == 0
        assert gpu.name == "NVIDIA RTX 4090"
        assert gpu.memory_used == 4000
        assert abs(gpu.memory_usage_percent - 16.67) < 0.1
    
    def test_gpu_info_validation(self):
        """测试GPU信息验证"""
        # 测试负GPU ID
        with pytest.raises(ValueError, match="GPU ID必须为非负整数"):
            GPUInfo(
                gpu_id=-1,
                name="Test GPU",
                memory_total=8000,
                memory_free=6000,
                compute_capability=(7, 5)
            )
        
        # 测试无效内存
        with pytest.raises(ValueError, match="GPU总内存必须大于0"):
            GPUInfo(
                gpu_id=0,
                name="Test GPU",
                memory_total=0,
                memory_free=0,
                compute_capability=(7, 5)
            )
        
        # 测试无效利用率
        with pytest.raises(ValueError, match="GPU利用率必须在0-100之间"):
            GPUInfo(
                gpu_id=0,
                name="Test GPU",
                memory_total=8000,
                memory_free=6000,
                compute_capability=(7, 5),
                utilization=150.0
            )
    
    def test_gpu_info_serialization(self):
        """测试GPU信息序列化"""
        gpu = GPUInfo(
            gpu_id=0,
            name="Test GPU",
            memory_total=8000,
            memory_free=6000,
            compute_capability=(7, 5),
            utilization=75.0
        )
        
        data = gpu.to_dict()
        assert data["gpu_id"] == 0
        assert data["name"] == "Test GPU"
        assert data["memory_used"] == 2000
        assert data["memory_usage_percent"] == 25.0


class TestGPUTopology:
    """GPU拓扑测试"""
    
    def create_test_topology(self, num_gpus: int = 2) -> GPUTopology:
        """创建测试用的GPU拓扑"""
        gpu_info = {}
        for i in range(num_gpus):
            gpu_info[i] = GPUInfo(
                gpu_id=i,
                name=f"Test GPU {i}",
                memory_total=8000,
                memory_free=6000,
                compute_capability=(7, 5),
                utilization=50.0
            )
        
        return GPUTopology(
            num_gpus=num_gpus,
            gpu_info=gpu_info,
            interconnect_bandwidth={(0, 1): 50.0, (1, 0): 50.0},
            numa_topology={0: 0, 1: 0} if num_gpus >= 2 else {0: 0},
            nvlink_connections={0: [1], 1: [0]} if num_gpus >= 2 else {}
        )
    
    def test_gpu_topology_creation(self):
        """测试GPU拓扑创建"""
        topology = self.create_test_topology(2)
        
        assert topology.num_gpus == 2
        assert len(topology.gpu_info) == 2
        assert topology.get_total_memory() == 16000
        assert topology.get_available_memory() == 12000
    
    def test_gpu_topology_validation(self):
        """测试GPU拓扑验证"""
        # 测试GPU数量为0
        with pytest.raises(ValueError, match="GPU数量必须大于0"):
            GPUTopology(
                num_gpus=0,
                gpu_info={},
                interconnect_bandwidth={},
                numa_topology={}
            )
        
        # 测试GPU信息数量不匹配
        gpu_info = {0: GPUInfo(0, "GPU 0", 8000, 6000, (7, 5))}
        with pytest.raises(ValueError, match="GPU信息数量.*与GPU数量.*不匹配"):
            GPUTopology(
                num_gpus=2,
                gpu_info=gpu_info,
                interconnect_bandwidth={},
                numa_topology={0: 0, 1: 0}
            )
        
        # 测试GPU ID不连续
        gpu_info = {
            0: GPUInfo(0, "GPU 0", 8000, 6000, (7, 5)),
            2: GPUInfo(2, "GPU 2", 8000, 6000, (7, 5))
        }
        with pytest.raises(ValueError, match="GPU ID不连续"):
            GPUTopology(
                num_gpus=2,
                gpu_info=gpu_info,
                interconnect_bandwidth={},
                numa_topology={0: 0, 2: 0}
            )
    
    def test_memory_balanced_gpus(self):
        """测试内存均衡GPU获取"""
        gpu_info = {
            0: GPUInfo(0, "GPU 0", 8000, 6000, (7, 5), utilization=25.0),  # 25%使用率
            1: GPUInfo(1, "GPU 1", 8000, 4000, (7, 5), utilization=50.0),  # 50%使用率
            2: GPUInfo(2, "GPU 2", 8000, 2000, (7, 5), utilization=75.0)   # 75%使用率
        }
        
        topology = GPUTopology(
            num_gpus=3,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0, 1: 0, 2: 1}
        )
        
        balanced_gpus = topology.get_memory_balanced_gpus()
        assert balanced_gpus == [0, 1, 2]  # 按使用率从低到高排序
    
    def test_nvlink_connections(self):
        """测试NVLink连接"""
        topology = self.create_test_topology(2)
        
        assert topology.has_nvlink(0, 1) is True
        assert topology.has_nvlink(1, 0) is True
        assert topology.has_nvlink(0, 0) is False
    
    def test_optimal_gpu_pairs(self):
        """测试最优GPU配对"""
        topology = self.create_test_topology(4)
        topology.nvlink_connections = {0: [1], 1: [0], 2: [3], 3: [2]}
        
        pairs = topology.get_optimal_gpu_pairs()
        assert len(pairs) == 2
        assert (0, 1) in pairs or (1, 0) in pairs
        assert (2, 3) in pairs or (3, 2) in pairs
    
    def test_topology_serialization(self):
        """测试拓扑序列化"""
        topology = self.create_test_topology(2)
        
        data = topology.to_dict()
        assert data["num_gpus"] == 2
        assert data["total_memory"] == 16000
        assert data["available_memory"] == 12000
        assert "0" in data["gpu_info"]
        assert "1" in data["gpu_info"]


class TestParallelConfig:
    """并行配置测试"""
    
    def test_parallel_config_creation(self):
        """测试并行配置创建"""
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            communication_backend=CommunicationBackend.NCCL,
            enable_zero_optimization=True,
            zero_stage=ZeroStage.OPTIMIZER_GRADIENT
        )
        
        assert config.strategy == ParallelStrategy.DATA_PARALLEL
        assert config.world_size == 4
        assert config.is_distributed is True
    
    def test_parallel_config_validation(self):
        """测试并行配置验证"""
        # 测试无效的并行大小
        with pytest.raises(ValueError, match="数据并行大小必须大于0"):
            ParallelConfig(
                strategy=ParallelStrategy.DATA_PARALLEL,
                data_parallel_size=0
            )
        
        # 测试无效的端口
        with pytest.raises(ValueError, match="主节点端口必须在1024-65535范围内"):
            ParallelConfig(
                strategy=ParallelStrategy.DATA_PARALLEL,
                master_port=100
            )
    
    def test_auto_adjust_config(self):
        """测试自动配置调整"""
        # 测试数据并行策略
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=4
        )
        
        assert config.model_parallel_enabled is False
        assert config.pipeline_parallel_enabled is False
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        
        # 测试单GPU配置
        config_single = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
        
        assert config_single.model_parallel_enabled is False
        assert config_single.pipeline_parallel_enabled is False
        assert config_single.enable_zero_optimization is False
    
    def test_process_group_config(self):
        """测试进程组配置"""
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2,
            master_addr="192.168.1.100",
            master_port=29500
        )
        
        pg_config = config.get_process_group_config()
        assert pg_config["backend"] == "nccl"
        assert pg_config["init_method"] == "tcp://192.168.1.100:29500"
        assert pg_config["world_size"] == 2
    
    def test_validate_gpu_topology(self):
        """测试GPU拓扑验证"""
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2
        )
        
        # 创建足够的GPU拓扑
        gpu_info = {
            0: GPUInfo(0, "GPU 0", 8000, 6000, (7, 5)),
            1: GPUInfo(1, "GPU 1", 8000, 6000, (7, 5))
        }
        topology = GPUTopology(
            num_gpus=2,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0, 1: 0}
        )
        
        assert config.validate_gpu_topology(topology) is True
        
        # 测试GPU不足的情况
        config_large = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=4
        )
        assert config_large.validate_gpu_topology(topology) is False
    
    def test_optimize_for_topology(self):
        """测试拓扑优化"""
        config = ParallelConfig(
            strategy=ParallelStrategy.AUTO,
            data_parallel_size=1
        )
        
        # 创建8GPU拓扑
        gpu_info = {}
        for i in range(8):
            gpu_info[i] = GPUInfo(i, f"GPU {i}", 16000, 12000, (8, 0))
        
        topology = GPUTopology(
            num_gpus=8,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={i: i // 4 for i in range(8)}
        )
        
        optimized = config.optimize_for_topology(topology)
        assert optimized.data_parallel_size == 8
        assert optimized.tensor_parallel_size == 1
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = ParallelConfig(
            strategy=ParallelStrategy.HYBRID_PARALLEL,
            data_parallel_size=2,
            tensor_parallel_size=2,
            enable_mixed_precision=True,
            mixed_precision_dtype="bf16"
        )
        
        data = config.to_dict()
        assert data["strategy"] == "hybrid_parallel"
        assert data["world_size"] == 4
        assert data["mixed_precision_dtype"] == "bf16"
        
        # 测试反序列化
        restored_config = ParallelConfig.from_dict(data)
        assert restored_config.strategy == ParallelStrategy.HYBRID_PARALLEL
        assert restored_config.data_parallel_size == 2
        assert restored_config.tensor_parallel_size == 2


class TestCommunicationMetrics:
    """通信指标测试"""
    
    def test_communication_metrics_creation(self):
        """测试通信指标创建"""
        metrics = CommunicationMetrics(
            total_communication_time=10.0,
            allreduce_time=6.0,
            broadcast_time=2.0,
            p2p_time=2.0,
            communication_volume=1024.0,
            bandwidth_utilization=75.0
        )
        
        assert metrics.total_communication_time == 10.0
        assert metrics.communication_efficiency == 0.8  # (6+2)/10
    
    def test_communication_metrics_validation(self):
        """测试通信指标验证"""
        # 测试负通信时间
        with pytest.raises(ValueError, match="通信时间不能为负数"):
            CommunicationMetrics(total_communication_time=-1.0)
        
        # 测试无效带宽利用率
        with pytest.raises(ValueError, match="带宽利用率必须在0-100之间"):
            CommunicationMetrics(bandwidth_utilization=150.0)
    
    def test_communication_efficiency(self):
        """测试通信效率计算"""
        # 测试零通信时间
        metrics = CommunicationMetrics(total_communication_time=0.0)
        assert metrics.communication_efficiency == 1.0
        
        # 测试正常情况
        metrics = CommunicationMetrics(
            total_communication_time=10.0,
            allreduce_time=4.0,
            broadcast_time=3.0
        )
        assert metrics.communication_efficiency == 0.7


class TestDistributedTrainingMetrics:
    """分布式训练指标测试"""
    
    def test_distributed_training_metrics_creation(self):
        """测试分布式训练指标创建"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            train_loss=0.5,
            val_loss=0.6,
            throughput_tokens_per_second=1000.0
        )
        
        assert metrics.epoch == 1
        assert metrics.global_step == 100
        assert metrics.train_loss == 0.5
    
    def test_metrics_validation(self):
        """测试指标验证"""
        # 测试负epoch
        with pytest.raises(ValueError, match="epoch必须为非负整数"):
            DistributedTrainingMetrics(epoch=-1, global_step=0)
        
        # 测试负global_step
        with pytest.raises(ValueError, match="global_step必须为非负整数"):
            DistributedTrainingMetrics(epoch=0, global_step=-1)
    
    def test_load_balance_calculation(self):
        """测试负载均衡计算"""
        metrics = DistributedTrainingMetrics(epoch=1, global_step=100)
        
        # 添加GPU指标
        metrics.update_gpu_metrics(0, {"utilization": 80.0, "memory_usage_percent": 70.0})
        metrics.update_gpu_metrics(1, {"utilization": 85.0, "memory_usage_percent": 75.0})
        metrics.update_gpu_metrics(2, {"utilization": 75.0, "memory_usage_percent": 65.0})
        
        load_balance = metrics.calculate_load_balance_score()
        assert 0.0 <= load_balance <= 1.0
        assert metrics.gpu_utilization_variance > 0
    
    def test_memory_efficiency_calculation(self):
        """测试内存效率计算"""
        metrics = DistributedTrainingMetrics(epoch=1, global_step=100)
        
        # 测试理想内存使用率（75%）
        metrics.update_gpu_metrics(0, {"memory_usage_percent": 75.0})
        efficiency = metrics.calculate_memory_efficiency()
        assert efficiency == 1.0
        
        # 测试低内存使用率
        metrics.gpu_metrics = {0: {"memory_usage_percent": 30.0}}
        efficiency = metrics.calculate_memory_efficiency()
        assert efficiency == 0.5  # 30/60
        
        # 测试高内存使用率
        metrics.gpu_metrics = {0: {"memory_usage_percent": 95.0}}
        efficiency = metrics.calculate_memory_efficiency()
        assert efficiency == 0.5  # (100-95)/10
    
    def test_performance_summary(self):
        """测试性能摘要"""
        metrics = DistributedTrainingMetrics(
            epoch=2,
            global_step=200,
            train_loss=0.3,
            val_loss=0.4,
            throughput_tokens_per_second=1500.0
        )
        
        summary = metrics.get_performance_summary()
        assert summary["epoch"] == 2
        assert summary["global_step"] == 200
        assert summary["train_loss"] == 0.3
        assert summary["throughput_tokens_per_second"] == 1500.0
    
    def test_metrics_serialization(self):
        """测试指标序列化"""
        comm_metrics = CommunicationMetrics(
            total_communication_time=5.0,
            allreduce_time=3.0
        )
        
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            communication_metrics=comm_metrics
        )
        
        data = metrics.to_dict()
        assert data["epoch"] == 1
        assert data["communication_metrics"]["total_communication_time"] == 5.0
        
        # 测试反序列化
        restored_metrics = DistributedTrainingMetrics.from_dict(data)
        assert restored_metrics.epoch == 1
        assert restored_metrics.communication_metrics.total_communication_time == 5.0


class TestParallelConfigValidator:
    """并行配置验证器测试"""
    
    def create_test_setup(self):
        """创建测试设置"""
        gpu_info = {
            0: GPUInfo(0, "GPU 0", 8000, 6000, (7, 5)),
            1: GPUInfo(1, "GPU 1", 8000, 6000, (7, 5))
        }
        topology = GPUTopology(
            num_gpus=2,
            gpu_info=gpu_info,
            interconnect_bandwidth={(0, 1): 50.0},
            numa_topology={0: 0, 1: 0}
        )
        
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2
        )
        
        return config, topology
    
    def test_validate_config_valid(self):
        """测试有效配置验证"""
        config, topology = self.create_test_setup()
        
        result = ParallelConfigValidator.validate_config(config, topology)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_config_invalid(self):
        """测试无效配置验证"""
        config, topology = self.create_test_setup()
        
        # 配置需要更多GPU
        config.data_parallel_size = 4
        result = ParallelConfigValidator.validate_config(config, topology)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "配置需要4个GPU，但只有2个可用" in result["errors"][0]
    
    def test_validate_config_warnings(self):
        """测试配置警告"""
        # 创建内存不足的GPU
        gpu_info = {
            0: GPUInfo(0, "GPU 0", 8000, 2000, (7, 5))  # 只有2GB可用内存
        }
        topology = GPUTopology(
            num_gpus=1,
            gpu_info=gpu_info,
            interconnect_bandwidth={},
            numa_topology={0: 0}
        )
        
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=1,
            communication_backend=CommunicationBackend.NCCL
        )
        
        result = ParallelConfigValidator.validate_config(config, topology)
        assert len(result["warnings"]) >= 1
        assert any("可用内存" in warning for warning in result["warnings"])
    
    def test_optimize_config(self):
        """测试配置优化"""
        config, topology = self.create_test_setup()
        
        optimized = ParallelConfigValidator.optimize_config(config, topology)
        assert isinstance(optimized, ParallelConfig)
        assert optimized.data_parallel_size <= topology.num_gpus
    
    def test_estimate_memory_usage(self):
        """测试内存使用估算"""
        config = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2,
            enable_zero_optimization=True,
            zero_stage=ZeroStage.OPTIMIZER_GRADIENT
        )
        
        memory_estimate = ParallelConfigValidator.estimate_memory_usage(config, 7.0)  # 7GB模型
        
        assert "model_memory_mb" in memory_estimate
        assert "optimizer_memory_mb" in memory_estimate
        assert "activation_memory_mb" in memory_estimate
        assert "total_per_gpu_mb" in memory_estimate
        assert memory_estimate["model_memory_mb"] == 7168  # 7GB * 1024
        assert memory_estimate["total_per_gpu_mb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])