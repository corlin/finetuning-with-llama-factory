"""
梯度检查点和累积管理器测试

测试梯度检查点保存和恢复、梯度累积和内存优化策略、
激活值重计算机制和混合精度训练内存优化功能。
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.gradient_manager import (
    GradientManager, ActivationCheckpointing, GradientAccumulator, MixedPrecisionManager,
    GradientCheckpointConfig, GradientAccumulationConfig, MixedPrecisionConfig,
    GradientStatistics
)
from src.memory_manager import MemoryManager


class SimpleModel(nn.Module):
    """简单的测试模型"""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_size, 10))
        
        # 注意力层（用于测试检查点）
        self.attention = nn.MultiheadAttention(hidden_size, 8)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        
        # 注意力机制
        x = x.unsqueeze(0)  # 添加序列维度
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)
        
        # 输出层
        x = self.layers[-1](x)
        return x


class TestGradientCheckpointConfig:
    """测试梯度检查点配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = GradientCheckpointConfig()
        
        assert config.enabled is True
        assert "attention" in config.checkpoint_layers
        assert "mlp" in config.checkpoint_layers
        assert config.checkpoint_ratio == 0.5
        assert config.preserve_rng_state is True
        assert config.use_reentrant is False
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = GradientCheckpointConfig(
            enabled=False,
            checkpoint_layers=["custom_layer"],
            checkpoint_ratio=0.3
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["enabled"] is False
        assert config_dict["checkpoint_layers"] == ["custom_layer"]
        assert config_dict["checkpoint_ratio"] == 0.3


class TestGradientAccumulationConfig:
    """测试梯度累积配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = GradientAccumulationConfig()
        
        assert config.enabled is True
        assert config.accumulation_steps == 4
        assert config.sync_gradients is True
        assert config.adaptive_accumulation is True
        assert config.gradient_clipping is True
        assert config.max_grad_norm == 1.0
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = GradientAccumulationConfig(
            accumulation_steps=8,
            max_grad_norm=2.0
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["accumulation_steps"] == 8
        assert config_dict["max_grad_norm"] == 2.0


class TestMixedPrecisionConfig:
    """测试混合精度配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MixedPrecisionConfig()
        
        assert config.enabled is True
        assert config.dtype == "bfloat16"
        assert config.init_scale == 2.**16
        assert config.enabled_auto_scaling is True
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = MixedPrecisionConfig(
            dtype="float16",
            init_scale=1024.0
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["dtype"] == "float16"
        assert config_dict["init_scale"] == 1024.0


class TestGradientStatistics:
    """测试梯度统计"""
    
    def test_statistics_creation(self):
        """测试统计信息创建"""
        stats = GradientStatistics(
            step=100,
            total_norm=1.5,
            max_grad=0.8,
            min_grad=-0.3,
            num_zero_grads=10,
            num_inf_grads=0,
            num_nan_grads=0,
            memory_usage=2048
        )
        
        assert stats.step == 100
        assert stats.total_norm == 1.5
        assert stats.max_grad == 0.8
        assert stats.min_grad == -0.3
        assert stats.num_zero_grads == 10
        assert stats.memory_usage == 2048
    
    def test_statistics_to_dict(self):
        """测试统计信息转换为字典"""
        stats = GradientStatistics(
            step=50,
            total_norm=2.0,
            max_grad=1.0,
            min_grad=0.0,
            num_zero_grads=5,
            num_inf_grads=0,
            num_nan_grads=0,
            memory_usage=1024
        )
        
        stats_dict = stats.to_dict()
        
        assert isinstance(stats_dict, dict)
        assert stats_dict["step"] == 50
        assert stats_dict["total_norm"] == 2.0
        assert "timestamp" in stats_dict


class TestActivationCheckpointing:
    """测试激活值检查点"""
    
    def test_checkpointing_initialization(self):
        """测试检查点初始化"""
        config = GradientCheckpointConfig()
        checkpointing = ActivationCheckpointing(config)
        
        assert checkpointing.config == config
        assert len(checkpointing.checkpointed_modules) == 0
        assert len(checkpointing.original_forwards) == 0
    
    def test_should_checkpoint_layer(self):
        """测试层检查点判断"""
        config = GradientCheckpointConfig(checkpoint_layers=["attention", "mlp"])
        checkpointing = ActivationCheckpointing(config)
        
        # 测试注意力层
        attention_layer = nn.MultiheadAttention(128, 8)
        assert checkpointing._should_checkpoint_layer("attention", attention_layer) is True
        
        # 测试大型线性层
        large_linear = nn.Linear(10000, 10000)
        assert checkpointing._should_checkpoint_layer("large_linear", large_linear) is True
        
        # 测试小型线性层
        small_linear = nn.Linear(10, 10)
        assert checkpointing._should_checkpoint_layer("small_linear", small_linear) is False
    
    def test_enable_disable_checkpointing(self):
        """测试启用和禁用检查点"""
        config = GradientCheckpointConfig()
        checkpointing = ActivationCheckpointing(config)
        model = SimpleModel()
        
        # 启用检查点
        success = checkpointing.enable_checkpointing(model)
        assert success is True
        assert len(checkpointing.checkpointed_modules) > 0
        
        # 禁用检查点
        success = checkpointing.disable_checkpointing(model)
        assert success is True
        assert len(checkpointing.checkpointed_modules) == 0
    
    def test_temporary_checkpointing(self):
        """测试临时检查点上下文管理器"""
        config = GradientCheckpointConfig()
        checkpointing = ActivationCheckpointing(config)
        model = SimpleModel()
        
        # 初始状态
        assert len(checkpointing.checkpointed_modules) == 0
        
        # 使用临时检查点
        with checkpointing.temporary_checkpointing(model):
            assert len(checkpointing.checkpointed_modules) > 0
        
        # 退出后应该恢复
        assert len(checkpointing.checkpointed_modules) == 0


class TestGradientAccumulator:
    """测试梯度累积器"""
    
    def test_accumulator_initialization(self):
        """测试累积器初始化"""
        config = GradientAccumulationConfig(accumulation_steps=8)
        accumulator = GradientAccumulator(config)
        
        assert accumulator.config == config
        assert accumulator.current_step == 0
        assert accumulator.accumulated_steps == 0
        assert accumulator.current_accumulation_steps == 8
    
    def test_should_accumulate_gradients(self):
        """测试梯度累积判断"""
        config = GradientAccumulationConfig(accumulation_steps=4)
        accumulator = GradientAccumulator(config)
        
        # 前3步应该累积
        for i in range(3):
            accumulator.accumulated_steps = i
            assert accumulator.should_accumulate_gradients() is True
        
        # 第4步不应该累积
        accumulator.accumulated_steps = 3
        assert accumulator.should_accumulate_gradients() is False
    
    def test_should_sync_gradients(self):
        """测试梯度同步判断"""
        config = GradientAccumulationConfig(accumulation_steps=4, sync_gradients=True)
        accumulator = GradientAccumulator(config)
        
        # 前3步不应该同步
        for i in range(3):
            accumulator.accumulated_steps = i
            assert accumulator.should_sync_gradients() is False
        
        # 第4步应该同步
        accumulator.accumulated_steps = 3
        assert accumulator.should_sync_gradients() is True
    
    def test_accumulate_gradients(self):
        """测试梯度累积"""
        config = GradientAccumulationConfig(accumulation_steps=2)
        accumulator = GradientAccumulator(config)
        model = SimpleModel()
        
        # 创建损失
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # 累积梯度
        success = accumulator.accumulate_gradients(loss, model)
        assert success is True
        assert accumulator.accumulated_steps == 1
    
    def test_step_optimizer(self):
        """测试优化器步骤"""
        config = GradientAccumulationConfig(accumulation_steps=2)
        accumulator = GradientAccumulator(config)
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        # 第一步不应该执行优化器
        accumulator.accumulated_steps = 0
        success = accumulator.step_optimizer(optimizer)
        assert success is False
        
        # 第二步应该执行优化器
        accumulator.accumulated_steps = 1
        success = accumulator.step_optimizer(optimizer)
        assert success is True
        assert accumulator.accumulated_steps == 0
        assert accumulator.current_step == 1
    
    def test_gradient_statistics_collection(self):
        """测试梯度统计收集"""
        config = GradientAccumulationConfig()
        accumulator = GradientAccumulator(config)
        model = SimpleModel()
        
        # 创建一些梯度
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # 收集统计信息
        accumulator._collect_gradient_statistics(model)
        
        stats = accumulator.get_gradient_statistics()
        assert len(stats) == 1
        assert isinstance(stats[0], GradientStatistics)
        assert stats[0].total_norm > 0


class TestMixedPrecisionManager:
    """测试混合精度管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        config = MixedPrecisionConfig(enabled=True, dtype="bfloat16")
        manager = MixedPrecisionManager(config)
        
        assert manager.config == config
        assert manager.dtype == torch.bfloat16
        assert manager.scaler is not None
    
    def test_manager_initialization_disabled(self):
        """测试禁用混合精度的管理器初始化"""
        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)
        
        assert manager.config == config
        assert manager.scaler is None
    
    def test_autocast_context(self):
        """测试自动混合精度上下文"""
        config = MixedPrecisionConfig(enabled=True, dtype="float16")
        manager = MixedPrecisionManager(config)
        
        with manager.autocast_context():
            x = torch.randn(10, 10)
            # 在autocast上下文中，操作应该使用混合精度
            assert x.dtype == torch.float32  # 输入仍然是float32
    
    def test_scale_loss(self):
        """测试损失缩放"""
        config = MixedPrecisionConfig(enabled=True)
        manager = MixedPrecisionManager(config)
        
        loss = torch.tensor(1.0)
        scaled_loss = manager.scale_loss(loss)
        
        # 缩放后的损失应该不同
        assert scaled_loss != loss
    
    def test_get_scale(self):
        """测试获取缩放因子"""
        config = MixedPrecisionConfig(enabled=True, init_scale=1024.0)
        manager = MixedPrecisionManager(config)
        
        scale = manager.get_scale()
        assert scale == 1024.0
    
    def test_state_dict_operations(self):
        """测试状态字典操作"""
        config = MixedPrecisionConfig(enabled=True)
        manager = MixedPrecisionManager(config)
        
        # 获取状态字典
        state_dict = manager.state_dict()
        assert isinstance(state_dict, dict)
        
        # 加载状态字典
        manager.load_state_dict(state_dict)  # 应该不抛出异常


class TestGradientManager:
    """测试梯度管理器主类"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = GradientManager()
        
        assert manager.checkpoint_config is not None
        assert manager.accumulation_config is not None
        assert manager.mixed_precision_config is not None
        assert manager.is_initialized is False
    
    def test_manager_initialization_with_configs(self):
        """测试使用自定义配置初始化管理器"""
        checkpoint_config = GradientCheckpointConfig(enabled=False)
        accumulation_config = GradientAccumulationConfig(accumulation_steps=8)
        mixed_precision_config = MixedPrecisionConfig(dtype="float16")
        
        manager = GradientManager(
            checkpoint_config=checkpoint_config,
            accumulation_config=accumulation_config,
            mixed_precision_config=mixed_precision_config
        )
        
        assert manager.checkpoint_config == checkpoint_config
        assert manager.accumulation_config == accumulation_config
        assert manager.mixed_precision_config == mixed_precision_config
    
    def test_initialize_manager(self):
        """测试初始化管理器"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        success = manager.initialize(model, optimizer)
        
        assert success is True
        assert manager.is_initialized is True
        assert manager.model == model
        assert manager.optimizer == optimizer
    
    def test_training_step(self):
        """测试训练步骤"""
        config = GradientAccumulationConfig(accumulation_steps=2)
        manager = GradientManager(accumulation_config=config)
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        # 初始化管理器
        manager.initialize(model, optimizer)
        
        # 创建损失
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # 执行训练步骤
        success = manager.training_step(loss)
        assert success is True
    
    def test_autocast_context(self):
        """测试混合精度上下文"""
        manager = GradientManager()
        
        with manager.autocast_context():
            x = torch.randn(10, 10)
            # 应该不抛出异常
            assert x is not None
    
    def test_get_gradient_statistics(self):
        """测试获取梯度统计"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        manager.initialize(model, optimizer)
        
        # 执行一些训练步骤生成统计数据
        for _ in range(3):
            x = torch.randn(32, 128)
            y = torch.randint(0, 10, (32,))
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            manager.training_step(loss)
        
        stats = manager.get_gradient_statistics()
        assert isinstance(stats, list)
    
    def test_save_load_checkpoint(self):
        """测试保存和加载检查点"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        manager.initialize(model, optimizer)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "gradient_checkpoint.pkl"
            
            # 保存检查点
            success = manager.save_checkpoint(str(checkpoint_path))
            assert success is True
            assert checkpoint_path.exists()
            
            # 加载检查点
            success = manager.load_checkpoint(str(checkpoint_path))
            assert success is True
    
    def test_optimize_memory_usage(self):
        """测试内存使用优化"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        manager.initialize(model, optimizer)
        
        optimizations = manager.optimize_memory_usage()
        
        assert isinstance(optimizations, dict)
        assert "gradient_checkpointing" in optimizations
        assert "increased_accumulation" in optimizations
        assert "cache_cleared" in optimizations
        assert "memory_saved_mb" in optimizations
    
    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        manager.initialize(model, optimizer)
        
        metrics = manager.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "gradient_accumulation_steps" in metrics
        assert "current_step" in metrics
        assert "gradient_checkpointing_enabled" in metrics
        assert "mixed_precision_enabled" in metrics
    
    def test_cleanup(self):
        """测试资源清理"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        manager.initialize(model, optimizer)
        assert manager.is_initialized is True
        
        manager.cleanup()
        assert manager.is_initialized is False
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with GradientManager() as manager:
            model = SimpleModel()
            optimizer = optim.Adam(model.parameters())
            manager.initialize(model, optimizer)
            assert manager.is_initialized is True
        
        # 退出上下文后应该自动清理
        assert manager.is_initialized is False


class TestGradientManagerIntegration:
    """梯度管理器集成测试"""
    
    def test_full_training_loop(self):
        """测试完整训练循环"""
        # 配置
        checkpoint_config = GradientCheckpointConfig(enabled=True)
        accumulation_config = GradientAccumulationConfig(accumulation_steps=4)
        mixed_precision_config = MixedPrecisionConfig(enabled=True, dtype="bfloat16")
        
        # 创建管理器
        manager = GradientManager(
            checkpoint_config=checkpoint_config,
            accumulation_config=accumulation_config,
            mixed_precision_config=mixed_precision_config
        )
        
        # 创建模型和优化器
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 初始化
        success = manager.initialize(model, optimizer)
        assert success is True
        
        # 模拟训练循环
        num_steps = 10
        for step in range(num_steps):
            # 创建批次数据
            x = torch.randn(16, 128)
            y = torch.randint(0, 10, (16,))
            
            # 前向传播（使用混合精度）
            with manager.autocast_context():
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)
            
            # 训练步骤
            success = manager.training_step(loss)
            assert success is True
        
        # 检查统计信息
        stats = manager.get_gradient_statistics()
        assert len(stats) > 0
        
        # 检查性能指标
        metrics = manager.get_performance_metrics()
        assert metrics["current_step"] > 0
        
        # 清理
        manager.cleanup()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_optimization_integration(self):
        """测试内存优化集成"""
        # 创建内存管理器
        memory_manager = MemoryManager({"monitoring_interval": 1})
        
        # 创建梯度管理器
        accumulation_config = GradientAccumulationConfig(
            adaptive_accumulation=True,
            min_accumulation_steps=1,
            max_accumulation_steps=8
        )
        
        gradient_manager = GradientManager(
            accumulation_config=accumulation_config
        )
        
        # 设置内存管理器
        gradient_manager.gradient_accumulator.memory_manager = memory_manager
        
        # 创建模型
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        # 初始化
        gradient_manager.initialize(model, optimizer)
        memory_manager.start()
        
        try:
            # 执行一些训练步骤
            for _ in range(5):
                x = torch.randn(32, 128)
                y = torch.randint(0, 10, (32,))
                
                with gradient_manager.autocast_context():
                    output = model(x)
                    loss = nn.CrossEntropyLoss()(output, y)
                
                gradient_manager.training_step(loss)
            
            # 检查是否有自适应调整
            metrics = gradient_manager.get_performance_metrics()
            assert "gradient_accumulation_steps" in metrics
            
        finally:
            memory_manager.stop()
            gradient_manager.cleanup()
    
    def test_checkpoint_persistence(self):
        """测试检查点持久化"""
        manager = GradientManager()
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        manager.initialize(model, optimizer)
        
        # 执行一些训练步骤
        for _ in range(5):
            x = torch.randn(16, 128)
            y = torch.randint(0, 10, (16,))
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            manager.training_step(loss)
        
        # 获取训练前的状态
        metrics_before = manager.get_performance_metrics()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pkl"
            
            # 保存检查点
            success = manager.save_checkpoint(str(checkpoint_path))
            assert success is True
            
            # 创建新的管理器并加载检查点
            new_manager = GradientManager()
            new_model = SimpleModel()
            new_optimizer = optim.Adam(new_model.parameters())
            
            new_manager.initialize(new_model, new_optimizer)
            success = new_manager.load_checkpoint(str(checkpoint_path))
            assert success is True
            
            # 验证状态恢复
            metrics_after = new_manager.get_performance_metrics()
            assert metrics_after["current_step"] == metrics_before["current_step"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])