"""
OOM预防和恢复管理器测试

测试OOM检测和自动恢复机制、训练参数自动调整策略、
内存溢出预警和处理流程、训练状态保存和恢复机制。
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.oom_manager import (
    OOMManager, OOMDetector, TrainingStateManager, OOMRecoveryManager,
    OOMPreventionConfig, OOMEvent, TrainingState, OOMSeverity, RecoveryStrategy
)
from src.memory_manager import MemoryManager, MemorySnapshot, MemoryPressureLevel
from src.gradient_manager import GradientManager


class SimpleTestModel(nn.Module):
    """简单的测试模型"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestOOMPreventionConfig:
    """测试OOM预防配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = OOMPreventionConfig()
        
        assert config.enabled is True
        assert config.warning_threshold == 0.8
        assert config.critical_threshold == 0.9
        assert config.auto_adjust_batch_size is True
        assert config.min_batch_size == 1
        assert config.max_batch_size == 64
        assert config.enable_auto_checkpointing is True
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = OOMPreventionConfig(
            warning_threshold=0.75,
            critical_threshold=0.85,
            min_batch_size=2
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["warning_threshold"] == 0.75
        assert config_dict["critical_threshold"] == 0.85
        assert config_dict["min_batch_size"] == 2


class TestOOMEvent:
    """测试OOM事件"""
    
    def test_oom_event_creation(self):
        """测试OOM事件创建"""
        event = OOMEvent(
            timestamp=datetime.now(),
            severity=OOMSeverity.MODERATE,
            error_message="CUDA out of memory",
            memory_usage_mb=8192,
            available_memory_mb=2048,
            batch_size=16,
            sequence_length=1024,
            gradient_accumulation_steps=4
        )
        
        assert event.severity == OOMSeverity.MODERATE
        assert event.error_message == "CUDA out of memory"
        assert event.memory_usage_mb == 8192
        assert event.batch_size == 16
        assert event.recovery_successful is False
    
    def test_oom_event_to_dict(self):
        """测试OOM事件转换为字典"""
        event = OOMEvent(
            timestamp=datetime.now(),
            severity=OOMSeverity.SEVERE,
            error_message="Out of memory error",
            memory_usage_mb=12000,
            available_memory_mb=1000,
            batch_size=32,
            sequence_length=2048,
            gradient_accumulation_steps=2,
            recovery_strategies_applied=[RecoveryStrategy.REDUCE_BATCH_SIZE],
            recovery_successful=True,
            recovery_time_seconds=5.5
        )
        
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert event_dict["severity"] == "severe"
        assert event_dict["memory_usage_mb"] == 12000
        assert event_dict["recovery_successful"] is True
        assert event_dict["recovery_time_seconds"] == 5.5
        assert "reduce_batch_size" in event_dict["recovery_strategies_applied"]


class TestTrainingState:
    """测试训练状态"""
    
    def test_training_state_creation(self):
        """测试训练状态创建"""
        state = TrainingState(
            epoch=5,
            step=1000,
            batch_size=16,
            sequence_length=1024,
            gradient_accumulation_steps=4,
            learning_rate=0.001
        )
        
        assert state.epoch == 5
        assert state.step == 1000
        assert state.batch_size == 16
        assert state.sequence_length == 1024
        assert state.gradient_accumulation_steps == 4
        assert state.learning_rate == 0.001
        assert state.model_state_dict is None
    
    def test_training_state_to_dict(self):
        """测试训练状态转换为字典"""
        state = TrainingState(
            epoch=3,
            step=500,
            batch_size=8,
            sequence_length=512,
            gradient_accumulation_steps=2,
            learning_rate=0.0001,
            model_state_dict={"layer.weight": torch.randn(10, 10)},
            optimizer_state_dict={"state": {}}
        )
        
        state_dict = state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict["epoch"] == 3
        assert state_dict["step"] == 500
        assert state_dict["has_model_state"] is True
        assert state_dict["has_optimizer_state"] is True
        assert "timestamp" in state_dict


class TestOOMDetector:
    """测试OOM检测器"""
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        config = OOMPreventionConfig()
        detector = OOMDetector(config)
        
        assert detector.config == config
        assert detector.is_monitoring is False
        assert len(detector.warning_callbacks) == 0
        assert len(detector.critical_callbacks) == 0
    
    def test_detect_oom_from_exception(self):
        """测试从异常检测OOM"""
        config = OOMPreventionConfig()
        detector = OOMDetector(config)
        
        # CUDA OOM异常 (2GB > 1GB, 应该是SEVERE)
        cuda_oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        severity = detector.detect_oom_from_exception(cuda_oom)
        assert severity == OOMSeverity.SEVERE
        
        # 严重OOM异常
        severe_oom = RuntimeError("CUDA error: out of memory")
        severity = detector.detect_oom_from_exception(severe_oom)
        assert severity == OOMSeverity.CRITICAL
        
        # 中等OOM异常
        moderate_oom = RuntimeError("CUDA out of memory. Tried to allocate 600.00 MiB")
        severity = detector.detect_oom_from_exception(moderate_oom)
        assert severity == OOMSeverity.MODERATE
        
        # 非OOM异常
        other_error = ValueError("Invalid input")
        severity = detector.detect_oom_from_exception(other_error)
        assert severity is None
    
    def test_add_callbacks(self):
        """测试添加回调函数"""
        config = OOMPreventionConfig()
        detector = OOMDetector(config)
        
        warning_callback = Mock()
        critical_callback = Mock()
        
        detector.add_warning_callback(warning_callback)
        detector.add_critical_callback(critical_callback)
        
        assert len(detector.warning_callbacks) == 1
        assert len(detector.critical_callbacks) == 1
        assert warning_callback in detector.warning_callbacks
        assert critical_callback in detector.critical_callbacks
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        config = OOMPreventionConfig(monitoring_interval=0.1)
        detector = OOMDetector(config)
        
        # 启动监控
        success = detector.start_monitoring()
        assert success is True
        assert detector.is_monitoring is True
        
        # 等待一段时间让监控运行
        time.sleep(0.5)
        
        # 停止监控
        success = detector.stop_monitoring()
        assert success is True
        assert detector.is_monitoring is False
    
    def test_memory_growth_tracking(self):
        """测试内存增长跟踪"""
        config = OOMPreventionConfig()
        detector = OOMDetector(config)
        
        # 模拟内存增长
        detector._track_memory_growth(1000)
        detector._track_memory_growth(1100)
        detector._track_memory_growth(1200)
        
        assert len(detector.memory_growth_history) == 3
        
        # 测试增长率计算
        growth_rate = detector.get_memory_growth_rate()
        assert isinstance(growth_rate, float)


class TestTrainingStateManager:
    """测试训练状态管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingStateManager(temp_dir)
            
            assert manager.checkpoint_dir == Path(temp_dir)
            assert manager.checkpoint_dir.exists()
            assert len(manager.state_history) == 0
    
    def test_save_load_training_state(self):
        """测试保存和加载训练状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingStateManager(temp_dir)
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            
            # 保存训练状态
            checkpoint_path = manager.save_training_state(
                epoch=2,
                step=100,
                batch_size=8,
                sequence_length=512,
                gradient_accumulation_steps=2,
                learning_rate=0.001,
                model=model,
                optimizer=optimizer
            )
            
            assert Path(checkpoint_path).exists()
            assert len(manager.state_history) == 1
            
            # 加载训练状态
            loaded_state = manager.load_training_state(checkpoint_path)
            
            assert loaded_state is not None
            assert loaded_state.epoch == 2
            assert loaded_state.step == 100
            assert loaded_state.batch_size == 8
            assert loaded_state.model_state_dict is not None
            assert loaded_state.optimizer_state_dict is not None
    
    def test_restore_training_state(self):
        """测试恢复训练状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingStateManager(temp_dir)
            
            # 创建原始模型和优化器
            original_model = SimpleTestModel()
            original_optimizer = optim.Adam(original_model.parameters(), lr=0.001)
            
            # 保存状态
            checkpoint_path = manager.save_training_state(
                epoch=1,
                step=50,
                batch_size=4,
                sequence_length=256,
                gradient_accumulation_steps=1,
                learning_rate=0.001,
                model=original_model,
                optimizer=original_optimizer
            )
            
            # 创建新的模型和优化器
            new_model = SimpleTestModel()
            new_optimizer = optim.Adam(new_model.parameters(), lr=0.002)
            
            # 加载并恢复状态
            loaded_state = manager.load_training_state(checkpoint_path)
            success = manager.restore_training_state(
                loaded_state, new_model, new_optimizer
            )
            
            assert success is True
            
            # 验证学习率是否恢复
            assert new_optimizer.param_groups[0]['lr'] == 0.001
    
    def test_get_latest_checkpoint(self):
        """测试获取最新检查点"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingStateManager(temp_dir)
            
            # 没有检查点时
            latest = manager.get_latest_checkpoint()
            assert latest is None
            
            # 创建多个检查点
            checkpoint1 = manager.save_training_state(1, 10, 4, 512, 1, 0.001)
            time.sleep(0.1)  # 确保时间戳不同
            checkpoint2 = manager.save_training_state(1, 20, 4, 512, 1, 0.001)
            
            # 获取最新检查点
            latest = manager.get_latest_checkpoint()
            assert latest == checkpoint2
    
    def test_cleanup_old_checkpoints(self):
        """测试清理旧检查点"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingStateManager(temp_dir)
            
            # 创建多个检查点
            checkpoints = []
            for i in range(5):
                checkpoint = manager.save_training_state(1, i*10, 4, 512, 1, 0.001)
                checkpoints.append(checkpoint)
                time.sleep(0.1)
            
            # 清理，只保留3个
            manager.cleanup_old_checkpoints(max_files=3)
            
            # 检查文件数量
            remaining_files = list(manager.checkpoint_dir.glob("training_state_*.pkl"))
            assert len(remaining_files) == 3
            
            # 检查最新的3个文件仍然存在
            for checkpoint in checkpoints[-3:]:
                assert Path(checkpoint).exists()


class TestOOMRecoveryManager:
    """测试OOM恢复管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        assert manager.config == config
        assert len(manager.oom_events) == 0
        assert manager.current_batch_size == config.max_batch_size
        assert manager.current_sequence_length == config.max_sequence_length
    
    def test_detect_oom_severity(self):
        """测试OOM严重程度检测"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        # 测试不同类型的异常
        critical_error = RuntimeError("CUDA error: out of memory")
        assert manager._detect_oom_severity(critical_error) == OOMSeverity.CRITICAL
        
        severe_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert manager._detect_oom_severity(severe_error) == OOMSeverity.SEVERE
        
        moderate_error = RuntimeError("CUDA out of memory")
        assert manager._detect_oom_severity(moderate_error) == OOMSeverity.MODERATE
        
        non_oom_error = ValueError("Invalid input")
        assert manager._detect_oom_severity(non_oom_error) is None
    
    def test_select_recovery_strategies(self):
        """测试恢复策略选择"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        # 创建OOM事件
        oom_event = OOMEvent(
            timestamp=datetime.now(),
            severity=OOMSeverity.SEVERE,
            error_message="CUDA out of memory",
            memory_usage_mb=8000,
            available_memory_mb=1000,
            batch_size=16,
            sequence_length=1024,
            gradient_accumulation_steps=4
        )
        
        strategies = manager._select_recovery_strategies(OOMSeverity.SEVERE, oom_event)
        
        assert len(strategies) > 0
        assert RecoveryStrategy.CLEAR_CACHE in strategies
        assert RecoveryStrategy.REDUCE_BATCH_SIZE in strategies
    
    def test_execute_single_strategy(self):
        """测试执行单个恢复策略"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        oom_event = OOMEvent(
            timestamp=datetime.now(),
            severity=OOMSeverity.MODERATE,
            error_message="CUDA out of memory",
            memory_usage_mb=8000,
            available_memory_mb=1000,
            batch_size=16,
            sequence_length=1024,
            gradient_accumulation_steps=4
        )
        
        # 测试清理缓存策略
        success = manager._execute_single_strategy(RecoveryStrategy.CLEAR_CACHE, oom_event)
        assert success is True
        
        # 测试减小批次大小策略
        original_batch_size = manager.current_batch_size
        success = manager._execute_single_strategy(RecoveryStrategy.REDUCE_BATCH_SIZE, oom_event)
        assert success is True
        assert manager.current_batch_size < original_batch_size
    
    def test_handle_oom_event(self):
        """测试处理OOM事件"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        # 创建OOM异常
        oom_exception = RuntimeError("CUDA out of memory")
        
        # 处理OOM事件
        recovery_successful, strategies = manager.handle_oom_event(
            oom_exception, batch_size=16, sequence_length=1024, gradient_accumulation_steps=4
        )
        
        assert isinstance(recovery_successful, bool)
        assert isinstance(strategies, list)
        assert len(manager.oom_events) == 1
        
        # 检查事件记录
        event = manager.oom_events[0]
        assert event.error_message == str(oom_exception)
        assert event.batch_size == 16
        assert event.sequence_length == 1024
    
    def test_get_recommended_parameters(self):
        """测试获取推荐参数"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        params = manager.get_recommended_parameters()
        
        assert isinstance(params, dict)
        assert "batch_size" in params
        assert "sequence_length" in params
        assert "gradient_accumulation_steps" in params
    
    def test_get_oom_statistics(self):
        """测试获取OOM统计"""
        config = OOMPreventionConfig()
        manager = OOMRecoveryManager(config)
        
        # 没有事件时
        stats = manager.get_oom_statistics()
        assert stats["total_events"] == 0
        
        # 添加一些事件
        for i in range(3):
            oom_exception = RuntimeError(f"CUDA out of memory {i}")
            manager.handle_oom_event(oom_exception, 16, 1024, 4)
        
        stats = manager.get_oom_statistics()
        assert stats["total_events"] == 3
        assert "recovery_success_rate" in stats
        assert "severity_distribution" in stats


class TestOOMManager:
    """测试OOM管理器主类"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = OOMManager()
        
        assert manager.config is not None
        assert manager.detector is not None
        assert manager.state_manager is not None
        assert manager.recovery_manager is not None
        assert manager.is_active is False
    
    def test_manager_initialization_with_components(self):
        """测试使用外部组件初始化管理器"""
        config = OOMPreventionConfig(warning_threshold=0.75)
        memory_manager = Mock()
        gradient_manager = Mock()
        
        manager = OOMManager(
            config=config,
            memory_manager=memory_manager,
            gradient_manager=gradient_manager
        )
        
        assert manager.config == config
        assert manager.memory_manager == memory_manager
        assert manager.gradient_manager == gradient_manager
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_start_stop_manager(self):
        """测试启动和停止管理器"""
        config = OOMPreventionConfig(monitoring_interval=0.1)
        manager = OOMManager(config)
        
        # 启动管理器
        success = manager.start()
        assert success is True
        assert manager.is_active is True
        
        # 等待一段时间
        time.sleep(0.5)
        
        # 停止管理器
        success = manager.stop()
        assert success is True
        assert manager.is_active is False
    
    def test_disabled_manager(self):
        """测试禁用的管理器"""
        config = OOMPreventionConfig(enabled=False)
        manager = OOMManager(config)
        
        success = manager.start()
        assert success is False
        assert manager.is_active is False
    
    def test_is_oom_error(self):
        """测试OOM错误判断"""
        manager = OOMManager()
        
        # OOM错误
        oom_error = RuntimeError("CUDA out of memory")
        assert manager._is_oom_error(oom_error) is True
        
        # 非OOM错误
        other_error = ValueError("Invalid input")
        assert manager._is_oom_error(other_error) is False
    
    def test_save_load_checkpoint(self):
        """测试保存和加载检查点"""
        manager = OOMManager()
        model = SimpleTestModel()
        optimizer = optim.Adam(model.parameters())
        
        # 保存检查点
        checkpoint_path = manager.save_checkpoint(
            epoch=1,
            step=100,
            batch_size=8,
            sequence_length=512,
            gradient_accumulation_steps=2,
            learning_rate=0.001,
            model=model,
            optimizer=optimizer
        )
        
        assert Path(checkpoint_path).exists()
        
        # 加载最新检查点
        state = manager.load_latest_checkpoint()
        assert state is not None
        assert state.epoch == 1
        assert state.step == 100
    
    def test_restore_from_checkpoint(self):
        """测试从检查点恢复"""
        manager = OOMManager()
        model = SimpleTestModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 保存检查点
        checkpoint_path = manager.save_checkpoint(
            epoch=2,
            step=200,
            batch_size=16,
            sequence_length=1024,
            gradient_accumulation_steps=4,
            learning_rate=0.0005,
            model=model,
            optimizer=optimizer
        )
        
        # 创建新的模型和优化器
        new_model = SimpleTestModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.002)
        
        # 从检查点恢复
        restored_state = manager.restore_from_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        assert restored_state is not None
        assert restored_state.epoch == 2
        assert restored_state.step == 200
        assert new_optimizer.param_groups[0]['lr'] == 0.0005
    
    def test_get_recommended_parameters(self):
        """测试获取推荐参数"""
        manager = OOMManager()
        
        params = manager.get_recommended_parameters()
        
        assert isinstance(params, dict)
        assert "batch_size" in params
        assert "sequence_length" in params
        assert "gradient_accumulation_steps" in params
    
    def test_get_oom_statistics(self):
        """测试获取OOM统计"""
        manager = OOMManager()
        
        stats = manager.get_oom_statistics()
        
        assert isinstance(stats, dict)
        assert "total_events" in stats
        assert "memory_growth_rate_mb_per_min" in stats
        assert "is_monitoring" in stats
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        manager = OOMManager()
        
        status = manager.get_system_status()
        
        assert isinstance(status, dict)
        assert "oom_manager_active" in status
        assert "config" in status
        assert "system_memory" in status
        assert "last_checkpoint_step" in status
    
    def test_oom_safe_training_step_normal(self):
        """测试正常情况下的OOM安全训练步骤"""
        manager = OOMManager()
        model = SimpleTestModel()
        optimizer = optim.Adam(model.parameters())
        
        # 正常训练步骤
        with manager.oom_safe_training_step(
            epoch=1, step=10, batch_size=4, sequence_length=512,
            gradient_accumulation_steps=2, learning_rate=0.001,
            model=model, optimizer=optimizer
        ):
            # 模拟正常训练操作
            x = torch.randn(4, 100)
            y = model(x)
            loss = y.sum()
            loss.backward()
    
    def test_oom_safe_training_step_with_oom(self):
        """测试OOM情况下的训练步骤"""
        manager = OOMManager()
        model = SimpleTestModel()
        optimizer = optim.Adam(model.parameters())
        
        # 模拟OOM错误
        with pytest.raises(RuntimeError):
            with manager.oom_safe_training_step(
                epoch=1, step=10, batch_size=4, sequence_length=512,
                gradient_accumulation_steps=2, learning_rate=0.001,
                model=model, optimizer=optimizer
            ):
                # 抛出OOM异常
                raise RuntimeError("CUDA out of memory")
        
        # 检查是否记录了OOM事件
        stats = manager.get_oom_statistics()
        assert stats["total_events"] > 0
    
    def test_context_manager(self):
        """测试上下文管理器"""
        config = OOMPreventionConfig(enabled=False)  # 禁用以避免实际监控
        
        with OOMManager(config) as manager:
            assert manager is not None
        
        # 退出上下文后应该自动清理
        # 由于禁用了监控，这里主要测试上下文管理器的结构


class TestOOMManagerIntegration:
    """OOM管理器集成测试"""
    
    def test_memory_warning_handling(self):
        """测试内存警告处理"""
        memory_manager = Mock()
        manager = OOMManager(memory_manager=memory_manager)
        
        # 创建内存快照
        memory_snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=13000,  # 80%使用率
            cached_memory=1024,
            free_memory=3384,
            utilization_rate=0.8,
            pressure_level=MemoryPressureLevel.HIGH,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        # 触发内存警告处理
        manager._handle_memory_warning(memory_snapshot)
        
        # 验证内存管理器的优化方法被调用
        memory_manager.optimize_memory.assert_called_once()
    
    def test_memory_critical_handling(self):
        """测试内存紧急情况处理"""
        gradient_manager = Mock()
        manager = OOMManager(gradient_manager=gradient_manager)
        
        # 创建紧急内存快照
        memory_snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=15000,  # 91%使用率
            cached_memory=1024,
            free_memory=1384,
            utilization_rate=0.91,
            pressure_level=MemoryPressureLevel.CRITICAL,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        # 触发紧急情况处理
        manager._handle_memory_critical(memory_snapshot)
        
        # 验证梯度管理器的优化方法被调用
        gradient_manager.optimize_memory_usage.assert_called_once()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_checkpointing(self):
        """测试自动检查点功能"""
        config = OOMPreventionConfig(
            enable_auto_checkpointing=True,
            checkpoint_interval_steps=5
        )
        manager = OOMManager(config)
        model = SimpleTestModel()
        optimizer = optim.Adam(model.parameters())
        
        # 模拟多个训练步骤
        for step in range(10):
            with manager.oom_safe_training_step(
                epoch=1, step=step, batch_size=4, sequence_length=512,
                gradient_accumulation_steps=2, learning_rate=0.001,
                model=model, optimizer=optimizer
            ):
                # 模拟训练操作
                x = torch.randn(4, 100)
                y = model(x)
                loss = y.sum()
                loss.backward()
        
        # 检查是否创建了检查点
        latest_checkpoint = manager.load_latest_checkpoint()
        assert latest_checkpoint is not None
    
    def test_recovery_strategy_effectiveness(self):
        """测试恢复策略有效性"""
        config = OOMPreventionConfig()
        manager = OOMManager(config)
        
        # 模拟多个OOM事件
        oom_exceptions = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("Out of memory")
        ]
        
        for i, exception in enumerate(oom_exceptions):
            recovery_successful, strategies = manager.recovery_manager.handle_oom_event(
                exception, batch_size=16-i*2, sequence_length=1024, gradient_accumulation_steps=4
            )
            
            # 验证恢复策略被应用
            assert len(strategies) > 0
            assert isinstance(recovery_successful, bool)
        
        # 检查统计信息
        stats = manager.get_oom_statistics()
        assert stats["total_events"] == 3
        assert "strategy_success_rates" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])