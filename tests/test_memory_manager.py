"""
动态内存管理器测试

测试内存监控、动态批次大小调整、内存压力检测和响应机制的功能。
"""

import pytest
import torch
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from src.memory_manager import (
    MemoryManager, MemoryMonitor, DynamicBatchSizeAdjuster, MemoryPredictor,
    MemorySnapshot, MemoryPressureLevel, OptimizationStrategy,
    MemoryOptimizationRecommendation, BatchSizeAdjustment, MemoryPrediction
)


class TestMemorySnapshot:
    """测试内存快照"""
    
    def test_memory_snapshot_creation(self):
        """测试内存快照创建"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=8192,
            cached_memory=1024,
            free_memory=8192,
            utilization_rate=0.5,
            pressure_level=MemoryPressureLevel.MODERATE,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        assert snapshot.gpu_id == 0
        assert snapshot.total_memory == 16384
        assert snapshot.utilization_rate == 0.5
        assert snapshot.pressure_level == MemoryPressureLevel.MODERATE
    
    def test_memory_snapshot_to_dict(self):
        """测试内存快照转换为字典"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=8192,
            cached_memory=1024,
            free_memory=8192,
            utilization_rate=0.5,
            pressure_level=MemoryPressureLevel.MODERATE,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        snapshot_dict = snapshot.to_dict()
        
        assert isinstance(snapshot_dict, dict)
        assert snapshot_dict["gpu_id"] == 0
        assert snapshot_dict["total_memory"] == 16384
        assert snapshot_dict["pressure_level"] == "moderate"
        assert "timestamp" in snapshot_dict


class TestMemoryOptimizationRecommendation:
    """测试内存优化建议"""
    
    def test_recommendation_creation(self):
        """测试优化建议创建"""
        recommendation = MemoryOptimizationRecommendation(
            strategy=OptimizationStrategy.REDUCE_BATCH_SIZE,
            priority=8,
            description="减小批次大小以降低内存使用",
            expected_memory_saving=2048,
            implementation_difficulty=1,
            side_effects=["训练时间增加"]
        )
        
        assert recommendation.strategy == OptimizationStrategy.REDUCE_BATCH_SIZE
        assert recommendation.priority == 8
        assert recommendation.expected_memory_saving == 2048
        assert len(recommendation.side_effects) == 1
    
    def test_recommendation_to_dict(self):
        """测试优化建议转换为字典"""
        recommendation = MemoryOptimizationRecommendation(
            strategy=OptimizationStrategy.ENABLE_GRADIENT_CHECKPOINTING,
            priority=7,
            description="启用梯度检查点",
            expected_memory_saving=4096,
            implementation_difficulty=2
        )
        
        rec_dict = recommendation.to_dict()
        
        assert isinstance(rec_dict, dict)
        assert rec_dict["strategy"] == "enable_gradient_checkpointing"
        assert rec_dict["priority"] == 7
        assert rec_dict["expected_memory_saving"] == 4096


class TestDynamicBatchSizeAdjuster:
    """测试动态批次大小调整器"""
    
    def test_adjuster_initialization(self):
        """测试调整器初始化"""
        adjuster = DynamicBatchSizeAdjuster(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16
        )
        
        assert adjuster.initial_batch_size == 4
        assert adjuster.min_batch_size == 1
        assert adjuster.max_batch_size == 16
        assert adjuster.current_batch_size == 4
    
    def test_should_adjust_batch_size_high_memory(self):
        """测试高内存使用率时的批次调整判断"""
        adjuster = DynamicBatchSizeAdjuster(initial_batch_size=8, min_batch_size=1)
        
        # 创建高内存使用率快照
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=14000,  # 85%+ 使用率
            cached_memory=1024,
            free_memory=2384,
            utilization_rate=0.87,
            pressure_level=MemoryPressureLevel.HIGH,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        should_adjust, reason = adjuster.should_adjust_batch_size(snapshot)
        
        assert should_adjust is True
        assert "内存使用率过高" in reason
    
    def test_should_adjust_batch_size_low_memory(self):
        """测试低内存使用率时的批次调整判断"""
        adjuster = DynamicBatchSizeAdjuster(initial_batch_size=2, max_batch_size=16)
        
        # 创建低内存使用率快照
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=8192,  # 50% 使用率
            cached_memory=1024,
            free_memory=8192,
            utilization_rate=0.5,
            pressure_level=MemoryPressureLevel.LOW,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        should_adjust, reason = adjuster.should_adjust_batch_size(snapshot)
        
        assert should_adjust is True
        assert "内存使用率较低" in reason
    
    def test_adjust_batch_size_reduce(self):
        """测试减小批次大小"""
        adjuster = DynamicBatchSizeAdjuster(initial_batch_size=8, min_batch_size=1)
        
        # 高内存使用率快照
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=14000,
            cached_memory=1024,
            free_memory=2384,
            utilization_rate=0.87,
            pressure_level=MemoryPressureLevel.HIGH,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        adjustment = adjuster.adjust_batch_size(snapshot)
        
        assert adjustment is not None
        assert adjustment.old_batch_size == 8
        assert adjustment.new_batch_size < 8
        assert adjustment.success is True
        assert adjuster.current_batch_size < 8
    
    def test_adjust_batch_size_increase(self):
        """测试增大批次大小"""
        adjuster = DynamicBatchSizeAdjuster(initial_batch_size=2, max_batch_size=16)
        
        # 低内存使用率快照
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=8192,
            cached_memory=1024,
            free_memory=8192,
            utilization_rate=0.5,
            pressure_level=MemoryPressureLevel.LOW,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        adjustment = adjuster.adjust_batch_size(snapshot)
        
        assert adjustment is not None
        assert adjustment.old_batch_size == 2
        assert adjustment.new_batch_size > 2
        assert adjustment.success is True
        assert adjuster.current_batch_size > 2
    
    def test_cooldown_period(self):
        """测试冷却期机制"""
        adjuster = DynamicBatchSizeAdjuster(initial_batch_size=4)
        adjuster.cooldown_period = 1  # 1秒冷却期
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=14000,
            cached_memory=1024,
            free_memory=2384,
            utilization_rate=0.87,
            pressure_level=MemoryPressureLevel.HIGH,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        # 第一次调整
        adjustment1 = adjuster.adjust_batch_size(snapshot)
        assert adjustment1 is not None
        
        # 立即再次调整（应该被冷却期阻止）
        adjustment2 = adjuster.adjust_batch_size(snapshot)
        assert adjustment2 is None
        
        # 等待冷却期结束
        time.sleep(1.1)
        adjustment3 = adjuster.adjust_batch_size(snapshot)
        assert adjustment3 is not None


class TestMemoryPredictor:
    """测试内存使用预测器"""
    
    def test_predictor_initialization(self):
        """测试预测器初始化"""
        predictor = MemoryPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'prediction_models')
    
    def test_predict_memory_usage_no_history(self):
        """测试无历史数据时的内存预测"""
        predictor = MemoryPredictor()
        
        prediction = predictor.predict_memory_usage(
            memory_history=[],
            batch_size=4,
            sequence_length=2048,
            model_parameters=4000000000
        )
        
        assert isinstance(prediction, MemoryPrediction)
        assert prediction.predicted_peak_memory > 0
        assert 0 <= prediction.confidence_score <= 1
        assert prediction.prediction_horizon > 0
        assert isinstance(prediction.factors, dict)
    
    def test_predict_memory_usage_with_history(self):
        """测试有历史数据时的内存预测"""
        predictor = MemoryPredictor()
        
        # 创建模拟历史数据
        history = []
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=datetime.now() - timedelta(minutes=i),
                gpu_id=0,
                total_memory=16384,
                allocated_memory=8000 + i * 100,  # 递增趋势
                cached_memory=1024,
                free_memory=16384 - (8000 + i * 100),
                utilization_rate=(8000 + i * 100) / 16384,
                pressure_level=MemoryPressureLevel.MODERATE,
                system_total_memory=32768,
                system_used_memory=16384,
                system_available_memory=16384,
                process_memory=2048,
                process_memory_percent=6.25
            )
            history.append(snapshot)
        
        prediction = predictor.predict_memory_usage(
            memory_history=history,
            batch_size=4,
            sequence_length=2048,
            model_parameters=4000000000
        )
        
        assert isinstance(prediction, MemoryPrediction)
        assert prediction.predicted_peak_memory > 0
        assert prediction.confidence_score > 0.3  # 有历史数据，置信度应该更高
    
    def test_estimate_base_memory(self):
        """测试基础内存估算"""
        predictor = MemoryPredictor()
        
        base_memory = predictor._estimate_base_memory(4000000000)  # 4B参数
        
        assert base_memory > 0
        assert isinstance(base_memory, int)
        # 4B参数 * 6字节/参数 ≈ 22GB
        assert 20000 < base_memory < 30000  # 20-30GB范围
    
    def test_estimate_batch_memory(self):
        """测试批次内存估算"""
        predictor = MemoryPredictor()
        
        batch_memory = predictor._estimate_batch_memory(4, 2048)
        
        assert batch_memory > 0
        assert isinstance(batch_memory, int)
    
    def test_generate_recommendations_high_utilization(self):
        """测试高内存使用率时的建议生成"""
        predictor = MemoryPredictor()
        
        recommendations = predictor._generate_recommendations(15000, 16384)  # 91%使用率
        
        assert len(recommendations) > 0
        assert any(rec.strategy == OptimizationStrategy.REDUCE_BATCH_SIZE 
                  for rec in recommendations)
        assert any(rec.priority >= 8 for rec in recommendations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemoryMonitor:
    """测试内存监控器（需要CUDA）"""
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        monitor = MemoryMonitor(monitoring_interval=1)
        
        assert monitor.monitoring_interval == 1
        assert monitor.is_monitoring is False
        assert len(monitor.memory_history) == 0
    
    def test_create_memory_snapshot(self):
        """测试创建内存快照"""
        monitor = MemoryMonitor()
        
        snapshot = monitor._create_memory_snapshot(0)
        
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.gpu_id == 0
        assert snapshot.total_memory > 0
        assert snapshot.allocated_memory >= 0
        assert 0 <= snapshot.utilization_rate <= 1
        assert isinstance(snapshot.pressure_level, MemoryPressureLevel)
    
    def test_get_current_memory_status(self):
        """测试获取当前内存状态"""
        monitor = MemoryMonitor()
        
        status = monitor.get_current_memory_status(0)
        
        assert status is not None
        assert isinstance(status, MemorySnapshot)
        assert status.gpu_id == 0
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        monitor = MemoryMonitor(monitoring_interval=1)
        
        # 启动监控
        success = monitor.start_monitoring()
        assert success is True
        assert monitor.is_monitoring is True
        
        # 等待一段时间收集数据
        time.sleep(2)
        
        # 检查是否收集到数据
        assert len(monitor.memory_history) > 0
        
        # 停止监控
        success = monitor.stop_monitoring()
        assert success is True
        assert monitor.is_monitoring is False
    
    def test_pressure_callback(self):
        """测试内存压力回调"""
        monitor = MemoryMonitor()
        callback_called = threading.Event()
        received_snapshot = None
        
        def pressure_callback(snapshot):
            nonlocal received_snapshot
            received_snapshot = snapshot
            callback_called.set()
        
        monitor.add_pressure_callback(pressure_callback)
        
        # 创建高压力快照并手动触发回调
        high_pressure_snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=15000,
            cached_memory=1024,
            free_memory=1384,
            utilization_rate=0.92,
            pressure_level=MemoryPressureLevel.CRITICAL,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        # 手动触发回调
        for callback in monitor.pressure_callbacks:
            callback(high_pressure_snapshot)
        
        # 验证回调被调用
        assert callback_called.is_set()
        assert received_snapshot is not None
        assert received_snapshot.pressure_level == MemoryPressureLevel.CRITICAL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemoryManager:
    """测试内存管理器主类（需要CUDA）"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        config = {
            "monitoring_interval": 2,
            "enable_auto_adjustment": True,
            "initial_batch_size": 4
        }
        
        manager = MemoryManager(config)
        
        assert manager.config == config
        assert manager.monitoring_interval == 2
        assert manager.enable_auto_adjustment is True
        assert manager.is_active is False
    
    def test_start_stop_manager(self):
        """测试启动和停止管理器"""
        manager = MemoryManager({"monitoring_interval": 1})
        
        # 启动管理器
        success = manager.start()
        assert success is True
        assert manager.is_active is True
        
        # 等待一段时间
        time.sleep(2)
        
        # 停止管理器
        success = manager.stop()
        assert success is True
        assert manager.is_active is False
    
    def test_get_current_memory_status(self):
        """测试获取当前内存状态"""
        manager = MemoryManager()
        
        status = manager.get_current_memory_status(0)
        
        assert status is not None
        assert isinstance(status, MemorySnapshot)
    
    def test_predict_memory_usage(self):
        """测试内存使用预测"""
        manager = MemoryManager()
        
        prediction = manager.predict_memory_usage(
            batch_size=4,
            sequence_length=2048,
            model_parameters=4000000000
        )
        
        assert isinstance(prediction, MemoryPrediction)
        assert prediction.predicted_peak_memory > 0
    
    def test_optimize_memory(self):
        """测试内存优化"""
        manager = MemoryManager()
        
        recommendations = manager.optimize_memory(0)
        
        assert isinstance(recommendations, list)
        # 至少应该有清理缓存的建议
        assert any(rec.strategy == OptimizationStrategy.CLEAR_CACHE 
                  for rec in recommendations)
    
    def test_batch_size_management(self):
        """测试批次大小管理"""
        manager = MemoryManager({"initial_batch_size": 4})
        
        # 获取当前批次大小
        current_size = manager.get_current_batch_size()
        assert current_size == 4
        
        # 设置新的批次大小
        success = manager.set_batch_size(8)
        assert success is True
        assert manager.get_current_batch_size() == 8
    
    def test_memory_analysis(self):
        """测试内存分析"""
        manager = MemoryManager({"monitoring_interval": 1})
        
        # 启动监控收集一些数据
        manager.start()
        time.sleep(2)
        
        analysis = manager.get_memory_analysis(0, 5)  # 5分钟分析
        
        assert isinstance(analysis, dict)
        assert "gpu_id" in analysis
        assert "memory_statistics" in analysis
        assert "pressure_distribution" in analysis
        
        manager.stop()
    
    def test_export_memory_report(self):
        """测试导出内存报告"""
        manager = MemoryManager({"monitoring_interval": 1})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "memory_report.json"
            
            # 启动监控收集数据
            manager.start()
            time.sleep(2)
            
            # 导出报告
            success = manager.export_memory_report(str(report_path), 0, 1)
            
            assert success is True
            assert report_path.exists()
            
            # 验证报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            assert "report_info" in report
            assert "memory_analysis" in report
            assert "memory_history" in report
            
            manager.stop()
    
    def test_context_manager(self):
        """测试上下文管理器"""
        config = {"monitoring_interval": 1}
        
        with MemoryManager(config) as manager:
            assert manager.is_active is True
            time.sleep(1)
        
        # 退出上下文后应该自动停止
        assert manager.is_active is False


class TestMemoryManagerIntegration:
    """内存管理器集成测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_pressure_handling(self):
        """测试内存压力处理集成"""
        manager = MemoryManager({
            "monitoring_interval": 1,
            "enable_auto_adjustment": True,
            "initial_batch_size": 8,
            "min_batch_size": 1
        })
        
        optimization_called = threading.Event()
        received_recommendations = None
        
        def optimization_callback(recommendations):
            nonlocal received_recommendations
            received_recommendations = recommendations
            optimization_called.set()
        
        manager.add_optimization_callback(optimization_callback)
        
        # 模拟高内存压力情况
        high_pressure_snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=15000,
            cached_memory=1024,
            free_memory=1384,
            utilization_rate=0.92,
            pressure_level=MemoryPressureLevel.CRITICAL,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=2048,
            process_memory_percent=6.25
        )
        
        # 手动触发内存压力处理
        manager._handle_memory_pressure(high_pressure_snapshot)
        
        # 验证批次大小被调整
        assert manager.get_current_batch_size() < 8
    
    def test_memory_prediction_accuracy(self):
        """测试内存预测准确性"""
        manager = MemoryManager()
        
        # 创建一系列内存快照模拟稳定增长
        history = []
        base_memory = 8000
        for i in range(20):
            snapshot = MemorySnapshot(
                timestamp=datetime.now() - timedelta(minutes=20-i),
                gpu_id=0,
                total_memory=16384,
                allocated_memory=base_memory + i * 50,
                cached_memory=1024,
                free_memory=16384 - (base_memory + i * 50),
                utilization_rate=(base_memory + i * 50) / 16384,
                pressure_level=MemoryPressureLevel.MODERATE,
                system_total_memory=32768,
                system_used_memory=16384,
                system_available_memory=16384,
                process_memory=2048,
                process_memory_percent=6.25
            )
            history.append(snapshot)
        
        # 手动设置历史数据
        manager.monitor.memory_history[0] = history
        
        # 进行预测
        prediction = manager.predict_memory_usage(4, 2048, 4000000000)
        
        # 验证预测结果
        assert prediction.confidence_score > 0.3  # 有历史数据，置信度应该较高
        assert prediction.predicted_peak_memory > base_memory + 19 * 50  # 应该考虑增长趋势


if __name__ == "__main__":
    pytest.main([__file__, "-v"])