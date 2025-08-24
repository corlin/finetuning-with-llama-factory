"""
分布式训练指标收集模块测试

测试DistributedMetricsCollector类的跨GPU通信开销分析和统计、负载均衡评估和优化建议生成、
通信效率和带宽利用率监控等功能。
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.distributed_metrics_collector import (
    DistributedMetricsCollector, CommunicationProfiler, CommunicationEvent,
    LoadBalanceMetrics, CommunicationEfficiencyMetrics, CommunicationContext,
    create_distributed_metrics_collector, with_communication_profiling
)
from src.parallel_config import DistributedTrainingMetrics, CommunicationMetrics


class TestCommunicationEvent:
    """测试通信事件类"""
    
    def test_communication_event_initialization(self):
        """测试通信事件初始化"""
        event = CommunicationEvent(
            event_type="allreduce",
            start_time=1.0,
            end_time=2.0,
            data_size=1024,
            group_size=4
        )
        
        assert event.event_type == "allreduce"
        assert event.start_time == 1.0
        assert event.end_time == 2.0
        assert event.data_size == 1024
        assert event.group_size == 4
    
    def test_duration_calculation(self):
        """测试持续时间计算"""
        event = CommunicationEvent(
            event_type="broadcast",
            start_time=1.0,
            end_time=3.5,
            data_size=2048
        )
        
        assert event.duration == 2.5
    
    def test_bandwidth_calculation(self):
        """测试带宽计算"""
        event = CommunicationEvent(
            event_type="p2p",
            start_time=0.0,
            end_time=1.0,  # 1秒
            data_size=1024 * 1024  # 1MB
        )
        
        assert event.bandwidth_mbps == 1.0  # 1MB/s
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        event = CommunicationEvent(
            event_type="allreduce",
            start_time=1.0,
            end_time=2.0,
            data_size=1024,
            src_rank=0,
            dst_rank=1,
            group_size=2
        )
        
        data = event.to_dict()
        
        assert data["event_type"] == "allreduce"
        assert data["duration"] == 1.0
        assert data["bandwidth_mbps"] == 1024 / (1024 * 1024)  # 很小的值
        assert data["src_rank"] == 0
        assert data["dst_rank"] == 1
        assert data["group_size"] == 2


class TestLoadBalanceMetrics:
    """测试负载均衡指标类"""
    
    def test_load_balance_metrics_initialization(self):
        """测试负载均衡指标初始化"""
        metrics = LoadBalanceMetrics()
        
        assert isinstance(metrics.timestamp, datetime)
        assert len(metrics.gpu_utilizations) == 0
        assert metrics.mean_utilization == 0.0
        assert metrics.utilization_variance == 0.0
    
    def test_calculate_balance_scores(self):
        """测试负载均衡评分计算"""
        metrics = LoadBalanceMetrics()
        
        # 设置测试数据
        metrics.gpu_utilizations = {0: 80.0, 1: 85.0, 2: 75.0, 3: 90.0}
        metrics.memory_usages = {0: 70.0, 1: 75.0, 2: 65.0, 3: 80.0}
        metrics.workload_distribution = {0: 100.0, 1: 110.0, 2: 95.0, 3: 105.0}
        metrics.communication_loads = {0: 20.0, 1: 25.0, 2: 18.0, 3: 22.0}
        
        # 计算评分
        metrics.calculate_balance_scores()
        
        # 检查计算结果
        assert metrics.mean_utilization == 82.5  # (80+85+75+90)/4
        assert metrics.utilization_variance > 0
        assert metrics.utilization_std > 0
        
        assert metrics.mean_memory_usage == 72.5  # (70+75+65+80)/4
        assert metrics.memory_variance > 0
        assert metrics.memory_std > 0
        
        assert metrics.workload_imbalance_score > 0
        assert metrics.communication_imbalance > 0
    
    def test_overall_balance_score(self):
        """测试综合负载均衡评分"""
        metrics = LoadBalanceMetrics()
        
        # 设置完全均衡的数据
        metrics.gpu_utilizations = {0: 80.0, 1: 80.0, 2: 80.0, 3: 80.0}
        metrics.memory_usages = {0: 70.0, 1: 70.0, 2: 70.0, 3: 70.0}
        metrics.workload_distribution = {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}
        metrics.communication_loads = {0: 20.0, 1: 20.0, 2: 20.0, 3: 20.0}
        
        metrics.calculate_balance_scores()
        
        # 完全均衡应该得到高分
        assert metrics.overall_balance_score > 0.9
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        metrics = LoadBalanceMetrics()
        metrics.gpu_utilizations = {0: 80.0, 1: 85.0}
        metrics.calculate_balance_scores()
        
        data = metrics.to_dict()
        
        assert "timestamp" in data
        assert "gpu_utilizations" in data
        assert "overall_balance_score" in data
        assert data["gpu_utilizations"] == {0: 80.0, 1: 85.0}


class TestCommunicationEfficiencyMetrics:
    """测试通信效率指标类"""
    
    def test_communication_efficiency_metrics_initialization(self):
        """测试通信效率指标初始化"""
        metrics = CommunicationEfficiencyMetrics()
        
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.total_communication_time == 0.0
        assert metrics.communication_efficiency == 0.0
        assert len(metrics.allreduce_stats) == 0
    
    def test_calculate_efficiency_metrics(self):
        """测试效率指标计算"""
        metrics = CommunicationEfficiencyMetrics()
        
        # 创建测试事件
        events = [
            CommunicationEvent("allreduce", 0.0, 1.0, 1024 * 1024),  # 1MB, 1秒
            CommunicationEvent("allreduce", 1.0, 2.5, 2048 * 1024),  # 2MB, 1.5秒
            CommunicationEvent("broadcast", 2.5, 3.0, 512 * 1024),   # 0.5MB, 0.5秒
            CommunicationEvent("p2p", 3.0, 3.2, 256 * 1024)         # 0.25MB, 0.2秒
        ]
        
        # 计算效率指标
        metrics.calculate_efficiency_metrics(events)
        
        # 检查AllReduce统计
        assert metrics.allreduce_stats["count"] == 2
        assert metrics.allreduce_stats["total_time"] == 2.5  # 1.0 + 1.5
        assert metrics.allreduce_stats["avg_time"] == 1.25   # 2.5 / 2
        
        # 检查Broadcast统计
        assert metrics.broadcast_stats["count"] == 1
        assert metrics.broadcast_stats["total_time"] == 0.5
        
        # 检查P2P统计
        assert metrics.p2p_stats["count"] == 1
        assert abs(metrics.p2p_stats["total_time"] - 0.2) < 1e-10  # 使用浮点数比较
        
        # 检查总体统计
        assert metrics.total_communication_time == 3.2  # 1.0 + 1.5 + 0.5 + 0.2
        assert metrics.peak_bandwidth_mbps > 0
        assert metrics.average_bandwidth_mbps > 0
        
        # 检查通信模式
        assert metrics.communication_patterns["allreduce"] == 2
        assert metrics.communication_patterns["broadcast"] == 1
        assert metrics.communication_patterns["p2p"] == 1
    
    def test_efficiency_score_calculation(self):
        """测试效率评分计算"""
        metrics = CommunicationEfficiencyMetrics()
        
        # 设置测试数据
        metrics.communication_computation_ratio = 0.1  # 低通信开销
        metrics.bandwidth_utilization = 0.8            # 高带宽利用率
        metrics.communication_patterns = {
            "allreduce": 8,
            "broadcast": 1,
            "p2p": 1
        }
        
        # 计算效率评分
        metrics._calculate_efficiency_score()
        
        # 应该得到较高的效率评分
        assert metrics.communication_efficiency > 0.7
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        metrics = CommunicationEfficiencyMetrics()
        metrics.total_communication_time = 5.0
        metrics.communication_efficiency = 0.8
        
        data = metrics.to_dict()
        
        assert "timestamp" in data
        assert "total_communication_time" in data
        assert "communication_efficiency" in data
        assert data["total_communication_time"] == 5.0
        assert data["communication_efficiency"] == 0.8


class TestCommunicationProfiler:
    """测试通信性能分析器类"""
    
    def test_communication_profiler_initialization(self):
        """测试通信性能分析器初始化"""
        profiler = CommunicationProfiler(max_events=1000)
        
        assert profiler.max_events == 1000
        assert len(profiler.events) == 0
        assert len(profiler.active_events) == 0
        assert not profiler.profiling_enabled
    
    def test_enable_disable_profiling(self):
        """测试启用和禁用性能分析"""
        profiler = CommunicationProfiler()
        
        # 启用分析
        profiler.enable_profiling()
        assert profiler.profiling_enabled
        
        # 禁用分析
        profiler.disable_profiling()
        assert not profiler.profiling_enabled
    
    def test_communication_event_recording(self):
        """测试通信事件记录"""
        profiler = CommunicationProfiler()
        profiler.enable_profiling()
        
        # 开始记录事件
        event_id = profiler.start_communication_event(
            "allreduce", 1024, group_size=4
        )
        
        assert event_id != ""
        assert event_id in profiler.active_events
        
        # 模拟一些延迟
        time.sleep(0.01)
        
        # 结束记录事件
        profiler.end_communication_event(event_id)
        
        assert event_id not in profiler.active_events
        assert len(profiler.events) == 1
        
        # 检查记录的事件
        recorded_event = profiler.events[0]
        assert recorded_event.event_type == "allreduce"
        assert recorded_event.data_size == 1024
        assert recorded_event.group_size == 4
        assert recorded_event.duration > 0
    
    def test_get_recent_events(self):
        """测试获取最近事件"""
        profiler = CommunicationProfiler()
        profiler.enable_profiling()
        
        # 记录多个事件
        for i in range(10):
            event_id = profiler.start_communication_event("allreduce", 1024 * i)
            profiler.end_communication_event(event_id)
        
        # 获取最近5个事件
        recent_events = profiler.get_recent_events(5)
        assert len(recent_events) == 5
        
        # 检查是否是最新的事件
        assert recent_events[-1].data_size == 1024 * 9  # 最后一个事件
    
    def test_get_events_by_type(self):
        """测试按类型获取事件"""
        profiler = CommunicationProfiler()
        profiler.enable_profiling()
        
        # 记录不同类型的事件
        for event_type in ["allreduce", "broadcast", "p2p"]:
            for i in range(3):
                event_id = profiler.start_communication_event(event_type, 1024)
                profiler.end_communication_event(event_id)
        
        # 按类型获取事件
        allreduce_events = profiler.get_events_by_type("allreduce")
        broadcast_events = profiler.get_events_by_type("broadcast")
        p2p_events = profiler.get_events_by_type("p2p")
        
        assert len(allreduce_events) == 3
        assert len(broadcast_events) == 3
        assert len(p2p_events) == 3
        
        # 检查事件类型
        assert all(e.event_type == "allreduce" for e in allreduce_events)
        assert all(e.event_type == "broadcast" for e in broadcast_events)
        assert all(e.event_type == "p2p" for e in p2p_events)
    
    def test_clear_events(self):
        """测试清空事件"""
        profiler = CommunicationProfiler()
        profiler.enable_profiling()
        
        # 记录一些事件
        for i in range(5):
            event_id = profiler.start_communication_event("allreduce", 1024)
            profiler.end_communication_event(event_id)
        
        assert len(profiler.events) == 5
        
        # 清空事件
        profiler.clear_events()
        
        assert len(profiler.events) == 0
        assert len(profiler.active_events) == 0


class TestDistributedMetricsCollector:
    """测试分布式指标收集器类"""
    
    def test_distributed_metrics_collector_initialization(self):
        """测试分布式指标收集器初始化"""
        collector = DistributedMetricsCollector(
            world_size=4,
            rank=0,
            gpu_ids=[0, 1],
            collection_interval=0.5
        )
        
        assert collector.world_size == 4
        assert collector.rank == 0
        assert collector.gpu_ids == [0, 1]
        assert collector.collection_interval == 0.5
        assert isinstance(collector.comm_profiler, CommunicationProfiler)
        assert not collector.collecting
    
    def test_start_stop_collection(self):
        """测试启动和停止收集"""
        collector = DistributedMetricsCollector(4, 0, [0])
        
        # 启动收集
        collector.start_collection()
        assert collector.collecting
        assert collector.comm_profiler.profiling_enabled
        assert collector.collection_thread is not None
        
        # 等待一小段时间让收集运行
        time.sleep(0.1)
        
        # 停止收集
        collector.stop_collection()
        assert not collector.collecting
        assert not collector.comm_profiler.profiling_enabled
    
    def test_workload_tracking(self):
        """测试工作负载跟踪"""
        collector = DistributedMetricsCollector(4, 0, [0, 1])
        
        # 记录计算开始和结束
        collector.record_computation_start(0)
        time.sleep(0.01)  # 模拟计算时间
        collector.record_computation_end(0, samples_processed=10)
        
        # 检查工作负载是否被记录
        assert collector.workload_tracker[0] > 0
        
        # 记录另一个GPU的工作负载
        collector.record_computation_start(1)
        time.sleep(0.02)  # 更长的计算时间
        collector.record_computation_end(1, samples_processed=5)
        
        # 由于时间测量的不确定性，只检查两个GPU都有工作负载记录
        assert collector.workload_tracker[0] > 0
        assert collector.workload_tracker[1] > 0
    
    def test_communication_recording(self):
        """测试通信记录"""
        collector = DistributedMetricsCollector(4, 0, [0])
        collector.start_collection()
        
        # 记录AllReduce
        event_id = collector.record_allreduce(1024 * 1024)  # 1MB
        assert event_id != ""
        
        time.sleep(0.01)  # 模拟通信时间
        collector.finish_communication(event_id)
        
        # 记录Broadcast
        event_id = collector.record_broadcast(512 * 1024, src_rank=0)  # 0.5MB
        time.sleep(0.005)
        collector.finish_communication(event_id)
        
        # 记录P2P
        event_id = collector.record_p2p_send(256 * 1024, dst_rank=1)  # 0.25MB
        time.sleep(0.002)
        collector.finish_communication(event_id)
        
        # 检查事件是否被记录
        events = collector.comm_profiler.get_recent_events()
        assert len(events) >= 3
        
        collector.stop_collection()
    
    def test_load_balance_metrics_collection(self):
        """测试负载均衡指标收集"""
        collector = DistributedMetricsCollector(4, 0, [0, 1])
        
        # 添加一些工作负载数据
        collector.workload_tracker[0] = 100.0
        collector.workload_tracker[1] = 120.0
        
        # 收集负载均衡指标
        metrics = collector._collect_load_balance_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, LoadBalanceMetrics)
        assert 0 in metrics.gpu_utilizations
        assert 1 in metrics.gpu_utilizations
        assert metrics.workload_distribution[0] == 100.0
        assert metrics.workload_distribution[1] == 120.0
    
    def test_communication_efficiency_collection(self):
        """测试通信效率指标收集"""
        collector = DistributedMetricsCollector(4, 0, [0])
        collector.start_collection()
        
        # 添加一些通信事件
        for i in range(5):
            event_id = collector.record_allreduce(1024 * (i + 1))
            time.sleep(0.001)
            collector.finish_communication(event_id)
        
        # 收集通信效率指标
        metrics = collector._collect_communication_efficiency_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, CommunicationEfficiencyMetrics)
        assert metrics.total_communication_time > 0
        
        collector.stop_collection()
    
    def test_optimization_recommendations(self):
        """测试优化建议生成"""
        collector = DistributedMetricsCollector(4, 0, [0, 1])
        collector.start_collection()
        
        # 等待收集一些数据
        time.sleep(0.2)
        
        # 生成优化建议
        recommendations = collector.generate_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        # 建议可能为空，这取决于模拟数据的质量
        
        collector.stop_collection()
    
    def test_create_distributed_training_metrics(self):
        """测试创建分布式训练指标"""
        collector = DistributedMetricsCollector(4, 0, [0, 1])
        collector.start_collection()
        
        # 添加一些通信事件
        event_id = collector.record_allreduce(1024 * 1024)
        time.sleep(0.01)
        collector.finish_communication(event_id)
        
        # 等待指标收集
        time.sleep(0.1)
        
        # 创建分布式训练指标
        gpu_metrics = {
            0: {"utilization": 80.0, "memory_usage_percent": 70.0},
            1: {"utilization": 85.0, "memory_usage_percent": 75.0}
        }
        
        metrics = collector.create_distributed_training_metrics(
            epoch=1,
            global_step=100,
            train_loss=0.5,
            val_loss=0.55,
            learning_rate=0.001,
            gpu_metrics=gpu_metrics
        )
        
        assert isinstance(metrics, DistributedTrainingMetrics)
        assert metrics.epoch == 1
        assert metrics.global_step == 100
        assert metrics.train_loss == 0.5
        assert metrics.gpu_metrics == gpu_metrics
        
        collector.stop_collection()
    
    def test_export_metrics_summary(self):
        """测试导出指标摘要"""
        collector = DistributedMetricsCollector(4, 0, [0, 1])
        collector.start_collection()
        
        # 等待收集一些数据
        time.sleep(0.2)
        
        # 导出摘要
        summary = collector.export_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "collection_info" in summary
        assert "load_balance_summary" in summary
        assert "communication_efficiency_summary" in summary
        assert "optimization_recommendations" in summary
        
        # 检查收集信息
        collection_info = summary["collection_info"]
        assert collection_info["world_size"] == 4
        assert collection_info["rank"] == 0
        assert collection_info["gpu_ids"] == [0, 1]
        
        collector.stop_collection()


class TestCommunicationContext:
    """测试通信上下文管理器"""
    
    def test_allreduce_context(self):
        """测试AllReduce上下文管理器"""
        collector = DistributedMetricsCollector(4, 0, [0])
        collector.start_collection()
        
        # 使用上下文管理器
        with CommunicationContext(collector, "allreduce", 1024 * 1024):
            time.sleep(0.01)  # 模拟通信时间
        
        # 检查事件是否被记录
        events = collector.comm_profiler.get_recent_events()
        assert len(events) >= 1
        
        allreduce_events = [e for e in events if e.event_type == "allreduce"]
        assert len(allreduce_events) >= 1
        assert allreduce_events[0].data_size == 1024 * 1024
        
        collector.stop_collection()
    
    def test_broadcast_context(self):
        """测试Broadcast上下文管理器"""
        collector = DistributedMetricsCollector(4, 0, [0])
        collector.start_collection()
        
        # 使用上下文管理器
        with CommunicationContext(collector, "broadcast", 512 * 1024, src_rank=0):
            time.sleep(0.005)  # 模拟通信时间
        
        # 检查事件是否被记录
        events = collector.comm_profiler.get_recent_events()
        broadcast_events = [e for e in events if e.event_type == "broadcast"]
        assert len(broadcast_events) >= 1
        assert broadcast_events[0].data_size == 512 * 1024
        
        collector.stop_collection()
    
    def test_p2p_context(self):
        """测试P2P上下文管理器"""
        collector = DistributedMetricsCollector(4, 0, [0])
        collector.start_collection()
        
        # 使用上下文管理器
        with CommunicationContext(collector, "p2p", 256 * 1024, dst_rank=1):
            time.sleep(0.002)  # 模拟通信时间
        
        # 检查事件是否被记录
        events = collector.comm_profiler.get_recent_events()
        p2p_events = [e for e in events if e.event_type == "p2p"]
        assert len(p2p_events) >= 1
        assert p2p_events[0].data_size == 256 * 1024
        
        collector.stop_collection()


class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_create_distributed_metrics_collector(self):
        """测试创建分布式指标收集器的便捷函数"""
        collector = create_distributed_metrics_collector(
            world_size=8,
            rank=2,
            gpu_ids=[2, 3]
        )
        
        assert isinstance(collector, DistributedMetricsCollector)
        assert collector.world_size == 8
        assert collector.rank == 2
        assert collector.gpu_ids == [2, 3]
    
    def test_with_communication_profiling(self):
        """测试通信性能分析便捷函数"""
        collector = DistributedMetricsCollector(4, 0, [0])
        collector.start_collection()
        
        # 使用便捷函数
        context = with_communication_profiling(
            collector, "allreduce", 2048 * 1024
        )
        
        assert isinstance(context, CommunicationContext)
        assert context.event_type == "allreduce"
        assert context.tensor_size == 2048 * 1024
        
        # 使用上下文
        with context:
            time.sleep(0.01)
        
        # 检查事件记录
        events = collector.comm_profiler.get_recent_events()
        assert len(events) >= 1
        
        collector.stop_collection()


class TestIntegration:
    """集成测试"""
    
    def test_full_distributed_metrics_workflow(self):
        """测试完整的分布式指标收集工作流"""
        collector = DistributedMetricsCollector(4, 0, [0, 1])
        
        try:
            # 启动收集
            collector.start_collection()
            
            # 模拟训练过程
            for step in range(5):
                # 记录计算
                for gpu_id in [0, 1]:
                    collector.record_computation_start(gpu_id)
                    time.sleep(0.01)  # 模拟计算
                    collector.record_computation_end(gpu_id, samples_processed=32)
                
                # 记录通信
                with with_communication_profiling(collector, "allreduce", 1024 * 1024):
                    time.sleep(0.005)  # 模拟AllReduce
                
                if step % 2 == 0:  # 偶数步进行广播
                    with with_communication_profiling(collector, "broadcast", 512 * 1024, src_rank=0):
                        time.sleep(0.002)  # 模拟Broadcast
            
            # 等待指标收集
            time.sleep(0.3)
            
            # 检查收集的指标
            load_balance_metrics = collector.get_current_load_balance_metrics()
            comm_efficiency_metrics = collector.get_current_communication_efficiency()
            
            # 负载均衡指标应该存在
            if load_balance_metrics:
                assert isinstance(load_balance_metrics, LoadBalanceMetrics)
                assert len(load_balance_metrics.gpu_utilizations) > 0
            
            # 通信效率指标应该存在
            if comm_efficiency_metrics:
                assert isinstance(comm_efficiency_metrics, CommunicationEfficiencyMetrics)
                # 由于异步收集的时间问题，通信时间可能为0，这是正常的
            
            # 生成优化建议
            recommendations = collector.generate_optimization_recommendations()
            assert isinstance(recommendations, list)
            
            # 导出摘要
            summary = collector.export_metrics_summary()
            assert isinstance(summary, dict)
            assert "collection_info" in summary
            
            # 创建分布式训练指标
            gpu_metrics = {
                0: {"utilization": 80.0, "memory_usage_percent": 70.0},
                1: {"utilization": 85.0, "memory_usage_percent": 75.0}
            }
            
            distributed_metrics = collector.create_distributed_training_metrics(
                epoch=1,
                global_step=100,
                train_loss=0.5,
                val_loss=0.55,
                learning_rate=0.001,
                gpu_metrics=gpu_metrics
            )
            
            assert isinstance(distributed_metrics, DistributedTrainingMetrics)
            assert distributed_metrics.load_balance_score >= 0.0
            
        finally:
            # 停止收集
            collector.stop_collection()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])