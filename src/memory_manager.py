"""
动态内存管理器

本模块实现了GPU内存监控、动态批次大小调整、内存压力检测和响应机制。
专门针对Qwen3-4B-Thinking模型的内存需求进行优化，支持多GPU环境下的内存管理。
"""

import torch
import psutil
import gc
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings

from gpu_utils import GPUDetector, GPUInfo, GPUTopology
from data_models import ChineseMetrics


class MemoryPressureLevel(Enum):
    """内存压力级别"""
    LOW = "low"           # < 60% 使用率
    MODERATE = "moderate" # 60-80% 使用率
    HIGH = "high"         # 80-90% 使用率
    CRITICAL = "critical" # > 90% 使用率


class OptimizationStrategy(Enum):
    """内存优化策略"""
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    ENABLE_GRADIENT_CHECKPOINTING = "enable_gradient_checkpointing"
    ENABLE_CPU_OFFLOAD = "enable_cpu_offload"
    CLEAR_CACHE = "clear_cache"
    REDUCE_SEQUENCE_LENGTH = "reduce_sequence_length"
    ENABLE_ACTIVATION_RECOMPUTATION = "enable_activation_recomputation"


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: datetime
    gpu_id: int
    total_memory: int  # MB
    allocated_memory: int  # MB
    cached_memory: int  # MB
    free_memory: int  # MB
    utilization_rate: float  # 0-1
    pressure_level: MemoryPressureLevel
    
    # 系统内存
    system_total_memory: int  # MB
    system_used_memory: int  # MB
    system_available_memory: int  # MB
    
    # 进程内存
    process_memory: int  # MB
    process_memory_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_id": self.gpu_id,
            "total_memory": self.total_memory,
            "allocated_memory": self.allocated_memory,
            "cached_memory": self.cached_memory,
            "free_memory": self.free_memory,
            "utilization_rate": self.utilization_rate,
            "pressure_level": self.pressure_level.value,
            "system_total_memory": self.system_total_memory,
            "system_used_memory": self.system_used_memory,
            "system_available_memory": self.system_available_memory,
            "process_memory": self.process_memory,
            "process_memory_percent": self.process_memory_percent
        }


@dataclass
class MemoryOptimizationRecommendation:
    """内存优化建议"""
    strategy: OptimizationStrategy
    priority: int  # 1-10, 10为最高优先级
    description: str
    expected_memory_saving: int  # MB
    implementation_difficulty: int  # 1-5, 5为最难实现
    side_effects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "strategy": self.strategy.value,
            "priority": self.priority,
            "description": self.description,
            "expected_memory_saving": self.expected_memory_saving,
            "implementation_difficulty": self.implementation_difficulty,
            "side_effects": self.side_effects
        }


@dataclass
class BatchSizeAdjustment:
    """批次大小调整记录"""
    timestamp: datetime
    gpu_id: int
    old_batch_size: int
    new_batch_size: int
    reason: str
    memory_before: int  # MB
    memory_after: int  # MB
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_id": self.gpu_id,
            "old_batch_size": self.old_batch_size,
            "new_batch_size": self.new_batch_size,
            "reason": self.reason,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "success": self.success
        }


@dataclass
class MemoryPrediction:
    """内存使用预测"""
    predicted_peak_memory: int  # MB
    confidence_score: float  # 0-1
    prediction_horizon: int  # 预测时间范围（秒）
    factors: Dict[str, float]  # 影响因素及其权重
    recommendations: List[MemoryOptimizationRecommendation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "predicted_peak_memory": self.predicted_peak_memory,
            "confidence_score": self.confidence_score,
            "prediction_horizon": self.prediction_horizon,
            "factors": self.factors,
            "recommendations": [rec.to_dict() for rec in self.recommendations]
        }


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, monitoring_interval: int = 5):
        """初始化内存监控器"""
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        self.gpu_detector = GPUDetector()
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 内存历史记录
        self.memory_history: Dict[int, List[MemorySnapshot]] = {}
        self.max_history_length = 1000
        
        # 回调函数
        self.pressure_callbacks: List[Callable[[MemorySnapshot], None]] = []
        
    def start_monitoring(self) -> bool:
        """开始内存监控"""
        if self.is_monitoring:
            self.logger.warning("内存监控已在运行")
            return False
        
        try:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("内存监控已启动")
            return True
        except Exception as e:
            self.logger.error(f"启动内存监控失败: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """停止内存监控"""
        if not self.is_monitoring:
            return True
        
        try:
            self.is_monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            self.logger.info("内存监控已停止")
            return True
        except Exception as e:
            self.logger.error(f"停止内存监控失败: {e}")
            return False
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_memory_snapshots()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"内存监控循环出错: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_memory_snapshots(self):
        """收集内存快照"""
        if not torch.cuda.is_available():
            return
        
        device_count = torch.cuda.device_count()
        for gpu_id in range(device_count):
            try:
                snapshot = self._create_memory_snapshot(gpu_id)
                
                # 存储历史记录
                if gpu_id not in self.memory_history:
                    self.memory_history[gpu_id] = []
                
                self.memory_history[gpu_id].append(snapshot)
                
                # 限制历史记录长度
                if len(self.memory_history[gpu_id]) > self.max_history_length:
                    self.memory_history[gpu_id] = self.memory_history[gpu_id][-self.max_history_length:]
                
                # 检查内存压力并触发回调
                if snapshot.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    for callback in self.pressure_callbacks:
                        try:
                            callback(snapshot)
                        except Exception as e:
                            self.logger.error(f"内存压力回调执行失败: {e}")
                            
            except Exception as e:
                self.logger.error(f"收集GPU {gpu_id}内存快照失败: {e}")
    
    def _create_memory_snapshot(self, gpu_id: int) -> MemorySnapshot:
        """创建内存快照"""
        # GPU内存信息
        torch.cuda.set_device(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory // (1024**2)
        allocated_memory = torch.cuda.memory_allocated(gpu_id) // (1024**2)
        cached_memory = torch.cuda.memory_reserved(gpu_id) // (1024**2)
        free_memory = total_memory - allocated_memory
        utilization_rate = allocated_memory / total_memory if total_memory > 0 else 0.0
        
        # 确定压力级别
        if utilization_rate < 0.6:
            pressure_level = MemoryPressureLevel.LOW
        elif utilization_rate < 0.8:
            pressure_level = MemoryPressureLevel.MODERATE
        elif utilization_rate < 0.9:
            pressure_level = MemoryPressureLevel.HIGH
        else:
            pressure_level = MemoryPressureLevel.CRITICAL
        
        # 系统内存信息
        system_memory = psutil.virtual_memory()
        system_total = system_memory.total // (1024**2)
        system_used = system_memory.used // (1024**2)
        system_available = system_memory.available // (1024**2)
        
        # 进程内存信息
        process = psutil.Process()
        process_memory = process.memory_info().rss // (1024**2)
        process_memory_percent = process.memory_percent()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=gpu_id,
            total_memory=total_memory,
            allocated_memory=allocated_memory,
            cached_memory=cached_memory,
            free_memory=free_memory,
            utilization_rate=utilization_rate,
            pressure_level=pressure_level,
            system_total_memory=system_total,
            system_used_memory=system_used,
            system_available_memory=system_available,
            process_memory=process_memory,
            process_memory_percent=process_memory_percent
        )
    
    def get_current_memory_status(self, gpu_id: int = 0) -> Optional[MemorySnapshot]:
        """获取当前内存状态"""
        try:
            return self._create_memory_snapshot(gpu_id)
        except Exception as e:
            self.logger.error(f"获取GPU {gpu_id}内存状态失败: {e}")
            return None
    
    def get_memory_history(self, gpu_id: int = 0, 
                          duration_minutes: int = 60) -> List[MemorySnapshot]:
        """获取内存历史记录"""
        if gpu_id not in self.memory_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [
            snapshot for snapshot in self.memory_history[gpu_id]
            if snapshot.timestamp >= cutoff_time
        ]
    
    def add_pressure_callback(self, callback: Callable[[MemorySnapshot], None]):
        """添加内存压力回调函数"""
        self.pressure_callbacks.append(callback)
    
    def remove_pressure_callback(self, callback: Callable[[MemorySnapshot], None]):
        """移除内存压力回调函数"""
        if callback in self.pressure_callbacks:
            self.pressure_callbacks.remove(callback)


class DynamicBatchSizeAdjuster:
    """动态批次大小调整器"""
    
    def __init__(self, initial_batch_size: int = 4, min_batch_size: int = 1, max_batch_size: int = 32):
        """初始化批次大小调整器"""
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
        self.logger = logging.getLogger(__name__)
        self.adjustment_history: List[BatchSizeAdjustment] = []
        
        # 调整策略参数
        self.memory_threshold_high = 0.85  # 85%内存使用率触发减小
        self.memory_threshold_low = 0.6    # 60%内存使用率可以尝试增大
        self.adjustment_factor = 0.75      # 调整因子
        self.cooldown_period = 30          # 冷却期（秒）
        
        self.last_adjustment_time = datetime.min
    
    def should_adjust_batch_size(self, memory_snapshot: MemorySnapshot) -> Tuple[bool, str]:
        """判断是否需要调整批次大小"""
        # 检查冷却期
        if datetime.now() - self.last_adjustment_time < timedelta(seconds=self.cooldown_period):
            return False, "在冷却期内"
        
        # 检查内存压力
        if memory_snapshot.utilization_rate > self.memory_threshold_high:
            if self.current_batch_size > self.min_batch_size:
                return True, f"内存使用率过高: {memory_snapshot.utilization_rate:.2%}"
        
        elif memory_snapshot.utilization_rate < self.memory_threshold_low:
            if self.current_batch_size < self.max_batch_size:
                return True, f"内存使用率较低，可以增大批次: {memory_snapshot.utilization_rate:.2%}"
        
        return False, "内存使用率正常"
    
    def adjust_batch_size(self, memory_snapshot: MemorySnapshot) -> Optional[BatchSizeAdjustment]:
        """调整批次大小"""
        should_adjust, reason = self.should_adjust_batch_size(memory_snapshot)
        
        if not should_adjust:
            return None
        
        old_batch_size = self.current_batch_size
        
        # 计算新的批次大小
        if memory_snapshot.utilization_rate > self.memory_threshold_high:
            # 减小批次大小
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * self.adjustment_factor)
            )
        else:
            # 增大批次大小
            new_batch_size = min(
                self.max_batch_size,
                max(self.current_batch_size + 1, int(self.current_batch_size / self.adjustment_factor))
            )
        
        # 确保批次大小有实际变化
        if new_batch_size == old_batch_size:
            return None
        
        # 执行调整
        self.current_batch_size = new_batch_size
        self.last_adjustment_time = datetime.now()
        
        # 记录调整
        adjustment = BatchSizeAdjustment(
            timestamp=datetime.now(),
            gpu_id=memory_snapshot.gpu_id,
            old_batch_size=old_batch_size,
            new_batch_size=new_batch_size,
            reason=reason,
            memory_before=memory_snapshot.allocated_memory,
            memory_after=0,  # 将在调整后更新
            success=True
        )
        
        self.adjustment_history.append(adjustment)
        
        self.logger.info(f"批次大小已调整: {old_batch_size} -> {new_batch_size}, 原因: {reason}")
        
        return adjustment
    
    def get_current_batch_size(self) -> int:
        """获取当前批次大小"""
        return self.current_batch_size
    
    def reset_batch_size(self):
        """重置批次大小为初始值"""
        self.current_batch_size = self.initial_batch_size
        self.last_adjustment_time = datetime.min
        self.logger.info(f"批次大小已重置为: {self.initial_batch_size}")
    
    def get_adjustment_history(self, duration_hours: int = 24) -> List[BatchSizeAdjustment]:
        """获取调整历史"""
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        return [
            adj for adj in self.adjustment_history
            if adj.timestamp >= cutoff_time
        ]


class MemoryPredictor:
    """内存使用预测器"""
    
    def __init__(self):
        """初始化内存预测器"""
        self.logger = logging.getLogger(__name__)
        self.prediction_models = {}  # 可以扩展为机器学习模型
    
    def predict_memory_usage(self, memory_history: List[MemorySnapshot],
                           batch_size: int, sequence_length: int,
                           model_parameters: int) -> MemoryPrediction:
        """预测内存使用"""
        if not memory_history:
            return self._create_default_prediction(batch_size, sequence_length, model_parameters)
        
        # 分析历史趋势
        recent_snapshots = memory_history[-10:]  # 最近10个快照
        avg_utilization = sum(s.utilization_rate for s in recent_snapshots) / len(recent_snapshots)
        max_utilization = max(s.utilization_rate for s in recent_snapshots)
        
        # 基于经验公式预测
        base_memory = self._estimate_base_memory(model_parameters)
        batch_memory = self._estimate_batch_memory(batch_size, sequence_length)
        overhead_memory = self._estimate_overhead_memory()
        
        predicted_memory = base_memory + batch_memory + overhead_memory
        
        # 考虑历史趋势
        trend_factor = 1.0
        if len(memory_history) >= 5:
            recent_trend = (recent_snapshots[-1].allocated_memory - 
                          recent_snapshots[0].allocated_memory) / len(recent_snapshots)
            if recent_trend > 0:
                trend_factor = 1.1  # 上升趋势，增加10%预测
        
        predicted_memory = int(predicted_memory * trend_factor)
        
        # 计算置信度
        confidence = self._calculate_confidence(memory_history, predicted_memory)
        
        # 影响因素分析
        factors = {
            "model_size": 0.4,
            "batch_size": 0.3,
            "sequence_length": 0.2,
            "historical_trend": 0.1
        }
        
        # 生成建议
        recommendations = self._generate_recommendations(
            predicted_memory, recent_snapshots[-1].total_memory
        )
        
        return MemoryPrediction(
            predicted_peak_memory=predicted_memory,
            confidence_score=confidence,
            prediction_horizon=300,  # 5分钟预测
            factors=factors,
            recommendations=recommendations
        )
    
    def _estimate_base_memory(self, model_parameters: int) -> int:
        """估算模型基础内存需求（MB）"""
        # Qwen3-4B模型大约4B参数
        # 假设FP16，每个参数2字节，加上优化器状态等
        bytes_per_param = 6  # 参数(2) + 梯度(2) + 优化器状态(2)
        base_memory_bytes = model_parameters * bytes_per_param
        return base_memory_bytes // (1024**2)
    
    def _estimate_batch_memory(self, batch_size: int, sequence_length: int) -> int:
        """估算批次相关内存需求（MB）"""
        # 激活值内存估算
        hidden_size = 3584  # Qwen3-4B hidden size
        num_layers = 32     # Qwen3-4B layers
        
        # 每个token的激活值内存（字节）
        activation_per_token = hidden_size * num_layers * 2  # FP16
        
        # 总激活值内存
        total_activation_bytes = batch_size * sequence_length * activation_per_token
        
        return total_activation_bytes // (1024**2)
    
    def _estimate_overhead_memory(self) -> int:
        """估算系统开销内存（MB）"""
        # CUDA上下文、缓存等开销
        return 1024  # 1GB开销
    
    def _calculate_confidence(self, memory_history: List[MemorySnapshot], 
                            predicted_memory: int) -> float:
        """计算预测置信度"""
        if len(memory_history) < 3:
            return 0.5  # 数据不足，中等置信度
        
        # 计算历史预测准确性
        recent_memories = [s.allocated_memory for s in memory_history[-5:]]
        if not recent_memories:
            return 0.3
            
        mean_memory = sum(recent_memories) / len(recent_memories)
        variance = sum((m - mean_memory)**2 for m in recent_memories) / len(recent_memories)
        
        # 方差越小，置信度越高，调整计算方式
        if variance == 0:
            confidence = 0.9  # 完全稳定
        else:
            # 使用相对方差来计算置信度
            relative_variance = variance / (mean_memory ** 2) if mean_memory > 0 else 1.0
            confidence = max(0.3, min(0.95, 1.0 - relative_variance))
        
        return confidence
    
    def _generate_recommendations(self, predicted_memory: int, 
                                total_memory: int) -> List[MemoryOptimizationRecommendation]:
        """生成优化建议"""
        recommendations = []
        utilization_rate = predicted_memory / total_memory
        
        if utilization_rate > 0.9:
            recommendations.append(MemoryOptimizationRecommendation(
                strategy=OptimizationStrategy.REDUCE_BATCH_SIZE,
                priority=10,
                description="预测内存使用率超过90%，强烈建议减小批次大小",
                expected_memory_saving=predicted_memory // 4,
                implementation_difficulty=1,
                side_effects=["训练时间可能增加", "梯度累积步数需要调整"]
            ))
            
            recommendations.append(MemoryOptimizationRecommendation(
                strategy=OptimizationStrategy.ENABLE_GRADIENT_CHECKPOINTING,
                priority=8,
                description="启用梯度检查点以减少激活值内存",
                expected_memory_saving=predicted_memory // 3,
                implementation_difficulty=2,
                side_effects=["训练速度降低约20%"]
            ))
        
        elif utilization_rate > 0.8:
            recommendations.append(MemoryOptimizationRecommendation(
                strategy=OptimizationStrategy.CLEAR_CACHE,
                priority=6,
                description="定期清理CUDA缓存",
                expected_memory_saving=500,
                implementation_difficulty=1,
                side_effects=["可能导致短暂的性能下降"]
            ))
        
        return recommendations
    
    def _create_default_prediction(self, batch_size: int, sequence_length: int,
                                 model_parameters: int) -> MemoryPrediction:
        """创建默认预测（无历史数据时）"""
        base_memory = self._estimate_base_memory(model_parameters)
        batch_memory = self._estimate_batch_memory(batch_size, sequence_length)
        overhead_memory = self._estimate_overhead_memory()
        
        predicted_memory = base_memory + batch_memory + overhead_memory
        
        return MemoryPrediction(
            predicted_peak_memory=predicted_memory,
            confidence_score=0.3,  # 低置信度
            prediction_horizon=300,
            factors={"model_size": 0.5, "batch_size": 0.3, "sequence_length": 0.2},
            recommendations=[]
        )


class MemoryManager:
    """动态内存管理器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化内存管理器"""
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.config = config or {}
        self.monitoring_interval = self.config.get("monitoring_interval", 5)
        self.enable_auto_adjustment = self.config.get("enable_auto_adjustment", True)
        self.enable_prediction = self.config.get("enable_prediction", True)
        
        # 组件初始化
        self.monitor = MemoryMonitor(self.monitoring_interval)
        self.batch_adjuster = DynamicBatchSizeAdjuster(
            initial_batch_size=self.config.get("initial_batch_size", 4),
            min_batch_size=self.config.get("min_batch_size", 1),
            max_batch_size=self.config.get("max_batch_size", 32)
        )
        self.predictor = MemoryPredictor()
        
        # 状态管理
        self.is_active = False
        self.optimization_callbacks: List[Callable[[List[MemoryOptimizationRecommendation]], None]] = []
        
        # 注册内存压力回调
        if self.enable_auto_adjustment:
            self.monitor.add_pressure_callback(self._handle_memory_pressure)
    
    def start(self) -> bool:
        """启动内存管理器"""
        if self.is_active:
            self.logger.warning("内存管理器已在运行")
            return False
        
        try:
            success = self.monitor.start_monitoring()
            if success:
                self.is_active = True
                self.logger.info("动态内存管理器已启动")
            return success
        except Exception as e:
            self.logger.error(f"启动内存管理器失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止内存管理器"""
        if not self.is_active:
            return True
        
        try:
            success = self.monitor.stop_monitoring()
            if success:
                self.is_active = False
                self.logger.info("动态内存管理器已停止")
            return success
        except Exception as e:
            self.logger.error(f"停止内存管理器失败: {e}")
            return False
    
    def _handle_memory_pressure(self, memory_snapshot: MemorySnapshot):
        """处理内存压力"""
        self.logger.warning(f"检测到GPU {memory_snapshot.gpu_id}内存压力: "
                          f"{memory_snapshot.pressure_level.value} "
                          f"({memory_snapshot.utilization_rate:.2%})")
        
        # 尝试调整批次大小
        adjustment = self.batch_adjuster.adjust_batch_size(memory_snapshot)
        if adjustment:
            self.logger.info(f"已自动调整批次大小: {adjustment.old_batch_size} -> {adjustment.new_batch_size}")
        
        # 生成优化建议
        if self.enable_prediction:
            memory_history = self.monitor.get_memory_history(memory_snapshot.gpu_id, 30)
            prediction = self.predictor.predict_memory_usage(
                memory_history, 
                self.batch_adjuster.get_current_batch_size(),
                2048,  # 默认序列长度
                4000000000  # Qwen3-4B参数量
            )
            
            if prediction.recommendations:
                for callback in self.optimization_callbacks:
                    try:
                        callback(prediction.recommendations)
                    except Exception as e:
                        self.logger.error(f"优化建议回调执行失败: {e}")
    
    def get_current_memory_status(self, gpu_id: int = 0) -> Optional[MemorySnapshot]:
        """获取当前内存状态"""
        return self.monitor.get_current_memory_status(gpu_id)
    
    def get_memory_analysis(self, gpu_id: int = 0, duration_minutes: int = 60) -> Dict[str, Any]:
        """获取内存分析报告"""
        memory_history = self.monitor.get_memory_history(gpu_id, duration_minutes)
        
        if not memory_history:
            return {"error": "没有内存历史数据"}
        
        # 统计分析
        utilizations = [s.utilization_rate for s in memory_history]
        allocated_memories = [s.allocated_memory for s in memory_history]
        
        analysis = {
            "gpu_id": gpu_id,
            "duration_minutes": duration_minutes,
            "sample_count": len(memory_history),
            "memory_statistics": {
                "avg_utilization": sum(utilizations) / len(utilizations),
                "max_utilization": max(utilizations),
                "min_utilization": min(utilizations),
                "avg_allocated_memory": sum(allocated_memories) / len(allocated_memories),
                "max_allocated_memory": max(allocated_memories),
                "min_allocated_memory": min(allocated_memories)
            },
            "pressure_distribution": {
                level.value: sum(1 for s in memory_history if s.pressure_level == level)
                for level in MemoryPressureLevel
            },
            "batch_adjustments": len(self.batch_adjuster.get_adjustment_history(duration_minutes // 60)),
            "current_batch_size": self.batch_adjuster.get_current_batch_size()
        }
        
        return analysis
    
    def predict_memory_usage(self, batch_size: int, sequence_length: int = 2048,
                           model_parameters: int = 4000000000) -> MemoryPrediction:
        """预测内存使用"""
        memory_history = self.monitor.get_memory_history(0, 30)  # 30分钟历史
        return self.predictor.predict_memory_usage(
            memory_history, batch_size, sequence_length, model_parameters
        )
    
    def optimize_memory(self, gpu_id: int = 0) -> List[MemoryOptimizationRecommendation]:
        """执行内存优化"""
        recommendations = []
        
        # 清理CUDA缓存
        try:
            torch.cuda.empty_cache()
            gc.collect()
            recommendations.append(MemoryOptimizationRecommendation(
                strategy=OptimizationStrategy.CLEAR_CACHE,
                priority=5,
                description="已清理CUDA缓存和Python垃圾回收",
                expected_memory_saving=0,  # 实际节省量难以预估
                implementation_difficulty=1
            ))
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
        
        # 获取当前内存状态
        current_status = self.get_current_memory_status(gpu_id)
        if current_status and current_status.utilization_rate > 0.8:
            # 建议减小批次大小
            recommendations.append(MemoryOptimizationRecommendation(
                strategy=OptimizationStrategy.REDUCE_BATCH_SIZE,
                priority=8,
                description=f"当前内存使用率{current_status.utilization_rate:.2%}，建议减小批次大小",
                expected_memory_saving=current_status.allocated_memory // 4,
                implementation_difficulty=1
            ))
        
        return recommendations
    
    def add_optimization_callback(self, callback: Callable[[List[MemoryOptimizationRecommendation]], None]):
        """添加优化建议回调"""
        self.optimization_callbacks.append(callback)
    
    def get_current_batch_size(self) -> int:
        """获取当前批次大小"""
        return self.batch_adjuster.get_current_batch_size()
    
    def set_batch_size(self, batch_size: int) -> bool:
        """设置批次大小"""
        try:
            self.batch_adjuster.current_batch_size = max(
                self.batch_adjuster.min_batch_size,
                min(self.batch_adjuster.max_batch_size, batch_size)
            )
            return True
        except Exception as e:
            self.logger.error(f"设置批次大小失败: {e}")
            return False
    
    def export_memory_report(self, output_path: str, gpu_id: int = 0, 
                           duration_hours: int = 24) -> bool:
        """导出内存报告"""
        try:
            # 收集数据
            memory_history = self.monitor.get_memory_history(gpu_id, duration_hours * 60)
            analysis = self.get_memory_analysis(gpu_id, duration_hours * 60)
            adjustment_history = self.batch_adjuster.get_adjustment_history(duration_hours)
            
            # 构建报告
            report = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "gpu_id": gpu_id,
                    "duration_hours": duration_hours,
                    "total_samples": len(memory_history)
                },
                "memory_analysis": analysis,
                "memory_history": [snapshot.to_dict() for snapshot in memory_history],
                "batch_adjustments": [adj.to_dict() for adj in adjustment_history],
                "configuration": self.config
            }
            
            # 保存报告
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"内存报告已导出: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出内存报告失败: {e}")
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()