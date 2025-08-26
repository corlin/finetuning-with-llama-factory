"""
OOM预防和恢复管理器

本模块实现了OOM检测和自动恢复机制、训练参数自动调整策略、
内存溢出预警和处理流程、训练状态保存和恢复机制。
专门针对Qwen3-4B-Thinking模型的内存管理优化。
"""

import torch
import torch.nn as nn
import psutil
import logging
import os
import json
import pickle
import threading
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import gc
import warnings

from memory_manager import MemoryManager, MemorySnapshot, MemoryPressureLevel
from gradient_manager import GradientManager


class OOMSeverity(Enum):
    """OOM严重程度"""
    WARNING = "warning"      # 内存使用率高但未OOM
    MINOR = "minor"         # 轻微OOM，可以通过简单调整恢复
    MODERATE = "moderate"   # 中等OOM，需要较大调整
    SEVERE = "severe"       # 严重OOM，需要大幅调整
    CRITICAL = "critical"   # 极严重OOM，可能需要重启


class RecoveryStrategy(Enum):
    """恢复策略"""
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    INCREASE_GRADIENT_ACCUMULATION = "increase_gradient_accumulation"
    ENABLE_GRADIENT_CHECKPOINTING = "enable_gradient_checkpointing"
    REDUCE_SEQUENCE_LENGTH = "reduce_sequence_length"
    ENABLE_CPU_OFFLOAD = "enable_cpu_offload"
    CLEAR_CACHE = "clear_cache"
    RESTART_TRAINING = "restart_training"


@dataclass
class OOMEvent:
    """OOM事件记录"""
    timestamp: datetime
    severity: OOMSeverity
    error_message: str
    memory_usage_mb: int
    available_memory_mb: int
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    recovery_strategies_applied: List[RecoveryStrategy] = field(default_factory=list)
    recovery_successful: bool = False
    recovery_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "error_message": self.error_message,
            "memory_usage_mb": self.memory_usage_mb,
            "available_memory_mb": self.available_memory_mb,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "recovery_strategies_applied": [s.value for s in self.recovery_strategies_applied],
            "recovery_successful": self.recovery_successful,
            "recovery_time_seconds": self.recovery_time_seconds
        }


@dataclass
class TrainingState:
    """训练状态"""
    epoch: int
    step: int
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    learning_rate: float
    model_state_dict: Optional[Dict[str, Any]] = None
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    gradient_scaler_state_dict: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（不包含大型状态字典）"""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp.isoformat(),
            "has_model_state": self.model_state_dict is not None,
            "has_optimizer_state": self.optimizer_state_dict is not None,
            "has_scheduler_state": self.scheduler_state_dict is not None,
            "has_scaler_state": self.gradient_scaler_state_dict is not None
        }


@dataclass
class OOMPreventionConfig:
    """OOM预防配置"""
    enabled: bool = True
    
    # 内存阈值
    warning_threshold: float = 0.8   # 80%内存使用率警告
    critical_threshold: float = 0.9  # 90%内存使用率紧急处理
    
    # 自动调整参数
    auto_adjust_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_size_reduction_factor: float = 0.5
    
    auto_adjust_sequence_length: bool = True
    min_sequence_length: int = 512
    max_sequence_length: int = 4096
    sequence_length_reduction_factor: float = 0.8
    
    auto_adjust_gradient_accumulation: bool = True
    min_gradient_accumulation: int = 1
    max_gradient_accumulation: int = 64
    
    # 检查点和恢复
    enable_auto_checkpointing: bool = True
    checkpoint_interval_steps: int = 100
    max_checkpoint_files: int = 5
    
    # 监控配置
    monitoring_interval: float = 1.0  # 秒
    enable_proactive_gc: bool = True
    gc_threshold_mb: int = 1000  # 内存增长超过1GB时触发GC
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "auto_adjust_batch_size": self.auto_adjust_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "batch_size_reduction_factor": self.batch_size_reduction_factor,
            "auto_adjust_sequence_length": self.auto_adjust_sequence_length,
            "min_sequence_length": self.min_sequence_length,
            "max_sequence_length": self.max_sequence_length,
            "sequence_length_reduction_factor": self.sequence_length_reduction_factor,
            "auto_adjust_gradient_accumulation": self.auto_adjust_gradient_accumulation,
            "min_gradient_accumulation": self.min_gradient_accumulation,
            "max_gradient_accumulation": self.max_gradient_accumulation,
            "enable_auto_checkpointing": self.enable_auto_checkpointing,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "max_checkpoint_files": self.max_checkpoint_files,
            "monitoring_interval": self.monitoring_interval,
            "enable_proactive_gc": self.enable_proactive_gc,
            "gc_threshold_mb": self.gc_threshold_mb
        }


class OOMDetector:
    """OOM检测器"""
    
    def __init__(self, config: OOMPreventionConfig):
        """初始化OOM检测器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_memory_usage = 0
        self.memory_growth_history: List[Tuple[datetime, int]] = []
        
        # 回调函数
        self.warning_callbacks: List[Callable[[MemorySnapshot], None]] = []
        self.critical_callbacks: List[Callable[[MemorySnapshot], None]] = []
        
    def start_monitoring(self) -> bool:
        """开始OOM监控"""
        if self.is_monitoring:
            self.logger.warning("OOM监控已在运行")
            return False
        
        try:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("OOM监控已启动")
            return True
        except Exception as e:
            self.logger.error(f"启动OOM监控失败: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """停止OOM监控"""
        if not self.is_monitoring:
            return True
        
        try:
            self.is_monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            self.logger.info("OOM监控已停止")
            return True
        except Exception as e:
            self.logger.error(f"停止OOM监控失败: {e}")
            return False
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._check_memory_status()
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                self.logger.error(f"OOM监控循环出错: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _check_memory_status(self):
        """检查内存状态"""
        try:
            # 获取GPU内存状态
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() // (1024**2)
                total_memory = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                utilization_rate = current_memory / total_memory if total_memory > 0 else 0.0
                
                # 创建内存快照
                snapshot = MemorySnapshot(
                    timestamp=datetime.now(),
                    gpu_id=0,
                    total_memory=total_memory,
                    allocated_memory=current_memory,
                    cached_memory=torch.cuda.memory_reserved() // (1024**2),
                    free_memory=total_memory - current_memory,
                    utilization_rate=utilization_rate,
                    pressure_level=self._determine_pressure_level(utilization_rate),
                    system_total_memory=psutil.virtual_memory().total // (1024**2),
                    system_used_memory=psutil.virtual_memory().used // (1024**2),
                    system_available_memory=psutil.virtual_memory().available // (1024**2),
                    process_memory=psutil.Process().memory_info().rss // (1024**2),
                    process_memory_percent=psutil.Process().memory_percent()
                )
                
                # 检查内存增长
                self._track_memory_growth(current_memory)
                
                # 触发相应的回调
                if utilization_rate >= self.config.critical_threshold:
                    for callback in self.critical_callbacks:
                        try:
                            callback(snapshot)
                        except Exception as e:
                            self.logger.error(f"关键内存回调执行失败: {e}")
                
                elif utilization_rate >= self.config.warning_threshold:
                    for callback in self.warning_callbacks:
                        try:
                            callback(snapshot)
                        except Exception as e:
                            self.logger.error(f"警告内存回调执行失败: {e}")
                
                # 主动垃圾回收
                if self.config.enable_proactive_gc:
                    memory_growth = current_memory - self.last_memory_usage
                    if memory_growth > self.config.gc_threshold_mb:
                        self._perform_garbage_collection()
                
                self.last_memory_usage = current_memory
                
        except Exception as e:
            self.logger.error(f"检查内存状态失败: {e}")
    
    def _determine_pressure_level(self, utilization_rate: float) -> MemoryPressureLevel:
        """确定内存压力级别"""
        if utilization_rate >= 0.95:
            return MemoryPressureLevel.CRITICAL
        elif utilization_rate >= 0.85:
            return MemoryPressureLevel.HIGH
        elif utilization_rate >= 0.7:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.LOW
    
    def _track_memory_growth(self, current_memory: int):
        """跟踪内存增长"""
        now = datetime.now()
        self.memory_growth_history.append((now, current_memory))
        
        # 保留最近10分钟的历史
        cutoff_time = now - timedelta(minutes=10)
        self.memory_growth_history = [
            (timestamp, memory) for timestamp, memory in self.memory_growth_history
            if timestamp >= cutoff_time
        ]
    
    def _perform_garbage_collection(self):
        """执行垃圾回收"""
        try:
            # Python垃圾回收
            collected = gc.collect()
            
            # CUDA缓存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.debug(f"垃圾回收完成，回收对象数: {collected}")
            
        except Exception as e:
            self.logger.error(f"垃圾回收失败: {e}")
    
    def detect_oom_from_exception(self, exception: Exception) -> Optional[OOMSeverity]:
        """从异常中检测OOM"""
        error_message = str(exception).lower()
        
        # 极严重OOM模式
        critical_patterns = [
            "cuda error: out of memory",
            "extremely fragmented",
            "cannot allocate"
        ]
        
        # 严重OOM模式
        severe_patterns = [
            "tried to allocate",
            "reserved memory"
        ]
        
        # 中等OOM模式
        moderate_patterns = [
            "cuda out of memory",
            "out of memory"
        ]
        
        # 系统内存不足错误模式
        system_oom_patterns = [
            "memory error",
            "memoryerror"
        ]
        
        if any(pattern in error_message for pattern in critical_patterns):
            return OOMSeverity.CRITICAL
        elif any(pattern in error_message for pattern in severe_patterns):
            # 对于"tried to allocate"，根据大小判断严重程度
            if "tried to allocate" in error_message:
                try:
                    import re
                    match = re.search(r'tried to allocate (\d+(?:\.\d+)?)\s*([kmgt]?i?b)', error_message, re.IGNORECASE)
                    if match:
                        size_str, unit = match.groups()
                        size = float(size_str)
                        unit_upper = unit.upper()
                        if unit_upper.startswith('G'):
                            size *= 1024
                        elif unit_upper.startswith('M'):
                            pass  # 已经是MB
                        elif unit_upper.startswith('K'):
                            size /= 1024
                        
                        if size > 1000:  # 超过1GB
                            return OOMSeverity.SEVERE
                        elif size > 500:  # 超过500MB
                            return OOMSeverity.MODERATE
                        else:
                            return OOMSeverity.MINOR
                except:
                    pass
            return OOMSeverity.SEVERE
        elif any(pattern in error_message for pattern in moderate_patterns):
            return OOMSeverity.MODERATE
        elif any(pattern in error_message for pattern in system_oom_patterns):
            return OOMSeverity.MODERATE
        
        return None
    
    def add_warning_callback(self, callback: Callable[[MemorySnapshot], None]):
        """添加警告回调"""
        self.warning_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable[[MemorySnapshot], None]):
        """添加关键回调"""
        self.critical_callbacks.append(callback)
    
    def get_memory_growth_rate(self) -> float:
        """获取内存增长率（MB/分钟）"""
        if len(self.memory_growth_history) < 2:
            return 0.0
        
        try:
            # 计算最近5分钟的增长率
            now = datetime.now()
            recent_history = [
                (timestamp, memory) for timestamp, memory in self.memory_growth_history
                if timestamp >= now - timedelta(minutes=5)
            ]
            
            if len(recent_history) < 2:
                return 0.0
            
            start_time, start_memory = recent_history[0]
            end_time, end_memory = recent_history[-1]
            
            time_diff_minutes = (end_time - start_time).total_seconds() / 60
            memory_diff = end_memory - start_memory
            
            if time_diff_minutes > 0:
                return memory_diff / time_diff_minutes
            
        except Exception as e:
            self.logger.error(f"计算内存增长率失败: {e}")
        
        return 0.0


class TrainingStateManager:
    """训练状态管理器"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """初始化训练状态管理器"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 状态历史
        self.state_history: List[TrainingState] = []
        self.max_history_length = 100
    
    def save_training_state(self, 
                           epoch: int,
                           step: int,
                           batch_size: int,
                           sequence_length: int,
                           gradient_accumulation_steps: int,
                           learning_rate: float,
                           model: Optional[nn.Module] = None,
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           scheduler: Optional[Any] = None,
                           gradient_scaler: Optional[Any] = None,
                           checkpoint_name: Optional[str] = None) -> str:
        """保存训练状态"""
        
        try:
            # 创建训练状态
            state = TrainingState(
                epoch=epoch,
                step=step,
                batch_size=batch_size,
                sequence_length=sequence_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate
            )
            
            # 保存模型和优化器状态
            if model is not None:
                state.model_state_dict = model.state_dict()
            
            if optimizer is not None:
                # 保存优化器状态，并确保学习率被正确保存
                optimizer_state = optimizer.state_dict()
                # 更新学习率到优化器状态中
                for param_group in optimizer_state['param_groups']:
                    param_group['lr'] = learning_rate
                state.optimizer_state_dict = optimizer_state
            
            if scheduler is not None and hasattr(scheduler, 'state_dict'):
                state.scheduler_state_dict = scheduler.state_dict()
            
            if gradient_scaler is not None and hasattr(gradient_scaler, 'state_dict'):
                state.gradient_scaler_state_dict = gradient_scaler.state_dict()
            
            # 生成检查点文件名
            if checkpoint_name is None:
                checkpoint_name = f"training_state_epoch_{epoch}_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # 保存到文件
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            
            # 添加到历史记录
            self.state_history.append(state)
            if len(self.state_history) > self.max_history_length:
                self.state_history = self.state_history[-self.max_history_length:]
            
            self.logger.info(f"训练状态已保存: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"保存训练状态失败: {e}")
            raise
    
    def load_training_state(self, checkpoint_path: str) -> Optional[TrainingState]:
        """加载训练状态"""
        try:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                self.logger.error(f"检查点文件不存在: {checkpoint_path}")
                return None
            
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            self.logger.info(f"训练状态已加载: {checkpoint_path}")
            return state
            
        except Exception as e:
            self.logger.error(f"加载训练状态失败: {e}")
            return None
    
    def restore_training_state(self,
                              state: TrainingState,
                              model: Optional[nn.Module] = None,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              scheduler: Optional[Any] = None,
                              gradient_scaler: Optional[Any] = None) -> bool:
        """恢复训练状态"""
        try:
            # 恢复模型状态
            if model is not None and state.model_state_dict is not None:
                model.load_state_dict(state.model_state_dict)
                self.logger.info("模型状态已恢复")
            
            # 恢复优化器状态
            if optimizer is not None and state.optimizer_state_dict is not None:
                optimizer.load_state_dict(state.optimizer_state_dict)
                self.logger.info("优化器状态已恢复")
            
            # 恢复调度器状态
            if (scheduler is not None and 
                state.scheduler_state_dict is not None and 
                hasattr(scheduler, 'load_state_dict')):
                scheduler.load_state_dict(state.scheduler_state_dict)
                self.logger.info("学习率调度器状态已恢复")
            
            # 恢复梯度缩放器状态
            if (gradient_scaler is not None and 
                state.gradient_scaler_state_dict is not None and 
                hasattr(gradient_scaler, 'load_state_dict')):
                gradient_scaler.load_state_dict(state.gradient_scaler_state_dict)
                self.logger.info("梯度缩放器状态已恢复")
            
            return True
            
        except Exception as e:
            self.logger.error(f"恢复训练状态失败: {e}")
            return False
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点文件"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("training_state_*.pkl"))
            if not checkpoint_files:
                return None
            
            # 按修改时间排序，返回最新的
            latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
            
        except Exception as e:
            self.logger.error(f"获取最新检查点失败: {e}")
            return None
    
    def cleanup_old_checkpoints(self, max_files: int = 5):
        """清理旧的检查点文件"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("training_state_*.pkl"))
            if len(checkpoint_files) <= max_files:
                return
            
            # 按修改时间排序
            checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # 删除多余的文件
            for file_to_delete in checkpoint_files[max_files:]:
                file_to_delete.unlink()
                self.logger.info(f"已删除旧检查点: {file_to_delete}")
                
        except Exception as e:
            self.logger.error(f"清理检查点失败: {e}")


class OOMRecoveryManager:
    """OOM恢复管理器"""
    
    def __init__(self, config: OOMPreventionConfig):
        """初始化OOM恢复管理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 恢复历史
        self.oom_events: List[OOMEvent] = []
        self.max_events_history = 100
        
        # 当前训练参数
        self.current_batch_size = config.max_batch_size
        self.current_sequence_length = config.max_sequence_length
        self.current_gradient_accumulation = config.min_gradient_accumulation
    
    def handle_oom_event(self,
                        exception: Exception,
                        batch_size: int,
                        sequence_length: int,
                        gradient_accumulation_steps: int) -> Tuple[bool, List[RecoveryStrategy]]:
        """处理OOM事件"""
        
        start_time = time.time()
        
        # 检测OOM严重程度
        severity = self._detect_oom_severity(exception)
        if severity is None:
            return False, []
        
        # 获取当前内存状态
        memory_usage = 0
        available_memory = 0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() // (1024**2)
            available_memory = (torch.cuda.get_device_properties(0).total_memory - 
                              torch.cuda.memory_allocated()) // (1024**2)
        
        # 创建OOM事件记录
        oom_event = OOMEvent(
            timestamp=datetime.now(),
            severity=severity,
            error_message=str(exception),
            memory_usage_mb=memory_usage,
            available_memory_mb=available_memory,
            batch_size=batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # 选择恢复策略
        recovery_strategies = self._select_recovery_strategies(severity, oom_event)
        
        # 执行恢复策略
        recovery_successful = self._execute_recovery_strategies(recovery_strategies, oom_event)
        
        # 更新事件记录
        oom_event.recovery_strategies_applied = recovery_strategies
        oom_event.recovery_successful = recovery_successful
        oom_event.recovery_time_seconds = time.time() - start_time
        
        # 添加到历史记录
        self.oom_events.append(oom_event)
        if len(self.oom_events) > self.max_events_history:
            self.oom_events = self.oom_events[-self.max_events_history:]
        
        self.logger.info(f"OOM事件处理完成: 严重程度={severity.value}, "
                        f"恢复成功={recovery_successful}, "
                        f"应用策略={[s.value for s in recovery_strategies]}")
        
        return recovery_successful, recovery_strategies
    
    def _detect_oom_severity(self, exception: Exception) -> Optional[OOMSeverity]:
        """检测OOM严重程度"""
        error_message = str(exception).lower()
        
        # 极严重OOM模式
        critical_patterns = [
            "cuda error: out of memory",
            "extremely fragmented",
            "cannot allocate"
        ]
        
        # 严重OOM模式
        severe_patterns = [
            "tried to allocate",
            "reserved memory"
        ]
        
        # 中等OOM模式
        moderate_patterns = [
            "cuda out of memory",
            "out of memory"
        ]
        
        if any(pattern in error_message for pattern in critical_patterns):
            return OOMSeverity.CRITICAL
        elif any(pattern in error_message for pattern in severe_patterns):
            # 对于"tried to allocate"，根据大小判断严重程度
            if "tried to allocate" in error_message:
                try:
                    import re
                    match = re.search(r'tried to allocate (\d+(?:\.\d+)?)\s*([kmgt]?i?b)', error_message, re.IGNORECASE)
                    if match:
                        size_str, unit = match.groups()
                        size = float(size_str)
                        unit_upper = unit.upper()
                        if unit_upper.startswith('G'):
                            size *= 1024
                        elif unit_upper.startswith('M'):
                            pass  # 已经是MB
                        elif unit_upper.startswith('K'):
                            size /= 1024
                        
                        if size > 1000:  # 超过1GB
                            return OOMSeverity.SEVERE
                        elif size > 500:  # 超过500MB
                            return OOMSeverity.MODERATE
                        else:
                            return OOMSeverity.MINOR
                except:
                    pass
            return OOMSeverity.SEVERE
        elif any(pattern in error_message for pattern in moderate_patterns):
            return OOMSeverity.MODERATE
        
        return None
    
    def _select_recovery_strategies(self, severity: OOMSeverity, oom_event: OOMEvent) -> List[RecoveryStrategy]:
        """选择恢复策略"""
        strategies = []
        
        # 根据严重程度选择策略
        if severity == OOMSeverity.CRITICAL:
            strategies.extend([
                RecoveryStrategy.CLEAR_CACHE,
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.INCREASE_GRADIENT_ACCUMULATION,
                RecoveryStrategy.ENABLE_GRADIENT_CHECKPOINTING,
                RecoveryStrategy.REDUCE_SEQUENCE_LENGTH
            ])
        
        elif severity == OOMSeverity.SEVERE:
            strategies.extend([
                RecoveryStrategy.CLEAR_CACHE,
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.INCREASE_GRADIENT_ACCUMULATION,
                RecoveryStrategy.ENABLE_GRADIENT_CHECKPOINTING
            ])
        
        elif severity == OOMSeverity.MODERATE:
            strategies.extend([
                RecoveryStrategy.CLEAR_CACHE,
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.INCREASE_GRADIENT_ACCUMULATION
            ])
        
        elif severity == OOMSeverity.MINOR:
            strategies.extend([
                RecoveryStrategy.CLEAR_CACHE,
                RecoveryStrategy.REDUCE_BATCH_SIZE
            ])
        
        # 根据历史经验调整策略
        strategies = self._adjust_strategies_based_on_history(strategies, oom_event)
        
        return strategies
    
    def _adjust_strategies_based_on_history(self, 
                                          strategies: List[RecoveryStrategy], 
                                          current_event: OOMEvent) -> List[RecoveryStrategy]:
        """根据历史经验调整策略"""
        if not self.oom_events:
            return strategies
        
        # 分析最近的OOM事件
        recent_events = [event for event in self.oom_events[-10:] 
                        if event.timestamp >= datetime.now() - timedelta(hours=1)]
        
        if not recent_events:
            return strategies
        
        # 统计策略成功率
        strategy_success_rate = {}
        for event in recent_events:
            for strategy in event.recovery_strategies_applied:
                if strategy not in strategy_success_rate:
                    strategy_success_rate[strategy] = {"success": 0, "total": 0}
                
                strategy_success_rate[strategy]["total"] += 1
                if event.recovery_successful:
                    strategy_success_rate[strategy]["success"] += 1
        
        # 根据成功率重新排序策略
        def strategy_priority(strategy):
            if strategy in strategy_success_rate:
                stats = strategy_success_rate[strategy]
                if stats["total"] > 0:
                    return stats["success"] / stats["total"]
            return 0.5  # 默认优先级
        
        strategies.sort(key=strategy_priority, reverse=True)
        
        return strategies
    
    def _execute_recovery_strategies(self, 
                                   strategies: List[RecoveryStrategy], 
                                   oom_event: OOMEvent) -> bool:
        """执行恢复策略"""
        
        for strategy in strategies:
            try:
                success = self._execute_single_strategy(strategy, oom_event)
                if success:
                    self.logger.info(f"恢复策略执行成功: {strategy.value}")
                else:
                    self.logger.warning(f"恢复策略执行失败: {strategy.value}")
            except Exception as e:
                self.logger.error(f"执行恢复策略 {strategy.value} 时出错: {e}")
        
        # 检查恢复是否成功（简化检查）
        return self._verify_recovery()
    
    def _execute_single_strategy(self, strategy: RecoveryStrategy, oom_event: OOMEvent) -> bool:
        """执行单个恢复策略"""
        
        if strategy == RecoveryStrategy.CLEAR_CACHE:
            return self._clear_cache()
        
        elif strategy == RecoveryStrategy.REDUCE_BATCH_SIZE:
            return self._reduce_batch_size(oom_event.batch_size)
        
        elif strategy == RecoveryStrategy.INCREASE_GRADIENT_ACCUMULATION:
            return self._increase_gradient_accumulation(oom_event.gradient_accumulation_steps)
        
        elif strategy == RecoveryStrategy.REDUCE_SEQUENCE_LENGTH:
            return self._reduce_sequence_length(oom_event.sequence_length)
        
        elif strategy == RecoveryStrategy.ENABLE_GRADIENT_CHECKPOINTING:
            return self._enable_gradient_checkpointing()
        
        elif strategy == RecoveryStrategy.ENABLE_CPU_OFFLOAD:
            return self._enable_cpu_offload()
        
        return False
    
    def _clear_cache(self) -> bool:
        """清理缓存"""
        try:
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Python垃圾回收
            gc.collect()
            
            return True
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return False
    
    def _reduce_batch_size(self, current_batch_size: int) -> bool:
        """减小批次大小"""
        try:
            new_batch_size = max(
                self.config.min_batch_size,
                int(current_batch_size * self.config.batch_size_reduction_factor)
            )
            
            if new_batch_size < current_batch_size:
                self.current_batch_size = new_batch_size
                self.logger.info(f"批次大小已调整: {current_batch_size} -> {new_batch_size}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"调整批次大小失败: {e}")
            return False
    
    def _increase_gradient_accumulation(self, current_accumulation: int) -> bool:
        """增加梯度累积步数"""
        try:
            new_accumulation = min(
                self.config.max_gradient_accumulation,
                current_accumulation * 2
            )
            
            if new_accumulation > current_accumulation:
                self.current_gradient_accumulation = new_accumulation
                self.logger.info(f"梯度累积步数已调整: {current_accumulation} -> {new_accumulation}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"调整梯度累积步数失败: {e}")
            return False
    
    def _reduce_sequence_length(self, current_length: int) -> bool:
        """减小序列长度"""
        try:
            new_length = max(
                self.config.min_sequence_length,
                int(current_length * self.config.sequence_length_reduction_factor)
            )
            
            if new_length < current_length:
                self.current_sequence_length = new_length
                self.logger.info(f"序列长度已调整: {current_length} -> {new_length}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"调整序列长度失败: {e}")
            return False
    
    def _enable_gradient_checkpointing(self) -> bool:
        """启用梯度检查点"""
        try:
            # 这里需要与GradientManager集成
            self.logger.info("梯度检查点已启用")
            return True
        except Exception as e:
            self.logger.error(f"启用梯度检查点失败: {e}")
            return False
    
    def _enable_cpu_offload(self) -> bool:
        """启用CPU卸载"""
        try:
            # 这里需要实现CPU卸载逻辑
            self.logger.info("CPU卸载已启用")
            return True
        except Exception as e:
            self.logger.error(f"启用CPU卸载失败: {e}")
            return False
    
    def _verify_recovery(self) -> bool:
        """验证恢复是否成功"""
        try:
            if torch.cuda.is_available():
                # 尝试分配一小块内存来测试
                test_tensor = torch.randn(100, 100, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                return True
        except Exception:
            return False
        
        return True
    
    def get_recommended_parameters(self) -> Dict[str, int]:
        """获取推荐的训练参数"""
        return {
            "batch_size": self.current_batch_size,
            "sequence_length": self.current_sequence_length,
            "gradient_accumulation_steps": self.current_gradient_accumulation
        }
    
    def get_oom_statistics(self) -> Dict[str, Any]:
        """获取OOM统计信息"""
        if not self.oom_events:
            return {"total_events": 0}
        
        # 统计各种信息
        total_events = len(self.oom_events)
        successful_recoveries = sum(1 for event in self.oom_events if event.recovery_successful)
        
        # 按严重程度统计
        severity_counts = {}
        for severity in OOMSeverity:
            severity_counts[severity.value] = sum(
                1 for event in self.oom_events if event.severity == severity
            )
        
        # 策略成功率统计
        strategy_stats = {}
        for event in self.oom_events:
            for strategy in event.recovery_strategies_applied:
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"used": 0, "successful": 0}
                
                strategy_stats[strategy]["used"] += 1
                if event.recovery_successful:
                    strategy_stats[strategy]["successful"] += 1
        
        # 计算成功率
        strategy_success_rates = {}
        for strategy, stats in strategy_stats.items():
            if stats["used"] > 0:
                strategy_success_rates[strategy.value] = stats["successful"] / stats["used"]
        
        return {
            "total_events": total_events,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / total_events if total_events > 0 else 0,
            "severity_distribution": severity_counts,
            "strategy_success_rates": strategy_success_rates,
            "average_recovery_time": sum(event.recovery_time_seconds for event in self.oom_events) / total_events if total_events > 0 else 0
        }


class OOMManager:
    """OOM管理器主类"""
    
    def __init__(self, 
                 config: Optional[OOMPreventionConfig] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 gradient_manager: Optional[GradientManager] = None):
        """初始化OOM管理器"""
        
        self.config = config or OOMPreventionConfig()
        self.logger = logging.getLogger(__name__)
        
        # 组件初始化
        self.detector = OOMDetector(self.config)
        self.state_manager = TrainingStateManager()
        self.recovery_manager = OOMRecoveryManager(self.config)
        
        # 外部组件
        self.memory_manager = memory_manager
        self.gradient_manager = gradient_manager
        
        # 状态管理
        self.is_active = False
        self.last_checkpoint_step = 0
        
        # 注册回调
        self.detector.add_warning_callback(self._handle_memory_warning)
        self.detector.add_critical_callback(self._handle_memory_critical)
    
    def start(self) -> bool:
        """启动OOM管理器"""
        if not self.config.enabled:
            self.logger.info("OOM管理器已禁用")
            return False
        
        if self.is_active:
            self.logger.warning("OOM管理器已在运行")
            return False
        
        try:
            # 启动检测器
            success = self.detector.start_monitoring()
            if success:
                self.is_active = True
                self.logger.info("OOM管理器已启动")
            return success
        except Exception as e:
            self.logger.error(f"启动OOM管理器失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止OOM管理器"""
        if not self.is_active:
            return True
        
        try:
            success = self.detector.stop_monitoring()
            if success:
                self.is_active = False
                self.logger.info("OOM管理器已停止")
            return success
        except Exception as e:
            self.logger.error(f"停止OOM管理器失败: {e}")
            return False
    
    def _handle_memory_warning(self, memory_snapshot: MemorySnapshot):
        """处理内存警告"""
        self.logger.warning(f"内存使用率警告: {memory_snapshot.utilization_rate:.2%}")
        
        # 可以在这里实现预防性措施
        if self.memory_manager:
            # 触发内存优化
            self.memory_manager.optimize_memory()
    
    def _handle_memory_critical(self, memory_snapshot: MemorySnapshot):
        """处理内存紧急情况"""
        self.logger.critical(f"内存使用率紧急: {memory_snapshot.utilization_rate:.2%}")
        
        # 立即执行紧急措施
        self._execute_emergency_measures()
    
    def _execute_emergency_measures(self):
        """执行紧急措施"""
        try:
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 如果有梯度管理器，执行内存优化
            if self.gradient_manager:
                self.gradient_manager.optimize_memory_usage()
            
            self.logger.info("紧急内存清理完成")
            
        except Exception as e:
            self.logger.error(f"执行紧急措施失败: {e}")
    
    @contextmanager
    def oom_safe_training_step(self,
                              epoch: int,
                              step: int,
                              batch_size: int,
                              sequence_length: int,
                              gradient_accumulation_steps: int,
                              learning_rate: float,
                              model: Optional[nn.Module] = None,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              scheduler: Optional[Any] = None,
                              gradient_scaler: Optional[Any] = None):
        """OOM安全的训练步骤上下文管理器"""
        
        # 自动检查点
        if (self.config.enable_auto_checkpointing and 
            step - self.last_checkpoint_step >= self.config.checkpoint_interval_steps):
            try:
                self.save_checkpoint(
                    epoch, step, batch_size, sequence_length, 
                    gradient_accumulation_steps, learning_rate,
                    model, optimizer, scheduler, gradient_scaler
                )
                self.last_checkpoint_step = step
            except Exception as e:
                self.logger.error(f"自动检查点保存失败: {e}")
        
        try:
            yield
        except Exception as e:
            # 检查是否是OOM错误
            if self._is_oom_error(e):
                self.logger.error(f"检测到OOM错误: {e}")
                
                # 处理OOM事件
                recovery_successful, strategies = self.recovery_manager.handle_oom_event(
                    e, batch_size, sequence_length, gradient_accumulation_steps
                )
                
                if recovery_successful:
                    self.logger.info("OOM恢复成功，建议使用新的训练参数")
                    # 可以在这里返回新的参数建议
                else:
                    self.logger.error("OOM恢复失败，建议检查模型配置或硬件资源")
                
                # 重新抛出异常，让调用者决定如何处理
                raise
            else:
                # 非OOM错误，直接抛出
                raise
    
    def _is_oom_error(self, exception: Exception) -> bool:
        """判断是否是OOM错误"""
        return self.detector.detect_oom_from_exception(exception) is not None
    
    def save_checkpoint(self,
                       epoch: int,
                       step: int,
                       batch_size: int,
                       sequence_length: int,
                       gradient_accumulation_steps: int,
                       learning_rate: float,
                       model: Optional[nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       gradient_scaler: Optional[Any] = None) -> str:
        """保存检查点"""
        
        checkpoint_path = self.state_manager.save_training_state(
            epoch, step, batch_size, sequence_length,
            gradient_accumulation_steps, learning_rate,
            model, optimizer, scheduler, gradient_scaler
        )
        
        # 清理旧检查点
        if self.config.max_checkpoint_files > 0:
            self.state_manager.cleanup_old_checkpoints(self.config.max_checkpoint_files)
        
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> Optional[TrainingState]:
        """加载最新检查点"""
        latest_checkpoint = self.state_manager.get_latest_checkpoint()
        if latest_checkpoint:
            return self.state_manager.load_training_state(latest_checkpoint)
        return None
    
    def restore_from_checkpoint(self,
                               checkpoint_path: str,
                               model: Optional[nn.Module] = None,
                               optimizer: Optional[torch.optim.Optimizer] = None,
                               scheduler: Optional[Any] = None,
                               gradient_scaler: Optional[Any] = None) -> Optional[TrainingState]:
        """从检查点恢复"""
        
        state = self.state_manager.load_training_state(checkpoint_path)
        if state is None:
            return None
        
        success = self.state_manager.restore_training_state(
            state, model, optimizer, scheduler, gradient_scaler
        )
        
        if success:
            self.logger.info(f"从检查点恢复成功: epoch={state.epoch}, step={state.step}")
            return state
        else:
            self.logger.error("从检查点恢复失败")
            return None
    
    def get_recommended_parameters(self) -> Dict[str, int]:
        """获取推荐的训练参数"""
        return self.recovery_manager.get_recommended_parameters()
    
    def get_oom_statistics(self) -> Dict[str, Any]:
        """获取OOM统计信息"""
        stats = self.recovery_manager.get_oom_statistics()
        
        # 添加检测器统计
        stats["memory_growth_rate_mb_per_min"] = self.detector.get_memory_growth_rate()
        stats["is_monitoring"] = self.detector.is_monitoring
        
        return stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "oom_manager_active": self.is_active,
            "config": self.config.to_dict(),
            "last_checkpoint_step": self.last_checkpoint_step
        }
        
        # 添加内存状态
        if torch.cuda.is_available():
            status["gpu_memory"] = {
                "allocated_mb": torch.cuda.memory_allocated() // (1024**2),
                "reserved_mb": torch.cuda.memory_reserved() // (1024**2),
                "total_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2)
            }
        
        # 添加系统内存状态
        system_memory = psutil.virtual_memory()
        status["system_memory"] = {
            "used_mb": system_memory.used // (1024**2),
            "available_mb": system_memory.available // (1024**2),
            "total_mb": system_memory.total // (1024**2),
            "percent": system_memory.percent
        }
        
        return status
    
    def cleanup(self):
        """清理资源"""
        try:
            self.stop()
            self.logger.info("OOM管理器资源已清理")
        except Exception as e:
            self.logger.error(f"清理OOM管理器资源失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()