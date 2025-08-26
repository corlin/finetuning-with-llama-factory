"""
训练监控核心模块

本模块实现了多GPU训练状态跟踪、实时损失曲线和学习率监控、训练进度估算和收敛检测算法、
GPU利用率和内存使用监控等功能。支持分布式训练环境下的综合监控和异常检测。
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import logging
from pathlib import Path
import numpy as np
import torch
import psutil

from src.data_models import ChineseMetrics
from src.parallel_config import DistributedTrainingMetrics, GPUInfo, CommunicationMetrics


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """训练状态数据结构"""
    epoch: int = 0
    global_step: int = 0
    local_step: int = 0
    is_training: bool = False
    start_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    
    # 损失历史
    train_loss_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    val_loss_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # 学习率历史
    learning_rate_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # 性能指标
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_loss(self, train_loss: float, val_loss: Optional[float] = None):
        """更新损失值"""
        self.train_loss_history.append((self.global_step, train_loss))
        if val_loss is not None:
            self.val_loss_history.append((self.global_step, val_loss))
        self.last_update_time = datetime.now()
    
    def update_learning_rate(self, lr: float):
        """更新学习率"""
        self.learning_rate_history.append((self.global_step, lr))
    
    def update_throughput(self, samples_per_second: float):
        """更新吞吐量"""
        self.throughput_history.append((time.time(), samples_per_second))
    
    def get_recent_train_loss(self, steps: int = 10) -> List[float]:
        """获取最近的训练损失"""
        recent_losses = list(self.train_loss_history)[-steps:]
        return [loss for _, loss in recent_losses]
    
    def get_recent_val_loss(self, steps: int = 10) -> List[float]:
        """获取最近的验证损失"""
        recent_losses = list(self.val_loss_history)[-steps:]
        return [loss for _, loss in recent_losses]


@dataclass
class ConvergenceMetrics:
    """收敛指标"""
    loss_smoothness: float = 0.0      # 损失平滑度
    loss_trend: float = 0.0           # 损失趋势（负值表示下降）
    convergence_score: float = 0.0    # 收敛评分
    plateau_steps: int = 0            # 平台期步数
    is_converged: bool = False        # 是否收敛
    early_stop_patience: int = 0      # 早停耐心值
    
    def calculate_convergence_score(self, recent_losses: List[float], window_size: int = 20) -> float:
        """计算收敛评分"""
        if len(recent_losses) < window_size:
            return 0.0
        
        # 计算损失趋势
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)
        
        # 线性回归计算趋势
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            self.loss_trend = slope
        
        # 计算平滑度（方差的倒数）
        if len(y) > 1:
            variance = np.var(y)
            self.loss_smoothness = 1.0 / (1.0 + variance)
        
        # 综合评分
        trend_score = max(0.0, -self.loss_trend)  # 下降趋势得分更高
        smoothness_score = self.loss_smoothness
        
        self.convergence_score = 0.7 * trend_score + 0.3 * smoothness_score
        
        # 检测平台期
        if abs(self.loss_trend) < 0.001 and variance < 0.01:
            self.plateau_steps += 1
        else:
            self.plateau_steps = 0
        
        # 收敛判断
        self.is_converged = (self.convergence_score > 0.8 and 
                           self.plateau_steps > 50)
        
        return self.convergence_score


@dataclass
class GPUMonitoringMetrics:
    """GPU监控指标"""
    gpu_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 基础指标
    memory_used: float = 0.0          # MB
    memory_total: float = 0.0         # MB
    memory_free: float = 0.0          # MB
    utilization: float = 0.0          # %
    temperature: float = 0.0          # °C
    power_usage: float = 0.0          # W
    
    # 计算指标
    compute_utilization: float = 0.0  # %
    memory_utilization: float = 0.0   # %
    
    # 性能指标
    sm_clock: float = 0.0             # MHz
    memory_clock: float = 0.0         # MHz
    
    @property
    def memory_usage_percent(self) -> float:
        """内存使用百分比"""
        return (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "gpu_id": self.gpu_id,
            "timestamp": self.timestamp.isoformat(),
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
            "memory_free": self.memory_free,
            "memory_usage_percent": self.memory_usage_percent,
            "utilization": self.utilization,
            "temperature": self.temperature,
            "power_usage": self.power_usage,
            "compute_utilization": self.compute_utilization,
            "memory_utilization": self.memory_utilization,
            "sm_clock": self.sm_clock,
            "memory_clock": self.memory_clock
        }


class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history: Dict[int, deque] = {
            gpu_id: deque(maxlen=1000) for gpu_id in gpu_ids
        }
        self.callbacks: List[Callable[[Dict[int, GPUMonitoringMetrics]], None]] = []
    
    def add_callback(self, callback: Callable[[Dict[int, GPUMonitoringMetrics]], None]):
        """添加监控回调函数"""
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"开始监控GPU {self.gpu_ids}，间隔{interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("GPU监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_gpu_metrics()
                
                # 存储历史数据
                for gpu_id, metric in metrics.items():
                    self.metrics_history[gpu_id].append(metric)
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"监控回调函数执行失败: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"GPU监控出错: {e}")
                time.sleep(interval)
    
    def _collect_gpu_metrics(self) -> Dict[int, GPUMonitoringMetrics]:
        """收集GPU指标"""
        metrics = {}
        
        for gpu_id in self.gpu_ids:
            try:
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    # 使用PyTorch获取GPU信息
                    torch.cuda.set_device(gpu_id)
                    
                    # 内存信息
                    memory_info = torch.cuda.mem_get_info(gpu_id)
                    memory_free = memory_info[0] / (1024 ** 2)  # 转换为MB
                    memory_total = memory_info[1] / (1024 ** 2)
                    memory_used = memory_total - memory_free
                    
                    # GPU利用率（简化实现）
                    utilization = torch.cuda.utilization(gpu_id) if hasattr(torch.cuda, 'utilization') else 0.0
                    
                    metric = GPUMonitoringMetrics(
                        gpu_id=gpu_id,
                        memory_used=memory_used,
                        memory_total=memory_total,
                        memory_free=memory_free,
                        utilization=utilization,
                        compute_utilization=utilization,
                        memory_utilization=memory_used / memory_total * 100 if memory_total > 0 else 0.0
                    )
                    
                    # 尝试获取更详细的信息（如果nvidia-ml-py可用）
                    try:
                        import pynvml
                        if not hasattr(self, '_nvml_initialized'):
                            pynvml.nvmlInit()
                            self._nvml_initialized = True
                        
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        
                        # 温度
                        try:
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            metric.temperature = float(temp)
                        except:
                            pass
                        
                        # 功耗
                        try:
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                            metric.power_usage = float(power)
                        except:
                            pass
                        
                        # 时钟频率
                        try:
                            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                            metric.sm_clock = float(sm_clock)
                            metric.memory_clock = float(memory_clock)
                        except:
                            pass
                            
                    except ImportError:
                        # pynvml不可用，使用基础信息
                        pass
                    except Exception as e:
                        logger.warning(f"获取GPU {gpu_id}详细信息失败: {e}")
                    
                    metrics[gpu_id] = metric
                else:
                    # CUDA不可用或GPU ID超出范围，创建默认指标
                    metrics[gpu_id] = GPUMonitoringMetrics(gpu_id=gpu_id)
                    
            except Exception as e:
                logger.error(f"收集GPU {gpu_id}指标失败: {e}")
                # 创建默认指标
                metrics[gpu_id] = GPUMonitoringMetrics(gpu_id=gpu_id)
        
        return metrics
    
    def get_current_metrics(self) -> Dict[int, GPUMonitoringMetrics]:
        """获取当前GPU指标"""
        return self._collect_gpu_metrics()
    
    def get_metrics_history(self, gpu_id: int, steps: int = 100) -> List[GPUMonitoringMetrics]:
        """获取GPU指标历史"""
        if gpu_id not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[gpu_id])
        return history[-steps:] if steps > 0 else history
    
    def get_average_metrics(self, gpu_id: int, steps: int = 10) -> Optional[GPUMonitoringMetrics]:
        """获取平均GPU指标"""
        history = self.get_metrics_history(gpu_id, steps)
        if not history:
            return None
        
        # 计算平均值
        avg_metric = GPUMonitoringMetrics(gpu_id=gpu_id)
        
        avg_metric.memory_used = sum(m.memory_used for m in history) / len(history)
        avg_metric.memory_total = history[-1].memory_total  # 使用最新的总内存
        avg_metric.memory_free = sum(m.memory_free for m in history) / len(history)
        avg_metric.utilization = sum(m.utilization for m in history) / len(history)
        avg_metric.temperature = sum(m.temperature for m in history) / len(history)
        avg_metric.power_usage = sum(m.power_usage for m in history) / len(history)
        avg_metric.compute_utilization = sum(m.compute_utilization for m in history) / len(history)
        avg_metric.memory_utilization = sum(m.memory_utilization for m in history) / len(history)
        
        return avg_metric


class TrainingMonitor:
    """训练监控核心类"""
    
    def __init__(self, 
                 gpu_ids: List[int],
                 log_dir: str = "logs/training",
                 save_interval: int = 100,
                 convergence_window: int = 50):
        """
        初始化训练监控器
        
        Args:
            gpu_ids: GPU ID列表
            log_dir: 日志保存目录
            save_interval: 保存间隔（步数）
            convergence_window: 收敛检测窗口大小
        """
        self.gpu_ids = gpu_ids
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.convergence_window = convergence_window
        
        # 训练状态
        self.training_state = TrainingState()
        self.convergence_metrics = ConvergenceMetrics()
        
        # GPU监控
        self.gpu_monitor = GPUMonitor(gpu_ids)
        self.gpu_monitor.add_callback(self._on_gpu_metrics_update)
        
        # 分布式训练指标历史
        self.distributed_metrics_history: deque = deque(maxlen=10000)
        
        # 回调函数
        self.step_callbacks: List[Callable[[DistributedTrainingMetrics], None]] = []
        self.epoch_callbacks: List[Callable[[int, DistributedTrainingMetrics], None]] = []
        
        # 异常检测
        self.anomaly_detectors: List[Callable[[DistributedTrainingMetrics], List[str]]] = []
        
        # 监控状态
        self.is_monitoring = False
        
        logger.info(f"训练监控器初始化完成，监控GPU: {gpu_ids}")
    
    def start_monitoring(self, gpu_monitor_interval: float = 1.0):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已经在运行中")
            return
        
        self.is_monitoring = True
        self.training_state.start_time = datetime.now()
        
        # 启动GPU监控
        self.gpu_monitor.start_monitoring(gpu_monitor_interval)
        
        logger.info("训练监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # 停止GPU监控
        self.gpu_monitor.stop_monitoring()
        
        # 保存最终报告
        self._save_final_report()
        
        logger.info("训练监控已停止")
    
    def update_training_step(self, 
                           epoch: int,
                           global_step: int,
                           train_loss: float,
                           learning_rate: float,
                           val_loss: Optional[float] = None,
                           chinese_metrics: Optional[ChineseMetrics] = None,
                           additional_metrics: Optional[Dict[str, Any]] = None):
        """更新训练步骤"""
        if not self.is_monitoring:
            logger.warning("监控未启动，无法更新训练步骤")
            return
        
        # 更新训练状态
        self.training_state.epoch = epoch
        self.training_state.global_step = global_step
        self.training_state.local_step += 1
        self.training_state.is_training = True
        
        # 更新损失和学习率
        self.training_state.update_loss(train_loss, val_loss)
        self.training_state.update_learning_rate(learning_rate)
        
        # 计算收敛指标
        recent_losses = self.training_state.get_recent_train_loss(self.convergence_window)
        self.convergence_metrics.calculate_convergence_score(recent_losses)
        
        # 创建分布式训练指标
        distributed_metrics = self._create_distributed_metrics(
            epoch, global_step, train_loss, val_loss, learning_rate,
            chinese_metrics, additional_metrics
        )
        
        # 存储历史数据
        self.distributed_metrics_history.append(distributed_metrics)
        
        # 执行步骤回调
        for callback in self.step_callbacks:
            try:
                callback(distributed_metrics)
            except Exception as e:
                logger.error(f"步骤回调执行失败: {e}")
        
        # 异常检测
        self._detect_anomalies(distributed_metrics)
        
        # 定期保存
        if global_step % self.save_interval == 0:
            self._save_metrics(distributed_metrics)
        
        logger.debug(f"更新训练步骤: epoch={epoch}, step={global_step}, loss={train_loss:.4f}")
    
    def update_epoch(self, epoch: int):
        """更新训练轮次"""
        if not self.is_monitoring:
            return
        
        self.training_state.epoch = epoch
        
        # 获取最新的分布式指标
        if self.distributed_metrics_history:
            latest_metrics = self.distributed_metrics_history[-1]
            
            # 执行轮次回调
            for callback in self.epoch_callbacks:
                try:
                    callback(epoch, latest_metrics)
                except Exception as e:
                    logger.error(f"轮次回调执行失败: {e}")
        
        logger.info(f"完成训练轮次: {epoch}")
    
    def _create_distributed_metrics(self,
                                  epoch: int,
                                  global_step: int,
                                  train_loss: float,
                                  val_loss: Optional[float],
                                  learning_rate: float,
                                  chinese_metrics: Optional[ChineseMetrics],
                                  additional_metrics: Optional[Dict[str, Any]]) -> DistributedTrainingMetrics:
        """创建分布式训练指标"""
        
        # 获取当前GPU指标
        current_gpu_metrics = {}
        gpu_metrics_dict = self.gpu_monitor.get_current_metrics()
        
        for gpu_id, gpu_metric in gpu_metrics_dict.items():
            current_gpu_metrics[gpu_id] = {
                "memory_used": gpu_metric.memory_used,
                "memory_total": gpu_metric.memory_total,
                "memory_usage_percent": gpu_metric.memory_usage_percent,
                "utilization": gpu_metric.utilization,
                "temperature": gpu_metric.temperature,
                "power_usage": gpu_metric.power_usage,
                "compute_utilization": gpu_metric.compute_utilization,
                "memory_utilization": gpu_metric.memory_utilization
            }
        
        # 计算吞吐量
        throughput_tokens_per_second = self._calculate_throughput()
        
        # 创建通信指标（简化实现）
        communication_metrics = CommunicationMetrics()
        
        # 创建分布式训练指标
        metrics = DistributedTrainingMetrics(
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss or 0.0,
            learning_rate=learning_rate,
            gpu_metrics=current_gpu_metrics,
            communication_metrics=communication_metrics,
            throughput_tokens_per_second=throughput_tokens_per_second,
            convergence_score=self.convergence_metrics.convergence_score,
            gradient_norm=additional_metrics.get("gradient_norm", 0.0) if additional_metrics else 0.0
        )
        
        # 计算负载均衡和内存效率
        metrics.calculate_load_balance_score()
        metrics.calculate_memory_efficiency()
        
        return metrics
    
    def _calculate_throughput(self) -> float:
        """计算训练吞吐量"""
        if len(self.training_state.throughput_history) < 2:
            return 0.0
        
        recent_throughput = list(self.training_state.throughput_history)[-10:]
        if not recent_throughput:
            return 0.0
        
        return sum(throughput for _, throughput in recent_throughput) / len(recent_throughput)
    
    def _on_gpu_metrics_update(self, gpu_metrics: Dict[int, GPUMonitoringMetrics]):
        """GPU指标更新回调"""
        # 这里可以添加实时GPU指标处理逻辑
        pass
    
    def _detect_anomalies(self, metrics: DistributedTrainingMetrics) -> List[str]:
        """检测训练异常"""
        anomalies = []
        
        # 内置异常检测
        # 1. 损失异常
        if metrics.train_loss > 100 or np.isnan(metrics.train_loss) or np.isinf(metrics.train_loss):
            anomalies.append(f"训练损失异常: {metrics.train_loss}")
        
        # 2. GPU内存异常
        for gpu_id, gpu_metric in metrics.gpu_metrics.items():
            memory_usage = gpu_metric.get("memory_usage_percent", 0)
            if memory_usage > 95:
                anomalies.append(f"GPU {gpu_id} 内存使用率过高: {memory_usage:.1f}%")
            
            temperature = gpu_metric.get("temperature", 0)
            if temperature > 85:
                anomalies.append(f"GPU {gpu_id} 温度过高: {temperature:.1f}°C")
        
        # 3. 梯度异常
        if metrics.gradient_norm > 10.0:
            anomalies.append(f"梯度范数过大: {metrics.gradient_norm:.4f}")
        
        # 4. 收敛异常
        if self.convergence_metrics.plateau_steps > 200:
            anomalies.append(f"训练可能陷入平台期: {self.convergence_metrics.plateau_steps} 步")
        
        # 执行自定义异常检测器
        for detector in self.anomaly_detectors:
            try:
                custom_anomalies = detector(metrics)
                anomalies.extend(custom_anomalies)
            except Exception as e:
                logger.error(f"自定义异常检测器执行失败: {e}")
        
        # 记录异常
        if anomalies:
            logger.warning(f"检测到训练异常: {anomalies}")
        
        return anomalies
    
    def _save_metrics(self, metrics: DistributedTrainingMetrics):
        """保存训练指标"""
        try:
            # 保存到JSON文件
            metrics_file = self.log_dir / f"metrics_step_{metrics.global_step}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 保存汇总指标
            summary_file = self.log_dir / "training_summary.json"
            summary_data = {
                "last_update": datetime.now().isoformat(),
                "current_epoch": metrics.epoch,
                "current_step": metrics.global_step,
                "current_loss": metrics.train_loss,
                "convergence_score": metrics.convergence_score,
                "load_balance_score": metrics.load_balance_score,
                "memory_efficiency": metrics.memory_efficiency
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存训练指标失败: {e}")
    
    def _save_final_report(self):
        """保存最终训练报告"""
        try:
            if not self.distributed_metrics_history:
                logger.warning("没有训练指标历史，无法生成最终报告")
                return
            
            final_metrics = self.distributed_metrics_history[-1]
            training_duration = datetime.now() - self.training_state.start_time
            
            report = {
                "training_summary": {
                    "start_time": self.training_state.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": training_duration.total_seconds(),
                    "total_epochs": final_metrics.epoch,
                    "total_steps": final_metrics.global_step,
                    "final_train_loss": final_metrics.train_loss,
                    "final_val_loss": final_metrics.val_loss,
                    "convergence_achieved": self.convergence_metrics.is_converged
                },
                "performance_summary": {
                    "average_throughput": self._calculate_throughput(),
                    "final_load_balance_score": final_metrics.load_balance_score,
                    "final_memory_efficiency": final_metrics.memory_efficiency,
                    "convergence_score": final_metrics.convergence_score
                },
                "gpu_summary": {
                    gpu_id: self.gpu_monitor.get_average_metrics(gpu_id, 100).to_dict()
                    for gpu_id in self.gpu_ids
                    if self.gpu_monitor.get_average_metrics(gpu_id, 100) is not None
                }
            }
            
            report_file = self.log_dir / "final_training_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"最终训练报告已保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"保存最终训练报告失败: {e}")
    
    # 公共接口方法
    def add_step_callback(self, callback: Callable[[DistributedTrainingMetrics], None]):
        """添加步骤回调函数"""
        self.step_callbacks.append(callback)
    
    def add_epoch_callback(self, callback: Callable[[int, DistributedTrainingMetrics], None]):
        """添加轮次回调函数"""
        self.epoch_callbacks.append(callback)
    
    def add_anomaly_detector(self, detector: Callable[[DistributedTrainingMetrics], List[str]]):
        """添加异常检测器"""
        self.anomaly_detectors.append(detector)
    
    def get_current_metrics(self) -> Optional[DistributedTrainingMetrics]:
        """获取当前训练指标"""
        if not self.distributed_metrics_history:
            return None
        return self.distributed_metrics_history[-1]
    
    def get_metrics_history(self, steps: int = 100) -> List[DistributedTrainingMetrics]:
        """获取训练指标历史"""
        history = list(self.distributed_metrics_history)
        return history[-steps:] if steps > 0 else history
    
    def get_loss_curve_data(self, steps: int = 1000) -> Dict[str, List[Tuple[int, float]]]:
        """获取损失曲线数据"""
        train_losses = list(self.training_state.train_loss_history)[-steps:]
        val_losses = list(self.training_state.val_loss_history)[-steps:]
        
        return {
            "train_loss": train_losses,
            "val_loss": val_losses
        }
    
    def get_learning_rate_curve_data(self, steps: int = 1000) -> List[Tuple[int, float]]:
        """获取学习率曲线数据"""
        return list(self.training_state.learning_rate_history)[-steps:]
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """获取收敛状态"""
        return {
            "is_converged": self.convergence_metrics.is_converged,
            "convergence_score": self.convergence_metrics.convergence_score,
            "loss_trend": self.convergence_metrics.loss_trend,
            "plateau_steps": self.convergence_metrics.plateau_steps,
            "loss_smoothness": self.convergence_metrics.loss_smoothness
        }
    
    def estimate_remaining_time(self, target_steps: int) -> Optional[timedelta]:
        """估算剩余训练时间"""
        if not self.distributed_metrics_history or len(self.distributed_metrics_history) < 2:
            return None
        
        current_step = self.training_state.global_step
        if current_step >= target_steps:
            return timedelta(0)
        
        # 计算平均步骤时间
        recent_metrics = list(self.distributed_metrics_history)[-10:]
        if len(recent_metrics) < 2:
            return None
        
        time_diffs = []
        for i in range(1, len(recent_metrics)):
            prev_time = recent_metrics[i-1].timestamp
            curr_time = recent_metrics[i].timestamp
            time_diff = (curr_time - prev_time).total_seconds()
            time_diffs.append(time_diff)
        
        if not time_diffs:
            return None
        
        avg_step_time = sum(time_diffs) / len(time_diffs)
        remaining_steps = target_steps - current_step
        remaining_seconds = remaining_steps * avg_step_time
        
        return timedelta(seconds=remaining_seconds)
    
    def get_gpu_utilization_summary(self) -> Dict[int, Dict[str, float]]:
        """获取GPU利用率摘要"""
        summary = {}
        
        for gpu_id in self.gpu_ids:
            avg_metrics = self.gpu_monitor.get_average_metrics(gpu_id, 100)
            if avg_metrics:
                summary[gpu_id] = {
                    "avg_utilization": avg_metrics.utilization,
                    "avg_memory_usage": avg_metrics.memory_usage_percent,
                    "avg_temperature": avg_metrics.temperature,
                    "avg_power_usage": avg_metrics.power_usage
                }
        
        return summary