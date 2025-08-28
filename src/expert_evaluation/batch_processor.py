"""
批量评估处理器

本模块实现了大规模QA数据的批量处理功能，包括智能批处理大小调整、
并行评估和多线程支持，集成现有的内存管理和GPU工具。
"""

import asyncio
import threading
import time
import gc
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import json
import queue
import multiprocessing as mp
from enum import Enum

import torch
import psutil

from ..memory_manager import MemoryManager, MemorySnapshot, MemoryPressureLevel
from ..gpu_utils import GPUDetector, GPUInfo
from .interfaces import ExpertEvaluationEngine as ExpertEvaluationEngineInterface
from .config import ExpertEvaluationConfig, BatchProcessingConfig, ExpertiseLevel
from .data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    BatchEvaluationResult,
    EvaluationDataset,
    ValidationResult
)
from .exceptions import (
    BatchProcessingError,
    ResourceError,
    TimeoutError,
    ConfigurationError
)


class ProcessingStrategy(Enum):
    """批处理策略枚举"""
    SEQUENTIAL = "sequential"           # 顺序处理
    PARALLEL_THREAD = "parallel_thread"  # 多线程并行
    PARALLEL_PROCESS = "parallel_process"  # 多进程并行
    ADAPTIVE = "adaptive"               # 自适应策略


class BatchStatus(Enum):
    """批次状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchTask:
    """批次任务"""
    batch_id: str
    qa_items: List[QAEvaluationItem]
    batch_size: int
    priority: int = 5  # 1-10, 10为最高优先级
    timeout: Optional[int] = None  # 超时时间（秒）
    retry_count: int = 0
    max_retries: int = 3
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[ExpertEvaluationResult] = None
    error: Optional[str] = None
    
    @property
    def processing_time(self) -> Optional[float]:
        """获取处理时间"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "priority": self.priority,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time,
            "error": self.error,
            "qa_items_count": len(self.qa_items)
        }


@dataclass
class BatchProcessingStats:
    """批处理统计信息"""
    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    cancelled_batches: int = 0
    total_qa_items: int = 0
    processed_qa_items: int = 0
    total_processing_time: float = 0.0
    average_batch_time: float = 0.0
    throughput_items_per_second: float = 0.0
    memory_peak_usage: int = 0  # MB
    gpu_peak_utilization: float = 0.0
    
    def update_stats(self, batch_task: BatchTask):
        """更新统计信息"""
        if batch_task.status == BatchStatus.COMPLETED:
            self.completed_batches += 1
            self.processed_qa_items += len(batch_task.qa_items)
            if batch_task.processing_time:
                self.total_processing_time += batch_task.processing_time
                self.average_batch_time = self.total_processing_time / self.completed_batches
                self.throughput_items_per_second = self.processed_qa_items / self.total_processing_time
        elif batch_task.status == BatchStatus.FAILED:
            self.failed_batches += 1
        elif batch_task.status == BatchStatus.CANCELLED:
            self.cancelled_batches += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_batches": self.total_batches,
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "cancelled_batches": self.cancelled_batches,
            "total_qa_items": self.total_qa_items,
            "processed_qa_items": self.processed_qa_items,
            "completion_rate": self.completed_batches / self.total_batches if self.total_batches > 0 else 0.0,
            "total_processing_time": self.total_processing_time,
            "average_batch_time": self.average_batch_time,
            "throughput_items_per_second": self.throughput_items_per_second,
            "memory_peak_usage": self.memory_peak_usage,
            "gpu_peak_utilization": self.gpu_peak_utilization
        }


class IntelligentBatchSizer:
    """智能批处理大小调整器"""
    
    def __init__(self, initial_batch_size: int = 4, min_batch_size: int = 1, max_batch_size: int = 32):
        """初始化批处理大小调整器"""
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
        self.logger = logging.getLogger(__name__)
        
        # 性能历史记录
        self.performance_history: List[Tuple[int, float, float]] = []  # (batch_size, processing_time, memory_usage)
        self.max_history_length = 50
        
        # 调整策略参数
        self.memory_threshold = 0.85  # 85%内存使用率
        self.performance_window = 10   # 性能评估窗口
        self.adjustment_factor = 0.8   # 调整因子
        
    def suggest_batch_size(self, memory_snapshot: Optional[MemorySnapshot] = None,
                          processing_time: Optional[float] = None) -> int:
        """建议批处理大小"""
        suggested_size = self.current_batch_size
        
        # 基于内存使用率调整
        if memory_snapshot:
            if memory_snapshot.utilization_rate > self.memory_threshold:
                # 内存压力大，减小批次
                suggested_size = max(self.min_batch_size, 
                                   int(self.current_batch_size * self.adjustment_factor))
                self.logger.info(f"内存压力大({memory_snapshot.utilization_rate:.2%})，建议减小批次: {suggested_size}")
            
            elif memory_snapshot.utilization_rate < 0.6 and self.current_batch_size < self.max_batch_size:
                # 内存使用率低，可以尝试增大批次
                suggested_size = min(self.max_batch_size, self.current_batch_size + 1)
                self.logger.info(f"内存使用率低({memory_snapshot.utilization_rate:.2%})，建议增大批次: {suggested_size}")
        
        # 基于性能历史调整
        if processing_time and len(self.performance_history) >= self.performance_window:
            optimal_size = self._analyze_optimal_batch_size()
            if optimal_size != self.current_batch_size:
                suggested_size = optimal_size
                self.logger.info(f"基于性能分析，建议批次大小: {suggested_size}")
        
        return suggested_size
    
    def record_performance(self, batch_size: int, processing_time: float, memory_usage: float):
        """记录性能数据"""
        self.performance_history.append((batch_size, processing_time, memory_usage))
        
        # 限制历史记录长度
        if len(self.performance_history) > self.max_history_length:
            self.performance_history = self.performance_history[-self.max_history_length:]
    
    def _analyze_optimal_batch_size(self) -> int:
        """分析最优批处理大小"""
        if len(self.performance_history) < self.performance_window:
            return self.current_batch_size
        
        # 计算不同批次大小的平均性能
        batch_performance = {}
        for batch_size, processing_time, memory_usage in self.performance_history[-self.performance_window:]:
            if batch_size not in batch_performance:
                batch_performance[batch_size] = []
            
            # 计算吞吐量（items/second）
            throughput = batch_size / processing_time if processing_time > 0 else 0
            batch_performance[batch_size].append((throughput, memory_usage))
        
        # 找到最优批次大小
        best_batch_size = self.current_batch_size
        best_score = 0.0
        
        for batch_size, performances in batch_performance.items():
            if len(performances) >= 3:  # 至少需要3个样本
                avg_throughput = sum(p[0] for p in performances) / len(performances)
                avg_memory = sum(p[1] for p in performances) / len(performances)
                
                # 综合评分：吞吐量权重0.7，内存效率权重0.3
                memory_efficiency = 1.0 - (avg_memory / 100.0)  # 假设内存使用率百分比
                score = avg_throughput * 0.7 + memory_efficiency * 0.3
                
                if score > best_score:
                    best_score = score
                    best_batch_size = batch_size
        
        return best_batch_size
    
    def update_batch_size(self, new_size: int):
        """更新当前批处理大小"""
        old_size = self.current_batch_size
        self.current_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_size))
        
        if self.current_batch_size != old_size:
            self.logger.info(f"批处理大小已更新: {old_size} -> {self.current_batch_size}")
    
    def get_current_batch_size(self) -> int:
        """获取当前批处理大小"""
        return self.current_batch_size


class ParallelEvaluationManager:
    """并行评估管理器"""
    
    def __init__(self, max_workers: Optional[int] = None, strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE):
        """初始化并行评估管理器"""
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1) + 4)
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # 线程池和进程池
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # 任务队列
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_queue: queue.Queue = queue.Queue()
        
        # 状态管理
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        
    def start(self):
        """启动并行评估管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 根据策略初始化执行器
        if self.strategy in [ProcessingStrategy.PARALLEL_THREAD, ProcessingStrategy.ADAPTIVE]:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        if self.strategy in [ProcessingStrategy.PARALLEL_PROCESS, ProcessingStrategy.ADAPTIVE]:
            # 进程池使用较少的worker，因为每个进程消耗更多资源
            process_workers = min(4, mp.cpu_count() or 1)
            self.process_executor = ProcessPoolExecutor(max_workers=process_workers)
        
        self.logger.info(f"并行评估管理器已启动，策略: {self.strategy.value}, 最大工作线程: {self.max_workers}")
    
    def stop(self):
        """停止并行评估管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 关闭执行器
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
            self.thread_executor = None
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
            self.process_executor = None
        
        self.logger.info("并行评估管理器已停止")
    
    def submit_batch_task(self, batch_task: BatchTask, 
                         evaluation_engine: ExpertEvaluationEngineInterface) -> asyncio.Future:
        """提交批次任务"""
        if not self.is_running:
            raise BatchProcessingError("并行评估管理器未启动")
        
        # 根据策略选择执行器
        executor = self._select_executor(batch_task)
        
        if executor is None:
            # 顺序处理
            future = asyncio.Future()
            try:
                result = self._process_batch_sequential(batch_task, evaluation_engine)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        
        # 并行处理
        future = executor.submit(self._process_batch_parallel, batch_task, evaluation_engine)
        return future
    
    def _select_executor(self, batch_task: BatchTask) -> Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]]:
        """选择合适的执行器"""
        if self.strategy == ProcessingStrategy.SEQUENTIAL:
            return None
        
        elif self.strategy == ProcessingStrategy.PARALLEL_THREAD:
            return self.thread_executor
        
        elif self.strategy == ProcessingStrategy.PARALLEL_PROCESS:
            return self.process_executor
        
        elif self.strategy == ProcessingStrategy.ADAPTIVE:
            # 自适应策略：根据批次大小和系统资源选择
            batch_size = len(batch_task.qa_items)
            
            # 小批次使用线程池，大批次使用进程池
            if batch_size <= 10:
                return self.thread_executor
            else:
                return self.process_executor
        
        return None
    
    def _process_batch_sequential(self, batch_task: BatchTask, 
                                evaluation_engine: ExpertEvaluationEngineInterface) -> ExpertEvaluationResult:
        """顺序处理批次"""
        batch_task.status = BatchStatus.PROCESSING
        batch_task.started_at = datetime.now()
        
        try:
            result = evaluation_engine.evaluate_model(batch_task.qa_items)
            batch_task.status = BatchStatus.COMPLETED
            batch_task.result = result
            batch_task.completed_at = datetime.now()
            return result
        
        except Exception as e:
            batch_task.status = BatchStatus.FAILED
            batch_task.error = str(e)
            batch_task.completed_at = datetime.now()
            raise BatchProcessingError(f"批次处理失败: {e}")
    
    def _process_batch_parallel(self, batch_task: BatchTask, 
                              evaluation_engine: ExpertEvaluationEngineInterface) -> ExpertEvaluationResult:
        """并行处理批次"""
        batch_task.status = BatchStatus.PROCESSING
        batch_task.started_at = datetime.now()
        
        try:
            # 将批次分割为更小的子批次进行并行处理
            sub_batches = self._split_batch(batch_task.qa_items, self.max_workers)
            
            # 并行处理子批次
            sub_results = []
            futures = []
            
            for sub_batch in sub_batches:
                if self.thread_executor:
                    future = self.thread_executor.submit(
                        evaluation_engine.evaluate_model, sub_batch
                    )
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures, timeout=batch_task.timeout):
                try:
                    result = future.result()
                    sub_results.append(result)
                except Exception as e:
                    self.logger.error(f"子批次处理失败: {e}")
                    raise
            
            # 聚合结果
            if sub_results:
                aggregated_result = self._aggregate_results(sub_results)
                batch_task.status = BatchStatus.COMPLETED
                batch_task.result = aggregated_result
                batch_task.completed_at = datetime.now()
                return aggregated_result
            else:
                raise BatchProcessingError("没有成功的子批次结果")
        
        except Exception as e:
            batch_task.status = BatchStatus.FAILED
            batch_task.error = str(e)
            batch_task.completed_at = datetime.now()
            raise BatchProcessingError(f"并行批次处理失败: {e}")
    
    def _split_batch(self, qa_items: List[QAEvaluationItem], num_splits: int) -> List[List[QAEvaluationItem]]:
        """将批次分割为子批次"""
        if num_splits <= 1 or len(qa_items) <= num_splits:
            return [qa_items]
        
        chunk_size = len(qa_items) // num_splits
        remainder = len(qa_items) % num_splits
        
        sub_batches = []
        start_idx = 0
        
        for i in range(num_splits):
            # 前remainder个子批次多分配一个项目
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            if start_idx < len(qa_items):
                sub_batches.append(qa_items[start_idx:end_idx])
            
            start_idx = end_idx
        
        return [batch for batch in sub_batches if batch]  # 过滤空批次
    
    def _aggregate_results(self, results: List[ExpertEvaluationResult]) -> ExpertEvaluationResult:
        """聚合多个评估结果"""
        if not results:
            raise BatchProcessingError("没有结果可以聚合")
        
        if len(results) == 1:
            return results[0]
        
        # 计算平均分数
        total_score = sum(r.overall_score for r in results)
        avg_score = total_score / len(results)
        
        # 聚合维度分数
        all_dimensions = set()
        for result in results:
            all_dimensions.update(result.dimension_scores.keys())
        
        aggregated_dimensions = {}
        for dim in all_dimensions:
            scores = [r.dimension_scores[dim].score for r in results if dim in r.dimension_scores]
            if scores:
                avg_dim_score = sum(scores) / len(scores)
                # 使用第一个结果的维度分数结构作为模板
                template_score = next(r.dimension_scores[dim] for r in results if dim in r.dimension_scores)
                aggregated_dimensions[dim] = type(template_score)(
                    dimension=dim,
                    score=avg_dim_score,
                    confidence=template_score.confidence,
                    details={"aggregated_from": len(scores), "total_results": len(results)}
                )
        
        # 聚合反馈和建议
        all_feedback = {}
        all_suggestions = []
        
        for result in results:
            all_feedback.update(result.detailed_feedback)
            all_suggestions.extend(result.improvement_suggestions)
        
        # 去重建议
        unique_suggestions = list(set(all_suggestions))[:10]  # 限制建议数量
        
        # 创建聚合结果
        from .data_models import ExpertEvaluationResult
        aggregated_result = ExpertEvaluationResult(
            question_id="aggregated_batch_result",
            overall_score=avg_score,
            dimension_scores=aggregated_dimensions,
            detailed_feedback=all_feedback,
            improvement_suggestions=unique_suggestions,
            processing_time=sum(r.processing_time for r in results)
        )
        
        return aggregated_result


class BatchProcessor:
    """批量评估处理器主类"""
    
    def __init__(self, config: BatchProcessingConfig, evaluation_engine: ExpertEvaluationEngineInterface):
        """初始化批量处理器"""
        self.config = config
        self.evaluation_engine = evaluation_engine
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.memory_manager = MemoryManager({
            "monitoring_interval": 5,
            "enable_auto_adjustment": True,
            "initial_batch_size": config.initial_batch_size,
            "min_batch_size": config.min_batch_size,
            "max_batch_size": config.max_batch_size
        })
        
        self.gpu_detector = GPUDetector()
        self.batch_sizer = IntelligentBatchSizer(
            initial_batch_size=config.initial_batch_size,
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size
        )
        
        self.parallel_manager = ParallelEvaluationManager(
            max_workers=config.max_workers,
            strategy=config.processing_strategy
        )
        
        # 状态管理
        self.is_running = False
        self.batch_queue: List[BatchTask] = []
        self.completed_batches: List[BatchTask] = []
        self.stats = BatchProcessingStats()
        
        # 回调函数
        self.progress_callbacks: List[Callable[[BatchTask], None]] = []
        self.completion_callbacks: List[Callable[[BatchEvaluationResult], None]] = []
    
    def start(self) -> bool:
        """启动批量处理器"""
        if self.is_running:
            self.logger.warning("批量处理器已在运行")
            return False
        
        try:
            # 启动内存管理器
            if not self.memory_manager.start():
                raise BatchProcessingError("内存管理器启动失败")
            
            # 启动并行管理器
            self.parallel_manager.start()
            
            self.is_running = True
            self.logger.info("批量评估处理器已启动")
            return True
        
        except Exception as e:
            self.logger.error(f"启动批量处理器失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止批量处理器"""
        if not self.is_running:
            return True
        
        try:
            # 停止并行管理器
            self.parallel_manager.stop()
            
            # 停止内存管理器
            self.memory_manager.stop()
            
            self.is_running = False
            self.logger.info("批量评估处理器已停止")
            return True
        
        except Exception as e:
            self.logger.error(f"停止批量处理器失败: {e}")
            return False
    
    def process_large_dataset(self, qa_datasets: List[EvaluationDataset]) -> BatchEvaluationResult:
        """处理大规模QA数据集"""
        if not self.is_running:
            raise BatchProcessingError("批量处理器未启动")
        
        start_time = datetime.now()
        batch_id = f"large_dataset_{int(time.time())}"
        
        try:
            self.logger.info(f"开始处理大规模数据集，包含 {len(qa_datasets)} 个数据集")
            
            # 准备批次任务
            batch_tasks = self._prepare_batch_tasks(qa_datasets)
            self.stats.total_batches = len(batch_tasks)
            self.stats.total_qa_items = sum(len(task.qa_items) for task in batch_tasks)
            
            # 处理批次
            individual_results = []
            
            for batch_task in batch_tasks:
                try:
                    # 动态调整批处理大小
                    self._adjust_batch_size_if_needed(batch_task)
                    
                    # 提交批次任务
                    future = self.parallel_manager.submit_batch_task(batch_task, self.evaluation_engine)
                    
                    # 等待结果
                    result = future.result(timeout=batch_task.timeout)
                    individual_results.append(result)
                    
                    # 更新统计信息
                    self.stats.update_stats(batch_task)
                    self.completed_batches.append(batch_task)
                    
                    # 触发进度回调
                    for callback in self.progress_callbacks:
                        try:
                            callback(batch_task)
                        except Exception as e:
                            self.logger.error(f"进度回调执行失败: {e}")
                    
                    # 记录性能数据
                    if batch_task.processing_time:
                        memory_snapshot = self.memory_manager.get_current_memory_status()
                        memory_usage = memory_snapshot.utilization_rate * 100 if memory_snapshot else 0
                        self.batch_sizer.record_performance(
                            len(batch_task.qa_items), 
                            batch_task.processing_time, 
                            memory_usage
                        )
                
                except Exception as e:
                    self.logger.error(f"批次 {batch_task.batch_id} 处理失败: {e}")
                    batch_task.status = BatchStatus.FAILED
                    batch_task.error = str(e)
                    self.stats.update_stats(batch_task)
            
            # 创建批量评估结果
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds()
            
            batch_result = BatchEvaluationResult(
                batch_id=batch_id,
                individual_results=individual_results,
                total_processing_time=total_processing_time,
                start_time=start_time,
                end_time=end_time
            )
            
            # 触发完成回调
            for callback in self.completion_callbacks:
                try:
                    callback(batch_result)
                except Exception as e:
                    self.logger.error(f"完成回调执行失败: {e}")
            
            self.logger.info(f"大规模数据集处理完成，成功处理 {len(individual_results)} 个批次")
            return batch_result
        
        except Exception as e:
            self.logger.error(f"大规模数据集处理失败: {e}")
            raise BatchProcessingError(f"大规模数据集处理失败: {e}")
    
    def _prepare_batch_tasks(self, qa_datasets: List[EvaluationDataset]) -> List[BatchTask]:
        """准备批次任务"""
        batch_tasks = []
        
        for i, dataset in enumerate(qa_datasets):
            # 将数据集分割为批次
            current_batch_size = self.batch_sizer.get_current_batch_size()
            
            for j in range(0, len(dataset.qa_items), current_batch_size):
                batch_qa_items = dataset.qa_items[j:j + current_batch_size]
                
                batch_task = BatchTask(
                    batch_id=f"{dataset.dataset_id}_batch_{j // current_batch_size}",
                    qa_items=batch_qa_items,
                    batch_size=len(batch_qa_items),
                    priority=dataset.priority if hasattr(dataset, 'priority') else 5,
                    timeout=self.config.batch_timeout,
                    max_retries=self.config.max_retries
                )
                
                batch_tasks.append(batch_task)
        
        # 按优先级排序
        batch_tasks.sort(key=lambda x: x.priority, reverse=True)
        
        return batch_tasks
    
    def _adjust_batch_size_if_needed(self, batch_task: BatchTask):
        """根据需要调整批处理大小"""
        # 获取当前内存状态
        memory_snapshot = self.memory_manager.get_current_memory_status()
        
        if memory_snapshot:
            # 建议新的批处理大小
            suggested_size = self.batch_sizer.suggest_batch_size(
                memory_snapshot=memory_snapshot
            )
            
            # 如果建议的大小与当前批次大小差异较大，记录警告
            if abs(suggested_size - len(batch_task.qa_items)) > 2:
                self.logger.warning(
                    f"建议批处理大小 {suggested_size} 与当前批次大小 {len(batch_task.qa_items)} 差异较大"
                )
            
            # 更新批处理器的批次大小（影响后续批次）
            self.batch_sizer.update_batch_size(suggested_size)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats_dict = self.stats.to_dict()
        
        # 添加实时信息
        memory_status = self.memory_manager.get_current_memory_status()
        if memory_status:
            stats_dict["current_memory_usage"] = memory_status.utilization_rate
            stats_dict["current_memory_pressure"] = memory_status.pressure_level.value
        
        stats_dict["current_batch_size"] = self.batch_sizer.get_current_batch_size()
        stats_dict["pending_batches"] = len([b for b in self.batch_queue if b.status == BatchStatus.PENDING])
        stats_dict["processing_batches"] = len([b for b in self.batch_queue if b.status == BatchStatus.PROCESSING])
        
        return stats_dict
    
    def add_progress_callback(self, callback: Callable[[BatchTask], None]):
        """添加进度回调函数"""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[BatchEvaluationResult], None]):
        """添加完成回调函数"""
        self.completion_callbacks.append(callback)
    
    def optimize_performance(self) -> Dict[str, Any]:
        """优化性能"""
        optimization_results = {
            "memory_optimization": {},
            "batch_size_optimization": {},
            "gpu_optimization": {}
        }
        
        try:
            # 内存优化
            memory_recommendations = self.memory_manager.optimize_memory()
            optimization_results["memory_optimization"] = {
                "recommendations": [rec.to_dict() for rec in memory_recommendations],
                "current_memory_status": self.memory_manager.get_memory_status()
            }
            
            # 批处理大小优化
            current_batch_size = self.batch_sizer.get_current_batch_size()
            memory_snapshot = self.memory_manager.get_current_memory_status()
            suggested_batch_size = self.batch_sizer.suggest_batch_size(memory_snapshot)
            
            optimization_results["batch_size_optimization"] = {
                "current_batch_size": current_batch_size,
                "suggested_batch_size": suggested_batch_size,
                "performance_history_length": len(self.batch_sizer.performance_history)
            }
            
            # GPU优化
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            optimization_results["gpu_optimization"] = {
                "gpu_count": len(gpu_infos),
                "gpu_info": [
                    {
                        "gpu_id": gpu.gpu_id,
                        "name": gpu.name,
                        "memory_utilization": gpu.used_memory / gpu.total_memory if gpu.total_memory > 0 else 0,
                        "temperature": gpu.temperature,
                        "utilization": gpu.utilization
                    }
                    for gpu in gpu_infos
                ]
            }
            
        except Exception as e:
            self.logger.error(f"性能优化失败: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results


# 测试函数
def test_batch_evaluation():
    """测试批量评估功能"""
    print("开始测试批量评估功能...")
    
    try:
        # 创建测试配置
        from .config import BatchProcessingConfig
        config = BatchProcessingConfig(
            initial_batch_size=2,
            min_batch_size=1,
            max_batch_size=8,
            max_workers=4,
            processing_strategy=ProcessingStrategy.ADAPTIVE,
            batch_timeout=300,
            max_retries=2
        )
        
        # 创建模拟评估引擎
        class MockEvaluationEngine:
            def evaluate_model(self, qa_items):
                # 模拟评估过程
                time.sleep(0.1 * len(qa_items))  # 模拟处理时间
                
                from .data_models import ExpertEvaluationResult, DimensionScore
                from .config import EvaluationDimension
                
                return ExpertEvaluationResult(
                    question_id=f"mock_batch_{len(qa_items)}",
                    overall_score=0.8,
                    dimension_scores={
                        EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
                            dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
                            score=0.8,
                            confidence=0.9,
                            details={"mock": True}
                        )
                    },
                    detailed_feedback={"mock": "测试反馈"},
                    improvement_suggestions=["测试建议"],
                    processing_time=0.1 * len(qa_items)
                )
        
        mock_engine = MockEvaluationEngine()
        
        # 创建批量处理器
        processor = BatchProcessor(config, mock_engine)
        
        # 启动处理器
        if not processor.start():
            raise Exception("批量处理器启动失败")
        
        print("✓ 批量处理器启动成功")
        
        # 创建测试数据
        from .data_models import QAEvaluationItem, EvaluationDataset
        
        test_qa_items = [
            QAEvaluationItem(
                question_id=f"test_q_{i}",
                question=f"测试问题 {i}",
                reference_answer=f"测试答案 {i}",
                model_answer=f"模型答案 {i}",
                context=f"测试上下文 {i}",
                domain_tags=["测试"],
                difficulty_level=ExpertiseLevel.INTERMEDIATE,
                expected_concepts=["测试概念"]
            )
            for i in range(10)
        ]
        
        test_dataset = EvaluationDataset(
            dataset_id="test_dataset",
            name="测试数据集",
            qa_items=test_qa_items,
            description="批量评估测试数据集"
        )
        
        # 执行批量评估
        print("开始批量评估...")
        result = processor.process_large_dataset([test_dataset])
        
        print(f"✓ 批量评估完成")
        print(f"  - 处理时间: {result.total_processing_time:.2f}秒")
        print(f"  - 成功结果数: {len(result.individual_results)}")
        
        # 获取统计信息
        stats = processor.get_processing_stats()
        print(f"✓ 处理统计:")
        print(f"  - 总批次数: {stats['total_batches']}")
        print(f"  - 完成批次数: {stats['completed_batches']}")
        print(f"  - 平均批次时间: {stats['average_batch_time']:.2f}秒")
        print(f"  - 吞吐量: {stats['throughput_items_per_second']:.2f} items/s")
        
        # 性能优化测试
        optimization = processor.optimize_performance()
        print(f"✓ 性能优化建议已生成")
        
        # 停止处理器
        processor.stop()
        print("✓ 批量处理器已停止")
        
        print("批量评估功能测试完成！")
        return True
        
    except Exception as e:
        print(f"✗ 批量评估测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_batch_evaluation()