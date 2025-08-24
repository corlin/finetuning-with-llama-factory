"""
分布式训练引擎核心实现

本模块实现了分布式训练的核心功能，包括：
- 分布式后端初始化和管理
- 多GPU训练进程管理
- 梯度同步和参数更新
- 通信后端配置和优化

支持NCCL、GLOO等通信后端，提供完整的分布式训练生命周期管理。
"""

import os
import sys
import time
import signal
import logging
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import contextmanager
import socket
import subprocess
import psutil

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks

from .parallel_config import (
    ParallelConfig, GPUTopology, CommunicationBackend, 
    DistributedTrainingMetrics, CommunicationMetrics
)


class ProcessStatus(Enum):
    """进程状态枚举"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class ProcessInfo:
    """进程信息"""
    rank: int
    local_rank: int
    world_size: int
    pid: Optional[int] = None
    status: ProcessStatus = ProcessStatus.INITIALIZING
    gpu_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def runtime(self) -> Optional[float]:
        """计算运行时间"""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "pid": self.pid,
            "status": self.status.value,
            "gpu_id": self.gpu_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "runtime": self.runtime,
            "error_message": self.error_message
        }


class DistributedBackendInitializer:
    """分布式后端初始化器"""
    
    def __init__(self, config: ParallelConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.is_initialized = False
        self.backend_info = {}
        
    def initialize_backend(self, rank: int, world_size: int) -> bool:
        """
        初始化分布式后端
        
        Args:
            rank: 当前进程的rank
            world_size: 总进程数
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 设置环境变量
            self._setup_environment_variables(rank, world_size)
            
            # 选择通信后端
            backend = self._select_backend()
            
            # 初始化进程组
            init_method = f"tcp://{self.config.master_addr}:{self.config.master_port}"
            
            self.logger.info(f"初始化分布式后端: backend={backend}, rank={rank}, world_size={world_size}")
            self.logger.info(f"初始化方法: {init_method}")
            
            # 设置超时时间
            timeout = torch.distributed.default_pg_timeout
            if hasattr(self.config, 'timeout'):
                timeout = torch.timedelta(seconds=self.config.timeout)
            
            init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                timeout=timeout
            )
            
            # 验证初始化
            if not dist.is_initialized():
                raise RuntimeError("分布式后端初始化失败")
            
            # 设置设备
            local_rank = rank
            device = torch.device("cpu")  # 默认使用CPU
            
            if torch.cuda.is_available():
                try:
                    local_rank = rank % torch.cuda.device_count()
                    torch.cuda.set_device(local_rank)
                    device = torch.device(f"cuda:{local_rank}")
                except Exception as e:
                    self.logger.warning(f"CUDA设备设置失败，使用CPU: {e}")
            
            # 存储后端信息
            self.backend_info = {
                "backend": backend,
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "device": device,
                "master_addr": self.config.master_addr,
                "master_port": self.config.master_port,
                "init_method": init_method
            }
            
            self.is_initialized = True
            self.logger.info(f"分布式后端初始化成功: {self.backend_info}")
            
            # 执行连接测试
            self._test_communication()
            
            return True
            
        except Exception as e:
            self.logger.error(f"分布式后端初始化失败: {e}")
            return False
    
    def _setup_environment_variables(self, rank: int, world_size: int):
        """设置环境变量"""
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.master_port)
        
        # 设置CUDA相关环境变量
        if torch.cuda.is_available():
            local_rank = rank % torch.cuda.device_count()
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        
        # 设置NCCL相关环境变量
        if self.config.communication_backend == CommunicationBackend.NCCL:
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
            # 设置NCCL超时
            os.environ["NCCL_TIMEOUT"] = "1800"  # 30分钟
    
    def _select_backend(self) -> str:
        """选择通信后端"""
        if self.config.communication_backend == CommunicationBackend.NCCL:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA不可用，自动切换到GLOO后端")
                return "gloo"
            return "nccl"
        elif self.config.communication_backend == CommunicationBackend.GLOO:
            return "gloo"
        elif self.config.communication_backend == CommunicationBackend.MPI:
            return "mpi"
        else:
            # 自动选择
            if torch.cuda.is_available() and self.config.world_size > 1:
                return "nccl"
            else:
                return "gloo"
    
    def _test_communication(self):
        """测试通信连接"""
        try:
            # 检查是否真正初始化了分布式环境
            if not dist.is_initialized():
                self.logger.info("分布式环境未初始化，跳过通信测试")
                return
            
            self.logger.info("开始通信连接测试...")
            
            # 创建测试张量
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                test_tensor = torch.ones(10, device=device) * dist.get_rank()
            else:
                test_tensor = torch.ones(10) * dist.get_rank()
            
            # 执行AllReduce测试
            start_time = time.time()
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            comm_time = time.time() - start_time
            
            # 验证结果
            expected_sum = sum(range(dist.get_world_size()))
            if torch.allclose(test_tensor, torch.ones_like(test_tensor) * expected_sum):
                self.logger.info(f"通信测试成功，耗时: {comm_time:.4f}秒")
            else:
                self.logger.warning("通信测试结果不正确")
            
            # 执行Broadcast测试
            if dist.get_rank() == 0:
                broadcast_tensor = torch.tensor([42.0])
            else:
                broadcast_tensor = torch.tensor([0.0])
            
            if torch.cuda.is_available():
                broadcast_tensor = broadcast_tensor.cuda()
            
            dist.broadcast(broadcast_tensor, src=0)
            
            if torch.allclose(broadcast_tensor, torch.tensor([42.0]).to(broadcast_tensor.device)):
                self.logger.info("广播测试成功")
            else:
                self.logger.warning("广播测试失败")
                
        except Exception as e:
            self.logger.error(f"通信测试失败: {e}")
    
    def cleanup(self):
        """清理分布式后端"""
        try:
            if self.is_initialized and dist.is_initialized():
                self.logger.info("清理分布式后端...")
                destroy_process_group()
                self.is_initialized = False
                self.logger.info("分布式后端清理完成")
        except Exception as e:
            self.logger.error(f"清理分布式后端失败: {e}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        return self.backend_info.copy()
    
    def is_master(self) -> bool:
        """检查是否为主进程"""
        return self.backend_info.get("rank", -1) == 0
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        return self.backend_info.get("device", torch.device("cpu"))


class MultiGPUProcessManager:
    """多GPU训练进程管理器"""
    
    def __init__(self, config: ParallelConfig, topology: GPUTopology, logger: Optional[logging.Logger] = None):
        self.config = config
        self.topology = topology
        self.logger = logger or logging.getLogger(__name__)
        
        self.processes: Dict[int, mp.Process] = {}
        self.process_info: Dict[int, ProcessInfo] = {}
        self.shared_state = mp.Manager().dict()
        self.error_queue = mp.Queue()
        self.status_lock = mp.Lock()
        
        # 故障检测和恢复
        self.failure_detector = None
        self.recovery_enabled = True
        self.max_retries = 3
        
    def spawn_training_processes(self, train_fn: Callable, *args, **kwargs) -> bool:
        """
        启动训练进程
        
        Args:
            train_fn: 训练函数
            *args, **kwargs: 传递给训练函数的参数
            
        Returns:
            bool: 启动是否成功
        """
        try:
            self.logger.info(f"启动{self.config.world_size}个训练进程...")
            
            # 检查端口可用性
            if not self._check_port_availability():
                return False
            
            # 为每个rank创建进程
            for rank in range(self.config.world_size):
                local_rank = rank % self.topology.num_gpus
                gpu_id = list(self.topology.gpu_info.keys())[local_rank]
                
                # 创建进程信息
                process_info = ProcessInfo(
                    rank=rank,
                    local_rank=local_rank,
                    world_size=self.config.world_size,
                    gpu_id=gpu_id,
                    start_time=time.time()
                )
                self.process_info[rank] = process_info
                
                # 创建进程
                process = mp.Process(
                    target=self._process_wrapper,
                    args=(rank, train_fn, args, kwargs),
                    name=f"TrainingProcess-{rank}"
                )
                
                self.processes[rank] = process
                process.start()
                process_info.pid = process.pid
                
                self.logger.info(f"启动进程 rank={rank}, pid={process.pid}, gpu={gpu_id}")
            
            # 启动故障检测
            self._start_failure_detector()
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动训练进程失败: {e}")
            self._cleanup_processes()
            return False
    
    def _process_wrapper(self, rank: int, train_fn: Callable, args: tuple, kwargs: dict):
        """进程包装器，处理异常和状态更新"""
        try:
            # 更新进程状态
            self._update_process_status(rank, ProcessStatus.RUNNING)
            
            # 设置信号处理
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # 执行训练函数
            result = train_fn(rank, self.config, *args, **kwargs)
            
            # 更新完成状态
            self._update_process_status(rank, ProcessStatus.COMPLETED)
            
            return result
            
        except Exception as e:
            error_msg = f"进程 rank={rank} 执行失败: {e}"
            self.logger.error(error_msg)
            
            # 更新失败状态
            self._update_process_status(rank, ProcessStatus.FAILED, error_msg)
            
            # 将错误信息放入队列
            self.error_queue.put((rank, str(e)))
            
            raise
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，正在终止进程...")
        sys.exit(0)
    
    def _update_process_status(self, rank: int, status: ProcessStatus, error_msg: Optional[str] = None):
        """更新进程状态"""
        with self.status_lock:
            if rank in self.process_info:
                self.process_info[rank].status = status
                if error_msg:
                    self.process_info[rank].error_message = error_msg
                if status in [ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.TERMINATED]:
                    self.process_info[rank].end_time = time.time()
    
    def _check_port_availability(self) -> bool:
        """检查端口可用性"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.config.master_addr, self.config.master_port))
                return True
        except OSError:
            self.logger.error(f"端口 {self.config.master_port} 不可用")
            return False
    
    def _start_failure_detector(self):
        """启动故障检测器"""
        if not self.recovery_enabled:
            return
        
        self.failure_detector = threading.Thread(
            target=self._monitor_processes,
            daemon=True,
            name="FailureDetector"
        )
        self.failure_detector.start()
        self.logger.info("故障检测器已启动")
    
    def _monitor_processes(self):
        """监控进程状态"""
        while True:
            try:
                time.sleep(5)  # 每5秒检查一次
                
                failed_ranks = []
                for rank, process in self.processes.items():
                    if not process.is_alive() and self.process_info[rank].status == ProcessStatus.RUNNING:
                        self.logger.warning(f"检测到进程 rank={rank} 异常终止")
                        self._update_process_status(rank, ProcessStatus.FAILED, "进程异常终止")
                        failed_ranks.append(rank)
                
                # 处理故障恢复
                if failed_ranks and self.recovery_enabled:
                    self._handle_process_failures(failed_ranks)
                
                # 检查是否所有进程都完成
                all_completed = all(
                    info.status in [ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.TERMINATED]
                    for info in self.process_info.values()
                )
                
                if all_completed:
                    self.logger.info("所有进程已完成，停止监控")
                    break
                    
            except Exception as e:
                self.logger.error(f"进程监控异常: {e}")
    
    def _handle_process_failures(self, failed_ranks: List[int]):
        """处理进程故障"""
        self.logger.info(f"处理进程故障: {failed_ranks}")
        
        # 简单的故障处理：记录错误，不自动重启
        # 在生产环境中，可以实现更复杂的恢复策略
        for rank in failed_ranks:
            process_info = self.process_info[rank]
            self.logger.error(f"进程 rank={rank} 故障，运行时间: {process_info.runtime:.2f}秒")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有进程完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否所有进程都成功完成
        """
        try:
            start_time = time.time()
            
            for rank, process in self.processes.items():
                remaining_time = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_time = max(0, timeout - elapsed)
                    if remaining_time <= 0:
                        self.logger.warning("等待进程完成超时")
                        return False
                
                process.join(timeout=remaining_time)
                
                if process.is_alive():
                    self.logger.warning(f"进程 rank={rank} 仍在运行，强制终止")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    self._update_process_status(rank, ProcessStatus.TERMINATED)
            
            # 检查是否有失败的进程
            failed_processes = [
                rank for rank, info in self.process_info.items()
                if info.status == ProcessStatus.FAILED
            ]
            
            if failed_processes:
                self.logger.error(f"以下进程执行失败: {failed_processes}")
                return False
            
            self.logger.info("所有进程成功完成")
            return True
            
        except Exception as e:
            self.logger.error(f"等待进程完成时发生错误: {e}")
            return False
    
    def _cleanup_processes(self):
        """清理进程"""
        self.logger.info("清理训练进程...")
        
        for rank, process in self.processes.items():
            if process.is_alive():
                self.logger.info(f"终止进程 rank={rank}")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                self._update_process_status(rank, ProcessStatus.TERMINATED)
        
        self.processes.clear()
    
    def get_process_status(self) -> Dict[int, Dict[str, Any]]:
        """获取所有进程状态"""
        return {rank: info.to_dict() for rank, info in self.process_info.items()}
    
    def get_failed_processes(self) -> List[int]:
        """获取失败的进程列表"""
        return [
            rank for rank, info in self.process_info.items()
            if info.status == ProcessStatus.FAILED
        ]
    
    def cleanup(self):
        """清理资源"""
        self._cleanup_processes()
        
        # 清理队列
        while not self.error_queue.empty():
            try:
                self.error_queue.get_nowait()
            except:
                break


class GradientSynchronizer:
    """梯度同步器"""
    
    def __init__(self, config: ParallelConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.communication_metrics = CommunicationMetrics()
        self.gradient_hooks = []
        self.sync_enabled = True
        
    def setup_ddp_hooks(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        设置DDP通信钩子
        
        Args:
            model: 要包装的模型
            
        Returns:
            torch.nn.Module: 包装后的DDP模型
        """
        try:
            # 确保模型在正确的设备上
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
            model = model.to(device)
            
            # 创建DDP模型
            ddp_model = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=True,  # 处理动态图
                gradient_as_bucket_view=True,  # 内存优化
                broadcast_buffers=True,
                bucket_cap_mb=25  # 桶大小限制
            )
            
            # 注册通信钩子
            if self.config.enable_mixed_precision:
                # 使用FP16压缩钩子
                ddp_model.register_comm_hook(None, ddp_hooks.fp16_compress_hook)
                self.logger.info("已注册FP16压缩通信钩子")
            else:
                # 使用AllReduce钩子
                ddp_model.register_comm_hook(None, ddp_hooks.allreduce_hook)
                self.logger.info("已注册AllReduce通信钩子")
            
            return ddp_model
            
        except Exception as e:
            self.logger.error(f"设置DDP钩子失败: {e}")
            raise
    
    def synchronize_gradients(self, model: torch.nn.Module) -> bool:
        """
        手动同步梯度
        
        Args:
            model: 模型
            
        Returns:
            bool: 同步是否成功
        """
        if not self.sync_enabled or not dist.is_initialized():
            return True
        
        try:
            start_time = time.time()
            
            # 收集所有参数的梯度
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.data)
            
            if not gradients:
                self.logger.warning("没有找到梯度，跳过同步")
                return True
            
            # 执行AllReduce操作
            for grad in gradients:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                # 平均化梯度
                grad.div_(dist.get_world_size())
            
            # 更新通信指标
            sync_time = time.time() - start_time
            self.communication_metrics.allreduce_time += sync_time
            self.communication_metrics.total_communication_time += sync_time
            
            # 计算通信数据量
            total_params = sum(grad.numel() for grad in gradients)
            data_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
            self.communication_metrics.communication_volume += data_size_mb
            
            self.logger.debug(f"梯度同步完成，耗时: {sync_time:.4f}秒，数据量: {data_size_mb:.2f}MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"梯度同步失败: {e}")
            return False
    
    def aggregate_gradients(self, gradients: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        聚合来自不同GPU的梯度
        
        Args:
            gradients: GPU ID到梯度张量的映射
            
        Returns:
            torch.Tensor: 聚合后的梯度
        """
        try:
            if not gradients:
                raise ValueError("没有提供梯度数据")
            
            # 获取第一个梯度作为模板
            first_grad = next(iter(gradients.values()))
            aggregated = torch.zeros_like(first_grad)
            
            # 累加所有梯度
            for gpu_id, grad in gradients.items():
                if grad.shape != first_grad.shape:
                    raise ValueError(f"GPU {gpu_id} 的梯度形状不匹配")
                aggregated += grad
            
            # 平均化
            aggregated /= len(gradients)
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"梯度聚合失败: {e}")
            raise
    
    def clip_gradients(self, model: torch.nn.Module, max_norm: float) -> float:
        """
        梯度裁剪
        
        Args:
            model: 模型
            max_norm: 最大梯度范数
            
        Returns:
            float: 实际的梯度范数
        """
        try:
            # 计算梯度范数
            parameters = [p for p in model.parameters() if p.grad is not None]
            if not parameters:
                return 0.0
            
            # 使用PyTorch内置的梯度裁剪
            total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
            
            return total_norm.item()
            
        except Exception as e:
            self.logger.error(f"梯度裁剪失败: {e}")
            return 0.0
    
    def validate_gradient_consistency(self, model: torch.nn.Module) -> bool:
        """
        验证梯度一致性
        
        Args:
            model: 模型
            
        Returns:
            bool: 梯度是否一致
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return True
        
        try:
            # 收集本地梯度信息
            local_grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    norm = param.grad.data.norm().item()
                    local_grad_norms.append(norm)
            
            if not local_grad_norms:
                return True
            
            # 创建张量用于通信
            local_tensor = torch.tensor(local_grad_norms, dtype=torch.float32)
            if torch.cuda.is_available():
                local_tensor = local_tensor.cuda()
            
            # 收集所有进程的梯度范数
            gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_tensors, local_tensor)
            
            # 检查一致性
            tolerance = 1e-6
            for i, tensor in enumerate(gathered_tensors):
                if not torch.allclose(local_tensor, tensor, atol=tolerance):
                    self.logger.warning(f"梯度不一致，rank {dist.get_rank()} vs rank {i}")
                    return False
            
            self.logger.debug("梯度一致性验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"梯度一致性验证失败: {e}")
            return False
    
    def get_communication_metrics(self) -> CommunicationMetrics:
        """获取通信指标"""
        return self.communication_metrics
    
    def reset_metrics(self):
        """重置通信指标"""
        self.communication_metrics = CommunicationMetrics()
    
    def enable_sync(self):
        """启用梯度同步"""
        self.sync_enabled = True
    
    def disable_sync(self):
        """禁用梯度同步"""
        self.sync_enabled = False


class ParameterUpdateManager:
    """参数更新管理器"""
    
    def __init__(self, config: ParallelConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.update_count = 0
        self.last_update_time = None
        
    def update_parameters(self, optimizer: torch.optim.Optimizer, 
                         gradient_synchronizer: GradientSynchronizer,
                         model: torch.nn.Module) -> bool:
        """
        更新模型参数
        
        Args:
            optimizer: 优化器
            gradient_synchronizer: 梯度同步器
            model: 模型
            
        Returns:
            bool: 更新是否成功
        """
        try:
            start_time = time.time()
            
            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                grad_norm = gradient_synchronizer.clip_gradients(model, self.config.max_grad_norm)
                if grad_norm > self.config.max_grad_norm * 2:
                    self.logger.warning(f"梯度范数过大: {grad_norm:.4f}")
            
            # 验证梯度一致性
            if not gradient_synchronizer.validate_gradient_consistency(model):
                self.logger.error("梯度一致性验证失败，跳过参数更新")
                return False
            
            # 执行参数更新
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新统计信息
            self.update_count += 1
            self.last_update_time = time.time()
            update_time = self.last_update_time - start_time
            
            self.logger.debug(f"参数更新完成，耗时: {update_time:.4f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"参数更新失败: {e}")
            return False
    
    def broadcast_parameters(self, model: torch.nn.Module, src_rank: int = 0) -> bool:
        """
        广播模型参数
        
        Args:
            model: 模型
            src_rank: 源rank
            
        Returns:
            bool: 广播是否成功
        """
        if not dist.is_initialized():
            return True
        
        try:
            self.logger.info(f"从rank {src_rank}广播模型参数...")
            
            for param in model.parameters():
                dist.broadcast(param.data, src=src_rank)
            
            self.logger.info("模型参数广播完成")
            return True
            
        except Exception as e:
            self.logger.error(f"参数广播失败: {e}")
            return False
    
    def synchronize_buffers(self, model: torch.nn.Module) -> bool:
        """
        同步模型缓冲区（如BatchNorm的running_mean和running_var）
        
        Args:
            model: 模型
            
        Returns:
            bool: 同步是否成功
        """
        if not dist.is_initialized():
            return True
        
        try:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
            
            self.logger.debug("模型缓冲区同步完成")
            return True
            
        except Exception as e:
            self.logger.error(f"缓冲区同步失败: {e}")
            return False
    
    def get_update_stats(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        return {
            "update_count": self.update_count,
            "last_update_time": self.last_update_time,
            "average_updates_per_second": self._calculate_update_rate()
        }
    
    def _calculate_update_rate(self) -> float:
        """计算更新速率"""
        if self.update_count == 0 or self.last_update_time is None:
            return 0.0
        
        # 简化计算，假设从开始到现在的平均速率
        elapsed_time = time.time() - (self.last_update_time - self.update_count * 0.1)  # 粗略估算
        return self.update_count / max(elapsed_time, 1.0)


class DistributedTrainingEngine:
    """分布式训练引擎主类"""
    
    def __init__(self, config: ParallelConfig, topology: GPUTopology, logger: Optional[logging.Logger] = None):
        self.config = config
        self.topology = topology
        self.logger = logger or logging.getLogger(__name__)
        
        # 核心组件
        self.backend_initializer = DistributedBackendInitializer(config, logger)
        self.process_manager = MultiGPUProcessManager(config, topology, logger)
        self.gradient_synchronizer = GradientSynchronizer(config, logger)
        self.parameter_manager = ParameterUpdateManager(config, logger)
        
        # 状态管理
        self.is_initialized = False
        self.training_active = False
        
    def initialize(self, rank: int, world_size: int) -> bool:
        """
        初始化分布式训练引擎
        
        Args:
            rank: 当前进程rank
            world_size: 总进程数
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("初始化分布式训练引擎...")
            
            # 初始化分布式后端
            if not self.backend_initializer.initialize_backend(rank, world_size):
                return False
            
            self.is_initialized = True
            self.logger.info("分布式训练引擎初始化成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"分布式训练引擎初始化失败: {e}")
            return False
    
    def setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        设置分布式模型
        
        Args:
            model: 原始模型
            
        Returns:
            torch.nn.Module: 分布式包装后的模型
        """
        if not self.is_initialized:
            raise RuntimeError("训练引擎未初始化")
        
        return self.gradient_synchronizer.setup_ddp_hooks(model)
    
    def train_step(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   loss: torch.Tensor) -> bool:
        """
        执行一个训练步骤
        
        Args:
            model: 模型
            optimizer: 优化器
            loss: 损失值
            
        Returns:
            bool: 训练步骤是否成功
        """
        try:
            # 反向传播
            loss.backward()
            
            # 更新参数
            success = self.parameter_manager.update_parameters(
                optimizer, self.gradient_synchronizer, model
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"训练步骤失败: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("清理分布式训练引擎...")
        
        self.training_active = False
        
        # 清理各个组件
        self.process_manager.cleanup()
        self.backend_initializer.cleanup()
        
        self.is_initialized = False
        self.logger.info("分布式训练引擎清理完成")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取训练指标"""
        return {
            "communication_metrics": self.gradient_synchronizer.get_communication_metrics().to_dict(),
            "parameter_update_stats": self.parameter_manager.get_update_stats(),
            "process_status": self.process_manager.get_process_status(),
            "backend_info": self.backend_initializer.get_backend_info()
        }
    
    @contextmanager
    def distributed_context(self, rank: int, world_size: int):
        """分布式训练上下文管理器"""
        try:
            # 初始化
            if not self.initialize(rank, world_size):
                raise RuntimeError("分布式训练引擎初始化失败")
            
            yield self
            
        finally:
            # 清理
            self.cleanup()