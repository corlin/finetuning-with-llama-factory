"""
训练流水线编排器

本模块实现了端到端训练流水线控制器，包括：
- 数据预处理到模型训练的自动化流程
- 训练状态管理和检查点机制
- 多GPU训练调度和监控
- 完整的训练生命周期管理
"""

import os
import json
import yaml
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.data_models import TrainingExample, ThinkingExample
from src.config_manager import TrainingConfig, DataConfig, SystemConfig
from src.lora_config_optimizer import LoRAMemoryProfile, MultiGPULoRAConfig
from src.parallel_config import ParallelConfig, GPUTopology, ParallelStrategy
from src.gpu_utils import GPUDetector

# Import training monitor with error handling
try:
    from src.training_monitor import TrainingMonitor
except ImportError:
    TrainingMonitor = None

# Import distributed training engine with error handling  
try:
    from src.distributed_training_engine import MultiGPUProcessManager, DistributedBackendInitializer
except ImportError:
    MultiGPUProcessManager = None
    DistributedBackendInitializer = None

# Import memory manager with error handling
try:
    from src.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None


class PipelineStage(Enum):
    """流水线阶段枚举"""
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    CONFIG_GENERATION = "config_generation"
    ENVIRONMENT_SETUP = "environment_setup"
    TRAINING_EXECUTION = "training_execution"
    MONITORING = "monitoring"
    CHECKPOINT_MANAGEMENT = "checkpoint_management"
    EVALUATION = "evaluation"
    COMPLETION = "completion"
    ERROR = "error"


class PipelineStatus(Enum):
    """流水线状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineCheckpoint:
    """流水线检查点"""
    checkpoint_id: str
    stage: PipelineStage
    timestamp: datetime
    state_data: Dict[str, Any]
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage.value,
            "timestamp": self.timestamp.isoformat(),
            "state_data": self.state_data,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineCheckpoint':
        return cls(
            checkpoint_id=data["checkpoint_id"],
            stage=PipelineStage(data["stage"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_data=data["state_data"],
            model_path=data.get("model_path"),
            config_path=data.get("config_path"),
            metrics=data.get("metrics")
        )


@dataclass
class PipelineState:
    """流水线状态"""
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    
    # 阶段进度跟踪
    stage_progress: Dict[PipelineStage, float] = field(default_factory=dict)
    stage_start_times: Dict[PipelineStage, datetime] = field(default_factory=dict)
    stage_end_times: Dict[PipelineStage, datetime] = field(default_factory=dict)
    
    # 检查点管理
    checkpoints: List[PipelineCheckpoint] = field(default_factory=list)
    latest_checkpoint: Optional[PipelineCheckpoint] = None
    
    @property
    def runtime(self) -> Optional[timedelta]:
        """计算运行时间"""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def current_stage_runtime(self) -> Optional[timedelta]:
        """计算当前阶段运行时间"""
        if self.current_stage not in self.stage_start_times:
            return None
        start = self.stage_start_times[self.current_stage]
        end = self.stage_end_times.get(self.current_stage, datetime.now())
        return end - start
    
    def update_stage(self, stage: PipelineStage, progress: float = 0.0):
        """更新阶段"""
        # 结束当前阶段
        if self.current_stage in self.stage_start_times:
            self.stage_end_times[self.current_stage] = datetime.now()
            self.stage_progress[self.current_stage] = 100.0
        
        # 开始新阶段
        self.current_stage = stage
        self.stage_start_times[stage] = datetime.now()
        self.stage_progress[stage] = progress
        
        # 更新总体进度
        self._update_overall_progress()
    
    def update_stage_progress(self, stage: PipelineStage, progress: float):
        """更新阶段进度"""
        self.stage_progress[stage] = progress
        self._update_overall_progress()
    
    def _update_overall_progress(self):
        """更新总体进度"""
        stage_weights = {
            PipelineStage.INITIALIZATION: 5,
            PipelineStage.DATA_PREPARATION: 15,
            PipelineStage.CONFIG_GENERATION: 5,
            PipelineStage.ENVIRONMENT_SETUP: 10,
            PipelineStage.TRAINING_EXECUTION: 50,
            PipelineStage.MONITORING: 5,
            PipelineStage.CHECKPOINT_MANAGEMENT: 5,
            PipelineStage.EVALUATION: 5,
            PipelineStage.COMPLETION: 0
        }
        
        total_weight = sum(stage_weights.values())
        weighted_progress = 0.0
        
        for stage, weight in stage_weights.items():
            stage_progress = self.stage_progress.get(stage, 0.0)
            weighted_progress += (stage_progress / 100.0) * weight
        
        self.progress = (weighted_progress / total_weight) * 100.0
    
    def add_checkpoint(self, checkpoint: PipelineCheckpoint):
        """添加检查点"""
        self.checkpoints.append(checkpoint)
        self.latest_checkpoint = checkpoint
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "progress": self.progress,
            "runtime_seconds": self.runtime.total_seconds() if self.runtime else None,
            "error_message": self.error_message,
            "stage_progress": {stage.value: progress for stage, progress in self.stage_progress.items()},
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "latest_checkpoint": self.latest_checkpoint.to_dict() if self.latest_checkpoint else None
        }


class TrainingPipelineOrchestrator:
    """训练流水线编排器"""
    
    def __init__(self, 
                 pipeline_id: str,
                 output_dir: str = "pipeline_output",
                 logger: Optional[logging.Logger] = None):
        """
        初始化训练流水线编排器
        
        Args:
            pipeline_id: 流水线ID
            output_dir: 输出目录
            logger: 日志记录器
        """
        self.pipeline_id = pipeline_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # 流水线状态
        self.state = PipelineState(pipeline_id=pipeline_id)
        
        # 组件初始化
        self.gpu_detector = GPUDetector()
        self.training_monitor: Optional[TrainingMonitor] = None
        self.process_manager: Optional[MultiGPUProcessManager] = None
        
        # 配置存储
        self.training_config: Optional[TrainingConfig] = None
        self.data_config: Optional[DataConfig] = None
        self.lora_config: Optional[Union[LoRAMemoryProfile, MultiGPULoRAConfig]] = None
        self.parallel_config: Optional[ParallelConfig] = None
        self.system_config: Optional[SystemConfig] = None
        
        # 数据和文件路径
        self.training_data: List[Union[TrainingExample, ThinkingExample]] = []
        self.data_files: Dict[str, str] = {}
        self.config_files: Dict[str, str] = {}
        
        # 回调函数
        self.stage_callbacks: Dict[PipelineStage, List[Callable]] = {}
        self.progress_callbacks: List[Callable[[PipelineState], None]] = []
        
        # 控制标志
        self.should_stop = False
        self.should_pause = False
        
        self.logger.info(f"训练流水线编排器初始化完成: {pipeline_id}")
    
    def add_stage_callback(self, stage: PipelineStage, callback: Callable):
        """添加阶段回调函数"""
        if stage not in self.stage_callbacks:
            self.stage_callbacks[stage] = []
        self.stage_callbacks[stage].append(callback)
    
    def add_progress_callback(self, callback: Callable[[PipelineState], None]):
        """添加进度回调函数"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """通知进度更新"""
        for callback in self.progress_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                self.logger.error(f"进度回调执行失败: {e}")
    
    def _execute_stage_callbacks(self, stage: PipelineStage):
        """执行阶段回调"""
        if stage in self.stage_callbacks:
            for callback in self.stage_callbacks[stage]:
                try:
                    callback(self.state)
                except Exception as e:
                    self.logger.error(f"阶段回调执行失败: {e}")
    
    def configure_pipeline(self,
                         training_data: List[Union[TrainingExample, ThinkingExample]],
                         training_config: TrainingConfig,
                         data_config: DataConfig,
                         lora_config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                         parallel_config: ParallelConfig,
                         system_config: Optional[SystemConfig] = None):
        """
        配置流水线
        
        Args:
            training_data: 训练数据
            training_config: 训练配置
            data_config: 数据配置
            lora_config: LoRA配置
            parallel_config: 并行配置
            system_config: 系统配置
        """
        self.training_data = training_data
        self.training_config = training_config
        self.data_config = data_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.system_config = system_config or SystemConfig()
        
        self.logger.info(f"流水线配置完成，训练数据: {len(training_data)}条")
    
    def run_pipeline(self, resume_from_checkpoint: Optional[str] = None) -> bool:
        """
        运行训练流水线
        
        Args:
            resume_from_checkpoint: 从检查点恢复
            
        Returns:
            bool: 是否成功完成
        """
        try:
            self.state.status = PipelineStatus.RUNNING
            self.state.start_time = datetime.now()
            self._notify_progress()
            
            self.logger.info(f"开始执行训练流水线: {self.pipeline_id}")
            
            # 从检查点恢复
            if resume_from_checkpoint:
                if not self._resume_from_checkpoint(resume_from_checkpoint):
                    return False
            
            # 执行流水线阶段
            pipeline_stages = [
                (PipelineStage.INITIALIZATION, self._stage_initialization),
                (PipelineStage.DATA_PREPARATION, self._stage_data_preparation),
                (PipelineStage.CONFIG_GENERATION, self._stage_config_generation),
                (PipelineStage.ENVIRONMENT_SETUP, self._stage_environment_setup),
                (PipelineStage.TRAINING_EXECUTION, self._stage_training_execution),
                (PipelineStage.EVALUATION, self._stage_evaluation),
                (PipelineStage.COMPLETION, self._stage_completion)
            ]
            
            for stage, stage_func in pipeline_stages:
                if self.should_stop:
                    self.logger.info("流水线被停止")
                    self.state.status = PipelineStatus.CANCELLED
                    return False
                
                while self.should_pause:
                    self.logger.info("流水线已暂停")
                    self.state.status = PipelineStatus.PAUSED
                    time.sleep(1)
                
                self.state.status = PipelineStatus.RUNNING
                self.state.update_stage(stage)
                self._notify_progress()
                self._execute_stage_callbacks(stage)
                
                self.logger.info(f"开始执行阶段: {stage.value}")
                
                if not stage_func():
                    self.logger.error(f"阶段执行失败: {stage.value}")
                    self.state.status = PipelineStatus.FAILED
                    return False
                
                # 创建检查点
                self._create_checkpoint(stage)
                
                self.logger.info(f"阶段执行完成: {stage.value}")
            
            self.state.status = PipelineStatus.COMPLETED
            self.state.end_time = datetime.now()
            self._notify_progress()
            
            self.logger.info(f"训练流水线执行完成: {self.pipeline_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"训练流水线执行失败: {e}")
            self.state.status = PipelineStatus.FAILED
            self.state.error_message = str(e)
            self.state.end_time = datetime.now()
            self._notify_progress()
            return False
    
    def _stage_initialization(self) -> bool:
        """初始化阶段"""
        try:
            self.logger.info("执行初始化阶段...")
            
            # 验证配置
            if not self.training_data:
                raise ValueError("训练数据为空")
            
            if not all([self.training_config, self.data_config, self.lora_config, self.parallel_config]):
                raise ValueError("配置不完整")
            
            # 检测GPU环境
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            if not gpu_infos:
                self.logger.warning("未检测到GPU，将使用CPU训练")
            else:
                self.logger.info(f"检测到{len(gpu_infos)}个GPU")
            
            # 创建输出目录结构
            directories = ["data", "configs", "checkpoints", "logs", "models"]
            for dir_name in directories:
                (self.output_dir / dir_name).mkdir(exist_ok=True)
            
            self.state.update_stage_progress(PipelineStage.INITIALIZATION, 100.0)
            return True
            
        except Exception as e:
            self.logger.error(f"初始化阶段失败: {e}")
            return False
    
    def _stage_data_preparation(self) -> bool:
        """数据准备阶段"""
        try:
            self.logger.info("执行数据准备阶段...")
            
            # 准备训练数据
            data_output_dir = self.output_dir / "data"
            dataset_name = f"{self.pipeline_id}_dataset"
            
            self.state.update_stage_progress(PipelineStage.DATA_PREPARATION, 20.0)
            
            # 直接准备训练数据，不依赖LlamaFactory
            self.data_files = self._prepare_direct_training_data(
                self.training_data,
                str(data_output_dir),
                dataset_name
            )
            
            self.state.update_stage_progress(PipelineStage.DATA_PREPARATION, 80.0)
            
            # 验证数据文件
            if not self._validate_direct_training_data(self.data_files):
                raise RuntimeError("训练数据验证失败")
            
            self.state.update_stage_progress(PipelineStage.DATA_PREPARATION, 100.0)
            self.logger.info(f"数据准备完成: {self.data_files}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据准备阶段失败: {e}")
            return False
    
    def _stage_config_generation(self) -> bool:
        """配置生成阶段"""
        try:
            self.logger.info("执行配置生成阶段...")
            
            config_output_dir = self.output_dir / "configs"
            dataset_name = f"{self.pipeline_id}_dataset"
            
            self.state.update_stage_progress(PipelineStage.CONFIG_GENERATION, 30.0)
            
            # 生成直接训练配置
            config_file = self._create_direct_training_config(
                self.training_config,
                self.data_config,
                self.lora_config,
                self.parallel_config,
                dataset_name,
                str(config_output_dir)
            )
            
            self.config_files["direct_training_config"] = config_file
            
            self.state.update_stage_progress(PipelineStage.CONFIG_GENERATION, 70.0)
            
            # 生成直接训练脚本
            script_file = self._generate_direct_training_script(
                config_file,
                str(self.output_dir / "direct_train_script.py")
            )
            
            self.config_files["training_script"] = script_file
            
            self.state.update_stage_progress(PipelineStage.CONFIG_GENERATION, 100.0)
            self.logger.info(f"配置生成完成: {self.config_files}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置生成阶段失败: {e}")
            return False
    
    def _stage_environment_setup(self) -> bool:
        """环境设置阶段 - 集成自研GPU检测和内存管理"""
        try:
            self.logger.info("执行环境设置阶段...")
            
            # 使用GPU检测器验证环境
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            if not gpu_infos:
                self.logger.warning("未检测到GPU，将使用CPU训练")
            else:
                self.logger.info(f"检测到 {len(gpu_infos)} 个GPU")
                for gpu_info in gpu_infos:
                    self.logger.info(f"GPU {gpu_info.gpu_id}: {gpu_info.name}, "
                                   f"内存: {gpu_info.total_memory}MB")
            
            self.state.update_stage_progress(PipelineStage.ENVIRONMENT_SETUP, 30.0)
            
            # 设置分布式训练环境变量
            if self.parallel_config.world_size > 1:
                os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
                os.environ["MASTER_ADDR"] = self.parallel_config.master_addr
                os.environ["MASTER_PORT"] = str(self.parallel_config.master_port)
                
                # 设置NCCL环境变量
                os.environ["NCCL_DEBUG"] = "INFO"
                os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
                os.environ["NCCL_TIMEOUT"] = "1800"
                
                self.logger.info(f"分布式训练环境设置: world_size={self.parallel_config.world_size}, "
                               f"master_addr={self.parallel_config.master_addr}, "
                               f"master_port={self.parallel_config.master_port}")
            
            self.state.update_stage_progress(PipelineStage.ENVIRONMENT_SETUP, 60.0)
            
            # 初始化训练监控器
            if TrainingMonitor:
                try:
                    gpu_ids = list(range(min(len(gpu_infos), self.parallel_config.world_size)))
                    self.training_monitor = TrainingMonitor(
                        gpu_ids=gpu_ids,
                        log_dir=str(self.output_dir / "logs"),
                        save_interval=100
                    )
                    self.logger.info("训练监控器初始化成功")
                except Exception as e:
                    self.logger.warning(f"训练监控器初始化失败: {e}")
                    self.training_monitor = None
            
            self.state.update_stage_progress(PipelineStage.ENVIRONMENT_SETUP, 100.0)
            self.logger.info("环境设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"环境设置阶段失败: {e}")
            return False
    
    def _stage_training_execution(self) -> bool:
        """训练执行阶段 - 使用自研分布式训练引擎"""
        try:
            self.logger.info("执行训练阶段...")
            
            # 启动训练监控
            if self.training_monitor:
                self.training_monitor.start_monitoring()
            
            self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 10.0)
            
            # 使用分布式训练引擎执行训练
            success = self._execute_distributed_training()
            
            if success:
                self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 100.0)
                self.logger.info("分布式训练执行完成")
                return True
            else:
                self.logger.error("分布式训练执行失败")
                return False
            
        except Exception as e:
            self.logger.error(f"训练执行阶段失败: {e}")
            return False
        finally:
            # 停止训练监控
            if self.training_monitor:
                self.training_monitor.stop_monitoring()
    
    def _execute_distributed_training(self) -> bool:
        """执行分布式训练 - 使用自研训练引擎"""
        try:
            # 导入分布式训练引擎
            from src.distributed_training_engine import MultiGPUProcessManager
            from src.memory_manager import MemoryManager
            
            # 检测GPU拓扑
            gpu_topology = self.gpu_detector.detect_gpu_topology()
            
            # 初始化内存管理器
            memory_manager = MemoryManager({
                "monitoring_interval": 5,
                "enable_auto_adjustment": True,
                "initial_batch_size": self.training_config.per_device_train_batch_size
            })
            
            # 启动内存管理器
            if not memory_manager.start():
                self.logger.warning("内存管理器启动失败，继续训练")
            
            self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 20.0)
            
            # 如果是单GPU训练
            if self.parallel_config.world_size == 1:
                return self._execute_single_gpu_training(memory_manager)
            
            # 多GPU分布式训练
            process_manager = MultiGPUProcessManager(
                config=self.parallel_config,
                topology=gpu_topology,
                logger=self.logger
            )
            
            self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 30.0)
            
            # 启动训练进程
            success = process_manager.spawn_training_processes(
                self._distributed_training_worker,
                self.training_data,
                self.training_config,
                self.data_config,
                self.lora_config,
                memory_manager
            )
            
            if not success:
                self.logger.error("启动分布式训练进程失败")
                return False
            
            self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 50.0)
            
            # 等待训练完成
            success = process_manager.wait_for_completion(timeout=3600 * 8)  # 8小时超时
            
            if success:
                self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 90.0)
                self.logger.info("分布式训练进程全部完成")
            else:
                failed_processes = process_manager.get_failed_processes()
                self.logger.error(f"分布式训练失败，失败进程: {failed_processes}")
            
            # 清理资源
            process_manager.cleanup()
            memory_manager.stop()
            
            return success
            
        except Exception as e:
            self.logger.error(f"分布式训练执行失败: {e}")
            return False
    
    def _execute_single_gpu_training(self, memory_manager) -> bool:
        """执行单GPU训练"""
        try:
            # 导入直接训练模块
            from direct_finetuning_with_existing_modules import DirectTrainer, DirectTrainingConfig
            
            # 创建训练配置
            direct_config = DirectTrainingConfig(
                model_name="Qwen/Qwen3-4B-Thinking-2507",
                data_path=self.data_files.get("train_file", ""),
                output_dir=self.training_config.output_dir,
                max_seq_length=self.data_config.max_samples or 2048,
                batch_size=memory_manager.get_current_batch_size(),
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                learning_rate=self.training_config.learning_rate,
                num_epochs=int(self.training_config.num_train_epochs),
                warmup_ratio=self.training_config.warmup_ratio,
                save_steps=self.training_config.save_steps,
                logging_steps=self.training_config.logging_steps,
                use_gradient_checkpointing=True,
                use_fp16=self.training_config.fp16 or self.training_config.bf16
            )
            
            # 设置LoRA配置
            if hasattr(self.lora_config, 'global_config'):
                direct_config.lora_r = self.lora_config.global_config.rank
                direct_config.lora_alpha = self.lora_config.global_config.alpha
                direct_config.target_modules = self.lora_config.global_config.target_modules
            
            # 创建训练器
            trainer = DirectTrainer(direct_config)
            
            # 设置内存管理器回调
            def memory_callback(recommendations):
                for rec in recommendations:
                    if rec.strategy.value == "reduce_batch_size":
                        new_batch_size = max(1, direct_config.batch_size // 2)
                        direct_config.batch_size = new_batch_size
                        self.logger.info(f"根据内存建议调整批次大小为: {new_batch_size}")
            
            memory_manager.add_optimization_callback(memory_callback)
            
            # 运行训练
            trainer.run()
            
            return True
            
        except Exception as e:
            self.logger.error(f"单GPU训练失败: {e}")
            return False
    
    def _distributed_training_worker(self, rank: int, config: 'ParallelConfig', 
                                   training_data, training_config, data_config, 
                                   lora_config, memory_manager):
        """分布式训练工作进程"""
        try:
            # 初始化分布式后端
            from src.distributed_training_engine import DistributedBackendInitializer
            
            backend_init = DistributedBackendInitializer(config, self.logger)
            if not backend_init.initialize_backend(rank, config.world_size):
                raise RuntimeError(f"Rank {rank} 分布式后端初始化失败")
            
            # 导入直接训练模块
            from direct_finetuning_with_existing_modules import DirectTrainer, DirectTrainingConfig
            
            # 创建分布式训练配置
            direct_config = DirectTrainingConfig(
                model_name="Qwen/Qwen3-4B-Thinking-2507",
                data_path=self.data_files.get("train_file", ""),
                output_dir=f"{training_config.output_dir}/rank_{rank}",
                max_seq_length=data_config.max_samples or 2048,
                batch_size=training_config.per_device_train_batch_size,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                learning_rate=training_config.learning_rate,
                num_epochs=int(training_config.num_train_epochs),
                warmup_ratio=training_config.warmup_ratio,
                save_steps=training_config.save_steps,
                logging_steps=training_config.logging_steps,
                use_gradient_checkpointing=True,
                use_fp16=training_config.fp16 or training_config.bf16
            )
            
            # 设置LoRA配置
            if hasattr(lora_config, 'global_config'):
                direct_config.lora_r = lora_config.global_config.rank
                direct_config.lora_alpha = lora_config.global_config.alpha
                direct_config.target_modules = lora_config.global_config.target_modules
            
            # 创建分布式训练器
            trainer = DirectTrainer(direct_config)
            
            # 运行分布式训练
            trainer.run()
            
            # 清理分布式后端
            backend_init.cleanup()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rank {rank} 训练工作进程失败: {e}")
            return False
    
    def _stage_evaluation(self) -> bool:
        """评估阶段"""
        try:
            self.logger.info("执行评估阶段...")
            
            # 简化的评估实现
            # 在实际应用中，这里应该调用评估框架
            
            self.state.update_stage_progress(PipelineStage.EVALUATION, 50.0)
            
            # 检查输出模型是否存在
            model_output_dir = Path(self.training_config.output_dir)
            if model_output_dir.exists():
                self.logger.info(f"找到训练输出: {model_output_dir}")
            else:
                self.logger.warning("未找到训练输出")
            
            self.state.update_stage_progress(PipelineStage.EVALUATION, 100.0)
            self.logger.info("评估阶段完成")
            return True
            
        except Exception as e:
            self.logger.error(f"评估阶段失败: {e}")
            return False
    
    def _stage_completion(self) -> bool:
        """完成阶段"""
        try:
            self.logger.info("执行完成阶段...")
            
            # 生成最终报告
            report = self._generate_final_report()
            
            report_file = self.output_dir / "pipeline_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"流水线报告已生成: {report_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"完成阶段失败: {e}")
            return False
    
    def _create_checkpoint(self, stage: PipelineStage):
        """创建检查点"""
        try:
            checkpoint_id = f"{self.pipeline_id}_{stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            checkpoint = PipelineCheckpoint(
                checkpoint_id=checkpoint_id,
                stage=stage,
                timestamp=datetime.now(),
                state_data={
                    "data_files": self.data_files,
                    "config_files": self.config_files,
                    "progress": self.state.progress
                }
            )
            
            self.state.add_checkpoint(checkpoint)
            
            # 保存检查点文件
            checkpoint_file = self.output_dir / "checkpoints" / f"{checkpoint_id}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"检查点已创建: {checkpoint_id}")
            
        except Exception as e:
            self.logger.error(f"创建检查点失败: {e}")
    
    def _resume_from_checkpoint(self, checkpoint_id: str) -> bool:
        """从检查点恢复"""
        try:
            checkpoint_file = self.output_dir / "checkpoints" / f"{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                self.logger.error(f"检查点文件不存在: {checkpoint_file}")
                return False
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            checkpoint = PipelineCheckpoint.from_dict(checkpoint_data)
            
            # 恢复状态
            self.data_files = checkpoint.state_data.get("data_files", {})
            self.config_files = checkpoint.state_data.get("config_files", {})
            self.state.current_stage = checkpoint.stage
            self.state.progress = checkpoint.state_data.get("progress", 0.0)
            
            self.logger.info(f"从检查点恢复: {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"从检查点恢复失败: {e}")
            return False
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        return {
            "pipeline_id": self.pipeline_id,
            "execution_summary": self.state.to_dict(),
            "data_files": self.data_files,
            "config_files": self.config_files,
            "training_data_count": len(self.training_data),
            "output_directory": str(self.output_dir),
            "generated_at": datetime.now().isoformat()
        }
    
    def pause_pipeline(self):
        """暂停流水线"""
        self.should_pause = True
        self.logger.info("流水线暂停请求已发送")
    
    def resume_pipeline(self):
        """恢复流水线"""
        self.should_pause = False
        self.logger.info("流水线恢复请求已发送")
    
    def stop_pipeline(self):
        """停止流水线"""
        self.should_stop = True
        self.logger.info("流水线停止请求已发送")
    
    def get_state(self) -> PipelineState:
        """获取流水线状态"""
        return self.state
    
    def get_progress(self) -> float:
        """获取进度百分比"""
        return self.state.progress
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.state.status == PipelineStatus.RUNNING
    
    def is_completed(self) -> bool:
        """检查是否已完成"""
        return self.state.status == PipelineStatus.COMPLETED
    
    def has_failed(self) -> bool:
        """检查是否失败"""
        return self.state.status == PipelineStatus.FAILED
    
    def _prepare_direct_training_data(self, 
                                    training_data: List[Union[TrainingExample, ThinkingExample]],
                                    output_dir: str,
                                    dataset_name: str) -> Dict[str, str]:
        """
        直接准备训练数据，不依赖LlamaFactory
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 分割数据为训练集和验证集
        split_idx = int(len(training_data) * 0.9)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:] if split_idx < len(training_data) else []
        
        # 转换为直接训练格式
        def convert_to_direct_format(examples):
            converted = []
            for example in examples:
                if isinstance(example, ThinkingExample):
                    # 处理ThinkingExample
                    output = example.final_response
                    if example.thinking_process:
                        output = f"<thinking>\n{example.thinking_process}\n</thinking>\n\n{example.final_response}"
                    
                    converted.append({
                        "instruction": example.instruction,
                        "input": "",
                        "output": output,
                        "system": "你是一个专业的密码学专家，请仔细思考后回答问题。"
                    })
                else:
                    # 处理TrainingExample
                    output = example.output
                    if example.thinking:
                        output = f"<thinking>\n{example.thinking}\n</thinking>\n\n{example.output}"
                    
                    converted.append({
                        "instruction": example.instruction,
                        "input": example.input,
                        "output": output,
                        "system": "你是一个专业的密码学专家，请仔细思考后回答问题。"
                    })
            return converted
        
        # 保存训练数据
        train_file = output_path / f"{dataset_name}_train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(convert_to_direct_format(train_data), f, ensure_ascii=False, indent=2)
        
        result = {"train_file": str(train_file)}
        
        # 保存验证数据
        if val_data:
            val_file = output_path / f"{dataset_name}_val.json"
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(convert_to_direct_format(val_data), f, ensure_ascii=False, indent=2)
            result["val_file"] = str(val_file)
        
        self.logger.info(f"直接训练数据准备完成: {result}")
        return result
    
    def _validate_direct_training_data(self, data_files: Dict[str, str]) -> bool:
        """
        验证直接训练数据
        """
        import json
        
        try:
            for file_type, file_path in data_files.items():
                if file_type in ["train_file", "val_file"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not isinstance(data, list) or len(data) == 0:
                        self.logger.error(f"{file_type} 格式错误或为空")
                        return False
                    
                    # 检查必要字段
                    for i, item in enumerate(data[:5]):  # 检查前5个样本
                        required_fields = ["instruction", "output"]
                        for field in required_fields:
                            if field not in item or not item[field]:
                                self.logger.error(f"{file_type} 第{i}个样本缺少{field}字段")
                                return False
            
            self.logger.info("直接训练数据验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
    
    def _create_direct_training_config(self,
                                     training_config: TrainingConfig,
                                     data_config: DataConfig,
                                     lora_config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                                     parallel_config: ParallelConfig,
                                     dataset_name: str,
                                     output_dir: str) -> str:
        """
        创建直接训练配置文件
        """
        import yaml
        from pathlib import Path
        
        # 提取LoRA参数
        if isinstance(lora_config, LoRAMemoryProfile):
            lora_rank = lora_config.rank
            lora_alpha = lora_config.alpha
            target_modules = lora_config.target_modules
        else:
            lora_rank = lora_config.global_config.rank
            lora_alpha = lora_config.global_config.alpha
            target_modules = lora_config.global_config.target_modules
        
        # 使用GPU检测器获取硬件信息
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        total_gpu_memory = sum(gpu.total_memory for gpu in gpu_infos) if gpu_infos else 0
        
        # 根据GPU内存自动调整批次大小
        recommended_batch_size = training_config.per_device_train_batch_size
        if gpu_infos and total_gpu_memory > 0:
            # 对于Qwen3-4B模型，每GB GPU内存大约可以支持batch_size=1
            max_batch_per_gpu = max(1, gpu_infos[0].total_memory // 8192)  # 8GB per batch
            recommended_batch_size = min(recommended_batch_size, max_batch_per_gpu)
        
        # 创建直接训练配置 - 集成自研模块配置
        config = {
            "model_name": "Qwen/Qwen3-4B-Thinking-2507",
            "data_path": str(self.data_files.get("train_file", "")),
            "val_data_path": str(self.data_files.get("val_file", "")),
            "output_dir": training_config.output_dir,
            "max_seq_length": data_config.max_samples or 2048,
            "batch_size": recommended_batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_train_epochs,
            "warmup_ratio": training_config.warmup_ratio,
            "save_steps": training_config.save_steps,
            "logging_steps": training_config.logging_steps,
            "lora_r": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.1,
            "target_modules": target_modules,
            "use_gradient_checkpointing": True,
            "use_fp16": training_config.fp16 or training_config.bf16,
            "world_size": parallel_config.world_size,
            "enable_distributed": parallel_config.world_size > 1,
            # 添加自研模块配置
            "memory_management": {
                "enable_auto_adjustment": True,
                "monitoring_interval": 5,
                "memory_threshold_high": 0.85,
                "memory_threshold_low": 0.6
            },
            "gpu_topology": {
                "num_gpus": len(gpu_infos),
                "total_memory": total_gpu_memory,
                "enable_topology_detection": True
            },
            "distributed_config": {
                "backend": parallel_config.communication_backend.value if hasattr(parallel_config, 'communication_backend') else "nccl",
                "master_addr": parallel_config.master_addr,
                "master_port": parallel_config.master_port,
                "timeout": 1800
            }
        }
        
        # 保存配置文件
        config_path = Path(output_dir) / f"direct_training_config_{dataset_name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        self.logger.info(f"直接训练配置文件已生成: {config_path}")
        self.logger.info(f"推荐批次大小: {recommended_batch_size} (基于GPU内存: {total_gpu_memory}MB)")
        return str(config_path)
    
    def _generate_direct_training_script(self, config_file: str, output_file: str) -> str:
        """
        生成直接训练脚本
        """
        script_content = '''#!/usr/bin/env python3
"""
直接训练脚本 - 不依赖LlamaFactory
自动生成于: {datetime.now().isoformat()}
"""

import sys
import os
sys.path.append('src')

from direct_finetuning_with_existing_modules import DirectTrainer, DirectTrainingConfig
import yaml

def main():
    """主训练函数"""
    try:
        # 加载配置
        config_file = "''' + config_file + '''"
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 创建训练配置
        training_config = DirectTrainingConfig(
            model_name=config_dict.get("model_name", "Qwen/Qwen3-4B-Thinking-2507"),
            data_path=config_dict.get("data_path", ""),
            output_dir=config_dict.get("output_dir", "output"),
            max_seq_length=config_dict.get("max_seq_length", 2048),
            batch_size=config_dict.get("batch_size", 1),
            gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", 8),
            learning_rate=config_dict.get("learning_rate", 1e-4),
            num_epochs=config_dict.get("num_epochs", 3),
            warmup_ratio=config_dict.get("warmup_ratio", 0.1),
            save_steps=config_dict.get("save_steps", 500),
            logging_steps=config_dict.get("logging_steps", 10),
            lora_r=config_dict.get("lora_r", 8),
            lora_alpha=config_dict.get("lora_alpha", 16),
            lora_dropout=config_dict.get("lora_dropout", 0.1),
            target_modules=config_dict.get("target_modules", ["q_proj", "v_proj"]),
            use_gradient_checkpointing=config_dict.get("use_gradient_checkpointing", True),
            use_fp16=config_dict.get("use_fp16", True)
        )
        
        # 创建训练器
        trainer = DirectTrainer(training_config)
        
        # 集成自研模块
        if config_dict.get("memory_management", {}).get("enable_auto_adjustment", False):
            print("🔧 启用内存管理器...")
            from memory_manager import MemoryManager
            memory_manager = MemoryManager(config_dict["memory_management"])
            memory_manager.start()
        
        if config_dict.get("gpu_topology", {}).get("enable_topology_detection", False):
            print("🔧 启用GPU拓扑检测...")
            from gpu_utils import GPUDetector
            gpu_detector = GPUDetector()
            topology = gpu_detector.detect_gpu_topology()
            print(f"检测到GPU拓扑: {topology.topology_type}")
        
        # 运行训练
        trainer.run()
        
        print("✅ 直接训练完成！")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
'''
        
        # 保存脚本文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限
        import stat
        os.chmod(output_file, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        self.logger.info(f"直接训练脚本已生成: {output_file}")
        return output_file


def main():
    """测试训练流水线编排器"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    test_data = [
        TrainingExample(
            instruction="什么是密码学？",
            input="",
            output="密码学是研究信息安全的科学。",
            thinking="<thinking>用户询问密码学的基本概念。</thinking>",
            crypto_terms=["密码学", "信息安全"]
        )
    ]
    
    # 创建配置
    training_config = TrainingConfig(num_train_epochs=1, per_device_train_batch_size=1)
    data_config = DataConfig()
    lora_config = LoRAMemoryProfile(rank=8, alpha=16, target_modules=["q_proj"])
    parallel_config = ParallelConfig(strategy=ParallelStrategy.DATA_PARALLEL)
    
    # 创建流水线编排器
    orchestrator = TrainingPipelineOrchestrator("test_pipeline")
    
    # 配置流水线
    orchestrator.configure_pipeline(
        test_data, training_config, data_config, lora_config, parallel_config
    )
    
    # 添加进度回调
    def progress_callback(state: PipelineState):
        print(f"进度: {state.progress:.1f}% - 阶段: {state.current_stage.value}")
    
    orchestrator.add_progress_callback(progress_callback)
    
    # 运行流水线
    success = orchestrator.run_pipeline()
    
    if success:
        print("流水线执行成功！")
    else:
        print("流水线执行失败！")
    
    print(f"最终状态: {orchestrator.get_state().to_dict()}")


if __name__ == "__main__":
    main()