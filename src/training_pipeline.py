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

from data_models import TrainingExample, ThinkingExample
from config_manager import TrainingConfig, DataConfig, SystemConfig
from lora_config_optimizer import LoRAMemoryProfile, MultiGPULoRAConfig
from parallel_config import ParallelConfig, GPUTopology, ParallelStrategy
from llamafactory_adapter import LlamaFactoryAdapter
from gpu_utils import GPUDetector

# Import training monitor with error handling
try:
    from training_monitor import TrainingMonitor
except ImportError:
    TrainingMonitor = None

# Import distributed training engine with error handling  
try:
    from distributed_training_engine import MultiGPUProcessManager
except ImportError:
    MultiGPUProcessManager = None


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
        self.llamafactory_adapter = LlamaFactoryAdapter(logger)
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
            
            self.data_files = self.llamafactory_adapter.prepare_training_data(
                self.training_data,
                str(data_output_dir),
                dataset_name,
                "alpaca",  # 使用Alpaca格式
                0.9  # 90%训练，10%验证
            )
            
            self.state.update_stage_progress(PipelineStage.DATA_PREPARATION, 80.0)
            
            # 验证数据文件
            validation = self.llamafactory_adapter.validate_integration(
                "dummy_config.yaml",  # 临时配置文件
                self.data_files
            )
            
            if not validation["data_validation"]["train_file"]["valid"]:
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
            
            # 生成LLaMA Factory配置
            config_file = self.llamafactory_adapter.create_training_config(
                self.training_config,
                self.data_config,
                self.lora_config,
                self.parallel_config,
                dataset_name,
                str(config_output_dir)
            )
            
            self.config_files["llamafactory_config"] = config_file
            
            self.state.update_stage_progress(PipelineStage.CONFIG_GENERATION, 70.0)
            
            # 生成训练脚本
            script_file = self.llamafactory_adapter.generate_training_script(
                config_file,
                str(self.output_dir / "train_script.py")
            )
            
            self.config_files["training_script"] = script_file
            
            self.state.update_stage_progress(PipelineStage.CONFIG_GENERATION, 100.0)
            self.logger.info(f"配置生成完成: {self.config_files}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置生成阶段失败: {e}")
            return False
    
    def _stage_environment_setup(self) -> bool:
        """环境设置阶段"""
        try:
            self.logger.info("执行环境设置阶段...")
            
            # 设置环境变量
            if self.parallel_config.data_parallel_size > 1:
                os.environ["WORLD_SIZE"] = str(self.parallel_config.data_parallel_size)
                os.environ["MASTER_ADDR"] = self.parallel_config.master_addr
                os.environ["MASTER_PORT"] = str(self.parallel_config.master_port)
            
            self.state.update_stage_progress(PipelineStage.ENVIRONMENT_SETUP, 50.0)
            
            # 初始化训练监控器
            gpu_ids = list(range(self.parallel_config.data_parallel_size))
            self.training_monitor = TrainingMonitor(
                gpu_ids=gpu_ids,
                log_dir=str(self.output_dir / "logs"),
                save_interval=100
            )
            
            self.state.update_stage_progress(PipelineStage.ENVIRONMENT_SETUP, 100.0)
            self.logger.info("环境设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"环境设置阶段失败: {e}")
            return False
    
    def _stage_training_execution(self) -> bool:
        """训练执行阶段"""
        try:
            self.logger.info("执行训练阶段...")
            
            # 启动训练监控
            if self.training_monitor:
                self.training_monitor.start_monitoring()
            
            self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 10.0)
            
            # 执行训练脚本
            script_file = self.config_files["training_script"]
            
            # 使用subprocess执行训练
            process = subprocess.Popen(
                [f"uv run python {script_file}"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.output_dir)
            )
            
            # 监控训练进程
            training_thread = threading.Thread(
                target=self._monitor_training_process,
                args=(process,),
                daemon=True
            )
            training_thread.start()
            
            # 等待训练完成
            return_code = process.wait()
            
            if return_code == 0:
                self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, 100.0)
                self.logger.info("训练执行完成")
                return True
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"训练执行失败，返回码: {return_code}")
                self.logger.error(f"错误输出: {stderr}")
                return False
            
        except Exception as e:
            self.logger.error(f"训练执行阶段失败: {e}")
            return False
        finally:
            # 停止训练监控
            if self.training_monitor:
                self.training_monitor.stop_monitoring()
    
    def _monitor_training_process(self, process: subprocess.Popen):
        """监控训练进程"""
        try:
            while process.poll() is None:
                if self.should_stop:
                    process.terminate()
                    break
                
                # 更新训练进度（简化实现）
                current_progress = self.state.stage_progress.get(PipelineStage.TRAINING_EXECUTION, 10.0)
                if current_progress < 90.0:
                    new_progress = min(current_progress + 1.0, 90.0)
                    self.state.update_stage_progress(PipelineStage.TRAINING_EXECUTION, new_progress)
                    self._notify_progress()
                
                time.sleep(10)  # 每10秒更新一次
                
        except Exception as e:
            self.logger.error(f"训练进程监控失败: {e}")
    
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