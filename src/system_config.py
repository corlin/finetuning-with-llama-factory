"""
系统配置管理

本模块实现了系统配置的加载、验证、保存和管理功能。
支持YAML和JSON格式的配置文件，提供配置模板生成和环境变量覆盖功能。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import json
import os
from enum import Enum
import logging
from datetime import datetime

from src.parallel_config import ParallelStrategy, CommunicationBackend, ZeroStage


class ConfigFormat(Enum):
    """配置文件格式"""
    YAML = "yaml"
    JSON = "json"


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class EnvironmentConfig:
    """环境配置"""
    python_version: str = "3.12+"
    cuda_version: str = "12.9"
    pytorch_version: str = "2.1.0"
    transformers_version: str = "4.36.0"
    
    # 环境变量
    cuda_visible_devices: Optional[str] = None
    master_addr: str = "localhost"
    master_port: int = 29500
    world_size: int = 1
    rank: int = 0
    
    def __post_init__(self):
        """从环境变量读取配置"""
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", self.cuda_visible_devices)
        self.master_addr = os.getenv("MASTER_ADDR", self.master_addr)
        self.master_port = int(os.getenv("MASTER_PORT", str(self.master_port)))
        self.world_size = int(os.getenv("WORLD_SIZE", str(self.world_size)))
        self.rank = int(os.getenv("RANK", str(self.rank)))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # 模型参数
    max_seq_length: int = 2048
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    
    # 特殊配置
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    padding_side: str = "right"
    
    def __post_init__(self):
        """配置验证"""
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length必须大于0")
        
        if self.padding_side not in ["left", "right"]:
            raise ValueError("padding_side必须是'left'或'right'")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 3
    batch_size: int = 4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # 优化器配置
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler: str = "cosine"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # LoRA配置
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # 混合精度
    fp16: bool = False
    bf16: bool = True
    dataloader_pin_memory: bool = True
    
    # 保存和日志
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 500
    
    def __post_init__(self):
        """配置验证和调整"""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs必须大于0")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate必须在0-1之间")
        
        if self.lora_rank <= 0:
            raise ValueError("lora_rank必须大于0")
        
        # 自动调整配置
        if self.per_device_batch_size * self.gradient_accumulation_steps != self.batch_size:
            self.gradient_accumulation_steps = max(1, self.batch_size // self.per_device_batch_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    
    # 数据处理
    data_format: str = "json"  # json, jsonl, csv
    max_samples: Optional[int] = None
    shuffle: bool = True
    
    # 分割配置
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 中文处理
    enable_chinese_processing: bool = True
    chinese_tokenizer: str = "chinese-roberta-wwm-ext"
    enable_traditional_conversion: bool = True
    
    # 密码学配置
    crypto_vocab_path: str = "data/crypto_vocab.json"
    enable_crypto_term_detection: bool = True
    
    def __post_init__(self):
        """配置验证"""
        if not abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6:
            raise ValueError("训练、验证、测试集比例之和必须等于1.0")
        
        if self.data_format not in ["json", "jsonl", "csv"]:
            raise ValueError("data_format必须是'json'、'jsonl'或'csv'")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class SystemConfig:
    """系统总配置"""
    # 子配置
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 并行配置
    parallel_strategy: ParallelStrategy = ParallelStrategy.AUTO
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    communication_backend: CommunicationBackend = CommunicationBackend.NCCL
    enable_zero_optimization: bool = True
    zero_stage: ZeroStage = ZeroStage.OPTIMIZER_GRADIENT
    
    # 系统配置
    output_dir: str = "output"
    cache_dir: str = "cache"
    log_dir: str = "logs"
    log_level: LogLevel = LogLevel.INFO
    
    # GPU配置
    gpu_memory_limit: Optional[int] = None  # MB
    enable_multi_gpu: bool = False
    gpu_ids: List[int] = field(default_factory=list)
    
    # 监控配置
    enable_monitoring: bool = True
    enable_distributed_monitoring: bool = True
    log_gpu_stats: bool = True
    monitoring_interval: int = 60  # 秒
    
    # 专家评估配置
    enable_expert_evaluation: bool = True
    expert_evaluation_interval: int = 1000  # steps
    
    # 其他配置
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    
    def __post_init__(self):
        """配置验证和初始化"""
        # 创建必要的目录
        for dir_path in [self.output_dir, self.cache_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # 验证GPU配置
        if self.enable_multi_gpu and not self.gpu_ids:
            # 自动检测GPU
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_ids = list(range(torch.cuda.device_count()))
                else:
                    self.enable_multi_gpu = False
            except ImportError:
                self.enable_multi_gpu = False
        
        # 调整并行配置
        if len(self.gpu_ids) == 1:
            self.enable_multi_gpu = False
            self.data_parallel_size = 1
            self.tensor_parallel_size = 1
            self.pipeline_parallel_size = 1
    
    @property
    def world_size(self) -> int:
        """计算总进程数"""
        return self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
    
    @property
    def num_gpus(self) -> int:
        """获取GPU数量"""
        return len(self.gpu_ids) if self.gpu_ids else 1
    
    def get_device_map(self) -> Dict[str, int]:
        """获取设备映射"""
        if not self.enable_multi_gpu or not self.gpu_ids:
            return {"": 0}
        
        device_map = {}
        for i, gpu_id in enumerate(self.gpu_ids):
            device_map[f"gpu_{i}"] = gpu_id
        
        return device_map
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        config_dict = {
            "environment": self.environment.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "parallel_strategy": self.parallel_strategy.value,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "communication_backend": self.communication_backend.value,
            "enable_zero_optimization": self.enable_zero_optimization,
            "zero_stage": self.zero_stage.value,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "log_dir": self.log_dir,
            "log_level": self.log_level.value,
            "gpu_memory_limit": self.gpu_memory_limit,
            "enable_multi_gpu": self.enable_multi_gpu,
            "gpu_ids": self.gpu_ids,
            "enable_monitoring": self.enable_monitoring,
            "enable_distributed_monitoring": self.enable_distributed_monitoring,
            "log_gpu_stats": self.log_gpu_stats,
            "monitoring_interval": self.monitoring_interval,
            "enable_expert_evaluation": self.enable_expert_evaluation,
            "expert_evaluation_interval": self.expert_evaluation_interval,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "benchmark": self.benchmark,
            "world_size": self.world_size,
            "num_gpus": self.num_gpus
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建实例"""
        data_copy = data.copy()
        
        # 处理子配置
        if "environment" in data_copy:
            data_copy["environment"] = EnvironmentConfig(**data["environment"])
        
        if "model" in data_copy:
            data_copy["model"] = ModelConfig(**data["model"])
        
        if "training" in data_copy:
            data_copy["training"] = TrainingConfig(**data["training"])
        
        if "data" in data_copy:
            data_copy["data"] = DataConfig(**data["data"])
        
        # 处理枚举类型
        if "parallel_strategy" in data_copy:
            data_copy["parallel_strategy"] = ParallelStrategy(data["parallel_strategy"])
        
        if "communication_backend" in data_copy:
            data_copy["communication_backend"] = CommunicationBackend(data["communication_backend"])
        
        if "zero_stage" in data_copy:
            data_copy["zero_stage"] = ZeroStage(data["zero_stage"])
        
        if "log_level" in data_copy:
            data_copy["log_level"] = LogLevel(data["log_level"])
        
        # 移除计算属性
        data_copy.pop("world_size", None)
        data_copy.pop("num_gpus", None)
        
        return cls(**data_copy)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器"""
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """加载配置文件"""
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            self.logger.info("未指定配置文件，使用默认配置")
            self.config = SystemConfig()
            return self.config
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            self.logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            self.config = SystemConfig()
            return self.config
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")
            
            self.config = SystemConfig.from_dict(data)
            self.logger.info(f"成功加载配置文件: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            self.logger.info("使用默认配置")
            self.config = SystemConfig()
        
        return self.config
    
    def save_config(self, config: Optional[SystemConfig] = None, 
                   config_path: Optional[str] = None,
                   format: ConfigFormat = ConfigFormat.YAML) -> bool:
        """保存配置文件"""
        if config:
            self.config = config
        
        if not self.config:
            self.logger.error("没有配置可保存")
            return False
        
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            self.logger.error("未指定配置文件路径")
            return False
        
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = self.config.to_dict()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.YAML:
                    yaml.dump(data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                else:  # JSON
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"配置文件保存成功: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            return False
    
    def validate_config(self, config: Optional[SystemConfig] = None) -> Dict[str, Any]:
        """验证配置"""
        if config:
            self.config = config
        
        if not self.config:
            return {"valid": False, "errors": ["没有配置可验证"]}
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # 验证环境配置
            env_result = self._validate_environment_config()
            result["errors"].extend(env_result.get("errors", []))
            result["warnings"].extend(env_result.get("warnings", []))
            
            # 验证模型配置
            model_result = self._validate_model_config()
            result["errors"].extend(model_result.get("errors", []))
            result["warnings"].extend(model_result.get("warnings", []))
            
            # 验证训练配置
            training_result = self._validate_training_config()
            result["errors"].extend(training_result.get("errors", []))
            result["warnings"].extend(training_result.get("warnings", []))
            
            # 验证数据配置
            data_result = self._validate_data_config()
            result["errors"].extend(data_result.get("errors", []))
            result["warnings"].extend(data_result.get("warnings", []))
            
            # 验证并行配置
            parallel_result = self._validate_parallel_config()
            result["errors"].extend(parallel_result.get("errors", []))
            result["warnings"].extend(parallel_result.get("warnings", []))
            
            result["valid"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"配置验证过程中出错: {str(e)}")
        
        return result
    
    def _validate_environment_config(self) -> Dict[str, List[str]]:
        """验证环境配置"""
        result = {"errors": [], "warnings": []}
        
        env = self.config.environment
        
        # 检查CUDA版本兼容性
        if env.cuda_version and not env.cuda_version.startswith("12"):
            result["warnings"].append(f"CUDA版本 {env.cuda_version} 可能与PyTorch不兼容")
        
        # 检查端口可用性
        if not 1024 <= env.master_port <= 65535:
            result["errors"].append(f"主节点端口 {env.master_port} 不在有效范围内")
        
        return result
    
    def _validate_model_config(self) -> Dict[str, List[str]]:
        """验证模型配置"""
        result = {"errors": [], "warnings": []}
        
        model = self.config.model
        
        # 检查序列长度
        if model.max_seq_length > 4096:
            result["warnings"].append(f"序列长度 {model.max_seq_length} 较大，可能导致内存不足")
        
        # 检查模型路径
        if model.model_path and not Path(model.model_path).exists():
            result["warnings"].append(f"模型路径不存在: {model.model_path}")
        
        return result
    
    def _validate_training_config(self) -> Dict[str, List[str]]:
        """验证训练配置"""
        result = {"errors": [], "warnings": []}
        
        training = self.config.training
        
        # 检查批次大小
        if training.batch_size < training.per_device_batch_size:
            result["errors"].append("总批次大小不能小于单设备批次大小")
        
        # 检查LoRA配置
        if training.use_lora and training.lora_rank > 64:
            result["warnings"].append(f"LoRA rank {training.lora_rank} 较大，可能影响训练效率")
        
        # 检查学习率
        if training.learning_rate > 1e-3:
            result["warnings"].append(f"学习率 {training.learning_rate} 较大，可能导致训练不稳定")
        
        return result
    
    def _validate_data_config(self) -> Dict[str, List[str]]:
        """验证数据配置"""
        result = {"errors": [], "warnings": []}
        
        data = self.config.data
        
        # 检查数据路径
        for path_name, path_value in [
            ("训练数据路径", data.train_data_path),
            ("验证数据路径", data.val_data_path),
            ("测试数据路径", data.test_data_path)
        ]:
            if not Path(path_value).exists():
                result["warnings"].append(f"{path_name}不存在: {path_value}")
        
        # 检查密码学词典
        if data.enable_crypto_term_detection and not Path(data.crypto_vocab_path).exists():
            result["warnings"].append(f"密码学词典不存在: {data.crypto_vocab_path}")
        
        return result
    
    def _validate_parallel_config(self) -> Dict[str, List[str]]:
        """验证并行配置"""
        result = {"errors": [], "warnings": []}
        
        # 检查GPU配置
        if self.config.enable_multi_gpu and not self.config.gpu_ids:
            result["errors"].append("启用多GPU但未指定GPU ID")
        
        # 检查并行度
        if self.config.world_size > self.config.num_gpus:
            result["errors"].append(f"并行度 {self.config.world_size} 超过GPU数量 {self.config.num_gpus}")
        
        # 检查通信后端
        if (self.config.communication_backend == CommunicationBackend.NCCL and 
            self.config.num_gpus == 1):
            result["warnings"].append("单GPU环境建议使用GLOO通信后端")
        
        return result
    
    def create_template_config(self, template_path: str, 
                             format: ConfigFormat = ConfigFormat.YAML) -> bool:
        """创建配置模板"""
        try:
            template_config = SystemConfig()
            
            # 添加注释信息
            config_dict = template_config.to_dict()
            config_dict["_template_info"] = {
                "description": "Qwen3-4B-Thinking微调系统配置模板",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "usage": "请根据实际需求修改配置参数"
            }
            
            template_file = Path(template_path)
            template_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(template_file, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False,
                             allow_unicode=True, indent=2)
                else:  # JSON
                    json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"配置模板创建成功: {template_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建配置模板失败: {e}")
            return False
    
    def merge_configs(self, base_config: SystemConfig, 
                     override_config: Dict[str, Any]) -> SystemConfig:
        """合并配置"""
        base_dict = base_config.to_dict()
        
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """深度合并字典"""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(base_dict, override_config)
        return SystemConfig.from_dict(merged_dict)
    
    def get_config(self) -> Optional[SystemConfig]:
        """获取当前配置"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        if not self.config:
            self.logger.error("没有配置可更新")
            return False
        
        try:
            current_dict = self.config.to_dict()
            
            def deep_update(base: Dict, updates: Dict) -> Dict:
                """深度更新字典"""
                for key, value in updates.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_update(base[key], value)
                    else:
                        base[key] = value
                return base
            
            updated_dict = deep_update(current_dict, updates)
            self.config = SystemConfig.from_dict(updated_dict)
            
            self.logger.info("配置更新成功")
            return True
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
            return False