"""
基础配置管理系统
统一管理项目配置，支持YAML和环境变量配置
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from src.model_config import QwenModelConfig, LoRATrainingConfig, ChineseProcessingConfig
from src.gpu_utils import SystemRequirements
from src.expert_evaluation.config import ExpertEvaluationConfig


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    run_name: str = "qwen3-4b-crypto-finetuning"
    
    # 训练超参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # 优化器配置
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 混合精度
    fp16: bool = False
    bf16: bool = True
    dataloader_pin_memory: bool = True
    
    # 保存和评估
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # 日志和监控
    logging_steps: int = 10
    report_to: str = "tensorboard"
    disable_tqdm: bool = False
    
    # 数据处理
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # 特殊配置
    gradient_checkpointing: bool = True
    use_cache: bool = False  # 训练时关闭cache
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    train_data_path: str = "./data/train"
    eval_data_path: str = "./data/eval"
    test_data_path: str = "./data/test"
    
    # 数据格式
    data_format: str = "json"  # json, jsonl, csv
    text_column: str = "text"
    label_column: str = "label"
    
    # 数据分割
    train_split_ratio: float = 0.7
    eval_split_ratio: float = 0.15
    test_split_ratio: float = 0.15
    
    # 数据预处理
    max_samples: Optional[int] = None
    shuffle_data: bool = True
    remove_duplicates: bool = True
    
    # 中文数据特殊处理
    enable_chinese_preprocessing: bool = True
    normalize_chinese_text: bool = True
    handle_traditional_chinese: bool = True
    
    # 密码学数据
    crypto_vocab_file: Optional[str] = None
    preserve_crypto_terms: bool = True
    
    # 思考数据处理
    preserve_thinking_tags: bool = True
    validate_thinking_structure: bool = True


@dataclass
class MultiGPUConfig:
    """多GPU配置"""
    # 分布式训练
    enable_distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"  # nccl, gloo, mpi
    
    # 并行策略
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    
    # 并行参数
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # 内存优化
    zero_stage: int = 2  # DeepSpeed ZeRO stage
    cpu_offload: bool = False
    nvme_offload: bool = False
    
    # 通信优化
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


@dataclass
class SystemConfig:
    """系统配置"""
    # 环境配置
    cuda_visible_devices: Optional[str] = None
    mixed_precision: str = "bf16"  # no, fp16, bf16
    
    # 缓存配置
    cache_dir: str = "./cache"
    hf_cache_dir: Optional[str] = None
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 资源限制
    max_memory_per_gpu: Optional[str] = None
    cpu_count: Optional[int] = None
    
    # 安全配置
    trust_remote_code: bool = True
    use_auth_token: bool = False


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file or "config.yaml"
        self.config_dir = Path("./configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # 初始化配置
        self.model_config = QwenModelConfig()
        self.lora_config = LoRATrainingConfig()
        self.chinese_config = ChineseProcessingConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        self.multigpu_config = MultiGPUConfig()
        self.system_config = SystemConfig()
        self.expert_evaluation_config = ExpertEvaluationConfig()
        
        # 加载配置
        self.load_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        config_path = self.config_dir / self.config_file
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                self._update_configs_from_dict(config_data)
                self.logger.info(f"配置文件加载成功: {config_path}")
                
            except Exception as e:
                self.logger.error(f"配置文件加载失败: {e}")
                self.logger.info("使用默认配置")
        else:
            self.logger.info(f"配置文件不存在: {config_path}，使用默认配置")
            self.save_config()  # 保存默认配置
        
        # 从环境变量更新配置
        self._update_from_env()
    
    def _update_configs_from_dict(self, config_data: Dict[str, Any]) -> None:
        """从字典更新配置"""
        config_mapping = {
            "model": self.model_config,
            "lora": self.lora_config,
            "chinese": self.chinese_config,
            "training": self.training_config,
            "data": self.data_config,
            "multigpu": self.multigpu_config,
            "system": self.system_config,
            "expert_evaluation": self.expert_evaluation_config,
        }
        
        for section_name, section_data in config_data.items():
            if section_name in config_mapping and isinstance(section_data, dict):
                config_obj = config_mapping[section_name]
                
                # 特殊处理专家评估配置
                if section_name == "expert_evaluation":
                    try:
                        self.expert_evaluation_config = ExpertEvaluationConfig.from_dict(section_data)
                    except Exception as e:
                        self.logger.error(f"专家评估配置加载失败: {e}")
                        self.expert_evaluation_config = ExpertEvaluationConfig()
                else:
                    for key, value in section_data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                        else:
                            self.logger.warning(f"未知配置项: {section_name}.{key}")
    
    def _update_from_env(self) -> None:
        """从环境变量更新配置"""
        env_mappings = {
            "CUDA_VISIBLE_DEVICES": ("system_config", "cuda_visible_devices"),
            "HF_CACHE_DIR": ("system_config", "hf_cache_dir"),
            "WORLD_SIZE": ("multigpu_config", "world_size"),
            "LOCAL_RANK": ("multigpu_config", "local_rank"),
            "MASTER_ADDR": ("multigpu_config", "master_addr"),
            "MASTER_PORT": ("multigpu_config", "master_port"),
        }
        
        for env_var, (config_name, attr_name) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                config_obj = getattr(self, config_name)
                
                # 类型转换
                if attr_name in ["world_size", "local_rank", "master_port"]:
                    env_value = int(env_value)
                
                setattr(config_obj, attr_name, env_value)
                self.logger.info(f"从环境变量更新配置: {env_var}={env_value}")
    
    def save_config(self) -> None:
        """保存配置到文件"""
        config_data = {
            "model": asdict(self.model_config),
            "lora": asdict(self.lora_config),
            "chinese": asdict(self.chinese_config),
            "training": asdict(self.training_config),
            "data": asdict(self.data_config),
            "multigpu": asdict(self.multigpu_config),
            "system": asdict(self.system_config),
            "expert_evaluation": self.expert_evaluation_config.to_dict(),
        }
        
        config_path = self.config_dir / self.config_file
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            self.logger.info(f"配置文件保存成功: {config_path}")
            
        except Exception as e:
            self.logger.error(f"配置文件保存失败: {e}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "model": self.model_config,
            "lora": self.lora_config,
            "chinese": self.chinese_config,
            "training": self.training_config,
            "data": self.data_config,
            "multigpu": self.multigpu_config,
            "system": self.system_config,
            "expert_evaluation": self.expert_evaluation_config,
        }
    
    def validate_configs(self) -> Dict[str, bool]:
        """验证配置有效性"""
        validation_results = {
            "paths_exist": True,
            "gpu_config_valid": True,
            "training_params_valid": True,
            "data_splits_valid": True,
            "lora_params_valid": True,
        }
        
        # 验证路径
        required_dirs = [
            self.training_config.output_dir,
            self.training_config.logging_dir,
            self.system_config.cache_dir,
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).parent.exists():
                validation_results["paths_exist"] = False
                self.logger.warning(f"路径不存在: {dir_path}")
        
        # 验证GPU配置
        if self.multigpu_config.enable_distributed:
            if self.multigpu_config.world_size <= 0:
                validation_results["gpu_config_valid"] = False
                self.logger.error("world_size必须大于0")
        
        # 验证训练参数
        if self.training_config.learning_rate <= 0:
            validation_results["training_params_valid"] = False
            self.logger.error("learning_rate必须大于0")
        
        # 验证数据分割比例
        total_ratio = (self.data_config.train_split_ratio + 
                      self.data_config.eval_split_ratio + 
                      self.data_config.test_split_ratio)
        if abs(total_ratio - 1.0) > 0.01:
            validation_results["data_splits_valid"] = False
            self.logger.error(f"数据分割比例总和应为1.0，当前为: {total_ratio}")
        
        # 验证LoRA参数
        if self.lora_config.r <= 0 or self.lora_config.lora_alpha <= 0:
            validation_results["lora_params_valid"] = False
            self.logger.error("LoRA参数r和alpha必须大于0")
        
        # 验证专家评估配置
        expert_validation = self.expert_evaluation_config.validate_config()
        validation_results["expert_evaluation_valid"] = all(expert_validation.values())
        if not validation_results["expert_evaluation_valid"]:
            self.logger.error(f"专家评估配置验证失败: {expert_validation}")
        
        return validation_results
    
    def optimize_for_hardware(self) -> None:
        """根据硬件配置优化参数"""
        from gpu_utils import GPUDetector
        
        detector = GPUDetector()
        gpu_infos = detector.get_all_gpu_info()
        
        if not gpu_infos:
            self.logger.warning("未检测到GPU，使用CPU配置")
            self.training_config.fp16 = False
            self.training_config.bf16 = False
            self.training_config.per_device_train_batch_size = 1
            return
        
        # 单GPU优化
        if len(gpu_infos) == 1:
            gpu_info = gpu_infos[0]
            
            # 根据GPU内存调整批次大小
            if gpu_info.total_memory < 8192:  # 8GB
                self.training_config.per_device_train_batch_size = 1
                self.training_config.gradient_accumulation_steps = 8
                self.model_config.load_in_4bit = True
            elif gpu_info.total_memory < 16384:  # 16GB
                self.training_config.per_device_train_batch_size = 2
                self.training_config.gradient_accumulation_steps = 4
                self.model_config.load_in_8bit = True
            else:
                self.training_config.per_device_train_batch_size = 4
                self.training_config.gradient_accumulation_steps = 2
        
        # 多GPU优化
        elif len(gpu_infos) > 1:
            self.multigpu_config.enable_distributed = True
            self.multigpu_config.world_size = len(gpu_infos)
            
            # 调整批次大小
            total_memory = sum(gpu.total_memory for gpu in gpu_infos)
            if total_memory > 32768:  # 32GB总内存
                self.training_config.per_device_train_batch_size = 2
            else:
                self.training_config.per_device_train_batch_size = 1
        
        self.logger.info("已根据硬件配置优化参数")
    
    def create_training_args(self) -> Dict[str, Any]:
        """创建训练参数字典"""
        args = asdict(self.training_config)
        
        # 添加其他配置
        args.update({
            "model_name_or_path": self.model_config.model_name,
            "tokenizer_name": self.chinese_config.tokenizer_name,
            "cache_dir": self.system_config.cache_dir,
            "trust_remote_code": self.system_config.trust_remote_code,
        })
        
        return args
    
    def reload_config(self) -> bool:
        """热重载配置文件"""
        try:
            old_config = self.get_all_configs().copy()
            self.load_config()
            new_config = self.get_all_configs()
            
            # 检查配置是否有变化
            changes_detected = False
            for section_name, new_section in new_config.items():
                if section_name in old_config:
                    if str(new_section) != str(old_config[section_name]):
                        changes_detected = True
                        self.logger.info(f"配置节 '{section_name}' 已更新")
                else:
                    changes_detected = True
                    self.logger.info(f"新增配置节 '{section_name}'")
            
            if changes_detected:
                self.logger.info("配置热重载完成")
                return True
            else:
                self.logger.info("配置无变化")
                return False
                
        except Exception as e:
            self.logger.error(f"配置热重载失败: {e}")
            return False
    
    def update_expert_evaluation_config(self, 
                                      config_updates: Dict[str, Any]) -> bool:
        """动态更新专家评估配置"""
        try:
            # 获取当前配置字典
            current_config = self.expert_evaluation_config.to_dict()
            
            # 应用更新
            for key, value in config_updates.items():
                if key in current_config:
                    current_config[key] = value
                    self.logger.info(f"更新专家评估配置: {key} = {value}")
                else:
                    self.logger.warning(f"未知的专家评估配置项: {key}")
            
            # 重新创建配置对象
            self.expert_evaluation_config = ExpertEvaluationConfig.from_dict(current_config)
            
            # 验证更新后的配置
            validation_result = self.expert_evaluation_config.validate_config()
            if not all(validation_result.values()):
                self.logger.error(f"配置更新后验证失败: {validation_result}")
                return False
            
            self.logger.info("专家评估配置动态更新成功")
            return True
            
        except Exception as e:
            self.logger.error(f"专家评估配置动态更新失败: {e}")
            return False
    
    def get_expert_evaluation_config(self) -> ExpertEvaluationConfig:
        """获取专家评估配置"""
        return self.expert_evaluation_config
    
    def set_expert_evaluation_config(self, config: ExpertEvaluationConfig) -> bool:
        """设置专家评估配置"""
        try:
            # 验证配置
            validation_result = config.validate_config()
            if not all(validation_result.values()):
                self.logger.error(f"专家评估配置验证失败: {validation_result}")
                return False
            
            self.expert_evaluation_config = config
            self.logger.info("专家评估配置设置成功")
            return True
            
        except Exception as e:
            self.logger.error(f"专家评估配置设置失败: {e}")
            return False
    
    def create_expert_evaluation_config_template(self, 
                                               output_path: Optional[str] = None) -> str:
        """创建专家评估配置模板文件"""
        template_config = ExpertEvaluationConfig()
        
        if output_path is None:
            output_path = self.config_dir / "expert_evaluation_template.json"
        
        try:
            template_config.save_to_file(str(output_path))
            self.logger.info(f"专家评估配置模板已创建: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"创建专家评估配置模板失败: {e}")
            raise


def create_default_config_file():
    """创建默认配置文件"""
    config_manager = ConfigManager()
    config_manager.save_config()
    print(f"默认配置文件已创建: {config_manager.config_dir / config_manager.config_file}")


def main():
    """测试配置管理功能"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 验证配置
    validation_results = config_manager.validate_configs()
    print("配置验证结果:", validation_results)
    
    # 硬件优化
    config_manager.optimize_for_hardware()
    
    # 保存优化后的配置
    config_manager.save_config()
    
    print("配置管理测试完成")


if __name__ == "__main__":
    main()