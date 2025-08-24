"""
LLaMA Factory适配器

本模块实现了与LLaMA Factory框架的集成，包括：
- LLaMA Factory配置文件生成器
- 数据格式转换为LLaMA Factory兼容格式
- 模型配置和训练参数映射
- LoRA配置到LLaMA Factory训练流程的集成
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from data_models import TrainingExample, ThinkingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig, MultiGPUConfig, SystemConfig
from lora_config_optimizer import LoRAMemoryProfile, MultiGPULoRAConfig
from parallel_config import ParallelConfig, GPUTopology


@dataclass
class LlamaFactoryDatasetConfig:
    """LLaMA Factory数据集配置"""
    dataset_name: str
    dataset_info: Dict[str, Any]
    formatting: str = "alpaca"  # alpaca, sharegpt, etc.
    ranking: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_info": self.dataset_info,
            "formatting": self.formatting,
            "ranking": self.ranking
        }


@dataclass
class LlamaFactoryModelConfig:
    """LLaMA Factory模型配置"""
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    model_revision: str = "main"
    quantization_bit: Optional[int] = None
    quantization_type: Optional[str] = None
    rope_scaling: Optional[str] = None
    flash_attn: str = "auto"
    shift_attn: bool = False
    mixture_of_depths: Optional[str] = None
    use_unsloth: bool = False
    visual_inputs: bool = False
    moe_aux_loss_coef: Optional[float] = None
    disable_gradient_checkpointing: bool = False
    upcast_layernorm: bool = False
    upcast_lmhead_output: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        # 移除None值
        return {k: v for k, v in config.items() if v is not None}


@dataclass
class LlamaFactoryLoRAConfig:
    """LLaMA Factory LoRA配置"""
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target: str = "all"
    loraplus_lr_ratio: Optional[float] = None
    loraplus_lr_embedding: Optional[float] = None
    use_rslora: bool = False
    use_dora: bool = False
    pissa_init: bool = False
    pissa_iter: int = 16
    pissa_convert: bool = False
    create_new_adapter: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        return {k: v for k, v in config.items() if v is not None}


@dataclass
class LlamaFactoryTrainingConfig:
    """LLaMA Factory训练配置"""
    # 基础配置
    stage: str = "sft"  # pt, sft, rm, ppo, dpo, kto
    do_train: bool = True
    finetuning_type: str = "lora"
    
    # 数据配置
    dataset: str = ""
    template: str = "qwen"
    cutoff_len: int = 2048
    train_on_prompt: bool = False
    mask_history: bool = True
    
    # 训练参数
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # 优化器
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 混合精度
    fp16: bool = False
    bf16: bool = True
    pure_bf16: bool = False
    tf32: bool = True
    
    # 保存和评估
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    eval_delay: int = 0
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 日志
    logging_steps: int = 10
    log_level: str = "info"
    log_on_each_node: bool = True
    logging_first_step: bool = False
    
    # 其他
    seed: int = 42
    data_seed: Optional[int] = None
    remove_unused_columns: bool = False
    label_names: Optional[List[str]] = None
    resume_from_checkpoint: Optional[str] = None
    
    # 分布式训练
    ddp_timeout: int = 1800
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    ddp_broadcast_buffers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0
    
    # 特殊配置
    plot_loss: bool = True
    val_size: float = 0.1
    preprocessing_num_workers: int = 16
    max_samples: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        return {k: v for k, v in config.items() if v is not None}


class LlamaFactoryDataConverter:
    """LLaMA Factory数据格式转换器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def convert_training_examples(self, 
                                examples: List[TrainingExample],
                                output_file: str,
                                format_type: str = "alpaca") -> bool:
        """
        转换训练样例为LLaMA Factory格式
        
        Args:
            examples: 训练样例列表
            output_file: 输出文件路径
            format_type: 格式类型 (alpaca, sharegpt)
            
        Returns:
            bool: 转换是否成功
        """
        try:
            if format_type == "alpaca":
                converted_data = self._convert_to_alpaca_format(examples)
            elif format_type == "sharegpt":
                converted_data = self._convert_to_sharegpt_format(examples)
            else:
                raise ValueError(f"不支持的格式类型: {format_type}")
            
            # 保存转换后的数据
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"成功转换{len(examples)}个样例到{output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据格式转换失败: {e}")
            return False
    
    def _convert_to_alpaca_format(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """转换为Alpaca格式"""
        converted = []
        
        for example in examples:
            # 构建输出内容
            output = example.output
            if example.has_thinking():
                output = f"<thinking>\n{example.thinking}\n</thinking>\n\n{example.output}"
            
            alpaca_item = {
                "instruction": example.instruction,
                "input": example.input,
                "output": output,
                "system": "你是一个专业的密码学专家，请仔细思考后回答问题。"
            }
            
            # 添加元数据
            if example.crypto_terms:
                alpaca_item["crypto_terms"] = example.crypto_terms
            
            if example.difficulty_level != DifficultyLevel.INTERMEDIATE:
                alpaca_item["difficulty"] = example.difficulty_level.value
            
            converted.append(alpaca_item)
        
        return converted
    
    def _convert_to_sharegpt_format(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """转换为ShareGPT格式"""
        converted = []
        
        for example in examples:
            # 构建对话
            conversations = []
            
            # 系统消息
            conversations.append({
                "from": "system",
                "value": "你是一个专业的密码学专家，请仔细思考后回答问题。"
            })
            
            # 用户消息
            user_message = example.instruction
            if example.input:
                user_message += f"\n\n{example.input}"
            
            conversations.append({
                "from": "human",
                "value": user_message
            })
            
            # 助手回复
            assistant_message = example.output
            if example.has_thinking():
                assistant_message = f"<thinking>\n{example.thinking}\n</thinking>\n\n{example.output}"
            
            conversations.append({
                "from": "gpt",
                "value": assistant_message
            })
            
            sharegpt_item = {
                "conversations": conversations
            }
            
            # 添加元数据
            if example.crypto_terms:
                sharegpt_item["crypto_terms"] = example.crypto_terms
            
            converted.append(sharegpt_item)
        
        return converted
    
    def convert_thinking_examples(self, 
                                examples: List[ThinkingExample],
                                output_file: str,
                                format_type: str = "alpaca") -> bool:
        """
        转换思考样例为LLaMA Factory格式
        
        Args:
            examples: 思考样例列表
            output_file: 输出文件路径
            format_type: 格式类型
            
        Returns:
            bool: 转换是否成功
        """
        try:
            # 转换为TrainingExample格式
            training_examples = []
            for thinking_example in examples:
                training_example = TrainingExample(
                    instruction=thinking_example.instruction,
                    input="",
                    output=thinking_example.final_response,
                    thinking=thinking_example.thinking_process,
                    crypto_terms=thinking_example.crypto_terms,
                    difficulty_level=thinking_example.difficulty_level
                )
                training_examples.append(training_example)
            
            # 使用通用转换方法
            return self.convert_training_examples(training_examples, output_file, format_type)
            
        except Exception as e:
            self.logger.error(f"思考样例转换失败: {e}")
            return False
    
    def validate_converted_data(self, data_file: str, format_type: str = "alpaca") -> Dict[str, Any]:
        """
        验证转换后的数据格式
        
        Args:
            data_file: 数据文件路径
            format_type: 格式类型
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                result["errors"].append("数据应该是列表格式")
                return result
            
            if format_type == "alpaca":
                result = self._validate_alpaca_format(data, result)
            elif format_type == "sharegpt":
                result = self._validate_sharegpt_format(data, result)
            
            # 统计信息
            result["statistics"] = {
                "total_samples": len(data),
                "avg_instruction_length": self._calculate_avg_length(data, "instruction"),
                "avg_output_length": self._calculate_avg_length(data, "output"),
                "thinking_samples": sum(1 for item in data if "<thinking>" in str(item))
            }
            
            result["valid"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"验证过程出错: {e}")
        
        return result
    
    def _validate_alpaca_format(self, data: List[Dict], result: Dict) -> Dict:
        """验证Alpaca格式"""
        required_fields = ["instruction", "input", "output"]
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                result["errors"].append(f"第{i}项不是字典格式")
                continue
            
            for field in required_fields:
                if field not in item:
                    result["errors"].append(f"第{i}项缺少必需字段: {field}")
                elif not isinstance(item[field], str):
                    result["errors"].append(f"第{i}项的{field}字段不是字符串")
            
            # 检查thinking格式
            if "<thinking>" in item.get("output", ""):
                if not self._validate_thinking_tags(item["output"]):
                    result["warnings"].append(f"第{i}项的thinking标签格式可能有问题")
        
        return result
    
    def _validate_sharegpt_format(self, data: List[Dict], result: Dict) -> Dict:
        """验证ShareGPT格式"""
        for i, item in enumerate(data):
            if "conversations" not in item:
                result["errors"].append(f"第{i}项缺少conversations字段")
                continue
            
            conversations = item["conversations"]
            if not isinstance(conversations, list):
                result["errors"].append(f"第{i}项的conversations不是列表")
                continue
            
            for j, conv in enumerate(conversations):
                if not isinstance(conv, dict):
                    result["errors"].append(f"第{i}项第{j}个对话不是字典")
                    continue
                
                if "from" not in conv or "value" not in conv:
                    result["errors"].append(f"第{i}项第{j}个对话缺少必需字段")
        
        return result
    
    def _validate_thinking_tags(self, text: str) -> bool:
        """验证thinking标签格式"""
        open_count = text.count("<thinking>")
        close_count = text.count("</thinking>")
        return open_count == close_count and open_count > 0
    
    def _calculate_avg_length(self, data: List[Dict], field: str) -> float:
        """计算字段平均长度"""
        lengths = []
        for item in data:
            if field in item and isinstance(item[field], str):
                lengths.append(len(item[field]))
        
        return sum(lengths) / len(lengths) if lengths else 0.0


class LlamaFactoryConfigGenerator:
    """LLaMA Factory配置生成器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_training_config(self,
                               training_config: TrainingConfig,
                               data_config: DataConfig,
                               lora_config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                               parallel_config: ParallelConfig,
                               dataset_name: str,
                               output_dir: str = "configs") -> str:
        """
        生成LLaMA Factory训练配置文件
        
        Args:
            training_config: 训练配置
            data_config: 数据配置
            lora_config: LoRA配置
            parallel_config: 并行配置
            dataset_name: 数据集名称
            output_dir: 输出目录
            
        Returns:
            str: 配置文件路径
        """
        try:
            # 创建LLaMA Factory配置
            llamafactory_config = self._create_llamafactory_config(
                training_config, data_config, lora_config, parallel_config, dataset_name
            )
            
            # 保存配置文件
            config_path = Path(output_dir) / f"llamafactory_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(llamafactory_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.logger.info(f"LLaMA Factory配置文件已生成: {config_path}")
            return str(config_path)
            
        except Exception as e:
            self.logger.error(f"生成LLaMA Factory配置失败: {e}")
            raise
    
    def _create_llamafactory_config(self,
                                  training_config: TrainingConfig,
                                  data_config: DataConfig,
                                  lora_config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                                  parallel_config: ParallelConfig,
                                  dataset_name: str) -> Dict[str, Any]:
        """创建LLaMA Factory配置字典"""
        
        # 基础模型配置
        model_config = LlamaFactoryModelConfig()
        
        # LoRA配置
        if isinstance(lora_config, LoRAMemoryProfile):
            lora_rank = lora_config.rank
            lora_alpha = lora_config.alpha
        else:
            lora_rank = lora_config.global_config.rank
            lora_alpha = lora_config.global_config.alpha
        
        llamafactory_lora_config = LlamaFactoryLoRAConfig(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            lora_target="all"
        )
        
        # 训练配置
        llamafactory_training_config = LlamaFactoryTrainingConfig(
            dataset=dataset_name,
            output_dir=training_config.output_dir,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            num_train_epochs=training_config.num_train_epochs,
            lr_scheduler_type=training_config.lr_scheduler_type,
            warmup_ratio=training_config.warmup_ratio,
            weight_decay=training_config.weight_decay,
            adam_beta1=training_config.adam_beta1,
            adam_beta2=training_config.adam_beta2,
            adam_epsilon=training_config.adam_epsilon,
            max_grad_norm=training_config.max_grad_norm,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            save_strategy=training_config.save_strategy,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            evaluation_strategy=training_config.eval_strategy,
            eval_steps=training_config.eval_steps,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            logging_steps=training_config.logging_steps,
            cutoff_len=data_config.max_samples or 2048,
            seed=training_config.seed
        )
        
        # 合并所有配置
        config = {}
        config.update(model_config.to_dict())
        config.update(llamafactory_lora_config.to_dict())
        config.update(llamafactory_training_config.to_dict())
        
        # 添加分布式训练配置
        if parallel_config.world_size > 1:
            config.update({
                "ddp_timeout": 1800,
                "ddp_backend": "nccl",
                "ddp_find_unused_parameters": False,
                "dataloader_pin_memory": True
            })
        
        return config
    
    def generate_dataset_info(self,
                            dataset_name: str,
                            train_file: str,
                            val_file: Optional[str] = None,
                            format_type: str = "alpaca",
                            output_dir: str = "configs") -> str:
        """
        生成数据集信息文件
        
        Args:
            dataset_name: 数据集名称
            train_file: 训练文件路径
            val_file: 验证文件路径
            format_type: 格式类型
            output_dir: 输出目录
            
        Returns:
            str: 数据集信息文件路径
        """
        try:
            dataset_info = {
                dataset_name: {
                    "file_name": train_file,
                    "formatting": format_type,
                    "columns": {
                        "prompt": "instruction",
                        "query": "input", 
                        "response": "output",
                        "system": "system"
                    }
                }
            }
            
            if val_file:
                dataset_info[dataset_name]["file_name"] = [train_file, val_file]
            
            # 保存数据集信息文件
            info_path = Path(output_dir) / "dataset_info.json"
            info_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果文件已存在，合并配置
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    existing_info = json.load(f)
                existing_info.update(dataset_info)
                dataset_info = existing_info
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据集信息文件已生成: {info_path}")
            return str(info_path)
            
        except Exception as e:
            self.logger.error(f"生成数据集信息文件失败: {e}")
            raise


class LlamaFactoryAdapter:
    """LLaMA Factory适配器主类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_converter = LlamaFactoryDataConverter(logger)
        self.config_generator = LlamaFactoryConfigGenerator(logger)
    
    def prepare_training_data(self,
                            examples: List[Union[TrainingExample, ThinkingExample]],
                            output_dir: str,
                            dataset_name: str,
                            format_type: str = "alpaca",
                            train_ratio: float = 0.9) -> Dict[str, str]:
        """
        准备训练数据
        
        Args:
            examples: 训练样例列表
            output_dir: 输出目录
            dataset_name: 数据集名称
            format_type: 格式类型
            train_ratio: 训练集比例
            
        Returns:
            Dict[str, str]: 文件路径字典
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 分割数据
            split_idx = int(len(examples) * train_ratio)
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:] if split_idx < len(examples) else []
            
            # 转换训练数据
            train_file = output_path / f"{dataset_name}_train.json"
            if isinstance(train_examples[0], ThinkingExample):
                success = self.data_converter.convert_thinking_examples(
                    train_examples, str(train_file), format_type
                )
            else:
                success = self.data_converter.convert_training_examples(
                    train_examples, str(train_file), format_type
                )
            
            if not success:
                raise RuntimeError("训练数据转换失败")
            
            result = {"train_file": str(train_file)}
            
            # 转换验证数据
            if val_examples:
                val_file = output_path / f"{dataset_name}_val.json"
                if isinstance(val_examples[0], ThinkingExample):
                    success = self.data_converter.convert_thinking_examples(
                        val_examples, str(val_file), format_type
                    )
                else:
                    success = self.data_converter.convert_training_examples(
                        val_examples, str(val_file), format_type
                    )
                
                if success:
                    result["val_file"] = str(val_file)
            
            # 生成数据集信息文件
            dataset_info_file = self.config_generator.generate_dataset_info(
                dataset_name,
                str(train_file),
                result.get("val_file"),
                format_type,
                str(output_path)
            )
            result["dataset_info_file"] = dataset_info_file
            
            self.logger.info(f"训练数据准备完成: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {e}")
            raise
    
    def create_training_config(self,
                             training_config: TrainingConfig,
                             data_config: DataConfig,
                             lora_config: Union[LoRAMemoryProfile, MultiGPULoRAConfig],
                             parallel_config: ParallelConfig,
                             dataset_name: str,
                             output_dir: str = "configs") -> str:
        """
        创建训练配置
        
        Args:
            training_config: 训练配置
            data_config: 数据配置
            lora_config: LoRA配置
            parallel_config: 并行配置
            dataset_name: 数据集名称
            output_dir: 输出目录
            
        Returns:
            str: 配置文件路径
        """
        return self.config_generator.generate_training_config(
            training_config, data_config, lora_config, parallel_config, dataset_name, output_dir
        )
    
    def validate_integration(self, config_file: str, data_files: Dict[str, str]) -> Dict[str, Any]:
        """
        验证LLaMA Factory集成
        
        Args:
            config_file: 配置文件路径
            data_files: 数据文件路径字典
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "config_validation": {},
            "data_validation": {}
        }
        
        try:
            # 验证配置文件
            if not Path(config_file).exists():
                result["errors"].append(f"配置文件不存在: {config_file}")
                result["valid"] = False
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                result["config_validation"] = self._validate_config(config)
                if not result["config_validation"]["valid"]:
                    result["valid"] = False
                    result["errors"].extend(result["config_validation"]["errors"])
            
            # 验证数据文件
            for file_type, file_path in data_files.items():
                if file_type in ["train_file", "val_file"]:
                    validation = self.data_converter.validate_converted_data(file_path)
                    result["data_validation"][file_type] = validation
                    
                    if not validation["valid"]:
                        result["valid"] = False
                        result["errors"].extend([f"{file_type}: {error}" for error in validation["errors"]])
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"验证过程出错: {e}")
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置文件"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 检查必需字段
        required_fields = [
            "model_name", "dataset", "output_dir", "learning_rate",
            "num_train_epochs", "lora_rank", "lora_alpha"
        ]
        
        for field in required_fields:
            if field not in config:
                result["errors"].append(f"缺少必需配置项: {field}")
        
        # 检查数值范围
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                result["errors"].append("learning_rate必须是正数")
        
        if "lora_rank" in config:
            rank = config["lora_rank"]
            if not isinstance(rank, int) or rank <= 0:
                result["errors"].append("lora_rank必须是正整数")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def generate_training_script(self,
                               config_file: str,
                               output_file: str = "train_llamafactory.py") -> str:
        """
        生成训练脚本
        
        Args:
            config_file: 配置文件路径
            output_file: 输出脚本文件路径
            
        Returns:
            str: 脚本文件路径
        """
        script_content = f'''#!/usr/bin/env python3
"""
LLaMA Factory训练脚本
自动生成于: {datetime.now().isoformat()}
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主训练函数"""
    try:
        # 加载配置
        config_file = "{config_file}"
        if not Path(config_file).exists():
            logger.error(f"配置文件不存在: {{config_file}}")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"加载配置文件: {{config_file}}")
        logger.info(f"模型: {{config.get('model_name', 'Unknown')}}")
        logger.info(f"数据集: {{config.get('dataset', 'Unknown')}}")
        logger.info(f"输出目录: {{config.get('output_dir', 'Unknown')}}")
        
        # 检查LLaMA Factory是否可用
        try:
            from llamafactory.train.tuner import run_exp
            logger.info("LLaMA Factory导入成功")
        except ImportError as e:
            logger.error(f"LLaMA Factory导入失败: {{e}}")
            logger.error("请确保已正确安装LLaMA Factory")
            return False
        
        # 设置环境变量
        os.environ["WANDB_DISABLED"] = "true"  # 禁用wandb
        
        # 启动训练
        logger.info("开始训练...")
        run_exp(config)
        
        logger.info("训练完成!")
        return True
        
    except Exception as e:
        logger.error(f"训练过程出错: {{e}}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        try:
            script_path = Path(output_file)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 设置执行权限
            script_path.chmod(0o755)
            
            self.logger.info(f"训练脚本已生成: {script_path}")
            return str(script_path)
            
        except Exception as e:
            self.logger.error(f"生成训练脚本失败: {e}")
            raise


def main():
    """测试LLaMA Factory适配器"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建适配器
    adapter = LlamaFactoryAdapter()
    
    # 创建测试数据
    test_examples = [
        TrainingExample(
            instruction="解释AES加密算法的工作原理",
            input="",
            output="AES（高级加密标准）是一种对称分组密码算法...",
            thinking="<thinking>用户询问AES算法原理，我需要解释其基本概念、工作流程和安全特性。</thinking>",
            crypto_terms=["AES", "对称加密", "分组密码"],
            difficulty_level=DifficultyLevel.INTERMEDIATE
        )
    ]
    
    try:
        # 准备训练数据
        data_files = adapter.prepare_training_data(
            test_examples,
            "test_output",
            "crypto_test",
            "alpaca"
        )
        
        print(f"数据文件: {data_files}")
        
        # 验证集成
        validation = adapter.validate_integration(
            "test_config.yaml",  # 假设的配置文件
            data_files
        )
        
        print(f"验证结果: {validation}")
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()