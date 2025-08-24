"""
Qwen3-4B-Thinking模型配置模块
集成模型和tokenizer配置，支持中文密码学领域优化
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType


@dataclass
class QwenModelConfig:
    """Qwen3-4B-Thinking模型配置"""
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    model_revision: str = "main"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    trust_remote_code: bool = True
    
    # 模型加载配置
    torch_dtype: str = "auto"  # auto, float16, bfloat16, float32
    device_map: str = "auto"   # auto, balanced, sequential
    low_cpu_mem_usage: bool = True
    
    # 量化配置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # 中文处理优化
    pad_token: str = "<|endoftext|>"
    eos_token: str = "<|im_end|>"
    bos_token: str = "<|im_start|>"
    unk_token: str = "<|endoftext|>"
    
    # 序列长度配置
    max_seq_length: int = 2048
    max_position_embeddings: int = 32768
    
    # 特殊配置
    use_cache: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False


@dataclass
class LoRATrainingConfig:
    """LoRA微调配置"""
    # LoRA基础参数
    r: int = 16  # rank
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    
    # 目标模块
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 任务类型
    task_type: str = "CAUSAL_LM"
    
    # 推理模式
    inference_mode: bool = False
    
    # 内存优化
    use_rslora: bool = False
    use_dora: bool = False


@dataclass
class ChineseProcessingConfig:
    """中文处理配置"""
    # 分词器配置
    tokenizer_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    add_special_tokens: bool = True
    padding_side: str = "left"
    truncation_side: str = "right"
    
    # 中文特殊处理
    enable_traditional_conversion: bool = True
    normalize_punctuation: bool = True
    handle_emoji: bool = True
    
    # 密码学词汇
    crypto_vocab_path: Optional[str] = None
    extend_vocab: bool = False
    
    # 思考标签处理
    thinking_start_token: str = "<thinking>"
    thinking_end_token: str = "</thinking>"
    preserve_thinking_structure: bool = True


class QwenModelManager:
    """Qwen3-4B-Thinking模型管理器"""
    
    def __init__(self, config: QwenModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            self.logger.warning("CUDA不可用，使用CPU")
            return torch.device("cpu")
    
    def _create_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """创建BitsAndBytes量化配置"""
        if not (self.config.load_in_8bit or self.config.load_in_4bit):
            return None
        
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            )
        elif self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        
        return None
    
    def load_tokenizer(self, chinese_config: Optional[ChineseProcessingConfig] = None) -> AutoTokenizer:
        """加载和配置tokenizer"""
        try:
            self.logger.info(f"加载tokenizer: {self.config.model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=self.config.model_revision,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # 配置特殊token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = self.config.pad_token
            if tokenizer.eos_token is None:
                tokenizer.eos_token = self.config.eos_token
            if tokenizer.bos_token is None:
                tokenizer.bos_token = self.config.bos_token
            if tokenizer.unk_token is None:
                tokenizer.unk_token = self.config.unk_token
            
            # 中文处理配置
            if chinese_config:
                tokenizer.padding_side = chinese_config.padding_side
                tokenizer.truncation_side = chinese_config.truncation_side
                
                # 添加思考标签
                if chinese_config.preserve_thinking_structure:
                    special_tokens = [
                        chinese_config.thinking_start_token,
                        chinese_config.thinking_end_token
                    ]
                    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            
            self.tokenizer = tokenizer
            self.logger.info(f"Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Tokenizer加载失败: {e}")
            raise
    
    def load_model(self) -> AutoModelForCausalLM:
        """加载Qwen3-4B-Thinking模型"""
        try:
            self.logger.info(f"加载模型: {self.config.model_name}")
            
            # 创建量化配置
            quantization_config = self._create_bnb_config()
            
            # 确定torch数据类型
            torch_dtype = self.config.torch_dtype
            if torch_dtype == "auto":
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            else:
                torch_dtype = getattr(torch, torch_dtype)
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                revision=self.config.model_revision,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map=self.config.device_map,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                quantization_config=quantization_config,
                use_cache=self.config.use_cache,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
            
            # 如果tokenizer已加载且添加了新token，需要调整embedding
            if self.tokenizer and len(self.tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(self.tokenizer))
                self.logger.info(f"调整embedding大小到: {len(self.tokenizer)}")
            
            self.model = model
            self.logger.info("模型加载成功")
            
            # 打印模型信息
            self._print_model_info()
            
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def create_lora_config(self, lora_config: LoRATrainingConfig) -> LoraConfig:
        """创建LoRA配置"""
        return LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=lora_config.inference_mode,
            use_rslora=lora_config.use_rslora,
            use_dora=lora_config.use_dora,
        )
    
    def _print_model_info(self) -> None:
        """打印模型信息"""
        if self.model is None:
            return
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型总参数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        self.logger.info(f"参数占用内存: {total_params * 4 / 1024**2:.2f} MB (FP32)")
        
        # GPU内存使用
        if torch.cuda.is_available() and self.model.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(f"GPU内存使用: {memory_allocated:.2f} MB")
            self.logger.info(f"GPU内存预留: {memory_reserved:.2f} MB")
    
    def validate_model_compatibility(self) -> Dict[str, bool]:
        """验证模型兼容性"""
        validation_results = {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "cuda_compatible": False,
            "thinking_tokens_added": False,
            "chinese_support": False
        }
        
        # CUDA兼容性
        if self.model and torch.cuda.is_available():
            validation_results["cuda_compatible"] = self.model.device.type == 'cuda'
        
        # 思考标签支持
        if self.tokenizer:
            thinking_tokens = ["<thinking>", "</thinking>"]
            validation_results["thinking_tokens_added"] = all(
                token in self.tokenizer.get_vocab() for token in thinking_tokens
            )
        
        # 中文支持测试
        if self.tokenizer:
            chinese_text = "这是一个中文测试文本，包含密码学术语：AES加密算法"
            try:
                tokens = self.tokenizer.encode(chinese_text)
                decoded = self.tokenizer.decode(tokens)
                validation_results["chinese_support"] = "中文" in decoded
            except Exception:
                validation_results["chinese_support"] = False
        
        return validation_results
    
    def get_model_memory_usage(self) -> Dict[str, float]:
        """获取模型内存使用情况"""
        memory_info = {}
        
        if self.model is None:
            return memory_info
        
        # 计算模型参数内存
        total_params = sum(p.numel() for p in self.model.parameters())
        param_memory = total_params * 4 / 1024**2  # FP32 MB
        
        memory_info["total_parameters"] = total_params
        memory_info["parameter_memory_mb"] = param_memory
        
        # GPU内存使用
        if torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            memory_info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        
        return memory_info
    
    def optimize_for_training(self, enable_gradient_checkpointing: bool = True) -> None:
        """优化模型用于训练"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 启用梯度检查点
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.logger.info("已启用梯度检查点")
        
        # 设置训练模式
        self.model.train()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("已清理GPU缓存")


def create_default_configs() -> tuple[QwenModelConfig, LoRATrainingConfig, ChineseProcessingConfig]:
    """创建默认配置"""
    model_config = QwenModelConfig()
    lora_config = LoRATrainingConfig()
    chinese_config = ChineseProcessingConfig()
    
    return model_config, lora_config, chinese_config


def main():
    """测试模型配置功能"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    model_config, lora_config, chinese_config = create_default_configs()
    
    # 创建模型管理器
    manager = QwenModelManager(model_config)
    
    try:
        # 加载tokenizer
        tokenizer = manager.load_tokenizer(chinese_config)
        print(f"Tokenizer词汇表大小: {len(tokenizer)}")
        
        # 测试中文编码
        chinese_text = "这是一个包含<thinking>思考过程</thinking>的中文测试"
        tokens = tokenizer.encode(chinese_text)
        decoded = tokenizer.decode(tokens)
        print(f"中文编码测试: {decoded}")
        
        # 验证兼容性
        compatibility = manager.validate_model_compatibility()
        print("兼容性检查:", compatibility)
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()