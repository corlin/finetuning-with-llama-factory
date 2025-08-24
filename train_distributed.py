#!/usr/bin/env python3
"""
直接分布式训练脚本
使用transformers和PEFT进行多GPU训练
"""

import os
import sys
import json
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_logging():
    """设置日志"""
    log_file = f"distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                try:
                    stream = self.stream
                    stream.write(msg + self.terminator)
                    self.flush()
                except UnicodeEncodeError:
                    safe_msg = msg.replace('🎉', '[SUCCESS]').replace('✅', '[OK]').replace('❌', '[FAIL]').replace('💥', '[ERROR]').replace('⚠️', '[WARN]')
                    stream.write(safe_msg + self.terminator)
                    self.flush()
            except Exception:
                self.handleError(record)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            SafeStreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """加载配置"""
    with open('validation_output/training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_dataset():
    """加载数据集"""
    logger = logging.getLogger(__name__)
    
    # 加载训练数据
    with open('validation_output/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 加载验证数据
    with open('validation_output/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    logger.info(f"加载训练数据: {len(train_data)} 个样例")
    logger.info(f"加载验证数据: {len(val_data)} 个样例")
    
    # 转换为Dataset格式
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def preprocess_function(examples, tokenizer, max_length=2048):
    """数据预处理函数"""
    inputs = []
    targets = []
    
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output = examples['output'][i]
        
        # 构建输入文本
        if input_text:
            full_input = f"指令: {instruction}\n输入: {input_text}\n回答: "
        else:
            full_input = f"指令: {instruction}\n回答: "
        
        # 构建完整文本
        full_text = full_input + output
        
        inputs.append(full_input)
        targets.append(full_text)
    
    # Tokenize
    model_inputs = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # 创建labels
    labels = model_inputs["input_ids"].copy()
    
    # 计算输入部分的长度，用于mask
    input_lengths = []
    for inp in inputs:
        input_tokens = tokenizer(inp, add_special_tokens=False)["input_ids"]
        input_lengths.append(len(input_tokens))
    
    # Mask输入部分的labels
    for i, input_length in enumerate(input_lengths):
        labels[i][:input_length] = [-100] * input_length
    
    model_inputs["labels"] = labels
    return model_inputs

def setup_model_and_tokenizer(config):
    """设置模型和tokenizer"""
    logger = logging.getLogger(__name__)
    
    model_name = config['model_name_or_path']
    logger.info(f"加载模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./cache"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.get('bf16', True) else torch.float16,
        device_map=None,  # 我们手动处理设备分配
        cache_dir="./cache"
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config.get('lora_rank', 8),
        lora_alpha=config.get('lora_alpha', 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.get('lora_dropout', 0.1),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
    
    return model, tokenizer

def main():
    """主训练函数"""
    logger = setup_logging()
    logger.info("开始分布式训练")
    
    # 检查GPU
    if not torch.cuda.is_available():
        logger.error("CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_count} 个GPU")
    
    # 加载配置
    config = load_config()
    logger.info("配置加载完成")
    
    # 加载数据
    train_dataset, val_dataset = load_dataset()
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 预处理数据
    logger.info("开始数据预处理...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config.get('cutoff_len', 2048)),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config.get('cutoff_len', 2048)),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    logger.info("数据预处理完成")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config.get('output_dir', './output'),
        overwrite_output_dir=config.get('overwrite_output_dir', True),
        num_train_epochs=config.get('num_train_epochs', 1),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        learning_rate=config.get('learning_rate', 2e-4),
        weight_decay=config.get('weight_decay', 0.01),
        logging_steps=config.get('logging_steps', 5),
        save_steps=config.get('save_steps', 50),
        eval_steps=config.get('eval_steps', 50),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=config.get('warmup_ratio', 0.1),
        bf16=config.get('bf16', True),
        fp16=config.get('fp16', False),
        dataloader_pin_memory=config.get('dataloader_pin_memory', True),
        remove_unused_columns=False,
        report_to=None,  # 禁用wandb等
        seed=42,
        data_seed=42,
        # 分布式训练设置
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练...")
    try:
        trainer.train()
        logger.info("训练完成！")
        
        # 保存模型
        trainer.save_model()
        logger.info(f"模型已保存到: {training_args.output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)