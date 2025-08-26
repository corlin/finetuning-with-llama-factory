#!/usr/bin/env python3
"""
直接使用已实现模块进行微调的脚本
不依赖LlamaFactory，直接使用PyTorch和已实现的功能模块
使用uv包管理器进行依赖管理
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time
from datetime import datetime
import re

# 导入已实现的模块
sys.path.append('src')
from data_models import TrainingExample, ThinkingExample, ChineseMetrics
from config_manager import TrainingConfig
from memory_manager import MemoryManager, MemoryPressureLevel
from gpu_utils import GPUDetector
from parallel_strategy_recommender import ParallelStrategyRecommender, ParallelStrategy
from training_monitor import TrainingMonitor
from chinese_nlp_processor import ChineseNLPProcessor
from crypto_term_processor import CryptoTermProcessor


@dataclass
class DirectTrainingConfig:
    """直接训练配置"""
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"  # 目标微调模型
    data_path: str = "data/raw"  # 原始markdown数据目录
    output_dir: str = "qwen3_4b_thinking_output"
    max_seq_length: int = 2048  # Thinking模型需要更长序列
    batch_size: int = 1  # 4B模型需要更小批次
    gradient_accumulation_steps: int = 8  # 增加梯度累积
    learning_rate: float = 1e-4  # 更保守的学习率
    num_epochs: int = 15
    warmup_ratio: float = 0.1
    save_steps: int = 50  # 更频繁保存
    logging_steps: int = 5
    
    # LoRA配置 - 针对4B模型优化
    lora_r: int = 240  # 增加rank
    lora_alpha: int = 480  # 增加alpha
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # 内存优化 - 4B模型需要更激进的优化
    use_gradient_checkpointing: bool = True
    use_fp16: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            # Qwen3-4B-Thinking模型的注意力和MLP层
            self.target_modules = [
                #"q_proj","v_proj"
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj"      # MLP层
            ]


class CryptoQADataset(Dataset):
    """密码学QA数据集 - 使用现有预处理方法"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 初始化中文处理器
        try:
            self.chinese_processor = ChineseNLPProcessor()
            print("✅ 初始化中文NLP处理器")
        except Exception as e:
            print(f"⚠️ 中文NLP处理器初始化失败: {e}")
            self.chinese_processor = None
        
        # 初始化密码学术语处理器
        try:
            self.crypto_processor = CryptoTermProcessor()
            print("✅ 初始化密码学术语处理器")
        except Exception as e:
            print(f"⚠️ 密码学术语处理器初始化失败: {e}")
            self.crypto_processor = None
        
        # 处理原始markdown数据
        self.data = self.process_raw_markdown_data(data_path)
        
        print(f"✅ 处理了 {len(self.data)} 条训练数据")
    
    def process_raw_markdown_data(self, data_path: str) -> List[Dict[str, Any]]:
        """处理原始markdown数据 - 使用现有预处理模块"""
        processed_data = []
        
        # 获取所有markdown文件
        markdown_files = []
        if os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(data_path, file))
        else:
            markdown_files = [data_path]
        
        print(f"🔄 处理 {len(markdown_files)} 个markdown文件...")
        
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析markdown内容
                qa_pairs = self.parse_markdown_content(content, file_path)
                
                # 使用现有模块进行数据增强
                enhanced_qa_pairs = self.enhance_qa_pairs(qa_pairs)
                
                processed_data.extend(enhanced_qa_pairs)
                print(f"  ✅ {os.path.basename(file_path)}: 提取了 {len(qa_pairs)} 个QA对，增强后 {len(enhanced_qa_pairs)} 个")
                
            except Exception as e:
                print(f"  ❌ {file_path}: 处理失败 - {e}")
        
        return processed_data
    
    def parse_markdown_content(self, content: str, file_path: str = "") -> List[Dict[str, Any]]:
        """解析markdown内容提取QA对 - 改进的解析逻辑"""
        qa_pairs = []
        
        # 分割内容为问题块
        import re
        
        # 更精确的问题模式匹配
        question_pattern = r'### Q(\d+)[:\s]+(.*?)(?=### Q\d+|$)'
        matches = re.findall(question_pattern, content, re.DOTALL)
        
        print(f"    🔍 在 {os.path.basename(file_path)} 中找到 {len(matches)} 个问题块")
        
        for question_num, question_block in matches:
            try:
                qa_pair = self.extract_qa_from_block(question_block.strip(), question_num, file_path)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                print(f"    ⚠️ 解析问题 Q{question_num} 失败: {e}")
                continue
        
        return qa_pairs
    
    def extract_qa_from_block(self, block: str, question_num: str, file_path: str = "") -> Optional[Dict[str, Any]]:
        """从问题块中提取QA对 - 改进的提取逻辑"""
        lines = block.split('\n')
        if not lines:
            return None
        
        # 第一行是问题
        instruction = lines[0].strip()
        if not instruction:
            return None
        
        # 查找thinking部分和答案部分
        thinking_content = ""
        answer_content = ""
        
        in_thinking = False
        thinking_lines = []
        answer_lines = []
        current_section = "none"
        
        for line in lines[1:]:
            original_line = line
            line = line.strip()
            
            if line.startswith('<thinking>'):
                in_thinking = True
                current_section = "thinking"
                thinking_lines.append(original_line)
            elif line.startswith('</thinking>'):
                in_thinking = False
                current_section = "answer"
                thinking_lines.append(original_line)
            elif in_thinking:
                thinking_lines.append(original_line)
            elif line.startswith(f'A{question_num}:') or (line.startswith('A') and ':' in line and current_section != "thinking"):
                # 答案开始
                current_section = "answer"
                if ':' in line:
                    answer_content_start = line.split(':', 1)[1].strip()
                    if answer_content_start:
                        answer_lines.append(answer_content_start)
            elif current_section == "answer" and line:
                answer_lines.append(original_line.rstrip())
            elif not in_thinking and line and not line.startswith('#') and not line.startswith('###'):
                # 可能是答案内容
                if current_section == "none":
                    current_section = "answer"
                if current_section == "answer":
                    answer_lines.append(original_line.rstrip())
        
        # 组装thinking内容
        if thinking_lines:
            thinking_content = '\n'.join(thinking_lines).strip()
        
        # 组装答案内容
        if answer_lines:
            answer_content = '\n'.join(answer_lines).strip()
        
        # 构建完整输出
        if thinking_content and answer_content:
            output = thinking_content + '\n\n' + answer_content
        elif thinking_content:
            output = thinking_content
        elif answer_content:
            output = answer_content
        else:
            return None
        
        # 估算难度级别
        difficulty = self.estimate_difficulty(instruction, output)
        
        return {
            'instruction': instruction,
            'input': '',
            'output': output,
            'system': '你是一个专业的密码学专家，请仔细思考后回答问题。',
            'difficulty': difficulty,
            'source_file': os.path.basename(file_path),
            'question_id': f"{os.path.basename(file_path)}_Q{question_num}"
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        system = item.get('system', '你是一个专业的密码学专家，请仔细思考后回答问题。')
        
        # 构建完整的对话格式 - 适配Qwen3-4B-Thinking
        if input_text:
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        # 确保thinking标签格式正确
        if '<thinking>' in output_text and '</thinking>' in output_text:
            # 保持thinking格式不变
            full_text = prompt + output_text + "<|im_end|>"
        else:
            # 如果没有thinking标签，直接添加
            full_text = prompt + output_text + "<|im_end|>"
        
        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 计算标签（只对assistant部分计算损失）
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        labels = encoding["input_ids"].clone()
        prompt_length = prompt_encoding["input_ids"].shape[1]
        labels[:, :prompt_length] = -100  # 忽略prompt部分的损失
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    def enhance_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用现有模块增强QA对"""
        enhanced_pairs = []
        
        for qa_pair in qa_pairs:
            try:
                # 基础QA对
                enhanced_pair = qa_pair.copy()
                
                # 使用中文NLP处理器进行文本质量分析
                if self.chinese_processor:
                    instruction_metrics = self.chinese_processor.analyze_text_quality(qa_pair['instruction'])
                    output_metrics = self.chinese_processor.analyze_text_quality(qa_pair['output'])
                    
                    enhanced_pair['chinese_metrics'] = {
                        'instruction_quality': instruction_metrics.overall_score if instruction_metrics else 0.0,
                        'output_quality': output_metrics.overall_score if output_metrics else 0.0
                    }
                
                # 使用密码学术语处理器进行术语分析
                if self.crypto_processor:
                    # 分析问题中的密码学术语
                    instruction_terms = self.crypto_processor.extract_crypto_terms(qa_pair['instruction'])
                    output_terms = self.crypto_processor.extract_crypto_terms(qa_pair['output'])
                    
                    enhanced_pair['crypto_terms'] = {
                        'instruction_terms': [term.term for term in instruction_terms],
                        'output_terms': [term.term for term in output_terms],
                        'total_terms': len(instruction_terms) + len(output_terms)
                    }
                    
                    # 根据术语复杂度调整难度
                    if instruction_terms or output_terms:
                        avg_complexity = np.mean([term.complexity for term in instruction_terms + output_terms])
                        enhanced_pair['difficulty'] = max(enhanced_pair['difficulty'], int(avg_complexity))
                
                enhanced_pairs.append(enhanced_pair)
                
            except Exception as e:
                print(f"    ⚠️ 增强QA对失败: {e}")
                # 如果增强失败，至少保留原始数据
                enhanced_pairs.append(qa_pair)
        
        return enhanced_pairs
    
    def estimate_difficulty(self, instruction: str, output: str) -> int:
        """估算问题难度级别"""
        difficulty_score = 1
        
        # 基于文本长度的初步估算
        total_length = len(instruction) + len(output)
        if total_length > 1000:
            difficulty_score = max(difficulty_score, 3)
        elif total_length > 500:
            difficulty_score = max(difficulty_score, 2)
        
        # 基于thinking标签的存在
        if '<thinking>' in output and '</thinking>' in output:
            difficulty_score = max(difficulty_score, 2)
            
            # 分析thinking内容的复杂度
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1)
                # 如果thinking内容很长或包含多个步骤，提高难度
                if len(thinking_content) > 500:
                    difficulty_score = max(difficulty_score, 3)
                if thinking_content.count('步骤') > 3 or thinking_content.count('分析') > 2:
                    difficulty_score = max(difficulty_score, 3)
        
        # 基于专业术语密度
        crypto_keywords = [
            '加密', '解密', '密钥', '哈希', '签名', '证书', '算法', '协议',
            '对称', '非对称', '公钥', '私钥', '数字签名', '消息认证',
            'AES', 'RSA', 'SHA', 'MD5', 'DES', 'ECC'
        ]
        
        combined_text = instruction + output
        keyword_count = sum(1 for keyword in crypto_keywords if keyword in combined_text)
        if keyword_count > 5:
            difficulty_score = max(difficulty_score, 3)
        elif keyword_count > 2:
            difficulty_score = max(difficulty_score, 2)
        
        return min(difficulty_score, 4)  # 最高难度为4


class DirectTrainer:
    """直接训练器"""
    
    def __init__(self, config: DirectTrainingConfig):
        self.config = config
        self.setup_logging()
        
        # 初始化GPU检测器
        self.gpu_detector = GPUDetector()
        self.gpu_info = self.gpu_detector.get_all_gpu_info()
        print(f"✅ 检测到 {len(self.gpu_info)} 个GPU")
        
        # 初始化内存管理器
        try:
            self.memory_manager = MemoryManager()
            print("✅ 初始化内存管理器")
        except Exception as e:
            print(f"⚠️ 内存管理器初始化失败: {e}")
            self.memory_manager = None
        
        # 初始化训练监控器（暂时跳过以避免依赖问题）
        print("⚠️ 跳过训练监控器初始化（避免依赖问题）")
        self.training_monitor = None
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ 使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def setup_logging(self):
        """设置日志"""
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print(f"🔄 加载模型: {self.config.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 启用梯度检查点
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        print(f"✅ 模型加载完成，参数量: {self.model.num_parameters():,}")
        
        # 配置LoRA
        self.setup_lora()
    
    def setup_lora(self):
        """设置LoRA配置"""
        print("🔄 配置LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("✅ LoRA配置完成")
    
    def create_dataset(self):
        """创建数据集"""
        print("🔄 创建数据集...")
        
        self.train_dataset = CryptoQADataset(
            self.config.data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        print(f"✅ 训练数据集创建完成，样本数: {len(self.train_dataset)}")
        
        # 数据集统计分析
        self.analyze_dataset_statistics()
    
    def create_dataloader(self):
        """创建数据加载器"""
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows兼容
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✅ 数据加载器创建完成，批次数: {len(self.train_dataloader)}")
    
    def setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # 计算总步数
        total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # 学习率调度器
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"✅ 优化器和调度器设置完成，总步数: {total_steps}, 预热步数: {warmup_steps}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始训练...")
        
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📊 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            epoch_steps = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                epoch_steps += 1
                
                # 梯度累积
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # 优化器步骤
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 日志记录
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]
                        
                        print(f"Step {global_step}: Loss = {avg_loss:.4f}, LR = {lr:.2e}")
                        
                        # 内存监控
                        if self.memory_manager and torch.cuda.is_available():
                            try:
                                memory_info = self.memory_manager.get_memory_snapshot(0)
                                print(f"GPU内存: {memory_info.allocated_memory}MB / {memory_info.total_memory}MB")
                            except Exception as e:
                                pass
                        
                        total_loss = 0
                    
                    # 保存检查点
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)
            
            # Epoch结束统计
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            print(f"Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.4f}")
        
        print("✅ 训练完成！")
        
        # 保存训练统计
        self.save_training_statistics(global_step)
        
        # 保存最终模型
        self.save_final_model()
    
    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        torch.save({
            'step': step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        print(f"💾 检查点已保存: {checkpoint_dir}")
    
    def save_final_model(self):
        """保存最终模型"""
        final_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        print(f"💾 最终模型已保存: {final_dir}")
    
    def analyze_dataset_statistics(self):
        """分析数据集统计信息"""
        print("\n📊 数据集统计分析:")
        
        # 基础统计
        total_samples = len(self.train_dataset.data)
        
        # 难度分布
        difficulty_dist = {}
        source_file_dist = {}
        crypto_terms_dist = {}
        
        total_instruction_length = 0
        total_output_length = 0
        thinking_count = 0
        
        for item in self.train_dataset.data:
            # 难度分布
            difficulty = item.get('difficulty', 1)
            difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
            
            # 来源文件分布
            source_file = item.get('source_file', 'unknown')
            source_file_dist[source_file] = source_file_dist.get(source_file, 0) + 1
            
            # 文本长度统计
            total_instruction_length += len(item.get('instruction', ''))
            total_output_length += len(item.get('output', ''))
            
            # thinking数据统计
            if '<thinking>' in item.get('output', ''):
                thinking_count += 1
            
            # 密码学术语统计
            crypto_terms = item.get('crypto_terms', {})
            total_terms = crypto_terms.get('total_terms', 0)
            if total_terms > 0:
                crypto_terms_dist[total_terms] = crypto_terms_dist.get(total_terms, 0) + 1
        
        print(f"  📈 总样本数: {total_samples}")
        print(f"  📝 平均问题长度: {total_instruction_length / total_samples:.1f} 字符")
        print(f"  📝 平均答案长度: {total_output_length / total_samples:.1f} 字符")
        print(f"  🧠 包含thinking的样本: {thinking_count} ({thinking_count/total_samples*100:.1f}%)")
        
        print(f"  📊 难度分布:")
        for difficulty in sorted(difficulty_dist.keys()):
            count = difficulty_dist[difficulty]
            percentage = count / total_samples * 100
            print(f"    难度{difficulty}: {count} 样本 ({percentage:.1f}%)")
        
        print(f"  📁 来源文件分布:")
        for source_file in sorted(source_file_dist.keys()):
            count = source_file_dist[source_file]
            percentage = count / total_samples * 100
            print(f"    {source_file}: {count} 样本 ({percentage:.1f}%)")
        
        if crypto_terms_dist:
            print(f"  🔐 密码学术语分布:")
            for term_count in sorted(crypto_terms_dist.keys()):
                count = crypto_terms_dist[term_count]
                print(f"    {term_count}个术语: {count} 样本")
    
    def save_training_statistics(self, final_step: int):
        """保存训练统计信息"""
        stats = {
            'training_config': {
                'model_name': self.config.model_name,
                'data_path': self.config.data_path,
                'max_seq_length': self.config.max_seq_length,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha
            },
            'dataset_stats': {
                'total_samples': len(self.train_dataset.data),
                'total_batches': len(self.train_dataloader),
                'final_training_steps': final_step
            },
            'model_stats': {
                'total_parameters': self.model.num_parameters(),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_completed_at': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(self.config.output_dir, 'training_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 训练统计已保存: {stats_file}")
    
    def validate_data_quality(self):
        """验证数据质量"""
        print("\n🔍 数据质量验证:")
        
        issues = []
        valid_samples = 0
        
        for i, item in enumerate(self.train_dataset.data):
            try:
                # 检查必要字段
                if not item.get('instruction'):
                    issues.append(f"样本 {i}: 缺少instruction")
                    continue
                
                if not item.get('output'):
                    issues.append(f"样本 {i}: 缺少output")
                    continue
                
                # 检查thinking格式
                output = item.get('output', '')
                if '<thinking>' in output:
                    if '</thinking>' not in output:
                        issues.append(f"样本 {i}: thinking标签不完整")
                        continue
                    
                    # 检查thinking内容是否为空
                    thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
                    if thinking_match and not thinking_match.group(1).strip():
                        issues.append(f"样本 {i}: thinking内容为空")
                
                # 检查文本长度
                if len(item.get('instruction', '')) < 5:
                    issues.append(f"样本 {i}: instruction过短")
                    continue
                
                if len(output) < 10:
                    issues.append(f"样本 {i}: output过短")
                    continue
                
                # 检查tokenization
                test_text = f"{item.get('instruction', '')} {output}"
                try:
                    tokens = self.tokenizer(test_text, max_length=self.config.max_seq_length, truncation=True)
                    if len(tokens['input_ids']) < 10:
                        issues.append(f"样本 {i}: tokenization结果过短")
                        continue
                except Exception as e:
                    issues.append(f"样本 {i}: tokenization失败 - {e}")
                    continue
                
                valid_samples += 1
                
            except Exception as e:
                issues.append(f"样本 {i}: 验证过程出错 - {e}")
        
        print(f"  ✅ 有效样本: {valid_samples}/{len(self.train_dataset.data)}")
        
        if issues:
            print(f"  ⚠️ 发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                print(f"    {issue}")
            if len(issues) > 10:
                print(f"    ... 还有 {len(issues) - 10} 个问题")
            
            # 保存问题报告
            issues_file = os.path.join(self.config.output_dir, 'data_quality_issues.txt')
            with open(issues_file, 'w', encoding='utf-8') as f:
                for issue in issues:
                    f.write(f"{issue}\n")
            print(f"  📝 问题报告已保存: {issues_file}")
        else:
            print("  ✅ 所有数据质量检查通过")
    
    def run(self):
        """运行完整训练流程"""
        try:
            print("🎯 开始直接微调流程")
            print("=" * 60)
            
            # 1. 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 2. 创建数据集
            self.create_dataset()
            
            # 3. 创建数据加载器
            self.create_dataloader()
            
            # 4. 数据质量验证
            self.validate_data_quality()
            
            # 5. 设置优化器和调度器
            self.setup_optimizer_and_scheduler()
            
            # 6. 开始训练
            self.train()
            
            print("🎉 微调流程完成！")
            return True
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def check_uv_environment():
    """检查uv环境"""
    try:
        import subprocess
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv未安装或不可用")
            return False
    except FileNotFoundError:
        print("❌ uv未找到，请先安装uv")
        print("   安装命令: pip install uv 或访问 https://docs.astral.sh/uv/getting-started/installation/")
        return False


def install_dependencies_with_uv():
    """使用uv安装依赖"""
    print("🔄 使用uv安装依赖...")
    
    required_packages = [
        "torch",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "datasets",
        "accelerate",
        "jieba",
        "opencc-python-reimplemented",
        "numpy",
        "tqdm"
    ]
    
    try:
        import subprocess
        for package in required_packages:
            print(f"  📦 安装 {package}...")
            result = subprocess.run(['uv', 'pip', 'install', package], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ 安装 {package} 失败: {result.stderr}")
                return False
            else:
                print(f"  ✅ {package} 安装成功")
        
        print("✅ 所有依赖安装完成")
        return True
        
    except Exception as e:
        print(f"❌ 依赖安装过程出错: {e}")
        return False


def main():
    """主函数"""
    print("🎯 直接使用已实现模块进行微调 (uv版本)")
    print("=" * 60)
    
    # 检查uv环境
    if not check_uv_environment():
        return False
    
    # 安装依赖
    if not install_dependencies_with_uv():
        print("❌ 依赖安装失败，无法继续")
        return False
    
    # 检查环境
    print("🔍 检查训练环境...")
    if not torch.cuda.is_available():
        print("⚠️ 未检测到CUDA，将使用CPU训练（速度较慢）")
    else:
        print(f"✅ 检测到CUDA，GPU数量: {torch.cuda.device_count()}")
    
    # 创建配置
    config = DirectTrainingConfig()
    
    # 检查数据文件
    if not os.path.exists(config.data_path):
        print(f"❌ 数据文件不存在: {config.data_path}")
        return False
    
    print(f"✅ 数据文件存在: {config.data_path}")
    
    # 创建训练器并运行
    trainer = DirectTrainer(config)
    success = trainer.run()
    
    if success:
        print("🎉 微调成功完成！")
        print(f"📁 输出目录: {config.output_dir}")
        print("\n📋 使用uv运行命令:")
        print(f"   uv run python {__file__}")
    else:
        print("❌ 微调失败")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)