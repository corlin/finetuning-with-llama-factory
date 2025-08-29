#!/usr/bin/env python3
"""
简化版CLI工具和用户接口

本模块实现了命令行训练工具、配置文件模板和验证等基本功能。
不依赖rich库，使用标准库实现基本的用户交互界面。
"""

import os
import sys
import json
import yaml
import click
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.data_models import TrainingExample, ThinkingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig, SystemConfig
from lora_config_optimizer import LoRAMemoryProfile, LoRAOptimizationStrategy
from parallel_config import ParallelConfig, ParallelStrategy
from training_pipeline import TrainingPipelineOrchestrator, PipelineStage, PipelineStatus
# 使用原生PyTorch训练引擎
from gpu_utils import GPUDetector


class ConfigTemplate:
    """配置模板生成器"""
    
    @staticmethod
    def generate_training_config_template() -> Dict[str, Any]:
        """生成训练配置模板"""
        return {
            "model": {
                "model_name": "Qwen/Qwen3-4B-Thinking-2507",
                "model_revision": "main",
                "trust_remote_code": True
            },
            "training": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
                "fp16": False,
                "bf16": True,
                "save_strategy": "steps",
                "save_steps": 500,
                "eval_strategy": "steps",
                "eval_steps": 500,
                "logging_steps": 10,
                "max_seq_length": 2048,
                "seed": 42
            },
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "optimization_strategy": "auto"
            },
            "data": {
                "train_split_ratio": 0.9,
                "eval_split_ratio": 0.1,
                "max_samples": None,
                "shuffle_data": True,
                "data_format": "alpaca"
            },
            "parallel": {
                "strategy": "data_parallel",
                "world_size": 1,
                "enable_distributed": False
            },
            "system": {
                "output_dir": "./output",
                "logging_dir": "./logs",
                "cache_dir": "./cache",
                "log_level": "INFO"
            }
        }
    
    @staticmethod
    def save_template(template: Dict[str, Any], output_file: str):
        """保存配置模板"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"✓ 配置模板已保存到: {output_path}")


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_config_file(config_file: str) -> Tuple[bool, List[str]]:
        """验证配置文件"""
        errors = []
        
        try:
            if not Path(config_file).exists():
                errors.append(f"配置文件不存在: {config_file}")
                return False, errors
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证必需的配置节
            required_sections = ["model", "training", "lora", "data", "system"]
            for section in required_sections:
                if section not in config:
                    errors.append(f"缺少配置节: {section}")
            
            # 验证模型配置
            if "model" in config:
                model_config = config["model"]
                if "model_name" not in model_config:
                    errors.append("model.model_name 是必需的")
            
            # 验证训练配置
            if "training" in config:
                training_config = config["training"]
                
                # 检查数值范围
                if "learning_rate" in training_config:
                    lr = training_config["learning_rate"]
                    if not isinstance(lr, (int, float)) or lr <= 0:
                        errors.append("training.learning_rate 必须是正数")
                
                if "num_train_epochs" in training_config:
                    epochs = training_config["num_train_epochs"]
                    if not isinstance(epochs, int) or epochs <= 0:
                        errors.append("training.num_train_epochs 必须是正整数")
            
            # 验证LoRA配置
            if "lora" in config:
                lora_config = config["lora"]
                
                if "rank" in lora_config:
                    rank = lora_config["rank"]
                    if not isinstance(rank, int) or rank <= 0:
                        errors.append("lora.rank 必须是正整数")
                
                if "alpha" in lora_config:
                    alpha = lora_config["alpha"]
                    if not isinstance(alpha, int) or alpha <= 0:
                        errors.append("lora.alpha 必须是正整数")
            
            # 验证数据配置
            if "data" in config:
                data_config = config["data"]
                
                if "train_split_ratio" in data_config:
                    ratio = data_config["train_split_ratio"]
                    if not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
                        errors.append("data.train_split_ratio 必须在0和1之间")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"配置文件解析失败: {e}")
            return False, errors


class SimpleProgressDisplay:
    """简单的进度显示器"""
    
    def __init__(self):
        self.is_running = False
        self.current_state = None
        
    def start_display(self, orchestrator: TrainingPipelineOrchestrator):
        """启动进度显示"""
        self.is_running = True
        
        print("=" * 60)
        print("训练流水线监控")
        print("=" * 60)
        
        while self.is_running:
            state = orchestrator.get_state()
            self.current_state = state
            
            # 清屏并显示状态
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"流水线ID: {state.pipeline_id}")
            print(f"状态: {state.status.value.upper()}")
            print(f"当前阶段: {state.current_stage.value}")
            print(f"总体进度: {state.progress:.1f}%")
            
            if state.runtime:
                runtime = state.runtime.total_seconds()
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)
                seconds = int(runtime % 60)
                print(f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # 显示进度条
            progress_bar = "█" * int(state.progress / 5) + "░" * (20 - int(state.progress / 5))
            print(f"进度: [{progress_bar}] {state.progress:.1f}%")
            
            # 显示阶段状态
            print("\n阶段状态:")
            for stage in PipelineStage:
                if stage == PipelineStage.ERROR:
                    continue
                    
                stage_progress = state.stage_progress.get(stage, 0.0)
                if stage == state.current_stage:
                    status_icon = "🔄"
                elif stage_progress >= 100.0:
                    status_icon = "✅"
                elif stage_progress > 0.0:
                    status_icon = "⏳"
                else:
                    status_icon = "⭕"
                
                print(f"  {status_icon} {stage.value}: {stage_progress:.0f}%")
            
            if state.error_message:
                print(f"\n错误信息: {state.error_message}")
            
            print("\n按 Ctrl+C 停止监控")
            
            time.sleep(2)
    
    def stop_display(self):
        """停止进度显示"""
        self.is_running = False


# CLI命令组
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='启用详细输出')
@click.option('--config', '-c', help='配置文件路径')
@click.pass_context
def cli(ctx, verbose, config):
    """Qwen3-4B-Thinking 密码学微调工具"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    # 设置日志级别
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--output', '-o', default='config.yaml', help='输出配置文件路径')
def init_config(output):
    """生成配置文件模板"""
    print("🔧 生成配置文件模板...")
    
    template = ConfigTemplate.generate_training_config_template()
    ConfigTemplate.save_template(template, output)
    
    print(f"\n📝 配置文件模板已生成: {output}")
    print("请编辑配置文件后使用 'train' 命令开始训练")


@cli.command()
@click.argument('config_file')
def validate_config(config_file):
    """验证配置文件"""
    print(f"🔍 验证配置文件: {config_file}")
    
    is_valid, errors = ConfigValidator.validate_config_file(config_file)
    
    if is_valid:
        print("✅ 配置文件验证通过")
    else:
        print("❌ 配置文件验证失败:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)


@cli.command()
@click.argument('data_file')
@click.argument('config_file')
@click.option('--output-dir', '-o', default='./training_output', help='输出目录')
@click.option('--pipeline-id', help='流水线ID')
@click.option('--resume', help='从检查点恢复')
@click.option('--dry-run', is_flag=True, help='仅验证配置，不执行训练')
@click.pass_context
def train(ctx, data_file, config_file, output_dir, pipeline_id, resume, dry_run):
    """开始训练"""
    print("🚀 开始训练流程...")
    
    # 验证配置文件
    is_valid, errors = ConfigValidator.validate_config_file(config_file)
    if not is_valid:
        print("❌ 配置文件验证失败:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    
    # 验证数据文件
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        sys.exit(1)
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载训练数据
    print(f"📊 加载训练数据: {data_file}")
    training_data = load_training_data(data_file)
    print(f"✓ 加载了 {len(training_data)} 条训练数据")
    
    if dry_run:
        print("🔍 仅验证模式，不执行实际训练")
        print("✅ 配置和数据验证通过")
        return
    
    # 创建配置对象
    training_config = create_training_config(config)
    data_config = create_data_config(config)
    lora_config = create_lora_config(config)
    parallel_config = create_parallel_config(config)
    system_config = create_system_config(config)
    
    # 生成流水线ID
    if not pipeline_id:
        pipeline_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建流水线编排器
    orchestrator = TrainingPipelineOrchestrator(pipeline_id, output_dir)
    orchestrator.configure_pipeline(
        training_data, training_config, data_config, lora_config, parallel_config, system_config
    )
    
    # 启动进度显示
    progress_display = SimpleProgressDisplay()
    display_thread = threading.Thread(
        target=progress_display.start_display,
        args=(orchestrator,),
        daemon=True
    )
    display_thread.start()
    
    try:
        # 运行训练流水线
        success = orchestrator.run_pipeline(resume_from_checkpoint=resume)
        
        progress_display.stop_display()
        
        if success:
            print("\n🎉 训练完成！")
            print(f"📁 输出目录: {output_dir}")
        else:
            print("\n❌ 训练失败！")
            if orchestrator.get_state().error_message:
                print(f"错误信息: {orchestrator.get_state().error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        orchestrator.stop_pipeline()
        progress_display.stop_display()
        sys.exit(0)


@cli.command()
@click.argument('pipeline_dir')
def status(pipeline_dir):
    """查看训练状态"""
    print(f"📊 查看训练状态: {pipeline_dir}")
    
    # 查找最新的状态文件
    pipeline_path = Path(pipeline_dir)
    if not pipeline_path.exists():
        print(f"❌ 流水线目录不存在: {pipeline_dir}")
        sys.exit(1)
    
    # 查找检查点文件
    checkpoints_dir = pipeline_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.json"))
        if checkpoint_files:
            # 获取最新的检查点
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 显示状态信息
            print("\n训练状态:")
            print(f"  检查点ID: {checkpoint_data['checkpoint_id']}")
            print(f"  阶段: {checkpoint_data['stage']}")
            print(f"  时间: {checkpoint_data['timestamp']}")
            print(f"  进度: {checkpoint_data['state_data'].get('progress', 0):.1f}%")
        else:
            print("❌ 未找到检查点文件")
    else:
        print("❌ 检查点目录不存在")


@cli.command()
def list_gpus():
    """列出可用的GPU"""
    print("🔍 检测GPU设备...")
    
    detector = GPUDetector()
    gpu_infos = detector.get_all_gpu_info()
    
    if not gpu_infos:
        print("❌ 未检测到GPU设备")
        sys.exit(0)
    
    print("\nGPU设备信息:")
    print("-" * 80)
    print(f"{'ID':<4} {'名称':<30} {'内存':<20} {'利用率':<10} {'温度':<10}")
    print("-" * 80)
    
    for gpu in gpu_infos:
        memory_text = f"{gpu.used_memory}MB / {gpu.total_memory}MB"
        utilization_text = f"{gpu.utilization:.1f}%"
        temperature_text = f"{gpu.temperature:.1f}°C" if gpu.temperature and gpu.temperature > 0 else "N/A"
        
        print(f"{gpu.gpu_id:<4} {gpu.name:<30} {memory_text:<20} {utilization_text:<10} {temperature_text:<10}")


@cli.command()
@click.argument('data_file')
@click.option('--format', 'data_format', default='json', help='数据格式 (json, jsonl)')
@click.option('--sample', type=int, help='显示样本数量')
def inspect_data(data_file, data_format, sample):
    """检查训练数据"""
    print(f"🔍 检查训练数据: {data_file}")
    
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        sys.exit(1)
    
    try:
        training_data = load_training_data(data_file)
        
        # 显示数据统计
        print("\n数据统计:")
        print(f"  总样本数: {len(training_data)}")
        
        # 统计难度级别
        difficulty_counts = {}
        thinking_count = 0
        crypto_terms_count = 0
        
        for example in training_data:
            # 难度级别统计
            difficulty = example.difficulty_level.name
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            # thinking数据统计
            if example.has_thinking():
                thinking_count += 1
            
            # 密码学术语统计
            if example.crypto_terms:
                crypto_terms_count += len(example.crypto_terms)
        
        print(f"  包含thinking数据: {thinking_count}")
        print(f"  密码学术语总数: {crypto_terms_count}")
        
        for difficulty, count in difficulty_counts.items():
            print(f"  难度-{difficulty}: {count}")
        
        # 显示样本
        if sample and sample > 0:
            print(f"\n显示前 {min(sample, len(training_data))} 个样本:")
            print("=" * 60)
            
            for i, example in enumerate(training_data[:sample]):
                print(f"\n样本 {i+1}:")
                print(f"  指令: {example.instruction}")
                if example.input:
                    print(f"  输入: {example.input}")
                print(f"  输出: {example.output[:200]}...")
                if example.has_thinking():
                    print(f"  思考: {example.thinking[:100]}...")
                print("-" * 40)
        
    except Exception as e:
        print(f"❌ 数据检查失败: {e}")
        sys.exit(1)


# 辅助函数
def load_training_data(data_file: str) -> List[TrainingExample]:
    """加载训练数据"""
    data_path = Path(data_file)
    
    if data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.suffix == '.jsonl':
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"不支持的数据格式: {data_path.suffix}")
    
    # 转换为TrainingExample对象
    training_examples = []
    for item in data:
        if isinstance(item, dict):
            example = TrainingExample(
                instruction=item.get('instruction', ''),
                input=item.get('input', ''),
                output=item.get('output', ''),
                thinking=item.get('thinking'),
                crypto_terms=item.get('crypto_terms', []),
                difficulty_level=DifficultyLevel(item.get('difficulty', 2))
            )
            training_examples.append(example)
    
    return training_examples


def create_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """创建训练配置"""
    training_section = config.get('training', {})
    
    return TrainingConfig(
        num_train_epochs=training_section.get('num_train_epochs', 3),
        per_device_train_batch_size=training_section.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=training_section.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_section.get('gradient_accumulation_steps', 4),
        learning_rate=training_section.get('learning_rate', 2e-4),
        weight_decay=training_section.get('weight_decay', 0.01),
        warmup_ratio=training_section.get('warmup_ratio', 0.1),
        lr_scheduler_type=training_section.get('lr_scheduler_type', 'cosine'),
        fp16=training_section.get('fp16', False),
        bf16=training_section.get('bf16', True),
        save_strategy=training_section.get('save_strategy', 'steps'),
        save_steps=training_section.get('save_steps', 500),
        eval_strategy=training_section.get('eval_strategy', 'steps'),
        eval_steps=training_section.get('eval_steps', 500),
        logging_steps=training_section.get('logging_steps', 10),
        seed=training_section.get('seed', 42)
    )


def create_data_config(config: Dict[str, Any]) -> DataConfig:
    """创建数据配置"""
    data_section = config.get('data', {})
    
    return DataConfig(
        train_split_ratio=data_section.get('train_split_ratio', 0.9),
        eval_split_ratio=data_section.get('eval_split_ratio', 0.1),
        max_samples=data_section.get('max_samples'),
        shuffle_data=data_section.get('shuffle_data', True),
        data_format=data_section.get('data_format', 'alpaca')
    )


def create_lora_config(config: Dict[str, Any]) -> LoRAMemoryProfile:
    """创建LoRA配置"""
    lora_section = config.get('lora', {})
    
    return LoRAMemoryProfile(
        rank=lora_section.get('rank', 8),
        alpha=lora_section.get('alpha', 16),
        target_modules=lora_section.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    )


def create_parallel_config(config: Dict[str, Any]) -> ParallelConfig:
    """创建并行配置"""
    parallel_section = config.get('parallel', {})
    
    strategy_map = {
        'data_parallel': ParallelStrategy.DATA_PARALLEL,
        'model_parallel': ParallelStrategy.MODEL_PARALLEL,
        'pipeline_parallel': ParallelStrategy.PIPELINE_PARALLEL,
        'hybrid_parallel': ParallelStrategy.HYBRID_PARALLEL,
        'auto': ParallelStrategy.AUTO
    }
    
    strategy = strategy_map.get(parallel_section.get('strategy', 'data_parallel'), ParallelStrategy.DATA_PARALLEL)
    
    return ParallelConfig(
        strategy=strategy,
        data_parallel_size=parallel_section.get('world_size', 1),
        enable_zero_optimization=parallel_section.get('enable_distributed', False)
    )


def create_system_config(config: Dict[str, Any]) -> SystemConfig:
    """创建系统配置"""
    system_section = config.get('system', {})
    
    return SystemConfig(
        cache_dir=system_section.get('cache_dir', './cache'),
        log_level=system_section.get('log_level', 'INFO')
    )


if __name__ == '__main__':
    cli()