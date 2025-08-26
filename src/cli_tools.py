#!/usr/bin/env python3
"""
CLI工具和用户接口

本模块实现了命令行训练工具、配置文件模板和验证、训练进度可视化界面等功能。
提供完整的用户交互界面，支持训练流水线的启动、监控和管理。
"""

import os
import sys
import json
import yaml
import click
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

from src.data_models import TrainingExample, ThinkingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig, SystemConfig
from lora_config_optimizer import LoRAMemoryProfile, LoRAOptimizationStrategy
from parallel_config import ParallelConfig, ParallelStrategy
from training_pipeline import TrainingPipelineOrchestrator, PipelineStage, PipelineStatus
# LlamaFactory adapter removed - using direct training engine
from gpu_utils import GPUDetector


# 全局控制台对象
console = Console()


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
        
        console.print(f"✓ 配置模板已保存到: {output_path}", style="green")


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


class TrainingProgressDisplay:
    """训练进度显示器"""
    
    def __init__(self):
        self.console = Console()
        self.is_running = False
        self.current_state = None
        
    def start_display(self, orchestrator: TrainingPipelineOrchestrator):
        """启动进度显示"""
        self.is_running = True
        
        # 创建布局
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="status", ratio=1)
        )
        
        with Live(layout, console=self.console, refresh_per_second=2) as live:
            while self.is_running:
                state = orchestrator.get_state()
                self.current_state = state
                
                # 更新布局内容
                layout["header"].update(self._create_header(state))
                layout["progress"].update(self._create_progress_panel(state))
                layout["status"].update(self._create_status_panel(state))
                layout["footer"].update(self._create_footer(state))
                
                time.sleep(0.5)
    
    def stop_display(self):
        """停止进度显示"""
        self.is_running = False
    
    def _create_header(self, state) -> Panel:
        """创建头部面板"""
        title = f"训练流水线: {state.pipeline_id}"
        status_color = {
            PipelineStatus.PENDING: "yellow",
            PipelineStatus.RUNNING: "green",
            PipelineStatus.PAUSED: "orange",
            PipelineStatus.COMPLETED: "blue",
            PipelineStatus.FAILED: "red",
            PipelineStatus.CANCELLED: "gray"
        }.get(state.status, "white")
        
        header_text = Text(title, style="bold")
        header_text.append(f" [{state.status.value.upper()}]", style=f"bold {status_color}")
        
        return Panel(header_text, box=box.ROUNDED)
    
    def _create_progress_panel(self, state) -> Panel:
        """创建进度面板"""
        # 总体进度条
        progress_bar = "█" * int(state.progress / 5) + "░" * (20 - int(state.progress / 5))
        progress_text = f"总体进度: [{progress_bar}] {state.progress:.1f}%\n\n"
        
        # 当前阶段
        progress_text += f"当前阶段: {state.current_stage.value}\n"
        
        # 阶段进度
        if state.current_stage_runtime:
            runtime = state.current_stage_runtime.total_seconds()
            progress_text += f"阶段运行时间: {runtime:.0f}秒\n"
        
        # 各阶段状态
        progress_text += "\n阶段状态:\n"
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
            
            progress_text += f"  {status_icon} {stage.value}: {stage_progress:.0f}%\n"
        
        return Panel(progress_text, title="训练进度", box=box.ROUNDED)
    
    def _create_status_panel(self, state) -> Panel:
        """创建状态面板"""
        status_text = ""
        
        # 运行时间
        if state.runtime:
            runtime = state.runtime.total_seconds()
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            seconds = int(runtime % 60)
            status_text += f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}\n\n"
        
        # 检查点信息
        if state.latest_checkpoint:
            status_text += f"最新检查点:\n"
            status_text += f"  ID: {state.latest_checkpoint.checkpoint_id}\n"
            status_text += f"  阶段: {state.latest_checkpoint.stage.value}\n"
            status_text += f"  时间: {state.latest_checkpoint.timestamp.strftime('%H:%M:%S')}\n\n"
        
        # 错误信息
        if state.error_message:
            status_text += f"错误信息:\n{state.error_message}\n"
        
        return Panel(status_text, title="状态信息", box=box.ROUNDED)
    
    def _create_footer(self, state) -> Panel:
        """创建底部面板"""
        footer_text = "按 Ctrl+C 停止监控"
        if state.status == PipelineStatus.RUNNING:
            footer_text += " | 训练正在进行中..."
        elif state.status == PipelineStatus.COMPLETED:
            footer_text += " | 训练已完成！"
        elif state.status == PipelineStatus.FAILED:
            footer_text += " | 训练失败！"
        
        return Panel(footer_text, box=box.ROUNDED)


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
    console.print("🔧 生成配置文件模板...", style="blue")
    
    template = ConfigTemplate.generate_training_config_template()
    ConfigTemplate.save_template(template, output)
    
    console.print(f"\n📝 配置文件模板已生成: {output}", style="green")
    console.print("请编辑配置文件后使用 'train' 命令开始训练", style="yellow")


@cli.command()
@click.argument('config_file')
def validate_config(config_file):
    """验证配置文件"""
    console.print(f"🔍 验证配置文件: {config_file}", style="blue")
    
    is_valid, errors = ConfigValidator.validate_config_file(config_file)
    
    if is_valid:
        console.print("✅ 配置文件验证通过", style="green")
    else:
        console.print("❌ 配置文件验证失败:", style="red")
        for error in errors:
            console.print(f"  • {error}", style="red")
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
    console.print("🚀 开始训练流程...", style="blue")
    
    # 验证配置文件
    is_valid, errors = ConfigValidator.validate_config_file(config_file)
    if not is_valid:
        console.print("❌ 配置文件验证失败:", style="red")
        for error in errors:
            console.print(f"  • {error}", style="red")
        sys.exit(1)
    
    # 验证数据文件
    if not Path(data_file).exists():
        console.print(f"❌ 数据文件不存在: {data_file}", style="red")
        sys.exit(1)
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载训练数据
    console.print(f"📊 加载训练数据: {data_file}", style="blue")
    training_data = load_training_data(data_file)
    console.print(f"✓ 加载了 {len(training_data)} 条训练数据", style="green")
    
    if dry_run:
        console.print("🔍 仅验证模式，不执行实际训练", style="yellow")
        console.print("✅ 配置和数据验证通过", style="green")
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
    progress_display = TrainingProgressDisplay()
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
            console.print("\n🎉 训练完成！", style="green")
            console.print(f"📁 输出目录: {output_dir}", style="blue")
        else:
            console.print("\n❌ 训练失败！", style="red")
            if orchestrator.get_state().error_message:
                console.print(f"错误信息: {orchestrator.get_state().error_message}", style="red")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n⏹️ 训练被用户中断", style="yellow")
        orchestrator.stop_pipeline()
        progress_display.stop_display()
        sys.exit(0)


@cli.command()
@click.argument('pipeline_dir')
def status(pipeline_dir):
    """查看训练状态"""
    console.print(f"📊 查看训练状态: {pipeline_dir}", style="blue")
    
    # 查找最新的状态文件
    pipeline_path = Path(pipeline_dir)
    if not pipeline_path.exists():
        console.print(f"❌ 流水线目录不存在: {pipeline_dir}", style="red")
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
            
            # 显示状态表格
            table = Table(title="训练状态")
            table.add_column("项目", style="cyan")
            table.add_column("值", style="green")
            
            table.add_row("检查点ID", checkpoint_data["checkpoint_id"])
            table.add_row("阶段", checkpoint_data["stage"])
            table.add_row("时间", checkpoint_data["timestamp"])
            table.add_row("进度", f"{checkpoint_data['state_data'].get('progress', 0):.1f}%")
            
            console.print(table)
        else:
            console.print("❌ 未找到检查点文件", style="red")
    else:
        console.print("❌ 检查点目录不存在", style="red")


@cli.command()
def list_gpus():
    """列出可用的GPU"""
    console.print("🔍 检测GPU设备...", style="blue")
    
    detector = GPUDetector()
    gpu_infos = detector.get_all_gpu_info()
    
    if not gpu_infos:
        console.print("❌ 未检测到GPU设备", style="red")
        return
    
    # 创建GPU信息表格
    table = Table(title="GPU设备信息")
    table.add_column("ID", style="cyan")
    table.add_column("名称", style="green")
    table.add_column("内存", style="yellow")
    table.add_column("利用率", style="blue")
    table.add_column("温度", style="red")
    
    for gpu in gpu_infos:
        memory_text = f"{gpu.memory_used}MB / {gpu.memory_total}MB"
        utilization_text = f"{gpu.utilization:.1f}%"
        temperature_text = f"{gpu.temperature:.1f}°C" if gpu.temperature > 0 else "N/A"
        
        table.add_row(
            str(gpu.gpu_id),
            gpu.name,
            memory_text,
            utilization_text,
            temperature_text
        )
    
    console.print(table)


@cli.command()
@click.argument('data_file')
@click.option('--format', 'data_format', default='json', help='数据格式 (json, jsonl)')
@click.option('--sample', type=int, help='显示样本数量')
def inspect_data(data_file, data_format, sample):
    """检查训练数据"""
    console.print(f"🔍 检查训练数据: {data_file}", style="blue")
    
    if not Path(data_file).exists():
        console.print(f"❌ 数据文件不存在: {data_file}", style="red")
        sys.exit(1)
    
    try:
        training_data = load_training_data(data_file)
        
        # 显示数据统计
        table = Table(title="数据统计")
        table.add_column("项目", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("总样本数", str(len(training_data)))
        
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
        
        table.add_row("包含thinking数据", str(thinking_count))
        table.add_row("密码学术语总数", str(crypto_terms_count))
        
        for difficulty, count in difficulty_counts.items():
            table.add_row(f"难度-{difficulty}", str(count))
        
        console.print(table)
        
        # 显示样本
        if sample and sample > 0:
            console.print(f"\n📝 显示前 {min(sample, len(training_data))} 个样本:", style="blue")
            
            for i, example in enumerate(training_data[:sample]):
                panel_content = f"指令: {example.instruction}\n"
                if example.input:
                    panel_content += f"输入: {example.input}\n"
                panel_content += f"输出: {example.output[:200]}..."
                if example.has_thinking():
                    panel_content += f"\n思考: {example.thinking[:100]}..."
                
                console.print(Panel(panel_content, title=f"样本 {i+1}"))
        
    except Exception as e:
        console.print(f"❌ 数据检查失败: {e}", style="red")
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