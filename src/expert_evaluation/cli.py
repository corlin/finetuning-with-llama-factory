"""
专家评估系统命令行工具

提供完整的CLI接口，支持配置文件处理、进度显示、交互功能等。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from .engine import ExpertEvaluationEngine
from .config import ExpertEvaluationConfig, EvaluationDimension, ExpertiseLevel
from .data_models import QAEvaluationItem, ExpertEvaluationResult
from .exceptions import (
    ModelLoadError,
    EvaluationProcessError,
    DataFormatError,
    ConfigurationError
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局控制台对象
console = Console()

class ExpertEvaluationCLI:
    """专家评估CLI主类"""
    
    def __init__(self):
        self.engine: Optional[ExpertEvaluationEngine] = None
        self.config: Optional[ExpertEvaluationConfig] = None
        self.console = Console()
    
    def load_config(self, config_path: Optional[str] = None) -> ExpertEvaluationConfig:
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 创建配置对象
                config = ExpertEvaluationConfig()
                
                # 更新配置
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                self.console.print(f"[green]✓[/green] 配置文件加载成功: {config_path}")
                return config
                
            except Exception as e:
                self.console.print(f"[red]✗[/red] 配置文件加载失败: {e}")
                raise ConfigurationError(f"配置文件加载失败: {e}")
        else:
            # 使用默认配置
            self.console.print("[yellow]![/yellow] 使用默认配置")
            return ExpertEvaluationConfig()
    
    def initialize_engine(self, config: ExpertEvaluationConfig) -> ExpertEvaluationEngine:
        """初始化评估引擎"""
        try:
            with self.console.status("[bold green]初始化专家评估引擎..."):
                engine = ExpertEvaluationEngine(config)
            
            self.console.print("[green]✓[/green] 专家评估引擎初始化成功")
            return engine
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] 引擎初始化失败: {e}")
            raise
    
    def load_qa_data(self, data_path: str) -> List[QAEvaluationItem]:
        """加载QA数据"""
        try:
            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.suffix.lower() == '.json':
                    raw_data = json.load(f)
                else:
                    raise DataFormatError(f"不支持的文件格式: {data_path.suffix}")
            
            # 转换为QA评估项
            qa_items = []
            for item in raw_data:
                # 处理难度级别
                difficulty_level = ExpertiseLevel.INTERMEDIATE
                if 'difficulty_level' in item:
                    level_map = {
                        'beginner': ExpertiseLevel.BEGINNER,
                        'intermediate': ExpertiseLevel.INTERMEDIATE,
                        'advanced': ExpertiseLevel.ADVANCED,
                        'expert': ExpertiseLevel.EXPERT
                    }
                    difficulty_level = level_map.get(item['difficulty_level'].lower(), ExpertiseLevel.INTERMEDIATE)
                
                qa_item = QAEvaluationItem(
                    question_id=item.get('question_id', ''),
                    question=item.get('question', ''),
                    context=item.get('context'),
                    reference_answer=item.get('reference_answer', ''),
                    model_answer=item.get('model_answer', ''),
                    domain_tags=item.get('domain_tags', []),
                    difficulty_level=difficulty_level,
                    expected_concepts=item.get('expected_concepts', [])
                )
                qa_items.append(qa_item)
            
            self.console.print(f"[green]✓[/green] 成功加载 {len(qa_items)} 个QA项")
            return qa_items
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] 数据加载失败: {e}")
            raise
    
    def run_evaluation(self, qa_items: List[QAEvaluationItem], show_progress: bool = True) -> ExpertEvaluationResult:
        """运行评估"""
        if not self.engine:
            raise RuntimeError("评估引擎未初始化")
        
        try:
            if show_progress:
                with Progress() as progress:
                    task = progress.add_task("[green]评估进行中...", total=100)
                    
                    # 模拟进度更新
                    for i in range(0, 101, 10):
                        progress.update(task, completed=i)
                        time.sleep(0.1)
                    
                    result = self.engine.evaluate_model(qa_items)
            else:
                result = self.engine.evaluate_model(qa_items)
            
            self.console.print("[green]✓[/green] 评估完成")
            return result
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] 评估失败: {e}")
            raise
    
    def display_results(self, result: ExpertEvaluationResult, detailed: bool = False):
        """显示评估结果"""
        # 总分面板
        score_panel = Panel(
            f"[bold green]{result.overall_score:.2f}[/bold green]",
            title="总体评分",
            border_style="green"
        )
        self.console.print(score_panel)
        
        # 维度得分表格
        table = Table(title="维度评分详情")
        table.add_column("评估维度", style="cyan", no_wrap=True)
        table.add_column("得分", style="magenta")
        table.add_column("置信区间", style="yellow")
        
        for dimension, score in result.dimension_scores.items():
            confidence = result.confidence_intervals.get(dimension.value, (0, 0))
            table.add_row(
                dimension.value,
                f"{score:.2f}",
                f"[{confidence[0]:.2f}, {confidence[1]:.2f}]"
            )
        
        self.console.print(table)
        
        # 行业指标
        if result.industry_metrics:
            industry_table = Table(title="行业指标")
            industry_table.add_column("指标", style="cyan")
            industry_table.add_column("数值", style="magenta")
            
            for metric, value in result.industry_metrics.items():
                industry_table.add_row(metric, f"{value:.2f}")
            
            self.console.print(industry_table)
        
        # 详细反馈
        if detailed and result.detailed_feedback:
            feedback_text = Text()
            for key, value in result.detailed_feedback.items():
                feedback_text.append(f"{key}: {value}\n", style="white")
            
            feedback_panel = Panel(
                feedback_text,
                title="详细反馈",
                border_style="blue"
            )
            self.console.print(feedback_panel)
        
        # 改进建议
        if result.improvement_suggestions:
            suggestions_text = Text()
            for i, suggestion in enumerate(result.improvement_suggestions, 1):
                suggestions_text.append(f"{i}. {suggestion}\n", style="yellow")
            
            suggestions_panel = Panel(
                suggestions_text,
                title="改进建议",
                border_style="yellow"
            )
            self.console.print(suggestions_panel)
    
    def save_results(self, result: ExpertEvaluationResult, output_path: str, format: str = "json"):
        """保存评估结果"""
        try:
            output_path = Path(output_path)
            
            if format.lower() == "json":
                # 转换结果为可序列化格式
                result_dict = {
                    "overall_score": result.overall_score,
                    "dimension_scores": {dim.value: score for dim, score in result.dimension_scores.items()},
                    "industry_metrics": result.industry_metrics,
                    "detailed_feedback": result.detailed_feedback,
                    "improvement_suggestions": result.improvement_suggestions,
                    "confidence_intervals": {k: list(v) for k, v in result.confidence_intervals.items()},
                    "statistical_significance": result.statistical_significance,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            self.console.print(f"[green]✓[/green] 结果已保存到: {output_path}")
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] 结果保存失败: {e}")
            raise

# Click CLI 接口
@click.group()
@click.option('--config', '-c', help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.pass_context
def cli(ctx, config, verbose):
    """专家评估系统命令行工具"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='输出文件路径')
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'html', 'csv']), help='输出格式')
@click.option('--detailed', '-d', is_flag=True, help='显示详细结果')
@click.option('--no-progress', is_flag=True, help='不显示进度条')
@click.pass_context
def evaluate(ctx, data_path, output, format, detailed, no_progress):
    """运行模型评估"""
    try:
        cli_tool = ExpertEvaluationCLI()
        
        # 加载配置
        config = cli_tool.load_config(ctx.obj.get('config_path'))
        cli_tool.config = config
        
        # 初始化引擎
        cli_tool.engine = cli_tool.initialize_engine(config)
        
        # 加载数据
        qa_items = cli_tool.load_qa_data(data_path)
        
        # 运行评估
        result = cli_tool.run_evaluation(qa_items, show_progress=not no_progress)
        
        # 显示结果
        cli_tool.display_results(result, detailed=detailed)
        
        # 保存结果
        if output:
            cli_tool.save_results(result, output, format)
        
    except Exception as e:
        console.print(f"[red]评估失败: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.pass_context
def load_model(ctx, model_path):
    """加载评估模型"""
    try:
        cli_tool = ExpertEvaluationCLI()
        
        # 加载配置
        config = cli_tool.load_config(ctx.obj.get('config_path'))
        
        # 设置模型路径
        config.model_path = model_path
        
        # 初始化引擎
        engine = cli_tool.initialize_engine(config)
        
        console.print(f"[green]✓[/green] 模型加载成功: {model_path}")
        
    except Exception as e:
        console.print(f"[red]模型加载失败: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--output', '-o', default='config.json', help='配置文件输出路径')
def init_config(output):
    """初始化配置文件"""
    try:
        config = ExpertEvaluationConfig()
        
        # 转换为字典
        config_dict = {
            "model_path": "",
            "evaluation_dimensions": [dim.value for dim in EvaluationDimension],
            "industry_weights": {
                "semantic_similarity": 0.25,
                "domain_accuracy": 0.25,
                "response_relevance": 0.20,
                "factual_correctness": 0.15,
                "completeness": 0.15
            },
            "threshold_settings": {
                "min_score": 0.6,
                "confidence_level": 0.95
            },
            "enable_detailed_analysis": True,
            "output_format": "json"
        }
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✓[/green] 配置文件已创建: {output}")
        
    except Exception as e:
        console.print(f"[red]配置文件创建失败: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
def validate_data(data_path):
    """验证QA数据格式"""
    try:
        cli_tool = ExpertEvaluationCLI()
        qa_items = cli_tool.load_qa_data(data_path)
        
        # 数据统计
        stats_table = Table(title="数据统计")
        stats_table.add_column("项目", style="cyan")
        stats_table.add_column("数量/值", style="magenta")
        
        stats_table.add_row("总QA项数", str(len(qa_items)))
        
        # 难度级别分布
        difficulty_counts = {}
        domain_counts = {}
        
        for item in qa_items:
            level = item.difficulty_level.value
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
            
            for tag in item.domain_tags:
                domain_counts[tag] = domain_counts.get(tag, 0) + 1
        
        stats_table.add_row("难度级别分布", str(difficulty_counts))
        stats_table.add_row("领域标签分布", str(domain_counts))
        
        console.print(stats_table)
        console.print("[green]✓[/green] 数据格式验证通过")
        
    except Exception as e:
        console.print(f"[red]数据验证失败: {e}[/red]")
        sys.exit(1)

@cli.command()
def version():
    """显示版本信息"""
    console.print("[bold blue]专家评估系统 CLI v1.0.0[/bold blue]")
    console.print("基于FastAPI的专家级行业化模型评估工具")

# 传统argparse接口（兼容性）
def create_parser():
    """创建argparse解析器"""
    parser = argparse.ArgumentParser(
        description='专家评估系统命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行评估
  python -m src.expert_evaluation.cli evaluate data.json -o results.json
  
  # 加载模型
  python -m src.expert_evaluation.cli load-model /path/to/model
  
  # 验证数据
  python -m src.expert_evaluation.cli validate-data data.json
  
  # 初始化配置
  python -m src.expert_evaluation.cli init-config -o my_config.json
        """
    )
    
    parser.add_argument('--version', action='version', version='专家评估系统 CLI v1.0.0')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # evaluate 子命令
    eval_parser = subparsers.add_parser('evaluate', help='运行模型评估')
    eval_parser.add_argument('data_path', help='QA数据文件路径')
    eval_parser.add_argument('--output', '-o', help='输出文件路径')
    eval_parser.add_argument('--format', '-f', choices=['json', 'html', 'csv'], default='json', help='输出格式')
    eval_parser.add_argument('--detailed', '-d', action='store_true', help='显示详细结果')
    eval_parser.add_argument('--no-progress', action='store_true', help='不显示进度条')
    
    # load-model 子命令
    model_parser = subparsers.add_parser('load-model', help='加载评估模型')
    model_parser.add_argument('model_path', help='模型文件路径')
    
    # init-config 子命令
    config_parser = subparsers.add_parser('init-config', help='初始化配置文件')
    config_parser.add_argument('--output', '-o', default='config.json', help='配置文件输出路径')
    
    # validate-data 子命令
    validate_parser = subparsers.add_parser('validate-data', help='验证QA数据格式')
    validate_parser.add_argument('data_path', help='QA数据文件路径')
    
    return parser

def main():
    """主入口函数"""
    try:
        # 优先使用Click CLI
        cli()
    except SystemExit:
        # 如果Click失败，回退到argparse
        parser = create_parser()
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        if args.command == 'evaluate':
            # 实现argparse版本的evaluate
            pass
        elif args.command == 'load-model':
            # 实现argparse版本的load-model
            pass
        # ... 其他命令
        else:
            parser.print_help()

if __name__ == "__main__":
    main()