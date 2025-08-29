#!/usr/bin/env python3
"""
综合微调演示程序

基于当前已完成的功能，使用 data/raw 数据进行模型微调的完整演示。

功能特性：
- 自动加载和处理 data/raw 中的所有数据文件
- 智能数据预处理和格式转换
- 自动配置 GPU 并行策略和 LoRA 参数
- 集成原生PyTorch进行模型微调
- 实时训练监控和进度跟踪
- 完整的训练流水线管理

使用方法：
    uv run python demo_comprehensive_finetuning.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入核心模块
from data_models import TrainingExample, ThinkingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig, SystemConfig
from lora_config_optimizer import LoRAConfigOptimizer, LoRAMemoryProfile
from parallel_config import ParallelConfig, ParallelStrategy
from gpu_utils import GPUDetector
# LlamaFactory adapter removed - using direct training engine
from training_pipeline import TrainingPipelineOrchestrator, PipelineState
from thinking_generator import ThinkingDataGenerator


class ComprehensiveFinetuningDemo:
    """综合微调演示类"""
    
    def __init__(self, output_dir: str = "demo_output"):
        """
        初始化演示程序
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.gpu_detector = GPUDetector()
        self.lora_optimizer = LoRAConfigOptimizer()
        # LlamaFactory adapter removed - using direct training engine
        self.thinking_processor = ThinkingDataGenerator()
        
        # 数据存储
        self.training_data: List[TrainingExample] = []
        self.thinking_data: List[ThinkingExample] = []
        
        self.logger.info("综合微调演示程序初始化完成")
    
    def setup_logging(self):
        """设置日志配置"""
        log_file = self.output_dir / f"demo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_raw_data(self, data_dir: str = "data/raw") -> bool:
        """
        加载原始数据文件
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            bool: 是否成功加载数据
        """
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                self.logger.error(f"数据目录不存在: {data_path}")
                return False
            
            # 查找所有markdown文件
            md_files = list(data_path.glob("*.md"))
            if not md_files:
                self.logger.error(f"在 {data_path} 中未找到markdown文件")
                return False
            
            self.logger.info(f"找到 {len(md_files)} 个数据文件")
            
            total_examples = 0
            
            for md_file in md_files:
                self.logger.info(f"处理文件: {md_file.name}")
                
                # 读取文件内容
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析QA对
                examples = self.parse_qa_content(content, str(md_file))
                
                if examples:
                    self.training_data.extend(examples)
                    total_examples += len(examples)
                    self.logger.info(f"从 {md_file.name} 解析出 {len(examples)} 个训练样例")
                else:
                    self.logger.warning(f"从 {md_file.name} 未解析出任何训练样例")
            
            self.logger.info(f"总共加载了 {total_examples} 个训练样例")
            
            # 处理thinking数据
            self.process_thinking_data()
            
            return total_examples > 0
            
        except Exception as e:
            self.logger.error(f"加载原始数据失败: {e}")
            return False
    
    def parse_qa_content(self, content: str, source_file: str) -> List[TrainingExample]:
        """
        解析QA内容为训练样例
        
        Args:
            content: 文件内容
            source_file: 源文件路径
            
        Returns:
            List[TrainingExample]: 训练样例列表
        """
        examples = []
        
        try:
            # 直接使用简单解析方法
            examples = self.simple_parse_qa(content, source_file)
                
        except Exception as e:
            self.logger.error(f"解析QA内容失败: {e}")
            examples = []
        
        return examples
    
    def simple_parse_qa(self, content: str, source_file: str) -> List[TrainingExample]:
        """
        简单的QA解析方法
        
        Args:
            content: 文件内容
            source_file: 源文件路径
            
        Returns:
            List[TrainingExample]: 训练样例列表
        """
        examples = []
        
        # 分割内容为段落
        sections = content.split('\n\n')
        
        current_question = ""
        current_thinking = ""
        current_answer = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # 检查是否是问题
            if section.startswith('### Q') or section.startswith('##') and '?' in section:
                # 保存之前的QA对
                if current_question and current_answer:
                    example = self.create_training_example(
                        current_question, current_answer, current_thinking, source_file
                    )
                    if example:
                        examples.append(example)
                
                # 开始新的QA对
                current_question = section
                current_thinking = ""
                current_answer = ""
            
            # 检查是否是thinking内容
            elif '<thinking>' in section:
                current_thinking = section
            
            # 检查是否是答案
            elif section.startswith('A') and ':' in section:
                current_answer = section
        
        # 处理最后一个QA对
        if current_question and current_answer:
            example = self.create_training_example(
                current_question, current_answer, current_thinking, source_file
            )
            if example:
                examples.append(example)
        
        return examples
    
    def create_training_example(self, question: str, answer: str, thinking: str, source_file: str) -> Optional[TrainingExample]:
        """
        创建训练样例
        
        Args:
            question: 问题
            answer: 答案
            thinking: 思考过程
            source_file: 源文件
            
        Returns:
            Optional[TrainingExample]: 训练样例
        """
        try:
            # 清理问题文本
            instruction = question.replace('###', '').replace('##', '').strip()
            if ':' in instruction:
                instruction = instruction.split(':', 1)[1].strip()
            
            # 清理答案文本
            output = answer
            if ':' in output and output.startswith('A'):
                output = output.split(':', 1)[1].strip()
            
            # 处理thinking内容
            thinking_content = None
            if thinking and '<thinking>' in thinking:
                thinking_content = thinking
            
            if not instruction or not output:
                return None
            
            # 提取密码学术语
            crypto_terms = self.extract_crypto_terms(instruction + " " + output)
            
            # 判断难度级别
            difficulty = self.determine_difficulty(instruction, output)
            
            return TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                thinking=thinking_content,
                crypto_terms=crypto_terms,
                difficulty_level=difficulty,
                source_file=source_file
            )
            
        except Exception as e:
            self.logger.error(f"创建训练样例失败: {e}")
            return None
    
    def extract_crypto_terms(self, text: str) -> List[str]:
        """
        提取密码学术语
        
        Args:
            text: 文本内容
            
        Returns:
            List[str]: 密码学术语列表
        """
        # 常见密码学术语列表
        crypto_keywords = [
            "密码学", "加密", "解密", "哈希", "数字签名", "公钥", "私钥", "对称加密", "非对称加密",
            "AES", "RSA", "SHA", "MD5", "DES", "3DES", "ECC", "DSA", "ECDSA",
            "密钥管理", "证书", "PKI", "CA", "数字证书", "身份认证", "访问控制",
            "完整性", "机密性", "真实性", "不可否认性", "随机数", "密钥交换",
            "区块链", "比特币", "以太坊", "智能合约", "共识算法", "工作量证明"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in crypto_keywords:
            if term.lower() in text_lower or term in text:
                found_terms.append(term)
        
        return list(set(found_terms))  # 去重
    
    def determine_difficulty(self, instruction: str, output: str) -> DifficultyLevel:
        """
        判断难度级别
        
        Args:
            instruction: 指令
            output: 输出
            
        Returns:
            DifficultyLevel: 难度级别
        """
        text = instruction + " " + output
        text_len = len(text)
        
        # 基于文本长度和复杂度判断
        if text_len < 200:
            return DifficultyLevel.BEGINNER
        elif text_len < 500:
            return DifficultyLevel.INTERMEDIATE
        elif text_len < 1000:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT
    
    def process_thinking_data(self):
        """处理thinking数据"""
        thinking_examples = []
        
        for example in self.training_data:
            if example.has_thinking():
                thinking_example = example.to_thinking_example()
                if thinking_example:
                    thinking_examples.append(thinking_example)
        
        self.thinking_data = thinking_examples
        self.logger.info(f"处理了 {len(thinking_examples)} 个thinking样例")
    
    def auto_configure_training(self) -> tuple:
        """
        自动配置训练参数
        
        Returns:
            tuple: (training_config, data_config, lora_config, parallel_config, system_config)
        """
        self.logger.info("开始自动配置训练参数...")
        
        # 检测GPU环境
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        gpu_count = len(gpu_infos) if gpu_infos else 1
        
        self.logger.info(f"检测到 {gpu_count} 个GPU")
        
        # 配置训练参数
        training_config = TrainingConfig(
            output_dir=str(self.output_dir / "model_output"),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=True,
            fp16=False,
            seed=42
        )
        
        # 配置数据参数
        data_config = DataConfig(
            max_samples=len(self.training_data),
            train_split_ratio=0.9,
            eval_split_ratio=0.1,
            test_split_ratio=0.0,
            shuffle_data=True
        )
        
        # 配置并行策略
        if gpu_count > 1:
            parallel_config = ParallelConfig(
                strategy=ParallelStrategy.DATA_PARALLEL,
                data_parallel_size=gpu_count,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                master_addr="localhost",
                master_port=29500
            )
        else:
            parallel_config = ParallelConfig(
                strategy=ParallelStrategy.DATA_PARALLEL,
                data_parallel_size=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            )
        
        # 配置LoRA参数
        if gpu_infos:
            # 使用第一个GPU的信息来配置LoRA
            gpu_memory = gpu_infos[0].total_memory  # 以MB为单位
            lora_config = self.lora_optimizer.optimize_for_single_gpu(
                available_memory_mb=gpu_memory,
                batch_size=training_config.per_device_train_batch_size,
                sequence_length=2048
            )
        else:
            # 默认LoRA配置
            lora_config = LoRAMemoryProfile(
                rank=8,
                alpha=16,
                dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        
        # 系统配置
        system_config = SystemConfig(
            mixed_precision="bf16" if gpu_count > 0 else "no",
            cuda_visible_devices=",".join(str(i) for i in range(gpu_count)) if gpu_count > 0 else None,
            log_level="INFO"
        )
        
        self.logger.info("训练参数配置完成")
        self.logger.info(f"- 训练轮数: {training_config.num_train_epochs}")
        self.logger.info(f"- 批次大小: {training_config.per_device_train_batch_size}")
        self.logger.info(f"- 学习率: {training_config.learning_rate}")
        self.logger.info(f"- LoRA rank: {lora_config.rank}")
        self.logger.info(f"- 并行策略: {parallel_config.strategy.value}")
        
        return training_config, data_config, lora_config, parallel_config, system_config
    
    def run_training_pipeline(self, 
                            training_config: TrainingConfig,
                            data_config: DataConfig,
                            lora_config: LoRAMemoryProfile,
                            parallel_config: ParallelConfig,
                            system_config: SystemConfig) -> bool:
        """
        运行训练流水线
        
        Args:
            training_config: 训练配置
            data_config: 数据配置
            lora_config: LoRA配置
            parallel_config: 并行配置
            system_config: 系统配置
            
        Returns:
            bool: 是否成功完成训练
        """
        try:
            # 创建流水线编排器
            pipeline_id = f"demo_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            orchestrator = TrainingPipelineOrchestrator(
                pipeline_id=pipeline_id,
                output_dir=str(self.output_dir / "pipeline"),
                logger=self.logger
            )
            
            # 配置流水线
            orchestrator.configure_pipeline(
                training_data=self.training_data,
                training_config=training_config,
                data_config=data_config,
                lora_config=lora_config,
                parallel_config=parallel_config,
                system_config=system_config
            )
            
            # 添加进度回调
            def progress_callback(state: PipelineState):
                self.logger.info(f"训练进度: {state.progress:.1f}% - 当前阶段: {state.current_stage.value}")
                if state.current_stage_runtime:
                    self.logger.info(f"当前阶段运行时间: {state.current_stage_runtime}")
            
            orchestrator.add_progress_callback(progress_callback)
            
            # 运行流水线
            self.logger.info("开始执行训练流水线...")
            success = orchestrator.run_pipeline()
            
            if success:
                self.logger.info("训练流水线执行成功！")
                
                # 生成训练报告
                self.generate_training_report(orchestrator)
                
            else:
                self.logger.error("训练流水线执行失败！")
                if orchestrator.state.error_message:
                    self.logger.error(f"错误信息: {orchestrator.state.error_message}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"运行训练流水线失败: {e}")
            return False
    
    def generate_training_report(self, orchestrator: TrainingPipelineOrchestrator):
        """
        生成训练报告
        
        Args:
            orchestrator: 流水线编排器
        """
        try:
            state = orchestrator.get_state()
            
            report = {
                "demo_info": {
                    "name": "综合微调演示程序",
                    "version": "1.0.0",
                    "execution_time": datetime.now().isoformat()
                },
                "data_summary": {
                    "total_training_examples": len(self.training_data),
                    "thinking_examples": len(self.thinking_data),
                    "data_sources": list(set(ex.source_file for ex in self.training_data)),
                    "difficulty_distribution": self.get_difficulty_distribution(),
                    "crypto_terms_count": len(set(term for ex in self.training_data for term in ex.crypto_terms))
                },
                "training_summary": state.to_dict(),
                "output_files": {
                    "model_output": str(self.output_dir / "model_output"),
                    "pipeline_output": str(self.output_dir / "pipeline"),
                    "logs": str(self.output_dir)
                }
            }
            
            # 保存报告
            report_file = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"训练报告已生成: {report_file}")
            
            # 打印摘要
            self.print_training_summary(report)
            
        except Exception as e:
            self.logger.error(f"生成训练报告失败: {e}")
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """获取难度分布"""
        distribution = {}
        for example in self.training_data:
            level = example.difficulty_level.name
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
    
    def print_training_summary(self, report: Dict[str, Any]):
        """打印训练摘要"""
        print("\n" + "="*60)
        print("🎉 综合微调演示程序执行完成！")
        print("="*60)
        
        data_summary = report["data_summary"]
        print(f"📊 数据统计:")
        print(f"   - 训练样例总数: {data_summary['total_training_examples']}")
        print(f"   - Thinking样例: {data_summary['thinking_examples']}")
        print(f"   - 数据源文件: {len(data_summary['data_sources'])}")
        print(f"   - 密码学术语: {data_summary['crypto_terms_count']}")
        
        print(f"\n📈 难度分布:")
        for level, count in data_summary["difficulty_distribution"].items():
            print(f"   - {level}: {count}")
        
        training_summary = report["training_summary"]
        print(f"\n🚀 训练状态:")
        print(f"   - 状态: {training_summary['status']}")
        print(f"   - 进度: {training_summary['progress']:.1f}%")
        print(f"   - 当前阶段: {training_summary['current_stage']}")
        
        if training_summary.get('runtime_seconds'):
            runtime = training_summary['runtime_seconds']
            print(f"   - 运行时间: {runtime//3600:.0f}h {(runtime%3600)//60:.0f}m {runtime%60:.0f}s")
        
        output_files = report["output_files"]
        print(f"\n📁 输出文件:")
        print(f"   - 模型输出: {output_files['model_output']}")
        print(f"   - 流水线输出: {output_files['pipeline_output']}")
        print(f"   - 日志文件: {output_files['logs']}")
        
        print("\n" + "="*60)
    
    def run_demo(self, data_dir: str = "data/raw") -> bool:
        """
        运行完整演示
        
        Args:
            data_dir: 数据目录
            
        Returns:
            bool: 是否成功完成
        """
        try:
            print("🚀 启动综合微调演示程序...")
            print(f"📂 数据目录: {data_dir}")
            print(f"📁 输出目录: {self.output_dir}")
            
            # 1. 加载原始数据
            print("\n📖 步骤 1: 加载原始数据...")
            if not self.load_raw_data(data_dir):
                print("❌ 数据加载失败")
                return False
            
            print(f"✅ 成功加载 {len(self.training_data)} 个训练样例")
            
            # 2. 自动配置训练参数
            print("\n⚙️ 步骤 2: 自动配置训练参数...")
            configs = self.auto_configure_training()
            training_config, data_config, lora_config, parallel_config, system_config = configs
            
            print("✅ 训练参数配置完成")
            
            # 3. 运行训练流水线
            print("\n🎯 步骤 3: 运行训练流水线...")
            success = self.run_training_pipeline(
                training_config, data_config, lora_config, parallel_config, system_config
            )
            
            if success:
                print("✅ 训练流水线执行成功")
                return True
            else:
                print("❌ 训练流水线执行失败")
                return False
                
        except Exception as e:
            self.logger.error(f"演示程序执行失败: {e}")
            print(f"❌ 演示程序执行失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合微调演示程序")
    parser.add_argument("--data-dir", default="data/raw", help="数据目录路径")
    parser.add_argument("--output-dir", default="demo_output", help="输出目录路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建演示程序
    demo = ComprehensiveFinetuningDemo(output_dir=args.output_dir)
    
    # 运行演示
    success = demo.run_demo(data_dir=args.data_dir)
    
    if success:
        print("\n🎉 演示程序执行成功！")
        sys.exit(0)
    else:
        print("\n💥 演示程序执行失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()