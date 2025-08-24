#!/usr/bin/env python3
"""
最终演示程序

基于当前已完成的功能，使用 data/raw 数据，展示完整的数据处理和配置生成流程。
专注于核心功能展示，避免复杂的训练流水线。

功能特性：
- ✅ 自动加载和处理 data/raw 中的所有数据文件
- ✅ 智能数据预处理和格式转换
- ✅ 自动配置 GPU 并行策略和 LoRA 参数
- ✅ 生成 LLaMA Factory 兼容的配置文件
- ✅ 提供完整的训练准备和指导

使用方法：
    uv run python demo_final.py
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入核心模块
from data_models import TrainingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig, SystemConfig
from lora_config_optimizer import LoRAConfigOptimizer, LoRAMemoryProfile
from parallel_config import ParallelConfig, ParallelStrategy
from gpu_utils import GPUDetector
from llamafactory_adapter import LlamaFactoryAdapter


class FinalDemo:
    """最终演示类"""
    
    def __init__(self, output_dir: str = "final_demo_output"):
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
        self.llamafactory_adapter = LlamaFactoryAdapter(self.logger)
        
        # 数据存储
        self.training_data: List[TrainingExample] = []
        
        self.logger.info("最终演示程序初始化完成")
    
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
    
    def load_and_process_data(self, data_dir: str = "data/raw") -> bool:
        """
        加载和处理原始数据
        
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
            
            self.logger.info(f"总共加载了 {total_examples} 个训练样例")
            return total_examples > 0
            
        except Exception as e:
            self.logger.error(f"加载和处理数据失败: {e}")
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
            if section.startswith('### Q') or (section.startswith('##') and '?' in section):
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
        """提取密码学术语"""
        crypto_keywords = [
            "密码学", "加密", "解密", "哈希", "数字签名", "公钥", "私钥", "对称加密", "非对称加密",
            "AES", "RSA", "SHA", "MD5", "DES", "3DES", "ECC", "DSA", "ECDSA",
            "密钥管理", "证书", "PKI", "CA", "数字证书", "身份认证", "访问控制",
            "完整性", "机密性", "真实性", "不可否认性", "随机数", "密钥交换"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in crypto_keywords:
            if term.lower() in text_lower or term in text:
                found_terms.append(term)
        
        return list(set(found_terms))
    
    def determine_difficulty(self, instruction: str, output: str) -> DifficultyLevel:
        """判断难度级别"""
        text = instruction + " " + output
        text_len = len(text)
        
        if text_len < 200:
            return DifficultyLevel.BEGINNER
        elif text_len < 500:
            return DifficultyLevel.INTERMEDIATE
        elif text_len < 1000:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT
    
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
    
    def prepare_training_files(self, configs: tuple) -> Dict[str, str]:
        """
        准备训练文件
        
        Args:
            configs: 配置元组
            
        Returns:
            Dict[str, str]: 生成的文件路径
        """
        training_config, data_config, lora_config, parallel_config, system_config = configs
        
        try:
            # 准备训练数据
            dataset_name = "crypto_qa_dataset"
            data_files = self.llamafactory_adapter.prepare_training_data(
                self.training_data,
                str(self.output_dir / "data"),
                dataset_name,
                "alpaca",
                0.9
            )
            
            # 生成训练配置
            config_file = self.llamafactory_adapter.create_training_config(
                training_config,
                data_config,
                lora_config,
                parallel_config,
                dataset_name,
                str(self.output_dir / "configs")
            )
            
            # 生成训练脚本
            script_file = self.llamafactory_adapter.generate_training_script(
                config_file,
                str(self.output_dir / "train.py")
            )
            
            result = {
                "train_data": data_files.get("train_file", ""),
                "val_data": data_files.get("val_file", ""),
                "dataset_info": data_files.get("dataset_info_file", ""),
                "config_file": config_file,
                "script_file": script_file
            }
            
            self.logger.info("训练文件准备完成")
            return result
            
        except Exception as e:
            self.logger.error(f"准备训练文件失败: {e}")
            return {}
    
    def generate_comprehensive_report(self, configs: tuple, files: Dict[str, str]) -> str:
        """
        生成综合报告
        
        Args:
            configs: 配置元组
            files: 文件路径字典
            
        Returns:
            str: 报告文件路径
        """
        try:
            training_config, data_config, lora_config, parallel_config, system_config = configs
            
            # 统计信息
            difficulty_dist = {}
            crypto_terms_count = {}
            thinking_count = 0
            
            for example in self.training_data:
                # 难度分布
                level = example.difficulty_level.name
                difficulty_dist[level] = difficulty_dist.get(level, 0) + 1
                
                # 术语统计
                for term in example.crypto_terms:
                    crypto_terms_count[term] = crypto_terms_count.get(term, 0) + 1
                
                # thinking样例统计
                if example.has_thinking():
                    thinking_count += 1
            
            # GPU信息
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            gpu_summary = []
            for gpu in gpu_infos:
                gpu_summary.append({
                    "name": gpu.name,
                    "memory_gb": round(gpu.total_memory / 1024, 1),
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}" if gpu.compute_capability else "Unknown",
                    "utilization": f"{gpu.utilization}%",
                    "temperature": f"{gpu.temperature}°C" if gpu.temperature else "N/A"
                })
            
            report = {
                "demo_info": {
                    "name": "最终微调演示程序",
                    "version": "1.0.0",
                    "execution_time": datetime.now().isoformat(),
                    "output_directory": str(self.output_dir)
                },
                "data_analysis": {
                    "total_examples": len(self.training_data),
                    "thinking_examples": thinking_count,
                    "thinking_ratio": round(thinking_count / len(self.training_data) * 100, 1) if self.training_data else 0,
                    "difficulty_distribution": difficulty_dist,
                    "top_crypto_terms": dict(sorted(crypto_terms_count.items(), key=lambda x: x[1], reverse=True)[:15]),
                    "avg_instruction_length": round(sum(len(ex.instruction) for ex in self.training_data) / len(self.training_data), 1) if self.training_data else 0,
                    "avg_output_length": round(sum(len(ex.output) for ex in self.training_data) / len(self.training_data), 1) if self.training_data else 0
                },
                "hardware_analysis": {
                    "gpu_count": len(gpu_infos),
                    "total_gpu_memory_gb": sum(gpu.total_memory / 1024 for gpu in gpu_infos),
                    "gpu_details": gpu_summary,
                    "recommended_batch_size": 1 if gpu_infos else "CPU模式",
                    "parallel_strategy": parallel_config.strategy.value
                },
                "training_configuration": {
                    "model": "Qwen/Qwen3-4B-Thinking-2507",
                    "epochs": training_config.num_train_epochs,
                    "learning_rate": training_config.learning_rate,
                    "batch_size": training_config.per_device_train_batch_size,
                    "lora_rank": lora_config.rank,
                    "lora_alpha": lora_config.alpha,
                    "sequence_length": 2048,
                    "mixed_precision": system_config.mixed_precision
                },
                "generated_files": files,
                "training_instructions": {
                    "prerequisites": [
                        "确保已安装 LLaMA Factory: pip install llamafactory",
                        "检查CUDA环境（如果使用GPU）",
                        "确认有足够的磁盘空间用于模型输出"
                    ],
                    "training_commands": [
                        f"cd {self.output_dir}",
                        f"llamafactory-cli train {files.get('config_file', 'config.yaml')}",
                        "或者运行: python train.py"
                    ],
                    "monitoring": [
                        "训练过程中可以通过 tensorboard 监控",
                        f"tensorboard --logdir {training_config.output_dir}/logs",
                        "检查生成的模型文件和检查点"
                    ]
                },
                "performance_estimates": {
                    "estimated_training_time": self.estimate_training_time(len(self.training_data), gpu_infos),
                    "memory_usage": self.estimate_memory_usage(lora_config, gpu_infos),
                    "disk_space_needed": "约 2-5GB（取决于检查点数量）"
                }
            }
            
            # 保存报告
            report_file = self.output_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"综合报告已生成: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"生成综合报告失败: {e}")
            return ""
    
    def estimate_training_time(self, num_examples: int, gpu_infos: List) -> str:
        """估算训练时间"""
        if not gpu_infos:
            return "CPU模式：预计数小时"
        
        # 简单估算：基于样例数量和GPU性能
        samples_per_minute = 10 * len(gpu_infos)  # 假设每个GPU每分钟处理10个样例
        total_minutes = (num_examples * 2) / samples_per_minute  # 2个epoch
        
        if total_minutes < 60:
            return f"约 {int(total_minutes)} 分钟"
        else:
            hours = int(total_minutes // 60)
            minutes = int(total_minutes % 60)
            return f"约 {hours} 小时 {minutes} 分钟"
    
    def estimate_memory_usage(self, lora_config, gpu_infos: List) -> str:
        """估算内存使用"""
        if not gpu_infos:
            return "CPU模式：约 8-16GB 系统内存"
        
        # 基于LoRA配置估算GPU内存使用
        base_memory = 6  # Qwen3-4B基础内存需求（GB）
        lora_memory = lora_config.rank * 0.1  # LoRA额外内存
        total_memory = base_memory + lora_memory
        
        return f"约 {total_memory:.1f}GB GPU内存 / GPU"
    
    def print_final_summary(self, report_data: Dict[str, Any]):
        """打印最终总结"""
        print("\n" + "="*80)
        print("🎉 最终微调演示程序执行完成！")
        print("="*80)
        
        data_analysis = report_data["data_analysis"]
        print(f"📊 数据分析:")
        print(f"   - 总训练样例: {data_analysis['total_examples']}")
        print(f"   - Thinking样例: {data_analysis['thinking_examples']} ({data_analysis['thinking_ratio']}%)")
        print(f"   - 平均指令长度: {data_analysis['avg_instruction_length']} 字符")
        print(f"   - 平均输出长度: {data_analysis['avg_output_length']} 字符")
        
        print(f"\n📈 难度分布:")
        for level, count in data_analysis["difficulty_distribution"].items():
            print(f"   - {level}: {count}")
        
        print(f"\n🔑 热门密码学术语:")
        for term, count in list(data_analysis["top_crypto_terms"].items())[:5]:
            print(f"   - {term}: {count}")
        
        hardware_analysis = report_data["hardware_analysis"]
        print(f"\n🖥️ 硬件配置:")
        print(f"   - GPU数量: {hardware_analysis['gpu_count']}")
        print(f"   - 总GPU内存: {hardware_analysis['total_gpu_memory_gb']:.1f}GB")
        print(f"   - 并行策略: {hardware_analysis['parallel_strategy']}")
        
        training_config = report_data["training_configuration"]
        print(f"\n⚙️ 训练配置:")
        print(f"   - 模型: {training_config['model']}")
        print(f"   - 训练轮数: {training_config['epochs']}")
        print(f"   - 学习率: {training_config['learning_rate']}")
        print(f"   - LoRA rank: {training_config['lora_rank']}")
        
        performance = report_data["performance_estimates"]
        print(f"\n⏱️ 性能预估:")
        print(f"   - 预计训练时间: {performance['estimated_training_time']}")
        print(f"   - 内存使用: {performance['memory_usage']}")
        print(f"   - 磁盘空间: {performance['disk_space_needed']}")
        
        print(f"\n📁 生成的文件:")
        files = report_data["generated_files"]
        for file_type, file_path in files.items():
            if file_path:
                print(f"   - {file_type}: {file_path}")
        
        print(f"\n🚀 开始训练:")
        instructions = report_data["training_instructions"]
        print("   前置条件:")
        for prereq in instructions["prerequisites"]:
            print(f"     • {prereq}")
        
        print("   训练命令:")
        for cmd in instructions["training_commands"]:
            print(f"     • {cmd}")
        
        print("\n" + "="*80)
    
    def run_demo(self, data_dir: str = "data/raw") -> bool:
        """
        运行完整演示
        
        Args:
            data_dir: 数据目录
            
        Returns:
            bool: 是否成功完成
        """
        try:
            print("🚀 启动最终微调演示程序...")
            print(f"📂 数据目录: {data_dir}")
            print(f"📁 输出目录: {self.output_dir}")
            
            # 1. 加载和处理数据
            print("\n📖 步骤 1: 加载和处理数据...")
            if not self.load_and_process_data(data_dir):
                print("❌ 数据加载失败")
                return False
            
            print(f"✅ 成功加载 {len(self.training_data)} 个训练样例")
            
            # 2. 自动配置训练参数
            print("\n⚙️ 步骤 2: 自动配置训练参数...")
            configs = self.auto_configure_training()
            print("✅ 训练参数配置完成")
            
            # 3. 准备训练文件
            print("\n📝 步骤 3: 准备训练文件...")
            files = self.prepare_training_files(configs)
            if not files:
                print("❌ 训练文件准备失败")
                return False
            
            print("✅ 训练文件准备完成")
            
            # 4. 生成综合报告
            print("\n📊 步骤 4: 生成综合报告...")
            report_file = self.generate_comprehensive_report(configs, files)
            if not report_file:
                print("❌ 综合报告生成失败")
                return False
            
            print("✅ 综合报告生成完成")
            
            # 5. 显示最终总结
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            self.print_final_summary(report_data)
            
            return True
                
        except Exception as e:
            self.logger.error(f"演示程序执行失败: {e}")
            print(f"❌ 演示程序执行失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="最终微调演示程序")
    parser.add_argument("--data-dir", default="data/raw", help="数据目录路径")
    parser.add_argument("--output-dir", default="final_demo_output", help="输出目录路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建演示程序
    demo = FinalDemo(output_dir=args.output_dir)
    
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