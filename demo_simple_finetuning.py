#!/usr/bin/env python3
"""
简化版微调演示程序

基于当前已完成的功能，使用 data/raw 数据，进行模型微调的简化演示。
避免复杂的流水线管理，专注于核心功能展示。

功能特性：
- 自动加载和处理 data/raw 中的所有数据文件
- 智能数据预处理和格式转换
- 自动配置 GPU 并行策略和 LoRA 参数
- 生成 LLaMA Factory 兼容的配置文件
- 提供训练脚本和配置

使用方法：
    uv run python demo_simple_finetuning.py
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
from data_models import TrainingExample, ThinkingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig
from gpu_utils import GPUDetector


class SimpleFinetuningDemo:
    """简化版微调演示类"""
    
    def __init__(self, output_dir: str = "simple_demo_output"):
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
        
        # 数据存储
        self.training_data: List[TrainingExample] = []
        
        self.logger.info("简化版微调演示程序初始化完成")
    
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
    
    def convert_to_training_format(self) -> Dict[str, Any]:
        """
        转换数据为训练格式（通用格式，不依赖特定框架）
        
        Returns:
            Dict[str, Any]: 转换结果信息
        """
        try:
            # 分割训练和验证数据
            train_ratio = 0.9
            split_idx = int(len(self.training_data) * train_ratio)
            train_examples = self.training_data[:split_idx]
            val_examples = self.training_data[split_idx:] if split_idx < len(self.training_data) else []
            
            # 转换为LLaMA Factory格式
            train_data = [example.to_llama_factory_format() for example in train_examples]
            val_data = [example.to_llama_factory_format() for example in val_examples] if val_examples else []
            
            # 保存训练数据
            train_file = self.output_dir / "train_data.json"
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            result = {
                "train_file": str(train_file),
                "train_samples": len(train_data)
            }
            
            # 保存验证数据
            if val_data:
                val_file = self.output_dir / "val_data.json"
                with open(val_file, 'w', encoding='utf-8') as f:
                    json.dump(val_data, f, ensure_ascii=False, indent=2)
                result["val_file"] = str(val_file)
                result["val_samples"] = len(val_data)
            
            self.logger.info(f"数据转换完成: 训练样例 {len(train_data)}, 验证样例 {len(val_data)}")
            return result
            
        except Exception as e:
            self.logger.error(f"数据格式转换失败: {e}")
            return {}
    
    def generate_dataset_info(self, data_files: Dict[str, Any]) -> str:
        """
        生成数据集信息文件
        
        Args:
            data_files: 数据文件信息
            
        Returns:
            str: 数据集信息文件路径
        """
        try:
            dataset_info = {
                "crypto_qa_dataset": {
                    "file_name": data_files["train_file"],
                    "formatting": "alpaca",
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                        "system": "system"
                    }
                }
            }
            
            # 如果有验证数据，添加到配置中
            if "val_file" in data_files:
                dataset_info["crypto_qa_dataset"]["file_name"] = [
                    data_files["train_file"],
                    data_files["val_file"]
                ]
            
            # 保存数据集信息
            info_file = self.output_dir / "dataset_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据集信息文件已生成: {info_file}")
            return str(info_file)
            
        except Exception as e:
            self.logger.error(f"生成数据集信息文件失败: {e}")
            return ""
    
    def generate_training_config(self) -> str:
        """
        生成训练配置文件
        
        Returns:
            str: 配置文件路径
        """
        try:
            # 检测GPU环境
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            gpu_count = len(gpu_infos) if gpu_infos else 1
            
            # 基础配置
            config = {
                # 模型配置
                "model_name": "Qwen/Qwen3-4B-Thinking-2507",
                "model_revision": "main",
                "template": "qwen",
                "flash_attn": "auto",
                
                # 训练配置
                "stage": "sft",
                "do_train": True,
                "finetuning_type": "lora",
                "dataset": "crypto_qa_dataset",
                "cutoff_len": 2048,
                "train_on_prompt": False,
                "mask_history": True,
                
                # 输出配置
                "output_dir": str(self.output_dir / "model_output"),
                "overwrite_output_dir": True,
                
                # 训练参数
                "num_train_epochs": 2,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                
                # 优化器
                "optim": "adamw_torch",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0,
                
                # 混合精度
                "bf16": True,
                "fp16": False,
                "tf32": True,
                
                # LoRA配置
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_target": "all",
                
                # 保存和评估
                "save_strategy": "steps",
                "save_steps": 100,
                "save_total_limit": 3,
                "evaluation_strategy": "steps",
                "eval_steps": 100,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                
                # 日志
                "logging_steps": 10,
                "log_level": "info",
                "plot_loss": True,
                
                # 其他
                "seed": 42,
                "val_size": 0.1,
                "preprocessing_num_workers": 4
            }
            
            # 多GPU配置
            if gpu_count > 1:
                config.update({
                    "ddp_timeout": 1800,
                    "ddp_backend": "nccl",
                    "ddp_find_unused_parameters": False,
                    "dataloader_pin_memory": True
                })
            
            # 保存配置文件
            config_file = self.output_dir / "training_config.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.logger.info(f"训练配置文件已生成: {config_file}")
            return str(config_file)
            
        except Exception as e:
            self.logger.error(f"生成训练配置文件失败: {e}")
            return ""
    
    def generate_training_script(self, config_file: str) -> str:
        """
        生成训练脚本
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            str: 训练脚本路径
        """
        try:
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
        # 配置文件路径
        config_file = "{config_file}"
        dataset_info_file = "{self.output_dir / 'dataset_info.json'}"
        
        # 检查文件是否存在
        if not Path(config_file).exists():
            logger.error(f"配置文件不存在: {{config_file}}")
            return False
        
        if not Path(dataset_info_file).exists():
            logger.error(f"数据集信息文件不存在: {{dataset_info_file}}")
            return False
        
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("配置信息:")
        logger.info(f"- 模型: {{config.get('model_name', 'Unknown')}}")
        logger.info(f"- 数据集: {{config.get('dataset', 'Unknown')}}")
        logger.info(f"- 输出目录: {{config.get('output_dir', 'Unknown')}}")
        logger.info(f"- 训练轮数: {{config.get('num_train_epochs', 'Unknown')}}")
        logger.info(f"- 学习率: {{config.get('learning_rate', 'Unknown')}}")
        logger.info(f"- LoRA rank: {{config.get('lora_rank', 'Unknown')}}")
        
        # 设置环境变量
        os.environ["DATASET_INFO_FILE"] = dataset_info_file
        
        # 检查LLaMA Factory是否可用
        try:
            # 这里应该导入并调用LLaMA Factory的训练函数
            # from llamafactory.train.tuner import run_exp
            # run_exp(config)
            
            logger.info("注意: 这是一个演示脚本")
            logger.info("要进行实际训练，请:")
            logger.info("1. 安装 LLaMA Factory: pip install llamafactory")
            logger.info("2. 取消注释上面的导入和调用代码")
            logger.info("3. 或者使用 LLaMA Factory CLI:")
            logger.info(f"   llamafactory-cli train {{config_file}}")
            
            return True
            
        except ImportError as e:
            logger.error(f"LLaMA Factory未安装: {{e}}")
            logger.info("请安装 LLaMA Factory: pip install llamafactory")
            return False
        
    except Exception as e:
        logger.error(f"训练执行失败: {{e}}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
            
            script_file = self.output_dir / "train.py"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 设置执行权限（在Unix系统上）
            if os.name != 'nt':  # 不是Windows
                os.chmod(script_file, 0o755)
            
            self.logger.info(f"训练脚本已生成: {script_file}")
            return str(script_file)
            
        except Exception as e:
            self.logger.error(f"生成训练脚本失败: {e}")
            return ""
    
    def generate_summary_report(self, data_files: Dict[str, Any], config_file: str, script_file: str) -> str:
        """
        生成总结报告
        
        Args:
            data_files: 数据文件信息
            config_file: 配置文件路径
            script_file: 训练脚本路径
            
        Returns:
            str: 报告文件路径
        """
        try:
            # 统计信息
            difficulty_dist = {}
            crypto_terms_count = {}
            
            for example in self.training_data:
                # 难度分布
                level = example.difficulty_level.name
                difficulty_dist[level] = difficulty_dist.get(level, 0) + 1
                
                # 术语统计
                for term in example.crypto_terms:
                    crypto_terms_count[term] = crypto_terms_count.get(term, 0) + 1
            
            # GPU信息
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            gpu_summary = []
            for gpu in gpu_infos:
                gpu_summary.append({
                    "name": gpu.name,
                    "memory_gb": round(gpu.total_memory / 1024, 1),
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}" if gpu.compute_capability else "Unknown"
                })
            
            report = {
                "demo_info": {
                    "name": "简化版微调演示程序",
                    "version": "1.0.0",
                    "execution_time": datetime.now().isoformat(),
                    "output_directory": str(self.output_dir)
                },
                "data_summary": {
                    "total_examples": len(self.training_data),
                    "train_examples": data_files.get("train_samples", 0),
                    "val_examples": data_files.get("val_samples", 0),
                    "difficulty_distribution": difficulty_dist,
                    "top_crypto_terms": dict(sorted(crypto_terms_count.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                "hardware_summary": {
                    "gpu_count": len(gpu_infos),
                    "gpu_details": gpu_summary
                },
                "generated_files": {
                    "train_data": data_files.get("train_file", ""),
                    "val_data": data_files.get("val_file", ""),
                    "dataset_info": str(self.output_dir / "dataset_info.json"),
                    "training_config": config_file,
                    "training_script": script_file
                },
                "next_steps": [
                    "检查生成的配置文件和数据文件",
                    "安装 LLaMA Factory: pip install llamafactory",
                    "运行训练脚本: python train.py",
                    "或使用 CLI: llamafactory-cli train training_config.yaml"
                ]
            }
            
            # 保存报告
            report_file = self.output_dir / f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"总结报告已生成: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"生成总结报告失败: {e}")
            return ""
    
    def run_demo(self, data_dir: str = "data/raw") -> bool:
        """
        运行完整演示
        
        Args:
            data_dir: 数据目录
            
        Returns:
            bool: 是否成功完成
        """
        try:
            print("🚀 启动简化版微调演示程序...")
            print(f"📂 数据目录: {data_dir}")
            print(f"📁 输出目录: {self.output_dir}")
            
            # 1. 加载和处理数据
            print("\n📖 步骤 1: 加载和处理数据...")
            if not self.load_and_process_data(data_dir):
                print("❌ 数据加载失败")
                return False
            
            print(f"✅ 成功加载 {len(self.training_data)} 个训练样例")
            
            # 2. 转换数据格式
            print("\n🔄 步骤 2: 转换数据格式...")
            data_files = self.convert_to_training_format()
            if not data_files:
                print("❌ 数据格式转换失败")
                return False
            
            print(f"✅ 数据格式转换完成")
            
            # 3. 生成数据集信息
            print("\n📋 步骤 3: 生成数据集信息...")
            dataset_info_file = self.generate_dataset_info(data_files)
            if not dataset_info_file:
                print("❌ 数据集信息生成失败")
                return False
            
            print(f"✅ 数据集信息生成完成")
            
            # 4. 生成训练配置
            print("\n⚙️ 步骤 4: 生成训练配置...")
            config_file = self.generate_training_config()
            if not config_file:
                print("❌ 训练配置生成失败")
                return False
            
            print(f"✅ 训练配置生成完成")
            
            # 5. 生成训练脚本
            print("\n📝 步骤 5: 生成训练脚本...")
            script_file = self.generate_training_script(config_file)
            if not script_file:
                print("❌ 训练脚本生成失败")
                return False
            
            print(f"✅ 训练脚本生成完成")
            
            # 6. 生成总结报告
            print("\n📊 步骤 6: 生成总结报告...")
            report_file = self.generate_summary_report(data_files, config_file, script_file)
            
            # 打印总结
            self.print_summary(data_files, config_file, script_file, report_file)
            
            return True
                
        except Exception as e:
            self.logger.error(f"演示程序执行失败: {e}")
            print(f"❌ 演示程序执行失败: {e}")
            return False
    
    def print_summary(self, data_files: Dict[str, Any], config_file: str, script_file: str, report_file: str):
        """打印总结信息"""
        print("\n" + "="*60)
        print("🎉 简化版微调演示程序执行完成！")
        print("="*60)
        
        print(f"📊 数据统计:")
        print(f"   - 训练样例: {data_files.get('train_samples', 0)}")
        print(f"   - 验证样例: {data_files.get('val_samples', 0)}")
        print(f"   - 总样例数: {len(self.training_data)}")
        
        # GPU信息
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        if gpu_infos:
            print(f"\n🖥️ GPU信息:")
            for i, gpu in enumerate(gpu_infos):
                print(f"   - GPU {i}: {gpu.name} ({gpu.total_memory/1024:.1f}GB)")
        
        print(f"\n📁 生成的文件:")
        print(f"   - 训练数据: {data_files.get('train_file', 'N/A')}")
        if 'val_file' in data_files:
            print(f"   - 验证数据: {data_files['val_file']}")
        print(f"   - 数据集信息: {self.output_dir}/dataset_info.json")
        print(f"   - 训练配置: {config_file}")
        print(f"   - 训练脚本: {script_file}")
        print(f"   - 总结报告: {report_file}")
        
        print(f"\n🚀 下一步操作:")
        print(f"   1. 检查生成的配置文件: {config_file}")
        print(f"   2. 安装 LLaMA Factory: pip install llamafactory")
        print(f"   3. 运行训练: python {script_file}")
        print(f"   4. 或使用 CLI: llamafactory-cli train {config_file}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版微调演示程序")
    parser.add_argument("--data-dir", default="data/raw", help="数据目录路径")
    parser.add_argument("--output-dir", default="simple_demo_output", help="输出目录路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建演示程序
    demo = SimpleFinetuningDemo(output_dir=args.output_dir)
    
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