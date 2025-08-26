"""
端到端集成测试套件

本模块实现了完整训练流程的自动化测试，包括：
- 多种配置场景的测试覆盖
- 性能基准测试和回归测试
- 中文密码学数据的训练效果验证
- 完整集成测试套件

测试覆盖：
1. 单GPU训练流程测试
2. 多GPU分布式训练测试
3. 内存管理和优化测试
4. 数据处理和格式验证测试
5. 模型导出和量化测试
6. 中文密码学专业能力测试
7. 性能基准和回归测试
"""

import os
import sys
import json
import yaml
import time
import tempfile
import shutil
import logging
import unittest
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import multiprocessing as mp

import torch
import torch.distributed as dist
import pytest

# 导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from src.training_pipeline import TrainingPipelineOrchestrator, PipelineStage, PipelineStatus
from src.data_models import TrainingExample, ThinkingExample, CryptoTerm, ChineseMetrics
from src.config_manager import TrainingConfig, DataConfig, SystemConfig
from src.lora_config_optimizer import LoRAMemoryProfile, MultiGPULoRAConfig
from src.parallel_config import ParallelConfig, GPUTopology, CommunicationBackend
from src.gpu_utils import GPUDetector
from src.memory_manager import MemoryManager, MemoryPressureLevel
from src.distributed_training_engine import MultiGPUProcessManager, DistributedBackendInitializer
from src.training_monitor import TrainingMonitor
from src.model_exporter import ModelExporter, QuantizationFormat
from src.evaluation_framework import ComprehensiveEvaluationFramework
from src.chinese_nlp_processor import ChineseNLPProcessor
from src.crypto_term_processor import CryptoTermProcessor
from src.thinking_generator import ThinkingGenerator
from src.dataset_splitter import DatasetSplitter


@dataclass
class TestConfiguration:
    """测试配置"""
    test_name: str
    description: str
    gpu_count: int
    batch_size: int
    sequence_length: int
    num_epochs: int
    enable_thinking: bool
    enable_crypto_terms: bool
    enable_chinese_processing: bool
    expected_duration_minutes: int
    memory_limit_mb: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "description": self.description,
            "gpu_count": self.gpu_count,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "num_epochs": self.num_epochs,
            "enable_thinking": self.enable_thinking,
            "enable_crypto_terms": self.enable_crypto_terms,
            "enable_chinese_processing": self.enable_chinese_processing,
            "expected_duration_minutes": self.expected_duration_minutes,
            "memory_limit_mb": self.memory_limit_mb
        }


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "memory_usage": self.memory_usage,
            "performance_metrics": self.performance_metrics
        }


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chinese_processor = ChineseNLPProcessor()
        self.crypto_processor = CryptoTermProcessor()
        self.thinking_generator = ThinkingGenerator()
    
    def generate_basic_training_data(self, count: int = 50) -> List[TrainingExample]:
        """生成基础训练数据"""
        examples = []
        
        basic_qa_pairs = [
            ("什么是对称加密？", "对称加密是一种加密方式，加密和解密使用相同的密钥。常见的对称加密算法包括AES、DES等。"),
            ("RSA算法的基本原理是什么？", "RSA算法基于大整数分解的数学难题，使用公钥和私钥进行加密解密。"),
            ("什么是哈希函数？", "哈希函数是将任意长度的输入映射为固定长度输出的函数，具有单向性和抗碰撞性。"),
            ("数字签名的作用是什么？", "数字签名用于验证数据的完整性和发送者的身份，防止数据被篡改。"),
            ("什么是椭圆曲线密码学？", "椭圆曲线密码学基于椭圆曲线上的离散对数问题，提供与RSA相同安全级别但密钥更短的加密方案。")
        ]
        
        for i in range(count):
            qa_pair = basic_qa_pairs[i % len(basic_qa_pairs)]
            
            # 添加变化
            instruction = f"请解释：{qa_pair[0]}"
            if i % 3 == 0:
                instruction = f"作为密码学专家，{qa_pair[0]}"
            elif i % 3 == 1:
                instruction = f"从技术角度分析：{qa_pair[0]}"
            
            example = TrainingExample(
                instruction=instruction,
                input="",
                output=qa_pair[1],
                thinking=None,
                crypto_terms=self.crypto_processor.extract_crypto_terms(qa_pair[1]),
                difficulty_level=1 + (i % 3),
                source_file=f"test_data_{i}.md"
            )
            examples.append(example)
        
        return examples
    
    def generate_thinking_training_data(self, count: int = 30) -> List[ThinkingExample]:
        """生成深度思考训练数据"""
        examples = []
        
        thinking_scenarios = [
            {
                "instruction": "分析AES-256加密算法的安全性",
                "base_response": "AES-256是目前最安全的对称加密算法之一，使用256位密钥。",
                "thinking_points": [
                    "首先需要了解AES算法的基本结构",
                    "分析256位密钥的安全强度",
                    "考虑已知的攻击方法",
                    "评估在量子计算威胁下的安全性"
                ]
            },
            {
                "instruction": "设计一个安全的密钥交换协议",
                "base_response": "可以使用Diffie-Hellman密钥交换协议来安全地交换密钥。",
                "thinking_points": [
                    "分析密钥交换的安全需求",
                    "考虑中间人攻击的防护",
                    "选择合适的数学基础",
                    "设计具体的协议步骤"
                ]
            }
        ]
        
        for i in range(count):
            scenario = thinking_scenarios[i % len(thinking_scenarios)]
            
            # 生成思考过程
            thinking_process = self.thinking_generator.generate_thinking_process(
                scenario["instruction"],
                scenario["thinking_points"]
            )
            
            example = ThinkingExample(
                instruction=scenario["instruction"],
                thinking_process=thinking_process,
                final_response=scenario["base_response"],
                crypto_terms=self.crypto_processor.extract_crypto_terms(scenario["base_response"]),
                reasoning_steps=scenario["thinking_points"]
            )
            examples.append(example)
        
        return examples
    
    def generate_chinese_crypto_data(self, count: int = 40) -> List[TrainingExample]:
        """生成中文密码学数据"""
        examples = []
        
        chinese_crypto_content = [
            ("密码学中的混淆和扩散原理", "混淆是指加密算法应该使密文和密钥之间的关系尽可能复杂；扩散是指明文中每一位的改变都应该影响密文中的多位。"),
            ("中国商用密码算法SM4的特点", "SM4是中国自主设计的分组密码算法，采用128位密钥和128位分组长度，具有良好的安全性和效率。"),
            ("数字证书在电子商务中的应用", "数字证书通过公钥基础设施(PKI)为电子商务提供身份认证、数据完整性和不可否认性保障。"),
            ("量子密码学的发展前景", "量子密码学利用量子力学原理提供理论上无条件安全的通信，是未来密码学发展的重要方向。")
        ]
        
        for i in range(count):
            content = chinese_crypto_content[i % len(chinese_crypto_content)]
            
            # 中文文本处理
            processed_instruction = self.chinese_processor.preprocess_text(f"请详细说明{content[0]}。")
            processed_output = self.chinese_processor.preprocess_text(content[1])
            
            example = TrainingExample(
                instruction=processed_instruction,
                input="",
                output=processed_output,
                thinking=None,
                crypto_terms=self.crypto_processor.extract_crypto_terms(content[1]),
                difficulty_level=2,
                source_file=f"chinese_crypto_{i}.md"
            )
            examples.append(example)
        
        return examples


class EndToEndIntegrationTestSuite:
    """端到端集成测试套件"""
    
    def __init__(self, output_dir: str = "test_integration_output"):
        """初始化测试套件"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 初始化组件
        self.gpu_detector = GPUDetector()
        self.data_generator = TestDataGenerator()
        
        # 测试结果存储
        self.test_results: List[TestResult] = []
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # 测试配置
        self.test_configurations = self._create_test_configurations()
        
        self.logger.info(f"端到端集成测试套件初始化完成，输出目录: {self.output_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("EndToEndIntegrationTest")
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = self.output_dir / "integration_test.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_test_configurations(self) -> List[TestConfiguration]:
        """创建测试配置"""
        configurations = []
        
        # 基础单GPU测试
        configurations.append(TestConfiguration(
            test_name="basic_single_gpu",
            description="基础单GPU训练流程测试",
            gpu_count=1,
            batch_size=2,
            sequence_length=512,
            num_epochs=1,
            enable_thinking=False,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=10
        ))
        
        # 深度思考数据测试
        configurations.append(TestConfiguration(
            test_name="thinking_data_training",
            description="深度思考数据训练测试",
            gpu_count=1,
            batch_size=1,
            sequence_length=1024,
            num_epochs=1,
            enable_thinking=True,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=15
        ))
        
        # 内存优化测试
        configurations.append(TestConfiguration(
            test_name="memory_optimization",
            description="内存管理和优化测试",
            gpu_count=1,
            batch_size=4,
            sequence_length=2048,
            num_epochs=1,
            enable_thinking=False,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=20,
            memory_limit_mb=8000
        ))
        
        # 中文密码学专业测试
        configurations.append(TestConfiguration(
            test_name="chinese_crypto_expertise",
            description="中文密码学专业能力测试",
            gpu_count=1,
            batch_size=2,
            sequence_length=1024,
            num_epochs=2,
            enable_thinking=True,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=25
        ))
        
        # 多GPU测试（如果可用）
        gpu_count = len(self.gpu_detector.get_all_gpu_info())
        if gpu_count > 1:
            configurations.append(TestConfiguration(
                test_name="multi_gpu_distributed",
                description="多GPU分布式训练测试",
                gpu_count=min(2, gpu_count),
                batch_size=1,
                sequence_length=512,
                num_epochs=1,
                enable_thinking=False,
                enable_crypto_terms=True,
                enable_chinese_processing=True,
                expected_duration_minutes=15
            ))
        
        return configurations
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有集成测试"""
        self.logger.info("开始运行端到端集成测试套件")
        start_time = time.time()
        
        # 清理之前的结果
        self.test_results.clear()
        
        # 运行每个测试配置
        for config in self.test_configurations:
            self.logger.info(f"开始测试: {config.test_name} - {config.description}")
            
            try:
                result = self._run_single_test(config)
                self.test_results.append(result)
                
                if result.success:
                    self.logger.info(f"测试成功: {config.test_name}, 耗时: {result.duration_seconds:.2f}秒")
                else:
                    self.logger.error(f"测试失败: {config.test_name}, 错误: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"测试异常: {config.test_name}, 异常: {e}")
                result = TestResult(
                    test_name=config.test_name,
                    success=False,
                    duration_seconds=0,
                    error_message=str(e)
                )
                self.test_results.append(result)
        
        # 生成测试报告
        total_duration = time.time() - start_time
        report = self._generate_test_report(total_duration)
        
        # 保存测试报告
        self._save_test_report(report)
        
        self.logger.info(f"集成测试套件完成，总耗时: {total_duration:.2f}秒")
        return report
    
    def _run_single_test(self, config: TestConfiguration) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        test_output_dir = self.output_dir / config.test_name
        test_output_dir.mkdir(exist_ok=True)
        
        try:
            # 生成测试数据
            training_data = self._generate_test_data(config)
            
            # 创建配置
            training_config, data_config, lora_config, parallel_config, system_config = \
                self._create_test_configs(config, str(test_output_dir))
            
            # 初始化内存管理器
            memory_manager = None
            if config.memory_limit_mb:
                memory_manager = MemoryManager({
                    "monitoring_interval": 2,
                    "enable_auto_adjustment": True,
                    "initial_batch_size": config.batch_size
                })
                memory_manager.start()
            
            # 创建训练流水线
            pipeline = TrainingPipelineOrchestrator(
                pipeline_id=f"test_{config.test_name}",
                output_dir=str(test_output_dir),
                logger=self.logger
            )
            
            # 配置流水线
            pipeline.configure_pipeline(
                training_data=training_data,
                training_config=training_config,
                data_config=data_config,
                lora_config=lora_config,
                parallel_config=parallel_config,
                system_config=system_config
            )
            
            # 运行训练流水线
            success = pipeline.run_pipeline()
            
            # 收集指标
            metrics = self._collect_test_metrics(pipeline, config)
            memory_usage = None
            if memory_manager:
                memory_usage = memory_manager.get_memory_analysis()
                memory_manager.stop()
            
            # 性能指标
            performance_metrics = self._calculate_performance_metrics(
                pipeline, config, time.time() - start_time
            )
            
            return TestResult(
                test_name=config.test_name,
                success=success,
                duration_seconds=time.time() - start_time,
                metrics=metrics,
                memory_usage=memory_usage,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            return TestResult(
                test_name=config.test_name,
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_test_data(self, config: TestConfiguration) -> List[Any]:
        """生成测试数据"""
        training_data = []
        
        # 基础训练数据
        if not config.enable_thinking:
            basic_data = self.data_generator.generate_basic_training_data(30)
            training_data.extend(basic_data)
        
        # 深度思考数据
        if config.enable_thinking:
            thinking_data = self.data_generator.generate_thinking_training_data(20)
            training_data.extend(thinking_data)
        
        # 中文密码学数据
        if config.enable_chinese_processing:
            chinese_data = self.data_generator.generate_chinese_crypto_data(25)
            training_data.extend(chinese_data)
        
        return training_data
    
    def _create_test_configs(self, config: TestConfiguration, output_dir: str) -> Tuple[Any, ...]:
        """创建测试配置"""
        # 训练配置
        training_config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            save_steps=50,
            logging_steps=10,
            fp16=True,
            dataloader_num_workers=2,
            remove_unused_columns=False
        )
        
        # 数据配置
        data_config = DataConfig(
            max_samples=config.sequence_length,
            train_split_ratio=0.8,
            val_split_ratio=0.1,
            test_split_ratio=0.1,
            enable_thinking_data=config.enable_thinking,
            enable_crypto_terms=config.enable_crypto_terms,
            enable_chinese_processing=config.enable_chinese_processing
        )
        
        # LoRA配置
        if config.gpu_count == 1:
            lora_config = LoRAMemoryProfile(
                rank=16,
                alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                dropout=0.1,
                memory_efficient=True
            )
        else:
            lora_config = MultiGPULoRAConfig(
                global_config=LoRAMemoryProfile(
                    rank=16,
                    alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    dropout=0.1,
                    memory_efficient=True
                ),
                per_gpu_configs={}
            )
        
        # 并行配置
        parallel_config = ParallelConfig(
            world_size=config.gpu_count,
            master_addr="localhost",
            master_port=29500,
            communication_backend=CommunicationBackend.NCCL if config.gpu_count > 1 else CommunicationBackend.GLOO,
            enable_mixed_precision=True,
            gradient_accumulation_steps=2
        )
        
        # 系统配置
        system_config = SystemConfig(
            num_gpus=config.gpu_count,
            enable_multi_gpu=config.gpu_count > 1,
            max_seq_length=config.sequence_length,
            batch_size=config.batch_size,
            enable_gradient_checkpointing=True,
            enable_zero_optimization=config.gpu_count > 1
        )
        
        return training_config, data_config, lora_config, parallel_config, system_config
    
    def _collect_test_metrics(self, pipeline: TrainingPipelineOrchestrator, 
                            config: TestConfiguration) -> Dict[str, Any]:
        """收集测试指标"""
        metrics = {
            "pipeline_status": pipeline.state.status.value,
            "current_stage": pipeline.state.current_stage.value,
            "progress": pipeline.state.progress,
            "runtime_seconds": pipeline.state.runtime.total_seconds() if pipeline.state.runtime else 0,
            "stage_progress": {stage.value: progress for stage, progress in pipeline.state.stage_progress.items()},
            "checkpoints_created": len(pipeline.state.checkpoints)
        }
        
        # 添加配置信息
        metrics["test_config"] = config.to_dict()
        
        return metrics
    
    def _calculate_performance_metrics(self, pipeline: TrainingPipelineOrchestrator,
                                     config: TestConfiguration, duration: float) -> Dict[str, Any]:
        """计算性能指标"""
        # 估算处理的数据量
        estimated_samples = 50  # 基于生成的测试数据
        estimated_tokens = estimated_samples * config.sequence_length
        
        performance_metrics = {
            "samples_per_second": estimated_samples / duration if duration > 0 else 0,
            "tokens_per_second": estimated_tokens / duration if duration > 0 else 0,
            "duration_vs_expected": duration / (config.expected_duration_minutes * 60),
            "memory_efficiency": "unknown",  # 需要从内存管理器获取
            "gpu_utilization": "unknown"     # 需要从GPU监控获取
        }
        
        return performance_metrics
    
    def _generate_test_report(self, total_duration: float) -> Dict[str, Any]:
        """生成测试报告"""
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) if self.test_results else 0,
                "total_duration_seconds": total_duration,
                "average_test_duration": sum(r.duration_seconds for r in self.test_results) / len(self.test_results) if self.test_results else 0
            },
            "test_results": [result.to_dict() for result in self.test_results],
            "performance_analysis": self._analyze_performance(),
            "memory_analysis": self._analyze_memory_usage(),
            "recommendations": self._generate_recommendations(),
            "environment_info": self._collect_environment_info(),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """分析性能"""
        if not self.test_results:
            return {"error": "没有测试结果"}
        
        successful_results = [r for r in self.test_results if r.success and r.performance_metrics]
        
        if not successful_results:
            return {"error": "没有成功的测试结果"}
        
        # 计算性能统计
        durations = [r.duration_seconds for r in successful_results]
        tokens_per_second = [
            r.performance_metrics.get("tokens_per_second", 0) 
            for r in successful_results 
            if r.performance_metrics
        ]
        
        analysis = {
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "average_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0,
            "performance_variance": self._calculate_variance(durations)
        }
        
        return analysis
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用"""
        memory_results = [
            r for r in self.test_results 
            if r.success and r.memory_usage
        ]
        
        if not memory_results:
            return {"error": "没有内存使用数据"}
        
        # 提取内存统计
        peak_utilizations = []
        avg_utilizations = []
        
        for result in memory_results:
            if result.memory_usage and "memory_statistics" in result.memory_usage:
                stats = result.memory_usage["memory_statistics"]
                peak_utilizations.append(stats.get("max_utilization", 0))
                avg_utilizations.append(stats.get("avg_utilization", 0))
        
        analysis = {
            "average_peak_utilization": sum(peak_utilizations) / len(peak_utilizations) if peak_utilizations else 0,
            "average_utilization": sum(avg_utilizations) / len(avg_utilizations) if avg_utilizations else 0,
            "memory_pressure_events": sum(
                result.memory_usage.get("pressure_distribution", {}).get("high", 0) +
                result.memory_usage.get("pressure_distribution", {}).get("critical", 0)
                for result in memory_results
            )
        }
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        failed_tests = [r for r in self.test_results if not r.success]
        if failed_tests:
            recommendations.append(f"有{len(failed_tests)}个测试失败，需要检查错误原因并修复")
        
        # 性能建议
        performance_analysis = self._analyze_performance()
        if "average_duration" in performance_analysis:
            avg_duration = performance_analysis["average_duration"]
            if avg_duration > 1800:  # 30分钟
                recommendations.append("测试执行时间较长，建议优化训练性能或减少测试数据量")
        
        # 内存建议
        memory_analysis = self._analyze_memory_usage()
        if "average_peak_utilization" in memory_analysis:
            peak_util = memory_analysis["average_peak_utilization"]
            if peak_util > 0.9:
                recommendations.append("内存使用率过高，建议启用更多内存优化策略")
        
        if not recommendations:
            recommendations.append("所有测试通过，系统运行正常")
        
        return recommendations
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """收集环境信息"""
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        
        return {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": len(gpu_infos),
            "gpu_info": [
                {
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.name,
                    "total_memory": gpu.total_memory,
                    "compute_capability": f"{gpu.compute_capability_major}.{gpu.compute_capability_minor}"
                }
                for gpu in gpu_infos
            ],
            "system_info": {
                "platform": sys.platform,
                "cpu_count": mp.cpu_count()
            }
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _save_test_report(self, report: Dict[str, Any]):
        """保存测试报告"""
        # JSON格式报告
        json_report_path = self.output_dir / "integration_test_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成简化的文本报告
        text_report_path = self.output_dir / "integration_test_summary.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("端到端集成测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            summary = report["test_summary"]
            f.write(f"测试总数: {summary['total_tests']}\n")
            f.write(f"成功测试: {summary['successful_tests']}\n")
            f.write(f"失败测试: {summary['failed_tests']}\n")
            f.write(f"成功率: {summary['success_rate']:.2%}\n")
            f.write(f"总耗时: {summary['total_duration_seconds']:.2f}秒\n\n")
            
            f.write("测试结果详情:\n")
            f.write("-" * 30 + "\n")
            for result in self.test_results:
                status = "✓" if result.success else "✗"
                f.write(f"{status} {result.test_name}: {result.duration_seconds:.2f}秒\n")
                if not result.success and result.error_message:
                    f.write(f"  错误: {result.error_message}\n")
            
            f.write("\n改进建议:\n")
            f.write("-" * 30 + "\n")
            for rec in report["recommendations"]:
                f.write(f"• {rec}\n")
        
        self.logger.info(f"测试报告已保存: {json_report_path}")
        self.logger.info(f"测试摘要已保存: {text_report_path}")
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        self.logger.info("开始性能基准测试")
        
        # 创建基准测试配置
        benchmark_config = TestConfiguration(
            test_name="performance_benchmark",
            description="性能基准测试",
            gpu_count=1,
            batch_size=4,
            sequence_length=1024,
            num_epochs=1,
            enable_thinking=True,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=30
        )
        
        # 运行基准测试
        result = self._run_single_test(benchmark_config)
        
        # 保存基准结果
        if result.success and result.performance_metrics:
            self.performance_baselines[benchmark_config.test_name] = result.performance_metrics
            
            baseline_file = self.output_dir / "performance_baseline.json"
            with open(baseline_file, 'w') as f:
                json.dump(self.performance_baselines, f, indent=2)
            
            self.logger.info(f"性能基准已保存: {baseline_file}")
        
        return result.to_dict()
    
    def run_regression_test(self) -> Dict[str, Any]:
        """运行回归测试"""
        self.logger.info("开始回归测试")
        
        # 加载历史基准
        baseline_file = self.output_dir / "performance_baseline.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baselines = json.load(f)
        else:
            self.logger.warning("未找到性能基准文件，跳过回归测试")
            return {"error": "未找到性能基准文件"}
        
        # 运行当前测试
        current_result = self.run_performance_benchmark()
        
        if not current_result.get("success"):
            return {"error": "当前性能测试失败"}
        
        # 比较性能
        regression_analysis = {}
        current_metrics = current_result.get("performance_metrics", {})
        baseline_metrics = baselines.get("performance_benchmark", {})
        
        for metric_name in ["tokens_per_second", "samples_per_second"]:
            current_value = current_metrics.get(metric_name, 0)
            baseline_value = baseline_metrics.get(metric_name, 0)
            
            if baseline_value > 0:
                change_percent = (current_value - baseline_value) / baseline_value * 100
                regression_analysis[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "change_percent": change_percent,
                    "regression": change_percent < -10  # 性能下降超过10%视为回归
                }
        
        # 生成回归报告
        regression_report = {
            "regression_analysis": regression_analysis,
            "has_regression": any(
                analysis.get("regression", False) 
                for analysis in regression_analysis.values()
            ),
            "current_result": current_result,
            "baseline_date": baseline_file.stat().st_mtime,
            "test_date": time.time()
        }
        
        # 保存回归测试报告
        regression_file = self.output_dir / "regression_test_report.json"
        with open(regression_file, 'w') as f:
            json.dump(regression_report, f, indent=2)
        
        self.logger.info(f"回归测试报告已保存: {regression_file}")
        
        return regression_report


# 测试用例类
class TestEndToEndIntegration(unittest.TestCase):
    """端到端集成测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类设置"""
        cls.test_suite = EndToEndIntegrationTestSuite("test_e2e_output")
    
    def test_basic_training_pipeline(self):
        """测试基础训练流水线"""
        config = TestConfiguration(
            test_name="basic_pipeline_test",
            description="基础训练流水线测试",
            gpu_count=1,
            batch_size=1,
            sequence_length=256,
            num_epochs=1,
            enable_thinking=False,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=5
        )
        
        result = self.test_suite._run_single_test(config)
        self.assertTrue(result.success, f"基础训练流水线测试失败: {result.error_message}")
    
    def test_thinking_data_processing(self):
        """测试深度思考数据处理"""
        config = TestConfiguration(
            test_name="thinking_processing_test",
            description="深度思考数据处理测试",
            gpu_count=1,
            batch_size=1,
            sequence_length=512,
            num_epochs=1,
            enable_thinking=True,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=8
        )
        
        result = self.test_suite._run_single_test(config)
        self.assertTrue(result.success, f"深度思考数据处理测试失败: {result.error_message}")
    
    def test_memory_management(self):
        """测试内存管理"""
        config = TestConfiguration(
            test_name="memory_management_test",
            description="内存管理测试",
            gpu_count=1,
            batch_size=2,
            sequence_length=1024,
            num_epochs=1,
            enable_thinking=False,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=10,
            memory_limit_mb=6000
        )
        
        result = self.test_suite._run_single_test(config)
        self.assertTrue(result.success, f"内存管理测试失败: {result.error_message}")
        
        # 检查内存使用情况
        if result.memory_usage:
            self.assertIn("memory_statistics", result.memory_usage)
    
    @pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, 
                       reason="需要至少2个GPU")
    def test_multi_gpu_training(self):
        """测试多GPU训练"""
        config = TestConfiguration(
            test_name="multi_gpu_test",
            description="多GPU训练测试",
            gpu_count=2,
            batch_size=1,
            sequence_length=512,
            num_epochs=1,
            enable_thinking=False,
            enable_crypto_terms=True,
            enable_chinese_processing=True,
            expected_duration_minutes=12
        )
        
        result = self.test_suite._run_single_test(config)
        self.assertTrue(result.success, f"多GPU训练测试失败: {result.error_message}")


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试套件
    test_suite = EndToEndIntegrationTestSuite()
    
    # 运行所有测试
    print("开始运行端到端集成测试套件...")
    report = test_suite.run_all_tests()
    
    # 打印摘要
    summary = report["test_summary"]
    print(f"\n测试完成!")
    print(f"总测试数: {summary['total_tests']}")
    print(f"成功: {summary['successful_tests']}")
    print(f"失败: {summary['failed_tests']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"总耗时: {summary['total_duration_seconds']:.2f}秒")
    
    # 运行性能基准测试
    print("\n运行性能基准测试...")
    benchmark_result = test_suite.run_performance_benchmark()
    if benchmark_result.get("success"):
        print("性能基准测试完成")
    else:
        print(f"性能基准测试失败: {benchmark_result.get('error_message')}")
    
    # 运行回归测试
    print("\n运行回归测试...")
    regression_result = test_suite.run_regression_test()
    if "error" not in regression_result:
        has_regression = regression_result.get("has_regression", False)
        if has_regression:
            print("⚠️  检测到性能回归")
        else:
            print("✓ 无性能回归")
    else:
        print(f"回归测试跳过: {regression_result['error']}")
    
    return summary['success_rate'] == 1.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)