"""
系统集成验证测试

本模块验证所有系统组件的集成和协作，包括：
- 配置文件验证和兼容性检查
- 模块间接口验证
- 数据流完整性验证
- 错误处理和恢复机制验证
- 资源管理和清理验证
"""

import os
import sys
import json
import yaml
import tempfile
import shutil
import logging
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

# 导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from src.config_manager import TrainingConfig, DataConfig, SystemConfig
from src.data_models import TrainingExample, ThinkingExample, CryptoTerm, ChineseMetrics
from src.lora_config_optimizer import LoRAMemoryProfile, MultiGPULoRAConfig
from src.parallel_config import ParallelConfig, GPUTopology, CommunicationBackend
from src.gpu_utils import GPUDetector
from src.memory_manager import MemoryManager
from src.training_monitor import TrainingMonitor
from src.chinese_nlp_processor import ChineseNLPProcessor
from src.crypto_term_processor import CryptoTermProcessor
from src.thinking_generator import ThinkingGenerator
from src.dataset_splitter import DatasetSplitter
from src.evaluation_framework import ComprehensiveEvaluationFramework
from src.model_exporter import ModelExporter


@dataclass
class ValidationResult:
    """验证结果"""
    component_name: str
    test_name: str
    success: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "test_name": self.test_name,
            "success": self.success,
            "error_message": self.error_message,
            "details": self.details
        }


class SystemIntegrationValidator:
    """系统集成验证器"""
    
    def __init__(self, output_dir: str = "validation_output"):
        """初始化验证器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []
        
        # 初始化组件
        self.gpu_detector = GPUDetector()
        self.chinese_processor = ChineseNLPProcessor()
        self.crypto_processor = CryptoTermProcessor()
        self.thinking_generator = ThinkingGenerator()
        self.dataset_splitter = DatasetSplitter()
        
        self.logger.info(f"系统集成验证器初始化完成，输出目录: {self.output_dir}")
    
    def validate_configuration_compatibility(self) -> List[ValidationResult]:
        """验证配置兼容性"""
        self.logger.info("开始配置兼容性验证...")
        results = []
        
        # 测试基础配置创建
        try:
            training_config = TrainingConfig(
                output_dir=str(self.output_dir / "test_training"),
                num_train_epochs=1,
                per_device_train_batch_size=2,
                learning_rate=2e-4
            )
            
            data_config = DataConfig(
                max_samples=512,
                enable_thinking_data=True,
                enable_crypto_terms=True,
                enable_chinese_processing=True
            )
            
            system_config = SystemConfig(
                num_gpus=1,
                max_seq_length=512,
                batch_size=2
            )
            
            results.append(ValidationResult(
                component_name="configuration",
                test_name="basic_config_creation",
                success=True,
                details={
                    "training_config": training_config.__dict__,
                    "data_config": data_config.__dict__,
                    "system_config": system_config.__dict__
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="configuration",
                test_name="basic_config_creation",
                success=False,
                error_message=str(e)
            ))
        
        # 测试LoRA配置兼容性
        try:
            lora_config = LoRAMemoryProfile(
                rank=16,
                alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                dropout=0.1,
                memory_efficient=True
            )
            
            # 验证LoRA配置与训练配置的兼容性
            compatible = self._validate_lora_training_compatibility(lora_config, training_config)
            
            results.append(ValidationResult(
                component_name="configuration",
                test_name="lora_compatibility",
                success=compatible,
                details={"lora_config": lora_config.__dict__}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="configuration",
                test_name="lora_compatibility",
                success=False,
                error_message=str(e)
            ))
        
        # 测试并行配置
        try:
            gpu_count = len(self.gpu_detector.get_all_gpu_info())
            
            parallel_config = ParallelConfig(
                world_size=min(2, max(1, gpu_count)),
                master_addr="localhost",
                master_port=29500,
                communication_backend=CommunicationBackend.NCCL if gpu_count > 1 else CommunicationBackend.GLOO
            )
            
            # 验证并行配置的有效性
            valid = self._validate_parallel_config(parallel_config)
            
            results.append(ValidationResult(
                component_name="configuration",
                test_name="parallel_config_validation",
                success=valid,
                details={"parallel_config": parallel_config.__dict__}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="configuration",
                test_name="parallel_config_validation",
                success=False,
                error_message=str(e)
            ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_data_processing_pipeline(self) -> List[ValidationResult]:
        """验证数据处理流水线"""
        self.logger.info("开始数据处理流水线验证...")
        results = []
        
        # 创建测试数据
        test_data = self._create_test_data()
        
        # 测试中文处理
        try:
            chinese_texts = [
                "这是一个中文密码学测试文本。",
                "RSA算法是一种非对称加密算法。",
                "数字签名可以验证数据的完整性。"
            ]
            
            processed_texts = []
            for text in chinese_texts:
                processed = self.chinese_processor.preprocess_text(text)
                processed_texts.append(processed)
            
            # 验证处理结果
            all_processed = all(isinstance(text, str) and len(text) > 0 for text in processed_texts)
            
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="chinese_text_processing",
                success=all_processed,
                details={
                    "original_texts": chinese_texts,
                    "processed_texts": processed_texts
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="chinese_text_processing",
                success=False,
                error_message=str(e)
            ))
        
        # 测试密码学术语处理
        try:
            crypto_texts = [
                "AES是一种对称加密算法，使用相同的密钥进行加密和解密。",
                "RSA算法基于大整数分解的数学难题。",
                "SHA-256是一种安全的哈希函数。"
            ]
            
            extracted_terms = []
            for text in crypto_texts:
                terms = self.crypto_processor.extract_crypto_terms(text)
                extracted_terms.extend(terms)
            
            # 验证术语提取
            has_terms = len(extracted_terms) > 0
            valid_terms = all(isinstance(term, str) for term in extracted_terms)
            
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="crypto_term_extraction",
                success=has_terms and valid_terms,
                details={
                    "crypto_texts": crypto_texts,
                    "extracted_terms": extracted_terms
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="crypto_term_extraction",
                success=False,
                error_message=str(e)
            ))
        
        # 测试深度思考数据生成
        try:
            thinking_examples = []
            for i in range(3):
                instruction = f"分析密码学算法 {i+1}"
                thinking_points = [
                    f"首先分析算法 {i+1} 的基本原理",
                    f"然后评估算法 {i+1} 的安全性",
                    f"最后考虑算法 {i+1} 的应用场景"
                ]
                
                thinking_process = self.thinking_generator.generate_thinking_process(
                    instruction, thinking_points
                )
                
                example = ThinkingExample(
                    instruction=instruction,
                    thinking_process=thinking_process,
                    final_response=f"算法 {i+1} 的分析结果",
                    crypto_terms=["加密", "安全", "算法"],
                    reasoning_steps=thinking_points
                )
                
                thinking_examples.append(example)
            
            # 验证生成结果
            valid_examples = all(
                isinstance(ex.thinking_process, str) and len(ex.thinking_process) > 0
                for ex in thinking_examples
            )
            
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="thinking_data_generation",
                success=valid_examples,
                details={
                    "generated_examples": len(thinking_examples),
                    "sample_thinking": thinking_examples[0].thinking_process[:200] if thinking_examples else ""
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="thinking_data_generation",
                success=False,
                error_message=str(e)
            ))
        
        # 测试数据集分割
        try:
            # 创建更多测试数据用于分割
            split_test_data = []
            for i in range(20):
                example = TrainingExample(
                    instruction=f"测试指令 {i}",
                    input="",
                    output=f"测试输出 {i}",
                    thinking=None,
                    crypto_terms=["测试"],
                    difficulty_level=1,
                    source_file=f"test_{i}.md"
                )
                split_test_data.append(example)
            
            # 执行分割
            splits = self.dataset_splitter.split_dataset(
                split_test_data,
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1
            )
            
            # 验证分割结果
            total_original = len(split_test_data)
            total_split = len(splits.train_data) + len(splits.val_data) + len(splits.test_data)
            split_valid = total_original == total_split
            
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="dataset_splitting",
                success=split_valid,
                details={
                    "original_count": total_original,
                    "train_count": len(splits.train_data),
                    "val_count": len(splits.val_data),
                    "test_count": len(splits.test_data),
                    "total_split": total_split
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="data_processing",
                test_name="dataset_splitting",
                success=False,
                error_message=str(e)
            ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_resource_management(self) -> List[ValidationResult]:
        """验证资源管理"""
        self.logger.info("开始资源管理验证...")
        results = []
        
        # 测试GPU检测
        try:
            gpu_infos = self.gpu_detector.get_all_gpu_info()
            gpu_topology = self.gpu_detector.detect_gpu_topology()
            
            # 验证GPU信息
            gpu_detection_valid = True
            if gpu_infos:
                # 如果有GPU，验证信息完整性
                for gpu_info in gpu_infos:
                    if not all([
                        hasattr(gpu_info, 'gpu_id'),
                        hasattr(gpu_info, 'name'),
                        hasattr(gpu_info, 'total_memory')
                    ]):
                        gpu_detection_valid = False
                        break
            
            results.append(ValidationResult(
                component_name="resource_management",
                test_name="gpu_detection",
                success=gpu_detection_valid,
                details={
                    "gpu_count": len(gpu_infos),
                    "gpu_names": [gpu.name for gpu in gpu_infos],
                    "topology_valid": gpu_topology is not None
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="resource_management",
                test_name="gpu_detection",
                success=False,
                error_message=str(e)
            ))
        
        # 测试内存管理器
        try:
            memory_manager = MemoryManager({
                "monitoring_interval": 1,
                "enable_auto_adjustment": True,
                "initial_batch_size": 2
            })
            
            # 启动内存管理器
            start_success = memory_manager.start()
            time.sleep(2)  # 等待监控启动
            
            # 获取内存状态
            memory_status = memory_manager.get_current_memory_status()
            
            # 停止内存管理器
            stop_success = memory_manager.stop()
            
            memory_manager_valid = start_success and stop_success
            
            results.append(ValidationResult(
                component_name="resource_management",
                test_name="memory_manager",
                success=memory_manager_valid,
                details={
                    "start_success": start_success,
                    "stop_success": stop_success,
                    "memory_status_available": memory_status is not None
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="resource_management",
                test_name="memory_manager",
                success=False,
                error_message=str(e)
            ))
        
        # 测试训练监控器
        try:
            gpu_ids = list(range(min(2, len(self.gpu_detector.get_all_gpu_info()))))
            if not gpu_ids:
                gpu_ids = [0]  # 默认使用CPU
            
            training_monitor = TrainingMonitor(
                gpu_ids=gpu_ids,
                log_dir=str(self.output_dir / "monitor_logs"),
                save_interval=10
            )
            
            # 启动监控
            monitor_start = training_monitor.start_monitoring()
            time.sleep(1)  # 短暂等待
            
            # 停止监控
            monitor_stop = training_monitor.stop_monitoring()
            
            monitor_valid = monitor_start and monitor_stop
            
            results.append(ValidationResult(
                component_name="resource_management",
                test_name="training_monitor",
                success=monitor_valid,
                details={
                    "monitor_start": monitor_start,
                    "monitor_stop": monitor_stop,
                    "gpu_ids": gpu_ids
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="resource_management",
                test_name="training_monitor",
                success=False,
                error_message=str(e)
            ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_error_handling(self) -> List[ValidationResult]:
        """验证错误处理机制"""
        self.logger.info("开始错误处理机制验证...")
        results = []
        
        # 测试配置错误处理
        try:
            # 尝试创建无效配置
            error_caught = False
            try:
                invalid_config = TrainingConfig(
                    output_dir="",  # 无效的输出目录
                    num_train_epochs=-1,  # 无效的epoch数
                    per_device_train_batch_size=0,  # 无效的批次大小
                    learning_rate=-0.1  # 无效的学习率
                )
            except (ValueError, TypeError) as e:
                error_caught = True
            
            results.append(ValidationResult(
                component_name="error_handling",
                test_name="invalid_config_handling",
                success=error_caught,
                details={"error_caught": error_caught}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="error_handling",
                test_name="invalid_config_handling",
                success=False,
                error_message=str(e)
            ))
        
        # 测试数据处理错误处理
        try:
            # 测试空数据处理
            empty_data_handled = True
            try:
                empty_splits = self.dataset_splitter.split_dataset([], 0.7, 0.2, 0.1)
                # 应该返回空的分割结果而不是崩溃
                if not (hasattr(empty_splits, 'train_data') and 
                       hasattr(empty_splits, 'val_data') and 
                       hasattr(empty_splits, 'test_data')):
                    empty_data_handled = False
            except Exception:
                empty_data_handled = False
            
            results.append(ValidationResult(
                component_name="error_handling",
                test_name="empty_data_handling",
                success=empty_data_handled,
                details={"empty_data_handled": empty_data_handled}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="error_handling",
                test_name="empty_data_handling",
                success=False,
                error_message=str(e)
            ))
        
        # 测试资源清理
        try:
            # 创建临时资源
            temp_dir = self.output_dir / "temp_test"
            temp_dir.mkdir(exist_ok=True)
            
            # 创建一些临时文件
            temp_files = []
            for i in range(3):
                temp_file = temp_dir / f"temp_{i}.txt"
                temp_file.write_text(f"临时文件 {i}")
                temp_files.append(temp_file)
            
            # 验证文件创建成功
            files_created = all(f.exists() for f in temp_files)
            
            # 清理资源
            shutil.rmtree(temp_dir)
            
            # 验证清理成功
            cleanup_success = not temp_dir.exists()
            
            results.append(ValidationResult(
                component_name="error_handling",
                test_name="resource_cleanup",
                success=files_created and cleanup_success,
                details={
                    "files_created": files_created,
                    "cleanup_success": cleanup_success
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="error_handling",
                test_name="resource_cleanup",
                success=False,
                error_message=str(e)
            ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_module_interfaces(self) -> List[ValidationResult]:
        """验证模块间接口"""
        self.logger.info("开始模块接口验证...")
        results = []
        
        # 测试数据模型接口
        try:
            # 创建训练示例
            training_example = TrainingExample(
                instruction="测试指令",
                input="测试输入",
                output="测试输出",
                thinking=None,
                crypto_terms=["测试"],
                difficulty_level=1,
                source_file="test.md"
            )
            
            # 验证序列化
            example_dict = training_example.__dict__
            serializable = all(
                isinstance(v, (str, int, float, list, type(None)))
                for v in example_dict.values()
            )
            
            results.append(ValidationResult(
                component_name="module_interfaces",
                test_name="data_model_serialization",
                success=serializable,
                details={"example_dict": example_dict}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="module_interfaces",
                test_name="data_model_serialization",
                success=False,
                error_message=str(e)
            ))
        
        # 测试配置接口兼容性
        try:
            # 创建各种配置
            training_config = TrainingConfig(output_dir=str(self.output_dir))
            data_config = DataConfig()
            system_config = SystemConfig()
            
            # 验证配置可以转换为字典
            configs_serializable = all([
                isinstance(training_config.__dict__, dict),
                isinstance(data_config.__dict__, dict),
                isinstance(system_config.__dict__, dict)
            ])
            
            results.append(ValidationResult(
                component_name="module_interfaces",
                test_name="config_interface_compatibility",
                success=configs_serializable,
                details={
                    "training_config_keys": list(training_config.__dict__.keys()),
                    "data_config_keys": list(data_config.__dict__.keys()),
                    "system_config_keys": list(system_config.__dict__.keys())
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="module_interfaces",
                test_name="config_interface_compatibility",
                success=False,
                error_message=str(e)
            ))
        
        # 测试处理器接口
        try:
            test_text = "这是一个测试文本，包含AES加密算法。"
            
            # 中文处理器接口
            chinese_result = self.chinese_processor.preprocess_text(test_text)
            chinese_interface_ok = isinstance(chinese_result, str)
            
            # 密码学术语处理器接口
            crypto_terms = self.crypto_processor.extract_crypto_terms(test_text)
            crypto_interface_ok = isinstance(crypto_terms, list)
            
            # 思考生成器接口
            thinking_result = self.thinking_generator.generate_thinking_process(
                "测试指令", ["步骤1", "步骤2"]
            )
            thinking_interface_ok = isinstance(thinking_result, str)
            
            all_interfaces_ok = all([
                chinese_interface_ok,
                crypto_interface_ok,
                thinking_interface_ok
            ])
            
            results.append(ValidationResult(
                component_name="module_interfaces",
                test_name="processor_interfaces",
                success=all_interfaces_ok,
                details={
                    "chinese_interface_ok": chinese_interface_ok,
                    "crypto_interface_ok": crypto_interface_ok,
                    "thinking_interface_ok": thinking_interface_ok
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="module_interfaces",
                test_name="processor_interfaces",
                success=False,
                error_message=str(e)
            ))
        
        self.validation_results.extend(results)
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整的系统集成验证"""
        self.logger.info("开始完整系统集成验证...")
        start_time = time.time()
        
        # 清空之前的结果
        self.validation_results.clear()
        
        # 运行各项验证
        validation_suites = [
            ("configuration_compatibility", self.validate_configuration_compatibility),
            ("data_processing_pipeline", self.validate_data_processing_pipeline),
            ("resource_management", self.validate_resource_management),
            ("error_handling", self.validate_error_handling),
            ("module_interfaces", self.validate_module_interfaces)
        ]
        
        suite_results = {}
        for suite_name, validation_func in validation_suites:
            self.logger.info(f"运行验证套件: {suite_name}")
            try:
                suite_results[suite_name] = validation_func()
            except Exception as e:
                self.logger.error(f"验证套件 {suite_name} 失败: {e}")
                suite_results[suite_name] = [ValidationResult(
                    component_name=suite_name,
                    test_name="suite_execution",
                    success=False,
                    error_message=str(e)
                )]
        
        total_time = time.time() - start_time
        
        # 生成验证报告
        report = self._generate_validation_report(suite_results, total_time)
        
        # 保存报告
        self._save_validation_report(report)
        
        self.logger.info(f"系统集成验证完成，耗时 {total_time:.2f}秒")
        
        return report
    
    def _create_test_data(self) -> List[TrainingExample]:
        """创建测试数据"""
        test_data = []
        
        examples = [
            ("什么是AES加密？", "AES是一种对称加密算法。"),
            ("RSA算法的原理是什么？", "RSA基于大整数分解难题。"),
            ("数字签名的作用？", "数字签名用于验证数据完整性。")
        ]
        
        for i, (instruction, output) in enumerate(examples):
            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                thinking=None,
                crypto_terms=["加密", "算法"],
                difficulty_level=1,
                source_file=f"test_{i}.md"
            )
            test_data.append(example)
        
        return test_data
    
    def _validate_lora_training_compatibility(self, lora_config: LoRAMemoryProfile, 
                                            training_config: TrainingConfig) -> bool:
        """验证LoRA配置与训练配置的兼容性"""
        try:
            # 检查基本兼容性
            if lora_config.rank <= 0 or lora_config.alpha <= 0:
                return False
            
            if not lora_config.target_modules:
                return False
            
            if training_config.per_device_train_batch_size <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def _validate_parallel_config(self, parallel_config: ParallelConfig) -> bool:
        """验证并行配置"""
        try:
            if parallel_config.world_size <= 0:
                return False
            
            if parallel_config.master_port <= 0 or parallel_config.master_port > 65535:
                return False
            
            if not parallel_config.master_addr:
                return False
            
            return True
        except Exception:
            return False
    
    def _generate_validation_report(self, suite_results: Dict[str, List[ValidationResult]], 
                                  total_time: float) -> Dict[str, Any]:
        """生成验证报告"""
        # 统计结果
        total_tests = len(self.validation_results)
        successful_tests = sum(1 for r in self.validation_results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # 按组件分类结果
        results_by_component = {}
        for result in self.validation_results:
            component = result.component_name
            if component not in results_by_component:
                results_by_component[component] = []
            results_by_component[component].append(result)
        
        # 生成报告
        report = {
            "validation_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_time_seconds": total_time,
                "timestamp": time.time()
            },
            "results_by_suite": {
                suite_name: [r.to_dict() for r in results]
                for suite_name, results in suite_results.items()
            },
            "results_by_component": {
                component: [r.to_dict() for r in results]
                for component, results in results_by_component.items()
            },
            "failed_tests": [
                r.to_dict() for r in self.validation_results if not r.success
            ],
            "recommendations": self._generate_validation_recommendations()
        }
        
        return report
    
    def _generate_validation_recommendations(self) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        failed_results = [r for r in self.validation_results if not r.success]
        
        if not failed_results:
            recommendations.append("所有系统集成验证通过，系统组件协作正常")
            return recommendations
        
        # 按组件分析失败
        failed_by_component = {}
        for result in failed_results:
            component = result.component_name
            if component not in failed_by_component:
                failed_by_component[component] = []
            failed_by_component[component].append(result)
        
        for component, failures in failed_by_component.items():
            if component == "configuration":
                recommendations.append("配置兼容性验证失败，请检查配置参数的有效性和兼容性")
            elif component == "data_processing":
                recommendations.append("数据处理流水线验证失败，请检查数据处理模块的实现")
            elif component == "resource_management":
                recommendations.append("资源管理验证失败，请检查GPU检测和内存管理功能")
            elif component == "error_handling":
                recommendations.append("错误处理机制验证失败，请完善异常处理和资源清理")
            elif component == "module_interfaces":
                recommendations.append("模块接口验证失败，请检查模块间的接口兼容性")
        
        # 通用建议
        if len(failed_results) > len(self.validation_results) * 0.5:
            recommendations.append("超过50%的验证失败，建议全面检查系统实现")
        
        return recommendations
    
    def _save_validation_report(self, report: Dict[str, Any]):
        """保存验证报告"""
        # JSON报告
        json_file = self.output_dir / "system_integration_validation_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 文本摘要
        summary_file = self.output_dir / "validation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("系统集成验证报告\n")
            f.write("=" * 40 + "\n\n")
            
            summary = report["validation_summary"]
            f.write(f"总测试数: {summary['total_tests']}\n")
            f.write(f"成功测试: {summary['successful_tests']}\n")
            f.write(f"失败测试: {summary['failed_tests']}\n")
            f.write(f"成功率: {summary['success_rate']:.2%}\n")
            f.write(f"总耗时: {summary['total_time_seconds']:.2f}秒\n\n")
            
            if report["failed_tests"]:
                f.write("失败的测试:\n")
                f.write("-" * 20 + "\n")
                for failed_test in report["failed_tests"]:
                    f.write(f"✗ {failed_test['component_name']}.{failed_test['test_name']}\n")
                    if failed_test['error_message']:
                        f.write(f"  错误: {failed_test['error_message']}\n")
                f.write("\n")
            
            f.write("改进建议:\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report["recommendations"], 1):
                f.write(f"{i}. {rec}\n")
        
        self.logger.info(f"验证报告已保存: {json_file}")
        self.logger.info(f"验证摘要已保存: {summary_file}")


# 测试用例
class TestSystemIntegrationValidation(unittest.TestCase):
    """系统集成验证测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类设置"""
        cls.validator = SystemIntegrationValidator("test_validation_output")
    
    def test_configuration_compatibility(self):
        """测试配置兼容性验证"""
        results = self.validator.validate_configuration_compatibility()
        self.assertGreater(len(results), 0, "应该有配置兼容性验证结果")
        
        # 检查是否有成功的验证
        successful_results = [r for r in results if r.success]
        self.assertGreater(len(successful_results), 0, "应该有成功的配置验证")
    
    def test_data_processing_pipeline(self):
        """测试数据处理流水线验证"""
        results = self.validator.validate_data_processing_pipeline()
        self.assertGreater(len(results), 0, "应该有数据处理验证结果")
        
        # 检查关键验证项
        test_names = [r.test_name for r in results]
        self.assertIn("chinese_text_processing", test_names, "应该包含中文处理验证")
        self.assertIn("crypto_term_extraction", test_names, "应该包含密码学术语验证")
    
    def test_resource_management(self):
        """测试资源管理验证"""
        results = self.validator.validate_resource_management()
        self.assertGreater(len(results), 0, "应该有资源管理验证结果")
        
        # 检查GPU检测验证
        gpu_results = [r for r in results if r.test_name == "gpu_detection"]
        self.assertEqual(len(gpu_results), 1, "应该有GPU检测验证")
    
    def test_full_validation(self):
        """测试完整验证"""
        report = self.validator.run_full_validation()
        
        self.assertIn("validation_summary", report, "报告应该包含验证摘要")
        self.assertIn("results_by_suite", report, "报告应该包含套件结果")
        self.assertIn("recommendations", report, "报告应该包含建议")
        
        # 检查摘要信息
        summary = report["validation_summary"]
        self.assertGreater(summary["total_tests"], 0, "应该有测试执行")
        self.assertIsInstance(summary["success_rate"], float, "成功率应该是浮点数")


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建验证器
    validator = SystemIntegrationValidator()
    
    # 运行完整验证
    print("开始系统集成验证...")
    report = validator.run_full_validation()
    
    # 显示结果
    summary = report["validation_summary"]
    print(f"\n验证完成!")
    print(f"总测试数: {summary['total_tests']}")
    print(f"成功: {summary['successful_tests']}")
    print(f"失败: {summary['failed_tests']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"耗时: {summary['total_time_seconds']:.2f}秒")
    
    if summary['failed_tests'] > 0:
        print(f"\n失败的测试:")
        for failed_test in report["failed_tests"]:
            print(f"  ✗ {failed_test['component_name']}.{failed_test['test_name']}")
    
    return summary['success_rate'] >= 0.8  # 80%成功率


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)