"""
性能基准测试和回归测试模块

本模块实现了系统性能的基准测试和回归检测，包括：
- 训练性能基准测试
- 内存使用效率测试
- GPU利用率基准测试
- 数据处理性能测试
- 多GPU通信效率测试
- 性能回归检测和报告
"""

import os
import sys
import json
import time
import logging
import statistics
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import unittest

import torch
import torch.distributed as dist
import psutil
import numpy as np

# 导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from src.gpu_utils import GPUDetector, GPUInfo
from src.memory_manager import MemoryManager, MemorySnapshot
from src.training_monitor import TrainingMonitor
from src.distributed_training_engine import MultiGPUProcessManager
from src.chinese_nlp_processor import ChineseNLPProcessor
from src.crypto_term_processor import CryptoTermProcessor
from src.thinking_generator import ThinkingDataGenerator
from src.dataset_splitter import DatasetSplitter
from src.data_models import TrainingExample, ThinkingExample


@dataclass
class PerformanceBenchmark:
    """性能基准"""
    benchmark_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    environment_info: Dict[str, Any]
    test_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "environment_info": self.environment_info,
            "test_config": self.test_config
        }


@dataclass
class PerformanceRegression:
    """性能回归"""
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    threshold_percent: float
    is_regression: bool
    severity: str  # "minor", "major", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "change_percent": self.change_percent,
            "threshold_percent": self.threshold_percent,
            "is_regression": self.is_regression,
            "severity": self.severity
        }


class TrainingPerformanceBenchmark:
    """训练性能基准测试"""
    
    def __init__(self, output_dir: str = "benchmark_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.gpu_detector = GPUDetector()
        self.chinese_processor = ChineseNLPProcessor()
        self.crypto_processor = CryptoTermProcessor()
        self.thinking_generator = ThinkingDataGenerator()
        
        # 基准配置 - 最小样本数量以加快测试速度
        self.benchmark_configs = {
            "small_batch": {"batch_size": 1, "sequence_length": 512, "samples": 5},
            "medium_batch": {"batch_size": 2, "sequence_length": 1024, "samples": 5},
            "large_batch": {"batch_size": 4, "sequence_length": 2048, "samples": 5},
            "thinking_data": {"batch_size": 1, "sequence_length": 1024, "samples": 5, "enable_thinking": True},
            "chinese_crypto": {"batch_size": 2, "sequence_length": 1024, "samples": 5, "enable_chinese": True}
        }
    
    def run_data_processing_benchmark(self) -> Dict[str, PerformanceBenchmark]:
        """运行数据处理性能基准测试"""
        self.logger.info("开始数据处理性能基准测试")
        benchmarks = {}
        
        for config_name, config in self.benchmark_configs.items():
            self.logger.info(f"测试配置: {config_name}")
            
            # 生成测试数据
            test_data = self._generate_benchmark_data(config)
            
            # 测试数据预处理性能
            start_time = time.time()
            processed_data = self._process_benchmark_data(test_data, config)
            processing_time = time.time() - start_time
            
            # 计算性能指标
            samples_per_second = len(test_data) / processing_time if processing_time > 0 else 0
            tokens_per_second = (len(test_data) * config["sequence_length"]) / processing_time if processing_time > 0 else 0
            
            # 创建基准
            benchmarks[f"{config_name}_samples_per_sec"] = PerformanceBenchmark(
                benchmark_name="data_processing",
                metric_name=f"{config_name}_samples_per_second",
                value=samples_per_second,
                unit="samples/sec",
                timestamp=datetime.now(),
                environment_info=self._get_environment_info(),
                test_config=config
            )
            
            benchmarks[f"{config_name}_tokens_per_sec"] = PerformanceBenchmark(
                benchmark_name="data_processing",
                metric_name=f"{config_name}_tokens_per_second",
                value=tokens_per_second,
                unit="tokens/sec",
                timestamp=datetime.now(),
                environment_info=self._get_environment_info(),
                test_config=config
            )
            
            self.logger.info(f"{config_name}: {samples_per_second:.2f} samples/sec, {tokens_per_second:.2f} tokens/sec")
        
        return benchmarks
    
    def run_memory_efficiency_benchmark(self) -> Dict[str, PerformanceBenchmark]:
        """运行内存效率基准测试"""
        self.logger.info("开始内存效率基准测试")
        benchmarks = {}
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，跳过GPU内存测试")
            return benchmarks
        
        # 初始化内存管理器
        memory_manager = MemoryManager({
            "monitoring_interval": 1,
            "enable_auto_adjustment": True,
            "initial_batch_size": 4
        })
        
        memory_manager.start()
        
        try:
            for config_name, config in self.benchmark_configs.items():
                self.logger.info(f"内存测试配置: {config_name}")
                
                # 模拟内存使用
                memory_snapshots = []
                
                # 创建测试张量模拟训练
                batch_size = config["batch_size"]
                seq_length = config["sequence_length"]
                
                test_tensors = []
                for i in range(10):  # 模拟10个训练步骤
                    # 创建模拟的激活张量
                    tensor = torch.randn(batch_size, seq_length, 3584, device='cuda')  # Qwen3-4B hidden size
                    test_tensors.append(tensor)
                    
                    # 收集内存快照
                    snapshot = memory_manager.get_current_memory_status()
                    if snapshot:
                        memory_snapshots.append(snapshot)
                    
                    time.sleep(0.1)  # 短暂等待
                
                # 清理张量
                del test_tensors
                torch.cuda.empty_cache()
                
                # 分析内存使用
                if memory_snapshots:
                    peak_memory = max(s.allocated_memory for s in memory_snapshots)
                    avg_memory = sum(s.allocated_memory for s in memory_snapshots) / len(memory_snapshots)
                    memory_efficiency = avg_memory / peak_memory if peak_memory > 0 else 0
                    
                    benchmarks[f"{config_name}_peak_memory"] = PerformanceBenchmark(
                        benchmark_name="memory_efficiency",
                        metric_name=f"{config_name}_peak_memory_mb",
                        value=peak_memory,
                        unit="MB",
                        timestamp=datetime.now(),
                        environment_info=self._get_environment_info(),
                        test_config=config
                    )
                    
                    benchmarks[f"{config_name}_memory_efficiency"] = PerformanceBenchmark(
                        benchmark_name="memory_efficiency",
                        metric_name=f"{config_name}_memory_efficiency",
                        value=memory_efficiency,
                        unit="ratio",
                        timestamp=datetime.now(),
                        environment_info=self._get_environment_info(),
                        test_config=config
                    )
                    
                    self.logger.info(f"{config_name}: 峰值内存 {peak_memory:.2f}MB, 效率 {memory_efficiency:.3f}")
        
        finally:
            memory_manager.stop()
        
        return benchmarks
    
    def run_gpu_utilization_benchmark(self) -> Dict[str, PerformanceBenchmark]:
        """运行GPU利用率基准测试"""
        self.logger.info("开始GPU利用率基准测试")
        benchmarks = {}
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，跳过GPU利用率测试")
            return benchmarks
        
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        
        for gpu_info in gpu_infos:
            gpu_id = gpu_info.gpu_id
            self.logger.info(f"测试GPU {gpu_id}: {gpu_info.name}")
            
            # 设置设备
            torch.cuda.set_device(gpu_id)
            
            # 运行计算密集型任务
            utilization_samples = []
            start_time = time.time()
            
            for i in range(20):  # 20次采样
                # 创建计算任务
                a = torch.randn(1024, 1024, device=f'cuda:{gpu_id}')
                b = torch.randn(1024, 1024, device=f'cuda:{gpu_id}')
                
                # 执行矩阵乘法
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # 获取GPU利用率（简化版本，实际需要nvidia-ml-py）
                # 这里使用内存使用率作为代理指标
                memory_info = torch.cuda.memory_stats(gpu_id)
                allocated = memory_info.get('allocated_bytes.all.current', 0)
                reserved = memory_info.get('reserved_bytes.all.current', 0)
                utilization = allocated / gpu_info.total_memory if gpu_info.total_memory > 0 else 0
                
                utilization_samples.append(utilization)
                time.sleep(0.1)
            
            total_time = time.time() - start_time
            
            # 计算统计指标
            avg_utilization = statistics.mean(utilization_samples)
            max_utilization = max(utilization_samples)
            utilization_variance = statistics.variance(utilization_samples) if len(utilization_samples) > 1 else 0
            
            benchmarks[f"gpu_{gpu_id}_avg_utilization"] = PerformanceBenchmark(
                benchmark_name="gpu_utilization",
                metric_name=f"gpu_{gpu_id}_average_utilization",
                value=avg_utilization,
                unit="ratio",
                timestamp=datetime.now(),
                environment_info=self._get_environment_info(),
                test_config={"gpu_id": gpu_id, "test_duration": total_time}
            )
            
            benchmarks[f"gpu_{gpu_id}_max_utilization"] = PerformanceBenchmark(
                benchmark_name="gpu_utilization",
                metric_name=f"gpu_{gpu_id}_max_utilization",
                value=max_utilization,
                unit="ratio",
                timestamp=datetime.now(),
                environment_info=self._get_environment_info(),
                test_config={"gpu_id": gpu_id, "test_duration": total_time}
            )
            
            self.logger.info(f"GPU {gpu_id}: 平均利用率 {avg_utilization:.3f}, 最大利用率 {max_utilization:.3f}")
        
        return benchmarks
    
    def run_communication_benchmark(self) -> Dict[str, PerformanceBenchmark]:
        """运行多GPU通信效率基准测试"""
        self.logger.info("开始多GPU通信效率基准测试")
        benchmarks = {}
        
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count < 2:
            self.logger.warning("需要至少2个GPU进行通信测试")
            return benchmarks
        
        # 测试不同大小的张量通信
        tensor_sizes = [
            (1024, 1024),      # 4MB (float32)
            (2048, 2048),      # 16MB
            (4096, 4096),      # 64MB
        ]
        
        for size in tensor_sizes:
            size_name = f"{size[0]}x{size[1]}"
            self.logger.info(f"测试张量大小: {size_name}")
            
            # 创建测试张量
            tensor = torch.randn(size, device='cuda:0')
            tensor_size_mb = tensor.numel() * 4 / (1024 * 1024)  # float32 = 4 bytes
            
            # 测试点对点通信
            if gpu_count >= 2:
                start_time = time.time()
                for i in range(10):  # 10次测试
                    # 从GPU 0发送到GPU 1
                    tensor_copy = tensor.to('cuda:1')
                    torch.cuda.synchronize()
                    
                    # 从GPU 1发送回GPU 0
                    tensor_back = tensor_copy.to('cuda:0')
                    torch.cuda.synchronize()
                
                p2p_time = (time.time() - start_time) / 20  # 20次传输（来回10次）
                p2p_bandwidth = tensor_size_mb / p2p_time if p2p_time > 0 else 0
                
                benchmarks[f"p2p_bandwidth_{size_name}"] = PerformanceBenchmark(
                    benchmark_name="communication",
                    metric_name=f"p2p_bandwidth_{size_name}",
                    value=p2p_bandwidth,
                    unit="MB/s",
                    timestamp=datetime.now(),
                    environment_info=self._get_environment_info(),
                    test_config={"tensor_size": size, "tensor_size_mb": tensor_size_mb}
                )
                
                self.logger.info(f"P2P带宽 {size_name}: {p2p_bandwidth:.2f} MB/s")
        
        return benchmarks
    
    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Any]:
        """生成基准测试数据"""
        data = []
        samples = config.get("samples", 100)
        enable_thinking = config.get("enable_thinking", False)
        enable_chinese = config.get("enable_chinese", False)
        
        for i in range(samples):
            if enable_thinking:
                # 生成思考数据
                example = ThinkingExample(
                    instruction=f"分析密码学算法 {i}",
                    thinking_process=f"<thinking>这是第{i}个思考过程</thinking>",
                    final_response=f"这是第{i}个回答",
                    crypto_terms=["AES", "RSA", "SHA"],
                    reasoning_steps=[f"步骤{j}" for j in range(3)]
                )
            else:
                # 生成普通训练数据
                instruction = f"解释密码学概念 {i}"
                if enable_chinese:
                    instruction = f"请用中文解释密码学概念 {i}"
                
                example = TrainingExample(
                    instruction=instruction,
                    input="",
                    output=f"这是第{i}个密码学概念的解释",
                    thinking=None,
                    crypto_terms=["加密", "解密", "密钥"],
                    difficulty_level=1,
                    source_file=f"benchmark_{i}.md"
                )
            
            data.append(example)
        
        return data
    
    def _process_benchmark_data(self, data: List[Any], config: Dict[str, Any]) -> List[Any]:
        """处理基准测试数据"""
        processed_data = []
        enable_chinese = config.get("enable_chinese", False)
        
        for item in data:
            if enable_chinese and hasattr(item, 'instruction'):
                # 中文处理 - 简化处理，只做分词测试
                tokens = self.chinese_processor.segment_text(item.instruction)
                processed_instruction = item.instruction  # 保持原文本
                
                if hasattr(item, 'output'):
                    output_tokens = self.chinese_processor.segment_text(item.output)
                    processed_output = item.output  # 保持原文本
                    # 创建处理后的副本
                    if isinstance(item, TrainingExample):
                        processed_item = TrainingExample(
                            instruction=processed_instruction,
                            input=item.input,
                            output=processed_output,
                            thinking=item.thinking,
                            crypto_terms=item.crypto_terms,
                            difficulty_level=item.difficulty_level,
                            source_file=item.source_file
                        )
                    else:
                        processed_item = item
                else:
                    processed_item = item
            else:
                processed_item = item
            
            # 密码学术语处理
            if hasattr(processed_item, 'output'):
                term_annotations = self.crypto_processor.identify_crypto_terms(processed_item.output)
                crypto_terms = [ann.term for ann in term_annotations]
                if hasattr(processed_item, 'crypto_terms'):
                    processed_item.crypto_terms = crypto_terms
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        
        return {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": len(gpu_infos),
            "gpu_info": [
                {
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.name,
                    "total_memory": gpu.total_memory
                }
                for gpu in gpu_infos
            ],
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total // (1024**3),  # GB
            "timestamp": datetime.now().isoformat()
        }


class PerformanceRegressionDetector:
    """性能回归检测器"""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = Path(baseline_file)
        self.logger = logging.getLogger(__name__)
        
        # 回归阈值配置
        self.regression_thresholds = {
            "samples_per_second": -10.0,      # 10%下降
            "tokens_per_second": -10.0,       # 10%下降
            "memory_efficiency": -15.0,       # 15%下降
            "peak_memory_mb": 20.0,           # 20%增加
            "average_utilization": -15.0,     # 15%下降
            "p2p_bandwidth": -20.0            # 20%下降
        }
        
        # 严重程度分类
        self.severity_thresholds = {
            "minor": 15.0,     # 15%以内
            "major": 30.0,     # 30%以内
            "critical": 50.0   # 50%以上
        }
    
    def load_baselines(self) -> Dict[str, PerformanceBenchmark]:
        """加载性能基准"""
        if not self.baseline_file.exists():
            self.logger.warning(f"基准文件不存在: {self.baseline_file}")
            return {}
        
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
            
            baselines = {}
            for key, value in data.items():
                baselines[key] = PerformanceBenchmark(
                    benchmark_name=value["benchmark_name"],
                    metric_name=value["metric_name"],
                    value=value["value"],
                    unit=value["unit"],
                    timestamp=datetime.fromisoformat(value["timestamp"]),
                    environment_info=value["environment_info"],
                    test_config=value["test_config"]
                )
            
            self.logger.info(f"加载了{len(baselines)}个性能基准")
            return baselines
            
        except Exception as e:
            self.logger.error(f"加载基准文件失败: {e}")
            return {}
    
    def save_baselines(self, benchmarks: Dict[str, PerformanceBenchmark]):
        """保存性能基准"""
        try:
            data = {key: benchmark.to_dict() for key, benchmark in benchmarks.items()}
            
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"保存了{len(benchmarks)}个性能基准到 {self.baseline_file}")
            
        except Exception as e:
            self.logger.error(f"保存基准文件失败: {e}")
    
    def detect_regressions(self, current_benchmarks: Dict[str, PerformanceBenchmark],
                          baseline_benchmarks: Dict[str, PerformanceBenchmark]) -> List[PerformanceRegression]:
        """检测性能回归"""
        regressions = []
        
        for metric_name, current_benchmark in current_benchmarks.items():
            if metric_name not in baseline_benchmarks:
                self.logger.warning(f"未找到基准: {metric_name}")
                continue
            
            baseline_benchmark = baseline_benchmarks[metric_name]
            
            # 计算变化百分比
            current_value = current_benchmark.value
            baseline_value = baseline_benchmark.value
            
            if baseline_value == 0:
                self.logger.warning(f"基准值为0，跳过: {metric_name}")
                continue
            
            change_percent = (current_value - baseline_value) / baseline_value * 100
            
            # 确定回归阈值
            threshold = self._get_regression_threshold(metric_name)
            
            # 检查是否为回归
            is_regression = False
            if threshold < 0:  # 值应该更高（如吞吐量）
                is_regression = change_percent < threshold
            else:  # 值应该更低（如内存使用）
                is_regression = change_percent > threshold
            
            # 确定严重程度
            severity = self._determine_severity(abs(change_percent))
            
            if is_regression:
                regression = PerformanceRegression(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    change_percent=change_percent,
                    threshold_percent=threshold,
                    is_regression=True,
                    severity=severity
                )
                regressions.append(regression)
                
                self.logger.warning(f"检测到回归: {metric_name}, 变化: {change_percent:.2f}%, 严重程度: {severity}")
        
        return regressions
    
    def _get_regression_threshold(self, metric_name: str) -> float:
        """获取回归阈值"""
        # 根据指标名称匹配阈值
        for pattern, threshold in self.regression_thresholds.items():
            if pattern in metric_name.lower():
                return threshold
        
        # 默认阈值
        return -10.0
    
    def _determine_severity(self, change_percent: float) -> str:
        """确定严重程度"""
        if change_percent <= self.severity_thresholds["minor"]:
            return "minor"
        elif change_percent <= self.severity_thresholds["major"]:
            return "major"
        else:
            return "critical"
    
    def generate_regression_report(self, regressions: List[PerformanceRegression],
                                 current_benchmarks: Dict[str, PerformanceBenchmark],
                                 baseline_benchmarks: Dict[str, PerformanceBenchmark]) -> Dict[str, Any]:
        """生成回归报告"""
        # 按严重程度分类
        regressions_by_severity = {
            "critical": [r for r in regressions if r.severity == "critical"],
            "major": [r for r in regressions if r.severity == "major"],
            "minor": [r for r in regressions if r.severity == "minor"]
        }
        
        # 统计信息
        total_metrics = len(current_benchmarks)
        regression_count = len(regressions)
        regression_rate = regression_count / total_metrics if total_metrics > 0 else 0
        
        # 生成报告
        report = {
            "summary": {
                "total_metrics": total_metrics,
                "regression_count": regression_count,
                "regression_rate": regression_rate,
                "critical_regressions": len(regressions_by_severity["critical"]),
                "major_regressions": len(regressions_by_severity["major"]),
                "minor_regressions": len(regressions_by_severity["minor"])
            },
            "regressions": {
                "critical": [r.to_dict() for r in regressions_by_severity["critical"]],
                "major": [r.to_dict() for r in regressions_by_severity["major"]],
                "minor": [r.to_dict() for r in regressions_by_severity["minor"]]
            },
            "recommendations": self._generate_recommendations(regressions),
            "baseline_info": {
                "baseline_count": len(baseline_benchmarks),
                "oldest_baseline": min(b.timestamp for b in baseline_benchmarks.values()).isoformat() if baseline_benchmarks else None,
                "newest_baseline": max(b.timestamp for b in baseline_benchmarks.values()).isoformat() if baseline_benchmarks else None
            },
            "current_test_info": {
                "test_time": datetime.now().isoformat(),
                "environment": list(current_benchmarks.values())[0].environment_info if current_benchmarks else {}
            }
        }
        
        return report
    
    def _generate_recommendations(self, regressions: List[PerformanceRegression]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 按类型分析回归
        memory_regressions = [r for r in regressions if "memory" in r.metric_name.lower()]
        throughput_regressions = [r for r in regressions if any(x in r.metric_name.lower() for x in ["samples_per_second", "tokens_per_second"])]
        utilization_regressions = [r for r in regressions if "utilization" in r.metric_name.lower()]
        communication_regressions = [r for r in regressions if "bandwidth" in r.metric_name.lower()]
        
        if memory_regressions:
            recommendations.append("检测到内存相关回归，建议检查内存管理策略和批次大小设置")
        
        if throughput_regressions:
            recommendations.append("检测到吞吐量回归，建议检查数据加载和预处理性能")
        
        if utilization_regressions:
            recommendations.append("检测到GPU利用率回归，建议检查计算负载和并行策略")
        
        if communication_regressions:
            recommendations.append("检测到通信性能回归，建议检查多GPU通信配置和网络环境")
        
        # 严重程度建议
        critical_regressions = [r for r in regressions if r.severity == "critical"]
        if critical_regressions:
            recommendations.append("存在严重性能回归，建议立即调查并修复")
        
        if not recommendations:
            recommendations.append("未检测到显著性能回归，系统性能稳定")
        
        return recommendations


class PerformanceBenchmarkTestSuite:
    """性能基准测试套件"""
    
    def __init__(self, output_dir: str = "performance_benchmark_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.benchmark_runner = TrainingPerformanceBenchmark(str(self.output_dir))
        self.regression_detector = PerformanceRegressionDetector(
            str(self.output_dir / "performance_baselines.json")
        )
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整的性能基准测试套件"""
        self.logger.info("开始运行完整性能基准测试套件")
        start_time = time.time()
        
        all_benchmarks = {}
        
        # 运行各项基准测试
        test_suites = [
            ("data_processing", self.benchmark_runner.run_data_processing_benchmark),
            ("memory_efficiency", self.benchmark_runner.run_memory_efficiency_benchmark),
            ("gpu_utilization", self.benchmark_runner.run_gpu_utilization_benchmark),
            ("communication", self.benchmark_runner.run_communication_benchmark)
        ]
        
        for suite_name, test_func in test_suites:
            self.logger.info(f"运行 {suite_name} 基准测试")
            try:
                benchmarks = test_func()
                all_benchmarks.update(benchmarks)
                self.logger.info(f"{suite_name} 完成，获得 {len(benchmarks)} 个基准")
            except Exception as e:
                self.logger.error(f"{suite_name} 测试失败: {e}")
        
        total_time = time.time() - start_time
        
        # 保存基准结果
        self.regression_detector.save_baselines(all_benchmarks)
        
        # 生成基准报告
        report = {
            "benchmark_summary": {
                "total_benchmarks": len(all_benchmarks),
                "test_duration_seconds": total_time,
                "test_suites": len(test_suites),
                "timestamp": datetime.now().isoformat()
            },
            "benchmarks": {key: benchmark.to_dict() for key, benchmark in all_benchmarks.items()},
            "environment_info": self.benchmark_runner._get_environment_info()
        }
        
        # 保存报告
        report_file = self.output_dir / "benchmark_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"性能基准测试完成，耗时 {total_time:.2f}秒，报告保存至 {report_file}")
        
        return report
    
    def run_regression_test(self) -> Dict[str, Any]:
        """运行回归测试"""
        self.logger.info("开始运行性能回归测试")
        
        # 加载历史基准
        baseline_benchmarks = self.regression_detector.load_baselines()
        if not baseline_benchmarks:
            self.logger.warning("未找到历史基准，无法进行回归测试")
            return {"error": "未找到历史基准"}
        
        # 运行当前基准测试
        current_report = self.run_full_benchmark_suite()
        current_benchmarks = {}
        
        for key, benchmark_dict in current_report["benchmarks"].items():
            current_benchmarks[key] = PerformanceBenchmark(
                benchmark_name=benchmark_dict["benchmark_name"],
                metric_name=benchmark_dict["metric_name"],
                value=benchmark_dict["value"],
                unit=benchmark_dict["unit"],
                timestamp=datetime.fromisoformat(benchmark_dict["timestamp"]),
                environment_info=benchmark_dict["environment_info"],
                test_config=benchmark_dict["test_config"]
            )
        
        # 检测回归
        regressions = self.regression_detector.detect_regressions(
            current_benchmarks, baseline_benchmarks
        )
        
        # 生成回归报告
        regression_report = self.regression_detector.generate_regression_report(
            regressions, current_benchmarks, baseline_benchmarks
        )
        
        # 保存回归报告
        regression_file = self.output_dir / "regression_report.json"
        with open(regression_file, 'w') as f:
            json.dump(regression_report, f, indent=2)
        
        self.logger.info(f"回归测试完成，发现 {len(regressions)} 个回归，报告保存至 {regression_file}")
        
        return regression_report


# 测试用例
class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类设置"""
        cls.benchmark_suite = PerformanceBenchmarkTestSuite("test_benchmark_output")
    
    def test_data_processing_benchmark(self):
        """测试数据处理性能基准"""
        benchmarks = self.benchmark_suite.benchmark_runner.run_data_processing_benchmark()
        self.assertGreater(len(benchmarks), 0, "应该生成数据处理基准")
        
        # 检查基准指标
        for benchmark in benchmarks.values():
            self.assertGreater(benchmark.value, 0, f"基准值应该大于0: {benchmark.metric_name}")
    
    def test_memory_efficiency_benchmark(self):
        """测试内存效率基准"""
        if not torch.cuda.is_available():
            self.skipTest("需要CUDA支持")
        
        benchmarks = self.benchmark_suite.benchmark_runner.run_memory_efficiency_benchmark()
        
        # 检查内存基准
        memory_benchmarks = [b for b in benchmarks.values() if "memory" in b.metric_name]
        self.assertGreater(len(memory_benchmarks), 0, "应该生成内存基准")
    
    def test_gpu_utilization_benchmark(self):
        """测试GPU利用率基准"""
        if not torch.cuda.is_available():
            self.skipTest("需要CUDA支持")
        
        benchmarks = self.benchmark_suite.benchmark_runner.run_gpu_utilization_benchmark()
        
        # 检查GPU利用率基准
        gpu_benchmarks = [b for b in benchmarks.values() if "utilization" in b.metric_name]
        self.assertGreater(len(gpu_benchmarks), 0, "应该生成GPU利用率基准")
    
    def test_regression_detection(self):
        """测试回归检测"""
        # 创建模拟基准
        baseline = PerformanceBenchmark(
            benchmark_name="test",
            metric_name="test_metric",
            value=100.0,
            unit="ops/sec",
            timestamp=datetime.now() - timedelta(days=1),
            environment_info={},
            test_config={}
        )
        
        # 创建回归的当前基准
        current = PerformanceBenchmark(
            benchmark_name="test",
            metric_name="test_metric",
            value=80.0,  # 20%下降
            unit="ops/sec",
            timestamp=datetime.now(),
            environment_info={},
            test_config={}
        )
        
        # 检测回归
        regressions = self.benchmark_suite.regression_detector.detect_regressions(
            {"test_metric": current},
            {"test_metric": baseline}
        )
        
        self.assertEqual(len(regressions), 1, "应该检测到1个回归")
        self.assertTrue(regressions[0].is_regression, "应该标记为回归")
        self.assertEqual(regressions[0].change_percent, -20.0, "变化百分比应该是-20%")


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建性能基准测试套件
    benchmark_suite = PerformanceBenchmarkTestSuite()
    
    # 运行完整基准测试
    print("开始运行性能基准测试套件...")
    benchmark_report = benchmark_suite.run_full_benchmark_suite()
    
    print(f"基准测试完成，生成了 {benchmark_report['benchmark_summary']['total_benchmarks']} 个基准")
    
    # 运行回归测试
    print("\n开始运行回归测试...")
    regression_report = benchmark_suite.run_regression_test()
    
    if "error" not in regression_report:
        regression_count = regression_report["summary"]["regression_count"]
        if regression_count > 0:
            print(f"⚠️  检测到 {regression_count} 个性能回归")
            critical_count = regression_report["summary"]["critical_regressions"]
            if critical_count > 0:
                print(f"🚨 其中 {critical_count} 个为严重回归")
        else:
            print("✓ 未检测到性能回归")
    else:
        print(f"回归测试跳过: {regression_report['error']}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)