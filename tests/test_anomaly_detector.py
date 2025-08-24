"""
异常检测和报告生成模块测试

测试AnomalyDetector和TrainingReportGenerator类的训练异常检测算法、
综合训练报告生成功能、可视化训练曲线和指标图表等功能。
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from src.anomaly_detector import (
    AnomalyDetector, TrainingReportGenerator, AnomalyEvent, TrainingReport,
    create_anomaly_detector, create_report_generator
)
from src.parallel_config import DistributedTrainingMetrics, CommunicationMetrics
from src.data_models import ChineseMetrics
from src.chinese_metrics_calculator import CryptoTermLearningProgress


class TestAnomalyEvent:
    """测试异常事件类"""
    
    def test_anomaly_event_initialization(self):
        """测试异常事件初始化"""
        event = AnomalyEvent(
            timestamp=datetime.now(),
            anomaly_type="gradient_explosion",
            severity="critical",
            description="梯度爆炸检测",
            affected_metrics=["gradient_norm"],
            suggested_actions=["降低学习率", "启用梯度裁剪"]
        )
        
        assert event.anomaly_type == "gradient_explosion"
        assert event.severity == "critical"
        assert len(event.affected_metrics) == 1
        assert len(event.suggested_actions) == 2
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        event = AnomalyEvent(
            timestamp=datetime.now(),
            anomaly_type="loss_spike",
            severity="high",
            description="损失突增",
            affected_metrics=["train_loss"],
            suggested_actions=["检查数据"],
            context_data={"spike_ratio": 2.5}
        )
        
        data = event.to_dict()
        
        assert data["anomaly_type"] == "loss_spike"
        assert data["severity"] == "high"
        assert data["context_data"]["spike_ratio"] == 2.5


class TestTrainingReport:
    """测试训练报告类"""
    
    def test_training_report_initialization(self):
        """测试训练报告初始化"""
        report = TrainingReport(report_id="test_report_001")
        
        assert report.report_id == "test_report_001"
        assert isinstance(report.generation_time, datetime)
        assert report.total_epochs == 0
        assert len(report.anomalies_detected) == 0
        assert len(report.optimization_recommendations) == 0
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        report = TrainingReport(
            report_id="test_report_002",
            total_epochs=5,
            total_steps=1000,
            final_train_loss=0.5
        )
        
        data = report.to_dict()
        
        assert data["report_id"] == "test_report_002"
        assert data["total_epochs"] == 5
        assert data["total_steps"] == 1000
        assert data["final_train_loss"] == 0.5


class TestAnomalyDetector:
    """测试异常检测器类"""
    
    def setup_method(self):
        """测试前设置"""
        self.detector = AnomalyDetector(
            gradient_explosion_threshold=5.0,
            loss_spike_threshold=2.0,
            convergence_patience=50,
            memory_threshold=90.0,
            temperature_threshold=80.0
        )
    
    def test_anomaly_detector_initialization(self):
        """测试异常检测器初始化"""
        assert self.detector.gradient_explosion_threshold == 5.0
        assert self.detector.loss_spike_threshold == 2.0
        assert self.detector.convergence_patience == 50
        assert len(self.detector.anomaly_history) == 0
    
    def test_detect_gradient_explosion(self):
        """测试梯度爆炸检测"""
        # 创建梯度爆炸的指标
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            gradient_norm=10.0  # 超过阈值5.0
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到梯度爆炸异常
        gradient_anomalies = [a for a in anomalies if a.anomaly_type == "gradient_explosion"]
        assert len(gradient_anomalies) == 1
        assert gradient_anomalies[0].severity == "critical"
    
    def test_detect_loss_anomalies_nan(self):
        """测试NaN损失检测"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            train_loss=float('nan')
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到无效损失异常
        loss_anomalies = [a for a in anomalies if a.anomaly_type == "invalid_loss"]
        assert len(loss_anomalies) == 1
        assert loss_anomalies[0].severity == "critical"
    
    def test_detect_loss_spike(self):
        """测试损失突增检测"""
        # 先添加一些正常的损失历史
        for i in range(10):
            normal_metrics = DistributedTrainingMetrics(
                epoch=1,
                global_step=i,
                train_loss=0.5 + i * 0.01  # 缓慢增长
            )
            self.detector.detect_anomalies(normal_metrics)
        
        # 然后添加一个突增的损失
        spike_metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=10,
            train_loss=2.0  # 突然增大
        )
        
        anomalies = self.detector.detect_anomalies(spike_metrics)
        
        # 应该检测到损失突增
        spike_anomalies = [a for a in anomalies if a.anomaly_type == "loss_spike"]
        assert len(spike_anomalies) == 1
    
    def test_detect_overfitting(self):
        """测试过拟合检测"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            train_loss=0.2,
            val_loss=0.8  # 验证损失远高于训练损失
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到过拟合
        overfitting_anomalies = [a for a in anomalies if a.anomaly_type == "overfitting"]
        assert len(overfitting_anomalies) == 1
        assert overfitting_anomalies[0].severity == "medium"
    
    def test_detect_gpu_overheating(self):
        """测试GPU过热检测"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            gpu_metrics={
                0: {"temperature": 90.0}  # 超过阈值80.0
            }
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到GPU过热
        overheating_anomalies = [a for a in anomalies if a.anomaly_type == "gpu_overheating"]
        assert len(overheating_anomalies) == 1
        assert overheating_anomalies[0].severity == "high"
    
    def test_detect_low_gpu_utilization(self):
        """测试GPU利用率过低检测"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            gpu_metrics={
                0: {"utilization": 20.0}  # 低于30%
            }
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到GPU利用率过低
        low_util_anomalies = [a for a in anomalies if a.anomaly_type == "low_gpu_utilization"]
        assert len(low_util_anomalies) == 1
        assert low_util_anomalies[0].severity == "low"
    
    def test_detect_high_memory_usage(self):
        """测试高内存使用检测"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            gpu_metrics={
                0: {"memory_usage_percent": 95.0}  # 超过阈值90.0
            }
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到高内存使用
        memory_anomalies = [a for a in anomalies if a.anomaly_type == "high_memory_usage"]
        assert len(memory_anomalies) == 1
        assert memory_anomalies[0].severity == "high"
    
    def test_detect_convergence_stagnation(self):
        """测试收敛停滞检测"""
        # 添加足够多的相似损失值来触发收敛停滞检测
        for i in range(60):  # 超过patience=50
            stagnant_metrics = DistributedTrainingMetrics(
                epoch=1,
                global_step=i,
                train_loss=0.5 + np.random.normal(0, 0.001)  # 基本不变的损失
            )
            self.detector.detect_anomalies(stagnant_metrics)
        
        # 最后一次调用应该检测到收敛停滞
        final_metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=60,
            train_loss=0.5
        )
        
        anomalies = self.detector.detect_anomalies(final_metrics)
        
        # 检查是否有收敛停滞异常
        stagnation_anomalies = [a for a in anomalies if a.anomaly_type == "convergence_stagnation"]
        assert len(stagnation_anomalies) >= 1
    
    def test_detect_performance_anomalies(self):
        """测试性能异常检测"""
        metrics = DistributedTrainingMetrics(
            epoch=1,
            global_step=100,
            throughput_tokens_per_second=50.0,  # 低吞吐量
            load_balance_score=0.5  # 低负载均衡评分
        )
        
        anomalies = self.detector.detect_anomalies(metrics)
        
        # 应该检测到低吞吐量和负载不均衡
        throughput_anomalies = [a for a in anomalies if a.anomaly_type == "low_throughput"]
        balance_anomalies = [a for a in anomalies if a.anomaly_type == "load_imbalance"]
        
        assert len(throughput_anomalies) == 1
        assert len(balance_anomalies) == 1
    
    def test_get_anomaly_summary(self):
        """测试异常摘要获取"""
        # 添加一些异常
        metrics1 = DistributedTrainingMetrics(
            epoch=1, global_step=100, gradient_norm=10.0
        )
        metrics2 = DistributedTrainingMetrics(
            epoch=1, global_step=101, train_loss=float('nan')
        )
        
        self.detector.detect_anomalies(metrics1)
        self.detector.detect_anomalies(metrics2)
        
        summary = self.detector.get_anomaly_summary()
        
        assert summary["total_anomalies"] >= 2
        assert "gradient_explosion" in summary["anomaly_types"]
        assert "invalid_loss" in summary["anomaly_types"]
        assert summary["severity_distribution"]["critical"] >= 2


class TestTrainingReportGenerator:
    """测试训练报告生成器类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = TrainingReportGenerator(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_report_generator_initialization(self):
        """测试报告生成器初始化"""
        assert self.generator.output_dir.exists()
        assert self.generator.output_dir.is_dir()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_comprehensive_report(self, mock_close, mock_savefig):
        """测试生成综合报告"""
        # 创建测试数据
        metrics_history = [
            DistributedTrainingMetrics(
                epoch=1,
                global_step=i,
                train_loss=1.0 - i * 0.01,
                val_loss=1.1 - i * 0.01,
                gpu_metrics={0: {"utilization": 80.0, "memory_usage_percent": 70.0}}
            )
            for i in range(10)
        ]
        
        chinese_metrics_history = [
            ChineseMetrics(
                character_accuracy=0.8 + i * 0.01,
                word_accuracy=0.75 + i * 0.01,
                rouge_l_chinese=0.7 + i * 0.01,
                bleu_chinese=0.65 + i * 0.01,
                crypto_term_accuracy=0.6 + i * 0.01
            )
            for i in range(5)
        ]
        
        crypto_progress = CryptoTermLearningProgress()
        crypto_progress.total_terms_encountered = 20
        crypto_progress.correctly_used_terms = 16
        
        anomalies = [
            AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type="test_anomaly",
                severity="medium",
                description="测试异常",
                affected_metrics=["test_metric"],
                suggested_actions=["测试操作"]
            )
        ]
        
        start_time = datetime.now() - timedelta(hours=2)
        end_time = datetime.now()
        
        # 生成报告
        report = self.generator.generate_comprehensive_report(
            metrics_history=metrics_history,
            chinese_metrics_history=chinese_metrics_history,
            crypto_progress=crypto_progress,
            anomalies=anomalies,
            training_start_time=start_time,
            training_end_time=end_time
        )
        
        # 验证报告内容
        assert isinstance(report, TrainingReport)
        assert report.total_epochs == 1
        assert report.total_steps == 9
        assert report.training_duration == end_time - start_time
        assert len(report.anomalies_detected) == 1
        assert len(report.chinese_metrics_summary) > 0
        assert len(report.optimization_recommendations) > 0
        
        # 验证文件生成
        json_files = list(Path(self.temp_dir).glob("*.json"))
        html_files = list(Path(self.temp_dir).glob("*.html"))
        
        assert len(json_files) >= 1
        assert len(html_files) >= 1


class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_create_anomaly_detector(self):
        """测试创建异常检测器的便捷函数"""
        detector = create_anomaly_detector(
            gradient_explosion_threshold=8.0,
            memory_threshold=85.0
        )
        
        assert isinstance(detector, AnomalyDetector)
        assert detector.gradient_explosion_threshold == 8.0
        assert detector.memory_threshold == 85.0
    
    def test_create_report_generator(self):
        """测试创建报告生成器的便捷函数"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = create_report_generator(output_dir=temp_dir)
            
            assert isinstance(generator, TrainingReportGenerator)
            assert generator.output_dir == Path(temp_dir)


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = AnomalyDetector()
        self.generator = TrainingReportGenerator(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_full_anomaly_detection_and_reporting_workflow(self, mock_close, mock_savefig):
        """测试完整的异常检测和报告生成工作流"""
        
        # 模拟训练过程中的指标收集
        metrics_history = []
        chinese_metrics_history = []
        all_anomalies = []
        
        # 模拟正常训练阶段
        for i in range(20):
            metrics = DistributedTrainingMetrics(
                epoch=1,
                global_step=i,
                train_loss=1.0 - i * 0.02,
                val_loss=1.1 - i * 0.02,
                gradient_norm=1.0 + np.random.normal(0, 0.1),
                gpu_metrics={
                    0: {
                        "utilization": 75.0 + np.random.normal(0, 5),
                        "memory_usage_percent": 60.0 + np.random.normal(0, 5),
                        "temperature": 65.0 + np.random.normal(0, 3)
                    }
                },
                throughput_tokens_per_second=800.0 + np.random.normal(0, 50),
                load_balance_score=0.85 + np.random.normal(0, 0.05)
            )
            
            # 检测异常
            anomalies = self.detector.detect_anomalies(metrics)
            all_anomalies.extend(anomalies)
            
            metrics_history.append(metrics)
        
        # 模拟异常阶段
        for i in range(20, 25):
            # 添加一些异常情况
            abnormal_metrics = DistributedTrainingMetrics(
                epoch=1,
                global_step=i,
                train_loss=2.0,  # 损失突增
                gradient_norm=15.0,  # 梯度爆炸
                gpu_metrics={
                    0: {
                        "utilization": 25.0,  # 低利用率
                        "memory_usage_percent": 95.0,  # 高内存使用
                        "temperature": 90.0  # 高温度
                    }
                },
                throughput_tokens_per_second=50.0,  # 低吞吐量
                load_balance_score=0.3  # 负载不均衡
            )
            
            anomalies = self.detector.detect_anomalies(abnormal_metrics)
            all_anomalies.extend(anomalies)
            
            metrics_history.append(abnormal_metrics)
        
        # 添加中文指标历史
        for i in range(10):
            chinese_metrics = ChineseMetrics(
                character_accuracy=0.85 + i * 0.01,
                word_accuracy=0.80 + i * 0.01,
                rouge_l_chinese=0.75 + i * 0.01,
                bleu_chinese=0.70 + i * 0.01,
                crypto_term_accuracy=0.65 + i * 0.02
            )
            chinese_metrics_history.append(chinese_metrics)
        
        # 创建密码学学习进度
        crypto_progress = CryptoTermLearningProgress()
        crypto_progress.total_terms_encountered = 50
        crypto_progress.correctly_used_terms = 40
        
        # 生成综合报告
        start_time = datetime.now() - timedelta(hours=3)
        end_time = datetime.now()
        
        report = self.generator.generate_comprehensive_report(
            metrics_history=metrics_history,
            chinese_metrics_history=chinese_metrics_history,
            crypto_progress=crypto_progress,
            anomalies=all_anomalies,
            training_start_time=start_time,
            training_end_time=end_time
        )
        
        # 验证报告质量
        assert isinstance(report, TrainingReport)
        assert report.total_steps == 24  # 0-24
        assert len(report.anomalies_detected) > 0  # 应该检测到异常
        assert len(report.optimization_recommendations) > 0
        assert len(report.warnings) >= 0
        
        # 验证异常检测效果
        anomaly_summary = self.detector.get_anomaly_summary()
        assert anomaly_summary["total_anomalies"] > 0
        
        # 验证文件生成
        output_files = list(Path(self.temp_dir).glob("*"))
        assert len(output_files) > 0  # 应该生成了一些文件
        
        # 验证JSON报告文件存在
        json_files = list(Path(self.temp_dir).glob("*.json"))
        assert len(json_files) >= 1
        
        # 验证HTML报告文件存在
        html_files = list(Path(self.temp_dir).glob("*.html"))
        assert len(html_files) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])