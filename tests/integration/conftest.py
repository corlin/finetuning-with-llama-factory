"""
集成测试配置和共享fixtures

提供集成测试所需的配置、模拟数据和工具函数。
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch

from src.expert_evaluation.config import (
    ExpertEvaluationConfig,
    EvaluationDimension,
    ExpertiseLevel,
    EvaluationMode
)
from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    EvaluationDataset
)


@pytest.fixture(scope="session")
def integration_test_dir():
    """创建集成测试临时目录"""
    temp_dir = tempfile.mkdtemp(prefix="expert_eval_integration_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def integration_config():
    """集成测试配置"""
    from src.expert_evaluation.config import BatchProcessingConfig
    
    batch_config = BatchProcessingConfig(
        initial_batch_size=5,
        max_workers=2,
        enable_memory_monitoring=True,
        enable_auto_adjustment=True
    )
    
    return ExpertEvaluationConfig(
        evaluation_mode=EvaluationMode.COMPREHENSIVE,
        log_level="INFO",
        enable_caching=True,
        batch_processing=batch_config
    )


@pytest.fixture
def large_qa_dataset():
    """创建大规模QA数据集用于性能测试"""
    qa_items = []
    
    # 密码学基础问题
    crypto_questions = [
        ("什么是对称加密？", "对称加密使用相同的密钥进行加密和解密"),
        ("RSA算法的原理是什么？", "RSA基于大数分解的数学难题，使用公私钥对"),
        ("数字签名的作用是什么？", "数字签名提供身份认证、数据完整性和不可否认性"),
        ("哈希函数有什么特点？", "哈希函数具有单向性、确定性和抗碰撞性"),
        ("什么是椭圆曲线密码学？", "椭圆曲线密码学基于椭圆曲线离散对数问题")
    ]
    
    for i in range(100):  # 创建100个测试项目
        base_idx = i % len(crypto_questions)
        question, answer = crypto_questions[base_idx]
        
        qa_item = QAEvaluationItem(
            question_id=f"large_test_{i:04d}",
            question=f"{question} (变体 {i})",
            context=f"密码学学习上下文 {i}",
            reference_answer=f"{answer}。详细说明 {i}。",
            model_answer=f"模型回答：{answer}。补充信息 {i}。",
            domain_tags=["密码学", "安全", f"标签{i%5}"],
            difficulty_level=list(ExpertiseLevel)[i % 4],
            expected_concepts=[f"概念{j}" for j in range(3)]
        )
        qa_items.append(qa_item)
    
    return qa_items


@pytest.fixture
def multi_domain_datasets():
    """创建多领域数据集用于比较测试"""
    datasets = {}
    
    # 密码学数据集
    crypto_items = []
    for i in range(20):
        item = QAEvaluationItem(
            question_id=f"crypto_{i:03d}",
            question=f"密码学问题 {i}：加密算法的安全性如何评估？",
            context="密码学安全评估",
            reference_answer=f"加密算法安全性评估包括理论分析和实际攻击测试 {i}",
            model_answer=f"评估加密算法需要考虑密钥长度、算法复杂度等因素 {i}",
            domain_tags=["密码学", "安全评估"],
            difficulty_level=ExpertiseLevel.ADVANCED
        )
        crypto_items.append(item)
    
    datasets["cryptography"] = EvaluationDataset(
        dataset_id="crypto_dataset",
        name="密码学专业数据集",
        description="专门用于测试密码学知识的数据集",
        qa_items=crypto_items
    )
    
    # 网络安全数据集
    security_items = []
    for i in range(15):
        item = QAEvaluationItem(
            question_id=f"security_{i:03d}",
            question=f"网络安全问题 {i}：如何防范网络攻击？",
            context="网络安全防护",
            reference_answer=f"网络安全防护需要多层防御策略 {i}",
            model_answer=f"防范网络攻击可以采用防火墙、入侵检测等技术 {i}",
            domain_tags=["网络安全", "防护"],
            difficulty_level=ExpertiseLevel.INTERMEDIATE
        )
        security_items.append(item)
    
    datasets["security"] = EvaluationDataset(
        dataset_id="security_dataset",
        name="网络安全数据集",
        description="网络安全相关问题数据集",
        qa_items=security_items
    )
    
    return datasets


@pytest.fixture
def mock_model_comparison_data():
    """模拟多模型比较数据"""
    models = {
        "model_a": {
            "name": "专家模型A",
            "path": "/mock/model_a",
            "expected_performance": {
                "semantic_similarity": 0.85,
                "domain_accuracy": 0.78,
                "completeness": 0.72
            }
        },
        "model_b": {
            "name": "专家模型B", 
            "path": "/mock/model_b",
            "expected_performance": {
                "semantic_similarity": 0.82,
                "domain_accuracy": 0.85,
                "completeness": 0.75
            }
        },
        "model_c": {
            "name": "基准模型C",
            "path": "/mock/model_c",
            "expected_performance": {
                "semantic_similarity": 0.75,
                "domain_accuracy": 0.70,
                "completeness": 0.68
            }
        }
    }
    return models


@pytest.fixture
def mock_existing_systems():
    """模拟现有系统组件"""
    systems = {}
    
    # 模拟评估框架
    mock_eval_framework = Mock()
    mock_eval_framework.evaluate_response.return_value = {
        "semantic_similarity": 0.85,
        "professional_accuracy": 0.78,
        "chinese_quality": 0.82
    }
    systems["evaluation_framework"] = mock_eval_framework
    
    # 模拟模型服务
    mock_model_service = Mock()
    mock_model_service.load_model.return_value = True
    mock_model_service.is_model_loaded = True
    mock_model_service.generate_response.return_value = "模拟回答"
    systems["model_service"] = mock_model_service
    
    # 模拟配置管理器
    mock_config_manager = Mock()
    mock_config_manager.get_config.return_value = {"test": "config"}
    systems["config_manager"] = mock_config_manager
    
    # 模拟中文NLP处理器
    mock_nlp_processor = Mock()
    mock_nlp_processor.preprocess_text.return_value = "预处理后的文本"
    mock_nlp_processor.calculate_similarity.return_value = 0.85
    systems["nlp_processor"] = mock_nlp_processor
    
    # 模拟密码学术语处理器
    mock_crypto_processor = Mock()
    mock_crypto_processor.extract_crypto_terms.return_value = [
        Mock(term="RSA", confidence=0.95),
        Mock(term="AES", confidence=0.92)
    ]
    systems["crypto_processor"] = mock_crypto_processor
    
    return systems


@pytest.fixture
def performance_benchmarks():
    """性能基准数据"""
    return {
        "evaluation_speed": {
            "single_item_max_time": 5.0,  # 单个项目最大评估时间（秒）
            "batch_throughput_min": 10,   # 批处理最小吞吐量（项目/分钟）
            "large_dataset_max_time": 300  # 大数据集最大处理时间（秒）
        },
        "memory_usage": {
            "max_memory_increase": 500,   # 最大内存增长（MB）
            "memory_leak_threshold": 50   # 内存泄漏阈值（MB）
        },
        "accuracy_thresholds": {
            "min_overall_score": 0.6,
            "min_semantic_similarity": 0.7,
            "min_domain_accuracy": 0.65
        }
    }


@pytest.fixture
def integration_test_data(integration_test_dir):
    """创建集成测试数据文件"""
    test_data = {
        "qa_data": [
            {
                "question_id": "integration_001",
                "question": "集成测试：解释RSA和AES的区别",
                "context": "密码学算法比较",
                "reference_answer": "RSA是非对称加密算法，AES是对称加密算法。RSA用于密钥交换和数字签名，AES用于大量数据加密。",
                "model_answer": "RSA和AES是两种不同类型的加密算法，各有其适用场景。",
                "domain_tags": ["密码学", "加密算法"],
                "difficulty_level": 2,
                "expected_concepts": ["非对称加密", "对称加密", "密钥交换"]
            }
        ],
        "evaluation_config": {
            "evaluation_mode": "comprehensive",
            "dimensions": ["semantic_similarity", "domain_accuracy", "completeness"],
            "weights": {"semantic_similarity": 0.4, "domain_accuracy": 0.4, "completeness": 0.2}
        }
    }
    
    # 保存测试数据
    data_file = integration_test_dir / "integration_test_data.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return data_file


def create_mock_evaluation_pipeline():
    """创建模拟评估流水线"""
    pipeline = Mock()
    pipeline.initialize.return_value = True
    pipeline.load_model.return_value = True
    pipeline.evaluate.return_value = Mock(overall_score=0.8)
    pipeline.generate_report.return_value = Mock(report_id="test_report")
    pipeline.cleanup.return_value = True
    return pipeline


def assert_integration_result(result, expected_min_score=0.6):
    """验证集成测试结果"""
    assert result is not None, "集成测试结果不能为空"
    assert hasattr(result, 'overall_score'), "结果必须包含总体评分"
    assert result.overall_score >= expected_min_score, f"总体评分过低: {result.overall_score}"
    assert hasattr(result, 'dimension_scores'), "结果必须包含维度评分"
    assert len(result.dimension_scores) > 0, "维度评分不能为空"


def measure_performance(func, *args, **kwargs):
    """测量函数执行性能"""
    import time
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "result": result,
        "execution_time": end_time - start_time,
        "memory_usage": final_memory - initial_memory,
        "peak_memory": final_memory
    }


def setup_engine_mocks(engine):
    """设置引擎的通用mock"""
    from unittest.mock import Mock, patch
    
    # 模拟异步模型加载
    async def mock_async_load_model(path):
        return True
    
    # 模拟异步回答生成
    async def mock_generate_answer(prompt):
        return "模拟的模型回答内容"
    
    # 模拟评估框架结果
    mock_framework_result = Mock()
    mock_framework_result.overall_score = 0.78
    mock_framework_result.scores = {
        "TECHNICAL_ACCURACY": 0.78,
        "CONCEPTUAL_UNDERSTANDING": 0.85,
        "PRACTICAL_APPLICABILITY": 0.72
    }
    mock_framework_result.detailed_feedback = {"test": "feedback"}
    
    return {
        'async_load_model': mock_async_load_model,
        'generate_answer': mock_generate_answer,
        'framework_result': mock_framework_result
    }


def mock_engine_evaluation(engine, qa_items):
    """模拟引擎评估过程"""
    from unittest.mock import patch
    
    mocks = setup_engine_mocks(engine)
    
    with patch.object(engine, '_async_load_model', side_effect=mocks['async_load_model']), \
         patch.object(engine, '_async_generate_answer', side_effect=mocks['generate_answer']), \
         patch.object(engine.evaluation_framework, 'evaluate_with_expert_integration', return_value=mocks['framework_result']):
        
        # 确保模型已加载
        if not engine.is_model_loaded:
            engine.load_model("/test/model")
        
        # 执行评估
        return engine.evaluate_model(qa_items)