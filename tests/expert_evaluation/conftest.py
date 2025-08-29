"""
专家评估测试配置和共享fixtures

提供测试所需的共享配置、模拟数据和工具函数。
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

from src.expert_evaluation.config import (
    ExpertEvaluationConfig,
    EvaluationDimension,
    ExpertiseLevel,
    EvaluationMode,
    OutputFormat
)
from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult,
    BatchEvaluationResult,
    EvaluationDataset,
    DimensionScore
)


@pytest.fixture(scope="session")
def test_data_dir():
    """创建临时测试数据目录"""
    temp_dir = tempfile.mkdtemp(prefix="expert_eval_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_config():
    """创建示例配置"""
    return ExpertEvaluationConfig(
        evaluation_mode=EvaluationMode.COMPREHENSIVE,
        log_level="INFO",
        enable_caching=True
    )


@pytest.fixture
def sample_qa_item():
    """创建单个示例QA项目"""
    return QAEvaluationItem(
        question_id="test_001",
        question="什么是RSA加密算法的基本原理？",
        context="密码学基础知识学习",
        reference_answer="RSA是一种非对称加密算法，基于大数分解的数学难题。它使用一对密钥：公钥用于加密，私钥用于解密。算法的安全性依赖于分解大整数的计算复杂性。",
        model_answer="RSA算法是一种公钥加密算法，由Rivest、Shamir和Adleman在1977年提出。它基于大整数分解的数学难题，使用公钥和私钥进行加密解密操作。",
        domain_tags=["密码学", "非对称加密", "RSA"],
        difficulty_level=ExpertiseLevel.INTERMEDIATE,
        expected_concepts=["非对称加密", "公钥", "私钥", "大数分解", "数学难题"]
    )


@pytest.fixture
def sample_qa_items():
    """创建多个示例QA项目"""
    items = []
    
    # RSA算法问题
    items.append(QAEvaluationItem(
        question_id="crypto_001",
        question="解释RSA算法的工作原理",
        context="密码学算法学习",
        reference_answer="RSA算法包括密钥生成、加密和解密三个步骤。密钥生成时选择两个大质数p和q，计算n=p*q和φ(n)=(p-1)(q-1)，选择公钥指数e和计算私钥指数d。",
        model_answer="RSA算法是非对称加密算法，使用公钥加密、私钥解密。其安全性基于大数分解难题。",
        domain_tags=["密码学", "RSA", "非对称加密"],
        difficulty_level=ExpertiseLevel.ADVANCED,
        expected_concepts=["密钥生成", "公钥", "私钥", "大质数", "模运算"]
    ))
    
    # AES算法问题
    items.append(QAEvaluationItem(
        question_id="crypto_002",
        question="AES加密算法的特点是什么？",
        context="对称加密算法比较",
        reference_answer="AES是高级加密标准，采用对称加密方式，支持128、192、256位密钥长度。它使用替换-置换网络结构，包含字节替换、行移位、列混合和轮密钥加等操作。",
        model_answer="AES是对称加密算法，使用相同密钥进行加密解密。它是当前广泛使用的加密标准，安全性高、效率好。",
        domain_tags=["密码学", "AES", "对称加密"],
        difficulty_level=ExpertiseLevel.INTERMEDIATE,
        expected_concepts=["对称加密", "密钥长度", "轮函数", "字节替换", "行移位"]
    ))
    
    # 数字签名问题
    items.append(QAEvaluationItem(
        question_id="crypto_003",
        question="数字签名的作用和实现原理",
        context="数字签名技术应用",
        reference_answer="数字签名提供身份认证、数据完整性和不可否认性。实现时使用私钥对消息哈希值进行签名，接收方用公钥验证签名的有效性。",
        model_answer="数字签名用于验证消息的真实性和完整性。发送方用私钥签名，接收方用公钥验证。",
        domain_tags=["密码学", "数字签名", "身份认证"],
        difficulty_level=ExpertiseLevel.BEGINNER,
        expected_concepts=["身份认证", "数据完整性", "不可否认性", "哈希函数", "公钥验证"]
    ))
    
    return items


@pytest.fixture
def sample_dimension_scores():
    """创建示例维度评分"""
    return {
        EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
            dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
            score=0.85,
            confidence=0.92,
            details={"method": "cosine_similarity", "embedding_model": "bert"},
            sub_scores={"lexical": 0.8, "semantic": 0.9}
        ),
        EvaluationDimension.DOMAIN_ACCURACY: DimensionScore(
            dimension=EvaluationDimension.DOMAIN_ACCURACY,
            score=0.78,
            confidence=0.88,
            details={"crypto_terms": 15, "accuracy_rate": 0.93},
            sub_scores={"terminology": 0.85, "concepts": 0.72}
        ),
        EvaluationDimension.COMPLETENESS: DimensionScore(
            dimension=EvaluationDimension.COMPLETENESS,
            score=0.72,
            confidence=0.85,
            details={"coverage_ratio": 0.8, "depth_score": 0.65},
            sub_scores={"breadth": 0.8, "depth": 0.65}
        )
    }


@pytest.fixture
def sample_evaluation_result(sample_dimension_scores):
    """创建示例评估结果"""
    return ExpertEvaluationResult(
        question_id="test_001",
        overall_score=0.78,
        dimension_scores=sample_dimension_scores,
        detailed_feedback={
            "strengths": "语义理解准确，专业术语使用恰当",
            "weaknesses": "技术深度不够，缺少实现细节",
            "suggestions": "建议增加算法实现步骤的详细说明"
        },
        improvement_suggestions=[
            "增加具体的算法实现步骤",
            "提供更多的技术细节",
            "加强实际应用场景的描述"
        ],
        confidence_intervals={
            "overall_score": (0.75, 0.81),
            "semantic_similarity": (0.82, 0.88)
        },
        statistical_significance={
            "overall_score": 0.95,
            "dimension_comparison": 0.89
        },
        processing_time=2.34
    )


@pytest.fixture
def sample_batch_result(sample_evaluation_result):
    """创建示例批量评估结果"""
    # 创建多个评估结果
    individual_results = []
    for i in range(5):
        result = ExpertEvaluationResult(
            question_id=f"batch_test_{i:03d}",
            overall_score=0.7 + i * 0.05,
            dimension_scores={},
            processing_time=1.5 + i * 0.2
        )
        individual_results.append(result)
    
    return BatchEvaluationResult(
        batch_id="test_batch_001",
        individual_results=individual_results,
        total_processing_time=12.5
    )


@pytest.fixture
def sample_dataset(sample_qa_items):
    """创建示例评估数据集"""
    return EvaluationDataset(
        dataset_id="test_dataset_001",
        name="密码学基础测试数据集",
        description="用于测试密码学基础知识评估的数据集",
        qa_items=sample_qa_items
    )


@pytest.fixture
def mock_model_service():
    """模拟模型服务"""
    mock_service = Mock()
    mock_service.load_model.return_value = True
    mock_service.is_model_loaded = True
    mock_service.model_path = "/mock/model/path"
    mock_service.generate_response.return_value = "模拟的模型回答内容"
    mock_service.get_model_info.return_value = {
        "model_name": "test_model",
        "model_size": "7B",
        "model_type": "llama"
    }
    return mock_service


@pytest.fixture
def mock_evaluation_framework():
    """模拟评估框架"""
    mock_framework = Mock()
    mock_framework.evaluate_response.return_value = {
        "semantic_similarity": 0.85,
        "professional_accuracy": 0.78,
        "chinese_quality": 0.82,
        "factual_correctness": 0.76
    }
    mock_framework.calculate_bleu_score.return_value = 0.65
    mock_framework.calculate_rouge_score.return_value = 0.72
    return mock_framework


@pytest.fixture
def mock_crypto_processor():
    """模拟密码学术语处理器"""
    mock_processor = Mock()
    
    # 模拟术语识别结果
    mock_terms = [
        Mock(term="RSA", category="asymmetric_encryption", confidence=0.95, complexity=3),
        Mock(term="AES", category="symmetric_encryption", confidence=0.92, complexity=2),
        Mock(term="SHA-256", category="hash_function", confidence=0.88, complexity=3),
        Mock(term="数字签名", category="digital_signature", confidence=0.90, complexity=2)
    ]
    
    mock_processor.identify_terms.return_value = mock_terms
    mock_processor.analyze_complexity.return_value = {
        "average_complexity": 2.5,
        "complexity_distribution": {1: 0, 2: 2, 3: 2, 4: 0}
    }
    
    return mock_processor


@pytest.fixture
def mock_nlp_processor():
    """模拟中文NLP处理器"""
    mock_processor = Mock()
    
    mock_processor.tokenize.return_value = ["RSA", "是", "一种", "非对称", "加密", "算法"]
    mock_processor.extract_keywords.return_value = ["RSA", "非对称加密", "算法", "密钥"]
    mock_processor.calculate_similarity.return_value = 0.85
    mock_processor.analyze_sentiment.return_value = {"polarity": 0.1, "subjectivity": 0.3}
    
    return mock_processor


@pytest.fixture
def test_json_data(test_data_dir):
    """创建测试用的JSON数据文件"""
    test_data = [
        {
            "question_id": "json_test_001",
            "question": "什么是哈希函数？",
            "context": "密码学基础",
            "reference_answer": "哈希函数是将任意长度的输入映射为固定长度输出的函数。",
            "model_answer": "哈希函数用于数据完整性验证，具有单向性和抗碰撞性。",
            "domain_tags": ["密码学", "哈希函数"],
            "difficulty_level": 2,
            "expected_concepts": ["单向性", "抗碰撞性", "固定长度"]
        },
        {
            "question_id": "json_test_002",
            "question": "解释对称加密和非对称加密的区别",
            "context": "加密算法比较",
            "reference_answer": "对称加密使用相同密钥，非对称加密使用公私钥对。",
            "model_answer": "对称加密速度快但密钥分发困难，非对称加密解决了密钥分发问题。",
            "domain_tags": ["密码学", "加密算法"],
            "difficulty_level": 1,
            "expected_concepts": ["密钥管理", "加密速度", "安全性"]
        }
    ]
    
    json_file = test_data_dir / "test_qa_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return json_file


@pytest.fixture
def performance_test_data():
    """创建性能测试数据"""
    large_qa_items = []
    
    for i in range(100):
        item = QAEvaluationItem(
            question_id=f"perf_test_{i:04d}",
            question=f"性能测试问题 {i} " + "这是一个较长的问题描述 " * 20,
            context=f"性能测试上下文 {i} " + "上下文信息 " * 10,
            reference_answer=f"性能测试参考答案 {i} " + "详细的参考答案内容 " * 50,
            model_answer=f"性能测试模型答案 {i} " + "模型生成的答案内容 " * 50,
            domain_tags=[f"标签{j}" for j in range(5)],
            difficulty_level=list(ExpertiseLevel)[i % 4],
            expected_concepts=[f"概念{j}" for j in range(10)]
        )
        large_qa_items.append(item)
    
    return large_qa_items


@pytest.fixture
def error_test_cases():
    """创建错误测试用例"""
    return {
        "invalid_qa_items": [
            # 缺少必需字段
            {
                "question_id": "",
                "question": "测试问题",
                "reference_answer": "参考答案",
                "model_answer": "模型答案"
            },
            # 空问题内容
            {
                "question_id": "test_001",
                "question": "",
                "reference_answer": "参考答案", 
                "model_answer": "模型答案"
            }
        ],
        "invalid_scores": [
            {"score": -0.1, "confidence": 0.9},  # 负分数
            {"score": 1.5, "confidence": 0.9},   # 超出范围
            {"score": 0.8, "confidence": 1.2}    # 置信度超出范围
        ],
        "malformed_data": [
            {"invalid": "data"},
            None,
            [],
            ""
        ]
    }


# 测试工具函数
def create_mock_evaluation_result(question_id: str, overall_score: float) -> ExpertEvaluationResult:
    """创建模拟评估结果的工具函数"""
    dimension_scores = {
        EvaluationDimension.SEMANTIC_SIMILARITY: DimensionScore(
            dimension=EvaluationDimension.SEMANTIC_SIMILARITY,
            score=overall_score + 0.05,
            confidence=0.9
        ),
        EvaluationDimension.DOMAIN_ACCURACY: DimensionScore(
            dimension=EvaluationDimension.DOMAIN_ACCURACY,
            score=overall_score - 0.05,
            confidence=0.85
        )
    }
    
    return ExpertEvaluationResult(
        question_id=question_id,
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        processing_time=1.0
    )


def assert_valid_score(score: float, score_name: str = "score"):
    """验证评分有效性的工具函数"""
    assert isinstance(score, (int, float)), f"{score_name} 必须是数字"
    assert 0.0 <= score <= 1.0, f"{score_name} 必须在0.0-1.0范围内，当前值: {score}"


def assert_valid_evaluation_result(result: ExpertEvaluationResult):
    """验证评估结果有效性的工具函数"""
    assert isinstance(result, ExpertEvaluationResult), "结果必须是ExpertEvaluationResult类型"
    assert result.question_id, "问题ID不能为空"
    assert_valid_score(result.overall_score, "总体评分")
    
    for dimension, score_obj in result.dimension_scores.items():
        assert isinstance(dimension, EvaluationDimension), "维度必须是EvaluationDimension类型"
        assert isinstance(score_obj, DimensionScore), "维度评分必须是DimensionScore类型"
        assert_valid_score(score_obj.score, f"{dimension.value}评分")
        assert_valid_score(score_obj.confidence, f"{dimension.value}置信度")


def create_test_config(**overrides) -> ExpertEvaluationConfig:
    """创建测试配置的工具函数"""
    default_config = ExpertEvaluationConfig(
        evaluation_mode=EvaluationMode.QUICK,
        log_level="WARNING",
        enable_caching=False
    )
    
    # 应用覆盖参数
    for key, value in overrides.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
    
    return default_config


# Pytest标记定义
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: 标记运行缓慢的测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "performance: 标记性能测试"
    )
    config.addinivalue_line(
        "markers", "unit: 标记单元测试"
    )


# 测试跳过条件
def skip_if_no_gpu():
    """如果没有GPU则跳过测试"""
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True


def skip_if_no_model():
    """如果没有模型则跳过测试"""
    # 这里可以检查模型文件是否存在
    return False  # 暂时不跳过


# 清理函数
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 在这里添加清理逻辑，如清理临时文件、重置全局状态等
    import gc
    gc.collect()  # 强制垃圾回收