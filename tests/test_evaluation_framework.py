"""
评估框架测试

测试专业领域评估框架的各个组件，包括：
- 专业准确性评估
- 中文语义评估
- 专家QA数据集管理
- 综合评估流程
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.evaluation_framework import (
    ProfessionalAccuracyEvaluator,
    ChineseSemanticEvaluator,
    ExpertQADatasetBuilder,
    ComprehensiveEvaluationFramework,
    EvaluationDimension,
    ExpertiseLevel,
    ExpertQAItem,
    EvaluationResult
)
from src.expert_qa_manager import (
    ExpertQAManager,
    ExpertProfile,
    ExpertRole,
    AnnotationTask,
    AnnotationStatus
)


@pytest.fixture
def crypto_knowledge_base():
    """创建密码学知识库"""
    return {
        "AES": {
            "category": "对称加密",
            "description": "高级加密标准",
            "security_level": "high"
        },
        "RSA": {
            "category": "非对称加密", 
            "description": "RSA公钥密码算法",
            "security_level": "high"
        },
        "MD5": {
            "category": "哈希函数",
            "description": "消息摘要算法5",
            "security_level": "low"
        },
        "SHA-256": {
            "category": "哈希函数",
            "description": "安全哈希算法256位",
            "security_level": "high"
        }
    }


@pytest.fixture
def temp_data_dir():
    """创建临时数据目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_expert():
    """创建示例专家"""
    return ExpertProfile(
        expert_id="expert_001",
        name="张教授",
        role=ExpertRole.DOMAIN_EXPERT,
        expertise_areas=["对称加密", "哈希函数"],
        experience_years=15,
        certification_level=5,
        contact_info={"email": "zhang@example.com"},
        quality_score=0.95
    )


@pytest.fixture
def sample_qa_item():
    """创建示例QA项"""
    return ExpertQAItem(
        question_id="qa_001",
        question="请解释AES加密算法的工作原理",
        context="在现代密码学应用中",
        reference_answer="AES是一种对称分组密码算法，使用128位分组长度...",
        expert_annotations={
            "key_concepts": ["AES", "对称加密", "分组密码"],
            "difficulty": "中等"
        },
        difficulty_level=ExpertiseLevel.INTERMEDIATE,
        crypto_categories=["对称加密"],
        evaluation_criteria={
            EvaluationDimension.TECHNICAL_ACCURACY: 0.9,
            EvaluationDimension.CONCEPTUAL_UNDERSTANDING: 0.8
        },
        thinking_required=True
    )


class TestProfessionalAccuracyEvaluator:
    """测试专业准确性评估器"""
    
    def test_init(self, crypto_knowledge_base):
        """测试初始化"""
        evaluator = ProfessionalAccuracyEvaluator(crypto_knowledge_base)
        assert evaluator.crypto_kb == crypto_knowledge_base
        assert evaluator.logger is not None
    
    def test_evaluate_technical_accuracy(self, crypto_knowledge_base):
        """测试技术准确性评估"""
        evaluator = ProfessionalAccuracyEvaluator(crypto_knowledge_base)
        
        # 测试正确的技术回答
        correct_answer = "AES是一种高级加密标准，属于对称加密算法，使用相同的密钥进行加密和解密"
        reference = "AES是对称加密算法的标准"
        
        score = evaluator.evaluate_technical_accuracy(correct_answer, reference)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # 正确答案应该得到较高分数
    
    def test_evaluate_technical_accuracy_with_errors(self, crypto_knowledge_base):
        """测试包含技术错误的回答"""
        evaluator = ProfessionalAccuracyEvaluator(crypto_knowledge_base)
        
        # 包含明显错误的回答
        wrong_answer = "DES是安全的加密算法，MD5适用于密码存储"
        reference = "DES已经不安全，MD5不应用于密码存储"
        
        score = evaluator.evaluate_technical_accuracy(wrong_answer, reference)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # 错误答案应该得到较低分数
    
    def test_evaluate_conceptual_understanding(self, crypto_knowledge_base):
        """测试概念理解评估"""
        evaluator = ProfessionalAccuracyEvaluator(crypto_knowledge_base)
        
        question = "请分析AES和RSA的区别"
        answer = "AES是对称加密，速度快，适合大量数据；RSA是非对称加密，安全性高，适合密钥交换"
        
        score = evaluator.evaluate_conceptual_understanding(answer, question)
        assert 0.0 <= score <= 1.0
        assert score > 0.6  # 好的概念理解应该得到较高分数
    
    def test_evaluate_practical_applicability(self, crypto_knowledge_base):
        """测试实用性评估"""
        evaluator = ProfessionalAccuracyEvaluator(crypto_knowledge_base)
        
        practical_answer = "AES在网络安全中广泛应用，实现简单，性能优秀，但需要注意密钥管理的安全性"
        
        score = evaluator.evaluate_practical_applicability(practical_answer)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # 实用的回答应该得到较高分数


class TestChineseSemanticEvaluator:
    """测试中文语义评估器"""
    
    def test_init(self):
        """测试初始化"""
        evaluator = ChineseSemanticEvaluator()
        assert evaluator.logger is not None
    
    def test_chinese_tokenize(self):
        """测试中文分词"""
        evaluator = ChineseSemanticEvaluator()
        
        text = "AES是一种对称加密算法"
        tokens = evaluator._chinese_tokenize(text)
        
        assert len(tokens) > 0
        assert "AES" in tokens
        assert "对称加密算法" in tokens or "对称" in tokens
    
    def test_evaluate_semantic_similarity(self):
        """测试语义相似性评估"""
        evaluator = ChineseSemanticEvaluator()
        
        # 相似的句子
        answer = "AES是高级加密标准，属于对称密码"
        reference = "AES是对称加密算法的标准"
        
        score = evaluator.evaluate_semantic_similarity(answer, reference)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # 相似句子应该有一定的相似性分数
    
    def test_evaluate_semantic_similarity_identical(self):
        """测试相同句子的语义相似性"""
        evaluator = ChineseSemanticEvaluator()
        
        text = "AES是对称加密算法"
        score = evaluator.evaluate_semantic_similarity(text, text)
        assert score == 1.0  # 相同句子相似性应该为1
    
    def test_evaluate_linguistic_quality(self):
        """测试语言质量评估"""
        evaluator = ChineseSemanticEvaluator()
        
        good_text = "AES加密算法是现代密码学中重要的对称加密标准，具有高安全性和良好的性能表现。"
        score = evaluator.evaluate_linguistic_quality(good_text)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # 高质量文本应该得到高分
    
    def test_evaluate_linguistic_quality_poor(self):
        """测试低质量文本的语言质量评估"""
        evaluator = ChineseSemanticEvaluator()
        
        poor_text = "AES好用"  # 过于简短，缺乏信息
        score = evaluator.evaluate_linguistic_quality(poor_text)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.8  # 低质量文本应该得到较低分数


class TestExpertQADatasetBuilder:
    """测试专家QA数据集构建器"""
    
    def test_init(self):
        """测试初始化"""
        builder = ExpertQADatasetBuilder()
        assert builder.logger is not None
        assert builder.qa_items == []
    
    def test_build_crypto_qa_dataset(self):
        """测试构建密码学QA数据集"""
        builder = ExpertQADatasetBuilder()
        
        domain_areas = ["对称加密", "非对称加密"]
        qa_items = builder.build_crypto_qa_dataset(domain_areas)
        
        assert isinstance(qa_items, list)
        # 至少应该生成一些问题
        assert len(qa_items) >= 0
    
    def test_generate_symmetric_crypto_questions(self):
        """测试生成对称加密问题"""
        builder = ExpertQADatasetBuilder()
        
        questions = builder._generate_symmetric_crypto_questions()
        assert isinstance(questions, list)
        
        if questions:  # 如果生成了问题
            question = questions[0]
            assert isinstance(question, ExpertQAItem)
            assert question.question_id is not None
            assert question.question is not None
            assert "对称加密" in question.crypto_categories


class TestComprehensiveEvaluationFramework:
    """测试综合评估框架"""
    
    def test_init(self, crypto_knowledge_base):
        """测试初始化"""
        framework = ComprehensiveEvaluationFramework(crypto_knowledge_base)
        
        assert framework.accuracy_evaluator is not None
        assert framework.semantic_evaluator is not None
        assert framework.qa_builder is not None
        assert len(framework.dimension_weights) > 0
    
    def test_evaluate_model_response(self, crypto_knowledge_base):
        """测试模型回答评估"""
        framework = ComprehensiveEvaluationFramework(crypto_knowledge_base)
        
        question = "什么是AES加密算法？"
        model_answer = "AES是高级加密标准，是一种对称分组密码算法，广泛用于数据加密"
        reference_answer = "AES是对称加密算法标准"
        
        result = framework.evaluate_model_response(question, model_answer, reference_answer)
        
        assert isinstance(result, EvaluationResult)
        assert result.model_answer == model_answer
        assert len(result.scores) > 0
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.detailed_feedback) > 0
    
    def test_batch_evaluate(self, crypto_knowledge_base):
        """测试批量评估"""
        framework = ComprehensiveEvaluationFramework(crypto_knowledge_base)
        
        qa_pairs = [
            ("什么是AES？", "AES是对称加密算法", "AES是加密标准"),
            ("RSA的特点？", "RSA是非对称加密", "RSA是公钥密码")
        ]
        
        results = framework.batch_evaluate(qa_pairs)
        
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert all(0.0 <= r.overall_score <= 1.0 for r in results)
    
    def test_generate_evaluation_report(self, crypto_knowledge_base):
        """测试生成评估报告"""
        framework = ComprehensiveEvaluationFramework(crypto_knowledge_base)
        
        # 创建模拟评估结果
        results = [
            EvaluationResult(
                question_id="test_1",
                model_answer="测试回答1",
                scores={EvaluationDimension.TECHNICAL_ACCURACY: 0.8},
                detailed_feedback={"test": "测试反馈"},
                overall_score=0.8
            ),
            EvaluationResult(
                question_id="test_2", 
                model_answer="测试回答2",
                scores={EvaluationDimension.TECHNICAL_ACCURACY: 0.6},
                detailed_feedback={"test": "测试反馈"},
                overall_score=0.6
            )
        ]
        
        report = framework.generate_evaluation_report(results)
        
        assert "evaluation_summary" in report
        assert "dimension_scores" in report
        assert "score_distribution" in report
        assert "detailed_results" in report
        assert report["evaluation_summary"]["total_questions"] == 2
        assert 0.0 <= report["evaluation_summary"]["average_overall_score"] <= 1.0


class TestExpertQAManager:
    """测试专家QA管理器"""
    
    def test_init(self, temp_data_dir):
        """测试初始化"""
        manager = ExpertQAManager(temp_data_dir)
        
        assert manager.data_dir == Path(temp_data_dir)
        assert manager.data_dir.exists()
        assert isinstance(manager.experts, dict)
        assert isinstance(manager.qa_items, dict)
        assert isinstance(manager.annotation_tasks, dict)
    
    def test_register_expert(self, temp_data_dir, sample_expert):
        """测试注册专家"""
        manager = ExpertQAManager(temp_data_dir)
        
        result = manager.register_expert(sample_expert)
        assert result is True
        assert sample_expert.expert_id in manager.experts
        assert manager.experts[sample_expert.expert_id].name == sample_expert.name
    
    def test_register_duplicate_expert(self, temp_data_dir, sample_expert):
        """测试注册重复专家"""
        manager = ExpertQAManager(temp_data_dir)
        
        # 第一次注册应该成功
        result1 = manager.register_expert(sample_expert)
        assert result1 is True
        
        # 第二次注册应该失败
        result2 = manager.register_expert(sample_expert)
        assert result2 is False
    
    def test_add_qa_item(self, temp_data_dir, sample_qa_item):
        """测试添加QA项"""
        manager = ExpertQAManager(temp_data_dir)
        
        result = manager.add_qa_item(sample_qa_item)
        assert result is True
        assert sample_qa_item.question_id in manager.qa_items
    
    def test_create_annotation_task(self, temp_data_dir, sample_expert, sample_qa_item):
        """测试创建标注任务"""
        manager = ExpertQAManager(temp_data_dir)
        
        # 先注册专家和添加QA项
        manager.register_expert(sample_expert)
        manager.add_qa_item(sample_qa_item)
        
        task_id = manager.create_annotation_task(
            sample_qa_item.question_id,
            [sample_expert.expert_id],
            deadline_days=7
        )
        
        assert task_id is not None
        assert task_id in manager.annotation_tasks
        
        task = manager.annotation_tasks[task_id]
        assert task.qa_item.question_id == sample_qa_item.question_id
        assert sample_expert.expert_id in task.assigned_experts
        assert task.status == AnnotationStatus.PENDING
    
    def test_submit_annotation(self, temp_data_dir, sample_expert, sample_qa_item):
        """测试提交标注"""
        manager = ExpertQAManager(temp_data_dir)
        
        # 准备数据
        manager.register_expert(sample_expert)
        manager.add_qa_item(sample_qa_item)
        task_id = manager.create_annotation_task(
            sample_qa_item.question_id,
            [sample_expert.expert_id]
        )
        
        # 提交标注
        annotation = {
            "scores": {
                EvaluationDimension.TECHNICAL_ACCURACY.value: 0.8,
                EvaluationDimension.CONCEPTUAL_UNDERSTANDING.value: 0.7
            },
            "feedback": "回答准确，但可以更详细"
        }
        
        result = manager.submit_annotation(task_id, sample_expert.expert_id, annotation)
        assert result is True
        
        task = manager.annotation_tasks[task_id]
        assert sample_expert.expert_id in task.annotations
        assert task.status == AnnotationStatus.COMPLETED  # 单个专家完成后状态变为已完成
    
    def test_calculate_inter_annotator_agreement(self, temp_data_dir):
        """测试计算标注者间一致性"""
        manager = ExpertQAManager(temp_data_dir)
        
        # 创建两个专家
        expert1 = ExpertProfile(
            expert_id="expert_001",
            name="专家1",
            role=ExpertRole.DOMAIN_EXPERT,
            expertise_areas=["密码学"],
            experience_years=10,
            certification_level=4,
            contact_info={}
        )
        
        expert2 = ExpertProfile(
            expert_id="expert_002", 
            name="专家2",
            role=ExpertRole.DOMAIN_EXPERT,
            expertise_areas=["密码学"],
            experience_years=12,
            certification_level=5,
            contact_info={}
        )
        
        qa_item = ExpertQAItem(
            question_id="qa_test",
            question="测试问题",
            context=None,
            reference_answer="测试答案",
            expert_annotations={},
            difficulty_level=ExpertiseLevel.INTERMEDIATE,
            crypto_categories=["测试"],
            evaluation_criteria={}
        )
        
        # 注册专家和QA项
        manager.register_expert(expert1)
        manager.register_expert(expert2)
        manager.add_qa_item(qa_item)
        
        # 创建任务
        task_id = manager.create_annotation_task(
            qa_item.question_id,
            [expert1.expert_id, expert2.expert_id]
        )
        
        # 提交相似的标注
        annotation1 = {
            "scores": {EvaluationDimension.TECHNICAL_ACCURACY.value: 0.8},
            "feedback": "好"
        }
        annotation2 = {
            "scores": {EvaluationDimension.TECHNICAL_ACCURACY.value: 0.82},
            "feedback": "很好"
        }
        
        manager.submit_annotation(task_id, expert1.expert_id, annotation1)
        manager.submit_annotation(task_id, expert2.expert_id, annotation2)
        
        # 计算一致性
        agreement = manager.calculate_inter_annotator_agreement(task_id)
        
        assert isinstance(agreement, dict)
        if EvaluationDimension.TECHNICAL_ACCURACY.value in agreement:
            # 相似分数应该有高一致性
            assert agreement[EvaluationDimension.TECHNICAL_ACCURACY.value] > 0.8
    
    def test_get_quality_control_report(self, temp_data_dir, sample_expert, sample_qa_item):
        """测试生成质量控制报告"""
        manager = ExpertQAManager(temp_data_dir)
        
        # 准备数据
        manager.register_expert(sample_expert)
        manager.add_qa_item(sample_qa_item)
        task_id = manager.create_annotation_task(
            sample_qa_item.question_id,
            [sample_expert.expert_id]
        )
        
        # 提交标注
        annotation = {
            "scores": {EvaluationDimension.TECHNICAL_ACCURACY.value: 0.9},
            "feedback": "优秀"
        }
        manager.submit_annotation(task_id, sample_expert.expert_id, annotation)
        
        # 生成报告
        report = manager.get_quality_control_report()
        
        assert "summary" in report
        assert "expert_performance" in report
        assert "recommendations" in report
        assert report["summary"]["total_tasks"] >= 1
        assert sample_expert.expert_id in report["expert_performance"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])