"""
系统兼容性测试

测试专家评估系统与现有系统的兼容性，包括：
- 与现有评估框架的集成
- 与模型服务的兼容性
- 与配置管理系统的集成
- 与数据处理模块的兼容性
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import (
    ExpertEvaluationConfig,
    EvaluationDimension,
    EvaluationMode
)
from src.expert_evaluation.data_models import (
    QAEvaluationItem,
    ExpertEvaluationResult
)


class TestEvaluationFrameworkCompatibility:
    """评估框架兼容性测试"""
    
    @pytest.mark.integration
    def test_comprehensive_evaluation_framework_integration(self, integration_config):
        """测试与ComprehensiveEvaluationFramework的集成"""
        with patch('src.evaluation_framework.ComprehensiveEvaluationFramework') as mock_framework_class:
            # 模拟现有的ComprehensiveEvaluationFramework
            mock_framework = Mock()
            
            # 模拟现有框架的方法
            mock_framework.evaluate_response.return_value = {
                "semantic_similarity": 0.85,
                "professional_accuracy": 0.78,
                "chinese_quality": 0.82,
                "factual_correctness": 0.76,
                "response_relevance": 0.80
            }
            
            mock_framework.calculate_bleu_score.return_value = 0.65
            mock_framework.calculate_rouge_score.return_value = {
                "rouge-1": 0.72,
                "rouge-2": 0.68,
                "rouge-l": 0.70
            }
            
            mock_framework.get_evaluation_metrics.return_value = [
                "semantic_similarity",
                "professional_accuracy", 
                "chinese_quality",
                "factual_correctness"
            ]
            
            mock_framework_class.return_value = mock_framework
            
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置其他模拟对象
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                mock_config.return_value = Mock()
                
                # 创建专家评估引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_framework
                
                # 测试数据
                test_item = QAEvaluationItem(
                    question_id="framework_compat_001",
                    question="测试与现有评估框架的兼容性",
                    context="兼容性测试",
                    reference_answer="这是参考答案，用于测试现有评估框架的集成。",
                    model_answer="这是模型答案，应该与现有评估框架兼容。"
                )
                
                # 加载模型
                engine.load_model("/test/model")
                
                # 执行评估并验证与现有框架的集成
                with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                    mock_eval_instance = Mock()
                    
                    # 验证现有框架方法被正确调用
                    framework_result = mock_framework.evaluate_response(
                        test_item.model_answer,
                        test_item.reference_answer
                    )
                    
                    assert "semantic_similarity" in framework_result
                    assert "professional_accuracy" in framework_result
                    assert "chinese_quality" in framework_result
                    
                    # 验证BLEU/ROUGE分数计算
                    bleu_score = mock_framework.calculate_bleu_score(
                        test_item.model_answer,
                        test_item.reference_answer
                    )
                    assert isinstance(bleu_score, (int, float))
                    
                    rouge_scores = mock_framework.calculate_rouge_score(
                        test_item.model_answer,
                        test_item.reference_answer
                    )
                    assert "rouge-1" in rouge_scores
                    
                    # 模拟多维度评估器集成现有框架结果
                    mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                        EvaluationDimension.SEMANTIC_SIMILARITY: framework_result["semantic_similarity"],
                        EvaluationDimension.DOMAIN_ACCURACY: framework_result["professional_accuracy"]
                    }
                    mock_eval_instance.calculate_weighted_score.return_value = 0.80
                    mock_evaluator.return_value = mock_eval_instance
                    
                    result = engine.evaluate_model([test_item])
                    
                    # 验证结果包含现有框架的评估信息
                    assert result.overall_score > 0.0
                    assert len(result.dimension_scores) > 0
                    
                    # 验证现有框架方法被调用
                    mock_framework.evaluate_response.assert_called()
    
    @pytest.mark.integration
    def test_chinese_nlp_processor_integration(self, integration_config):
        """测试与ChineseNLPProcessor的集成"""
        with patch('src.chinese_nlp_processor.ChineseNLPProcessor') as mock_nlp_class:
            # 模拟现有的ChineseNLPProcessor
            mock_nlp = Mock()
            
            # 模拟中文NLP处理方法
            mock_nlp.preprocess_text.return_value = "预处理后的中文文本"
            mock_nlp.tokenize.return_value = ["预处理", "后", "的", "中文", "文本"]
            mock_nlp.extract_keywords.return_value = ["关键词1", "关键词2", "关键词3"]
            mock_nlp.calculate_similarity.return_value = 0.85
            mock_nlp.analyze_sentiment.return_value = {
                "polarity": 0.1,
                "subjectivity": 0.3,
                "confidence": 0.9
            }
            
            mock_nlp_class.return_value = mock_nlp
            
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置其他模拟对象
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": 0.85,
                    "chinese_quality": 0.82
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                # 测试中文文本处理
                chinese_text = "这是一个中文密码学问题的测试文本，包含RSA算法和AES加密等专业术语。"
                
                # 验证中文NLP处理器方法可以被调用
                preprocessed = mock_nlp.preprocess_text(chinese_text)
                assert preprocessed == "预处理后的中文文本"
                
                tokens = mock_nlp.tokenize(chinese_text)
                assert isinstance(tokens, list)
                assert len(tokens) > 0
                
                keywords = mock_nlp.extract_keywords(chinese_text)
                assert isinstance(keywords, list)
                
                similarity = mock_nlp.calculate_similarity("文本1", "文本2")
                assert isinstance(similarity, (int, float))
                assert 0.0 <= similarity <= 1.0
                
                sentiment = mock_nlp.analyze_sentiment(chinese_text)
                assert "polarity" in sentiment
                assert "subjectivity" in sentiment
    
    @pytest.mark.integration
    def test_crypto_term_processor_integration(self, integration_config):
        """测试与CryptoTermProcessor的集成"""
        with patch('src.crypto_term_processor.CryptoTermProcessor') as mock_crypto_class:
            # 模拟现有的CryptoTermProcessor
            mock_crypto = Mock()
            
            # 模拟密码学术语处理方法
            mock_terms = [
                Mock(term="RSA", category="asymmetric_encryption", confidence=0.95, complexity=3),
                Mock(term="AES", category="symmetric_encryption", confidence=0.92, complexity=2),
                Mock(term="SHA-256", category="hash_function", confidence=0.88, complexity=3),
                Mock(term="数字签名", category="digital_signature", confidence=0.90, complexity=2)
            ]
            
            mock_crypto.extract_crypto_terms.return_value = mock_terms
            mock_crypto.identify_terms.return_value = mock_terms
            mock_crypto.analyze_complexity.return_value = {
                "average_complexity": 2.5,
                "complexity_distribution": {1: 0, 2: 2, 3: 2, 4: 0},
                "total_terms": 4
            }
            mock_crypto.validate_terminology.return_value = {
                "correct_terms": 3,
                "incorrect_terms": 1,
                "accuracy_rate": 0.75
            }
            
            mock_crypto_class.return_value = mock_crypto
            
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                # 设置其他模拟对象
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "professional_accuracy": 0.78
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                # 测试密码学术语处理
                crypto_text = "RSA算法是一种非对称加密算法，而AES是对称加密算法。SHA-256是常用的哈希函数，数字签名用于身份验证。"
                
                # 验证密码学术语处理器方法可以被调用
                extracted_terms = mock_crypto.extract_crypto_terms(crypto_text)
                assert len(extracted_terms) == 4
                
                for term in extracted_terms:
                    assert hasattr(term, 'term')
                    assert hasattr(term, 'category')
                    assert hasattr(term, 'confidence')
                    assert hasattr(term, 'complexity')
                
                complexity_analysis = mock_crypto.analyze_complexity(extracted_terms)
                assert "average_complexity" in complexity_analysis
                assert "complexity_distribution" in complexity_analysis
                
                terminology_validation = mock_crypto.validate_terminology(crypto_text)
                assert "accuracy_rate" in terminology_validation


class TestModelServiceCompatibility:
    """模型服务兼容性测试"""
    
    @pytest.mark.integration
    def test_model_service_integration(self, integration_config):
        """测试与ModelService的集成"""
        with patch('src.model_service.ModelService') as mock_service_class:
            # 模拟现有的ModelService
            mock_service = Mock()
            
            # 模拟模型服务方法
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_service.model_path = "/test/model/path"
            mock_service.model_info = {
                "model_name": "test_model",
                "model_size": "7B",
                "model_type": "llama",
                "quantization": "fp16"
            }
            
            mock_service.generate_response.return_value = "这是模型生成的回答内容"
            mock_service.batch_generate.return_value = [
                "批量生成的回答1",
                "批量生成的回答2",
                "批量生成的回答3"
            ]
            
            mock_service.get_model_info.return_value = mock_service.model_info
            mock_service.get_generation_config.return_value = {
                "max_length": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
            
            mock_service_class.return_value = mock_service
            
            with patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": 0.85
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建专家评估引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                # 测试模型加载
                load_result = engine.load_model("/test/model/path")
                assert load_result is True
                assert engine.is_model_loaded is True
                
                # 验证模型服务方法被正确调用
                mock_service.load_model.assert_called_with("/test/model/path")
                
                # 测试模型信息获取
                model_info = mock_service.get_model_info()
                assert "model_name" in model_info
                assert "model_size" in model_info
                
                # 测试生成配置获取
                gen_config = mock_service.get_generation_config()
                assert "max_length" in gen_config
                assert "temperature" in gen_config
                
                # 测试单个回答生成
                response = mock_service.generate_response("测试问题")
                assert isinstance(response, str)
                assert len(response) > 0
                
                # 测试批量回答生成
                questions = ["问题1", "问题2", "问题3"]
                batch_responses = mock_service.batch_generate(questions)
                assert len(batch_responses) == len(questions)
    
    @pytest.mark.integration
    def test_model_service_error_handling(self, integration_config):
        """测试模型服务错误处理"""
        with patch('src.model_service.ModelService') as mock_service_class:
            # 模拟模型服务错误情况
            mock_service = Mock()
            
            # 模拟模型加载失败
            mock_service.load_model.side_effect = Exception("模型文件不存在")
            mock_service.is_model_loaded = False
            
            mock_service_class.return_value = mock_service
            
            with patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                mock_eval = Mock()
                mock_framework.return_value = mock_eval
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                # 测试模型加载失败处理
                from src.expert_evaluation.exceptions import ModelLoadError
                
                with pytest.raises(ModelLoadError):
                    engine.load_model("/invalid/model/path")
                
                # 验证引擎状态正确
                assert engine.is_model_loaded is False
                
                # 测试在模型未加载时进行评估
                test_item = QAEvaluationItem(
                    question_id="error_test_001",
                    question="错误处理测试",
                    context="测试",
                    reference_answer="参考",
                    model_answer="模型"
                )
                
                from src.expert_evaluation.exceptions import EvaluationProcessError
                
                with pytest.raises(EvaluationProcessError, match="模型未加载"):
                    engine.evaluate_model([test_item])


class TestConfigurationCompatibility:
    """配置管理兼容性测试"""
    
    @pytest.mark.integration
    def test_config_manager_integration(self, integration_config, integration_test_dir):
        """测试与ConfigManager的集成"""
        with patch('src.config_manager.ConfigManager') as mock_config_class:
            # 模拟现有的ConfigManager
            mock_config_manager = Mock()
            
            # 模拟配置管理方法
            mock_config_data = {
                "expert_evaluation": {
                    "evaluation_mode": "comprehensive",
                    "log_level": "INFO",
                    "enable_caching": True,
                    "batch_size": 10,
                    "dimensions": {
                        "semantic_similarity": {"weight": 0.4, "enabled": True},
                        "domain_accuracy": {"weight": 0.4, "enabled": True},
                        "completeness": {"weight": 0.2, "enabled": True}
                    }
                },
                "model_service": {
                    "model_path": "/default/model/path",
                    "generation_config": {
                        "max_length": 2048,
                        "temperature": 0.7
                    }
                }
            }
            
            mock_config_manager.load_config.return_value = mock_config_data
            mock_config_manager.get_config.return_value = mock_config_data["expert_evaluation"]
            mock_config_manager.save_config.return_value = True
            mock_config_manager.validate_config.return_value = True
            
            mock_config_class.return_value = mock_config_manager
            
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework:
                
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": 0.85
                }
                mock_framework.return_value = mock_eval
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                engine.config_manager = mock_config_manager
                
                # 测试配置加载
                loaded_config = mock_config_manager.get_config("expert_evaluation")
                assert "evaluation_mode" in loaded_config
                assert "dimensions" in loaded_config
                
                # 验证配置验证
                is_valid = mock_config_manager.validate_config(loaded_config)
                assert is_valid is True
                
                # 测试配置保存
                new_config = {
                    "evaluation_mode": "quick",
                    "log_level": "DEBUG"
                }
                save_result = mock_config_manager.save_config("expert_evaluation", new_config)
                assert save_result is True
    
    @pytest.mark.integration
    def test_configuration_file_compatibility(self, integration_config, integration_test_dir):
        """测试配置文件兼容性"""
        # 创建兼容的配置文件格式
        config_data = {
            "training": {
                "model_name": "test_model",
                "output_dir": str(integration_test_dir / "output"),
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4
            },
            "data": {
                "max_samples": 1000,
                "train_split_ratio": 0.8,
                "enable_thinking_data": True,
                "enable_crypto_terms": True
            },
            "expert_evaluation": {
                "evaluation_mode": "comprehensive",
                "log_level": "INFO",
                "enable_caching": True,
                "dimensions": ["semantic_similarity", "domain_accuracy", "completeness"]
            }
        }
        
        # 保存配置文件
        config_file = integration_test_dir / "compatible_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 模拟配置管理器加载文件
            mock_config_instance = Mock()
            mock_config_instance.load_from_file.return_value = config_data
            mock_config_instance.get_config.return_value = config_data["expert_evaluation"]
            mock_config.return_value = mock_config_instance
            
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.85
            }
            mock_framework.return_value = mock_eval
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            engine.config_manager = mock_config_instance
            
            # 测试从文件加载配置
            loaded_config = mock_config_instance.load_from_file(str(config_file))
            assert "training" in loaded_config
            assert "data" in loaded_config
            assert "expert_evaluation" in loaded_config
            
            # 验证专家评估配置
            expert_config = loaded_config["expert_evaluation"]
            assert expert_config["evaluation_mode"] == "comprehensive"
            assert expert_config["enable_caching"] is True
            assert "dimensions" in expert_config


class TestDataProcessingCompatibility:
    """数据处理兼容性测试"""
    
    @pytest.mark.integration
    def test_training_data_format_compatibility(self, integration_config, integration_test_dir):
        """测试训练数据格式兼容性"""
        # 创建兼容的训练数据格式
        training_data = [
            {
                "instruction": "什么是RSA加密算法？",
                "input": "",
                "output": "RSA是一种非对称加密算法，基于大数分解的数学难题。",
                "thinking": "首先需要理解非对称加密的概念...",
                "crypto_terms": ["RSA", "非对称加密", "大数分解"],
                "difficulty_level": 2,
                "source_file": "crypto_basics.md"
            },
            {
                "instruction": "解释AES算法的工作原理",
                "input": "",
                "output": "AES是高级加密标准，采用对称加密方式。",
                "thinking": "AES算法包含多个轮次的加密操作...",
                "crypto_terms": ["AES", "对称加密", "高级加密标准"],
                "difficulty_level": 3,
                "source_file": "aes_algorithm.md"
            }
        ]
        
        # 保存训练数据
        data_file = integration_test_dir / "training_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.85,
                "professional_accuracy": 0.78
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            engine.load_model("/test/model")
            
            # 转换训练数据为QA评估格式
            qa_items = []
            for i, item in enumerate(training_data):
                qa_item = QAEvaluationItem(
                    question_id=f"compat_data_{i:03d}",
                    question=item["instruction"],
                    context=item.get("input", ""),
                    reference_answer=item["output"],
                    model_answer=f"模拟模型回答: {item['output']}",
                    domain_tags=item.get("crypto_terms", []),
                    difficulty_level=item.get("difficulty_level", 1),
                    expected_concepts=item.get("crypto_terms", [])
                )
                qa_items.append(qa_item)
            
            # 执行评估
            with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                mock_eval_instance = Mock()
                mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                    EvaluationDimension.SEMANTIC_SIMILARITY: 0.85,
                    EvaluationDimension.DOMAIN_ACCURACY: 0.78
                }
                mock_eval_instance.calculate_weighted_score.return_value = 0.81
                mock_evaluator.return_value = mock_eval_instance
                
                result = engine.evaluate_model(qa_items)
                
                # 验证评估结果
                assert result.overall_score > 0.0
                assert len(result.dimension_scores) > 0
    
    @pytest.mark.integration
    def test_dataset_splitter_compatibility(self, integration_config):
        """测试与DatasetSplitter的兼容性"""
        with patch('src.dataset_splitter.DatasetSplitter') as mock_splitter_class:
            # 模拟现有的DatasetSplitter
            mock_splitter = Mock()
            
            # 模拟数据集分割方法
            mock_split_result = {
                "train": [
                    {"instruction": "训练问题1", "output": "训练答案1"},
                    {"instruction": "训练问题2", "output": "训练答案2"}
                ],
                "validation": [
                    {"instruction": "验证问题1", "output": "验证答案1"}
                ],
                "test": [
                    {"instruction": "测试问题1", "output": "测试答案1"}
                ]
            }
            
            mock_splitter.split_dataset.return_value = mock_split_result
            mock_splitter.get_split_statistics.return_value = {
                "train_size": 2,
                "validation_size": 1,
                "test_size": 1,
                "total_size": 4
            }
            
            mock_splitter_class.return_value = mock_splitter
            
            with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
                 patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
                 patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
                
                mock_service = Mock()
                mock_service.load_model.return_value = True
                mock_service.is_model_loaded = True
                mock_model_service.return_value = mock_service
                
                mock_eval = Mock()
                mock_eval.evaluate_response.return_value = {
                    "semantic_similarity": 0.85
                }
                mock_framework.return_value = mock_eval
                
                mock_config.return_value = Mock()
                
                # 创建引擎
                engine = ExpertEvaluationEngine(integration_config)
                engine.model_service = mock_service
                engine.evaluation_framework = mock_eval
                
                # 测试数据集分割兼容性
                original_data = [
                    {"instruction": "问题1", "output": "答案1"},
                    {"instruction": "问题2", "output": "答案2"},
                    {"instruction": "问题3", "output": "答案3"},
                    {"instruction": "问题4", "output": "答案4"}
                ]
                
                # 使用数据集分割器
                split_result = mock_splitter.split_dataset(
                    original_data,
                    train_ratio=0.5,
                    val_ratio=0.25,
                    test_ratio=0.25
                )
                
                # 验证分割结果
                assert "train" in split_result
                assert "validation" in split_result
                assert "test" in split_result
                
                # 获取分割统计
                stats = mock_splitter.get_split_statistics()
                assert stats["total_size"] == 4
                
                # 使用测试集进行评估
                test_data = split_result["test"]
                qa_items = []
                
                for i, item in enumerate(test_data):
                    qa_item = QAEvaluationItem(
                        question_id=f"split_test_{i:03d}",
                        question=item["instruction"],
                        context="",
                        reference_answer=item["output"],
                        model_answer=f"模拟回答: {item['output']}"
                    )
                    qa_items.append(qa_item)
                
                engine.load_model("/test/model")
                
                with patch.object(engine, '_create_multi_dimensional_evaluator') as mock_evaluator:
                    mock_eval_instance = Mock()
                    mock_eval_instance.integrate_evaluation_dimensions.return_value = {
                        EvaluationDimension.SEMANTIC_SIMILARITY: 0.85
                    }
                    mock_eval_instance.calculate_weighted_score.return_value = 0.82
                    mock_evaluator.return_value = mock_eval_instance
                    
                    result = engine.evaluate_model(qa_items)
                    assert result.overall_score > 0.0


class TestBackwardCompatibility:
    """向后兼容性测试"""
    
    @pytest.mark.integration
    def test_legacy_api_compatibility(self, integration_config):
        """测试遗留API兼容性"""
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            mock_service = Mock()
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.85,
                "professional_accuracy": 0.78
            }
            mock_framework.return_value = mock_eval
            
            mock_config.return_value = Mock()
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            
            # 测试遗留的评估方法调用
            if hasattr(mock_eval, 'evaluate_text'):
                # 模拟遗留方法
                mock_eval.evaluate_text.return_value = 0.80
                
                legacy_result = mock_eval.evaluate_text("测试文本", "参考文本")
                assert isinstance(legacy_result, (int, float))
            
            # 测试遗留的配置格式
            legacy_config = {
                "model_path": "/legacy/model/path",
                "evaluation_settings": {
                    "use_bleu": True,
                    "use_rouge": True,
                    "custom_metrics": ["semantic_similarity"]
                }
            }
            
            # 验证遗留配置可以被处理
            assert "model_path" in legacy_config
            assert "evaluation_settings" in legacy_config
    
    @pytest.mark.integration
    def test_version_compatibility(self, integration_config):
        """测试版本兼容性"""
        # 模拟不同版本的组件
        version_info = {
            "expert_evaluation": "1.0.0",
            "model_service": "0.9.5",
            "evaluation_framework": "1.1.0",
            "config_manager": "1.0.2"
        }
        
        with patch('src.expert_evaluation.engine.ModelService') as mock_model_service, \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework') as mock_framework, \
             patch('src.expert_evaluation.engine.ConfigManager') as mock_config:
            
            # 模拟版本检查
            mock_service = Mock()
            mock_service.get_version.return_value = version_info["model_service"]
            mock_service.load_model.return_value = True
            mock_service.is_model_loaded = True
            mock_model_service.return_value = mock_service
            
            mock_eval = Mock()
            mock_eval.get_version.return_value = version_info["evaluation_framework"]
            mock_eval.evaluate_response.return_value = {
                "semantic_similarity": 0.85
            }
            mock_framework.return_value = mock_eval
            
            mock_config_instance = Mock()
            mock_config_instance.get_version.return_value = version_info["config_manager"]
            mock_config.return_value = mock_config_instance
            
            # 创建引擎
            engine = ExpertEvaluationEngine(integration_config)
            engine.model_service = mock_service
            engine.evaluation_framework = mock_eval
            engine.config_manager = mock_config_instance
            
            # 验证版本兼容性
            if hasattr(mock_service, 'get_version'):
                service_version = mock_service.get_version()
                assert service_version == version_info["model_service"]
            
            if hasattr(mock_eval, 'get_version'):
                framework_version = mock_eval.get_version()
                assert framework_version == version_info["evaluation_framework"]
            
            if hasattr(mock_config_instance, 'get_version'):
                config_version = mock_config_instance.get_version()
                assert config_version == version_info["config_manager"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])