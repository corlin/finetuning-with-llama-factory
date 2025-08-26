"""
数据模型单元测试

测试训练数据模型、深度思考数据结构、密码学术语和中文指标等核心数据类的功能。
"""

import pytest
import json
from datetime import datetime
from src.data_models import (
    TrainingExample, ThinkingExample, ThinkingStructure, CryptoTerm, 
    ChineseMetrics, ReasoningStep, DifficultyLevel, CryptoCategory,
    DataModelValidator
)


class TestReasoningStep:
    """推理步骤测试"""
    
    def test_reasoning_step_creation(self):
        """测试推理步骤创建"""
        step = ReasoningStep(
            step_number=1,
            description="分析问题",
            input_data="RSA加密算法",
            reasoning_process="首先需要理解RSA的基本原理",
            output_result="RSA是非对称加密算法",
            confidence_score=0.9
        )
        
        assert step.step_number == 1
        assert step.description == "分析问题"
        assert step.confidence_score == 0.9
    
    def test_reasoning_step_serialization(self):
        """测试推理步骤序列化"""
        step = ReasoningStep(
            step_number=1,
            description="测试步骤",
            input_data="输入",
            reasoning_process="推理",
            output_result="输出"
        )
        
        data = step.to_dict()
        assert data["step_number"] == 1
        assert data["description"] == "测试步骤"
        
        # 测试反序列化
        restored_step = ReasoningStep.from_dict(data)
        assert restored_step.step_number == step.step_number
        assert restored_step.description == step.description


class TestCryptoTerm:
    """密码学术语测试"""
    
    def test_crypto_term_creation(self):
        """测试密码学术语创建"""
        term = CryptoTerm(
            term="RSA",
            definition="一种非对称加密算法",
            category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
            complexity=7,
            aliases=["RSA算法", "RSA加密"],
            related_terms=["公钥加密", "数字签名"]
        )
        
        assert term.term == "RSA"
        assert term.category == CryptoCategory.ASYMMETRIC_ENCRYPTION
        assert term.complexity == 7
        assert "RSA算法" in term.aliases
    
    def test_crypto_term_validation(self):
        """测试密码学术语验证"""
        # 测试空术语名称
        with pytest.raises(ValueError, match="术语名称不能为空"):
            CryptoTerm(
                term="",
                definition="定义",
                category=CryptoCategory.OTHER,
                complexity=5
            )
        
        # 测试空定义
        with pytest.raises(ValueError, match="术语定义不能为空"):
            CryptoTerm(
                term="测试术语",
                definition="",
                category=CryptoCategory.OTHER,
                complexity=5
            )
        
        # 测试复杂度范围
        with pytest.raises(ValueError, match="复杂度必须在1-10之间"):
            CryptoTerm(
                term="测试术语",
                definition="定义",
                category=CryptoCategory.OTHER,
                complexity=11
            )
    
    def test_crypto_term_serialization(self):
        """测试密码学术语序列化"""
        term = CryptoTerm(
            term="AES",
            definition="高级加密标准",
            category=CryptoCategory.SYMMETRIC_ENCRYPTION,
            complexity=6
        )
        
        data = term.to_dict()
        assert data["term"] == "AES"
        assert data["category"] == "对称加密"
        
        # 测试反序列化
        restored_term = CryptoTerm.from_dict(data)
        assert restored_term.term == term.term
        assert restored_term.category == term.category


class TestChineseMetrics:
    """中文指标测试"""
    
    def test_chinese_metrics_creation(self):
        """测试中文指标创建"""
        metrics = ChineseMetrics(
            character_accuracy=0.95,
            word_accuracy=0.90,
            rouge_l_chinese=0.85,
            bleu_chinese=0.80,
            crypto_term_accuracy=0.92,
            semantic_similarity=0.88,
            fluency_score=0.87,
            coherence_score=0.89
        )
        
        assert metrics.character_accuracy == 0.95
        assert metrics.crypto_term_accuracy == 0.92
    
    def test_chinese_metrics_validation(self):
        """测试中文指标验证"""
        # 测试超出范围的值
        with pytest.raises(ValueError, match="所有指标值必须在0.0-1.0之间"):
            ChineseMetrics(
                character_accuracy=1.5,  # 超出范围
                word_accuracy=0.90,
                rouge_l_chinese=0.85,
                bleu_chinese=0.80,
                crypto_term_accuracy=0.92
            )
    
    def test_overall_score_calculation(self):
        """测试综合评分计算"""
        metrics = ChineseMetrics(
            character_accuracy=0.9,
            word_accuracy=0.9,
            rouge_l_chinese=0.9,
            bleu_chinese=0.9,
            crypto_term_accuracy=0.9,
            semantic_similarity=0.9,
            fluency_score=0.9,
            coherence_score=0.9
        )
        
        overall = metrics.overall_score()
        assert 0.0 <= overall <= 1.0
        assert abs(overall - 0.9) < 0.01  # 应该接近0.9


class TestThinkingStructure:
    """深度思考结构测试"""
    
    def test_thinking_structure_creation(self):
        """测试深度思考结构创建"""
        reasoning_steps = [
            ReasoningStep(1, "步骤1", "输入1", "推理1", "输出1"),
            ReasoningStep(2, "步骤2", "输入2", "推理2", "输出2")
        ]
        
        structure = ThinkingStructure(
            raw_thinking="<thinking>这是思考过程</thinking>",
            parsed_steps=["步骤1", "步骤2"],
            reasoning_chain=reasoning_steps,
            validation_result=True
        )
        
        assert structure.thinking_depth == 2
        assert structure.validation_result is True
        assert len(structure.reasoning_chain) == 2
    
    def test_thinking_format_validation(self):
        """测试thinking格式验证"""
        # 正确格式
        structure = ThinkingStructure(
            raw_thinking="<thinking>正确的思考过程</thinking>",
            parsed_steps=["步骤1"],
            reasoning_chain=[],
            validation_result=True
        )
        assert structure.validate_thinking_format() is True
        
        # 错误格式 - 缺少标签
        structure_invalid = ThinkingStructure(
            raw_thinking="没有thinking标签的内容",
            parsed_steps=["步骤1"],
            reasoning_chain=[],
            validation_result=False
        )
        assert structure_invalid.validate_thinking_format() is False
    
    def test_extract_thinking_content(self):
        """测试提取thinking内容"""
        structure = ThinkingStructure(
            raw_thinking="<thinking>第一段思考</thinking>其他内容<thinking>第二段思考</thinking>",
            parsed_steps=[],
            reasoning_chain=[],
            validation_result=True
        )
        
        content = structure.extract_thinking_content()
        assert len(content) == 2
        assert "第一段思考" in content
        assert "第二段思考" in content
    
    def test_thinking_structure_serialization(self):
        """测试深度思考结构序列化"""
        reasoning_steps = [
            ReasoningStep(1, "步骤1", "输入1", "推理1", "输出1")
        ]
        
        structure = ThinkingStructure(
            raw_thinking="<thinking>测试</thinking>",
            parsed_steps=["步骤1"],
            reasoning_chain=reasoning_steps,
            validation_result=True
        )
        
        data = structure.to_dict()
        assert data["raw_thinking"] == "<thinking>测试</thinking>"
        assert data["thinking_depth"] == 1
        
        # 测试反序列化
        restored_structure = ThinkingStructure.from_dict(data)
        assert restored_structure.raw_thinking == structure.raw_thinking
        assert len(restored_structure.reasoning_chain) == 1


class TestThinkingExample:
    """深度思考样例测试"""
    
    def test_thinking_example_creation(self):
        """测试深度思考样例创建"""
        example = ThinkingExample(
            instruction="解释RSA算法的工作原理",
            thinking_process="首先需要理解公钥加密的概念...",
            final_response="RSA算法是一种非对称加密算法...",
            crypto_terms=["RSA", "公钥加密", "非对称加密"],
            difficulty_level=DifficultyLevel.ADVANCED
        )
        
        assert example.instruction == "解释RSA算法的工作原理"
        assert example.difficulty_level == DifficultyLevel.ADVANCED
        assert "RSA" in example.crypto_terms
    
    def test_thinking_example_validation(self):
        """测试深度思考样例验证"""
        # 测试空指令
        with pytest.raises(ValueError, match="指令不能为空"):
            ThinkingExample(
                instruction="",
                thinking_process="思考过程",
                final_response="回答"
            )
        
        # 测试空思考过程
        with pytest.raises(ValueError, match="思考过程不能为空"):
            ThinkingExample(
                instruction="指令",
                thinking_process="",
                final_response="回答"
            )
    
    def test_validate_thinking_tags(self):
        """测试thinking标签验证"""
        example = ThinkingExample(
            instruction="测试指令",
            thinking_process="<thinking>正确的思考过程</thinking>",
            final_response="回答"
        )
        assert example.validate_thinking_tags() is True
        
        example_invalid = ThinkingExample(
            instruction="测试指令",
            thinking_process="没有标签的思考过程",
            final_response="回答"
        )
        assert example_invalid.validate_thinking_tags() is False
    
    def test_extract_reasoning_steps(self):
        """测试提取推理步骤"""
        example = ThinkingExample(
            instruction="测试指令",
            thinking_process="步骤1：分析问题\n步骤2：寻找解决方案\n步骤3：验证结果",
            final_response="回答"
        )
        
        steps = example.extract_reasoning_steps()
        assert len(steps) == 3
        assert "步骤1：分析问题" in steps
    
    def test_to_direct_training_format(self):
        """测试转换为直接训练格式"""
        example = ThinkingExample(
            instruction="测试指令",
            thinking_process="思考过程",
            final_response="最终回答"
        )
        
        # Manual conversion to direct training format
        direct_format = {
            "instruction": example.instruction,
            "input": "",
            "output": f"<thinking>\n{example.thinking_process}\n</thinking>\n\n{example.final_response}",
            "system": "你是一个专业的密码学专家，请仔细思考后回答问题。"
        }
        assert direct_format["instruction"] == "测试指令"
        assert "<thinking>" in direct_format["output"]
        assert "最终回答" in direct_format["output"]
    
    def test_thinking_example_serialization(self):
        """测试深度思考样例序列化"""
        example = ThinkingExample(
            instruction="测试指令",
            thinking_process="思考过程",
            final_response="回答",
            crypto_terms=["术语1", "术语2"]
        )
        
        data = example.to_dict()
        assert data["instruction"] == "测试指令"
        assert data["crypto_terms"] == ["术语1", "术语2"]
        
        # 测试反序列化
        restored_example = ThinkingExample.from_dict(data)
        assert restored_example.instruction == example.instruction
        assert restored_example.crypto_terms == example.crypto_terms


class TestTrainingExample:
    """训练样例测试"""
    
    def test_training_example_creation(self):
        """测试训练样例创建"""
        example = TrainingExample(
            instruction="解释AES加密",
            input="什么是AES？",
            output="AES是高级加密标准...",
            thinking="<thinking>需要解释AES的基本概念</thinking>",
            crypto_terms=["AES", "对称加密"],
            source_file="crypto_basics.md"
        )
        
        assert example.instruction == "解释AES加密"
        assert example.has_thinking() is True
        assert "AES" in example.crypto_terms
    
    def test_training_example_validation(self):
        """测试训练样例验证"""
        # 测试空指令
        with pytest.raises(ValueError, match="指令不能为空"):
            TrainingExample(
                instruction="",
                input="输入",
                output="输出"
            )
        
        # 测试空输出
        with pytest.raises(ValueError, match="输出不能为空"):
            TrainingExample(
                instruction="指令",
                input="输入",
                output=""
            )
    
    def test_validate_format(self):
        """测试格式验证"""
        # 正确格式
        example = TrainingExample(
            instruction="指令",
            input="输入",
            output="输出",
            thinking="<thinking>思考</thinking>"
        )
        assert example.validate_format() is True
        
        # 错误的thinking格式
        example_invalid = TrainingExample(
            instruction="指令",
            input="输入",
            output="输出",
            thinking="没有标签的思考"
        )
        assert example_invalid.validate_format() is False
    
    def test_to_thinking_example(self):
        """测试转换为ThinkingExample"""
        example = TrainingExample(
            instruction="指令",
            input="输入",
            output="输出",
            thinking="<thinking>思考过程</thinking>",
            crypto_terms=["术语1"]
        )
        
        thinking_example = example.to_thinking_example()
        assert thinking_example is not None
        assert thinking_example.instruction == "指令"
        assert thinking_example.thinking_process == "<thinking>思考过程</thinking>"
        
        # 测试没有thinking的情况
        example_no_thinking = TrainingExample(
            instruction="指令",
            input="输入",
            output="输出"
        )
        assert example_no_thinking.to_thinking_example() is None
    
    def test_to_direct_training_format(self):
        """测试转换为直接训练格式"""
        example = TrainingExample(
            instruction="指令",
            input="输入",
            output="输出",
            thinking="思考过程"
        )
        
        # Manual conversion to direct training format
        output = example.output
        if example.thinking:
            output = f"<thinking>\n{example.thinking}\n</thinking>\n\n{example.output}"
        
        direct_format = {
            "instruction": example.instruction,
            "input": example.input,
            "output": output,
            "system": "你是一个专业的密码学专家，请仔细思考后回答问题。"
        }
        assert direct_format["instruction"] == "指令"
        assert direct_format["input"] == "输入"
        assert "<thinking>" in direct_format["output"]
    
    def test_training_example_serialization(self):
        """测试训练样例序列化"""
        example = TrainingExample(
            instruction="指令",
            input="输入",
            output="输出",
            crypto_terms=["术语1", "术语2"],
            metadata={"key": "value"}
        )
        
        data = example.to_dict()
        assert data["instruction"] == "指令"
        assert data["metadata"]["key"] == "value"
        
        # 测试反序列化
        restored_example = TrainingExample.from_dict(data)
        assert restored_example.instruction == example.instruction
        assert restored_example.metadata == example.metadata


class TestDataModelValidator:
    """数据模型验证器测试"""
    
    def test_validate_thinking_data_valid(self):
        """测试有效thinking数据验证"""
        thinking_text = "<thinking>这是一个有效的思考过程</thinking>"
        result = DataModelValidator.validate_thinking_data(thinking_text)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_thinking_data_invalid(self):
        """测试无效thinking数据验证"""
        # 测试空内容
        result = DataModelValidator.validate_thinking_data("")
        assert result["valid"] is False
        assert "thinking内容不能为空" in result["errors"]
        
        # 测试缺少标签
        result = DataModelValidator.validate_thinking_data("没有标签的内容")
        assert result["valid"] is False
        assert "缺少<thinking>开始标签" in result["errors"]
        
        # 测试标签不平衡
        result = DataModelValidator.validate_thinking_data("<thinking>不平衡的标签")
        assert result["valid"] is False
        assert any("thinking标签不平衡" in error or "缺少</thinking>结束标签" in error for error in result["errors"])
    
    def test_validate_thinking_data_warnings(self):
        """测试thinking数据警告"""
        # 测试深度嵌套
        nested_thinking = "<thinking><thinking><thinking><thinking>深度嵌套</thinking></thinking></thinking></thinking>"
        result = DataModelValidator.validate_thinking_data(nested_thinking)
        
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "thinking嵌套深度较深" in result["warnings"][0]
    
    def test_serialize_deserialize_training_examples(self):
        """测试训练样例序列化和反序列化"""
        examples = [
            TrainingExample(
                instruction="指令1",
                input="输入1",
                output="输出1",
                crypto_terms=["术语1"]
            ),
            TrainingExample(
                instruction="指令2",
                input="输入2",
                output="输出2",
                crypto_terms=["术语2"]
            )
        ]
        
        # 序列化
        json_str = DataModelValidator.serialize_training_examples(examples)
        assert isinstance(json_str, str)
        assert "指令1" in json_str
        
        # 反序列化
        restored_examples = DataModelValidator.deserialize_training_examples(json_str)
        assert len(restored_examples) == 2
        assert restored_examples[0].instruction == "指令1"
        assert restored_examples[1].crypto_terms == ["术语2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])