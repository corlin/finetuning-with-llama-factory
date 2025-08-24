"""
深度思考数据生成器测试

测试thinking数据生成、逻辑连贯性验证、密码学推理和质量评估功能。
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thinking_generator import (
    ThinkingDataGenerator, ThinkingType, ReasoningPattern,
    ThinkingTemplate
)
from data_models import (
    ThinkingExample, ThinkingStructure, ReasoningStep,
    CryptoTerm, CryptoCategory, DifficultyLevel
)


class TestThinkingDataGenerator(unittest.TestCase):
    """思考数据生成器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.generator = ThinkingDataGenerator()
        
        # 创建测试用的密码学术语
        self.test_crypto_terms = [
            CryptoTerm(
                term="AES",
                definition="高级加密标准",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=7,
                aliases=["Advanced Encryption Standard"],
                related_terms=["DES", "3DES"],
                examples=["AES-128", "AES-256"]
            ),
            CryptoTerm(
                term="RSA",
                definition="一种非对称加密算法",
                category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
                complexity=8,
                aliases=["Rivest-Shamir-Adleman"],
                related_terms=["ECC", "DSA"],
                examples=["RSA-2048", "RSA-4096"]
            )
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.generator, ThinkingDataGenerator)
        self.assertIsInstance(self.generator.thinking_templates, dict)
        self.assertIsInstance(self.generator.reasoning_patterns, dict)
        
        # 检查模板是否正确初始化
        self.assertIn("crypto_analysis", self.generator.thinking_templates)
        self.assertIn("algorithm_comparison", self.generator.thinking_templates)
        self.assertIn("problem_solving", self.generator.thinking_templates)
        self.assertIn("concept_explanation", self.generator.thinking_templates)
    
    def test_analyze_instruction(self):
        """测试指令分析"""
        # 测试密码学相关指令
        crypto_instruction = "请解释AES加密算法的工作原理和安全性"
        analysis = self.generator._analyze_instruction(crypto_instruction)
        
        self.assertIn("crypto_concepts", analysis)
        self.assertIn("problem_core", analysis)
        self.assertTrue(len(analysis["crypto_concepts"]) > 0)
        
        # 测试概念解释指令
        concept_instruction = "什么是数字签名？"
        analysis = self.generator._analyze_instruction(concept_instruction)
        self.assertEqual(analysis["problem_core"], "概念理解和应用")
        
        # 测试比较类指令
        comparison_instruction = "比较RSA和ECC算法的优缺点"
        analysis = self.generator._analyze_instruction(comparison_instruction)
        self.assertEqual(analysis["problem_core"], "算法或概念比较")
    
    def test_generate_thinking_process(self):
        """测试thinking过程生成"""
        instruction = "解释AES加密算法的安全性"
        
        thinking_process = self.generator.generate_thinking_process(
            instruction=instruction,
            thinking_type=ThinkingType.CRYPTOGRAPHIC,
            target_length=300
        )
        
        self.assertIsInstance(thinking_process, str)
        self.assertTrue(len(thinking_process) > 50)
        self.assertIn("AES", thinking_process)
        
        # 测试不同思考类型
        analytical_thinking = self.generator.generate_thinking_process(
            instruction=instruction,
            thinking_type=ThinkingType.ANALYTICAL
        )
        
        self.assertIsInstance(analytical_thinking, str)
        self.assertTrue(len(analytical_thinking) > 0)
    
    def test_generate_crypto_reasoning(self):
        """测试密码学推理生成"""
        crypto_problem = "如何选择合适的加密算法来保护敏感数据？"
        
        thinking_structure = self.generator.generate_crypto_reasoning(
            crypto_problem=crypto_problem,
            crypto_terms=self.test_crypto_terms
        )
        
        self.assertIsInstance(thinking_structure, ThinkingStructure)
        self.assertTrue(thinking_structure.validation_result)
        self.assertTrue(len(thinking_structure.reasoning_chain) > 0)
        self.assertTrue(thinking_structure.thinking_depth > 0)
        self.assertGreater(thinking_structure.logical_consistency, 0.5)
        
        # 检查推理步骤
        for step in thinking_structure.reasoning_chain:
            self.assertIsInstance(step, ReasoningStep)
            self.assertTrue(step.step_number > 0)
            self.assertTrue(len(step.description) > 0)
            self.assertTrue(len(step.reasoning_process) > 0)
    
    def test_classify_crypto_problem(self):
        """测试密码学问题分类"""
        test_cases = [
            ("如何加密数据？", "encryption_decryption"),
            ("SHA-256哈希函数的特点", "hash_function"),
            ("数字签名的验证过程", "digital_signature"),
            ("密钥管理的最佳实践", "key_management"),
            ("针对RSA的攻击方法", "cryptanalysis"),
            ("区块链的共识机制", "general_crypto")
        ]
        
        for problem, expected_type in test_cases:
            result = self.generator._classify_crypto_problem(problem)
            self.assertEqual(result, expected_type)
    
    def test_validate_thinking_coherence(self):
        """测试思考连贯性验证"""
        # 测试连贯的thinking文本
        coherent_text = """
        首先，我需要理解这个问题的核心要求。
        然后，分析相关的技术概念和安全需求。
        接下来，考虑不同的解决方案。
        最后，得出最佳方案并验证其可行性。
        """
        
        result = self.generator.validate_thinking_coherence(coherent_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn("coherence_score", result)
        self.assertIn("issues", result)
        self.assertIn("suggestions", result)
        self.assertIn("logical_flow", result)
        self.assertIn("completeness", result)
        
        self.assertGreater(result["coherence_score"], 0.5)
        self.assertTrue(result["logical_flow"])
        
        # 测试不连贯的文本
        incoherent_text = "AES很好。RSA也不错。"
        result = self.generator.validate_thinking_coherence(incoherent_text)
        
        self.assertLess(result["coherence_score"], 0.6)
        self.assertTrue(len(result["issues"]) > 0)
        
        # 测试空文本
        empty_result = self.generator.validate_thinking_coherence("")
        self.assertEqual(empty_result["coherence_score"], 0.0)
        self.assertIn("思考内容为空", empty_result["issues"])
    
    def test_assess_thinking_quality(self):
        """测试thinking质量评估"""
        # 创建高质量的thinking样例
        high_quality_example = ThinkingExample(
            instruction="解释AES加密算法的工作原理",
            thinking_process="""
            首先，我需要理解AES算法的基本概念。AES是一种对称加密算法，
            使用相同的密钥进行加密和解密。接下来，分析其工作原理：
            AES采用替换-置换网络结构，通过多轮变换来实现加密。
            然后，考虑其安全性：AES-128、AES-192和AES-256提供不同级别的安全性。
            最后，验证其在实际应用中的效果和适用场景。
            """,
            final_response="AES是一种安全可靠的对称加密算法...",
            crypto_terms=["AES", "对称加密", "密钥"]
        )
        
        assessment = self.generator.assess_thinking_quality(high_quality_example)
        
        self.assertIsInstance(assessment, dict)
        self.assertIn("overall_quality", assessment)
        self.assertIn("dimensions", assessment)
        self.assertIn("strengths", assessment)
        self.assertIn("weaknesses", assessment)
        self.assertIn("improvement_suggestions", assessment)
        
        # 检查评估维度
        dimensions = assessment["dimensions"]
        self.assertIn("coherence", dimensions)
        self.assertIn("completeness", dimensions)
        self.assertIn("accuracy", dimensions)
        self.assertIn("depth", dimensions)
        self.assertIn("clarity", dimensions)
        
        # 高质量样例应该有较高的总体评分
        self.assertGreater(assessment["overall_quality"], 0.6)
        
        # 测试低质量样例
        low_quality_example = ThinkingExample(
            instruction="AES是什么？",
            thinking_process="AES就是加密。",
            final_response="AES是加密算法。",
            crypto_terms=[]
        )
        
        low_assessment = self.generator.assess_thinking_quality(low_quality_example)
        self.assertLess(low_assessment["overall_quality"], 0.6)
        self.assertTrue(len(low_assessment["weaknesses"]) > 0)
        self.assertTrue(len(low_assessment["improvement_suggestions"]) > 0)
    
    def test_assess_thinking_depth(self):
        """测试思考深度评估"""
        # 深度思考文本
        deep_text = """
        这个问题需要深入分析多个层面。首先从技术角度进行根本原因分析，
        然后从安全角度进行更深层次的考虑。我们需要综合考虑各种因素，
        权衡不同方案的trade-off，进一步探讨其本质特征。
        """
        
        depth_score = self.generator._assess_thinking_depth(deep_text)
        self.assertGreaterEqual(depth_score, 0.7)
        
        # 浅层思考文本
        shallow_text = "这很简单。"
        shallow_score = self.generator._assess_thinking_depth(shallow_text)
        self.assertLess(shallow_score, 0.5)
    
    def test_assess_thinking_clarity(self):
        """测试思考清晰度评估"""
        # 清晰的文本
        clear_text = """
        具体来说，这个算法有三个主要步骤。
        换句话说，我们需要按顺序执行这些操作。
        例如，第一步是初始化密钥。
        总结来说，整个过程是安全可靠的。
        """
        
        clarity_score = self.generator._assess_thinking_clarity(clear_text)
        self.assertGreater(clarity_score, 0.6)
        
        # 不清晰的文本
        unclear_text = "算法步骤很复杂需要仔细考虑各种情况然后执行相应操作最终得到结果。"
        unclear_score = self.generator._assess_thinking_clarity(unclear_text)
        self.assertLess(unclear_score, 0.7)
    
    def test_assess_crypto_accuracy(self):
        """测试密码学准确性评估"""
        # 准确使用术语的文本
        accurate_text = """
        AES是一种对称加密算法，使用相同的密钥进行加密和解密。
        它提供了强大的安全性保护，能够抵御各种密码学攻击。
        """
        
        accuracy_score = self.generator._assess_crypto_accuracy(
            accurate_text, ["AES", "对称加密", "密钥"]
        )
        self.assertGreater(accuracy_score, 0.5)
        
        # 不准确使用术语的文本
        inaccurate_text = "AES用于网络通信。"
        inaccuracy_score = self.generator._assess_crypto_accuracy(
            inaccurate_text, ["AES"]
        )
        self.assertLessEqual(inaccuracy_score, 1.0)
    
    def test_convert_to_thinking_format(self):
        """测试数据格式转换"""
        instruction = "什么是RSA算法？"
        response = "RSA是一种非对称加密算法，由Rivest、Shamir和Adleman发明。"
        crypto_terms = ["RSA", "非对称加密"]
        
        thinking_example = self.generator.convert_to_thinking_format(
            instruction=instruction,
            response=response,
            crypto_terms=crypto_terms
        )
        
        self.assertIsInstance(thinking_example, ThinkingExample)
        self.assertEqual(thinking_example.instruction, instruction)
        self.assertEqual(thinking_example.final_response, response)
        self.assertEqual(thinking_example.crypto_terms, crypto_terms)
        self.assertTrue(len(thinking_example.thinking_process) > 0)
        
        # 测试没有密码学术语的情况
        simple_thinking = self.generator.convert_to_thinking_format(
            instruction="今天天气怎么样？",
            response="今天天气很好。"
        )
        
        self.assertIsInstance(simple_thinking, ThinkingExample)
        self.assertEqual(simple_thinking.crypto_terms, [])
    
    def test_thinking_template(self):
        """测试思考模板"""
        template = ThinkingTemplate(
            name="测试模板",
            pattern=ReasoningPattern.LINEAR,
            steps=[
                "第一步：{step1}",
                "第二步：{step2}",
                "结论：{conclusion}"
            ]
        )
        
        context = {
            "step1": "分析问题",
            "step2": "设计方案",
            "conclusion": "验证结果"
        }
        
        generated_steps = template.generate_structure(context)
        
        self.assertEqual(len(generated_steps), 3)
        self.assertEqual(generated_steps[0], "第一步：分析问题")
        self.assertEqual(generated_steps[1], "第二步：设计方案")
        self.assertEqual(generated_steps[2], "结论：验证结果")
    
    def test_select_template(self):
        """测试模板选择"""
        # 测试密码学分析模板选择
        crypto_analysis = {"crypto_concepts": ["AES"], "problem_core": "安全性分析"}
        template = self.generator._select_template(
            ThinkingType.CRYPTOGRAPHIC, crypto_analysis
        )
        self.assertEqual(template.name, "密码学分析")
        
        # 测试算法比较模板选择
        comparison_analysis = {"crypto_concepts": ["RSA", "ECC"], "problem_core": "算法或概念比较"}
        template = self.generator._select_template(
            ThinkingType.COMPARATIVE, comparison_analysis
        )
        self.assertEqual(template.name, "算法比较")
        
        # 测试概念解释模板选择
        concept_analysis = {"crypto_concepts": [], "problem_core": "概念理解和应用"}
        template = self.generator._select_template(
            ThinkingType.ANALYTICAL, concept_analysis
        )
        self.assertEqual(template.name, "概念解释")
    
    def test_elaborate_thinking_steps(self):
        """测试思考步骤扩展"""
        steps = ["分析密码学概念", "考虑安全性要求", "设计解决方案"]
        analysis = {
            "crypto_concepts": ["AES", "RSA"],
            "problem_core": "密码学分析"
        }
        
        detailed_steps = self.generator._elaborate_thinking_steps(
            steps, analysis, target_length=200
        )
        
        self.assertEqual(len(detailed_steps), len(steps))
        for detailed_step in detailed_steps:
            self.assertIsInstance(detailed_step, str)
            self.assertTrue(len(detailed_step) > 0)
        
        # 检查是否包含了分析结果
        combined_text = " ".join(detailed_steps)
        self.assertIn("AES", combined_text)
        self.assertIn("RSA", combined_text)
    
    def test_assemble_thinking_process(self):
        """测试thinking过程组装"""
        steps = [
            "首先分析问题",
            "然后设计方案",
            "最后验证结果"
        ]
        
        assembled = self.generator._assemble_thinking_process(steps)
        
        self.assertIsInstance(assembled, str)
        self.assertTrue(len(assembled) > 0)
        
        # 检查是否包含所有步骤
        for step in steps:
            self.assertIn(step, assembled)
        
        # 检查是否有连接词
        connectors = ["接下来，", "然后，", "进一步，", "此外，", "最后，"]
        has_connector = any(connector in assembled for connector in connectors)
        self.assertTrue(has_connector)


class TestThinkingTemplate(unittest.TestCase):
    """思考模板测试类"""
    
    def test_template_creation(self):
        """测试模板创建"""
        template = ThinkingTemplate(
            name="测试模板",
            pattern=ReasoningPattern.HIERARCHICAL,
            steps=["步骤1", "步骤2"],
            crypto_focus=True,
            difficulty=DifficultyLevel.ADVANCED
        )
        
        self.assertEqual(template.name, "测试模板")
        self.assertEqual(template.pattern, ReasoningPattern.HIERARCHICAL)
        self.assertEqual(len(template.steps), 2)
        self.assertTrue(template.crypto_focus)
        self.assertEqual(template.difficulty, DifficultyLevel.ADVANCED)
    
    def test_generate_structure_with_context(self):
        """测试基于上下文的结构生成"""
        template = ThinkingTemplate(
            name="参数化模板",
            pattern=ReasoningPattern.LINEAR,
            steps=[
                "分析{topic}的{aspect}",
                "考虑{factor}因素",
                "得出{result}"
            ]
        )
        
        context = {
            "topic": "AES算法",
            "aspect": "安全性",
            "factor": "密钥长度",
            "result": "安全评估结论"
        }
        
        structure = template.generate_structure(context)
        
        self.assertEqual(len(structure), 3)
        self.assertEqual(structure[0], "分析AES算法的安全性")
        self.assertEqual(structure[1], "考虑密钥长度因素")
        self.assertEqual(structure[2], "得出安全评估结论")


if __name__ == '__main__':
    unittest.main()