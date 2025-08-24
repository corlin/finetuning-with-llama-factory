#!/usr/bin/env python3
"""
任务2.1实际数据测试脚本

使用data/raw目录中的真实密码学QA数据测试训练数据模型和深度思考数据结构的实现。
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 导入我们实现的数据模型
from src.data_models import (
    TrainingExample, ThinkingExample, ThinkingStructure, CryptoTerm, 
    ChineseMetrics, ReasoningStep, DifficultyLevel, CryptoCategory,
    DataModelValidator
)


class RealDataTester:
    """真实数据测试器"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.qa_files = list(self.data_dir.glob("QA*.md"))
        self.crypto_terms = []
        self.training_examples = []
        self.thinking_examples = []
        
    def extract_crypto_terms_from_qa(self, content: str) -> List[CryptoTerm]:
        """从QA内容中提取密码学术语"""
        terms = []
        
        # 定义密码学术语模式和分类
        crypto_patterns = {
            CryptoCategory.SYMMETRIC_ENCRYPTION: [
                "对称密码", "对称加密", "AES", "DES", "3DES", "SM4"
            ],
            CryptoCategory.ASYMMETRIC_ENCRYPTION: [
                "非对称密码", "非对称加密", "RSA", "ECC", "SM2", "公钥", "私钥"
            ],
            CryptoCategory.HASH_FUNCTION: [
                "杂凑算法", "哈希函数", "SHA", "MD5", "SM3", "消息摘要"
            ],
            CryptoCategory.DIGITAL_SIGNATURE: [
                "数字签名", "电子签名", "签名算法", "签名验证"
            ],
            CryptoCategory.KEY_MANAGEMENT: [
                "密钥管理", "密钥生成", "密钥分发", "密钥存储", "密钥销毁"
            ],
            CryptoCategory.CRYPTOGRAPHIC_PROTOCOL: [
                "密码协议", "SSL", "TLS", "IPSec", "VPN"
            ],
            CryptoCategory.CRYPTANALYSIS: [
                "密码分析", "攻击", "破解", "安全性分析"
            ],
            CryptoCategory.OTHER: [
                "身份鉴别", "访问控制", "完整性", "机密性", "真实性", "不可否认性"
            ]
        }
        
        for category, keywords in crypto_patterns.items():
            for keyword in keywords:
                if keyword in content:
                    # 尝试提取术语的定义
                    definition_pattern = rf"什么是{keyword}[？?].*?A\d+:\s*(.+?)(?=\n\n|\n###|$)"
                    match = re.search(definition_pattern, content, re.DOTALL)
                    
                    if match:
                        definition = match.group(1).strip()
                        # 计算复杂度（基于定义长度和技术词汇数量）
                        complexity = min(10, max(1, len(definition) // 20 + 
                                               len(re.findall(r'[技术|算法|协议|系统|安全]', definition))))
                        
                        term = CryptoTerm(
                            term=keyword,
                            definition=definition,
                            category=category,
                            complexity=complexity
                        )
                        terms.append(term)
                        break
        
        return terms
    
    def create_training_examples_from_qa(self, content: str, source_file: str) -> List[TrainingExample]:
        """从QA内容创建训练样例"""
        examples = []
        
        # 提取QA对
        qa_pattern = r"### (Q\d+): (.+?)\n\n(A\d+): (.+?)(?=\n\n###|\n\n##|$)"
        matches = re.findall(qa_pattern, content, re.DOTALL)
        
        for q_num, question, a_num, answer in matches:
            # 提取密码学术语
            crypto_terms = []
            for term in ["密码", "加密", "解密", "密钥", "算法", "安全", "认证", "签名"]:
                if term in question or term in answer:
                    crypto_terms.append(term)
            
            # 判断难度级别
            difficulty = DifficultyLevel.INTERMEDIATE
            if any(word in question + answer for word in ["基础", "什么是", "定义"]):
                difficulty = DifficultyLevel.BEGINNER
            elif any(word in question + answer for word in ["技术要求", "实施", "测评"]):
                difficulty = DifficultyLevel.ADVANCED
            elif any(word in question + answer for word in ["第四级", "第五级", "复杂"]):
                difficulty = DifficultyLevel.EXPERT
            
            # 创建训练样例
            example = TrainingExample(
                instruction=f"请回答以下关于密码应用的问题：{question}",
                input="",
                output=answer.strip(),
                crypto_terms=crypto_terms,
                difficulty_level=difficulty,
                source_file=source_file,
                metadata={
                    "question_id": q_num,
                    "answer_id": a_num,
                    "category": "密码应用标准"
                }
            )
            examples.append(example)
        
        return examples
    
    def create_thinking_examples_with_reasoning(self, training_examples: List[TrainingExample]) -> List[ThinkingExample]:
        """为训练样例创建带有深度思考的版本"""
        thinking_examples = []
        
        for example in training_examples[:10]:  # 只处理前10个作为示例
            # 生成思考过程
            thinking_process = self.generate_thinking_process(example.instruction, example.output)
            
            thinking_example = ThinkingExample(
                instruction=example.instruction,
                thinking_process=thinking_process,
                final_response=example.output,
                crypto_terms=example.crypto_terms,
                difficulty_level=example.difficulty_level,
                source_domain="密码学标准"
            )
            thinking_examples.append(thinking_example)
        
        return thinking_examples
    
    def generate_thinking_process(self, question: str, answer: str) -> str:
        """生成思考过程"""
        # 简化的思考过程生成
        thinking_steps = []
        
        # 分析问题类型
        if "什么是" in question:
            thinking_steps.append("这是一个概念定义类问题，需要准确解释相关术语的含义。")
        elif "要求" in question:
            thinking_steps.append("这是一个要求类问题，需要列出具体的技术或管理要求。")
        elif "区别" in question:
            thinking_steps.append("这是一个比较类问题，需要分析不同概念或等级之间的差异。")
        else:
            thinking_steps.append("需要仔细分析问题的核心要点。")
        
        # 分析答案结构
        if "包括" in answer:
            thinking_steps.append("答案需要列举多个要点，应该确保完整性和准确性。")
        elif "是指" in answer:
            thinking_steps.append("这是一个定义性回答，需要准确表达概念的本质。")
        
        # 检查密码学术语
        crypto_terms_found = []
        for term in ["密码", "加密", "密钥", "算法", "安全", "认证"]:
            if term in question or term in answer:
                crypto_terms_found.append(term)
        
        if crypto_terms_found:
            thinking_steps.append(f"涉及的密码学术语包括：{', '.join(crypto_terms_found)}，需要确保术语使用的准确性。")
        
        thinking_steps.append("基于GB/T 39786-2021标准的要求，给出准确的回答。")
        
        return "<thinking>\n" + "\n".join(thinking_steps) + "\n</thinking>"
    
    def test_crypto_term_extraction_and_validation(self):
        """测试密码学术语提取和验证"""
        print("=== 测试密码学术语提取和验证 ===")
        
        all_terms = []
        for qa_file in self.qa_files[:2]:  # 只测试前2个文件
            print(f"\n处理文件: {qa_file.name}")
            
            with open(qa_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            terms = self.extract_crypto_terms_from_qa(content)
            all_terms.extend(terms)
            
            print(f"提取到 {len(terms)} 个密码学术语")
            for term in terms[:3]:  # 显示前3个
                print(f"  - {term.term} ({term.category.value}): {term.definition[:50]}...")
        
        # 测试术语序列化
        if all_terms:
            print(f"\n测试术语序列化...")
            term_dict = all_terms[0].to_dict()
            restored_term = CryptoTerm.from_dict(term_dict)
            assert restored_term.term == all_terms[0].term
            print("✓ 术语序列化测试通过")
        
        self.crypto_terms = all_terms
        return all_terms
    
    def test_training_example_creation_and_validation(self):
        """测试训练样例创建和验证"""
        print("\n=== 测试训练样例创建和验证 ===")
        
        all_examples = []
        for qa_file in self.qa_files[:2]:  # 只测试前2个文件
            print(f"\n处理文件: {qa_file.name}")
            
            with open(qa_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            examples = self.create_training_examples_from_qa(content, qa_file.name)
            all_examples.extend(examples)
            
            print(f"创建了 {len(examples)} 个训练样例")
            
            # 显示第一个样例
            if examples:
                example = examples[0]
                print(f"  示例: {example.instruction[:50]}...")
                print(f"  难度: {example.difficulty_level.name}")
                print(f"  术语: {example.crypto_terms[:3]}")
        
        # 测试样例验证
        if all_examples:
            print(f"\n测试样例验证...")
            example = all_examples[0]
            assert example.validate_format() == True
            print("✓ 样例格式验证通过")
            
            # 测试LLaMA Factory格式转换
            llama_format = example.to_llama_factory_format()
            assert "instruction" in llama_format
            assert "output" in llama_format
            print("✓ LLaMA Factory格式转换通过")
            
            # 测试序列化
            example_dict = example.to_dict()
            restored_example = TrainingExample.from_dict(example_dict)
            assert restored_example.instruction == example.instruction
            print("✓ 样例序列化测试通过")
        
        self.training_examples = all_examples
        return all_examples
    
    def test_thinking_example_creation_and_validation(self):
        """测试深度思考样例创建和验证"""
        print("\n=== 测试深度思考样例创建和验证 ===")
        
        if not self.training_examples:
            print("需要先创建训练样例")
            return []
        
        thinking_examples = self.create_thinking_examples_with_reasoning(self.training_examples)
        print(f"创建了 {len(thinking_examples)} 个深度思考样例")
        
        if thinking_examples:
            example = thinking_examples[0]
            print(f"\n示例思考过程:")
            print(example.thinking_process[:200] + "...")
            
            # 测试thinking标签验证
            assert example.validate_thinking_tags() == True
            print("✓ thinking标签验证通过")
            
            # 测试推理步骤提取
            steps = example.extract_reasoning_steps()
            print(f"✓ 提取到 {len(steps)} 个推理步骤")
            
            # 测试LLaMA Factory格式转换
            llama_format = example.to_llama_factory_format()
            assert "<thinking>" in llama_format["output"]
            print("✓ LLaMA Factory格式转换通过")
            
            # 测试序列化
            example_dict = example.to_dict()
            restored_example = ThinkingExample.from_dict(example_dict)
            assert restored_example.instruction == example.instruction
            print("✓ 思考样例序列化测试通过")
        
        self.thinking_examples = thinking_examples
        return thinking_examples
    
    def test_thinking_structure_analysis(self):
        """测试深度思考结构分析"""
        print("\n=== 测试深度思考结构分析 ===")
        
        if not self.thinking_examples:
            print("需要先创建深度思考样例")
            return
        
        example = self.thinking_examples[0]
        
        # 创建推理步骤
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                description="问题分析",
                input_data=example.instruction,
                reasoning_process="分析问题类型和关键词",
                output_result="确定这是一个密码学概念问题",
                confidence_score=0.9
            ),
            ReasoningStep(
                step_number=2,
                description="知识检索",
                input_data="密码学概念问题",
                reasoning_process="从GB/T 39786-2021标准中检索相关信息",
                output_result="找到相关标准条款",
                confidence_score=0.85
            )
        ]
        
        # 创建思考结构
        thinking_structure = ThinkingStructure(
            raw_thinking=example.thinking_process,
            parsed_steps=example.extract_reasoning_steps(),
            reasoning_chain=reasoning_steps,
            validation_result=example.validate_thinking_tags()
        )
        
        print(f"思考深度: {thinking_structure.thinking_depth}")
        print(f"逻辑一致性: {thinking_structure.logical_consistency:.2f}")
        print(f"格式验证: {thinking_structure.validate_thinking_format()}")
        
        # 测试内容提取
        content = thinking_structure.extract_thinking_content()
        print(f"✓ 提取到 {len(content)} 段思考内容")
        
        # 测试序列化
        structure_dict = thinking_structure.to_dict()
        restored_structure = ThinkingStructure.from_dict(structure_dict)
        assert restored_structure.thinking_depth == thinking_structure.thinking_depth
        print("✓ 思考结构序列化测试通过")
    
    def test_chinese_metrics_evaluation(self):
        """测试中文指标评估"""
        print("\n=== 测试中文指标评估 ===")
        
        # 创建中文评估指标
        metrics = ChineseMetrics(
            character_accuracy=0.95,
            word_accuracy=0.92,
            rouge_l_chinese=0.88,
            bleu_chinese=0.85,
            crypto_term_accuracy=0.90,
            semantic_similarity=0.87,
            fluency_score=0.89,
            coherence_score=0.86
        )
        
        print(f"字符准确率: {metrics.character_accuracy}")
        print(f"词汇准确率: {metrics.word_accuracy}")
        print(f"密码术语准确率: {metrics.crypto_term_accuracy}")
        print(f"综合评分: {metrics.overall_score():.3f}")
        
        # 测试序列化
        metrics_dict = metrics.to_dict()
        assert "overall_score" in metrics_dict
        print("✓ 中文指标评估测试通过")
    
    def test_data_validation_and_serialization(self):
        """测试数据验证和序列化"""
        print("\n=== 测试数据验证和序列化 ===")
        
        # 测试thinking数据验证
        valid_thinking = "<thinking>这是一个有效的思考过程</thinking>"
        result = DataModelValidator.validate_thinking_data(valid_thinking)
        assert result["valid"] == True
        print("✓ 有效thinking数据验证通过")
        
        invalid_thinking = "<thinking>不平衡的标签"
        result = DataModelValidator.validate_thinking_data(invalid_thinking)
        assert result["valid"] == False
        print("✓ 无效thinking数据验证通过")
        
        # 测试训练样例序列化
        if self.training_examples:
            json_str = DataModelValidator.serialize_training_examples(self.training_examples[:5])
            restored_examples = DataModelValidator.deserialize_training_examples(json_str)
            assert len(restored_examples) == 5
            print("✓ 训练样例批量序列化测试通过")
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("任务2.1实际数据测试报告")
        print("="*60)
        
        print(f"测试数据源: {len(self.qa_files)} 个QA文件")
        print(f"提取密码学术语: {len(self.crypto_terms)} 个")
        print(f"创建训练样例: {len(self.training_examples)} 个")
        print(f"创建思考样例: {len(self.thinking_examples)} 个")
        
        # 统计难度分布
        if self.training_examples:
            difficulty_stats = {}
            for example in self.training_examples:
                level = example.difficulty_level.name
                difficulty_stats[level] = difficulty_stats.get(level, 0) + 1
            
            print(f"\n难度级别分布:")
            for level, count in difficulty_stats.items():
                print(f"  {level}: {count} 个")
        
        # 统计术语分类
        if self.crypto_terms:
            category_stats = {}
            for term in self.crypto_terms:
                category = term.category.value
                category_stats[category] = category_stats.get(category, 0) + 1
            
            print(f"\n密码学术语分类:")
            for category, count in category_stats.items():
                print(f"  {category}: {count} 个")
        
        print(f"\n✅ 所有测试通过！任务2.1实现验证成功。")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始任务2.1实际数据测试...")
        print(f"数据源目录: {self.data_dir}")
        print(f"找到QA文件: {[f.name for f in self.qa_files]}")
        
        try:
            # 1. 测试密码学术语提取和验证
            self.test_crypto_term_extraction_and_validation()
            
            # 2. 测试训练样例创建和验证
            self.test_training_example_creation_and_validation()
            
            # 3. 测试深度思考样例创建和验证
            self.test_thinking_example_creation_and_validation()
            
            # 4. 测试深度思考结构分析
            self.test_thinking_structure_analysis()
            
            # 5. 测试中文指标评估
            self.test_chinese_metrics_evaluation()
            
            # 6. 测试数据验证和序列化
            self.test_data_validation_and_serialization()
            
            # 7. 生成测试报告
            self.generate_test_report()
            
        except Exception as e:
            print(f"\n❌ 测试失败: {str(e)}")
            raise


def main():
    """主函数"""
    # 检查数据目录
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    qa_files = list(data_dir.glob("QA*.md"))
    if not qa_files:
        print(f"❌ 在 {data_dir} 中未找到QA*.md文件")
        return
    
    # 运行测试
    tester = RealDataTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()