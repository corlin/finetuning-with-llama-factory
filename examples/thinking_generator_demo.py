"""
深度思考数据生成器演示

展示如何使用ThinkingDataGenerator生成高质量的thinking数据。
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thinking_generator import ThinkingDataGenerator, ThinkingType
from data_models import CryptoTerm, CryptoCategory, DifficultyLevel


def main():
    """主演示函数"""
    print("=== 深度思考数据生成器演示 ===\n")
    
    # 初始化生成器
    generator = ThinkingDataGenerator()
    
    # 示例1：生成基础thinking过程
    print("1. 生成基础thinking过程")
    print("-" * 40)
    
    instruction1 = "解释AES加密算法的工作原理"
    thinking_process1 = generator.generate_thinking_process(
        instruction=instruction1,
        thinking_type=ThinkingType.CRYPTOGRAPHIC,
        target_length=300
    )
    
    print(f"指令: {instruction1}")
    print(f"生成的thinking过程:\n{thinking_process1}\n")
    
    # 示例2：生成密码学推理过程
    print("2. 生成密码学推理过程")
    print("-" * 40)
    
    # 创建密码学术语
    crypto_terms = [
        CryptoTerm(
            term="RSA",
            definition="一种非对称加密算法",
            category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
            complexity=8
        ),
        CryptoTerm(
            term="AES",
            definition="高级加密标准",
            category=CryptoCategory.SYMMETRIC_ENCRYPTION,
            complexity=7
        )
    ]
    
    crypto_problem = "如何选择合适的加密算法来保护敏感数据？"
    thinking_structure = generator.generate_crypto_reasoning(
        crypto_problem=crypto_problem,
        crypto_terms=crypto_terms
    )
    
    print(f"密码学问题: {crypto_problem}")
    print(f"推理步骤数: {len(thinking_structure.reasoning_chain)}")
    print(f"逻辑一致性: {thinking_structure.logical_consistency:.2f}")
    print(f"生成的推理过程:\n{thinking_structure.raw_thinking}\n")
    
    # 示例3：验证thinking连贯性
    print("3. 验证thinking连贯性")
    print("-" * 40)
    
    test_thinking = """
    首先，我需要理解这个问题的核心要求。
    然后，分析相关的技术概念和安全需求。
    接下来，考虑不同的解决方案。
    最后，得出最佳方案并验证其可行性。
    """
    
    coherence_result = generator.validate_thinking_coherence(test_thinking)
    print(f"测试文本: {test_thinking.strip()}")
    print(f"连贯性分数: {coherence_result['coherence_score']:.2f}")
    print(f"完整性分数: {coherence_result['completeness']:.2f}")
    print(f"发现的问题: {coherence_result['issues']}")
    print(f"改进建议: {coherence_result['suggestions']}\n")
    
    # 示例4：转换为thinking格式
    print("4. 转换现有数据为thinking格式")
    print("-" * 40)
    
    original_instruction = "什么是数字签名？"
    original_response = "数字签名是一种用于验证数字文档真实性和完整性的密码学技术。"
    
    thinking_example = generator.convert_to_thinking_format(
        instruction=original_instruction,
        response=original_response,
        crypto_terms=["数字签名", "密码学"]
    )
    
    print(f"原始指令: {original_instruction}")
    print(f"原始回答: {original_response}")
    print(f"生成的thinking过程:\n{thinking_example.thinking_process}")
    print(f"最终回答: {thinking_example.final_response}\n")
    
    # 示例5：质量评估
    print("5. thinking质量评估")
    print("-" * 40)
    
    assessment = generator.assess_thinking_quality(thinking_example)
    print(f"总体质量分数: {assessment['overall_quality']:.2f}")
    print("各维度评分:")
    for dimension, score in assessment['dimensions'].items():
        print(f"  {dimension}: {score:.2f}")
    print(f"优势: {assessment['strengths']}")
    print(f"弱点: {assessment['weaknesses']}")
    print(f"改进建议: {assessment['improvement_suggestions']}")


if __name__ == "__main__":
    main()