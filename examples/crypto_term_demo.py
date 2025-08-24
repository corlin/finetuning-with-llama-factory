"""
密码学术语处理器演示

演示密码学术语词典构建、术语识别标注、复杂度评估算法、
以及thinking数据中的专业术语处理功能。
"""

import sys
import os
import json
from typing import List

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from crypto_term_processor import CryptoTermProcessor, TermComplexity
    from data_models import ThinkingExample, DifficultyLevel, CryptoCategory
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def demo_basic_term_identification():
    """演示基础术语识别功能"""
    print("=" * 60)
    print("1. 基础术语识别演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 示例文本
    sample_texts = [
        "RSA是一种非对称加密算法，基于大整数分解难题。",
        "AES-256是目前最安全的对称加密标准之一。",
        "数字签名可以确保消息的完整性和不可否认性。",
        "SHA-256哈希函数被广泛应用于区块链技术中。",
        "椭圆曲线密码学提供了更高效的安全解决方案。",
        "PKI公钥基础设施是现代网络安全的重要组成部分。"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n文本 {i}: {text}")
        annotations = processor.identify_crypto_terms(text)
        
        if annotations:
            print("识别的术语:")
            for ann in annotations:
                print(f"  - {ann.term} ({ann.category.value})")
                print(f"    复杂度: {ann.complexity}/10")
                print(f"    置信度: {ann.confidence:.2f}")
                print(f"    定义: {ann.definition[:50]}..." if ann.definition else "")
        else:
            print("  未识别到密码学术语")


def demo_complexity_evaluation():
    """演示术语复杂度评估"""
    print("\n" + "=" * 60)
    print("2. 术语复杂度评估演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 不同复杂度的术语组合
    term_sets = {
        "基础术语": ["加密", "密钥", "哈希函数", "数字签名"],
        "中级术语": ["RSA", "AES", "SHA-256", "对称加密", "非对称加密"],
        "高级术语": ["椭圆曲线", "差分分析", "侧信道攻击", "零知识证明"],
        "混合术语": ["AES", "椭圆曲线", "PKI", "区块链", "密码分析"]
    }
    
    for category, terms in term_sets.items():
        complexity_score = processor.calculate_term_complexity(terms)
        print(f"\n{category}:")
        print(f"  术语: {', '.join(terms)}")
        print(f"  复杂度评分: {complexity_score:.2f}/10")


def demo_term_distribution_analysis():
    """演示术语分布分析"""
    print("\n" + "=" * 60)
    print("3. 术语分布分析演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 示例文档集合
    documents = [
        """
        RSA加密算法是一种广泛使用的非对称加密技术。它基于大整数分解的数学难题，
        使用公钥和私钥对进行加密和解密操作。RSA不仅用于数据加密，
        还广泛应用于数字签名领域。
        """,
        """
        AES（高级加密标准）是目前最常用的对称加密算法。与RSA不同，
        AES使用相同的密钥进行加密和解密。AES支持128位、192位和256位密钥长度，
        其中AES-256提供最高级别的安全性。
        """,
        """
        哈希函数是密码学的重要组成部分。SHA-256是最常用的哈希算法之一，
        被广泛应用于区块链技术中。与MD5不同，SHA-256具有更强的抗碰撞能力，
        是目前推荐使用的安全哈希算法。
        """,
        """
        椭圆曲线密码学(ECC)是一种基于椭圆曲线离散对数问题的密码技术。
        ECDSA是椭圆曲线数字签名算法，相比RSA签名，ECDSA在提供相同安全级别的
        情况下使用更短的密钥长度，因此在移动设备和物联网中应用广泛。
        """
    ]
    
    # 分析术语分布
    distribution = processor.analyze_term_distribution(documents)
    
    print(f"总术语数量: {distribution.total_terms}")
    print(f"唯一术语数量: {distribution.unique_terms}")
    print(f"术语覆盖率: {distribution.coverage_ratio:.2%}")
    
    print("\n类别分布:")
    for category, count in distribution.category_distribution.items():
        print(f"  {category.value}: {count}")
    
    print("\n复杂度分布:")
    for complexity, count in sorted(distribution.complexity_distribution.items()):
        print(f"  复杂度 {complexity}: {count}")
    
    print("\n高频术语 (前10个):")
    sorted_terms = sorted(distribution.term_frequency.items(), 
                         key=lambda x: x[1], reverse=True)
    for term, freq in sorted_terms[:10]:
        print(f"  {term}: {freq}")


def demo_thinking_term_processing():
    """演示thinking数据术语处理"""
    print("\n" + "=" * 60)
    print("4. Thinking数据术语处理演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 创建示例thinking数据
    thinking_example = ThinkingExample(
        instruction="请详细解释RSA加密算法的工作原理，并与AES进行比较。",
        thinking_process="""
        <thinking>
        这个问题需要我从多个角度来分析RSA加密算法：
        
        1. RSA的数学基础
        RSA基于大整数分解难题，这是一个在计算上困难的数学问题。
        算法使用两个大质数的乘积作为模数。
        
        2. 密钥生成过程
        - 选择两个大质数p和q
        - 计算n = p × q（模数）
        - 计算φ(n) = (p-1)(q-1)（欧拉函数）
        - 选择公钥指数e，通常是65537
        - 计算私钥指数d，使得ed ≡ 1 (mod φ(n))
        
        3. 加密和解密过程
        - 加密：c = m^e mod n（使用公钥）
        - 解密：m = c^d mod n（使用私钥）
        
        4. 与AES的比较
        RSA是非对称加密，AES是对称加密：
        - RSA使用公钥/私钥对，AES使用相同密钥
        - RSA速度较慢，适合加密小数据，AES速度快，适合大数据
        - RSA常用于密钥交换和数字签名，AES用于数据加密
        - RSA密钥长度通常2048位或更长，AES密钥128/192/256位
        
        5. 安全性考虑
        RSA的安全性依赖于大整数分解的困难性，而AES依赖于替换-置换网络的复杂性。
        </thinking>
        """,
        final_response="""
        RSA加密算法是一种非对称加密算法，由Rivest、Shamir和Adleman在1977年提出。
        
        **工作原理：**
        RSA基于大整数分解的数学难题。算法使用一对密钥：公钥用于加密，私钥用于解密。
        
        密钥生成过程包括选择两个大质数，计算它们的乘积作为模数，然后生成公钥和私钥指数。
        
        **与AES的比较：**
        1. **加密类型**：RSA是非对称加密，AES是对称加密
        2. **密钥管理**：RSA使用公钥/私钥对，AES使用相同密钥
        3. **性能**：AES速度更快，适合大数据加密；RSA较慢，适合小数据和密钥交换
        4. **应用场景**：RSA常用于数字签名和密钥交换，AES用于数据加密
        
        在实际应用中，通常结合使用：用RSA交换AES密钥，用AES加密实际数据。
        """,
        crypto_terms=["RSA", "非对称加密", "AES", "对称加密", "公钥", "私钥", "数字签名"],
        difficulty_level=DifficultyLevel.ADVANCED
    )
    
    # 分析thinking数据中的术语
    analysis = processor.process_thinking_terms(thinking_example)
    
    print(f"Thinking ID: {analysis.thinking_id}")
    print(f"总术语数量: {analysis.total_terms}")
    print(f"唯一术语数量: {analysis.unique_terms}")
    print(f"复杂度评分: {analysis.complexity_score:.2f}/10")
    print(f"专业性评分: {analysis.professional_score:.2f}")
    print(f"术语连贯性: {analysis.term_coherence:.2f}")
    
    print("\n识别的术语:")
    for ann in analysis.term_annotations[:10]:  # 显示前10个
        print(f"  - {ann.term} ({ann.category.value}, 复杂度: {ann.complexity})")


def demo_thinking_enhancement():
    """演示thinking文本增强"""
    print("\n" + "=" * 60)
    print("5. Thinking文本增强演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    original_thinking = """
    <thinking>
    RSA是一种加密算法，它使用数学方法来保护数据。
    这种算法比较安全，被广泛使用。
    </thinking>
    """
    
    print("原始thinking文本:")
    print(original_thinking)
    
    enhanced_thinking = processor.enhance_thinking_with_terms(original_thinking)
    
    print("\n增强后的thinking文本:")
    print(enhanced_thinking)


def demo_term_validation():
    """演示术语使用验证"""
    print("\n" + "=" * 60)
    print("6. 术语使用验证演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 测试不同质量的文本
    test_texts = [
        {
            "title": "正确使用示例",
            "text": "RSA是一种非对称加密算法，使用公钥和私钥进行加密和解密操作。"
        },
        {
            "title": "可能有问题的使用",
            "text": "RSA使用相同的密钥进行加密和解密，是一种对称加密算法。"
        },
        {
            "title": "术语密度较低",
            "text": "这是一个关于计算机安全的文档，讨论了一些保护数据的方法。"
        }
    ]
    
    for test_case in test_texts:
        print(f"\n{test_case['title']}:")
        print(f"文本: {test_case['text']}")
        
        validation_result = processor.validate_term_usage(test_case['text'])
        
        print(f"总术语数: {validation_result['total_terms']}")
        print(f"有效术语数: {validation_result['valid_terms']}")
        
        if validation_result['invalid_terms']:
            print("问题术语:")
            for invalid in validation_result['invalid_terms']:
                print(f"  - {invalid['term']}: {invalid['issue']}")
        
        if validation_result['suggestions']:
            print("改进建议:")
            for suggestion in validation_result['suggestions']:
                print(f"  - {suggestion}")


def demo_dictionary_statistics():
    """演示词典统计信息"""
    print("\n" + "=" * 60)
    print("7. 词典统计信息演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    stats = processor.get_term_statistics()
    
    print(f"词典统计信息:")
    print(f"  唯一术语数量: {stats['total_unique_terms']}")
    print(f"  总条目数量: {stats['total_entries']} (包括别名)")
    print(f"  平均复杂度: {stats['average_complexity']:.2f}")
    
    print(f"\n类别分布:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
    
    print(f"\n复杂度分布:")
    for complexity, count in sorted(stats['complexity_distribution'].items()):
        print(f"  复杂度 {complexity}: {count}")


def main():
    """主演示函数"""
    print("密码学术语处理器功能演示")
    print("=" * 60)
    
    try:
        # 运行各个演示
        demo_basic_term_identification()
        demo_complexity_evaluation()
        demo_term_distribution_analysis()
        demo_thinking_term_processing()
        demo_thinking_enhancement()
        demo_term_validation()
        demo_dictionary_statistics()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()