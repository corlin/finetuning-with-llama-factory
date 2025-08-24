#!/usr/bin/env python3
"""
数据集分割模块演示

本演示展示了智能数据集分割模块的主要功能：
- 基础数据分割功能
- 专业术语分布优化
- thinking数据完整性保护
- 语义完整性保护
- 分割质量评估和验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_splitter import DatasetSplitter, SplitConfig, SplitStrategy
from src.data_models import TrainingExample, DifficultyLevel
import json


def create_demo_dataset():
    """创建演示数据集"""
    examples = []
    
    # 密码学术语集合
    crypto_terms_sets = [
        ["AES", "对称加密"],
        ["RSA", "非对称加密"],
        ["SHA-256", "哈希函数"],
        ["ECDSA", "数字签名"],
        ["DES", "对称加密"],
        ["ECC", "椭圆曲线密码"],
        ["HMAC", "消息认证码"],
        ["MD5", "哈希函数"]
    ]
    
    difficulties = list(DifficultyLevel)
    
    for i in range(80):
        crypto_terms = crypto_terms_sets[i % len(crypto_terms_sets)]
        difficulty = difficulties[i % len(difficulties)]
        
        # 60%的样例包含thinking数据
        thinking = None
        if i % 5 < 3:
            thinking = f"""<thinking>
让我分析{crypto_terms[0]}的特点：

1. 首先考虑其基本原理
   - {crypto_terms[0]}属于{crypto_terms[1]}范畴
   - 需要理解其核心机制

2. 然后分析安全性
   - 评估其抗攻击能力
   - 考虑实际应用中的安全风险

3. 最后总结应用场景
   - 适用的具体场景
   - 与其他算法的比较

综上所述，{crypto_terms[0]}是一个重要的密码学工具。
</thinking>"""
        
        # 创建不同长度和复杂度的指令
        if difficulty == DifficultyLevel.BEGINNER:
            instruction = f"问题{i}：什么是{crypto_terms[0]}？"
            output = f"回答{i}：{crypto_terms[0]}是一种{crypto_terms[1]}技术，具有基本的安全特性。"
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            instruction = f"问题{i}：请详细解释{crypto_terms[0]}的工作原理和主要特点？"
            output = f"回答{i}：{crypto_terms[0]}是{crypto_terms[1]}技术，工作原理包括：1. 基本机制；2. 安全特性；3. 应用场景。具有高安全性和广泛应用的特点。"
        elif difficulty == DifficultyLevel.ADVANCED:
            instruction = f"问题{i}：在实际应用中，{crypto_terms[0]}算法面临哪些安全挑战，如何进行优化？"
            output = f"回答{i}：{crypto_terms[0]}在实际应用中面临的主要挑战包括：1. 性能优化；2. 安全威胁防护；3. 标准化实现。优化策略包括算法改进、硬件加速和安全增强等方面。"
        else:  # EXPERT
            instruction = f"问题{i}：请从密码学理论角度深入分析{crypto_terms[0]}的数学基础、安全证明和前沿研究方向？"
            output = f"回答{i}：从理论角度分析，{crypto_terms[0]}的数学基础涉及复杂的代数结构和计算复杂性理论。安全证明基于困难问题假设，当前研究方向包括抗量子攻击、零知识证明集成和形式化验证等前沿领域。"
        
        example = TrainingExample(
            instruction=instruction,
            input="",
            output=output,
            thinking=thinking,
            crypto_terms=crypto_terms,
            difficulty_level=difficulty,
            source_file=f"demo_{i}.md"
        )
        examples.append(example)
    
    return examples


def demo_basic_splitting():
    """演示基础分割功能"""
    print("=" * 60)
    print("1. 基础数据分割功能演示")
    print("=" * 60)
    
    examples = create_demo_dataset()
    print(f"创建了 {len(examples)} 个演示样例")
    
    # 使用默认配置进行分割
    splitter = DatasetSplitter()
    splits = splitter.split_dataset(examples)
    
    sizes = splits.get_split_sizes()
    ratios = splits.get_split_ratios()
    
    print(f"\n分割结果：")
    print(f"  训练集: {sizes['train']} 样例 ({ratios['train']:.2%})")
    print(f"  验证集: {sizes['val']} 样例 ({ratios['val']:.2%})")
    print(f"  测试集: {sizes['test']} 样例 ({ratios['test']:.2%})")
    print(f"  总计: {sizes['total']} 样例")
    
    return splits


def demo_advanced_splitting():
    """演示高级分割功能"""
    print("\n" + "=" * 60)
    print("2. 高级分割功能演示（均衡分割 + 术语优化）")
    print("=" * 60)
    
    examples = create_demo_dataset()
    
    # 使用高级配置
    config = SplitConfig(
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.1,
        strategy=SplitStrategy.BALANCED,
        balance_crypto_terms=True,
        preserve_thinking_integrity=True,
        balance_difficulty_levels=True,
        random_seed=42
    )
    
    splitter = DatasetSplitter(config)
    splits = splitter.split_dataset(examples)
    
    sizes = splits.get_split_sizes()
    ratios = splits.get_split_ratios()
    
    print(f"高级分割结果：")
    print(f"  训练集: {sizes['train']} 样例 ({ratios['train']:.2%})")
    print(f"  验证集: {sizes['val']} 样例 ({ratios['val']:.2%})")
    print(f"  测试集: {sizes['test']} 样例 ({ratios['test']:.2%})")
    
    # 显示元数据
    metadata = splits.split_metadata
    print(f"\n难度分布统计：")
    for split_name in ['train', 'val', 'test']:
        dist = metadata['difficulty_distribution'][split_name]
        print(f"  {split_name}: {dist}")
    
    print(f"\nthinking数据统计：")
    for split_name in ['train', 'val', 'test']:
        stats = metadata['thinking_data_stats'][split_name]
        print(f"  {split_name}: {stats['with_thinking']}/{stats['total']} ({stats['thinking_ratio']:.2%})")
    
    return splits


def demo_quality_assessment():
    """演示质量评估功能"""
    print("\n" + "=" * 60)
    print("3. 分割质量评估演示")
    print("=" * 60)
    
    examples = create_demo_dataset()
    splitter = DatasetSplitter()
    splits = splitter.split_dataset(examples)
    
    # 评估分割质量
    quality_report = splitter.evaluate_split_quality(splits)
    
    print(f"质量评估结果：")
    print(f"  综合评分: {quality_report.overall_score:.3f}")
    print(f"  分布均衡性: {quality_report.distribution_balance_score:.3f}")
    print(f"  术语均衡性: {quality_report.crypto_term_balance_score:.3f}")
    print(f"  难度均衡性: {quality_report.difficulty_balance_score:.3f}")
    print(f"  thinking完整性: {quality_report.thinking_integrity_score:.3f}")
    print(f"  语义完整性: {quality_report.semantic_integrity_score:.3f}")
    print(f"  过拟合风险: {quality_report.overfitting_risk_score:.3f}")
    
    print(f"\n质量等级: {'高质量' if quality_report.is_high_quality() else '需要改进'}")
    print(f"风险等级: {quality_report.get_risk_level()}")
    
    if quality_report.warnings:
        print(f"\n警告:")
        for warning in quality_report.warnings:
            print(f"  - {warning}")
    
    if quality_report.recommendations:
        print(f"\n建议:")
        for rec in quality_report.recommendations:
            print(f"  - {rec}")


def demo_format_validation():
    """演示格式验证功能"""
    print("\n" + "=" * 60)
    print("4. Qwen格式兼容性验证演示")
    print("=" * 60)
    
    examples = create_demo_dataset()
    splitter = DatasetSplitter()
    splits = splitter.split_dataset(examples)
    
    # 验证格式兼容性
    validation_result = splitter.validate_qwen_format_compatibility(splits)
    
    print(f"格式验证结果:")
    print(f"  兼容性: {'通过' if validation_result['compatible'] else '失败'}")
    print(f"  问题数量: {len(validation_result['issues'])}")
    print(f"  警告数量: {len(validation_result['warnings'])}")
    
    stats = validation_result['statistics']
    print(f"\n统计信息:")
    print(f"  总样例数: {stats['total_examples']}")
    print(f"  thinking样例数: {stats['thinking_examples']}")
    print(f"  平均指令长度: {stats['avg_instruction_length']:.1f} 字符")
    print(f"  平均输出长度: {stats['avg_output_length']:.1f} 字符")
    print(f"  密码学术语种类: {stats['crypto_terms_count']}")


def demo_overfitting_assessment():
    """演示过拟合风险评估"""
    print("\n" + "=" * 60)
    print("5. 过拟合风险评估演示")
    print("=" * 60)
    
    examples = create_demo_dataset()
    splitter = DatasetSplitter()
    splits = splitter.split_dataset(examples)
    
    # 评估过拟合风险
    risk_assessment = splitter.assess_overfitting_risk(splits)
    
    print(f"过拟合风险评估:")
    print(f"  风险等级: {risk_assessment['risk_level']}")
    print(f"  风险评分: {risk_assessment['risk_score']:.3f}")
    
    if risk_assessment['risk_factors']:
        print(f"\n风险因素:")
        for factor in risk_assessment['risk_factors']:
            print(f"  - {factor}")
    
    if risk_assessment['recommendations']:
        print(f"\n改进建议:")
        for rec in risk_assessment['recommendations']:
            print(f"  - {rec}")


def demo_comprehensive_report():
    """演示综合报告生成"""
    print("\n" + "=" * 60)
    print("6. 综合报告生成演示")
    print("=" * 60)
    
    examples = create_demo_dataset()
    splitter = DatasetSplitter()
    splits = splitter.split_dataset(examples)
    
    # 生成综合报告
    report = splitter.generate_comprehensive_report(splits)
    
    print(f"综合报告摘要:")
    summary = report['summary']
    print(f"  总样例数: {summary['total_examples']}")
    print(f"  综合质量评分: {summary['overall_quality_score']:.3f}")
    print(f"  格式兼容性: {'通过' if summary['format_compatible'] else '失败'}")
    print(f"  过拟合风险: {summary['overfitting_risk']}")
    
    # 保存报告到文件
    output_path = "dataset_split_report.json"
    splitter.save_comprehensive_report(splits, output_path)
    print(f"\n详细报告已保存到: {output_path}")
    
    return report


def demo_save_and_load():
    """演示保存和加载功能"""
    print("\n" + "=" * 60)
    print("7. 数据保存和加载演示")
    print("=" * 60)
    
    examples = create_demo_dataset()
    splitter = DatasetSplitter()
    splits = splitter.split_dataset(examples)
    
    # 保存分割结果
    output_dir = "demo_splits"
    splitter.save_splits(splits, output_dir)
    print(f"分割结果已保存到: {output_dir}/")
    
    # 加载分割结果
    loaded_splits = splitter.load_splits(output_dir)
    loaded_sizes = loaded_splits.get_split_sizes()
    
    print(f"加载验证:")
    print(f"  原始总数: {splits.get_split_sizes()['total']}")
    print(f"  加载总数: {loaded_sizes['total']}")
    print(f"  数据完整性: {'通过' if loaded_sizes['total'] == splits.get_split_sizes()['total'] else '失败'}")


def main():
    """主演示函数"""
    print("智能数据集分割模块演示")
    print("支持中文密码学领域的深度思考(thinking)数据处理")
    
    try:
        # 运行各个演示
        demo_basic_splitting()
        demo_advanced_splitting()
        demo_quality_assessment()
        demo_format_validation()
        demo_overfitting_assessment()
        demo_comprehensive_report()
        demo_save_and_load()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n主要功能特点:")
        print("✓ 支持多种分割策略（随机、分层、均衡、语义）")
        print("✓ 密码学术语分布优化")
        print("✓ thinking数据完整性保护")
        print("✓ 语义完整性保护")
        print("✓ 全面的质量评估和验证")
        print("✓ Qwen3-4B-Thinking格式兼容性")
        print("✓ 过拟合风险评估")
        print("✓ 综合报告生成")
        print("✓ 数据保存和加载")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()