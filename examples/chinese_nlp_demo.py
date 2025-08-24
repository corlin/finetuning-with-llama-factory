#!/usr/bin/env python3
"""
中文NLP处理工具演示

演示中文分词、繁简体转换、标点符号规范化、文本质量评估和Qwen tokenizer优化功能。
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chinese_nlp_processor import (
    ChineseNLPProcessor, TextVariant, PunctuationStyle
)


def main():
    """主演示函数"""
    print("=" * 60)
    print("中文NLP处理工具演示")
    print("=" * 60)
    
    # 创建处理器实例
    processor = ChineseNLPProcessor()
    
    # 示例文本
    sample_texts = [
        "RSA算法使用大整数分解难题,确保加密安全性.",
        "AES是对称加密算法的代表，广泛应用于数据保护。",
        "哈希函数如SHA-256用于数据完整性验证和数字签名。",
        "Hello世界,这是mixed text测试!"
    ]
    
    print("\n1. 中文分词和词性标注演示")
    print("-" * 40)
    for text in sample_texts[:2]:
        print(f"原文: {text}")
        tokens = processor.segment_text(text)
        print("分词结果:")
        for token in tokens:
            crypto_mark = " [密码学术语]" if token.is_crypto_term else ""
            print(f"  {token.word} ({token.pos}){crypto_mark}")
        print()
    
    print("\n2. 繁简体转换演示")
    print("-" * 40)
    test_text = "网络安全和数据加密"
    print(f"原文 (简体): {test_text}")
    
    traditional = processor.convert_traditional_simplified(test_text, TextVariant.TRADITIONAL)
    print(f"转换为繁体: {traditional}")
    
    back_to_simplified = processor.convert_traditional_simplified(traditional, TextVariant.SIMPLIFIED)
    print(f"转回简体: {back_to_simplified}")
    
    variant = processor.detect_text_variant(test_text)
    print(f"文本变体检测: {variant.value}")
    print()
    
    print("\n3. 标点符号规范化演示")
    print("-" * 40)
    mixed_punct_text = "Hello,world.How are you?这是测试!"
    print(f"原文: {mixed_punct_text}")
    
    chinese_punct = processor.normalize_punctuation(mixed_punct_text, PunctuationStyle.CHINESE)
    print(f"中文标点: {chinese_punct}")
    
    english_punct = processor.normalize_punctuation(chinese_punct, PunctuationStyle.ENGLISH)
    print(f"英文标点: {english_punct}")
    print()
    
    print("\n4. 文本质量评估演示")
    print("-" * 40)
    quality_text = "RSA是一种非对称加密算法，它使用公钥和私钥进行加密和解密。首先，算法生成密钥对。然后，使用公钥加密数据。最后，用私钥解密数据。"
    print(f"评估文本: {quality_text}")
    
    quality_metrics = processor.assess_text_quality(quality_text)
    print("质量指标:")
    print(f"  可读性: {quality_metrics.readability_score:.3f}")
    print(f"  流畅度: {quality_metrics.fluency_score:.3f}")
    print(f"  连贯性: {quality_metrics.coherence_score:.3f}")
    print(f"  复杂度: {quality_metrics.complexity_score:.3f}")
    print(f"  标点规范性: {quality_metrics.punctuation_score:.3f}")
    print(f"  字符多样性: {quality_metrics.character_diversity:.3f}")
    print(f"  词汇多样性: {quality_metrics.word_diversity:.3f}")
    print(f"  平均句长: {quality_metrics.avg_sentence_length:.1f}")
    print(f"  综合质量: {quality_metrics.overall_quality():.3f}")
    print()
    
    print("\n5. 密码学术语提取演示")
    print("-" * 40)
    crypto_text = "RSA、AES、DES、SHA-256、MD5都是重要的密码学算法，用于数据加密和哈希计算。"
    print(f"原文: {crypto_text}")
    
    crypto_terms = processor.extract_crypto_terms_from_text(crypto_text)
    print(f"提取的密码学术语: {crypto_terms}")
    print()
    
    print("\n6. 训练预处理演示")
    print("-" * 40)
    raw_text = "Hello,world.這是繁體中文測試!"
    print(f"原文: {raw_text}")
    
    processed = processor.preprocess_for_training(raw_text)
    print(f"预处理后: {processed}")
    print()
    
    print("\n7. Qwen Tokenizer优化演示")
    print("-" * 40)
    training_texts = [
        "RSA加密算法基于大整数分解的数学难题",
        "AES对称加密算法具有高效的加密性能",
        "哈希函数SHA-256确保数据完整性验证",
        "数字签名技术提供身份认证和不可否认性"
    ]
    
    print("训练文本样例:")
    for i, text in enumerate(training_texts, 1):
        print(f"  {i}. {text}")
    
    optimization = processor.optimize_qwen_tokenizer(training_texts, vocab_size=1000)
    print(f"\nTokenizer优化结果:")
    print(f"  词汇表大小: {optimization.vocab_size}")
    print(f"  中文词汇比例: {optimization.chinese_vocab_ratio:.3f}")
    print(f"  密码学术语数量: {optimization.crypto_terms_count}")
    print(f"  OOV率: {optimization.oov_rate:.3f}")
    print(f"  压缩比: {optimization.compression_ratio:.2f}")
    print(f"  是否已优化: {optimization.is_optimized()}")
    print(f"  特殊Token数量: {len(optimization.special_tokens)}")
    print(f"  部分特殊Token: {optimization.special_tokens[:10]}")
    print()
    
    print("\n8. 中文评估指标演示")
    print("-" * 40)
    predictions = ["RSA是非对称加密算法", "AES用于对称加密"]
    references = ["RSA是一种非对称加密算法", "AES是对称加密算法"]
    
    print("预测文本:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred}")
    
    print("参考文本:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    
    chinese_metrics = processor.calculate_chinese_metrics(predictions, references)
    print(f"\n中文评估指标:")
    print(f"  字符准确率: {chinese_metrics.character_accuracy:.3f}")
    print(f"  词汇准确率: {chinese_metrics.word_accuracy:.3f}")
    print(f"  中文ROUGE-L: {chinese_metrics.rouge_l_chinese:.3f}")
    print(f"  中文BLEU: {chinese_metrics.bleu_chinese:.3f}")
    print(f"  密码学术语准确率: {chinese_metrics.crypto_term_accuracy:.3f}")
    print()
    
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()