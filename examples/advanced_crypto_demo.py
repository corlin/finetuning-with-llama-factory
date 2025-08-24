"""
高级密码学术语处理演示

展示扩展后的密码学术语词典，包含更多专业术语和高级概念。
"""

import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from crypto_term_processor import CryptoTermProcessor
    from data_models import CryptoCategory
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


def demo_advanced_terms():
    """演示高级密码学术语识别"""
    print("=" * 60)
    print("高级密码学术语识别演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 高级密码学文本示例
    advanced_texts = [
        """
        零知识证明允许证明者在不泄露秘密信息的情况下证明自己知道某个秘密。
        zk-SNARKs是一种简洁的非交互式零知识证明系统，广泛应用于区块链隐私保护。
        """,
        """
        同态加密是一种特殊的加密方法，允许在密文上直接进行计算操作。
        全同态加密(FHE)可以在加密数据上执行任意计算，是隐私计算的重要技术。
        """,
        """
        多方安全计算(MPC)允许多个参与方在不泄露各自输入的情况下共同计算函数。
        秘密分享是MPC的基础技术，Shamir方案是经典的门限秘密分享方案。
        """,
        """
        椭圆曲线密码学基于椭圆曲线离散对数问题的困难性。
        Ed25519是基于Curve25519的EdDSA签名算法，提供了高性能的数字签名。
        ECDH密钥交换协议使用椭圆曲线实现高效的密钥协商。
        """,
        """
        侧信道攻击通过分析密码设备的物理特征来获取密钥信息。
        功耗分析攻击通过监测设备的功耗变化来推断密钥。
        时间攻击利用算法执行时间的差异来获取敏感信息。
        """,
        """
        区块链技术使用工作量证明(PoW)或权益证明(PoS)等共识算法。
        智能合约是运行在区块链上的自动执行程序。
        DeFi(去中心化金融)基于智能合约提供金融服务。
        NFT(非同质化代币)用于表示独特的数字资产。
        """,
        """
        AES-GCM是一种认证加密(AEAD)算法，同时提供机密性和完整性保护。
        ChaCha20-Poly1305是现代流密码与认证的组合。
        CBC模式需要初始化向量，而CTR模式可以并行处理。
        """,
        """
        PKI公钥基础设施使用CA颁发数字证书。
        OCSP协议提供实时的证书状态检查，比CRL更高效。
        HSM硬件安全模块为密钥提供硬件级保护。
        """,
        """
        盲签名允许签名者在不知道消息内容的情况下进行签名。
        环签名提供了签名者的匿名性，门罗币使用环签名保护隐私。
        门限签名需要多个参与者协作才能生成有效签名。
        """,
        """
        PBKDF2、scrypt和Argon2都是密钥派生函数。
        Argon2是密码哈希竞赛的获胜者，提供了更好的安全性。
        scrypt具有内存困难特性，可以抵抗ASIC攻击。
        """
    ]
    
    total_terms = 0
    total_unique_terms = set()
    category_stats = {}
    complexity_stats = {}
    
    for i, text in enumerate(advanced_texts, 1):
        print(f"\n高级文本 {i}:")
        print(f"内容: {text.strip()}")
        
        annotations = processor.identify_crypto_terms(text)
        
        if annotations:
            print(f"识别的术语 ({len(annotations)}个):")
            for ann in annotations:
                print(f"  - {ann.term} ({ann.category.value}, 复杂度: {ann.complexity})")
                total_unique_terms.add(ann.term)
                
                # 统计类别
                if ann.category not in category_stats:
                    category_stats[ann.category] = 0
                category_stats[ann.category] += 1
                
                # 统计复杂度
                if ann.complexity not in complexity_stats:
                    complexity_stats[ann.complexity] = 0
                complexity_stats[ann.complexity] += 1
            
            total_terms += len(annotations)
            
            # 计算复杂度评分
            term_names = [ann.term for ann in annotations]
            complexity_score = processor.calculate_term_complexity(term_names)
            print(f"  复杂度评分: {complexity_score:.2f}/10")
        else:
            print("  未识别到密码学术语")
    
    # 总体统计
    print(f"\n" + "=" * 60)
    print("总体统计信息")
    print("=" * 60)
    print(f"总术语数量: {total_terms}")
    print(f"唯一术语数量: {len(total_unique_terms)}")
    
    print(f"\n类别分布:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category.value}: {count}")
    
    print(f"\n复杂度分布:")
    for complexity, count in sorted(complexity_stats.items()):
        print(f"  复杂度 {complexity}: {count}")
    
    # 高复杂度术语统计
    high_complexity_terms = [term for term in total_unique_terms 
                           if term in processor.complexity_mapping 
                           and processor.complexity_mapping[term] >= 6]
    
    print(f"\n高复杂度术语 (≥6级, {len(high_complexity_terms)}个):")
    for term in sorted(high_complexity_terms):
        complexity = processor.complexity_mapping[term]
        print(f"  - {term} (复杂度: {complexity})")


def demo_category_coverage():
    """演示各类别术语覆盖情况"""
    print(f"\n" + "=" * 60)
    print("术语类别覆盖情况")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    stats = processor.get_term_statistics()
    
    print(f"词典总览:")
    print(f"  唯一术语数量: {stats['total_unique_terms']}")
    print(f"  平均复杂度: {stats['average_complexity']:.2f}")
    
    print(f"\n各类别术语数量:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
    
    print(f"\n复杂度分布:")
    for complexity, count in sorted(stats['complexity_distribution'].items()):
        print(f"  复杂度 {complexity}: {count}")
    
    # 展示每个类别的代表性术语
    print(f"\n各类别代表性术语:")
    
    category_examples = {
        CryptoCategory.SYMMETRIC_ENCRYPTION: ["AES", "ChaCha20", "GCM", "分组密码"],
        CryptoCategory.ASYMMETRIC_ENCRYPTION: ["RSA", "椭圆曲线", "Ed25519", "ECDH"],
        CryptoCategory.HASH_FUNCTION: ["SHA-256", "SHA-3", "BLAKE2", "Merkle树"],
        CryptoCategory.DIGITAL_SIGNATURE: ["ECDSA", "EdDSA", "盲签名", "环签名"],
        CryptoCategory.KEY_MANAGEMENT: ["PKI", "HSM", "Argon2", "PBKDF2"],
        CryptoCategory.CRYPTOGRAPHIC_PROTOCOL: ["TLS", "SSH", "OAuth", "JWT"],
        CryptoCategory.CRYPTANALYSIS: ["差分分析", "侧信道攻击", "时间攻击", "功耗分析"],
        CryptoCategory.BLOCKCHAIN: ["区块链", "智能合约", "DeFi", "NFT"],
        CryptoCategory.OTHER: ["零知识证明", "同态加密", "多方安全计算", "秘密分享"]
    }
    
    for category, examples in category_examples.items():
        available_examples = [term for term in examples if term in processor.crypto_dict]
        if available_examples:
            print(f"  {category.value}: {', '.join(available_examples[:4])}")


def demo_professional_enhancement():
    """演示专业性提升效果"""
    print(f"\n" + "=" * 60)
    print("专业性提升效果演示")
    print("=" * 60)
    
    processor = CryptoTermProcessor()
    
    # 对比基础文本和高级文本
    basic_text = "RSA是一种加密算法，用于保护数据安全。"
    advanced_text = """
    RSA是一种基于大整数分解难题的非对称加密算法。它使用公钥和私钥对，
    支持数字签名和密钥交换。相比之下，AES是对称加密算法，使用相同密钥
    进行加密解密。现代系统常采用混合加密方案：用RSA交换AES密钥，
    用AES加密实际数据。在区块链应用中，ECDSA椭圆曲线数字签名算法
    因其高效性而被广泛采用。
    """
    
    print("基础文本分析:")
    print(f"文本: {basic_text}")
    basic_annotations = processor.identify_crypto_terms(basic_text)
    basic_complexity = processor.calculate_term_complexity([ann.term for ann in basic_annotations])
    print(f"识别术语数: {len(basic_annotations)}")
    print(f"复杂度评分: {basic_complexity:.2f}/10")
    
    print(f"\n高级文本分析:")
    print(f"文本: {advanced_text.strip()}")
    advanced_annotations = processor.identify_crypto_terms(advanced_text)
    advanced_complexity = processor.calculate_term_complexity([ann.term for ann in advanced_annotations])
    print(f"识别术语数: {len(advanced_annotations)}")
    print(f"复杂度评分: {advanced_complexity:.2f}/10")
    
    print(f"\n专业性提升:")
    print(f"术语数量提升: {len(advanced_annotations) - len(basic_annotations)}个")
    print(f"复杂度提升: {advanced_complexity - basic_complexity:.2f}分")
    
    # 展示高级文本中的术语
    print(f"\n高级文本识别的术语:")
    for ann in advanced_annotations:
        print(f"  - {ann.term} ({ann.category.value}, 复杂度: {ann.complexity})")


def main():
    """主演示函数"""
    print("高级密码学术语处理系统演示")
    print("=" * 60)
    
    try:
        demo_advanced_terms()
        demo_category_coverage()
        demo_professional_enhancement()
        
        print(f"\n" + "=" * 60)
        print("高级演示完成！")
        print("=" * 60)
        print(f"系统现已支持 80+ 专业密码学术语")
        print(f"涵盖 9 大类别，复杂度范围 1-8 级")
        print(f"包含最新的密码学技术和区块链概念")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()