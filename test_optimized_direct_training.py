#!/usr/bin/env python3
"""
测试优化后的直接训练流程
验证中文NLP处理器、密码学术语处理器、训练监控器和并行策略推荐器的集成
"""

import os
import sys
import json
import torch
from pathlib import Path

# 添加src目录到路径
sys.path.append('src')

def test_module_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from chinese_nlp_processor import ChineseNLPProcessor
        print("✅ ChineseNLPProcessor 导入成功")
    except Exception as e:
        print(f"❌ ChineseNLPProcessor 导入失败: {e}")
        return False
    
    try:
        from crypto_term_processor import CryptoTermProcessor
        print("✅ CryptoTermProcessor 导入成功")
    except Exception as e:
        print(f"❌ CryptoTermProcessor 导入失败: {e}")
        return False
    
    try:
        from training_monitor import TrainingMonitor
        print("✅ TrainingMonitor 导入成功")
    except Exception as e:
        print(f"❌ TrainingMonitor 导入失败: {e}")
        return False
    
    try:
        from parallel_strategy_recommender import ParallelStrategyRecommender
        print("✅ ParallelStrategyRecommender 导入成功")
    except Exception as e:
        print(f"❌ ParallelStrategyRecommender 导入失败: {e}")
        return False
    
    return True


def test_chinese_nlp_processor():
    """测试中文NLP处理器"""
    print("\n🔍 测试中文NLP处理器...")
    
    try:
        from chinese_nlp_processor import ChineseNLPProcessor
        
        processor = ChineseNLPProcessor()
        
        # 测试文本预处理
        test_text = "这是一个测试文本，包含AES加密算法和RSA数字签名。"
        processed_text = processor.preprocess_for_training(test_text)
        print(f"原文: {test_text}")
        print(f"预处理后: {processed_text}")
        
        # 测试文本质量评估
        quality_metrics = processor.assess_text_quality(test_text)
        print(f"文本质量评分: {quality_metrics.overall_quality():.3f}")
        
        # 测试密码学术语提取
        crypto_terms = processor.extract_crypto_terms_from_text(test_text)
        print(f"提取的密码学术语: {crypto_terms}")
        
        print("✅ 中文NLP处理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 中文NLP处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crypto_term_processor():
    """测试密码学术语处理器"""
    print("\n🔍 测试密码学术语处理器...")
    
    try:
        from crypto_term_processor import CryptoTermProcessor
        
        processor = CryptoTermProcessor()
        
        # 测试术语提取
        test_text = "RSA算法是一种非对称加密算法，使用公钥和私钥进行加密解密。"
        terms = processor.identify_crypto_terms(test_text)
        
        print(f"测试文本: {test_text}")
        print(f"提取的术语数量: {len(terms)}")
        for term in terms:
            print(f"  - {term.term} (复杂度: {term.complexity}, 类别: {term.category.value})")
        
        print("✅ 密码学术语处理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 密码学术语处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_monitor():
    """测试训练监控器"""
    print("\n🔍 测试训练监控器...")
    
    try:
        from training_monitor import TrainingMonitor
        
        # 创建临时监控器
        gpu_ids = [0] if torch.cuda.is_available() else []
        monitor = TrainingMonitor(
            gpu_ids=gpu_ids,
            log_dir="test_output/training_logs",
            save_interval=10
        )
        
        # 测试启动和停止
        monitor.start_monitoring()
        print("✅ 训练监控器启动成功")
        
        # 模拟训练步骤更新
        monitor.update_training_step(
            epoch=1,
            global_step=1,
            train_loss=2.5,
            learning_rate=1e-4,
            val_loss=2.3
        )
        print("✅ 训练步骤更新成功")
        
        # 获取当前指标
        current_metrics = monitor.get_current_metrics()
        if current_metrics:
            print(f"当前训练损失: {current_metrics.train_loss:.4f}")
        
        monitor.stop_monitoring()
        print("✅ 训练监控器停止成功")
        
        print("✅ 训练监控器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练监控器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_strategy_recommender():
    """测试并行策略推荐器"""
    print("\n🔍 测试并行策略推荐器...")
    
    try:
        from parallel_strategy_recommender import ParallelStrategyRecommender
        
        recommender = ParallelStrategyRecommender()
        
        # 测试策略推荐
        recommendation = recommender.recommend_strategy(
            batch_size=4,
            sequence_length=2048,
            enable_lora=True,
            lora_rank=64
        )
        
        print(f"推荐策略: {recommendation.strategy.value}")
        print(f"置信度: {recommendation.confidence:.2f}")
        print(f"数据并行: {recommendation.config.data_parallel}")
        print(f"模型并行: {recommendation.config.model_parallel}")
        print(f"梯度累积步数: {recommendation.config.gradient_accumulation_steps}")
        
        if recommendation.reasoning:
            print("推荐理由:")
            for reason in recommendation.reasoning:
                print(f"  - {reason}")
        
        if recommendation.warnings:
            print("警告:")
            for warning in recommendation.warnings:
                print(f"  ⚠️ {warning}")
        
        print("✅ 并行策略推荐器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 并行策略推荐器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimized_training_integration():
    """测试优化后的训练集成"""
    print("\n🔍 测试优化后的训练集成...")
    
    try:
        # 导入优化后的训练器
        from direct_finetuning_with_existing_modules import DirectTrainingConfig, DirectTrainer
        
        # 创建测试配置
        config = DirectTrainingConfig()
        config.data_path = "data/raw"  # 假设存在测试数据
        config.output_dir = "test_output/optimized_training"
        config.num_epochs = 1  # 只测试一个epoch
        config.batch_size = 1
        config.max_seq_length = 512  # 减小序列长度以节省内存
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 创建训练器
        trainer = DirectTrainer(config)
        
        print("✅ 优化后的训练器创建成功")
        
        # 测试各个组件是否正确初始化
        if hasattr(trainer, 'parallel_recommender') and trainer.parallel_recommender:
            print("✅ 并行策略推荐器集成成功")
        
        if hasattr(trainer, 'training_monitor') and trainer.training_monitor:
            print("✅ 训练监控器集成成功")
        
        print("✅ 优化后的训练集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 优化后的训练集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_data():
    """创建测试数据"""
    print("\n🔍 创建测试数据...")
    
    test_data_dir = Path("data/raw")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试markdown文件
    test_content = """# 密码学测试数据

### Q1: 什么是AES加密算法？

<thinking>
用户询问AES加密算法，这是一个基础的密码学问题。我需要解释AES的基本概念、特点和应用。

AES（Advanced Encryption Standard）是一种对称加密算法，也就是说加密和解密使用相同的密钥。它是目前最广泛使用的加密算法之一。

我应该从以下几个方面来回答：
1. AES的基本定义
2. AES的特点（对称加密、分组密码等）
3. AES的密钥长度选项
4. AES的应用场景
</thinking>

AES（Advanced Encryption Standard，高级加密标准）是一种对称加密算法，由美国国家标准与技术研究院（NIST）在2001年确立为加密标准。

AES的主要特点：
1. **对称加密**：加密和解密使用相同的密钥
2. **分组密码**：将明文分成128位的块进行处理
3. **密钥长度**：支持128位、192位和256位三种密钥长度
4. **安全性高**：目前被认为是安全可靠的加密算法

AES广泛应用于：
- HTTPS网站加密
- 文件和磁盘加密
- VPN连接
- 无线网络安全（WPA2/WPA3）

### Q2: RSA和AES有什么区别？

<thinking>
这是一个比较两种不同类型加密算法的问题。RSA是非对称加密算法，AES是对称加密算法，它们在原理、用途、性能等方面都有显著差异。

我需要从以下几个维度来比较：
1. 加密类型（对称vs非对称）
2. 密钥管理
3. 加密速度
4. 安全性基础
5. 典型应用场景
6. 密钥长度要求
</thinking>

RSA和AES是两种不同类型的加密算法，主要区别如下：

**加密类型**：
- RSA：非对称加密算法，使用公钥加密、私钥解密
- AES：对称加密算法，加密解密使用相同密钥

**性能对比**：
- RSA：加密速度较慢，适合加密少量数据
- AES：加密速度快，适合大量数据加密

**密钥管理**：
- RSA：密钥分发相对简单，公钥可以公开
- AES：密钥分发复杂，需要安全信道传输密钥

**典型应用**：
- RSA：数字签名、密钥交换、身份认证
- AES：数据加密、文件保护、通信加密

**实际应用中的结合**：
通常将RSA和AES结合使用，用RSA加密AES密钥，用AES加密实际数据，这样既保证了安全性又提高了效率。
"""
    
    test_file = test_data_dir / "test_crypto_qa.md"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"✅ 测试数据已创建: {test_file}")
    return True


def main():
    """主测试函数"""
    print("🎯 测试优化后的直接训练流程")
    print("=" * 60)
    
    # 创建测试输出目录
    os.makedirs("test_output", exist_ok=True)
    
    # 创建测试数据
    if not create_test_data():
        print("❌ 创建测试数据失败")
        return False
    
    # 测试各个组件
    tests = [
        ("模块导入", test_module_imports),
        ("中文NLP处理器", test_chinese_nlp_processor),
        ("密码学术语处理器", test_crypto_term_processor),
        ("训练监控器", test_training_monitor),
        ("并行策略推荐器", test_parallel_strategy_recommender),
        ("优化后的训练集成", test_optimized_training_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
    
    # 测试结果汇总
    print(f"\n{'='*60}")
    print(f"📊 测试结果汇总: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！优化后的直接训练流程集成成功")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} 个测试失败，需要修复")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)