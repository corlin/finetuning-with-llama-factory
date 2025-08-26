#!/usr/bin/env python3
"""
快速测试已实现模块的核心功能
不需要下载大模型，只测试模块导入和基本功能
"""

import os
import sys
import json
import torch

# 添加src路径
sys.path.append('src')

def test_module_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    modules_to_test = [
        ("data_models", "TrainingExample"),
        ("config_manager", "TrainingConfig"),
        ("gpu_utils", "GPUDetector"),
        ("memory_manager", "MemoryManager"),
        ("chinese_nlp_processor", "ChineseNLPProcessor"),
        ("crypto_term_processor", "CryptoTermProcessor"),
        ("parallel_strategy_recommender", "ParallelStrategyRecommender"),
        ("training_monitor", "TrainingMonitor"),
        ("dataset_splitter", "DatasetSplitter"),
    ]
    
    results = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            results.append(True)
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n模块导入测试: {passed}/{total} 通过")
    return passed == total

def test_data_models():
    """测试数据模型"""
    print("\n🔍 测试数据模型...")
    
    try:
        from data_models import TrainingExample, ThinkingExample, CryptoTerm, ChineseMetrics
        
        # 测试TrainingExample
        example = TrainingExample(
            instruction="什么是AES？",
            input="",
            output="AES是高级加密标准",
            thinking="<thinking>这是关于对称加密的问题</thinking>",
            crypto_terms=["AES", "对称加密"],
            difficulty_level=1,
            source_file="test.json"
        )
        print(f"✅ TrainingExample创建成功: {example.instruction}")
        
        # 测试CryptoTerm
        term = CryptoTerm(
            term="AES",
            definition="高级加密标准",
            category="对称加密",
            complexity=2
        )
        print(f"✅ CryptoTerm创建成功: {term.term}")
        
        return True
    except Exception as e:
        print(f"❌ 数据模型测试失败: {e}")
        return False

def test_config_manager():
    """测试配置管理器"""
    print("\n🔍 测试配置管理器...")
    
    try:
        from config_manager import TrainingConfig
        
        config = TrainingConfig()
        print(f"✅ TrainingConfig创建成功")
        print(f"  输出目录: {config.output_dir}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  批次大小: {config.per_device_train_batch_size}")
        
        return True
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False

def test_gpu_utils():
    """测试GPU工具"""
    print("\n🔍 测试GPU工具...")
    
    try:
        from gpu_utils import GPUDetector
        
        detector = GPUDetector()
        gpu_info = detector.get_all_gpu_info()
        
        print(f"✅ GPU检测成功，发现 {len(gpu_info)} 个GPU")
        for i, gpu in enumerate(gpu_info):
            print(f"  GPU {i}: {gpu.name}, 内存: {gpu.total_memory}MB")
        
        return True
    except Exception as e:
        print(f"❌ GPU工具测试失败: {e}")
        return False

def test_chinese_processor():
    """测试中文处理器"""
    print("\n🔍 测试中文处理器...")
    
    try:
        from chinese_nlp_processor import ChineseNLPProcessor
        
        processor = ChineseNLPProcessor()
        
        # 测试文本分词
        test_text = "什么是对称加密算法？它有哪些特点？"
        tokens = processor.segment_text(test_text)
        print(f"✅ 文本分词: '{test_text}'")
        print(f"  分词结果: {[token.word for token in tokens[:5]]}...")  # 只显示前5个
        
        # 测试文本质量评估
        try:
            quality = processor.assess_text_quality(test_text)
            print(f"✅ 文本质量评估: 综合评分 {quality.overall_quality():.2f}")
        except:
            print("⚠️ 文本质量评估功能不可用")
        
        return True
    except Exception as e:
        print(f"❌ 中文处理器测试失败: {e}")
        return False

def test_crypto_processor():
    """测试密码学处理器"""
    print("\n🔍 测试密码学处理器...")
    
    try:
        from crypto_term_processor import CryptoTermProcessor
        
        processor = CryptoTermProcessor()
        
        test_text = "AES是一种对称加密算法，RSA是非对称加密算法，SHA-256是哈希函数。"
        terms = processor.identify_crypto_terms(test_text)
        
        print(f"✅ 密码学术语识别成功")
        print(f"  输入: {test_text}")
        print(f"  识别的术语: {[term.term for term in terms]}")
        
        return True
    except Exception as e:
        print(f"❌ 密码学处理器测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        data_path = "final_demo_output/data/crypto_qa_dataset_train.json"
        
        if not os.path.exists(data_path):
            print(f"⚠️ 数据文件不存在: {data_path}")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 数据加载成功，共 {len(data)} 条记录")
        
        # 分析数据结构
        if data:
            sample = data[0]
            keys = list(sample.keys())
            print(f"  数据字段: {keys}")
            print(f"  样本指令: {sample.get('instruction', '')[:50]}...")
            
            # 统计thinking数据
            thinking_count = sum(1 for item in data if '<thinking>' in item.get('output', ''))
            print(f"  包含thinking的样本: {thinking_count}/{len(data)}")
        
        return True
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_pytorch_environment():
    """测试PyTorch环境"""
    print("\n🔍 测试PyTorch环境...")
    
    try:
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // 1024**2
                print(f"  GPU {i}: {gpu_name}, {gpu_memory}MB")
        
        # 测试基本张量操作
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print(f"✅ 张量运算测试通过: {z.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 快速测试已实现模块")
    print("=" * 40)
    
    tests = [
        ("模块导入", test_module_imports),
        ("数据模型", test_data_models),
        ("配置管理器", test_config_manager),
        ("GPU工具", test_gpu_utils),
        ("中文处理器", test_chinese_processor),
        ("密码学处理器", test_crypto_processor),
        ("数据加载", test_data_loading),
        ("PyTorch环境", test_pytorch_environment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed >= len(results) * 0.8:  # 80%通过率
        print("🎉 大部分测试通过！环境基本可用。")
        print("\n📝 下一步建议:")
        print("1. 运行 'uv run python test_direct_finetuning.py' 进行完整测试")
        print("2. 运行 'uv run python direct_finetuning_with_existing_modules.py' 开始微调")
        return True
    else:
        print("⚠️ 多个测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)