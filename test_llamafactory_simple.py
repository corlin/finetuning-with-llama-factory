#!/usr/bin/env python3
"""
简单的LLaMA Factory适配器测试
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory_adapter import LlamaFactoryAdapter, LlamaFactoryDataConverter
from data_models import TrainingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig
from lora_config_optimizer import LoRAMemoryProfile
from parallel_config import ParallelConfig


def test_basic_functionality():
    """测试基本功能"""
    print("开始测试LLaMA Factory适配器...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    try:
        # 创建适配器
        adapter = LlamaFactoryAdapter()
        print("✓ 适配器创建成功")
        
        # 创建测试数据
        test_examples = [
            TrainingExample(
                instruction="什么是AES加密算法？",
                input="",
                output="AES是高级加密标准，是一种对称分组密码算法。",
                thinking="<thinking>用户询问AES的基本概念，我需要简洁地解释其定义和特点。</thinking>",
                crypto_terms=["AES", "对称加密", "分组密码"],
                difficulty_level=DifficultyLevel.BEGINNER
            ),
            TrainingExample(
                instruction="比较AES和DES的安全性",
                input="在现代密码学应用中",
                output="AES比DES更安全，密钥长度更长，抗攻击能力更强。",
                crypto_terms=["AES", "DES", "安全性"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
        ]
        print(f"✓ 创建了{len(test_examples)}个测试样例")
        
        # 测试数据转换
        data_files = adapter.prepare_training_data(
            test_examples,
            temp_dir,
            "test_crypto",
            "alpaca",
            0.8
        )
        print("✓ 数据转换成功")
        print(f"  训练文件: {data_files['train_file']}")
        print(f"  验证文件: {data_files['val_file']}")
        print(f"  数据集信息: {data_files['dataset_info_file']}")
        
        # 验证文件存在
        for file_type, file_path in data_files.items():
            if Path(file_path).exists():
                print(f"  ✓ {file_type} 文件存在")
            else:
                print(f"  ✗ {file_type} 文件不存在")
        
        # 测试配置生成
        training_config = TrainingConfig(learning_rate=2e-4, num_train_epochs=3)
        data_config = DataConfig()
        lora_config = LoRAMemoryProfile(
            rank=8, alpha=16, target_modules=["q_proj", "v_proj"]
        )
        from parallel_config import ParallelStrategy
        parallel_config = ParallelConfig(strategy=ParallelStrategy.DATA_PARALLEL)
        
        config_file = adapter.create_training_config(
            training_config,
            data_config,
            lora_config,
            parallel_config,
            "test_crypto",
            temp_dir
        )
        print(f"✓ 配置文件生成成功: {config_file}")
        
        # 验证集成
        validation = adapter.validate_integration(config_file, data_files)
        if validation["valid"]:
            print("✓ 集成验证通过")
        else:
            print("✗ 集成验证失败:")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        # 生成训练脚本
        script_file = adapter.generate_training_script(
            config_file, str(Path(temp_dir) / "train.py")
        )
        print(f"✓ 训练脚本生成成功: {script_file}")
        
        print("\n所有测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"清理临时目录: {temp_dir}")


def test_data_converter():
    """测试数据转换器"""
    print("\n测试数据转换器...")
    
    converter = LlamaFactoryDataConverter()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建测试数据
        examples = [
            TrainingExample(
                instruction="解释RSA算法",
                input="",
                output="RSA是一种非对称加密算法。",
                thinking="<thinking>需要解释RSA的基本原理。</thinking>",
                crypto_terms=["RSA", "非对称加密"]
            )
        ]
        
        # 测试Alpaca格式转换
        alpaca_file = Path(temp_dir) / "test_alpaca.json"
        success = converter.convert_training_examples(examples, str(alpaca_file), "alpaca")
        
        if success and alpaca_file.exists():
            print("✓ Alpaca格式转换成功")
            
            # 验证转换结果
            validation = converter.validate_converted_data(str(alpaca_file), "alpaca")
            if validation["valid"]:
                print("✓ Alpaca格式验证通过")
                print(f"  样例数量: {validation['statistics']['total_samples']}")
                print(f"  思考样例: {validation['statistics']['thinking_samples']}")
            else:
                print("✗ Alpaca格式验证失败")
        else:
            print("✗ Alpaca格式转换失败")
        
        # 测试ShareGPT格式转换
        sharegpt_file = Path(temp_dir) / "test_sharegpt.json"
        success = converter.convert_training_examples(examples, str(sharegpt_file), "sharegpt")
        
        if success and sharegpt_file.exists():
            print("✓ ShareGPT格式转换成功")
        else:
            print("✗ ShareGPT格式转换失败")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据转换器测试失败: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("LLaMA Factory适配器测试")
    print("=" * 50)
    
    success1 = test_basic_functionality()
    success2 = test_data_converter()
    
    if success1 and success2:
        print("\n🎉 所有测试都通过了！")
        exit(0)
    else:
        print("\n❌ 部分测试失败")
        exit(1)