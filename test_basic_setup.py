#!/usr/bin/env python3
"""
基础设置测试脚本
测试项目结构和基本功能，不依赖外部库
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """测试项目结构"""
    print("=== 测试项目结构 ===")
    
    required_files = [
        "pyproject.toml",
        "README.md", 
        "setup.py",
        "src/__init__.py",
        "src/gpu_utils.py",
        "src/model_config.py",
        "src/config_manager.py",
        "src/environment_setup.py",
        "scripts/check_environment.py",
        "scripts/train.py",
        "tests/test_environment.py"
    ]
    
    required_dirs = [
        "src",
        "scripts", 
        "tests",
        "data",
        "configs",
        "output",
        "logs",
        "cache"
    ]
    
    # 检查文件
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    # 检查目录
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}/")
    
    if missing_files:
        print(f"\n❌ 缺失文件: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"\n❌ 缺失目录: {missing_dirs}")
        return False
    
    print("\n✅ 项目结构检查通过")
    return True

def test_python_imports():
    """测试Python模块导入"""
    print("\n=== 测试模块导入 ===")
    
    # 添加src到路径
    sys.path.insert(0, str(Path("src")))
    
    modules_to_test = [
        "gpu_utils",
        "model_config", 
        "config_manager",
        "environment_setup"
    ]
    
    import_results = {}
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
            import_results[module_name] = True
        except ImportError as e:
            print(f"⚠️  {module_name}: {e}")
            import_results[module_name] = False
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            import_results[module_name] = False
    
    success_count = sum(import_results.values())
    total_count = len(import_results)
    
    print(f"\n模块导入结果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("✅ 所有模块导入成功")
        return True
    else:
        print("⚠️  部分模块导入失败（可能缺少依赖）")
        return False

def test_config_files():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    
    # 检查pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding='utf-8')
        if "llama-factory-finetuning" in content:
            print("✓ pyproject.toml 配置正确")
        else:
            print("❌ pyproject.toml 配置有误")
            return False
    else:
        print("❌ pyproject.toml 不存在")
        return False
    
    # 检查README.md
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding='utf-8')
        if "Qwen3-4B-Thinking" in content:
            print("✓ README.md 内容正确")
        else:
            print("❌ README.md 内容有误")
            return False
    else:
        print("❌ README.md 不存在")
        return False
    
    print("✅ 配置文件检查通过")
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    sys.path.insert(0, str(Path("src")))
    
    try:
        # 测试数据类创建
        from model_config import QwenModelConfig, LoRATrainingConfig
        
        model_config = QwenModelConfig()
        lora_config = LoRATrainingConfig()
        
        print(f"✓ 模型配置: {model_config.model_name}")
        print(f"✓ LoRA配置: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # 测试配置管理（不依赖外部库的部分）
        from config_manager import TrainingConfig, DataConfig
        
        training_config = TrainingConfig()
        data_config = DataConfig()
        
        print(f"✓ 训练配置: batch_size={training_config.per_device_train_batch_size}")
        print(f"✓ 数据配置: format={data_config.data_format}")
        
        print("✅ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== LLaMA Factory Finetuning 基础设置测试 ===")
    print()
    
    tests = [
        ("项目结构", test_project_structure),
        ("模块导入", test_python_imports),
        ("配置文件", test_config_files),
        ("基本功能", test_basic_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    print("\n" + "="*50)
    print("测试结果汇总:")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n总体结果: {success_count}/{total_count} 测试通过")
    
    if success_count == total_count:
        print("\n🎉 所有基础测试通过！")
        print("\n下一步:")
        print("1. 安装依赖: uv sync")
        print("2. 运行完整设置: python setup.py")
        print("3. 检查环境: python scripts/check_environment.py")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())