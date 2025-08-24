#!/usr/bin/env python3
"""
CLI工具测试
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
from click.testing import CliRunner

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cli_tools_simple import cli, ConfigTemplate, ConfigValidator, load_training_data
from data_models import TrainingExample, DifficultyLevel


def test_config_template():
    """测试配置模板生成"""
    print("测试配置模板生成...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 生成配置模板
        template = ConfigTemplate.generate_training_config_template()
        
        # 验证模板结构
        required_sections = ["model", "training", "lora", "data", "parallel", "system"]
        for section in required_sections:
            assert section in template, f"缺少配置节: {section}"
        
        print("✓ 配置模板结构验证通过")
        
        # 保存模板
        template_file = Path(temp_dir) / "test_config.yaml"
        ConfigTemplate.save_template(template, str(template_file))
        
        assert template_file.exists(), "配置模板文件未创建"
        print("✓ 配置模板保存成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置模板测试失败: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir)


def test_config_validator():
    """测试配置验证器"""
    print("\n测试配置验证器...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建有效配置文件
        valid_config = ConfigTemplate.generate_training_config_template()
        valid_config_file = Path(temp_dir) / "valid_config.yaml"
        ConfigTemplate.save_template(valid_config, str(valid_config_file))
        
        # 验证有效配置
        is_valid, errors = ConfigValidator.validate_config_file(str(valid_config_file))
        assert is_valid, f"有效配置验证失败: {errors}"
        print("✓ 有效配置验证通过")
        
        # 创建无效配置文件
        invalid_config = {"invalid": "config"}
        invalid_config_file = Path(temp_dir) / "invalid_config.yaml"
        ConfigTemplate.save_template(invalid_config, str(invalid_config_file))
        
        # 验证无效配置
        is_valid, errors = ConfigValidator.validate_config_file(str(invalid_config_file))
        assert not is_valid, "无效配置应该验证失败"
        assert len(errors) > 0, "应该有错误信息"
        print("✓ 无效配置验证通过")
        
        # 测试不存在的文件
        is_valid, errors = ConfigValidator.validate_config_file("nonexistent.yaml")
        assert not is_valid, "不存在的文件应该验证失败"
        print("✓ 不存在文件验证通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置验证器测试失败: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir)


def test_data_loading():
    """测试数据加载功能"""
    print("\n测试数据加载功能...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建测试数据
        test_data = [
            {
                "instruction": "什么是AES加密？",
                "input": "",
                "output": "AES是高级加密标准。",
                "thinking": "<thinking>需要解释AES的基本概念。</thinking>",
                "crypto_terms": ["AES", "加密"],
                "difficulty": 1
            },
            {
                "instruction": "比较RSA和ECC",
                "input": "在现代密码学中",
                "output": "RSA基于大整数分解，ECC基于椭圆曲线。",
                "crypto_terms": ["RSA", "ECC"],
                "difficulty": 2
            }
        ]
        
        # 保存为JSON文件
        json_file = Path(temp_dir) / "test_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # 测试加载
        loaded_data = load_training_data(str(json_file))
        
        assert len(loaded_data) == 2, f"期望2条数据，实际{len(loaded_data)}条"
        assert isinstance(loaded_data[0], TrainingExample), "数据类型不正确"
        assert loaded_data[0].instruction == "什么是AES加密？", "指令内容不匹配"
        assert loaded_data[0].has_thinking(), "应该包含thinking数据"
        assert loaded_data[0].difficulty_level == DifficultyLevel.BEGINNER, "难度级别不匹配"
        
        print("✓ JSON数据加载测试通过")
        
        # 保存为JSONL文件
        jsonl_file = Path(temp_dir) / "test_data.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 测试JSONL加载
        loaded_jsonl_data = load_training_data(str(jsonl_file))
        assert len(loaded_jsonl_data) == 2, "JSONL数据加载数量不正确"
        
        print("✓ JSONL数据加载测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir)


def test_cli_commands():
    """测试CLI命令"""
    print("\n测试CLI命令...")
    
    temp_dir = tempfile.mkdtemp()
    runner = CliRunner()
    
    try:
        # 测试init-config命令
        config_file = Path(temp_dir) / "test_config.yaml"
        result = runner.invoke(cli, ['init-config', '--output', str(config_file)])
        
        assert result.exit_code == 0, f"init-config命令失败: {result.output}"
        assert config_file.exists(), "配置文件未创建"
        print("✓ init-config命令测试通过")
        
        # 测试validate-config命令
        result = runner.invoke(cli, ['validate-config', str(config_file)])
        assert result.exit_code == 0, f"validate-config命令失败: {result.output}"
        print("✓ validate-config命令测试通过")
        
        # 测试list-gpus命令
        result = runner.invoke(cli, ['list-gpus'])
        # list-gpus命令应该正常运行，无论是否有GPU
        print("✓ list-gpus命令测试通过")
        
        # 测试inspect-data命令
        # 首先创建测试数据文件
        test_data = [
            {
                "instruction": "测试问题",
                "input": "",
                "output": "测试回答",
                "crypto_terms": ["测试"],
                "difficulty": 1
            }
        ]
        
        data_file = Path(temp_dir) / "test_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        result = runner.invoke(cli, ['inspect-data', str(data_file), '--sample', '1'])
        assert result.exit_code == 0, f"inspect-data命令失败: {result.output}"
        print("✓ inspect-data命令测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI命令测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir)


def test_config_creation_functions():
    """测试配置创建函数"""
    print("\n测试配置创建函数...")
    
    try:
        from cli_tools_simple import (
            create_training_config, create_data_config, 
            create_lora_config, create_parallel_config, create_system_config
        )
        
        # 创建测试配置
        test_config = ConfigTemplate.generate_training_config_template()
        
        # 测试训练配置创建
        training_config = create_training_config(test_config)
        assert training_config.num_train_epochs == 3, "训练轮次不匹配"
        assert training_config.learning_rate == 2e-4, "学习率不匹配"
        print("✓ 训练配置创建测试通过")
        
        # 测试数据配置创建
        data_config = create_data_config(test_config)
        assert data_config.train_split_ratio == 0.9, "训练分割比例不匹配"
        print("✓ 数据配置创建测试通过")
        
        # 测试LoRA配置创建
        lora_config = create_lora_config(test_config)
        assert lora_config.rank == 8, "LoRA rank不匹配"
        assert lora_config.alpha == 16, "LoRA alpha不匹配"
        print("✓ LoRA配置创建测试通过")
        
        # 测试并行配置创建
        parallel_config = create_parallel_config(test_config)
        assert parallel_config.data_parallel_size == 1, "并行大小不匹配"
        print("✓ 并行配置创建测试通过")
        
        # 测试系统配置创建
        system_config = create_system_config(test_config)
        assert system_config.cache_dir == './cache', "缓存目录不匹配"
        print("✓ 系统配置创建测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置创建函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CLI工具测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_config_template())
    test_results.append(test_config_validator())
    test_results.append(test_data_loading())
    test_results.append(test_cli_commands())
    test_results.append(test_config_creation_functions())
    
    # 汇总结果
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
        exit(0)
    else:
        print("❌ 部分测试失败")
        exit(1)