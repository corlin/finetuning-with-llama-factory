#!/usr/bin/env python3
"""
CLI工具使用演示

展示如何使用Qwen3-4B-Thinking密码学微调工具的完整工作流程
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cli_tools_simple import (
    ConfigTemplate, ConfigValidator, load_training_data,
    create_training_config, create_data_config, create_lora_config,
    create_parallel_config, create_system_config
)
from data_models import TrainingExample, DifficultyLevel
from training_pipeline import TrainingPipelineOrchestrator
from gpu_utils import GPUDetector


def demo_complete_workflow():
    """演示完整的CLI工具工作流程"""
    print("🚀 Qwen3-4B-Thinking 密码学微调工具演示")
    print("=" * 60)
    
    # 创建临时工作目录
    work_dir = Path(tempfile.mkdtemp(prefix="qwen_demo_"))
    print(f"📁 工作目录: {work_dir}")
    
    try:
        # 步骤1: 生成配置文件模板
        print("\n📝 步骤1: 生成配置文件模板")
        config_file = work_dir / "config.yaml"
        template = ConfigTemplate.generate_training_config_template()
        ConfigTemplate.save_template(template, str(config_file))
        
        # 步骤2: 验证配置文件
        print("\n🔍 步骤2: 验证配置文件")
        is_valid, errors = ConfigValidator.validate_config_file(str(config_file))
        if is_valid:
            print("✅ 配置文件验证通过")
        else:
            print("❌ 配置文件验证失败:")
            for error in errors:
                print(f"  • {error}")
        
        # 步骤3: 创建示例训练数据
        print("\n📊 步骤3: 创建示例训练数据")
        training_data = create_sample_training_data()
        data_file = work_dir / "training_data.json"
        
        # 保存训练数据
        data_dict = [example.to_dict() for example in training_data]
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 创建了 {len(training_data)} 条训练数据")
        
        # 步骤4: 检查训练数据
        print("\n🔍 步骤4: 检查训练数据")
        loaded_data = load_training_data(str(data_file))
        
        print(f"数据统计:")
        print(f"  总样本数: {len(loaded_data)}")
        
        thinking_count = sum(1 for ex in loaded_data if ex.has_thinking())
        print(f"  包含thinking数据: {thinking_count}")
        
        difficulty_counts = {}
        for example in loaded_data:
            difficulty = example.difficulty_level.name
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        for difficulty, count in difficulty_counts.items():
            print(f"  难度-{difficulty}: {count}")
        
        # 步骤5: 检测GPU环境
        print("\n🔍 步骤5: 检测GPU环境")
        detector = GPUDetector()
        gpu_infos = detector.get_all_gpu_info()
        
        if gpu_infos:
            print(f"检测到 {len(gpu_infos)} 个GPU:")
            for gpu in gpu_infos:
                print(f"  GPU {gpu.gpu_id}: {gpu.name} ({gpu.total_memory}MB)")
        else:
            print("未检测到GPU，将使用CPU模式")
        
        # 步骤6: 创建训练流水线配置
        print("\n⚙️ 步骤6: 创建训练流水线配置")
        
        # 加载配置文件
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建配置对象
        training_config = create_training_config(config)
        data_config = create_data_config(config)
        lora_config = create_lora_config(config)
        parallel_config = create_parallel_config(config)
        system_config = create_system_config(config)
        
        print("✅ 训练配置创建完成")
        print(f"  训练轮次: {training_config.num_train_epochs}")
        print(f"  学习率: {training_config.learning_rate}")
        print(f"  LoRA rank: {lora_config.rank}")
        print(f"  LoRA alpha: {lora_config.alpha}")
        
        # 步骤7: 创建训练流水线编排器
        print("\n🔧 步骤7: 创建训练流水线编排器")
        pipeline_id = "demo_pipeline"
        orchestrator = TrainingPipelineOrchestrator(
            pipeline_id, 
            output_dir=str(work_dir / "pipeline_output")
        )
        
        # 配置流水线
        orchestrator.configure_pipeline(
            loaded_data, training_config, data_config, 
            lora_config, parallel_config, system_config
        )
        
        print("✅ 训练流水线配置完成")
        
        # 步骤8: 模拟流水线执行（仅前几个阶段）
        print("\n🚀 步骤8: 模拟流水线执行")
        
        # 执行初始化阶段
        print("  执行初始化阶段...")
        if orchestrator._stage_initialization():
            print("  ✅ 初始化阶段完成")
        else:
            print("  ❌ 初始化阶段失败")
            return
        
        # 执行数据准备阶段
        print("  执行数据准备阶段...")
        if orchestrator._stage_data_preparation():
            print("  ✅ 数据准备阶段完成")
            print(f"    数据文件: {list(orchestrator.data_files.keys())}")
        else:
            print("  ❌ 数据准备阶段失败")
            return
        
        # 执行配置生成阶段
        print("  执行配置生成阶段...")
        if orchestrator._stage_config_generation():
            print("  ✅ 配置生成阶段完成")
            print(f"    配置文件: {list(orchestrator.config_files.keys())}")
        else:
            print("  ❌ 配置生成阶段失败")
            return
        
        # 步骤9: 显示生成的文件
        print("\n📁 步骤9: 生成的文件")
        output_dir = Path(work_dir) / "pipeline_output"
        
        if output_dir.exists():
            print("生成的目录结构:")
            for item in output_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(output_dir)
                    print(f"  📄 {rel_path}")
        
        # 步骤10: 生成系统报告
        print("\n📊 步骤10: 生成系统环境报告")
        report = detector.generate_system_report()
        
        report_file = work_dir / "system_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 系统报告已保存到: {report_file}")
        
        # 显示部分报告内容
        print("\n系统环境摘要:")
        report_lines = report.split('\n')
        for line in report_lines[:15]:  # 显示前15行
            print(f"  {line}")
        print("  ...")
        
        print(f"\n🎉 演示完成！所有文件保存在: {work_dir}")
        print("\n💡 下一步操作:")
        print("  1. 编辑配置文件以调整训练参数")
        print("  2. 准备更多训练数据")
        print("  3. 使用 CLI 工具开始实际训练:")
        print(f"     python src/cli_tools_simple.py train {data_file} {config_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 询问是否保留文件
        try:
            keep_files = input(f"\n是否保留演示文件？(y/N): ").lower().strip()
            if keep_files != 'y':
                shutil.rmtree(work_dir)
                print(f"🗑️ 已清理临时文件: {work_dir}")
            else:
                print(f"📁 文件保留在: {work_dir}")
        except KeyboardInterrupt:
            print(f"\n📁 文件保留在: {work_dir}")


def create_sample_training_data():
    """创建示例训练数据"""
    examples = [
        TrainingExample(
            instruction="什么是AES加密算法？",
            input="",
            output="AES（高级加密标准）是一种对称分组密码算法，使用128位分组长度和128、192或256位的密钥长度。它是目前最广泛使用的加密算法之一。",
            thinking="<thinking>用户询问AES的基本概念。我需要解释AES的全称、基本特性和应用情况。AES是Advanced Encryption Standard的缩写，是一种对称加密算法，意味着加密和解密使用相同的密钥。</thinking>",
            crypto_terms=["AES", "对称加密", "分组密码", "高级加密标准"],
            difficulty_level=DifficultyLevel.BEGINNER
        ),
        TrainingExample(
            instruction="比较RSA和ECC算法的优缺点",
            input="在现代密码学应用中",
            output="RSA和ECC都是非对称加密算法，但各有特点：\n\nRSA优点：成熟稳定、广泛支持、实现简单\nRSA缺点：密钥长度大、计算开销高\n\nECC优点：密钥长度短、计算效率高、安全性强\nECC缺点：实现复杂、专利问题、标准化程度低",
            thinking="<thinking>这是一个比较性问题，需要从多个维度分析RSA和ECC。RSA基于大整数分解难题，而ECC基于椭圆曲线离散对数难题。我需要比较它们的安全性、效率、实用性等方面。</thinking>",
            crypto_terms=["RSA", "ECC", "非对称加密", "椭圆曲线", "大整数分解"],
            difficulty_level=DifficultyLevel.INTERMEDIATE
        ),
        TrainingExample(
            instruction="解释数字签名的工作原理和安全性保证",
            input="",
            output="数字签名基于非对称密码学原理，提供身份认证、数据完整性和不可否认性保证。工作流程：\n1. 发送方用私钥对消息摘要进行签名\n2. 接收方用发送方公钥验证签名\n3. 验证成功则确认消息来源和完整性\n\n安全性基于私钥的保密性和哈希函数的抗碰撞性。",
            thinking="<thinking>数字签名是密码学中的重要概念，涉及非对称加密、哈希函数等多个知识点。我需要解释其工作原理、安全性保证，以及为什么能够提供身份认证、完整性和不可否认性。关键是要说明私钥签名、公钥验证的过程。</thinking>",
            crypto_terms=["数字签名", "非对称加密", "哈希函数", "身份认证", "不可否认性"],
            difficulty_level=DifficultyLevel.ADVANCED
        ),
        TrainingExample(
            instruction="分析量子计算对现有密码算法的威胁",
            input="考虑到量子计算技术的发展",
            output="量子计算对密码学构成重大威胁：\n\n受威胁算法：\n- RSA：Shor算法可破解大整数分解\n- ECC：Shor算法可解椭圆曲线离散对数\n- DH密钥交换：同样受Shor算法威胁\n\n相对安全算法：\n- AES：Grover算法仅将安全强度减半\n- SHA系列：受影响较小\n\n应对策略：\n- 发展后量子密码学\n- 增加密钥长度\n- 混合加密方案",
            thinking="<thinking>这是一个高级话题，涉及量子计算和密码学的交叉领域。需要分析Shor算法和Grover算法对不同类型密码算法的影响。Shor算法主要威胁基于数论难题的算法（如RSA、ECC），而Grover算法对对称密码的影响相对较小。还需要提及后量子密码学的发展。</thinking>",
            crypto_terms=["量子计算", "Shor算法", "Grover算法", "后量子密码学", "RSA", "ECC"],
            difficulty_level=DifficultyLevel.EXPERT
        ),
        TrainingExample(
            instruction="什么是哈希函数的雪崩效应？",
            input="",
            output="雪崩效应是指输入的微小变化会导致输出发生剧烈变化的特性。在密码学哈希函数中，即使输入只改变一个比特，输出也应该有大约50%的比特发生变化。这个特性对于确保哈希函数的安全性至关重要。",
            thinking="<thinking>雪崩效应是哈希函数的重要安全特性之一。它确保了即使输入有微小变化，输出也会显著不同，这有助于防止攻击者通过分析输入输出关系来破解哈希函数。</thinking>",
            crypto_terms=["哈希函数", "雪崩效应", "密码学特性"],
            difficulty_level=DifficultyLevel.INTERMEDIATE
        )
    ]
    
    return examples


def demo_cli_commands():
    """演示CLI命令的使用"""
    print("\n" + "=" * 60)
    print("CLI命令使用演示")
    print("=" * 60)
    
    print("可用的CLI命令:")
    print("  python src/cli_tools_simple.py --help")
    print("  python src/cli_tools_simple.py init-config --output config.yaml")
    print("  python src/cli_tools_simple.py validate-config config.yaml")
    print("  python src/cli_tools_simple.py list-gpus")
    print("  python src/cli_tools_simple.py inspect-data data.json --sample 3")
    print("  python src/cli_tools_simple.py train data.json config.yaml --dry-run")
    print("  python src/cli_tools_simple.py status ./training_output")
    
    print("\n典型工作流程:")
    print("  1. 生成配置模板: init-config")
    print("  2. 编辑配置文件")
    print("  3. 验证配置: validate-config")
    print("  4. 检查GPU环境: list-gpus")
    print("  5. 检查训练数据: inspect-data")
    print("  6. 开始训练: train")
    print("  7. 监控状态: status")


if __name__ == "__main__":
    print("Qwen3-4B-Thinking 密码学微调工具 - 完整演示")
    
    try:
        # 运行完整工作流程演示
        success = demo_complete_workflow()
        
        # 显示CLI命令使用说明
        demo_cli_commands()
        
        if success:
            print("\n🎉 演示成功完成！")
            print("\n📚 更多信息:")
            print("  - 查看生成的配置文件了解可调整的参数")
            print("  - 查看系统报告了解硬件环境")
            print("  - 使用 --help 参数查看各命令的详细说明")
        else:
            print("\n❌ 演示过程中出现问题")
            
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()