#!/usr/bin/env python3
"""
依赖修复脚本

修复 LLaMA Factory 和 transformers 版本兼容性问题
"""

import subprocess
import sys
import os

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_dependencies():
    """修复依赖版本问题"""
    print("🔧 开始修复依赖版本问题...")
    
    # 检查当前版本
    print("\n📋 检查当前版本:")
    success, stdout, stderr = run_command("uv pip list | findstr -i \"transformers llamafactory\"")
    if success:
        print(stdout)
    
    # 尝试降级 transformers
    print("\n⬇️ 降级 transformers 到兼容版本...")
    
    # 方案1: 降级到 4.44.x
    commands = [
        "uv pip install \"transformers>=4.41.0,<4.45.0\"",
        "uv pip install \"transformers==4.44.2\"",
        "uv pip install \"transformers==4.43.4\"",
        "uv pip install \"transformers==4.42.4\""
    ]
    
    for cmd in commands:
        print(f"尝试: {cmd}")
        success, stdout, stderr = run_command(cmd)
        if success:
            print("✅ 成功!")
            break
        else:
            print(f"❌ 失败: {stderr}")
    
    # 验证修复结果
    print("\n🔍 验证修复结果:")
    success, stdout, stderr = run_command("uv pip list | findstr -i \"transformers llamafactory\"")
    if success:
        print(stdout)
    
    # 测试 llamafactory 导入
    print("\n🧪 测试 LLaMA Factory 导入:")
    try:
        import llamafactory
        print("✅ LLaMA Factory 导入成功!")
        print(f"版本: {llamafactory.__version__}")
    except ImportError as e:
        print(f"❌ LLaMA Factory 导入失败: {e}")
        return False
    
    return True

def create_alternative_training_script():
    """创建替代训练脚本"""
    print("\n📝 创建替代训练脚本...")
    
    script_content = '''#!/usr/bin/env python3
"""
替代训练脚本

当 LLaMA Factory CLI 不可用时的备用方案
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_file):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return None

def load_dataset_info(dataset_info_file):
    """加载数据集信息"""
    try:
        with open(dataset_info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载数据集信息失败: {e}")
        return None

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 查找配置文件
    config_files = list(Path("configs").glob("llamafactory_config_*.yaml"))
    if not config_files:
        logger.error("未找到配置文件")
        return False
    
    config_file = config_files[0]
    logger.info(f"使用配置文件: {config_file}")
    
    # 加载配置
    config = load_config(config_file)
    if not config:
        return False
    
    # 查找数据集信息文件
    dataset_info_file = Path("data/dataset_info.json")
    if not dataset_info_file.exists():
        logger.error("未找到数据集信息文件")
        return False
    
    dataset_info = load_dataset_info(dataset_info_file)
    if not dataset_info:
        return False
    
    logger.info("配置信息:")
    logger.info(f"- 模型: {config.get('model_name', 'Unknown')}")
    logger.info(f"- 数据集: {config.get('dataset', 'Unknown')}")
    logger.info(f"- 输出目录: {config.get('output_dir', 'Unknown')}")
    logger.info(f"- 训练轮数: {config.get('num_train_epochs', 'Unknown')}")
    logger.info(f"- 学习率: {config.get('learning_rate', 'Unknown')}")
    logger.info(f"- LoRA rank: {config.get('lora_rank', 'Unknown')}")
    
    # 尝试导入 LLaMA Factory
    try:
        from llamafactory.train.tuner import run_exp
        logger.info("✅ LLaMA Factory 导入成功，开始训练...")
        
        # 设置环境变量
        os.environ["DATASET_INFO_FILE"] = str(dataset_info_file)
        
        # 运行训练
        run_exp(config)
        logger.info("✅ 训练完成!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ LLaMA Factory 导入失败: {e}")
        logger.info("请尝试以下解决方案:")
        logger.info("1. 运行依赖修复脚本: python fix_dependencies.py")
        logger.info("2. 手动安装兼容版本: uv pip install 'transformers==4.44.2'")
        logger.info("3. 重新安装 LLaMA Factory: uv pip install llamafactory")
        return False
    
    except Exception as e:
        logger.error(f"❌ 训练执行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    # 保存脚本到各个输出目录
    output_dirs = ["final_demo_output", "simple_demo_output", "demo_output"]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            script_path = os.path.join(output_dir, "alternative_train.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print(f"✅ 创建替代训练脚本: {script_path}")

def main():
    """主函数"""
    print("🚀 LLaMA Factory 依赖修复工具")
    print("="*50)
    
    # 修复依赖
    if fix_dependencies():
        print("\n✅ 依赖修复成功!")
    else:
        print("\n❌ 依赖修复失败，创建替代方案...")
        create_alternative_training_script()
    
    print("\n📋 使用说明:")
    print("1. 如果修复成功，可以直接使用 llamafactory-cli")
    print("2. 如果修复失败，使用替代脚本:")
    print("   cd final_demo_output")
    print("   python alternative_train.py")
    
    print("\n🔧 手动修复命令:")
    print("uv pip install 'transformers==4.44.2'")
    print("uv pip install --upgrade llamafactory")

if __name__ == "__main__":
    main()