#!/usr/bin/env python3
"""
项目环境设置主脚本
运行此脚本来完成项目的初始化设置
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """主设置函数"""
    print("=== LLaMA Factory Finetuning 环境设置 ===")
    print("正在初始化Qwen3-4B-Thinking微调环境...")
    print()
    
    try:
        from environment_setup import EnvironmentSetup
        
        # 创建环境设置实例
        setup = EnvironmentSetup(project_root)
        
        # 运行完整设置
        results = setup.run_full_setup()
        
        # 检查设置结果
        success = all(results.values())
        
        if success:
            print("\n🎉 环境设置完成！")
            print("\n下一步操作:")
            print("1. 检查环境: python scripts/check_environment.py")
            print("2. 准备数据: 将训练数据放入 data/ 目录")
            print("3. 开始训练: python scripts/train.py")
            return 0
        else:
            print("\n⚠️  环境设置部分完成，请检查上述报告中的问题")
            return 1
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所需依赖")
        return 1
    except Exception as e:
        print(f"❌ 设置失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())