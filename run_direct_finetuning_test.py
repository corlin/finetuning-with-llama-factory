#!/usr/bin/env python3
"""
运行直接微调测试的脚本
使用uv环境管理器运行测试
"""

import os
import sys
import subprocess
from pathlib import Path

def check_uv():
    """检查uv是否可用"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv 版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv 不可用")
            return False
    except FileNotFoundError:
        print("❌ uv 未安装")
        return False

def run_test():
    """运行测试"""
    print("🚀 开始运行直接微调测试...")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    # 运行测试
    cmd = ["uv", "run", "python", "test_direct_finetuning.py"]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 测试完成！")
            return True
        else:
            print(f"❌ 测试失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def run_full_training():
    """运行完整训练"""
    print("🚀 开始运行完整微调...")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    # 运行训练
    cmd = ["uv", "run", "python", "direct_finetuning_with_existing_modules.py"]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 微调完成！")
            return True
        else:
            print(f"❌ 微调失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 直接微调测试启动器")
    print("=" * 40)
    
    # 检查uv
    if not check_uv():
        print("请先安装uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    # 询问用户选择
    print("\n请选择操作:")
    print("1. 运行功能测试")
    print("2. 运行完整微调")
    print("3. 先测试后微调")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        return run_test()
    elif choice == "2":
        return run_full_training()
    elif choice == "3":
        print("🔄 先运行测试...")
        if run_test():
            print("\n🔄 测试通过，开始微调...")
            return run_full_training()
        else:
            print("❌ 测试失败，跳过微调")
            return False
    else:
        print("❌ 无效选择")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)