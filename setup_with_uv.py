#!/usr/bin/env python3
"""
使用uv进行项目环境设置的主脚本
确保使用uv包管理器进行依赖管理和虚拟环境管理
"""

import sys
import os
import subprocess
from pathlib import Path

def check_uv_installation():
    """检查uv是否已安装"""
    try:
        result = subprocess.run(["uv", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✓ uv已安装: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv未安装或不在PATH中")
        print("\n请安装uv包管理器:")
        print("Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("或使用pip: pip install uv")
        return False

def setup_uv_project():
    """设置uv项目"""
    project_root = Path.cwd()
    
    print("=== 使用uv设置项目环境 ===")
    
    # 检查uv安装
    if not check_uv_installation():
        return False
    
    try:
        # 同步依赖
        print("\n正在使用uv同步依赖...")
        result = subprocess.run(
            ["uv", "sync"],
            cwd=project_root,
            check=True
        )
        print("✓ 生产依赖安装完成")
        
        # 安装开发依赖
        print("\n正在安装开发依赖...")
        dev_result = subprocess.run(
            ["uv", "sync", "--extra", "dev"],
            cwd=project_root,
            check=True
        )
        print("✓ 开发依赖安装完成")
        
        # 显示虚拟环境信息
        print("\n获取虚拟环境信息...")
        venv_result = subprocess.run(
            ["uv", "venv", "--python", "3.12"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        print("✓ uv项目设置完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ uv设置失败: {e}")
        return False

def run_environment_setup():
    """运行环境设置脚本"""
    print("\n=== 运行环境设置 ===")
    
    try:
        # 使用uv run执行环境设置
        result = subprocess.run(
            ["uv", "run", "python", "src/environment_setup.py"],
            check=True
        )
        print("✓ 环境设置完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 环境设置失败: {e}")
        return False

def run_basic_tests():
    """运行基础测试"""
    print("\n=== 运行基础测试 ===")
    
    try:
        # 使用uv run执行测试
        result = subprocess.run(
            ["uv", "run", "python", "test_basic_setup.py"],
            check=True
        )
        print("✓ 基础测试通过")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 基础测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== LLaMA Factory Finetuning - uv环境设置 ===")
    print("专门针对Qwen3-4B-Thinking模型的微调环境")
    print()
    
    steps = [
        ("uv项目设置", setup_uv_project),
        ("环境设置", run_environment_setup),
        ("基础测试", run_basic_tests),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*50}")
        print(f"步骤: {step_name}")
        print('='*50)
        
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name}执行异常: {e}")
            results[step_name] = False
        
        if not results[step_name]:
            print(f"\n⚠️  {step_name}失败，停止后续步骤")
            break
    
    # 显示结果
    print(f"\n{'='*50}")
    print("设置结果汇总:")
    print('='*50)
    
    for step_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {step_name}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n总体结果: {success_count}/{total_count} 步骤成功")
    
    if success_count == total_count:
        print("\n🎉 uv环境设置完成！")
        print("\n下一步操作:")
        print("1. 检查环境: uv run python scripts/check_environment.py")
        print("2. 准备数据: 将训练数据放入 data/ 目录")
        print("3. 开始训练: uv run python scripts/train.py")
        print("\n常用uv命令:")
        print("- 运行脚本: uv run python <script.py>")
        print("- 安装包: uv add <package>")
        print("- 移除包: uv remove <package>")
        print("- 查看依赖: uv tree")
        print("- 激活环境: source .venv/bin/activate (Linux/Mac) 或 .venv\\Scripts\\activate (Windows)")
        return 0
    else:
        print("\n⚠️  部分步骤失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())