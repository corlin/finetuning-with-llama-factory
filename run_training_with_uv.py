#!/usr/bin/env python3
"""
使用 uv 运行 LlamaFactory 训练的脚本
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def run_training():
    """运行训练"""
    print("🚀 开始使用 uv 运行 LlamaFactory 训练...")
    
    # 配置文件路径
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"  # 禁用 libuv 支持以避免 Windows 上的分布式训练错误
    env["NCCL_P2P_DISABLE"] = "1"  # 禁用 NCCL P2P 通信
    env["NCCL_IB_DISABLE"] = "1"   # 禁用 InfiniBand
    env["OMP_NUM_THREADS"] = "1"   # 设置 OpenMP 线程数
    
    # 构建命令
    cmd = [
        "uv", "run", "llamafactory-cli", "train", config_file
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,  # 显示实时输出
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 训练完成!")
            return True
        else:
            print(f"❌ 训练失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")
    
    # 检查 uv
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv 版本: {result.stdout.strip()}")
        else:
            print("❌ uv 未安装")
            return False
    except:
        print("❌ uv 未安装")
        return False
    
    # 检查 LlamaFactory
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import llamafactory; print(llamafactory.__version__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ LlamaFactory 版本: {result.stdout.strip()}")
        else:
            print("❌ LlamaFactory 导入失败")
            return False
    except:
        print("❌ LlamaFactory 检查失败")
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 LlamaFactory 训练启动器 (使用 uv)")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 运行训练
    return run_training()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)