#!/usr/bin/env python3
"""
使用单 GPU 运行 LlamaFactory 训练的脚本 (避免分布式训练问题)
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_single_gpu_config():
    """创建单 GPU 配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    single_gpu_config = "final_demo_output/configs/llamafactory_config_single_gpu.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改为单 GPU 配置
    config['ddp_backend'] = None  # 禁用分布式训练
    config['dataloader_num_workers'] = 0  # Windows 上设为 0 避免多进程问题
    config['per_device_train_batch_size'] = 2  # 增加批次大小补偿单 GPU
    config['per_device_eval_batch_size'] = 2
    config['gradient_accumulation_steps'] = 2  # 减少梯度累积步数
    
    # 修复评估和保存策略匹配问题
    if config.get('load_best_model_at_end', False):
        # 确保评估策略和保存策略匹配
        if config.get('save_strategy') == 'steps':
            config['evaluation_strategy'] = 'steps'
        elif config.get('save_strategy') == 'epoch':
            config['evaluation_strategy'] = 'epoch'
        else:
            # 如果不匹配，禁用 load_best_model_at_end
            config['load_best_model_at_end'] = False
    
    # 保存单 GPU 配置
    os.makedirs(os.path.dirname(single_gpu_config), exist_ok=True)
    with open(single_gpu_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建单 GPU 配置文件: {single_gpu_config}")
    return single_gpu_config

def run_training():
    """运行训练"""
    print("🚀 开始使用单 GPU 运行 LlamaFactory 训练...")
    
    # 创建单 GPU 配置
    config_file = create_single_gpu_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"  # 禁用 libuv 支持
    env["OMP_NUM_THREADS"] = "1"   # 设置 OpenMP 线程数
    env["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个 GPU
    
    # 构建命令 (不使用 torchrun)
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
    
    # 检查 GPU
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import torch; print(f'GPU 可用: {torch.cuda.is_available()}, GPU 数量: {torch.cuda.device_count()}')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print("❌ GPU 检查失败")
    except:
        print("❌ GPU 检查失败")
    
    return True

def main():
    """主函数"""
    print("🎯 LlamaFactory 单 GPU 训练启动器")
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