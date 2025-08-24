#!/usr/bin/env python3
"""
修复分布式训练问题的 LlamaFactory 训练脚本
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_windows_distributed_config():
    """创建适用于 Windows 的分布式配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    windows_config = "final_demo_output/configs/llamafactory_config_windows_distributed.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 启用 Windows 兼容的分布式训练
    config['ddp_backend'] = 'gloo'  # 使用 gloo 后端，Windows 兼容性更好
    config['dataloader_num_workers'] = 0  # Windows 上设为 0 避免多进程问题
    config['ddp_timeout'] = 3600  # 增加超时时间
    config['ddp_find_unused_parameters'] = True  # 启用未使用参数查找
    config['ddp_broadcast_buffers'] = False  # 禁用缓冲区广播以提高稳定性
    
    # 修复模型名称参数
    if 'model_name' in config and 'model_name_or_path' not in config:
        config['model_name_or_path'] = config['model_name']
    
    # 移除不被 LlamaFactory 识别的键
    unused_keys = ['model_name', 'visual_inputs', 'evaluation_strategy']
    for key in unused_keys:
        config.pop(key, None)
    
    # 简单修复：禁用 load_best_model_at_end 以避免策略匹配问题
    config['load_best_model_at_end'] = False
    
    # 确保保存策略正确设置
    config['save_strategy'] = 'steps'
    
    # 修复数据集路径问题
    config['dataset_dir'] = 'final_demo_output/data'
    
    # 调整批次大小适应分布式训练
    config['per_device_train_batch_size'] = 1  # 每个GPU的批次大小
    config['per_device_eval_batch_size'] = 1
    config['gradient_accumulation_steps'] = 4  # 增加梯度累积以保持有效批次大小
    
    # 保存 Windows 分布式配置
    os.makedirs(os.path.dirname(windows_config), exist_ok=True)
    with open(windows_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建 Windows 分布式配置文件: {windows_config}")
    return windows_config

def run_training():
    """运行训练"""
    print("🚀 开始使用修复的分布式训练...")
    
    # 创建 Windows 兼容配置
    config_file = create_windows_distributed_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"  # 禁用 libuv 支持
    env["NCCL_P2P_DISABLE"] = "1"  # 禁用 NCCL P2P 通信
    env["NCCL_IB_DISABLE"] = "1"   # 禁用 InfiniBand
    env["OMP_NUM_THREADS"] = "1"   # 设置 OpenMP 线程数
    env["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用两个 GPU
    env["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # 启用分布式调试信息
    
    # 构建命令
    cmd = [
        "uv", "run", "llamafactory-cli", "train", config_file
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"🔧 使用 GPU: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"🔧 分布式后端: gloo (Windows 兼容)")
    
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
            print("✅ 分布式训练完成!")
            return True
        else:
            print(f"❌ 分布式训练失败，退出码: {result.returncode}")
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
    print("🎯 LlamaFactory 真正的分布式训练启动器")
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