#!/usr/bin/env python3
"""
使用 DataParallel 的多 GPU 训练脚本 - Windows 兼容
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_dataparallel_config():
    """创建 DataParallel 配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    dp_config = "final_demo_output/configs/llamafactory_config_dataparallel.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 禁用分布式训练，使用 DataParallel
    config.pop('ddp_backend', None)  # 移除分布式后端
    config.pop('ddp_timeout', None)
    config.pop('ddp_find_unused_parameters', None)
    config.pop('ddp_broadcast_buffers', None)
    
    # Windows 兼容设置
    config['dataloader_num_workers'] = 0
    
    # 修复模型名称参数
    if 'model_name' in config and 'model_name_or_path' not in config:
        config['model_name_or_path'] = config['model_name']
    
    # 移除不被识别的键
    unused_keys = ['model_name', 'visual_inputs', 'evaluation_strategy']
    for key in unused_keys:
        config.pop(key, None)
    
    # 修复策略匹配问题
    config['load_best_model_at_end'] = False
    config['save_strategy'] = 'steps'
    
    # 数据集路径
    config['dataset_dir'] = 'final_demo_output/data'
    
    # DataParallel 批次大小 - 可以更大因为是多 GPU
    config['per_device_train_batch_size'] = 2  # 每个GPU 2个样本
    config['per_device_eval_batch_size'] = 2
    config['gradient_accumulation_steps'] = 2  # 总有效批次大小 = 2 GPUs × 2 batch × 2 accumulation = 8
    
    # 保存配置
    os.makedirs(os.path.dirname(dp_config), exist_ok=True)
    with open(dp_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建 DataParallel 配置文件: {dp_config}")
    return dp_config

def run_training():
    """运行 DataParallel 训练"""
    print("🚀 开始 DataParallel 多 GPU 训练...")
    
    # 创建配置
    config_file = create_dataparallel_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用两个 GPU
    env["OMP_NUM_THREADS"] = "1"
    
    # 使用 LlamaFactory CLI
    cmd = ["uv", "run", "llamafactory-cli", "train", config_file]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"🔧 使用 GPU: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"🔧 训练模式: DataParallel (非分布式)")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ DataParallel 训练完成!")
            return True
        else:
            print(f"❌ DataParallel 训练失败，退出码: {result.returncode}")
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
    print("🎯 LlamaFactory DataParallel 多 GPU 训练启动器")
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