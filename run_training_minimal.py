#!/usr/bin/env python3
"""
最小化配置的 LlamaFactory 训练脚本
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_minimal_config():
    """创建最小化配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    minimal_config = "final_demo_output/configs/llamafactory_config_minimal.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建最小化配置
    minimal_config_dict = {
        # 基本模型配置
        'model_name_or_path': config.get('model_name', 'Qwen/Qwen3-4B-Thinking-2507'),
        'template': config.get('template', 'qwen'),
        
        # 数据配置
        'dataset': config.get('dataset', 'crypto_qa_dataset'),
        'cutoff_len': config.get('cutoff_len', 76),
        'val_size': config.get('val_size', 0.1),
        
        # 训练配置
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'lora',
        'lora_target': 'all',
        'lora_rank': 64,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        
        # 训练参数
        'num_train_epochs': 2,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 2,
        'learning_rate': 0.0002,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        
        # 保存和评估
        'output_dir': config.get('output_dir', 'final_demo_output/model_output'),
        'save_strategy': 'steps',
        'save_steps': 100,
        'save_total_limit': 3,
        'evaluation_strategy': 'no',  # 禁用评估以避免问题
        'load_best_model_at_end': False,  # 禁用以避免策略匹配问题
        
        # 日志
        'logging_steps': 10,
        'log_level': 'info',
        'plot_loss': True,
        
        # 系统配置
        'dataloader_num_workers': 0,  # Windows 兼容
        'bf16': True,
        'tf32': True,
        'seed': 42,
        'overwrite_output_dir': True,
    }
    
    # 保存最小化配置
    os.makedirs(os.path.dirname(minimal_config), exist_ok=True)
    with open(minimal_config, 'w', encoding='utf-8') as f:
        yaml.dump(minimal_config_dict, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建最小化配置文件: {minimal_config}")
    return minimal_config

def run_training():
    """运行训练"""
    print("🚀 开始使用最小化配置训练...")
    
    # 创建最小化配置
    config_file = create_minimal_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"  # 禁用 libuv 支持
    env["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个 GPU
    env["OMP_NUM_THREADS"] = "1"   # 设置 OpenMP 线程数
    
    # 构建命令
    cmd = ["uv", "run", "llamafactory-cli", "train", config_file]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
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
    print("🎯 LlamaFactory 最小化配置训练启动器")
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