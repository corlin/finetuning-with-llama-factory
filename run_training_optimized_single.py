#!/usr/bin/env python3
"""
优化的单 GPU 训练脚本 - 使用大批次大小充分利用 GPU 显存
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_optimized_single_config():
    """创建优化的单 GPU 配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    optimized_config = "final_demo_output/configs/llamafactory_config_optimized_single.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 强制单 GPU 配置
    config.pop('ddp_backend', None)  # 完全移除分布式后端
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
    
    # 优化批次大小 - 使用更大的批次充分利用单 GPU 显存
    config['per_device_train_batch_size'] = 4  # 增大批次大小
    config['per_device_eval_batch_size'] = 4
    config['gradient_accumulation_steps'] = 2  # 总有效批次大小 = 4 × 2 = 8
    
    # 优化内存使用
    config['fp16'] = True  # 使用半精度以节省显存
    config['bf16'] = False  # 禁用 bf16，使用 fp16
    config['gradient_checkpointing'] = True  # 启用梯度检查点节省显存
    
    # 保存配置
    os.makedirs(os.path.dirname(optimized_config), exist_ok=True)
    with open(optimized_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建优化单 GPU 配置文件: {optimized_config}")
    return optimized_config

def run_training():
    """运行优化的单 GPU 训练"""
    print("🚀 开始优化的单 GPU 训练...")
    
    # 创建配置
    config_file = create_optimized_single_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["CUDA_VISIBLE_DEVICES"] = "0"  # 强制只使用第一个 GPU
    env["OMP_NUM_THREADS"] = "1"
    env["USE_LIBUV"] = "0"  # 预防性设置
    
    # 使用 LlamaFactory CLI
    cmd = ["uv", "run", "llamafactory-cli", "train", config_file]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"🔧 使用 GPU: {env['CUDA_VISIBLE_DEVICES']} (强制单 GPU)")
    print(f"🔧 优化设置: 大批次大小 + FP16 + 梯度检查点")
    print(f"🔧 有效批次大小: 4 × 2 = 8 (相当于双 GPU 的性能)")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 优化单 GPU 训练完成!")
            return True
        else:
            print(f"❌ 优化单 GPU 训练失败，退出码: {result.returncode}")
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
    
    # 检查 GPU 显存
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import torch; print(f'GPU 0 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB') if torch.cuda.is_available() else print('GPU 不可用')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
    except:
        print("❌ GPU 显存检查失败")
    
    return True

def main():
    """主函数"""
    print("🎯 LlamaFactory 优化单 GPU 训练启动器")
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