#!/usr/bin/env python3
"""
真正的分布式训练脚本 - 通过直接调用 torchrun 来避免环境变量传递问题
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_distributed_config():
    """创建分布式配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    distributed_config = "final_demo_output/configs/llamafactory_config_true_distributed.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 配置真正的分布式训练
    config['ddp_backend'] = 'gloo'  # Windows 兼容的后端
    config['dataloader_num_workers'] = 0  # Windows 兼容
    config['ddp_timeout'] = 3600
    config['ddp_find_unused_parameters'] = True
    config['ddp_broadcast_buffers'] = False
    
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
    
    # 分布式训练批次大小
    config['per_device_train_batch_size'] = 1
    config['per_device_eval_batch_size'] = 1
    config['gradient_accumulation_steps'] = 4
    
    # 保存配置
    os.makedirs(os.path.dirname(distributed_config), exist_ok=True)
    with open(distributed_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建真正的分布式配置文件: {distributed_config}")
    return distributed_config

def run_training():
    """运行分布式训练"""
    print("🚀 开始真正的分布式训练...")
    
    # 创建配置
    config_file = create_distributed_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"  # 关键：禁用 libuv
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    
    # 获取 LlamaFactory launcher 路径
    launcher_path = None
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))"],
            capture_output=True, text=True, env=env
        )
        if result.returncode == 0:
            launcher_path = result.stdout.strip()
            print(f"✅ 找到 LlamaFactory launcher: {launcher_path}")
        else:
            print("❌ 无法找到 LlamaFactory launcher")
            return False
    except Exception as e:
        print(f"❌ 查找 launcher 失败: {e}")
        return False
    
    # 直接使用 torchrun 命令
    cmd = [
        "uv", "run", "torchrun",
        "--nnodes", "1",
        "--node_rank", "0", 
        "--nproc_per_node", "2",
        "--master_addr", "127.0.0.1",
        "--master_port", "29500",
        launcher_path,
        config_file
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"🔧 使用 GPU: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"🔧 环境变量 USE_LIBUV: {env['USE_LIBUV']}")
    
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
    
    # 检查 torchrun
    try:
        result = subprocess.run(["uv", "run", "torchrun", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ torchrun 可用")
        else:
            print("❌ torchrun 不可用")
            return False
    except:
        print("❌ torchrun 检查失败")
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 LlamaFactory 真正的分布式训练启动器 (直接 torchrun)")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 运行训练
    return run_training()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)