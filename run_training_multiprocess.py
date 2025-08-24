#!/usr/bin/env python3
"""
使用 Python 多进程的分布式训练脚本 - 避免 torchrun 的 libuv 问题
"""

import os
import sys
import yaml
import subprocess
import multiprocessing as mp
from pathlib import Path

def create_multiprocess_config():
    """创建多进程配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    mp_config = "final_demo_output/configs/llamafactory_config_multiprocess.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 配置多进程训练
    config['ddp_backend'] = 'gloo'  # Windows 兼容
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
    
    # 多进程训练批次大小
    config['per_device_train_batch_size'] = 1
    config['per_device_eval_batch_size'] = 1
    config['gradient_accumulation_steps'] = 4
    
    # 保存配置
    os.makedirs(os.path.dirname(mp_config), exist_ok=True)
    with open(mp_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建多进程配置文件: {mp_config}")
    return mp_config

def run_worker(rank, world_size, config_file, master_port):
    """运行单个工作进程"""
    print(f"🔧 启动工作进程 {rank}/{world_size}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"  # 关键：禁用 libuv
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(rank)  # 每个进程使用一个 GPU
    
    # 分布式训练环境变量
    env["RANK"] = str(rank)
    env["LOCAL_RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    
    print(f"🔧 进程 {rank} 使用 GPU: {env['CUDA_VISIBLE_DEVICES']}")
    
    # 直接调用 LlamaFactory 训练
    cmd = ["uv", "run", "python", "-m", "llamafactory.train.tuner", config_file]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ 进程 {rank} 训练完成!")
            return True
        else:
            print(f"❌ 进程 {rank} 训练失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 进程 {rank} 执行失败: {e}")
        return False

def run_training():
    """运行多进程训练"""
    print("🚀 开始多进程分布式训练...")
    
    # 创建配置
    config_file = create_multiprocess_config()
    if not config_file:
        return False
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 设置参数
    world_size = 2  # 使用 2 个 GPU
    master_port = 29500
    
    print(f"🔧 启动 {world_size} 个训练进程")
    print(f"🔧 主端口: {master_port}")
    
    # 创建进程池
    processes = []
    
    try:
        # 启动工作进程
        for rank in range(world_size):
            p = mp.Process(
                target=run_worker,
                args=(rank, world_size, config_file, master_port)
            )
            p.start()
            processes.append(p)
            print(f"✅ 启动进程 {rank}")
        
        # 等待所有进程完成
        results = []
        for i, p in enumerate(processes):
            p.join()
            results.append(p.exitcode == 0)
            print(f"✅ 进程 {i} 完成，退出码: {p.exitcode}")
        
        # 检查结果
        if all(results):
            print("✅ 所有进程训练完成!")
            return True
        else:
            print("❌ 部分进程训练失败")
            return False
            
    except Exception as e:
        print(f"❌ 多进程训练失败: {e}")
        # 清理进程
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
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
    print("🎯 LlamaFactory 多进程分布式训练启动器")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 运行训练
    return run_training()

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    success = main()
    sys.exit(0 if success else 1)