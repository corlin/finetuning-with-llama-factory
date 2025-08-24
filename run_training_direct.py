#!/usr/bin/env python3
"""
直接调用 LlamaFactory 训练的脚本，避免 torchrun 问题
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_direct_training_script():
    """创建直接训练脚本"""
    script_content = '''
import os
import sys
import yaml
from llamafactory.train.tuner import run_exp

def main():
    # 设置环境变量
    os.environ["USE_LIBUV"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # 配置文件路径
    config_file = sys.argv[1] if len(sys.argv) > 1 else "final_demo_output/configs/llamafactory_config_direct.yaml"
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 读取配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保单 GPU 配置
    config.pop('ddp_backend', None)
    config['dataloader_num_workers'] = 0
    
    print("🚀 开始直接训练...")
    
    try:
        # 直接调用训练函数
        run_exp(config)
        print("✅ 训练完成!")
        return True
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    script_path = "direct_training.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 创建直接训练脚本: {script_path}")
    return script_path

def create_direct_config():
    """创建直接训练配置文件"""
    config_file = "final_demo_output/configs/llamafactory_config_20250824_212935.yaml"
    direct_config = "final_demo_output/configs/llamafactory_config_direct.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 原配置文件不存在: {config_file}")
        return None
    
    # 读取原配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 完全移除分布式相关配置
    config.pop('ddp_backend', None)
    config.pop('ddp_timeout', None)
    config.pop('ddp_find_unused_parameters', None)
    config.pop('ddp_broadcast_buffers', None)
    
    # 修复模型名称参数
    if 'model_name' in config and 'model_name_or_path' not in config:
        config['model_name_or_path'] = config['model_name']
    
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
    
    # 单 GPU 优化配置
    config['dataloader_num_workers'] = 0
    config['per_device_train_batch_size'] = 2
    config['per_device_eval_batch_size'] = 2
    config['gradient_accumulation_steps'] = 2
    
    # 保存直接训练配置
    os.makedirs(os.path.dirname(direct_config), exist_ok=True)
    with open(direct_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建直接训练配置文件: {direct_config}")
    return direct_config

def run_training():
    """运行训练"""
    print("🚀 开始使用直接调用方式训练...")
    
    # 创建直接训练脚本和配置
    script_path = create_direct_training_script()
    config_file = create_direct_config()
    
    if not config_file:
        return False
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATASET_INFO_FILE"] = "final_demo_output/data/dataset_info.json"
    env["USE_LIBUV"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    
    # 构建命令
    cmd = ["uv", "run", "python", script_path, config_file]
    
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
    print("🎯 LlamaFactory 直接调用训练启动器")
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