
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
