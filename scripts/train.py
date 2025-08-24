#!/usr/bin/env python3
"""
训练启动脚本
启动Qwen3-4B-Thinking模型的微调训练

使用方法: uv run python scripts/train.py
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """主训练函数"""
    try:
        from config_manager import ConfigManager
        from gpu_utils import GPUDetector
        
        print("=== Qwen3-4B-Thinking 微调训练 ===")
        print()
        
        # 初始化配置管理器
        config_manager = ConfigManager()
        
        # 检查GPU状态
        detector = GPUDetector()
        gpu_infos = detector.get_all_gpu_info()
        
        if not gpu_infos:
            print("⚠️  未检测到GPU，将使用CPU模式（不推荐）")
        else:
            print(f"检测到 {len(gpu_infos)} 个GPU:")
            for gpu in gpu_infos:
                print(f"  GPU {gpu.gpu_id}: {gpu.name} ({gpu.total_memory}MB)")
        
        print()
        
        # 验证配置
        validation = config_manager.validate_configs()
        if not all(validation.values()):
            print("❌ 配置验证失败:")
            for key, value in validation.items():
                if not value:
                    print(f"  ✗ {key}")
            print("\n请修复配置问题后重试")
            return 1
        
        print("✓ 配置验证通过")
        
        # 获取训练配置
        all_configs = config_manager.get_all_configs()
        training_config = all_configs["training"]
        
        print(f"✓ 训练配置:")
        print(f"  - 批次大小: {training_config.per_device_train_batch_size}")
        print(f"  - 梯度累积: {training_config.gradient_accumulation_steps}")
        print(f"  - 学习率: {training_config.learning_rate}")
        print(f"  - 训练轮数: {training_config.num_train_epochs}")
        
        # 检查数据目录
        data_dirs = ["data/train", "data/eval"]
        for data_dir in data_dirs:
            data_path = project_root / data_dir
            if not data_path.exists() or not any(data_path.iterdir()):
                print(f"⚠️  数据目录为空: {data_dir}")
                print("请准备训练数据后重试")
                return 1
        
        print("✓ 数据目录检查通过")
        print()
        
        print("🚀 准备开始训练...")
        print("注意: 实际训练逻辑将在后续任务中实现")
        print()
        
        # TODO: 在后续任务中实现实际的训练逻辑
        # 这里只是验证环境和配置
        
        print("训练准备完成！")
        print("\n下一步:")
        print("1. 实现数据处理模块（任务2）")
        print("2. 实现训练引擎（任务6）")
        print("3. 实现监控系统（任务8）")
        
        return 0
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请先运行 python setup.py 初始化环境")
        return 1
    except Exception as e:
        print(f"❌ 训练准备失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())