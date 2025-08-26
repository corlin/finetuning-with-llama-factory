#!/usr/bin/env python3
"""
测试直接训练引擎配置
"""

from direct_finetuning_with_existing_modules import DirectTrainer, DirectTrainingConfig

def test_direct_training_config():
    """测试直接训练配置"""
    try:
        config = DirectTrainingConfig(
            model_name='Qwen/Qwen3-4B-Thinking-2507',
            data_path='test_data.json',
            output_dir='test_output',
            num_epochs=1,
            batch_size=1
        )
        print('✅ DirectTrainingConfig 创建成功')
        print(f'✅ 模型名称: {config.model_name}')
        print(f'✅ 输出目录: {config.output_dir}')
        print('✅ 直接训练引擎配置正常')
        return True
    except Exception as e:
        print(f'❌ 配置创建失败: {e}')
        return False

if __name__ == "__main__":
    test_direct_training_config()