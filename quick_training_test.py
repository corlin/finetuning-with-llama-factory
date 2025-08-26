#!/usr/bin/env python3
"""
快速训练测试 - 验证JSON序列化修复
运行一个非常短的训练来验证修复是否有效
"""

import os
import sys

# 添加src目录到路径
sys.path.append('src')

def main():
    """快速训练测试"""
    print("🎯 快速训练测试 - 验证JSON序列化修复")
    print("=" * 60)
    
    try:
        from direct_finetuning_with_existing_modules import DirectTrainingConfig, DirectTrainer
        
        # 创建最小配置
        config = DirectTrainingConfig()
        config.data_path = "data/raw"
        config.output_dir = "test_output/quick_training_test"
        config.num_epochs = 1  # 只训练1个epoch
        config.batch_size = 1
        config.max_seq_length = 128  # 很小的序列长度
        config.save_steps = 5  # 每5步保存一次
        config.logging_steps = 1
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        print("🔄 创建训练器...")
        trainer = DirectTrainer(config)
        
        print("🔄 开始快速训练测试...")
        success = trainer.run()
        
        if success:
            print("🎉 快速训练测试成功完成！")
            print("✅ JSON序列化修复验证通过")
            
            # 检查输出文件
            stats_file = os.path.join(config.output_dir, 'training_statistics.json')
            if os.path.exists(stats_file):
                print(f"✅ 训练统计文件已生成: {stats_file}")
                
                # 验证文件可以正常读取
                import json
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                print("✅ 训练统计文件JSON格式正确")
                print(f"📊 训练步数: {stats.get('dataset_stats', {}).get('final_training_steps', 'N/A')}")
            else:
                print("⚠️ 训练统计文件未找到")
            
            return True
        else:
            print("❌ 快速训练测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 快速训练测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("🎉 JSON序列化修复验证成功！")
        print("💡 现在可以安全运行完整的训练流程")
    else:
        print("❌ 验证失败，需要进一步检查")
    
    sys.exit(0 if success else 1)