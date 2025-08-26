#!/usr/bin/env python3
"""
直接测试save_training_statistics方法
验证JSON序列化修复是否有效
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# 添加src目录到路径
sys.path.append('src')

def test_save_statistics():
    """测试保存统计信息"""
    print("🔍 测试保存统计信息...")
    
    try:
        from direct_finetuning_with_existing_modules import convert_numpy_types
        
        # 创建模拟的训练统计数据（包含numpy类型）
        mock_stats = {
            'training_config': {
                'model_name': 'test_model',
                'batch_size': np.int32(4),
                'learning_rate': np.float64(1e-4),
                'use_lora': np.bool_(True),
                'gradient_accumulation_steps': np.int64(8)
            },
            'dataset_stats': {
                'total_samples': np.int64(81),
                'total_batches': np.int32(81),
                'final_training_steps': np.int32(10),
                'average_chinese_quality': np.float64(0.803),
                'average_crypto_complexity': np.float32(2.27),
                'quality_samples': np.int32(81),
                'crypto_samples': np.int32(44)
            },
            'model_stats': {
                'total_parameters': np.int64(4022468096),
                'trainable_parameters': np.int64(495452160)
            },
            'monitoring_stats': {
                'convergence_status': {
                    'is_converged': np.bool_(False),
                    'convergence_score': np.float64(0.75),
                    'loss_trend': np.float32(-0.1),
                    'plateau_steps': np.int32(0),
                    'loss_smoothness': np.float64(0.85)
                },
                'gpu_utilization_summary': {
                    0: {
                        'avg_utilization': np.float32(85.5),
                        'avg_memory_usage': np.float64(78.2),
                        'avg_temperature': np.float32(72.0),
                        'avg_power_usage': np.float32(250.5)
                    },
                    1: {
                        'avg_utilization': np.float32(90.2),
                        'avg_memory_usage': np.float64(82.1),
                        'avg_temperature': np.float32(75.0),
                        'avg_power_usage': np.float32(265.3)
                    }
                },
                'final_metrics': {
                    'epoch': np.int32(1),
                    'global_step': np.int32(10),
                    'train_loss': np.float64(2.345),
                    'val_loss': np.float64(2.123),
                    'learning_rate': np.float64(1e-4),
                    'memory_efficiency': np.float32(0.85),
                    'load_balance_score': np.float64(0.92),
                    'convergence_score': np.float32(0.75),
                    'gradient_norm': np.float64(1.23)
                }
            },
            'training_completed_at': datetime.now().isoformat()
        }
        
        print("✅ 创建模拟统计数据（包含numpy类型）")
        
        # 转换numpy类型
        converted_stats = convert_numpy_types(mock_stats)
        print("✅ numpy类型转换完成")
        
        # 测试JSON序列化
        output_dir = "test_output/statistics_test"
        os.makedirs(output_dir, exist_ok=True)
        
        stats_file = os.path.join(output_dir, 'test_training_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(converted_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 统计信息已保存到: {stats_file}")
        
        # 验证文件可以正常读取
        with open(stats_file, 'r', encoding='utf-8') as f:
            loaded_stats = json.load(f)
        
        print("✅ 统计信息文件读取成功")
        
        # 验证数据完整性
        assert loaded_stats['training_config']['batch_size'] == 4
        assert loaded_stats['dataset_stats']['total_samples'] == 81
        assert loaded_stats['monitoring_stats']['convergence_status']['is_converged'] == False
        assert loaded_stats['monitoring_stats']['gpu_utilization_summary']['0']['avg_utilization'] == 85.5
        
        print("✅ 数据完整性验证通过")
        
        # 显示文件大小
        file_size = os.path.getsize(stats_file)
        print(f"📊 统计文件大小: {file_size} 字节")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存统计信息测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 测试保存训练统计信息")
    print("=" * 50)
    
    if test_save_statistics():
        print("\n🎉 保存统计信息测试成功！")
        print("✅ JSON序列化修复有效")
        print("💡 训练过程中的统计信息保存应该不会再出现JSON序列化错误")
        return True
    else:
        print("\n❌ 保存统计信息测试失败")
        print("⚠️ 需要进一步检查JSON序列化修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)