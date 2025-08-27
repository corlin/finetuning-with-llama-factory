#!/usr/bin/env python3
"""
测试TensorBoard集成功能
验证训练过程中的TensorBoard日志记录
"""

import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime

def test_tensorboard_basic():
    """测试基础TensorBoard功能"""
    print("🔍 测试TensorBoard基础功能...")
    
    # 创建测试日志目录
    test_log_dir = "test_tensorboard_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(test_log_dir, f"test_run_{timestamp}")
    
    try:
        # 初始化TensorBoard writer
        writer = SummaryWriter(log_dir=run_dir)
        print(f"✅ TensorBoard writer创建成功: {run_dir}")
        
        # 模拟训练数据
        for step in range(100):
            # 模拟损失曲线
            loss = 2.0 * np.exp(-step * 0.02) + 0.1 * np.random.random()
            writer.add_scalar('Training/Loss', loss, step)
            
            # 模拟学习率
            lr = 1e-4 * (0.95 ** (step // 10))
            writer.add_scalar('Training/Learning_Rate', lr, step)
            
            # 模拟梯度范数
            grad_norm = 1.0 + 0.5 * np.sin(step * 0.1) + 0.2 * np.random.random()
            writer.add_scalar('Training/Gradient_Norm', grad_norm, step)
            
            # 模拟GPU内存使用
            if torch.cuda.is_available():
                memory_used = 8000 + 2000 * np.sin(step * 0.05) + 500 * np.random.random()
                writer.add_scalar('Memory/GPU_Allocated_MB', memory_used, step)
                writer.add_scalar('Memory/GPU_Utilization_Percent', 
                                (memory_used / 16000) * 100, step)
        
        # 添加文本摘要
        writer.add_text('Test/Summary', 
                       'TensorBoard集成测试完成，所有基础功能正常工作', 
                       100)
        
        # 添加直方图数据
        weights = np.random.normal(0, 1, 1000)
        writer.add_histogram('Model/Weights_Distribution', weights, 100)
        
        # 关闭writer
        writer.close()
        print("✅ TensorBoard测试数据写入完成")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorBoard测试失败: {e}")
        return False

def test_tensorboard_advanced():
    """测试高级TensorBoard功能"""
    print("🔍 测试TensorBoard高级功能...")
    
    test_log_dir = "test_tensorboard_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(test_log_dir, f"advanced_test_{timestamp}")
    
    try:
        writer = SummaryWriter(log_dir=run_dir)
        
        # 模拟多GPU训练监控
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 2
        for step in range(50):
            for gpu_id in range(num_gpus):
                # 每个GPU的利用率
                utilization = 70 + 20 * np.sin(step * 0.1 + gpu_id) + 10 * np.random.random()
                writer.add_scalar(f'GPU_{gpu_id}/Utilization_Percent', utilization, step)
                
                # 每个GPU的内存使用
                memory_usage = 60 + 30 * np.cos(step * 0.08 + gpu_id) + 10 * np.random.random()
                writer.add_scalar(f'GPU_{gpu_id}/Memory_Usage_Percent', memory_usage, step)
                
                # 每个GPU的温度
                temperature = 65 + 15 * np.sin(step * 0.05 + gpu_id) + 5 * np.random.random()
                writer.add_scalar(f'GPU_{gpu_id}/Temperature_C', temperature, step)
        
        # 模拟数据集统计
        writer.add_scalar('Dataset/Total_Samples', 1000, 0)
        writer.add_scalar('Dataset/Avg_Instruction_Length', 150.5, 0)
        writer.add_scalar('Dataset/Avg_Output_Length', 300.2, 0)
        writer.add_scalar('Dataset/Thinking_Samples_Percent', 75.0, 0)
        
        # 模拟中文质量指标
        writer.add_scalar('Dataset/Avg_Instruction_Quality', 0.85, 0)
        writer.add_scalar('Dataset/Avg_Output_Quality', 0.78, 0)
        writer.add_scalar('Dataset/Avg_Crypto_Complexity', 2.3, 0)
        
        # 模拟难度分布
        for difficulty in range(1, 5):
            count = np.random.randint(50, 300)
            writer.add_scalar(f'Dataset/Difficulty_{difficulty}_Count', count, 0)
            writer.add_scalar(f'Dataset/Difficulty_{difficulty}_Percent', count/1000*100, 0)
        
        # 模拟收敛监控
        for step in range(50):
            convergence_score = min(1.0, step * 0.02 + 0.1 * np.random.random())
            writer.add_scalar('Monitoring/Convergence_Score', convergence_score, step)
            
            loss_trend = -0.01 * step + 0.005 * np.random.random()
            writer.add_scalar('Monitoring/Loss_Trend', loss_trend, step)
        
        # 添加配置信息
        config_info = """
        训练配置:
        - 模型: Qwen3-4B-Thinking
        - 批次大小: 1
        - 学习率: 1e-4
        - LoRA rank: 240
        - 序列长度: 2048
        """
        writer.add_text('Config/Training_Settings', config_info, 0)
        
        writer.close()
        print("✅ TensorBoard高级功能测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorBoard高级功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 TensorBoard集成测试")
    print("=" * 50)
    
    # 检查TensorBoard是否可用
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard模块可用")
    except ImportError:
        print("❌ TensorBoard模块不可用，请安装: pip install tensorboard")
        return False
    
    # 运行测试
    tests = [
        ("基础功能", test_tensorboard_basic),
        ("高级功能", test_tensorboard_advanced),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 TensorBoard集成测试全部通过！")
        print("\n📊 启动TensorBoard查看测试结果:")
        print("   tensorboard --logdir=test_tensorboard_logs")
        print("   然后在浏览器中打开: http://localhost:6006")
        return True
    else:
        print("⚠️ 部分测试失败，请检查TensorBoard配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)