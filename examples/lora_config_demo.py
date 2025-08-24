"""
LoRA配置优化演示

演示如何使用LoRA配置优化器为不同的GPU环境动态配置LoRA参数。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lora_config_optimizer import (
    LoRAConfigOptimizer,
    LoRAOptimizationStrategy,
    LoRATargetModule
)
from src.parallel_config import GPUTopology, GPUInfo, ParallelConfig, ParallelStrategy


def demo_single_gpu_optimization():
    """演示单GPU LoRA配置优化"""
    print("=" * 60)
    print("单GPU LoRA配置优化演示")
    print("=" * 60)
    
    # 创建优化器
    optimizer = LoRAConfigOptimizer(
        model_size_gb=4.0,  # Qwen3-4B模型
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32
    )
    
    # 模拟不同内存容量的GPU
    gpu_scenarios = [
        ("RTX 4090 (24GB)", 20000),  # 20GB可用
        ("RTX 3080 (10GB)", 8000),   # 8GB可用
        ("RTX 3060 (12GB)", 10000),  # 10GB可用
    ]
    
    strategies = [
        LoRAOptimizationStrategy.MEMORY_EFFICIENT,
        LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
        LoRAOptimizationStrategy.QUALITY_FOCUSED
    ]
    
    for gpu_name, available_memory in gpu_scenarios:
        print(f"\n{gpu_name} - 可用内存: {available_memory}MB")
        print("-" * 50)
        
        for strategy in strategies:
            config = optimizer.optimize_for_single_gpu(
                available_memory_mb=available_memory,
                strategy=strategy,
                target_module_type=LoRATargetModule.ATTENTION_MLP,
                batch_size=4,
                sequence_length=2048
            )
            
            print(f"  {strategy.value:20s}: rank={config.rank:2d}, alpha={config.alpha:2d}, "
                  f"内存={config.total_memory_mb:6.1f}MB, "
                  f"效率={config.memory_efficiency:.2f}")


def demo_multi_gpu_optimization():
    """演示多GPU LoRA配置优化"""
    print("\n" + "=" * 60)
    print("多GPU LoRA配置优化演示")
    print("=" * 60)
    
    # 创建优化器
    optimizer = LoRAConfigOptimizer()
    
    # 场景1: 均衡的多GPU环境
    print("\n场景1: 均衡的4x RTX 4090环境")
    print("-" * 40)
    
    gpu_info_balanced = {
        0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
        1: GPUInfo(gpu_id=1, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
        2: GPUInfo(gpu_id=2, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
        3: GPUInfo(gpu_id=3, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
    }
    
    topology_balanced = GPUTopology(
        num_gpus=4,
        gpu_info=gpu_info_balanced,
        interconnect_bandwidth={
            (0, 1): 50.0, (1, 0): 50.0, (0, 2): 50.0, (2, 0): 50.0,
            (0, 3): 50.0, (3, 0): 50.0, (1, 2): 50.0, (2, 1): 50.0,
            (1, 3): 50.0, (3, 1): 50.0, (2, 3): 50.0, (3, 2): 50.0
        },
        numa_topology={0: 0, 1: 0, 2: 1, 3: 1}
    )
    
    parallel_config = ParallelConfig(
        strategy=ParallelStrategy.DATA_PARALLEL,
        data_parallel_size=4
    )
    
    config_balanced = optimizer.optimize_for_multi_gpu(
        topology=topology_balanced,
        parallel_config=parallel_config,
        strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED,
        target_module_type=LoRATargetModule.ATTENTION_MLP,
        batch_size=16,
        sequence_length=2048
    )
    
    print(f"全局配置: rank={config_balanced.global_config.rank}, alpha={config_balanced.global_config.alpha}")
    print(f"平均内存使用: {config_balanced.average_memory_usage:.1f}MB")
    print(f"负载均衡评分: {config_balanced.get_memory_balance_score():.3f}")
    print(f"总LoRA参数: {config_balanced.total_lora_parameters:,}")
    
    # 场景2: 不均衡的多GPU环境
    print("\n场景2: 不均衡的混合GPU环境")
    print("-" * 40)
    
    gpu_info_mixed = {
        0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
        1: GPUInfo(gpu_id=1, name="RTX 3080", memory_total=10000, memory_free=8000, compute_capability=(8, 6)),
        2: GPUInfo(gpu_id=2, name="RTX 3060", memory_total=12000, memory_free=10000, compute_capability=(8, 6))
    }
    
    topology_mixed = GPUTopology(
        num_gpus=3,
        gpu_info=gpu_info_mixed,
        interconnect_bandwidth={
            (0, 1): 25.0, (1, 0): 25.0,
            (0, 2): 25.0, (2, 0): 25.0,
            (1, 2): 16.0, (2, 1): 16.0
        },
        numa_topology={0: 0, 1: 0, 2: 1}
    )
    
    parallel_config_mixed = ParallelConfig(
        strategy=ParallelStrategy.DATA_PARALLEL,
        data_parallel_size=3
    )
    
    config_mixed = optimizer.optimize_for_multi_gpu(
        topology=topology_mixed,
        parallel_config=parallel_config_mixed,
        strategy=LoRAOptimizationStrategy.AUTO,
        target_module_type=LoRATargetModule.ATTENTION_MLP,
        batch_size=12,
        sequence_length=2048
    )
    
    print(f"平均内存使用: {config_mixed.average_memory_usage:.1f}MB")
    print(f"负载均衡评分: {config_mixed.get_memory_balance_score():.3f}")
    
    print("\n各GPU配置详情:")
    for gpu_id, gpu_config in config_mixed.per_gpu_configs.items():
        gpu_name = gpu_info_mixed[gpu_id].name
        print(f"  GPU {gpu_id} ({gpu_name}): rank={gpu_config.rank}, alpha={gpu_config.alpha}, "
              f"内存={gpu_config.total_memory_mb:.1f}MB")


def demo_configuration_validation():
    """演示配置验证功能"""
    print("\n" + "=" * 60)
    print("LoRA配置验证演示")
    print("=" * 60)
    
    optimizer = LoRAConfigOptimizer()
    
    # 创建一个GPU拓扑用于验证
    gpu_info = {
        0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9))
    }
    
    topology = GPUTopology(
        num_gpus=1,
        gpu_info=gpu_info,
        interconnect_bandwidth={},
        numa_topology={0: 0}
    )
    
    # 测试有效配置
    print("\n测试有效配置:")
    valid_config = optimizer.optimize_for_single_gpu(
        available_memory_mb=20000,
        strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED
    )
    
    validation_result = optimizer.validate_lora_config(valid_config, topology)
    print(f"配置有效性: {validation_result['valid']}")
    print(f"错误数量: {len(validation_result['errors'])}")
    print(f"警告数量: {len(validation_result['warnings'])}")
    print(f"建议数量: {len(validation_result['recommendations'])}")
    
    if validation_result['warnings']:
        print("警告信息:")
        for warning in validation_result['warnings']:
            print(f"  - {warning}")
    
    if validation_result['recommendations']:
        print("优化建议:")
        for rec in validation_result['recommendations']:
            print(f"  - {rec}")


def demo_configuration_report():
    """演示配置报告生成"""
    print("\n" + "=" * 60)
    print("LoRA配置报告演示")
    print("=" * 60)
    
    optimizer = LoRAConfigOptimizer()
    
    # 创建配置
    gpu_info = {
        0: GPUInfo(gpu_id=0, name="RTX 4090", memory_total=24000, memory_free=20000, compute_capability=(8, 9)),
        1: GPUInfo(gpu_id=1, name="RTX 4090", memory_total=24000, memory_free=18000, compute_capability=(8, 9))
    }
    
    topology = GPUTopology(
        num_gpus=2,
        gpu_info=gpu_info,
        interconnect_bandwidth={(0, 1): 50.0, (1, 0): 50.0},
        numa_topology={0: 0, 1: 0}
    )
    
    parallel_config = ParallelConfig(
        strategy=ParallelStrategy.DATA_PARALLEL,
        data_parallel_size=2
    )
    
    config = optimizer.optimize_for_multi_gpu(
        topology=topology,
        parallel_config=parallel_config,
        strategy=LoRAOptimizationStrategy.PERFORMANCE_BALANCED
    )
    
    # 生成报告
    report = optimizer.generate_config_report(config, topology)
    
    print(f"\n配置类型: {report['config_type']}")
    print(f"GPU数量: {report['gpu_topology']['num_gpus']}")
    print(f"配置有效性: {report['validation_result']['valid']}")
    
    print("\n性能估算:")
    perf = report['performance_estimates']
    print(f"  平均内存效率: {perf['average_memory_efficiency']:.3f}")
    print(f"  内存平衡评分: {perf['memory_balance_score']:.3f}")
    print(f"  总参数数量: {perf['total_parameters']:,}")
    print(f"  通信开销估算: {perf['communication_overhead_estimate']:.1%}")


def main():
    """主函数"""
    print("LoRA配置优化器演示程序")
    print("本演示展示了如何为不同GPU环境动态配置LoRA参数")
    
    try:
        demo_single_gpu_optimization()
        demo_multi_gpu_optimization()
        demo_configuration_validation()
        demo_configuration_report()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()