"""
分布式训练引擎演示

演示如何使用分布式训练引擎进行多GPU训练，包括：
- 分布式后端初始化
- 多GPU进程管理
- 梯度同步和参数更新
- 训练监控和指标收集
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.distributed_training_engine import (
    DistributedTrainingEngine,
    DistributedBackendInitializer,
    GradientSynchronizer,
    ParameterUpdateManager
)
from src.parallel_config import (
    ParallelConfig, GPUTopology, GPUInfo,
    ParallelStrategy, CommunicationBackend
)


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoModel(nn.Module):
    """演示用的简单模型"""
    
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


def create_demo_dataset(num_samples=1000, input_size=100, output_size=10):
    """创建演示数据集"""
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, output_size)
    
    dataset = TensorDataset(X, y)
    return dataset


def create_demo_config(num_gpus=1, backend="gloo"):
    """创建演示配置"""
    return ParallelConfig(
        strategy=ParallelStrategy.DATA_PARALLEL if num_gpus > 1 else ParallelStrategy.AUTO,
        data_parallel_enabled=num_gpus > 1,
        data_parallel_size=num_gpus,
        communication_backend=CommunicationBackend.GLOO if backend == "gloo" else CommunicationBackend.NCCL,
        master_addr="localhost",
        master_port=29500,
        enable_zero_optimization=False,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        enable_mixed_precision=False  # 简化演示
    )


def create_demo_topology(num_gpus=1):
    """创建演示GPU拓扑"""
    gpu_info = {}
    for i in range(num_gpus):
        gpu_info[i] = GPUInfo(
            gpu_id=i,
            name=f"Demo GPU {i}",
            memory_total=8000,  # 8GB
            memory_free=6000,   # 6GB可用
            compute_capability=(7, 5)
        )
    
    # 创建互联带宽矩阵
    interconnect_bandwidth = {}
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                interconnect_bandwidth[(i, j)] = 50.0  # 50 GB/s
    
    # NUMA拓扑（简化）
    numa_topology = {i: i // 4 for i in range(num_gpus)}  # 每4个GPU一个NUMA节点
    
    return GPUTopology(
        num_gpus=num_gpus,
        gpu_info=gpu_info,
        interconnect_bandwidth=interconnect_bandwidth,
        numa_topology=numa_topology
    )


def demo_backend_initialization():
    """演示分布式后端初始化"""
    logger.info("=== 分布式后端初始化演示 ===")
    
    config = create_demo_config(num_gpus=1, backend="gloo")
    initializer = DistributedBackendInitializer(config, logger)
    
    logger.info("测试环境变量设置...")
    initializer._setup_environment_variables(rank=0, world_size=1)
    
    logger.info("测试后端选择...")
    backend = initializer._select_backend()
    logger.info(f"选择的后端: {backend}")
    
    logger.info("测试端口可用性...")
    # 注意：实际的分布式初始化需要多进程环境
    logger.info("后端初始化演示完成")


def demo_gradient_synchronization():
    """演示梯度同步"""
    logger.info("=== 梯度同步演示 ===")
    
    config = create_demo_config()
    synchronizer = GradientSynchronizer(config, logger)
    
    # 创建模型
    model = DemoModel()
    
    # 创建虚拟梯度
    logger.info("创建虚拟梯度...")
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.1
    
    # 测试梯度裁剪
    logger.info("测试梯度裁剪...")
    grad_norm = synchronizer.clip_gradients(model, max_norm=1.0)
    logger.info(f"梯度范数: {grad_norm:.4f}")
    
    # 测试梯度聚合
    logger.info("测试梯度聚合...")
    test_gradients = {
        0: torch.tensor([1.0, 2.0, 3.0]),
        1: torch.tensor([4.0, 5.0, 6.0])
    }
    aggregated = synchronizer.aggregate_gradients(test_gradients)
    logger.info(f"聚合结果: {aggregated}")
    
    # 获取通信指标
    metrics = synchronizer.get_communication_metrics()
    logger.info(f"通信指标: {metrics.to_dict()}")
    
    logger.info("梯度同步演示完成")


def demo_parameter_update():
    """演示参数更新"""
    logger.info("=== 参数更新演示 ===")
    
    config = create_demo_config()
    manager = ParameterUpdateManager(config, logger)
    synchronizer = GradientSynchronizer(config, logger)
    
    # 创建模型和优化器
    model = DemoModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 设置梯度
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.01
    
    # 执行参数更新
    logger.info("执行参数更新...")
    success = manager.update_parameters(optimizer, synchronizer, model)
    logger.info(f"参数更新结果: {success}")
    
    # 获取更新统计
    stats = manager.get_update_stats()
    logger.info(f"更新统计: {stats}")
    
    logger.info("参数更新演示完成")


def demo_training_loop():
    """演示完整的训练循环"""
    logger.info("=== 完整训练循环演示 ===")
    
    # 创建配置和拓扑
    config = create_demo_config(num_gpus=1)
    topology = create_demo_topology(num_gpus=1)
    
    # 创建训练引擎
    engine = DistributedTrainingEngine(config, topology, logger)
    
    # 创建模型、数据和优化器
    model = DemoModel()
    dataset = create_demo_dataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    logger.info("开始训练循环...")
    
    # 模拟训练（不使用实际的分布式初始化）
    num_epochs = 3
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 模拟梯度同步（在实际分布式环境中会自动进行）
            if config.is_distributed:
                engine.gradient_synchronizer.synchronize_gradients(model)
            
            # 参数更新
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                logger.info(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"  Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
    
    # 获取训练指标
    metrics = engine.get_metrics()
    logger.info("训练指标:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("训练循环演示完成")


def demo_multi_gpu_simulation():
    """演示多GPU训练模拟"""
    logger.info("=== 多GPU训练模拟演示 ===")
    
    # 创建多GPU配置
    num_gpus = 2
    config = create_demo_config(num_gpus=num_gpus, backend="gloo")
    topology = create_demo_topology(num_gpus=num_gpus)
    
    logger.info(f"配置信息:")
    logger.info(f"  GPU数量: {topology.num_gpus}")
    logger.info(f"  并行策略: {config.strategy.value}")
    logger.info(f"  通信后端: {config.communication_backend.value}")
    logger.info(f"  数据并行大小: {config.data_parallel_size}")
    
    # 验证配置
    from src.parallel_config import ParallelConfigValidator
    validator = ParallelConfigValidator()
    validation_result = validator.validate_config(config, topology)
    
    logger.info("配置验证结果:")
    logger.info(f"  有效: {validation_result['valid']}")
    if validation_result['errors']:
        logger.info(f"  错误: {validation_result['errors']}")
    if validation_result['warnings']:
        logger.info(f"  警告: {validation_result['warnings']}")
    if validation_result['recommendations']:
        logger.info(f"  建议: {validation_result['recommendations']}")
    
    # 估算内存使用
    memory_usage = validator.estimate_memory_usage(config, model_size_gb=1.0)
    logger.info("内存使用估算:")
    for key, value in memory_usage.items():
        logger.info(f"  {key}: {value:.2f}")
    
    logger.info("多GPU训练模拟演示完成")


def demo_error_handling():
    """演示错误处理"""
    logger.info("=== 错误处理演示 ===")
    
    config = create_demo_config()
    synchronizer = GradientSynchronizer(config, logger)
    
    # 测试空梯度聚合
    logger.info("测试空梯度聚合错误处理...")
    try:
        synchronizer.aggregate_gradients({})
    except ValueError as e:
        logger.info(f"捕获预期错误: {e}")
    
    # 测试形状不匹配的梯度聚合
    logger.info("测试形状不匹配错误处理...")
    try:
        gradients = {
            0: torch.tensor([1.0, 2.0]),
            1: torch.tensor([1.0, 2.0, 3.0])
        }
        synchronizer.aggregate_gradients(gradients)
    except ValueError as e:
        logger.info(f"捕获预期错误: {e}")
    
    # 测试未初始化引擎的模型设置
    logger.info("测试未初始化引擎错误处理...")
    topology = create_demo_topology()
    engine = DistributedTrainingEngine(config, topology, logger)
    
    try:
        model = DemoModel()
        engine.setup_model(model)
    except RuntimeError as e:
        logger.info(f"捕获预期错误: {e}")
    
    logger.info("错误处理演示完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分布式训练引擎演示")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "backend", "gradient", "parameter", "training", "multi_gpu", "error"],
                       help="选择要运行的演示")
    
    args = parser.parse_args()
    
    logger.info("开始分布式训练引擎演示")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
    
    try:
        if args.demo == "all" or args.demo == "backend":
            demo_backend_initialization()
            print()
        
        if args.demo == "all" or args.demo == "gradient":
            demo_gradient_synchronization()
            print()
        
        if args.demo == "all" or args.demo == "parameter":
            demo_parameter_update()
            print()
        
        if args.demo == "all" or args.demo == "training":
            demo_training_loop()
            print()
        
        if args.demo == "all" or args.demo == "multi_gpu":
            demo_multi_gpu_simulation()
            print()
        
        if args.demo == "all" or args.demo == "error":
            demo_error_handling()
            print()
        
        logger.info("所有演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()