#!/usr/bin/env python3
"""
跨平台GPU检测功能测试
"""

import sys
import os
import platform

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gpu_utils import GPUDetector


def test_cross_platform_gpu_detection():
    """测试跨平台GPU检测功能"""
    print(f"当前操作系统: {platform.system()}")
    print("=" * 60)
    
    # 创建GPU检测器
    detector = GPUDetector()
    
    # 测试CUDA可用性检测
    print("🔍 检测CUDA可用性...")
    cuda_available = detector.detect_cuda_availability()
    print(f"CUDA可用: {'是' if cuda_available else '否'}")
    
    # 测试GPU信息获取
    print("\n🔍 获取GPU信息...")
    gpu_infos = detector.get_all_gpu_info()
    
    if gpu_infos:
        print(f"检测到 {len(gpu_infos)} 个GPU:")
        for gpu in gpu_infos:
            print(f"  GPU {gpu.gpu_id}: {gpu.name}")
            print(f"    总内存: {gpu.total_memory}MB")
            print(f"    可用内存: {gpu.free_memory}MB")
            print(f"    利用率: {gpu.utilization}%")
            if gpu.temperature is not None:
                print(f"    温度: {gpu.temperature}°C")
            if gpu.power_usage is not None:
                print(f"    功耗: {gpu.power_usage}W")
            if gpu.pci_bus_id:
                print(f"    PCI总线ID: {gpu.pci_bus_id}")
            if gpu.numa_node is not None:
                print(f"    NUMA节点: {gpu.numa_node}")
            print()
    else:
        print("未检测到GPU")
    
    # 测试NUMA拓扑信息
    print("🔍 获取NUMA拓扑信息...")
    numa_info = detector.get_numa_topology_info()
    print(f"NUMA库可用: {'是' if numa_info['numa_available'] else '否'}")
    print(f"平台: {numa_info['platform']}")
    
    if numa_info["numa_nodes"]:
        print(f"NUMA节点: {numa_info['numa_nodes']}")
        for node_id in numa_info["numa_nodes"]:
            memory = numa_info["memory_per_node"].get(node_id, 0)
            if memory:
                memory_gb = memory // (1024**3)
                print(f"  节点{node_id}: {memory_gb}GB内存")
    else:
        print("未检测到NUMA节点信息")
    
    # 测试GPU拓扑检测
    if len(gpu_infos) > 1:
        print("\n🔍 检测GPU拓扑...")
        topology = detector.detect_gpu_topology()
        print(f"GPU数量: {topology.num_gpus}")
        print(f"拓扑类型: {topology.topology_type}")
        
        if topology.numa_topology:
            print("GPU NUMA分布:")
            for gpu_id, numa_node in topology.numa_topology.items():
                print(f"  GPU {gpu_id} -> NUMA节点 {numa_node}")
        
        if topology.interconnects:
            print("GPU互联:")
            for ic in topology.interconnects:
                print(f"  GPU {ic.gpu_a} <-> GPU {ic.gpu_b}: "
                      f"{ic.interconnect_type.value} ({ic.bandwidth_gbps:.1f} GB/s)")
    
    # 测试Qwen3-4B需求验证
    print("\n🔍 验证Qwen3-4B运行需求...")
    validation = detector.validate_qwen_requirements()
    
    print("需求验证结果:")
    for requirement, passed in validation.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {requirement}")
    
    # 测试优化建议
    print("\n💡 获取优化建议...")
    recommendations = detector.get_optimization_recommendations()
    for rec in recommendations:
        print(f"  {rec}")
    
    # 测试硬件兼容性检查
    print("\n🔍 检查硬件兼容性...")
    compatibility = detector.check_hardware_compatibility()
    
    print(f"兼容性评分: {compatibility.compatibility_score:.1f}/100")
    print(f"是否兼容: {'是' if compatibility.is_compatible else '否'}")
    
    if compatibility.issues:
        print("问题:")
        for issue in compatibility.issues:
            print(f"  ❌ {issue}")
    
    if compatibility.warnings:
        print("警告:")
        for warning in compatibility.warnings:
            print(f"  ⚠️ {warning}")
    
    if compatibility.recommendations:
        print("建议:")
        for rec in compatibility.recommendations:
            print(f"  💡 {rec}")
    
    return len(gpu_infos) > 0


def test_system_report():
    """测试系统报告生成"""
    print("\n" + "=" * 60)
    print("生成系统环境报告")
    print("=" * 60)
    
    detector = GPUDetector()
    report = detector.generate_system_report()
    print(report)


def test_platform_specific_features():
    """测试平台特定功能"""
    print("\n" + "=" * 60)
    print("测试平台特定功能")
    print("=" * 60)
    
    detector = GPUDetector()
    
    if platform.system() == "Windows":
        print("Windows平台特定测试:")
        
        # 测试WMI检测
        try:
            wmi_gpus = detector._detect_gpus_wmi()
            print(f"  WMI检测到 {len(wmi_gpus)} 个GPU")
        except Exception as e:
            print(f"  WMI检测失败: {e}")
        
        # 测试WMIC检测
        try:
            wmic_gpus = detector._detect_gpus_wmic()
            print(f"  WMIC检测到 {len(wmic_gpus)} 个GPU")
        except Exception as e:
            print(f"  WMIC检测失败: {e}")
    
    elif platform.system() == "Linux":
        print("Linux平台特定测试:")
        
        # 测试lspci检测
        try:
            lspci_gpus = detector._detect_gpus_lspci()
            print(f"  lspci检测到 {len(lspci_gpus)} 个GPU")
        except Exception as e:
            print(f"  lspci检测失败: {e}")
        
        # 测试proc检测
        try:
            proc_gpus = detector._detect_gpus_proc_nvidia()
            print(f"  /proc检测到 {len(proc_gpus)} 个GPU")
        except Exception as e:
            print(f"  /proc检测失败: {e}")
    
    elif platform.system() == "Darwin":
        print("macOS平台特定测试:")
        
        # 测试system_profiler检测
        try:
            profiler_gpus = detector._detect_gpus_system_profiler()
            print(f"  system_profiler检测到 {len(profiler_gpus)} 个GPU")
        except Exception as e:
            print(f"  system_profiler检测失败: {e}")
    
    # 测试nvidia-smi检测（所有平台）
    try:
        smi_gpus = detector._detect_gpus_nvidia_smi()
        print(f"  nvidia-smi检测到 {len(smi_gpus)} 个GPU")
    except Exception as e:
        print(f"  nvidia-smi检测失败: {e}")


if __name__ == "__main__":
    print("跨平台GPU检测功能测试")
    print("=" * 60)
    
    try:
        # 基本GPU检测测试
        has_gpu = test_cross_platform_gpu_detection()
        
        # 系统报告测试
        test_system_report()
        
        # 平台特定功能测试
        test_platform_specific_features()
        
        print("\n" + "=" * 60)
        if has_gpu:
            print("🎉 GPU检测功能测试完成，检测到GPU设备！")
        else:
            print("✅ GPU检测功能测试完成，未检测到GPU设备（这在某些环境中是正常的）")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()