"""
GPU检测和验证功能模块
针对Qwen3-4B模型内存需求优化的GPU管理工具
支持GPU拓扑检测、互联带宽分析和NUMA拓扑检测
"""

import torch
import psutil
import logging
import subprocess
import platform
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")

# 尝试导入numa库（主要在Linux系统上可用）
NUMA_AVAILABLE = False
LIBNUMA_AVAILABLE = False

# 只在Linux系统上尝试导入numa库
if platform.system() == "Linux":
    try:
        import numa
        NUMA_AVAILABLE = True
        LIBNUMA_AVAILABLE = True
    except ImportError:
        pass
    except Exception as e:
        # 处理其他导入错误（如库依赖问题）
        logging.debug(f"numa库导入失败: {e}")

# 如果numa库不可用，记录警告（但这在Windows上是正常的）
if not NUMA_AVAILABLE and platform.system() == "Linux":
    logging.warning("numa library not available on Linux, NUMA topology detection will be limited")
elif not NUMA_AVAILABLE and platform.system() != "Linux":
    logging.info("numa library not available on non-Linux system, this is expected")


class InterconnectType(Enum):
    """GPU互联类型枚举"""
    NVLINK = "NVLink"
    PCIE = "PCIe"
    NVSWITCH = "NVSwitch"
    UNKNOWN = "Unknown"


@dataclass
class GPUInfo:
    """GPU信息数据类"""
    gpu_id: int
    name: str
    total_memory: int  # MB
    free_memory: int   # MB
    used_memory: int   # MB
    utilization: float # 百分比
    temperature: Optional[int] = None
    power_usage: Optional[int] = None
    # 新增拓扑相关字段
    pci_bus_id: Optional[str] = None
    numa_node: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    multi_processor_count: Optional[int] = None
    max_threads_per_block: Optional[int] = None


@dataclass
class GPUInterconnect:
    """GPU互联信息"""
    gpu_a: int
    gpu_b: int
    interconnect_type: InterconnectType
    bandwidth_gbps: float
    bidirectional: bool = True
    link_count: Optional[int] = None


@dataclass
class GPUTopology:
    """GPU拓扑结构信息"""
    num_gpus: int
    gpu_info: Dict[int, GPUInfo] = field(default_factory=dict)
    interconnects: List[GPUInterconnect] = field(default_factory=list)
    numa_topology: Dict[int, int] = field(default_factory=dict)  # gpu_id -> numa_node
    bandwidth_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    topology_type: str = "Unknown"  # "Single", "NVLink", "PCIe", "Mixed"
    
    def get_bandwidth(self, gpu_a: int, gpu_b: int) -> float:
        """获取两个GPU之间的带宽"""
        key = (min(gpu_a, gpu_b), max(gpu_a, gpu_b))
        return self.bandwidth_matrix.get(key, 0.0)
    
    def get_numa_distance(self, gpu_a: int, gpu_b: int) -> int:
        """获取两个GPU之间的NUMA距离"""
        numa_a = self.numa_topology.get(gpu_a, -1)
        numa_b = self.numa_topology.get(gpu_b, -1)
        
        if numa_a == -1 or numa_b == -1:
            return -1  # 未知
        
        if numa_a == numa_b:
            return 0  # 同一NUMA节点
        else:
            return 1  # 不同NUMA节点，简化为距离1
    
    def get_optimal_gpu_pairs(self) -> List[Tuple[int, int]]:
        """获取最优的GPU配对（基于带宽和NUMA距离）"""
        pairs = []
        gpus = list(self.gpu_info.keys())
        
        # 按带宽排序所有GPU对
        gpu_pairs = []
        for i in range(len(gpus)):
            for j in range(i + 1, len(gpus)):
                gpu_a, gpu_b = gpus[i], gpus[j]
                bandwidth = self.get_bandwidth(gpu_a, gpu_b)
                numa_distance = self.get_numa_distance(gpu_a, gpu_b)
                # 优先级：带宽高，NUMA距离小
                priority = bandwidth * 1000 - numa_distance * 100
                gpu_pairs.append((gpu_a, gpu_b, priority))
        
        # 按优先级排序
        gpu_pairs.sort(key=lambda x: x[2], reverse=True)
        return [(pair[0], pair[1]) for pair in gpu_pairs]


@dataclass
class SystemRequirements:
    """Qwen3-4B模型系统需求"""
    min_gpu_memory: int = 8192  # MB, 最小8GB
    recommended_gpu_memory: int = 16384  # MB, 推荐16GB
    min_system_memory: int = 16384  # MB, 最小16GB系统内存
    min_python_version: Tuple[int, int] = (3, 12)
    required_cuda_version: str = "12.9"
    # 新增硬件兼容性需求
    min_compute_capability: Tuple[int, int] = (7, 0)  # 最小计算能力7.0
    recommended_compute_capability: Tuple[int, int] = (8, 0)  # 推荐8.0+
    min_nvlink_bandwidth: float = 25.0  # GB/s, NVLink最小带宽
    min_pcie_bandwidth: float = 16.0  # GB/s, PCIe最小带宽
    max_numa_distance: int = 1  # 最大NUMA距离


@dataclass
class HardwareCompatibility:
    """硬件兼容性检查结果"""
    is_compatible: bool
    compatibility_score: float  # 0-100分
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str):
        """添加兼容性问题"""
        self.issues.append(issue)
        self.is_compatible = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
    
    def add_recommendation(self, recommendation: str):
        """添加建议"""
        self.recommendations.append(recommendation)


class GPUDetector:
    """GPU检测和验证类，支持拓扑检测和硬件兼容性检查"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = SystemRequirements()
        self._initialize_pynvml()
        self._topology_cache: Optional[GPUTopology] = None
    
    def _initialize_pynvml(self) -> None:
        """初始化NVIDIA管理库"""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.logger.info("NVIDIA管理库初始化成功")
            except Exception as e:
                self.logger.warning(f"NVIDIA管理库初始化失败: {e}")
                globals()['PYNVML_AVAILABLE'] = False
    
    def detect_cuda_availability(self) -> bool:
        """检测CUDA可用性"""
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                self.logger.info(f"CUDA可用: 版本 {cuda_version}, 设备数量: {device_count}")
                return True
            else:
                self.logger.error("CUDA不可用")
                return False
        except Exception as e:
            self.logger.error(f"CUDA检测失败: {e}")
            return False
    
    def get_gpu_info(self, gpu_id: int = 0) -> Optional[GPUInfo]:
        """获取指定GPU信息，包含拓扑相关信息"""
        try:
            if not torch.cuda.is_available():
                return None
            
            device = torch.device(f'cuda:{gpu_id}')
            props = torch.cuda.get_device_properties(device)
            
            # 获取内存信息
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory // (1024**2)
            torch.cuda.empty_cache()  # 清理缓存以获得准确的空闲内存
            free_memory = (torch.cuda.get_device_properties(gpu_id).total_memory - 
                          torch.cuda.memory_allocated(gpu_id)) // (1024**2)
            used_memory = total_memory - free_memory
            
            gpu_info = GPUInfo(
                gpu_id=gpu_id,
                name=props.name,
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
                utilization=0.0,  # PyTorch不直接提供利用率
                compute_capability=(props.major, props.minor),
                multi_processor_count=props.multi_processor_count,
                max_threads_per_block=getattr(props, 'max_threads_per_block', None)
            )
            
            # 如果pynvml可用，获取更详细信息
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
                    
                    # 获取PCI总线ID
                    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                    if hasattr(pci_info.busId, 'decode'):
                        gpu_info.pci_bus_id = pci_info.busId.decode('utf-8')
                    else:
                        gpu_info.pci_bus_id = str(pci_info.busId)
                    
                    gpu_info.utilization = util.gpu
                    gpu_info.temperature = temp
                    gpu_info.power_usage = power
                except Exception as e:
                    self.logger.warning(f"获取GPU详细信息失败: {e}")
            
            # 获取NUMA节点信息
            gpu_info.numa_node = self._get_gpu_numa_node(gpu_id)
            
            return gpu_info
            
        except Exception as e:
            self.logger.error(f"获取GPU信息失败: {e}")
            return None
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有GPU信息"""
        gpu_infos = []
        if not torch.cuda.is_available():
            return gpu_infos
        
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpu_info = self.get_gpu_info(i)
            if gpu_info:
                gpu_infos.append(gpu_info)
        
        return gpu_infos
    
    def validate_qwen_requirements(self, gpu_id: int = 0) -> Dict[str, bool]:
        """验证Qwen3-4B模型运行需求"""
        validation_results = {
            "cuda_available": False,
            "sufficient_gpu_memory": False,
            "sufficient_system_memory": False,
            "python_version_ok": False,
            "recommended_setup": False
        }
        
        # 检查CUDA
        validation_results["cuda_available"] = self.detect_cuda_availability()
        
        # 检查GPU内存
        gpu_info = self.get_gpu_info(gpu_id)
        if gpu_info:
            validation_results["sufficient_gpu_memory"] = (
                gpu_info.free_memory >= self.requirements.min_gpu_memory
            )
            validation_results["recommended_setup"] = (
                gpu_info.total_memory >= self.requirements.recommended_gpu_memory
            )
        
        # 检查系统内存
        system_memory = psutil.virtual_memory().total // (1024**2)  # MB
        validation_results["sufficient_system_memory"] = (
            system_memory >= self.requirements.min_system_memory
        )
        
        # 检查Python版本
        import sys
        python_version = sys.version_info[:2]
        validation_results["python_version_ok"] = (
            python_version >= self.requirements.min_python_version
        )
        
        return validation_results
    
    def get_optimization_recommendations(self, gpu_id: int = 0) -> List[str]:
        """获取针对Qwen3-4B模型的优化建议"""
        recommendations = []
        
        gpu_info = self.get_gpu_info(gpu_id)
        if not gpu_info:
            recommendations.append("无法检测到GPU，建议检查CUDA安装")
            return recommendations
        
        # 内存优化建议
        if gpu_info.total_memory < self.requirements.recommended_gpu_memory:
            recommendations.append(
                f"GPU内存 ({gpu_info.total_memory}MB) 低于推荐值 "
                f"({self.requirements.recommended_gpu_memory}MB)，建议启用以下优化："
            )
            recommendations.append("- 启用梯度检查点 (gradient_checkpointing=True)")
            recommendations.append("- 使用混合精度训练 (fp16=True)")
            recommendations.append("- 减小批次大小 (batch_size=1-2)")
            recommendations.append("- 启用LoRA微调以减少参数量")
        
        if gpu_info.total_memory >= self.requirements.recommended_gpu_memory:
            recommendations.append("GPU内存充足，可以使用标准配置进行训练")
            recommendations.append("- 可以使用较大的批次大小 (batch_size=4-8)")
            recommendations.append("- 可以考虑全参数微调")
        
        # 多GPU建议
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            total_memory = sum(self.get_gpu_info(i).total_memory for i in range(gpu_count))
            recommendations.append(f"检测到 {gpu_count} 个GPU，总内存 {total_memory}MB")
            recommendations.append("- 建议启用数据并行训练")
            recommendations.append("- 可以考虑模型并行以处理更大的模型")
        
        return recommendations
    
    def generate_system_report(self) -> str:
        """生成系统环境报告"""
        report_lines = []
        report_lines.append("=== Qwen3-4B-Thinking 系统环境报告 ===")
        report_lines.append("")
        
        # 基本信息
        import sys
        report_lines.append(f"Python版本: {sys.version}")
        report_lines.append(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            report_lines.append(f"CUDA版本: {torch.version.cuda}")
            report_lines.append(f"cuDNN版本: {torch.backends.cudnn.version()}")
        
        # 系统内存
        memory = psutil.virtual_memory()
        report_lines.append(f"系统内存: {memory.total // (1024**3)}GB "
                          f"(可用: {memory.available // (1024**3)}GB)")
        
        # GPU信息
        gpu_infos = self.get_all_gpu_info()
        if gpu_infos:
            report_lines.append("")
            report_lines.append("GPU信息:")
            for gpu_info in gpu_infos:
                report_lines.append(f"  GPU {gpu_info.gpu_id}: {gpu_info.name}")
                report_lines.append(f"    总内存: {gpu_info.total_memory}MB")
                report_lines.append(f"    可用内存: {gpu_info.free_memory}MB")
                if gpu_info.utilization is not None:
                    report_lines.append(f"    利用率: {gpu_info.utilization}%")
                if gpu_info.temperature is not None:
                    report_lines.append(f"    温度: {gpu_info.temperature}°C")
        else:
            report_lines.append("未检测到可用GPU")
        
        # 需求验证
        validation = self.validate_qwen_requirements()
        report_lines.append("")
        report_lines.append("Qwen3-4B需求验证:")
        for requirement, passed in validation.items():
            status = "✓" if passed else "✗"
            report_lines.append(f"  {status} {requirement}")
        
        # NUMA拓扑信息
        numa_info = self.get_numa_topology_info()
        report_lines.append("")
        report_lines.append("NUMA拓扑信息:")
        report_lines.append(f"  NUMA库可用: {'是' if numa_info['numa_available'] else '否'}")
        if numa_info["numa_nodes"]:
            report_lines.append(f"  NUMA节点: {numa_info['numa_nodes']}")
            for node_id, memory in numa_info["memory_per_node"].items():
                memory_gb = memory // (1024**3) if memory else "未知"
                report_lines.append(f"    节点{node_id}: {memory_gb}GB内存")
        else:
            report_lines.append("  未检测到NUMA节点信息")
        
        # GPU拓扑信息
        topology = self.detect_gpu_topology()
        if topology.num_gpus > 1:
            report_lines.append("")
            report_lines.append("GPU拓扑信息:")
            report_lines.append(f"  拓扑类型: {topology.topology_type}")
            if topology.numa_topology:
                report_lines.append("  GPU NUMA分布:")
                for gpu_id, numa_node in topology.numa_topology.items():
                    report_lines.append(f"    GPU {gpu_id} -> NUMA节点 {numa_node}")
            
            if topology.interconnects:
                report_lines.append("  GPU互联:")
                for ic in topology.interconnects:
                    report_lines.append(f"    GPU {ic.gpu_a} <-> GPU {ic.gpu_b}: "
                                      f"{ic.interconnect_type.value} ({ic.bandwidth_gbps:.1f} GB/s)")
        
        # 优化建议
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            report_lines.append("")
            report_lines.append("优化建议:")
            for rec in recommendations:
                report_lines.append(f"  {rec}")
        
        return "\n".join(report_lines)
    
    def _get_gpu_numa_node(self, gpu_id: int) -> Optional[int]:
        """获取GPU所在的NUMA节点，支持多种检测方法"""
        try:
            if platform.system() != "Linux":
                return None
            
            # 方法1: 使用numa库
            if NUMA_AVAILABLE:
                try:
                    # 获取当前进程的NUMA节点
                    current_node = numa.get_mempolicy()[1]
                    if current_node is not None and len(current_node) > 0:
                        return list(current_node)[0]
                except Exception as e:
                    self.logger.debug(f"numa库检测失败: {e}")
            
            # 方法2: 使用numa库的其他功能
            if NUMA_AVAILABLE:
                try:
                    # 尝试获取NUMA节点信息
                    from numa import info
                    numa_nodes = info.numa_hardware_info()
                    if numa_nodes:
                        # 简化处理：根据GPU ID分配NUMA节点
                        return gpu_id % len(numa_nodes.get('nodes', [0]))
                except Exception as e:
                    self.logger.debug(f"numa info检测失败: {e}")
            
            # 方法3: 通过nvidia-ml-py获取PCI总线ID并查询sysfs
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                if hasattr(pci_info.busId, 'decode'):
                    pci_bus_id = pci_info.busId.decode('utf-8')
                else:
                    pci_bus_id = str(pci_info.busId)
                
                # 尝试多种PCI路径格式
                possible_paths = [
                    f"/sys/bus/pci/devices/0000:{pci_bus_id}/numa_node",
                    f"/sys/bus/pci/devices/{pci_bus_id}/numa_node",
                    f"/sys/class/pci_bus/0000:{pci_bus_id.split(':')[0]}/device/numa_node"
                ]
                
                for numa_path in possible_paths:
                    if Path(numa_path).exists():
                        try:
                            with open(numa_path, 'r') as f:
                                numa_node = int(f.read().strip())
                                return numa_node if numa_node >= 0 else None
                        except (ValueError, IOError):
                            continue
            
            # 方法4: 通过/proc/cpuinfo和GPU亲和性推断
            return self._infer_numa_from_cpu_affinity(gpu_id)
            
        except Exception as e:
            self.logger.debug(f"获取GPU NUMA节点失败: {e}")
            return None
    
    def _infer_numa_from_cpu_affinity(self, gpu_id: int) -> Optional[int]:
        """通过CPU亲和性推断NUMA节点"""
        try:
            # 检查/proc/cpuinfo获取NUMA信息
            numa_nodes = set()
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'physical id' in line:
                        numa_nodes.add(int(line.split(':')[1].strip()))
            
            if numa_nodes:
                # 简化处理：根据GPU ID分配到不同NUMA节点
                return gpu_id % len(numa_nodes)
            
            return None
        except Exception:
            return None
    
    def get_numa_topology_info(self) -> Dict[str, any]:
        """获取系统NUMA拓扑信息"""
        numa_info = {
            "numa_available": NUMA_AVAILABLE or LIBNUMA_AVAILABLE,
            "numa_nodes": [],
            "memory_per_node": {},
            "cpu_per_node": {},
            "distances": {}
        }
        
        try:
            if NUMA_AVAILABLE:
                # 使用numa库获取详细信息
                try:
                    numa_info["numa_nodes"] = list(range(numa.get_max_node() + 1))
                    for node in numa_info["numa_nodes"]:
                        numa_info["memory_per_node"][node] = numa.node_memsize(node)
                except Exception as e:
                    self.logger.debug(f"numa库获取拓扑信息失败: {e}")
            
            elif NUMA_AVAILABLE:
                # 使用numa库的info模块获取信息
                try:
                    from numa import info
                    hardware_info = info.numa_hardware_info()
                    if hardware_info and 'nodes' in hardware_info:
                        numa_info["numa_nodes"] = list(hardware_info['nodes'].keys())
                        for node_id in numa_info["numa_nodes"]:
                            # 尝试获取内存信息
                            try:
                                from numa import memory
                                # 这里简化处理，实际可能需要更复杂的逻辑
                                numa_info["memory_per_node"][node_id] = "未知"
                            except Exception:
                                pass
                except Exception as e:
                    self.logger.debug(f"numa info获取拓扑信息失败: {e}")
            
            else:
                # 从/sys/devices/system/node/读取NUMA信息
                node_path = Path("/sys/devices/system/node")
                if node_path.exists():
                    numa_dirs = [d for d in node_path.iterdir() 
                               if d.is_dir() and d.name.startswith('node')]
                    numa_info["numa_nodes"] = [int(d.name[4:]) for d in numa_dirs]
                    
                    for node_id in numa_info["numa_nodes"]:
                        # 读取内存信息
                        meminfo_path = node_path / f"node{node_id}" / "meminfo"
                        if meminfo_path.exists():
                            try:
                                with open(meminfo_path, 'r') as f:
                                    for line in f:
                                        if 'MemTotal:' in line:
                                            memory_kb = int(line.split()[3])
                                            numa_info["memory_per_node"][node_id] = memory_kb * 1024
                                            break
                            except Exception:
                                pass
        
        except Exception as e:
            self.logger.debug(f"获取NUMA拓扑信息失败: {e}")
        
        return numa_info
    
    def detect_gpu_topology(self) -> GPUTopology:
        """检测GPU拓扑结构"""
        if self._topology_cache is not None:
            return self._topology_cache
        
        try:
            if not torch.cuda.is_available():
                return GPUTopology(num_gpus=0)
            
            num_gpus = torch.cuda.device_count()
            topology = GPUTopology(num_gpus=num_gpus)
            
            # 获取所有GPU信息
            for i in range(num_gpus):
                gpu_info = self.get_gpu_info(i)
                if gpu_info:
                    topology.gpu_info[i] = gpu_info
                    if gpu_info.numa_node is not None:
                        topology.numa_topology[i] = gpu_info.numa_node
            
            # 检测GPU互联
            topology.interconnects = self._detect_gpu_interconnects(num_gpus)
            
            # 构建带宽矩阵
            topology.bandwidth_matrix = self._build_bandwidth_matrix(topology.interconnects)
            
            # 确定拓扑类型
            topology.topology_type = self._determine_topology_type(topology.interconnects)
            
            self._topology_cache = topology
            return topology
            
        except Exception as e:
            self.logger.error(f"检测GPU拓扑失败: {e}")
            return GPUTopology(num_gpus=0)
    
    def _detect_gpu_interconnects(self, num_gpus: int) -> List[GPUInterconnect]:
        """检测GPU之间的互联"""
        interconnects = []
        
        try:
            if not PYNVML_AVAILABLE or num_gpus < 2:
                return interconnects
            
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    interconnect = self._detect_gpu_pair_interconnect(i, j)
                    if interconnect:
                        interconnects.append(interconnect)
            
            return interconnects
            
        except Exception as e:
            self.logger.error(f"检测GPU互联失败: {e}")
            return interconnects
    
    def _detect_gpu_pair_interconnect(self, gpu_a: int, gpu_b: int) -> Optional[GPUInterconnect]:
        """检测两个GPU之间的互联类型和带宽"""
        try:
            if not PYNVML_AVAILABLE:
                return None
            
            handle_a = pynvml.nvmlDeviceGetHandleByIndex(gpu_a)
            handle_b = pynvml.nvmlDeviceGetHandleByIndex(gpu_b)
            
            # 检查NVLink连接
            try:
                # 尝试获取NVLink状态
                nvlink_bandwidth = 0.0
                nvlink_count = 0
                
                # 检查所有可能的NVLink端口
                for link_id in range(6):  # 大多数GPU最多6个NVLink端口
                    try:
                        # 检查从gpu_a到gpu_b的NVLink
                        remote_info = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle_a, link_id)
                        if hasattr(remote_info.busId, 'decode'):
                            remote_pci = remote_info.busId.decode('utf-8')
                        else:
                            remote_pci = str(remote_info.busId)
                        
                        # 获取gpu_b的PCI信息进行比较
                        pci_info_b = pynvml.nvmlDeviceGetPciInfo(handle_b)
                        if hasattr(pci_info_b.busId, 'decode'):
                            pci_b = pci_info_b.busId.decode('utf-8')
                        else:
                            pci_b = str(pci_info_b.busId)
                        
                        if remote_pci == pci_b:
                            # 找到NVLink连接
                            nvlink_count += 1
                            # 假设每个NVLink 3.0连接提供25GB/s带宽
                            nvlink_bandwidth += 25.0
                    except:
                        continue
                
                if nvlink_count > 0:
                    return GPUInterconnect(
                        gpu_a=gpu_a,
                        gpu_b=gpu_b,
                        interconnect_type=InterconnectType.NVLINK,
                        bandwidth_gbps=nvlink_bandwidth,
                        link_count=nvlink_count
                    )
            except:
                pass
            
            # 如果没有NVLink，假设是PCIe连接
            # 估算PCIe带宽（简化处理）
            pcie_bandwidth = 16.0  # PCIe 4.0 x16的理论带宽
            
            return GPUInterconnect(
                gpu_a=gpu_a,
                gpu_b=gpu_b,
                interconnect_type=InterconnectType.PCIE,
                bandwidth_gbps=pcie_bandwidth
            )
            
        except Exception as e:
            self.logger.debug(f"检测GPU {gpu_a}-{gpu_b}互联失败: {e}")
            return None
    
    def _build_bandwidth_matrix(self, interconnects: List[GPUInterconnect]) -> Dict[Tuple[int, int], float]:
        """构建带宽矩阵"""
        bandwidth_matrix = {}
        
        for interconnect in interconnects:
            key = (min(interconnect.gpu_a, interconnect.gpu_b), 
                   max(interconnect.gpu_a, interconnect.gpu_b))
            bandwidth_matrix[key] = interconnect.bandwidth_gbps
        
        return bandwidth_matrix
    
    def _determine_topology_type(self, interconnects: List[GPUInterconnect]) -> str:
        """确定拓扑类型"""
        if not interconnects:
            return "Single"
        
        nvlink_count = sum(1 for ic in interconnects if ic.interconnect_type == InterconnectType.NVLINK)
        pcie_count = sum(1 for ic in interconnects if ic.interconnect_type == InterconnectType.PCIE)
        
        if nvlink_count > 0 and pcie_count == 0:
            return "NVLink"
        elif pcie_count > 0 and nvlink_count == 0:
            return "PCIe"
        elif nvlink_count > 0 and pcie_count > 0:
            return "Mixed"
        else:
            return "Unknown"
    
    def analyze_interconnect_bandwidth(self) -> Dict[str, float]:
        """分析GPU互联带宽"""
        topology = self.detect_gpu_topology()
        
        analysis = {
            "total_bandwidth": 0.0,
            "avg_bandwidth": 0.0,
            "min_bandwidth": float('inf'),
            "max_bandwidth": 0.0,
            "nvlink_bandwidth": 0.0,
            "pcie_bandwidth": 0.0
        }
        
        if not topology.interconnects:
            return analysis
        
        bandwidths = []
        for interconnect in topology.interconnects:
            bandwidth = interconnect.bandwidth_gbps
            bandwidths.append(bandwidth)
            analysis["total_bandwidth"] += bandwidth
            analysis["min_bandwidth"] = min(analysis["min_bandwidth"], bandwidth)
            analysis["max_bandwidth"] = max(analysis["max_bandwidth"], bandwidth)
            
            if interconnect.interconnect_type == InterconnectType.NVLINK:
                analysis["nvlink_bandwidth"] += bandwidth
            elif interconnect.interconnect_type == InterconnectType.PCIE:
                analysis["pcie_bandwidth"] += bandwidth
        
        if bandwidths:
            analysis["avg_bandwidth"] = sum(bandwidths) / len(bandwidths)
        
        if analysis["min_bandwidth"] == float('inf'):
            analysis["min_bandwidth"] = 0.0
        
        return analysis
    
    def check_hardware_compatibility(self) -> HardwareCompatibility:
        """检查硬件兼容性"""
        compatibility = HardwareCompatibility(is_compatible=True, compatibility_score=100.0)
        
        try:
            # 检查基本CUDA可用性
            if not self.detect_cuda_availability():
                compatibility.add_issue("CUDA不可用，无法进行GPU训练")
                compatibility.compatibility_score -= 50
                return compatibility
            
            topology = self.detect_gpu_topology()
            
            if topology.num_gpus == 0 or len(topology.gpu_info) == 0:
                compatibility.add_issue("未检测到可用GPU")
                compatibility.compatibility_score -= 50
                return compatibility
            
            # 检查GPU内存
            insufficient_memory_gpus = []
            for gpu_id, gpu_info in topology.gpu_info.items():
                if gpu_info.total_memory < self.requirements.min_gpu_memory:
                    insufficient_memory_gpus.append(gpu_id)
            
            if insufficient_memory_gpus:
                compatibility.add_issue(
                    f"GPU {insufficient_memory_gpus} 内存不足 "
                    f"(需要至少 {self.requirements.min_gpu_memory}MB)"
                )
                compatibility.compatibility_score -= 30
            
            # 检查计算能力
            low_compute_gpus = []
            for gpu_id, gpu_info in topology.gpu_info.items():
                if (gpu_info.compute_capability and 
                    gpu_info.compute_capability < self.requirements.min_compute_capability):
                    low_compute_gpus.append(gpu_id)
            
            if low_compute_gpus:
                compatibility.add_warning(
                    f"GPU {low_compute_gpus} 计算能力较低，可能影响性能"
                )
                compatibility.compatibility_score -= 10
            
            # 检查多GPU互联
            if topology.num_gpus > 1:
                bandwidth_analysis = self.analyze_interconnect_bandwidth()
                
                if bandwidth_analysis["min_bandwidth"] < self.requirements.min_pcie_bandwidth:
                    compatibility.add_warning(
                        f"GPU间带宽较低 ({bandwidth_analysis['min_bandwidth']:.1f} GB/s)，"
                        f"可能影响多GPU训练效率"
                    )
                    compatibility.compatibility_score -= 15
                
                # 检查NUMA拓扑
                numa_distances = []
                gpus = list(topology.gpu_info.keys())
                for i in range(len(gpus)):
                    for j in range(i + 1, len(gpus)):
                        distance = topology.get_numa_distance(gpus[i], gpus[j])
                        if distance > 0:
                            numa_distances.append(distance)
                
                if numa_distances and max(numa_distances) > self.requirements.max_numa_distance:
                    compatibility.add_warning(
                        "检测到跨NUMA节点的GPU配置，可能影响通信效率"
                    )
                    compatibility.compatibility_score -= 10
            
            # 生成建议
            self._generate_compatibility_recommendations(compatibility, topology)
            
            return compatibility
            
        except Exception as e:
            self.logger.error(f"硬件兼容性检查失败: {e}")
            compatibility.add_issue(f"兼容性检查失败: {e}")
            compatibility.is_compatible = False
            compatibility.compatibility_score = 0.0
            return compatibility
    
    def _generate_compatibility_recommendations(self, compatibility: HardwareCompatibility, 
                                              topology: GPUTopology):
        """生成兼容性建议"""
        if topology.num_gpus == 1:
            gpu_info = list(topology.gpu_info.values())[0]
            if gpu_info.total_memory < self.requirements.recommended_gpu_memory:
                compatibility.add_recommendation("启用LoRA微调以减少内存使用")
                compatibility.add_recommendation("使用混合精度训练(FP16)")
                compatibility.add_recommendation("启用梯度检查点")
        
        elif topology.num_gpus > 1:
            if topology.topology_type == "NVLink":
                compatibility.add_recommendation("检测到NVLink连接，建议启用数据并行训练")
            elif topology.topology_type == "PCIe":
                compatibility.add_recommendation("使用PCIe连接，建议优化通信策略")
            
            bandwidth_analysis = self.analyze_interconnect_bandwidth()
            if bandwidth_analysis["avg_bandwidth"] > 50.0:
                compatibility.add_recommendation("高带宽互联，可考虑模型并行训练")


def main():
    """主函数，用于测试GPU检测功能"""
    logging.basicConfig(level=logging.INFO)
    
    detector = GPUDetector()
    print(detector.generate_system_report())
    
    # 测试拓扑检测
    print("\n=== GPU拓扑检测 ===")
    topology = detector.detect_gpu_topology()
    print(f"GPU数量: {topology.num_gpus}")
    print(f"拓扑类型: {topology.topology_type}")
    
    if topology.interconnects:
        print("GPU互联:")
        for ic in topology.interconnects:
            print(f"  GPU {ic.gpu_a} <-> GPU {ic.gpu_b}: "
                  f"{ic.interconnect_type.value} ({ic.bandwidth_gbps:.1f} GB/s)")
    
    # 测试硬件兼容性
    print("\n=== 硬件兼容性检查 ===")
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


if __name__ == "__main__":
    main()