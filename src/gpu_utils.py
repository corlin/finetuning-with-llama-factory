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
import os
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# 获取当前操作系统
CURRENT_OS = platform.system()
IS_WINDOWS = CURRENT_OS == "Windows"
IS_LINUX = CURRENT_OS == "Linux"
IS_MACOS = CURRENT_OS == "Darwin"

# 尝试导入pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")

# 尝试导入numa库（主要在Linux系统上可用）
NUMA_AVAILABLE = False
LIBNUMA_AVAILABLE = False

if IS_LINUX:
    try:
        import numa
        NUMA_AVAILABLE = True
        LIBNUMA_AVAILABLE = True
    except ImportError:
        pass
    except Exception as e:
        logging.debug(f"numa库导入失败: {e}")

# Windows特定的导入
WMI_AVAILABLE = False
if IS_WINDOWS:
    try:
        import wmi
        WMI_AVAILABLE = True
    except ImportError:
        logging.debug("WMI not available on Windows, some features will be limited")

# 记录NUMA库状态
if not NUMA_AVAILABLE and IS_LINUX:
    logging.warning("numa library not available on Linux, NUMA topology detection will be limited")
elif not NUMA_AVAILABLE and not IS_LINUX:
    logging.debug("numa library not available on non-Linux system, this is expected")


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
                    
                    # 获取更准确的内存信息
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.total_memory = mem_info.total // (1024**2)  # 转换为MB
                    gpu_info.used_memory = mem_info.used // (1024**2)   # 转换为MB
                    gpu_info.free_memory = mem_info.free // (1024**2)   # 转换为MB
                    
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
        """获取所有GPU信息，支持跨平台检测"""
        gpu_infos = []
        
        # 首先尝试PyTorch CUDA检测
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                gpu_info = self.get_gpu_info(i)
                if gpu_info:
                    gpu_infos.append(gpu_info)
        
        # 如果PyTorch没有检测到GPU，尝试其他方法
        if not gpu_infos:
            gpu_infos = self._fallback_gpu_detection()
        
        return gpu_infos
    
    def _fallback_gpu_detection(self) -> List[GPUInfo]:
        """备用GPU检测方法，支持多平台"""
        gpu_infos = []
        
        try:
            if IS_WINDOWS:
                gpu_infos = self._detect_gpus_windows()
            elif IS_LINUX:
                gpu_infos = self._detect_gpus_linux()
            elif IS_MACOS:
                gpu_infos = self._detect_gpus_macos()
        except Exception as e:
            self.logger.debug(f"备用GPU检测失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_windows(self) -> List[GPUInfo]:
        """Windows平台GPU检测"""
        gpu_infos = []
        
        try:
            # 方法1: 使用WMI
            if WMI_AVAILABLE:
                gpu_infos.extend(self._detect_gpus_wmi())
            
            # 方法2: 使用nvidia-smi命令
            if not gpu_infos:
                gpu_infos.extend(self._detect_gpus_nvidia_smi())
            
            # 方法3: 使用WMIC命令
            if not gpu_infos:
                gpu_infos.extend(self._detect_gpus_wmic())
                
        except Exception as e:
            self.logger.debug(f"Windows GPU检测失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_linux(self) -> List[GPUInfo]:
        """Linux平台GPU检测"""
        gpu_infos = []
        
        try:
            # 方法1: 使用nvidia-smi命令
            gpu_infos.extend(self._detect_gpus_nvidia_smi())
            
            # 方法2: 读取/proc/driver/nvidia/gpus/
            if not gpu_infos:
                gpu_infos.extend(self._detect_gpus_proc_nvidia())
            
            # 方法3: 使用lspci命令
            if not gpu_infos:
                gpu_infos.extend(self._detect_gpus_lspci())
                
        except Exception as e:
            self.logger.debug(f"Linux GPU检测失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_macos(self) -> List[GPUInfo]:
        """macOS平台GPU检测"""
        gpu_infos = []
        
        try:
            # 方法1: 使用system_profiler命令
            gpu_infos.extend(self._detect_gpus_system_profiler())
            
            # 方法2: 使用nvidia-smi（如果安装了NVIDIA GPU）
            if not gpu_infos:
                gpu_infos.extend(self._detect_gpus_nvidia_smi())
                
        except Exception as e:
            self.logger.debug(f"macOS GPU检测失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_wmi(self) -> List[GPUInfo]:
        """使用WMI检测Windows GPU"""
        gpu_infos = []
        
        try:
            if not WMI_AVAILABLE:
                return gpu_infos
            
            c = wmi.WMI()
            for i, gpu in enumerate(c.Win32_VideoController()):
                if gpu.Name and 'NVIDIA' in gpu.Name.upper():
                    # 尝试获取内存信息
                    memory_mb = 0
                    if gpu.AdapterRAM:
                        memory_mb = int(gpu.AdapterRAM) // (1024 * 1024)
                    
                    gpu_info = GPUInfo(
                        gpu_id=i,
                        name=gpu.Name,
                        total_memory=memory_mb,
                        free_memory=memory_mb,  # WMI无法直接获取空闲内存
                        used_memory=0,
                        utilization=0.0,
                        pci_bus_id=gpu.PNPDeviceID if gpu.PNPDeviceID else None
                    )
                    gpu_infos.append(gpu_info)
                    
        except Exception as e:
            self.logger.debug(f"WMI GPU检测失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_nvidia_smi(self) -> List[GPUInfo]:
        """使用nvidia-smi命令检测GPU"""
        gpu_infos = []
        
        try:
            # 尝试运行nvidia-smi命令
            cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw,pci.bus_id", "--format=csv,noheader,nounits"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            try:
                                gpu_info = GPUInfo(
                                    gpu_id=int(parts[0]),
                                    name=parts[1],
                                    total_memory=int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                                    free_memory=int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                                    used_memory=int(parts[4]) if parts[4] != '[Not Supported]' else 0,
                                    utilization=float(parts[5]) if parts[5] != '[Not Supported]' else 0.0,
                                    temperature=int(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else None,
                                    power_usage=int(float(parts[7])) if len(parts) > 7 and parts[7] != '[Not Supported]' else None,
                                    pci_bus_id=parts[8] if len(parts) > 8 else None
                                )
                                gpu_infos.append(gpu_info)
                            except (ValueError, IndexError) as e:
                                self.logger.debug(f"解析nvidia-smi输出失败: {e}")
                                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.debug(f"nvidia-smi命令执行失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_wmic(self) -> List[GPUInfo]:
        """使用WMIC命令检测Windows GPU"""
        gpu_infos = []
        
        try:
            cmd = ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,PNPDeviceID", "/format:csv"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                gpu_id = 0
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3 and parts[2]:  # 确保有名称
                            name = parts[2]
                            if 'NVIDIA' in name.upper():
                                memory_mb = 0
                                if parts[1] and parts[1] != '':
                                    try:
                                        memory_mb = int(parts[1]) // (1024 * 1024)
                                    except ValueError:
                                        pass
                                
                                gpu_info = GPUInfo(
                                    gpu_id=gpu_id,
                                    name=name,
                                    total_memory=memory_mb,
                                    free_memory=memory_mb,
                                    used_memory=0,
                                    utilization=0.0,
                                    pci_bus_id=parts[3] if len(parts) > 3 else None
                                )
                                gpu_infos.append(gpu_info)
                                gpu_id += 1
                                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.debug(f"WMIC命令执行失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_proc_nvidia(self) -> List[GPUInfo]:
        """通过/proc/driver/nvidia/gpus/检测Linux GPU"""
        gpu_infos = []
        
        try:
            nvidia_proc_path = Path("/proc/driver/nvidia/gpus")
            if nvidia_proc_path.exists():
                gpu_dirs = [d for d in nvidia_proc_path.iterdir() if d.is_dir()]
                
                for i, gpu_dir in enumerate(sorted(gpu_dirs)):
                    try:
                        # 读取GPU信息文件
                        info_file = gpu_dir / "information"
                        if info_file.exists():
                            with open(info_file, 'r') as f:
                                content = f.read()
                                
                            # 解析GPU名称
                            name_match = re.search(r'Model:\s+(.+)', content)
                            name = name_match.group(1).strip() if name_match else f"NVIDIA GPU {i}"
                            
                            # 解析PCI总线ID
                            pci_match = re.search(r'Bus Location:\s+(.+)', content)
                            pci_bus_id = pci_match.group(1).strip() if pci_match else None
                            
                            gpu_info = GPUInfo(
                                gpu_id=i,
                                name=name,
                                total_memory=0,  # 无法从proc获取内存信息
                                free_memory=0,
                                used_memory=0,
                                utilization=0.0,
                                pci_bus_id=pci_bus_id
                            )
                            gpu_infos.append(gpu_info)
                            
                    except Exception as e:
                        self.logger.debug(f"读取GPU {i}信息失败: {e}")
                        
        except Exception as e:
            self.logger.debug(f"proc nvidia检测失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_lspci(self) -> List[GPUInfo]:
        """使用lspci命令检测Linux GPU"""
        gpu_infos = []
        
        try:
            cmd = ["lspci", "-nn"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_id = 0
                
                for line in lines:
                    if 'VGA compatible controller' in line and 'NVIDIA' in line.upper():
                        # 解析PCI总线ID和GPU名称
                        parts = line.split(' ', 1)
                        pci_bus_id = parts[0] if parts else None
                        
                        # 提取GPU名称
                        name_match = re.search(r'NVIDIA[^[]*', line)
                        name = name_match.group(0).strip() if name_match else f"NVIDIA GPU {gpu_id}"
                        
                        gpu_info = GPUInfo(
                            gpu_id=gpu_id,
                            name=name,
                            total_memory=0,  # lspci无法获取内存信息
                            free_memory=0,
                            used_memory=0,
                            utilization=0.0,
                            pci_bus_id=pci_bus_id
                        )
                        gpu_infos.append(gpu_info)
                        gpu_id += 1
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.debug(f"lspci命令执行失败: {e}")
        
        return gpu_infos
    
    def _detect_gpus_system_profiler(self) -> List[GPUInfo]:
        """使用system_profiler命令检测macOS GPU"""
        gpu_infos = []
        
        try:
            cmd = ["system_profiler", "SPDisplaysDataType", "-json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays = data.get('SPDisplaysDataType', [])
                
                gpu_id = 0
                for display in displays:
                    name = display.get('sppci_model', 'Unknown GPU')
                    if 'NVIDIA' in name.upper() or 'GeForce' in name.upper() or 'Quadro' in name.upper():
                        # 尝试获取VRAM信息
                        vram_str = display.get('sppci_vram', '0 MB')
                        memory_mb = 0
                        if 'MB' in vram_str:
                            try:
                                memory_mb = int(re.search(r'(\d+)', vram_str).group(1))
                            except:
                                pass
                        elif 'GB' in vram_str:
                            try:
                                memory_gb = float(re.search(r'(\d+(?:\.\d+)?)', vram_str).group(1))
                                memory_mb = int(memory_gb * 1024)
                            except:
                                pass
                        
                        gpu_info = GPUInfo(
                            gpu_id=gpu_id,
                            name=name,
                            total_memory=memory_mb,
                            free_memory=memory_mb,
                            used_memory=0,
                            utilization=0.0,
                            pci_bus_id=display.get('sppci_bus', None)
                        )
                        gpu_infos.append(gpu_info)
                        gpu_id += 1
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.debug(f"system_profiler命令执行失败: {e}")
        
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
        """获取GPU所在的NUMA节点，支持多平台检测"""
        try:
            if IS_LINUX:
                return self._get_gpu_numa_node_linux(gpu_id)
            elif IS_WINDOWS:
                return self._get_gpu_numa_node_windows(gpu_id)
            else:
                return None
            
        except Exception as e:
            self.logger.debug(f"获取GPU NUMA节点失败: {e}")
            return None
    
    def _get_gpu_numa_node_linux(self, gpu_id: int) -> Optional[int]:
        """Linux平台获取GPU NUMA节点"""
        try:
            # 方法1: 使用numa库
            if NUMA_AVAILABLE:
                try:
                    current_node = numa.get_mempolicy()[1]
                    if current_node is not None and len(current_node) > 0:
                        return list(current_node)[0]
                except Exception as e:
                    self.logger.debug(f"numa库检测失败: {e}")
            
            # 方法2: 通过nvidia-ml-py获取PCI总线ID并查询sysfs
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
            
            # 方法3: 通过/proc/cpuinfo和GPU亲和性推断
            return self._infer_numa_from_cpu_affinity_linux(gpu_id)
            
        except Exception as e:
            self.logger.debug(f"Linux GPU NUMA节点检测失败: {e}")
            return None
    
    def _get_gpu_numa_node_windows(self, gpu_id: int) -> Optional[int]:
        """Windows平台获取GPU NUMA节点"""
        try:
            # 方法1: 使用WMI查询NUMA信息
            if WMI_AVAILABLE:
                try:
                    c = wmi.WMI()
                    # 查询NUMA节点信息
                    numa_nodes = c.Win32_NumaNode()
                    if numa_nodes:
                        # 简化处理：根据GPU ID分配NUMA节点
                        return gpu_id % len(numa_nodes)
                except Exception as e:
                    self.logger.debug(f"WMI NUMA检测失败: {e}")
            
            # 方法2: 使用PowerShell命令查询
            try:
                cmd = ["powershell", "-Command", "Get-WmiObject -Class Win32_NumaNode | Select-Object NodeId"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[2:]  # 跳过标题行
                    numa_nodes = []
                    for line in lines:
                        if line.strip().isdigit():
                            numa_nodes.append(int(line.strip()))
                    
                    if numa_nodes:
                        return gpu_id % len(numa_nodes)
                        
            except Exception as e:
                self.logger.debug(f"PowerShell NUMA检测失败: {e}")
            
            # 方法3: 通过CPU核心数推断
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count and cpu_count > 8:  # 假设超过8核心可能有多个NUMA节点
                return gpu_id % 2  # 简化为2个NUMA节点
            
            return 0  # 默认返回节点0
            
        except Exception as e:
            self.logger.debug(f"Windows GPU NUMA节点检测失败: {e}")
            return None
    
    def _infer_numa_from_cpu_affinity_linux(self, gpu_id: int) -> Optional[int]:
        """通过CPU亲和性推断NUMA节点（Linux）"""
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
        """获取系统NUMA拓扑信息，支持多平台"""
        numa_info = {
            "numa_available": NUMA_AVAILABLE or LIBNUMA_AVAILABLE or WMI_AVAILABLE,
            "numa_nodes": [],
            "memory_per_node": {},
            "cpu_per_node": {},
            "distances": {},
            "platform": CURRENT_OS
        }
        
        try:
            if IS_LINUX:
                numa_info.update(self._get_numa_topology_linux())
            elif IS_WINDOWS:
                numa_info.update(self._get_numa_topology_windows())
            elif IS_MACOS:
                numa_info.update(self._get_numa_topology_macos())
                
        except Exception as e:
            self.logger.debug(f"获取NUMA拓扑信息失败: {e}")
        
        return numa_info
    
    def _get_numa_topology_linux(self) -> Dict[str, any]:
        """获取Linux NUMA拓扑信息"""
        numa_info = {}
        
        try:
            if NUMA_AVAILABLE:
                # 使用numa库获取详细信息
                try:
                    numa_info["numa_nodes"] = list(range(numa.get_max_node() + 1))
                    for node in numa_info["numa_nodes"]:
                        numa_info["memory_per_node"][node] = numa.node_memsize(node)
                except Exception as e:
                    self.logger.debug(f"numa库获取拓扑信息失败: {e}")
            
            # 从/sys/devices/system/node/读取NUMA信息
            if not numa_info.get("numa_nodes"):
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
            self.logger.debug(f"Linux NUMA拓扑信息获取失败: {e}")
        
        return numa_info
    
    def _get_numa_topology_windows(self) -> Dict[str, any]:
        """获取Windows NUMA拓扑信息"""
        numa_info = {}
        
        try:
            # 方法1: 使用WMI
            if WMI_AVAILABLE:
                try:
                    c = wmi.WMI()
                    numa_nodes = c.Win32_NumaNode()
                    
                    if numa_nodes:
                        numa_info["numa_nodes"] = [node.NodeId for node in numa_nodes]
                        
                        # 获取每个NUMA节点的内存信息
                        for node in numa_nodes:
                            try:
                                # 查询该NUMA节点的内存
                                memory_query = f"SELECT * FROM Win32_PhysicalMemory WHERE PositionInRow = {node.NodeId}"
                                memory_modules = c.query(memory_query)
                                total_memory = sum(int(mem.Capacity) for mem in memory_modules if mem.Capacity)
                                numa_info["memory_per_node"][node.NodeId] = total_memory
                            except Exception:
                                numa_info["memory_per_node"][node.NodeId] = 0
                                
                except Exception as e:
                    self.logger.debug(f"WMI NUMA信息获取失败: {e}")
            
            # 方法2: 使用PowerShell命令
            if not numa_info.get("numa_nodes"):
                try:
                    cmd = ["powershell", "-Command", 
                          "Get-WmiObject -Class Win32_NumaNode | Select-Object NodeId, @{Name='Memory';Expression={(Get-WmiObject -Class Win32_PhysicalMemory | Where-Object {$_.PositionInRow -eq $_.NodeId} | Measure-Object -Property Capacity -Sum).Sum}}"]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')[2:]  # 跳过标题行
                        for line in lines:
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    try:
                                        node_id = int(parts[0])
                                        memory = int(parts[1]) if parts[1].isdigit() else 0
                                        numa_info.setdefault("numa_nodes", []).append(node_id)
                                        numa_info.setdefault("memory_per_node", {})[node_id] = memory
                                    except ValueError:
                                        pass
                                        
                except Exception as e:
                    self.logger.debug(f"PowerShell NUMA信息获取失败: {e}")
            
            # 方法3: 简化推断
            if not numa_info.get("numa_nodes"):
                cpu_count = psutil.cpu_count(logical=False)
                if cpu_count and cpu_count > 8:
                    # 假设超过8核心的系统可能有多个NUMA节点
                    numa_info["numa_nodes"] = [0, 1]
                    total_memory = psutil.virtual_memory().total
                    numa_info["memory_per_node"] = {0: total_memory // 2, 1: total_memory // 2}
                else:
                    numa_info["numa_nodes"] = [0]
                    numa_info["memory_per_node"] = {0: psutil.virtual_memory().total}
        
        except Exception as e:
            self.logger.debug(f"Windows NUMA拓扑信息获取失败: {e}")
        
        return numa_info
    
    def _get_numa_topology_macos(self) -> Dict[str, any]:
        """获取macOS NUMA拓扑信息"""
        numa_info = {}
        
        try:
            # macOS通常不支持NUMA，但我们可以提供基本信息
            numa_info["numa_nodes"] = [0]  # 单一节点
            numa_info["memory_per_node"] = {0: psutil.virtual_memory().total}
            
            # 尝试使用system_profiler获取更详细的硬件信息
            try:
                cmd = ["system_profiler", "SPHardwareDataType", "-json"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    hardware = data.get('SPHardwareDataType', [])
                    if hardware:
                        memory_str = hardware[0].get('physical_memory', '0 GB')
                        # 解析内存大小
                        memory_match = re.search(r'(\d+(?:\.\d+)?)\s*GB', memory_str)
                        if memory_match:
                            memory_gb = float(memory_match.group(1))
                            numa_info["memory_per_node"][0] = int(memory_gb * 1024 * 1024 * 1024)
                            
            except Exception as e:
                self.logger.debug(f"macOS硬件信息获取失败: {e}")
        
        except Exception as e:
            self.logger.debug(f"macOS NUMA拓扑信息获取失败: {e}")
        
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