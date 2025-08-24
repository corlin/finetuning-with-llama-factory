"""
GPU硬件检测功能测试
测试GPU拓扑检测、互联带宽分析和硬件兼容性检查
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.gpu_utils import (
    GPUDetector, GPUInfo, GPUTopology, GPUInterconnect, 
    InterconnectType, HardwareCompatibility, SystemRequirements
)


class TestGPUHardwareDetection:
    """GPU硬件检测测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.detector = GPUDetector()
    
    def test_gpu_info_creation(self):
        """测试GPU信息数据结构创建"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            total_memory=24576,
            free_memory=20480,
            used_memory=4096,
            utilization=50.0,
            compute_capability=(8, 9),
            numa_node=0
        )
        
        assert gpu_info.gpu_id == 0
        assert gpu_info.name == "NVIDIA RTX 4090"
        assert gpu_info.total_memory == 24576
        assert gpu_info.compute_capability == (8, 9)
        assert gpu_info.numa_node == 0
    
    def test_gpu_topology_creation(self):
        """测试GPU拓扑结构创建"""
        topology = GPUTopology(num_gpus=2)
        
        # 添加GPU信息
        gpu0 = GPUInfo(0, "GPU0", 16384, 14000, 2384, 30.0)
        gpu1 = GPUInfo(1, "GPU1", 16384, 13000, 3384, 40.0)
        topology.gpu_info[0] = gpu0
        topology.gpu_info[1] = gpu1
        
        # 添加互联信息
        interconnect = GPUInterconnect(
            gpu_a=0, gpu_b=1, 
            interconnect_type=InterconnectType.NVLINK,
            bandwidth_gbps=50.0
        )
        topology.interconnects.append(interconnect)
        topology.bandwidth_matrix[(0, 1)] = 50.0
        
        assert topology.num_gpus == 2
        assert len(topology.gpu_info) == 2
        assert len(topology.interconnects) == 1
        assert topology.get_bandwidth(0, 1) == 50.0
        assert topology.get_bandwidth(1, 0) == 50.0  # 对称
    
    def test_numa_distance_calculation(self):
        """测试NUMA距离计算"""
        topology = GPUTopology(num_gpus=4)
        topology.numa_topology = {0: 0, 1: 0, 2: 1, 3: 1}
        
        # 同一NUMA节点
        assert topology.get_numa_distance(0, 1) == 0
        assert topology.get_numa_distance(2, 3) == 0
        
        # 不同NUMA节点
        assert topology.get_numa_distance(0, 2) == 1
        assert topology.get_numa_distance(1, 3) == 1
        
        # 未知NUMA节点
        assert topology.get_numa_distance(0, 4) == -1
    
    def test_optimal_gpu_pairs(self):
        """测试最优GPU配对算法"""
        topology = GPUTopology(num_gpus=4)
        topology.gpu_info = {
            0: GPUInfo(0, "GPU0", 16384, 14000, 2384, 30.0),
            1: GPUInfo(1, "GPU1", 16384, 13000, 3384, 40.0),
            2: GPUInfo(2, "GPU2", 16384, 12000, 4384, 50.0),
            3: GPUInfo(3, "GPU3", 16384, 11000, 5384, 60.0)
        }
        
        # 设置带宽矩阵（NVLink > PCIe）
        topology.bandwidth_matrix = {
            (0, 1): 50.0,  # NVLink
            (0, 2): 16.0,  # PCIe
            (0, 3): 16.0,  # PCIe
            (1, 2): 16.0,  # PCIe
            (1, 3): 50.0,  # NVLink
            (2, 3): 16.0   # PCIe
        }
        
        # 设置NUMA拓扑
        topology.numa_topology = {0: 0, 1: 0, 2: 1, 3: 1}
        
        pairs = topology.get_optimal_gpu_pairs()
        
        # 应该优先选择高带宽的配对
        assert len(pairs) == 6  # C(4,2) = 6种配对
        # 第一个应该是高带宽配对之一
        assert pairs[0] in [(0, 1), (1, 3)]
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_detect_cuda_availability(self, mock_device_count, mock_cuda_available):
        """测试CUDA可用性检测"""
        # 测试CUDA可用
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        assert self.detector.detect_cuda_availability() is True
        
        # 测试CUDA不可用
        mock_cuda_available.return_value = False
        
        assert self.detector.detect_cuda_availability() is False
    
    @patch('src.gpu_utils.PYNVML_AVAILABLE', True)
    @patch('src.gpu_utils.pynvml')
    @patch('platform.system')
    def test_get_gpu_numa_node(self, mock_platform, mock_pynvml):
        """测试GPU NUMA节点检测"""
        # 模拟Linux平台
        mock_platform.return_value = "Linux"
        
        # 模拟pynvml调用
        mock_handle = Mock()
        mock_pci_info = Mock()
        mock_pci_info.busId = b"0000:01:00.0"
        
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info
        
        # 模拟文件系统
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open_numa_file("0")):
            
            numa_node = self.detector._get_gpu_numa_node(0)
            assert numa_node == 0
        
        # 测试非Linux平台
        mock_platform.return_value = "Windows"
        numa_node = self.detector._get_gpu_numa_node(0)
        assert numa_node is None
    
    def test_interconnect_type_enum(self):
        """测试互联类型枚举"""
        assert InterconnectType.NVLINK.value == "NVLink"
        assert InterconnectType.PCIE.value == "PCIe"
        assert InterconnectType.NVSWITCH.value == "NVSwitch"
        assert InterconnectType.UNKNOWN.value == "Unknown"
    
    def test_hardware_compatibility_creation(self):
        """测试硬件兼容性结果创建"""
        compatibility = HardwareCompatibility(is_compatible=True, compatibility_score=85.0)
        
        compatibility.add_issue("测试问题")
        compatibility.add_warning("测试警告")
        compatibility.add_recommendation("测试建议")
        
        assert compatibility.is_compatible is False  # 添加问题后变为False
        assert len(compatibility.issues) == 1
        assert len(compatibility.warnings) == 1
        assert len(compatibility.recommendations) == 1
        assert compatibility.issues[0] == "测试问题"
    
    @patch('torch.cuda.is_available')
    def test_check_hardware_compatibility_no_cuda(self, mock_cuda_available):
        """测试无CUDA环境的兼容性检查"""
        mock_cuda_available.return_value = False
        
        compatibility = self.detector.check_hardware_compatibility()
        
        assert compatibility.is_compatible is False
        assert compatibility.compatibility_score <= 50.0
        assert any("CUDA不可用" in issue for issue in compatibility.issues)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_check_hardware_compatibility_no_gpu(self, mock_device_count, mock_cuda_available):
        """测试无GPU环境的兼容性检查"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 0
        
        compatibility = self.detector.check_hardware_compatibility()
        
        assert compatibility.is_compatible is False
        assert any("未检测到可用GPU" in issue for issue in compatibility.issues)
    
    def test_bandwidth_matrix_construction(self):
        """测试带宽矩阵构建"""
        interconnects = [
            GPUInterconnect(0, 1, InterconnectType.NVLINK, 50.0),
            GPUInterconnect(1, 2, InterconnectType.PCIE, 16.0),
            GPUInterconnect(0, 2, InterconnectType.PCIE, 16.0)
        ]
        
        bandwidth_matrix = self.detector._build_bandwidth_matrix(interconnects)
        
        assert bandwidth_matrix[(0, 1)] == 50.0
        assert bandwidth_matrix[(1, 2)] == 16.0
        assert bandwidth_matrix[(0, 2)] == 16.0
        assert len(bandwidth_matrix) == 3
    
    def test_topology_type_determination(self):
        """测试拓扑类型判断"""
        # 纯NVLink拓扑
        nvlink_interconnects = [
            GPUInterconnect(0, 1, InterconnectType.NVLINK, 50.0),
            GPUInterconnect(1, 2, InterconnectType.NVLINK, 50.0)
        ]
        assert self.detector._determine_topology_type(nvlink_interconnects) == "NVLink"
        
        # 纯PCIe拓扑
        pcie_interconnects = [
            GPUInterconnect(0, 1, InterconnectType.PCIE, 16.0),
            GPUInterconnect(1, 2, InterconnectType.PCIE, 16.0)
        ]
        assert self.detector._determine_topology_type(pcie_interconnects) == "PCIe"
        
        # 混合拓扑
        mixed_interconnects = [
            GPUInterconnect(0, 1, InterconnectType.NVLINK, 50.0),
            GPUInterconnect(1, 2, InterconnectType.PCIE, 16.0)
        ]
        assert self.detector._determine_topology_type(mixed_interconnects) == "Mixed"
        
        # 无互联
        assert self.detector._determine_topology_type([]) == "Single"
    
    def test_bandwidth_analysis(self):
        """测试带宽分析"""
        # 创建模拟拓扑
        topology = GPUTopology(num_gpus=3)
        topology.interconnects = [
            GPUInterconnect(0, 1, InterconnectType.NVLINK, 50.0),
            GPUInterconnect(1, 2, InterconnectType.PCIE, 16.0),
            GPUInterconnect(0, 2, InterconnectType.NVLINK, 25.0)
        ]
        
        # 缓存拓扑以供分析使用
        self.detector._topology_cache = topology
        
        analysis = self.detector.analyze_interconnect_bandwidth()
        
        assert analysis["total_bandwidth"] == 91.0  # 50 + 16 + 25
        assert analysis["avg_bandwidth"] == 91.0 / 3
        assert analysis["min_bandwidth"] == 16.0
        assert analysis["max_bandwidth"] == 50.0
        assert analysis["nvlink_bandwidth"] == 75.0  # 50 + 25
        assert analysis["pcie_bandwidth"] == 16.0
    
    def test_system_requirements(self):
        """测试系统需求配置"""
        requirements = SystemRequirements()
        
        assert requirements.min_gpu_memory == 8192
        assert requirements.recommended_gpu_memory == 16384
        assert requirements.min_compute_capability == (7, 0)
        assert requirements.recommended_compute_capability == (8, 0)
        assert requirements.min_nvlink_bandwidth == 25.0
        assert requirements.min_pcie_bandwidth == 16.0


def mock_open_numa_file(numa_node_value):
    """模拟NUMA节点文件读取"""
    from unittest.mock import mock_open
    return mock_open(read_data=numa_node_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])