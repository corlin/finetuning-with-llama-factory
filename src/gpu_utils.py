"""
GPU检测和验证功能模块
针对Qwen3-4B模型内存需求优化的GPU管理工具
"""

import torch
import psutil
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")


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


@dataclass
class SystemRequirements:
    """Qwen3-4B模型系统需求"""
    min_gpu_memory: int = 8192  # MB, 最小8GB
    recommended_gpu_memory: int = 16384  # MB, 推荐16GB
    min_system_memory: int = 16384  # MB, 最小16GB系统内存
    min_python_version: Tuple[int, int] = (3, 12)
    required_cuda_version: str = "12.9"


class GPUDetector:
    """GPU检测和验证类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = SystemRequirements()
        self._initialize_pynvml()
    
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
        """获取指定GPU信息"""
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
                utilization=0.0  # PyTorch不直接提供利用率
            )
            
            # 如果pynvml可用，获取更详细信息
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
                    
                    gpu_info.utilization = util.gpu
                    gpu_info.temperature = temp
                    gpu_info.power_usage = power
                except Exception as e:
                    self.logger.warning(f"获取GPU详细信息失败: {e}")
            
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
        
        # 优化建议
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            report_lines.append("")
            report_lines.append("优化建议:")
            for rec in recommendations:
                report_lines.append(f"  {rec}")
        
        return "\n".join(report_lines)


def main():
    """主函数，用于测试GPU检测功能"""
    logging.basicConfig(level=logging.INFO)
    
    detector = GPUDetector()
    print(detector.generate_system_report())


if __name__ == "__main__":
    main()