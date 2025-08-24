"""
环境设置测试
"""

import unittest
import sys
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

class TestEnvironmentSetup(unittest.TestCase):
    """环境设置测试类"""
    
    def setUp(self):
        """测试设置"""
        from environment_setup import EnvironmentSetup
        self.setup = EnvironmentSetup(project_root)
    
    def test_python_version(self):
        """测试Python版本检查"""
        result = self.setup.check_python_version()
        self.assertIsInstance(result, bool)
    
    def test_directory_structure(self):
        """测试目录结构创建"""
        required_dirs = [
            "src", "data", "configs", "output", 
            "logs", "cache", "scripts", "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"目录不存在: {dir_name}")
    
    def test_gpu_detector(self):
        """测试GPU检测器"""
        from gpu_utils import GPUDetector
        
        detector = GPUDetector()
        
        # 测试CUDA检测
        cuda_available = detector.detect_cuda_availability()
        self.assertIsInstance(cuda_available, bool)
        
        # 测试GPU信息获取
        gpu_infos = detector.get_all_gpu_info()
        self.assertIsInstance(gpu_infos, list)
        
        # 测试系统报告生成
        report = detector.generate_system_report()
        self.assertIsInstance(report, str)
        self.assertIn("系统环境报告", report)
    
    def test_config_manager(self):
        """测试配置管理器"""
        from config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # 测试配置验证
        validation = config_manager.validate_configs()
        self.assertIsInstance(validation, dict)
        
        # 测试配置获取
        all_configs = config_manager.get_all_configs()
        self.assertIsInstance(all_configs, dict)
        self.assertIn("model", all_configs)
        self.assertIn("training", all_configs)
    
    def test_model_config(self):
        """测试模型配置"""
        from model_config import create_default_configs, QwenModelManager
        
        # 测试默认配置创建
        model_config, lora_config, chinese_config = create_default_configs()
        
        self.assertEqual(model_config.model_name, "Qwen/Qwen3-4B-Thinking-2507")
        self.assertEqual(lora_config.r, 16)
        self.assertEqual(chinese_config.thinking_start_token, "<thinking>")
        
        # 测试模型管理器创建
        manager = QwenModelManager(model_config)
        self.assertIsNotNone(manager)


class TestGPUUtils(unittest.TestCase):
    """GPU工具测试类"""
    
    def setUp(self):
        """测试设置"""
        from gpu_utils import GPUDetector, SystemRequirements
        self.detector = GPUDetector()
        self.requirements = SystemRequirements()
    
    def test_system_requirements(self):
        """测试系统需求"""
        self.assertGreater(self.requirements.min_gpu_memory, 0)
        self.assertGreater(self.requirements.recommended_gpu_memory, 0)
        self.assertEqual(self.requirements.min_python_version, (3, 12))
    
    def test_validation(self):
        """测试需求验证"""
        validation = self.detector.validate_qwen_requirements()
        
        required_keys = [
            "cuda_available", "sufficient_gpu_memory", 
            "sufficient_system_memory", "python_version_ok"
        ]
        
        for key in required_keys:
            self.assertIn(key, validation)
            self.assertIsInstance(validation[key], bool)
    
    def test_recommendations(self):
        """测试优化建议"""
        recommendations = self.detector.get_optimization_recommendations()
        self.assertIsInstance(recommendations, list)


if __name__ == "__main__":
    unittest.main()