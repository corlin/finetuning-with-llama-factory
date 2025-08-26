#!/usr/bin/env python3
"""
测试服务组件功能（无需运行服务器）
验证模型服务的核心组件和数据模型
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_models():
    """测试数据模型"""
    logger.info("测试数据模型...")
    
    try:
        from src.data_models import (
            ModelMetadata, TrainingExample, ThinkingExample, 
            CryptoTerm, ChineseMetrics, ThinkingStructure,
            DifficultyLevel, CryptoCategory
        )
        
        # 测试ModelMetadata
        metadata = ModelMetadata(
            model_name="Qwen3-4B-Thinking",
            model_type="CausalLM",
            model_path="./models/qwen3-4b-thinking",
            quantization_format="int8",
            parameters="4B",
            language="Chinese",
            domain="Cryptography"
        )
        
        logger.info(f"✓ ModelMetadata创建成功: {metadata.model_name}")
        
        # 测试CryptoTerm
        crypto_term = CryptoTerm(
            term="AES",
            definition="高级加密标准",
            category=CryptoCategory.SYMMETRIC_ENCRYPTION,
            complexity=5,
            aliases=["Advanced Encryption Standard"],
            related_terms=["DES", "3DES"]
        )
        
        logger.info(f"✓ CryptoTerm创建成功: {crypto_term.term}")
        
        # 测试ThinkingExample
        thinking_example = ThinkingExample(
            instruction="解释AES加密算法",
            thinking_process="<thinking>AES是一种对称加密算法...</thinking>",
            final_response="AES是高级加密标准...",
            crypto_terms=["AES", "对称加密"],
            difficulty_level=DifficultyLevel.INTERMEDIATE
        )
        
        logger.info(f"✓ ThinkingExample创建成功: {thinking_example.instruction[:20]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据模型测试失败: {e}")
        return False

def test_model_service_components():
    """测试模型服务组件"""
    logger.info("测试模型服务组件...")
    
    try:
        from src.model_service import (
            GenerateRequest, ThinkingRequest, BatchGenerateRequest,
            GenerateResponse, ThinkingResponse, HealthResponse,
            ModelService
        )
        
        # 测试请求模型
        gen_request = GenerateRequest(
            prompt="什么是RSA加密？",
            max_length=100,
            temperature=0.7
        )
        logger.info(f"✓ GenerateRequest创建成功: {gen_request.prompt[:20]}...")
        
        thinking_request = ThinkingRequest(
            question="RSA和AES的区别是什么？",
            thinking_depth=3
        )
        logger.info(f"✓ ThinkingRequest创建成功: {thinking_request.question[:20]}...")
        
        batch_request = BatchGenerateRequest(
            prompts=["什么是对称加密？", "什么是非对称加密？"],
            max_length=50
        )
        logger.info(f"✓ BatchGenerateRequest创建成功，批次大小: {len(batch_request.prompts)}")
        
        # 测试ModelService类
        service = ModelService()
        logger.info("✓ ModelService实例创建成功")
        
        # 测试健康检查
        health = service.get_health_status()
        logger.info(f"✓ 健康检查功能正常: {health.status}")
        
        # 测试服务统计
        stats = service.get_service_stats()
        logger.info(f"✓ 服务统计功能正常: {stats.total_requests} 总请求")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模型服务组件测试失败: {e}")
        return False

def test_gpu_detection():
    """测试GPU检测功能"""
    logger.info("测试GPU检测功能...")
    
    try:
        from src.gpu_utils import GPUDetector, GPUInfo, SystemRequirements
        
        detector = GPUDetector()
        logger.info("✓ GPUDetector实例创建成功")
        
        # 检测GPU
        gpu_info = detector.detect_gpus()
        logger.info(f"✓ GPU检测完成，发现 {len(gpu_info)} 个GPU")
        
        # 检查系统需求
        requirements = SystemRequirements()
        logger.info(f"✓ 系统需求检查: 最小GPU内存 {requirements.min_gpu_memory}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ GPU检测测试失败: {e}")
        return False

def test_memory_manager():
    """测试内存管理器"""
    logger.info("测试内存管理器...")
    
    try:
        from src.memory_manager import MemoryManager
        
        memory_manager = MemoryManager()
        logger.info("✓ MemoryManager实例创建成功")
        
        # 获取内存状态
        memory_status = memory_manager.get_memory_status()
        logger.info(f"✓ 内存状态获取成功: {memory_status.get('total_memory_gb', 0):.1f}GB 总内存")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 内存管理器测试失败: {e}")
        return False

def test_validation_script():
    """测试验证脚本组件"""
    logger.info("测试验证脚本组件...")
    
    try:
        from scripts.validate_service import ServiceValidator, ValidationResult
        
        # 创建验证器（不连接服务）
        validator = ServiceValidator("http://localhost:8000")
        logger.info("✓ ServiceValidator实例创建成功")
        
        # 测试验证结果数据结构
        result = ValidationResult(
            test_name="测试",
            success=True,
            message="测试成功",
            duration=1.0
        )
        logger.info(f"✓ ValidationResult创建成功: {result.test_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证脚本测试失败: {e}")
        return False

def test_deployment_components():
    """测试部署组件"""
    logger.info("测试部署组件...")
    
    try:
        from scripts.deploy_service import DeploymentManager
        
        # 创建部署管理器
        manager = DeploymentManager()
        logger.info("✓ DeploymentManager实例创建成功")
        
        # 测试配置加载
        config = manager.config
        logger.info(f"✓ 配置加载成功: {config['deployment']['type']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 部署组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始服务组件测试...")
    
    tests = [
        ("数据模型", test_data_models),
        ("模型服务组件", test_model_service_components),
        ("GPU检测", test_gpu_detection),
        ("内存管理器", test_memory_manager),
        ("验证脚本", test_validation_script),
        ("部署组件", test_deployment_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 生成测试报告
    logger.info(f"\n{'='*60}")
    logger.info("测试结果汇总")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        logger.info(f"{status} {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 所有组件测试通过！")
        return True
    else:
        logger.warning(f"⚠️  {total-passed} 个测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)