"""
环境设置和初始化模块
负责项目环境的完整设置和验证
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from gpu_utils import GPUDetector
from model_config import QwenModelManager, create_default_configs
from config_manager import ConfigManager


class EnvironmentSetup:
    """环境设置管理器"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 创建必要的目录结构
        self.create_directory_structure()
        
        # 初始化组件
        self.gpu_detector = GPUDetector()
        self.config_manager = ConfigManager()
    
    def setup_logging(self) -> None:
        """设置日志系统"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "setup.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def create_directory_structure(self) -> None:
        """创建项目目录结构"""
        directories = [
            "src",
            "data/raw",
            "data/processed",
            "data/train",
            "data/eval", 
            "data/test",
            "configs",
            "output",
            "logs",
            "cache",
            "models",
            "scripts",
            "tests",
            "docs",
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # 创建__init__.py文件（如果是Python包目录）
            if directory in ["src", "tests"]:
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
        
        self.logger.info("项目目录结构创建完成")
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        required_version = (3, 12)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            self.logger.info(f"Python版本检查通过: {sys.version}")
            return True
        else:
            self.logger.error(f"Python版本不符合要求。需要: {required_version}, 当前: {current_version}")
            return False
    
    def check_uv_installation(self) -> bool:
        """检查uv包管理器安装"""
        try:
            result = subprocess.run(["uv", "--version"], 
                                  capture_output=True, text=True, check=True)
            self.logger.info(f"uv版本: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("uv未安装或不在PATH中")
            return False
    
    def install_dependencies(self) -> bool:
        """安装项目依赖"""
        try:
            self.logger.info("开始使用uv安装项目依赖...")
            
            # 首先初始化uv项目（如果需要）
            if not (self.project_root / "uv.lock").exists():
                self.logger.info("初始化uv项目...")
                init_result = subprocess.run(
                    ["uv", "sync"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.logger.info("uv项目初始化完成")
            
            # 安装依赖包括开发依赖
            self.logger.info("安装生产依赖...")
            result = subprocess.run(
                ["uv", "sync"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info("安装开发依赖...")
            dev_result = subprocess.run(
                ["uv", "sync", "--extra", "dev"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info("uv依赖安装成功")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"uv依赖安装失败: {e}")
            self.logger.error(f"错误输出: {e.stderr}")
            self.logger.info("请确保已安装uv包管理器")
            return False
    
    def verify_pytorch_cuda(self) -> bool:
        """验证PyTorch CUDA安装"""
        try:
            import torch
            
            self.logger.info(f"PyTorch版本: {torch.__version__}")
            
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                self.logger.info(f"CUDA版本: {cuda_version}")
                self.logger.info(f"可用GPU数量: {device_count}")
                
                # 测试GPU
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    self.logger.info(f"GPU {i}: {gpu_name}")
                
                return True
            else:
                self.logger.warning("CUDA不可用，将使用CPU模式")
                return False
                
        except ImportError as e:
            self.logger.error(f"PyTorch导入失败: {e}")
            return False
    
    def verify_transformers_installation(self) -> bool:
        """验证Transformers库安装"""
        try:
            import transformers
            self.logger.info(f"Transformers版本: {transformers.__version__}")
            return True
        except ImportError as e:
            self.logger.error(f"Transformers导入失败: {e}")
            return False
    
    def test_qwen_model_loading(self) -> bool:
        """测试Qwen模型加载（仅tokenizer）"""
        try:
            self.logger.info("测试Qwen3-4B-Thinking模型配置...")
            
            # 创建模型配置
            model_config, lora_config, chinese_config = create_default_configs()
            
            # 创建模型管理器
            manager = QwenModelManager(model_config)
            
            # 仅测试tokenizer加载（避免下载大模型）
            try:
                tokenizer = manager.load_tokenizer(chinese_config)
                self.logger.info("Tokenizer加载测试成功")
                
                # 测试中文编码
                test_text = "这是一个包含<thinking>思考过程</thinking>的中文测试"
                tokens = tokenizer.encode(test_text)
                decoded = tokenizer.decode(tokens)
                
                if "中文" in decoded:
                    self.logger.info("中文处理测试通过")
                    return True
                else:
                    self.logger.warning("中文处理可能存在问题")
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Tokenizer加载失败（可能需要网络连接）: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"模型配置测试失败: {e}")
            return False
    
    def create_sample_config(self) -> None:
        """创建示例配置文件"""
        # 优化配置以适应硬件
        self.config_manager.optimize_for_hardware()
        
        # 保存配置
        self.config_manager.save_config()
        
        # 创建环境变量示例文件
        env_file = self.project_root / ".env.example"
        env_content = """# 环境变量配置示例
# 复制为.env文件并根据需要修改

# CUDA设备配置
CUDA_VISIBLE_DEVICES=0,1

# Hugging Face缓存目录
HF_CACHE_DIR=./cache/huggingface

# 分布式训练配置
WORLD_SIZE=1
LOCAL_RANK=-1
MASTER_ADDR=localhost
MASTER_PORT=29500

# 日志级别
LOG_LEVEL=INFO

# 模型配置
MODEL_NAME=Qwen/Qwen3-4B-Thinking-2507
TRUST_REMOTE_CODE=true
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        self.logger.info("示例配置文件创建完成")
    
    def create_startup_scripts(self) -> None:
        """创建启动脚本"""
        # 训练脚本
        train_script = self.project_root / "scripts" / "train.py"
        train_content = '''#!/usr/bin/env python3
"""
训练启动脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config_manager import ConfigManager
from gpu_utils import GPUDetector

def main():
    """主训练函数"""
    print("=== Qwen3-4B-Thinking 微调训练 ===")
    
    # 初始化配置
    config_manager = ConfigManager()
    
    # 检查GPU
    detector = GPUDetector()
    print(detector.generate_system_report())
    
    # 验证配置
    validation = config_manager.validate_configs()
    if not all(validation.values()):
        print("配置验证失败:", validation)
        return
    
    print("环境检查完成，准备开始训练...")
    # TODO: 实现训练逻辑

if __name__ == "__main__":
    main()
'''
        
        with open(train_script, 'w', encoding='utf-8') as f:
            f.write(train_content)
        
        # 环境检查脚本
        check_script = self.project_root / "scripts" / "check_environment.py"
        check_content = '''#!/usr/bin/env python3
"""
环境检查脚本
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from environment_setup import EnvironmentSetup

def main():
    """环境检查主函数"""
    setup = EnvironmentSetup()
    
    print("=== 环境检查报告 ===")
    
    # 生成GPU报告
    print(setup.gpu_detector.generate_system_report())
    
    # 验证配置
    validation = setup.config_manager.validate_configs()
    print("\\n配置验证结果:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}")

if __name__ == "__main__":
    main()
'''
        
        with open(check_script, 'w', encoding='utf-8') as f:
            f.write(check_content)
        
        # 设置脚本可执行权限（Unix系统）
        if os.name != 'nt':
            os.chmod(train_script, 0o755)
            os.chmod(check_script, 0o755)
        
        self.logger.info("启动脚本创建完成")
    
    def run_full_setup(self) -> Dict[str, bool]:
        """运行完整的环境设置"""
        setup_results = {}
        
        self.logger.info("开始完整环境设置...")
        
        # 1. 检查Python版本
        setup_results["python_version"] = self.check_python_version()
        
        # 2. 检查uv安装
        setup_results["uv_installed"] = self.check_uv_installation()
        
        # 3. 验证PyTorch CUDA
        setup_results["pytorch_cuda"] = self.verify_pytorch_cuda()
        
        # 4. 验证Transformers
        setup_results["transformers"] = self.verify_transformers_installation()
        
        # 5. GPU检测
        gpu_infos = self.gpu_detector.get_all_gpu_info()
        setup_results["gpu_detected"] = len(gpu_infos) > 0
        
        # 6. 验证Qwen模型配置
        setup_results["qwen_config"] = self.test_qwen_model_loading()
        
        # 7. 创建配置文件
        try:
            self.create_sample_config()
            setup_results["config_created"] = True
        except Exception as e:
            self.logger.error(f"配置文件创建失败: {e}")
            setup_results["config_created"] = False
        
        # 8. 创建启动脚本
        try:
            self.create_startup_scripts()
            setup_results["scripts_created"] = True
        except Exception as e:
            self.logger.error(f"启动脚本创建失败: {e}")
            setup_results["scripts_created"] = False
        
        # 生成设置报告
        self._generate_setup_report(setup_results)
        
        return setup_results
    
    def _generate_setup_report(self, results: Dict[str, bool]) -> None:
        """生成设置报告"""
        report_lines = []
        report_lines.append("=== 环境设置完成报告 ===")
        report_lines.append("")
        
        for item, success in results.items():
            status = "✓" if success else "✗"
            report_lines.append(f"{status} {item}")
        
        report_lines.append("")
        
        # 添加建议
        failed_items = [item for item, success in results.items() if not success]
        if failed_items:
            report_lines.append("需要注意的问题:")
            for item in failed_items:
                if item == "uv_installed":
                    report_lines.append("  - 请安装uv包管理器: https://docs.astral.sh/uv/")
                elif item == "pytorch_cuda":
                    report_lines.append("  - 请检查CUDA安装或使用CPU模式")
                elif item == "qwen_config":
                    report_lines.append("  - 模型配置可能需要网络连接，可稍后测试")
        else:
            report_lines.append("所有检查项目都通过了！")
            report_lines.append("")
            report_lines.append("下一步:")
            report_lines.append("  1. 运行 python scripts/check_environment.py 检查环境")
            report_lines.append("  2. 准备训练数据到 data/ 目录")
            report_lines.append("  3. 运行 python scripts/train.py 开始训练")
        
        report = "\n".join(report_lines)
        print(report)
        
        # 保存报告到文件
        report_file = self.project_root / "logs" / "setup_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"设置报告已保存到: {report_file}")


def main():
    """主函数"""
    setup = EnvironmentSetup()
    results = setup.run_full_setup()
    
    # 返回设置是否成功
    success = all(results.values())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()