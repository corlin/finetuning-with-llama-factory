#!/usr/bin/env python3
"""
Qwen3-4B-Thinking 模型服务自动化部署脚本
支持Docker和本地部署，包含完整的环境检查和配置
"""

import os
import sys
import subprocess
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.project_root = Path.cwd()
        self.deployment_log = []
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "deployment": {
                "type": "docker",  # docker, local
                "model_path": "./models/qwen3-4b-thinking",
                "quantization_format": "int8",
                "port": 8000,
                "workers": 1,
                "enable_monitoring": False,
                "enable_cache": False
            },
            "docker": {
                "image_name": "qwen-thinking-service",
                "container_name": "qwen-thinking-service",
                "build_args": {},
                "environment": {
                    "MODEL_PATH": "/app/models/qwen3-4b-thinking",
                    "QUANTIZATION_FORMAT": "int8",
                    "LOG_LEVEL": "INFO",
                    "MAX_WORKERS": "1"
                }
            },
            "local": {
                "python_executable": "python",
                "use_uv": True,
                "virtual_env": ".venv",
                "requirements_file": "pyproject.toml"
            },
            "monitoring": {
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "enable_alerts": True
            },
            "validation": {
                "run_tests": True,
                "test_endpoints": True,
                "performance_test": True
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # 合并配置
                self._merge_config(default_config, user_config)
                logger.info(f"已加载配置文件: {config_file}")
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def _merge_config(self, default: Dict, user: Dict):
        """递归合并配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _log_step(self, step: str, success: bool = True, details: str = ""):
        """记录部署步骤"""
        status = "✓" if success else "✗"
        message = f"{status} {step}"
        if details:
            message += f": {details}"
        
        logger.info(message)
        self.deployment_log.append({
            "step": step,
            "success": success,
            "details": details,
            "timestamp": time.time()
        })
    
    def check_prerequisites(self) -> bool:
        """检查部署前提条件"""
        logger.info("检查部署前提条件...")
        
        checks = []
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version >= (3, 12):
            checks.append(("Python版本", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python版本", False, f"需要Python 3.12+，当前: {python_version.major}.{python_version.minor}"))
        
        # 检查Docker（如果需要）
        if self.config["deployment"]["type"] == "docker":
            try:
                result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    checks.append(("Docker", True, result.stdout.strip()))
                else:
                    checks.append(("Docker", False, "Docker未安装或不可用"))
            except FileNotFoundError:
                checks.append(("Docker", False, "Docker未安装"))
            
            # 检查Docker Compose
            try:
                result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    checks.append(("Docker Compose", True, result.stdout.strip()))
                else:
                    checks.append(("Docker Compose", False, "Docker Compose未安装或不可用"))
            except FileNotFoundError:
                checks.append(("Docker Compose", False, "Docker Compose未安装"))
        
        # 检查uv（如果需要）
        if self.config["local"]["use_uv"]:
            try:
                result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    checks.append(("uv", True, result.stdout.strip()))
                else:
                    checks.append(("uv", False, "uv未安装或不可用"))
            except FileNotFoundError:
                checks.append(("uv", False, "uv未安装"))
        
        # 检查GPU（可选）
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("NVIDIA GPU", True, "GPU可用"))
            else:
                checks.append(("NVIDIA GPU", False, "GPU不可用，将使用CPU"))
        except FileNotFoundError:
            checks.append(("NVIDIA GPU", False, "nvidia-smi未找到"))
        
        # 检查模型文件
        model_path = Path(self.config["deployment"]["model_path"])
        if model_path.exists():
            checks.append(("模型文件", True, f"路径: {model_path}"))
        else:
            checks.append(("模型文件", False, f"模型路径不存在: {model_path}"))
        
        # 检查端口可用性
        port = self.config["deployment"]["port"]
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result != 0:
                    checks.append(("端口可用性", True, f"端口 {port} 可用"))
                else:
                    checks.append(("端口可用性", False, f"端口 {port} 已被占用"))
        except Exception as e:
            checks.append(("端口可用性", False, f"端口检查失败: {e}"))
        
        # 记录检查结果
        all_passed = True
        for check_name, success, details in checks:
            self._log_step(f"检查{check_name}", success, details)
            if not success:
                all_passed = False
        
        return all_passed
    
    def prepare_environment(self) -> bool:
        """准备部署环境"""
        logger.info("准备部署环境...")
        
        try:
            # 创建必要的目录
            directories = [
                "logs",
                "models",
                "configs",
                "monitoring"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                self._log_step(f"创建目录 {directory}", True)
            
            # 生成配置文件
            self._generate_config_files()
            
            # 设置环境变量
            self._setup_environment_variables()
            
            return True
            
        except Exception as e:
            self._log_step("准备环境", False, str(e))
            return False
    
    def _generate_config_files(self):
        """生成配置文件"""
        # 生成服务配置
        service_config = {
            "model": {
                "path": self.config["deployment"]["model_path"],
                "quantization": {
                    "format": self.config["deployment"]["quantization_format"],
                    "enable": True
                }
            },
            "server": {
                "host": "0.0.0.0",
                "port": self.config["deployment"]["port"],
                "workers": self.config["deployment"]["workers"]
            },
            "logging": {
                "level": "INFO",
                "file": "logs/service.log"
            }
        }
        
        config_path = self.project_root / "configs" / "service_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(service_config, f, default_flow_style=False, allow_unicode=True)
        
        self._log_step("生成服务配置文件", True, str(config_path))
        
        # 生成Docker环境变量文件
        if self.config["deployment"]["type"] == "docker":
            env_content = []
            for key, value in self.config["docker"]["environment"].items():
                env_content.append(f"{key}={value}")
            
            env_path = self.project_root / ".env"
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(env_content))
            
            self._log_step("生成Docker环境文件", True, str(env_path))
    
    def _setup_environment_variables(self):
        """设置环境变量"""
        env_vars = {
            "PYTHONPATH": str(self.project_root),
            "MODEL_PATH": self.config["deployment"]["model_path"],
            "QUANTIZATION_FORMAT": self.config["deployment"]["quantization_format"],
            "LOG_LEVEL": "INFO"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        self._log_step("设置环境变量", True)
    
    def deploy_docker(self) -> bool:
        """Docker部署"""
        logger.info("开始Docker部署...")
        
        try:
            # 构建Docker镜像
            build_cmd = [
                "docker", "build",
                "-t", self.config["docker"]["image_name"],
                "."
            ]
            
            # 添加构建参数
            for key, value in self.config["docker"]["build_args"].items():
                build_cmd.extend(["--build-arg", f"{key}={value}"])
            
            logger.info("构建Docker镜像...")
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log_step("构建Docker镜像", True)
            else:
                self._log_step("构建Docker镜像", False, result.stderr)
                return False
            
            # 启动服务
            compose_cmd = ["docker-compose", "up", "-d"]
            
            # 添加profile
            if self.config["deployment"]["enable_monitoring"]:
                compose_cmd.extend(["--profile", "monitoring"])
            
            if self.config["deployment"]["enable_cache"]:
                compose_cmd.extend(["--profile", "cache"])
            
            compose_cmd.append("qwen-model-service")
            
            logger.info("启动Docker服务...")
            result = subprocess.run(compose_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log_step("启动Docker服务", True)
                return True
            else:
                self._log_step("启动Docker服务", False, result.stderr)
                return False
                
        except Exception as e:
            self._log_step("Docker部署", False, str(e))
            return False
    
    def deploy_local(self) -> bool:
        """本地部署"""
        logger.info("开始本地部署...")
        
        try:
            # 安装依赖
            if self.config["local"]["use_uv"]:
                install_cmd = ["uv", "sync"]
            else:
                install_cmd = ["pip", "install", "-e", "."]
            
            logger.info("安装依赖...")
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log_step("安装依赖", True)
            else:
                self._log_step("安装依赖", False, result.stderr)
                return False
            
            # 启动服务
            if self.config["local"]["use_uv"]:
                start_cmd = [
                    "uv", "run", "python", "-m", "uvicorn",
                    "src.model_service:app",
                    "--host", "0.0.0.0",
                    "--port", str(self.config["deployment"]["port"]),
                    "--workers", str(self.config["deployment"]["workers"])
                ]
            else:
                start_cmd = [
                    self.config["local"]["python_executable"], "-m", "uvicorn",
                    "src.model_service:app",
                    "--host", "0.0.0.0",
                    "--port", str(self.config["deployment"]["port"]),
                    "--workers", str(self.config["deployment"]["workers"])
                ]
            
            logger.info("启动本地服务...")
            # 在后台启动服务
            process = subprocess.Popen(start_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待服务启动
            time.sleep(10)
            
            if process.poll() is None:
                self._log_step("启动本地服务", True, f"PID: {process.pid}")
                
                # 保存进程ID
                with open("service.pid", "w") as f:
                    f.write(str(process.pid))
                
                return True
            else:
                stdout, stderr = process.communicate()
                self._log_step("启动本地服务", False, stderr.decode())
                return False
                
        except Exception as e:
            self._log_step("本地部署", False, str(e))
            return False
    
    def validate_deployment(self) -> bool:
        """验证部署"""
        if not self.config["validation"]["run_tests"]:
            return True
        
        logger.info("验证部署...")
        
        try:
            # 等待服务启动
            time.sleep(5)
            
            # 运行验证脚本
            if self.config["local"]["use_uv"]:
                validate_cmd = ["uv", "run", "python", "scripts/validate_service.py"]
            else:
                validate_cmd = ["python", "scripts/validate_service.py"]
            
            result = subprocess.run(validate_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log_step("服务验证", True)
                return True
            else:
                self._log_step("服务验证", False, result.stderr)
                return False
                
        except Exception as e:
            self._log_step("部署验证", False, str(e))
            return False
    
    def generate_deployment_report(self) -> str:
        """生成部署报告"""
        total_steps = len(self.deployment_log)
        successful_steps = sum(1 for step in self.deployment_log if step["success"])
        
        report = f"""
# 部署报告

## 总体结果
- 部署类型: {self.config["deployment"]["type"]}
- 总步骤数: {total_steps}
- 成功步骤: {successful_steps}
- 失败步骤: {total_steps - successful_steps}
- 成功率: {successful_steps/total_steps*100:.1f}%

## 配置信息
- 模型路径: {self.config["deployment"]["model_path"]}
- 量化格式: {self.config["deployment"]["quantization_format"]}
- 服务端口: {self.config["deployment"]["port"]}
- 工作进程: {self.config["deployment"]["workers"]}

## 部署步骤详情
"""
        
        for step in self.deployment_log:
            status = "✓" if step["success"] else "✗"
            report += f"- {status} {step['step']}"
            if step["details"]:
                report += f": {step['details']}"
            report += "\n"
        
        # 添加访问信息
        if successful_steps == total_steps:
            report += f"""
## 服务访问信息
- 服务地址: http://localhost:{self.config["deployment"]["port"]}
- 健康检查: http://localhost:{self.config["deployment"]["port"]}/health
- API文档: http://localhost:{self.config["deployment"]["port"]}/docs
- 统计信息: http://localhost:{self.config["deployment"]["port"]}/stats
"""
            
            if self.config["deployment"]["enable_monitoring"]:
                report += f"""
## 监控访问信息
- Prometheus: http://localhost:{self.config["monitoring"]["prometheus_port"]}
- Grafana: http://localhost:{self.config["monitoring"]["grafana_port"]} (admin/admin)
"""
        
        # 保存报告
        timestamp = int(time.time())
        report_file = f"deployment_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"部署报告已保存: {report_file}")
        return report_file
    
    def deploy(self) -> bool:
        """执行完整部署流程"""
        logger.info("开始部署 Qwen3-4B-Thinking 模型服务...")
        
        # 1. 检查前提条件
        if not self.check_prerequisites():
            logger.error("前提条件检查失败，部署终止")
            return False
        
        # 2. 准备环境
        if not self.prepare_environment():
            logger.error("环境准备失败，部署终止")
            return False
        
        # 3. 执行部署
        deployment_type = self.config["deployment"]["type"]
        if deployment_type == "docker":
            success = self.deploy_docker()
        elif deployment_type == "local":
            success = self.deploy_local()
        else:
            logger.error(f"不支持的部署类型: {deployment_type}")
            return False
        
        if not success:
            logger.error("部署失败")
            return False
        
        # 4. 验证部署
        if not self.validate_deployment():
            logger.warning("部署验证失败，但服务可能仍在运行")
        
        # 5. 生成报告
        report_file = self.generate_deployment_report()
        
        logger.info("部署完成！")
        logger.info(f"服务地址: http://localhost:{self.config['deployment']['port']}")
        logger.info(f"部署报告: {report_file}")
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking 模型服务部署工具")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--type", choices=["docker", "local"], help="部署类型")
    parser.add_argument("--port", type=int, help="服务端口")
    parser.add_argument("--model-path", help="模型路径")
    parser.add_argument("--quantization", choices=["int8", "int4", "gptq"], help="量化格式")
    parser.add_argument("--monitoring", action="store_true", help="启用监控")
    parser.add_argument("--cache", action="store_true", help="启用缓存")
    parser.add_argument("--workers", type=int, help="工作进程数")
    
    args = parser.parse_args()
    
    # 创建部署管理器
    manager = DeploymentManager(args.config)
    
    # 覆盖命令行参数
    if args.type:
        manager.config["deployment"]["type"] = args.type
    if args.port:
        manager.config["deployment"]["port"] = args.port
    if args.model_path:
        manager.config["deployment"]["model_path"] = args.model_path
    if args.quantization:
        manager.config["deployment"]["quantization_format"] = args.quantization
    if args.monitoring:
        manager.config["deployment"]["enable_monitoring"] = True
    if args.cache:
        manager.config["deployment"]["enable_cache"] = True
    if args.workers:
        manager.config["deployment"]["workers"] = args.workers
    
    # 执行部署
    try:
        success = manager.deploy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("用户中断部署")
        sys.exit(1)
    except Exception as e:
        logger.error(f"部署异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()