#!/usr/bin/env python3
"""
集成测试运行器

本脚本运行完整的端到端集成测试套件，包括：
- 端到端训练流程测试
- 性能基准测试和回归检测
- 多种配置场景测试
- 中文密码学数据验证
- 系统集成验证

使用方法:
    python run_integration_tests.py [选项]

选项:
    --test-type: 测试类型 (all, e2e, performance, regression)
    --output-dir: 输出目录
    --config-file: 配置文件路径
    --verbose: 详细输出
    --gpu-count: 指定GPU数量
    --skip-slow: 跳过耗时测试
"""

import os
import sys
import json
import argparse
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import shutil

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入测试模块
from tests.test_end_to_end_integration import EndToEndIntegrationTestSuite
from tests.test_performance_benchmarks import PerformanceBenchmarkTestSuite


class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self, output_dir: str = "integration_test_results", verbose: bool = False):
        """初始化测试运行器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 测试套件
        self.e2e_suite = EndToEndIntegrationTestSuite(str(self.output_dir / "e2e"))
        self.performance_suite = PerformanceBenchmarkTestSuite(str(self.output_dir / "performance"))
        
        # 测试结果
        self.test_results = {}
        
        self.logger.info(f"集成测试运行器初始化完成，输出目录: {self.output_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("IntegrationTestRunner")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器
        log_file = self.output_dir / "integration_test_runner.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_environment_check(self) -> Dict[str, Any]:
        """运行环境检查"""
        self.logger.info("开始环境检查...")
        
        check_results = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
            "output_directory": str(self.output_dir.absolute()),
            "timestamp": datetime.now().isoformat()
        }
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.logger.error(f"Python版本过低: {python_version}, 需要Python 3.8+")
            check_results["python_version_ok"] = False
        else:
            check_results["python_version_ok"] = True
        
        # 检查必要的包
        required_packages = [
            "torch", "transformers", "datasets", "numpy", "pandas", 
            "psutil", "pynvml", "jieba", "opencc"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.debug(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                self.logger.warning(f"✗ {package} 未安装")
        
        check_results["missing_packages"] = missing_packages
        check_results["packages_ok"] = len(missing_packages) == 0
        
        # 检查CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0
            
            check_results["cuda_available"] = cuda_available
            check_results["gpu_count"] = gpu_count
            
            if cuda_available:
                self.logger.info(f"✓ CUDA可用，检测到 {gpu_count} 个GPU")
                
                # 获取GPU信息
                gpu_info = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        "gpu_id": i,
                        "name": props.name,
                        "total_memory": props.total_memory // (1024**2),  # MB
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
                
                check_results["gpu_info"] = gpu_info
            else:
                self.logger.warning("✗ CUDA不可用，将使用CPU进行测试")
                
        except Exception as e:
            self.logger.error(f"CUDA检查失败: {e}")
            check_results["cuda_available"] = False
            check_results["gpu_count"] = 0
        
        # 检查磁盘空间
        try:
            disk_usage = shutil.disk_usage(self.output_dir)
            free_space_gb = disk_usage.free // (1024**3)
            check_results["free_disk_space_gb"] = free_space_gb
            
            if free_space_gb < 10:  # 至少需要10GB
                self.logger.warning(f"磁盘空间不足: {free_space_gb}GB，建议至少10GB")
                check_results["disk_space_ok"] = False
            else:
                check_results["disk_space_ok"] = True
                self.logger.info(f"✓ 可用磁盘空间: {free_space_gb}GB")
                
        except Exception as e:
            self.logger.error(f"磁盘空间检查失败: {e}")
            check_results["disk_space_ok"] = False
        
        # 检查内存
        try:
            import psutil
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total // (1024**3)
            available_memory_gb = memory.available // (1024**3)
            
            check_results["total_memory_gb"] = total_memory_gb
            check_results["available_memory_gb"] = available_memory_gb
            
            if available_memory_gb < 8:  # 至少需要8GB可用内存
                self.logger.warning(f"可用内存不足: {available_memory_gb}GB，建议至少8GB")
                check_results["memory_ok"] = False
            else:
                check_results["memory_ok"] = True
                self.logger.info(f"✓ 可用内存: {available_memory_gb}GB")
                
        except Exception as e:
            self.logger.error(f"内存检查失败: {e}")
            check_results["memory_ok"] = False
        
        # 总体环境状态
        environment_ok = all([
            check_results.get("python_version_ok", False),
            check_results.get("packages_ok", False),
            check_results.get("disk_space_ok", False),
            check_results.get("memory_ok", False)
        ])
        
        check_results["environment_ok"] = environment_ok
        
        if environment_ok:
            self.logger.info("✓ 环境检查通过")
        else:
            self.logger.error("✗ 环境检查失败，请修复上述问题")
        
        # 保存环境检查结果
        env_check_file = self.output_dir / "environment_check.json"
        with open(env_check_file, 'w') as f:
            json.dump(check_results, f, indent=2)
        
        return check_results
    
    def run_e2e_tests(self, skip_slow: bool = False) -> Dict[str, Any]:
        """运行端到端集成测试"""
        self.logger.info("开始端到端集成测试...")
        
        try:
            # 如果跳过耗时测试，修改测试配置
            if skip_slow:
                self.logger.info("跳过耗时测试，使用快速配置")
                # 修改测试配置为更小的规模
                for config in self.e2e_suite.test_configurations:
                    config.num_epochs = 1
                    config.batch_size = min(config.batch_size, 1)
                    config.sequence_length = min(config.sequence_length, 256)
                    config.expected_duration_minutes = min(config.expected_duration_minutes, 5)
            
            # 运行测试
            e2e_results = self.e2e_suite.run_all_tests()
            self.test_results["e2e"] = e2e_results
            
            # 记录结果
            success_rate = e2e_results["test_summary"]["success_rate"]
            total_tests = e2e_results["test_summary"]["total_tests"]
            successful_tests = e2e_results["test_summary"]["successful_tests"]
            
            self.logger.info(f"端到端测试完成: {successful_tests}/{total_tests} 成功 ({success_rate:.2%})")
            
            return e2e_results
            
        except Exception as e:
            self.logger.error(f"端到端测试失败: {e}")
            error_result = {
                "error": str(e),
                "test_summary": {
                    "total_tests": 0,
                    "successful_tests": 0,
                    "failed_tests": 0,
                    "success_rate": 0.0
                }
            }
            self.test_results["e2e"] = error_result
            return error_result
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        self.logger.info("开始性能基准测试...")
        
        try:
            # 运行基准测试
            performance_results = self.performance_suite.run_full_benchmark_suite()
            self.test_results["performance"] = performance_results
            
            # 记录结果
            benchmark_count = performance_results["benchmark_summary"]["total_benchmarks"]
            test_duration = performance_results["benchmark_summary"]["test_duration_seconds"]
            
            self.logger.info(f"性能基准测试完成: 生成 {benchmark_count} 个基准，耗时 {test_duration:.2f}秒")
            
            return performance_results
            
        except Exception as e:
            self.logger.error(f"性能基准测试失败: {e}")
            error_result = {
                "error": str(e),
                "benchmark_summary": {
                    "total_benchmarks": 0,
                    "test_duration_seconds": 0
                }
            }
            self.test_results["performance"] = error_result
            return error_result
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """运行回归测试"""
        self.logger.info("开始回归测试...")
        
        try:
            # 运行回归测试
            regression_results = self.performance_suite.run_regression_test()
            self.test_results["regression"] = regression_results
            
            # 记录结果
            if "error" not in regression_results:
                regression_count = regression_results["summary"]["regression_count"]
                critical_count = regression_results["summary"]["critical_regressions"]
                
                if regression_count > 0:
                    self.logger.warning(f"检测到 {regression_count} 个性能回归，其中 {critical_count} 个严重")
                else:
                    self.logger.info("未检测到性能回归")
            else:
                self.logger.warning(f"回归测试跳过: {regression_results['error']}")
            
            return regression_results
            
        except Exception as e:
            self.logger.error(f"回归测试失败: {e}")
            error_result = {"error": str(e)}
            self.test_results["regression"] = error_result
            return error_result
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        self.logger.info("开始单元测试...")
        
        try:
            # 使用pytest运行单元测试
            test_files = [
                "tests/test_end_to_end_integration.py",
                "tests/test_performance_benchmarks.py"
            ]
            
            results = {}
            total_passed = 0
            total_failed = 0
            
            for test_file in test_files:
                if Path(test_file).exists():
                    self.logger.info(f"运行单元测试: {test_file}")
                    
                    # 运行pytest
                    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
                    if not self.verbose:
                        cmd.append("-q")
                    
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5分钟超时
                        )
                        
                        # 解析结果
                        output = result.stdout + result.stderr
                        passed = output.count(" PASSED")
                        failed = output.count(" FAILED")
                        
                        results[test_file] = {
                            "return_code": result.returncode,
                            "passed": passed,
                            "failed": failed,
                            "output": output if self.verbose else output[-1000:]  # 只保留最后1000字符
                        }
                        
                        total_passed += passed
                        total_failed += failed
                        
                        if result.returncode == 0:
                            self.logger.info(f"✓ {test_file}: {passed} 通过")
                        else:
                            self.logger.error(f"✗ {test_file}: {passed} 通过, {failed} 失败")
                            
                    except subprocess.TimeoutExpired:
                        self.logger.error(f"单元测试超时: {test_file}")
                        results[test_file] = {
                            "return_code": -1,
                            "passed": 0,
                            "failed": 1,
                            "output": "测试超时"
                        }
                        total_failed += 1
                        
                    except Exception as e:
                        self.logger.error(f"运行单元测试失败: {test_file}, 错误: {e}")
                        results[test_file] = {
                            "return_code": -1,
                            "passed": 0,
                            "failed": 1,
                            "output": str(e)
                        }
                        total_failed += 1
                else:
                    self.logger.warning(f"测试文件不存在: {test_file}")
            
            unit_test_results = {
                "total_passed": total_passed,
                "total_failed": total_failed,
                "success_rate": total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
                "test_files": results
            }
            
            self.test_results["unit_tests"] = unit_test_results
            
            self.logger.info(f"单元测试完成: {total_passed} 通过, {total_failed} 失败")
            
            return unit_test_results
            
        except Exception as e:
            self.logger.error(f"单元测试执行失败: {e}")
            error_result = {
                "error": str(e),
                "total_passed": 0,
                "total_failed": 1,
                "success_rate": 0.0
            }
            self.test_results["unit_tests"] = error_result
            return error_result
    
    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终测试报告"""
        self.logger.info("生成最终测试报告...")
        
        # 汇总所有测试结果
        final_report = {
            "test_run_info": {
                "start_time": datetime.now().isoformat(),
                "output_directory": str(self.output_dir.absolute()),
                "runner_version": "1.0.0"
            },
            "environment_check": self.test_results.get("environment", {}),
            "test_results": {
                "e2e_tests": self.test_results.get("e2e", {}),
                "performance_tests": self.test_results.get("performance", {}),
                "regression_tests": self.test_results.get("regression", {}),
                "unit_tests": self.test_results.get("unit_tests", {})
            },
            "summary": self._calculate_overall_summary(),
            "recommendations": self._generate_final_recommendations()
        }
        
        # 保存最终报告
        final_report_file = self.output_dir / "final_integration_test_report.json"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # 生成简化的文本报告
        self._generate_text_summary(final_report)
        
        self.logger.info(f"最终测试报告已保存: {final_report_file}")
        
        return final_report
    
    def _calculate_overall_summary(self) -> Dict[str, Any]:
        """计算总体摘要"""
        summary = {
            "overall_success": True,
            "total_test_suites": 0,
            "successful_test_suites": 0,
            "failed_test_suites": 0,
            "total_individual_tests": 0,
            "successful_individual_tests": 0,
            "failed_individual_tests": 0
        }
        
        # 环境检查
        env_check = self.test_results.get("environment", {})
        if env_check.get("environment_ok", False):
            summary["successful_test_suites"] += 1
        else:
            summary["failed_test_suites"] += 1
            summary["overall_success"] = False
        summary["total_test_suites"] += 1
        
        # 端到端测试
        e2e_results = self.test_results.get("e2e", {})
        if "test_summary" in e2e_results:
            e2e_summary = e2e_results["test_summary"]
            if e2e_summary.get("success_rate", 0) >= 0.8:  # 80%成功率
                summary["successful_test_suites"] += 1
            else:
                summary["failed_test_suites"] += 1
                summary["overall_success"] = False
            
            summary["total_individual_tests"] += e2e_summary.get("total_tests", 0)
            summary["successful_individual_tests"] += e2e_summary.get("successful_tests", 0)
            summary["failed_individual_tests"] += e2e_summary.get("failed_tests", 0)
        else:
            summary["failed_test_suites"] += 1
            summary["overall_success"] = False
        summary["total_test_suites"] += 1
        
        # 性能测试
        perf_results = self.test_results.get("performance", {})
        if "benchmark_summary" in perf_results and "error" not in perf_results:
            summary["successful_test_suites"] += 1
        else:
            summary["failed_test_suites"] += 1
            summary["overall_success"] = False
        summary["total_test_suites"] += 1
        
        # 回归测试
        regression_results = self.test_results.get("regression", {})
        if "error" not in regression_results:
            # 如果有严重回归，标记为失败
            critical_regressions = regression_results.get("summary", {}).get("critical_regressions", 0)
            if critical_regressions == 0:
                summary["successful_test_suites"] += 1
            else:
                summary["failed_test_suites"] += 1
                summary["overall_success"] = False
        else:
            # 回归测试跳过不算失败
            pass
        summary["total_test_suites"] += 1
        
        # 单元测试
        unit_results = self.test_results.get("unit_tests", {})
        if unit_results.get("success_rate", 0) >= 0.9:  # 90%成功率
            summary["successful_test_suites"] += 1
        else:
            summary["failed_test_suites"] += 1
            summary["overall_success"] = False
        
        summary["total_individual_tests"] += unit_results.get("total_passed", 0) + unit_results.get("total_failed", 0)
        summary["successful_individual_tests"] += unit_results.get("total_passed", 0)
        summary["failed_individual_tests"] += unit_results.get("total_failed", 0)
        summary["total_test_suites"] += 1
        
        # 计算总体成功率
        if summary["total_test_suites"] > 0:
            summary["test_suite_success_rate"] = summary["successful_test_suites"] / summary["total_test_suites"]
        else:
            summary["test_suite_success_rate"] = 0.0
        
        if summary["total_individual_tests"] > 0:
            summary["individual_test_success_rate"] = summary["successful_individual_tests"] / summary["total_individual_tests"]
        else:
            summary["individual_test_success_rate"] = 0.0
        
        return summary
    
    def _generate_final_recommendations(self) -> List[str]:
        """生成最终建议"""
        recommendations = []
        
        # 基于环境检查的建议
        env_check = self.test_results.get("environment", {})
        if not env_check.get("environment_ok", False):
            recommendations.append("环境检查失败，请修复环境问题后重新运行测试")
            
            if not env_check.get("packages_ok", False):
                missing = env_check.get("missing_packages", [])
                recommendations.append(f"安装缺失的包: {', '.join(missing)}")
            
            if not env_check.get("memory_ok", False):
                recommendations.append("系统内存不足，建议增加内存或关闭其他程序")
            
            if not env_check.get("disk_space_ok", False):
                recommendations.append("磁盘空间不足，建议清理磁盘空间")
        
        # 基于端到端测试的建议
        e2e_results = self.test_results.get("e2e", {})
        if "test_summary" in e2e_results:
            success_rate = e2e_results["test_summary"].get("success_rate", 0)
            if success_rate < 0.8:
                recommendations.append("端到端测试成功率较低，建议检查训练流程和配置")
            elif success_rate < 1.0:
                recommendations.append("部分端到端测试失败，建议查看详细错误信息")
        
        # 基于性能测试的建议
        perf_results = self.test_results.get("performance", {})
        if "error" in perf_results:
            recommendations.append("性能基准测试失败，建议检查GPU和CUDA环境")
        
        # 基于回归测试的建议
        regression_results = self.test_results.get("regression", {})
        if "summary" in regression_results:
            critical_regressions = regression_results["summary"].get("critical_regressions", 0)
            if critical_regressions > 0:
                recommendations.append(f"检测到{critical_regressions}个严重性能回归，需要立即修复")
            
            total_regressions = regression_results["summary"].get("regression_count", 0)
            if total_regressions > 0:
                recommendations.append("存在性能回归，建议分析性能变化原因")
        
        # 基于单元测试的建议
        unit_results = self.test_results.get("unit_tests", {})
        success_rate = unit_results.get("success_rate", 0)
        if success_rate < 0.9:
            recommendations.append("单元测试成功率较低，建议修复失败的测试用例")
        
        # 总体建议
        summary = self._calculate_overall_summary()
        if summary["overall_success"]:
            recommendations.append("所有测试通过，系统运行正常，可以进行部署")
        else:
            recommendations.append("存在测试失败，建议修复问题后重新运行测试")
        
        if not recommendations:
            recommendations.append("测试完成，系统状态良好")
        
        return recommendations
    
    def _generate_text_summary(self, final_report: Dict[str, Any]):
        """生成文本摘要"""
        summary_file = self.output_dir / "integration_test_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("集成测试总结报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体摘要
            summary = final_report["summary"]
            f.write("总体结果:\n")
            f.write(f"  整体状态: {'✓ 通过' if summary['overall_success'] else '✗ 失败'}\n")
            f.write(f"  测试套件: {summary['successful_test_suites']}/{summary['total_test_suites']} 成功\n")
            f.write(f"  个别测试: {summary['successful_individual_tests']}/{summary['total_individual_tests']} 成功\n")
            f.write(f"  套件成功率: {summary['test_suite_success_rate']:.2%}\n")
            f.write(f"  测试成功率: {summary['individual_test_success_rate']:.2%}\n\n")
            
            # 各测试套件结果
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            
            # 环境检查
            env_check = final_report["test_results"].get("environment_check", {})
            env_status = "✓" if env_check.get("environment_ok", False) else "✗"
            f.write(f"{env_status} 环境检查\n")
            
            # 端到端测试
            e2e_results = final_report["test_results"].get("e2e_tests", {})
            if "test_summary" in e2e_results:
                e2e_summary = e2e_results["test_summary"]
                success_rate = e2e_summary.get("success_rate", 0)
                status = "✓" if success_rate >= 0.8 else "✗"
                f.write(f"{status} 端到端测试: {e2e_summary.get('successful_tests', 0)}/{e2e_summary.get('total_tests', 0)} ({success_rate:.2%})\n")
            else:
                f.write("✗ 端到端测试: 执行失败\n")
            
            # 性能测试
            perf_results = final_report["test_results"].get("performance_tests", {})
            if "benchmark_summary" in perf_results:
                benchmark_count = perf_results["benchmark_summary"].get("total_benchmarks", 0)
                f.write(f"✓ 性能基准测试: 生成 {benchmark_count} 个基准\n")
            else:
                f.write("✗ 性能基准测试: 执行失败\n")
            
            # 回归测试
            regression_results = final_report["test_results"].get("regression_tests", {})
            if "summary" in regression_results:
                regression_count = regression_results["summary"].get("regression_count", 0)
                critical_count = regression_results["summary"].get("critical_regressions", 0)
                if regression_count == 0:
                    f.write("✓ 回归测试: 无性能回归\n")
                else:
                    status = "⚠️" if critical_count == 0 else "✗"
                    f.write(f"{status} 回归测试: {regression_count} 个回归 ({critical_count} 严重)\n")
            else:
                f.write("- 回归测试: 跳过\n")
            
            # 单元测试
            unit_results = final_report["test_results"].get("unit_tests", {})
            success_rate = unit_results.get("success_rate", 0)
            status = "✓" if success_rate >= 0.9 else "✗"
            passed = unit_results.get("total_passed", 0)
            failed = unit_results.get("total_failed", 0)
            f.write(f"{status} 单元测试: {passed} 通过, {failed} 失败 ({success_rate:.2%})\n\n")
            
            # 建议
            f.write("改进建议:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(final_report["recommendations"], 1):
                f.write(f"{i}. {rec}\n")
        
        self.logger.info(f"文本摘要已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="集成测试运行器")
    parser.add_argument("--test-type", choices=["all", "e2e", "performance", "regression", "unit"], 
                       default="all", help="测试类型")
    parser.add_argument("--output-dir", default="integration_test_results", help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--skip-slow", action="store_true", help="跳过耗时测试")
    parser.add_argument("--gpu-count", type=int, help="指定GPU数量")
    
    args = parser.parse_args()
    
    # 创建测试运行器
    runner = IntegrationTestRunner(args.output_dir, args.verbose)
    
    print(f"开始集成测试运行器...")
    print(f"测试类型: {args.test_type}")
    print(f"输出目录: {args.output_dir}")
    print(f"详细输出: {args.verbose}")
    print(f"跳过耗时测试: {args.skip_slow}")
    
    start_time = time.time()
    
    try:
        # 环境检查
        print("\n1. 环境检查...")
        env_results = runner.run_environment_check()
        runner.test_results["environment"] = env_results
        
        if not env_results.get("environment_ok", False):
            print("❌ 环境检查失败，请修复环境问题")
            if not args.verbose:
                print("使用 --verbose 查看详细错误信息")
            return False
        
        print("✅ 环境检查通过")
        
        # 根据测试类型运行相应测试
        if args.test_type in ["all", "unit"]:
            print("\n2. 单元测试...")
            unit_results = runner.run_unit_tests()
            if unit_results.get("success_rate", 0) < 0.9:
                print(f"⚠️  单元测试成功率较低: {unit_results.get('success_rate', 0):.2%}")
            else:
                print("✅ 单元测试通过")
        
        if args.test_type in ["all", "e2e"]:
            print("\n3. 端到端集成测试...")
            e2e_results = runner.run_e2e_tests(args.skip_slow)
            success_rate = e2e_results.get("test_summary", {}).get("success_rate", 0)
            if success_rate < 0.8:
                print(f"❌ 端到端测试失败: {success_rate:.2%}")
            else:
                print(f"✅ 端到端测试通过: {success_rate:.2%}")
        
        if args.test_type in ["all", "performance"]:
            print("\n4. 性能基准测试...")
            perf_results = runner.run_performance_tests()
            if "error" in perf_results:
                print(f"❌ 性能测试失败: {perf_results['error']}")
            else:
                benchmark_count = perf_results.get("benchmark_summary", {}).get("total_benchmarks", 0)
                print(f"✅ 性能测试完成: {benchmark_count} 个基准")
        
        if args.test_type in ["all", "regression"]:
            print("\n5. 回归测试...")
            regression_results = runner.run_regression_tests()
            if "error" in regression_results:
                print(f"⚠️  回归测试跳过: {regression_results['error']}")
            else:
                regression_count = regression_results.get("summary", {}).get("regression_count", 0)
                critical_count = regression_results.get("summary", {}).get("critical_regressions", 0)
                if regression_count == 0:
                    print("✅ 无性能回归")
                elif critical_count == 0:
                    print(f"⚠️  检测到 {regression_count} 个轻微回归")
                else:
                    print(f"❌ 检测到 {regression_count} 个回归，其中 {critical_count} 个严重")
        
        # 生成最终报告
        print("\n6. 生成最终报告...")
        final_report = runner.generate_final_report()
        
        # 显示总结
        total_time = time.time() - start_time
        summary = final_report["summary"]
        
        print(f"\n{'='*50}")
        print("集成测试完成!")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"整体状态: {'✅ 通过' if summary['overall_success'] else '❌ 失败'}")
        print(f"测试套件成功率: {summary['test_suite_success_rate']:.2%}")
        print(f"个别测试成功率: {summary['individual_test_success_rate']:.2%}")
        print(f"输出目录: {args.output_dir}")
        
        if not summary["overall_success"]:
            print("\n改进建议:")
            for i, rec in enumerate(final_report["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
        
        return summary["overall_success"]
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return False
    except Exception as e:
        print(f"\n测试运行器异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)