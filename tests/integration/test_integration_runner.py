"""
集成测试运行器

统一运行所有集成测试，提供测试报告和性能分析。
"""

import pytest
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import ExpertEvaluationConfig, EvaluationMode


class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self, output_dir: str = "integration_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        self.performance_metrics = {}
    
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """运行所有集成测试"""
        print("开始运行专家评估系统集成测试...")
        
        start_time = time.time()
        
        # 定义测试套件
        test_suites = [
            {
                "name": "evaluation_pipeline",
                "description": "完整评估流程测试",
                "module": "tests.integration.test_evaluation_pipeline",
                "critical": True
            },
            {
                "name": "multi_model_comparison", 
                "description": "多模型比较测试",
                "module": "tests.integration.test_multi_model_comparison",
                "critical": True
            },
            {
                "name": "large_scale_processing",
                "description": "大规模数据处理测试",
                "module": "tests.integration.test_large_scale_processing",
                "critical": False
            },
            {
                "name": "system_compatibility",
                "description": "系统兼容性测试",
                "module": "tests.integration.test_system_compatibility",
                "critical": True
            }
        ]
        
        # 运行每个测试套件
        for suite in test_suites:
            print(f"\n运行测试套件: {suite['name']} - {suite['description']}")
            
            suite_start_time = time.time()
            
            try:
                # 运行pytest
                result = pytest.main([
                    f"{suite['module']}",
                    "-v",
                    "--tb=short",
                    f"--junitxml={self.output_dir}/{suite['name']}_results.xml",
                    "--timeout=300"  # 5分钟超时
                ])
                
                suite_duration = time.time() - suite_start_time
                
                self.test_results[suite['name']] = {
                    "status": "passed" if result == 0 else "failed",
                    "duration": suite_duration,
                    "critical": suite['critical'],
                    "description": suite['description'],
                    "exit_code": result
                }
                
                if result == 0:
                    print(f"✓ {suite['name']} 测试通过 ({suite_duration:.2f}s)")
                else:
                    print(f"✗ {suite['name']} 测试失败 ({suite_duration:.2f}s)")
                    
            except Exception as e:
                suite_duration = time.time() - suite_start_time
                self.test_results[suite['name']] = {
                    "status": "error",
                    "duration": suite_duration,
                    "critical": suite['critical'],
                    "description": suite['description'],
                    "error": str(e)
                }
                print(f"✗ {suite['name']} 测试异常: {e}")
        
        total_duration = time.time() - start_time
        
        # 生成测试报告
        report = self._generate_integration_report(total_duration)
        
        # 保存报告
        self._save_integration_report(report)
        
        print(f"\n集成测试完成，总耗时: {total_duration:.2f}秒")
        print(f"测试报告已保存到: {self.output_dir}")
        
        return report
    
    def run_quick_integration_test(self) -> Dict[str, Any]:
        """运行快速集成测试"""
        print("运行快速集成测试...")
        
        # 只运行关键的集成测试
        critical_tests = [
            "tests/integration/test_evaluation_pipeline.py::TestCompleteEvaluationPipeline::test_end_to_end_evaluation_flow",
            "tests/integration/test_system_compatibility.py::TestEvaluationFrameworkCompatibility::test_comprehensive_evaluation_framework_integration"
        ]
        
        start_time = time.time()
        
        for test in critical_tests:
            print(f"运行: {test}")
            result = pytest.main([test, "-v", "--tb=short"])
            
            if result != 0:
                print(f"快速测试失败: {test}")
                return {"status": "failed", "failed_test": test}
        
        duration = time.time() - start_time
        print(f"快速集成测试通过 ({duration:.2f}s)")
        
        return {"status": "passed", "duration": duration}
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        print("运行性能基准测试...")
        
        benchmark_tests = [
            "tests/integration/test_large_scale_processing.py::TestLargeScaleProcessing::test_large_batch_evaluation",
            "tests/integration/test_multi_model_comparison.py::TestMultiModelComparison::test_parallel_model_evaluation"
        ]
        
        performance_results = {}
        
        for test in benchmark_tests:
            print(f"基准测试: {test}")
            
            start_time = time.time()
            result = pytest.main([test, "-v", "--tb=short", "-s"])
            duration = time.time() - start_time
            
            test_name = test.split("::")[-1]
            performance_results[test_name] = {
                "duration": duration,
                "status": "passed" if result == 0 else "failed"
            }
        
        return performance_results
    
    def _generate_integration_report(self, total_duration: float) -> Dict[str, Any]:
        """生成集成测试报告"""
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "passed"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "failed"])
        error_tests = len([r for r in self.test_results.values() if r["status"] == "error"])
        
        # 关键测试结果
        critical_tests = [name for name, result in self.test_results.items() if result["critical"]]
        critical_passed = len([name for name in critical_tests if self.test_results[name]["status"] == "passed"])
        
        # 性能分析
        performance_analysis = {
            "total_duration": total_duration,
            "average_test_duration": total_duration / total_tests if total_tests > 0 else 0,
            "slowest_test": max(self.test_results.items(), key=lambda x: x[1]["duration"]) if self.test_results else None,
            "fastest_test": min(self.test_results.items(), key=lambda x: x[1]["duration"]) if self.test_results else None
        }
        
        # 生成建议
        recommendations = self._generate_recommendations()
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "critical_tests_passed": critical_passed,
                "critical_tests_total": len(critical_tests),
                "overall_status": "passed" if critical_passed == len(critical_tests) and failed_tests == 0 else "failed"
            },
            "test_results": self.test_results,
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "environment": self._get_environment_info(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        failed_tests = [name for name, result in self.test_results.items() if result["status"] == "failed"]
        error_tests = [name for name, result in self.test_results.items() if result["status"] == "error"]
        
        if failed_tests:
            recommendations.append(f"有 {len(failed_tests)} 个测试失败，需要检查并修复: {', '.join(failed_tests)}")
        
        if error_tests:
            recommendations.append(f"有 {len(error_tests)} 个测试出现异常，需要调试: {', '.join(error_tests)}")
        
        # 性能建议
        if self.test_results:
            durations = [result["duration"] for result in self.test_results.values()]
            avg_duration = sum(durations) / len(durations)
            
            if avg_duration > 60:  # 平均超过1分钟
                recommendations.append("测试执行时间较长，建议优化测试性能或使用并行执行")
        
        # 关键测试建议
        critical_failed = [
            name for name, result in self.test_results.items() 
            if result["critical"] and result["status"] != "passed"
        ]
        
        if critical_failed:
            recommendations.append(f"关键测试失败，必须优先修复: {', '.join(critical_failed)}")
        
        if not recommendations:
            recommendations.append("所有集成测试通过，系统集成状态良好")
        
        return recommendations
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        import platform
        import sys
        
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except ImportError:
            torch_version = "未安装"
            cuda_available = False
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "torch_version": torch_version,
            "cuda_available": cuda_available
        }
    
    def _save_integration_report(self, report: Dict[str, Any]):
        """保存集成测试报告"""
        # JSON格式报告
        json_report_path = self.output_dir / "integration_test_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 文本格式摘要
        text_report_path = self.output_dir / "integration_test_summary.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("专家评估系统集成测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            summary = report["summary"]
            f.write(f"测试概况:\n")
            f.write(f"  总测试数: {summary['total_tests']}\n")
            f.write(f"  通过测试: {summary['passed_tests']}\n")
            f.write(f"  失败测试: {summary['failed_tests']}\n")
            f.write(f"  异常测试: {summary['error_tests']}\n")
            f.write(f"  成功率: {summary['success_rate']:.2%}\n")
            f.write(f"  整体状态: {summary['overall_status']}\n\n")
            
            f.write(f"关键测试:\n")
            f.write(f"  关键测试通过: {summary['critical_tests_passed']}/{summary['critical_tests_total']}\n\n")
            
            f.write("测试结果详情:\n")
            f.write("-" * 30 + "\n")
            for test_name, result in report["test_results"].items():
                status_symbol = "✓" if result["status"] == "passed" else "✗"
                critical_mark = " [关键]" if result["critical"] else ""
                f.write(f"{status_symbol} {test_name}{critical_mark}: {result['duration']:.2f}s\n")
                if result["status"] != "passed":
                    error_msg = result.get("error", "测试失败")
                    f.write(f"    错误: {error_msg}\n")
            
            f.write(f"\n性能分析:\n")
            f.write("-" * 30 + "\n")
            perf = report["performance_analysis"]
            f.write(f"总耗时: {perf['total_duration']:.2f}秒\n")
            f.write(f"平均测试时间: {perf['average_test_duration']:.2f}秒\n")
            
            if perf["slowest_test"]:
                slowest_name, slowest_data = perf["slowest_test"]
                f.write(f"最慢测试: {slowest_name} ({slowest_data['duration']:.2f}s)\n")
            
            f.write(f"\n改进建议:\n")
            f.write("-" * 30 + "\n")
            for rec in report["recommendations"]:
                f.write(f"• {rec}\n")
        
        print(f"详细报告: {json_report_path}")
        print(f"测试摘要: {text_report_path}")


def run_integration_tests():
    """运行集成测试的主函数"""
    runner = IntegrationTestRunner()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            return runner.run_quick_integration_test()
        elif sys.argv[1] == "--benchmark":
            return runner.run_performance_benchmark()
    
    # 运行完整集成测试
    return runner.run_all_integration_tests()


def test_integration_runner_functionality():
    """测试集成测试运行器本身的功能"""
    runner = IntegrationTestRunner("test_runner_output")
    
    # 测试环境信息获取
    env_info = runner._get_environment_info()
    assert "python_version" in env_info
    assert "platform" in env_info
    
    # 测试建议生成
    runner.test_results = {
        "test1": {"status": "passed", "duration": 10.0, "critical": True},
        "test2": {"status": "failed", "duration": 5.0, "critical": False},
        "test3": {"status": "error", "duration": 2.0, "critical": True}
    }
    
    recommendations = runner._generate_recommendations()
    assert len(recommendations) > 0
    assert any("失败" in rec for rec in recommendations)
    assert any("异常" in rec for rec in recommendations)
    
    # 测试报告生成
    report = runner._generate_integration_report(30.0)
    assert "summary" in report
    assert "test_results" in report
    assert "performance_analysis" in report
    assert "recommendations" in report
    
    # 验证摘要统计
    summary = report["summary"]
    assert summary["total_tests"] == 3
    assert summary["passed_tests"] == 1
    assert summary["failed_tests"] == 1
    assert summary["error_tests"] == 1
    assert summary["overall_status"] == "failed"  # 因为有关键测试失败


if __name__ == "__main__":
    # 如果直接运行此文件，执行集成测试
    result = run_integration_tests()
    
    # 根据结果设置退出码
    if isinstance(result, dict):
        if result.get("summary", {}).get("overall_status") == "failed":
            sys.exit(1)
        elif result.get("status") == "failed":
            sys.exit(1)
    
    sys.exit(0)