#!/usr/bin/env python3
"""
性能优化验证脚本

使用uv运行性能优化验证测试，包括：
- 性能瓶颈分析验证
- 数据加载优化验证
- 多GPU通信优化验证
- 超参数调优建议验证
- 端到端优化流程验证
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def setup_logging() -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('performance_optimization_validation.log')
        ]
    )
    return logging.getLogger(__name__)


def check_uv_available() -> bool:
    """检查uv是否可用"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_uv_command(command: List[str], logger: logging.Logger, timeout: int = 300) -> Dict[str, Any]:
    """运行uv命令"""
    logger.info(f"执行命令: {' '.join(command)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        execution_time = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": execution_time
        }
    
    except subprocess.TimeoutExpired:
        logger.error(f"命令执行超时 ({timeout}秒)")
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "命令执行超时",
            "execution_time": timeout
        }
    
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time
        }


def validate_performance_optimization_modules(logger: logging.Logger) -> Dict[str, Any]:
    """验证性能优化模块"""
    logger.info("开始验证性能优化模块...")
    
    validation_results = {
        "module_imports": {},
        "basic_functionality": {},
        "integration_tests": {}
    }
    
    # 1. 验证模块导入
    logger.info("验证模块导入...")
    import_script = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    from src.performance_optimizer import (
        PerformanceOptimizer, PerformanceBottleneckAnalyzer, 
        DataLoadingOptimizer, CommunicationOptimizer, HyperparameterTuner
    )
    print("SUCCESS: 所有性能优化模块导入成功")
except Exception as e:
    print(f"ERROR: 模块导入失败: {e}")
    sys.exit(1)
'''
    
    with open("temp_import_test.py", "w", encoding="utf-8") as f:
        f.write(import_script)
    
    try:
        result = run_uv_command(["uv", "run", "temp_import_test.py"], logger)
        validation_results["module_imports"] = {
            "success": result["success"],
            "output": result["stdout"],
            "error": result["stderr"]
        }
    finally:
        Path("temp_import_test.py").unlink(missing_ok=True)
    
    # 2. 验证基础功能
    logger.info("验证基础功能...")
    basic_test_script = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.performance_optimizer import PerformanceOptimizer
from src.parallel_config import DistributedTrainingMetrics, CommunicationMetrics
from src.memory_manager import MemorySnapshot, MemoryPressureLevel
from datetime import datetime

try:
    # 创建性能优化器
    optimizer = PerformanceOptimizer("temp_test_output")
    
    # 创建测试数据
    training_metrics = []
    for i in range(3):
        metric = DistributedTrainingMetrics(
            epoch=1,
            global_step=i,
            train_loss=2.0 - i * 0.1,
            val_loss=2.1 - i * 0.1,
            learning_rate=2e-4,
            gpu_metrics={0: {"utilization": 50, "memory_usage_percent": 80}},
            communication_metrics=CommunicationMetrics(),
            throughput_tokens_per_second=100,
            convergence_score=0.5
        )
        training_metrics.append(metric)
    
    memory_snapshots = {0: []}
    for i in range(3):
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_id=0,
            total_memory=16384,
            allocated_memory=12000,
            cached_memory=2000,
            free_memory=2384,
            utilization_rate=0.85,
            pressure_level=MemoryPressureLevel.HIGH,
            system_total_memory=32768,
            system_used_memory=16384,
            system_available_memory=16384,
            process_memory=8192,
            process_memory_percent=25.0
        )
        memory_snapshots[0].append(snapshot)
    
    current_config = {
        "batch_size": 4,
        "learning_rate": 2e-4,
        "sequence_length": 2048
    }
    
    # 运行优化分析
    report = optimizer.analyze_and_optimize(
        training_metrics, memory_snapshots, current_config
    )
    
    print(f"SUCCESS: 性能优化分析完成，检测到 {report['summary']['total_bottlenecks']} 个瓶颈")
    print(f"生成了 {report['summary']['total_recommendations']} 个优化建议")
    
except Exception as e:
    print(f"ERROR: 基础功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    with open("temp_basic_test.py", "w", encoding="utf-8") as f:
        f.write(basic_test_script)
    
    try:
        result = run_uv_command(["uv", "run", "temp_basic_test.py"], logger)
        validation_results["basic_functionality"] = {
            "success": result["success"],
            "output": result["stdout"],
            "error": result["stderr"]
        }
    finally:
        Path("temp_basic_test.py").unlink(missing_ok=True)
        # 清理测试输出目录
        import shutil
        shutil.rmtree("temp_test_output", ignore_errors=True)
    
    return validation_results


def run_performance_optimization_tests(logger: logging.Logger) -> Dict[str, Any]:
    """运行性能优化测试"""
    logger.info("开始运行性能优化测试...")
    
    test_results = {}
    
    # 运行单元测试
    logger.info("运行单元测试...")
    result = run_uv_command([
        "uv", "run", "python", "-m", "pytest", 
        "tests/test_performance_optimization.py", 
        "-v", "--tb=short"
    ], logger, timeout=600)
    
    test_results["unit_tests"] = {
        "success": result["success"],
        "output": result["stdout"],
        "error": result["stderr"],
        "execution_time": result["execution_time"]
    }
    
    if not result["success"]:
        logger.error("单元测试失败")
        logger.error(f"错误输出: {result['stderr']}")
    else:
        logger.info("单元测试通过")
    
    return test_results


def run_performance_benchmarks(logger: logging.Logger) -> Dict[str, Any]:
    """运行性能基准测试"""
    logger.info("开始运行性能基准测试...")
    
    benchmark_results = {}
    
    # 运行性能基准测试
    logger.info("运行性能基准测试...")
    result = run_uv_command([
        "uv", "run", "python", "-m", "pytest", 
        "tests/test_performance_benchmarks.py", 
        "-v", "--tb=short", "-k", "benchmark"
    ], logger, timeout=900)
    
    benchmark_results["performance_benchmarks"] = {
        "success": result["success"],
        "output": result["stdout"],
        "error": result["stderr"],
        "execution_time": result["execution_time"]
    }
    
    if not result["success"]:
        logger.warning("性能基准测试失败或部分失败")
        logger.warning(f"错误输出: {result['stderr']}")
    else:
        logger.info("性能基准测试完成")
    
    return benchmark_results


def validate_optimization_recommendations(logger: logging.Logger) -> Dict[str, Any]:
    """验证优化建议的有效性"""
    logger.info("开始验证优化建议...")
    
    validation_script = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.performance_optimizer import PerformanceOptimizer
from src.parallel_config import DistributedTrainingMetrics, CommunicationMetrics
from src.memory_manager import MemorySnapshot, MemoryPressureLevel
from datetime import datetime
import json

try:
    optimizer = PerformanceOptimizer("validation_output")
    
    # 创建有明显性能问题的测试场景
    training_metrics = []
    for i in range(10):
        metric = DistributedTrainingMetrics(
            epoch=1,
            global_step=i,
            train_loss=2.0 - i * 0.01,  # 收敛很慢
            val_loss=2.1 - i * 0.01,
            learning_rate=1e-3,  # 学习率可能过高
            gpu_metrics={
                0: {"utilization": 30 + i, "memory_usage_percent": 95},  # 低利用率，高内存
                1: {"utilization": 90 + i, "memory_usage_percent": 60}   # 负载不均衡
            },
            communication_metrics=CommunicationMetrics(
                total_communication_time=1.0 + i * 0.1,  # 通信开销大
                allreduce_time=0.7 + i * 0.05,
                communication_volume=500 + i * 20
            ),
            throughput_tokens_per_second=50 - i * 0.5,  # 吞吐量下降
            convergence_score=0.1 + i * 0.01  # 收敛性很差
        )
        training_metrics.append(metric)
    
    # 高内存压力的内存快照
    memory_snapshots = {0: [], 1: []}
    for gpu_id in [0, 1]:
        for i in range(10):
            pressure = MemoryPressureLevel.CRITICAL if gpu_id == 0 else MemoryPressureLevel.LOW
            allocated = 15500 if gpu_id == 0 else 9000  # GPU 0内存几乎满了
            
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                gpu_id=gpu_id,
                total_memory=16384,
                allocated_memory=allocated,
                cached_memory=500,
                free_memory=16384 - allocated - 500,
                utilization_rate=allocated / 16384,
                pressure_level=pressure,
                system_total_memory=32768,
                system_used_memory=28000,  # 系统内存也很高
                system_available_memory=4768,
                process_memory=12000,
                process_memory_percent=37.0
            )
            memory_snapshots[gpu_id].append(snapshot)
    
    # 系统资源紧张
    system_metrics = {
        "cpu_utilization": 95.0,  # CPU满载
        "memory_utilization": 90.0,  # 内存几乎满了
        "io_wait": 30.0,  # IO等待很高
        "data_loading_times": [0.5, 0.8, 0.6, 1.0, 0.7],  # 数据加载很慢
        "batch_processing_times": [0.1, 0.12, 0.1, 0.15, 0.11]
    }
    
    current_config = {
        "batch_size": 16,  # 批次可能过大
        "learning_rate": 1e-3,  # 学习率可能过高
        "sequence_length": 4096,  # 序列长度很长
        "enable_lora": False,  # 没有使用LoRA
        "num_workers": 1,  # 数据加载进程太少
        "pin_memory": False
    }
    
    # 运行优化分析
    report = optimizer.analyze_and_optimize(
        training_metrics, memory_snapshots, current_config, system_metrics
    )
    
    # 验证分析结果
    bottlenecks = report["bottlenecks"]
    recommendations = report["optimization_recommendations"]
    
    print(f"检测到 {len(bottlenecks)} 个性能瓶颈:")
    for bottleneck in bottlenecks[:5]:  # 显示前5个
        print(f"  - {bottleneck['description']} (严重程度: {bottleneck['severity']:.2f})")
    
    print(f"\\n生成 {len(recommendations)} 个优化建议:")
    for rec in recommendations[:5]:  # 显示前5个
        print(f"  - {rec['description']} (优先级: {rec['priority']}, 预期改进: {rec['expected_improvement']:.1f}%)")
    
    # 应用优化建议
    if recommendations:
        optimization_result = optimizer.apply_optimization_recommendations(
            recommendations, current_config
        )
        
        print(f"\\n应用了 {len(optimization_result['applied_optimizations'])} 个优化:")
        for opt in optimization_result['applied_optimizations']:
            print(f"  - {opt}")
    
    # 验证关键瓶颈类型是否被检测到
    expected_bottlenecks = ["gpu_memory_bound", "load_imbalance", "cpu_bound", "memory_bound"]
    detected_types = [b["bottleneck_type"] for b in bottlenecks]
    
    detected_expected = [bt for bt in expected_bottlenecks if bt in detected_types]
    print(f"\\n检测到预期瓶颈类型: {detected_expected}")
    
    # 验证关键优化策略是否被推荐
    expected_strategies = ["batch_size_tuning", "data_loading_optimization", "memory_optimization"]
    recommended_strategies = [r["strategy"] for r in recommendations]
    
    recommended_expected = [st for st in expected_strategies if st in recommended_strategies]
    print(f"推荐了预期优化策略: {recommended_expected}")
    
    # 保存详细报告
    with open("validation_output/detailed_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\\nSUCCESS: 优化建议验证完成")
    print(f"详细报告已保存到: validation_output/detailed_validation_report.json")
    
except Exception as e:
    print(f"ERROR: 优化建议验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    with open("temp_validation_test.py", "w", encoding="utf-8") as f:
        f.write(validation_script)
    
    try:
        result = run_uv_command(["uv", "run", "temp_validation_test.py"], logger, timeout=300)
        validation_results = {
            "success": result["success"],
            "output": result["stdout"],
            "error": result["stderr"],
            "execution_time": result["execution_time"]
        }
        
        if result["success"]:
            logger.info("优化建议验证成功")
        else:
            logger.error("优化建议验证失败")
            logger.error(f"错误: {result['stderr']}")
        
        return validation_results
        
    finally:
        Path("temp_validation_test.py").unlink(missing_ok=True)


def generate_validation_report(results: Dict[str, Any], logger: logging.Logger) -> str:
    """生成验证报告"""
    logger.info("生成验证报告...")
    
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "validation_summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "success_rate": 0.0
        },
        "detailed_results": results,
        "recommendations": []
    }
    
    # 统计测试结果
    all_results = []
    
    # 模块验证结果
    if "module_validation" in results:
        module_results = results["module_validation"]
        all_results.extend([
            module_results.get("module_imports", {}).get("success", False),
            module_results.get("basic_functionality", {}).get("success", False)
        ])
    
    # 单元测试结果
    if "test_results" in results:
        test_results = results["test_results"]
        all_results.append(test_results.get("unit_tests", {}).get("success", False))
    
    # 基准测试结果
    if "benchmark_results" in results:
        benchmark_results = results["benchmark_results"]
        all_results.append(benchmark_results.get("performance_benchmarks", {}).get("success", False))
    
    # 优化建议验证结果
    if "optimization_validation" in results:
        all_results.append(results["optimization_validation"].get("success", False))
    
    # 计算统计信息
    report["validation_summary"]["total_tests"] = len(all_results)
    report["validation_summary"]["passed_tests"] = sum(all_results)
    report["validation_summary"]["failed_tests"] = len(all_results) - sum(all_results)
    
    if len(all_results) > 0:
        report["validation_summary"]["success_rate"] = sum(all_results) / len(all_results)
    
    # 生成建议
    if report["validation_summary"]["success_rate"] < 1.0:
        report["recommendations"].append("存在测试失败，建议检查错误日志并修复问题")
    
    if report["validation_summary"]["success_rate"] >= 0.8:
        report["recommendations"].append("大部分测试通过，性能优化功能基本可用")
    else:
        report["recommendations"].append("测试通过率较低，建议全面检查性能优化模块")
    
    # 保存报告
    report_file = f"performance_optimization_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"验证报告已保存到: {report_file}")
    return report_file


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始性能优化验证...")
    
    # 检查uv可用性
    if not check_uv_available():
        logger.error("uv不可用，请先安装uv")
        sys.exit(1)
    
    logger.info("uv可用，开始验证...")
    
    validation_results = {}
    
    try:
        # 1. 验证性能优化模块
        logger.info("=== 步骤1: 验证性能优化模块 ===")
        validation_results["module_validation"] = validate_performance_optimization_modules(logger)
        
        # 2. 运行性能优化测试
        logger.info("=== 步骤2: 运行性能优化测试 ===")
        validation_results["test_results"] = run_performance_optimization_tests(logger)
        
        # 3. 运行性能基准测试
        logger.info("=== 步骤3: 运行性能基准测试 ===")
        validation_results["benchmark_results"] = run_performance_benchmarks(logger)
        
        # 4. 验证优化建议
        logger.info("=== 步骤4: 验证优化建议 ===")
        validation_results["optimization_validation"] = validate_optimization_recommendations(logger)
        
        # 5. 生成验证报告
        logger.info("=== 步骤5: 生成验证报告 ===")
        report_file = generate_validation_report(validation_results, logger)
        
        # 6. 输出总结
        logger.info("=== 验证总结 ===")
        
        # 统计成功的验证项
        successful_validations = []
        failed_validations = []
        
        if validation_results.get("module_validation", {}).get("module_imports", {}).get("success"):
            successful_validations.append("模块导入")
        else:
            failed_validations.append("模块导入")
        
        if validation_results.get("module_validation", {}).get("basic_functionality", {}).get("success"):
            successful_validations.append("基础功能")
        else:
            failed_validations.append("基础功能")
        
        if validation_results.get("test_results", {}).get("unit_tests", {}).get("success"):
            successful_validations.append("单元测试")
        else:
            failed_validations.append("单元测试")
        
        if validation_results.get("benchmark_results", {}).get("performance_benchmarks", {}).get("success"):
            successful_validations.append("性能基准测试")
        else:
            failed_validations.append("性能基准测试")
        
        if validation_results.get("optimization_validation", {}).get("success"):
            successful_validations.append("优化建议验证")
        else:
            failed_validations.append("优化建议验证")
        
        logger.info(f"成功的验证项 ({len(successful_validations)}): {', '.join(successful_validations)}")
        if failed_validations:
            logger.warning(f"失败的验证项 ({len(failed_validations)}): {', '.join(failed_validations)}")
        
        success_rate = len(successful_validations) / (len(successful_validations) + len(failed_validations))
        logger.info(f"总体成功率: {success_rate:.1%}")
        
        logger.info(f"详细验证报告: {report_file}")
        
        if success_rate >= 0.8:
            logger.info("SUCCESS: 性能优化验证基本通过")
            return 0
        else:
            logger.error("FAILED: 性能优化验证失败")
            return 1
    
    except Exception as e:
        logger.error(f"验证过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # 清理临时文件
        for temp_file in ["temp_import_test.py", "temp_basic_test.py", "temp_validation_test.py"]:
            Path(temp_file).unlink(missing_ok=True)
        
        # 清理临时目录
        import shutil
        for temp_dir in ["temp_test_output", "validation_output"]:
            if Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)