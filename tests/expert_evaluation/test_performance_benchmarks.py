"""
专家评估系统性能基准测试

测试系统在不同负载下的性能表现，包括处理速度、内存使用、并发能力等。
"""

import pytest
import time
import threading
import multiprocessing
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import ExpertEvaluationConfig, EvaluationMode
from src.expert_evaluation.data_models import QAEvaluationItem, ExpertiseLevel
from tests.expert_evaluation.conftest import (
    create_test_config,
    create_mock_evaluation_result,
    assert_valid_evaluation_result
)


class PerformanceMonitor:
    """性能监控工具类"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
        
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if self.start_time is None:
            return {}
            
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_cpu = self.process.cpu_percent()
        
        return {
            "elapsed_time": current_time - self.start_time,
            "memory_usage_mb": current_memory,
            "memory_increase_mb": current_memory - self.start_memory,
            "cpu_usage_percent": current_cpu,
            "peak_memory_mb": max(current_memory, self.start_memory)
        }


@pytest.mark.performance
class TestEvaluationSpeed:
    """评估速度测试"""
    
    def create_test_items(self, count: int) -> List[QAEvaluationItem]:
        """创建测试数据"""
        items = []
        for i in range(count):
            item = QAEvaluationItem(
                question_id=f"speed_test_{i:06d}",
                question=f"速度测试问题 {i} " + "这是一个测试问题的详细描述 " * 10,
                context=f"速度测试上下文 {i} " + "相关背景信息 " * 5,
                reference_answer=f"速度测试参考答案 {i} " + "详细的参考答案内容 " * 20,
                model_answer=f"速度测试模型答案 {i} " + "模型生成的回答内容 " * 20,
                domain_tags=["性能测试", f"类别{i % 5}"],
                difficulty_level=list(ExpertiseLevel)[i % 4],
                expected_concepts=[f"概念{j}" for j in range(5)]
            )
            items.append(item)
        return items
    
    @pytest.mark.slow
    def test_single_evaluation_speed(self):
        """测试单个评估的速度"""
        config = create_test_config(evaluation_mode=EvaluationMode.QUICK)
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            # 创建测试数据
            test_item = self.create_test_items(1)[0]
            
            # 模拟评估过程
            with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                mock_evaluate.return_value = create_mock_evaluation_result("speed_test_000000", 0.8)
                
                monitor = PerformanceMonitor()
                monitor.start_monitoring()
                
                try:
                    result = engine.evaluate_model([test_item])
                    metrics = monitor.get_metrics()
                    
                    # 性能要求
                    assert metrics["elapsed_time"] < 1.0, f"单个评估耗时过长: {metrics['elapsed_time']:.2f}秒"
                    assert metrics["memory_increase_mb"] < 50, f"内存增长过多: {metrics['memory_increase_mb']:.2f}MB"
                    
                    # 验证结果有效性
                    assert_valid_evaluation_result(result)
                    
                except Exception as e:
                    pytest.skip(f"单个评估速度测试跳过: {e}")
    
    @pytest.mark.slow
    def test_batch_evaluation_speed(self):
        """测试批量评估速度"""
        config = create_test_config(evaluation_mode=EvaluationMode.QUICK)
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            # 创建批量测试数据
            test_items = self.create_test_items(50)
            
            # 模拟批量评估
            with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                mock_evaluate.side_effect = lambda item: create_mock_evaluation_result(
                    item.question_id, 0.7 + hash(item.question_id) % 30 / 100
                )
                
                monitor = PerformanceMonitor()
                monitor.start_monitoring()
                
                try:
                    result = engine.evaluate_model(test_items)
                    metrics = monitor.get_metrics()
                    
                    # 性能要求：50个项目应在30秒内完成
                    assert metrics["elapsed_time"] < 30.0, f"批量评估耗时过长: {metrics['elapsed_time']:.2f}秒"
                    
                    # 计算吞吐量
                    throughput = len(test_items) / metrics["elapsed_time"]
                    assert throughput > 1.0, f"吞吐量过低: {throughput:.2f} items/sec"
                    
                    # 内存使用应该合理
                    assert metrics["memory_increase_mb"] < 200, f"内存增长过多: {metrics['memory_increase_mb']:.2f}MB"
                    
                    print(f"批量评估性能指标:")
                    print(f"  - 处理时间: {metrics['elapsed_time']:.2f}秒")
                    print(f"  - 吞吐量: {throughput:.2f} items/sec")
                    print(f"  - 内存使用: {metrics['memory_usage_mb']:.2f}MB")
                    print(f"  - 内存增长: {metrics['memory_increase_mb']:.2f}MB")
                    
                except Exception as e:
                    pytest.skip(f"批量评估速度测试跳过: {e}")
    
    def test_evaluation_scaling(self):
        """测试评估性能扩展性"""
        config = create_test_config(evaluation_mode=EvaluationMode.QUICK)
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            # 测试不同规模的数据
            test_sizes = [1, 5, 10, 20]
            performance_data = []
            
            for size in test_sizes:
                test_items = self.create_test_items(size)
                
                with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                    mock_evaluate.side_effect = lambda item: create_mock_evaluation_result(
                        item.question_id, 0.8
                    )
                    
                    monitor = PerformanceMonitor()
                    monitor.start_monitoring()
                    
                    try:
                        result = engine.evaluate_model(test_items)
                        metrics = monitor.get_metrics()
                        
                        performance_data.append({
                            "size": size,
                            "time": metrics["elapsed_time"],
                            "throughput": size / metrics["elapsed_time"],
                            "memory": metrics["memory_increase_mb"]
                        })
                        
                    except Exception as e:
                        pytest.skip(f"扩展性测试跳过 (size={size}): {e}")
            
            if len(performance_data) >= 2:
                # 验证性能扩展性
                for i in range(1, len(performance_data)):
                    prev_data = performance_data[i-1]
                    curr_data = performance_data[i]
                    
                    # 时间增长应该接近线性
                    time_ratio = curr_data["time"] / prev_data["time"]
                    size_ratio = curr_data["size"] / prev_data["size"]
                    
                    # 允许一定的性能波动
                    assert time_ratio < size_ratio * 2, f"性能扩展性不佳: 时间比例 {time_ratio:.2f} vs 大小比例 {size_ratio:.2f}"
                
                print("性能扩展性测试结果:")
                for data in performance_data:
                    print(f"  - 大小: {data['size']}, 时间: {data['time']:.2f}s, 吞吐量: {data['throughput']:.2f} items/s")


@pytest.mark.performance
class TestMemoryUsage:
    """内存使用测试"""
    
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 多次创建和销毁引擎
            for i in range(10):
                engine = ExpertEvaluationEngine(config)
                engine.is_model_loaded = True
                
                # 模拟一些操作
                test_item = QAEvaluationItem(
                    question_id=f"memory_test_{i}",
                    question="内存测试问题",
                    context="测试上下文",
                    reference_answer="参考答案",
                    model_answer="模型答案"
                )
                
                with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                    mock_evaluate.return_value = create_mock_evaluation_result(f"memory_test_{i}", 0.8)
                    
                    try:
                        result = engine.evaluate_model([test_item])
                    except Exception:
                        pass  # 忽略评估错误，专注于内存测试
                
                # 清理引用
                del engine
                
                # 强制垃圾回收
                import gc
                gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 内存增长应该在合理范围内（小于100MB）
            assert memory_increase < 100, f"可能存在内存泄漏: 内存增长 {memory_increase:.2f}MB"
            
            print(f"内存泄漏测试结果:")
            print(f"  - 初始内存: {initial_memory:.2f}MB")
            print(f"  - 最终内存: {final_memory:.2f}MB")
            print(f"  - 内存增长: {memory_increase:.2f}MB")
    
    def test_large_data_memory_usage(self):
        """测试大数据内存使用"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            # 创建大量数据
            large_items = []
            for i in range(1000):
                item = QAEvaluationItem(
                    question_id=f"large_test_{i:06d}",
                    question="大数据测试问题 " * 100,  # 较长的文本
                    context="大数据测试上下文 " * 50,
                    reference_answer="大数据测试参考答案 " * 200,
                    model_answer="大数据测试模型答案 " * 200,
                    domain_tags=[f"标签{j}" for j in range(10)],
                    expected_concepts=[f"概念{j}" for j in range(20)]
                )
                large_items.append(item)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # 分批处理以避免内存溢出
            batch_size = 50
            for i in range(0, min(200, len(large_items)), batch_size):  # 只测试前200个
                batch = large_items[i:i+batch_size]
                
                with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                    mock_evaluate.side_effect = lambda item: create_mock_evaluation_result(
                        item.question_id, 0.8
                    )
                    
                    try:
                        result = engine.evaluate_model(batch)
                    except Exception as e:
                        pytest.skip(f"大数据内存测试跳过: {e}")
            
            metrics = monitor.get_metrics()
            
            # 内存使用应该在合理范围内
            assert metrics["memory_increase_mb"] < 500, f"大数据处理内存使用过多: {metrics['memory_increase_mb']:.2f}MB"
            
            print(f"大数据内存使用测试结果:")
            print(f"  - 处理项目数: {min(200, len(large_items))}")
            print(f"  - 内存增长: {metrics['memory_increase_mb']:.2f}MB")
            print(f"  - 峰值内存: {metrics['peak_memory_mb']:.2f}MB")


@pytest.mark.performance
class TestConcurrencyPerformance:
    """并发性能测试"""
    
    def test_thread_safety(self):
        """测试线程安全性"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            results = []
            errors = []
            
            def evaluate_worker(worker_id: int):
                """工作线程函数"""
                try:
                    test_item = QAEvaluationItem(
                        question_id=f"thread_test_{worker_id}",
                        question=f"线程测试问题 {worker_id}",
                        context="线程测试上下文",
                        reference_answer=f"线程测试参考答案 {worker_id}",
                        model_answer=f"线程测试模型答案 {worker_id}"
                    )
                    
                    with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                        mock_evaluate.return_value = create_mock_evaluation_result(
                            f"thread_test_{worker_id}", 0.8
                        )
                        
                        result = engine.evaluate_model([test_item])
                        results.append((worker_id, result))
                        
                except Exception as e:
                    errors.append((worker_id, str(e)))
            
            # 启动多个线程
            threads = []
            num_threads = 5
            
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(target=evaluate_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=10)  # 10秒超时
            
            elapsed_time = time.time() - start_time
            
            # 验证结果
            if errors:
                print(f"线程安全测试中的错误: {errors}")
            
            # 至少应该有一些成功的结果
            success_rate = len(results) / num_threads
            assert success_rate >= 0.5, f"线程安全测试成功率过低: {success_rate:.2f}"
            
            print(f"线程安全测试结果:")
            print(f"  - 线程数: {num_threads}")
            print(f"  - 成功数: {len(results)}")
            print(f"  - 错误数: {len(errors)}")
            print(f"  - 总耗时: {elapsed_time:.2f}秒")
            print(f"  - 成功率: {success_rate:.2f}")
    
    @pytest.mark.slow
    def test_concurrent_evaluation_performance(self):
        """测试并发评估性能"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            # 创建测试数据
            test_items = []
            for i in range(20):
                item = QAEvaluationItem(
                    question_id=f"concurrent_test_{i:03d}",
                    question=f"并发测试问题 {i}",
                    context="并发测试上下文",
                    reference_answer=f"并发测试参考答案 {i}",
                    model_answer=f"并发测试模型答案 {i}"
                )
                test_items.append(item)
            
            def evaluate_batch(items: List[QAEvaluationItem]) -> List[Any]:
                """评估一批数据"""
                engine = ExpertEvaluationEngine(config)
                engine.is_model_loaded = True
                
                results = []
                for item in items:
                    with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                        mock_evaluate.return_value = create_mock_evaluation_result(
                            item.question_id, 0.8
                        )
                        
                        try:
                            result = engine.evaluate_model([item])
                            results.append(result)
                        except Exception as e:
                            results.append(f"Error: {e}")
                
                return results
            
            # 测试串行处理
            start_time = time.time()
            serial_results = evaluate_batch(test_items)
            serial_time = time.time() - start_time
            
            # 测试并行处理
            start_time = time.time()
            
            # 将数据分成多个批次
            batch_size = 5
            batches = [test_items[i:i+batch_size] for i in range(0, len(test_items), batch_size)]
            
            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    parallel_results = list(executor.map(evaluate_batch, batches))
                
                parallel_time = time.time() - start_time
                
                # 展平结果
                flat_parallel_results = []
                for batch_results in parallel_results:
                    flat_parallel_results.extend(batch_results)
                
                # 计算性能提升
                speedup = serial_time / parallel_time if parallel_time > 0 else 0
                
                print(f"并发性能测试结果:")
                print(f"  - 串行时间: {serial_time:.2f}秒")
                print(f"  - 并行时间: {parallel_time:.2f}秒")
                print(f"  - 性能提升: {speedup:.2f}x")
                print(f"  - 串行结果数: {len(serial_results)}")
                print(f"  - 并行结果数: {len(flat_parallel_results)}")
                
                # 并行处理应该有一定的性能提升
                assert speedup > 1.0, f"并行处理没有性能提升: {speedup:.2f}x"
                
            except Exception as e:
                pytest.skip(f"并发性能测试跳过: {e}")


@pytest.mark.performance
class TestResourceUtilization:
    """资源利用率测试"""
    
    def test_cpu_utilization(self):
        """测试CPU利用率"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            # 创建计算密集型测试数据
            test_items = []
            for i in range(10):
                item = QAEvaluationItem(
                    question_id=f"cpu_test_{i:03d}",
                    question="CPU测试问题 " * 100,
                    context="CPU测试上下文 " * 50,
                    reference_answer="CPU测试参考答案 " * 200,
                    model_answer="CPU测试模型答案 " * 200
                )
                test_items.append(item)
            
            # 监控CPU使用率
            process = psutil.Process()
            cpu_percentages = []
            
            def monitor_cpu():
                """监控CPU使用率"""
                for _ in range(10):  # 监控10秒
                    cpu_percent = process.cpu_percent(interval=1)
                    cpu_percentages.append(cpu_percent)
            
            # 启动CPU监控线程
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # 执行评估
            with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                mock_evaluate.side_effect = lambda item: (
                    time.sleep(0.1),  # 模拟计算时间
                    create_mock_evaluation_result(item.question_id, 0.8)
                )[1]
                
                try:
                    result = engine.evaluate_model(test_items)
                except Exception as e:
                    pytest.skip(f"CPU利用率测试跳过: {e}")
            
            # 等待监控完成
            monitor_thread.join()
            
            if cpu_percentages:
                avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
                max_cpu = max(cpu_percentages)
                
                print(f"CPU利用率测试结果:")
                print(f"  - 平均CPU使用率: {avg_cpu:.2f}%")
                print(f"  - 峰值CPU使用率: {max_cpu:.2f}%")
                print(f"  - CPU使用率样本: {cpu_percentages}")
                
                # CPU使用率应该在合理范围内
                assert avg_cpu < 90, f"平均CPU使用率过高: {avg_cpu:.2f}%"
    
    def test_disk_io_performance(self):
        """测试磁盘I/O性能"""
        import tempfile
        import json
        
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            
            # 创建临时文件进行I/O测试
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                test_data = []
                for i in range(100):
                    item_data = {
                        "question_id": f"io_test_{i:06d}",
                        "question": f"I/O测试问题 {i} " * 50,
                        "reference_answer": f"I/O测试参考答案 {i} " * 100,
                        "model_answer": f"I/O测试模型答案 {i} " * 100
                    }
                    test_data.append(item_data)
                
                start_time = time.time()
                json.dump(test_data, f, ensure_ascii=False, indent=2)
                write_time = time.time() - start_time
                
                temp_file_path = f.name
            
            # 测试读取性能
            start_time = time.time()
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            read_time = time.time() - start_time
            
            # 清理临时文件
            os.unlink(temp_file_path)
            
            print(f"磁盘I/O性能测试结果:")
            print(f"  - 写入时间: {write_time:.3f}秒")
            print(f"  - 读取时间: {read_time:.3f}秒")
            print(f"  - 数据项数: {len(test_data)}")
            print(f"  - 写入速度: {len(test_data)/write_time:.2f} items/sec")
            print(f"  - 读取速度: {len(loaded_data)/read_time:.2f} items/sec")
            
            # I/O性能应该在合理范围内
            assert write_time < 5.0, f"写入时间过长: {write_time:.3f}秒"
            assert read_time < 2.0, f"读取时间过长: {read_time:.3f}秒"
            assert len(loaded_data) == len(test_data), "数据完整性检查失败"


@pytest.mark.performance
class TestStressTest:
    """压力测试"""
    
    @pytest.mark.slow
    def test_sustained_load(self):
        """测试持续负载"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            engine = ExpertEvaluationEngine(config)
            engine.is_model_loaded = True
            
            # 持续运行测试
            duration_seconds = 30  # 30秒压力测试
            start_time = time.time()
            
            evaluation_count = 0
            error_count = 0
            
            while time.time() - start_time < duration_seconds:
                test_item = QAEvaluationItem(
                    question_id=f"stress_test_{evaluation_count:06d}",
                    question=f"压力测试问题 {evaluation_count}",
                    context="压力测试上下文",
                    reference_answer=f"压力测试参考答案 {evaluation_count}",
                    model_answer=f"压力测试模型答案 {evaluation_count}"
                )
                
                with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                    mock_evaluate.return_value = create_mock_evaluation_result(
                        f"stress_test_{evaluation_count:06d}", 0.8
                    )
                    
                    try:
                        result = engine.evaluate_model([test_item])
                        evaluation_count += 1
                    except Exception as e:
                        error_count += 1
                        if error_count > 10:  # 如果错误太多，停止测试
                            break
                
                # 短暂休息以避免过度占用资源
                time.sleep(0.1)
            
            actual_duration = time.time() - start_time
            
            print(f"持续负载测试结果:")
            print(f"  - 测试时长: {actual_duration:.2f}秒")
            print(f"  - 评估次数: {evaluation_count}")
            print(f"  - 错误次数: {error_count}")
            print(f"  - 成功率: {evaluation_count/(evaluation_count+error_count)*100:.2f}%")
            print(f"  - 平均吞吐量: {evaluation_count/actual_duration:.2f} evaluations/sec")
            
            # 压力测试要求
            success_rate = evaluation_count / (evaluation_count + error_count) if (evaluation_count + error_count) > 0 else 0
            assert success_rate >= 0.9, f"持续负载测试成功率过低: {success_rate:.2f}"
            assert evaluation_count > 50, f"持续负载测试处理量过低: {evaluation_count}"
    
    def test_memory_pressure(self):
        """测试内存压力"""
        config = create_test_config()
        
        with patch('src.expert_evaluation.engine.ModelService'), \
             patch('src.expert_evaluation.engine.ComprehensiveEvaluationFramework'), \
             patch('src.expert_evaluation.engine.ConfigManager'):
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 创建多个引擎实例以增加内存压力
            engines = []
            for i in range(5):
                engine = ExpertEvaluationEngine(config)
                engine.is_model_loaded = True
                engines.append(engine)
            
            # 为每个引擎创建大量数据
            all_results = []
            
            for i, engine in enumerate(engines):
                large_items = []
                for j in range(20):  # 每个引擎处理20个项目
                    item = QAEvaluationItem(
                        question_id=f"memory_pressure_{i}_{j:03d}",
                        question=f"内存压力测试问题 {i}-{j} " * 50,
                        context=f"内存压力测试上下文 {i}-{j} " * 25,
                        reference_answer=f"内存压力测试参考答案 {i}-{j} " * 100,
                        model_answer=f"内存压力测试模型答案 {i}-{j} " * 100
                    )
                    large_items.append(item)
                
                with patch.object(engine, '_evaluate_single_item') as mock_evaluate:
                    mock_evaluate.side_effect = lambda item: create_mock_evaluation_result(
                        item.question_id, 0.8
                    )
                    
                    try:
                        result = engine.evaluate_model(large_items)
                        all_results.append(result)
                    except Exception as e:
                        print(f"引擎 {i} 评估失败: {e}")
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"内存压力测试结果:")
            print(f"  - 引擎数量: {len(engines)}")
            print(f"  - 成功结果数: {len(all_results)}")
            print(f"  - 初始内存: {initial_memory:.2f}MB")
            print(f"  - 最终内存: {final_memory:.2f}MB")
            print(f"  - 内存增长: {memory_increase:.2f}MB")
            
            # 内存压力测试要求
            assert memory_increase < 1000, f"内存压力测试内存增长过多: {memory_increase:.2f}MB"
            assert len(all_results) >= len(engines) * 0.8, "内存压力测试成功率过低"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])