#!/usr/bin/env python3
"""
专家评估系统性能基准测试

本脚本提供全面的性能基准测试，包括：
1. 单项评估性能测试
2. 批量处理性能测试
3. 内存使用分析
4. 并发性能测试
5. 不同配置下的性能对比

使用方法:
    uv run python examples/expert_evaluation_performance_benchmark.py

作者: 专家评估系统开发团队
"""

import json
import time
import sys
import threading
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import ExpertEvaluationConfig

class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.output_dir = Path("benchmark_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试数据
        self.test_qa_items = self._generate_test_data()
        
        print("🚀 专家评估系统性能基准测试")
        print("=" * 60)
        print(f"📊 测试数据: {len(self.test_qa_items)} 个QA项")
        print(f"📁 输出目录: {self.output_dir}")
    
    def _generate_test_data(self) -> List[Dict[str, Any]]:
        """生成测试数据"""
        test_data = []
        
        # 不同复杂度的测试用例
        test_cases = [
            {
                "complexity": "simple",
                "question": "什么是Python？",
                "reference_answer": "Python是一种高级编程语言。",
                "model_answer": "Python是一种解释型编程语言。"
            },
            {
                "complexity": "medium",
                "question": "解释面向对象编程的三大特性",
                "reference_answer": "面向对象编程的三大特性是封装、继承和多态。封装是将数据和方法组合在一起，继承允许子类获得父类的属性和方法，多态允许不同类的对象对同一消息做出不同的响应。",
                "model_answer": "OOP的三个主要特性包括：1）封装-隐藏内部实现细节；2）继承-代码重用机制；3）多态-同一接口的不同实现。"
            },
            {
                "complexity": "complex",
                "question": "详细说明分布式系统中的CAP定理，并分析在实际系统设计中如何权衡一致性、可用性和分区容错性",
                "reference_answer": "CAP定理指出，在分布式系统中，一致性(Consistency)、可用性(Availability)和分区容错性(Partition tolerance)三者不能同时满足，最多只能同时保证其中两个。一致性要求所有节点在同一时间看到相同的数据；可用性要求系统在合理时间内返回合理的响应；分区容错性要求系统在网络分区故障时仍能继续运行。在实际设计中，由于网络分区是不可避免的，通常需要在CP和AP之间选择。",
                "model_answer": "CAP定理是分布式系统的重要理论，说明了一致性、可用性、分区容错性三者的权衡关系。在网络分区发生时，系统必须在一致性和可用性之间做出选择。例如，银行系统通常选择一致性，而社交媒体系统可能更注重可用性。"
            }
        ]
        
        # 为每种复杂度生成多个测试用例
        for i, case in enumerate(test_cases):
            for j in range(10):  # 每种复杂度10个用例
                test_data.append({
                    "question_id": f"{case['complexity']}_{j:03d}",
                    "question": case["question"],
                    "reference_answer": case["reference_answer"],
                    "model_answer": case["model_answer"],
                    "domain_tags": ["测试", case["complexity"]],
                    "difficulty_level": case["complexity"],
                    "complexity": case["complexity"]
                })
        
        return test_data
    
    def benchmark_1_single_evaluation_performance(self):
        """基准测试1: 单项评估性能"""
        print("\n⚡ 基准测试1: 单项评估性能")
        print("-" * 40)
        
        # 不同配置的性能测试
        configs = {
            "minimal": {
                "evaluation": {
                    "dimensions": ["semantic_similarity"],
                    "weights": {"semantic_similarity": 1.0}
                }
            },
            "standard": {
                "evaluation": {
                    "dimensions": ["semantic_similarity", "domain_accuracy"],
                    "weights": {"semantic_similarity": 0.6, "domain_accuracy": 0.4}
                }
            },
            "comprehensive": {
                "evaluation": {
                    "dimensions": [
                        "semantic_similarity", "domain_accuracy", 
                        "response_relevance", "factual_correctness", "completeness"
                    ],
                    "weights": {
                        "semantic_similarity": 0.25,
                        "domain_accuracy": 0.25,
                        "response_relevance": 0.20,
                        "factual_correctness": 0.15,
                        "completeness": 0.15
                    }
                }
            }
        }
        
        results = {}
        
        for config_name, config_dict in configs.items():
            print(f"🔍 测试配置: {config_name}")
            
            config = ExpertEvaluationConfig.from_dict(config_dict)
            engine = ExpertEvaluationEngine(config)
            
            # 预热
            engine.evaluate_single_qa(self.test_qa_items[0])
            
            # 性能测试
            times = []
            memory_usage = []
            
            for qa_item in self.test_qa_items[:10]:  # 测试前10个项目
                # 记录内存使用
                gc.collect()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 执行评估
                start_time = time.time()
                result = engine.evaluate_single_qa(qa_item)
                end_time = time.time()
                
                # 记录结果
                evaluation_time = end_time - start_time
                times.append(evaluation_time)
                
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage.append(memory_after - memory_before)
            
            # 计算统计信息
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            results[config_name] = {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "average_memory_delta": avg_memory,
                "throughput": 1 / avg_time,
                "dimension_count": len(config_dict["evaluation"]["dimensions"])
            }
            
            print(f"   平均耗时: {avg_time:.3f}秒")
            print(f"   吞吐量: {1/avg_time:.1f} QA项/秒")
            print(f"   内存增量: {avg_memory:.1f}MB")
        
        # 保存结果
        with open(self.output_dir / "single_evaluation_benchmark.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成性能对比图
        self._plot_single_evaluation_performance(results)
        
        print("✅ 单项评估性能测试完成")
        return results
    
    def benchmark_2_batch_processing_performance(self):
        """基准测试2: 批量处理性能"""
        print("\n📦 基准测试2: 批量处理性能")
        print("-" * 40)
        
        # 不同批量大小的测试
        batch_sizes = [1, 5, 10, 20, 30]
        config = ExpertEvaluationConfig()
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"🔍 测试批量大小: {batch_size}")
            
            # 配置批量处理
            config.batch_size = batch_size
            engine = ExpertEvaluationEngine(config)
            
            # 选择测试数据
            test_data = self.test_qa_items[:batch_size]
            
            # 预热
            if len(test_data) > 0:
                engine.evaluate_single_qa(test_data[0])
            
            # 性能测试
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            batch_result = engine.evaluate_batch(test_data)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            # 计算指标
            total_time = end_time - start_time
            throughput = len(test_data) / total_time
            memory_delta = memory_after - memory_before
            avg_time_per_item = total_time / len(test_data)
            
            results[batch_size] = {
                "batch_size": batch_size,
                "total_time": total_time,
                "throughput": throughput,
                "memory_delta": memory_delta,
                "avg_time_per_item": avg_time_per_item,
                "average_score": batch_result.average_overall_score
            }
            
            print(f"   总耗时: {total_time:.2f}秒")
            print(f"   吞吐量: {throughput:.1f} QA项/秒")
            print(f"   平均每项: {avg_time_per_item:.3f}秒")
            print(f"   内存增量: {memory_delta:.1f}MB")
        
        # 保存结果
        with open(self.output_dir / "batch_processing_benchmark.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成批量处理性能图
        self._plot_batch_processing_performance(results)
        
        print("✅ 批量处理性能测试完成")
        return results
    
    def benchmark_3_memory_analysis(self):
        """基准测试3: 内存使用分析"""
        print("\n💾 基准测试3: 内存使用分析")
        print("-" * 40)
        
        config = ExpertEvaluationConfig()
        engine = ExpertEvaluationEngine(config)
        
        memory_timeline = []
        evaluation_count = 0
        
        # 监控内存使用
        def memory_monitor():
            while evaluation_count < 50:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_timeline.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_mb,
                    "evaluation_count": evaluation_count
                })
                time.sleep(0.1)  # 每100ms记录一次
        
        # 启动内存监控线程
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        # 执行评估任务
        print("🔍 执行50次评估并监控内存使用...")
        
        for i in range(50):
            qa_item = self.test_qa_items[i % len(self.test_qa_items)]
            result = engine.evaluate_single_qa(qa_item)
            evaluation_count = i + 1
            
            if (i + 1) % 10 == 0:
                print(f"   完成 {i + 1}/50 次评估")
        
        # 等待监控线程结束
        monitor_thread.join()
        
        # 分析内存使用
        initial_memory = memory_timeline[0]["memory_mb"]
        peak_memory = max(point["memory_mb"] for point in memory_timeline)
        final_memory = memory_timeline[-1]["memory_mb"]
        
        memory_analysis = {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": final_memory - initial_memory,
            "peak_memory_increase_mb": peak_memory - initial_memory,
            "timeline": memory_timeline
        }
        
        print(f"📊 内存分析结果:")
        print(f"   初始内存: {initial_memory:.1f}MB")
        print(f"   峰值内存: {peak_memory:.1f}MB")
        print(f"   最终内存: {final_memory:.1f}MB")
        print(f"   内存增长: {final_memory - initial_memory:.1f}MB")
        print(f"   峰值增长: {peak_memory - initial_memory:.1f}MB")
        
        # 保存内存分析结果
        with open(self.output_dir / "memory_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(memory_analysis, f, indent=2, ensure_ascii=False)
        
        # 生成内存使用图
        self._plot_memory_usage(memory_timeline)
        
        print("✅ 内存使用分析完成")
        return memory_analysis
    
    def benchmark_4_concurrent_performance(self):
        """基准测试4: 并发性能测试"""
        print("\n🔄 基准测试4: 并发性能测试")
        print("-" * 40)
        
        # 不同并发级别的测试
        thread_counts = [1, 2, 4, 8]
        test_data = self.test_qa_items[:20]  # 使用20个测试项目
        
        results = {}
        
        for thread_count in thread_counts:
            print(f"🔍 测试并发数: {thread_count}")
            
            config = ExpertEvaluationConfig()
            config.max_workers = thread_count
            
            # 并发评估函数
            def evaluate_qa_item(qa_item):
                engine = ExpertEvaluationEngine(config)
                start_time = time.time()
                result = engine.evaluate_single_qa(qa_item)
                end_time = time.time()
                return {
                    "evaluation_time": end_time - start_time,
                    "score": result.overall_score
                }
            
            # 执行并发测试
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(evaluate_qa_item, qa_item) for qa_item in test_data]
                concurrent_results = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            
            # 计算并发性能指标
            total_time = end_time - start_time
            avg_evaluation_time = sum(r["evaluation_time"] for r in concurrent_results) / len(concurrent_results)
            throughput = len(test_data) / total_time
            
            results[thread_count] = {
                "thread_count": thread_count,
                "total_time": total_time,
                "avg_evaluation_time": avg_evaluation_time,
                "throughput": throughput,
                "efficiency": throughput / thread_count,  # 每线程吞吐量
                "individual_results": concurrent_results
            }
            
            print(f"   总耗时: {total_time:.2f}秒")
            print(f"   吞吐量: {throughput:.1f} QA项/秒")
            print(f"   效率: {throughput/thread_count:.1f} QA项/秒/线程")
        
        # 保存并发测试结果
        with open(self.output_dir / "concurrent_performance.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成并发性能图
        self._plot_concurrent_performance(results)
        
        print("✅ 并发性能测试完成")
        return results
    
    def benchmark_5_configuration_comparison(self):
        """基准测试5: 不同配置下的性能对比"""
        print("\n⚙️ 基准测试5: 配置性能对比")
        print("-" * 40)
        
        # 不同配置方案
        configurations = {
            "fast": {
                "name": "快速配置",
                "config": {
                    "evaluation": {
                        "dimensions": ["semantic_similarity"],
                        "algorithms": {
                            "semantic_similarity": {"method": "fast_cosine"}
                        }
                    },
                    "performance": {
                        "max_workers": 1,
                        "cache_size": "256MB"
                    }
                }
            },
            "balanced": {
                "name": "平衡配置",
                "config": {
                    "evaluation": {
                        "dimensions": ["semantic_similarity", "domain_accuracy"],
                        "weights": {"semantic_similarity": 0.6, "domain_accuracy": 0.4}
                    },
                    "performance": {
                        "max_workers": 2,
                        "cache_size": "512MB"
                    }
                }
            },
            "accurate": {
                "name": "高精度配置",
                "config": {
                    "evaluation": {
                        "dimensions": [
                            "semantic_similarity", "domain_accuracy", 
                            "response_relevance", "factual_correctness"
                        ],
                        "algorithms": {
                            "semantic_similarity": {"method": "bert_score"}
                        }
                    },
                    "performance": {
                        "max_workers": 4,
                        "cache_size": "1GB"
                    }
                }
            }
        }
        
        test_data = self.test_qa_items[:15]  # 使用15个测试项目
        results = {}
        
        for config_id, config_info in configurations.items():
            print(f"🔍 测试配置: {config_info['name']}")
            
            config = ExpertEvaluationConfig.from_dict(config_info['config'])
            engine = ExpertEvaluationEngine(config)
            
            # 预热
            engine.evaluate_single_qa(test_data[0])
            
            # 性能测试
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            batch_result = engine.evaluate_batch(test_data)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            # 计算性能指标
            total_time = end_time - start_time
            throughput = len(test_data) / total_time
            memory_delta = memory_after - memory_before
            
            results[config_id] = {
                "name": config_info['name'],
                "total_time": total_time,
                "throughput": throughput,
                "memory_delta": memory_delta,
                "average_score": batch_result.average_overall_score,
                "dimension_count": len(config_info['config']['evaluation']['dimensions']),
                "performance_score": self._calculate_performance_score(total_time, throughput, memory_delta)
            }
            
            print(f"   耗时: {total_time:.2f}秒")
            print(f"   吞吐量: {throughput:.1f} QA项/秒")
            print(f"   内存: {memory_delta:.1f}MB")
            print(f"   平均得分: {batch_result.average_overall_score:.3f}")
        
        # 保存配置对比结果
        with open(self.output_dir / "configuration_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成配置对比图
        self._plot_configuration_comparison(results)
        
        print("✅ 配置性能对比完成")
        return results
    
    def _calculate_performance_score(self, total_time: float, throughput: float, memory_delta: float) -> float:
        """计算综合性能得分"""
        # 时间得分 (越快越好)
        time_score = max(0, 100 - total_time * 10)
        
        # 吞吐量得分 (越高越好)
        throughput_score = min(100, throughput * 20)
        
        # 内存得分 (越少越好)
        memory_score = max(0, 100 - memory_delta * 2)
        
        # 综合得分
        return (time_score + throughput_score + memory_score) / 3
    
    def _plot_single_evaluation_performance(self, results):
        """绘制单项评估性能图"""
        configs = list(results.keys())
        times = [results[config]["average_time"] for config in configs]
        throughputs = [results[config]["throughput"] for config in configs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 评估时间对比
        ax1.bar(configs, times, color='skyblue')
        ax1.set_title('平均评估时间对比')
        ax1.set_ylabel('时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 吞吐量对比
        ax2.bar(configs, throughputs, color='lightgreen')
        ax2.set_title('吞吐量对比')
        ax2.set_ylabel('QA项/秒')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "single_evaluation_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_processing_performance(self, results):
        """绘制批量处理性能图"""
        batch_sizes = list(results.keys())
        throughputs = [results[size]["throughput"] for size in batch_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, throughputs, marker='o', linewidth=2, markersize=8)
        plt.title('批量处理性能 - 吞吐量 vs 批量大小')
        plt.xlabel('批量大小')
        plt.ylabel('吞吐量 (QA项/秒)')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "batch_processing_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, memory_timeline):
        """绘制内存使用图"""
        timestamps = [point["timestamp"] for point in memory_timeline]
        memory_values = [point["memory_mb"] for point in memory_timeline]
        
        # 转换为相对时间
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        plt.figure(figsize=(12, 6))
        plt.plot(relative_times, memory_values, linewidth=2)
        plt.title('内存使用时间线')
        plt.xlabel('时间 (秒)')
        plt.ylabel('内存使用 (MB)')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_concurrent_performance(self, results):
        """绘制并发性能图"""
        thread_counts = list(results.keys())
        throughputs = [results[count]["throughput"] for count in thread_counts]
        efficiencies = [results[count]["efficiency"] for count in thread_counts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 总吞吐量
        ax1.plot(thread_counts, throughputs, marker='o', linewidth=2, markersize=8, color='blue')
        ax1.set_title('并发吞吐量')
        ax1.set_xlabel('线程数')
        ax1.set_ylabel('总吞吐量 (QA项/秒)')
        ax1.grid(True, alpha=0.3)
        
        # 每线程效率
        ax2.plot(thread_counts, efficiencies, marker='s', linewidth=2, markersize=8, color='red')
        ax2.set_title('每线程效率')
        ax2.set_xlabel('线程数')
        ax2.set_ylabel('效率 (QA项/秒/线程)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "concurrent_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_configuration_comparison(self, results):
        """绘制配置对比图"""
        configs = list(results.keys())
        names = [results[config]["name"] for config in configs]
        throughputs = [results[config]["throughput"] for config in configs]
        scores = [results[config]["average_score"] for config in configs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 性能对比
        ax1.bar(names, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('配置性能对比')
        ax1.set_ylabel('吞吐量 (QA项/秒)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 准确性对比
        ax2.bar(names, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('配置准确性对比')
        ax2.set_ylabel('平均得分')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "configuration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        benchmarks = [
            ("单项评估性能", self.benchmark_1_single_evaluation_performance),
            ("批量处理性能", self.benchmark_2_batch_processing_performance),
            ("内存使用分析", self.benchmark_3_memory_analysis),
            ("并发性能测试", self.benchmark_4_concurrent_performance),
            ("配置性能对比", self.benchmark_5_configuration_comparison)
        ]
        
        all_results = {}
        
        for benchmark_name, benchmark_func in benchmarks:
            try:
                print(f"\n🎯 开始执行: {benchmark_name}")
                result = benchmark_func()
                all_results[benchmark_name] = result
                print(f"✅ {benchmark_name} 完成")
            except Exception as e:
                print(f"❌ {benchmark_name} 失败: {e}")
                all_results[benchmark_name] = {"error": str(e)}
        
        # 生成综合报告
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "python_version": sys.version
            },
            "test_data_size": len(self.test_qa_items),
            "completed_benchmarks": len([r for r in all_results.values() if "error" not in r]),
            "total_benchmarks": len(benchmarks),
            "benchmark_results": all_results
        }
        
        with open(self.output_dir / "benchmark_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 性能基准测试完成!")
        print(f"📊 完成测试: {summary['completed_benchmarks']}/{summary['total_benchmarks']}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"💾 综合报告: {self.output_dir / 'benchmark_summary.json'}")

def main():
    """主函数"""
    try:
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()