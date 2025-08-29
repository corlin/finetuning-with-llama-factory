#!/usr/bin/env python3
"""
专家评估系统高级场景示例

本脚本展示了专家评估系统的高级使用场景，包括：
1. 自定义评估维度
2. 多模型对比评估
3. 大规模批量处理
4. 实时评估监控
5. 结果分析和可视化

使用方法:
    uv run python examples/expert_evaluation_advanced_scenarios.py

作者: 专家评估系统开发团队
"""

import json
import time
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import ExpertEvaluationConfig
from src.expert_evaluation.multi_dimensional import MultiDimensionalEvaluator
from src.expert_evaluation.performance import PerformanceMonitor

class AdvancedEvaluationScenarios:
    """高级评估场景演示类"""
    
    def __init__(self):
        self.output_dir = Path("advanced_demo_output")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🚀 专家评估系统高级场景演示")
        print("=" * 60)
    
    def scenario_1_custom_dimensions(self):
        """场景1: 自定义评估维度"""
        print("\n🎯 场景1: 自定义评估维度")
        print("-" * 40)
        
        # 创建自定义评估配置
        custom_config = {
            "evaluation": {
                "dimensions": [
                    "semantic_similarity",
                    "domain_accuracy", 
                    "creativity_score",
                    "technical_depth",
                    "practical_applicability"
                ],
                "weights": {
                    "semantic_similarity": 0.20,
                    "domain_accuracy": 0.25,
                    "creativity_score": 0.20,
                    "technical_depth": 0.20,
                    "practical_applicability": 0.15
                },
                "algorithms": {
                    "creativity_score": {
                        "method": "novelty_detection",
                        "baseline_size": 50,
                        "diversity_weight": 0.4
                    },
                    "technical_depth": {
                        "method": "concept_complexity",
                        "min_concepts": 3,
                        "depth_threshold": 0.7
                    }
                }
            }
        }
        
        config = ExpertEvaluationConfig.from_dict(custom_config)
        engine = ExpertEvaluationEngine(config)
        
        # 测试数据
        qa_item = {
            "question_id": "custom_001",
            "question": "设计一个基于区块链的去中心化身份验证系统",
            "reference_answer": "该系统应包含分布式身份标识符(DID)、可验证凭证(VC)、智能合约验证机制等核心组件...",
            "model_answer": "可以使用以太坊智能合约创建一个身份注册系统，用户通过私钥签名证明身份，结合IPFS存储身份信息...",
            "domain_tags": ["区块链", "身份验证", "去中心化"],
            "difficulty_level": "expert",
            "expected_concepts": ["DID", "智能合约", "密码学", "分布式系统"]
        }
        
        result = engine.evaluate_single_qa(qa_item)
        
        print("📊 自定义维度评估结果:")
        for dimension, score in result.dimension_scores.items():
            print(f"   - {dimension}: {score:.3f}")
        
        # 保存结果
        with open(self.output_dir / "custom_dimensions_result.json", 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print("✅ 自定义维度评估完成")
        return result
    
    def scenario_2_model_comparison(self):
        """场景2: 多模型对比评估"""
        print("\n🔄 场景2: 多模型对比评估")
        print("-" * 40)
        
        # 模拟不同模型的配置
        model_configs = {
            "model_a": {
                "name": "模型A (基础版)",
                "config": {
                    "evaluation": {
                        "weights": {
                            "semantic_similarity": 0.4,
                            "domain_accuracy": 0.6
                        }
                    }
                }
            },
            "model_b": {
                "name": "模型B (增强版)",
                "config": {
                    "evaluation": {
                        "weights": {
                            "semantic_similarity": 0.3,
                            "domain_accuracy": 0.4,
                            "innovation": 0.3
                        }
                    }
                }
            },
            "model_c": {
                "name": "模型C (专业版)",
                "config": {
                    "evaluation": {
                        "weights": {
                            "semantic_similarity": 0.25,
                            "domain_accuracy": 0.25,
                            "innovation": 0.25,
                            "practical_value": 0.25
                        }
                    }
                }
            }
        }
        
        # 测试数据集
        test_qa_items = [
            {
                "question_id": "comp_001",
                "question": "解释深度学习中的注意力机制",
                "reference_answer": "注意力机制允许模型在处理序列数据时动态地关注输入的不同部分...",
                "model_answer": "注意力机制是深度学习中的重要技术，它帮助模型识别输入中的关键信息...",
                "domain_tags": ["深度学习", "注意力机制"]
            },
            {
                "question_id": "comp_002", 
                "question": "什么是联邦学习？它解决了什么问题？",
                "reference_answer": "联邦学习是一种分布式机器学习方法，允许多个参与方在不共享原始数据的情况下协作训练模型...",
                "model_answer": "联邦学习让不同的设备可以一起训练AI模型，同时保护用户隐私...",
                "domain_tags": ["联邦学习", "隐私保护"]
            }
        ]
        
        comparison_results = {}
        
        # 对每个模型进行评估
        for model_id, model_info in model_configs.items():
            print(f"🔍 评估 {model_info['name']}...")
            
            config = ExpertEvaluationConfig.from_dict(model_info['config'])
            engine = ExpertEvaluationEngine(config)
            
            model_results = []
            for qa_item in test_qa_items:
                result = engine.evaluate_single_qa(qa_item)
                model_results.append(result)
            
            # 计算平均得分
            avg_score = sum(r.overall_score for r in model_results) / len(model_results)
            comparison_results[model_id] = {
                "name": model_info['name'],
                "average_score": avg_score,
                "individual_results": [r.to_dict() for r in model_results]
            }
            
            print(f"   平均得分: {avg_score:.3f}")
        
        # 生成对比报告
        print("\n📊 模型对比结果:")
        sorted_models = sorted(comparison_results.items(), 
                             key=lambda x: x[1]['average_score'], 
                             reverse=True)
        
        for i, (model_id, result) in enumerate(sorted_models, 1):
            print(f"   {i}. {result['name']}: {result['average_score']:.3f}")
        
        # 保存对比结果
        with open(self.output_dir / "model_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        # 生成对比图表
        self._generate_comparison_chart(comparison_results)
        
        print("✅ 多模型对比评估完成")
        return comparison_results
    
    def scenario_3_large_scale_processing(self):
        """场景3: 大规模批量处理"""
        print("\n📦 场景3: 大规模批量处理")
        print("-" * 40)
        
        # 生成大规模测试数据
        large_dataset = []
        for i in range(100):  # 生成100个QA项
            qa_item = {
                "question_id": f"large_scale_{i:03d}",
                "question": f"这是第{i+1}个测试问题，请详细回答相关技术原理。",
                "reference_answer": f"这是第{i+1}个参考答案，包含详细的技术解释和实现方案。",
                "model_answer": f"这是第{i+1}个模型答案，提供了相应的技术分析。",
                "domain_tags": ["技术", "测试"],
                "difficulty_level": "intermediate"
            }
            large_dataset.append(qa_item)
        
        print(f"📊 生成测试数据: {len(large_dataset)} 个QA项")
        
        # 配置批量处理
        batch_config = {
            "performance": {
                "max_workers": 4,
                "batch_size": 10,
                "timeout": 300
            },
            "evaluation": {
                "dimensions": ["semantic_similarity", "domain_accuracy"]
            }
        }
        
        config = ExpertEvaluationConfig.from_dict(batch_config)
        engine = ExpertEvaluationEngine(config)
        
        # 执行大规模批量评估
        print("⏳ 开始大规模批量评估...")
        start_time = time.time()
        
        batch_result = engine.evaluate_batch(large_dataset)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ 批量评估完成")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"🚀 处理速度: {len(large_dataset)/total_time:.1f} QA项/秒")
        print(f"📊 平均得分: {batch_result.average_overall_score:.3f}")
        
        # 分析处理性能
        performance_stats = {
            "total_items": len(large_dataset),
            "total_time": total_time,
            "throughput": len(large_dataset) / total_time,
            "average_score": batch_result.average_overall_score,
            "score_distribution": self._calculate_score_distribution(batch_result)
        }
        
        # 保存性能统计
        with open(self.output_dir / "large_scale_performance.json", 'w', encoding='utf-8') as f:
            json.dump(performance_stats, f, indent=2, ensure_ascii=False)
        
        print("✅ 大规模批量处理完成")
        return performance_stats
    
    def scenario_4_realtime_monitoring(self):
        """场景4: 实时评估监控"""
        print("\n📡 场景4: 实时评估监控")
        print("-" * 40)
        
        # 初始化性能监控器
        monitor = PerformanceMonitor()
        
        # 模拟实时评估流
        config = ExpertEvaluationConfig()
        engine = ExpertEvaluationEngine(config)
        
        print("🔄 开始实时评估监控 (10秒演示)...")
        
        monitoring_data = []
        start_time = time.time()
        
        # 模拟10秒的实时评估
        while time.time() - start_time < 10:
            # 生成随机QA项
            qa_item = {
                "question_id": f"realtime_{int(time.time())}",
                "question": "实时评估测试问题",
                "reference_answer": "实时评估参考答案",
                "model_answer": "实时评估模型答案",
                "domain_tags": ["实时测试"]
            }
            
            # 执行评估并监控
            eval_start = time.time()
            result = engine.evaluate_single_qa(qa_item)
            eval_time = time.time() - eval_start
            
            # 收集监控数据
            monitoring_data.append({
                "timestamp": time.time(),
                "evaluation_time": eval_time,
                "score": result.overall_score,
                "memory_usage": monitor.get_memory_usage(),
                "cpu_usage": monitor.get_cpu_usage()
            })
            
            print(f"⚡ 评估完成 - 得分: {result.overall_score:.3f}, 耗时: {eval_time:.3f}秒")
            
            time.sleep(1)  # 每秒一次评估
        
        print("✅ 实时监控演示完成")
        
        # 分析监控数据
        avg_eval_time = sum(d['evaluation_time'] for d in monitoring_data) / len(monitoring_data)
        avg_score = sum(d['score'] for d in monitoring_data) / len(monitoring_data)
        avg_memory = sum(d['memory_usage'] for d in monitoring_data) / len(monitoring_data)
        avg_cpu = sum(d['cpu_usage'] for d in monitoring_data) / len(monitoring_data)
        
        print(f"\n📊 监控统计:")
        print(f"   - 平均评估时间: {avg_eval_time:.3f}秒")
        print(f"   - 平均得分: {avg_score:.3f}")
        print(f"   - 平均内存使用: {avg_memory:.1f}MB")
        print(f"   - 平均CPU使用: {avg_cpu:.1f}%")
        
        # 保存监控数据
        with open(self.output_dir / "realtime_monitoring.json", 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False)
        
        return monitoring_data
    
    def scenario_5_result_visualization(self):
        """场景5: 结果分析和可视化"""
        print("\n📈 场景5: 结果分析和可视化")
        print("-" * 40)
        
        # 生成多维度评估数据
        evaluation_data = []
        dimensions = ["semantic_similarity", "domain_accuracy", "innovation", "practical_value"]
        
        for i in range(50):
            scores = {}
            for dim in dimensions:
                # 生成模拟得分 (添加一些随机性)
                base_score = 0.7 + (i % 10) * 0.03
                noise = (hash(f"{i}_{dim}") % 100) / 1000  # -0.05 到 0.05 的噪声
                scores[dim] = max(0, min(1, base_score + noise))
            
            overall_score = sum(scores.values()) / len(scores)
            
            evaluation_data.append({
                "item_id": f"viz_{i:03d}",
                "overall_score": overall_score,
                "dimension_scores": scores
            })
        
        # 创建数据分析
        df = pd.DataFrame(evaluation_data)
        
        # 展开维度得分
        for dim in dimensions:
            df[dim] = df['dimension_scores'].apply(lambda x: x[dim])
        
        print("📊 生成可视化图表...")
        
        # 1. 得分分布直方图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['overall_score'], bins=20, alpha=0.7, color='skyblue')
        plt.title('总体得分分布')
        plt.xlabel('得分')
        plt.ylabel('频次')
        
        # 2. 维度得分箱线图
        plt.subplot(2, 2, 2)
        dimension_data = [df[dim] for dim in dimensions]
        plt.boxplot(dimension_data, labels=dimensions)
        plt.title('各维度得分分布')
        plt.xticks(rotation=45)
        
        # 3. 得分趋势图
        plt.subplot(2, 2, 3)
        plt.plot(df.index, df['overall_score'], marker='o', markersize=3)
        plt.title('得分趋势')
        plt.xlabel('样本序号')
        plt.ylabel('总体得分')
        
        # 4. 维度相关性热力图
        plt.subplot(2, 2, 4)
        correlation_matrix = df[dimensions].corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('维度相关性')
        plt.xticks(range(len(dimensions)), dimensions, rotation=45)
        plt.yticks(range(len(dimensions)), dimensions)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成统计报告
        stats_report = {
            "summary": {
                "total_samples": len(evaluation_data),
                "mean_score": float(df['overall_score'].mean()),
                "std_score": float(df['overall_score'].std()),
                "min_score": float(df['overall_score'].min()),
                "max_score": float(df['overall_score'].max())
            },
            "dimension_stats": {}
        }
        
        for dim in dimensions:
            stats_report["dimension_stats"][dim] = {
                "mean": float(df[dim].mean()),
                "std": float(df[dim].std()),
                "min": float(df[dim].min()),
                "max": float(df[dim].max())
            }
        
        # 保存统计报告
        with open(self.output_dir / "visualization_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats_report, f, indent=2, ensure_ascii=False)
        
        print("✅ 可视化分析完成")
        print(f"📈 图表已保存: {self.output_dir / 'evaluation_analysis.png'}")
        
        return stats_report
    
    def _generate_comparison_chart(self, comparison_results):
        """生成模型对比图表"""
        models = list(comparison_results.keys())
        scores = [comparison_results[model]['average_score'] for model in models]
        names = [comparison_results[model]['name'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title('模型对比评估结果')
        plt.ylabel('平均得分')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_score_distribution(self, batch_result):
        """计算得分分布"""
        scores = [result.overall_score for result in batch_result.individual_results]
        
        ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.0, 0.6)]
        distribution = {}
        
        for min_score, max_score in ranges:
            count = sum(1 for score in scores if min_score <= score < max_score)
            distribution[f"{min_score}-{max_score}"] = count
        
        return distribution
    
    def run_all_scenarios(self):
        """运行所有高级场景"""
        scenarios = [
            ("自定义评估维度", self.scenario_1_custom_dimensions),
            ("多模型对比评估", self.scenario_2_model_comparison),
            ("大规模批量处理", self.scenario_3_large_scale_processing),
            ("实时评估监控", self.scenario_4_realtime_monitoring),
            ("结果分析可视化", self.scenario_5_result_visualization)
        ]
        
        results = {}
        
        for scenario_name, scenario_func in scenarios:
            try:
                print(f"\n🎯 开始执行: {scenario_name}")
                result = scenario_func()
                results[scenario_name] = result
                print(f"✅ {scenario_name} 完成")
            except Exception as e:
                print(f"❌ {scenario_name} 失败: {e}")
                results[scenario_name] = {"error": str(e)}
        
        # 生成总结报告
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_scenarios": len([r for r in results.values() if "error" not in r]),
            "total_scenarios": len(scenarios),
            "output_directory": str(self.output_dir),
            "scenario_results": results
        }
        
        with open(self.output_dir / "advanced_scenarios_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 高级场景演示完成!")
        print(f"📊 完成场景: {summary['completed_scenarios']}/{summary['total_scenarios']}")
        print(f"📁 输出目录: {self.output_dir}")

def main():
    """主函数"""
    try:
        demo = AdvancedEvaluationScenarios()
        demo.run_all_scenarios()
    except Exception as e:
        print(f"❌ 高级场景演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()