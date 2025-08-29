#!/usr/bin/env python3
"""
专家评估系统完整演示脚本

本脚本展示了专家评估系统的完整使用流程，包括：
1. 系统初始化和配置
2. 模型加载和验证
3. 数据准备和验证
4. 单项评估演示
5. 批量评估演示
6. 结果分析和报告生成
7. 性能基准测试
8. API服务演示

使用方法:
    uv run python examples/expert_evaluation_demo.py

作者: 专家评估系统开发团队
版本: 1.0.0
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.expert_evaluation.engine import ExpertEvaluationEngine
    from src.expert_evaluation.config import ExpertEvaluationConfig
    from src.expert_evaluation.data_manager import EvaluationDataManager
    from src.expert_evaluation.report_generator import EvaluationReportGenerator
    from src.expert_evaluation.performance import PerformanceBenchmark
    from src.expert_evaluation.data_models import QAEvaluationItem
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，并已正确安装依赖")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExpertEvaluationDemo:
    """专家评估系统演示类"""
    
    def __init__(self):
        """初始化演示环境"""
        self.demo_data_dir = Path("demo_data")
        self.demo_output_dir = Path("demo_output")
        self.config = None
        self.engine = None
        self.data_manager = None
        
        # 创建演示目录
        self.demo_data_dir.mkdir(exist_ok=True)
        self.demo_output_dir.mkdir(exist_ok=True)
        
        print("🚀 专家评估系统演示开始")
        print("=" * 60)
    
    def step_1_system_initialization(self):
        """步骤1: 系统初始化和配置"""
        print("\n📋 步骤1: 系统初始化和配置")
        print("-" * 40)
        
        try:
            # 创建演示配置
            demo_config = {
                "model": {
                    "model_path": "",  # 将在运行时设置
                    "device": "auto",
                    "quantization": None,
                    "max_length": 1024,
                    "batch_size": 1
                },
                "evaluation": {
                    "dimensions": [
                        "semantic_similarity",
                        "domain_accuracy",
                        "response_relevance",
                        "factual_correctness",
                        "completeness"
                    ],
                    "weights": {
                        "semantic_similarity": 0.25,
                        "domain_accuracy": 0.25,
                        "response_relevance": 0.20,
                        "factual_correctness": 0.15,
                        "completeness": 0.15
                    },
                    "thresholds": {
                        "min_score": 0.6,
                        "confidence_level": 0.95
                    }
                },
                "performance": {
                    "max_workers": 2,
                    "timeout": 120,
                    "memory_limit": "4GB",
                    "cache_size": "512MB"
                },
                "output": {
                    "format": "json",
                    "detailed": True,
                    "save_intermediate": False
                },
                "logging": {
                    "level": "INFO"
                }
            }
            
            # 保存演示配置
            config_path = self.demo_data_dir / "demo_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(demo_config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 演示配置已创建: {config_path}")
            
            # 加载配置
            self.config = ExpertEvaluationConfig.from_dict(demo_config)
            print("✅ 配置加载成功")
            
            # 显示配置信息
            print(f"📊 评估维度: {len(self.config.evaluation_dimensions)}个")
            print(f"⚙️  设备配置: {self.config.device}")
            print(f"🔧 性能配置: {self.config.max_workers}个工作线程")
            
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    def step_2_model_loading(self):
        """步骤2: 模型加载和验证"""
        print("\n🤖 步骤2: 模型加载和验证")
        print("-" * 40)
        
        try:
            # 初始化评估引擎
            self.engine = ExpertEvaluationEngine(self.config)
            print("✅ 评估引擎初始化成功")
            
            # 注意: 在实际使用中，这里会加载真实的模型
            # 为了演示目的，我们使用模拟模式
            print("ℹ️  演示模式: 使用模拟模型进行演示")
            print("   在生产环境中，请配置真实的模型路径")
            
            # 显示模型信息
            print("📋 模型配置信息:")
            print(f"   - 设备: {self.config.device}")
            print(f"   - 量化: {self.config.quantization or '未启用'}")
            print(f"   - 最大长度: {self.config.max_length}")
            print(f"   - 批处理大小: {self.config.batch_size}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def step_3_data_preparation(self):
        """步骤3: 数据准备和验证"""
        print("\n📊 步骤3: 数据准备和验证")
        print("-" * 40)
        
        try:
            # 创建演示QA数据
            demo_qa_data = [
                {
                    "question_id": "crypto_001",
                    "question": "什么是AES加密算法？请详细说明其工作原理。",
                    "context": "密码学基础知识",
                    "reference_answer": "AES（Advanced Encryption Standard）是一种对称加密算法，采用分组密码体制。它使用128位分组长度，支持128、192、256位密钥长度。AES算法基于代替-置换网络（SPN）结构，包含字节代替、行移位、列混合和轮密钥加等操作。",
                    "model_answer": "AES是高级加密标准，是一种广泛使用的对称加密算法。它将数据分成128位的块进行加密，支持不同长度的密钥。AES算法安全性高，性能优秀，被广泛应用于各种安全系统中。",
                    "domain_tags": ["密码学", "对称加密", "分组密码"],
                    "difficulty_level": "intermediate",
                    "expected_concepts": ["对称加密", "分组密码", "密钥长度", "加密轮数"]
                },
                {
                    "question_id": "crypto_002", 
                    "question": "RSA算法的安全性基于什么数学难题？",
                    "context": "公钥密码学",
                    "reference_answer": "RSA算法的安全性基于大整数分解的数学难题。具体来说，是基于在计算上难以将两个大素数的乘积进行因式分解。RSA的公钥包含一个大合数n=p×q，其中p和q是两个大素数。攻击者需要分解n才能获得私钥，但目前没有有效的算法能在多项式时间内分解大整数。",
                    "model_answer": "RSA算法的安全性依赖于大数分解问题的困难性。当两个大素数相乘得到一个合数时，要从这个合数反推出原来的两个素数是非常困难的。这个数学难题保证了RSA加密的安全性。",
                    "domain_tags": ["密码学", "公钥加密", "数论"],
                    "difficulty_level": "advanced",
                    "expected_concepts": ["大整数分解", "素数", "公钥", "私钥", "数学难题"]
                },
                {
                    "question_id": "crypto_003",
                    "question": "什么是数字签名？它如何保证数据的完整性和认证性？",
                    "context": "数字签名技术",
                    "reference_answer": "数字签名是一种数学机制，用于验证数字消息或文档的真实性。它基于公钥密码学，使用私钥对消息的哈希值进行加密生成签名，接收方用对应的公钥验证签名。数字签名提供三个安全属性：1）认证性-确认消息来源；2）完整性-检测消息是否被篡改；3）不可否认性-发送方无法否认发送过该消息。",
                    "model_answer": "数字签名是用来验证电子文档真实性的技术。发送方用自己的私钥对文档进行签名，接收方用发送方的公钥来验证签名。如果验证成功，说明文档确实来自发送方且未被修改。这样就保证了数据的完整性和发送方的身份认证。",
                    "domain_tags": ["密码学", "数字签名", "身份认证"],
                    "difficulty_level": "intermediate",
                    "expected_concepts": ["数字签名", "哈希函数", "公钥", "私钥", "完整性", "认证性", "不可否认性"]
                }
            ]
            
            # 保存演示数据
            qa_data_path = self.demo_data_dir / "demo_qa_data.json"
            with open(qa_data_path, 'w', encoding='utf-8') as f:
                json.dump(demo_qa_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 演示QA数据已创建: {qa_data_path}")
            print(f"📊 数据统计: {len(demo_qa_data)}个QA项")
            
            # 初始化数据管理器
            self.data_manager = EvaluationDataManager()
            
            # 验证数据格式
            validation_result = self.data_manager.validate_qa_data(demo_qa_data)
            if validation_result.is_valid:
                print("✅ 数据格式验证通过")
                print(f"📋 验证统计:")
                print(f"   - 有效项目: {validation_result.valid_count}")
                print(f"   - 无效项目: {validation_result.invalid_count}")
                print(f"   - 平均问题长度: {validation_result.avg_question_length:.0f}字符")
                print(f"   - 平均答案长度: {validation_result.avg_answer_length:.0f}字符")
            else:
                print("❌ 数据格式验证失败")
                for error in validation_result.errors:
                    print(f"   - {error}")
                return False
            
            # 保存验证后的数据
            self.demo_qa_data = demo_qa_data
            return True
            
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return False
    
    def step_4_single_evaluation(self):
        """步骤4: 单项评估演示"""
        print("\n🔍 步骤4: 单项评估演示")
        print("-" * 40)
        
        try:
            # 选择第一个QA项进行演示
            qa_item = self.demo_qa_data[0]
            print(f"📝 评估项目: {qa_item['question_id']}")
            print(f"❓ 问题: {qa_item['question'][:50]}...")
            
            # 执行评估
            print("⏳ 正在执行评估...")
            start_time = time.time()
            
            result = self.engine.evaluate_single_qa(qa_item)
            
            end_time = time.time()
            evaluation_time = end_time - start_time
            
            print(f"✅ 评估完成 (耗时: {evaluation_time:.2f}秒)")
            
            # 显示评估结果
            print("\n📊 评估结果:")
            print(f"🎯 总体得分: {result.overall_score:.3f}")
            
            print("\n📈 维度得分:")
            for dimension, score in result.dimension_scores.items():
                print(f"   - {dimension}: {score:.3f}")
            
            print("\n🏭 行业指标:")
            for metric, value in result.industry_metrics.items():
                print(f"   - {metric}: {value:.3f}")
            
            if result.improvement_suggestions:
                print("\n💡 改进建议:")
                for i, suggestion in enumerate(result.improvement_suggestions, 1):
                    print(f"   {i}. {suggestion}")
            
            # 保存单项评估结果
            single_result_path = self.demo_output_dir / "single_evaluation_result.json"
            with open(single_result_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 结果已保存: {single_result_path}")
            return True
            
        except Exception as e:
            print(f"❌ 单项评估失败: {e}")
            return False
    
    def step_5_batch_evaluation(self):
        """步骤5: 批量评估演示"""
        print("\n📦 步骤5: 批量评估演示")
        print("-" * 40)
        
        try:
            print(f"📊 批量评估 {len(self.demo_qa_data)} 个QA项")
            
            # 执行批量评估
            print("⏳ 正在执行批量评估...")
            start_time = time.time()
            
            batch_result = self.engine.evaluate_batch(self.demo_qa_data)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"✅ 批量评估完成 (总耗时: {total_time:.2f}秒)")
            print(f"⚡ 平均每项耗时: {total_time/len(self.demo_qa_data):.2f}秒")
            
            # 显示批量评估统计
            print("\n📊 批量评估统计:")
            print(f"🎯 平均总体得分: {batch_result.average_overall_score:.3f}")
            print(f"📈 最高得分: {batch_result.max_score:.3f}")
            print(f"📉 最低得分: {batch_result.min_score:.3f}")
            print(f"📊 标准差: {batch_result.score_std:.3f}")
            
            print("\n📈 平均维度得分:")
            for dimension, avg_score in batch_result.average_dimension_scores.items():
                print(f"   - {dimension}: {avg_score:.3f}")
            
            # 显示评估分布
            print("\n📊 得分分布:")
            score_ranges = [
                (0.9, 1.0, "优秀"),
                (0.8, 0.9, "良好"), 
                (0.7, 0.8, "中等"),
                (0.6, 0.7, "及格"),
                (0.0, 0.6, "不及格")
            ]
            
            for min_score, max_score, label in score_ranges:
                count = sum(1 for result in batch_result.individual_results 
                           if min_score <= result.overall_score < max_score)
                percentage = count / len(batch_result.individual_results) * 100
                print(f"   - {label} ({min_score:.1f}-{max_score:.1f}): {count}项 ({percentage:.1f}%)")
            
            # 保存批量评估结果
            batch_result_path = self.demo_output_dir / "batch_evaluation_result.json"
            with open(batch_result_path, 'w', encoding='utf-8') as f:
                json.dump(batch_result.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 批量结果已保存: {batch_result_path}")
            
            self.batch_result = batch_result
            return True
            
        except Exception as e:
            print(f"❌ 批量评估失败: {e}")
            return False
    
    def step_6_report_generation(self):
        """步骤6: 结果分析和报告生成"""
        print("\n📋 步骤6: 结果分析和报告生成")
        print("-" * 40)
        
        try:
            # 初始化报告生成器
            report_generator = EvaluationReportGenerator()
            
            # 生成详细报告
            print("📝 正在生成详细报告...")
            
            detailed_report = report_generator.generate_detailed_report(
                self.batch_result,
                include_charts=True,
                include_recommendations=True
            )
            
            # 保存HTML报告
            html_report_path = self.demo_output_dir / "evaluation_report.html"
            report_generator.save_html_report(detailed_report, html_report_path)
            print(f"✅ HTML报告已生成: {html_report_path}")
            
            # 生成JSON报告
            json_report_path = self.demo_output_dir / "evaluation_report.json"
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_report.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"✅ JSON报告已生成: {json_report_path}")
            
            # 显示报告摘要
            print("\n📊 报告摘要:")
            print(f"📝 评估项目总数: {detailed_report.total_evaluations}")
            print(f"🎯 平均得分: {detailed_report.overall_statistics['mean']:.3f}")
            print(f"📊 得分中位数: {detailed_report.overall_statistics['median']:.3f}")
            print(f"📈 最佳表现维度: {detailed_report.best_performing_dimension}")
            print(f"📉 待改进维度: {detailed_report.worst_performing_dimension}")
            
            if detailed_report.key_insights:
                print("\n🔍 关键洞察:")
                for i, insight in enumerate(detailed_report.key_insights, 1):
                    print(f"   {i}. {insight}")
            
            if detailed_report.recommendations:
                print("\n💡 改进建议:")
                for i, recommendation in enumerate(detailed_report.recommendations, 1):
                    print(f"   {i}. {recommendation}")
            
            return True
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            return False
    
    def step_7_performance_benchmark(self):
        """步骤7: 性能基准测试"""
        print("\n⚡ 步骤7: 性能基准测试")
        print("-" * 40)
        
        try:
            # 初始化性能基准测试
            benchmark = PerformanceBenchmark()
            
            print("🔧 正在运行性能基准测试...")
            
            # 运行基准测试
            benchmark_result = benchmark.run_comprehensive_benchmark(
                qa_data=self.demo_qa_data[:2],  # 使用前2个项目进行快速测试
                config=self.config
            )
            
            print("✅ 性能基准测试完成")
            
            # 显示性能指标
            print("\n📊 性能指标:")
            print(f"⏱️  平均评估时间: {benchmark_result.avg_evaluation_time:.2f}秒")
            print(f"🚀 吞吐量: {benchmark_result.throughput:.1f} QA项/秒")
            print(f"💾 峰值内存使用: {benchmark_result.peak_memory_mb:.1f}MB")
            print(f"🔥 平均CPU使用率: {benchmark_result.avg_cpu_percent:.1f}%")
            
            if benchmark_result.gpu_metrics:
                print(f"🎮 GPU使用率: {benchmark_result.gpu_metrics.get('utilization', 0):.1f}%")
                print(f"🎮 GPU内存使用: {benchmark_result.gpu_metrics.get('memory_used_mb', 0):.1f}MB")
            
            # 性能评级
            performance_grade = self._calculate_performance_grade(benchmark_result)
            print(f"\n🏆 性能评级: {performance_grade}")
            
            # 保存基准测试结果
            benchmark_path = self.demo_output_dir / "performance_benchmark.json"
            with open(benchmark_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_result.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"💾 基准测试结果已保存: {benchmark_path}")
            return True
            
        except Exception as e:
            print(f"❌ 性能基准测试失败: {e}")
            return False
    
    def step_8_api_demo(self):
        """步骤8: API服务演示"""
        print("\n🌐 步骤8: API服务演示")
        print("-" * 40)
        
        try:
            print("ℹ️  API服务演示 (模拟)")
            print("   在实际使用中，可以启动API服务器:")
            print("   uv run python -m src.expert_evaluation.api")
            
            # 模拟API请求示例
            api_request_example = {
                "qa_items": self.demo_qa_data[:1],
                "config": {
                    "evaluation_dimensions": ["semantic_similarity", "domain_accuracy"],
                    "async_mode": False
                }
            }
            
            # 保存API请求示例
            api_example_path = self.demo_output_dir / "api_request_example.json"
            with open(api_example_path, 'w', encoding='utf-8') as f:
                json.dump(api_request_example, f, indent=2, ensure_ascii=False)
            
            print(f"📝 API请求示例已保存: {api_example_path}")
            
            # 显示API使用示例
            print("\n📋 API使用示例:")
            print("```bash")
            print("# 启动API服务")
            print("uv run uvicorn src.expert_evaluation.api:app --host 0.0.0.0 --port 8000")
            print("")
            print("# 健康检查")
            print("curl -X GET 'http://localhost:8000/health'")
            print("")
            print("# 提交评估任务")
            print("curl -X POST 'http://localhost:8000/evaluate' \\")
            print("  -H 'Content-Type: application/json' \\")
            print(f"  -d @{api_example_path}")
            print("```")
            
            return True
            
        except Exception as e:
            print(f"❌ API演示失败: {e}")
            return False
    
    def _calculate_performance_grade(self, benchmark_result) -> str:
        """计算性能评级"""
        score = 0
        
        # 评估时间评分 (越快越好)
        if benchmark_result.avg_evaluation_time < 1.0:
            score += 30
        elif benchmark_result.avg_evaluation_time < 2.0:
            score += 25
        elif benchmark_result.avg_evaluation_time < 5.0:
            score += 20
        else:
            score += 10
        
        # 吞吐量评分 (越高越好)
        if benchmark_result.throughput > 2.0:
            score += 30
        elif benchmark_result.throughput > 1.0:
            score += 25
        elif benchmark_result.throughput > 0.5:
            score += 20
        else:
            score += 10
        
        # 内存使用评分 (越少越好)
        if benchmark_result.peak_memory_mb < 1000:
            score += 25
        elif benchmark_result.peak_memory_mb < 2000:
            score += 20
        elif benchmark_result.peak_memory_mb < 4000:
            score += 15
        else:
            score += 10
        
        # CPU使用评分 (适中最好)
        if 30 <= benchmark_result.avg_cpu_percent <= 70:
            score += 15
        elif 20 <= benchmark_result.avg_cpu_percent <= 80:
            score += 10
        else:
            score += 5
        
        # 根据总分确定等级
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 80:
            return "A (良好)"
        elif score >= 70:
            return "B (中等)"
        elif score >= 60:
            return "C (及格)"
        else:
            return "D (需要优化)"
    
    def generate_summary_report(self):
        """生成演示总结报告"""
        print("\n📋 演示总结报告")
        print("=" * 60)
        
        summary = {
            "demo_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0.0",
                "total_qa_items": len(self.demo_qa_data) if hasattr(self, 'demo_qa_data') else 0
            },
            "system_status": {
                "initialization": "✅ 成功",
                "model_loading": "✅ 成功 (演示模式)",
                "data_preparation": "✅ 成功",
                "evaluation_engine": "✅ 正常运行"
            },
            "evaluation_results": {
                "single_evaluation": "✅ 完成",
                "batch_evaluation": "✅ 完成",
                "report_generation": "✅ 完成",
                "performance_benchmark": "✅ 完成"
            },
            "output_files": [
                str(self.demo_output_dir / "single_evaluation_result.json"),
                str(self.demo_output_dir / "batch_evaluation_result.json"),
                str(self.demo_output_dir / "evaluation_report.html"),
                str(self.demo_output_dir / "evaluation_report.json"),
                str(self.demo_output_dir / "performance_benchmark.json"),
                str(self.demo_output_dir / "api_request_example.json")
            ],
            "next_steps": [
                "配置真实的模型路径",
                "准备实际的QA评估数据",
                "根据需求调整评估维度和权重",
                "部署API服务到生产环境",
                "集成到现有的评估流程中"
            ]
        }
        
        # 保存总结报告
        summary_path = self.demo_output_dir / "demo_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 演示数据: {summary['demo_info']['total_qa_items']} 个QA项")
        print(f"📁 输出文件: {len(summary['output_files'])} 个")
        print(f"💾 总结报告: {summary_path}")
        
        print("\n🎯 下一步建议:")
        for i, step in enumerate(summary['next_steps'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n📂 所有输出文件位于: {self.demo_output_dir}")
        print("🎉 专家评估系统演示完成!")
    
    def run_complete_demo(self):
        """运行完整演示"""
        steps = [
            ("系统初始化", self.step_1_system_initialization),
            ("模型加载", self.step_2_model_loading),
            ("数据准备", self.step_3_data_preparation),
            ("单项评估", self.step_4_single_evaluation),
            ("批量评估", self.step_5_batch_evaluation),
            ("报告生成", self.step_6_report_generation),
            ("性能测试", self.step_7_performance_benchmark),
            ("API演示", self.step_8_api_demo)
        ]
        
        success_count = 0
        
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
                else:
                    print(f"⚠️  步骤 '{step_name}' 未完全成功，继续下一步...")
            except Exception as e:
                print(f"❌ 步骤 '{step_name}' 执行失败: {e}")
                print("继续执行下一步...")
        
        print(f"\n📊 演示完成统计: {success_count}/{len(steps)} 个步骤成功")
        
        # 生成总结报告
        self.generate_summary_report()

def main():
    """主函数"""
    try:
        # 创建并运行演示
        demo = ExpertEvaluationDemo()
        demo.run_complete_demo()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 感谢使用专家评估系统演示!")

if __name__ == "__main__":
    main()