#!/usr/bin/env python3
"""
专家评估系统简单演示

这是一个独立的演示脚本，展示专家评估系统的核心概念和使用方法，
不依赖完整的实现，适合快速了解系统功能。

使用方法:
    uv run python examples/expert_evaluation_simple_demo.py

作者: 专家评估系统开发团队
"""

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any

class SimpleEvaluationDemo:
    """简单评估演示类"""
    
    def __init__(self):
        self.output_dir = Path("simple_demo_output")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🚀 专家评估系统简单演示")
        print("=" * 50)
        print("本演示展示系统的核心概念和基本流程")
    
    def create_sample_data(self):
        """创建示例数据"""
        print("\n📊 步骤1: 创建示例数据")
        print("-" * 30)
        
        # 创建示例QA数据
        qa_data = [
            {
                "question_id": "demo_001",
                "question": "什么是人工智能？",
                "context": "计算机科学基础",
                "reference_answer": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。它包括机器学习、深度学习、自然语言处理等技术。",
                "model_answer": "人工智能是让计算机模拟人类智能的技术，包括学习、推理、感知等能力。现在广泛应用于语音识别、图像识别、自动驾驶等领域。",
                "domain_tags": ["人工智能", "计算机科学"],
                "difficulty_level": "beginner",
                "expected_concepts": ["机器学习", "智能系统", "算法"]
            },
            {
                "question_id": "demo_002",
                "question": "解释机器学习中的过拟合现象",
                "context": "机器学习理论",
                "reference_answer": "过拟合是指模型在训练数据上表现很好，但在新数据上表现较差的现象。这通常发生在模型过于复杂，学习了训练数据中的噪声和细节，而不是潜在的模式。解决方法包括正则化、交叉验证、增加训练数据等。",
                "model_answer": "过拟合就是模型记住了训练数据的具体内容，而不是学会了通用规律。就像学生死记硬背考试题目，遇到新题目就不会做了。可以通过减少模型复杂度或增加数据来解决。",
                "domain_tags": ["机器学习", "模型训练"],
                "difficulty_level": "intermediate",
                "expected_concepts": ["泛化能力", "正则化", "交叉验证"]
            },
            {
                "question_id": "demo_003",
                "question": "描述Transformer架构的核心创新",
                "context": "深度学习架构",
                "reference_answer": "Transformer的核心创新是注意力机制(Attention Mechanism)，特别是自注意力(Self-Attention)。它摒弃了传统的循环和卷积结构，完全基于注意力机制来处理序列数据。这使得模型能够并行处理，提高了训练效率，并且能够更好地捕捉长距离依赖关系。",
                "model_answer": "Transformer最重要的创新是注意力机制，让模型能够同时关注输入序列的所有位置。这比传统的RNN更高效，因为可以并行计算，而且能更好地处理长文本。",
                "domain_tags": ["深度学习", "自然语言处理", "神经网络"],
                "difficulty_level": "advanced",
                "expected_concepts": ["注意力机制", "并行计算", "序列建模"]
            }
        ]
        
        # 保存示例数据
        data_path = self.output_dir / "sample_qa_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 示例QA数据已创建: {data_path}")
        print(f"📊 数据统计: {len(qa_data)} 个QA项")
        
        return qa_data
    
    def create_sample_config(self):
        """创建示例配置"""
        print("\n⚙️ 步骤2: 创建示例配置")
        print("-" * 30)
        
        config = {
            "model": {
                "model_path": "/path/to/your/model",
                "device": "auto",
                "quantization": "int8",
                "batch_size": 4
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
                "max_workers": 4,
                "timeout": 300,
                "memory_limit": "8GB",
                "cache_size": "1GB"
            },
            "output": {
                "format": "json",
                "detailed": True,
                "save_intermediate": False
            }
        }
        
        # 保存配置
        config_path = self.output_dir / "sample_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 示例配置已创建: {config_path}")
        print("📋 配置包含:")
        print(f"   - 评估维度: {len(config['evaluation']['dimensions'])} 个")
        print(f"   - 工作线程: {config['performance']['max_workers']} 个")
        print(f"   - 内存限制: {config['performance']['memory_limit']}")
        
        return config
    
    def simulate_evaluation(self, qa_data: List[Dict], config: Dict):
        """模拟评估过程"""
        print("\n🔍 步骤3: 模拟评估过程")
        print("-" * 30)
        
        print("⏳ 正在执行评估...")
        
        results = []
        
        for i, qa_item in enumerate(qa_data):
            print(f"📝 评估项目 {i+1}/{len(qa_data)}: {qa_item['question_id']}")
            
            # 模拟评估计算 (生成随机但合理的分数)
            random.seed(hash(qa_item['question_id']))  # 确保结果可重现
            
            # 根据难度级别调整基础分数
            difficulty_multiplier = {
                "beginner": 0.85,
                "intermediate": 0.75,
                "advanced": 0.70,
                "expert": 0.65
            }
            
            base_score = difficulty_multiplier.get(qa_item.get('difficulty_level', 'intermediate'), 0.75)
            
            # 生成各维度得分
            dimension_scores = {}
            for dimension in config['evaluation']['dimensions']:
                # 添加一些随机变化
                variation = (random.random() - 0.5) * 0.2  # -0.1 到 0.1 的变化
                score = max(0.0, min(1.0, base_score + variation))
                dimension_scores[dimension] = round(score, 3)
            
            # 计算加权总分
            weights = config['evaluation']['weights']
            overall_score = sum(
                dimension_scores[dim] * weights.get(dim, 0) 
                for dim in dimension_scores
            )
            overall_score = round(overall_score, 3)
            
            # 生成行业指标
            industry_metrics = {
                "domain_relevance": round(base_score + random.uniform(-0.05, 0.05), 3),
                "practical_applicability": round(base_score + random.uniform(-0.08, 0.08), 3),
                "innovation_level": round(base_score + random.uniform(-0.1, 0.1), 3),
                "completeness": round(base_score + random.uniform(-0.06, 0.06), 3)
            }
            
            # 生成改进建议
            suggestions = []
            if dimension_scores.get('semantic_similarity', 0) < 0.8:
                suggestions.append("提高答案与参考答案的语义相似性")
            if dimension_scores.get('domain_accuracy', 0) < 0.8:
                suggestions.append("增强专业领域知识的准确性")
            if dimension_scores.get('completeness', 0) < 0.8:
                suggestions.append("补充更完整的信息和细节")
            if not suggestions:
                suggestions.append("继续保持当前的高质量水平")
            
            result = {
                "question_id": qa_item['question_id'],
                "overall_score": overall_score,
                "dimension_scores": dimension_scores,
                "industry_metrics": industry_metrics,
                "improvement_suggestions": suggestions,
                "confidence_intervals": {
                    "overall_score": [
                        round(overall_score - 0.03, 3),
                        round(overall_score + 0.03, 3)
                    ]
                },
                "evaluation_time": round(random.uniform(0.5, 2.0), 2)
            }
            
            results.append(result)
            
            # 显示单项结果
            print(f"   🎯 得分: {overall_score:.3f}")
            print(f"   ⏱️  耗时: {result['evaluation_time']}秒")
        
        print("✅ 评估完成")
        return results
    
    def analyze_results(self, results: List[Dict]):
        """分析评估结果"""
        print("\n📊 步骤4: 结果分析")
        print("-" * 30)
        
        # 计算统计信息
        scores = [r['overall_score'] for r in results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # 计算各维度平均分
        all_dimensions = set()
        for result in results:
            all_dimensions.update(result['dimension_scores'].keys())
        
        dimension_averages = {}
        for dim in all_dimensions:
            dim_scores = [r['dimension_scores'].get(dim, 0) for r in results]
            dimension_averages[dim] = sum(dim_scores) / len(dim_scores)
        
        # 分析结果
        analysis = {
            "summary": {
                "total_evaluations": len(results),
                "average_score": round(avg_score, 3),
                "max_score": round(max_score, 3),
                "min_score": round(min_score, 3),
                "score_range": round(max_score - min_score, 3)
            },
            "dimension_analysis": {
                dim: round(avg, 3) for dim, avg in dimension_averages.items()
            },
            "performance_distribution": {
                "excellent (≥0.9)": len([s for s in scores if s >= 0.9]),
                "good (0.8-0.9)": len([s for s in scores if 0.8 <= s < 0.9]),
                "fair (0.7-0.8)": len([s for s in scores if 0.7 <= s < 0.8]),
                "poor (<0.7)": len([s for s in scores if s < 0.7])
            },
            "top_performing_dimension": max(dimension_averages.items(), key=lambda x: x[1])[0],
            "lowest_performing_dimension": min(dimension_averages.items(), key=lambda x: x[1])[0]
        }
        
        # 显示分析结果
        print("📈 评估统计:")
        print(f"   总评估项目: {analysis['summary']['total_evaluations']}")
        print(f"   平均得分: {analysis['summary']['average_score']}")
        print(f"   最高得分: {analysis['summary']['max_score']}")
        print(f"   最低得分: {analysis['summary']['min_score']}")
        
        print("\n📊 维度表现:")
        for dim, score in analysis['dimension_analysis'].items():
            print(f"   {dim}: {score:.3f}")
        
        print("\n🏆 表现分布:")
        for level, count in analysis['performance_distribution'].items():
            percentage = (count / len(results)) * 100
            print(f"   {level}: {count}项 ({percentage:.1f}%)")
        
        print(f"\n🎯 最佳维度: {analysis['top_performing_dimension']}")
        print(f"📉 待改进维度: {analysis['lowest_performing_dimension']}")
        
        return analysis
    
    def generate_report(self, qa_data: List[Dict], results: List[Dict], analysis: Dict):
        """生成评估报告"""
        print("\n📋 步骤5: 生成评估报告")
        print("-" * 30)
        
        # 创建详细报告
        report = {
            "report_info": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "report_type": "专家评估系统演示报告",
                "version": "1.0.0"
            },
            "evaluation_summary": analysis['summary'],
            "dimension_performance": analysis['dimension_analysis'],
            "performance_distribution": analysis['performance_distribution'],
            "detailed_results": results,
            "recommendations": [
                f"重点提升 {analysis['lowest_performing_dimension']} 维度的表现",
                f"保持 {analysis['top_performing_dimension']} 维度的优势",
                "增加更多样化的评估数据",
                "根据实际需求调整评估权重",
                "定期更新评估标准和阈值"
            ],
            "next_steps": [
                "配置真实的模型路径",
                "准备实际的QA评估数据",
                "根据业务需求调整评估维度",
                "建立定期评估流程",
                "集成到现有的开发流程中"
            ]
        }
        
        # 保存JSON报告
        json_report_path = self.output_dir / "evaluation_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        html_report = self._generate_html_report(report)
        html_report_path = self.output_dir / "evaluation_report.html"
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ JSON报告已生成: {json_report_path}")
        print(f"✅ HTML报告已生成: {html_report_path}")
        
        return report
    
    def _generate_html_report(self, report: Dict) -> str:
        """生成HTML格式报告"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>专家评估系统演示报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background: #e9f5ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .recommendation {{ background: #f0f8f0; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .score-excellent {{ color: #28a745; font-weight: bold; }}
        .score-good {{ color: #17a2b8; font-weight: bold; }}
        .score-fair {{ color: #ffc107; font-weight: bold; }}
        .score-poor {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 专家评估系统演示报告</h1>
        <p><strong>生成时间:</strong> {report['report_info']['generated_at']}</p>
        <p><strong>报告版本:</strong> {report['report_info']['version']}</p>
    </div>
    
    <div class="section">
        <h2>📊 评估概要</h2>
        <div class="metric">总评估项目: {report['evaluation_summary']['total_evaluations']}</div>
        <div class="metric">平均得分: {report['evaluation_summary']['average_score']}</div>
        <div class="metric">最高得分: {report['evaluation_summary']['max_score']}</div>
        <div class="metric">最低得分: {report['evaluation_summary']['min_score']}</div>
    </div>
    
    <div class="section">
        <h2>📈 维度表现</h2>
        <table>
            <tr><th>评估维度</th><th>平均得分</th><th>表现等级</th></tr>
        """
        
        for dim, score in report['dimension_performance'].items():
            if score >= 0.9:
                grade_class = "score-excellent"
                grade = "优秀"
            elif score >= 0.8:
                grade_class = "score-good"
                grade = "良好"
            elif score >= 0.7:
                grade_class = "score-fair"
                grade = "一般"
            else:
                grade_class = "score-poor"
                grade = "待改进"
            
            html += f'<tr><td>{dim}</td><td class="{grade_class}">{score}</td><td class="{grade_class}">{grade}</td></tr>'
        
        html += f"""
        </table>
    </div>
    
    <div class="section">
        <h2>🏆 表现分布</h2>
        <table>
            <tr><th>表现等级</th><th>项目数量</th></tr>
        """
        
        for level, count in report['performance_distribution'].items():
            html += f"<tr><td>{level}</td><td>{count}</td></tr>"
        
        html += f"""
        </table>
    </div>
    
    <div class="section">
        <h2>💡 改进建议</h2>
        """
        
        for rec in report['recommendations']:
            html += f'<div class="recommendation">• {rec}</div>'
        
        html += f"""
    </div>
    
    <div class="section">
        <h2>🎯 下一步行动</h2>
        """
        
        for step in report['next_steps']:
            html += f'<div class="recommendation">• {step}</div>'
        
        html += """
    </div>
    
    <div class="section">
        <p><em>本报告由专家评估系统自动生成</em></p>
    </div>
</body>
</html>
        """
        
        return html
    
    def demonstrate_cli_usage(self):
        """演示CLI使用方法"""
        print("\n💻 步骤6: CLI使用演示")
        print("-" * 30)
        
        cli_examples = {
            "基本命令": [
                "# 显示帮助信息",
                "uv run python -m src.expert_evaluation.cli --help",
                "",
                "# 初始化配置文件", 
                "uv run python -m src.expert_evaluation.cli init-config",
                "",
                "# 验证数据格式",
                "uv run python -m src.expert_evaluation.cli validate-data sample_qa_data.json",
                "",
                "# 执行评估",
                "uv run python -m src.expert_evaluation.cli evaluate sample_qa_data.json"
            ],
            "高级用法": [
                "# 使用自定义配置",
                "uv run python -m src.expert_evaluation.cli -c sample_config.json evaluate sample_qa_data.json",
                "",
                "# 保存结果到文件",
                "uv run python -m src.expert_evaluation.cli evaluate sample_qa_data.json -o results.json",
                "",
                "# 生成详细报告",
                "uv run python -m src.expert_evaluation.cli evaluate sample_qa_data.json --detailed",
                "",
                "# 使用不同输出格式",
                "uv run python -m src.expert_evaluation.cli evaluate sample_qa_data.json -f html"
            ],
            "API使用": [
                "# 启动API服务",
                "uv run uvicorn src.expert_evaluation.api:app --host 0.0.0.0 --port 8000",
                "",
                "# 健康检查",
                "curl -X GET 'http://localhost:8000/health'",
                "",
                "# 提交评估任务",
                "curl -X POST 'http://localhost:8000/evaluate' \\",
                "  -H 'Content-Type: application/json' \\",
                "  -d @sample_qa_data.json"
            ]
        }
        
        # 保存CLI示例
        cli_path = self.output_dir / "cli_usage_examples.md"
        with open(cli_path, 'w', encoding='utf-8') as f:
            f.write("# 专家评估系统CLI使用示例\n\n")
            
            for category, commands in cli_examples.items():
                f.write(f"## {category}\n\n")
                f.write("```bash\n")
                for cmd in commands:
                    f.write(f"{cmd}\n")
                f.write("```\n\n")
        
        print(f"✅ CLI使用示例已保存: {cli_path}")
        
        # 显示关键命令
        print("🔧 关键CLI命令:")
        print("   1. 初始化配置: uv run python -m src.expert_evaluation.cli init-config")
        print("   2. 验证数据: uv run python -m src.expert_evaluation.cli validate-data data.json")
        print("   3. 执行评估: uv run python -m src.expert_evaluation.cli evaluate data.json")
        print("   4. 启动API: uv run uvicorn src.expert_evaluation.api:app --port 8000")
    
    def run_complete_demo(self):
        """运行完整演示"""
        try:
            # 执行演示步骤
            qa_data = self.create_sample_data()
            config = self.create_sample_config()
            results = self.simulate_evaluation(qa_data, config)
            analysis = self.analyze_results(results)
            report = self.generate_report(qa_data, results, analysis)
            self.demonstrate_cli_usage()
            
            # 生成演示总结
            print("\n🎉 演示完成总结")
            print("=" * 50)
            print(f"📊 评估了 {len(qa_data)} 个QA项目")
            print(f"📈 平均得分: {analysis['summary']['average_score']}")
            print(f"🏆 最佳维度: {analysis['top_performing_dimension']}")
            print(f"📁 输出目录: {self.output_dir}")
            
            # 列出生成的文件
            output_files = list(self.output_dir.glob("*"))
            print(f"\n📂 生成的文件 ({len(output_files)}个):")
            for file_path in output_files:
                print(f"   - {file_path.name}")
            
            print("\n🎯 下一步建议:")
            print("   1. 查看生成的HTML报告了解详细结果")
            print("   2. 参考CLI使用示例学习命令行操作")
            print("   3. 根据实际需求修改配置文件")
            print("   4. 准备真实的QA数据进行评估")
            print("   5. 探索API接口进行系统集成")
            
            print("\n✨ 感谢使用专家评估系统演示！")
            
        except Exception as e:
            print(f"\n❌ 演示执行失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    demo = SimpleEvaluationDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()