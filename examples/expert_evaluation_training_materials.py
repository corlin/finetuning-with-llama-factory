#!/usr/bin/env python3
"""
专家评估系统用户培训材料

本脚本提供交互式培训材料，帮助用户学习和掌握专家评估系统的使用。

培训内容包括：
1. 系统概述和基本概念
2. 配置文件详解
3. 数据格式说明
4. 基础操作演练
5. 高级功能介绍
6. 最佳实践指导
7. 常见问题解答

使用方法:
    uv run python examples/expert_evaluation_training_materials.py

作者: 专家评估系统开发团队
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class InteractiveTraining:
    """交互式培训类"""
    
    def __init__(self):
        self.training_dir = Path("training_materials")
        self.training_dir.mkdir(exist_ok=True)
        
        self.current_lesson = 0
        self.user_progress = {
            "completed_lessons": [],
            "quiz_scores": {},
            "practice_results": {}
        }
        
        print("🎓 专家评估系统用户培训")
        print("=" * 50)
        print("欢迎参加专家评估系统培训课程！")
        print("本培训将帮助您全面掌握系统的使用方法。")
    
    def lesson_1_system_overview(self):
        """第1课: 系统概述"""
        print("\n📚 第1课: 系统概述")
        print("-" * 30)
        
        content = """
🎯 学习目标:
- 了解专家评估系统的作用和价值
- 掌握系统的核心概念
- 理解评估流程

📖 课程内容:

1. 什么是专家评估系统？
   专家评估系统是一个全面的行业化评估框架，专门用于评估训练后已合并的最终模型。
   它提供比传统BLEU、ROUGE更适合行业场景的多维度评估能力。

2. 核心概念:
   - QA评估项: 包含问题、参考答案、模型答案的评估单元
   - 评估维度: 不同角度的评估指标（语义相似性、领域准确性等）
   - 评估权重: 各维度在总评分中的重要程度
   - 行业指标: 针对特定行业的专业评估指标

3. 评估流程:
   数据准备 → 配置设置 → 模型加载 → 执行评估 → 结果分析 → 报告生成

4. 系统优势:
   - 多维度评估: 8个评估维度全面覆盖
   - 行业适配: 针对不同行业的专业指标
   - 高性能: 支持批量处理和并发评估
   - 易扩展: 插件化架构支持自定义评估器
   - 可视化: 丰富的图表和报告功能
        """
        
        print(content)
        
        # 保存课程内容
        with open(self.training_dir / "lesson_1_overview.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 小测验
        print("\n🧠 知识检测:")
        questions = [
            {
                "question": "专家评估系统提供多少个评估维度？",
                "options": ["A. 5个", "B. 6个", "C. 8个", "D. 10个"],
                "answer": "C"
            },
            {
                "question": "系统的主要优势不包括以下哪项？",
                "options": ["A. 多维度评估", "B. 自动数据生成", "C. 高性能处理", "D. 可视化报告"],
                "answer": "B"
            }
        ]
        
        score = self._conduct_quiz("第1课", questions)
        self.user_progress["quiz_scores"]["lesson_1"] = score
        
        if score >= 80:
            print("🎉 恭喜！您已掌握系统概述，可以继续下一课。")
            self.user_progress["completed_lessons"].append(1)
            return True
        else:
            print("📚 建议复习本课内容后再继续。")
            return False
    
    def lesson_2_configuration_guide(self):
        """第2课: 配置文件详解"""
        print("\n📚 第2课: 配置文件详解")
        print("-" * 30)
        
        content = """
🎯 学习目标:
- 理解配置文件的结构和作用
- 掌握各配置项的含义和设置方法
- 学会根据需求调整配置

📖 课程内容:

1. 配置文件结构:
   配置文件采用JSON格式，包含以下主要部分：
   - model: 模型相关配置
   - evaluation: 评估相关配置
   - performance: 性能相关配置
   - output: 输出相关配置
   - logging: 日志相关配置

2. 模型配置 (model):
   - model_path: 模型文件路径
   - device: 运行设备 (cpu/cuda/auto)
   - quantization: 量化方式 (int8/int4/fp16)
   - batch_size: 批处理大小

3. 评估配置 (evaluation):
   - dimensions: 评估维度列表
   - weights: 各维度权重
   - thresholds: 评估阈值
   - algorithms: 算法配置

4. 性能配置 (performance):
   - max_workers: 最大工作线程数
   - timeout: 超时时间
   - memory_limit: 内存限制
   - cache_size: 缓存大小

5. 配置示例:
        """
        
        # 创建示例配置
        example_config = {
            "model": {
                "model_path": "/path/to/model",
                "device": "auto",
                "quantization": "int8",
                "batch_size": 4
            },
            "evaluation": {
                "dimensions": [
                    "semantic_similarity",
                    "domain_accuracy",
                    "response_relevance"
                ],
                "weights": {
                    "semantic_similarity": 0.4,
                    "domain_accuracy": 0.4,
                    "response_relevance": 0.2
                },
                "thresholds": {
                    "min_score": 0.6,
                    "confidence_level": 0.95
                }
            },
            "performance": {
                "max_workers": 4,
                "timeout": 300,
                "memory_limit": "8GB"
            }
        }
        
        # 保存示例配置
        config_path = self.training_dir / "example_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2, ensure_ascii=False)
        
        content += f"""
   
   示例配置已保存到: {config_path}

6. 配置最佳实践:
   - 根据硬件资源调整性能参数
   - 根据应用场景选择评估维度
   - 合理设置权重分配
   - 定期验证配置有效性
        """
        
        print(content)
        
        # 保存课程内容
        with open(self.training_dir / "lesson_2_configuration.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 实践练习
        print("\n🛠️ 实践练习:")
        print("请根据以下需求创建配置文件：")
        print("- 使用GPU加速")
        print("- 只评估语义相似性和领域准确性")
        print("- 权重比例为 6:4")
        print("- 设置4个工作线程")
        
        practice_config = {
            "model": {"device": "cuda"},
            "evaluation": {
                "dimensions": ["semantic_similarity", "domain_accuracy"],
                "weights": {"semantic_similarity": 0.6, "domain_accuracy": 0.4}
            },
            "performance": {"max_workers": 4}
        }
        
        # 保存练习答案
        with open(self.training_dir / "practice_config_answer.json", 'w', encoding='utf-8') as f:
            json.dump(practice_config, f, indent=2, ensure_ascii=False)
        
        print(f"💡 参考答案已保存到: {self.training_dir / 'practice_config_answer.json'}")
        
        # 小测验
        questions = [
            {
                "question": "配置文件采用什么格式？",
                "options": ["A. YAML", "B. JSON", "C. XML", "D. INI"],
                "answer": "B"
            },
            {
                "question": "device配置项的auto值表示什么？",
                "options": ["A. 只使用CPU", "B. 只使用GPU", "C. 自动选择最佳设备", "D. 使用所有设备"],
                "answer": "C"
            }
        ]
        
        score = self._conduct_quiz("第2课", questions)
        self.user_progress["quiz_scores"]["lesson_2"] = score
        
        if score >= 80:
            print("🎉 恭喜！您已掌握配置文件使用，可以继续下一课。")
            self.user_progress["completed_lessons"].append(2)
            return True
        else:
            print("📚 建议复习配置相关内容后再继续。")
            return False
    
    def lesson_3_data_format(self):
        """第3课: 数据格式说明"""
        print("\n📚 第3课: 数据格式说明")
        print("-" * 30)
        
        content = """
🎯 学习目标:
- 理解QA数据的格式要求
- 掌握数据字段的含义和用法
- 学会准备和验证评估数据

📖 课程内容:

1. QA数据格式:
   QA数据采用JSON数组格式，每个元素代表一个评估项目。

2. 必需字段:
   - question_id: 问题唯一标识符
   - question: 问题内容
   - reference_answer: 参考答案
   - model_answer: 模型生成的答案

3. 可选字段:
   - context: 问题上下文信息
   - domain_tags: 领域标签列表
   - difficulty_level: 难度级别 (beginner/intermediate/advanced/expert)
   - expected_concepts: 期望包含的概念列表

4. 数据质量要求:
   - 问题表述清晰明确
   - 参考答案准确完整
   - 模型答案真实有效
   - 标签信息准确标注

5. 数据示例:
        """
        
        # 创建示例数据
        example_data = [
            {
                "question_id": "example_001",
                "question": "什么是机器学习？请简要说明其基本原理。",
                "context": "人工智能基础概念",
                "reference_answer": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。基本原理是通过算法分析大量数据，识别模式和规律，然后使用这些模式对新数据进行预测或决策。",
                "model_answer": "机器学习是AI的重要组成部分，让计算机可以从数据中自动学习规律。它通过训练算法来识别数据中的模式，从而对未知数据做出预测。",
                "domain_tags": ["人工智能", "机器学习", "算法"],
                "difficulty_level": "beginner",
                "expected_concepts": ["算法", "数据", "模式识别", "预测"]
            },
            {
                "question_id": "example_002",
                "question": "解释深度学习中的反向传播算法。",
                "context": "深度学习算法",
                "reference_answer": "反向传播算法是训练神经网络的核心算法。它通过计算损失函数对网络参数的梯度，然后使用梯度下降法更新参数。算法分为前向传播和反向传播两个阶段：前向传播计算网络输出和损失，反向传播计算梯度并更新权重。",
                "model_answer": "反向传播是深度学习的关键算法，用于训练神经网络。它通过计算误差梯度来调整网络权重，使模型能够学习数据中的复杂模式。",
                "domain_tags": ["深度学习", "神经网络", "优化算法"],
                "difficulty_level": "advanced",
                "expected_concepts": ["梯度", "损失函数", "权重更新", "前向传播"]
            }
        ]
        
        # 保存示例数据
        data_path = self.training_dir / "example_qa_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, indent=2, ensure_ascii=False)
        
        content += f"""

   示例数据已保存到: {data_path}

6. 数据验证:
   使用以下命令验证数据格式：
   uv run python -m src.expert_evaluation.cli validate-data your_data.json

7. 数据准备建议:
   - 确保问题表述清晰
   - 参考答案要准确权威
   - 模型答案要真实反映模型输出
   - 合理设置难度级别
   - 准确标注领域标签
        """
        
        print(content)
        
        # 保存课程内容
        with open(self.training_dir / "lesson_3_data_format.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 实践练习
        print("\n🛠️ 实践练习:")
        print("请创建一个关于'什么是区块链？'的QA数据项")
        
        practice_answer = {
            "question_id": "practice_001",
            "question": "什么是区块链？",
            "reference_answer": "区块链是一种分布式账本技术，通过密码学方法将数据块按时间顺序链接，形成不可篡改的数据链。",
            "model_answer": "区块链是一种去中心化的数据存储技术，具有不可篡改和透明的特点。",
            "domain_tags": ["区块链", "分布式系统", "密码学"],
            "difficulty_level": "intermediate",
            "expected_concepts": ["分布式", "密码学", "不可篡改", "去中心化"]
        }
        
        # 保存练习答案
        with open(self.training_dir / "practice_qa_answer.json", 'w', encoding='utf-8') as f:
            json.dump(practice_answer, f, indent=2, ensure_ascii=False)
        
        print(f"💡 参考答案已保存到: {self.training_dir / 'practice_qa_answer.json'}")
        
        # 小测验
        questions = [
            {
                "question": "QA数据中哪个字段是必需的？",
                "options": ["A. context", "B. domain_tags", "C. question_id", "D. difficulty_level"],
                "answer": "C"
            },
            {
                "question": "difficulty_level字段不包括以下哪个值？",
                "options": ["A. beginner", "B. intermediate", "C. professional", "D. expert"],
                "answer": "C"
            }
        ]
        
        score = self._conduct_quiz("第3课", questions)
        self.user_progress["quiz_scores"]["lesson_3"] = score
        
        if score >= 80:
            print("🎉 恭喜！您已掌握数据格式，可以继续下一课。")
            self.user_progress["completed_lessons"].append(3)
            return True
        else:
            print("📚 建议复习数据格式相关内容后再继续。")
            return False
    
    def lesson_4_basic_operations(self):
        """第4课: 基础操作演练"""
        print("\n📚 第4课: 基础操作演练")
        print("-" * 30)
        
        content = """
🎯 学习目标:
- 掌握命令行工具的使用
- 学会执行基本的评估操作
- 理解评估结果的含义

📖 课程内容:

1. 命令行工具概述:
   专家评估系统提供了完整的CLI工具，支持各种评估操作。

2. 基本命令:
   
   # 显示帮助信息
   uv run python -m src.expert_evaluation.cli --help
   
   # 初始化配置文件
   uv run python -m src.expert_evaluation.cli init-config
   
   # 验证数据格式
   uv run python -m src.expert_evaluation.cli validate-data data.json
   
   # 执行评估
   uv run python -m src.expert_evaluation.cli evaluate data.json
   
   # 使用自定义配置
   uv run python -m src.expert_evaluation.cli -c config.json evaluate data.json
   
   # 保存结果到文件
   uv run python -m src.expert_evaluation.cli evaluate data.json -o results.json

3. 评估结果解读:
   
   评估结果包含以下主要部分：
   - overall_score: 总体得分 (0-1之间)
   - dimension_scores: 各维度得分
   - industry_metrics: 行业特定指标
   - improvement_suggestions: 改进建议
   - confidence_intervals: 置信区间

4. 结果示例:
        """
        
        # 创建示例结果
        example_result = {
            "overall_score": 0.85,
            "dimension_scores": {
                "semantic_similarity": 0.88,
                "domain_accuracy": 0.82,
                "response_relevance": 0.85,
                "factual_correctness": 0.83,
                "completeness": 0.87
            },
            "industry_metrics": {
                "domain_relevance": 0.84,
                "practical_applicability": 0.86,
                "innovation_level": 0.78,
                "completeness": 0.89
            },
            "improvement_suggestions": [
                "增加更多具体的技术细节",
                "补充实际应用案例",
                "提高答案的创新性"
            ],
            "confidence_intervals": {
                "overall_score": [0.82, 0.88],
                "semantic_similarity": [0.85, 0.91]
            },
            "timestamp": "2024-01-01T12:00:00"
        }
        
        # 保存示例结果
        result_path = self.training_dir / "example_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(example_result, f, indent=2, ensure_ascii=False)
        
        content += f"""

   示例结果已保存到: {result_path}

5. 操作流程:
   
   步骤1: 准备数据
   - 创建或准备QA数据文件
   - 验证数据格式
   
   步骤2: 配置系统
   - 创建或修改配置文件
   - 根据需求调整参数
   
   步骤3: 执行评估
   - 运行评估命令
   - 监控评估进度
   
   步骤4: 分析结果
   - 查看评估得分
   - 理解各维度表现
   - 参考改进建议

6. 常用技巧:
   - 使用 --detailed 参数获取详细结果
   - 使用 --no-progress 参数在脚本中运行
   - 使用不同输出格式 (json/html/csv)
        """
        
        print(content)
        
        # 保存课程内容
        with open(self.training_dir / "lesson_4_operations.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建操作练习脚本
        practice_script = """#!/bin/bash
# 基础操作练习脚本

echo "🚀 开始基础操作练习"

# 1. 创建配置文件
echo "📋 步骤1: 创建配置文件"
uv run python -m src.expert_evaluation.cli init-config -o practice_config.json

# 2. 验证数据
echo "📊 步骤2: 验证数据格式"
uv run python -m src.expert_evaluation.cli validate-data example_qa_data.json

# 3. 执行评估
echo "🔍 步骤3: 执行评估"
uv run python -m src.expert_evaluation.cli -c practice_config.json evaluate example_qa_data.json -o practice_results.json

echo "✅ 练习完成！"
echo "📁 结果文件: practice_results.json"
"""
        
        script_path = self.training_dir / "practice_operations.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(practice_script)
        
        print(f"\n🛠️ 实践练习脚本已创建: {script_path}")
        print("请在终端中运行此脚本进行操作练习")
        
        # 小测验
        questions = [
            {
                "question": "哪个命令用于验证数据格式？",
                "options": ["A. validate-config", "B. validate-data", "C. check-data", "D. verify-format"],
                "answer": "B"
            },
            {
                "question": "overall_score的取值范围是？",
                "options": ["A. 0-100", "B. 1-10", "C. 0-1", "D. -1到1"],
                "answer": "C"
            }
        ]
        
        score = self._conduct_quiz("第4课", questions)
        self.user_progress["quiz_scores"]["lesson_4"] = score
        
        if score >= 80:
            print("🎉 恭喜！您已掌握基础操作，可以继续下一课。")
            self.user_progress["completed_lessons"].append(4)
            return True
        else:
            print("📚 建议复习基础操作后再继续。")
            return False
    
    def lesson_5_advanced_features(self):
        """第5课: 高级功能介绍"""
        print("\n📚 第5课: 高级功能介绍")
        print("-" * 30)
        
        content = """
🎯 学习目标:
- 了解系统的高级功能
- 掌握API服务的使用
- 学会性能优化和监控

📖 课程内容:

1. API服务:
   
   启动API服务:
   uv run uvicorn src.expert_evaluation.api:app --host 0.0.0.0 --port 8000
   
   主要API端点:
   - GET /health: 健康检查
   - POST /evaluate: 执行评估
   - GET /task/{task_id}: 查询任务状态
   - POST /generate_report: 生成报告

2. 批量处理:
   
   系统支持大规模批量处理，可以：
   - 并行处理多个QA项
   - 智能调整批处理大小
   - 监控处理进度
   - 优化内存使用

3. 异步评估:
   
   对于大型任务，可以使用异步模式：
   - 提交任务后立即返回任务ID
   - 通过任务ID查询进度和结果
   - 支持任务取消和重试

4. 自定义评估器:
   
   系统支持插件化扩展：
   - 实现自定义评估维度
   - 添加特定领域的评估逻辑
   - 集成外部评估工具

5. 性能监控:
   
   内置性能监控功能：
   - 实时监控系统资源使用
   - 评估性能指标统计
   - 自动性能优化建议

6. 结果可视化:
   
   丰富的可视化功能：
   - 评估结果图表
   - 趋势分析图
   - 对比分析报告
   - 交互式仪表板

7. 高级配置:
   
   支持复杂的配置选项：
   - 算法参数调优
   - 缓存策略配置
   - 并发控制设置
   - 安全和隐私配置
        """
        
        print(content)
        
        # 保存课程内容
        with open(self.training_dir / "lesson_5_advanced.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建API使用示例
        api_example = """
# API使用示例

import requests
import json

# 基础配置
BASE_URL = "http://localhost:8000"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/health")
print("健康状态:", response.json())

# 2. 同步评估
qa_data = {
    "qa_items": [
        {
            "question_id": "api_test_001",
            "question": "什么是REST API？",
            "reference_answer": "REST API是一种基于HTTP协议的Web服务接口...",
            "model_answer": "REST API是网络服务的一种架构风格...",
            "domain_tags": ["Web开发", "API设计"]
        }
    ],
    "async_mode": False
}

response = requests.post(f"{BASE_URL}/evaluate", json=qa_data)
result = response.json()
print("评估结果:", result)

# 3. 异步评估
qa_data["async_mode"] = True
response = requests.post(f"{BASE_URL}/evaluate", json=qa_data)
task_info = response.json()
task_id = task_info["task_id"]

# 查询任务状态
response = requests.get(f"{BASE_URL}/task/{task_id}")
status = response.json()
print("任务状态:", status)
        """
        
        api_path = self.training_dir / "api_example.py"
        with open(api_path, 'w', encoding='utf-8') as f:
            f.write(api_example)
        
        print(f"\n💻 API使用示例已保存到: {api_path}")
        
        # 小测验
        questions = [
            {
                "question": "API服务的默认端口是？",
                "options": ["A. 8080", "B. 8000", "C. 3000", "D. 5000"],
                "answer": "B"
            },
            {
                "question": "异步评估的主要优势是？",
                "options": ["A. 更高精度", "B. 更低成本", "C. 处理大型任务", "D. 更简单操作"],
                "answer": "C"
            }
        ]
        
        score = self._conduct_quiz("第5课", questions)
        self.user_progress["quiz_scores"]["lesson_5"] = score
        
        if score >= 80:
            print("🎉 恭喜！您已了解高级功能，可以继续下一课。")
            self.user_progress["completed_lessons"].append(5)
            return True
        else:
            print("📚 建议复习高级功能相关内容后再继续。")
            return False
    
    def lesson_6_best_practices(self):
        """第6课: 最佳实践指导"""
        print("\n📚 第6课: 最佳实践指导")
        print("-" * 30)
        
        content = """
🎯 学习目标:
- 掌握系统使用的最佳实践
- 了解常见问题的解决方案
- 学会优化评估效果

📖 课程内容:

1. 数据准备最佳实践:
   
   ✅ 推荐做法:
   - 确保问题表述清晰明确
   - 参考答案准确权威
   - 模型答案真实反映输出
   - 合理标注领域标签和难度
   - 保持数据集的多样性和平衡性
   
   ❌ 避免做法:
   - 问题表述模糊不清
   - 参考答案存在错误
   - 人为修改模型答案
   - 标签信息不准确
   - 数据集过于单一

2. 配置优化最佳实践:
   
   性能优化:
   - 根据硬件资源调整batch_size
   - 合理设置max_workers数量
   - 启用适当的模型量化
   - 配置合理的缓存大小
   
   准确性优化:
   - 选择合适的评估维度
   - 根据应用场景调整权重
   - 设置合理的评估阈值
   - 使用高质量的算法配置

3. 评估流程最佳实践:
   
   评估前:
   - 验证数据格式和质量
   - 检查配置文件正确性
   - 确认系统资源充足
   - 备份重要数据
   
   评估中:
   - 监控评估进度
   - 关注系统资源使用
   - 及时处理异常情况
   - 保存中间结果
   
   评估后:
   - 仔细分析评估结果
   - 验证结果合理性
   - 生成详细报告
   - 记录改进建议

4. 结果分析最佳实践:
   
   多维度分析:
   - 不仅关注总体得分
   - 分析各维度表现
   - 识别优势和不足
   - 对比历史结果
   
   深入理解:
   - 结合业务场景解读
   - 考虑评估的局限性
   - 参考置信区间
   - 关注统计显著性

5. 生产环境最佳实践:
   
   部署配置:
   - 使用生产级配置
   - 启用安全认证
   - 配置监控告警
   - 建立备份策略
   
   运维管理:
   - 定期更新系统
   - 监控性能指标
   - 及时处理告警
   - 维护日志记录

6. 团队协作最佳实践:
   
   标准化:
   - 统一配置标准
   - 建立数据规范
   - 制定操作流程
   - 共享最佳实践
   
   知识管理:
   - 文档化配置和流程
   - 分享经验和技巧
   - 定期培训更新
   - 建立问题库

7. 持续改进最佳实践:
   
   定期评估:
   - 评估系统效果
   - 收集用户反馈
   - 分析性能数据
   - 识别改进机会
   
   优化迭代:
   - 调整配置参数
   - 更新评估标准
   - 改进数据质量
   - 升级系统版本
        """
        
        print(content)
        
        # 保存课程内容
        with open(self.training_dir / "lesson_6_best_practices.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建最佳实践检查清单
        checklist = {
            "数据准备检查清单": [
                "□ 问题表述清晰明确",
                "□ 参考答案准确完整",
                "□ 模型答案真实有效",
                "□ 领域标签准确标注",
                "□ 难度级别合理设置",
                "□ 数据格式验证通过"
            ],
            "配置优化检查清单": [
                "□ 设备配置符合硬件",
                "□ 批处理大小合理",
                "□ 工作线程数适当",
                "□ 内存限制设置",
                "□ 评估维度选择",
                "□ 权重分配合理"
            ],
            "评估执行检查清单": [
                "□ 系统资源充足",
                "□ 配置文件正确",
                "□ 数据文件可访问",
                "□ 监控评估进度",
                "□ 处理异常情况",
                "□ 保存评估结果"
            ],
            "结果分析检查清单": [
                "□ 总体得分合理",
                "□ 各维度表现分析",
                "□ 置信区间检查",
                "□ 改进建议理解",
                "□ 历史对比分析",
                "□ 业务价值评估"
            ]
        }
        
        checklist_path = self.training_dir / "best_practices_checklist.json"
        with open(checklist_path, 'w', encoding='utf-8') as f:
            json.dump(checklist, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 最佳实践检查清单已保存到: {checklist_path}")
        
        # 小测验
        questions = [
            {
                "question": "数据准备时最重要的是？",
                "options": ["A. 数据量大", "B. 数据质量高", "C. 处理速度快", "D. 格式统一"],
                "answer": "B"
            },
            {
                "question": "生产环境部署时不需要考虑？",
                "options": ["A. 安全认证", "B. 监控告警", "C. 开发调试", "D. 备份策略"],
                "answer": "C"
            }
        ]
        
        score = self._conduct_quiz("第6课", questions)
        self.user_progress["quiz_scores"]["lesson_6"] = score
        
        if score >= 80:
            print("🎉 恭喜！您已掌握最佳实践，培训即将完成。")
            self.user_progress["completed_lessons"].append(6)
            return True
        else:
            print("📚 建议复习最佳实践相关内容。")
            return False
    
    def _conduct_quiz(self, lesson_name: str, questions: List[Dict]) -> float:
        """进行小测验"""
        print(f"\n🧠 {lesson_name} 知识检测:")
        correct_answers = 0
        
        for i, q in enumerate(questions, 1):
            print(f"\n问题 {i}: {q['question']}")
            for option in q['options']:
                print(f"   {option}")
            
            # 模拟用户选择 (在实际使用中可以添加交互输入)
            user_answer = q['answer']  # 为演示目的，假设用户总是选择正确答案
            
            if user_answer.upper() == q['answer'].upper():
                correct_answers += 1
                print(f"✅ 正确！")
            else:
                print(f"❌ 错误。正确答案是: {q['answer']}")
        
        score = (correct_answers / len(questions)) * 100
        print(f"\n📊 测验得分: {score:.0f}% ({correct_answers}/{len(questions)})")
        
        return score
    
    def generate_completion_certificate(self):
        """生成培训完成证书"""
        completed_lessons = len(self.user_progress["completed_lessons"])
        total_lessons = 6
        
        if completed_lessons == total_lessons:
            certificate = {
                "certificate_id": f"CERT_{int(time.time())}",
                "recipient": "培训学员",
                "course_name": "专家评估系统用户培训",
                "completion_date": time.strftime("%Y-%m-%d"),
                "lessons_completed": completed_lessons,
                "total_lessons": total_lessons,
                "quiz_scores": self.user_progress["quiz_scores"],
                "average_score": sum(self.user_progress["quiz_scores"].values()) / len(self.user_progress["quiz_scores"]),
                "status": "已完成"
            }
            
            cert_path = self.training_dir / "completion_certificate.json"
            with open(cert_path, 'w', encoding='utf-8') as f:
                json.dump(certificate, f, indent=2, ensure_ascii=False)
            
            print(f"\n🏆 恭喜！培训完成证书已生成: {cert_path}")
            print(f"📊 平均测验得分: {certificate['average_score']:.1f}%")
            
            return certificate
        else:
            print(f"\n📚 培训进度: {completed_lessons}/{total_lessons} 课程")
            print("请完成所有课程后获取证书。")
            return None
    
    def run_training_program(self):
        """运行完整培训程序"""
        lessons = [
            ("系统概述", self.lesson_1_system_overview),
            ("配置详解", self.lesson_2_configuration_guide),
            ("数据格式", self.lesson_3_data_format),
            ("基础操作", self.lesson_4_basic_operations),
            ("高级功能", self.lesson_5_advanced_features),
            ("最佳实践", self.lesson_6_best_practices)
        ]
        
        print("\n🎓 开始培训课程")
        print("=" * 50)
        
        for i, (lesson_name, lesson_func) in enumerate(lessons, 1):
            print(f"\n📚 准备开始第{i}课: {lesson_name}")
            input("按回车键继续...")
            
            success = lesson_func()
            
            if not success:
                print(f"⚠️  第{i}课未完全掌握，建议复习后继续。")
                break
        
        # 生成培训总结
        self.generate_training_summary()
        
        # 尝试生成完成证书
        self.generate_completion_certificate()
    
    def generate_training_summary(self):
        """生成培训总结"""
        summary = {
            "training_info": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_lessons": 6,
                "completed_lessons": len(self.user_progress["completed_lessons"])
            },
            "progress": self.user_progress,
            "materials_generated": [
                "lesson_1_overview.md",
                "lesson_2_configuration.md", 
                "lesson_3_data_format.md",
                "lesson_4_operations.md",
                "lesson_5_advanced.md",
                "lesson_6_best_practices.md",
                "example_config.json",
                "example_qa_data.json",
                "example_result.json",
                "api_example.py",
                "best_practices_checklist.json"
            ],
            "next_steps": [
                "实际操作练习",
                "准备真实评估数据",
                "配置生产环境",
                "建立评估流程",
                "团队知识分享"
            ]
        }
        
        summary_path = self.training_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 培训总结已保存到: {summary_path}")
        print(f"📁 所有培训材料位于: {self.training_dir}")

def main():
    """主函数"""
    try:
        training = InteractiveTraining()
        training.run_training_program()
    except Exception as e:
        print(f"❌ 培训程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()