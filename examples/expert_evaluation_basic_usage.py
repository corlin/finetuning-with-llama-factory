#!/usr/bin/env python3
"""
专家评估系统基础使用示例

本脚本展示了专家评估系统的基本使用方法，适合初学者快速上手。

使用方法:
    uv run python examples/expert_evaluation_basic_usage.py

作者: 专家评估系统开发团队
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.expert_evaluation.engine import ExpertEvaluationEngine
from src.expert_evaluation.config import ExpertEvaluationConfig

def basic_usage_example():
    """基础使用示例"""
    
    print("🚀 专家评估系统基础使用示例")
    print("=" * 50)
    
    # 1. 创建简单配置
    print("\n📋 步骤1: 创建配置")
    config_dict = {
        "model": {
            "device": "auto",
            "batch_size": 1
        },
        "evaluation": {
            "dimensions": [
                "semantic_similarity",
                "domain_accuracy"
            ],
            "weights": {
                "semantic_similarity": 0.6,
                "domain_accuracy": 0.4
            }
        }
    }
    
    config = ExpertEvaluationConfig.from_dict(config_dict)
    print("✅ 配置创建成功")
    
    # 2. 初始化评估引擎
    print("\n🤖 步骤2: 初始化评估引擎")
    engine = ExpertEvaluationEngine(config)
    print("✅ 引擎初始化成功")
    
    # 3. 准备QA数据
    print("\n📊 步骤3: 准备QA数据")
    qa_item = {
        "question_id": "example_001",
        "question": "什么是机器学习？",
        "reference_answer": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
        "model_answer": "机器学习是AI的一部分，让计算机可以从数据中自动学习模式和规律。",
        "domain_tags": ["人工智能", "机器学习"],
        "difficulty_level": "beginner"
    }
    print("✅ QA数据准备完成")
    
    # 4. 执行评估
    print("\n🔍 步骤4: 执行评估")
    result = engine.evaluate_single_qa(qa_item)
    
    # 5. 显示结果
    print("\n📊 评估结果:")
    print(f"🎯 总体得分: {result.overall_score:.3f}")
    print(f"📈 语义相似性: {result.dimension_scores.get('semantic_similarity', 0):.3f}")
    print(f"📈 领域准确性: {result.dimension_scores.get('domain_accuracy', 0):.3f}")
    
    if result.improvement_suggestions:
        print("\n💡 改进建议:")
        for suggestion in result.improvement_suggestions:
            print(f"   - {suggestion}")
    
    print("\n✅ 基础使用示例完成!")

if __name__ == "__main__":
    try:
        basic_usage_example()
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()