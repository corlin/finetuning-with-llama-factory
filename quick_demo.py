#!/usr/bin/env python3
"""
快速Checkpoint合并与专家评估演示

简化版演示程序，展示核心功能：
1. 模拟checkpoint合并过程
2. 加载真实QA数据
3. 执行专家评估
4. 生成评估报告

使用方法:
    uv run python quick_demo.py
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def load_qa_data(data_dir: str = "data/raw", max_items: int = 5) -> List[Dict[str, Any]]:
    """加载QA数据"""
    print(f"📊 从 {data_dir} 加载QA数据...")
    
    data_path = Path(data_dir)
    all_qa_items = []
    
    # 加载enhanced QA文件
    enhanced_files = list(data_path.glob("enhanced_QA*.md"))
    
    for file_path in enhanced_files:
        print(f"📖 处理文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取Q&A对
        qa_pattern = r'### Q(\d+):\s*(.+?)\n\n<thinking>.*?</thinking>\n\nA\1:\s*(.+?)(?=\n### Q|\n## |$)'
        matches = re.findall(qa_pattern, content, re.DOTALL)
        
        for match in matches:
            q_num, question, answer = match
            question = question.strip()
            answer = answer.strip()
            
            if question and answer:
                qa_item = {
                    "question_id": f"qa_{q_num}",
                    "question": question,
                    "reference_answer": answer,
                    "context": "密码应用标准GB/T 39786-2021",
                    "domain_tags": ["密码学", "信息安全", "国家标准"],
                    "difficulty_level": "intermediate"
                }
                all_qa_items.append(qa_item)
                
                if len(all_qa_items) >= max_items:
                    break
        
        if len(all_qa_items) >= max_items:
            break
    
    print(f"✅ 加载了 {len(all_qa_items)} 个QA项")
    return all_qa_items

def simulate_model_answers(qa_data: List[Dict[str, Any]]) -> List[str]:
    """模拟模型生成答案"""
    print("🤖 模拟模型生成答案...")
    
    model_answers = []
    for qa_item in qa_data:
        # 模拟不同质量的答案
        question = qa_item["question"]
        
        if "密码" in question or "加密" in question:
            answer = f"关于{question[:20]}...的问题，这涉及密码学的核心概念。密码技术是信息安全的重要基础，包括加密算法、数字签名、身份认证等多个方面。在实际应用中需要根据具体场景选择合适的密码方案。"
        elif "安全" in question:
            answer = f"针对{question[:20]}...的安全问题，需要从多个维度考虑：技术安全、管理安全、物理安全等。建立完善的安全体系需要综合运用各种安全技术和管理措施。"
        else:
            answer = f"这是一个关于{question[:15]}...的专业问题。根据相关标准和最佳实践，需要考虑技术可行性、安全性、成本效益等多个因素来制定合适的解决方案。"
        
        model_answers.append(answer)
        print(f"   ✅ {qa_item['question_id']}: {answer[:50]}...")
    
    return model_answers

def evaluate_answers(qa_data: List[Dict[str, Any]], model_answers: List[str]) -> Dict[str, Any]:
    """评估答案质量"""
    print("📊 执行专家评估...")
    
    results = []
    
    for qa_item, model_answer in zip(qa_data, model_answers):
        # 简化评估逻辑
        ref_words = set(qa_item["reference_answer"].lower().split())
        model_words = set(model_answer.lower().split())
        
        # 计算词汇重叠度
        overlap = len(ref_words & model_words)
        union = len(ref_words | model_words)
        similarity = overlap / union if union > 0 else 0
        
        # 检查专业术语
        domain_terms = ["密码", "加密", "安全", "算法", "认证", "标准"]
        domain_count = sum(1 for term in domain_terms if term in model_answer)
        
        # 计算各维度得分
        scores = {
            "semantic_similarity": min(0.9, similarity + 0.3),
            "domain_accuracy": min(0.9, 0.5 + domain_count * 0.1),
            "response_relevance": min(0.9, 0.6 + len(model_answer) / 200),
            "completeness": min(0.9, 0.5 + len(model_answer) / 300),
            "clarity": 0.8 if len(model_answer) > 50 else 0.6
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        result = {
            "question_id": qa_item["question_id"],
            "question": qa_item["question"],
            "reference_answer": qa_item["reference_answer"],
            "model_answer": model_answer,
            "overall_score": round(overall_score, 3),
            "dimension_scores": {k: round(v, 3) for k, v in scores.items()}
        }
        
        results.append(result)
        print(f"   📈 {qa_item['question_id']}: {overall_score:.3f}")
    
    # 计算统计信息
    scores = [r["overall_score"] for r in results]
    avg_score = sum(scores) / len(scores)
    
    return {
        "summary": {
            "total_evaluations": len(results),
            "average_score": round(avg_score, 3),
            "max_score": round(max(scores), 3),
            "min_score": round(min(scores), 3)
        },
        "individual_results": results
    }

def generate_report(evaluation_results: Dict[str, Any], output_dir: str = "quick_demo_output"):
    """生成评估报告"""
    print("📋 生成评估报告...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 生成JSON报告
    json_path = output_path / "evaluation_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # 生成HTML报告
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>快速评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .metric {{ background: #f9f9f9; padding: 10px; margin: 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Checkpoint合并与专家评估报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>📊 评估概要</h2>
    <div class="metric">总评估项目: {evaluation_results['summary']['total_evaluations']}</div>
    <div class="metric">平均得分: {evaluation_results['summary']['average_score']}</div>
    <div class="metric">最高得分: {evaluation_results['summary']['max_score']}</div>
    <div class="metric">最低得分: {evaluation_results['summary']['min_score']}</div>
    
    <h2>📈 详细结果</h2>
    <table>
        <tr><th>问题ID</th><th>问题</th><th>得分</th><th>语义相似性</th><th>领域准确性</th></tr>
    """
    
    for result in evaluation_results['individual_results']:
        html_content += f"""
        <tr>
            <td>{result['question_id']}</td>
            <td>{result['question'][:50]}...</td>
            <td>{result['overall_score']}</td>
            <td>{result['dimension_scores']['semantic_similarity']}</td>
            <td>{result['dimension_scores']['domain_accuracy']}</td>
        </tr>
        """
    
    html_content += """
    </table>
</body>
</html>
    """
    
    html_path = output_path / "evaluation_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 报告已生成:")
    print(f"   JSON: {json_path}")
    print(f"   HTML: {html_path}")
    
    return str(html_path)

def main():
    """主函数"""
    print("🚀 快速Checkpoint合并与专家评估演示")
    print("=" * 50)
    
    try:
        # 步骤1: 模拟checkpoint合并
        print("\n📋 步骤1: 模拟Checkpoint合并")
        print("✅ 模拟合并LoRA checkpoint到基座模型 (Qwen/Qwen3-4B-Thinking-2507)")
        print("✅ 合并完成，模型已准备就绪")
        
        # 步骤2: 加载QA数据
        print("\n📋 步骤2: 加载评估数据")
        qa_data = load_qa_data(max_items=5)
        
        if not qa_data:
            print("❌ 没有加载到QA数据")
            return
        
        # 步骤3: 生成模型答案
        print("\n📋 步骤3: 生成模型答案")
        model_answers = simulate_model_answers(qa_data)
        
        # 步骤4: 执行评估
        print("\n📋 步骤4: 执行专家评估")
        evaluation_results = evaluate_answers(qa_data, model_answers)
        
        # 步骤5: 生成报告
        print("\n📋 步骤5: 生成评估报告")
        report_path = generate_report(evaluation_results)
        
        # 总结
        print("\n🎉 演示完成!")
        print(f"📊 平均评估得分: {evaluation_results['summary']['average_score']}")
        print(f"📋 详细报告: {report_path}")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()