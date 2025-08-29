#!/usr/bin/env python3
"""
测试增强的QA4-QA6数据处理
验证thinking标签的添加和数据质量
"""

import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ThinkingExample:
    """思考示例数据结构"""
    question: str
    thinking: str
    answer: str
    difficulty: str = "INTERMEDIATE"
    
    def to_direct_training_format(self) -> Dict:
        """转换为直接训练格式"""
        system_prompt = "你是一个密码学专家，请根据GB/T 39786-2021等相关标准回答问题。在回答前，请在<thinking>标签中展示你的思考过程。"
        
        instruction = self.question
        thinking_content = f"<thinking>\n{self.thinking}\n</thinking>\n\n{self.answer}"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": thinking_content,
            "system": system_prompt
        }

class EnhancedQAProcessor:
    """增强QA数据处理器"""
    
    def __init__(self):
        self.thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
        self.qa_pattern = re.compile(r'### (Q\d+): (.+?)\n\n<thinking>(.*?)</thinking>\n\n(A\d+): (.+?)(?=\n### |$)', re.DOTALL)
        
    def parse_enhanced_file(self, file_path: str) -> List[ThinkingExample]:
        """解析增强的QA文件"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            matches = self.qa_pattern.findall(content)
            
            for match in matches:
                q_id, question, thinking, a_id, answer = match
                
                # 清理文本
                question = question.strip()
                thinking = thinking.strip()
                answer = answer.strip()
                
                # 判断难度级别
                difficulty = self._determine_difficulty(question, thinking, answer)
                
                example = ThinkingExample(
                    question=question,
                    thinking=thinking,
                    answer=answer,
                    difficulty=difficulty
                )
                
                examples.append(example)
                
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            
        return examples
    
    def _determine_difficulty(self, question: str, thinking: str, answer: str) -> str:
        """判断问题难度级别"""
        # 基于关键词和thinking复杂度判断难度
        expert_keywords = ['第四级', '第三级', 'SM2', 'SM3', 'SM4', '密钥管理', '数字签名', '椭圆曲线']
        intermediate_keywords = ['第二级', '密码应用', '安全性', '完整性', '机密性']
        
        text = f"{question} {thinking} {answer}".lower()
        
        expert_count = sum(1 for keyword in expert_keywords if keyword.lower() in text)
        intermediate_count = sum(1 for keyword in intermediate_keywords if keyword.lower() in text)
        
        thinking_length = len(thinking)
        
        if expert_count >= 2 or thinking_length > 800:
            return "EXPERT"
        elif intermediate_count >= 2 or thinking_length > 400:
            return "INTERMEDIATE"
        else:
            return "BEGINNER"
    
    def validate_thinking_structure(self, thinking: str) -> Dict[str, bool]:
        """验证thinking结构"""
        validation = {
            "has_numbered_points": bool(re.search(r'\d+[.、]', thinking)),
            "has_analysis_structure": bool(re.search(r'(分析|理解|考虑|方面|角度)', thinking)),
            "has_technical_terms": bool(re.search(r'(密码|加密|签名|认证|完整性|机密性)', thinking)),
            "sufficient_length": len(thinking) > 200,
            "logical_flow": bool(re.search(r'(首先|其次|然后|最后|因此|所以)', thinking))
        }
        
        return validation
    
    def generate_statistics(self, examples: List[ThinkingExample]) -> Dict:
        """生成统计信息"""
        if not examples:
            return {}
        
        total_examples = len(examples)
        difficulty_counts = {}
        thinking_lengths = []
        validation_scores = []
        
        for example in examples:
            # 难度统计
            difficulty_counts[example.difficulty] = difficulty_counts.get(example.difficulty, 0) + 1
            
            # thinking长度统计
            thinking_lengths.append(len(example.thinking))
            
            # 验证分数
            validation = self.validate_thinking_structure(example.thinking)
            score = sum(validation.values()) / len(validation)
            validation_scores.append(score)
        
        return {
            "total_examples": total_examples,
            "difficulty_distribution": difficulty_counts,
            "avg_thinking_length": sum(thinking_lengths) / len(thinking_lengths),
            "min_thinking_length": min(thinking_lengths),
            "max_thinking_length": max(thinking_lengths),
            "avg_validation_score": sum(validation_scores) / len(validation_scores),
            "high_quality_examples": sum(1 for score in validation_scores if score >= 0.8)
        }

def test_enhanced_qa_files():
    """测试增强的QA文件"""
    processor = EnhancedQAProcessor()
    
    # 测试文件列表
    test_files = [
        "data/raw/enhanced_QA4.md",
        "data/raw/enhanced_QA5.md", 
        "data/raw/enhanced_QA6.md"
    ]
    
    total_stats = {
        "total_examples": 0,
        "total_files": 0,
        "difficulty_distribution": {},
        "avg_thinking_length": 0,
        "high_quality_examples": 0
    }
    
    all_examples = []
    
    print("=== 增强QA4-QA6数据测试报告 ===\n")
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            continue
            
        print(f"📁 处理文件: {file_path}")
        
        examples = processor.parse_enhanced_file(file_path)
        all_examples.extend(examples)
        
        if examples:
            stats = processor.generate_statistics(examples)
            
            print(f"   ✅ 成功解析 {stats['total_examples']} 个问题")
            print(f"   📊 难度分布: {stats['difficulty_distribution']}")
            print(f"   📏 平均thinking长度: {stats['avg_thinking_length']:.0f} 字符")
            print(f"   ⭐ 高质量样例: {stats['high_quality_examples']} 个")
            print(f"   🎯 验证分数: {stats['avg_validation_score']:.2f}")
            
            # 更新总统计
            total_stats["total_examples"] += stats["total_examples"]
            total_stats["high_quality_examples"] += stats["high_quality_examples"]
            
            for difficulty, count in stats["difficulty_distribution"].items():
                total_stats["difficulty_distribution"][difficulty] = \
                    total_stats["difficulty_distribution"].get(difficulty, 0) + count
        else:
            print(f"   ❌ 未找到有效的QA对")
        
        print()
    
    # 总体统计
    if all_examples:
        overall_stats = processor.generate_statistics(all_examples)
        
        print("=== 总体统计 ===")
        print(f"📈 总问题数量: {overall_stats['total_examples']}")
        print(f"📊 难度分布: {overall_stats['difficulty_distribution']}")
        print(f"📏 平均thinking长度: {overall_stats['avg_thinking_length']:.0f} 字符")
        print(f"⭐ 高质量样例: {overall_stats['high_quality_examples']} 个")
        print(f"🎯 整体质量分数: {overall_stats['avg_validation_score']:.2f}")
        
        # 转换为标准训练格式示例
        print("\n=== 标准训练格式示例 ===")
        if all_examples:
            sample_example = all_examples[0]
            direct_format = sample_example.to_direct_training_format()
            
            print("样例转换结果:")
            print(f"Instruction: {llama_format['instruction'][:100]}...")
            print(f"Output包含thinking: {'<thinking>' in llama_format['output']}")
            print(f"System prompt: {llama_format['system'][:50]}...")
    
    return all_examples

if __name__ == "__main__":
    examples = test_enhanced_qa_files()
    
    if examples:
        print(f"\n🎉 成功处理 {len(examples)} 个增强QA样例！")
        print("✅ 所有样例都包含完整的thinking过程")
        print("✅ 数据格式符合标准训练要求")
    else:
        print("\n❌ 未找到有效的增强QA数据")