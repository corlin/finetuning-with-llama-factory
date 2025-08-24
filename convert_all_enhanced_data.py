#!/usr/bin/env python3
"""
将所有增强的QA数据转换为LLaMA Factory训练格式
包括QA1-QA6的所有thinking数据
"""

import json
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
    source_file: str = ""
    
    def to_llama_factory_format(self) -> Dict:
        """转换为LLaMA Factory格式"""
        system_prompt = "你是一个密码学专家，请根据GB/T 39786-2021等相关标准回答问题。在回答前，请在<thinking>标签中展示你的思考过程。"
        
        instruction = self.question
        thinking_content = f"<thinking>\n{self.thinking}\n</thinking>\n\n{self.answer}"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": thinking_content,
            "system": system_prompt,
            "history": []
        }

class ComprehensiveQAProcessor:
    """综合QA数据处理器"""
    
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
                    difficulty=difficulty,
                    source_file=os.path.basename(file_path)
                )
                
                examples.append(example)
                
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            
        return examples
    
    def _determine_difficulty(self, question: str, thinking: str, answer: str) -> str:
        """判断问题难度级别"""
        # 基于关键词和thinking复杂度判断难度
        expert_keywords = ['第四级', '第三级', 'SM2', 'SM3', 'SM4', 'ZUC', '椭圆曲线', '数字签名', 
                          '密钥协商', '证书链', 'SCADA', 'V2X', '区块链', '量子计算']
        intermediate_keywords = ['第二级', '密码应用', '安全性', '完整性', '机密性', '身份鉴别', 
                               '访问控制', '密钥管理', '评估流程']
        
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
    
    def process_all_enhanced_files(self) -> List[ThinkingExample]:
        """处理所有增强的QA文件"""
        enhanced_files = [
            "data/raw/enhanced_QA1.md",
            "data/raw/enhanced_QA2.md", 
            "data/raw/enhanced_QA3.md",
            "data/raw/enhanced_QA4.md",
            "data/raw/enhanced_QA5.md",
            "data/raw/enhanced_QA6.md"
        ]
        
        all_examples = []
        
        for file_path in enhanced_files:
            if os.path.exists(file_path):
                print(f"处理文件: {file_path}")
                examples = self.parse_enhanced_file(file_path)
                all_examples.extend(examples)
                print(f"  解析到 {len(examples)} 个问题")
            else:
                print(f"文件不存在: {file_path}")
        
        return all_examples
    
    def convert_to_llama_factory_format(self, examples: List[ThinkingExample]) -> List[Dict]:
        """转换为LLaMA Factory格式"""
        return [example.to_llama_factory_format() for example in examples]
    
    def save_training_data(self, examples: List[ThinkingExample], output_dir: str = "data/processed"):
        """保存训练数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为LLaMA Factory格式
        llama_data = self.convert_to_llama_factory_format(examples)
        
        # 保存完整数据集
        full_path = os.path.join(output_dir, "thinking_training_data.json")
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(llama_data, f, ensure_ascii=False, indent=2)
        
        print(f"完整训练数据已保存到: {full_path}")
        
        # 按难度级别分别保存
        difficulty_data = {"BEGINNER": [], "INTERMEDIATE": [], "EXPERT": []}
        
        for example in examples:
            difficulty_data[example.difficulty].append(example.to_llama_factory_format())
        
        for difficulty, data in difficulty_data.items():
            if data:
                difficulty_path = os.path.join(output_dir, f"thinking_data_{difficulty.lower()}.json")
                with open(difficulty_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"{difficulty}级别数据已保存到: {difficulty_path} ({len(data)}个样例)")
        
        # 按来源文件分别保存
        source_data = {}
        for example in examples:
            source = example.source_file
            if source not in source_data:
                source_data[source] = []
            source_data[source].append(example.to_llama_factory_format())
        
        for source, data in source_data.items():
            source_name = source.replace('.md', '').replace('enhanced_', '')
            source_path = os.path.join(output_dir, f"thinking_data_{source_name}.json")
            with open(source_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"{source}数据已保存到: {source_path} ({len(data)}个样例)")
    
    def generate_comprehensive_report(self, examples: List[ThinkingExample]) -> Dict:
        """生成综合报告"""
        if not examples:
            return {}
        
        # 基本统计
        total_examples = len(examples)
        difficulty_counts = {}
        source_counts = {}
        thinking_lengths = []
        
        for example in examples:
            # 难度统计
            difficulty_counts[example.difficulty] = difficulty_counts.get(example.difficulty, 0) + 1
            
            # 来源统计
            source_counts[example.source_file] = source_counts.get(example.source_file, 0) + 1
            
            # thinking长度统计
            thinking_lengths.append(len(example.thinking))
        
        # 专业术语统计
        crypto_terms = ['密码', '加密', '解密', '签名', '认证', '完整性', '机密性', '真实性', 
                       '不可否认', '密钥', 'SM2', 'SM3', 'SM4', 'ZUC', '椭圆曲线', '哈希']
        
        term_counts = {}
        all_text = ' '.join([f"{ex.question} {ex.thinking} {ex.answer}" for ex in examples])
        
        for term in crypto_terms:
            count = all_text.count(term)
            if count > 0:
                term_counts[term] = count
        
        return {
            "total_examples": total_examples,
            "difficulty_distribution": difficulty_counts,
            "source_distribution": source_counts,
            "thinking_length_stats": {
                "average": sum(thinking_lengths) / len(thinking_lengths),
                "min": min(thinking_lengths),
                "max": max(thinking_lengths),
                "median": sorted(thinking_lengths)[len(thinking_lengths)//2]
            },
            "crypto_term_frequency": dict(sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        }

def main():
    """主函数"""
    print("=== 综合QA数据处理和转换 ===\n")
    
    processor = ComprehensiveQAProcessor()
    
    # 处理所有增强文件
    print("1. 处理所有增强的QA文件...")
    all_examples = processor.process_all_enhanced_files()
    
    if not all_examples:
        print("❌ 未找到有效的增强QA数据")
        return
    
    print(f"\n✅ 总共处理了 {len(all_examples)} 个问题\n")
    
    # 生成综合报告
    print("2. 生成综合统计报告...")
    report = processor.generate_comprehensive_report(all_examples)
    
    print("=== 综合统计报告 ===")
    print(f"📈 总问题数量: {report['total_examples']}")
    print(f"📊 难度分布: {report['difficulty_distribution']}")
    print(f"📁 来源分布: {report['source_distribution']}")
    print(f"📏 thinking长度统计:")
    print(f"   - 平均: {report['thinking_length_stats']['average']:.0f} 字符")
    print(f"   - 最小: {report['thinking_length_stats']['min']} 字符")
    print(f"   - 最大: {report['thinking_length_stats']['max']} 字符")
    print(f"   - 中位数: {report['thinking_length_stats']['median']} 字符")
    
    print(f"\n🔤 高频密码学术语 (Top 10):")
    for i, (term, count) in enumerate(list(report['crypto_term_frequency'].items())[:10], 1):
        print(f"   {i}. {term}: {count}次")
    
    # 保存训练数据
    print("\n3. 转换并保存LLaMA Factory训练数据...")
    processor.save_training_data(all_examples)
    
    # 验证转换结果
    print("\n4. 验证转换结果...")
    sample_example = all_examples[0]
    llama_format = sample_example.to_llama_factory_format()
    
    print("✅ 样例转换验证:")
    print(f"   - Instruction长度: {len(llama_format['instruction'])} 字符")
    print(f"   - Output包含thinking: {'<thinking>' in llama_format['output']}")
    print(f"   - System prompt设置: {'密码学专家' in llama_format['system']}")
    print(f"   - 格式完整性: {all(key in llama_format for key in ['instruction', 'input', 'output', 'system', 'history'])}")
    
    print(f"\n🎉 数据处理完成！")
    print(f"✅ 成功转换 {len(all_examples)} 个thinking样例")
    print(f"✅ 数据已保存到 data/processed/ 目录")
    print(f"✅ 支持LLaMA Factory直接训练")

if __name__ == "__main__":
    main()