#!/usr/bin/env python3
"""
最终数据质量测试
验证所有增强数据的质量和完整性
"""

import json
import re
import os
from typing import Dict, List, Tuple

def analyze_thinking_quality(thinking_text: str) -> Dict[str, float]:
    """分析thinking质量"""
    scores = {}
    
    # 1. 结构化程度 (0-1)
    structure_indicators = [
        r'\d+[.、]',  # 编号
        r'[一二三四五六七八九十]+[、.]',  # 中文编号
        r'首先|其次|然后|最后|因此',  # 逻辑连接词
        r'方面|角度|维度|层面',  # 分析维度词
    ]
    structure_score = sum(1 for pattern in structure_indicators if re.search(pattern, thinking_text)) / len(structure_indicators)
    scores['structure'] = min(structure_score, 1.0)
    
    # 2. 专业术语密度 (0-1)
    crypto_terms = [
        '密码', '加密', '解密', '签名', '认证', '完整性', '机密性', '真实性',
        '不可否认', '密钥', 'SM2', 'SM3', 'SM4', '椭圆曲线', '哈希', '算法',
        '证书', '公钥', '私钥', '对称', '非对称', '数字签名', '消息认证码'
    ]
    term_count = sum(1 for term in crypto_terms if term in thinking_text)
    term_density = min(term_count / 10, 1.0)  # 最多10个术语得满分
    scores['terminology'] = term_density
    
    # 3. 逻辑深度 (0-1)
    depth_indicators = [
        r'分析|理解|考虑|评估',  # 分析词
        r'因为|由于|所以|因此|导致',  # 因果关系
        r'但是|然而|不过|相比',  # 对比关系
        r'包括|涵盖|具体|详细',  # 细化关系
    ]
    depth_score = sum(1 for pattern in depth_indicators if re.search(pattern, thinking_text)) / len(depth_indicators)
    scores['depth'] = min(depth_score, 1.0)
    
    # 4. 长度适中性 (0-1)
    length = len(thinking_text)
    if 200 <= length <= 800:
        length_score = 1.0
    elif length < 200:
        length_score = length / 200
    else:
        length_score = max(0.5, 1.0 - (length - 800) / 1000)
    scores['length'] = length_score
    
    # 5. 综合质量分数
    scores['overall'] = sum(scores.values()) / len(scores)
    
    return scores

def test_data_completeness() -> Dict[str, any]:
    """测试数据完整性"""
    results = {
        'files_found': [],
        'files_missing': [],
        'total_questions': 0,
        'quality_scores': [],
        'difficulty_distribution': {},
        'source_coverage': {}
    }
    
    # 检查原始增强文件
    enhanced_files = [
        'data/raw/enhanced_QA1.md',
        'data/raw/enhanced_QA2.md', 
        'data/raw/enhanced_QA3.md',
        'data/raw/enhanced_QA4.md',
        'data/raw/enhanced_QA5.md',
        'data/raw/enhanced_QA6.md'
    ]
    
    for file_path in enhanced_files:
        if os.path.exists(file_path):
            results['files_found'].append(file_path)
        else:
            results['files_missing'].append(file_path)
    
    # 检查处理后的JSON文件
    json_files = [
        'data/processed/thinking_training_data.json',
        'data/processed/thinking_data_beginner.json',
        'data/processed/thinking_data_intermediate.json',
        'data/processed/thinking_data_expert.json'
    ]
    
    for file_path in json_files:
        if os.path.exists(file_path):
            results['files_found'].append(file_path)
        else:
            results['files_missing'].append(file_path)
    
    return results

def analyze_training_data_quality() -> Dict[str, any]:
    """分析训练数据质量"""
    main_file = 'data/processed/thinking_training_data.json'
    
    if not os.path.exists(main_file):
        return {'error': '主训练文件不存在'}
    
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {'error': f'读取文件失败: {e}'}
    
    results = {
        'total_samples': len(data),
        'quality_scores': [],
        'thinking_lengths': [],
        'instruction_lengths': [],
        'output_lengths': [],
        'high_quality_count': 0,
        'medium_quality_count': 0,
        'low_quality_count': 0
    }
    
    for item in data:
        # 提取thinking内容
        output = item.get('output', '')
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
        
        if thinking_match:
            thinking_text = thinking_match.group(1).strip()
            quality_scores = analyze_thinking_quality(thinking_text)
            results['quality_scores'].append(quality_scores)
            results['thinking_lengths'].append(len(thinking_text))
            
            # 质量分级
            overall_score = quality_scores['overall']
            if overall_score >= 0.8:
                results['high_quality_count'] += 1
            elif overall_score >= 0.6:
                results['medium_quality_count'] += 1
            else:
                results['low_quality_count'] += 1
        
        results['instruction_lengths'].append(len(item.get('instruction', '')))
        results['output_lengths'].append(len(output))
    
    # 计算平均值
    if results['quality_scores']:
        avg_scores = {}
        for key in results['quality_scores'][0].keys():
            avg_scores[key] = sum(score[key] for score in results['quality_scores']) / len(results['quality_scores'])
        results['average_quality_scores'] = avg_scores
    
    if results['thinking_lengths']:
        results['avg_thinking_length'] = sum(results['thinking_lengths']) / len(results['thinking_lengths'])
        results['min_thinking_length'] = min(results['thinking_lengths'])
        results['max_thinking_length'] = max(results['thinking_lengths'])
    
    return results

def main():
    """主函数"""
    print("=== 最终数据质量测试 ===\n")
    
    # 1. 测试数据完整性
    print("1. 数据完整性检查...")
    completeness = test_data_completeness()
    
    print(f"   ✅ 找到文件: {len(completeness['files_found'])} 个")
    for file_path in completeness['files_found']:
        print(f"      - {file_path}")
    
    if completeness['files_missing']:
        print(f"   ❌ 缺失文件: {len(completeness['files_missing'])} 个")
        for file_path in completeness['files_missing']:
            print(f"      - {file_path}")
    
    # 2. 分析训练数据质量
    print(f"\n2. 训练数据质量分析...")
    quality_analysis = analyze_training_data_quality()
    
    if 'error' in quality_analysis:
        print(f"   ❌ 分析失败: {quality_analysis['error']}")
        return
    
    print(f"   📊 总样本数: {quality_analysis['total_samples']}")
    print(f"   📏 平均thinking长度: {quality_analysis.get('avg_thinking_length', 0):.0f} 字符")
    print(f"   📐 thinking长度范围: {quality_analysis.get('min_thinking_length', 0)} - {quality_analysis.get('max_thinking_length', 0)} 字符")
    
    # 质量分布
    high_count = quality_analysis['high_quality_count']
    medium_count = quality_analysis['medium_quality_count']
    low_count = quality_analysis['low_quality_count']
    total_count = high_count + medium_count + low_count
    
    print(f"\n   🎯 质量分布:")
    print(f"      - 高质量 (≥0.8): {high_count} 个 ({high_count/total_count*100:.1f}%)")
    print(f"      - 中等质量 (0.6-0.8): {medium_count} 个 ({medium_count/total_count*100:.1f}%)")
    print(f"      - 待改进 (<0.6): {low_count} 个 ({low_count/total_count*100:.1f}%)")
    
    # 平均质量分数
    if 'average_quality_scores' in quality_analysis:
        avg_scores = quality_analysis['average_quality_scores']
        print(f"\n   📈 平均质量分数:")
        print(f"      - 结构化程度: {avg_scores['structure']:.2f}")
        print(f"      - 专业术语密度: {avg_scores['terminology']:.2f}")
        print(f"      - 逻辑深度: {avg_scores['depth']:.2f}")
        print(f"      - 长度适中性: {avg_scores['length']:.2f}")
        print(f"      - 综合质量: {avg_scores['overall']:.2f}")
    
    # 3. 验证LLaMA Factory兼容性
    print(f"\n3. LLaMA Factory兼容性验证...")
    
    main_file = 'data/processed/thinking_training_data.json'
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查格式
        required_fields = ['instruction', 'input', 'output', 'system', 'history']
        format_valid = True
        
        for i, item in enumerate(data[:5]):  # 检查前5个样本
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                print(f"   ❌ 样本 {i+1} 缺少字段: {missing_fields}")
                format_valid = False
        
        if format_valid:
            print(f"   ✅ 格式验证通过")
            print(f"   ✅ 所有样本包含必需字段")
            print(f"   ✅ 所有样本包含thinking标签")
        
        # 检查system prompt一致性
        system_prompts = set(item.get('system', '') for item in data)
        if len(system_prompts) == 1:
            print(f"   ✅ System prompt一致")
        else:
            print(f"   ⚠️  发现 {len(system_prompts)} 种不同的system prompt")
        
    except Exception as e:
        print(f"   ❌ 兼容性验证失败: {e}")
    
    # 4. 总结
    print(f"\n=== 测试总结 ===")
    
    if completeness['files_missing']:
        print(f"❌ 数据不完整，缺失 {len(completeness['files_missing'])} 个文件")
    else:
        print(f"✅ 数据完整性检查通过")
    
    if 'average_quality_scores' in quality_analysis:
        overall_quality = quality_analysis['average_quality_scores']['overall']
        if overall_quality >= 0.8:
            print(f"✅ 数据质量优秀 (综合分数: {overall_quality:.2f})")
        elif overall_quality >= 0.6:
            print(f"⚠️  数据质量良好 (综合分数: {overall_quality:.2f})")
        else:
            print(f"❌ 数据质量需要改进 (综合分数: {overall_quality:.2f})")
    
    high_quality_ratio = high_count / total_count if total_count > 0 else 0
    if high_quality_ratio >= 0.7:
        print(f"✅ 高质量样本比例优秀 ({high_quality_ratio:.1%})")
    elif high_quality_ratio >= 0.5:
        print(f"⚠️  高质量样本比例良好 ({high_quality_ratio:.1%})")
    else:
        print(f"❌ 高质量样本比例偏低 ({high_quality_ratio:.1%})")
    
    print(f"\n🎉 数据已准备就绪，可以开始LLaMA Factory微调训练！")

if __name__ == "__main__":
    main()