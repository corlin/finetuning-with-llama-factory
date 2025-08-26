#!/usr/bin/env python3
"""
验证生成的LLaMA Factory训练数据格式
"""

import json
import os
from pathlib import Path

def validate_direct_training_format(data_file: str) -> dict:
    """验证直接训练数据格式"""
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return {"valid": False, "error": "数据不是列表格式"}
        
        if len(data) == 0:
            return {"valid": False, "error": "数据为空"}
        
        # 检查必需字段
        required_fields = ["instruction", "input", "output", "system", "history"]
        
        valid_count = 0
        thinking_count = 0
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return {"valid": False, "error": f"第{i+1}项不是字典格式"}
            
            # 检查必需字段
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                return {"valid": False, "error": f"第{i+1}项缺少字段: {missing_fields}"}
            
            # 检查thinking标签
            if "<thinking>" in item["output"] and "</thinking>" in item["output"]:
                thinking_count += 1
            
            valid_count += 1
        
        return {
            "valid": True,
            "total_items": len(data),
            "valid_items": valid_count,
            "thinking_items": thinking_count,
            "thinking_ratio": thinking_count / len(data) if len(data) > 0 else 0
        }
        
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"JSON格式错误: {e}"}
    except Exception as e:
        return {"valid": False, "error": f"验证失败: {e}"}

def main():
    """主函数"""
    print("=== LLaMA Factory训练数据验证 ===\n")
    
    data_dir = "data/processed"
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ 在 {data_dir} 中未找到JSON文件")
        return
    
    print(f"📁 找到 {len(json_files)} 个JSON文件\n")
    
    total_items = 0
    total_thinking_items = 0
    
    for json_file in sorted(json_files):
        file_path = os.path.join(data_dir, json_file)
        print(f"🔍 验证文件: {json_file}")
        
        result = validate_direct_training_format(file_path)
        
        if result["valid"]:
            print(f"   ✅ 格式正确")
            print(f"   📊 数据项数量: {result['total_items']}")
            print(f"   🧠 包含thinking: {result['thinking_items']} 项 ({result['thinking_ratio']:.1%})")
            
            total_items += result['total_items']
            total_thinking_items += result['thinking_items']
            
            # 显示样例
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:
                    sample = data[0]
                    print(f"   📝 样例instruction: {sample['instruction'][:50]}...")
                    print(f"   🎯 样例system: {sample['system'][:50]}...")
                    print(f"   💭 包含thinking标签: {'<thinking>' in sample['output']}")
        else:
            print(f"   ❌ 验证失败: {result['error']}")
        
        print()
    
    # 总体统计
    print("=== 总体统计 ===")
    print(f"📈 总数据项: {total_items}")
    print(f"🧠 thinking项: {total_thinking_items}")
    print(f"📊 thinking覆盖率: {total_thinking_items/total_items:.1%}" if total_items > 0 else "📊 thinking覆盖率: 0%")
    
    # 验证主训练文件
    main_file = os.path.join(data_dir, "thinking_training_data.json")
    if os.path.exists(main_file):
        print(f"\n🎯 主训练文件验证:")
        result = validate_direct_training_format(main_file)
        if result["valid"]:
            print(f"   ✅ 主文件格式正确，包含 {result['total_items']} 个训练样例")
            print(f"   🧠 所有样例都包含thinking: {result['thinking_ratio'] == 1.0}")
        else:
            print(f"   ❌ 主文件验证失败: {result['error']}")
    
    print(f"\n🎉 验证完成！数据已准备好用于LLaMA Factory训练")

if __name__ == "__main__":
    main()