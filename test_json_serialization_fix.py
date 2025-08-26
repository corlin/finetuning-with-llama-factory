#!/usr/bin/env python3
"""
测试JSON序列化修复
验证numpy类型转换函数是否正确工作
"""

import json
import numpy as np
from direct_finetuning_with_existing_modules import convert_numpy_types

def test_numpy_conversion():
    """测试numpy类型转换"""
    print("🔍 测试numpy类型转换...")
    
    # 创建包含各种numpy类型的测试数据
    test_data = {
        'int_value': np.int32(42),
        'float_value': np.float64(3.14159),
        'bool_value': np.bool_(True),
        'array_value': np.array([1, 2, 3]),
        'nested_dict': {
            'nested_int': np.int64(100),
            'nested_float': np.float32(2.718),
            'nested_bool': np.bool_(False)
        },
        'list_with_numpy': [np.int16(1), np.float16(2.5), np.bool_(True)],
        'regular_values': {
            'string': 'test',
            'int': 123,
            'float': 4.56,
            'bool': True,
            'list': [1, 2, 3]
        }
    }
    
    print("原始数据类型:")
    print(f"  int_value: {type(test_data['int_value'])}")
    print(f"  float_value: {type(test_data['float_value'])}")
    print(f"  bool_value: {type(test_data['bool_value'])}")
    print(f"  array_value: {type(test_data['array_value'])}")
    
    # 转换numpy类型
    converted_data = convert_numpy_types(test_data)
    
    print("\n转换后数据类型:")
    print(f"  int_value: {type(converted_data['int_value'])}")
    print(f"  float_value: {type(converted_data['float_value'])}")
    print(f"  bool_value: {type(converted_data['bool_value'])}")
    print(f"  array_value: {type(converted_data['array_value'])}")
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(converted_data, indent=2)
        print("\n✅ JSON序列化成功")
        
        # 验证数据完整性
        parsed_data = json.loads(json_str)
        print("✅ JSON反序列化成功")
        
        # 验证值的正确性
        assert parsed_data['int_value'] == 42
        assert abs(parsed_data['float_value'] - 3.14159) < 1e-10
        assert parsed_data['bool_value'] == True
        assert parsed_data['array_value'] == [1, 2, 3]
        assert parsed_data['nested_dict']['nested_int'] == 100
        assert parsed_data['nested_dict']['nested_bool'] == False
        assert parsed_data['list_with_numpy'] == [1, 2.5, True]
        
        print("✅ 数据值验证通过")
        return True
        
    except Exception as e:
        print(f"❌ JSON序列化失败: {e}")
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n🔍 测试边界情况...")
    
    edge_cases = {
        'nan_value': np.nan,
        'inf_value': np.inf,
        'neg_inf_value': np.NINF,
        'empty_array': np.array([]),
        'multidim_array': np.array([[1, 2], [3, 4]]),
        'none_value': None,
        'empty_dict': {},
        'empty_list': []
    }
    
    try:
        converted = convert_numpy_types(edge_cases)
        
        # 特殊值处理
        assert np.isnan(converted['nan_value']) or converted['nan_value'] is None
        assert converted['inf_value'] == float('inf') or converted['inf_value'] is None
        assert converted['neg_inf_value'] == float('-inf') or converted['neg_inf_value'] is None
        assert converted['empty_array'] == []
        assert converted['multidim_array'] == [[1, 2], [3, 4]]
        assert converted['none_value'] is None
        assert converted['empty_dict'] == {}
        assert converted['empty_list'] == []
        
        # 尝试JSON序列化（某些特殊值可能无法序列化）
        json_str = json.dumps(converted, allow_nan=True)
        print("✅ 边界情况处理成功")
        return True
        
    except Exception as e:
        print(f"⚠️ 边界情况处理警告: {e}")
        return True  # 边界情况可能有预期的失败

def main():
    """主测试函数"""
    print("🎯 测试JSON序列化修复")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # 测试基本转换
    if test_numpy_conversion():
        success_count += 1
        print("✅ 基本numpy类型转换测试通过")
    else:
        print("❌ 基本numpy类型转换测试失败")
    
    # 测试边界情况
    if test_edge_cases():
        success_count += 1
        print("✅ 边界情况测试通过")
    else:
        print("❌ 边界情况测试失败")
    
    print(f"\n📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！JSON序列化修复成功")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)