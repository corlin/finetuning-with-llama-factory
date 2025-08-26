#!/usr/bin/env python3
"""
测试Qwen3-4B-Thinking模型的专用脚本
验证模型加载、thinking格式处理和基本推理
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 添加src路径
sys.path.append('src')

def test_model_availability():
    """测试模型是否可用"""
    print("🔍 测试Qwen3-4B-Thinking模型可用性...")
    
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    try:
        # 尝试加载tokenizer
        print(f"  正在加载tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ Tokenizer加载成功")
        
        # 检查特殊token
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  PAD token: {tokenizer.pad_token}")
        print(f"  词汇表大小: {len(tokenizer)}")
        
        return True
    except Exception as e:
        print(f"❌ 模型不可用: {e}")
        return False

def test_thinking_data_format():
    """测试thinking数据格式处理"""
    print("\n🔍 测试thinking数据格式处理...")
    
    try:
        # 读取训练数据
        data_path = "final_demo_output/data/crypto_qa_dataset_train.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 分析thinking数据
        thinking_samples = []
        for item in data:
            output = item.get('output', '')
            if '<thinking>' in output and '</thinking>' in output:
                thinking_samples.append(item)
        
        print(f"✅ 找到 {len(thinking_samples)} 个thinking样本")
        
        if thinking_samples:
            sample = thinking_samples[0]
            print("  样本示例:")
            print(f"    指令: {sample['instruction'][:50]}...")
            
            output = sample['output']
            # 提取thinking部分
            thinking_start = output.find('<thinking>')
            thinking_end = output.find('</thinking>') + len('</thinking>')
            
            if thinking_start != -1 and thinking_end != -1:
                thinking_part = output[thinking_start:thinking_end]
                response_part = output[thinking_end:].strip()
                
                print(f"    Thinking部分长度: {len(thinking_part)} 字符")
                print(f"    响应部分长度: {len(response_part)} 字符")
                print(f"    Thinking预览: {thinking_part[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ thinking数据格式测试失败: {e}")
        return False

def test_memory_requirements():
    """测试内存需求"""
    print("\n🔍 测试内存需求...")
    
    try:
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，跳过GPU内存测试")
            return True
        
        # 检查GPU内存
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory // 1024**2
            
            print(f"  GPU {i}: {gpu_name}")
            print(f"    总内存: {total_memory}MB")
            
            # 估算4B模型需要的内存
            # 4B参数 * 2字节(fp16) ≈ 8GB
            # 加上激活值、梯度等，大约需要12-16GB
            required_memory = 12000  # MB
            
            if total_memory >= required_memory:
                print(f"    ✅ 内存充足 ({total_memory}MB >= {required_memory}MB)")
            else:
                print(f"    ⚠️ 内存可能不足 ({total_memory}MB < {required_memory}MB)")
                print(f"    建议: 使用梯度检查点、更小批次大小或模型并行")
        
        return True
    except Exception as e:
        print(f"❌ 内存需求测试失败: {e}")
        return False

def test_tokenizer_with_thinking():
    """测试tokenizer处理thinking格式"""
    print("\n🔍 测试tokenizer处理thinking格式...")
    
    try:
        model_name = "Qwen/Qwen3-4B-Thinking-2507"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 测试thinking格式文本
        test_text = """<|im_start|>system
你是一个专业的密码学专家，请仔细思考后回答问题。<|im_end|>
<|im_start|>user
什么是AES加密算法？<|im_end|>
<|im_start|>assistant
<thinking>
这是一个关于对称加密算法的问题。AES是高级加密标准，我需要从以下几个方面来回答：
1. AES的全称和历史
2. AES的基本特征
3. AES的工作原理
</thinking>

AES（Advanced Encryption Standard，高级加密标准）是一种对称加密算法。<|im_end|>"""
        
        # 分词测试
        tokens = tokenizer(test_text, return_tensors="pt")
        
        print(f"✅ 分词成功")
        print(f"  输入长度: {len(test_text)} 字符")
        print(f"  Token数量: {tokens['input_ids'].shape[1]}")
        
        # 检查特殊token是否正确处理
        decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False)
        
        if '<thinking>' in decoded and '</thinking>' in decoded:
            print("✅ thinking标签保持完整")
        else:
            print("⚠️ thinking标签可能被处理")
        
        return True
    except Exception as e:
        print(f"❌ tokenizer测试失败: {e}")
        return False

def test_model_loading_with_optimization():
    """测试优化配置下的模型加载"""
    print("\n🔍 测试优化配置下的模型加载...")
    
    try:
        model_name = "Qwen/Qwen3-4B-Thinking-2507"
        
        print("  正在加载模型（这可能需要几分钟）...")
        
        # 使用内存优化配置
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用fp16减少内存
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 减少CPU内存使用
        )
        
        print(f"✅ 模型加载成功")
        print(f"  参数量: {model.num_parameters():,}")
        print(f"  模型大小: {model.num_parameters() * 2 / 1024**3:.2f}GB (fp16)")
        
        # 启用梯度检查点
        model.gradient_checkpointing_enable()
        print("✅ 梯度检查点已启用")
        
        # 检查模型设备分布
        if torch.cuda.is_available():
            device_map = {}
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in device_map:
                    device_map[device] = 0
                device_map[device] += param.numel()
            
            print("  模型设备分布:")
            for device, param_count in device_map.items():
                print(f"    {device}: {param_count:,} 参数")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 Qwen3-4B-Thinking模型专用测试")
    print("=" * 50)
    
    tests = [
        ("模型可用性", test_model_availability),
        ("Thinking数据格式", test_thinking_data_format),
        ("内存需求", test_memory_requirements),
        ("Tokenizer处理", test_tokenizer_with_thinking),
        ("优化模型加载", test_model_loading_with_optimization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed >= len(results) * 0.8:  # 80%通过率
        print("🎉 Qwen3-4B-Thinking模型测试基本通过！")
        print("\n📝 下一步建议:")
        print("1. 运行 'uv run python direct_finetuning_with_existing_modules.py' 开始微调")
        print("2. 监控GPU内存使用，必要时调整批次大小")
        print("3. 确保有足够的存储空间保存检查点")
        return True
    else:
        print("⚠️ 多个测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)