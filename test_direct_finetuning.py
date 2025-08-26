#!/usr/bin/env python3
"""
测试直接微调功能的简化脚本
用于验证已实现模块的基本功能
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import logging

# 添加src路径
sys.path.append('src')

def test_gpu_detection():
    """测试GPU检测功能"""
    print("🔍 测试GPU检测...")
    try:
        from gpu_utils import GPUDetector
        detector = GPUDetector()
        gpu_info = detector.detect_gpus()
        print(f"✅ 检测到 {len(gpu_info)} 个GPU")
        for i, gpu in enumerate(gpu_info):
            print(f"  GPU {i}: {gpu.name}, 内存: {gpu.memory_total}MB")
        return True
    except Exception as e:
        print(f"❌ GPU检测失败: {e}")
        return False

def test_memory_manager():
    """测试内存管理器"""
    print("🔍 测试内存管理器...")
    try:
        from memory_manager import MemoryManager
        manager = MemoryManager()
        if torch.cuda.is_available():
            snapshot = manager.get_memory_snapshot(0)
            print(f"✅ GPU内存快照: {snapshot.allocated_memory}MB / {snapshot.total_memory}MB")
        else:
            print("✅ 内存管理器初始化成功（CPU模式）")
        return True
    except Exception as e:
        print(f"❌ 内存管理器测试失败: {e}")
        return False

def test_chinese_processor():
    """测试中文处理器"""
    print("🔍 测试中文处理器...")
    try:
        from chinese_nlp_processor import ChineseNLPProcessor
        processor = ChineseNLPProcessor()
        
        test_text = "什么是对称加密算法？"
        processed = processor.preprocess_text(test_text)
        print(f"✅ 中文处理测试: '{test_text}' -> '{processed}'")
        return True
    except Exception as e:
        print(f"❌ 中文处理器测试失败: {e}")
        return False

def test_crypto_processor():
    """测试密码学术语处理器"""
    print("🔍 测试密码学术语处理器...")
    try:
        from crypto_term_processor import CryptoTermProcessor
        processor = CryptoTermProcessor()
        
        test_text = "AES是一种对称加密算法，使用相同的密钥进行加密和解密。"
        terms = processor.extract_crypto_terms(test_text)
        print(f"✅ 密码学术语提取: {[term.term for term in terms]}")
        return True
    except Exception as e:
        print(f"❌ 密码学术语处理器测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("🔍 测试数据加载...")
    try:
        data_path = "final_demo_output/data/crypto_qa_dataset_train.json"
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 成功加载 {len(data)} 条训练数据")
        
        # 显示第一条数据
        if data:
            sample = data[0]
            print(f"  样本示例:")
            print(f"    指令: {sample.get('instruction', '')[:50]}...")
            print(f"    输出: {sample.get('output', '')[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    try:
        model_name = "Qwen/Qwen3-4B-Thinking-2507"  # 目标微调模型
        print(f"  加载模型: {model_name}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ 分词器加载成功")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print(f"✅ 模型加载成功，参数量: {model.num_parameters():,}")
        
        # 测试LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✅ LoRA配置成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_inference():
    """测试简单推理"""
    print("🔍 测试简单推理...")
    try:
        model_name = "Qwen/Qwen3-4B-Thinking-2507"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 测试推理
        test_prompt = "<|im_start|>system\n你是一个专业的密码学专家。<|im_end|>\n<|im_start|>user\n什么是AES？<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("✅ 推理测试成功")
        print(f"  输入: {test_prompt[:50]}...")
        print(f"  输出: {response[len(test_prompt):100]}...")
        
        return True
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 测试直接微调功能")
    print("=" * 50)
    
    tests = [
        ("GPU检测", test_gpu_detection),
        ("内存管理器", test_memory_manager),
        ("中文处理器", test_chinese_processor),
        ("密码学术语处理器", test_crypto_processor),
        ("数据加载", test_data_loading),
        ("模型加载", test_model_loading),
        ("简单推理", test_simple_inference),
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
    
    if passed == len(results):
        print("🎉 所有测试通过！可以开始微调。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)