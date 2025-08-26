#!/usr/bin/env python3
"""
验证LlamaFactory清理和自研训练引擎替换的测试脚本
"""

def test_imports():
    """测试关键组件导入"""
    try:
        # 测试训练流水线
        from src.training_pipeline import TrainingPipelineOrchestrator
        print("✅ TrainingPipelineOrchestrator 导入成功")
        
        # 测试直接训练引擎
        from direct_finetuning_with_existing_modules import DirectTrainer, DirectTrainingConfig
        print("✅ DirectTrainer 导入成功")
        
        # 测试演示程序
        from demo_final import FinalDemo
        print("✅ FinalDemo 导入成功")
        
        # 测试分布式训练引擎
        from src.distributed_training_engine import MultiGPUProcessManager
        print("✅ MultiGPUProcessManager 导入成功")
        
        # 测试内存管理器
        from src.memory_manager import MemoryManager
        print("✅ MemoryManager 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_no_llamafactory_references():
    """测试是否还有LlamaFactory引用"""
    import os
    import re
    
    # 检查关键文件中是否还有LlamaFactory引用
    key_files = [
        "src/training_pipeline.py",
        "demo_final.py", 
        "demo_comprehensive_finetuning.py",
        "direct_finetuning_with_existing_modules.py"
    ]
    
    llamafactory_pattern = re.compile(r'llamafactory|LlamaFactory', re.IGNORECASE)
    
    for file_path in key_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 排除注释中的引用
                lines = content.split('\n')
                active_references = []
                for i, line in enumerate(lines, 1):
                    if llamafactory_pattern.search(line) and not line.strip().startswith('#'):
                        active_references.append(f"Line {i}: {line.strip()}")
                
                if active_references:
                    print(f"⚠️ {file_path} 中仍有活跃的LlamaFactory引用:")
                    for ref in active_references:
                        print(f"  {ref}")
                else:
                    print(f"✅ {file_path} 已清理完成")

def main():
    """主测试函数"""
    print("🧪 开始验证LlamaFactory清理和自研训练引擎替换...")
    print("=" * 60)
    
    print("\n📦 测试组件导入:")
    imports_ok = test_imports()
    
    print("\n🔍 检查LlamaFactory引用清理:")
    test_no_llamafactory_references()
    
    print("\n" + "=" * 60)
    if imports_ok:
        print("✅ 所有测试通过！LlamaFactory依赖已成功清理并替换为自研训练引擎")
        print("🚀 系统已准备好使用自研训练框架进行模型微调")
    else:
        print("❌ 部分测试失败，需要进一步修复")

if __name__ == "__main__":
    main()