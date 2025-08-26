#!/usr/bin/env python3
"""
快速启动演示脚本

简化的演示程序启动脚本，用于快速测试和验证功能。
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_environment():
    """检查环境"""
    logger = logging.getLogger(__name__)
    
    # 检查数据目录
    data_dir = Path("data/raw")
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return False
    
    # 检查数据文件
    md_files = list(data_dir.glob("*.md"))
    if not md_files:
        logger.error(f"在 {data_dir} 中未找到markdown文件")
        return False
    
    logger.info(f"找到 {len(md_files)} 个数据文件")
    
    # 检查src目录
    src_dir = Path("src")
    if not src_dir.exists():
        logger.error(f"源代码目录不存在: {src_dir}")
        return False
    
    logger.info("环境检查通过")
    return True

def simple_data_processing():
    """简单的数据处理演示"""
    logger = logging.getLogger(__name__)
    
    try:
        # 导入必要的模块
        from data_models import TrainingExample, DifficultyLevel
        
        # 读取一个示例文件
        data_dir = Path("data/raw")
        md_files = list(data_dir.glob("*.md"))
        
        if not md_files:
            logger.error("没有找到数据文件")
            return False
        
        # 读取第一个文件
        sample_file = md_files[0]
        logger.info(f"处理示例文件: {sample_file.name}")
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单解析QA对
        examples = []
        lines = content.split('\n')
        
        current_question = ""
        current_answer = ""
        current_thinking = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查问题
            if line.startswith('### Q') or (line.startswith('##') and '?' in line):
                # 保存之前的QA对
                if current_question and current_answer:
                    example = TrainingExample(
                        instruction=current_question.replace('###', '').replace('##', '').strip(),
                        input="",
                        output=current_answer.replace('A1:', '').replace('A2:', '').replace('A3:', '').strip(),
                        thinking=current_thinking if current_thinking else None,
                        crypto_terms=["密码学", "加密"],  # 简化处理
                        difficulty_level=DifficultyLevel.INTERMEDIATE
                    )
                    examples.append(example)
                
                current_question = line
                current_answer = ""
                current_thinking = ""
            
            elif line.startswith('<thinking>'):
                current_thinking = line
            
            elif line.startswith('A') and ':' in line:
                current_answer = line
        
        # 处理最后一个QA对
        if current_question and current_answer:
            example = TrainingExample(
                instruction=current_question.replace('###', '').replace('##', '').strip(),
                input="",
                output=current_answer.replace('A1:', '').replace('A2:', '').replace('A3:', '').strip(),
                thinking=current_thinking if current_thinking else None,
                crypto_terms=["密码学", "加密"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
            examples.append(example)
        
        logger.info(f"解析出 {len(examples)} 个训练样例")
        
        # 保存处理结果
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # 转换为训练格式（通用格式）
        training_data = []
        for example in examples:
            training_data.append(example.to_training_format())
        
        # 保存数据
        output_file = output_dir / "processed_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理结果已保存到: {output_file}")
        
        # 生成简单报告
        report = {
            "processing_time": datetime.now().isoformat(),
            "source_file": str(sample_file),
            "total_examples": len(examples),
            "output_file": str(output_file),
            "sample_data": training_data[:2] if training_data else []
        }
        
        report_file = output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理报告已保存到: {report_file}")
        
        # 打印摘要
        print("\n" + "="*50)
        print("🎉 数据处理演示完成！")
        print("="*50)
        print(f"📊 处理统计:")
        print(f"   - 源文件: {sample_file.name}")
        print(f"   - 训练样例: {len(examples)}")
        print(f"   - 输出文件: {output_file}")
        print(f"   - 报告文件: {report_file}")
        
        if examples:
            print(f"\n📝 示例数据:")
            example = examples[0]
            print(f"   - 问题: {example.instruction[:100]}...")
            print(f"   - 答案: {example.output[:100]}...")
            print(f"   - 包含thinking: {'是' if example.has_thinking() else '否'}")
        
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        return False

def gpu_detection_demo():
    """GPU检测演示"""
    logger = logging.getLogger(__name__)
    
    try:
        from gpu_utils import GPUDetector
        
        detector = GPUDetector()
        gpu_infos = detector.get_all_gpu_info()
        
        print("\n" + "="*50)
        print("🖥️ GPU检测结果")
        print("="*50)
        
        if gpu_infos:
            print(f"检测到 {len(gpu_infos)} 个GPU:")
            for i, gpu in enumerate(gpu_infos):
                print(f"   GPU {i}: {gpu.name}")
                # 安全地访问GPU属性
                try:
                    # 内存信息 (total_memory是以MB为单位)
                    if hasattr(gpu, 'total_memory') and gpu.total_memory:
                        print(f"   - 总内存: {gpu.total_memory/1024:.1f} GB")
                        print(f"   - 已用内存: {gpu.used_memory/1024:.1f} GB")
                        print(f"   - 空闲内存: {gpu.free_memory/1024:.1f} GB")
                    
                    # 利用率
                    if hasattr(gpu, 'utilization') and gpu.utilization is not None:
                        print(f"   - GPU利用率: {gpu.utilization}%")
                    
                    # 温度
                    if hasattr(gpu, 'temperature') and gpu.temperature is not None:
                        print(f"   - 温度: {gpu.temperature}°C")
                    
                    # 功耗
                    if hasattr(gpu, 'power_usage') and gpu.power_usage is not None:
                        print(f"   - 功耗: {gpu.power_usage}W")
                    
                    # 计算能力
                    if hasattr(gpu, 'compute_capability') and gpu.compute_capability:
                        major, minor = gpu.compute_capability
                        print(f"   - 计算能力: {major}.{minor}")
                    
                    # PCI总线ID
                    if hasattr(gpu, 'pci_bus_id') and gpu.pci_bus_id:
                        print(f"   - PCI总线: {gpu.pci_bus_id}")
                        
                except Exception as e:
                    print(f"   - 详细信息获取失败: {e}")
                    # 显示可用属性用于调试
                    available_attrs = [attr for attr in dir(gpu) if not attr.startswith('_') and not callable(getattr(gpu, attr))]
                    print(f"   - 可用属性: {available_attrs[:5]}...")  # 只显示前5个属性
        else:
            print("未检测到GPU，将使用CPU模式")
        
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"GPU检测失败: {e}")
        return False

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 启动快速演示程序...")
    
    # 1. 环境检查
    print("\n📋 步骤 1: 环境检查")
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 2. GPU检测
    print("\n🖥️ 步骤 2: GPU检测")
    gpu_detection_demo()
    
    # 3. 数据处理演示
    print("\n📊 步骤 3: 数据处理演示")
    if not simple_data_processing():
        print("❌ 数据处理失败")
        return False
    
    print("\n✅ 快速演示程序执行成功！")
    print("\n💡 提示:")
    print("   - 查看 demo_output/ 目录获取处理结果")
    print("   - 运行 'uv run python demo_comprehensive_finetuning.py' 进行完整训练")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)