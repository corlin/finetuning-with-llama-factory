#!/usr/bin/env python3
"""
最终测试摘要程序

整合所有导出模型的测试结果，生成完整的评估报告
"""

import json
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def find_all_models() -> Dict[str, str]:
    """查找所有可用的模型"""
    
    base_dirs = [
        "quantized_models_output",
        "fixed_quantized_output", 
        "size_demo_output",
        "safe_quantized_output",
        "integrated_export_test_output"
    ]
    
    models = {}
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue
            
        for format_dir in base_path.iterdir():
            if format_dir.is_dir():
                # 检查是否包含模型文件
                has_model = any([
                    list(format_dir.glob("*.safetensors")),
                    list(format_dir.glob("*.bin")),
                ])
                
                has_tokenizer = (format_dir / "tokenizer.json").exists()
                
                if has_model and has_tokenizer:
                    model_key = f"{base_dir}_{format_dir.name}"
                    models[model_key] = str(format_dir)
    
    return models


def quick_test_model(model_path: str, model_name: str) -> Dict[str, Any]:
    """快速测试单个模型"""
    
    result = {
        "model_name": model_name,
        "model_path": model_path,
        "test_time": datetime.now().isoformat(),
        "success": False,
        "load_time": 0,
        "model_size_mb": 0,
        "inference_test": {},
        "error": None
    }
    
    try:
        # 计算模型大小
        result["model_size_mb"] = get_directory_size(Path(model_path))
        
        # 加载模型
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        result["load_time"] = time.time() - start_time
        
        # 推理测试
        test_prompt = "什么是AES加密算法？"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        inference_time = time.time() - start_time
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 评估响应质量
        chinese_chars = sum(1 for char in response if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / len(response) if response else 0
        
        result["inference_test"] = {
            "prompt": test_prompt,
            "response": response[:100] + "..." if len(response) > 100 else response,
            "response_length": len(response),
            "inference_time": inference_time,
            "chinese_ratio": chinese_ratio,
            "has_content": len(response.strip()) > 10
        }
        
        result["success"] = True
        
        # 清理内存
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def get_directory_size(directory: Path) -> float:
    """计算目录大小（MB）"""
    try:
        total_size = sum(
            f.stat().st_size for f in directory.rglob('*') if f.is_file()
        )
        return total_size / 1024 / 1024
    except:
        return 0.0


def generate_final_report(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成最终报告"""
    
    successful_tests = [r for r in test_results if r["success"]]
    failed_tests = [r for r in test_results if not r["success"]]
    
    report = {
        "report_time": datetime.now().isoformat(),
        "summary": {
            "total_models": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(test_results) if test_results else 0
        },
        "test_results": test_results,
        "analysis": {},
        "recommendations": []
    }
    
    if successful_tests:
        # 性能分析
        sizes = [r["model_size_mb"] for r in successful_tests]
        load_times = [r["load_time"] for r in successful_tests]
        inference_times = [r["inference_test"]["inference_time"] for r in successful_tests 
                          if "inference_time" in r["inference_test"]]
        
        report["analysis"] = {
            "size_stats": {
                "min_mb": min(sizes),
                "max_mb": max(sizes),
                "avg_mb": sum(sizes) / len(sizes)
            },
            "performance_stats": {
                "avg_load_time": sum(load_times) / len(load_times),
                "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0
            }
        }
        
        # 找出最佳模型
        best_models = {
            "smallest": min(successful_tests, key=lambda x: x["model_size_mb"]),
            "fastest_load": min(successful_tests, key=lambda x: x["load_time"]),
            "fastest_inference": min(successful_tests, 
                                   key=lambda x: x["inference_test"].get("inference_time", float('inf')))
        }
        
        report["analysis"]["best_models"] = {
            "smallest": best_models["smallest"]["model_name"],
            "fastest_load": best_models["fastest_load"]["model_name"], 
            "fastest_inference": best_models["fastest_inference"]["model_name"]
        }
        
        # 生成建议
        if report["summary"]["success_rate"] > 0.8:
            report["recommendations"].append("大部分模型测试成功，可以考虑部署使用")
        
        if len([r for r in successful_tests if r["model_size_mb"] < 4000]) > 0:
            report["recommendations"].append("有可用的压缩模型，适合资源受限环境")
        
        if len([r for r in successful_tests if r["inference_test"].get("inference_time", 0) < 2.0]) > 0:
            report["recommendations"].append("有快速推理的模型，适合实时应用")
    
    if failed_tests:
        common_errors = {}
        for test in failed_tests:
            error = test.get("error", "未知错误")
            error_type = error.split(":")[0] if ":" in error else error
            common_errors[error_type] = common_errors.get(error_type, 0) + 1
        
        report["analysis"]["common_errors"] = common_errors
        
        if "CUDA" in str(common_errors):
            report["recommendations"].append("存在CUDA相关错误，建议检查量化算法")
        
        if len(failed_tests) > len(successful_tests):
            report["recommendations"].append("失败率较高，建议重新检查模型导出流程")
    
    return report


def display_summary(report: Dict[str, Any]):
    """显示测试摘要"""
    
    print("\n" + "=" * 80)
    print("🔍 导出模型最终测试摘要")
    print("=" * 80)
    
    summary = report["summary"]
    print(f"📊 测试统计:")
    print(f"   总模型数: {summary['total_models']}")
    print(f"   成功测试: {summary['successful_tests']}")
    print(f"   失败测试: {summary['failed_tests']}")
    print(f"   成功率: {summary['success_rate']:.1%}")
    
    if "analysis" in report:
        analysis = report["analysis"]
        
        if "size_stats" in analysis:
            size_stats = analysis["size_stats"]
            print(f"\n📏 模型大小统计:")
            print(f"   最小: {size_stats['min_mb']:.1f} MB")
            print(f"   最大: {size_stats['max_mb']:.1f} MB") 
            print(f"   平均: {size_stats['avg_mb']:.1f} MB")
        
        if "performance_stats" in analysis:
            perf_stats = analysis["performance_stats"]
            print(f"\n⚡ 性能统计:")
            print(f"   平均加载时间: {perf_stats['avg_load_time']:.1f}s")
            print(f"   平均推理时间: {perf_stats['avg_inference_time']:.2f}s")
        
        if "best_models" in analysis:
            best = analysis["best_models"]
            print(f"\n🏆 最佳模型:")
            print(f"   最小体积: {best['smallest']}")
            print(f"   最快加载: {best['fastest_load']}")
            print(f"   最快推理: {best['fastest_inference']}")
        
        if "common_errors" in analysis:
            errors = analysis["common_errors"]
            print(f"\n❌ 常见错误:")
            for error_type, count in errors.items():
                print(f"   {error_type}: {count} 次")
    
    print(f"\n✅ 成功的模型:")
    for result in report["test_results"]:
        if result["success"]:
            size = result["model_size_mb"]
            load_time = result["load_time"]
            inf_time = result["inference_test"].get("inference_time", 0)
            print(f"   ✓ {result['model_name']}")
            print(f"     大小: {size:.1f}MB, 加载: {load_time:.1f}s, 推理: {inf_time:.2f}s")
    
    if any(not r["success"] for r in report["test_results"]):
        print(f"\n❌ 失败的模型:")
        for result in report["test_results"]:
            if not result["success"]:
                print(f"   ✗ {result['model_name']}: {result['error']}")
    
    if report.get("recommendations"):
        print(f"\n💡 建议:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    
    print("🔍 开始最终测试摘要...")
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ transformers库不可用")
        return
    
    # 查找所有模型
    models = find_all_models()
    
    if not models:
        print("❌ 未找到任何可测试的模型")
        return
    
    print(f"📋 找到 {len(models)} 个模型:")
    for model_name, model_path in models.items():
        print(f"   - {model_name}: {model_path}")
    
    # 测试所有模型
    print(f"\n🧪 开始测试...")
    test_results = []
    
    for i, (model_name, model_path) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] 测试 {model_name}...")
        
        result = quick_test_model(model_path, model_name)
        test_results.append(result)
        
        if result["success"]:
            print(f"   ✅ 成功 ({result['model_size_mb']:.1f}MB, {result['load_time']:.1f}s)")
        else:
            print(f"   ❌ 失败: {result['error']}")
    
    # 生成最终报告
    print(f"\n📊 生成最终报告...")
    final_report = generate_final_report(test_results)
    
    # 保存报告
    report_path = Path("final_test_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    # 显示摘要
    display_summary(final_report)
    
    print(f"\n📄 详细报告已保存到: {report_path}")
    
    # 生成简化的使用指南
    generate_usage_guide(final_report)


def generate_usage_guide(report: Dict[str, Any]):
    """生成使用指南"""
    
    successful_models = [r for r in report["test_results"] if r["success"]]
    
    if not successful_models:
        return
    
    guide_content = f"""# 导出模型使用指南

## 测试摘要

- **测试时间**: {report['report_time']}
- **总模型数**: {report['summary']['total_models']}
- **可用模型**: {report['summary']['successful_tests']}
- **成功率**: {report['summary']['success_rate']:.1%}

## 推荐模型

"""
    
    if "analysis" in report and "best_models" in report["analysis"]:
        best = report["analysis"]["best_models"]
        guide_content += f"""### 🏆 最佳选择

- **最小体积**: `{best['smallest']}` - 适合存储受限环境
- **最快加载**: `{best['fastest_load']}` - 适合频繁重启场景  
- **最快推理**: `{best['fastest_inference']}` - 适合实时应用

"""
    
    guide_content += """## 可用模型列表

| 模型名称 | 大小(MB) | 加载时间(s) | 推理时间(s) | 状态 |
|----------|----------|-------------|-------------|------|
"""
    
    for result in successful_models:
        name = result["model_name"]
        size = result["model_size_mb"]
        load_time = result["load_time"]
        inf_time = result["inference_test"].get("inference_time", 0)
        
        guide_content += f"| {name} | {size:.1f} | {load_time:.1f} | {inf_time:.2f} | ✅ |\n"
    
    guide_content += f"""
## 使用方法

### 加载模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 选择一个可用的模型路径
model_path = "path/to/your/chosen/model"

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
```

### 推理示例

```python
# 基础推理
prompt = "什么是AES加密算法？"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# 思考推理
thinking_prompt = "<thinking>分析这个问题</thinking>请解释RSA算法。"
inputs = tokenizer(thinking_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=300, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 性能建议

"""
    
    if report.get("recommendations"):
        for rec in report["recommendations"]:
            guide_content += f"- {rec}\n"
    
    guide_content += f"""
## 故障排除

如果遇到问题，请检查：

1. **CUDA内存**: 确保GPU内存足够
2. **依赖版本**: 使用兼容的transformers版本
3. **模型完整性**: 确认模型文件完整下载
4. **设备兼容性**: 检查CUDA版本兼容性

---

*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*测试版本: v1.0*
"""
    
    with open("MODEL_USAGE_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("📖 使用指南已生成: MODEL_USAGE_GUIDE.md")


if __name__ == "__main__":
    main()