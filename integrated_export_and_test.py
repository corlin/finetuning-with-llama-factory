#!/usr/bin/env python3
"""
集成导出和测试程序

本程序结合了模型导出和测试功能，提供一站式的模型处理解决方案。
"""

import os
import sys
import torch
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers或peft库不可用")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedExportAndTest:
    """集成导出和测试器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 测试用例
        self.test_cases = [
            {
                "prompt": "什么是AES加密算法？",
                "expected_keywords": ["对称加密", "高级加密标准", "分组密码"],
                "max_length": 100
            },
            {
                "prompt": "<thinking>我需要解释这个概念</thinking>请解释数字签名的作用。",
                "expected_keywords": ["哈希函数", "私钥", "公钥", "完整性"],
                "max_length": 120
            },
            {
                "prompt": "椭圆曲线密码学有什么优势？",
                "expected_keywords": ["密钥长度", "安全性", "计算效率"],
                "max_length": 100
            }
        ]
    
    def load_and_merge_checkpoint(
        self, 
        checkpoint_path: str,
        base_model_path: str = "Qwen/Qwen3-4B-Thinking-2507"
    ):
        """加载并合并检查点"""
        logger.info("步骤1: 加载和合并检查点")
        
        # 加载基础模型
        logger.info(f"加载基础模型: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 检查是否为LoRA检查点
        checkpoint_dir = Path(checkpoint_path)
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        
        if adapter_config_path.exists():
            logger.info("检测到LoRA检查点，进行合并...")
            
            # 加载LoRA适配器
            model_with_lora = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                torch_dtype=torch.float16
            )
            
            # 合并LoRA权重
            merged_model = model_with_lora.merge_and_unload()
            logger.info("✅ LoRA权重合并完成")
            return merged_model, tokenizer
        else:
            logger.info("使用基础模型...")
            return base_model, tokenizer
    
    def export_multiple_formats(self, model, tokenizer, output_dir: str):
        """导出多种格式并立即测试"""
        logger.info("步骤2: 导出多种格式")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        formats_to_export = [
            ("fp16", "FP16精度模型"),
            ("int8_simulated", "模拟INT8量化"),
            ("int4_simulated", "模拟INT4量化")
        ]
        
        results = {}
        
        for format_name, description in formats_to_export:
            logger.info(f"\n导出 {format_name.upper()} 格式: {description}")
            
            format_dir = output_path / format_name
            format_dir.mkdir(exist_ok=True)
            
            try:
                # 导出模型
                export_result = self._export_format(model, tokenizer, format_dir, format_name)
                
                if export_result["success"]:
                    # 立即测试
                    logger.info(f"测试 {format_name.upper()} 模型...")
                    test_result = self._test_model_immediately(format_dir, format_name)
                    
                    results[format_name] = {
                        **export_result,
                        "test_result": test_result
                    }
                    
                    if test_result["success"]:
                        logger.info(f"✅ {format_name.upper()} 导出和测试成功")
                    else:
                        logger.warning(f"⚠️ {format_name.upper()} 导出成功但测试失败")
                else:
                    results[format_name] = export_result
                    logger.error(f"❌ {format_name.upper()} 导出失败")
                
            except Exception as e:
                logger.error(f"❌ {format_name.upper()} 处理失败: {e}")
                results[format_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _export_format(self, model, tokenizer, output_dir: Path, format_name: str):
        """导出特定格式"""
        try:
            if format_name == "fp16":
                # FP16格式
                model_fp16 = model.half()
                model_fp16.save_pretrained(
                    output_dir,
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                
            elif format_name == "int8_simulated":
                # 模拟INT8量化
                model_int8 = self._simulate_quantization(model, bits=8)
                model_int8.save_pretrained(
                    output_dir,
                    safe_serialization=True,
                    max_shard_size="1GB"
                )
                
            elif format_name == "int4_simulated":
                # 模拟INT4量化
                model_int4 = self._simulate_quantization(model, bits=4)
                model_int4.save_pretrained(
                    output_dir,
                    safe_serialization=True,
                    max_shard_size="500MB"
                )
            
            # 保存tokenizer
            tokenizer.save_pretrained(output_dir)
            
            # 计算大小
            size_mb = self._get_directory_size(output_dir)
            
            # 创建格式信息
            format_info = {
                "format": format_name,
                "export_time": datetime.now().isoformat(),
                "size_mb": size_mb,
                "compression_note": self._get_compression_note(format_name)
            }
            
            with open(output_dir / "format_info.json", "w", encoding="utf-8") as f:
                json.dump(format_info, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "format": format_name,
                "size_mb": size_mb,
                "output_path": str(output_dir)
            }
            
        except Exception as e:
            return {
                "success": False,
                "format": format_name,
                "error": str(e)
            }
    
    def _simulate_quantization(self, model, bits: int):
        """模拟量化过程"""
        model_copy = model.cpu()
        
        with torch.no_grad():
            for name, param in model_copy.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    # 模拟量化
                    param_min = param.min()
                    param_max = param.max()
                    
                    # 计算量化级别
                    levels = 2 ** bits - 1
                    scale = (param_max - param_min) / levels
                    
                    # 量化和反量化
                    quantized = torch.round((param - param_min) / scale)
                    quantized = torch.clamp(quantized, 0, levels)
                    dequantized = quantized * scale + param_min
                    
                    param.copy_(dequantized)
        
        return model_copy
    
    def _test_model_immediately(self, model_dir: Path, format_name: str):
        """立即测试导出的模型"""
        try:
            # 加载模型
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 运行测试用例
            test_results = []
            total_time = 0
            
            model.eval()
            with torch.no_grad():
                for i, test_case in enumerate(self.test_cases):
                    start_time = time.time()
                    
                    try:
                        inputs = tokenizer(
                            test_case["prompt"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                        
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        
                        outputs = model.generate(
                            **inputs,
                            max_length=inputs["input_ids"].shape[1] + test_case["max_length"],
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        
                        response = tokenizer.decode(
                            outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True
                        )
                        
                        inference_time = time.time() - start_time
                        total_time += inference_time
                        
                        # 评估响应
                        score = self._evaluate_response(response, test_case)
                        
                        test_results.append({
                            "test_id": i + 1,
                            "prompt": test_case["prompt"][:50] + "...",
                            "response_length": len(response),
                            "inference_time": inference_time,
                            "score": score,
                            "success": True
                        })
                        
                    except Exception as e:
                        test_results.append({
                            "test_id": i + 1,
                            "prompt": test_case["prompt"][:50] + "...",
                            "success": False,
                            "error": str(e)
                        })
            
            # 计算总体结果
            successful_tests = [t for t in test_results if t["success"]]
            avg_score = sum(t["score"] for t in successful_tests) / len(successful_tests) if successful_tests else 0
            avg_time = total_time / len(successful_tests) if successful_tests else 0
            
            # 清理内存
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "total_tests": len(self.test_cases),
                "successful_tests": len(successful_tests),
                "success_rate": len(successful_tests) / len(self.test_cases),
                "avg_score": avg_score,
                "avg_inference_time": avg_time,
                "test_details": test_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _evaluate_response(self, response: str, test_case: Dict[str, Any]) -> float:
        """评估响应质量"""
        if not response.strip():
            return 0.0
        
        score = 0.0
        
        # 关键词匹配
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            matched = sum(1 for kw in expected_keywords if kw in response)
            score += (matched / len(expected_keywords)) * 0.5
        
        # 中文字符比例
        chinese_chars = sum(1 for char in response if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / len(response) if response else 0
        score += min(chinese_ratio * 2, 1.0) * 0.3
        
        # 长度合理性
        if 20 <= len(response) <= 300:
            score += 0.2
        elif len(response) > 10:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_directory_size(self, directory: Path) -> float:
        """计算目录大小（MB）"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob('*') if f.is_file()
            )
            return total_size / 1024 / 1024
        except:
            return 0.0
    
    def _get_compression_note(self, format_name: str) -> str:
        """获取压缩说明"""
        notes = {
            "fp16": "半精度浮点，平衡精度和性能",
            "int8_simulated": "模拟8位量化，约50%压缩",
            "int4_simulated": "模拟4位量化，约75%压缩"
        }
        return notes.get(format_name, "未知格式")
    
    def generate_comprehensive_report(self, results: Dict[str, Any], output_dir: str):
        """生成综合报告"""
        logger.info("步骤3: 生成综合报告")
        
        # 创建详细报告
        report = {
            "export_and_test_time": datetime.now().isoformat(),
            "summary": {
                "total_formats": len(results),
                "successful_exports": sum(1 for r in results.values() if r.get("success", False)),
                "successful_tests": sum(1 for r in results.values() 
                                      if r.get("success", False) and r.get("test_result", {}).get("success", False))
            },
            "results": results,
            "performance_comparison": {},
            "recommendations": []
        }
        
        # 性能对比
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}
        
        if successful_results:
            for format_name, result in successful_results.items():
                test_result = result.get("test_result", {})
                
                report["performance_comparison"][format_name] = {
                    "size_mb": result.get("size_mb", 0),
                    "test_success": test_result.get("success", False),
                    "success_rate": test_result.get("success_rate", 0),
                    "avg_score": test_result.get("avg_score", 0),
                    "avg_inference_time": test_result.get("avg_inference_time", 0)
                }
        
        # 生成建议
        if report["summary"]["successful_tests"] > 0:
            # 找出最佳模型
            best_models = {}
            for format_name, perf in report["performance_comparison"].items():
                if perf["test_success"]:
                    if not best_models.get("quality") or perf["avg_score"] > report["performance_comparison"][best_models["quality"]]["avg_score"]:
                        best_models["quality"] = format_name
                    
                    if not best_models.get("speed") or perf["avg_inference_time"] < report["performance_comparison"][best_models["speed"]]["avg_inference_time"]:
                        best_models["speed"] = format_name
                    
                    if not best_models.get("size") or perf["size_mb"] < report["performance_comparison"][best_models["size"]]["size_mb"]:
                        best_models["size"] = format_name
            
            report["recommendations"] = [
                f"最佳质量: {best_models.get('quality', 'N/A')}",
                f"最快推理: {best_models.get('speed', 'N/A')}",
                f"最小体积: {best_models.get('size', 'N/A')}"
            ]
        
        # 保存报告
        report_path = Path(output_dir) / "comprehensive_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 显示摘要
        self._display_final_summary(report)
        
        logger.info(f"📊 综合报告已保存到: {report_path}")
        
        return report
    
    def _display_final_summary(self, report: Dict[str, Any]):
        """显示最终摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("集成导出和测试摘要")
        logger.info("=" * 60)
        
        summary = report["summary"]
        logger.info(f"总格式数: {summary['total_formats']}")
        logger.info(f"成功导出: {summary['successful_exports']}")
        logger.info(f"测试通过: {summary['successful_tests']}")
        
        if "performance_comparison" in report:
            logger.info("\n性能对比:")
            logger.info("-" * 40)
            
            for format_name, perf in report["performance_comparison"].items():
                status = "✅" if perf["test_success"] else "❌"
                logger.info(f"{status} {format_name.upper()}:")
                logger.info(f"   大小: {perf['size_mb']:.1f}MB")
                if perf["test_success"]:
                    logger.info(f"   质量: {perf['avg_score']:.2f}")
                    logger.info(f"   速度: {perf['avg_inference_time']:.2f}s")
                    logger.info(f"   成功率: {perf['success_rate']:.1%}")
        
        if report.get("recommendations"):
            logger.info("\n推荐:")
            logger.info("-" * 20)
            for rec in report["recommendations"]:
                logger.info(f"  {rec}")
    
    def run_integrated_demo(
        self,
        checkpoint_path: str = "qwen3_4b_thinking_output/final_model",
        base_model_path: str = "Qwen/Qwen3-4B-Thinking-2507",
        output_dir: str = "integrated_export_test_output"
    ):
        """运行集成演示"""
        logger.info("=" * 60)
        logger.info("集成导出和测试演示")
        logger.info("=" * 60)
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers库不可用")
            return
        
        try:
            # 步骤1: 加载模型
            model, tokenizer = self.load_and_merge_checkpoint(checkpoint_path, base_model_path)
            
            # 步骤2: 导出和测试
            results = self.export_multiple_formats(model, tokenizer, output_dir)
            
            # 步骤3: 生成报告
            report = self.generate_comprehensive_report(results, output_dir)
            
            logger.info("\n🎉 集成演示完成！")
            logger.info(f"📁 输出目录: {output_dir}")
            logger.info("📖 查看 comprehensive_report.json 了解详细信息")
            
        except Exception as e:
            logger.error(f"集成演示失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="集成导出和测试演示")
    parser.add_argument(
        "--checkpoint_path",
        default="qwen3_4b_thinking_output/final_model",
        help="检查点路径"
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="基础模型路径"
    )
    parser.add_argument(
        "--output_dir",
        default="integrated_export_test_output",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 运行集成演示
    demo = IntegratedExportAndTest()
    demo.run_integrated_demo(
        checkpoint_path=args.checkpoint_path,
        base_model_path=args.base_model,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()