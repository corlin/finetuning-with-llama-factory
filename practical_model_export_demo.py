#!/usr/bin/env python3
"""
实用模型导出演示程序

本程序使用现有的模型导出功能，演示如何：
1. 加载微调后的检查点
2. 使用自研模型导出器进行量化
3. 验证中文处理能力
4. 生成部署包

使用方法:
    python practical_model_export_demo.py
"""

import os
import sys
import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入自研模块
try:
    from model_exporter import (
        ModelExporter, 
        QuantizationConfig, 
        QuantizationFormat, 
        QuantizationBackend,
        ModelQuantizer,
        ChineseCapabilityValidator
    )
    MODEL_EXPORTER_AVAILABLE = True
    logger.info("✅ 自研模型导出器加载成功")
except ImportError as e:
    MODEL_EXPORTER_AVAILABLE = False
    logger.error(f"❌ 自研模块导入失败: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers库可用")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ Transformers库不可用")


class PracticalModelExporter:
    """实用模型导出器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MODEL_EXPORTER_AVAILABLE:
            self.exporter = ModelExporter()
            self.quantizer = ModelQuantizer()
            self.validator = ChineseCapabilityValidator()
        else:
            self.exporter = None
            self.quantizer = None
            self.validator = None
    
    def load_checkpoint_model(
        self, 
        checkpoint_path: str,
        base_model_path: str = "Qwen/Qwen3-4B-Thinking-2507"
    ) -> tuple:
        """
        加载检查点模型
        
        Args:
            checkpoint_path: 检查点路径
            base_model_path: 基础模型路径
            
        Returns:
            (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要transformers和peft库")
        
        self.logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"检查点目录不存在: {checkpoint_path}")
        
        # 检查是否为LoRA检查点
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if adapter_config_path.exists():
            self.logger.info("检测到LoRA检查点，进行合并...")
            return self._load_lora_checkpoint(checkpoint_path, base_model_path)
        else:
            self.logger.info("加载完整模型检查点...")
            return self._load_full_checkpoint(checkpoint_path)
    
    def _load_lora_checkpoint(self, checkpoint_path: str, base_model_path: str) -> tuple:
        """加载LoRA检查点"""
        try:
            # 加载基础模型
            self.logger.info(f"加载基础模型: {base_model_path}")
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
            
            # 加载LoRA适配器
            self.logger.info("加载LoRA适配器...")
            model_with_lora = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                torch_dtype=torch.float16
            )
            
            # 合并LoRA权重
            self.logger.info("合并LoRA权重...")
            merged_model = model_with_lora.merge_and_unload()
            
            return merged_model, tokenizer
            
        except Exception as e:
            self.logger.error(f"LoRA检查点加载失败: {e}")
            raise
    
    def _load_full_checkpoint(self, checkpoint_path: str) -> tuple:
        """加载完整模型检查点"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"完整模型加载失败: {e}")
            raise
    
    def export_quantized_models(
        self,
        model,
        tokenizer,
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, Any]:
        """
        导出量化模型
        
        Args:
            model: 模型
            tokenizer: 分词器
            output_dir: 输出目录
            formats: 量化格式列表
            
        Returns:
            导出结果
        """
        if not MODEL_EXPORTER_AVAILABLE:
            return self._fallback_export(model, tokenizer, output_dir, formats)
        
        if formats is None:
            formats = ["int8", "int4", "dynamic"]
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for format_name in formats:
            try:
                self.logger.info(f"开始导出 {format_name.upper()} 格式...")
                
                # 创建量化配置
                config = self._create_quantization_config(format_name)
                
                # 执行量化
                quantized_model, quant_result = self.quantizer.quantize_model(
                    model=model,
                    tokenizer=tokenizer,
                    config=config
                )
                
                if quant_result.success:
                    # 保存量化模型
                    format_dir = output_path / format_name
                    format_dir.mkdir(exist_ok=True)
                    
                    # 保存模型
                    if hasattr(quantized_model, 'save_pretrained'):
                        quantized_model.save_pretrained(format_dir)
                    else:
                        torch.save(quantized_model.state_dict(), format_dir / "model.pth")
                    
                    # 保存tokenizer
                    tokenizer.save_pretrained(format_dir)
                    
                    # 验证中文能力
                    chinese_validation = self.validator.validate_chinese_capability(
                        quantized_model, tokenizer
                    )
                    
                    results[format_name] = {
                        "success": True,
                        "quantization_result": quant_result.to_dict(),
                        "chinese_validation": chinese_validation,
                        "output_path": str(format_dir)
                    }
                    
                    self.logger.info(f"✅ {format_name.upper()} 导出成功")
                else:
                    results[format_name] = {
                        "success": False,
                        "error": quant_result.error_message
                    }
                    self.logger.error(f"❌ {format_name.upper()} 导出失败: {quant_result.error_message}")
                
            except Exception as e:
                results[format_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.logger.error(f"❌ {format_name.upper()} 导出异常: {e}")
        
        # 生成导出报告
        self._generate_export_report(results, output_path)
        
        return results
    
    def _create_quantization_config(self, format_name: str) -> QuantizationConfig:
        """创建量化配置"""
        if format_name == "int8":
            return QuantizationConfig(
                format=QuantizationFormat.INT8,
                backend=QuantizationBackend.BITSANDBYTES,
                load_in_8bit=True
            )
        elif format_name == "int4":
            return QuantizationConfig(
                format=QuantizationFormat.INT4,
                backend=QuantizationBackend.BITSANDBYTES,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif format_name == "gptq":
            return QuantizationConfig(
                format=QuantizationFormat.GPTQ,
                backend=QuantizationBackend.GPTQ,
                bits=4,
                group_size=128
            )
        elif format_name == "dynamic":
            return QuantizationConfig(
                format=QuantizationFormat.DYNAMIC,
                backend=QuantizationBackend.PYTORCH
            )
        else:
            raise ValueError(f"不支持的量化格式: {format_name}")
    
    def _fallback_export(
        self,
        model,
        tokenizer,
        output_dir: str,
        formats: List[str]
    ) -> Dict[str, Any]:
        """备用导出方法（当自研模块不可用时）"""
        self.logger.warning("使用备用导出方法...")
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 只导出FP16版本作为基准
        try:
            fp16_dir = output_path / "fp16"
            fp16_dir.mkdir(exist_ok=True)
            
            # 转换为FP16并保存
            model_fp16 = model.half()
            model_fp16.save_pretrained(fp16_dir)
            tokenizer.save_pretrained(fp16_dir)
            
            # 简单的推理测试
            test_result = self._simple_inference_test(model_fp16, tokenizer)
            
            results["fp16"] = {
                "success": True,
                "format": "FP16",
                "output_path": str(fp16_dir),
                "test_result": test_result,
                "note": "备用导出，仅FP16格式"
            }
            
            self.logger.info("✅ FP16备用导出成功")
            
        except Exception as e:
            results["fp16"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"❌ 备用导出失败: {e}")
        
        return results
    
    def _simple_inference_test(self, model, tokenizer) -> Dict[str, Any]:
        """简单推理测试"""
        test_prompts = [
            "什么是AES加密算法？",
            "请解释数字签名的作用。"
        ]
        
        results = []
        model.eval()
        
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + 50,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    results.append({
                        "prompt": prompt,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "success": True
                    })
                    
                except Exception as e:
                    results.append({
                        "prompt": prompt,
                        "response": "",
                        "success": False,
                        "error": str(e)
                    })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        
        return {
            "success_rate": success_rate,
            "test_cases": results
        }
    
    def _generate_export_report(self, results: Dict[str, Any], output_path: Path):
        """生成导出报告"""
        report = {
            "export_time": datetime.now().isoformat(),
            "exporter_version": "practical_demo_v1.0",
            "total_formats": len(results),
            "successful_exports": sum(1 for r in results.values() if r.get("success", False)),
            "results": results,
            "summary": {
                "formats_attempted": list(results.keys()),
                "successful_formats": [k for k, v in results.items() if v.get("success", False)],
                "failed_formats": [k for k, v in results.items() if not v.get("success", False)]
            }
        }
        
        with open(output_path / "export_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"导出报告已保存: {output_path / 'export_report.json'}")
    
    def generate_usage_examples(self, output_dir: str, results: Dict[str, Any]):
        """生成使用示例"""
        output_path = Path(output_dir)
        
        # 生成Python使用示例
        example_script = f'''#!/usr/bin/env python3
"""
模型使用示例脚本
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def load_model(model_path, use_quantization=False):
    """加载模型"""
    print(f"加载模型: {{model_path}}")
    
    if use_quantization:
        from transformers import BitsAndBytesConfig
        
        # INT8量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def test_basic_inference(model, tokenizer):
    """基础推理测试"""
    print("\\n=== 基础推理测试 ===")
    
    prompts = [
        "什么是AES加密算法？",
        "请解释RSA算法的工作原理。",
        "数字签名有什么作用？"
    ]
    
    for prompt in prompts:
        print(f"\\n输入: {{prompt}}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"输出: {{response}}")
        print("-" * 50)

def test_thinking_inference(model, tokenizer):
    """深度思考推理测试"""
    print("\\n=== 深度思考推理测试 ===")
    
    thinking_prompt = '''<thinking>
用户询问关于椭圆曲线密码学的问题。我需要：
1. 解释椭圆曲线密码学的基本概念
2. 说明其相比RSA的优势
3. 提及实际应用场景
</thinking>
请详细解释椭圆曲线密码学的优势。'''
    
    print(f"输入: {{thinking_prompt}}")
    
    inputs = tokenizer(thinking_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"输出: {{response}}")

def main():
    """主函数"""
    # 模型路径配置
    base_dir = Path("{output_dir}")
    
    # 可用的模型格式
    available_formats = {results.keys()}
    successful_formats = [k for k, v in {results}.items() if v.get("success", False)]
    
    print("可用的模型格式:")
    for fmt in successful_formats:
        print(f"- {{fmt}}: {{base_dir / fmt}}")
    
    # 选择一个可用的格式进行测试
    if successful_formats:
        test_format = successful_formats[0]
        model_path = base_dir / test_format
        
        print(f"\\n使用格式: {{test_format}}")
        
        try:
            # 加载模型
            use_quant = test_format in ["int8", "int4"]
            model, tokenizer = load_model(str(model_path), use_quantization=use_quant)
            
            # 运行测试
            test_basic_inference(model, tokenizer)
            test_thinking_inference(model, tokenizer)
            
            print("\\n✅ 所有测试完成！")
            
        except Exception as e:
            print(f"❌ 测试失败: {{e}}")
    else:
        print("❌ 没有可用的模型格式")

if __name__ == "__main__":
    main()
'''
        
        with open(output_path / "usage_examples.py", "w", encoding="utf-8") as f:
            f.write(example_script)
        
        self.logger.info("使用示例已生成: usage_examples.py")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("实用模型导出演示程序")
    logger.info("=" * 60)
    
    # 配置参数
    checkpoint_path = "qwen3_4b_thinking_output/final_model"
    base_model_path = "Qwen/Qwen3-4B-Thinking-2507"
    output_dir = "practical_export_output"
    
    exporter = PracticalModelExporter()
    
    try:
        # 检查环境
        logger.info("检查运行环境...")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️ CUDA不可用，将使用CPU")
        
        # 步骤1: 加载检查点模型
        logger.info(f"\\n步骤1: 加载检查点模型")
        logger.info(f"检查点路径: {checkpoint_path}")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ transformers库不可用，无法继续")
            logger.info("请安装: pip install transformers peft torch")
            return
        
        model, tokenizer = exporter.load_checkpoint_model(
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path
        )
        
        logger.info("✅ 模型加载成功")
        
        # 步骤2: 导出量化模型
        logger.info(f"\\n步骤2: 导出量化模型")
        
        # 根据可用性选择导出格式
        if MODEL_EXPORTER_AVAILABLE:
            formats = ["int8", "int4", "dynamic"]
            logger.info("使用自研量化器")
        else:
            formats = ["fp16"]
            logger.info("使用备用导出方法")
        
        export_results = exporter.export_quantized_models(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            formats=formats
        )
        
        # 步骤3: 生成使用示例
        logger.info(f"\\n步骤3: 生成使用示例和文档")
        exporter.generate_usage_examples(output_dir, export_results)
        
        # 步骤4: 显示结果摘要
        logger.info(f"\\n步骤4: 结果摘要")
        logger.info("=" * 40)
        
        successful_exports = [k for k, v in export_results.items() if v.get("success", False)]
        failed_exports = [k for k, v in export_results.items() if not v.get("success", False)]
        
        if successful_exports:
            logger.info(f"✅ 成功导出格式: {', '.join(successful_exports)}")
            
            for format_name in successful_exports:
                result = export_results[format_name]
                output_path = result.get("output_path", "未知")
                logger.info(f"  {format_name.upper()}: {output_path}")
                
                # 显示量化结果
                if "quantization_result" in result:
                    quant_result = result["quantization_result"]
                    if quant_result.get("success", False):
                        compression = quant_result.get("compression_ratio", 1.0)
                        accuracy = quant_result.get("accuracy_preserved", 0.0)
                        logger.info(f"    压缩比: {compression:.2f}x, 精度保持: {accuracy:.1%}")
                
                # 显示中文验证结果
                if "chinese_validation" in result:
                    chinese_result = result["chinese_validation"]
                    overall_score = chinese_result.get("overall_score", 0.0)
                    logger.info(f"    中文能力评分: {overall_score:.1%}")
        
        if failed_exports:
            logger.warning(f"⚠️ 失败格式: {', '.join(failed_exports)}")
            for format_name in failed_exports:
                error = export_results[format_name].get("error", "未知错误")
                logger.warning(f"  {format_name.upper()}: {error}")
        
        logger.info(f"\\n🎉 演示完成！")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info(f"📖 查看 export_report.json 了解详细信息")
        logger.info(f"🚀 运行 usage_examples.py 测试模型")
        
    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()