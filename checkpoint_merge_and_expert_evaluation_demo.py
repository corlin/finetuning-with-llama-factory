#!/usr/bin/env python3
"""
Checkpoint合并与专家评估完整演示

本程序演示完整的流程：
1. 将LoRA checkpoint与基座模型合并
2. 使用合并后的模型对评估数据进行专家评估
3. 生成详细的评估报告

使用方法:
    uv run python checkpoint_merge_and_expert_evaluation_demo.py

作者: 专家评估系统开发团队
"""

import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查依赖
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers和PEFT库可用")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"⚠️ Transformers库不可用: {e}")

class CheckpointMerger:
    """Checkpoint合并器"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        logger.info(f"🔧 使用设备: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def merge_lora_checkpoint(
        self, 
        checkpoint_path: str, 
        base_model_path: str = "Qwen/Qwen3-4B-Thinking-2507",
        output_path: str = "merged_model_output"
    ) -> Tuple[Any, Any]:
        """
        合并LoRA checkpoint到基座模型
        
        Args:
            checkpoint_path: LoRA checkpoint路径
            base_model_path: 基座模型路径
            output_path: 输出路径
            
        Returns:
            (merged_model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装transformers和peft库")
        
        try:
            logger.info(f"📥 加载基座模型: {base_model_path}")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 加载基座模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            logger.info(f"📥 加载LoRA checkpoint: {checkpoint_path}")
            
            # 加载LoRA模型
            model_with_lora = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("🔄 合并LoRA权重到基座模型...")
            
            # 合并权重
            merged_model = model_with_lora.merge_and_unload()
            
            # 保存合并后的模型
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 保存合并后的模型到: {output_path}")
            
            merged_model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            tokenizer.save_pretrained(output_path)
            
            # 生成合并报告
            merge_info = {
                "merge_time": datetime.now().isoformat(),
                "base_model": base_model_path,
                "checkpoint_path": checkpoint_path,
                "output_path": output_path,
                "device_used": self.device,
                "model_dtype": str(merged_model.dtype),
                "success": True
            }
            
            with open(output_dir / "merge_info.json", 'w', encoding='utf-8') as f:
                json.dump(merge_info, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ 模型合并完成!")
            return merged_model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ 模型合并失败: {e}")
            raise

class QADataProcessor:
    """QA数据处理器"""
    
    def __init__(self):
        pass
    
    def load_qa_data_from_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """从Markdown文件加载QA数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            qa_items = []
            
            # 使用正则表达式提取Q&A对
            # 匹配模式: ### Q数字: 问题内容 ... A数字: 答案内容
            qa_pattern = r'### Q(\d+):\s*(.+?)\n\n<thinking>.*?</thinking>\n\nA\1:\s*(.+?)(?=\n### Q|\n## |$)'
            
            matches = re.findall(qa_pattern, content, re.DOTALL)
            
            for match in matches:
                q_num, question, answer = match
                
                # 清理文本
                question = question.strip()
                answer = answer.strip()
                
                if question and answer:
                    qa_item = {
                        "question_id": f"qa_{q_num}",
                        "question": question,
                        "reference_answer": answer,
                        "context": "密码应用标准GB/T 39786-2021",
                        "domain_tags": ["密码学", "信息安全", "国家标准"],
                        "difficulty_level": "intermediate",
                        "expected_concepts": ["密码应用", "安全要求", "技术标准"]
                    }
                    qa_items.append(qa_item)
            
            logger.info(f"📊 从 {file_path} 提取了 {len(qa_items)} 个QA项")
            return qa_items
            
        except Exception as e:
            logger.error(f"❌ 加载QA数据失败: {e}")
            return []
    
    def load_all_qa_data(self, data_dir: str = "data/raw") -> List[Dict[str, Any]]:
        """加载所有QA数据文件"""
        data_path = Path(data_dir)
        all_qa_items = []
        
        # 加载enhanced QA文件
        enhanced_files = list(data_path.glob("enhanced_QA*.md"))
        
        for file_path in enhanced_files:
            logger.info(f"📖 处理文件: {file_path}")
            qa_items = self.load_qa_data_from_markdown(str(file_path))
            all_qa_items.extend(qa_items)
        
        logger.info(f"📊 总共加载了 {len(all_qa_items)} 个QA项")
        return all_qa_items

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 确保模型在正确的设备上
        if hasattr(model, 'to'):
            self.model = model.to(self.device)
    
    def generate_answer(self, question: str, context: str = "", max_length: int = 512) -> str:
        """使用模型生成答案"""
        try:
            # 构建prompt
            if context:
                prompt = f"上下文：{context}\n\n问题：{question}\n\n答案："
            else:
                prompt = f"问题：{question}\n\n答案："
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"❌ 生成答案失败: {e}")
            return "生成失败"

class ExpertEvaluator:
    """专家评估器（简化版）"""
    
    def __init__(self):
        self.evaluation_dimensions = [
            "semantic_similarity",
            "domain_accuracy", 
            "response_relevance",
            "factual_correctness",
            "completeness",
            "clarity",
            "technical_depth"
        ]
        
        self.dimension_weights = {
            "semantic_similarity": 0.20,
            "domain_accuracy": 0.25,
            "response_relevance": 0.15,
            "factual_correctness": 0.20,
            "completeness": 0.10,
            "clarity": 0.05,
            "technical_depth": 0.05
        }
    
    def evaluate_answer_pair(
        self, 
        question: str, 
        reference_answer: str, 
        model_answer: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """评估单个QA对"""
        
        # 简化的评估逻辑（实际应用中会使用更复杂的算法）
        dimension_scores = {}
        
        # 基于文本长度和关键词匹配的简单评估
        ref_words = set(reference_answer.lower().split())
        model_words = set(model_answer.lower().split())
        
        # 语义相似性（基于词汇重叠）
        if ref_words and model_words:
            overlap = len(ref_words & model_words)
            union = len(ref_words | model_words)
            jaccard_sim = overlap / union if union > 0 else 0
            dimension_scores["semantic_similarity"] = min(0.95, max(0.3, jaccard_sim + 0.2))
        else:
            dimension_scores["semantic_similarity"] = 0.3
        
        # 领域准确性（基于专业术语）
        domain_terms = ["密码", "加密", "安全", "算法", "认证", "完整性", "机密性", "标准"]
        ref_domain_count = sum(1 for term in domain_terms if term in reference_answer)
        model_domain_count = sum(1 for term in domain_terms if term in model_answer)
        
        if ref_domain_count > 0:
            domain_accuracy = min(1.0, model_domain_count / ref_domain_count)
        else:
            domain_accuracy = 0.7
        dimension_scores["domain_accuracy"] = max(0.4, domain_accuracy)
        
        # 响应相关性（基于问题关键词）
        question_words = set(question.lower().split())
        question_relevance = len(question_words & model_words) / len(question_words) if question_words else 0
        dimension_scores["response_relevance"] = min(0.95, max(0.5, question_relevance + 0.3))
        
        # 事实正确性（基于答案长度和结构）
        if len(model_answer) > 20 and "。" in model_answer:
            dimension_scores["factual_correctness"] = min(0.9, max(0.6, len(model_answer) / 200))
        else:
            dimension_scores["factual_correctness"] = 0.5
        
        # 完整性（基于答案长度比较）
        length_ratio = len(model_answer) / len(reference_answer) if reference_answer else 0
        if 0.5 <= length_ratio <= 1.5:
            dimension_scores["completeness"] = min(0.9, 0.7 + length_ratio * 0.1)
        else:
            dimension_scores["completeness"] = max(0.4, 0.8 - abs(length_ratio - 1.0) * 0.3)
        
        # 清晰度（基于句子结构）
        sentences = model_answer.count("。") + model_answer.count("！") + model_answer.count("？")
        if sentences > 0 and len(model_answer) / sentences < 100:
            dimension_scores["clarity"] = 0.8
        else:
            dimension_scores["clarity"] = 0.6
        
        # 技术深度（基于专业术语密度）
        tech_density = model_domain_count / len(model_answer.split()) if model_answer.split() else 0
        dimension_scores["technical_depth"] = min(0.9, max(0.4, tech_density * 10))
        
        # 计算加权总分
        overall_score = sum(
            dimension_scores[dim] * self.dimension_weights[dim]
            for dim in dimension_scores
        )
        
        # 生成改进建议
        suggestions = []
        if dimension_scores["semantic_similarity"] < 0.7:
            suggestions.append("提高答案与参考答案的语义相似性")
        if dimension_scores["domain_accuracy"] < 0.7:
            suggestions.append("增加更多专业术语和概念")
        if dimension_scores["completeness"] < 0.7:
            suggestions.append("提供更完整和详细的回答")
        if dimension_scores["technical_depth"] < 0.6:
            suggestions.append("增强技术深度和专业性")
        
        if not suggestions:
            suggestions.append("继续保持当前的高质量水平")
        
        return {
            "overall_score": round(overall_score, 3),
            "dimension_scores": {k: round(v, 3) for k, v in dimension_scores.items()},
            "improvement_suggestions": suggestions,
            "evaluation_time": datetime.now().isoformat()
        }
    
    def evaluate_batch(
        self, 
        qa_items: List[Dict[str, Any]], 
        model_answers: List[str]
    ) -> Dict[str, Any]:
        """批量评估"""
        
        if len(qa_items) != len(model_answers):
            raise ValueError("QA项目数量与模型答案数量不匹配")
        
        individual_results = []
        
        for i, (qa_item, model_answer) in enumerate(zip(qa_items, model_answers)):
            logger.info(f"📊 评估第 {i+1}/{len(qa_items)} 项: {qa_item['question_id']}")
            
            result = self.evaluate_answer_pair(
                question=qa_item["question"],
                reference_answer=qa_item["reference_answer"],
                model_answer=model_answer,
                context=qa_item.get("context", "")
            )
            
            result["question_id"] = qa_item["question_id"]
            result["question"] = qa_item["question"]
            result["reference_answer"] = qa_item["reference_answer"]
            result["model_answer"] = model_answer
            
            individual_results.append(result)
        
        # 计算统计信息
        scores = [r["overall_score"] for r in individual_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # 计算各维度平均分
        dimension_averages = {}
        for dim in self.evaluation_dimensions:
            dim_scores = [r["dimension_scores"][dim] for r in individual_results]
            dimension_averages[dim] = sum(dim_scores) / len(dim_scores)
        
        return {
            "summary": {
                "total_evaluations": len(individual_results),
                "average_score": round(avg_score, 3),
                "max_score": round(max_score, 3),
                "min_score": round(min_score, 3),
                "score_std": round(
                    (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5, 3
                )
            },
            "dimension_averages": {k: round(v, 3) for k, v in dimension_averages.items()},
            "individual_results": individual_results,
            "evaluation_time": datetime.now().isoformat()
        }

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "evaluation_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(
        self, 
        merge_info: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        qa_data: List[Dict[str, Any]]
    ) -> str:
        """生成综合评估报告"""
        
        report_data = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Checkpoint合并与专家评估报告",
                "version": "1.0.0"
            },
            "merge_summary": merge_info,
            "evaluation_summary": evaluation_results["summary"],
            "dimension_performance": evaluation_results["dimension_averages"],
            "detailed_results": evaluation_results["individual_results"],
            "data_statistics": {
                "total_qa_items": len(qa_data),
                "data_sources": list(set(item.get("context", "未知") for item in qa_data)),
                "difficulty_distribution": self._analyze_difficulty_distribution(qa_data),
                "domain_distribution": self._analyze_domain_distribution(qa_data)
            }
        }
        
        # 保存JSON报告
        json_path = self.output_dir / "comprehensive_evaluation_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        html_path = self.output_dir / "comprehensive_evaluation_report.html"
        html_content = self._generate_html_report(report_data)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📋 综合报告已生成:")
        logger.info(f"   JSON: {json_path}")
        logger.info(f"   HTML: {html_path}")
        
        return str(html_path)
    
    def _analyze_difficulty_distribution(self, qa_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析难度分布"""
        distribution = {}
        for item in qa_data:
            difficulty = item.get("difficulty_level", "unknown")
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _analyze_domain_distribution(self, qa_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析领域分布"""
        distribution = {}
        for item in qa_data:
            tags = item.get("domain_tags", [])
            for tag in tags:
                distribution[tag] = distribution.get(tag, 0) + 1
        return distribution
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkpoint合并与专家评估报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
        .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .score-excellent {{ color: #28a745; font-weight: bold; }}
        .score-good {{ color: #17a2b8; font-weight: bold; }}
        .score-fair {{ color: #ffc107; font-weight: bold; }}
        .score-poor {{ color: #dc3545; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
        .chart-container {{ margin: 20px 0; }}
        .recommendation {{ background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Checkpoint合并与专家评估报告</h1>
        <p><strong>生成时间:</strong> {report_data['report_info']['generated_at']}</p>
        <p><strong>报告版本:</strong> {report_data['report_info']['version']}</p>
    </div>
    
    <div class="section">
        <h2>🔄 模型合并信息</h2>
        <div class="metric">基座模型: {report_data['merge_summary'].get('base_model', 'N/A')}</div>
        <div class="metric">Checkpoint路径: {report_data['merge_summary'].get('checkpoint_path', 'N/A')}</div>
        <div class="metric">合并时间: {report_data['merge_summary'].get('merge_time', 'N/A')}</div>
        <div class="metric">使用设备: {report_data['merge_summary'].get('device_used', 'N/A')}</div>
        <div class="metric">模型精度: {report_data['merge_summary'].get('model_dtype', 'N/A')}</div>
    </div>
    
    <div class="section">
        <h2>📊 评估概要</h2>
        <div class="metric">总评估项目: {report_data['evaluation_summary']['total_evaluations']}</div>
        <div class="metric">平均得分: <span class="score-{self._get_score_class(report_data['evaluation_summary']['average_score'])}">{report_data['evaluation_summary']['average_score']}</span></div>
        <div class="metric">最高得分: <span class="score-{self._get_score_class(report_data['evaluation_summary']['max_score'])}">{report_data['evaluation_summary']['max_score']}</span></div>
        <div class="metric">最低得分: <span class="score-{self._get_score_class(report_data['evaluation_summary']['min_score'])}">{report_data['evaluation_summary']['min_score']}</span></div>
        <div class="metric">得分标准差: {report_data['evaluation_summary']['score_std']}</div>
    </div>
    
    <div class="section">
        <h2>📈 维度表现分析</h2>
        <table>
            <tr><th>评估维度</th><th>平均得分</th><th>表现等级</th><th>得分可视化</th></tr>
        """
        
        for dim, score in report_data['dimension_performance'].items():
            score_class = self._get_score_class(score)
            grade = self._get_grade_text(score)
            progress_width = int(score * 100)
            
            html += f"""
            <tr>
                <td>{self._translate_dimension(dim)}</td>
                <td class="{score_class}">{score}</td>
                <td class="{score_class}">{grade}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress_width}%"></div>
                    </div>
                </td>
            </tr>
            """
        
        html += f"""
        </table>
    </div>
    
    <div class="section">
        <h2>📊 数据统计</h2>
        <div class="metric">数据来源: {', '.join(report_data['data_statistics']['data_sources'])}</div>
        
        <h3>难度分布</h3>
        <table>
            <tr><th>难度级别</th><th>题目数量</th></tr>
        """
        
        for difficulty, count in report_data['data_statistics']['difficulty_distribution'].items():
            html += f"<tr><td>{difficulty}</td><td>{count}</td></tr>"
        
        html += f"""
        </table>
        
        <h3>领域分布</h3>
        <table>
            <tr><th>领域标签</th><th>出现次数</th></tr>
        """
        
        for domain, count in report_data['data_statistics']['domain_distribution'].items():
            html += f"<tr><td>{domain}</td><td>{count}</td></tr>"
        
        # 添加改进建议
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>💡 改进建议</h2>
        """
        
        avg_score = report_data['evaluation_summary']['average_score']
        if avg_score < 0.7:
            html += '<div class="warning">⚠️ 整体表现需要改进，建议重新训练或调整模型参数</div>'
        elif avg_score < 0.8:
            html += '<div class="recommendation">📈 表现良好，可以通过针对性优化进一步提升</div>'
        else:
            html += '<div class="recommendation">🎉 表现优秀，继续保持当前水平</div>'
        
        # 基于维度表现给出具体建议
        poor_dimensions = [dim for dim, score in report_data['dimension_performance'].items() if score < 0.7]
        if poor_dimensions:
            html += f'<div class="recommendation">🎯 重点改进维度: {", ".join([self._translate_dimension(d) for d in poor_dimensions])}</div>'
        
        html += """
    </div>
    
    <div class="section">
        <h2>🔍 详细结果</h2>
        <p>详细的单项评估结果请查看JSON报告文件。</p>
    </div>
    
    <div class="section">
        <p><em>本报告由专家评估系统自动生成 - {}</em></p>
    </div>
</body>
</html>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return html
    
    def _get_score_class(self, score: float) -> str:
        """获取分数对应的CSS类"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "fair"
        else:
            return "poor"
    
    def _get_grade_text(self, score: float) -> str:
        """获取分数对应的等级文本"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "一般"
        else:
            return "待改进"
    
    def _translate_dimension(self, dimension: str) -> str:
        """翻译维度名称"""
        translations = {
            "semantic_similarity": "语义相似性",
            "domain_accuracy": "领域准确性",
            "response_relevance": "响应相关性",
            "factual_correctness": "事实正确性",
            "completeness": "完整性",
            "clarity": "清晰度",
            "technical_depth": "技术深度"
        }
        return translations.get(dimension, dimension)

class ComprehensiveDemo:
    """综合演示类"""
    
    def __init__(self, output_dir: str = "comprehensive_demo_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个组件
        self.merger = CheckpointMerger()
        self.qa_processor = QADataProcessor()
        self.evaluator = ExpertEvaluator()
        self.report_generator = ReportGenerator(str(self.output_dir))
        
        logger.info(f"🚀 综合演示初始化完成，输出目录: {self.output_dir}")
    
    def run_complete_pipeline(
        self,
        checkpoint_path: str = "qwen3_4b_thinking_output/final_model",
        base_model_path: str = "Qwen/Qwen3-4B-Thinking-2507",
        qa_data_dir: str = "data/raw"
    ):
        """运行完整的pipeline"""
        
        try:
            logger.info("🎯 开始执行完整的Checkpoint合并与专家评估流程")
            
            # 步骤1: 合并模型
            logger.info("\n" + "="*60)
            logger.info("📋 步骤1: 合并LoRA Checkpoint到基座模型")
            logger.info("="*60)
            
            merged_model_path = self.output_dir / "merged_model"
            
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("⚠️ Transformers不可用，跳过模型合并步骤")
                merge_info = {
                    "merge_time": datetime.now().isoformat(),
                    "base_model": base_model_path,
                    "checkpoint_path": checkpoint_path,
                    "output_path": str(merged_model_path),
                    "success": False,
                    "error": "Transformers库不可用"
                }
                model = None
                tokenizer = None
            else:
                model, tokenizer = self.merger.merge_lora_checkpoint(
                    checkpoint_path=checkpoint_path,
                    base_model_path=base_model_path,
                    output_path=str(merged_model_path)
                )
                
                # 读取合并信息
                with open(merged_model_path / "merge_info.json", 'r', encoding='utf-8') as f:
                    merge_info = json.load(f)
            
            # 步骤2: 加载QA数据
            logger.info("\n" + "="*60)
            logger.info("📋 步骤2: 加载评估数据")
            logger.info("="*60)
            
            qa_data = self.qa_processor.load_all_qa_data(qa_data_dir)
            
            if not qa_data:
                logger.error("❌ 没有加载到QA数据，程序终止")
                return
            
            # 步骤3: 生成模型答案
            logger.info("\n" + "="*60)
            logger.info("📋 步骤3: 使用合并后的模型生成答案")
            logger.info("="*60)
            
            model_answers = []
            
            if model and tokenizer:
                model_evaluator = ModelEvaluator(model, tokenizer, self.merger.device)
                
                for i, qa_item in enumerate(qa_data):
                    logger.info(f"🤖 生成答案 {i+1}/{len(qa_data)}: {qa_item['question_id']}")
                    
                    try:
                        answer = model_evaluator.generate_answer(
                            question=qa_item["question"],
                            context=qa_item.get("context", ""),
                            max_length=256
                        )
                        model_answers.append(answer)
                        
                        # 显示生成的答案（截断显示）
                        display_answer = answer[:100] + "..." if len(answer) > 100 else answer
                        logger.info(f"   生成答案: {display_answer}")
                        
                    except Exception as e:
                        logger.error(f"❌ 生成答案失败: {e}")
                        model_answers.append("生成失败")
            else:
                logger.warning("⚠️ 模型不可用，使用模拟答案")
                # 使用模拟答案进行演示
                for qa_item in qa_data:
                    simulated_answer = f"这是对问题'{qa_item['question'][:20]}...'的模拟回答。在实际应用中，这里会是合并后模型生成的答案。"
                    model_answers.append(simulated_answer)
            
            # 步骤4: 专家评估
            logger.info("\n" + "="*60)
            logger.info("📋 步骤4: 执行专家评估")
            logger.info("="*60)
            
            evaluation_results = self.evaluator.evaluate_batch(qa_data, model_answers)
            
            logger.info(f"📊 评估完成:")
            logger.info(f"   总项目数: {evaluation_results['summary']['total_evaluations']}")
            logger.info(f"   平均得分: {evaluation_results['summary']['average_score']}")
            logger.info(f"   最高得分: {evaluation_results['summary']['max_score']}")
            logger.info(f"   最低得分: {evaluation_results['summary']['min_score']}")
            
            # 步骤5: 生成报告
            logger.info("\n" + "="*60)
            logger.info("📋 步骤5: 生成综合评估报告")
            logger.info("="*60)
            
            report_path = self.report_generator.generate_comprehensive_report(
                merge_info=merge_info,
                evaluation_results=evaluation_results,
                qa_data=qa_data
            )
            
            # 步骤6: 总结
            logger.info("\n" + "="*60)
            logger.info("🎉 完整流程执行完成")
            logger.info("="*60)
            
            logger.info(f"📁 输出目录: {self.output_dir}")
            logger.info(f"📋 HTML报告: {report_path}")
            logger.info(f"📊 平均评估得分: {evaluation_results['summary']['average_score']}")
            
            # 显示最佳和最差表现的维度
            dim_scores = evaluation_results['dimension_averages']
            best_dim = max(dim_scores.items(), key=lambda x: x[1])
            worst_dim = min(dim_scores.items(), key=lambda x: x[1])
            
            logger.info(f"🏆 最佳维度: {best_dim[0]} ({best_dim[1]:.3f})")
            logger.info(f"📉 待改进维度: {worst_dim[0]} ({worst_dim[1]:.3f})")
            
            # 生成简要建议
            avg_score = evaluation_results['summary']['average_score']
            if avg_score >= 0.8:
                logger.info("✨ 模型表现优秀，可以考虑部署使用")
            elif avg_score >= 0.7:
                logger.info("📈 模型表现良好，建议进一步优化")
            else:
                logger.info("⚠️ 模型表现需要改进，建议重新训练")
            
            logger.info("\n🎯 查看详细报告请打开生成的HTML文件")
            
        except Exception as e:
            logger.error(f"❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    
    print("🚀 Checkpoint合并与专家评估完整演示")
    print("=" * 60)
    
    # 检查必要的路径
    checkpoint_path = "qwen3_4b_thinking_output/final_model"
    qa_data_dir = "data/raw"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint路径不存在: {checkpoint_path}")
        return
    
    if not Path(qa_data_dir).exists():
        logger.error(f"❌ QA数据目录不存在: {qa_data_dir}")
        return
    
    try:
        # 创建并运行演示
        demo = ComprehensiveDemo()
        demo.run_complete_pipeline(
            checkpoint_path=checkpoint_path,
            qa_data_dir=qa_data_dir
        )
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ 用户中断程序执行")
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()