"""
深度思考数据生成器

本模块实现自动thinking过程生成算法、思考逻辑连贯性验证、
密码学推理过程生成器和thinking数据质量评估功能。
"""

import re
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import jieba
import jieba.posseg as pseg

from data_models import (
    ThinkingExample, ThinkingStructure, ReasoningStep, 
    CryptoTerm, CryptoCategory, DifficultyLevel, ChineseMetrics
)


class ThinkingType(Enum):
    """思考类型枚举"""
    ANALYTICAL = "分析型"
    DEDUCTIVE = "演绎型"
    INDUCTIVE = "归纳型"
    COMPARATIVE = "比较型"
    PROBLEM_SOLVING = "问题解决型"
    CRYPTOGRAPHIC = "密码学推理型"


class ReasoningPattern(Enum):
    """推理模式枚举"""
    LINEAR = "线性推理"
    BRANCHING = "分支推理"
    ITERATIVE = "迭代推理"
    HIERARCHICAL = "层次推理"
    COMPARATIVE = "比较推理"


@dataclass
class ThinkingTemplate:
    """思考模板"""
    name: str
    pattern: ReasoningPattern
    steps: List[str]
    crypto_focus: bool = False
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    
    def generate_structure(self, context: Dict[str, Any]) -> List[str]:
        """根据上下文生成思考结构"""
        generated_steps = []
        for step_template in self.steps:
            # 安全的模板替换，处理缺失的键
            try:
                step = step_template.format(**context)
            except KeyError as e:
                # 如果缺少键，用占位符替换
                missing_key = str(e).strip("'")
                context[missing_key] = f"[{missing_key}]"
                step = step_template.format(**context)
            generated_steps.append(step)
        return generated_steps


class ThinkingDataGenerator:
    """深度思考数据生成器"""
    
    def __init__(self):
        self.crypto_terms_db: Dict[str, CryptoTerm] = {}
        self.thinking_templates = self._initialize_templates()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        # 初始化jieba分词
        jieba.initialize()
    
    def _initialize_templates(self) -> Dict[str, ThinkingTemplate]:
        """初始化思考模板"""
        templates = {}
        
        # 密码学分析模板
        templates["crypto_analysis"] = ThinkingTemplate(
            name="密码学分析",
            pattern=ReasoningPattern.HIERARCHICAL,
            steps=[
                "首先，我需要理解这个密码学问题的核心：{problem_core}",
                "让我分析涉及的密码学概念：{crypto_concepts}",
                "接下来考虑安全性要求：{security_requirements}",
                "分析可能的攻击向量：{attack_vectors}",
                "评估不同解决方案的优缺点：{solution_analysis}",
                "得出最佳方案并验证其安全性：{final_solution}"
            ],
            crypto_focus=True,
            difficulty=DifficultyLevel.ADVANCED
        )
        
        # 算法比较模板
        templates["algorithm_comparison"] = ThinkingTemplate(
            name="算法比较",
            pattern=ReasoningPattern.COMPARATIVE,
            steps=[
                "我需要比较这些算法：{algorithms}",
                "从安全性角度分析：{security_analysis}",
                "从性能角度考虑：{performance_analysis}",
                "考虑实际应用场景：{application_scenarios}",
                "综合评估得出结论：{conclusion}"
            ],
            crypto_focus=True,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # 问题解决模板
        templates["problem_solving"] = ThinkingTemplate(
            name="问题解决",
            pattern=ReasoningPattern.LINEAR,
            steps=[
                "理解问题：{problem_understanding}",
                "分析已知条件：{known_conditions}",
                "确定解决策略：{solution_strategy}",
                "逐步实施解决方案：{implementation_steps}",
                "验证结果正确性：{result_verification}"
            ],
            crypto_focus=False,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # 概念解释模板
        templates["concept_explanation"] = ThinkingTemplate(
            name="概念解释",
            pattern=ReasoningPattern.HIERARCHICAL,
            steps=[
                "这个概念的定义是：{concept_definition}",
                "它的核心特征包括：{key_features}",
                "与相关概念的区别：{concept_differences}",
                "实际应用举例：{practical_examples}",
                "总结要点：{summary_points}"
            ],
            crypto_focus=False,
            difficulty=DifficultyLevel.BEGINNER
        )
        
        return templates
    
    def _initialize_reasoning_patterns(self) -> Dict[str, List[str]]:
        """初始化推理模式"""
        return {
            "cause_effect": [
                "分析原因", "推导过程", "得出结果", "验证因果关系"
            ],
            "hypothesis_testing": [
                "提出假设", "设计验证方法", "收集证据", "验证假设", "得出结论"
            ],
            "step_by_step": [
                "第一步", "第二步", "第三步", "综合分析", "最终结论"
            ],
            "pros_cons": [
                "列出优点", "分析缺点", "权衡利弊", "得出平衡方案"
            ]
        }
    
    def generate_thinking_process(
        self, 
        instruction: str, 
        context: Optional[Dict[str, Any]] = None,
        thinking_type: ThinkingType = ThinkingType.ANALYTICAL,
        target_length: int = 200
    ) -> str:
        """
        生成自动thinking过程
        
        Args:
            instruction: 指令文本
            context: 上下文信息
            thinking_type: 思考类型
            target_length: 目标长度（字符数）
            
        Returns:
            生成的thinking过程文本
        """
        if context is None:
            context = {}
        
        # 分析指令内容
        instruction_analysis = self._analyze_instruction(instruction)
        context.update(instruction_analysis)
        
        # 选择合适的模板
        template = self._select_template(thinking_type, instruction_analysis)
        
        # 生成思考步骤
        thinking_steps = template.generate_structure(context)
        
        # 扩展和细化步骤
        detailed_steps = self._elaborate_thinking_steps(
            thinking_steps, instruction_analysis, target_length
        )
        
        # 组装完整的thinking过程
        thinking_process = self._assemble_thinking_process(detailed_steps)
        
        return thinking_process
    
    def _analyze_instruction(self, instruction: str) -> Dict[str, Any]:
        """分析指令内容"""
        analysis = {
            "problem_core": "",
            "crypto_concepts": [],
            "security_requirements": [],
            "attack_vectors": [],
            "algorithms": [],
            "known_conditions": [],
            "concept_definition": "",
            "key_features": [],
            "practical_examples": []
        }
        
        # 使用jieba进行分词和词性标注
        words = pseg.cut(instruction)
        
        # 提取关键信息
        crypto_keywords = [
            "加密", "解密", "哈希", "签名", "密钥", "算法", "安全", "攻击",
            "RSA", "AES", "DES", "SHA", "MD5", "椭圆曲线", "数字签名"
        ]
        
        found_crypto_terms = []
        for word, flag in words:
            if word in crypto_keywords or any(kw in word for kw in crypto_keywords):
                found_crypto_terms.append(word)
        
        analysis["crypto_concepts"] = found_crypto_terms
        
        # 简单的问题核心提取
        if "什么是" in instruction or "如何" in instruction:
            analysis["problem_core"] = "概念理解和应用"
        elif "比较" in instruction or "区别" in instruction:
            analysis["problem_core"] = "算法或概念比较"
        elif "安全" in instruction or "攻击" in instruction:
            analysis["problem_core"] = "安全性分析"
        else:
            analysis["problem_core"] = "综合分析"
        
        return analysis
    
    def _select_template(
        self, 
        thinking_type: ThinkingType, 
        instruction_analysis: Dict[str, Any]
    ) -> ThinkingTemplate:
        """选择合适的思考模板"""
        
        # 根据指令分析选择模板
        problem_core = instruction_analysis.get("problem_core", "")
        
        if instruction_analysis["crypto_concepts"]:
            if "比较" in problem_core:
                return self.thinking_templates["algorithm_comparison"]
            else:
                return self.thinking_templates["crypto_analysis"]
        elif "概念理解和应用" in problem_core:
            return self.thinking_templates["concept_explanation"]
        else:
            return self.thinking_templates["problem_solving"]
    
    def _elaborate_thinking_steps(
        self, 
        steps: List[str], 
        analysis: Dict[str, Any], 
        target_length: int
    ) -> List[str]:
        """扩展和细化思考步骤"""
        detailed_steps = []
        
        for step in steps:
            # 基于分析结果扩展步骤
            if "密码学概念" in step and analysis["crypto_concepts"]:
                detailed_step = f"{step} 涉及的概念包括：{', '.join(analysis['crypto_concepts'])}。"
            elif "安全性" in step:
                detailed_step = f"{step} 需要考虑机密性、完整性、可用性等方面。"
            elif "算法" in step and analysis["crypto_concepts"]:
                detailed_step = f"{step} 主要算法：{', '.join(analysis['crypto_concepts'])}。"
            else:
                detailed_step = step
            
            # 添加推理细节
            detailed_step += self._add_reasoning_details(step, analysis)
            detailed_steps.append(detailed_step)
        
        return detailed_steps
    
    def _add_reasoning_details(self, step: str, analysis: Dict[str, Any]) -> str:
        """添加推理细节"""
        details = []
        
        if "分析" in step:
            details.append("我需要从多个角度来考虑这个问题。")
        if "安全" in step:
            details.append("安全性是密码学的核心要求。")
        if "算法" in step:
            details.append("不同算法有各自的特点和适用场景。")
        
        return " " + " ".join(details) if details else ""
    
    def _assemble_thinking_process(self, steps: List[str]) -> str:
        """组装完整的thinking过程"""
        thinking_parts = []
        
        for i, step in enumerate(steps, 1):
            thinking_parts.append(f"{step}")
            
            # 在步骤间添加连接词
            if i < len(steps):
                connectors = ["接下来，", "然后，", "进一步，", "此外，", "最后，"]
                if i < len(connectors):
                    thinking_parts.append(f"\n\n{connectors[i-1]}")
                else:
                    thinking_parts.append("\n\n")
        
        return "".join(thinking_parts)
    
    def generate_crypto_reasoning(
        self, 
        crypto_problem: str, 
        crypto_terms: List[CryptoTerm]
    ) -> ThinkingStructure:
        """
        生成密码学推理过程
        
        Args:
            crypto_problem: 密码学问题描述
            crypto_terms: 相关密码学术语
            
        Returns:
            结构化的思考数据
        """
        # 分析密码学问题类型
        problem_type = self._classify_crypto_problem(crypto_problem)
        
        # 生成推理步骤
        reasoning_steps = self._generate_crypto_reasoning_steps(
            crypto_problem, crypto_terms, problem_type
        )
        
        # 构建thinking结构
        raw_thinking = self._build_crypto_thinking_text(reasoning_steps)
        parsed_steps = [step.description for step in reasoning_steps]
        
        thinking_structure = ThinkingStructure(
            raw_thinking=raw_thinking,
            parsed_steps=parsed_steps,
            reasoning_chain=reasoning_steps,
            validation_result=True,
            thinking_depth=len(reasoning_steps),
            logical_consistency=0.9,  # 密码学推理通常逻辑性较强
            completeness_score=0.85
        )
        
        return thinking_structure
    
    def _classify_crypto_problem(self, problem: str) -> str:
        """分类密码学问题类型"""
        if any(word in problem for word in ["加密", "解密", "cipher"]):
            return "encryption_decryption"
        elif any(word in problem for word in ["哈希", "散列", "hash"]):
            return "hash_function"
        elif any(word in problem for word in ["签名", "signature"]):
            return "digital_signature"
        elif any(word in problem for word in ["密钥", "key"]):
            return "key_management"
        elif any(word in problem for word in ["攻击", "破解", "attack"]):
            return "cryptanalysis"
        else:
            return "general_crypto"
    
    def _generate_crypto_reasoning_steps(
        self, 
        problem: str, 
        crypto_terms: List[CryptoTerm], 
        problem_type: str
    ) -> List[ReasoningStep]:
        """生成密码学推理步骤"""
        steps = []
        
        # 步骤1：问题理解
        steps.append(ReasoningStep(
            step_number=1,
            description="理解密码学问题的核心要求",
            input_data=problem,
            reasoning_process=f"这是一个关于{problem_type}的问题，需要分析其安全需求和技术要求。",
            output_result="明确了问题的类型和核心要求",
            confidence_score=0.9
        ))
        
        # 步骤2：相关概念分析
        if crypto_terms:
            term_names = [term.term for term in crypto_terms]
            steps.append(ReasoningStep(
                step_number=2,
                description="分析相关密码学概念",
                input_data=f"涉及概念：{', '.join(term_names)}",
                reasoning_process="需要理解每个概念的定义、特性和相互关系。",
                output_result="建立了概念框架",
                confidence_score=0.85
            ))
        
        # 步骤3：安全性分析
        steps.append(ReasoningStep(
            step_number=len(steps) + 1,
            description="进行安全性分析",
            input_data="安全需求和威胁模型",
            reasoning_process="分析可能的攻击方式和防护措施，评估安全强度。",
            output_result="确定了安全要求和防护策略",
            confidence_score=0.8
        ))
        
        # 步骤4：解决方案设计
        steps.append(ReasoningStep(
            step_number=len(steps) + 1,
            description="设计解决方案",
            input_data="问题要求和安全约束",
            reasoning_process="基于分析结果，选择合适的算法和实现方案。",
            output_result="提出了具体的解决方案",
            confidence_score=0.85
        ))
        
        # 步骤5：方案验证
        steps.append(ReasoningStep(
            step_number=len(steps) + 1,
            description="验证方案的正确性和安全性",
            input_data="提出的解决方案",
            reasoning_process="检查方案是否满足所有要求，是否存在安全漏洞。",
            output_result="确认方案的可行性和安全性",
            confidence_score=0.9
        ))
        
        return steps
    
    def _build_crypto_thinking_text(self, reasoning_steps: List[ReasoningStep]) -> str:
        """构建密码学thinking文本"""
        thinking_parts = []
        
        for step in reasoning_steps:
            thinking_parts.append(
                f"{step.description}：{step.reasoning_process} "
                f"基于{step.input_data}，{step.output_result}。"
            )
        
        return "\n\n".join(thinking_parts)
    
    def validate_thinking_coherence(self, thinking_text: str) -> Dict[str, Any]:
        """
        验证思考逻辑连贯性
        
        Args:
            thinking_text: 思考过程文本
            
        Returns:
            验证结果字典
        """
        result = {
            "coherence_score": 0.0,
            "issues": [],
            "suggestions": [],
            "logical_flow": True,
            "completeness": 0.0
        }
        
        # 检查基本结构
        if not thinking_text.strip():
            result["issues"].append("思考内容为空")
            return result
        
        # 分析句子结构和逻辑连接
        sentences = re.split(r'[。！？]', thinking_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            result["issues"].append("思考过程过于简短，缺乏深度")
            result["suggestions"].append("建议增加更多推理步骤")
        
        # 检查逻辑连接词
        logical_connectors = [
            "因此", "所以", "然后", "接下来", "首先", "其次", "最后",
            "由于", "因为", "但是", "然而", "此外", "另外"
        ]
        
        connector_count = sum(1 for sentence in sentences 
                            for connector in logical_connectors 
                            if connector in sentence)
        
        connector_ratio = connector_count / len(sentences) if sentences else 0
        
        # 计算连贯性分数
        if connector_ratio > 0.3:
            result["coherence_score"] = 0.9
        elif connector_ratio > 0.2:
            result["coherence_score"] = 0.7
        elif connector_ratio > 0.1:
            result["coherence_score"] = 0.5
        else:
            result["coherence_score"] = 0.3
            result["issues"].append("缺乏逻辑连接词，思考过程不够连贯")
            result["suggestions"].append("建议使用更多逻辑连接词")
        
        # 检查完整性
        thinking_elements = {
            "问题分析": any(word in thinking_text for word in ["分析", "理解", "问题"]),
            "推理过程": any(word in thinking_text for word in ["推理", "推导", "因此"]),
            "结论验证": any(word in thinking_text for word in ["验证", "检查", "确认"])
        }
        
        completeness = sum(thinking_elements.values()) / len(thinking_elements)
        result["completeness"] = completeness
        
        if completeness < 0.6:
            result["issues"].append("思考过程不够完整")
            missing_elements = [k for k, v in thinking_elements.items() if not v]
            result["suggestions"].append(f"建议补充：{', '.join(missing_elements)}")
        
        return result
    
    def assess_thinking_quality(self, thinking_example: ThinkingExample) -> Dict[str, Any]:
        """
        评估thinking数据质量
        
        Args:
            thinking_example: 思考样例
            
        Returns:
            质量评估结果
        """
        assessment = {
            "overall_quality": 0.0,
            "dimensions": {
                "coherence": 0.0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "depth": 0.0,
                "clarity": 0.0
            },
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": []
        }
        
        # 连贯性评估
        coherence_result = self.validate_thinking_coherence(thinking_example.thinking_process)
        assessment["dimensions"]["coherence"] = coherence_result["coherence_score"]
        
        # 完整性评估
        assessment["dimensions"]["completeness"] = coherence_result["completeness"]
        
        # 深度评估
        depth_score = self._assess_thinking_depth(thinking_example.thinking_process)
        assessment["dimensions"]["depth"] = depth_score
        
        # 清晰度评估
        clarity_score = self._assess_thinking_clarity(thinking_example.thinking_process)
        assessment["dimensions"]["clarity"] = clarity_score
        
        # 准确性评估（基于密码学术语使用）
        accuracy_score = self._assess_crypto_accuracy(
            thinking_example.thinking_process, 
            thinking_example.crypto_terms
        )
        assessment["dimensions"]["accuracy"] = accuracy_score
        
        # 计算总体质量
        weights = {
            "coherence": 0.25,
            "completeness": 0.20,
            "accuracy": 0.25,
            "depth": 0.15,
            "clarity": 0.15
        }
        
        overall_quality = sum(
            assessment["dimensions"][dim] * weight 
            for dim, weight in weights.items()
        )
        assessment["overall_quality"] = overall_quality
        
        # 生成优缺点和建议
        self._generate_quality_feedback(assessment)
        
        return assessment
    
    def _assess_thinking_depth(self, thinking_text: str) -> float:
        """评估思考深度"""
        # 简单的深度评估：基于推理层次和复杂度
        depth_indicators = [
            "深入分析", "进一步", "更深层次", "根本原因", "本质",
            "多角度", "综合考虑", "权衡", "trade-off"
        ]
        
        depth_count = sum(1 for indicator in depth_indicators 
                         if indicator in thinking_text)
        
        # 基于文本长度和深度指标计算分数
        text_length = len(thinking_text)
        if text_length > 300 and depth_count > 2:
            return 0.9
        elif text_length > 200 and depth_count > 1:
            return 0.8
        elif text_length > 100 and depth_count > 0:
            return 0.7
        elif text_length > 100:
            return 0.6
        elif text_length > 50:
            return 0.5
        else:
            return 0.3
    
    def _assess_thinking_clarity(self, thinking_text: str) -> float:
        """评估思考清晰度"""
        # 基于句子结构和表达清晰度
        sentences = re.split(r'[。！？]', thinking_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 计算平均句子长度
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 理想句子长度在15-40字符之间
        if 15 <= avg_sentence_length <= 40:
            length_score = 1.0
        elif 10 <= avg_sentence_length <= 50:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # 检查表达清晰度指标
        clarity_indicators = [
            "具体来说", "换句话说", "也就是说", "例如", "比如",
            "总结", "综上", "简而言之"
        ]
        
        clarity_count = sum(1 for indicator in clarity_indicators 
                          if indicator in thinking_text)
        clarity_ratio = clarity_count / len(sentences)
        
        clarity_score = min(1.0, clarity_ratio * 3)  # 最多3个清晰度指标得满分
        
        return (length_score + clarity_score) / 2
    
    def _assess_crypto_accuracy(self, thinking_text: str, crypto_terms: List[str]) -> float:
        """评估密码学准确性"""
        if not crypto_terms:
            return 0.7  # 没有密码学术语时给中等分数
        
        # 检查术语使用的准确性
        correct_usage = 0
        total_usage = 0
        
        for term in crypto_terms:
            if term in thinking_text:
                total_usage += 1
                # 简单的上下文检查
                if self._check_term_context(term, thinking_text):
                    correct_usage += 1
        
        if total_usage == 0:
            return 0.5  # 没有使用术语
        
        return correct_usage / total_usage
    
    def _check_term_context(self, term: str, text: str) -> bool:
        """检查术语使用的上下文是否合理"""
        # 简化实现：检查术语周围是否有相关词汇
        term_index = text.find(term)
        if term_index == -1:
            return False
        
        # 获取术语前后的上下文
        start = max(0, term_index - 50)
        end = min(len(text), term_index + len(term) + 50)
        context = text[start:end]
        
        # 检查是否有相关的密码学词汇
        related_words = [
            "安全", "加密", "解密", "算法", "密钥", "哈希", "签名",
            "攻击", "防护", "协议", "认证", "完整性", "机密性"
        ]
        
        return any(word in context for word in related_words)
    
    def _generate_quality_feedback(self, assessment: Dict[str, Any]):
        """生成质量反馈"""
        dimensions = assessment["dimensions"]
        
        # 识别优势
        for dim, score in dimensions.items():
            if score >= 0.8:
                assessment["strengths"].append(f"{dim}表现优秀")
        
        # 识别弱点
        for dim, score in dimensions.items():
            if score < 0.6:
                assessment["weaknesses"].append(f"{dim}需要改进")
        
        # 生成改进建议
        if dimensions["coherence"] < 0.6:
            assessment["improvement_suggestions"].append("增加逻辑连接词，提高思考连贯性")
        
        if dimensions["completeness"] < 0.6:
            assessment["improvement_suggestions"].append("补充问题分析、推理过程或结论验证")
        
        if dimensions["depth"] < 0.6:
            assessment["improvement_suggestions"].append("增加思考深度，进行多层次分析")
        
        if dimensions["accuracy"] < 0.7:
            assessment["improvement_suggestions"].append("确保密码学术语使用准确，加强专业性")
        
        if dimensions["clarity"] < 0.6:
            assessment["improvement_suggestions"].append("优化表达方式，提高清晰度")
    
    def convert_to_thinking_format(
        self, 
        instruction: str, 
        response: str,
        crypto_terms: Optional[List[str]] = None
    ) -> ThinkingExample:
        """
        将现有数据转换为thinking格式
        
        Args:
            instruction: 原始指令
            response: 原始回答
            crypto_terms: 密码学术语列表
            
        Returns:
            转换后的ThinkingExample
        """
        if crypto_terms is None:
            crypto_terms = []
        
        # 生成thinking过程
        thinking_process = self.generate_thinking_process(
            instruction, 
            thinking_type=ThinkingType.CRYPTOGRAPHIC if crypto_terms else ThinkingType.ANALYTICAL
        )
        
        # 创建ThinkingExample
        thinking_example = ThinkingExample(
            instruction=instruction,
            thinking_process=thinking_process,
            final_response=response,
            crypto_terms=crypto_terms,
            difficulty_level=DifficultyLevel.INTERMEDIATE
        )
        
        return thinking_example