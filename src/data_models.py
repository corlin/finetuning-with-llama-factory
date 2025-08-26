"""
核心数据模型和深度思考数据结构

本模块实现了训练数据模型、深度思考数据结构、密码学术语和中文指标等核心数据类。
支持thinking数据验证、序列化和反序列化功能。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
import re
from datetime import datetime


class DifficultyLevel(Enum):
    """难度级别枚举"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class CryptoCategory(Enum):
    """密码学术语分类"""
    SYMMETRIC_ENCRYPTION = "对称加密"
    ASYMMETRIC_ENCRYPTION = "非对称加密"
    HASH_FUNCTION = "哈希函数"
    DIGITAL_SIGNATURE = "数字签名"
    KEY_MANAGEMENT = "密钥管理"
    CRYPTOGRAPHIC_PROTOCOL = "密码协议"
    CRYPTANALYSIS = "密码分析"
    BLOCKCHAIN = "区块链"
    OTHER = "其他"


@dataclass
class ReasoningStep:
    """推理步骤数据结构"""
    step_number: int
    description: str
    input_data: str
    reasoning_process: str
    output_result: str
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "input_data": self.input_data,
            "reasoning_process": self.reasoning_process,
            "output_result": self.output_result,
            "confidence_score": self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class CryptoTerm:
    """密码学术语数据结构"""
    term: str
    definition: str
    category: CryptoCategory
    complexity: int  # 1-10复杂度评分
    aliases: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """数据验证"""
        if not self.term.strip():
            raise ValueError("术语名称不能为空")
        if not self.definition.strip():
            raise ValueError("术语定义不能为空")
        if not 1 <= self.complexity <= 10:
            raise ValueError("复杂度必须在1-10之间")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "term": self.term,
            "definition": self.definition,
            "category": self.category.value,
            "complexity": self.complexity,
            "aliases": self.aliases,
            "related_terms": self.related_terms,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoTerm':
        """从字典创建实例"""
        data_copy = data.copy()
        data_copy["category"] = CryptoCategory(data["category"])
        return cls(**data_copy)


@dataclass
class ChineseMetrics:
    """中文文本评估指标"""
    character_accuracy: float
    word_accuracy: float
    rouge_l_chinese: float
    bleu_chinese: float
    crypto_term_accuracy: float
    semantic_similarity: float = 0.0
    fluency_score: float = 0.0
    coherence_score: float = 0.0
    
    def __post_init__(self):
        """数据验证"""
        metrics = [
            self.character_accuracy, self.word_accuracy, self.rouge_l_chinese,
            self.bleu_chinese, self.crypto_term_accuracy, self.semantic_similarity,
            self.fluency_score, self.coherence_score
        ]
        for metric in metrics:
            if not 0.0 <= metric <= 1.0:
                raise ValueError("所有指标值必须在0.0-1.0之间")
    
    def overall_score(self) -> float:
        """计算综合评分"""
        weights = {
            'character_accuracy': 0.15,
            'word_accuracy': 0.15,
            'rouge_l_chinese': 0.20,
            'bleu_chinese': 0.20,
            'crypto_term_accuracy': 0.15,
            'semantic_similarity': 0.10,
            'fluency_score': 0.025,
            'coherence_score': 0.025
        }
        
        return (
            self.character_accuracy * weights['character_accuracy'] +
            self.word_accuracy * weights['word_accuracy'] +
            self.rouge_l_chinese * weights['rouge_l_chinese'] +
            self.bleu_chinese * weights['bleu_chinese'] +
            self.crypto_term_accuracy * weights['crypto_term_accuracy'] +
            self.semantic_similarity * weights['semantic_similarity'] +
            self.fluency_score * weights['fluency_score'] +
            self.coherence_score * weights['coherence_score']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "character_accuracy": self.character_accuracy,
            "word_accuracy": self.word_accuracy,
            "rouge_l_chinese": self.rouge_l_chinese,
            "bleu_chinese": self.bleu_chinese,
            "crypto_term_accuracy": self.crypto_term_accuracy,
            "semantic_similarity": self.semantic_similarity,
            "fluency_score": self.fluency_score,
            "coherence_score": self.coherence_score,
            "overall_score": self.overall_score()
        }


@dataclass
class ThinkingStructure:
    """深度思考数据结构"""
    raw_thinking: str
    parsed_steps: List[str]
    reasoning_chain: List[ReasoningStep]
    validation_result: bool
    thinking_depth: int = 0
    logical_consistency: float = 0.0
    completeness_score: float = 0.0
    
    def __post_init__(self):
        """数据验证和后处理"""
        if not self.raw_thinking.strip():
            raise ValueError("原始thinking内容不能为空")
        
        # 自动计算thinking深度
        self.thinking_depth = len(self.reasoning_chain)
        
        # 验证逻辑一致性
        self._validate_logical_consistency()
    
    def _validate_logical_consistency(self):
        """验证逻辑一致性"""
        if not self.reasoning_chain:
            self.logical_consistency = 0.0
            return
        
        # 检查推理步骤的连贯性
        consistency_scores = []
        for i in range(len(self.reasoning_chain) - 1):
            current_step = self.reasoning_chain[i]
            next_step = self.reasoning_chain[i + 1]
            
            # 简单的一致性检查：下一步的输入应该与当前步的输出相关
            if current_step.output_result and next_step.input_data:
                # 这里可以实现更复杂的语义相似度检查
                consistency_scores.append(0.8)  # 简化实现
            else:
                consistency_scores.append(0.5)
        
        self.logical_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def validate_thinking_format(self) -> bool:
        """验证thinking格式的正确性"""
        # 检查是否包含thinking标签
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        matches = re.findall(thinking_pattern, self.raw_thinking, re.DOTALL)
        
        if not matches:
            return False
        
        # 检查嵌套标签的平衡性
        open_tags = self.raw_thinking.count('<thinking>')
        close_tags = self.raw_thinking.count('</thinking>')
        
        return open_tags == close_tags and open_tags > 0
    
    def extract_thinking_content(self) -> List[str]:
        """提取thinking标签内的内容"""
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        matches = re.findall(thinking_pattern, self.raw_thinking, re.DOTALL)
        return [match.strip() for match in matches]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "raw_thinking": self.raw_thinking,
            "parsed_steps": self.parsed_steps,
            "reasoning_chain": [step.to_dict() for step in self.reasoning_chain],
            "validation_result": self.validation_result,
            "thinking_depth": self.thinking_depth,
            "logical_consistency": self.logical_consistency,
            "completeness_score": self.completeness_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThinkingStructure':
        """从字典创建实例"""
        data_copy = data.copy()
        data_copy["reasoning_chain"] = [
            ReasoningStep.from_dict(step) for step in data["reasoning_chain"]
        ]
        return cls(**data_copy)


@dataclass
class ThinkingExample:
    """深度思考训练样例"""
    instruction: str
    thinking_process: str  # <thinking>标签内的内容
    final_response: str
    crypto_terms: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    source_domain: str = "密码学"
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据验证"""
        if not self.instruction.strip():
            raise ValueError("指令不能为空")
        if not self.thinking_process.strip():
            raise ValueError("思考过程不能为空")
        if not self.final_response.strip():
            raise ValueError("最终回答不能为空")
    
    def validate_thinking_tags(self) -> bool:
        """验证thinking标签格式"""
        return '<thinking>' in self.thinking_process and '</thinking>' in self.thinking_process
    
    def extract_reasoning_steps(self) -> List[str]:
        """从thinking过程中提取推理步骤"""
        if not self.reasoning_steps:
            # 简单的步骤提取：按行分割并过滤空行
            steps = [line.strip() for line in self.thinking_process.split('\n') if line.strip()]
            self.reasoning_steps = steps
        return self.reasoning_steps
    

    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "instruction": self.instruction,
            "thinking_process": self.thinking_process,
            "final_response": self.final_response,
            "crypto_terms": self.crypto_terms,
            "reasoning_steps": self.reasoning_steps,
            "difficulty_level": self.difficulty_level.value,
            "source_domain": self.source_domain,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThinkingExample':
        """从字典创建实例"""
        data_copy = data.copy()
        data_copy["difficulty_level"] = DifficultyLevel(data["difficulty_level"])
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data_copy)


@dataclass
class TrainingExample:
    """训练数据样例"""
    instruction: str
    input: str
    output: str
    thinking: Optional[str] = None  # 深度思考过程
    crypto_terms: List[str] = field(default_factory=list)
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据验证"""
        if not self.instruction.strip():
            raise ValueError("指令不能为空")
        if not self.output.strip():
            raise ValueError("输出不能为空")
    
    def has_thinking(self) -> bool:
        """检查是否包含thinking数据"""
        return self.thinking is not None and self.thinking.strip() != ""
    
    def validate_format(self) -> bool:
        """验证数据格式"""
        # 基本字段验证
        if not all([self.instruction, self.output]):
            return False
        
        # 如果有thinking数据，验证格式
        if self.has_thinking():
            return '<thinking>' in self.thinking and '</thinking>' in self.thinking
        
        return True
    
    def to_thinking_example(self) -> Optional[ThinkingExample]:
        """转换为ThinkingExample"""
        if not self.has_thinking():
            return None
        
        return ThinkingExample(
            instruction=self.instruction,
            thinking_process=self.thinking,
            final_response=self.output,
            crypto_terms=self.crypto_terms,
            difficulty_level=self.difficulty_level
        )
    

    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "thinking": self.thinking,
            "crypto_terms": self.crypto_terms,
            "difficulty_level": self.difficulty_level.value,
            "source_file": self.source_file,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        """从字典创建实例"""
        data_copy = data.copy()
        data_copy["difficulty_level"] = DifficultyLevel(data["difficulty_level"])
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data_copy)


# 数据验证和序列化工具函数
class DataModelValidator:
    """数据模型验证器"""
    
    @staticmethod
    def validate_thinking_data(thinking_text: str) -> Dict[str, Any]:
        """验证thinking数据格式"""
        result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        if not thinking_text.strip():
            result["errors"].append("thinking内容不能为空")
            return result
        
        # 检查thinking标签
        if '<thinking>' not in thinking_text:
            result["errors"].append("缺少<thinking>开始标签")
        
        if '</thinking>' not in thinking_text:
            result["errors"].append("缺少</thinking>结束标签")
        
        # 检查标签平衡
        open_count = thinking_text.count('<thinking>')
        close_count = thinking_text.count('</thinking>')
        
        if open_count != close_count:
            result["errors"].append(f"thinking标签不平衡：{open_count}个开始标签，{close_count}个结束标签")
        
        # 检查嵌套深度
        max_depth = 0
        current_depth = 0
        
        for match in re.finditer(r'</?thinking>', thinking_text):
            if match.group().startswith('</'):
                current_depth -= 1
            else:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
        
        if max_depth > 3:
            result["warnings"].append(f"thinking嵌套深度较深：{max_depth}层")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    @staticmethod
    def serialize_training_examples(examples: List[TrainingExample]) -> str:
        """序列化训练样例列表"""
        data = [example.to_dict() for example in examples]
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize_training_examples(json_str: str) -> List[TrainingExample]:
        """反序列化训练样例列表"""
        data = json.loads(json_str)
        return [TrainingExample.from_dict(item) for item in data]


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    model_type: str
    model_path: str
    quantization_format: str
    parameters: str
    language: str
    domain: str
    version: str = "1.0.0"
    loaded_at: Optional[datetime] = None
    file_size_mb: Optional[float] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """数据验证"""
        if not self.model_name.strip():
            raise ValueError("模型名称不能为空")
        if not self.model_type.strip():
            raise ValueError("模型类型不能为空")
        if not self.model_path.strip():
            raise ValueError("模型路径不能为空")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "quantization_format": self.quantization_format,
            "parameters": self.parameters,
            "language": self.language,
            "domain": self.domain,
            "version": self.version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "file_size_mb": self.file_size_mb,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建实例"""
        data_copy = data.copy()
        if data_copy.get("loaded_at"):
            data_copy["loaded_at"] = datetime.fromisoformat(data["loaded_at"])
        return cls(**data_copy)