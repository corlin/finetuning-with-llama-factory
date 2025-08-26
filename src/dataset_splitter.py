"""
智能数据集分割模块

本模块实现了智能数据集分割功能，支持：
- 基础训练/验证/测试集分割
- 自定义分割比例配置
- 数据分布均衡检查
- 专业术语分布优化
- thinking数据完整性保护
- 语义完整性保护
- 分割质量评估和验证
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
import random
import json
import logging
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime

from src.data_models import (
    TrainingExample, ThinkingExample, CryptoTerm, 
    DifficultyLevel, CryptoCategory, ChineseMetrics
)


class SplitStrategy(Enum):
    """数据分割策略"""
    RANDOM = "random"  # 随机分割
    STRATIFIED = "stratified"  # 分层分割
    BALANCED = "balanced"  # 均衡分割
    SEMANTIC = "semantic"  # 语义分割


@dataclass
class SplitConfig:
    """数据分割配置"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    strategy: SplitStrategy = SplitStrategy.BALANCED
    random_seed: int = 42
    preserve_thinking_integrity: bool = True
    balance_crypto_terms: bool = True
    balance_difficulty_levels: bool = True
    min_examples_per_split: int = 1
    
    def __post_init__(self):
        """验证配置参数"""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"分割比例总和必须为1.0，当前为{total_ratio}")
        
        if any(ratio <= 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("所有分割比例必须大于0")


@dataclass
class DatasetSplits:
    """数据集分割结果"""
    train_examples: List[TrainingExample]
    val_examples: List[TrainingExample]
    test_examples: List[TrainingExample]
    split_config: SplitConfig
    split_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_split_sizes(self) -> Dict[str, int]:
        """获取各分割的大小"""
        return {
            "train": len(self.train_examples),
            "val": len(self.val_examples),
            "test": len(self.test_examples),
            "total": len(self.train_examples) + len(self.val_examples) + len(self.test_examples)
        }
    
    def get_split_ratios(self) -> Dict[str, float]:
        """获取实际分割比例"""
        sizes = self.get_split_sizes()
        total = sizes["total"]
        if total == 0:
            return {"train": 0.0, "val": 0.0, "test": 0.0}
        
        return {
            "train": sizes["train"] / total,
            "val": sizes["val"] / total,
            "test": sizes["test"] / total
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "train_examples": [ex.to_dict() for ex in self.train_examples],
            "val_examples": [ex.to_dict() for ex in self.val_examples],
            "test_examples": [ex.to_dict() for ex in self.test_examples],
            "split_config": {
                "train_ratio": self.split_config.train_ratio,
                "val_ratio": self.split_config.val_ratio,
                "test_ratio": self.split_config.test_ratio,
                "strategy": self.split_config.strategy.value,
                "random_seed": self.split_config.random_seed,
                "preserve_thinking_integrity": self.split_config.preserve_thinking_integrity,
                "balance_crypto_terms": self.split_config.balance_crypto_terms,
                "balance_difficulty_levels": self.split_config.balance_difficulty_levels,
                "min_examples_per_split": self.split_config.min_examples_per_split
            },
            "split_metadata": self.split_metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class SplitQualityReport:
    """分割质量评估报告"""
    overall_score: float
    distribution_balance_score: float
    crypto_term_balance_score: float
    difficulty_balance_score: float
    thinking_integrity_score: float
    semantic_integrity_score: float
    overfitting_risk_score: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """判断分割质量是否达标"""
        return self.overall_score >= threshold
    
    def get_risk_level(self) -> str:
        """获取风险等级"""
        if self.overfitting_risk_score >= 0.8:
            return "高风险"
        elif self.overfitting_risk_score >= 0.5:
            return "中等风险"
        else:
            return "低风险"


class DatasetSplitter:
    """智能数据集分割器"""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """初始化分割器"""
        self.config = config or SplitConfig()
        self.logger = logging.getLogger(__name__)
        self._random_state = random.Random(self.config.random_seed)
        
        # 设置随机种子
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def split_dataset(
        self, 
        examples: List[TrainingExample],
        custom_ratios: Optional[Tuple[float, float, float]] = None
    ) -> DatasetSplits:
        """
        分割数据集
        
        Args:
            examples: 训练样例列表
            custom_ratios: 自定义分割比例 (train, val, test)
        
        Returns:
            DatasetSplits: 分割结果
        """
        if not examples:
            raise ValueError("数据集不能为空")
        
        # 更新配置
        if custom_ratios:
            self.config.train_ratio, self.config.val_ratio, self.config.test_ratio = custom_ratios
            # 重新验证配置
            self.config.__post_init__()
        
        self.logger.info(f"开始分割数据集，共{len(examples)}个样例")
        self.logger.info(f"分割比例：训练{self.config.train_ratio:.2f}，验证{self.config.val_ratio:.2f}，测试{self.config.test_ratio:.2f}")
        
        # 根据策略选择分割方法
        if self.config.strategy == SplitStrategy.RANDOM:
            splits = self._random_split(examples)
        elif self.config.strategy == SplitStrategy.STRATIFIED:
            splits = self._stratified_split(examples)
        elif self.config.strategy == SplitStrategy.BALANCED:
            splits = self._balanced_split(examples)
        elif self.config.strategy == SplitStrategy.SEMANTIC:
            splits = self._semantic_split(examples)
        else:
            raise ValueError(f"不支持的分割策略：{self.config.strategy}")
        
        # 验证分割结果
        self._validate_splits(splits)
        
        # 添加元数据
        splits.split_metadata = self._generate_split_metadata(splits)
        
        self.logger.info(f"数据集分割完成：训练{len(splits.train_examples)}，验证{len(splits.val_examples)}，测试{len(splits.test_examples)}")
        
        return splits
    
    def _random_split(self, examples: List[TrainingExample]) -> DatasetSplits:
        """随机分割"""
        shuffled_examples = examples.copy()
        self._random_state.shuffle(shuffled_examples)
        
        total_size = len(shuffled_examples)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        
        # 确保每个分割至少有最小数量的样例
        if total_size >= 3:  # 至少需要3个样例才能分割
            # 调整分割大小以满足最小要求
            min_size = self.config.min_examples_per_split
            if train_size < min_size:
                train_size = min_size
            if val_size < min_size:
                val_size = min_size
            
            # 确保总数不超过可用样例数
            if train_size + val_size >= total_size:
                # 重新分配：优先保证训练集，然后验证集，最后测试集
                train_size = max(min_size, total_size - 2 * min_size)
                val_size = min_size
                test_size = total_size - train_size - val_size
                if test_size < min_size:
                    val_size = total_size - train_size - min_size
                    test_size = min_size
        
        train_examples = shuffled_examples[:train_size]
        val_examples = shuffled_examples[train_size:train_size + val_size]
        test_examples = shuffled_examples[train_size + val_size:]
        
        return DatasetSplits(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            split_config=self.config
        )
    
    def _stratified_split(self, examples: List[TrainingExample]) -> DatasetSplits:
        """分层分割（按难度级别分层）"""
        # 按难度级别分组
        difficulty_groups = defaultdict(list)
        for example in examples:
            difficulty_groups[example.difficulty_level].append(example)
        
        train_examples = []
        val_examples = []
        test_examples = []
        
        # 对每个难度级别进行分割
        for difficulty, group_examples in difficulty_groups.items():
            if len(group_examples) < 3:  # 如果样例太少，全部放入训练集
                train_examples.extend(group_examples)
                continue
            
            self._random_state.shuffle(group_examples)
            
            group_size = len(group_examples)
            train_size = max(1, int(group_size * self.config.train_ratio))
            val_size = max(1, int(group_size * self.config.val_ratio))
            
            train_examples.extend(group_examples[:train_size])
            val_examples.extend(group_examples[train_size:train_size + val_size])
            test_examples.extend(group_examples[train_size + val_size:])
        
        # 如果验证集或测试集为空，从训练集中移动一些样例
        if len(val_examples) == 0 and len(train_examples) > 1:
            val_examples.append(train_examples.pop())
        
        if len(test_examples) == 0 and len(train_examples) > 1:
            test_examples.append(train_examples.pop())
        
        return DatasetSplits(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            split_config=self.config
        )
    
    def _balanced_split(self, examples: List[TrainingExample]) -> DatasetSplits:
        """均衡分割（考虑多个因素的均衡）"""
        # 首先按难度级别分层
        splits = self._stratified_split(examples)
        
        # 如果启用了密码学术语均衡，进行调整
        if self.config.balance_crypto_terms:
            splits = self._balance_crypto_terms(splits)
        
        # 如果启用了thinking数据完整性保护，进行调整
        if self.config.preserve_thinking_integrity:
            splits = self._preserve_thinking_integrity(splits)
        
        # 保护语义完整性
        splits = self._protect_semantic_integrity(splits)
        
        return splits
    
    def _semantic_split(self, examples: List[TrainingExample]) -> DatasetSplits:
        """语义分割（保持语义相关性）"""
        # 简化实现：按指令相似度分组
        instruction_groups = self._group_by_instruction_similarity(examples)
        
        train_examples = []
        val_examples = []
        test_examples = []
        
        # 对每个语义组进行分割
        for group in instruction_groups:
            if len(group) < 3:
                train_examples.extend(group)
                continue
            
            self._random_state.shuffle(group)
            
            group_size = len(group)
            train_size = max(1, int(group_size * self.config.train_ratio))
            val_size = max(1, int(group_size * self.config.val_ratio))
            
            train_examples.extend(group[:train_size])
            val_examples.extend(group[train_size:train_size + val_size])
            test_examples.extend(group[train_size + val_size:])
        
        # 如果某些分割为空，重新分配
        total_examples = train_examples + val_examples + test_examples
        if len(val_examples) == 0 or len(test_examples) == 0:
            # 回退到简单随机分割
            return self._random_split(total_examples)
        
        return DatasetSplits(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            split_config=self.config
        )
    
    def _group_by_instruction_similarity(self, examples: List[TrainingExample]) -> List[List[TrainingExample]]:
        """按指令相似度分组（简化实现）"""
        # 简化实现：按指令长度和关键词分组
        groups = defaultdict(list)
        
        for example in examples:
            # 使用指令的前20个字符作为分组键
            key = example.instruction[:20].strip()
            groups[key].append(example)
        
        return list(groups.values())
    
    def _balance_crypto_terms(self, splits: DatasetSplits) -> DatasetSplits:
        """均衡密码学术语分布"""
        # 统计各分割中的术语分布
        train_term_counts = self._count_crypto_terms(splits.train_examples)
        val_term_counts = self._count_crypto_terms(splits.val_examples)
        test_term_counts = self._count_crypto_terms(splits.test_examples)
        
        # 获取所有术语
        all_terms = set()
        all_terms.update(train_term_counts.keys())
        all_terms.update(val_term_counts.keys())
        all_terms.update(test_term_counts.keys())
        
        if not all_terms:
            return splits
        
        # 计算理想分布
        total_examples = len(splits.train_examples) + len(splits.val_examples) + len(splits.test_examples)
        ideal_train_ratio = len(splits.train_examples) / total_examples
        ideal_val_ratio = len(splits.val_examples) / total_examples
        ideal_test_ratio = len(splits.test_examples) / total_examples
        
        # 识别分布不均衡的术语
        imbalanced_terms = []
        for term in all_terms:
            train_count = train_term_counts.get(term, 0)
            val_count = val_term_counts.get(term, 0)
            test_count = test_term_counts.get(term, 0)
            total_term_count = train_count + val_count + test_count
            
            if total_term_count == 0:
                continue
            
            # 计算实际比例
            actual_train_ratio = train_count / total_term_count
            actual_val_ratio = val_count / total_term_count
            actual_test_ratio = test_count / total_term_count
            
            # 检查是否偏离理想比例太多
            train_deviation = abs(actual_train_ratio - ideal_train_ratio)
            val_deviation = abs(actual_val_ratio - ideal_val_ratio)
            test_deviation = abs(actual_test_ratio - ideal_test_ratio)
            
            if max(train_deviation, val_deviation, test_deviation) > 0.3:  # 30%的偏差阈值
                imbalanced_terms.append({
                    'term': term,
                    'train_count': train_count,
                    'val_count': val_count,
                    'test_count': test_count,
                    'total_count': total_term_count,
                    'max_deviation': max(train_deviation, val_deviation, test_deviation)
                })
        
        # 对于严重不均衡的术语，尝试重新分配样例
        if imbalanced_terms and len(imbalanced_terms) <= 5:  # 只处理少量不均衡术语
            splits = self._rebalance_examples_by_terms(splits, imbalanced_terms)
        
        return splits
    
    def _count_crypto_terms(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """统计密码学术语出现次数"""
        term_counts = defaultdict(int)
        for example in examples:
            for term in example.crypto_terms:
                term_counts[term] += 1
        return dict(term_counts)
    
    def _rebalance_examples_by_terms(self, splits: DatasetSplits, imbalanced_terms: List[Dict]) -> DatasetSplits:
        """根据术语重新平衡样例分布"""
        # 简化实现：对于最不均衡的术语，尝试移动一些样例
        if not imbalanced_terms:
            return splits
        
        # 按偏差程度排序，处理最严重的不均衡
        imbalanced_terms.sort(key=lambda x: x['max_deviation'], reverse=True)
        most_imbalanced = imbalanced_terms[0]
        target_term = most_imbalanced['term']
        
        # 找到包含该术语的样例
        all_examples = splits.train_examples + splits.val_examples + splits.test_examples
        term_examples = [ex for ex in all_examples if target_term in ex.crypto_terms]
        
        if len(term_examples) < 3:  # 样例太少，无法重新分配
            return splits
        
        # 重新分配包含该术语的样例
        self._random_state.shuffle(term_examples)
        
        total_term_examples = len(term_examples)
        new_train_count = int(total_term_examples * self.config.train_ratio)
        new_val_count = int(total_term_examples * self.config.val_ratio)
        
        new_train_term_examples = term_examples[:new_train_count]
        new_val_term_examples = term_examples[new_train_count:new_train_count + new_val_count]
        new_test_term_examples = term_examples[new_train_count + new_val_count:]
        
        # 从原分割中移除这些样例
        train_examples = [ex for ex in splits.train_examples if target_term not in ex.crypto_terms]
        val_examples = [ex for ex in splits.val_examples if target_term not in ex.crypto_terms]
        test_examples = [ex for ex in splits.test_examples if target_term not in ex.crypto_terms]
        
        # 添加重新分配的样例
        train_examples.extend(new_train_term_examples)
        val_examples.extend(new_val_term_examples)
        test_examples.extend(new_test_term_examples)
        
        return DatasetSplits(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            split_config=splits.split_config,
            split_metadata=splits.split_metadata,
            created_at=splits.created_at
        )
    
    def _preserve_thinking_integrity(self, splits: DatasetSplits) -> DatasetSplits:
        """保护thinking数据完整性"""
        # 检查thinking数据是否被正确保持
        integrity_issues = []
        
        for split_name, examples in [
            ("train", splits.train_examples),
            ("val", splits.val_examples),
            ("test", splits.test_examples)
        ]:
            for i, example in enumerate(examples):
                if example.has_thinking():
                    # 验证thinking标签的完整性
                    if not self._validate_thinking_integrity(example.thinking):
                        integrity_issues.append({
                            'split': split_name,
                            'index': i,
                            'example': example,
                            'issue': 'thinking_tag_mismatch'
                        })
                    
                    # 检查thinking内容的逻辑完整性
                    if not self._validate_thinking_logical_completeness(example.thinking):
                        integrity_issues.append({
                            'split': split_name,
                            'index': i,
                            'example': example,
                            'issue': 'thinking_logic_incomplete'
                        })
        
        # 修复发现的问题
        if integrity_issues:
            splits = self._fix_thinking_integrity_issues(splits, integrity_issues)
        
        # 确保thinking数据在各分割中的分布合理
        splits = self._balance_thinking_distribution(splits)
        
        return splits
    
    def _validate_thinking_logical_completeness(self, thinking_text: str) -> bool:
        """验证thinking逻辑完整性"""
        if not thinking_text:
            return True
        
        # 提取thinking内容
        thinking_content = self._extract_thinking_content(thinking_text)
        if not thinking_content:
            return False
        
        # 检查是否包含基本的推理元素
        content = thinking_content[0].lower()
        
        # 检查是否包含问题分析
        has_analysis = any(keyword in content for keyword in [
            '分析', '考虑', '思考', '问题', '需要', '要求'
        ])
        
        # 检查是否包含推理过程
        has_reasoning = any(keyword in content for keyword in [
            '因为', '所以', '由于', '因此', '首先', '然后', '最后', '步骤'
        ])
        
        # 检查是否包含结论
        has_conclusion = any(keyword in content for keyword in [
            '总结', '结论', '答案', '结果', '综上', '因此'
        ])
        
        # 至少需要包含分析和推理
        return has_analysis and (has_reasoning or has_conclusion)
    
    def _extract_thinking_content(self, thinking_text: str) -> List[str]:
        """提取thinking标签内的内容"""
        import re
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        matches = re.findall(thinking_pattern, thinking_text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def _fix_thinking_integrity_issues(self, splits: DatasetSplits, issues: List[Dict]) -> DatasetSplits:
        """修复thinking完整性问题"""
        # 对于标签不匹配的问题，尝试修复
        for issue in issues:
            if issue['issue'] == 'thinking_tag_mismatch':
                example = issue['example']
                if example.thinking:
                    # 尝试修复不平衡的标签
                    fixed_thinking = self._fix_thinking_tags(example.thinking)
                    example.thinking = fixed_thinking
        
        return splits
    
    def _fix_thinking_tags(self, thinking_text: str) -> str:
        """修复thinking标签"""
        if not thinking_text:
            return thinking_text
        
        # 统计开始和结束标签
        open_count = thinking_text.count('<thinking>')
        close_count = thinking_text.count('</thinking>')
        
        # 如果缺少结束标签，添加
        if open_count > close_count:
            missing_close = open_count - close_count
            thinking_text += '</thinking>' * missing_close
        
        # 如果缺少开始标签，添加
        elif close_count > open_count:
            missing_open = close_count - open_count
            thinking_text = '<thinking>' * missing_open + thinking_text
        
        return thinking_text
    
    def _balance_thinking_distribution(self, splits: DatasetSplits) -> DatasetSplits:
        """平衡thinking数据分布"""
        # 统计各分割中thinking数据的比例
        train_thinking_ratio = self._calculate_thinking_ratio(splits.train_examples)
        val_thinking_ratio = self._calculate_thinking_ratio(splits.val_examples)
        test_thinking_ratio = self._calculate_thinking_ratio(splits.test_examples)
        
        # 计算总体thinking比例
        all_examples = splits.train_examples + splits.val_examples + splits.test_examples
        overall_thinking_ratio = self._calculate_thinking_ratio(all_examples)
        
        # 如果某个分割的thinking比例偏离总体比例太多，进行调整
        max_deviation = 0.3  # 30%的偏差阈值
        
        adjustments_needed = []
        if abs(train_thinking_ratio - overall_thinking_ratio) > max_deviation:
            adjustments_needed.append('train')
        if abs(val_thinking_ratio - overall_thinking_ratio) > max_deviation:
            adjustments_needed.append('val')
        if abs(test_thinking_ratio - overall_thinking_ratio) > max_deviation:
            adjustments_needed.append('test')
        
        # 如果需要调整且样例数量足够，进行重新分配
        if adjustments_needed and len(all_examples) >= 10:
            splits = self._rebalance_thinking_examples(splits, overall_thinking_ratio)
        
        return splits
    
    def _calculate_thinking_ratio(self, examples: List[TrainingExample]) -> float:
        """计算thinking数据比例"""
        if not examples:
            return 0.0
        
        thinking_count = sum(1 for ex in examples if ex.has_thinking())
        return thinking_count / len(examples)
    
    def _rebalance_thinking_examples(self, splits: DatasetSplits, target_ratio: float) -> DatasetSplits:
        """重新平衡thinking样例分布"""
        # 分离thinking和非thinking样例
        all_examples = splits.train_examples + splits.val_examples + splits.test_examples
        thinking_examples = [ex for ex in all_examples if ex.has_thinking()]
        non_thinking_examples = [ex for ex in all_examples if not ex.has_thinking()]
        
        if not thinking_examples or not non_thinking_examples:
            return splits  # 无法平衡
        
        # 重新分配
        self._random_state.shuffle(thinking_examples)
        self._random_state.shuffle(non_thinking_examples)
        
        # 计算各分割需要的thinking样例数
        total_examples = len(all_examples)
        train_size = int(total_examples * self.config.train_ratio)
        val_size = int(total_examples * self.config.val_ratio)
        test_size = total_examples - train_size - val_size
        
        train_thinking_needed = int(train_size * target_ratio)
        val_thinking_needed = int(val_size * target_ratio)
        test_thinking_needed = int(test_size * target_ratio)
        
        # 确保不超过可用的thinking样例数
        total_thinking_needed = train_thinking_needed + val_thinking_needed + test_thinking_needed
        if total_thinking_needed > len(thinking_examples):
            # 按比例缩减
            scale = len(thinking_examples) / total_thinking_needed
            train_thinking_needed = int(train_thinking_needed * scale)
            val_thinking_needed = int(val_thinking_needed * scale)
            test_thinking_needed = len(thinking_examples) - train_thinking_needed - val_thinking_needed
        
        # 分配thinking样例
        train_thinking = thinking_examples[:train_thinking_needed]
        val_thinking = thinking_examples[train_thinking_needed:train_thinking_needed + val_thinking_needed]
        test_thinking = thinking_examples[train_thinking_needed + val_thinking_needed:train_thinking_needed + val_thinking_needed + test_thinking_needed]
        
        # 分配非thinking样例
        train_non_thinking_needed = train_size - len(train_thinking)
        val_non_thinking_needed = val_size - len(val_thinking)
        test_non_thinking_needed = test_size - len(test_thinking)
        
        train_non_thinking = non_thinking_examples[:train_non_thinking_needed]
        val_non_thinking = non_thinking_examples[train_non_thinking_needed:train_non_thinking_needed + val_non_thinking_needed]
        test_non_thinking = non_thinking_examples[train_non_thinking_needed + val_non_thinking_needed:train_non_thinking_needed + val_non_thinking_needed + test_non_thinking_needed]
        
        # 组合结果
        new_train_examples = train_thinking + train_non_thinking
        new_val_examples = val_thinking + val_non_thinking
        new_test_examples = test_thinking + test_non_thinking
        
        # 随机打乱
        self._random_state.shuffle(new_train_examples)
        self._random_state.shuffle(new_val_examples)
        self._random_state.shuffle(new_test_examples)
        
        return DatasetSplits(
            train_examples=new_train_examples,
            val_examples=new_val_examples,
            test_examples=new_test_examples,
            split_config=splits.split_config,
            split_metadata=splits.split_metadata,
            created_at=splits.created_at
        )
    
    def _validate_thinking_integrity(self, thinking_text: str) -> bool:
        """验证thinking数据完整性"""
        if not thinking_text:
            return True
        
        # 检查thinking标签平衡
        open_count = thinking_text.count('<thinking>')
        close_count = thinking_text.count('</thinking>')
        
        return open_count == close_count and open_count > 0
    
    def _extract_crypto_terms(self, examples: List[TrainingExample]) -> List[str]:
        """提取密码学术语"""
        terms = []
        for example in examples:
            terms.extend(example.crypto_terms)
        return terms
    
    def _validate_splits(self, splits: DatasetSplits) -> None:
        """验证分割结果"""
        sizes = splits.get_split_sizes()
        
        # 检查最小样例数
        if sizes["train"] < self.config.min_examples_per_split:
            raise ValueError(f"训练集样例数({sizes['train']})少于最小要求({self.config.min_examples_per_split})")
        
        if sizes["val"] < self.config.min_examples_per_split:
            raise ValueError(f"验证集样例数({sizes['val']})少于最小要求({self.config.min_examples_per_split})")
        
        if sizes["test"] < self.config.min_examples_per_split:
            raise ValueError(f"测试集样例数({sizes['test']})少于最小要求({self.config.min_examples_per_split})")
        
        # 检查总数是否匹配
        expected_total = sizes["train"] + sizes["val"] + sizes["test"]
        if sizes["total"] != expected_total:
            raise ValueError(f"分割后总数不匹配：期望{expected_total}，实际{sizes['total']}")
    
    def _generate_split_metadata(self, splits: DatasetSplits) -> Dict[str, Any]:
        """生成分割元数据"""
        sizes = splits.get_split_sizes()
        ratios = splits.get_split_ratios()
        
        # 统计各分割的特征
        metadata = {
            "sizes": sizes,
            "ratios": ratios,
            "difficulty_distribution": self._analyze_difficulty_distribution(splits),
            "crypto_term_distribution": self._analyze_crypto_term_distribution(splits),
            "thinking_data_stats": self._analyze_thinking_data(splits),
            "split_timestamp": datetime.now().isoformat()
        }
        
        return metadata
    
    def _analyze_difficulty_distribution(self, splits: DatasetSplits) -> Dict[str, Any]:
        """分析难度分布"""
        def count_difficulties(examples: List[TrainingExample]) -> Dict[str, int]:
            counter = Counter(ex.difficulty_level.name for ex in examples)
            return dict(counter)
        
        return {
            "train": count_difficulties(splits.train_examples),
            "val": count_difficulties(splits.val_examples),
            "test": count_difficulties(splits.test_examples)
        }
    
    def _analyze_crypto_term_distribution(self, splits: DatasetSplits) -> Dict[str, Any]:
        """分析密码学术语分布"""
        def count_terms(examples: List[TrainingExample]) -> Dict[str, int]:
            all_terms = []
            for ex in examples:
                all_terms.extend(ex.crypto_terms)
            return dict(Counter(all_terms))
        
        return {
            "train": count_terms(splits.train_examples),
            "val": count_terms(splits.val_examples),
            "test": count_terms(splits.test_examples)
        }
    
    def _analyze_thinking_data(self, splits: DatasetSplits) -> Dict[str, Any]:
        """分析thinking数据统计"""
        def count_thinking(examples: List[TrainingExample]) -> Dict[str, int]:
            thinking_count = sum(1 for ex in examples if ex.has_thinking())
            return {
                "total": len(examples),
                "with_thinking": thinking_count,
                "without_thinking": len(examples) - thinking_count,
                "thinking_ratio": thinking_count / len(examples) if examples else 0.0
            }
        
        return {
            "train": count_thinking(splits.train_examples),
            "val": count_thinking(splits.val_examples),
            "test": count_thinking(splits.test_examples)
        }
    
    def evaluate_split_quality(self, splits: DatasetSplits) -> SplitQualityReport:
        """评估分割质量"""
        # 计算各项质量指标
        distribution_score = self._evaluate_distribution_balance(splits)
        crypto_term_score = self._evaluate_crypto_term_balance(splits)
        difficulty_score = self._evaluate_difficulty_balance(splits)
        thinking_score = self._evaluate_thinking_integrity(splits)
        semantic_score = self._evaluate_semantic_integrity(splits)
        overfitting_score = self._evaluate_overfitting_risk(splits)
        
        # 计算综合评分
        weights = {
            'distribution': 0.2,
            'crypto_term': 0.2,
            'difficulty': 0.15,
            'thinking': 0.15,
            'semantic': 0.15,
            'overfitting': 0.15
        }
        
        overall_score = (
            distribution_score * weights['distribution'] +
            crypto_term_score * weights['crypto_term'] +
            difficulty_score * weights['difficulty'] +
            thinking_score * weights['thinking'] +
            semantic_score * weights['semantic'] +
            (1.0 - overfitting_score) * weights['overfitting']  # 过拟合风险越低越好
        )
        
        # 生成警告和建议
        warnings = []
        recommendations = []
        
        if distribution_score < 0.7:
            warnings.append("数据分布不够均衡")
            recommendations.append("考虑使用分层分割策略")
        
        if crypto_term_score < 0.7:
            warnings.append("密码学术语分布不均衡")
            recommendations.append("启用密码学术语均衡功能")
        
        if overfitting_score > 0.7:
            warnings.append("过拟合风险较高")
            recommendations.append("增加数据集大小或调整分割比例")
        
        return SplitQualityReport(
            overall_score=overall_score,
            distribution_balance_score=distribution_score,
            crypto_term_balance_score=crypto_term_score,
            difficulty_balance_score=difficulty_score,
            thinking_integrity_score=thinking_score,
            semantic_integrity_score=semantic_score,
            overfitting_risk_score=overfitting_score,
            warnings=warnings,
            recommendations=recommendations,
            detailed_metrics=splits.split_metadata
        )
    
    def _evaluate_distribution_balance(self, splits: DatasetSplits) -> float:
        """评估分布均衡性"""
        ratios = splits.get_split_ratios()
        expected_ratios = {
            "train": self.config.train_ratio,
            "val": self.config.val_ratio,
            "test": self.config.test_ratio
        }
        
        # 计算实际比例与期望比例的差异
        differences = [
            abs(ratios[key] - expected_ratios[key])
            for key in ["train", "val", "test"]
        ]
        
        # 转换为评分（差异越小评分越高）
        max_diff = max(differences)
        # 调整评分函数，使其更宽容一些
        if max_diff <= 0.05:  # 5%以内的偏差认为是很好的
            return 1.0
        elif max_diff <= 0.1:  # 10%以内的偏差认为是良好的
            return 0.8
        elif max_diff <= 0.2:  # 20%以内的偏差认为是可接受的
            return 0.6
        else:
            return max(0.0, 1.0 - max_diff * 2)  # 减少惩罚力度
    
    def _evaluate_crypto_term_balance(self, splits: DatasetSplits) -> float:
        """评估密码学术语均衡性"""
        train_terms = set(self._extract_crypto_terms(splits.train_examples))
        val_terms = set(self._extract_crypto_terms(splits.val_examples))
        test_terms = set(self._extract_crypto_terms(splits.test_examples))
        
        all_terms = train_terms | val_terms | test_terms
        
        if not all_terms:
            return 1.0  # 没有术语时认为是均衡的
        
        # 计算术语覆盖率
        train_coverage = len(train_terms) / len(all_terms)
        val_coverage = len(val_terms) / len(all_terms)
        test_coverage = len(test_terms) / len(all_terms)
        
        # 理想情况下，训练集应该覆盖大部分术语
        ideal_train_coverage = 0.8
        ideal_val_coverage = 0.5
        ideal_test_coverage = 0.5
        
        train_score = 1.0 - abs(train_coverage - ideal_train_coverage)
        val_score = 1.0 - abs(val_coverage - ideal_val_coverage)
        test_score = 1.0 - abs(test_coverage - ideal_test_coverage)
        
        return (train_score * 0.5 + val_score * 0.25 + test_score * 0.25)
    
    def _evaluate_difficulty_balance(self, splits: DatasetSplits) -> float:
        """评估难度均衡性"""
        # 简化实现：检查每个分割是否包含各种难度级别
        def has_all_difficulties(examples: List[TrainingExample]) -> float:
            difficulties = set(ex.difficulty_level for ex in examples)
            total_difficulties = len(DifficultyLevel)
            return len(difficulties) / total_difficulties
        
        train_score = has_all_difficulties(splits.train_examples)
        val_score = has_all_difficulties(splits.val_examples)
        test_score = has_all_difficulties(splits.test_examples)
        
        return (train_score * 0.5 + val_score * 0.25 + test_score * 0.25)
    
    def _evaluate_thinking_integrity(self, splits: DatasetSplits) -> float:
        """评估thinking数据完整性"""
        total_thinking_examples = 0
        valid_thinking_examples = 0
        
        for examples in [splits.train_examples, splits.val_examples, splits.test_examples]:
            for example in examples:
                if example.has_thinking():
                    total_thinking_examples += 1
                    if self._validate_thinking_integrity(example.thinking):
                        valid_thinking_examples += 1
        
        if total_thinking_examples == 0:
            return 1.0  # 没有thinking数据时认为是完整的
        
        return valid_thinking_examples / total_thinking_examples
    
    def _evaluate_semantic_integrity(self, splits: DatasetSplits) -> float:
        """评估语义完整性"""
        integrity_scores = []
        
        # 检查各分割的语义完整性
        for split_name, examples in [
            ("train", splits.train_examples),
            ("val", splits.val_examples),
            ("test", splits.test_examples)
        ]:
            if not examples:
                continue
            
            # 检查指令的完整性
            instruction_integrity = self._check_instruction_integrity(examples)
            
            # 检查输出的完整性
            output_integrity = self._check_output_integrity(examples)
            
            # 检查thinking数据的语义连贯性
            thinking_coherence = self._check_thinking_coherence(examples)
            
            split_score = (instruction_integrity + output_integrity + thinking_coherence) / 3
            integrity_scores.append(split_score)
        
        return sum(integrity_scores) / len(integrity_scores) if integrity_scores else 1.0
    
    def _check_instruction_integrity(self, examples: List[TrainingExample]) -> float:
        """检查指令完整性"""
        if not examples:
            return 1.0
        
        complete_instructions = 0
        for example in examples:
            instruction = example.instruction.strip()
            
            # 检查指令是否完整（不以标点符号意外结束）
            if instruction and not instruction.endswith(('...', '。。', '？？')):
                # 检查是否包含问号或冒号（表示完整的问题）
                if '？' in instruction or '：' in instruction or '?' in instruction or ':' in instruction:
                    complete_instructions += 1
                elif len(instruction) > 10:  # 足够长的指令认为是完整的
                    complete_instructions += 1
        
        return complete_instructions / len(examples)
    
    def _check_output_integrity(self, examples: List[TrainingExample]) -> float:
        """检查输出完整性"""
        if not examples:
            return 1.0
        
        complete_outputs = 0
        for example in examples:
            output = example.output.strip()
            
            # 检查输出是否完整
            if output and len(output) > 20:  # 足够长的输出
                # 检查是否以合适的标点结束
                if output.endswith(('。', '！', '？', '.', '!', '?')):
                    complete_outputs += 1
                elif not output.endswith(('...', '。。')):  # 不以省略号结束
                    complete_outputs += 1
        
        return complete_outputs / len(examples)
    
    def _check_thinking_coherence(self, examples: List[TrainingExample]) -> float:
        """检查thinking数据连贯性"""
        thinking_examples = [ex for ex in examples if ex.has_thinking()]
        
        if not thinking_examples:
            return 1.0  # 没有thinking数据时认为是连贯的
        
        coherent_thinking = 0
        for example in thinking_examples:
            if self._validate_thinking_logical_completeness(example.thinking):
                coherent_thinking += 1
        
        return coherent_thinking / len(thinking_examples)
    
    def _protect_semantic_integrity(self, splits: DatasetSplits) -> DatasetSplits:
        """保护语义完整性"""
        # 检查并修复语义完整性问题
        integrity_issues = []
        
        for split_name, examples in [
            ("train", splits.train_examples),
            ("val", splits.val_examples),
            ("test", splits.test_examples)
        ]:
            for i, example in enumerate(examples):
                issues = self._detect_semantic_issues(example)
                if issues:
                    integrity_issues.extend([{
                        'split': split_name,
                        'index': i,
                        'example': example,
                        'issues': issues
                    }])
        
        # 修复发现的语义问题
        if integrity_issues:
            splits = self._fix_semantic_issues(splits, integrity_issues)
        
        return splits
    
    def _detect_semantic_issues(self, example: TrainingExample) -> List[str]:
        """检测语义问题"""
        issues = []
        
        # 检查指令截断
        instruction = example.instruction.strip()
        if instruction.endswith(('...', '。。', '？？')):
            issues.append('instruction_truncated')
        
        # 检查输出截断
        output = example.output.strip()
        if output.endswith(('...', '。。')) and len(output) < 50:
            issues.append('output_truncated')
        
        # 检查指令和输出的语义一致性
        if not self._check_instruction_output_consistency(instruction, output):
            issues.append('instruction_output_mismatch')
        
        # 检查thinking和输出的一致性
        if example.has_thinking():
            if not self._check_thinking_output_consistency(example.thinking, output):
                issues.append('thinking_output_mismatch')
        
        return issues
    
    def _check_instruction_output_consistency(self, instruction: str, output: str) -> bool:
        """检查指令和输出的一致性"""
        # 简化实现：检查关键词匹配
        instruction_lower = instruction.lower()
        output_lower = output.lower()
        
        # 提取指令中的关键概念
        crypto_keywords = ['aes', 'rsa', 'sha', 'md5', 'des', 'ecdsa', 'hmac', '加密', '解密', '哈希', '签名']
        
        instruction_concepts = [kw for kw in crypto_keywords if kw in instruction_lower]
        output_concepts = [kw for kw in crypto_keywords if kw in output_lower]
        
        # 如果指令中有密码学概念，输出中也应该有相关概念
        if instruction_concepts:
            return len(set(instruction_concepts) & set(output_concepts)) > 0
        
        return True  # 没有明确概念时认为是一致的
    
    def _check_thinking_output_consistency(self, thinking: str, output: str) -> bool:
        """检查thinking和输出的一致性"""
        thinking_content = self._extract_thinking_content(thinking)
        if not thinking_content:
            return True
        
        thinking_text = thinking_content[0].lower()
        output_lower = output.lower()
        
        # 检查thinking中的结论是否与输出一致
        # 简化实现：检查关键词重叠
        thinking_words = set(thinking_text.split())
        output_words = set(output_lower.split())
        
        # 计算词汇重叠度
        common_words = thinking_words & output_words
        if len(thinking_words) > 0:
            overlap_ratio = len(common_words) / len(thinking_words)
            return overlap_ratio > 0.1  # 至少10%的词汇重叠
        
        return True
    
    def _fix_semantic_issues(self, splits: DatasetSplits, issues: List[Dict]) -> DatasetSplits:
        """修复语义问题"""
        # 对于严重的语义问题，可以选择移除或修复样例
        # 这里实现简化的修复策略
        
        examples_to_remove = []
        
        for issue_info in issues:
            example = issue_info['example']
            issues = issue_info['issues']
            
            # 对于截断问题，如果内容太短就移除
            if 'instruction_truncated' in issues and len(example.instruction) < 20:
                examples_to_remove.append(example)
            elif 'output_truncated' in issues and len(example.output) < 30:
                examples_to_remove.append(example)
        
        # 从分割中移除有问题的样例
        if examples_to_remove:
            new_train_examples = [ex for ex in splits.train_examples if ex not in examples_to_remove]
            new_val_examples = [ex for ex in splits.val_examples if ex not in examples_to_remove]
            new_test_examples = [ex for ex in splits.test_examples if ex not in examples_to_remove]
            
            # 确保移除后仍满足最小样例数要求
            if (len(new_train_examples) >= self.config.min_examples_per_split and
                len(new_val_examples) >= self.config.min_examples_per_split and
                len(new_test_examples) >= self.config.min_examples_per_split):
                
                splits = DatasetSplits(
                    train_examples=new_train_examples,
                    val_examples=new_val_examples,
                    test_examples=new_test_examples,
                    split_config=splits.split_config,
                    split_metadata=splits.split_metadata,
                    created_at=splits.created_at
                )
        
        return splits
    
    def _evaluate_overfitting_risk(self, splits: DatasetSplits) -> float:
        """评估过拟合风险"""
        sizes = splits.get_split_sizes()
        
        # 基于数据集大小评估风险
        total_size = sizes["total"]
        train_size = sizes["train"]
        
        # 数据集越小，过拟合风险越高
        if total_size < 100:
            size_risk = 0.9
        elif total_size < 500:
            size_risk = 0.6
        elif total_size < 1000:
            size_risk = 0.3
        else:
            size_risk = 0.1
        
        # 训练集比例过大也会增加过拟合风险
        train_ratio = train_size / total_size
        if train_ratio > 0.8:
            ratio_risk = 0.7
        elif train_ratio > 0.75:
            ratio_risk = 0.4
        else:
            ratio_risk = 0.1
        
        return max(size_risk, ratio_risk)
    
    def save_splits(self, splits: DatasetSplits, output_dir: str) -> None:
        """保存分割结果"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各个分割
        for split_name, examples in [
            ("train", splits.train_examples),
            ("val", splits.val_examples),
            ("test", splits.test_examples)
        ]:
            file_path = os.path.join(output_dir, f"{split_name}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                # Convert to direct training format
                data = []
                for ex in examples:
                    output = ex.output
                    if hasattr(ex, 'thinking') and ex.thinking:
                        output = f"<thinking>\n{ex.thinking}\n</thinking>\n\n{ex.output}"
                    
                    data.append({
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": output,
                        "system": "你是一个专业的密码学专家，请仔细思考后回答问题。"
                    })
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存分割元数据
        metadata_path = os.path.join(output_dir, "split_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(splits.to_dict(), f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"分割结果已保存到 {output_dir}")
    
    def validate_qwen_format_compatibility(self, splits: DatasetSplits) -> Dict[str, Any]:
        """验证Qwen3-4B-Thinking格式兼容性"""
        validation_result = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }
        
        all_examples = splits.train_examples + splits.val_examples + splits.test_examples
        
        # 检查基本格式要求
        for i, example in enumerate(all_examples):
            # 检查必需字段
            if not example.instruction.strip():
                validation_result["issues"].append(f"样例{i}: 指令为空")
                validation_result["compatible"] = False
            
            if not example.output.strip():
                validation_result["issues"].append(f"样例{i}: 输出为空")
                validation_result["compatible"] = False
            
            # 检查thinking格式
            if example.has_thinking():
                if not self._validate_thinking_integrity(example.thinking):
                    validation_result["issues"].append(f"样例{i}: thinking标签不平衡")
                    validation_result["compatible"] = False
                
                # 检查thinking内容质量
                if not self._validate_thinking_logical_completeness(example.thinking):
                    validation_result["warnings"].append(f"样例{i}: thinking逻辑不够完整")
            
            # 检查文本长度
            if len(example.instruction) > 2048:
                validation_result["warnings"].append(f"样例{i}: 指令过长({len(example.instruction)}字符)")
            
            if len(example.output) > 4096:
                validation_result["warnings"].append(f"样例{i}: 输出过长({len(example.output)}字符)")
        
        # 统计信息
        validation_result["statistics"] = {
            "total_examples": len(all_examples),
            "thinking_examples": len([ex for ex in all_examples if ex.has_thinking()]),
            "avg_instruction_length": sum(len(ex.instruction) for ex in all_examples) / len(all_examples),
            "avg_output_length": sum(len(ex.output) for ex in all_examples) / len(all_examples),
            "crypto_terms_count": len(set(term for ex in all_examples for term in ex.crypto_terms))
        }
        
        return validation_result
    
    def assess_overfitting_risk(self, splits: DatasetSplits) -> Dict[str, Any]:
        """评估过拟合风险"""
        risk_assessment = {
            "risk_level": "低风险",
            "risk_score": 0.0,
            "risk_factors": [],
            "recommendations": []
        }
        
        sizes = splits.get_split_sizes()
        total_size = sizes["total"]
        
        # 数据集大小风险
        if total_size < 50:
            risk_assessment["risk_factors"].append("数据集过小(< 50样例)")
            risk_assessment["risk_score"] += 0.4
        elif total_size < 200:
            risk_assessment["risk_factors"].append("数据集较小(< 200样例)")
            risk_assessment["risk_score"] += 0.2
        
        # 训练集比例风险
        train_ratio = sizes["train"] / total_size
        if train_ratio > 0.85:
            risk_assessment["risk_factors"].append(f"训练集比例过高({train_ratio:.2f})")
            risk_assessment["risk_score"] += 0.3
        elif train_ratio > 0.8:
            risk_assessment["risk_factors"].append(f"训练集比例较高({train_ratio:.2f})")
            risk_assessment["risk_score"] += 0.1
        
        # 验证集大小风险
        if sizes["val"] < 10:
            risk_assessment["risk_factors"].append(f"验证集过小({sizes['val']}样例)")
            risk_assessment["risk_score"] += 0.2
        
        # 测试集大小风险
        if sizes["test"] < 10:
            risk_assessment["risk_factors"].append(f"测试集过小({sizes['test']}样例)")
            risk_assessment["risk_score"] += 0.2
        
        # 数据多样性风险
        diversity_score = self._assess_data_diversity(splits)
        if diversity_score < 0.5:
            risk_assessment["risk_factors"].append("数据多样性不足")
            risk_assessment["risk_score"] += 0.3
        elif diversity_score < 0.7:
            risk_assessment["risk_factors"].append("数据多样性较低")
            risk_assessment["risk_score"] += 0.1
        
        # 确定风险等级
        if risk_assessment["risk_score"] >= 0.7:
            risk_assessment["risk_level"] = "高风险"
        elif risk_assessment["risk_score"] >= 0.4:
            risk_assessment["risk_level"] = "中等风险"
        else:
            risk_assessment["risk_level"] = "低风险"
        
        # 生成建议
        if total_size < 100:
            risk_assessment["recommendations"].append("增加数据集大小至少100个样例")
        
        if train_ratio > 0.8:
            risk_assessment["recommendations"].append("减少训练集比例，增加验证集和测试集")
        
        if diversity_score < 0.6:
            risk_assessment["recommendations"].append("增加数据多样性，包含更多不同类型的样例")
        
        if sizes["val"] < 10:
            risk_assessment["recommendations"].append("增加验证集大小至少10个样例")
        
        return risk_assessment
    
    def _assess_data_diversity(self, splits: DatasetSplits) -> float:
        """评估数据多样性"""
        all_examples = splits.train_examples + splits.val_examples + splits.test_examples
        
        if not all_examples:
            return 0.0
        
        # 计算多个维度的多样性
        diversity_scores = []
        
        # 难度级别多样性
        difficulty_types = len(set(ex.difficulty_level for ex in all_examples))
        max_difficulty_types = len(DifficultyLevel)
        difficulty_diversity = difficulty_types / max_difficulty_types
        diversity_scores.append(difficulty_diversity)
        
        # 密码学术语多样性
        all_terms = set(term for ex in all_examples for term in ex.crypto_terms)
        if len(all_terms) >= 5:
            term_diversity = 1.0
        elif len(all_terms) >= 3:
            term_diversity = 0.8
        elif len(all_terms) >= 2:
            term_diversity = 0.6
        else:
            term_diversity = 0.3
        diversity_scores.append(term_diversity)
        
        # 指令长度多样性
        instruction_lengths = [len(ex.instruction) for ex in all_examples]
        if instruction_lengths:
            length_std = np.std(instruction_lengths)
            length_mean = np.mean(instruction_lengths)
            length_cv = length_std / length_mean if length_mean > 0 else 0
            length_diversity = min(1.0, length_cv)  # 变异系数作为多样性指标
            diversity_scores.append(length_diversity)
        
        # thinking数据多样性
        thinking_ratio = len([ex for ex in all_examples if ex.has_thinking()]) / len(all_examples)
        # 50%左右的thinking比例认为是最佳的
        thinking_diversity = 1.0 - abs(thinking_ratio - 0.5) * 2
        diversity_scores.append(thinking_diversity)
        
        return sum(diversity_scores) / len(diversity_scores)
    
    def generate_comprehensive_report(self, splits: DatasetSplits) -> Dict[str, Any]:
        """生成综合分割报告"""
        quality_report = self.evaluate_split_quality(splits)
        format_validation = self.validate_qwen_format_compatibility(splits)
        overfitting_assessment = self.assess_overfitting_risk(splits)
        
        report = {
            "summary": {
                "total_examples": splits.get_split_sizes()["total"],
                "split_ratios": splits.get_split_ratios(),
                "overall_quality_score": quality_report.overall_score,
                "format_compatible": format_validation["compatible"],
                "overfitting_risk": overfitting_assessment["risk_level"]
            },
            "quality_assessment": {
                "overall_score": quality_report.overall_score,
                "distribution_balance": quality_report.distribution_balance_score,
                "crypto_term_balance": quality_report.crypto_term_balance_score,
                "difficulty_balance": quality_report.difficulty_balance_score,
                "thinking_integrity": quality_report.thinking_integrity_score,
                "semantic_integrity": quality_report.semantic_integrity_score,
                "is_high_quality": quality_report.is_high_quality()
            },
            "format_validation": format_validation,
            "overfitting_assessment": overfitting_assessment,
            "detailed_statistics": splits.split_metadata,
            "warnings": quality_report.warnings + format_validation["warnings"],
            "recommendations": quality_report.recommendations + overfitting_assessment["recommendations"],
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def save_comprehensive_report(self, splits: DatasetSplits, output_path: str) -> None:
        """保存综合报告"""
        report = self.generate_comprehensive_report(splits)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"综合报告已保存到 {output_path}")
    
    def validate_split_integrity(self, splits: DatasetSplits) -> bool:
        """验证分割完整性"""
        try:
            # 基本验证
            self._validate_splits(splits)
            
            # 格式验证
            format_validation = self.validate_qwen_format_compatibility(splits)
            if not format_validation["compatible"]:
                self.logger.error("格式兼容性验证失败")
                return False
            
            # 质量验证
            quality_report = self.evaluate_split_quality(splits)
            if not quality_report.is_high_quality(threshold=0.5):  # 使用较低的阈值
                self.logger.warning(f"分割质量较低: {quality_report.overall_score:.2f}")
            
            # 过拟合风险验证
            overfitting_assessment = self.assess_overfitting_risk(splits)
            if overfitting_assessment["risk_level"] == "高风险":
                self.logger.warning("检测到高过拟合风险")
            
            return True
            
        except Exception as e:
            self.logger.error(f"分割完整性验证失败: {e}")
            return False
    
    def load_splits(self, input_dir: str) -> DatasetSplits:
        """加载分割结果"""
        import os
        
        metadata_path = os.path.join(input_dir, "split_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到分割元数据文件：{metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建配置
        config_data = data["split_config"]
        config = SplitConfig(
            train_ratio=config_data["train_ratio"],
            val_ratio=config_data["val_ratio"],
            test_ratio=config_data["test_ratio"],
            strategy=SplitStrategy(config_data["strategy"]),
            random_seed=config_data["random_seed"],
            preserve_thinking_integrity=config_data["preserve_thinking_integrity"],
            balance_crypto_terms=config_data["balance_crypto_terms"],
            balance_difficulty_levels=config_data["balance_difficulty_levels"],
            min_examples_per_split=config_data["min_examples_per_split"]
        )
        
        # 重建样例
        train_examples = [TrainingExample.from_dict(ex) for ex in data["train_examples"]]
        val_examples = [TrainingExample.from_dict(ex) for ex in data["val_examples"]]
        test_examples = [TrainingExample.from_dict(ex) for ex in data["test_examples"]]
        
        return DatasetSplits(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            split_config=config,
            split_metadata=data["split_metadata"],
            created_at=datetime.fromisoformat(data["created_at"])
        )