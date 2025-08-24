"""
中文NLP处理工具

本模块实现中文分词和词性标注功能、繁简体转换和标点符号规范化、
中文文本质量评估方法，以及Qwen tokenizer的中文处理优化。
"""

import re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import opencc
import unicodedata
from collections import Counter
import math

try:
    from data_models import ChineseMetrics
except ImportError:
    from src.data_models import ChineseMetrics


class TextVariant(Enum):
    """文本变体枚举"""
    SIMPLIFIED = "简体中文"
    TRADITIONAL = "繁体中文"
    MIXED = "混合"


class PunctuationStyle(Enum):
    """标点符号风格枚举"""
    CHINESE = "中文标点"
    ENGLISH = "英文标点"
    MIXED = "混合标点"


@dataclass
class ChineseToken:
    """中文分词结果"""
    word: str
    pos: str  # 词性标注
    start_pos: int
    end_pos: int
    is_crypto_term: bool = False
    confidence: float = 1.0
    
    def __post_init__(self):
        """数据验证"""
        if not self.word.strip():
            raise ValueError("词汇不能为空")
        if self.start_pos < 0 or self.end_pos <= self.start_pos:
            raise ValueError("位置信息无效")


@dataclass
class TextQualityMetrics:
    """文本质量指标"""
    readability_score: float  # 可读性评分
    fluency_score: float  # 流畅度评分
    coherence_score: float  # 连贯性评分
    complexity_score: float  # 复杂度评分
    punctuation_score: float  # 标点符号规范性评分
    character_diversity: float  # 字符多样性
    word_diversity: float  # 词汇多样性
    avg_sentence_length: float  # 平均句长
    
    def overall_quality(self) -> float:
        """计算综合质量评分"""
        weights = {
            'readability_score': 0.20,
            'fluency_score': 0.20,
            'coherence_score': 0.15,
            'complexity_score': 0.10,
            'punctuation_score': 0.10,
            'character_diversity': 0.10,
            'word_diversity': 0.10,
            'avg_sentence_length': 0.05
        }
        
        # 对avg_sentence_length进行归一化处理
        normalized_sentence_length = min(1.0, self.avg_sentence_length / 50.0)  # 假设50字符为理想长度
        
        score = (
            self.readability_score * weights['readability_score'] +
            self.fluency_score * weights['fluency_score'] +
            self.coherence_score * weights['coherence_score'] +
            self.complexity_score * weights['complexity_score'] +
            self.punctuation_score * weights['punctuation_score'] +
            self.character_diversity * weights['character_diversity'] +
            self.word_diversity * weights['word_diversity'] +
            normalized_sentence_length * weights['avg_sentence_length']
        )
        
        return min(1.0, max(0.0, score))


@dataclass
class TokenizerOptimization:
    """Tokenizer优化配置"""
    vocab_size: int
    special_tokens: List[str]
    chinese_vocab_ratio: float
    crypto_terms_count: int
    oov_rate: float  # Out-of-vocabulary rate
    compression_ratio: float
    
    def is_optimized(self) -> bool:
        """判断是否已优化"""
        return (
            self.chinese_vocab_ratio > 0.6 and
            self.oov_rate < 0.05 and
            self.compression_ratio > 2.0
        )


class ChineseNLPProcessor:
    """中文NLP处理器"""
    
    def __init__(self, custom_dict_path: Optional[str] = None):
        """
        初始化中文NLP处理器
        
        Args:
            custom_dict_path: 自定义词典路径
        """
        # 初始化jieba分词器
        jieba.initialize()
        
        # 加载自定义词典
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
        
        # 初始化繁简转换器
        self.s2t_converter = opencc.OpenCC('s2t')  # 简体转繁体
        self.t2s_converter = opencc.OpenCC('t2s')  # 繁体转简体
        
        # 中文标点符号映射
        self.punctuation_map = {
            # 英文标点 -> 中文标点
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            ':': '：',
            ';': '；',
            '(': '（',
            ')': '）',
            '[': '［',
            ']': '］',
            '{': '｛',
            '}': '｝',
            '"': '"',
            # 中文标点 -> 英文标点（反向映射）
            '，': ',',
            '。': '.',
            '？': '?',
            '！': '!',
            '：': ':',
            '；': ';',
            '（': '(',
            '）': ')',
            '［': '[',
            '］': ']',
            '｛': '{',
            '｝': '}',
            '"': '"'
        }
        
        # 密码学术语词典（扩展版）
        self.crypto_terms = {
            # 基础术语
            "对称加密", "非对称加密", "公钥", "私钥", "数字签名", "哈希函数",
            "RSA", "AES", "DES", "SHA", "MD5", "椭圆曲线", "密钥交换",
            "数字证书", "PKI", "SSL", "TLS", "HTTPS", "加密算法", "解密",
            "密码学", "密码分析", "密码协议", "认证", "完整性", "机密性",
            "不可否认性", "随机数", "伪随机数", "密钥管理", "密钥分发",
            
            # 对称加密扩展
            "分组密码", "流密码", "CBC", "ECB", "CTR", "GCM", "ChaCha20",
            "3DES", "Rijndael", "密码分组链接", "电子密码本", "计数器模式",
            "伽罗瓦计数器模式", "AEAD", "认证加密",
            
            # 非对称加密扩展
            "Diffie-Hellman", "ECDH", "DSA", "EdDSA", "Ed25519", "DH",
            "椭圆曲线密码学", "ECC", "离散对数", "大整数分解",
            
            # 哈希函数扩展
            "SHA-1", "SHA-256", "SHA-3", "BLAKE2", "HMAC", "Merkle树",
            "默克尔树", "哈希树", "Keccak", "海绵构造", "消息认证码",
            
            # 密钥管理扩展
            "CA", "CRL", "OCSP", "HSM", "密钥派生", "PBKDF2", "scrypt", "Argon2",
            "证书颁发机构", "证书撤销列表", "硬件安全模块", "密钥导出",
            "KDF", "盐值", "内存困难",
            
            # 密码协议扩展
            "握手协议", "IPSec", "SSH", "SAML", "OAuth", "JWT",
            "安全外壳", "JSON Web Token", "单点登录", "API授权",
            
            # 密码分析扩展
            "线性分析", "差分分析", "时间攻击", "功耗分析", "电磁分析",
            "碰撞攻击", "生日攻击", "中间人攻击", "重放攻击", "侧信道攻击",
            "旁路攻击", "计时攻击", "功耗攻击", "电磁攻击", "MITM攻击",
            
            # 区块链扩展
            "区块链", "比特币", "智能合约", "工作量证明", "权益证明",
            "哈希指针", "挖矿", "以太坊", "DeFi", "NFT", "PoW", "PoS",
            "共识算法", "分布式账本", "去中心化金融", "非同质化代币",
            
            # 高级术语
            "零知识证明", "同态加密", "多方安全计算", "秘密分享",
            "盲签名", "环签名", "群签名", "门限签名", "ZKP", "MPC",
            "隐私计算", "联邦学习", "门限方案"
        }
        
        # 将密码学术语添加到jieba词典
        for term in self.crypto_terms:
            jieba.add_word(term, freq=1000, tag='crypto')
    
    def segment_text(self, text: str, use_hmm: bool = True) -> List[ChineseToken]:
        """
        中文分词和词性标注
        
        Args:
            text: 输入文本
            use_hmm: 是否使用HMM模型
            
        Returns:
            分词结果列表
        """
        if not text.strip():
            return []
        
        # 使用jieba进行分词和词性标注
        words_with_pos = list(pseg.cut(text, HMM=use_hmm))
        
        tokens = []
        current_pos = 0
        
        for word, pos in words_with_pos:
            # 跳过空词汇
            if not word.strip():
                continue
                
            start_pos = text.find(word, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(word)
            
            # 检查是否为密码学术语
            is_crypto_term = word in self.crypto_terms or pos == 'crypto'
            
            token = ChineseToken(
                word=word,
                pos=pos,
                start_pos=start_pos,
                end_pos=end_pos,
                is_crypto_term=is_crypto_term
            )
            
            tokens.append(token)
            current_pos = end_pos
        
        return tokens
    
    def convert_traditional_simplified(
        self, 
        text: str, 
        target_variant: TextVariant
    ) -> str:
        """
        繁简体转换
        
        Args:
            text: 输入文本
            target_variant: 目标文本变体
            
        Returns:
            转换后的文本
        """
        if not text.strip():
            return text
        
        if target_variant == TextVariant.SIMPLIFIED:
            return self.t2s_converter.convert(text)
        elif target_variant == TextVariant.TRADITIONAL:
            return self.s2t_converter.convert(text)
        else:
            return text  # 保持原样
    
    def detect_text_variant(self, text: str) -> TextVariant:
        """
        检测文本变体（简体/繁体/混合）
        
        Args:
            text: 输入文本
            
        Returns:
            文本变体类型
        """
        if not text.strip():
            return TextVariant.SIMPLIFIED
        
        # 转换为简体和繁体
        simplified = self.t2s_converter.convert(text)
        traditional = self.s2t_converter.convert(text)
        
        # 计算相似度
        simplified_similarity = self._calculate_text_similarity(text, simplified)
        traditional_similarity = self._calculate_text_similarity(text, traditional)
        
        if simplified_similarity > 0.95:
            return TextVariant.SIMPLIFIED
        elif traditional_similarity > 0.95:
            return TextVariant.TRADITIONAL
        else:
            return TextVariant.MIXED
    
    def normalize_punctuation(
        self, 
        text: str, 
        target_style: PunctuationStyle = PunctuationStyle.CHINESE
    ) -> str:
        """
        标点符号规范化
        
        Args:
            text: 输入文本
            target_style: 目标标点符号风格
            
        Returns:
            规范化后的文本
        """
        if not text.strip():
            return text
        
        normalized_text = text
        
        if target_style == PunctuationStyle.CHINESE:
            # 转换为中文标点
            for eng_punct, chi_punct in self.punctuation_map.items():
                if eng_punct in [',', '.', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']:
                    normalized_text = normalized_text.replace(eng_punct, chi_punct)
        
        elif target_style == PunctuationStyle.ENGLISH:
            # 转换为英文标点
            for chi_punct, eng_punct in self.punctuation_map.items():
                if chi_punct in ['，', '。', '？', '！', '：', '；', '（', '）', '［', '］', '｛', '｝']:
                    normalized_text = normalized_text.replace(chi_punct, eng_punct)
        
        # 处理引号
        normalized_text = self._normalize_quotes(normalized_text, target_style)
        
        # 处理空格
        normalized_text = self._normalize_spaces(normalized_text)
        
        return normalized_text
    
    def _normalize_quotes(self, text: str, style: PunctuationStyle) -> str:
        """规范化引号"""
        if style == PunctuationStyle.CHINESE:
            # 转换为中文引号 - 简化实现
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace("'", "'").replace("'", "'")
        elif style == PunctuationStyle.ENGLISH:
            # 转换为英文引号 - 简化实现
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace("'", "'").replace("'", "'")
        
        return text
    
    def _normalize_spaces(self, text: str) -> str:
        """规范化空格"""
        # 中英文之间添加空格
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
        
        # 数字和中文之间添加空格
        text = re.sub(r'([\u4e00-\u9fff])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([\u4e00-\u9fff])', r'\1 \2', text)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def assess_text_quality(self, text: str) -> TextQualityMetrics:
        """
        中文文本质量评估
        
        Args:
            text: 输入文本
            
        Returns:
            文本质量指标
        """
        if not text.strip():
            return TextQualityMetrics(
                readability_score=0.0,
                fluency_score=0.0,
                coherence_score=0.0,
                complexity_score=0.0,
                punctuation_score=0.0,
                character_diversity=0.0,
                word_diversity=0.0,
                avg_sentence_length=0.0
            )
        
        # 分词
        tokens = self.segment_text(text)
        words = [token.word for token in tokens]
        
        # 句子分割
        sentences = self._split_sentences(text)
        
        # 计算各项指标
        readability = self._calculate_readability(text, words, sentences)
        fluency = self._calculate_fluency(tokens)
        coherence = self._calculate_coherence(sentences)
        complexity = self._calculate_complexity(words, tokens)
        punctuation = self._assess_punctuation_quality(text)
        char_diversity = self._calculate_character_diversity(text)
        word_diversity = self._calculate_word_diversity(words)
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        return TextQualityMetrics(
            readability_score=readability,
            fluency_score=fluency,
            coherence_score=coherence,
            complexity_score=complexity,
            punctuation_score=punctuation,
            character_diversity=char_diversity,
            word_diversity=word_diversity,
            avg_sentence_length=avg_sentence_length
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 中文句子分割
        sentences = re.split(r'[。！？；]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_readability(
        self, 
        text: str, 
        words: List[str], 
        sentences: List[str]
    ) -> float:
        """计算可读性评分"""
        if not sentences or not words:
            return 0.0
        
        # 基于句长和词长的可读性评估
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 理想句长：10-20词，理想词长：1-3字符
        sentence_score = 1.0 - abs(avg_sentence_length - 15) / 15
        word_score = 1.0 - abs(avg_word_length - 2) / 2
        
        sentence_score = max(0.0, min(1.0, sentence_score))
        word_score = max(0.0, min(1.0, word_score))
        
        return (sentence_score + word_score) / 2
    
    def _calculate_fluency(self, tokens: List[ChineseToken]) -> float:
        """计算流畅度评分"""
        if not tokens:
            return 0.0
        
        # 基于词性序列的流畅度评估
        pos_sequence = [token.pos for token in tokens]
        
        # 检查常见的不流畅模式
        fluency_issues = 0
        
        # 连续的相同词性（除了名词）
        for i in range(len(pos_sequence) - 1):
            if pos_sequence[i] == pos_sequence[i + 1] and pos_sequence[i] not in ['n', 'nr', 'ns']:
                fluency_issues += 1
        
        # 动词后直接跟动词
        for i in range(len(pos_sequence) - 1):
            if pos_sequence[i].startswith('v') and pos_sequence[i + 1].startswith('v'):
                fluency_issues += 1
        
        # 计算流畅度分数
        fluency_score = 1.0 - (fluency_issues / len(tokens))
        return max(0.0, min(1.0, fluency_score))
    
    def _calculate_coherence(self, sentences: List[str]) -> float:
        """计算连贯性评分"""
        if len(sentences) < 2:
            return 1.0
        
        # 基于连接词和语义相似度的连贯性评估
        coherence_indicators = [
            '因此', '所以', '然后', '接下来', '首先', '其次', '最后',
            '由于', '因为', '但是', '然而', '此外', '另外', '同时',
            '相反', '类似', '同样', '不过', '而且', '并且'
        ]
        
        coherence_count = 0
        for sentence in sentences:
            if any(indicator in sentence for indicator in coherence_indicators):
                coherence_count += 1
        
        coherence_ratio = coherence_count / len(sentences)
        return min(1.0, coherence_ratio * 2)  # 最多50%的句子有连接词就得满分
    
    def _calculate_complexity(self, words: List[str], tokens: List[ChineseToken]) -> float:
        """计算复杂度评分"""
        if not words:
            return 0.0
        
        # 基于词汇复杂度和句法复杂度
        
        # 词汇复杂度：长词和专业术语的比例
        long_words = [w for w in words if len(w) > 3]
        crypto_terms = [t for t in tokens if t.is_crypto_term]
        
        vocab_complexity = (len(long_words) + len(crypto_terms)) / len(words)
        
        # 句法复杂度：基于词性多样性
        pos_types = set(token.pos for token in tokens)
        syntax_complexity = len(pos_types) / 20  # 假设最多20种词性
        
        complexity = (vocab_complexity + syntax_complexity) / 2
        return min(1.0, complexity)
    
    def _assess_punctuation_quality(self, text: str) -> float:
        """评估标点符号质量"""
        if not text.strip():
            return 0.0
        
        # 检查标点符号的规范性
        issues = 0
        
        # 检查混合标点
        has_chinese_punct = any(p in text for p in ['，', '。', '？', '！'])
        has_english_punct = any(p in text for p in [',', '.', '?', '!'])
        
        if has_chinese_punct and has_english_punct:
            issues += 1
        
        # 检查标点符号密度
        punct_count = len(re.findall(r'[，。？！,.\?!；;：:]', text))
        char_count = len(text)
        punct_density = punct_count / char_count if char_count > 0 else 0
        
        # 理想标点密度：5-15%
        if punct_density < 0.05 or punct_density > 0.15:
            issues += 1
        
        # 检查连续标点
        if re.search(r'[，。？！,.\?!]{2,}', text):
            issues += 1
        
        # 计算质量分数
        quality_score = 1.0 - (issues * 0.3)
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_character_diversity(self, text: str) -> float:
        """计算字符多样性"""
        if not text:
            return 0.0
        
        # 去除标点和空格
        chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        if not chars:
            return 0.0
        
        unique_chars = set(chars)
        diversity = len(unique_chars) / len(chars)
        return min(1.0, diversity * 2)  # 50%不重复字符得满分
    
    def _calculate_word_diversity(self, words: List[str]) -> float:
        """计算词汇多样性"""
        if not words:
            return 0.0
        
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        return min(1.0, diversity * 1.5)  # 67%不重复词汇得满分
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的字符级相似度
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)
        
        if not total_chars:
            return 1.0
        
        return len(common_chars) / len(total_chars)
    
    def optimize_qwen_tokenizer(
        self, 
        texts: List[str], 
        vocab_size: int = 32000
    ) -> TokenizerOptimization:
        """
        优化Qwen tokenizer的中文处理
        
        Args:
            texts: 训练文本列表
            vocab_size: 词汇表大小
            
        Returns:
            优化配置信息
        """
        if not texts:
            return TokenizerOptimization(
                vocab_size=vocab_size,
                special_tokens=[],
                chinese_vocab_ratio=0.0,
                crypto_terms_count=0,
                oov_rate=1.0,
                compression_ratio=1.0
            )
        
        # 统计中文字符和词汇
        all_chars = set()
        all_words = set()
        crypto_terms_found = set()
        
        total_chars = 0
        total_tokens = 0
        
        for text in texts:
            # 字符统计
            chinese_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            all_chars.update(chinese_chars)
            total_chars += len(chinese_chars)
            
            # 分词统计
            tokens = self.segment_text(text)
            words = [token.word for token in tokens]
            all_words.update(words)
            total_tokens += len(words)
            
            # 密码学术语统计
            for token in tokens:
                if token.is_crypto_term:
                    crypto_terms_found.add(token.word)
        
        # 计算指标
        chinese_vocab_ratio = len(all_chars) / vocab_size if vocab_size > 0 else 0
        crypto_terms_count = len(crypto_terms_found)
        
        # 估算OOV率（简化计算）
        common_vocab = set('的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞')
        common_chars = all_chars & common_vocab
        oov_rate = 1.0 - (len(common_chars) / len(all_chars)) if all_chars else 0.0
        
        # 估算压缩比
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 1.0
        
        # 特殊token建议
        special_tokens = ['<thinking>', '</thinking>', '<crypto>', '</crypto>']
        special_tokens.extend(list(crypto_terms_found)[:100])  # 添加前100个密码学术语
        
        return TokenizerOptimization(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            chinese_vocab_ratio=chinese_vocab_ratio,
            crypto_terms_count=crypto_terms_count,
            oov_rate=oov_rate,
            compression_ratio=compression_ratio
        )
    
    def preprocess_for_training(
        self, 
        text: str, 
        normalize_variant: bool = True,
        normalize_punctuation: bool = True,
        target_variant: TextVariant = TextVariant.SIMPLIFIED
    ) -> str:
        """
        为训练预处理文本
        
        Args:
            text: 输入文本
            normalize_variant: 是否规范化繁简体
            normalize_punctuation: 是否规范化标点符号
            target_variant: 目标文本变体
            
        Returns:
            预处理后的文本
        """
        if not text.strip():
            return text
        
        processed_text = text
        
        # 繁简体转换
        if normalize_variant:
            processed_text = self.convert_traditional_simplified(
                processed_text, target_variant
            )
        
        # 标点符号规范化
        if normalize_punctuation:
            processed_text = self.normalize_punctuation(
                processed_text, PunctuationStyle.CHINESE
            )
        
        # 清理多余空白字符
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text.strip()
        
        return processed_text
    
    def extract_crypto_terms_from_text(self, text: str) -> List[str]:
        """
        从文本中提取密码学术语
        
        Args:
            text: 输入文本
            
        Returns:
            提取的密码学术语列表
        """
        tokens = self.segment_text(text)
        crypto_terms = [token.word for token in tokens if token.is_crypto_term]
        return list(set(crypto_terms))  # 去重
    
    def calculate_chinese_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> ChineseMetrics:
        """
        计算中文特定的评估指标
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            中文评估指标
        """
        if len(predictions) != len(references):
            raise ValueError("预测文本和参考文本数量不匹配")
        
        if not predictions:
            return ChineseMetrics(
                character_accuracy=0.0,
                word_accuracy=0.0,
                rouge_l_chinese=0.0,
                bleu_chinese=0.0,
                crypto_term_accuracy=0.0
            )
        
        # 计算字符级准确率
        char_accuracy = self._calculate_character_accuracy(predictions, references)
        
        # 计算词级准确率
        word_accuracy = self._calculate_word_accuracy(predictions, references)
        
        # 计算中文ROUGE-L
        rouge_l = self._calculate_chinese_rouge_l(predictions, references)
        
        # 计算中文BLEU
        bleu = self._calculate_chinese_bleu(predictions, references)
        
        # 计算密码学术语准确率
        crypto_accuracy = self._calculate_crypto_term_accuracy(predictions, references)
        
        return ChineseMetrics(
            character_accuracy=char_accuracy,
            word_accuracy=word_accuracy,
            rouge_l_chinese=rouge_l,
            bleu_chinese=bleu,
            crypto_term_accuracy=crypto_accuracy
        )
    
    def _calculate_character_accuracy(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """计算字符级准确率"""
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred)
            ref_chars = list(ref)
            
            # 使用最长公共子序列计算准确率
            lcs_length = self._lcs_length(pred_chars, ref_chars)
            total_chars += max(len(pred_chars), len(ref_chars))
            correct_chars += lcs_length
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def _calculate_word_accuracy(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """计算词级准确率"""
        total_words = 0
        correct_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.segment_text(pred)
            ref_tokens = self.segment_text(ref)
            
            pred_words = [token.word for token in pred_tokens]
            ref_words = [token.word for token in ref_tokens]
            
            # 使用最长公共子序列计算准确率
            lcs_length = self._lcs_length(pred_words, ref_words)
            total_words += max(len(pred_words), len(ref_words))
            correct_words += lcs_length
        
        return correct_words / total_words if total_words > 0 else 0.0
    
    def _calculate_chinese_rouge_l(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """计算中文ROUGE-L分数"""
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred)
            ref_chars = list(ref)
            
            lcs_length = self._lcs_length(pred_chars, ref_chars)
            
            if len(pred_chars) == 0 and len(ref_chars) == 0:
                rouge_scores.append(1.0)
            elif len(pred_chars) == 0 or len(ref_chars) == 0:
                rouge_scores.append(0.0)
            else:
                precision = lcs_length / len(pred_chars)
                recall = lcs_length / len(ref_chars)
                
                if precision + recall == 0:
                    rouge_scores.append(0.0)
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                    rouge_scores.append(f1)
        
        return sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    def _calculate_chinese_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """计算中文BLEU分数"""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.segment_text(pred)
            ref_tokens = self.segment_text(ref)
            
            pred_words = [token.word for token in pred_tokens]
            ref_words = [token.word for token in ref_tokens]
            
            # 计算1-gram到4-gram的精确度
            precisions = []
            for n in range(1, 5):
                pred_ngrams = self._get_ngrams(pred_words, n)
                ref_ngrams = self._get_ngrams(ref_words, n)
                
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                
                matches = 0
                for ngram in pred_ngrams:
                    if ngram in ref_ngrams:
                        matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
                
                precision = matches / sum(pred_ngrams.values())
                precisions.append(precision)
            
            # 计算几何平均
            if all(p > 0 for p in precisions):
                geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
                
                # 简化的长度惩罚
                bp = min(1.0, len(pred_words) / len(ref_words)) if ref_words else 0.0
                bleu = bp * geo_mean
            else:
                bleu = 0.0
            
            bleu_scores.append(bleu)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    def _calculate_crypto_term_accuracy(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """计算密码学术语准确率"""
        total_terms = 0
        correct_terms = 0
        
        for pred, ref in zip(predictions, references):
            pred_crypto_terms = set(self.extract_crypto_terms_from_text(pred))
            ref_crypto_terms = set(self.extract_crypto_terms_from_text(ref))
            
            total_terms += len(ref_crypto_terms)
            correct_terms += len(pred_crypto_terms & ref_crypto_terms)
        
        return correct_terms / total_terms if total_terms > 0 else 1.0
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _get_ngrams(self, words: List[str], n: int) -> Dict[tuple, int]:
        """获取n-gram统计"""
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams