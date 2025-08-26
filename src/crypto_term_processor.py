"""
密码学专业术语处理模块

本模块实现密码学术语词典和分类系统、专业术语识别和标注功能、
术语复杂度评估算法，以及thinking数据中的专业术语处理。
"""

import re
import json
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict

try:
    from src.data_models import CryptoTerm, CryptoCategory, ThinkingStructure, ThinkingExample
    from src.chinese_nlp_processor import ChineseNLPProcessor, ChineseToken
except ImportError:
    from src.data_models import CryptoTerm, CryptoCategory, ThinkingStructure, ThinkingExample
    from src.chinese_nlp_processor import ChineseNLPProcessor, ChineseToken


class TermComplexity(Enum):
    """术语复杂度级别"""
    BASIC = 1      # 基础术语
    INTERMEDIATE = 2  # 中级术语
    ADVANCED = 3   # 高级术语
    EXPERT = 4     # 专家级术语


@dataclass
class TermAnnotation:
    """术语标注结果"""
    term: str
    category: CryptoCategory
    complexity: int
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    definition: Optional[str] = None
    related_terms: List[str] = field(default_factory=list)


@dataclass
class TermDistribution:
    """术语分布统计"""
    total_terms: int
    unique_terms: int
    category_distribution: Dict[CryptoCategory, int]
    complexity_distribution: Dict[int, int]
    term_frequency: Dict[str, int]
    coverage_ratio: float  # 术语覆盖率


@dataclass
class ThinkingTermAnalysis:
    """Thinking数据中的术语分析"""
    thinking_id: str
    total_terms: int
    unique_terms: int
    term_annotations: List[TermAnnotation]
    complexity_score: float
    professional_score: float  # 专业性评分
    term_coherence: float  # 术语连贯性


class CryptoTermProcessor:
    """密码学术语处理器"""
    
    def __init__(self, custom_dict_path: Optional[str] = None):
        """
        初始化密码学术语处理器
        
        Args:
            custom_dict_path: 自定义术语词典路径
        """
        self.chinese_processor = ChineseNLPProcessor()
        
        # 构建密码学术语词典
        self.crypto_dict = self._build_crypto_dictionary()
        
        # 加载自定义词典
        if custom_dict_path:
            self._load_custom_dictionary(custom_dict_path)
        
        # 术语复杂度映射
        self.complexity_mapping = self._build_complexity_mapping()
        
        # 术语关系图
        self.term_relations = self._build_term_relations()
        
        # 上下文模式
        self.context_patterns = self._build_context_patterns()
    
    def _build_crypto_dictionary(self) -> Dict[str, CryptoTerm]:
        """构建密码学术语词典"""
        crypto_terms = {
            # 对称加密
            "对称加密": CryptoTerm(
                term="对称加密",
                definition="使用相同密钥进行加密和解密的加密方式",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=2,
                aliases=["对称密码", "私钥加密"],
                related_terms=["AES", "DES", "密钥管理"],
                examples=["AES是最常用的对称加密算法"]
            ),
            "AES": CryptoTerm(
                term="AES",
                definition="高级加密标准，一种对称加密算法",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=3,
                aliases=["高级加密标准", "Rijndael"],
                related_terms=["对称加密", "分组密码", "密钥长度"],
                examples=["AES-256提供了很高的安全性"]
            ),
            "DES": CryptoTerm(
                term="DES",
                definition="数据加密标准，较老的对称加密算法",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=3,
                aliases=["数据加密标准"],
                related_terms=["3DES", "对称加密", "分组密码"],
                examples=["DES由于密钥长度较短已不推荐使用"]
            ),
            "3DES": CryptoTerm(
                term="3DES",
                definition="三重DES，对DES的改进版本",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=4,
                aliases=["三重DES", "TDES"],
                related_terms=["DES", "对称加密"],
                examples=["3DES通过三次DES操作提高安全性"]
            ),
            
            # 非对称加密
            "非对称加密": CryptoTerm(
                term="非对称加密",
                definition="使用不同密钥进行加密和解密的加密方式",
                category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
                complexity=3,
                aliases=["公钥加密", "公开密钥加密"],
                related_terms=["RSA", "椭圆曲线", "数字签名"],
                examples=["RSA是最著名的非对称加密算法"]
            ),
            "RSA": CryptoTerm(
                term="RSA",
                definition="基于大整数分解难题的非对称加密算法",
                category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
                complexity=4,
                aliases=["RSA算法"],
                related_terms=["非对称加密", "数字签名", "密钥交换"],
                examples=["RSA-2048是目前推荐的密钥长度"]
            ),
            "椭圆曲线": CryptoTerm(
                term="椭圆曲线",
                definition="基于椭圆曲线离散对数问题的密码学",
                category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
                complexity=5,
                aliases=["ECC", "椭圆曲线密码学"],
                related_terms=["ECDSA", "ECDH", "非对称加密"],
                examples=["椭圆曲线加密提供了更高的安全性和效率"]
            ),
            
            # 哈希函数
            "哈希函数": CryptoTerm(
                term="哈希函数",
                definition="将任意长度输入映射为固定长度输出的函数",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=2,
                aliases=["散列函数", "摘要函数"],
                related_terms=["SHA", "MD5", "完整性"],
                examples=["SHA-256是广泛使用的哈希函数"]
            ),
            "SHA": CryptoTerm(
                term="SHA",
                definition="安全哈希算法系列",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=3,
                aliases=["安全哈希算法"],
                related_terms=["SHA-1", "SHA-256", "SHA-3"],
                examples=["SHA-256被比特币网络广泛使用"]
            ),
            "MD5": CryptoTerm(
                term="MD5",
                definition="消息摘要算法5，已被认为不安全",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=2,
                aliases=["消息摘要5"],
                related_terms=["哈希函数", "碰撞攻击"],
                examples=["MD5由于存在碰撞攻击已不推荐使用"]
            ),
            
            # 数字签名
            "数字签名": CryptoTerm(
                term="数字签名",
                definition="用于验证消息完整性和发送者身份的密码学技术",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=3,
                aliases=["电子签名"],
                related_terms=["RSA签名", "ECDSA", "认证"],
                examples=["数字签名确保了消息的不可否认性"]
            ),
            "ECDSA": CryptoTerm(
                term="ECDSA",
                definition="椭圆曲线数字签名算法",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=5,
                aliases=["椭圆曲线数字签名算法"],
                related_terms=["椭圆曲线", "数字签名", "ECDH"],
                examples=["ECDSA被广泛用于区块链技术中"]
            ),
            
            # 密钥管理
            "密钥管理": CryptoTerm(
                term="密钥管理",
                definition="密钥的生成、分发、存储和销毁的管理过程",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=4,
                aliases=["密钥管理系统", "KMS"],
                related_terms=["密钥交换", "密钥分发", "PKI"],
                examples=["良好的密钥管理是密码系统安全的基础"]
            ),
            "密钥交换": CryptoTerm(
                term="密钥交换",
                definition="在不安全信道上安全交换密钥的协议",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=4,
                aliases=["密钥协商"],
                related_terms=["Diffie-Hellman", "ECDH", "密钥管理"],
                examples=["Diffie-Hellman是最著名的密钥交换协议"]
            ),
            "PKI": CryptoTerm(
                term="PKI",
                definition="公钥基础设施，管理数字证书的框架",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=5,
                aliases=["公钥基础设施", "公钥基础架构"],
                related_terms=["数字证书", "CA", "密钥管理"],
                examples=["PKI是现代网络安全的重要基础设施"]
            ),
            
            # 密码协议
            "SSL": CryptoTerm(
                term="SSL",
                definition="安全套接字层协议",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=4,
                aliases=["安全套接字层"],
                related_terms=["TLS", "HTTPS", "密码协议"],
                examples=["SSL已被更安全的TLS协议取代"]
            ),
            "TLS": CryptoTerm(
                term="TLS",
                definition="传输层安全协议",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=4,
                aliases=["传输层安全"],
                related_terms=["SSL", "HTTPS", "握手协议"],
                examples=["TLS 1.3是目前最新的版本"]
            ),
            "HTTPS": CryptoTerm(
                term="HTTPS",
                definition="基于TLS/SSL的安全HTTP协议",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=3,
                aliases=["安全HTTP"],
                related_terms=["TLS", "SSL", "Web安全"],
                examples=["现代网站都应该使用HTTPS"]
            ),
            
            # 密码分析
            "密码分析": CryptoTerm(
                term="密码分析",
                definition="研究破解密码系统的学科",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=4,
                aliases=["密码破译", "密码攻击"],
                related_terms=["差分分析", "线性分析", "侧信道攻击"],
                examples=["密码分析帮助发现密码算法的弱点"]
            ),
            "差分分析": CryptoTerm(
                term="差分分析",
                definition="一种密码分析技术，分析输入差分对输出的影响",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=6,
                aliases=["差分密码分析"],
                related_terms=["密码分析", "线性分析", "分组密码"],
                examples=["差分分析是分析分组密码的重要方法"]
            ),
            "侧信道攻击": CryptoTerm(
                term="侧信道攻击",
                definition="通过分析密码设备的物理特征进行攻击",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=6,
                aliases=["旁路攻击"],
                related_terms=["功耗分析", "时间攻击", "电磁分析"],
                examples=["侧信道攻击是实际密码设备面临的重要威胁"]
            ),
            
            # 更多对称加密术语
            "分组密码": CryptoTerm(
                term="分组密码",
                definition="将明文分成固定长度的块进行加密的密码算法",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=4,
                aliases=["块密码", "分块加密"],
                related_terms=["AES", "DES", "CBC", "ECB"],
                examples=["AES是最常用的分组密码算法"]
            ),
            "流密码": CryptoTerm(
                term="流密码",
                definition="逐位或逐字节加密明文的密码算法",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=4,
                aliases=["序列密码"],
                related_terms=["RC4", "ChaCha20", "密钥流"],
                examples=["RC4是著名的流密码算法"]
            ),
            "CBC": CryptoTerm(
                term="CBC",
                definition="密码分组链接模式，一种分组密码的工作模式",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=5,
                aliases=["密码分组链接", "Cipher Block Chaining"],
                related_terms=["分组密码", "初始化向量", "ECB"],
                examples=["CBC模式需要初始化向量来保证安全性"]
            ),
            "ECB": CryptoTerm(
                term="ECB",
                definition="电子密码本模式，最简单的分组密码工作模式",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=4,
                aliases=["电子密码本", "Electronic Codebook"],
                related_terms=["分组密码", "CBC", "CTR"],
                examples=["ECB模式存在安全缺陷，不推荐使用"]
            ),
            "CTR": CryptoTerm(
                term="CTR",
                definition="计数器模式，将分组密码转换为流密码的工作模式",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=5,
                aliases=["计数器模式", "Counter Mode"],
                related_terms=["分组密码", "流密码", "并行加密"],
                examples=["CTR模式支持并行加密和解密"]
            ),
            "GCM": CryptoTerm(
                term="GCM",
                definition="伽罗瓦计数器模式，提供加密和认证的工作模式",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=6,
                aliases=["伽罗瓦计数器模式", "Galois Counter Mode"],
                related_terms=["AEAD", "认证加密", "CTR"],
                examples=["AES-GCM广泛用于TLS协议中"]
            ),
            "ChaCha20": CryptoTerm(
                term="ChaCha20",
                definition="现代流密码算法，Salsa20的改进版本",
                category=CryptoCategory.SYMMETRIC_ENCRYPTION,
                complexity=5,
                aliases=["ChaCha20流密码"],
                related_terms=["流密码", "Poly1305", "AEAD"],
                examples=["ChaCha20-Poly1305是现代AEAD算法"]
            ),
            
            # 更多非对称加密术语
            "Diffie-Hellman": CryptoTerm(
                term="Diffie-Hellman",
                definition="第一个公开的密钥交换协议",
                category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
                complexity=5,
                aliases=["DH", "迪菲-赫尔曼"],
                related_terms=["密钥交换", "离散对数", "ECDH"],
                examples=["Diffie-Hellman协议奠定了现代密码学基础"]
            ),
            "ECDH": CryptoTerm(
                term="ECDH",
                definition="椭圆曲线Diffie-Hellman密钥交换协议",
                category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
                complexity=6,
                aliases=["椭圆曲线DH"],
                related_terms=["椭圆曲线", "密钥交换", "ECDSA"],
                examples=["ECDH提供了更高效的密钥交换"]
            ),
            "DSA": CryptoTerm(
                term="DSA",
                definition="数字签名算法，基于离散对数问题",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=5,
                aliases=["数字签名算法", "Digital Signature Algorithm"],
                related_terms=["数字签名", "离散对数", "ECDSA"],
                examples=["DSA是美国政府的数字签名标准"]
            ),
            "EdDSA": CryptoTerm(
                term="EdDSA",
                definition="爱德华兹曲线数字签名算法",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=6,
                aliases=["爱德华兹曲线签名"],
                related_terms=["椭圆曲线", "Ed25519", "数字签名"],
                examples=["Ed25519是EdDSA的一个具体实现"]
            ),
            "Ed25519": CryptoTerm(
                term="Ed25519",
                definition="基于Curve25519的EdDSA签名算法",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=6,
                aliases=["Ed25519签名"],
                related_terms=["EdDSA", "Curve25519", "椭圆曲线"],
                examples=["Ed25519提供了高性能的数字签名"]
            ),
            
            # 更多哈希函数术语
            "SHA-1": CryptoTerm(
                term="SHA-1",
                definition="安全哈希算法1，已被认为不安全",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=3,
                aliases=["SHA1"],
                related_terms=["SHA", "SHA-256", "碰撞攻击"],
                examples=["SHA-1由于碰撞攻击已被弃用"]
            ),
            "SHA-256": CryptoTerm(
                term="SHA-256",
                definition="SHA-2系列的256位哈希函数",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=4,
                aliases=["SHA256"],
                related_terms=["SHA-2", "比特币", "区块链"],
                examples=["SHA-256是比特币的核心哈希算法"]
            ),
            "SHA-3": CryptoTerm(
                term="SHA-3",
                definition="最新的安全哈希算法标准，基于Keccak",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=5,
                aliases=["SHA3", "Keccak"],
                related_terms=["哈希函数", "NIST", "海绵构造"],
                examples=["SHA-3使用了全新的海绵构造设计"]
            ),
            "BLAKE2": CryptoTerm(
                term="BLAKE2",
                definition="高性能密码哈希函数，BLAKE的改进版本",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=5,
                aliases=["BLAKE2哈希"],
                related_terms=["哈希函数", "BLAKE", "高性能"],
                examples=["BLAKE2在某些场景下比SHA-3更快"]
            ),
            "HMAC": CryptoTerm(
                term="HMAC",
                definition="基于哈希的消息认证码",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=4,
                aliases=["哈希消息认证码"],
                related_terms=["消息认证", "哈希函数", "密钥"],
                examples=["HMAC-SHA256广泛用于API认证"]
            ),
            "Merkle树": CryptoTerm(
                term="Merkle树",
                definition="二叉树结构，用于高效验证大量数据的完整性",
                category=CryptoCategory.HASH_FUNCTION,
                complexity=5,
                aliases=["默克尔树", "哈希树"],
                related_terms=["哈希函数", "区块链", "数据完整性"],
                examples=["比特币使用Merkle树组织交易数据"]
            ),
            
            # 更多密钥管理术语
            "CA": CryptoTerm(
                term="CA",
                definition="证书颁发机构，负责颁发和管理数字证书",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=4,
                aliases=["证书颁发机构", "Certificate Authority"],
                related_terms=["PKI", "数字证书", "信任链"],
                examples=["Let's Encrypt是著名的免费CA"]
            ),
            "数字证书": CryptoTerm(
                term="数字证书",
                definition="绑定公钥与身份信息的电子文档",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=4,
                aliases=["X.509证书", "公钥证书"],
                related_terms=["PKI", "CA", "公钥"],
                examples=["HTTPS网站使用数字证书验证身份"]
            ),
            "CRL": CryptoTerm(
                term="CRL",
                definition="证书撤销列表，记录已撤销的数字证书",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=5,
                aliases=["证书撤销列表", "Certificate Revocation List"],
                related_terms=["数字证书", "OCSP", "PKI"],
                examples=["CRL用于检查证书是否已被撤销"]
            ),
            "OCSP": CryptoTerm(
                term="OCSP",
                definition="在线证书状态协议，实时检查证书状态",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=5,
                aliases=["在线证书状态协议"],
                related_terms=["数字证书", "CRL", "证书验证"],
                examples=["OCSP提供了比CRL更及时的证书状态检查"]
            ),
            "HSM": CryptoTerm(
                term="HSM",
                definition="硬件安全模块，专用的密码处理硬件设备",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=6,
                aliases=["硬件安全模块", "Hardware Security Module"],
                related_terms=["密钥管理", "硬件加密", "FIPS"],
                examples=["银行使用HSM保护关键密钥"]
            ),
            "密钥派生": CryptoTerm(
                term="密钥派生",
                definition="从主密钥或密码生成其他密钥的过程",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=5,
                aliases=["密钥导出", "KDF"],
                related_terms=["PBKDF2", "scrypt", "Argon2"],
                examples=["PBKDF2是常用的密钥派生函数"]
            ),
            "PBKDF2": CryptoTerm(
                term="PBKDF2",
                definition="基于密码的密钥派生函数2",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=5,
                aliases=["Password-Based Key Derivation Function 2"],
                related_terms=["密钥派生", "密码哈希", "盐值"],
                examples=["PBKDF2通过多次迭代增强密码安全性"]
            ),
            "scrypt": CryptoTerm(
                term="scrypt",
                definition="内存困难的密钥派生函数",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=6,
                aliases=["scrypt算法"],
                related_terms=["密钥派生", "内存困难", "ASIC抗性"],
                examples=["scrypt被莱特币等加密货币使用"]
            ),
            "Argon2": CryptoTerm(
                term="Argon2",
                definition="密码哈希竞赛获胜者，现代密钥派生函数",
                category=CryptoCategory.KEY_MANAGEMENT,
                complexity=6,
                aliases=["Argon2算法"],
                related_terms=["密钥派生", "密码哈希", "内存困难"],
                examples=["Argon2是目前推荐的密码哈希算法"]
            ),
            
            # 更多密码协议术语
            "握手协议": CryptoTerm(
                term="握手协议",
                definition="建立安全连接时协商参数的协议",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=4,
                aliases=["握手过程"],
                related_terms=["TLS", "密钥交换", "身份认证"],
                examples=["TLS握手协议建立安全连接"]
            ),
            "IPSec": CryptoTerm(
                term="IPSec",
                definition="IP层安全协议套件",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=5,
                aliases=["IP安全协议"],
                related_terms=["VPN", "ESP", "AH"],
                examples=["IPSec广泛用于VPN连接"]
            ),
            "SSH": CryptoTerm(
                term="SSH",
                definition="安全外壳协议，用于安全远程登录",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=4,
                aliases=["安全外壳", "Secure Shell"],
                related_terms=["远程登录", "公钥认证", "端口转发"],
                examples=["SSH是Linux系统远程管理的标准协议"]
            ),
            "SAML": CryptoTerm(
                term="SAML",
                definition="安全断言标记语言，用于身份认证和授权",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=5,
                aliases=["安全断言标记语言"],
                related_terms=["SSO", "身份认证", "XML签名"],
                examples=["SAML广泛用于企业单点登录系统"]
            ),
            "OAuth": CryptoTerm(
                term="OAuth",
                definition="开放授权协议，用于第三方应用授权",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=4,
                aliases=["开放授权"],
                related_terms=["JWT", "API授权", "访问令牌"],
                examples=["OAuth 2.0是现代API授权的标准"]
            ),
            "JWT": CryptoTerm(
                term="JWT",
                definition="JSON Web Token，用于安全传输信息的令牌格式",
                category=CryptoCategory.CRYPTOGRAPHIC_PROTOCOL,
                complexity=4,
                aliases=["JSON Web Token"],
                related_terms=["OAuth", "数字签名", "无状态认证"],
                examples=["JWT广泛用于微服务架构的认证"]
            ),
            
            # 更多密码分析术语
            "线性分析": CryptoTerm(
                term="线性分析",
                definition="利用密码算法的线性特性进行攻击的方法",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=6,
                aliases=["线性密码分析"],
                related_terms=["差分分析", "分组密码", "密码分析"],
                examples=["线性分析是分析DES等算法的重要方法"]
            ),
            "时间攻击": CryptoTerm(
                term="时间攻击",
                definition="通过分析算法执行时间获取密钥信息的攻击",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=5,
                aliases=["计时攻击", "Timing Attack"],
                related_terms=["侧信道攻击", "常数时间", "密码实现"],
                examples=["RSA实现需要防范时间攻击"]
            ),
            "功耗分析": CryptoTerm(
                term="功耗分析",
                definition="通过分析设备功耗变化获取密钥的攻击方法",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=6,
                aliases=["功耗攻击", "Power Analysis"],
                related_terms=["侧信道攻击", "DPA", "SPA"],
                examples=["智能卡需要防范功耗分析攻击"]
            ),
            "电磁分析": CryptoTerm(
                term="电磁分析",
                definition="通过分析设备电磁辐射获取密钥的攻击方法",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=6,
                aliases=["电磁攻击", "EM Analysis"],
                related_terms=["侧信道攻击", "TEMPEST", "电磁泄露"],
                examples=["军用设备需要防范电磁分析攻击"]
            ),
            "碰撞攻击": CryptoTerm(
                term="碰撞攻击",
                definition="寻找哈希函数输出相同的不同输入的攻击",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=5,
                aliases=["哈希碰撞"],
                related_terms=["哈希函数", "生日攻击", "MD5"],
                examples=["MD5和SHA-1都存在碰撞攻击"]
            ),
            "生日攻击": CryptoTerm(
                term="生日攻击",
                definition="基于生日悖论的概率攻击方法",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=5,
                aliases=["生日悖论攻击"],
                related_terms=["碰撞攻击", "概率分析", "哈希函数"],
                examples=["生日攻击降低了寻找哈希碰撞的复杂度"]
            ),
            "中间人攻击": CryptoTerm(
                term="中间人攻击",
                definition="攻击者在通信双方之间进行窃听和篡改的攻击",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=4,
                aliases=["MITM攻击", "Man-in-the-Middle"],
                related_terms=["密钥交换", "身份认证", "证书验证"],
                examples=["HTTPS通过证书验证防范中间人攻击"]
            ),
            "重放攻击": CryptoTerm(
                term="重放攻击",
                definition="攻击者重复发送之前截获的有效消息的攻击",
                category=CryptoCategory.CRYPTANALYSIS,
                complexity=4,
                aliases=["回放攻击", "Replay Attack"],
                related_terms=["时间戳", "随机数", "序列号"],
                examples=["协议设计需要防范重放攻击"]
            ),
            
            # 区块链相关术语
            "区块链": CryptoTerm(
                term="区块链",
                definition="分布式账本技术，使用密码学保证数据完整性",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=4,
                aliases=["分布式账本"],
                related_terms=["比特币", "智能合约", "共识算法"],
                examples=["区块链技术被广泛应用于数字货币"]
            ),
            "比特币": CryptoTerm(
                term="比特币",
                definition="第一个成功的加密货币",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=3,
                aliases=["Bitcoin", "BTC"],
                related_terms=["区块链", "工作量证明", "数字签名"],
                examples=["比特币使用SHA-256和ECDSA"]
            ),
            "智能合约": CryptoTerm(
                term="智能合约",
                definition="自动执行的合约程序",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=4,
                aliases=["智能契约"],
                related_terms=["以太坊", "区块链", "去中心化"],
                examples=["智能合约在以太坊平台上广泛使用"]
            ),
            "工作量证明": CryptoTerm(
                term="工作量证明",
                definition="通过计算工作量达成共识的机制",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=5,
                aliases=["PoW", "Proof of Work"],
                related_terms=["挖矿", "共识算法", "比特币"],
                examples=["比特币使用工作量证明保证网络安全"]
            ),
            "权益证明": CryptoTerm(
                term="权益证明",
                definition="基于持有代币数量的共识机制",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=5,
                aliases=["PoS", "Proof of Stake"],
                related_terms=["共识算法", "质押", "以太坊2.0"],
                examples=["以太坊2.0采用权益证明机制"]
            ),
            "哈希指针": CryptoTerm(
                term="哈希指针",
                definition="包含哈希值的指针，用于链接区块",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=4,
                aliases=["哈希链接"],
                related_terms=["区块链", "哈希函数", "数据完整性"],
                examples=["区块链通过哈希指针连接各个区块"]
            ),
            "挖矿": CryptoTerm(
                term="挖矿",
                definition="通过计算寻找满足条件的哈希值的过程",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=4,
                aliases=["采矿", "Mining"],
                related_terms=["工作量证明", "比特币", "算力"],
                examples=["比特币挖矿需要大量的计算资源"]
            ),
            "以太坊": CryptoTerm(
                term="以太坊",
                definition="支持智能合约的区块链平台",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=4,
                aliases=["Ethereum", "ETH"],
                related_terms=["智能合约", "EVM", "DApp"],
                examples=["以太坊是最大的智能合约平台"]
            ),
            "DeFi": CryptoTerm(
                term="DeFi",
                definition="去中心化金融，基于区块链的金融服务",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=5,
                aliases=["去中心化金融", "Decentralized Finance"],
                related_terms=["智能合约", "流动性挖矿", "AMM"],
                examples=["DeFi协议提供了传统金融的替代方案"]
            ),
            "NFT": CryptoTerm(
                term="NFT",
                definition="非同质化代币，表示独特数字资产的代币",
                category=CryptoCategory.BLOCKCHAIN,
                complexity=4,
                aliases=["非同质化代币", "Non-Fungible Token"],
                related_terms=["ERC-721", "数字收藏品", "元宇宙"],
                examples=["NFT广泛用于数字艺术品交易"]
            ),
            
            # 其他重要术语
            "零知识证明": CryptoTerm(
                term="零知识证明",
                definition="证明者在不泄露秘密的情况下证明知道秘密的方法",
                category=CryptoCategory.OTHER,
                complexity=7,
                aliases=["ZKP", "Zero-Knowledge Proof"],
                related_terms=["zk-SNARKs", "隐私保护", "区块链"],
                examples=["零知识证明用于保护区块链隐私"]
            ),
            "同态加密": CryptoTerm(
                term="同态加密",
                definition="允许在密文上直接进行计算的加密方法",
                category=CryptoCategory.OTHER,
                complexity=8,
                aliases=["Homomorphic Encryption"],
                related_terms=["隐私计算", "云计算安全", "FHE"],
                examples=["同态加密实现了密文计算"]
            ),
            "多方安全计算": CryptoTerm(
                term="多方安全计算",
                definition="多个参与方在不泄露各自输入的情况下共同计算函数",
                category=CryptoCategory.OTHER,
                complexity=7,
                aliases=["MPC", "Secure Multi-party Computation"],
                related_terms=["隐私保护", "秘密分享", "联邦学习"],
                examples=["MPC用于隐私保护的联合计算"]
            ),
            "秘密分享": CryptoTerm(
                term="秘密分享",
                definition="将秘密分割成多个份额分发给不同参与者的方法",
                category=CryptoCategory.OTHER,
                complexity=6,
                aliases=["Secret Sharing", "门限方案"],
                related_terms=["多方安全计算", "Shamir方案", "门限签名"],
                examples=["Shamir秘密分享是经典的门限方案"]
            ),
            "盲签名": CryptoTerm(
                term="盲签名",
                definition="签名者在不知道消息内容的情况下进行签名",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=6,
                aliases=["Blind Signature"],
                related_terms=["数字签名", "隐私保护", "电子现金"],
                examples=["盲签名用于保护签名请求者的隐私"]
            ),
            "环签名": CryptoTerm(
                term="环签名",
                definition="一组用户中任意一个成员可以代表整个组进行签名",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=6,
                aliases=["Ring Signature"],
                related_terms=["匿名签名", "门罗币", "隐私保护"],
                examples=["门罗币使用环签名保护交易隐私"]
            ),
            "群签名": CryptoTerm(
                term="群签名",
                definition="群成员可以匿名代表群体签名的方案",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=6,
                aliases=["Group Signature"],
                related_terms=["匿名认证", "可追踪性", "撤销机制"],
                examples=["群签名平衡了匿名性和可追踪性"]
            ),
            "门限签名": CryptoTerm(
                term="门限签名",
                definition="需要多个参与者协作才能生成的签名方案",
                category=CryptoCategory.DIGITAL_SIGNATURE,
                complexity=6,
                aliases=["Threshold Signature"],
                related_terms=["秘密分享", "多重签名", "分布式签名"],
                examples=["门限签名提高了密钥管理的安全性"]
            ),
        }
        
        return crypto_terms
    
    def _load_custom_dictionary(self, dict_path: str):
        """加载自定义术语词典"""
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                custom_terms = json.load(f)
            
            for term_data in custom_terms:
                term = CryptoTerm.from_dict(term_data)
                self.crypto_dict[term.term] = term
                
                # 添加别名
                for alias in term.aliases:
                    if alias not in self.crypto_dict:
                        self.crypto_dict[alias] = term
                
                # 将术语添加到jieba词典以便分词识别
                import jieba
                jieba.add_word(term.term, freq=1000, tag='crypto')
                for alias in term.aliases:
                    jieba.add_word(alias, freq=1000, tag='crypto')
                        
        except Exception as e:
            print(f"加载自定义词典失败: {e}")
    
    def _build_complexity_mapping(self) -> Dict[str, int]:
        """构建术语复杂度映射"""
        complexity_map = {}
        
        for term, crypto_term in self.crypto_dict.items():
            complexity_map[term] = crypto_term.complexity
        
        return complexity_map
    
    def _build_term_relations(self) -> Dict[str, Set[str]]:
        """构建术语关系图"""
        relations = defaultdict(set)
        
        for term, crypto_term in self.crypto_dict.items():
            # 添加相关术语关系
            for related_term in crypto_term.related_terms:
                relations[term].add(related_term)
                relations[related_term].add(term)
            
            # 添加别名关系
            for alias in crypto_term.aliases:
                relations[term].add(alias)
                relations[alias].add(term)
        
        return dict(relations)
    
    def _build_context_patterns(self) -> Dict[str, List[str]]:
        """构建上下文模式"""
        patterns = {
            "定义模式": [
                r"(.+)是一种(.+)",
                r"(.+)指的是(.+)",
                r"(.+)定义为(.+)",
                r"所谓(.+)就是(.+)"
            ],
            "比较模式": [
                r"(.+)与(.+)的区别",
                r"(.+)相比(.+)",
                r"(.+)和(.+)都是",
                r"不同于(.+)，(.+)"
            ],
            "应用模式": [
                r"(.+)用于(.+)",
                r"(.+)应用在(.+)",
                r"使用(.+)来(.+)",
                r"(.+)的应用场景"
            ],
            "安全模式": [
                r"(.+)的安全性",
                r"(.+)存在(.+)漏洞",
                r"(.+)可以防止(.+)",
                r"(.+)攻击(.+)"
            ]
        }
        
        return patterns
    
    def identify_crypto_terms(self, text: str) -> List[TermAnnotation]:
        """
        识别和标注密码学术语
        
        Args:
            text: 输入文本
            
        Returns:
            术语标注结果列表
        """
        annotations = []
        
        # 使用中文NLP处理器进行分词
        tokens = self.chinese_processor.segment_text(text)
        
        # 识别单个术语
        for token in tokens:
            if token.word in self.crypto_dict:
                crypto_term = self.crypto_dict[token.word]
                
                # 获取上下文
                context = self._extract_context(text, token.start_pos, token.end_pos)
                
                # 计算置信度
                confidence = self._calculate_term_confidence(token.word, context)
                
                annotation = TermAnnotation(
                    term=token.word,
                    category=crypto_term.category,
                    complexity=crypto_term.complexity,
                    confidence=confidence,
                    start_pos=token.start_pos,
                    end_pos=token.end_pos,
                    context=context,
                    definition=crypto_term.definition,
                    related_terms=crypto_term.related_terms
                )
                
                annotations.append(annotation)
        
        # 识别复合术语
        compound_annotations = self._identify_compound_terms(text, tokens)
        annotations.extend(compound_annotations)
        
        # 去重和排序
        annotations = self._deduplicate_annotations(annotations)
        annotations.sort(key=lambda x: x.start_pos)
        
        return annotations
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, window_size: int = 50) -> str:
        """提取术语上下文"""
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)
        
        return text[context_start:context_end]
    
    def _calculate_term_confidence(self, term: str, context: str) -> float:
        """计算术语识别置信度"""
        base_confidence = 0.8
        
        # 基于上下文模式提升置信度
        for pattern_type, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context):
                    base_confidence += 0.05
                    break
        
        # 基于相关术语提升置信度
        if term in self.term_relations:
            related_terms = self.term_relations[term]
            for related_term in related_terms:
                if related_term in context:
                    base_confidence += 0.03
        
        return min(1.0, base_confidence)
    
    def _identify_compound_terms(self, text: str, tokens: List[ChineseToken]) -> List[TermAnnotation]:
        """识别复合术语"""
        compound_annotations = []
        
        # 定义复合术语模式
        compound_patterns = [
            (r"RSA[-_]?\d+", CryptoCategory.ASYMMETRIC_ENCRYPTION, 4),
            (r"AES[-_]?\d+", CryptoCategory.SYMMETRIC_ENCRYPTION, 3),
            (r"SHA[-_]?\d+", CryptoCategory.HASH_FUNCTION, 3),
            (r"TLS\s*\d+\.\d+", CryptoCategory.CRYPTOGRAPHIC_PROTOCOL, 4),
            (r"椭圆曲线\s*P[-_]?\d+", CryptoCategory.ASYMMETRIC_ENCRYPTION, 5),
        ]
        
        for pattern, category, complexity in compound_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                term = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                context = self._extract_context(text, start_pos, end_pos)
                confidence = self._calculate_term_confidence(term, context)
                
                annotation = TermAnnotation(
                    term=term,
                    category=category,
                    complexity=complexity,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context
                )
                
                compound_annotations.append(annotation)
        
        return compound_annotations
    
    def _deduplicate_annotations(self, annotations: List[TermAnnotation]) -> List[TermAnnotation]:
        """去重术语标注"""
        # 按位置去重，保留置信度最高的
        position_map = {}
        
        for annotation in annotations:
            key = (annotation.start_pos, annotation.end_pos)
            
            if key not in position_map or annotation.confidence > position_map[key].confidence:
                position_map[key] = annotation
        
        return list(position_map.values())
    
    def calculate_term_complexity(self, terms: List[str]) -> float:
        """
        计算术语复杂度评分
        
        Args:
            terms: 术语列表
            
        Returns:
            复杂度评分 (0-10)
        """
        if not terms:
            return 0.0
        
        total_complexity = 0
        valid_terms = 0
        
        for term in terms:
            if term in self.complexity_mapping:
                total_complexity += self.complexity_mapping[term]
                valid_terms += 1
        
        if valid_terms == 0:
            return 0.0
        
        # 计算平均复杂度
        avg_complexity = total_complexity / valid_terms
        
        # 考虑术语多样性
        unique_terms = len(set(terms))
        diversity_bonus = min(1.0, unique_terms / 10) * 2  # 最多加2分
        
        # 考虑术语密度
        density_bonus = min(1.0, len(terms) / 20) * 1  # 最多加1分
        
        final_complexity = avg_complexity + diversity_bonus + density_bonus
        
        return min(10.0, final_complexity)
    
    def analyze_term_distribution(self, texts: List[str]) -> TermDistribution:
        """
        分析术语分布
        
        Args:
            texts: 文本列表
            
        Returns:
            术语分布统计
        """
        all_terms = []
        category_counter = Counter()
        complexity_counter = Counter()
        term_counter = Counter()
        
        total_words = 0
        
        for text in texts:
            annotations = self.identify_crypto_terms(text)
            tokens = self.chinese_processor.segment_text(text)
            total_words += len(tokens)
            
            for annotation in annotations:
                all_terms.append(annotation.term)
                category_counter[annotation.category] += 1
                complexity_counter[annotation.complexity] += 1
                term_counter[annotation.term] += 1
        
        unique_terms = len(set(all_terms))
        coverage_ratio = len(all_terms) / total_words if total_words > 0 else 0.0
        
        return TermDistribution(
            total_terms=len(all_terms),
            unique_terms=unique_terms,
            category_distribution=dict(category_counter),
            complexity_distribution=dict(complexity_counter),
            term_frequency=dict(term_counter),
            coverage_ratio=coverage_ratio
        )
    
    def process_thinking_terms(self, thinking_example: ThinkingExample) -> ThinkingTermAnalysis:
        """
        处理thinking数据中的专业术语
        
        Args:
            thinking_example: thinking训练样例
            
        Returns:
            thinking术语分析结果
        """
        # 合并所有文本内容
        full_text = f"{thinking_example.instruction} {thinking_example.thinking_process} {thinking_example.final_response}"
        
        # 识别术语
        annotations = self.identify_crypto_terms(full_text)
        
        # 分别分析不同部分的术语
        instruction_terms = self.identify_crypto_terms(thinking_example.instruction)
        thinking_terms = self.identify_crypto_terms(thinking_example.thinking_process)
        response_terms = self.identify_crypto_terms(thinking_example.final_response)
        
        # 计算复杂度评分
        all_term_names = [ann.term for ann in annotations]
        complexity_score = self.calculate_term_complexity(all_term_names)
        
        # 计算专业性评分
        professional_score = self._calculate_professional_score(annotations)
        
        # 计算术语连贯性
        term_coherence = self._calculate_term_coherence(
            instruction_terms, thinking_terms, response_terms
        )
        
        return ThinkingTermAnalysis(
            thinking_id=f"thinking_{hash(thinking_example.instruction)}",
            total_terms=len(annotations),
            unique_terms=len(set(all_term_names)),
            term_annotations=annotations,
            complexity_score=complexity_score,
            professional_score=professional_score,
            term_coherence=term_coherence
        )
    
    def _calculate_professional_score(self, annotations: List[TermAnnotation]) -> float:
        """计算专业性评分"""
        if not annotations:
            return 0.0
        
        # 基于术语类别多样性
        categories = set(ann.category for ann in annotations)
        category_diversity = len(categories) / len(CryptoCategory)
        
        # 基于术语复杂度分布
        complexities = [ann.complexity for ann in annotations]
        avg_complexity = sum(complexities) / len(complexities)
        complexity_score = avg_complexity / 10.0
        
        # 基于术语置信度
        confidences = [ann.confidence for ann in annotations]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 综合评分
        professional_score = (
            category_diversity * 0.3 +
            complexity_score * 0.4 +
            avg_confidence * 0.3
        )
        
        return min(1.0, professional_score)
    
    def _calculate_term_coherence(
        self,
        instruction_terms: List[TermAnnotation],
        thinking_terms: List[TermAnnotation],
        response_terms: List[TermAnnotation]
    ) -> float:
        """计算术语连贯性"""
        # 提取术语名称
        inst_terms = set(ann.term for ann in instruction_terms)
        think_terms = set(ann.term for ann in thinking_terms)
        resp_terms = set(ann.term for ann in response_terms)
        
        # 计算术语重叠度
        inst_think_overlap = len(inst_terms & think_terms) / len(inst_terms | think_terms) if inst_terms | think_terms else 0
        think_resp_overlap = len(think_terms & resp_terms) / len(think_terms | resp_terms) if think_terms | resp_terms else 0
        inst_resp_overlap = len(inst_terms & resp_terms) / len(inst_terms | resp_terms) if inst_terms | resp_terms else 0
        
        # 计算相关术语连贯性
        related_coherence = 0.0
        total_relations = 0
        
        for term in think_terms:
            if term in self.term_relations:
                related_terms = self.term_relations[term]
                for related_term in related_terms:
                    total_relations += 1
                    if related_term in resp_terms:
                        related_coherence += 1
        
        related_coherence_score = related_coherence / total_relations if total_relations > 0 else 0
        
        # 综合连贯性评分
        coherence_score = (
            inst_think_overlap * 0.3 +
            think_resp_overlap * 0.4 +
            inst_resp_overlap * 0.2 +
            related_coherence_score * 0.1
        )
        
        return coherence_score
    
    def enhance_thinking_with_terms(self, thinking_text: str) -> str:
        """
        增强thinking文本的术语使用
        
        Args:
            thinking_text: 原始thinking文本
            
        Returns:
            增强后的thinking文本
        """
        # 识别现有术语
        annotations = self.identify_crypto_terms(thinking_text)
        existing_terms = set(ann.term for ann in annotations)
        
        enhanced_text = thinking_text
        
        # 为每个识别的术语添加相关术语建议
        for annotation in annotations:
            term = annotation.term
            
            if term in self.term_relations:
                related_terms = self.term_relations[term]
                
                # 选择未出现的相关术语
                missing_related = related_terms - existing_terms
                
                if missing_related:
                    # 在术语附近插入相关术语的解释或提及
                    suggestion = f"（相关概念：{', '.join(list(missing_related)[:2])}）"
                    
                    # 在术语后插入建议
                    enhanced_text = enhanced_text.replace(
                        term,
                        f"{term}{suggestion}",
                        1  # 只替换第一次出现
                    )
        
        return enhanced_text
    
    def validate_term_usage(self, text: str) -> Dict[str, Any]:
        """
        验证术语使用的准确性
        
        Args:
            text: 输入文本
            
        Returns:
            验证结果
        """
        annotations = self.identify_crypto_terms(text)
        
        validation_result = {
            "total_terms": len(annotations),
            "valid_terms": 0,
            "invalid_terms": [],
            "context_issues": [],
            "suggestions": []
        }
        
        for annotation in annotations:
            # 检查术语定义的上下文一致性
            if self._validate_term_context(annotation):
                validation_result["valid_terms"] += 1
            else:
                validation_result["invalid_terms"].append({
                    "term": annotation.term,
                    "issue": "上下文使用不当",
                    "context": annotation.context
                })
        
        # 检查术语组合的合理性
        term_combinations = self._check_term_combinations(annotations)
        validation_result["context_issues"].extend(term_combinations)
        
        # 生成改进建议
        suggestions = self._generate_improvement_suggestions(annotations)
        validation_result["suggestions"].extend(suggestions)
        
        return validation_result
    
    def _validate_term_context(self, annotation: TermAnnotation) -> bool:
        """验证术语上下文"""
        term = annotation.term
        context = annotation.context.lower()
        
        # 基于术语类别检查上下文
        if annotation.category == CryptoCategory.SYMMETRIC_ENCRYPTION:
            positive_indicators = ["加密", "解密", "密钥", "算法", "安全", "对称"]
            negative_indicators = ["公钥私钥分离", "不同密钥"]
        elif annotation.category == CryptoCategory.ASYMMETRIC_ENCRYPTION:
            positive_indicators = ["公钥", "私钥", "非对称", "数字签名", "加密", "算法"]
            negative_indicators = ["相同密钥进行", "对称加密算法"]
        else:
            return True  # 其他类别暂时通过
        
        # 检查正面指标
        has_positive = any(indicator in context for indicator in positive_indicators)
        has_negative = any(indicator in context for indicator in negative_indicators)
        
        # 如果没有明显的负面指标，且有正面指标，则认为有效
        # 如果没有正面指标但也没有负面指标，给予基本信任
        return has_positive or not has_negative
    
    def _check_term_combinations(self, annotations: List[TermAnnotation]) -> List[Dict[str, Any]]:
        """检查术语组合的合理性"""
        issues = []
        
        # 检查冲突的术语组合
        conflicting_pairs = [
            (CryptoCategory.SYMMETRIC_ENCRYPTION, CryptoCategory.ASYMMETRIC_ENCRYPTION),
        ]
        
        categories = [ann.category for ann in annotations]
        
        for cat1, cat2 in conflicting_pairs:
            if cat1 in categories and cat2 in categories:
                # 这不一定是问题，但需要检查上下文
                issues.append({
                    "type": "potential_conflict",
                    "categories": [cat1.value, cat2.value],
                    "message": "同时使用了对称和非对称加密术语，请确保上下文清晰"
                })
        
        return issues
    
    def _generate_improvement_suggestions(self, annotations: List[TermAnnotation]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if not annotations:
            suggestions.append("建议添加更多密码学专业术语以提高专业性")
            return suggestions
        
        # 基于术语复杂度分布给出建议
        complexities = [ann.complexity for ann in annotations]
        avg_complexity = sum(complexities) / len(complexities)
        
        if avg_complexity < 3:
            suggestions.append("可以考虑使用更高级的密码学术语来提升内容深度")
        elif avg_complexity > 6:
            suggestions.append("术语复杂度较高，建议添加基础概念的解释")
        
        # 基于术语类别分布给出建议
        categories = set(ann.category for ann in annotations)
        
        if len(categories) < 2:
            suggestions.append("建议涵盖更多密码学领域以增加内容广度")
        
        # 基于术语密度给出建议
        if len(annotations) < 3:
            suggestions.append("专业术语密度较低，建议增加相关术语的使用")
        
        return suggestions
    
    def export_term_dictionary(self, output_path: str):
        """导出术语词典"""
        export_data = []
        
        for term, crypto_term in self.crypto_dict.items():
            if term == crypto_term.term:  # 避免重复导出别名
                export_data.append(crypto_term.to_dict())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def get_term_statistics(self) -> Dict[str, Any]:
        """获取术语词典统计信息"""
        category_counts = Counter()
        complexity_counts = Counter()
        
        unique_terms = set()
        
        for term, crypto_term in self.crypto_dict.items():
            if term == crypto_term.term:  # 只统计主术语，不包括别名
                unique_terms.add(term)
                category_counts[crypto_term.category] += 1
                complexity_counts[crypto_term.complexity] += 1
        
        return {
            "total_unique_terms": len(unique_terms),
            "total_entries": len(self.crypto_dict),  # 包括别名
            "category_distribution": {cat.value: count for cat, count in category_counts.items()},
            "complexity_distribution": dict(complexity_counts),
            "average_complexity": sum(complexity_counts.keys()) / len(complexity_counts) if complexity_counts else 0
        }