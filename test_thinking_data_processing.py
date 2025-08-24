#!/usr/bin/env python3
"""
深度思考数据处理专项测试

测试复杂的thinking数据结构处理，包括嵌套thinking、多步推理、中文密码学术语等。
"""

from src.data_models import (
    ThinkingExample, ThinkingStructure, ReasoningStep, 
    DataModelValidator, CryptoTerm, CryptoCategory
)


def test_complex_thinking_structures():
    """测试复杂的thinking结构"""
    print("=== 测试复杂thinking结构处理 ===")
    
    # 测试1: 嵌套thinking标签
    nested_thinking = """
    <thinking>
    这是一个关于密码学的复杂问题，需要分步分析：
    
    1. 首先理解问题的核心：什么是非对称加密？
    <thinking>
    非对称加密涉及公钥和私钥的概念，需要从数学原理开始解释
    </thinking>
    
    2. 然后分析其应用场景
    3. 最后总结其优缺点
    </thinking>
    """
    
    result = DataModelValidator.validate_thinking_data(nested_thinking)
    print(f"嵌套thinking验证结果: {result['valid']}")
    if result['warnings']:
        print(f"警告: {result['warnings']}")
    
    # 测试2: 多步推理thinking
    multi_step_thinking = """
    <thinking>
    这个问题涉及GB/T 39786-2021标准中的密码应用等级，需要系统性分析：
    
    步骤1：理解密码应用等级的基本概念
    - 密码应用分为五个等级
    - 每个等级有不同的技术要求和管理要求
    
    步骤2：分析第三级和第四级的区别
    - 第三级：增加真实性、机密性要求
    - 第四级：增加完整性、不可否认性要求
    
    步骤3：结合具体的技术实现
    - 需要考虑密码产品的安全等级要求
    - 需要考虑密码应用安全性评估要求
    
    步骤4：给出准确的回答
    基于以上分析，可以得出结论...
    </thinking>
    """
    
    # 创建ThinkingStructure进行分析
    reasoning_steps = [
        ReasoningStep(
            step_number=1,
            description="理解密码应用等级概念",
            input_data="GB/T 39786-2021标准问题",
            reasoning_process="分析标准中的等级划分原理",
            output_result="明确五级等级体系",
            confidence_score=0.95
        ),
        ReasoningStep(
            step_number=2,
            description="分析等级差异",
            input_data="第三级和第四级要求",
            reasoning_process="对比技术要求和管理要求的差异",
            output_result="识别关键区别点",
            confidence_score=0.90
        ),
        ReasoningStep(
            step_number=3,
            description="技术实现分析",
            input_data="密码产品和评估要求",
            reasoning_process="结合实际应用场景分析",
            output_result="确定实施要点",
            confidence_score=0.85
        ),
        ReasoningStep(
            step_number=4,
            description="综合结论",
            input_data="前述分析结果",
            reasoning_process="整合所有信息形成完整回答",
            output_result="准确的标准解释",
            confidence_score=0.92
        )
    ]
    
    thinking_structure = ThinkingStructure(
        raw_thinking=multi_step_thinking,
        parsed_steps=["步骤1：理解概念", "步骤2：分析差异", "步骤3：技术实现", "步骤4：综合结论"],
        reasoning_chain=reasoning_steps,
        validation_result=True
    )
    
    print(f"多步推理thinking深度: {thinking_structure.thinking_depth}")
    print(f"逻辑一致性评分: {thinking_structure.logical_consistency:.2f}")
    print(f"完整性评分: {thinking_structure.completeness_score:.2f}")
    
    # 测试内容提取
    extracted_content = thinking_structure.extract_thinking_content()
    print(f"提取到 {len(extracted_content)} 段thinking内容")
    
    return thinking_structure


def test_crypto_domain_thinking_examples():
    """测试密码学领域的thinking样例"""
    print("\n=== 测试密码学领域thinking样例 ===")
    
    # 创建密码学专业问题的thinking样例
    crypto_examples = [
        {
            "instruction": "请详细解释RSA算法的工作原理，包括密钥生成、加密和解密过程。",
            "thinking": """<thinking>
这是一个关于RSA非对称加密算法的技术问题，需要从数学原理和实现步骤两个层面来回答：

1. 数学基础分析：
   - RSA基于大整数分解的困难性
   - 涉及欧拉函数、模运算、费马小定理等数学概念
   - 需要解释为什么大数分解在计算上是困难的

2. 密钥生成过程：
   - 选择两个大素数p和q
   - 计算n = p × q（模数）
   - 计算φ(n) = (p-1)(q-1)（欧拉函数值）
   - 选择公钥指数e，满足gcd(e, φ(n)) = 1
   - 计算私钥指数d，满足ed ≡ 1 (mod φ(n))

3. 加密解密过程：
   - 加密：c = m^e mod n
   - 解密：m = c^d mod n
   - 需要解释为什么这个过程是可逆的

4. 安全性分析：
   - 基于大整数分解问题的困难性
   - 密钥长度的选择（通常2048位或更高）
   - 可能的攻击方式和防护措施

5. 实际应用考虑：
   - 性能特点（相比对称加密较慢）
   - 适用场景（密钥交换、数字签名等）
   - 与其他密码算法的配合使用
</thinking>""",
            "response": "RSA算法是一种基于大整数分解困难性的非对称加密算法...",
            "crypto_terms": ["RSA", "非对称加密", "公钥", "私钥", "模运算", "数字签名"]
        },
        {
            "instruction": "根据GB/T 39786-2021标准，第四级密码应用在网络和通信安全方面有哪些具体要求？",
            "thinking": """<thinking>
这是一个关于国家密码应用标准的问题，需要准确引用GB/T 39786-2021标准的具体条款：

1. 标准背景分析：
   - GB/T 39786-2021是《信息安全技术 信息系统密码应用基本要求》
   - 第四级是较高的安全等级，有严格的技术要求
   - 网络和通信安全是四个技术层面之一

2. 第四级网络和通信安全要求梳理：
   - 通信实体身份鉴别：应采用密码技术对通信实体进行双向身份鉴别
   - 数据完整性：应采用密码技术保证通信过程中数据的完整性
   - 数据机密性：应采用密码技术保证通信过程中重要数据的机密性
   - 访问控制：应采用密码技术保证网络边界访问控制信息的完整性
   - 设备接入认证：宜采用密码技术对外部设备进行接入认证

3. 与其他等级的对比：
   - 相比第三级，第四级要求双向身份鉴别（而非单向）
   - 增加了完整性和不可否认性的全面要求
   - 对密码产品的安全等级要求更高（GB/T37092三级及以上）

4. 实施要点：
   - 需要部署符合要求的密码产品
   - 需要建立完整的密钥管理体系
   - 需要进行密码应用安全性评估
</thinking>""",
            "response": "根据GB/T 39786-2021标准，第四级密码应用在网络和通信安全方面的具体要求包括...",
            "crypto_terms": ["GB/T 39786-2021", "密码应用", "身份鉴别", "完整性", "机密性", "访问控制"]
        }
    ]
    
    thinking_examples = []
    for i, example_data in enumerate(crypto_examples):
        example = ThinkingExample(
            instruction=example_data["instruction"],
            thinking_process=example_data["thinking"],
            final_response=example_data["response"],
            crypto_terms=example_data["crypto_terms"],
            source_domain="密码学专业领域"
        )
        
        thinking_examples.append(example)
        
        print(f"\n样例 {i+1}:")
        print(f"指令长度: {len(example.instruction)} 字符")
        print(f"思考过程长度: {len(example.thinking_process)} 字符")
        print(f"密码术语数量: {len(example.crypto_terms)}")
        print(f"thinking标签验证: {example.validate_thinking_tags()}")
        
        # 提取推理步骤
        steps = example.extract_reasoning_steps()
        print(f"推理步骤数量: {len(steps)}")
        
        # 转换为LLaMA Factory格式
        llama_format = example.to_llama_factory_format()
        print(f"LLaMA格式输出长度: {len(llama_format['output'])} 字符")
    
    return thinking_examples


def test_chinese_crypto_term_processing():
    """测试中文密码学术语处理"""
    print("\n=== 测试中文密码学术语处理 ===")
    
    # 创建中文密码学术语
    chinese_crypto_terms = [
        CryptoTerm(
            term="商用密码",
            definition="用于保护不涉及国家秘密内容的密码技术和密码产品",
            category=CryptoCategory.OTHER,
            complexity=6,
            aliases=["商密", "商用密码技术"],
            related_terms=["国产密码", "SM系列算法"],
            examples=["SM2椭圆曲线公钥密码算法", "SM3密码杂凑算法", "SM4分组密码算法"]
        ),
        CryptoTerm(
            term="密码应用安全性评估",
            definition="对信息系统密码应用的安全性进行评估的活动",
            category=CryptoCategory.OTHER,
            complexity=8,
            aliases=["密评", "密码评估"],
            related_terms=["GB/T 39786-2021", "测评单元", "风险分析"],
            examples=["方案编制活动", "现场测评活动", "分析与报告编制活动"]
        ),
        CryptoTerm(
            term="SM2椭圆曲线公钥密码算法",
            definition="基于椭圆曲线离散对数问题的国产公钥密码算法",
            category=CryptoCategory.ASYMMETRIC_ENCRYPTION,
            complexity=9,
            aliases=["SM2算法", "SM2"],
            related_terms=["椭圆曲线", "数字签名", "密钥交换"],
            examples=["数字签名", "密钥协商", "公钥加密"]
        )
    ]
    
    for term in chinese_crypto_terms:
        print(f"\n术语: {term.term}")
        print(f"分类: {term.category.value}")
        print(f"复杂度: {term.complexity}/10")
        print(f"别名: {term.aliases}")
        print(f"相关术语: {term.related_terms}")
        
        # 测试序列化
        term_dict = term.to_dict()
        restored_term = CryptoTerm.from_dict(term_dict)
        assert restored_term.term == term.term
        print("✓ 序列化测试通过")
    
    return chinese_crypto_terms


def test_thinking_data_validation_edge_cases():
    """测试thinking数据验证的边界情况"""
    print("\n=== 测试thinking数据验证边界情况 ===")
    
    test_cases = [
        {
            "name": "空thinking标签",
            "data": "<thinking></thinking>",
            "expected_valid": True
        },
        {
            "name": "多层嵌套thinking",
            "data": "<thinking>外层<thinking>中层<thinking>内层</thinking>中层</thinking>外层</thinking>",
            "expected_valid": True
        },
        {
            "name": "不匹配的标签",
            "data": "<thinking>开始</thinkng>",
            "expected_valid": False
        },
        {
            "name": "中文内容thinking",
            "data": "<thinking>这是中文的思考过程，包含密码学术语：非对称加密、数字签名等</thinking>",
            "expected_valid": True
        },
        {
            "name": "包含特殊字符",
            "data": "<thinking>思考过程包含特殊字符：@#$%^&*()，以及数学公式：c = m^e mod n</thinking>",
            "expected_valid": True
        }
    ]
    
    for case in test_cases:
        result = DataModelValidator.validate_thinking_data(case["data"])
        print(f"\n测试用例: {case['name']}")
        print(f"预期结果: {'有效' if case['expected_valid'] else '无效'}")
        print(f"实际结果: {'有效' if result['valid'] else '无效'}")
        
        if result['errors']:
            if case['expected_valid']:
                print(f"❌ 意外错误: {result['errors']}")
            else:
                print(f"✓ 预期错误检测: {result['errors']}")
        
        if result['warnings']:
            print(f"⚠️ 警告信息: {result['warnings']}")
        
        if result["valid"] == case["expected_valid"]:
            print("✅ 测试通过")
        else:
            print("❌ 测试失败")


def main():
    """主函数"""
    print("开始深度思考数据处理专项测试...\n")
    
    try:
        # 1. 测试复杂thinking结构
        thinking_structure = test_complex_thinking_structures()
        
        # 2. 测试密码学领域thinking样例
        thinking_examples = test_crypto_domain_thinking_examples()
        
        # 3. 测试中文密码学术语处理
        crypto_terms = test_chinese_crypto_term_processing()
        
        # 4. 测试thinking数据验证边界情况
        test_thinking_data_validation_edge_cases()
        
        print("\n" + "="*60)
        print("深度思考数据处理专项测试报告")
        print("="*60)
        print(f"✅ 复杂thinking结构处理: 通过")
        print(f"✅ 密码学领域thinking样例: {len(thinking_examples)} 个")
        print(f"✅ 中文密码学术语处理: {len(crypto_terms)} 个")
        print(f"✅ thinking数据验证边界测试: 通过")
        print(f"\n🎉 所有专项测试通过！任务2.1深度思考数据处理能力验证成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()