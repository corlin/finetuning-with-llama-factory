#!/usr/bin/env python3
"""
Qwen3-4B-Thinking 模型服务 API 使用示例
演示所有API端点的使用方法
"""

import asyncio
import json
import time
import requests
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenAPIClient:
    """Qwen API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        logger.info("执行健康检查...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"服务状态: {data.get('status')}")
            logger.info(f"模型已加载: {data.get('model_loaded')}")
            logger.info(f"GPU可用: {data.get('gpu_available')}")
            
            return data
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            raise
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        logger.info("获取服务统计...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"总请求数: {data.get('total_requests')}")
            logger.info(f"成功请求: {data.get('successful_requests')}")
            logger.info(f"平均响应时间: {data.get('average_response_time'):.3f}s")
            
            return data
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        logger.info("获取模型信息...")
        
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            
            if response.status_code == 503:
                logger.warning("模型未加载")
                return {"error": "模型未加载"}
            
            response.raise_for_status()
            data = response.json()
            logger.info(f"模型类型: {data.get('model_type')}")
            logger.info(f"量化格式: {data.get('quantization')}")
            
            return data
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """文本生成"""
        logger.info(f"生成文本: {prompt[:50]}...")
        
        try:
            request_data = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/generate", json=request_data)
            
            if response.status_code == 503:
                logger.warning("模型未加载，无法生成文本")
                return {"error": "模型未加载"}
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"生成完成，耗时: {time.time() - start_time:.2f}s")
            logger.info(f"生成文本长度: {len(data.get('generated_text', ''))}")
            
            return data
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise
    
    def thinking_inference(self, question: str, context: str = None, thinking_depth: int = 3) -> Dict[str, Any]:
        """深度思考推理"""
        logger.info(f"思考推理: {question[:50]}...")
        
        try:
            request_data = {
                "question": question,
                "thinking_depth": thinking_depth,
                "include_reasoning": True
            }
            
            if context:
                request_data["context"] = context
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/thinking", json=request_data)
            
            if response.status_code == 503:
                logger.warning("模型未加载，无法进行思考推理")
                return {"error": "模型未加载"}
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"思考完成，耗时: {time.time() - start_time:.2f}s")
            logger.info(f"置信度: {data.get('confidence_score', 0):.2f}")
            logger.info(f"推理步骤数: {len(data.get('reasoning_steps', []))}")
            
            return data
        except Exception as e:
            logger.error(f"思考推理失败: {e}")
            raise
    
    def batch_generate(self, prompts: List[str], max_length: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        """批量生成"""
        logger.info(f"批量生成，批次大小: {len(prompts)}")
        
        try:
            request_data = {
                "prompts": prompts,
                "max_length": max_length,
                "temperature": temperature
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/batch_generate", json=request_data)
            
            if response.status_code == 503:
                logger.warning("模型未加载，无法批量生成")
                return {"error": "模型未加载"}
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"批量生成完成，总耗时: {data.get('total_time', 0):.2f}s")
            logger.info(f"平均每个请求: {data.get('total_time', 0) / len(prompts):.2f}s")
            
            return data
        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            raise
    
    def load_model(self, model_path: str, quantization_format: str = "int8") -> Dict[str, Any]:
        """加载模型"""
        logger.info(f"加载模型: {model_path}")
        
        try:
            params = {
                "model_path": model_path,
                "quantization_format": quantization_format
            }
            
            response = self.session.post(f"{self.base_url}/load_model", params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"模型加载结果: {data.get('message')}")
            
            return data
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

def demo_basic_usage():
    """基础使用示例"""
    print("\n" + "="*50)
    print("基础使用示例")
    print("="*50)
    
    client = QwenAPIClient()
    
    # 1. 健康检查
    print("\n1. 健康检查")
    health = client.health_check()
    print(f"服务状态: {health.get('status')}")
    
    # 2. 获取统计信息
    print("\n2. 服务统计")
    stats = client.get_service_stats()
    print(f"运行时间: {stats.get('uptime_hours', 0):.1f}小时")
    
    # 3. 获取模型信息
    print("\n3. 模型信息")
    model_info = client.get_model_info()
    if "error" not in model_info:
        print(f"模型: {model_info.get('model_type')}")
    else:
        print("模型未加载")

def demo_text_generation():
    """文本生成示例"""
    print("\n" + "="*50)
    print("文本生成示例")
    print("="*50)
    
    client = QwenAPIClient()
    
    # 密码学相关问题
    crypto_questions = [
        "什么是AES加密算法？请详细解释其工作原理。",
        "RSA和ECC加密算法有什么区别？",
        "解释一下哈希函数的特性和应用场景。",
        "什么是数字签名？它如何保证数据完整性？"
    ]
    
    for i, question in enumerate(crypto_questions, 1):
        print(f"\n{i}. 问题: {question}")
        
        try:
            result = client.generate_text(
                prompt=question,
                max_length=300,
                temperature=0.7
            )
            
            if "error" not in result:
                print(f"回答: {result['generated_text'][:200]}...")
                print(f"生成时间: {result['generation_time']:.2f}s")
                print(f"Token数量: {result['token_count']}")
            else:
                print(f"生成失败: {result['error']}")
                
        except Exception as e:
            print(f"请求失败: {e}")

def demo_thinking_inference():
    """深度思考推理示例"""
    print("\n" + "="*50)
    print("深度思考推理示例")
    print("="*50)
    
    client = QwenAPIClient()
    
    # 复杂的密码学问题
    complex_questions = [
        {
            "question": "如何设计一个安全的密码存储系统？",
            "context": "考虑到彩虹表攻击、暴力破解和内部威胁等安全风险"
        },
        {
            "question": "量子计算对现有加密算法的威胁是什么？",
            "context": "分析RSA、ECC等算法在量子计算环境下的安全性"
        },
        {
            "question": "零知识证明在区块链中的应用原理是什么？",
            "context": "解释zk-SNARKs和zk-STARKs的技术特点和应用场景"
        }
    ]
    
    for i, item in enumerate(complex_questions, 1):
        print(f"\n{i}. 问题: {item['question']}")
        print(f"   上下文: {item['context']}")
        
        try:
            result = client.thinking_inference(
                question=item['question'],
                context=item['context'],
                thinking_depth=3
            )
            
            if "error" not in result:
                print(f"\n思考过程:")
                print(result['thinking_process'][:300] + "..." if len(result['thinking_process']) > 300 else result['thinking_process'])
                
                print(f"\n最终答案:")
                print(result['final_answer'][:300] + "..." if len(result['final_answer']) > 300 else result['final_answer'])
                
                print(f"\n推理步骤:")
                for j, step in enumerate(result['reasoning_steps'][:3], 1):
                    print(f"  {j}. {step}")
                
                print(f"\n置信度: {result['confidence_score']:.2f}")
                print(f"处理时间: {result['processing_time']:.2f}s")
            else:
                print(f"推理失败: {result['error']}")
                
        except Exception as e:
            print(f"请求失败: {e}")

def demo_batch_generation():
    """批量生成示例"""
    print("\n" + "="*50)
    print("批量生成示例")
    print("="*50)
    
    client = QwenAPIClient()
    
    # 批量密码学概念解释
    batch_prompts = [
        "简要解释对称加密的特点",
        "简要解释非对称加密的特点", 
        "简要解释哈希函数的作用",
        "简要解释数字证书的用途",
        "简要解释密钥交换协议"
    ]
    
    print(f"批量生成 {len(batch_prompts)} 个问题的答案...")
    
    try:
        result = client.batch_generate(
            prompts=batch_prompts,
            max_length=100,
            temperature=0.7
        )
        
        if "error" not in result:
            print(f"\n批量生成完成:")
            print(f"总耗时: {result['total_time']:.2f}s")
            print(f"批次大小: {result['batch_size']}")
            print(f"平均每个: {result['total_time'] / result['batch_size']:.2f}s")
            
            print(f"\n结果预览:")
            for i, res in enumerate(result['results'][:3], 1):
                print(f"{i}. {res['prompt']}")
                print(f"   答案: {res['generated_text'][:100]}...")
                print(f"   耗时: {res['generation_time']:.2f}s")
                print()
        else:
            print(f"批量生成失败: {result['error']}")
            
    except Exception as e:
        print(f"请求失败: {e}")

def demo_performance_test():
    """性能测试示例"""
    print("\n" + "="*50)
    print("性能测试示例")
    print("="*50)
    
    client = QwenAPIClient()
    
    # 测试响应时间
    print("1. 响应时间测试")
    response_times = []
    
    for i in range(5):
        start_time = time.time()
        try:
            client.health_check()
            response_time = time.time() - start_time
            response_times.append(response_time)
            print(f"   请求 {i+1}: {response_time:.3f}s")
        except Exception as e:
            print(f"   请求 {i+1} 失败: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"   平均响应时间: {avg_time:.3f}s")
        print(f"   最大响应时间: {max(response_times):.3f}s")
        print(f"   最小响应时间: {min(response_times):.3f}s")
    
    # 测试并发请求
    print("\n2. 并发请求测试")
    import concurrent.futures
    
    def make_health_request():
        try:
            start_time = time.time()
            client.health_check()
            return time.time() - start_time
        except:
            return None
    
    concurrent_times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_health_request) for _ in range(10)]
        concurrent_times = [f.result() for f in futures if f.result() is not None]
    
    if concurrent_times:
        print(f"   并发请求数: {len(concurrent_times)}")
        print(f"   平均响应时间: {sum(concurrent_times) / len(concurrent_times):.3f}s")
        print(f"   成功率: {len(concurrent_times) / 10 * 100:.1f}%")

def demo_error_handling():
    """错误处理示例"""
    print("\n" + "="*50)
    print("错误处理示例")
    print("="*50)
    
    client = QwenAPIClient()
    
    # 测试各种错误情况
    error_tests = [
        ("空提示", lambda: client.generate_text("")),
        ("过长提示", lambda: client.generate_text("a" * 5000)),
        ("无效温度", lambda: client.generate_text("测试", temperature=3.0)),
        ("无效思考深度", lambda: client.thinking_inference("测试", thinking_depth=10)),
        ("空批次", lambda: client.batch_generate([])),
    ]
    
    for test_name, test_func in error_tests:
        print(f"\n测试: {test_name}")
        try:
            result = test_func()
            if "error" in result:
                print(f"   预期错误: {result['error']}")
            else:
                print(f"   意外成功: {result}")
        except requests.exceptions.HTTPError as e:
            print(f"   HTTP错误 (预期): {e.response.status_code}")
        except Exception as e:
            print(f"   其他错误: {e}")

def main():
    """主函数"""
    print("Qwen3-4B-Thinking 模型服务 API 使用示例")
    print("="*60)
    
    # 运行所有示例
    try:
        demo_basic_usage()
        demo_text_generation()
        demo_thinking_inference()
        demo_batch_generation()
        demo_performance_test()
        demo_error_handling()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行异常: {e}")

if __name__ == "__main__":
    main()