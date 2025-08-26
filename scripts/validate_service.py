#!/usr/bin/env python3
"""
模型服务验证脚本
使用uv验证服务接口功能完整性和API响应正确性
"""

import asyncio
import json
import time
import sys
import subprocess
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果"""
    test_name: str
    success: bool
    message: str
    duration: float
    details: Optional[Dict[str, Any]] = None

class ServiceValidator:
    """服务验证器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[ValidationResult] = []
        self.session = requests.Session()
        self.session.timeout = 30
    
    def run_validation(self) -> bool:
        """运行完整验证"""
        logger.info("开始服务验证...")
        
        # 验证测试列表
        validations = [
            ("环境检查", self.validate_environment),
            ("服务连通性", self.validate_connectivity),
            ("健康检查端点", self.validate_health_endpoint),
            ("统计端点", self.validate_stats_endpoint),
            ("模型信息端点", self.validate_model_info_endpoint),
            ("生成端点验证", self.validate_generate_endpoint),
            ("思考推理端点验证", self.validate_thinking_endpoint),
            ("批量生成端点验证", self.validate_batch_generate_endpoint),
            ("输入验证测试", self.validate_input_validation),
            ("错误处理测试", self.validate_error_handling),
            ("并发测试", self.validate_concurrent_requests),
            ("性能测试", self.validate_performance),
            ("API响应格式", self.validate_response_formats)
        ]
        
        # 执行验证
        for test_name, validation_func in validations:
            try:
                logger.info(f"执行验证: {test_name}")
                start_time = time.time()
                
                success, message, details = validation_func()
                duration = time.time() - start_time
                
                result = ValidationResult(
                    test_name=test_name,
                    success=success,
                    message=message,
                    duration=duration,
                    details=details
                )
                
                self.results.append(result)
                
                status = "✓" if success else "✗"
                logger.info(f"{status} {test_name}: {message} ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                result = ValidationResult(
                    test_name=test_name,
                    success=False,
                    message=f"验证异常: {str(e)}",
                    duration=duration
                )
                self.results.append(result)
                logger.error(f"✗ {test_name}: 验证异常 - {str(e)}")
        
        # 生成报告
        self.generate_report()
        
        # 返回总体结果
        return all(result.success for result in self.results)
    
    def validate_environment(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证环境"""
        details = {}
        
        try:
            # 检查uv
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                details['uv_version'] = result.stdout.strip()
            else:
                return False, "uv未安装或不可用", details
            
            # 检查Python版本
            import sys
            details['python_version'] = sys.version
            
            # 检查依赖包
            try:
                import torch
                details['torch_version'] = torch.__version__
                details['cuda_available'] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    details['gpu_count'] = torch.cuda.device_count()
                    details['gpu_name'] = torch.cuda.get_device_name(0)
            except ImportError:
                details['torch_error'] = "PyTorch未安装"
            
            try:
                import fastapi
                details['fastapi_version'] = fastapi.__version__
            except ImportError:
                details['fastapi_error'] = "FastAPI未安装"
            
            return True, "环境检查通过", details
            
        except Exception as e:
            return False, f"环境检查失败: {str(e)}", details
    
    def validate_connectivity(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证服务连通性"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            
            details = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'headers': dict(response.headers)
            }
            
            if response.status_code == 200:
                return True, "服务连通正常", details
            else:
                return False, f"服务返回状态码: {response.status_code}", details
                
        except requests.exceptions.ConnectionError:
            return False, "无法连接到服务", {}
        except requests.exceptions.Timeout:
            return False, "连接超时", {}
        except Exception as e:
            return False, f"连接异常: {str(e)}", {}
    
    def validate_health_endpoint(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证健康检查端点"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                return False, f"状态码错误: {response.status_code}", {}
            
            data = response.json()
            
            # 检查必需字段
            required_fields = ['status', 'timestamp', 'model_loaded', 'gpu_available', 'memory_usage', 'uptime_seconds']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return False, f"缺少字段: {missing_fields}", data
            
            # 验证数据类型
            if not isinstance(data['model_loaded'], bool):
                return False, "model_loaded字段类型错误", data
            
            if not isinstance(data['gpu_available'], bool):
                return False, "gpu_available字段类型错误", data
            
            if not isinstance(data['memory_usage'], dict):
                return False, "memory_usage字段类型错误", data
            
            return True, "健康检查端点正常", data
            
        except Exception as e:
            return False, f"健康检查失败: {str(e)}", {}
    
    def validate_stats_endpoint(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证统计端点"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            
            if response.status_code != 200:
                return False, f"状态码错误: {response.status_code}", {}
            
            data = response.json()
            
            # 检查必需字段
            required_fields = ['total_requests', 'successful_requests', 'failed_requests', 'average_response_time', 'uptime_hours']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return False, f"缺少字段: {missing_fields}", data
            
            # 验证数据类型和逻辑
            if not isinstance(data['total_requests'], int) or data['total_requests'] < 0:
                return False, "total_requests字段无效", data
            
            if data['successful_requests'] + data['failed_requests'] > data['total_requests']:
                return False, "请求统计逻辑错误", data
            
            return True, "统计端点正常", data
            
        except Exception as e:
            return False, f"统计端点失败: {str(e)}", {}
    
    def validate_model_info_endpoint(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证模型信息端点"""
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            
            # 模型未加载时应该返回503
            if response.status_code == 503:
                data = response.json()
                if "模型未加载" in data.get("detail", ""):
                    return True, "模型信息端点正常（模型未加载）", data
                else:
                    return False, "503错误信息不正确", data
            
            # 模型已加载时应该返回200
            elif response.status_code == 200:
                data = response.json()
                return True, "模型信息端点正常（模型已加载）", data
            
            else:
                return False, f"意外状态码: {response.status_code}", {}
                
        except Exception as e:
            return False, f"模型信息端点失败: {str(e)}", {}
    
    def validate_generate_endpoint(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证生成端点"""
        try:
            # 测试基本生成请求
            request_data = {
                "prompt": "什么是AES加密算法？",
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
            
            response = self.session.post(f"{self.base_url}/generate", json=request_data)
            
            # 模型未加载时应该返回503
            if response.status_code == 503:
                data = response.json()
                if "模型未加载" in data.get("detail", ""):
                    return True, "生成端点正常（模型未加载）", data
                else:
                    return False, "503错误信息不正确", data
            
            # 模型已加载时应该返回200
            elif response.status_code == 200:
                data = response.json()
                
                # 检查响应格式
                required_fields = ['generated_text', 'prompt', 'generation_time', 'token_count', 'model_info']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return False, f"响应缺少字段: {missing_fields}", data
                
                # 验证数据类型
                if not isinstance(data['generated_text'], str):
                    return False, "generated_text字段类型错误", data
                
                if not isinstance(data['generation_time'], (int, float)) or data['generation_time'] < 0:
                    return False, "generation_time字段无效", data
                
                return True, "生成端点正常（模型已加载）", data
            
            else:
                return False, f"意外状态码: {response.status_code}", {}
                
        except Exception as e:
            return False, f"生成端点失败: {str(e)}", {}
    
    def validate_thinking_endpoint(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证思考推理端点"""
        try:
            request_data = {
                "question": "RSA和AES加密算法有什么区别？",
                "context": "这是关于密码学的问题",
                "thinking_depth": 3,
                "include_reasoning": True
            }
            
            response = self.session.post(f"{self.base_url}/thinking", json=request_data)
            
            # 模型未加载时应该返回503
            if response.status_code == 503:
                data = response.json()
                if "模型未加载" in data.get("detail", ""):
                    return True, "思考推理端点正常（模型未加载）", data
                else:
                    return False, "503错误信息不正确", data
            
            # 模型已加载时应该返回200
            elif response.status_code == 200:
                data = response.json()
                
                # 检查响应格式
                required_fields = ['question', 'thinking_process', 'final_answer', 'reasoning_steps', 'confidence_score', 'processing_time']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return False, f"响应缺少字段: {missing_fields}", data
                
                # 验证数据类型
                if not isinstance(data['thinking_process'], str):
                    return False, "thinking_process字段类型错误", data
                
                if not isinstance(data['reasoning_steps'], list):
                    return False, "reasoning_steps字段类型错误", data
                
                if not isinstance(data['confidence_score'], (int, float)) or not (0 <= data['confidence_score'] <= 1):
                    return False, "confidence_score字段无效", data
                
                return True, "思考推理端点正常（模型已加载）", data
            
            else:
                return False, f"意外状态码: {response.status_code}", {}
                
        except Exception as e:
            return False, f"思考推理端点失败: {str(e)}", {}
    
    def validate_batch_generate_endpoint(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证批量生成端点"""
        try:
            request_data = {
                "prompts": [
                    "什么是对称加密？",
                    "什么是非对称加密？",
                    "什么是哈希函数？"
                ],
                "max_length": 50,
                "temperature": 0.7
            }
            
            response = self.session.post(f"{self.base_url}/batch_generate", json=request_data)
            
            # 模型未加载时应该返回503
            if response.status_code == 503:
                data = response.json()
                if "模型未加载" in data.get("detail", ""):
                    return True, "批量生成端点正常（模型未加载）", data
                else:
                    return False, "503错误信息不正确", data
            
            # 模型已加载时应该返回200
            elif response.status_code == 200:
                data = response.json()
                
                # 检查响应格式
                required_fields = ['results', 'total_time', 'batch_size']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return False, f"响应缺少字段: {missing_fields}", data
                
                # 验证数据类型
                if not isinstance(data['results'], list):
                    return False, "results字段类型错误", data
                
                if data['batch_size'] != len(request_data['prompts']):
                    return False, "batch_size与输入不匹配", data
                
                return True, "批量生成端点正常（模型已加载）", data
            
            else:
                return False, f"意外状态码: {response.status_code}", {}
                
        except Exception as e:
            return False, f"批量生成端点失败: {str(e)}", {}
    
    def validate_input_validation(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证输入验证功能"""
        test_cases = []
        
        # 测试空提示
        response = self.session.post(f"{self.base_url}/generate", json={"prompt": ""})
        test_cases.append(("空提示", response.status_code == 422))
        
        # 测试过长提示
        long_prompt = "a" * 5000
        response = self.session.post(f"{self.base_url}/generate", json={"prompt": long_prompt})
        test_cases.append(("过长提示", response.status_code == 422))
        
        # 测试无效温度
        response = self.session.post(f"{self.base_url}/generate", json={
            "prompt": "测试",
            "temperature": 3.0
        })
        test_cases.append(("无效温度", response.status_code == 422))
        
        # 测试无效思考深度
        response = self.session.post(f"{self.base_url}/thinking", json={
            "question": "测试",
            "thinking_depth": 10
        })
        test_cases.append(("无效思考深度", response.status_code == 422))
        
        # 测试空批次
        response = self.session.post(f"{self.base_url}/batch_generate", json={"prompts": []})
        test_cases.append(("空批次", response.status_code == 422))
        
        # 统计结果
        passed = sum(1 for _, success in test_cases if success)
        total = len(test_cases)
        
        details = {
            "test_cases": test_cases,
            "passed": passed,
            "total": total
        }
        
        if passed == total:
            return True, f"输入验证测试通过 ({passed}/{total})", details
        else:
            return False, f"输入验证测试失败 ({passed}/{total})", details
    
    def validate_error_handling(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证错误处理"""
        test_cases = []
        
        # 测试无效JSON
        try:
            response = self.session.post(f"{self.base_url}/generate", data="invalid json")
            test_cases.append(("无效JSON", response.status_code == 422))
        except:
            test_cases.append(("无效JSON", False))
        
        # 测试缺少必需字段
        response = self.session.post(f"{self.base_url}/generate", json={})
        test_cases.append(("缺少必需字段", response.status_code == 422))
        
        # 测试无效端点
        response = self.session.get(f"{self.base_url}/invalid_endpoint")
        test_cases.append(("无效端点", response.status_code == 404))
        
        # 测试无效HTTP方法
        response = self.session.delete(f"{self.base_url}/generate")
        test_cases.append(("无效HTTP方法", response.status_code == 405))
        
        # 统计结果
        passed = sum(1 for _, success in test_cases if success)
        total = len(test_cases)
        
        details = {
            "test_cases": test_cases,
            "passed": passed,
            "total": total
        }
        
        if passed == total:
            return True, f"错误处理测试通过 ({passed}/{total})", details
        else:
            return False, f"错误处理测试失败 ({passed}/{total})", details
    
    def validate_concurrent_requests(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证并发请求处理"""
        try:
            def make_request():
                return self.session.get(f"{self.base_url}/health")
            
            # 并发发送10个健康检查请求
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                responses = [future.result() for future in futures]
            
            # 检查所有响应
            success_count = sum(1 for r in responses if r.status_code == 200)
            total_count = len(responses)
            
            details = {
                "total_requests": total_count,
                "successful_requests": success_count,
                "success_rate": success_count / total_count
            }
            
            if success_count == total_count:
                return True, f"并发测试通过 ({success_count}/{total_count})", details
            else:
                return False, f"并发测试失败 ({success_count}/{total_count})", details
                
        except Exception as e:
            return False, f"并发测试异常: {str(e)}", {}
    
    def validate_performance(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证性能"""
        try:
            # 测试响应时间
            response_times = []
            for _ in range(5):
                start_time = time.time()
                response = self.session.get(f"{self.base_url}/health")
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
            
            if not response_times:
                return False, "无法获取响应时间", {}
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            details = {
                "average_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "response_times": response_times
            }
            
            # 响应时间应该小于1秒
            if avg_response_time < 1.0:
                return True, f"性能测试通过 (平均: {avg_response_time:.3f}s)", details
            else:
                return False, f"响应时间过长 (平均: {avg_response_time:.3f}s)", details
                
        except Exception as e:
            return False, f"性能测试异常: {str(e)}", {}
    
    def validate_response_formats(self) -> tuple[bool, str, Dict[str, Any]]:
        """验证API响应格式"""
        test_cases = []
        
        # 测试健康检查响应格式
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                # 检查是否为有效JSON且包含必需字段
                required_fields = ['status', 'timestamp', 'model_loaded']
                has_required = all(field in data for field in required_fields)
                test_cases.append(("健康检查响应格式", has_required))
            else:
                test_cases.append(("健康检查响应格式", False))
        except:
            test_cases.append(("健康检查响应格式", False))
        
        # 测试统计响应格式
        try:
            response = self.session.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                required_fields = ['total_requests', 'successful_requests']
                has_required = all(field in data for field in required_fields)
                test_cases.append(("统计响应格式", has_required))
            else:
                test_cases.append(("统计响应格式", False))
        except:
            test_cases.append(("统计响应格式", False))
        
        # 统计结果
        passed = sum(1 for _, success in test_cases if success)
        total = len(test_cases)
        
        details = {
            "test_cases": test_cases,
            "passed": passed,
            "total": total
        }
        
        if passed == total:
            return True, f"响应格式测试通过 ({passed}/{total})", details
        else:
            return False, f"响应格式测试失败 ({passed}/{total})", details
    
    def generate_report(self):
        """生成验证报告"""
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        failed_tests = total_tests - passed_tests
        
        total_time = sum(result.duration for result in self.results)
        
        report = f"""
# 服务验证报告

## 总体结果
- 总测试数: {total_tests}
- 通过测试: {passed_tests}
- 失败测试: {failed_tests}
- 成功率: {passed_tests/total_tests*100:.1f}%
- 总耗时: {total_time:.2f}秒

## 详细结果
"""
        
        for result in self.results:
            status = "✓" if result.success else "✗"
            report += f"- {status} {result.test_name}: {result.message} ({result.duration:.2f}s)\n"
        
        if failed_tests > 0:
            report += "\n## 失败测试详情\n"
            for result in self.results:
                if not result.success:
                    report += f"\n### {result.test_name}\n"
                    report += f"错误: {result.message}\n"
                    if result.details:
                        report += f"详情: {json.dumps(result.details, indent=2, ensure_ascii=False)}\n"
        
        # 保存报告
        timestamp = int(time.time())
        report_file = f"service_validation_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"验证报告已保存: {report_file}")
        
        # 打印摘要
        print("\n" + "="*50)
        print("服务验证完成")
        print(f"通过: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"报告: {report_file}")
        print("="*50)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking 服务验证")
    parser.add_argument("--url", default="http://localhost:8000", help="服务URL")
    parser.add_argument("--uv", action="store_true", help="使用uv运行验证")
    
    args = parser.parse_args()
    
    if args.uv:
        logger.info("使用uv运行验证...")
        # 确保在uv环境中运行
        try:
            result = subprocess.run(['uv', 'run', 'python', __file__, '--url', args.url], 
                                  capture_output=False)
            sys.exit(result.returncode)
        except FileNotFoundError:
            logger.error("uv未安装，使用标准Python运行")
    
    # 创建验证器并运行
    validator = ServiceValidator(args.url)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()