"""
专家评估系统 API 使用示例

提供完整的API使用示例，包括同步和异步评估、批量处理等。
"""

import asyncio
import json
import time
from typing import List, Dict, Any
import requests
from datetime import datetime

# API基础URL
BASE_URL = "http://localhost:8000"

class ExpertEvaluationAPIClient:
    """专家评估API客户端"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """加载模型"""
        data = {"model_path": model_path}
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/load_model", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        response = self.session.get(f"{self.base_url}/model_info")
        response.raise_for_status()
        return response.json()
    
    def evaluate_sync(self, qa_items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步评估"""
        data = {
            "qa_items": qa_items,
            "async_mode": False
        }
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/evaluate", json=data)
        response.raise_for_status()
        return response.json()
    
    def evaluate_async(self, qa_items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """异步评估，返回任务ID"""
        data = {
            "qa_items": qa_items,
            "async_mode": True
        }
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/evaluate", json=data)
        response.raise_for_status()
        return response.json()["task_id"]
    
    def batch_evaluate_async(self, datasets: List[List[Dict[str, Any]]], config: Dict[str, Any] = None) -> str:
        """批量异步评估，返回任务ID"""
        data = {
            "datasets": datasets,
            "async_mode": True
        }
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/batch_evaluate", json=data)
        response.raise_for_status()
        return response.json()["task_id"]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        response = self.session.get(f"{self.base_url}/task/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            print(f"任务 {task_id} 进度: {status['progress']:.1%}")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"任务 {task_id} 在 {timeout} 秒内未完成")
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消任务"""
        response = self.session.delete(f"{self.base_url}/task/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def list_tasks(self) -> Dict[str, Any]:
        """列出所有任务"""
        response = self.session.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()
    
    def generate_report(self, task_id: str, format: str = "json") -> Dict[str, Any]:
        """生成评估报告"""
        response = self.session.post(f"{self.base_url}/generate_report/{task_id}?format={format}")
        response.raise_for_status()
        return response.json()

def create_sample_qa_items() -> List[Dict[str, Any]]:
    """创建示例QA数据"""
    return [
        {
            "question_id": "crypto_001",
            "question": "什么是AES加密算法？请详细解释其工作原理。",
            "context": "密码学基础知识",
            "reference_answer": "AES（Advanced Encryption Standard）是一种对称加密算法，使用相同的密钥进行加密和解密。它采用替换-置换网络结构，支持128、192、256位密钥长度。",
            "model_answer": "AES是高级加密标准，是一种广泛使用的对称加密算法。它使用固定的128位数据块，支持128、192和256位的密钥长度。AES采用多轮加密过程，每轮包含字节替换、行移位、列混合和轮密钥加等操作。",
            "domain_tags": ["密码学", "对称加密", "AES"],
            "difficulty_level": "intermediate",
            "expected_concepts": ["对称加密", "密钥长度", "加密轮数", "替换置换网络"]
        },
        {
            "question_id": "crypto_002",
            "question": "RSA算法的安全性基于什么数学难题？",
            "context": "公钥密码学",
            "reference_answer": "RSA算法的安全性基于大整数分解的数学难题，即给定两个大素数的乘积，要找到这两个素数在计算上是困难的。",
            "model_answer": "RSA的安全性依赖于大数分解问题的困难性。当两个大素数相乘得到一个合数时，从这个合数反推出原来的两个素数是非常困难的，这就是RSA安全性的数学基础。",
            "domain_tags": ["密码学", "公钥加密", "RSA", "数学基础"],
            "difficulty_level": "advanced",
            "expected_concepts": ["大整数分解", "素数", "计算复杂性", "单向函数"]
        }
    ]

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    client = ExpertEvaluationAPIClient()
    
    # 1. 健康检查
    print("1. 检查服务健康状态...")
    health = client.health_check()
    print(f"服务状态: {health['status']}")
    print(f"模型已加载: {health['model_loaded']}")
    
    # 2. 获取服务统计
    print("\n2. 获取服务统计...")
    stats = client.get_stats()
    print(f"总请求数: {stats['total_requests']}")
    print(f"活跃任务数: {stats['active_tasks']}")
    
    # 3. 同步评估
    print("\n3. 执行同步评估...")
    qa_items = create_sample_qa_items()
    
    try:
        result = client.evaluate_sync(qa_items)
        print(f"评估完成，总分: {result['overall_score']:.2f}")
        print("维度得分:")
        for dim, score in result['dimension_scores'].items():
            print(f"  {dim}: {score:.2f}")
    except requests.exceptions.HTTPError as e:
        print(f"评估失败: {e}")

def example_async_evaluation():
    """异步评估示例"""
    print("\n=== 异步评估示例 ===")
    
    client = ExpertEvaluationAPIClient()
    qa_items = create_sample_qa_items()
    
    # 1. 提交异步评估任务
    print("1. 提交异步评估任务...")
    task_id = client.evaluate_async(qa_items)
    print(f"任务ID: {task_id}")
    
    # 2. 监控任务进度
    print("2. 监控任务进度...")
    try:
        final_status = client.wait_for_task(task_id, timeout=60)
        
        if final_status["status"] == "completed":
            print("任务完成！")
            result = final_status["result"]
            print(f"总分: {result['overall_score']:.2f}")
            
            # 3. 生成报告
            print("3. 生成评估报告...")
            report = client.generate_report(task_id, format="json")
            print("报告生成完成")
            
        else:
            print(f"任务失败: {final_status.get('error', '未知错误')}")
            
    except TimeoutError as e:
        print(f"任务超时: {e}")
        # 取消任务
        client.cancel_task(task_id)

def example_batch_evaluation():
    """批量评估示例"""
    print("\n=== 批量评估示例 ===")
    
    client = ExpertEvaluationAPIClient()
    
    # 创建多个数据集
    datasets = [
        create_sample_qa_items(),  # 数据集1
        create_sample_qa_items()   # 数据集2
    ]
    
    # 1. 提交批量评估任务
    print("1. 提交批量评估任务...")
    task_id = client.batch_evaluate_async(datasets)
    print(f"批量任务ID: {task_id}")
    
    # 2. 等待完成
    print("2. 等待批量评估完成...")
    try:
        final_status = client.wait_for_task(task_id, timeout=120)
        
        if final_status["status"] == "completed":
            print("批量评估完成！")
            result = final_status["result"]
            
            print(f"数据集数量: {len(result['overall_results'])}")
            for i, dataset_result in enumerate(result['overall_results']):
                print(f"数据集 {i+1} 得分: {dataset_result['overall_score']:.2f}")
                
        else:
            print(f"批量评估失败: {final_status.get('error', '未知错误')}")
            
    except TimeoutError as e:
        print(f"批量评估超时: {e}")

def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    client = ExpertEvaluationAPIClient()
    qa_items = create_sample_qa_items()
    
    # 自定义评估配置
    custom_config = {
        "evaluation_dimensions": [
            "semantic_similarity",
            "domain_accuracy", 
            "response_relevance"
        ],
        "industry_weights": {
            "semantic_similarity": 0.4,
            "domain_accuracy": 0.4,
            "response_relevance": 0.2
        },
        "threshold_settings": {
            "min_score": 0.6,
            "confidence_level": 0.95
        },
        "enable_detailed_analysis": True
    }
    
    print("1. 使用自定义配置进行评估...")
    try:
        result = client.evaluate_sync(qa_items, config=custom_config)
        print(f"自定义评估完成，总分: {result['overall_score']:.2f}")
        
        if result.get('improvement_suggestions'):
            print("改进建议:")
            for suggestion in result['improvement_suggestions']:
                print(f"  - {suggestion}")
                
    except requests.exceptions.HTTPError as e:
        print(f"自定义评估失败: {e}")

def example_task_management():
    """任务管理示例"""
    print("\n=== 任务管理示例 ===")
    
    client = ExpertEvaluationAPIClient()
    
    # 1. 列出所有任务
    print("1. 列出所有任务...")
    tasks = client.list_tasks()
    print(f"总任务数: {tasks['total_count']}")
    print(f"活跃任务数: {tasks['active_count']}")
    
    # 2. 提交一个长时间运行的任务
    qa_items = create_sample_qa_items() * 10  # 扩大数据集
    task_id = client.evaluate_async(qa_items)
    print(f"2. 提交长时间任务: {task_id}")
    
    # 3. 等待一段时间后取消
    time.sleep(2)
    print("3. 取消任务...")
    cancel_result = client.cancel_task(task_id)
    print(f"取消结果: {cancel_result['message']}")
    
    # 4. 检查任务状态
    status = client.get_task_status(task_id)
    print(f"4. 任务状态: {status['status']}")

def main():
    """主函数 - 运行所有示例"""
    print("专家评估系统 API 使用示例")
    print("=" * 50)
    
    try:
        # 基础使用
        example_basic_usage()
        
        # 异步评估
        example_async_evaluation()
        
        # 批量评估
        example_batch_evaluation()
        
        # 自定义配置
        example_custom_config()
        
        # 任务管理
        example_task_management()
        
        print("\n所有示例执行完成！")
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务器")
        print("请确保API服务器正在运行: uv run python -m src.expert_evaluation.api")
    except Exception as e:
        print(f"示例执行出错: {e}")

if __name__ == "__main__":
    main()