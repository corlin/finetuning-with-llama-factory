"""
模型服务接口测试
测试FastAPI服务的所有端点和功能
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import torch

# 导入被测试的模块
from src.model_service import app, model_service, ModelService
from src.model_service import GenerateRequest, ThinkingRequest, BatchGenerateRequest

class TestModelService:
    """模型服务测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model_service(self):
        """模拟模型服务"""
        with patch('src.model_service.model_service') as mock:
            mock.model = Mock()
            mock.tokenizer = Mock()
            mock.model_metadata = {
                "model_type": "Qwen3-4B-Thinking",
                "quantization": "int8",
                "loaded_at": "2025-01-26T10:00:00"
            }
            yield mock
    
    def test_health_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "memory_usage" in data
        assert "uptime_seconds" in data
    
    def test_stats_endpoint(self, client, mock_model_service):
        """测试统计端点"""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_response_time" in data
        assert "uptime_hours" in data
    
    def test_model_info_endpoint_no_model(self, client):
        """测试模型信息端点（未加载模型）"""
        response = client.get("/model_info")
        assert response.status_code == 503
        assert "模型未加载" in response.json()["detail"]
    
    def test_model_info_endpoint_with_model(self, client, mock_model_service):
        """测试模型信息端点（已加载模型）"""
        response = client.get("/model_info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_type"] == "Qwen3-4B-Thinking"
        assert data["quantization"] == "int8"
    
    def test_generate_endpoint_no_model(self, client):
        """测试生成端点（未加载模型）"""
        request_data = {
            "prompt": "什么是AES加密？",
            "max_length": 100,
            "temperature": 0.7
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 503
        assert "模型未加载" in response.json()["detail"]
    
    @patch('src.model_service.model_service')
    def test_generate_endpoint_success(self, mock_service, client):
        """测试生成端点成功情况"""
        # 模拟成功的生成响应
        mock_response = {
            "generated_text": "AES是一种对称加密算法...",
            "prompt": "什么是AES加密？",
            "generation_time": 2.5,
            "token_count": 50,
            "model_info": {"model_type": "Qwen3-4B-Thinking"}
        }
        
        mock_service.generate_text = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        request_data = {
            "prompt": "什么是AES加密？",
            "max_length": 100,
            "temperature": 0.7
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 200
    
    def test_generate_endpoint_validation(self, client):
        """测试生成端点输入验证"""
        # 测试空提示
        response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422
        
        # 测试过长提示
        long_prompt = "a" * 5000
        response = client.post("/generate", json={"prompt": long_prompt})
        assert response.status_code == 422
        
        # 测试无效参数
        response = client.post("/generate", json={
            "prompt": "测试",
            "temperature": 3.0  # 超出范围
        })
        assert response.status_code == 422
    
    @patch('src.model_service.model_service')
    def test_thinking_endpoint_success(self, mock_service, client):
        """测试思考推理端点成功情况"""
        mock_response = {
            "question": "RSA和AES的区别？",
            "thinking_process": "让我分析这两种加密算法...",
            "final_answer": "RSA是非对称加密，AES是对称加密...",
            "reasoning_steps": ["分析RSA特点", "分析AES特点", "对比差异"],
            "confidence_score": 0.85,
            "processing_time": 3.2
        }
        
        mock_service.thinking_inference = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        request_data = {
            "question": "RSA和AES的区别？",
            "thinking_depth": 3
        }
        
        response = client.post("/thinking", json=request_data)
        assert response.status_code == 200
    
    def test_thinking_endpoint_validation(self, client):
        """测试思考推理端点输入验证"""
        # 测试空问题
        response = client.post("/thinking", json={"question": ""})
        assert response.status_code == 422
        
        # 测试无效思考深度
        response = client.post("/thinking", json={
            "question": "测试问题",
            "thinking_depth": 10  # 超出范围
        })
        assert response.status_code == 422
    
    @patch('src.model_service.model_service')
    def test_batch_generate_endpoint_success(self, mock_service, client):
        """测试批量生成端点成功情况"""
        mock_response = {
            "results": [
                {
                    "generated_text": "对称加密使用相同密钥...",
                    "prompt": "什么是对称加密？",
                    "generation_time": 2.0,
                    "token_count": 30,
                    "model_info": {}
                }
            ],
            "total_time": 2.5,
            "batch_size": 1
        }
        
        mock_service.batch_generate = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        request_data = {
            "prompts": ["什么是对称加密？"],
            "max_length": 100
        }
        
        response = client.post("/batch_generate", json=request_data)
        assert response.status_code == 200
    
    def test_batch_generate_validation(self, client):
        """测试批量生成端点输入验证"""
        # 测试空批次
        response = client.post("/batch_generate", json={"prompts": []})
        assert response.status_code == 422
        
        # 测试过大批次
        large_batch = ["测试"] * 20
        response = client.post("/batch_generate", json={"prompts": large_batch})
        assert response.status_code == 422
    
    def test_load_model_endpoint(self, client):
        """测试加载模型端点"""
        with patch('src.model_service.model_service.load_model') as mock_load:
            mock_load.return_value = AsyncMock()
            
            response = client.post("/load_model?model_path=test_path&quantization_format=int8")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "模型加载成功" in data["message"]

class TestModelServiceClass:
    """ModelService类的单元测试"""
    
    @pytest.fixture
    def service(self):
        """创建ModelService实例"""
        return ModelService()
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, service):
        """测试模型加载成功"""
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            mock_model.return_value = Mock()
            mock_tokenizer.return_value = Mock()
            
            result = await service.load_model("test_path", "int8")
            assert result is True
            assert service.model is not None
            assert service.tokenizer is not None
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self, service):
        """测试模型加载失败"""
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            mock_model.side_effect = Exception("模型文件不存在")
            
            with pytest.raises(Exception):
                await service.load_model("invalid_path", "int8")
    
    @pytest.mark.asyncio
    async def test_generate_text_success(self, service):
        """测试文本生成成功"""
        # 设置模拟对象
        service.model = Mock()
        service.tokenizer = Mock()
        service.model_metadata = {"model_type": "test"}
        
        # 模拟tokenizer行为
        service.tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        service.tokenizer.decode.return_value = "生成的文本"
        service.tokenizer.eos_token_id = 2
        
        # 模拟模型生成
        service.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        request = GenerateRequest(
            prompt="测试提示",
            max_length=100,
            temperature=0.7
        )
        
        response = await service.generate_text(request)
        
        assert response.prompt == "测试提示"
        assert response.generated_text == "生成的文本"
        assert response.generation_time > 0
        assert response.token_count > 0
    
    @pytest.mark.asyncio
    async def test_generate_text_no_model(self, service):
        """测试未加载模型时的文本生成"""
        request = GenerateRequest(prompt="测试", max_length=100)
        
        with pytest.raises(Exception) as exc_info:
            await service.generate_text(request)
        
        assert "模型未加载" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_thinking_inference_success(self, service):
        """测试思考推理成功"""
        service.model = Mock()
        service.tokenizer = Mock()
        
        # 模拟tokenizer和模型行为
        service.tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        service.tokenizer.decode.return_value = "<thinking>分析问题...</thinking>这是答案"
        service.tokenizer.eos_token_id = 2
        service.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        request = ThinkingRequest(
            question="测试问题",
            thinking_depth=2
        )
        
        response = await service.thinking_inference(request)
        
        assert response.question == "测试问题"
        assert response.processing_time > 0
        assert isinstance(response.reasoning_steps, list)
    
    def test_build_thinking_prompt(self, service):
        """测试思考提示构建"""
        prompt = service._build_thinking_prompt("测试问题", "测试上下文", 3)
        
        assert "测试问题" in prompt
        assert "测试上下文" in prompt
        assert "<thinking>" in prompt
    
    def test_parse_thinking_response(self, service):
        """测试思考响应解析"""
        response = "<thinking>这是思考过程</thinking>这是最终答案"
        prompt = "原始提示"
        
        result = service._parse_thinking_response(response, prompt)
        
        assert result["thinking_process"] == "这是思考过程"
        assert result["final_answer"] == "这是最终答案"
        assert isinstance(result["reasoning_steps"], list)
        assert isinstance(result["confidence_score"], float)
    
    @pytest.mark.asyncio
    async def test_batch_generate_success(self, service):
        """测试批量生成成功"""
        service.model = Mock()
        service.tokenizer = Mock()
        service.model_metadata = {"model_type": "test"}
        
        # 模拟generate_text方法
        async def mock_generate_text(request):
            return type('obj', (object,), {
                'generated_text': f"回答: {request.prompt}",
                'prompt': request.prompt,
                'generation_time': 1.0,
                'token_count': 10,
                'model_info': {}
            })
        
        service.generate_text = mock_generate_text
        
        request = BatchGenerateRequest(
            prompts=["问题1", "问题2"],
            max_length=100
        )
        
        response = await service.batch_generate(request)
        
        assert response.batch_size == 2
        assert len(response.results) == 2
        assert response.total_time > 0
    
    def test_get_health_status(self, service):
        """测试健康状态获取"""
        health = service.get_health_status()
        
        assert health.status in ["healthy", "model_not_loaded", "error"]
        assert health.timestamp is not None
        assert isinstance(health.model_loaded, bool)
        assert isinstance(health.gpu_available, bool)
        assert isinstance(health.memory_usage, dict)
        assert isinstance(health.uptime_seconds, float)
    
    def test_get_service_stats(self, service):
        """测试服务统计获取"""
        stats = service.get_service_stats()
        
        assert isinstance(stats.total_requests, int)
        assert isinstance(stats.successful_requests, int)
        assert isinstance(stats.failed_requests, int)
        assert isinstance(stats.average_response_time, float)
        assert isinstance(stats.uptime_hours, float)

class TestRequestModels:
    """请求模型验证测试"""
    
    def test_generate_request_validation(self):
        """测试生成请求验证"""
        # 有效请求
        valid_request = GenerateRequest(
            prompt="测试提示",
            max_length=100,
            temperature=0.7
        )
        assert valid_request.prompt == "测试提示"
        
        # 无效提示（空字符串）
        with pytest.raises(ValueError):
            GenerateRequest(prompt="", max_length=100)
        
        # 无效参数范围
        with pytest.raises(ValueError):
            GenerateRequest(prompt="测试", temperature=3.0)
    
    def test_thinking_request_validation(self):
        """测试思考请求验证"""
        # 有效请求
        valid_request = ThinkingRequest(
            question="测试问题",
            thinking_depth=3
        )
        assert valid_request.question == "测试问题"
        
        # 无效问题（空字符串）
        with pytest.raises(ValueError):
            ThinkingRequest(question="", thinking_depth=2)
    
    def test_batch_generate_request_validation(self):
        """测试批量生成请求验证"""
        # 有效请求
        valid_request = BatchGenerateRequest(
            prompts=["问题1", "问题2"],
            max_length=100
        )
        assert len(valid_request.prompts) == 2
        
        # 无效批次（空列表）
        with pytest.raises(ValueError):
            BatchGenerateRequest(prompts=[], max_length=100)
        
        # 无效批次（包含空字符串）
        with pytest.raises(ValueError):
            BatchGenerateRequest(prompts=["问题1", ""], max_length=100)

class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_full_workflow(self, client):
        """测试完整工作流程"""
        # 1. 检查健康状态
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. 检查统计信息
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200
        
        # 3. 尝试生成（应该失败，因为模型未加载）
        generate_response = client.post("/generate", json={
            "prompt": "测试",
            "max_length": 50
        })
        assert generate_response.status_code == 503
    
    def test_error_handling(self, client):
        """测试错误处理"""
        # 测试无效JSON
        response = client.post("/generate", data="invalid json")
        assert response.status_code == 422
        
        # 测试缺少必需字段
        response = client.post("/generate", json={})
        assert response.status_code == 422
        
        # 测试无效端点
        response = client.get("/invalid_endpoint")
        assert response.status_code == 404
    
    def test_cors_headers(self, client):
        """测试CORS头部"""
        response = client.options("/health")
        assert response.status_code == 200
        
        # 检查CORS头部（如果配置了的话）
        # assert "Access-Control-Allow-Origin" in response.headers

class TestPerformance:
    """性能测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_concurrent_health_checks(self, client):
        """测试并发健康检查"""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        # 并发发送10个请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # 所有请求都应该成功
        for response in responses:
            assert response.status_code == 200
    
    def test_response_time(self, client):
        """测试响应时间"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # 响应时间应该小于1秒

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])