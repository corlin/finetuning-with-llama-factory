"""
模型服务化接口
基于FastAPI的模型推理服务，支持异步请求处理和批量推理
"""

import asyncio
import time
import traceback
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from contextlib import asynccontextmanager

import torch
import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from .model_exporter import ModelExporter
from .data_models import ModelMetadata
from .memory_manager import MemoryManager
from .gpu_utils import GPUDetector

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 请求和响应模型
class GenerateRequest(BaseModel):
    """文本生成请求模型"""
    prompt: str = Field(..., min_length=1, max_length=4096, description="输入提示文本")
    max_length: int = Field(default=512, ge=1, le=2048, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="生成温度")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p采样参数")
    do_sample: bool = Field(default=True, description="是否使用采样")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("提示文本不能为空")
        return v.strip()

class ThinkingRequest(BaseModel):
    """深度思考推理请求模型"""
    question: str = Field(..., min_length=1, max_length=2048, description="问题文本")
    context: Optional[str] = Field(default=None, max_length=2048, description="上下文信息")
    thinking_depth: int = Field(default=3, ge=1, le=5, description="思考深度级别")
    include_reasoning: bool = Field(default=True, description="是否包含推理过程")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("问题文本不能为空")
        return v.strip()

class BatchGenerateRequest(BaseModel):
    """批量生成请求模型"""
    prompts: List[str] = Field(..., min_items=1, max_items=10, description="批量提示文本")
    max_length: int = Field(default=512, ge=1, le=2048, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="生成温度")
    
    @validator('prompts')
    def validate_prompts(cls, v):
        if not v:
            raise ValueError("批量提示不能为空")
        for prompt in v:
            if not prompt.strip():
                raise ValueError("提示文本不能为空")
        return [p.strip() for p in v]

class GenerateResponse(BaseModel):
    """生成响应模型"""
    generated_text: str
    prompt: str
    generation_time: float
    token_count: int
    model_info: Dict[str, Any]

class ThinkingResponse(BaseModel):
    """思考推理响应模型"""
    question: str
    thinking_process: str
    final_answer: str
    reasoning_steps: List[str]
    confidence_score: float
    processing_time: float

class BatchGenerateResponse(BaseModel):
    """批量生成响应模型"""
    results: List[GenerateResponse]
    total_time: float
    batch_size: int

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: datetime
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime_seconds: float

class ServiceStats(BaseModel):
    """服务统计响应模型"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    model_memory_usage: float
    gpu_utilization: float
    uptime_hours: float

# 全局变量
model = None
tokenizer = None
model_metadata = None
service_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "response_times": [],
    "start_time": time.time()
}

class ModelService:
    """模型服务类"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_metadata = None
        self.memory_manager = MemoryManager()
        self.gpu_detector = GPUDetector()
        self.model_exporter = ModelExporter()
        
    async def load_model(self, model_path: str, quantization_format: str = "int8"):
        """异步加载模型"""
        try:
            logger.info(f"开始加载模型: {model_path}")
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("使用CPU进行推理")
            
            # 加载量化模型
            if quantization_format in ["int8", "int4", "gptq"]:
                self.model = await self._load_quantized_model(model_path, quantization_format)
            else:
                self.model = await self._load_standard_model(model_path)
            
            self.model.to(device)
            self.model.eval()
            
            # 加载tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 加载模型元数据
            self.model_metadata = self._load_model_metadata(model_path)
            
            logger.info("模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    
    async def _load_quantized_model(self, model_path: str, format: str):
        """加载量化模型"""
        # 这里应该集成实际的量化模型加载逻辑
        # 暂时使用标准模型加载作为示例
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    async def _load_standard_model(self, model_path: str):
        """加载标准模型"""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    def _load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """加载模型元数据"""
        return {
            "model_path": model_path,
            "model_type": "Qwen3-4B-Thinking",
            "quantization": "int8",
            "loaded_at": datetime.now().isoformat(),
            "parameters": "4B",
            "language": "Chinese",
            "domain": "Cryptography"
        }
    
    async def generate_text(self, request: GenerateRequest) -> GenerateResponse:
        """生成文本"""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        start_time = time.time()
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(request.prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(request.prompt):].strip()
            
            generation_time = time.time() - start_time
            token_count = len(outputs[0]) - len(inputs[0])
            
            return GenerateResponse(
                generated_text=generated_text,
                prompt=request.prompt,
                generation_time=generation_time,
                token_count=token_count,
                model_info=self.model_metadata or {}
            )
            
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"文本生成失败: {str(e)}")
    
    async def thinking_inference(self, request: ThinkingRequest) -> ThinkingResponse:
        """深度思考推理"""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        start_time = time.time()
        
        try:
            # 构建思考提示
            thinking_prompt = self._build_thinking_prompt(
                request.question, 
                request.context, 
                request.thinking_depth
            )
            
            # 生成思考过程
            inputs = self.tokenizer.encode(thinking_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解析思考过程
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            thinking_result = self._parse_thinking_response(full_response, thinking_prompt)
            
            processing_time = time.time() - start_time
            
            return ThinkingResponse(
                question=request.question,
                thinking_process=thinking_result["thinking_process"],
                final_answer=thinking_result["final_answer"],
                reasoning_steps=thinking_result["reasoning_steps"],
                confidence_score=thinking_result["confidence_score"],
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"思考推理失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"思考推理失败: {str(e)}")
    
    def _build_thinking_prompt(self, question: str, context: Optional[str], depth: int) -> str:
        """构建思考提示"""
        prompt = f"请深入思考以下问题：\n\n问题：{question}\n\n"
        
        if context:
            prompt += f"上下文：{context}\n\n"
        
        prompt += "<thinking>\n"
        prompt += "让我仔细分析这个问题...\n"
        
        return prompt
    
    def _parse_thinking_response(self, response: str, prompt: str) -> Dict[str, Any]:
        """解析思考响应"""
        # 移除原始提示
        response = response[len(prompt):].strip()
        
        # 简单的思考过程解析
        thinking_process = ""
        final_answer = ""
        reasoning_steps = []
        
        if "<thinking>" in response and "</thinking>" in response:
            thinking_start = response.find("<thinking>") + len("<thinking>")
            thinking_end = response.find("</thinking>")
            thinking_process = response[thinking_start:thinking_end].strip()
            final_answer = response[thinking_end + len("</thinking>"):].strip()
            
            # 提取推理步骤
            reasoning_steps = [step.strip() for step in thinking_process.split('\n') if step.strip()]
        else:
            final_answer = response
            reasoning_steps = [response]
        
        # 简单的置信度计算
        confidence_score = min(0.9, len(reasoning_steps) * 0.2)
        
        return {
            "thinking_process": thinking_process,
            "final_answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "confidence_score": confidence_score
        }
    
    async def batch_generate(self, request: BatchGenerateRequest) -> BatchGenerateResponse:
        """批量生成"""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        start_time = time.time()
        results = []
        
        try:
            # 并行处理批量请求
            tasks = []
            for prompt in request.prompts:
                gen_request = GenerateRequest(
                    prompt=prompt,
                    max_length=request.max_length,
                    temperature=request.temperature
                )
                tasks.append(self.generate_text(gen_request))
            
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            return BatchGenerateResponse(
                results=results,
                total_time=total_time,
                batch_size=len(request.prompts)
            )
            
        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """获取健康状态"""
        try:
            # 内存使用情况
            memory_info = psutil.virtual_memory()
            memory_usage = {
                "total_gb": round(memory_info.total / (1024**3), 2),
                "used_gb": round(memory_info.used / (1024**3), 2),
                "available_gb": round(memory_info.available / (1024**3), 2),
                "percent": memory_info.percent
            }
            
            # GPU内存使用（如果可用）
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                memory_usage["gpu_used_gb"] = round(gpu_memory, 2)
            
            uptime = time.time() - service_stats["start_time"]
            
            return HealthResponse(
                status="healthy" if self.model is not None else "model_not_loaded",
                timestamp=datetime.now(),
                model_loaded=self.model is not None,
                gpu_available=torch.cuda.is_available(),
                memory_usage=memory_usage,
                uptime_seconds=uptime
            )
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return HealthResponse(
                status="error",
                timestamp=datetime.now(),
                model_loaded=False,
                gpu_available=False,
                memory_usage={},
                uptime_seconds=0
            )
    
    def get_service_stats(self) -> ServiceStats:
        """获取服务统计"""
        try:
            uptime = time.time() - service_stats["start_time"]
            avg_response_time = (
                sum(service_stats["response_times"]) / len(service_stats["response_times"])
                if service_stats["response_times"] else 0
            )
            
            # GPU利用率
            gpu_utilization = 0.0
            if torch.cuda.is_available():
                gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            
            # 模型内存使用
            model_memory = 0.0
            if torch.cuda.is_available() and self.model:
                model_memory = torch.cuda.memory_allocated() / (1024**3)
            
            return ServiceStats(
                total_requests=service_stats["total_requests"],
                successful_requests=service_stats["successful_requests"],
                failed_requests=service_stats["failed_requests"],
                average_response_time=avg_response_time,
                model_memory_usage=model_memory,
                gpu_utilization=gpu_utilization,
                uptime_hours=uptime / 3600
            )
            
        except Exception as e:
            logger.error(f"获取服务统计失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取服务统计失败: {str(e)}")

# 全局服务实例
model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    try:
        # 这里可以配置默认模型路径
        # await model_service.load_model("path/to/model")
        logger.info("服务启动完成")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
    
    yield
    
    # 关闭时清理资源
    logger.info("服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="Qwen3-4B-Thinking 模型服务",
    description="基于FastAPI的中文密码学模型推理服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求统计中间件
@app.middleware("http")
async def stats_middleware(request, call_next):
    start_time = time.time()
    service_stats["total_requests"] += 1
    
    try:
        response = await call_next(request)
        service_stats["successful_requests"] += 1
        return response
    except Exception as e:
        service_stats["failed_requests"] += 1
        raise
    finally:
        process_time = time.time() - start_time
        service_stats["response_times"].append(process_time)
        # 保持最近1000次请求的响应时间
        if len(service_stats["response_times"]) > 1000:
            service_stats["response_times"] = service_stats["response_times"][-1000:]

# API端点
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """文本生成端点"""
    return await model_service.generate_text(request)

@app.post("/thinking", response_model=ThinkingResponse)
async def thinking_inference(request: ThinkingRequest):
    """深度思考推理端点"""
    return await model_service.thinking_inference(request)

@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest):
    """批量生成端点"""
    return await model_service.batch_generate(request)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return model_service.get_health_status()

@app.get("/stats", response_model=ServiceStats)
async def service_statistics():
    """服务统计端点"""
    return model_service.get_service_stats()

@app.post("/load_model")
async def load_model(model_path: str, quantization_format: str = "int8"):
    """加载模型端点"""
    try:
        await model_service.load_model(model_path, quantization_format)
        return {"status": "success", "message": "模型加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    """模型信息端点"""
    if not model_service.model_metadata:
        raise HTTPException(status_code=503, detail="模型未加载")
    return model_service.model_metadata

if __name__ == "__main__":
    uvicorn.run(
        "model_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )