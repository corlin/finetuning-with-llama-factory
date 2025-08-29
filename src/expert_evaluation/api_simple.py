"""
专家评估系统 RESTful API 接口 - 简化版本

提供基于FastAPI的Web API端点，支持异步评估任务处理。
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .engine import ExpertEvaluationEngine
from .config import ExpertEvaluationConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
evaluation_engine: Optional[ExpertEvaluationEngine] = None
service_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "active_tasks": 0,
    "response_times": []
}

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global evaluation_engine
    
    try:
        logger.info("初始化专家评估引擎...")
        config = ExpertEvaluationConfig()
        evaluation_engine = ExpertEvaluationEngine(config)
        logger.info("专家评估API服务启动完成")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        evaluation_engine = None
    
    yield
    
    logger.info("专家评估API服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="专家评估系统 API",
    description="基于FastAPI的专家级行业化模型评估服务",
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

# API 端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy" if evaluation_engine else "degraded",
        "timestamp": datetime.now(),
        "model_loaded": evaluation_engine is not None,
        "active_tasks": service_stats["active_tasks"]
    }

@app.get("/stats")
async def service_statistics():
    """服务统计端点"""
    return {
        "total_requests": service_stats["total_requests"],
        "successful_requests": service_stats["successful_requests"],
        "failed_requests": service_stats["failed_requests"],
        "active_tasks": service_stats["active_tasks"]
    }

# 主函数
def main():
    """启动API服务器"""
    uvicorn.run(
        "src.expert_evaluation.api_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()