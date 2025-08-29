#!/usr/bin/env python3

print("Starting API debug...")

try:
    print("1. Importing basic modules...")
    import asyncio
    import time
    import uuid
    from typing import List, Dict, Any, Optional, Union
    from datetime import datetime
    from contextlib import asynccontextmanager
    import logging
    print("   Basic modules imported")

    print("2. Importing FastAPI...")
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    print("   FastAPI modules imported")

    print("3. Importing expert evaluation modules...")
    from src.expert_evaluation.engine import ExpertEvaluationEngine
    print("   Engine imported")
    
    from src.expert_evaluation.config import ExpertEvaluationConfig, EvaluationDimension, ExpertiseLevel
    print("   Config imported")
    
    from src.expert_evaluation.data_models import (
        QAEvaluationItem, 
        ExpertEvaluationResult, 
        BatchEvaluationResult,
        EvaluationReport,
        EvaluationDataset
    )
    print("   Data models imported")
    
    from src.expert_evaluation.exceptions import (
        ModelLoadError,
        EvaluationProcessError,
        DataFormatError,
        ConfigurationError
    )
    print("   Exceptions imported")

    print("4. Setting up logging...")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("   Logging configured")

    print("5. Creating global variables...")
    evaluation_engine: Optional[ExpertEvaluationEngine] = None
    task_status: Dict[str, Dict[str, Any]] = {}
    service_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "active_tasks": 0,
        "response_times": []
    }
    print("   Global variables created")

    print("6. Creating lifespan function...")
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        global evaluation_engine
        
        try:
            logger.info("初始化专家评估引擎...")
            evaluation_engine = ExpertEvaluationEngine()
            logger.info("专家评估API服务启动完成")
        except Exception as e:
            logger.error(f"服务启动失败: {str(e)}")
            evaluation_engine = None
        
        yield
        
        if evaluation_engine:
            try:
                logger.info("清理评估引擎资源...")
            except Exception as e:
                logger.error(f"资源清理失败: {str(e)}")
        
        logger.info("专家评估API服务关闭")
    print("   Lifespan function created")

    print("7. Creating FastAPI app...")
    app = FastAPI(
        title="专家评估系统 API",
        description="基于FastAPI的专家级行业化模型评估服务",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    print("   FastAPI app created successfully")
    print(f"   App type: {type(app)}")

    print("8. Testing app import...")
    print(f"   App is defined: {app is not None}")
    
except Exception as e:
    print(f"Error during API creation: {e}")
    import traceback
    traceback.print_exc()