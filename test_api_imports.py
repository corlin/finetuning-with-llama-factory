#!/usr/bin/env python3

print("Testing individual imports from API file...")

try:
    print("1. Testing basic imports...")
    import asyncio
    import time
    import uuid
    from typing import List, Dict, Any, Optional, Union
    from datetime import datetime
    from contextlib import asynccontextmanager
    import logging
    print("   Basic imports OK")

    print("2. Testing FastAPI imports...")
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    print("   FastAPI imports OK")

    print("3. Testing expert evaluation imports...")
    try:
        from src.expert_evaluation.engine import ExpertEvaluationEngine
        print("   Engine import OK")
    except Exception as e:
        print(f"   Engine import failed: {e}")
        
    try:
        from src.expert_evaluation.config import ExpertEvaluationConfig, EvaluationDimension, ExpertiseLevel
        print("   Config import OK")
    except Exception as e:
        print(f"   Config import failed: {e}")
        
    try:
        from src.expert_evaluation.data_models import (
            QAEvaluationItem, 
            ExpertEvaluationResult, 
            BatchEvaluationResult,
            EvaluationReport,
            EvaluationDataset
        )
        print("   Data models import OK")
    except Exception as e:
        print(f"   Data models import failed: {e}")
        
    try:
        from src.expert_evaluation.exceptions import (
            ModelLoadError,
            EvaluationProcessError,
            DataFormatError,
            ConfigurationError
        )
        print("   Exceptions import OK")
    except Exception as e:
        print(f"   Exceptions import failed: {e}")

    print("4. Testing FastAPI app creation...")
    app = FastAPI(title="Test API")
    print("   FastAPI app creation OK")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()