#!/usr/bin/env python3

print("Testing minimal API step by step...")

try:
    print("1. Basic imports...")
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    print("   OK")

    print("2. Expert evaluation imports...")
    from src.expert_evaluation.engine import ExpertEvaluationEngine
    from src.expert_evaluation.config import ExpertEvaluationConfig
    print("   OK")

    print("3. Creating lifespan function...")
    evaluation_engine = None
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global evaluation_engine
        try:
            config = ExpertEvaluationConfig()
            evaluation_engine = ExpertEvaluationEngine(config)
            print("   Engine created in lifespan")
        except Exception as e:
            print(f"   Engine creation failed: {e}")
            evaluation_engine = None
        
        yield
        
        print("   Lifespan cleanup")
    
    print("   Lifespan function created")

    print("4. Creating FastAPI app...")
    app = FastAPI(
        title="Test API",
        lifespan=lifespan
    )
    print("   FastAPI app created")
    print(f"   App type: {type(app)}")

    print("5. Testing app availability...")
    print(f"   App is not None: {app is not None}")
    
    print("All steps successful!")
    
except Exception as e:
    print(f"Error at step: {e}")
    import traceback
    traceback.print_exc()