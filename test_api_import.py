#!/usr/bin/env python3

try:
    print("Testing API imports...")
    
    # Test individual imports
    print("1. Testing FastAPI import...")
    from fastapi import FastAPI
    print("   FastAPI import successful")
    
    print("2. Testing expert evaluation engine import...")
    from src.expert_evaluation.engine import ExpertEvaluationEngine
    print("   Engine import successful")
    
    print("3. Testing config import...")
    from src.expert_evaluation.config import ExpertEvaluationConfig
    print("   Config import successful")
    
    print("4. Testing data models import...")
    from src.expert_evaluation.data_models import QAEvaluationItem
    print("   Data models import successful")
    
    print("5. Testing exceptions import...")
    from src.expert_evaluation.exceptions import ModelLoadError
    print("   Exceptions import successful")
    
    print("6. Testing full API module import...")
    import src.expert_evaluation.api as api_module
    print("   API module import successful")
    
    print("7. Checking if app is defined...")
    if hasattr(api_module, 'app'):
        print("   app is defined")
        print(f"   app type: {type(api_module.app)}")
    else:
        print("   app is NOT defined")
        print(f"   Available attributes: {[attr for attr in dir(api_module) if not attr.startswith('_')]}")
    
    print("8. Testing direct app import...")
    from src.expert_evaluation.api import app
    print("   Direct app import successful")
    print(f"   App type: {type(app)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()