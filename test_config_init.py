#!/usr/bin/env python3

print("Testing ExpertEvaluationConfig initialization...")

try:
    from src.expert_evaluation.config import ExpertEvaluationConfig
    print("Config import successful")
    
    print("Creating config instance...")
    config = ExpertEvaluationConfig()
    print("Config created successfully")
    print(f"Config type: {type(config)}")
    
    from src.expert_evaluation.engine import ExpertEvaluationEngine
    print("Engine import successful")
    
    print("Creating engine with config...")
    engine = ExpertEvaluationEngine(config)
    print("Engine created successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()