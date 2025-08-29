#!/usr/bin/env python3

print("Testing ExpertEvaluationEngine initialization...")

try:
    from src.expert_evaluation.engine import ExpertEvaluationEngine
    print("Engine import successful")
    
    print("Creating engine instance...")
    engine = ExpertEvaluationEngine()
    print("Engine created successfully")
    print(f"Engine type: {type(engine)}")
    
except Exception as e:
    print(f"Error creating engine: {e}")
    import traceback
    traceback.print_exc()