#!/usr/bin/env python3

try:
    print("Testing API import...")
    import src.expert_evaluation.api as api_module
    print("Module imported successfully")
    
    print("Available attributes:", [attr for attr in dir(api_module) if not attr.startswith('_')])
    
    if hasattr(api_module, 'app'):
        print("app found")
        print(f"app type: {type(api_module.app)}")
    else:
        print("app not found")
        
    # Try direct import
    try:
        from src.expert_evaluation.api import app
        print("Direct app import successful")
        print(f"App type: {type(app)}")
    except ImportError as e:
        print(f"Direct import failed: {e}")
        
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()