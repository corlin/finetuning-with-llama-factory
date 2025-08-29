#!/usr/bin/env python3

print("Testing minimal API creation...")

try:
    from fastapi import FastAPI
    print("FastAPI imported successfully")
    
    app = FastAPI(title="Test API")
    print("FastAPI app created successfully")
    print(f"App type: {type(app)}")
    
except Exception as e:
    print(f"Error creating minimal API: {e}")
    import traceback
    traceback.print_exc()