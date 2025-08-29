"""
最小化API测试
"""

from fastapi import FastAPI

# 创建FastAPI应用
app = FastAPI(title="Test API")

@app.get("/")
async def root():
    return {"message": "Hello World"}

print("API module loaded successfully")