#!/usr/bin/env python3
"""
启动模型服务的简单脚本
用于测试API端点功能
"""

import uvicorn
import logging
from src.model_service import app

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """启动模型服务"""
    logger.info("启动Qwen3-4B-Thinking模型服务...")
    
    try:
        # 启动FastAPI服务
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"启动服务失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()