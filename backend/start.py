"""
快速启动脚本 - 用于测试
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

# 设置环境变量（测试用）
os.environ['DEEPSEEK_API_KEY'] = 'sk-test-key'
os.environ['DEEPSEEK_API_BASE'] = 'https://api.deepseek.com/v1'
os.environ['DEEPSEEK_MODEL'] = 'deepseek-chat'
os.environ['APP_HOST'] = '0.0.0.0'
os.environ['APP_PORT'] = '8000'
os.environ['DEBUG'] = 'True'

if __name__ == "__main__":
    import uvicorn
    from loguru import logger
    
    logger.info("=" * 60)
    logger.info("🚀 DocReviewer 启动中...")
    logger.info("=" * 60)
    logger.info("📝 注意: 请确保已配置 DEEPSEEK_API_KEY")
    logger.info("🌐 前端地址: 打开 frontend/index.html")
    logger.info("📚 API 文档: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

