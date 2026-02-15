"""
主应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import os

# 确保可以导入 app 模块
sys.path.insert(0, os.path.dirname(__file__))

from app.api import review, standards
from app.config import settings

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG"
)

# 创建应用
app = FastAPI(
    title="DocReviewer - 文档审核系统",
    description="基于 AI 的文档标准审核系统，支持长文档智能分块和跨段落检查",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(review.router)
app.include_router(standards.router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "DocReviewer",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model": settings.deepseek_model,
        "debug": settings.debug
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("DocReviewer 文档审核系统启动中...")
    logger.info(f"服务地址: http://{settings.app_host}:{settings.app_port}")
    logger.info(f"API 文档: http://{settings.app_host}:{settings.app_port}/docs")
    logger.info(f"使用模型: {settings.deepseek_model}")
    logger.info("=" * 60)
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )

