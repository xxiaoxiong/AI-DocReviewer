"""
配置管理模块
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # DeepSeek API 配置
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    
    # 备用模型配置
    local_model_api_base: Optional[str] = None
    local_model_name: Optional[str] = None
    
    # 应用配置
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    
    # 向量配置
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    
    # 文档处理配置
    max_file_size: int = 50  # MB
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # 缓存配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_enabled: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 全局配置实例
settings = Settings()

