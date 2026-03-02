"""
AI Research Orchestrator - Configuration Settings
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Google Gemini Configuration
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    
    # Search API Keys (optional)
    serpapi_key: Optional[str] = Field(default=None, env="SERPAPI_KEY")
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    
    # Orchestrator Settings
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: float = Field(default=2.0, env="RETRY_DELAY")
    max_concurrent_agents: int = Field(default=5, env="MAX_CONCURRENT_AGENTS")
    timeout_seconds: int = Field(default=120, env="TIMEOUT_SECONDS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_llm_config() -> dict:
    """Get LLM configuration based on available API keys."""
    config = {
        "gemini": {
            "api_key": settings.google_api_key,
            "model": settings.gemini_model,
            "available": bool(settings.google_api_key)
        },
        "openai": {
            "api_key": settings.openai_api_key,
            "available": bool(settings.openai_api_key)
        }
    }
    return config
