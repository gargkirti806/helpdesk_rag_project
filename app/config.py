from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Helpdesk RAG"
    DEBUG: bool = True
    VECTORSTORE_PATH: str = "/home/kirti/cutomer_support/vector_db"
    LLM_MODEL: str = "qwen2.5:latest"
    LLM_TEMPERATURE: float = 0.0
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600
    REDIS_URL: str = "redis://localhost:6379/0"
    class Config:
        env_file = ".env"

settings = Settings()
