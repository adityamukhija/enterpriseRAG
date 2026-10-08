from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    redis_url: str = "redis://localhost:6379"
    gemini_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 3072  # gemini-embedding-001 default output size
    default_model: str = "gemini-2.0-flash"
    flagship_model: str = "gemini-2.0-pro"
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 5
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600
    jwt_secret: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    class Config:
        env_file = ".env"

settings = Settings()
