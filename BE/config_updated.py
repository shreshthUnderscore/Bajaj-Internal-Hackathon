import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Basic Configuration
    debug: bool = False
    environment: str = "production"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://bfhldevapigw.healthrx.co.in/sp-gw/api/openai/v1/"
    
    # File Configuration
    upload_directory: str = "uploads"
    max_file_size: str = "50MB"
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Vector Database Configuration
    vector_db_type: str = "faiss"
    faiss_index_file: str = "faiss_index.bin"
    faiss_metadata_file: str = "embeddings_metadata.json"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Domain-specific thresholds
    insurance_confidence_threshold: float = 0.7
    legal_confidence_threshold: float = 0.8
    hr_confidence_threshold: float = 0.6
    compliance_confidence_threshold: float = 0.9
    
    # Performance Configuration
    max_chunks_per_document: int = 1000
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_search_results: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
