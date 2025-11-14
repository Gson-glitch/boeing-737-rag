
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    gemini_api_key: str

    # Model Configuration
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Chunking Configuration
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Retrieval Configuration
    hybrid_top_k: int = 100
    rerank_top_k: int = 20
    confidence_threshold: float = 0.6
    max_pages_default: int = 5

    # Storage Paths
    chroma_persist_dir: str = "./data/processed/chroma_db"
    raw_pdf_path: str = "./data/raw/boeing_737_manual.pdf"
    processed_chunks_path: str = "./data/processed/chunks.json"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()  # type: ignore [call-arg]
