from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: Optional[str] = "aa-BSFlJZgRVq7FPhrItgfG1nW6RE1ypZFJIBK5jifhWZ5XWlNY"
    openai_model: str = "gpt-4o-mini" #"gpt-5.4-mini"
    embedding_model: str = "text-embedding-3-small"
    base_url: str = "https://api.avalai.ir/v1"

    max_pdf_pages: int = 100

    retrieval_top_k: int = 10
    final_top_k: int = 3

    model_config = {"env_file": ".env", "extra": "ignore"}

    # weaviate
    weaviate_http_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_host: str = "localhost"
    weaviate_grpc_port: int = 50051
    weaviate_collection: str = "DocumentChunk"
    hybrid_alpha: float = 0.5   # 0 = pure BM25, 1 = pure vector

    # redis
    redis_url: str = "redis://localhost:6379/0"
    redis_parent_ttl_seconds: int | None = None  # None = no expiry

    # Phoenix
    phoenix_url: str = "http://127.0.0.1:6006"

    # chunking
    parent_chunk_size: int = 1700
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 300
    child_chunk_overlap: int = 40


    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return self.openai_api_key


settings = Settings()
