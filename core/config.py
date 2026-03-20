"""
Sentinel AI — Application Configuration
Loads settings from environment variables / .env file using Pydantic Settings.
"""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment / .env file.
    All fields have sensible defaults for local development with Ollama.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_vision_model: str = "llava:7b"
    ollama_timeout: int = 120

    # --- Vision ---
    default_vision_model: str = "llava"

    # --- LLM ---
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- RAG ---
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "sentinel_docs"
    embedding_model: str = "nomic-embed-text"
    rag_top_k: int = 5

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: str = "INFO"

    # --- Hardware ---
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # --- Monitoring ---
    otel_enabled: bool = True
    otel_service_name: str = "sentinel-ai"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sentinel-finetune"

    # --- Paths ---
    data_dir: str = "./data"
    models_dir: str = "./models"
    benchmark_output_dir: str = "./benchmark/reports"

    # --- Cloud Fallback ---
    openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    Call this from anywhere in the app:
        from core.config import get_settings
        settings = get_settings()
    """
    return Settings()
