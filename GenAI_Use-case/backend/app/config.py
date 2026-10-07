"""Settings loaded from environment / .env (app DB URL, LLM provider + model, encryption key, sampling limits).

Usage anywhere in the app:
    from app.config import settings
    settings.llm_model
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels above this file (backend/app/config.py -> GenAI_Use-case/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The engine's own database (sources, models, rules, reviews)
    app_db_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"
    # DEVELOPER ONLY: default source for the CLI (the demo DB). Users connect sources via the UI.
    source_db_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'source.db'}"

    # Security
    encryption_key: str = ""

    # LLM (any OpenAI-compatible endpoint: Groq, Ollama, vLLM, ...)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    llm_temperature: float = 0.2

    # Who is acting, until the UI brings real logins
    reviewer: str = "steward"

    # Extraction / privacy
    sample_row_limit: int = 10000
    send_raw_samples_to_llm: bool = False


settings = Settings()
