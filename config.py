# config.py
import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Loads all configuration settings from a .env file.
    This version loads credentials for TWO separate Azure OpenAI resources:
    1. A "CHAT" resource for the main LLM (NLU and responses)
    2. An "EMBED" resource for the embedding model (RAG)
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        frozen=True
    )

    # --- 1. CHAT Model Resource (Your existing 'spotn' resource) ---
    AZURE_OPENAI_CHAT_ENDPOINT: str
    AZURE_OPENAI_CHAT_KEY: str
    AZURE_OPENAI_CHAT_API_VERSION: str
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str  # Your 'gpt-4.1-nano'
    
    # --- 2. EMBEDDING Model Resource (Your new 'sp20254' resource) ---
    AZURE_OPENAI_EMBED_ENDPOINT: str
    AZURE_OPENAI_EMBED_KEY: str
    AZURE_OPENAI_EMBED_API_VERSION: str
    AZURE_OPENAI_EMBED_DEPLOYMENT_NAME: str # Your 'text-embedding-ada-002'

@lru_cache
def get_settings() -> Settings:
    """
    Dependency function to get a single, cached instance of the Settings.
    """
    try:
        return Settings()
    except ValueError as e:
        logging.error(f"Error loading settings. Is your .env file correct? {e}")
        raise