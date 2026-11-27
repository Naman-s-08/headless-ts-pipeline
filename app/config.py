"""
Application configuration.

Loads settings from .env file using Pydantic Settings.
All configuration is externalized - nothing is hardcoded.
"""

from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    model_id: str = "unitary/toxic-bert"
    toxicity_allow_threshold: float = 0.40
    toxicity_review_threshold: float = 0.80
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "trust_safety"
    mongo_collection: str = "audit_logs"
    max_text_chars: int = 5000
    log_level: str = "INFO"

    @model_validator(mode="after")
    def check_thresholds(self):
        """Ensure ALLOW threshold is less than REVIEW threshold, and both are between 0 and 1."""
        if not (0 <= self.toxicity_allow_threshold < self.toxicity_review_threshold <= 1):
            raise ValueError("need 0 <= ALLOW < REVIEW <= 1")
        return self


@lru_cache
def get_settings() -> Settings:
    """Returns a cached Settings instance. Created once and reused."""
    return Settings()
