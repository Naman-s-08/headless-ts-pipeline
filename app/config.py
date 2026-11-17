"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "headless-ts-pipeline"
    debug: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
