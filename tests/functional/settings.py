from pydantic import BaseSettings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class TestSettings(BaseSettings):
    service_url: str

    class Config:
        env_file = str(Path(BASE_DIR, '.env'))
        env_file_encoding = 'utf-8'


test_settings = TestSettings()
