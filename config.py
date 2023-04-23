from pathlib import Path
from pydantic import (
    BaseSettings,
    HttpUrl
)


BASE_DIR = Path(__file__).resolve().parent


class Config(BaseSettings):
    aii_admin_url: str
    aii_admin_secret_key: str

    class Config:
        env_file = str(Path(BASE_DIR, '.env'))
        env_file_encoding = 'utf-8'


config = Config()
