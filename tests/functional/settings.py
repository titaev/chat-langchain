from pydantic import BaseSettings


class TestSettings(BaseSettings):
    service_url: str


test_settings = TestSettings()