import os

from dotenv import dotenv_values, load_dotenv


def _is_production() -> bool:
    return os.environ.get("ENV", "").strip().lower() == "production"


if not _is_production():
    load_dotenv()


class Settings:
    def __init__(self, values: dict):
        self._values = values

    def __getattr__(self, key):
        return self._values.get(key, os.getenv(key, ""))

    def get(self, key, default=""):
        return self._values.get(key, os.getenv(key, default))


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings({} if _is_production() else dotenv_values())
    return _settings
