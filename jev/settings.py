import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 0.5


def _settings_source():
    try:
        from config import get_settings
        return get_settings()
    except ImportError:
        pass
    try:
        from analyzerEngine.config import get_settings
        return get_settings()
    except ImportError:
        return None


def _read(key: str) -> str | None:
    value = os.environ.get(key)
    if value in (None, ""):
        source = _settings_source()
        value = source.get(key) if source is not None else None
    return value if value not in (None, "") else None


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class JevSettings:
    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    enabled: bool = True

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_key)

    @classmethod
    def from_env(cls) -> "JevSettings":
        return cls(
            api_key=_read("TYPESAFE_API_KEY"),
            base_url=(_read("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            model=_read("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
            timeout_seconds=_as_float(_read("JEV_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS),
            max_retries=max(0, _as_int(_read("JEV_MAX_RETRIES"), DEFAULT_MAX_RETRIES)),
            backoff_seconds=max(0.0, _as_float(_read("JEV_BACKOFF_SECONDS"), DEFAULT_BACKOFF_SECONDS)),
            enabled=(_read("JEV_ENABLED") or "true").strip().lower() not in ("false", "0", "no", "off"),
        )
