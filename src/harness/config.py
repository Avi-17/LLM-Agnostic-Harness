"""Typed, validated settings for the harness.

Each component has its own settings *section*. A component is handed only
its section, so adding or removing a component never touches the others.
Nothing outside this module reads environment variables directly.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

_PLACEHOLDERS = {"", "replace_me", "changeme"}

class ConfigError(RuntimeError):
    """Settings are missing or invalid. The message never contains secret values."""

def _require_real_secret(value: SecretStr) -> SecretStr:
    if value.get_secret_value().strip().lower() in _PLACEHOLDERS:
        raise ValueError("is empty or still a placeholder")
    return value

class _Section(BaseSettings):
    """Behavior shared by every settings section."""
    model_config = SettingsConfigDict(
        extra="ignore",  # the .env file also holds other sections' keys
        frozen=True,  # settings are read-only after startup
        env_file_encoding="utf-8",
    )

#----------sections-----------
class AppSettings(_Section):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

class DatabaseSettings(_Section):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = Field(min_length=1)
    user: str = Field(min_length=1)
    password: SecretStr

    _check_password = field_validator("password")(_require_real_secret)

class GoogleSettings(_Section):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_")

    api_key: SecretStr

    _check_key = field_validator("api_key")(_require_real_secret)

class EmbeddingSettings(_Section):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["google"] = "google"
    model: str = "gemini-embedding-2"
    dims: int = Field(default=1536, gt=0, le=3072)

#-------------------------composite--------------------------------------
@dataclass(frozen=True)
class Settings:
    app: AppSettings
    database: DatabaseSettings
    google: GoogleSettings
    embedding: EmbeddingSettings

    def describe(self) -> dict[str, str]:
        """Safe-to-print summary: shows *which* config is active, never secrets."""
        db, emb = self.database, self.embedding
        return {
            "database": f"{db.user}@{db.host}:{db.port}/{db.name}",
            "embedding": f"{emb.provider}:{emb.model} ({emb.dims} dims)",
            "google_api_key": "set",
            "log_level": self.app.log_level,
        }

# To add a component: write its section class,add it here
# and as a field on Settings. Removing one is the reverse.
_SECTIONS: dict[str, type[_Section]] = {
    "app": AppSettings,
    "database": DatabaseSettings,
    "google": GoogleSettings,
    "embedding": EmbeddingSettings,
}

def load_settings(env_file: Path | None = DEFAULT_ENV_FILE) -> Settings:
    """Load every section, reporting *all* problems at once.

    Pass env_file=None to read only real environment variables (used in tests).
    """
    loaded: dict[str, _Section] = {}
    problems: list[str] = []

    for attr, section_cls in _SECTIONS.items():
        try:
            loaded[attr] = section_cls(_env_file=env_file)
        except ValidationError as exc:
            problems.extend(_format_errors(section_cls, exc))

    if problems:
        details = "\n".join(f"  - {p}" for p in problems)
        raise ConfigError(f"Invalid configuration:\n{details}")

    return Settings(**loaded)

def _format_errors(section_cls: type[_Section], exc: ValidationError) -> list[str]:
    prefix = section_cls.model_config.get("env_prefix", "")
    messages = []
    # include_input=False: never echo the offending value (it may be a secret)
    for err in exc.errors(include_input=False):
        field = str(err["loc"][0]) if err["loc"] else "?"
        messages.append(f"{(prefix + field).upper()}: {err['msg']}")
    return messages
