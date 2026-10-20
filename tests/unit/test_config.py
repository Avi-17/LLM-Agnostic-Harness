import pytest

from harness.config import ConfigError, load_settings

ALL_KEYS = [
    "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
    "GOOGLE_API_KEY", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
    "EMBEDDING_DIMS", "LOG_LEVEL",
]
REQUIRED = {
    "DB_NAME": "ai_harness_test",
    "DB_USER": "tester",
    "DB_PASSWORD": "p@ss:w/rd",
    "GOOGLE_API_KEY": "fake-key-123",
}


@pytest.fixture
def env(monkeypatch):
    for key in ALL_KEYS:  # isolate from whatever is exported in your shell
        monkeypatch.delenv(key, raising=False)
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def test_defaults_apply(env):
    s = load_settings(env_file=None)
    assert s.database.host == "127.0.0.1"
    assert s.database.port == 5432
    assert s.embedding.dims == 1536


def test_all_missing_values_reported_together(env):
    env.delenv("DB_NAME")
    env.delenv("GOOGLE_API_KEY")
    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None)
    assert "DB_NAME" in str(exc.value)
    assert "GOOGLE_API_KEY" in str(exc.value)


def test_placeholder_secret_rejected(env):
    env.setenv("DB_PASSWORD", "replace_me")
    with pytest.raises(ConfigError, match="DB_PASSWORD"):
        load_settings(env_file=None)


def test_invalid_port_rejected(env):
    env.setenv("DB_PORT", "99999")
    with pytest.raises(ConfigError, match="DB_PORT"):
        load_settings(env_file=None)


def test_log_level_is_case_insensitive(env):
    env.setenv("LOG_LEVEL", "debug")
    assert load_settings(env_file=None).app.log_level == "DEBUG"


def test_secrets_never_leak(env):
    s = load_settings(env_file=None)
    output = repr(s) + str(s.describe())
    assert "p@ss:w/rd" not in output
    assert "fake-key-123" not in output