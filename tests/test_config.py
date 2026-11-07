from __future__ import annotations

import pytest

from agents_app.config import Settings


def _clear_providers(monkeypatch) -> None:
    for key in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auto_resolves_to_mock_without_credentials(monkeypatch):
    _clear_providers(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    assert Settings().resolved_provider == "mock"


def test_auto_resolves_openai_then_azure(monkeypatch):
    _clear_providers(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings().resolved_provider == "openai"

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "support")
    assert Settings().resolved_provider == "azure"


def test_explicit_provider_is_not_overridden(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings().resolved_provider == "local"


def test_production_refuses_unsafe_config(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEYS", "")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("DEBUG_RESPONSES", "true")
    with pytest.raises(RuntimeError, match="unsafe production configuration"):
        Settings().validate_for_startup()


def test_production_accepts_safe_config(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEYS", "secret-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEBUG_RESPONSES", "false")
    Settings().validate_for_startup()  # must not raise


def test_public_summary_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-appear")
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    summary = Settings().public_summary()
    assert summary["provider"] == "openai"
    assert "sk-should-not-appear" not in str(summary)


def test_invalid_numbers_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("RETENTION_DAYS", "not-a-number")
    assert Settings().retention_days == 30
