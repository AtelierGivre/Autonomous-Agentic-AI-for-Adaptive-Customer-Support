"""Configuration for the support agents.

Plain dataclass on purpose: no pydantic-settings dependency, and every knob is
read from the environment once at import time. ``settings`` is a singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, matches the README quick start
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only when python-dotenv is absent
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str = "") -> tuple[str, ...]:
    raw = _env(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass
class Settings:
    # --- runtime ---
    environment: str = field(default_factory=lambda: _env("ENVIRONMENT", "development").lower())
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())
    log_json: bool = field(default_factory=lambda: _bool("LOG_JSON", False))
    debug_responses: bool = field(default_factory=lambda: _bool("DEBUG_RESPONSES", True))

    # --- HTTP / auth ---
    api_keys: tuple[str, ...] = field(default_factory=lambda: _list("API_KEYS"))
    rate_limit_per_min: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MIN", 30))
    rate_limit_key_per_min: int = field(default_factory=lambda: _int("RATE_LIMIT_KEY_PER_MIN", 600))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: _list("CORS_ORIGINS"))
    default_customer_id: str = field(default_factory=lambda: _env("DEFAULT_CUSTOMER_ID", "demo"))

    # --- data / retention ---
    db_path: str = field(default_factory=lambda: _env("DB_PATH", str(ROOT / "data" / "support.db")))
    retention_days: int = field(default_factory=lambda: _int("RETENTION_DAYS", 30))
    knowledge_base_dir: str = field(
        default_factory=lambda: _env("KNOWLEDGE_BASE_DIR", str(ROOT / "knowledge_base"))
    )

    # --- safety ---
    redact_pii: bool = field(default_factory=lambda: _bool("REDACT_PII", True))
    confirm_write_actions: bool = field(default_factory=lambda: _bool("CONFIRM_WRITE_ACTIONS", True))
    confirm_ttl_s: int = field(default_factory=lambda: _int("CONFIRM_TTL_S", 600))
    kb_skip_suspicious: bool = field(default_factory=lambda: _bool("KB_SKIP_SUSPICIOUS", True))
    escalate_after_failures: int = field(default_factory=lambda: _int("ESCALATE_AFTER_FAILURES", 2))

    # --- retrieval ---
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "hf").lower())
    hf_embedding_model: str = field(
        default_factory=lambda: _env("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    min_retrieval_score: float = field(default_factory=lambda: _float("MIN_RETRIEVAL_SCORE", 0.45))
    min_lexical_coverage: float = field(default_factory=lambda: _float("MIN_LEXICAL_COVERAGE", 0.5))
    retrieval_top_k: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 4))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 200))

    # --- LLM routing ---
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto").lower())
    json_mode: bool = field(default_factory=lambda: _bool("JSON_MODE", True))
    llm_timeout_s: float = field(default_factory=lambda: _float("LLM_TIMEOUT_S", 30))
    request_timeout_s: float = field(default_factory=lambda: _float("REQUEST_TIMEOUT_S", 60))
    max_llm_calls_per_request: int = field(default_factory=lambda: _int("MAX_LLM_CALLS_PER_REQUEST", 10))
    max_cost_per_request_usd: float = field(default_factory=lambda: _float("MAX_COST_PER_REQUEST_USD", 0.05))
    price_input_per_1k: float = field(default_factory=lambda: _float("PRICE_INPUT_PER_1K", 0.00015))
    price_output_per_1k: float = field(default_factory=lambda: _float("PRICE_OUTPUT_PER_1K", 0.0006))

    azure_openai_api_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY"))
    azure_openai_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT"))
    azure_openai_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT"))
    azure_openai_deployment_fast: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT_FAST"))
    azure_openai_api_version: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-10-21")
    )

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_model_fast: str = field(default_factory=lambda: _env("OPENAI_MODEL_FAST"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))

    local_model: str = field(default_factory=lambda: _env("LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    local_model_fast: str = field(default_factory=lambda: _env("LOCAL_MODEL_FAST"))
    local_model_revision: str = field(default_factory=lambda: _env("LOCAL_MODEL_REVISION"))
    local_device: str = field(default_factory=lambda: _env("LOCAL_DEVICE", "auto"))
    local_dtype: str = field(default_factory=lambda: _env("LOCAL_DTYPE", "auto"))
    local_max_new_tokens: int = field(default_factory=lambda: _int("LOCAL_MAX_NEW_TOKENS", 400))
    local_temperature: float = field(default_factory=lambda: _float("LOCAL_TEMPERATURE", 0.2))

    # ------------------------------------------------------------------ helpers
    @property
    def is_production(self) -> bool:
        return self.environment in {"production", "prod"}

    @property
    def resolved_provider(self) -> str:
        """Resolve ``auto`` to whichever provider actually has credentials."""
        provider = self.llm_provider
        if provider != "auto":
            return provider
        if self.azure_openai_api_key and self.azure_openai_endpoint and self.azure_openai_deployment:
            return "azure"
        if self.openai_api_key or self.openai_base_url:
            return "openai"
        return "mock"

    def validate_for_startup(self) -> None:
        """Refuse to boot with a configuration that silently disables safety."""
        if not self.is_production:
            return
        problems: list[str] = []
        if not self.api_keys:
            problems.append("API_KEYS must be set when ENVIRONMENT=production")
        if self.resolved_provider == "mock":
            problems.append("LLM_PROVIDER=mock is not allowed when ENVIRONMENT=production")
        if self.debug_responses:
            problems.append("DEBUG_RESPONSES must be false in production")
        if problems:
            raise RuntimeError("unsafe production configuration: " + "; ".join(problems))

    def public_summary(self) -> dict:
        return {
            "environment": self.environment,
            "provider": self.resolved_provider,
            "embedding_backend": self.embedding_backend,
            "auth": "on" if self.api_keys else "off (dev only)",
            "debug_responses": self.debug_responses,
        }


settings = Settings()
