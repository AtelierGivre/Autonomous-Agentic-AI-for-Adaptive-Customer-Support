"""Shared test setup.

The suite must run with no API keys, no network and no GPU. Settings are read
from the environment at import time, so these defaults are set before any
``agents_app`` module is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("MIN_RETRIEVAL_SCORE", "0.1")
os.environ.setdefault("MIN_LEXICAL_COVERAGE", "0.5")
os.environ.setdefault("VECTOR_STORE", "builtin")
os.environ.setdefault("INGEST_BACKEND", "builtin")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG_RESPONSES", "true")
os.environ.setdefault("DEFAULT_CUSTOMER_ID", "demo")
os.environ.setdefault("API_KEYS", "")

import pytest  # noqa: E402

from agents_app.config import Settings  # noqa: E402
from agents_app.memory import ConversationMemory  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def memory() -> ConversationMemory:
    return ConversationMemory()
