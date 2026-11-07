from __future__ import annotations

import importlib

import pytest

from agents_app import ingest
from agents_app.ingest import (
    BuiltinIngestor,
    LlamaIndexIngestor,
    _split_sentences,
    coverage_by_words,
    make_ingestor,
)

SAMPLE = (
    "# Router troubleshooting\n\n"
    "Use this order. Stop as soon as the connection is stable.\n\n"
    "1. Power cycle. Unplug the router, wait 30 seconds, plug it back in.\n"
    "2. Check the LEDs. Solid white means the WAN link is up. Flashing amber "
    "means the router cannot see the modem. Solid red means no internet.\n"
    "3. Check the cable. The modem cable must be in the WAN port, not a LAN port.\n\n"
    "If it still disconnects, capture the times of three disconnects and whether "
    "the internet LED changes."
)


def test_builtin_ingestor_splits_and_preserves_text():
    chunks = BuiltinIngestor(200, 20).split(SAMPLE)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert coverage_by_words(SAMPLE, chunks) == 1.0


def test_sentence_pieces_join_back_to_the_original():
    assert "".join(_split_sentences(SAMPLE)) == SAMPLE


def test_llamaindex_ingestor_is_bounded_and_lossless():
    pytest.importorskip("llama_index")
    chunks = LlamaIndexIngestor(200, 20).split(SAMPLE)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert coverage_by_words(SAMPLE, chunks) >= 0.99


def test_llamaindex_ingestor_handles_empty_text():
    pytest.importorskip("llama_index")
    assert LlamaIndexIngestor(200, 20).split("   \n\n ") == []


def test_factory_returns_the_requested_backend():
    assert make_ingestor("builtin", 200, 20).name == "builtin"
    if importlib.util.find_spec("llama_index") is not None:
        assert make_ingestor("llamaindex", 200, 20).name == "llamaindex"
    assert make_ingestor("not-a-chunker", 200, 20).name == "builtin"


def test_factory_falls_back_when_llamaindex_fails(monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise ImportError("llama-index is not installed")

    monkeypatch.setattr(ingest, "LlamaIndexIngestor", Broken)

    assert make_ingestor("llamaindex", 200, 20).name == "builtin"
    assert make_ingestor("auto", 200, 20).name == "builtin"
