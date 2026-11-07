from __future__ import annotations

import time

import pytest

from agents_app.memory import ConversationMemory
from agents_app.store import SessionStore


@pytest.fixture()
def store(tmp_path):
    session_store = SessionStore(str(tmp_path / "support.db"))
    yield session_store
    session_store.close()


async def test_save_and_load_round_trip(store):
    memory = ConversationMemory()
    memory.add_user("hello")
    memory.add_assistant("hi")
    await store.save("key", memory)
    loaded = await store.load("key")
    assert loaded.turns == memory.turns


async def test_missing_session_returns_empty_memory(store):
    loaded = await store.load("never-seen")
    assert loaded.turns == []


async def test_feedback_requires_a_known_trace(store):
    assert await store.add_feedback("missing", 1, None) is False
    await store.record_trace("trace-1", "key")
    assert await store.add_feedback("trace-1", -1, "wrong answer") is True
    assert await store.add_feedback("trace-1", 1, "changed my mind") is True


async def test_stats_and_erasure(store):
    memory = ConversationMemory()
    memory.add_user("hello")
    await store.save("key", memory)
    await store.record_trace("trace-1", "key")
    await store.add_feedback("trace-1", 1, None)
    assert await store.stats() == {"sessions": 1, "traces": 1, "feedback": 1}

    deleted = await store.delete_session("key")
    assert deleted >= 3
    assert await store.stats() == {"sessions": 0, "traces": 0, "feedback": 0}


async def test_retention_purge_removes_old_rows(store):
    await store.save("key", ConversationMemory())
    await store.record_trace("trace-1", "key")
    time.sleep(0.01)
    removed = await store.purge_older_than(0)
    assert removed >= 2
    assert await store.stats() == {"sessions": 0, "traces": 0, "feedback": 0}
