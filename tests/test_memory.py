from __future__ import annotations

import time

from agents_app.memory import ConversationMemory


def test_history_is_bounded_to_recent_turns():
    memory = ConversationMemory()
    for index in range(20):
        memory.add_user(f"user {index}")
    text = memory.history_text(limit=3)
    assert "user 17" in text
    assert "user 16" not in text


def test_pending_confirmation_expires():
    memory = ConversationMemory()
    memory.propose("create_ticket", {"subject": "x"}, "open a ticket")
    assert memory.pending_valid(ttl_s=60) is True
    memory.pending["created_at"] = time.time() - 120
    assert memory.pending_valid(ttl_s=60) is False


def test_pending_can_be_cleared():
    memory = ConversationMemory()
    memory.propose("create_ticket", {"subject": "x"}, "open a ticket")
    memory.clear_pending()
    assert memory.pending is None
    assert memory.pending_valid(ttl_s=60) is False


def test_failure_counter_and_reset():
    memory = ConversationMemory()
    memory.note_failure()
    memory.note_failure()
    assert memory.failure_count == 2
    memory.reset_failures()
    assert memory.failure_count == 0


def test_clear_resets_everything():
    memory = ConversationMemory()
    memory.add_user("hello")
    memory.propose("create_ticket", {}, "ticket")
    memory.note_failure()
    memory.clear()
    assert memory.turns == []
    assert memory.pending is None
    assert memory.failure_count == 0


def test_round_trip_serialisation():
    memory = ConversationMemory()
    memory.add_user("hello")
    memory.add_assistant("hi")
    memory.propose("create_ticket", {"subject": "x"}, "ticket")
    memory.note_failure()
    restored = ConversationMemory.from_dict(memory.to_dict())
    assert restored.turns == memory.turns
    assert restored.pending == memory.pending
    assert restored.failure_count == 1


def test_from_dict_tolerates_missing_data():
    restored = ConversationMemory.from_dict(None)
    assert restored.turns == []
    assert restored.pending is None
