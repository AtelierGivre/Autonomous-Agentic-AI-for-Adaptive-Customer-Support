"""Conversation memory.

Holds the recent turns, the pending side-effecting action (with an expiry) and
the repeated-failure counter used for adaptive escalation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_HISTORY_TURNS = 12


def _now() -> float:
    return time.time()


@dataclass
class ConversationMemory:
    turns: list[dict] = field(default_factory=list)
    pending: dict | None = None
    failure_count: int = 0
    last_trace_id: str | None = None

    # ------------------------------------------------------------------ history
    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})

    def history_text(self, limit: int = MAX_HISTORY_TURNS) -> str:
        recent = self.turns[-limit:]
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)

    def clear(self) -> None:
        self.turns.clear()
        self.pending = None
        self.failure_count = 0

    # ---------------------------------------------------------- confirmation
    def propose(self, action: str, args: dict, summary: str) -> None:
        self.pending = {
            "action": action,
            "args": args,
            "summary": summary,
            "created_at": _now(),
        }

    def pending_valid(self, ttl_s: int) -> bool:
        if not self.pending:
            return False
        return (_now() - float(self.pending.get("created_at", 0))) <= ttl_s

    def clear_pending(self) -> None:
        self.pending = None

    # ------------------------------------------------------- adaptive state
    def note_failure(self) -> None:
        self.failure_count += 1

    def reset_failures(self) -> None:
        self.failure_count = 0

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "turns": self.turns,
            "pending": self.pending,
            "failure_count": self.failure_count,
            "last_trace_id": self.last_trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ConversationMemory:
        if not data:
            return cls()
        return cls(
            turns=list(data.get("turns", [])),
            pending=data.get("pending"),
            failure_count=int(data.get("failure_count", 0)),
            last_trace_id=data.get("last_trace_id"),
        )
