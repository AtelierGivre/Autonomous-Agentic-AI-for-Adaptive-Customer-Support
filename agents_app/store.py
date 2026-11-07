"""SQLite persistence: sessions, audit trail, traces and feedback.

A single connection guarded by a lock is enough for the documented
single-replica deployment. Every public method is async and offloads the
blocking SQLite call, so it never stalls the event loop.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

from agents_app.memory import ConversationMemory


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    trace_id TEXT PRIMARY KEY,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_session ON audit(session_key);
                CREATE INDEX IF NOT EXISTS idx_audit_created ON audit(created_at);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------ sync core
    def _load(self, key: str) -> ConversationMemory:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM sessions WHERE session_key = ?", (key,)
            ).fetchone()
        if not row:
            return ConversationMemory()
        try:
            return ConversationMemory.from_dict(json.loads(row[0]))
        except json.JSONDecodeError:
            return ConversationMemory()

    def _save(self, key: str, memory: ConversationMemory) -> None:
        payload = json.dumps(memory.to_dict(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_key, data, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(session_key) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (key, payload, time.time()),
            )
            self._conn.commit()

    def _record_trace(self, trace_id: str, session_key: str, event: str = "turn") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, session_key, created_at) VALUES (?, ?, ?)",
                (trace_id, session_key, time.time()),
            )
            self._conn.execute(
                "INSERT INTO audit (trace_id, session_key, event, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (trace_id, session_key, event, None, time.time()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ async API
    async def load(self, key: str) -> ConversationMemory:
        return await asyncio.to_thread(self._load, key)

    async def save(self, key: str, memory: ConversationMemory) -> None:
        await asyncio.to_thread(self._save, key, memory)

    async def record_trace(self, trace_id: str, session_key: str, event: str = "turn") -> None:
        await asyncio.to_thread(self._record_trace, trace_id, session_key, event)

    async def add_feedback(self, trace_id: str, rating: int, comment: str | None) -> bool:
        def _write() -> bool:
            with self._lock:
                exists = self._conn.execute(
                    "SELECT 1 FROM traces WHERE trace_id = ?", (trace_id,)
                ).fetchone()
                if not exists:
                    return False
                self._conn.execute(
                    "INSERT OR REPLACE INTO feedback (trace_id, rating, comment, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (trace_id, rating, comment, time.time()),
                )
                self._conn.commit()
                return True

        return await asyncio.to_thread(_write)

    async def delete_session(self, key: str) -> int:
        def _delete() -> int:
            with self._lock:
                session_rows = self._conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE session_key = ?", (key,)
                ).fetchone()[0]
                trace_rows = self._conn.execute(
                    "SELECT COUNT(*) FROM traces WHERE session_key = ?", (key,)
                ).fetchone()[0]
                audit_rows = self._conn.execute(
                    "SELECT COUNT(*) FROM audit WHERE session_key = ?", (key,)
                ).fetchone()[0]
                self._conn.execute("DELETE FROM sessions WHERE session_key = ?", (key,))
                self._conn.execute(
                    "DELETE FROM feedback WHERE trace_id IN (SELECT trace_id FROM traces WHERE session_key = ?)",
                    (key,),
                )
                self._conn.execute("DELETE FROM traces WHERE session_key = ?", (key,))
                self._conn.execute("DELETE FROM audit WHERE session_key = ?", (key,))
                self._conn.commit()
                return session_rows + trace_rows + audit_rows

        return await asyncio.to_thread(_delete)

    async def purge_older_than(self, days: int) -> int:
        cutoff = time.time() - max(0, int(days)) * 86400

        def _purge() -> int:
            with self._lock:
                removed = self._conn.execute(
                    "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
                ).rowcount
                removed += self._conn.execute(
                    "DELETE FROM traces WHERE created_at < ?", (cutoff,)
                ).rowcount
                removed += self._conn.execute(
                    "DELETE FROM audit WHERE created_at < ?", (cutoff,)
                ).rowcount
                self._conn.execute("DELETE FROM feedback WHERE created_at < ?", (cutoff,))
                self._conn.commit()
                return max(0, removed)

        return await asyncio.to_thread(_purge)

    async def stats(self) -> dict:
        def _stats() -> dict:
            with self._lock:
                sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                traces = self._conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
                feedback = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            return {"sessions": sessions, "traces": traces, "feedback": feedback}

        return await asyncio.to_thread(_stats)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
