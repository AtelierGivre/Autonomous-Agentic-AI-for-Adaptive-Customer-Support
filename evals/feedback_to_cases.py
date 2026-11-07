"""Turn thumbs-down feedback into candidate eval cases.

Usage::

    python -m evals.feedback_to_cases --db data/support.db --out evals/candidates.jsonl

The store keeps feedback against a ``trace_id``; the trace maps to a session
whose memory holds the redacted conversation. This exporter reads the last
customer turn in that session and writes a *candidate* record. A human still
has to label the expected outcome and add it to ``dataset.jsonl``. Nothing is
ever promoted automatically, and text is redacted again on the way out.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_app.config import settings  # noqa: E402
from agents_app.security import redact_pii  # noqa: E402


def _turns(session_json: str | None) -> list[dict]:
    if not session_json:
        return []
    try:
        data = json.loads(session_json)
    except json.JSONDecodeError:
        return []
    turns = data.get("turns") or []
    return [turn for turn in turns if isinstance(turn, dict)]


def export(db_path: Path, out_path: Path) -> int:
    if not db_path.exists():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 1

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT f.trace_id, f.rating, f.comment, f.created_at,
                   t.session_key, s.data
            FROM feedback f
            LEFT JOIN traces t ON t.trace_id = f.trace_id
            LEFT JOIN sessions s ON s.session_key = t.session_key
            WHERE f.rating < 0
            ORDER BY f.created_at DESC
            """
        ).fetchall()
    except sqlite3.Error as exc:
        print(f"could not read feedback: {exc}", file=sys.stderr)
        return 1
    finally:
        connection.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            turns = _turns(row["data"])
            user_turns = [t.get("content", "") for t in turns if t.get("role") == "user"]
            if not user_turns:
                continue
            message, _ = redact_pii(user_turns[-1])
            if not message or message == "[blocked input]" or message in seen:
                continue
            seen.add(message)
            comment, _ = redact_pii(row["comment"] or "")
            record = {
                "id": f"feedback-{row['trace_id'][:12]}",
                "status": "needs-label",
                "turns": [message],
                "expect": {},
                "context": {
                    "trace_id": row["trace_id"],
                    "session_key": row["session_key"],
                    "rating": row["rating"],
                    "comment": comment,
                    "feedback_at": datetime.fromtimestamp(
                        row["created_at"], tz=UTC
                    ).isoformat(),
                    "note": "Last customer turn in the session; verify before labelling.",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} candidate case(s) to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export thumbs-down feedback as eval candidates.")
    parser.add_argument("--db", type=Path, default=Path(settings.db_path))
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "candidates.jsonl")
    args = parser.parse_args(argv)
    return export(args.db, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
