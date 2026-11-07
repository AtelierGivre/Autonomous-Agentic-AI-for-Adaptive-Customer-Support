"""Security primitives: API keys, rate limiting, PII redaction.

These are deliberately dependency-free and deterministic so they can be tested
offline and reasoned about during a review.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import defaultdict, deque

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
# 13-19 digits, optionally separated by spaces or dashes. Luhn-checked below.
_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
# Aadhaar: 12 digits, usually grouped 4-4-4.
_AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_PHONE = re.compile(r"(?<![\d+])(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?)?\d{3,5}[\s\-]\d{4,5}(?!\d)")
_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{12,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|"
    r"Bearer\s+[A-Za-z0-9._\-]{20,}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})\b"
)


def _luhn(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, int]:
    """Redact emails, phones, cards, government IDs and secrets.

    Returns ``(redacted_text, replacement_count)``. This is heuristic: free-text
    names and postal addresses are not detected (documented limitation).
    """
    if not text:
        return text, 0

    count = 0

    def _replace(regex: re.Pattern, label: str, value: str, validator=None) -> str:
        nonlocal count

        def _sub(match: re.Match) -> str:
            nonlocal count
            if validator and not validator(match.group(0)):
                return match.group(0)
            count += 1
            return f"[{label}]"

        return regex.sub(_sub, value)

    out = _replace(_SECRET, "SECRET", text)
    out = _replace(_EMAIL, "EMAIL", out)
    out = _replace(_CARD, "CARD", out, validator=_luhn)
    out = _replace(_AADHAAR, "AADHAAR", out)
    out = _replace(_PAN, "PAN", out)
    out = _replace(_PHONE, "PHONE", out)
    return out, count


def verify_api_key(candidate: str | None, configured) -> str | None:
    """Constant-time comparison against every configured key."""
    if not candidate:
        return None
    for key in configured:
        if hmac.compare_digest(candidate, key):
            return key
    return None


def key_fingerprint(key: str) -> str:
    """Stable, non-reversible identity for rate-limit buckets and logs."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class SlidingWindowRateLimiter:
    """Per-bucket sliding window, safe to call from multiple event-loop tasks."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window_s = float(window_s)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, bucket: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            events = self._events[bucket]
            cutoff = now - self.window_s
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_s - (now - events[0])))
                return False, retry_after
            events.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
