"""Guardrails: prompt-injection screening and untrusted-text fencing.

Heuristics, not a proof. They stop the common attacks cheaply and, importantly,
before the model is called (zero tokens spent on an obvious injection).
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(?:reveal|show|print|display|repeat)\s+(?:me\s+|us\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(?:a|an|in)\b", re.I),
    re.compile(r"\b(?:jailbreak|developer\s+mode|do\s+anything\s+now|DAN\s+mode)\b", re.I),
    re.compile(r"\bforget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions?|rules?)\b", re.I),
    re.compile(r"\bact\s+as\s+(?:a|an)\s+(?:unrestricted|uncensored|evil)\b", re.I),
    re.compile(r"<\s*/?\s*(?:system|assistant|developer)\s*>", re.I),
    re.compile(r"\boverride\s+(?:your\s+)?(?:safety|system)\b", re.I),
]

MAX_INPUT_CHARS = 4000


def screen_input(text: str) -> dict:
    """Return ``{blocked, category, reason}`` for a raw user message."""
    if len(text) > MAX_INPUT_CHARS:
        return {"blocked": True, "category": "length", "reason": "message too long"}
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return {
                "blocked": True,
                "category": "prompt_injection",
                "reason": f"matched {pattern.pattern}",
            }
    return {"blocked": False, "category": None, "reason": ""}


def looks_suspicious_document(text: str) -> bool:
    """Used to keep obviously poisoned knowledge-base docs out of the index."""
    return screen_input(text)["blocked"] and len(text) < MAX_INPUT_CHARS


def clean_untrusted(text: str) -> str:
    """Strip control characters and fence markers before prompt inclusion."""
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = text.replace("<untrusted>", "").replace("</untrusted>", "")
    return text.strip()[:2000]


def fence_untrusted(text: str) -> str:
    return f"<untrusted>\n{clean_untrusted(text)}\n</untrusted>"


def output_guard(text: str, max_chars: int = 4000) -> str:
    """Remove internal markers and cap size before returning to a customer."""
    text = (text or "").replace("<untrusted>", "").replace("</untrusted>", "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
