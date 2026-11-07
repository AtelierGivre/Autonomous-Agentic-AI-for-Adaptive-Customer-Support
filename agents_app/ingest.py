"""Pluggable document chunkers for the knowledge base.

``BuiltinIngestor`` is the original paragraph packer: it splits on blank lines
and packs paragraphs up to ``CHUNK_SIZE`` characters with a word-boundary
overlap. It is dependency-free and remains the default.

``LlamaIndexIngestor`` wraps LlamaIndex's ``SentenceSplitter``. It is opt-in
(``INGEST_BACKEND=llamaindex``) and calibrated in characters, not tokens, so
``CHUNK_SIZE`` and ``CHUNK_OVERLAP`` keep the same meaning as in the built-in
path. The tokenizer is a plain ``list(text)`` and the sentence splitter is a
local regex, so nothing downloads ``tiktoken``/NLTK data at import or query
time.

Both implementations return plain strings; the caller owns embedding,
tokenisation and scoring.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

_SPACE_RE = re.compile(r"[ \t]+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Sentence-sized pieces that keep their trailing whitespace. ``SentenceSplitter``
# joins the pieces it is given with an empty string, so the whitespace must
# stay attached.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+[\s]*|[^.!?]+$")


class Ingestor(Protocol):
    name: str

    def split(self, text: str) -> list[str]: ...


def _char_tokenizer(text: str) -> list[str]:
    """Measure chunk size in characters, matching the built-in settings."""
    return list(text)


def _split_sentences(text: str) -> list[str]:
    pieces = [piece for piece in _SENTENCE_RE.findall(text) if piece]
    return pieces or [text]


class BuiltinIngestor:
    """Paragraph packer used by the base install."""

    name = "builtin"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))

    def _tail_overlap(self, text: str) -> str:
        if not self.chunk_overlap:
            return ""
        tail = text[-self.chunk_overlap :]
        # Start on a word boundary so a chunk never opens with a half word.
        if " " not in tail:
            return ""
        return tail[tail.index(" ") + 1 :].strip()

    def split(self, text: str) -> list[str]:
        text = _SPACE_RE.sub(" ", text).strip()
        paragraphs = [
            paragraph.strip()
            for paragraph in _PARAGRAPH_RE.split(text)
            if paragraph.strip()
        ]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > self.chunk_size:
                chunks.append(current)
                overlap = self._tail_overlap(current)
                current = f"{overlap}\n\n{paragraph}".strip()
            else:
                current = f"{current}\n\n{paragraph}".strip()
        if current:
            chunks.append(current)
        return chunks or [text]


class LlamaIndexIngestor:
    """Sentence-aware chunking through LlamaIndex's ``SentenceSplitter``."""

    name = "llamaindex"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        from llama_index.core.node_parser import SentenceSplitter  # noqa: PLC0415

        size = max(1, int(chunk_size))
        overlap = max(0, min(int(chunk_overlap), size - 1))
        self.chunk_size = size
        self.chunk_overlap = overlap
        self._splitter = SentenceSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            tokenizer=_char_tokenizer,
            chunking_tokenizer_fn=_split_sentences,
            paragraph_separator="\n\n",
        )

    def split(self, text: str) -> list[str]:
        text = _SPACE_RE.sub(" ", text).strip()
        if not text:
            return []
        chunks = self._splitter.split_text(text)
        return chunks or [text]


def make_ingestor(
    backend: str,
    chunk_size: int,
    chunk_overlap: int,
) -> BuiltinIngestor | LlamaIndexIngestor:
    """Return the requested chunker, falling back to the built-in one.

    ``auto`` uses LlamaIndex when the optional package is importable. A failed
    import never breaks retrieval; the paragraph packer takes over.
    """

    wanted = (backend or "builtin").strip().lower()
    if wanted == "builtin":
        return BuiltinIngestor(chunk_size, chunk_overlap)
    if wanted not in {"auto", "llamaindex"}:
        return BuiltinIngestor(chunk_size, chunk_overlap)
    try:
        return LlamaIndexIngestor(chunk_size, chunk_overlap)
    except Exception:  # noqa: BLE001 - missing package or incompatible version
        return BuiltinIngestor(chunk_size, chunk_overlap)


def coverage_by_words(original: str, chunks: Sequence[str]) -> float:
    """Fraction of original words preserved in the chunks (diagnostics/tests)."""
    words = original.split()
    if not words:
        return 1.0
    joined = " ".join(chunks)
    remaining = list(joined.split())
    preserved = 0
    for word in words:
        try:
            remaining.remove(word)
        except ValueError:
            continue
        preserved += 1
    return preserved / len(words)
