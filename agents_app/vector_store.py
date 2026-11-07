"""Pluggable vector-index backends for the knowledge base.

The built-in store is an exact dot-product scan over the normalised chunk
vectors. It needs no dependencies and is the default, which keeps the base
install and the offline test suite unchanged.

``FaissVectorStore`` is an opt-in backend. It uses FAISS ``IndexFlatIP`` over
the same normalised vectors, so the ranking is mathematically identical to the
built-in scan; the benefits are index persistence and lower search overhead at
larger corpus sizes. Install it with ``pip install -r requirements-rag.txt``.

Persistence layout (``INDEX_DIR``)::

    index.faiss        FAISS serialised index
    index.meta.json    version, fingerprint, dimension, vector count

The fingerprint covers the knowledge-base files and the embedding/chunking
configuration. If anything changes, the index is rebuilt instead of silently
serving stale vectors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

INDEX_FILE = "index.faiss"
META_FILE = "index.meta.json"
META_VERSION = 1


def _import_faiss():
    """Import FAISS lazily so the base install never needs it."""
    import faiss  # noqa: PLC0415 - optional dependency, imported on demand

    return faiss


def _normalised(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class BuiltinVectorStore:
    """Exact inner-product search with no third-party dependency."""

    name = "builtin"

    def __init__(self) -> None:
        self._vectors: list[list[float]] = []

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        self._vectors = [_normalised(vector) for vector in vectors]

    def search(
        self,
        query: Sequence[float],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        normalised_query = _normalised(query)
        scores = [
            sum(a * b for a, b in zip(normalised_query, vector, strict=False))
            for vector in self._vectors
        ]
        ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        if top_k is not None:
            ranking = ranking[:top_k]
        return [(index, scores[index]) for index in ranking]

    def vectors(self) -> list[list[float]]:
        return [list(vector) for vector in self._vectors]

    def load(self, directory: Path, fingerprint: str) -> bool:
        return False

    def save(self, directory: Path, fingerprint: str) -> None:
        return None

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "vectors": len(self._vectors),
            "persisted": False,
        }


class FaissVectorStore:
    """FAISS ``IndexFlatIP`` over the same normalised vectors.

    ``IndexFlatIP`` is an exact index, not an approximate one. At this corpus
    size that means the returned ranking matches the built-in scan; parity is
    asserted in ``tests/test_vector_store.py``.
    """

    name = "faiss"

    def __init__(self) -> None:
        self._faiss = _import_faiss()
        import numpy as np  # noqa: PLC0415 - FAISS already depends on numpy

        self._np = np
        self._index = None
        self._dim = 0
        self._fingerprint = ""
        self._path: Path | None = None

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            self._index = None
            self._dim = 0
            return
        self._dim = len(vectors[0])
        index = self._faiss.IndexFlatIP(self._dim)
        matrix = self._np.asarray(
            [_normalised(vector) for vector in vectors],
            dtype="float32",
        )
        index.add(matrix)
        self._index = index

    def search(
        self,
        query: Sequence[float],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        limit = int(top_k) if top_k is not None else int(self._index.ntotal)
        limit = max(1, min(limit, int(self._index.ntotal)))
        query_matrix = self._np.asarray([_normalised(query)], dtype="float32")
        scores, indices = self._index.search(query_matrix, limit)
        return [
            (int(index), float(score))
            for score, index in zip(scores[0], indices[0], strict=False)
            if int(index) >= 0
        ]

    def vectors(self) -> list[list[float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        return self._index.reconstruct_n(0, int(self._index.ntotal)).tolist()

    def save(self, directory: Path, fingerprint: str) -> None:
        if self._index is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(directory / INDEX_FILE))
        meta = {
            "version": META_VERSION,
            "backend": self.name,
            "fingerprint": fingerprint,
            "dim": int(self._index.d),
            "count": int(self._index.ntotal),
        }
        (directory / META_FILE).write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._fingerprint = fingerprint
        self._path = directory

    def load(self, directory: Path, fingerprint: str) -> bool:
        index_path = directory / INDEX_FILE
        meta_path = directory / META_FILE
        if not index_path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            meta.get("version") != META_VERSION
            or meta.get("backend") != self.name
            or meta.get("fingerprint") != fingerprint
        ):
            return False
        try:
            index = self._faiss.read_index(str(index_path))
        except (OSError, RuntimeError, ValueError):
            return False
        if int(getattr(index, "ntotal", -1)) != int(meta.get("count", -2)):
            return False
        if int(getattr(index, "d", -1)) != int(meta.get("dim", -2)):
            return False
        self._index = index
        self._dim = int(index.d)
        self._fingerprint = fingerprint
        self._path = directory
        return True

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "vectors": int(self._index.ntotal) if self._index is not None else 0,
            "dim": self._dim,
            "persisted": self._path is not None,
            "path": str(self._path) if self._path is not None else "",
        }


def make_vector_store(backend: str) -> BuiltinVectorStore | FaissVectorStore:
    """Return the requested store, falling back to the built-in one.

    ``auto`` uses FAISS when the optional package is importable. ``faiss``
    requests it explicitly but still degrades cleanly rather than breaking the
    whole service when the wheel is missing.
    """

    wanted = (backend or "builtin").strip().lower()
    if wanted == "builtin":
        return BuiltinVectorStore()
    if wanted not in {"auto", "faiss"}:
        return BuiltinVectorStore()
    try:
        return FaissVectorStore()
    except Exception:  # noqa: BLE001 - missing package or unsupported platform
        return BuiltinVectorStore()
