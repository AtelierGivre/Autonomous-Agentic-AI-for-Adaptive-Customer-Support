"""Hybrid retrieval over the local knowledge base.

Vector search (Hugging Face embeddings when available, deterministic hashing
otherwise) is fused with BM25 using reciprocal rank fusion. A confidence gate
uses both the fused score and lexical coverage, so "I don't know" is a
first-class answer rather than a hallucinated one.

No network is required at query time. If ``sentence-transformers`` is not
installed the hash embedder keeps the whole pipeline runnable offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from agents_app.config import Settings
from agents_app.config import settings as default_settings
from agents_app.guardrails import looks_suspicious_document
from agents_app.ingest import make_ingestor
from agents_app.vector_store import make_vector_store

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do",
    "does", "for", "from", "get", "got", "has", "have", "help", "how", "i", "if",
    "in", "is", "it", "its", "keep", "keeps", "kept", "me", "might", "must", "my",
    "need", "needs", "of", "on", "or", "our", "please", "shall", "should", "so",
    "that", "the", "their", "then", "there", "these", "this", "to", "want", "wants",
    "was", "we", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
}

# Small domain thesaurus so a customer's wording still reaches the right chunk.
# Applied to the query only (not the index), which keeps it predictable.
# Keys and values are already stemmed (see ``_stem``); entries are applied to
# the query only, which keeps the index predictable.
_QUERY_SYNONYMS = {
    "twic": ("duplicat",),
    "doubl": ("duplicat",),
    "parcel": ("order", "delivery"),
    "packag": ("order", "delivery"),
}


def tokenize(text: str) -> list[str]:
    words = [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]
    return [_stem(t) for t in words]


def expand_query(text: str) -> list[str]:
    """Tokenise a question and add the small domain thesaurus entries."""
    tokens = tokenize(text)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_QUERY_SYNONYMS.get(token, ()))
    return list(dict.fromkeys(expanded))


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'disconnecting' matches 'disconnects'."""
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    # "change"/"changed" and "charge"/"charged" should land on one form.
    if len(token) >= 5 and token.endswith("e"):
        token = token[:-1]
    return token


def _hash_embed(tokens: list[str], dim: int = 256) -> list[float]:
    vector = [0.0] * dim
    features = list(tokens)
    features += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]  # bigrams keep phrases together
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dim
        sign = 1.0 if (value >> 8) & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class _HFEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            list(map(float, row))
            for row in self.model.encode(texts, normalize_embeddings=True).tolist()
        ]


@dataclass
class Chunk:
    source: str
    text: str
    tokens: list[str]
    vector: list[float]


class KnowledgeBase:
    def __init__(self, settings: Settings | None = None, kb_dir: str | None = None) -> None:
        self.settings = settings or default_settings
        self.kb_dir = Path(kb_dir or self.settings.knowledge_base_dir)
        self.chunks: list[Chunk] = []
        self.backend = "hash"
        self._embedder: _HFEmbedder | None = None
        self._signature: tuple = ()
        self._avg_len = 1.0
        self._df: dict[str, int] = {}
        self._min_score = self.settings.min_retrieval_score
        self._ingestor = make_ingestor(
            self.settings.ingest_backend,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        self._vector_store = make_vector_store(self.settings.vector_store)
        self.build()

    # ------------------------------------------------------------------ indexing
    def _resolve_embedder(self) -> None:
        wanted = self.settings.embedding_backend
        if wanted not in {"hf", "auto", "sentence-transformers"}:
            return
        try:
            self._embedder = _HFEmbedder(self.settings.hf_embedding_model)
            self.backend = "hf"
        except Exception:  # noqa: BLE001 - missing package/weights falls back cleanly
            self._embedder = None
            self.backend = "hash"

    def _files(self) -> list[Path]:
        if not self.kb_dir.exists():
            return []
        return sorted(self.kb_dir.glob("*.md"))

    def _current_signature(self) -> tuple:
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in self._files())

    def _split(self, text: str) -> list[str]:
        return self._ingestor.split(text)

    @property
    def ingest_backend(self) -> str:
        return self._ingestor.name

    @property
    def vector_store_backend(self) -> str:
        return self._vector_store.name

    def _fingerprint(self) -> str:
        """Identity of the corpus plus the retrieval configuration."""
        payload = {
            "files": self._signature,
            "embedding": self.backend,
            "hf_model": self.settings.hf_embedding_model if self.backend == "hf" else "",
            "ingestor": self._ingestor.name,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "vector_store": self._vector_store.name,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def build(self) -> None:
        self._resolve_embedder()
        self._signature = self._current_signature()
        documents: list[tuple[str, str]] = []
        for path in self._files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for chunk_text in self._split(text):
                if self.settings.kb_skip_suspicious and looks_suspicious_document(chunk_text):
                    continue
                documents.append((path.name, chunk_text))

        fingerprint = self._fingerprint()
        vectors: list[list[float]] | None = None
        if self._vector_store.name == "faiss" and self.settings.persist_index:
            if self._vector_store.load(Path(self.settings.index_dir), fingerprint):
                vectors = self._vector_store.vectors()
                if len(vectors) != len(documents):
                    vectors = None

        if vectors is None:
            if self._embedder and documents:
                vectors = self._embedder.encode([text for _, text in documents])
            else:
                self.backend = "hash"
                vectors = [_hash_embed(tokenize(text)) for _, text in documents]
            self._vector_store.build(vectors)
            if self._vector_store.name == "faiss" and self.settings.persist_index:
                self._vector_store.save(Path(self.settings.index_dir), fingerprint)

        self.chunks = [
            Chunk(source=source, text=text, tokens=tokenize(text), vector=vector)
            for (source, text), vector in zip(documents, vectors, strict=True)
        ]

        # BM25 corpus statistics.
        self._df = {}
        for chunk in self.chunks:
            for token in set(chunk.tokens):
                self._df[token] = self._df.get(token, 0) + 1
        self._avg_len = (sum(len(c.tokens) for c in self.chunks) / len(self.chunks)) if self.chunks else 1.0
        if self.backend == "hash" and "MIN_RETRIEVAL_SCORE" not in __import__("os").environ:
            # Hash similarity has a different scale from sentence embeddings.
            self._min_score = min(self._min_score, 0.10)

    def refresh_if_changed(self) -> bool:
        if self._current_signature() != self._signature:
            self.build()
            return True
        return False

    def is_ready(self) -> bool:
        return bool(self.chunks)

    # ------------------------------------------------------------------ search
    def _bm25(self, query_tokens: list[str]) -> list[float]:
        k1, b = 1.5, 0.75
        scores = []
        total = max(1, len(self.chunks))
        for chunk in self.chunks:
            length = len(chunk.tokens) or 1
            score = 0.0
            counts: dict[str, int] = {}
            for token in chunk.tokens:
                counts[token] = counts.get(token, 0) + 1
            for token in query_tokens:
                if token not in counts:
                    continue
                df = self._df.get(token, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                tf = counts[token]
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / self._avg_len))
            scores.append(score)
        peak = max(scores) if scores else 0.0
        return [s / peak if peak else 0.0 for s in scores]

    def search(
        self,
        query: str,
        k: int | None = None,
        prefer: tuple[str, ...] | None = None,
        prefer_bonus: float = 0.08,
    ) -> list[dict]:
        self.refresh_if_changed()
        if not self.chunks:
            return []
        k = k or self.settings.retrieval_top_k
        query_tokens = expand_query(query)
        if not query_tokens:
            return []
        query_vector = (
            self._embedder.encode([query])[0] if self._embedder else _hash_embed(query_tokens)
        )

        vector_scores = [0.0] * len(self.chunks)
        vector_rank: list[int] = []
        for index, score in self._vector_store.search(query_vector):
            if 0 <= index < len(self.chunks):
                vector_scores[index] = score
                vector_rank.append(index)
        lexical_scores = self._bm25(query_tokens)
        lexical_rank = sorted(range(len(self.chunks)), key=lambda i: lexical_scores[i], reverse=True)
        fused = [0.0] * len(self.chunks)
        for ranking, weight in ((vector_rank, 0.65), (lexical_rank, 0.35)):
            for rank, index in enumerate(ranking):
                fused[index] += weight / (60 + rank + 1)
        peak = max(fused) or 1.0

        query_set = set(query_tokens)
        results = []
        for index in range(len(self.chunks)):
            chunk = self.chunks[index]
            coverage = len(query_set & set(chunk.tokens)) / max(1, len(query_set))
            # Blend relevance with rank fusion and query coverage so a chunk
            # that actually contains the asked-about terms outranks a chunk
            # that merely shares a common word.
            score = (
                0.45 * max(0.0, vector_scores[index])
                + 0.20 * lexical_scores[index]
                + 0.15 * (fused[index] / peak)
                + 0.20 * coverage
            )
            if prefer and chunk.source in prefer:
                score = min(1.0, score + prefer_bonus)
            results.append(
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": round(min(1.0, score), 4),
                    "coverage": round(coverage, 4),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def confidence(self, results: list[dict]) -> tuple[float, float]:
        if not results:
            return 0.0, 0.0
        return float(results[0]["score"]), float(results[0]["coverage"])

    def is_confident(self, results: list[dict]) -> bool:
        score, coverage = self.confidence(results)
        return score >= self._min_score and coverage >= self.settings.min_lexical_coverage

    def stats(self) -> dict:
        return {
            "documents": len({chunk.source for chunk in self.chunks}),
            "files": len(self._files()),
            "chunks": len(self.chunks),
            "backend": self.backend,
            "ingest_backend": self._ingestor.name,
            "vector_store": self._vector_store.name,
            "ready": self.is_ready(),
            "min_score": self._min_score,
            "min_coverage": self.settings.min_lexical_coverage,
            "index": self._vector_store.stats(),
        }
