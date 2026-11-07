from __future__ import annotations

import random

import pytest

from agents_app import vector_store
from agents_app.vector_store import BuiltinVectorStore, make_vector_store


def _random_vectors(count: int = 40, dim: int = 32, seed: int = 7) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(count)]


def _query(seed: int = 11, dim: int = 32) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dim)]


def test_builtin_search_is_an_exact_inner_product():
    store = BuiltinVectorStore()
    store.build([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    results = store.search([1.0, 0.0])

    assert [index for index, _ in results] == [0, 2, 1]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(2**-0.5)


def test_builtin_top_k_limits_results():
    store = BuiltinVectorStore()
    store.build(_random_vectors())
    assert len(store.search(_query(), top_k=3)) == 3


def test_faiss_matches_builtin_ranking_and_scores():
    pytest.importorskip("faiss")
    vectors = _random_vectors()
    builtin = BuiltinVectorStore()
    builtin.build(vectors)

    faiss_store = make_vector_store("faiss")
    assert faiss_store.name == "faiss"
    faiss_store.build(vectors)

    for seed in range(5):
        expected = builtin.search(_query(seed))
        actual = faiss_store.search(_query(seed))
        assert [index for index, _ in actual] == [index for index, _ in expected]
        assert [score for _, score in actual] == pytest.approx(
            [score for _, score in expected], abs=1e-5
        )


def test_faiss_persistence_round_trip(tmp_path):
    pytest.importorskip("faiss")
    vectors = _random_vectors()
    store = make_vector_store("faiss")
    store.build(vectors)
    store.save(tmp_path, "fingerprint-1")

    assert store.stats()["persisted"] is True
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "index.meta.json").exists()

    reloaded = make_vector_store("faiss")
    assert reloaded.load(tmp_path, "fingerprint-1") is True
    assert reloaded.stats()["vectors"] == len(vectors)
    assert [
        index for index, _ in reloaded.search(_query(3))
    ] == [index for index, _ in store.search(_query(3))]

    stale = make_vector_store("faiss")
    assert stale.load(tmp_path, "different-fingerprint") is False


def test_faiss_missing_package_falls_back(monkeypatch):
    def broken_import():
        raise ImportError("faiss is not installed")

    monkeypatch.setattr(vector_store, "_import_faiss", broken_import)

    assert make_vector_store("faiss").name == "builtin"
    assert make_vector_store("auto").name == "builtin"


def test_unknown_or_disabled_backend_uses_builtin():
    assert make_vector_store("builtin").name == "builtin"
    assert make_vector_store("not-a-backend").name == "builtin"
