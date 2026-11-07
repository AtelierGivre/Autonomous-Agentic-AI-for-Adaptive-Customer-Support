"""Synthetic benchmark for the vector-store backends.

Usage::

    python -m evals.benchmark_rag
    python -m evals.benchmark_rag --backend faiss --sizes 1000,10000,100000

The benchmark isolates the index: it generates random normalised vectors, builds
the store, then times top-10 queries. It does not measure embedding, chunking or
LLM latency. Results are written to ``evals/benchmark_report.json``.

The built-in store is pure Python, so large synthetic runs are memory-hungry.
Use FAISS (``pip install -r requirements-rag.txt``) for the 100k row.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_app.vector_store import make_vector_store  # noqa: E402

DEFAULT_REPORT = Path(__file__).with_name("benchmark_report.json")


def _unit_vector(rng: random.Random, dim: int) -> list[float]:
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def benchmark(
    backend: str,
    sizes: list[int],
    dim: int,
    queries: int,
    top_k: int,
) -> list[dict]:
    rng = random.Random(7)
    rows: list[dict] = []
    for count in sizes:
        vectors = [_unit_vector(rng, dim) for _ in range(count)]
        store = make_vector_store(backend)
        build_start = time.perf_counter()
        store.build(vectors)
        build_seconds = time.perf_counter() - build_start

        query_vectors = [_unit_vector(rng, dim) for _ in range(queries)]
        latencies: list[float] = []
        for query_vector in query_vectors:
            start = time.perf_counter()
            store.search(query_vector, top_k=top_k)
            latencies.append((time.perf_counter() - start) * 1000)
        latencies.sort()

        row = {
            "backend": store.name,
            "vectors": count,
            "dim": dim,
            "build_seconds": round(build_seconds, 4),
            "query_p50_ms": round(statistics.median(latencies), 4),
            "query_p95_ms": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 4),
            "query_max_ms": round(latencies[-1], 4),
        }
        rows.append(row)
        print(
            f"{row['backend']:>7}  {count:>7} vectors  "
            f"build={row['build_seconds']:>8.3f}s  "
            f"p50={row['query_p50_ms']:>7.3f}ms  p95={row['query_p95_ms']:>7.3f}ms"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the vector-store backends.")
    parser.add_argument("--backend", default="builtin", choices=["builtin", "faiss", "auto"])
    parser.add_argument(
        "--sizes",
        default="1000,10000",
        help="comma-separated vector counts (default: 1000,10000)",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    rows = benchmark(args.backend, sizes, args.dim, args.queries, args.top_k)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
