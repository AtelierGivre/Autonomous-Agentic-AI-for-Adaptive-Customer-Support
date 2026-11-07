"""Retrieval-quality gate for the hybrid index.

Usage::

    python -m evals.run_retrieval --min-recall 0.8

Each line of ``evals/retrieval_cases.jsonl`` is one labelled query::

    {"id": "refund-timing", "query": "...", "expected_sources": ["billing.md"]}

The runner builds the knowledge base once, ranks the corpus for every query and
reports recall@1, recall@3, recall@5 and mean reciprocal rank (MRR). It is
backend-agnostic, so the same cases can be run against the built-in and optional
backends::

    VECTOR_STORE=faiss INGEST_BACKEND=llamaindex python -m evals.run_retrieval

Exit code is 0 when recall@3 meets ``--min-recall``, otherwise 1. The labelled
set is deliberately small: it detects ranking regressions, it is not a benchmark
of answer quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before agents_app.config is imported.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("MIN_RETRIEVAL_SCORE", "0.1")
os.environ.setdefault("MIN_LEXICAL_COVERAGE", "0.5")
os.environ.setdefault("VECTOR_STORE", "builtin")
os.environ.setdefault("INGEST_BACKEND", "builtin")

from agents_app.rag import KnowledgeBase  # noqa: E402

DEFAULT_CASES = Path(__file__).with_name("retrieval_cases.jsonl")
DEFAULT_REPORT = Path(__file__).with_name("retrieval_report.json")
KS = (1, 3, 5)


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not case.get("id") or not case.get("query") or not case.get("expected_sources"):
            raise SystemExit(
                f"{path}:{line_number}: every case needs id, query and expected_sources"
            )
        cases.append(case)
    if not cases:
        raise SystemExit(f"{path}: no cases found")
    return cases


def evaluate(kb: KnowledgeBase, cases: list[dict], max_k: int = 5) -> dict:
    rows: list[dict] = []
    hits = {k: 0 for k in KS}
    reciprocal_rank = 0.0

    for case in cases:
        expected = set(case["expected_sources"])
        results = kb.search(case["query"], k=max(KS[-1], max_k))
        sources = [row["source"] for row in results]
        first_rank = next(
            (rank for rank, source in enumerate(sources, start=1) if source in expected),
            None,
        )
        row = {
            "id": case["id"],
            "query": case["query"],
            "expected_sources": sorted(expected),
            "ranked_sources": sources,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
        }
        for k in KS:
            hit = bool(first_rank and first_rank <= k)
            row[f"recall@{k}"] = hit
            hits[k] += int(hit)
        reciprocal_rank += row["reciprocal_rank"]
        rows.append(row)

    total = len(rows)
    return {
        "total": total,
        "cases": rows,
        "recall@1": round(hits[1] / total, 4),
        "recall@3": round(hits[3] / total, 4),
        "recall@5": round(hits[5] / total, 4),
        "mrr": round(reciprocal_rank / total, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval-quality gate.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help="minimum recall@3 required to exit 0 (default: 0.8)",
    )
    args = parser.parse_args(argv)

    kb = KnowledgeBase()
    cases = load_cases(args.cases)
    metrics = evaluate(kb, cases)
    stats = kb.stats()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_backend": stats["backend"],
        "vector_store": stats["vector_store"],
        "ingest_backend": stats["ingest_backend"],
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        **metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for row in metrics["cases"]:
        mark = "PASS" if row["recall@3"] else "FAIL"
        first = row["first_relevant_rank"] or "-"
        print(f"{mark}  {row['id']:<28} first_relevant={first}  {row['ranked_sources'][:3]}")
        if not row["recall@3"]:
            print(f"      expected one of {row['expected_sources']}")

    print(
        f"\n{metrics['total']} queries | recall@1={metrics['recall@1']:.1%} "
        f"| recall@3={metrics['recall@3']:.1%} | recall@5={metrics['recall@5']:.1%} "
        f"| MRR={metrics['mrr']:.3f}"
    )
    print(
        f"backends: embeddings={report['embedding_backend']} "
        f"vector={report['vector_store']} ingest={report['ingest_backend']} "
        f"| chunks={report['chunks']}"
    )
    print(f"report: {args.report}")

    if report["recall@3"] < args.min_recall:
        print(f"FAILED retrieval gate: recall@3 {report['recall@3']:.1%} < {args.min_recall:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
