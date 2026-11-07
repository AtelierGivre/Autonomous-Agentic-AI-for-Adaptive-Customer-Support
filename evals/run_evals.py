"""Offline golden-dataset quality gate.

Usage::

    python -m evals.run_evals --min-pass 0.9

Each line of ``evals/dataset.jsonl`` is one conversation. The runner starts from
an empty memory per case, replays every turn, and checks the *final* response
against the expectations in the record. It needs no API key and no network:
set ``LLM_PROVIDER=mock`` and ``EMBEDDING_BACKEND=hash`` (the defaults below).

Exit code is 0 when the pass rate meets ``--min-pass``, otherwise 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before agents_app.config is imported.
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("MIN_RETRIEVAL_SCORE", "0.1")
os.environ.setdefault("MIN_LEXICAL_COVERAGE", "0.5")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG_RESPONSES", "true")
os.environ.setdefault("DEFAULT_CUSTOMER_ID", "demo")
os.environ.setdefault("API_KEYS", "")

from agents_app.config import settings  # noqa: E402
from agents_app.memory import ConversationMemory  # noqa: E402
from agents_app.workflow import SupportWorkflow  # noqa: E402

DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")
DEFAULT_REPORT = Path(__file__).with_name("report.json")


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
        if not case.get("id") or not case.get("turns"):
            raise SystemExit(f"{path}:{line_number}: every case needs an id and turns")
        if not case.get("expect"):
            # Feedback candidates start as {"expect": {}}. Refusing them here is
            # what stops an unlabelled export from turning into a free pass.
            raise SystemExit(
                f"{path}:{line_number}: case {case['id']!r} has no expectations; "
                "label it before adding it to the dataset"
            )
        cases.append(case)
    return cases


def _tool_found(result: dict) -> bool | None:
    calls = result.get("tool_calls") or []
    if not calls:
        return None
    return bool((calls[0].get("result") or {}).get("found"))


def check(case: dict, result: dict) -> list[str]:
    expect = case.get("expect") or {}
    answer = (result.get("answer") or "").lower()
    failures: list[str] = []

    if "outcome" in expect and result.get("outcome") != expect["outcome"]:
        failures.append(f"outcome {result.get('outcome')!r} != {expect['outcome']!r}")
    if "verified" in expect and bool(result.get("verified")) is not bool(expect["verified"]):
        failures.append(f"verified {result.get('verified')!r} != {expect['verified']!r}")
    if "needs_confirmation" in expect:
        if bool(result.get("needs_confirmation")) is not bool(expect["needs_confirmation"]):
            failures.append(
                f"needs_confirmation {result.get('needs_confirmation')!r} "
                f"!= {expect['needs_confirmation']!r}"
            )
    for phrase in expect.get("contains", []):
        if phrase.lower() not in answer:
            failures.append(f"answer is missing {phrase!r}")
    for phrase in expect.get("not_contains", []):
        if phrase.lower() in answer:
            failures.append(f"answer unexpectedly contains {phrase!r}")

    sources = result.get("sources") or []
    if "min_sources" in expect and len(sources) < expect["min_sources"]:
        failures.append(f"{len(sources)} sources < {expect['min_sources']}")
    if "max_sources" in expect and len(sources) > expect["max_sources"]:
        failures.append(f"{len(sources)} sources > {expect['max_sources']}")

    debug = result.get("debug") or {}
    metrics = debug.get("metrics") or {}
    if "max_llm_calls" in expect and metrics.get("llm_calls", 0) > expect["max_llm_calls"]:
        failures.append(f"{metrics.get('llm_calls')} LLM calls > {expect['max_llm_calls']}")
    if "min_pii_redactions" in expect:
        if debug.get("pii_redactions", 0) < expect["min_pii_redactions"]:
            failures.append(
                f"{debug.get('pii_redactions', 0)} PII redactions "
                f"< {expect['min_pii_redactions']}"
            )
    if "tool_found" in expect:
        found = _tool_found(result)
        if found is not expect["tool_found"]:
            failures.append(f"tool_found {found!r} != {expect['tool_found']!r}")

    return failures


async def run(dataset: Path, report_path: Path) -> dict:
    cases = load_cases(dataset)
    workflow = SupportWorkflow()
    rows: list[dict] = []
    passed = 0

    for case in cases:
        memory = ConversationMemory()
        result: dict = {}
        error = ""
        try:
            for turn in case["turns"]:
                result = await workflow.handle(turn, memory)
        except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a stopped run
            error = f"{type(exc).__name__}: {exc}"

        failures = [error] if error else check(case, result)
        if not failures:
            passed += 1
        rows.append(
            {
                "id": case["id"],
                "turns": case["turns"],
                "outcome": result.get("outcome"),
                "pass": not failures,
                "failures": failures,
                "answer": result.get("answer"),
            }
        )

    total = len(rows)
    pass_rate = round(passed / total, 4) if total else 0.0
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": settings.resolved_provider,
        "embedding_backend": workflow.rag.backend,
        "dataset": str(dataset),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "cases": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline golden-dataset quality gate.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--min-pass",
        type=float,
        default=0.9,
        help="minimum pass rate required to exit 0 (default: 0.9)",
    )
    args = parser.parse_args(argv)

    report = asyncio.run(run(args.dataset, args.report))

    for row in report["cases"]:
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"{mark}  {row['id']:<28} {row['outcome']}")
        for failure in row["failures"]:
            print(f"      - {failure}")

    print(
        f"\n{report['passed']}/{report['total']} passed "
        f"({report['pass_rate']:.1%}) | provider={report['provider']} "
        f"| embeddings={report['embedding_backend']}"
    )
    print(f"report: {args.report}")

    if report["pass_rate"] < args.min_pass:
        print(f"FAILED quality gate: {report['pass_rate']:.1%} < {args.min_pass:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
