from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.run_evals import DEFAULT_DATASET, load_cases


def test_golden_dataset_loads_and_unlabelled_cases_are_refused(tmp_path: Path):
    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) == 26
    assert all(case["expect"] for case in cases)

    candidate = tmp_path / "candidates.jsonl"
    candidate.write_text(
        json.dumps({"id": "feedback-abc", "status": "needs-label", "turns": ["hello"], "expect": {}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="has no expectations"):
        load_cases(candidate)
