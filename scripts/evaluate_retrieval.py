#!/usr/bin/env python3
"""Evaluate a JSON fixture without contacting a provider or corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.contracts import CandidateIdentity, EvaluationCase, RetrievalResult
from app.retrieval.benchmark import evaluate


class FixtureRetriever:
    def __init__(self, results: dict[str, RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, case: EvaluationCase) -> RetrievalResult:
        try:
            return self.results[case.case_id]
        except KeyError as exc:
            raise ValueError(f"fixture has no retrieval result for {case.case_id}") from exc


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    candidate = CandidateIdentity.model_validate(payload["candidate"])

    def cases(key: str) -> list[EvaluationCase]:
        return [EvaluationCase.model_validate(item) for item in payload.get(key, [])]

    all_cases = cases("cases")
    tuning = cases("tuning_cases")
    held_out = cases("held_out_cases")
    raw_results = payload.get("results", {})
    results = {
        case_id: RetrievalResult.model_validate({"candidate": candidate, **result})
        for case_id, result in raw_results.items()
    }
    report = evaluate(
        all_cases,
        FixtureRetriever(results),
        candidate=candidate,
        dataset_id=str(payload["dataset_id"]),
        tuning_cases=tuning,
        held_out_cases=held_out,
        strategy=str(payload.get("strategy", "dense")),
    )
    return report.as_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = load_report(args.fixture)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
