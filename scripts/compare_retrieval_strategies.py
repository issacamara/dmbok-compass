#!/usr/bin/env python3
"""Make a fail-closed hybrid-versus-dense adoption decision from held-out data.

The input is a JSON object. It intentionally contains only aggregate counts and
gate statuses: this script records a decision; it never tunes a strategy or
changes serving configuration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


REQUIRED_GATES = (
    "sm_02",
    "sm_05",
    "sm_06",
    "memory",
    "cold_start",
    "privacy",
    "approved_cost",
)
MINIMUM_SLICE_CASES = 40
MINIMUM_SLICE_GAIN_PERCENTAGE_POINTS = 5.0


@dataclass(frozen=True)
class StrategyEvidence:
    candidate_id: str
    strategy_id: str
    calibration_policy_version: str
    overall_cases: int
    overall_hits: int
    slice_id: str
    slice_digest: str
    slice_cases: int
    slice_hits: int

    @property
    def overall_hit_at_5(self) -> float:
        return _percentage(self.overall_hits, self.overall_cases)

    @property
    def slice_hit_at_5(self) -> float:
        return _percentage(self.slice_hits, self.slice_cases)


def compare(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable adoption decision without modifying either strategy."""

    dense = _strategy(payload, "dense")
    hybrid = _strategy(payload, "hybrid")
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("gates must be an object")

    reasons: list[str] = []
    _compare_identity(dense, hybrid, reasons)
    _check_sample_and_gain(dense, hybrid, reasons)
    _check_calibration(dense, hybrid, reasons)
    _check_gates(gates, reasons)

    recommendation = "recommend-hybrid" if not reasons else "retain-dense-only"
    return {
        "schema_version": "hybrid-adoption-decision-v1",
        "decision": recommendation,
        "dense": _strategy_report(dense),
        "hybrid": _strategy_report(hybrid),
        "slice_gain_percentage_points": round(
            hybrid.slice_hit_at_5 - dense.slice_hit_at_5, 2
        ),
        "gates": {gate: gates.get(gate, "unknown") for gate in REQUIRED_GATES},
        "reasons": reasons,
        "serving_change": "none",
        "approval_boundary": (
            "A hybrid recommendation creates a distinct non-production candidate "
            "and cannot reuse dense candidate approval."
        ),
    }


def render_markdown(decision: dict[str, Any]) -> str:
    """Render the portable evidence record stored with a decision."""

    dense = decision["dense"]
    hybrid = decision["hybrid"]
    reasons = decision["reasons"] or ["All prespecified comparison gates passed."]
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)
    return f"""# Hybrid retrieval adoption experiment\n\nStatus: **{decision['decision']}**\n\nThis is an offline decision record. It does not enable hybrid serving, tune either\nstrategy, or publish a release.\n\n## Prespecified held-out comparison\n\n| Measure | Dense baseline | Hybrid candidate |\n| --- | ---: | ---: |\n| Candidate identity | `{dense['candidate_id']}` | `{hybrid['candidate_id']}` |\n| Strategy identity | `{dense['strategy_id']}` | `{hybrid['strategy_id']}` |\n| Calibration policy | `{dense['calibration_policy_version']}` | `{hybrid['calibration_policy_version']}` |\n| Overall hit@5 | {dense['overall_hit_at_5_percentage']:.2f}% | {hybrid['overall_hit_at_5_percentage']:.2f}% |\n| Held-out slice (`{dense['slice_id']}`) | {dense['slice_hit_at_5_percentage']:.2f}% ({dense['slice_cases']} cases) | {hybrid['slice_hit_at_5_percentage']:.2f}% ({hybrid['slice_cases']} cases) |\n\nThe held-out slice digest is `{dense['slice_digest']}`. Hybrid's slice gain is\n{decision['slice_gain_percentage_points']:.2f} percentage points; the minimum is\n{MINIMUM_SLICE_GAIN_PERCENTAGE_POINTS:.2f} percentage points with no overall loss.\n\n## Gate result\n\n{reason_lines}\n\n## Approval boundary\n\n{decision['approval_boundary']} Dense-only release has no dependency on this\noptional experiment.\n"""


def _strategy(payload: dict[str, Any], name: str) -> StrategyEvidence:
    source = payload.get(name)
    if not isinstance(source, dict):
        raise ValueError(f"{name} must be an object")
    slice_data = source.get("held_out_slice")
    if not isinstance(slice_data, dict):
        raise ValueError(f"{name}.held_out_slice must be an object")
    return StrategyEvidence(
        candidate_id=_required_string(source, "candidate_id", name),
        strategy_id=_required_string(source, "strategy_id", name),
        calibration_policy_version=_required_string(
            source, "calibration_policy_version", name
        ),
        overall_cases=_count(source, "overall_cases", name),
        overall_hits=_count(source, "overall_hits", name),
        slice_id=_required_string(slice_data, "id", f"{name}.held_out_slice"),
        slice_digest=_required_string(
            slice_data, "digest", f"{name}.held_out_slice"
        ),
        slice_cases=_count(slice_data, "cases", f"{name}.held_out_slice"),
        slice_hits=_count(slice_data, "hits", f"{name}.held_out_slice"),
    )


def _required_string(source: dict[str, Any], key: str, context: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _count(source: dict[str, Any], key: str, context: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer")
    return value


def _percentage(hits: int, cases: int) -> float:
    if hits > cases:
        raise ValueError("hits cannot exceed cases")
    if cases < 1:
        return 0.0
    return hits / cases * 100


def _compare_identity(
    dense: StrategyEvidence, hybrid: StrategyEvidence, reasons: list[str]
) -> None:
    if dense.slice_id != hybrid.slice_id or dense.slice_digest != hybrid.slice_digest:
        reasons.append("strategies were not compared on the same prespecified held-out slice")
    if dense.slice_cases != hybrid.slice_cases:
        reasons.append("strategies have different held-out slice case counts")
    if dense.candidate_id == hybrid.candidate_id:
        reasons.append("hybrid must have a distinct candidate identity from dense")


def _check_sample_and_gain(
    dense: StrategyEvidence, hybrid: StrategyEvidence, reasons: list[str]
) -> None:
    if dense.slice_cases < MINIMUM_SLICE_CASES:
        reasons.append(
            f"held-out slice has {dense.slice_cases} cases; at least {MINIMUM_SLICE_CASES} are required"
        )
    gain = hybrid.slice_hit_at_5 - dense.slice_hit_at_5
    if gain < MINIMUM_SLICE_GAIN_PERCENTAGE_POINTS:
        reasons.append(
            f"hybrid slice hit@5 gain is {gain:.2f}pp; at least "
            f"{MINIMUM_SLICE_GAIN_PERCENTAGE_POINTS:.2f}pp is required"
        )
    if hybrid.overall_hit_at_5 < dense.overall_hit_at_5:
        reasons.append("hybrid has an overall hit@5 loss versus dense")


def _check_calibration(
    dense: StrategyEvidence, hybrid: StrategyEvidence, reasons: list[str]
) -> None:
    if dense.calibration_policy_version == hybrid.calibration_policy_version:
        reasons.append("strategy-specific score calibration must use distinct policy versions")


def _check_gates(gates: dict[str, Any], reasons: list[str]) -> None:
    for gate in REQUIRED_GATES:
        if gates.get(gate) != "passed":
            reasons.append(f"{gate} gate is {gates.get(gate, 'unknown')!r}, not 'passed'")


def _strategy_report(strategy: StrategyEvidence) -> dict[str, Any]:
    return {
        "candidate_id": strategy.candidate_id,
        "strategy_id": strategy.strategy_id,
        "calibration_policy_version": strategy.calibration_policy_version,
        "overall_cases": strategy.overall_cases,
        "overall_hits": strategy.overall_hits,
        "overall_hit_at_5_percentage": round(strategy.overall_hit_at_5, 2),
        "slice_id": strategy.slice_id,
        "slice_digest": strategy.slice_digest,
        "slice_cases": strategy.slice_cases,
        "slice_hits": strategy.slice_hits,
        "slice_hit_at_5_percentage": round(strategy.slice_hit_at_5, 2),
    }


def _read_input(path: str) -> dict[str, Any]:
    stream: TextIO
    if path == "-":
        import sys

        stream = sys.stdin
        payload = json.load(stream)
    else:
        payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="comparison JSON path, or - for standard input")
    parser.add_argument("--output", type=Path, help="write the decision JSON to this path")
    parser.add_argument("--markdown-output", type=Path, help="write the evidence record to this path")
    args = parser.parse_args()
    decision = compare(_read_input(args.input))
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    if args.markdown_output:
        args.markdown_output.write_text(render_markdown(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
