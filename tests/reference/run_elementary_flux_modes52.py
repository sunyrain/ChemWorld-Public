"""Prospective twelve-unit / 120-comparison bounded EFM development block."""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

from elementary_flux_modes52_cases import SPECIAL, STRUCTURAL, STRUCTURAL_FIELDS, source_scope_model
from elementary_flux_modes52_reference import exchanges, minimal_balanced

from chemworld.examples.sbml_flux_modes import flux_model, run_example
from chemworld.physchem.elementary_flux_modes import (
    FluxModePolicy,
    enumerate_elementary_flux_modes,
    sbml_elementary_flux_modes,
)
from chemworld.physchem.sbml_kinetics import SBMLKinetics


def modes(result):
    return sorted((tuple(int(v) for v in m["vector"]), m["bidirectional"]) for m in result["modes"])


def enumerated(case, **kwargs):
    return enumerate_elementary_flux_modes(
        case["matrix"],
        reaction_ids=tuple(f"r{i}" for i in range(len(case["irreversible"]))),
        irreversible=case["irreversible"],
        internal_species_ids=tuple(f"I{i}" for i in range(len(case["matrix"]))),
        external_matrix=case.get("external", ()),
        external_species_ids=tuple(f"E{i}" for i in range(len(case.get("external", ())))),
        **kwargs,
    )


def expected_structure(case):
    return dict(
        zip(
            STRUCTURAL_FIELDS,
            (
                "complete",
                True,
                sorted(zip(case["vectors"], case["bidirectional"], strict=True)),
                len(case["vectors"]),
                case["rank"],
                case["lineality"],
                case["total"],
                case["total"],
                True,
                True,
            ),
            strict=True,
        )
    )


def structural(case):
    report = enumerated(case)
    model = flux_model(
        case["matrix"],
        irreversible=case["irreversible"],
        external_matrix=case.get("external", ()),
        scaled=case.get("scaled", False),
    )
    frame = SBMLKinetics(model, controlled_parameters=("cf",))
    before = dict(frame.initial_values)
    native = sbml_elementary_flux_modes(frame, before)
    coherent = (
        modes(native) == modes(report)
        and before == dict(frame.initial_values)
        and native["complete"]
        and not report["extreme_rays_computed"]
    )
    for mode in report["modes"]:
        vector = tuple(int(v) for v in mode["vector"])
        coherent = coherent and all(
            v >= 0 for v, flag in zip(vector, case["irreversible"], strict=True) if flag
        )
        coherent = coherent and list(mode["external_exchange"].values()) == exchanges(
            case.get("external", ()), vector
        )
        coherent = coherent and mode["support"] == [f"r{i}" for i, v in enumerate(vector) if v]
        coherent = coherent and mode["internal_cycle"] == (
            not any(Fraction(v) for v in mode["external_exchange"].values())
        )
    coherent = coherent and all(
        abs(v) <= 1e-12
        for row in native["native_normalized_mode_balance"].values()
        for v in row.values()
    )
    if case.get("scaled"):
        coherent = coherent and Fraction(native["current_row_factors"]["I0"]) == Fraction(0.001)
    actual = dict(
        zip(
            STRUCTURAL_FIELDS,
            (
                report["status"],
                report["complete"],
                modes(report),
                len(report["modes"]),
                report["rank"],
                report["lineality_dimension"],
                report["total_supports"],
                report["evaluated_supports"],
                all(minimal_balanced(case["matrix"], m["vector"]) for m in report["modes"]),
                coherent,
            ),
            strict=True,
        )
    )
    return actual, {"analyses": [report, native], "source_xml": model.to_xml()}


def expected_special(name):
    if name == "native_current_control":
        values = (
            True,
            True,
            [((0, 1, 1), False), ((1, 0, 1), False)],
            [((0, 0, 1), False), ((0, 1, 0), False), ((1, 0, 0), False)],
            {"cf": 0.0},
            {"I0": "0"},
            True,
            True,
            True,
            True,
        )
    elif name == "budget_incomplete":
        values = (
            "incomplete",
            False,
            False,
            [],
            1,
            6,
            "support_evaluation_budget_exhausted",
            True,
            "incomplete",
            True,
        )
    else:
        values = (True,) * 10
    return dict(zip(dict(SPECIAL)[name], values, strict=True))


def native_control():
    result = run_example()
    before, after = result["before"], result["after"]
    frame = SBMLKinetics(flux_model(zero_rates=True))
    zero = sbml_elementary_flux_modes(frame, frame.initial_values)
    actual = dict(
        zip(
            dict(SPECIAL)["native_current_control"],
            (
                before["complete"],
                after["complete"],
                modes(before),
                modes(after),
                after["current_controls"],
                after["current_row_factors"],
                before["time_s"] == after["time_s"] == 1.0,
                result["snapshots_unchanged_by_analysis"],
                all(
                    info["transaction_status"] == "committed"
                    and info["operation_outcome"] == "success"
                    for info in result["operations"]
                ),
                not any(zero["current_reaction_rates_mol_s"].values()) and len(zero["modes"]) == 2,
            ),
            strict=True,
        )
    )
    return actual, {"analyses": [before, after, zero], "native_execution": result}


def budget():
    partial = enumerated(STRUCTURAL[2], policy=FluxModePolicy(1))
    empty = enumerate_elementary_flux_modes(
        ((1, 1),), reaction_ids=("a", "b"), irreversible=(True, True), internal_species_ids=("X",)
    )
    arithmetic = enumerate_elementary_flux_modes(
        ((1, 2),),
        reaction_ids=("a", "b"),
        irreversible=(False, False),
        internal_species_ids=("X",),
        policy=FluxModePolicy(max_integer_bits=1),
    )
    actual = dict(
        zip(
            dict(SPECIAL)["budget_incomplete"],
            (
                partial["status"],
                partial["complete"],
                partial["no_modes_proven"],
                partial["modes"],
                partial["evaluated_supports"],
                partial["total_supports"],
                partial["stop_reason"],
                empty["complete"] and empty["no_modes_proven"] and not empty["modes"],
                arithmetic["status"],
                "arithmetic size budget" in arithmetic["stop_reason"]
                and not arithmetic["no_modes_proven"],
            ),
            strict=True,
        )
    )
    return actual, {"analyses": [partial, empty, arithmetic]}


def scope():
    actual, rejections = {}, {}
    for defect in dict(SPECIAL)["scope_rejection"][:4]:
        # Construct the valid original source first. A malformed XML fixture is
        # an unexpected unit failure, not a claimed analysis-scope rejection.
        frame = SBMLKinetics(source_scope_model(defect))
        error = None
        try:
            sbml_elementary_flux_modes(frame, frame.initial_values)
        except ValueError as exc:
            error = str(exc)
        actual[defect] = error is not None
        rejections[defect] = error
    scenarios = {
        "boolean_coefficient": {"matrix": ((True,),)},
        "nonfinite_coefficient": {"matrix": ((float("inf"),),)},
        "duplicate_reaction": {
            "matrix": ((1, 1),),
            "reaction_ids": ("r", "r"),
            "irreversible": (True, True),
        },
        "dimension_mismatch": {"matrix": ((1, 1),)},
        "overlapping_pools": {"external_matrix": ((1,),), "external_species_ids": ("X",)},
    }
    for defect, changes in scenarios.items():
        kwargs = {
            "matrix": ((1,),),
            "reaction_ids": ("r",),
            "irreversible": (True,),
            "internal_species_ids": ("X",),
            **changes,
        }
        error = None
        try:
            enumerate_elementary_flux_modes(**kwargs)
        except ValueError as exc:
            error = str(exc)
        actual[defect], rejections[defect] = error is not None, error
    error = None
    try:
        FluxModePolicy(-1)
    except ValueError as exc:
        error = str(exc)
    actual["invalid_budget"], rejections["invalid_budget"] = error is not None, error
    return actual, {"rejections": rejections, "analyses": []}


def run_block(output, summary):
    if output.exists() or summary.exists():
        raise ValueError("attempt output names must be new")
    plan = [
        (case["name"], expected_structure(case), lambda case=case: structural(case))
        for case in STRUCTURAL
    ]
    plan += [
        (
            name,
            expected_special(name),
            {
                "native_current_control": native_control,
                "budget_incomplete": budget,
                "scope_rejection": scope,
            }[name],
        )
        for name, _ in SPECIAL
    ]
    if len(plan) != 12 or any(len(expected) != 10 for _, expected, _ in plan):
        raise ValueError("fixed block must contain twelve units with ten comparison slots each")
    started = time.perf_counter()
    units, failures = [], []
    for index, (name, expected, execute) in enumerate(plan):
        error = None
        try:
            actual, evidence = execute()
        except Exception as exc:
            actual, evidence = {}, {}
            error = f"{type(exc).__name__}: {exc}"
        checks = [
            {
                "label": key,
                "actual": actual.get(key),
                "expected": value,
                "passed": error is None and key in actual and actual[key] == value,
            }
            for key, value in expected.items()
        ]
        if error is not None:
            evidence["unexpected_exception"] = error
        failed = [{"unit": name, **check} for check in checks if not check["passed"]]
        failures.extend(failed)
        units.append({"unit": name, "checks": checks, "evidence": evidence, "passed": not failed})
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "fixed_52C_block",
                    "completed": index + 1,
                    "total": 12,
                    "units_per_s": (index + 1) / elapsed,
                    "eta_s": elapsed / (index + 1) * (11 - index),
                    "unit": name,
                    "passed": not failed,
                }
            ),
            flush=True,
        )
    analyses = [a for unit in units for a in unit["evidence"].get("analyses", ())]
    compact = {
        "mode": "development",
        "status": "failed" if failures else "passed",
        "units": 12,
        "passed_units": sum(u["passed"] for u in units),
        "comparisons": 120,
        "passed_comparisons": sum(c["passed"] for u in units for c in u["checks"]),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "resources": {
            "analyses": len(analyses),
            "complete_analyses": sum(a["complete"] for a in analyses),
            "incomplete_analyses": sum(not a["complete"] for a in analyses),
            **{
                key: sum(a[key] for a in analyses)
                for key in ("rank_evaluations", "attempted_supports", "evaluated_supports")
            },
            "native_operations": sum(
                len(u["evidence"].get("native_execution", {}).get("operations", ())) for u in units
            ),
        },
        "per_unit": [
            {
                "unit": u["unit"],
                "passed": u["passed"],
                "comparisons": 10,
                "analysis_outcomes": [
                    {
                        key: a[key]
                        for key in (
                            "status",
                            "complete",
                            "no_modes_proven",
                            "stop_reason",
                            "rank",
                            "lineality_dimension",
                            "total_supports",
                            "attempted_supports",
                            "evaluated_supports",
                        )
                    }
                    | {"modes": len(a["modes"])}
                    for a in u["evidence"].get("analyses", ())
                ],
                "rejections": u["evidence"].get("rejections", {}),
                "unexpected_exception": u["evidence"].get("unexpected_exception"),
            }
            for u in units
        ],
        "full_output": str(output.resolve()),
        "boundary": (
            "Exact bounded EFM development qualification; no extreme-ray or scalability claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": compact, "units": units}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {k: compact[k] for k in ("status", "units", "comparisons", "failures", "resources")},
            indent=2,
        )
    )
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = run_block(args.output, args.summary)
    if result["status"] != "passed":
        raise SystemExit(1)
