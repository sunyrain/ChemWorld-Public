"""Frozen, provider-free challenge probes for Experiment 1 candidate loci."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256

SCHEMA_VERSION = "chemworld-experiment-1-challenge-probes-1.0"
EXPECTED_BLOCKS = (
    "EC-E",
    "EC-P",
    "EC-S",
    "RX-P",
    "RX-S",
    "PA-E",
    "PA-P",
    "PA-S",
    "C-E",
    "C-P",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _midpoint(band: Any) -> float | None:
    values = _sequence(band)
    if len(values) != 2 or not all(_finite(value) for value in values):
        return None
    return (float(values[0]) + float(values[1])) / 2.0


def _entity_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    prior = _mapping(report.get("prior_audit"))
    checks = _mapping(prior.get("checks"))
    private = _mapping(report.get("private_world_audit"))
    mapping_rows = _sequence(private.get("mapping_rows"))
    comparisons = _sequence(report.get("comparisons"))
    anchors = _sequence(report.get("anchor_results"))

    registered_exchange = bool(
        checks.get("single_target_transposition")
        or checks.get("solvent_only_transposition")
        or (
            isinstance(report.get("descriptor_permutation"), list)
            and sorted(report["descriptor_permutation"])
            == list(range(len(report["descriptor_permutation"])))
        )
    )
    mapping_consistent = bool(
        (
            mapping_rows
            and all(_mapping(row).get("own_mapping_closer") is True for row in mapping_rows)
        )
        or (
            comparisons
            and all(_mapping(row).get("aligned_mapping_consistent") is True for row in comparisons)
        )
    )
    plausible = bool(prior.get("passed") and registered_exchange and mapping_consistent)

    if anchors:
        active_rows = [_mapping(row) for row in anchors]
        active_score = max(
            (
                float(
                    row.get(
                        "signal_to_noise_ratio",
                        row.get("support_signal_to_noise_ratio", 0.0),
                    )
                )
                for row in active_rows
            ),
            default=0.0,
        )
        active_passed = any(row.get("passed") is True for row in active_rows)
    else:
        active_rows = [_mapping(row) for row in comparisons]
        active_score = max(
            (float(row.get("signal_to_noise_ratio", 0.0)) for row in active_rows),
            default=0.0,
        )
        active_passed = any(row.get("resolved") is True for row in active_rows)
    active_passed = bool(active_passed and active_score >= float(rule["minimum_active_snr"]))
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": (
            "registered within-family label exchange with aligned mapping consistency"
        ),
        "default_one_shot_reliably_discriminates": False,
        "default_evidence": (
            "one unpaired outcome is invariant to label exchange plus nuisance baseline"
        ),
        "active_information_score": active_score,
        "active_information_passed": active_passed,
        "active_evidence": "paired registered entities under a shared context",
        "minimum_reliable_unique_condition_cost": 2,
    }


def _reflection_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    analysis = _mapping(report.get("legacy_analysis"))
    selected = _mapping(analysis.get("selected_reflection"))
    checks = _mapping(selected.get("checks"))
    matching = _mapping(analysis.get("prior_matching"))
    priors = _mapping(analysis.get("public_priors"))
    contexts = []
    for name in ("supplied_a", "supplied_b"):
        context = _mapping(_mapping(priors.get(name)).get("context_contract"))
        if _finite(context.get("coordinate_center")):
            centre: Any = {"coordinate_center": float(context["coordinate_center"])}
        else:
            centre = _mapping(context.get("approximate_reference_region"))
        contexts.append(
            {
                "centre": centre,
                "target_controls": list(_sequence(context.get("target_controls"))),
            }
        )
    centred = len(contexts) == 2 and contexts[0] == contexts[1] and bool(contexts[0]["centre"])
    plausible = bool(
        matching.get("passed")
        and checks.get("baseline_utility_matched")
        and selected.get("axis") in {"potential", "current", "temperature", "duration"}
        and centred
    )
    score = float(selected.get("disagreement_fraction", 0.0))
    active = bool(
        checks.get("low_side_falsification_region")
        and checks.get("high_side_falsification_region")
        and checks.get("representatives_separated")
        and score >= float(rule["minimum_active_disagreement_fraction"])
    )
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": (
            "matched-context directional reflection inside the legal public grid"
        ),
        "default_one_shot_reliably_discriminates": False,
        "default_evidence": (
            "the opposed directional claims share their registered reflection centre"
        ),
        "active_information_score": score,
        "active_information_passed": active,
        "active_evidence": "separated low/high public-grid regions with held-out disagreement",
        "minimum_reliable_unique_condition_cost": 2,
    }


def _ec_structural_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    analysis = _mapping(report.get("legacy_analysis"))
    checks = _mapping(analysis.get("checks"))
    model = _mapping(analysis.get("model_qualification"))
    score = float(model.get("disagreement_fraction", 0.0))
    plausible = bool(
        checks.get("prior_schema_matched")
        and checks.get("prior_word_count_matched")
        and checks.get("baseline_error_matched")
        and abs(float(model.get("baseline_error_gap", math.inf))) <= 1.0e-12
    )
    active = bool(
        model.get("low_counterexample_support", 0) >= 1
        and model.get("high_counterexample_support", 0) >= 1
        and score >= float(rule["minimum_active_disagreement_fraction"])
    )
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": (
            "registered topology alternatives are baseline-matched at the frozen centre"
        ),
        "default_one_shot_reliably_discriminates": False,
        "default_evidence": (
            "the frozen centre has equal baseline error and does not identify topology"
        ),
        "active_information_score": score,
        "active_information_passed": active,
        "active_evidence": "counterexamples occur in separated low and high regions",
        "minimum_reliable_unique_condition_cost": 2,
    }


def _rx_structural_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    analysis = _mapping(report.get("legacy_analysis"))
    checks = _mapping(analysis.get("checks"))
    mechanism = _mapping(analysis.get("mechanism_audit"))
    arms = _mapping(report.get("prior_arms"))
    aligned = _mapping(arms.get("aligned"))
    false = _mapping(arms.get("misspecified"))
    supporting = _sequence(analysis.get("supporting_cells"))
    accumulations = [
        _mapping(value) for value in _mapping(analysis.get("accumulation_reports")).values()
    ]
    plausible = bool(
        aligned
        and false
        and set(aligned) == set(false)
        and aligned.get("confidence") == false.get("confidence")
        and mechanism.get("added_reaction_count") == 1
        and mechanism.get("opposite_stoichiometric_channel") is True
        and mechanism.get("execution_mechanism_binding_matches") is True
    )
    score = float(analysis.get("minimum_candidate_effect", 0.0))
    active = bool(
        len(supporting) >= int(rule["minimum_supporting_conditions"])
        and accumulations
        and all(row.get("passed") is True for row in accumulations)
        and checks.get("two_separated_safe_supporting_cells")
    )
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": "one bound reverse channel with matched qualitative prior schema",
        "default_one_shot_reliably_discriminates": False,
        "default_evidence": (
            "one condition cannot separate reverse accumulation from nuisance kinetics"
        ),
        "active_information_score": score,
        "active_information_passed": active,
        "active_evidence": "separated duration/dose cells exhibit an accumulation signature",
        "minimum_reliable_unique_condition_cost": 2,
    }


def _pa_parametric_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    aligned = _midpoint(report.get("aligned_prior", {}).get("k_star_band"))
    false = _midpoint(report.get("misspecified_prior", {}).get("k_star_band"))
    ratio = false / aligned if aligned and false is not None else math.nan
    lower, upper = [float(value) for value in rule["plausible_false_to_aligned_center_ratio"]]
    plausible = bool(
        _finite(ratio)
        and lower <= ratio <= upper
        and report.get("aligned_contains_fit") is True
        and report.get("misspecified_contains_fit") is False
    )
    rows = [_mapping(row) for row in _sequence(report.get("phase_point_reports"))]
    default = next(
        (row for row in rows if row.get("point_id") == rule["default_point_id"]),
        {},
    )
    default_success = default.get("noise_robust_counterexample") is True
    active_score = max((float(row.get("prediction_gap", 0.0)) for row in rows), default=0.0)
    robust = sum(row.get("noise_robust_counterexample") is True for row in rows)
    active = bool(
        robust >= 2
        and active_score >= float(rule["minimum_active_prediction_gap"])
        and active_score > float(default.get("prediction_gap", 0.0))
    )
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": {
            "false_to_aligned_center_ratio": ratio,
            "allowed": [lower, upper],
        },
        "default_one_shot_reliably_discriminates": default_success,
        "default_evidence": {"point_id": rule["default_point_id"], "report": dict(default)},
        "active_information_score": active_score,
        "active_information_passed": active,
        "active_evidence": {"robust_candidate_points": robust},
        "minimum_reliable_unique_condition_cost": 1 if default_success else 2,
    }


def _pa_structural_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    arms = _mapping(report.get("prior_arms"))
    slope = _mapping(report.get("public_log_ratio_slope"))
    metrics = _mapping(report.get("metric_reports"))
    all_pairs = {
        _mapping(row).get("pair_id")
        for metric in metrics.values()
        for row in _sequence(_mapping(metric).get("pairs"))
        if _mapping(row).get("pair_id") is not None
    }
    plausible = bool(
        arms.get("schema_matched")
        and arms.get("text_template_matched")
        and report.get("law_binding_verified") is True
        and not arms.get("leakage_tokens")
    )
    score = float(slope.get("absolute_slope_deviation_from_one", 0.0))
    active = bool(
        slope.get("slope_signature_passed")
        and score >= float(rule["minimum_slope_deviation"])
        and len(all_pairs) >= 2
    )
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": (
            "two executable response families with matched qualitative prior schema"
        ),
        "default_one_shot_reliably_discriminates": False,
        "default_evidence": (
            "a response exponent is not identifiable from one phase-ratio condition"
        ),
        "active_information_score": score,
        "active_information_passed": active,
        "active_evidence": {"distinct_pair_conditions": len(all_pairs)},
        "minimum_reliable_unique_condition_cost": 2,
    }


def _c_parametric_probe(report: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    aligned = _sequence(report.get("aligned_effect_band"))
    false = _sequence(report.get("misspecified_effect_band"))
    low, high = [float(value) for value in rule["plausible_effect_interval"]]
    values = [*aligned, *false]
    plausible = bool(
        len(aligned) == len(false) == 2
        and all(_finite(value) and low <= float(value) <= high for value in values)
        and float(false[1]) < float(aligned[0])
    )
    score = float(report.get("endpoint_signal_to_noise_ratio", 0.0))
    temperatures = {
        _mapping(row).get("temperature_K") for row in _sequence(report.get("temperature_results"))
    }
    active = bool(score >= float(rule["minimum_active_snr"]) and {270.0, 310.0} <= temperatures)
    return {
        "plausibility_passed": plausible,
        "plausibility_evidence": {
            "aligned_effect_band": list(aligned),
            "false_effect_band": list(false),
        },
        "default_one_shot_reliably_discriminates": False,
        "default_evidence": "the registered cooling-response gain requires two endpoint conditions",
        "active_information_score": score,
        "active_information_passed": active,
        "active_evidence": "310 K to 270 K endpoint contrast",
        "minimum_reliable_unique_condition_cost": 2,
    }


PROBE_FAMILIES = {
    "entity_label_exchange": _entity_probe,
    "centered_directional_reflection": _reflection_probe,
    "ec_structural_two_region_trend": _ec_structural_probe,
    "rx_structural_accumulation_trend": _rx_structural_probe,
    "pa_numeric_band_reference_default": _pa_parametric_probe,
    "pa_structural_slope_trend": _pa_structural_probe,
    "c_endpoint_gain_contrast": _c_parametric_probe,
}


def validate_contract(contract: Mapping[str, Any], root: Path) -> None:
    if contract.get("schema_version") != "chemworld-experiment-1-challenge-probes-contract-1.0":
        raise ValueError("unexpected challenge contract schema")
    denominator = _mapping(contract.get("denominator"))
    if denominator.get("world_probe_rows") != 50:
        raise ValueError("challenge denominator must be 50 World-probe rows")
    loci = _mapping(contract.get("loci"))
    if tuple(loci) != EXPECTED_BLOCKS:
        raise ValueError("challenge contract must bind the ordered ten candidate loci")
    for binding_name in ("source_registry", "protocol", "experiment_note"):
        binding = _mapping(contract.get(binding_name))
        path = root / str(binding.get("path", ""))
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"{binding_name} binding mismatch")
    for block, rule_any in loci.items():
        rule = _mapping(rule_any)
        if rule.get("probe_family") not in PROBE_FAMILIES:
            raise ValueError(f"unknown probe family for {block}")
        for binding_any in _sequence(rule.get("source_contracts")):
            binding = _mapping(binding_any)
            path = root / str(binding.get("path", ""))
            if not path.is_file() or file_sha256(path) != binding.get("sha256"):
                raise ValueError(f"source contract binding mismatch for {block}")


def _resolve_report(relative: str, roots: Sequence[Path]) -> Path:
    hits = [(root / relative).resolve() for root in roots if (root / relative).is_file()]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one evidence report for {relative}; found {len(hits)}")
    return hits[0]


def _load_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_probe_summary(
    contract: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    root: Path,
    evidence_roots: Sequence[Path],
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Evaluate all frozen challenge probes without changing World evidence."""

    validate_contract(contract, root)
    if registry.get("registry_sha256") != _mapping(contract["source_registry"]).get(
        "registry_sha256"
    ):
        raise ValueError("source registry self hash mismatch")
    rows = registry.get("rows")
    if not isinstance(rows, list) or len(rows) != 105:
        raise ValueError("challenge probes require the complete 105-row registry")

    rules = _mapping(contract["loci"])
    selected = []
    for row_any in rows:
        row = _mapping(row_any)
        system = str(row.get("system_id", ""))
        locus = str(row.get("prior_locus", ""))
        block = f"{system}-{locus[:1].upper()}"
        if block in rules:
            selected.append((block, row))
    if len(selected) != 50:
        raise ValueError("challenge candidate selection did not yield 50 rows")

    probe_rows: list[dict[str, Any]] = []
    for block, row in selected:
        failures: list[str] = []
        payload: dict[str, Any] = {}
        evidence = _mapping(_mapping(row.get("qualification_run")).get("evidence"))
        relative = str(evidence.get("path", ""))
        try:
            if row.get("current_status") != "qualified-development":
                raise ValueError("source unit is not qualified-development")
            path = _resolve_report(relative, evidence_roots)
            if file_sha256(path) != evidence.get("sha256"):
                raise ValueError("source report file digest mismatch")
            report = _load_json(path)
            self_hash = report.get("report_sha256")
            unhashed = dict(report)
            unhashed.pop("report_sha256", None)
            if self_hash != evidence.get("report_sha256") or self_hash != canonical_json_sha256(
                unhashed
            ):
                raise ValueError("source report self hash mismatch")
            rule = _mapping(rules[block])
            payload = PROBE_FAMILIES[str(rule["probe_family"])](report, rule)
            min_cost = int(payload["minimum_reliable_unique_condition_cost"])
            budget = int(rule["participant_budget"])
            lower = int(_mapping(contract["denominator"])["trivial_lower_bound_unique_conditions"])
            payload["participant_budget"] = budget
            payload["trivial_lower_bound"] = lower
            payload["budget_window_passed"] = lower < min_cost <= budget
            if not payload["plausibility_passed"]:
                failures.append("plausibility")
            if not payload["active_information_passed"]:
                failures.append("information_choice")
            if not payload["budget_window_passed"]:
                failures.append("budget_window")
        except Exception as exc:  # fail closed while preserving the row
            failures.append(f"probe_error:{type(exc).__name__}:{exc}")
        probe_rows.append(
            {
                "block": block,
                "unit_id": row.get("unit_id"),
                "world_id": row.get("world_id"),
                "source_evidence_path": relative,
                "source_evidence_sha256": evidence.get("sha256"),
                "probe": payload,
                "failures": failures,
                "completed": bool(payload),
            }
        )

    block_rows = []
    for block in EXPECTED_BLOCKS:
        subset = [row for row in probe_rows if row["block"] == block]
        default_successes = sum(
            _mapping(row.get("probe")).get("default_one_shot_reliably_discriminates") is True
            for row in subset
        )
        checks = {
            "plausibility": len(subset) == 5
            and all(
                _mapping(row.get("probe")).get("plausibility_passed") is True for row in subset
            ),
            "non_triviality": len(subset) == 5 and default_successes < 5,
            "information_choice": len(subset) == 5
            and all(
                _mapping(row.get("probe")).get("active_information_passed") is True
                for row in subset
            ),
            "budget_window": len(subset) == 5
            and all(
                _mapping(row.get("probe")).get("budget_window_passed") is True for row in subset
            ),
        }
        block_rows.append(
            {
                "block": block,
                "world_probe_rows": len(subset),
                "default_one_shot_success_worlds": default_successes,
                "checks": checks,
                "challenge_probe_passed": all(checks.values()),
                "failures": [name for name, passed in checks.items() if not passed],
            }
        )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "participant_execution_authorized": False,
        "confirmation_execution_authorized": False,
        "source_commit": source_commit or "unbound-test",
        "contract_sha256": canonical_json_sha256(contract),
        "source_registry_sha256": registry.get("registry_sha256"),
        "denominators": {
            "planned_world_probe_rows": 50,
            "attempted_world_probe_rows": len(probe_rows),
            "completed_world_probe_rows": sum(row["completed"] for row in probe_rows),
            "failed_world_probe_rows": sum(bool(row["failures"]) for row in probe_rows),
            "candidate_loci": 10,
        },
        "world_probe_rows": probe_rows,
        "loci": block_rows,
        "challenge_passed_loci": sum(row["challenge_probe_passed"] for row in block_rows),
        "challenge_failed_loci": sum(not row["challenge_probe_passed"] for row in block_rows),
    }
    summary["challenge_probe_sha256"] = canonical_json_sha256(summary)
    return summary
