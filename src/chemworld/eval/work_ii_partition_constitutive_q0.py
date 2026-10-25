"""Frozen design, analysis, and artifact validation for partition A-S Q0."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from chemworld.envs.observation_noise import ObservationNoiseCoordinate
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_execution_mode import (
    ExecutionMode,
    WorkIIExecutionContext,
    validate_execution_envelope,
)

QUALIFICATION_VERSION = "chemworld-work-ii-partition-constitutive-q0-0.1"
TASK_REPORT_VERSION = "chemworld-work-ii-partition-constitutive-q0-task-report-0.1"
SUMMARY_VERSION = "chemworld-work-ii-partition-constitutive-q0-summary-0.1"
NOMINAL_PAIR_QUALIFICATION_VERSION = (
    "chemworld-work-ii-partition-nominal-pair-constitutive-q0-0.2"
)
NOMINAL_PAIR_TASK_REPORT_VERSION = (
    "chemworld-work-ii-partition-nominal-pair-constitutive-q0-task-report-0.2"
)
NOMINAL_PAIR_SUMMARY_VERSION = (
    "chemworld-work-ii-partition-nominal-pair-constitutive-q0-summary-0.2"
)
TASK_ID = "partition-discovery"
WORLD_SEED = 0
LAW_IDS = ("linear_response", "power_response")
INSTRUMENTS = ("hplc", "final_assay")
METRICS = ("product_in_organic", "product_in_aqueous", "phase_ratio")
DECLARED_SIGMA = {
    "hplc": {
        "product_in_organic": 0.015,
        "product_in_aqueous": 0.015,
        "phase_ratio": 0.018,
    },
    "final_assay": {
        "product_in_organic": 0.010,
        "product_in_aqueous": 0.010,
        "phase_ratio": 0.012,
    },
}
BASELINE_EXPONENT = 1.0
POWER_RESPONSE_EXPONENT = 1.75
LOAD_LEVELS_L = (0.006, 0.015, 0.024)
PHASE_VOLUME_LEVELS_L = (0.008, 0.019, 0.030)
NOMINAL_IDENTITIES = (0, 1, 2, 3)
NOMINAL_PAIR_AQUEOUS_VOLUME_L = 0.015
NOMINAL_PAIR_EXTRACTANT_VOLUME_L = 0.019
NOMINAL_PAIR_SOLVENT_VOLUME_L = 0.020
NOMINAL_PAIR_SLOPE_MINIMUM_DEVIATION = 0.20
FORBIDDEN_PUBLIC_TOKENS = (
    "mechanism_family",
    "world_intervention",
    "private_seed",
    "hidden_state",
    "evaluator_truth",
)


def constitutive_intervention() -> dict[str, Any]:
    """Return the frozen power-response intervention."""

    return {
        "kind": "mechanism_family",
        "mode": "constitutive_law_family",
        "severity": 1.0,
        "constitutive_law_change": {
            "transform_id": "partition_power_response_stress_v1",
            "partition_coefficient_exponent_at_full_severity": (
                POWER_RESPONSE_EXPONENT
            ),
        },
    }


def registered_cells() -> list[dict[str, Any]]:
    return [
        {
            "cell_id": f"load-{load_index}-phase-volume-{phase_index}",
            "load_index": load_index,
            "phase_volume_index": phase_index,
            "aqueous_volume_L": float(load),
            "extractant_volume_L": float(phase_volume),
        }
        for load_index, load in enumerate(LOAD_LEVELS_L)
        for phase_index, phase_volume in enumerate(PHASE_VOLUME_LEVELS_L)
    ]


def frozen_action_plan(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the immutable public protocol for one registered grid cell."""

    return [
        {"operation": "add_solvent", "volume_L": 0.020, "solvent": 0},
        {
            "operation": "add_phase",
            "phase": "aqueous",
            "volume_L": float(cell["aqueous_volume_L"]),
        },
        {
            "operation": "add_extractant",
            "extractant": 1,
            "volume_L": float(cell["extractant_volume_L"]),
        },
        {
            "operation": "mix",
            "duration_s": 420.0,
            "stirring_speed_rpm": 800.0,
        },
        {"operation": "settle", "duration_s": 900.0},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def observation_binding(cell_id: str) -> tuple[int, str]:
    digest = sha256(
        f"work-ii-partition-constitutive-q0:{WORLD_SEED}:{cell_id}".encode()
    ).hexdigest()
    return (
        int(digest[:8], 16) % 2_147_483_647,
        f"work-ii-partition-constitutive-w{WORLD_SEED}-{digest[:12]}",
    )


def noise_coordinate(cell_id: str, instrument: str) -> ObservationNoiseCoordinate:
    observation_seed, namespace = observation_binding(cell_id)
    return ObservationNoiseCoordinate(
        namespace=namespace,
        base_observation_seed=observation_seed,
        experiment_index=0,
        operation_type="measure",
        instrument=instrument,
        # The registered HPLC result is the second, post-separation assay.
        replicate_index=1 if instrument == "hplc" else 0,
    )


def effect_gate(instrument: str, metric: str) -> float:
    return max(0.03, 6.0 * float(DECLARED_SIGMA[instrument][metric]))


def registered_nominal_pair_cells() -> list[dict[str, Any]]:
    """Return the frozen full categorical solvent-by-extractant design."""

    return [
        {
            "cell_id": f"solvent-{solvent}-extractant-{extractant}",
            "solvent": solvent,
            "extractant": extractant,
            "aqueous_volume_L": NOMINAL_PAIR_AQUEOUS_VOLUME_L,
            "extractant_volume_L": NOMINAL_PAIR_EXTRACTANT_VOLUME_L,
        }
        for solvent in NOMINAL_IDENTITIES
        for extractant in NOMINAL_IDENTITIES
    ]


def frozen_nominal_pair_action_plan(
    cell: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the immutable protocol for one nominal identity pair."""

    return [
        {
            "operation": "add_solvent",
            "volume_L": NOMINAL_PAIR_SOLVENT_VOLUME_L,
            "solvent": int(cell["solvent"]),
        },
        {
            "operation": "add_phase",
            "phase": "aqueous",
            "volume_L": NOMINAL_PAIR_AQUEOUS_VOLUME_L,
        },
        {
            "operation": "add_extractant",
            "extractant": int(cell["extractant"]),
            "volume_L": NOMINAL_PAIR_EXTRACTANT_VOLUME_L,
        },
        {"operation": "mix", "duration_s": 420.0, "stirring_speed_rpm": 800.0},
        {"operation": "settle", "duration_s": 900.0},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def nominal_pair_observation_binding(cell_id: str) -> tuple[int, str]:
    digest = sha256(
        f"work-ii-partition-nominal-pair-constitutive-q0:{WORLD_SEED}:{cell_id}".encode()
    ).hexdigest()
    return (
        int(digest[:8], 16) % 2_147_483_647,
        f"work-ii-partition-nominal-pair-w{WORLD_SEED}-{digest[:12]}",
    )


def nominal_pair_noise_coordinate(
    cell_id: str, instrument: str
) -> ObservationNoiseCoordinate:
    observation_seed, namespace = nominal_pair_observation_binding(cell_id)
    return ObservationNoiseCoordinate(
        namespace=namespace,
        base_observation_seed=observation_seed,
        experiment_index=0,
        operation_type="measure",
        instrument=instrument,
        replicate_index=1 if instrument == "hplc" else 0,
    )


def analyze(
    rows: Sequence[Mapping[str, Any]],
    constitutive_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "fixed_execution_denominator": len(rows) == 18,
        "all_outcomes_classified": all(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in rows
        ),
        "all_executions_completed": all(
            row.get("status") == "completed" for row in rows
        ),
        "registered_task_world_and_cell_bindings": _registered_row_bindings(rows),
        "all_exact_replay": all(row.get("exact_replay") is True for row in rows),
        "zero_platform_failures": not any(
            row.get("status") == "platform_failure" for row in rows
        ),
        "paired_action_plans": _paired_equal(rows, "action_plan_sha256"),
        "frozen_action_plans_bound": _frozen_action_plans_bound(rows),
        "paired_hplc_noise": _paired_noise_equal(rows, "hplc"),
        "paired_final_assay_noise": _paired_noise_equal(rows, "final_assay"),
        "frozen_noise_coordinates_bound": _frozen_noise_coordinates_bound(rows),
        "reaction_network_unchanged": (
            constitutive_audit.get("reaction_network_unchanged") is True
        ),
        "public_contract_unchanged": (
            constitutive_audit.get("public_contract_unchanged") is True
        ),
        "baseline_exponent_is_frozen": math.isclose(
            float(constitutive_audit.get("baseline_exponent", math.nan)),
            BASELINE_EXPONENT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "power_response_exponent_is_frozen": math.isclose(
            float(constitutive_audit.get("power_response_exponent", math.nan)),
            POWER_RESPONSE_EXPONENT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "only_registered_constitutive_parameter_changed": (
            constitutive_audit.get("only_registered_constitutive_parameter_changed")
            is True
            and constitutive_audit.get("changed_domain_parameter_keys")
            == ["partition_coefficient_exponent"]
        ),
        "intervention_binding_deterministic": (
            constitutive_audit.get("intervention_binding_deterministic") is True
        ),
        "execution_constitutive_binding_matches": (
            constitutive_audit.get("execution_constitutive_binding_matches") is True
            and _execution_constitutive_binding_matches(rows, constitutive_audit)
        ),
    }
    if not all(checks.values()):
        return _early_result(rows, constitutive_audit, checks)

    completed = [row for row in rows if row.get("status") == "completed"]
    metrics_finite = all(
        all(
            math.isfinite(float(row["measurements"][instrument][metric]))
            for instrument in INSTRUMENTS
            for metric in METRICS
        )
        for row in completed
    )
    metrics_observed = all(
        all(
            row["observed_masks"][instrument].get(metric) is True
            for instrument in INSTRUMENTS
            for metric in METRICS
        )
        for row in completed
    )
    leakage_matches = sorted(
        {
            str(token)
            for row in rows
            for token in row.get("participant_visible_leakage_matches", [])
        }
        | {
            token
            for row in rows
            for token in FORBIDDEN_PUBLIC_TOKENS
            if token
            in json.dumps(row.get("participant_visible_payload", {}), sort_keys=True)
        }
    )
    pairs = _paired_rows(rows)
    channel_reports: dict[str, dict[str, Any]] = {}
    supporting_cells: set[str] = set()
    for instrument in INSTRUMENTS:
        for metric in METRICS:
            channel_id = f"{instrument}:{metric}"
            gate = effect_gate(instrument, metric)
            cell_gaps = []
            for pair in pairs:
                baseline = float(
                    pair[LAW_IDS[0]]["measurements"][instrument][metric]
                )
                power = float(pair[LAW_IDS[1]]["measurements"][instrument][metric])
                gap = power - baseline
                cell_gaps.append(
                    {
                        "cell_id": pair["cell_id"],
                        "load_index": pair["load_index"],
                        "phase_volume_index": pair["phase_volume_index"],
                        "linear_response": baseline,
                        "power_response": power,
                        "signed_gap": gap,
                    }
                )
                if abs(gap) >= gate:
                    supporting_cells.add(str(pair["cell_id"]))

            load_ranges = _axis_ranges(pairs, instrument, metric, "load_index")
            phase_ranges = _axis_ranges(
                pairs,
                instrument,
                metric,
                "phase_volume_index",
            )
            gap_grid = {
                (int(row["load_index"]), int(row["phase_volume_index"])): float(
                    row["signed_gap"]
                )
                for row in cell_gaps
            }
            load_curvatures = [
                abs(gap_grid[2, phase] - 2.0 * gap_grid[1, phase] + gap_grid[0, phase])
                for phase in range(3)
            ]
            interaction = abs(
                gap_grid[2, 2]
                - gap_grid[2, 0]
                - gap_grid[0, 2]
                + gap_grid[0, 0]
            )
            signature = max(*load_curvatures, interaction)
            channel_reports[channel_id] = {
                "instrument": instrument,
                "metric": metric,
                "declared_sigma": DECLARED_SIGMA[instrument][metric],
                "effect_gate": gate,
                "max_absolute_paired_gap": max(
                    abs(float(row["signed_gap"])) for row in cell_gaps
                ),
                "load_axis_max_range": max(load_ranges),
                "load_axis_observable": max(load_ranges) >= gate,
                "phase_volume_axis_max_range": max(phase_ranges),
                "phase_volume_axis_observable": max(phase_ranges) >= gate,
                "load_gap_curvatures": load_curvatures,
                "load_phase_gap_interaction": interaction,
                "functional_form_signature": signature,
                "functional_form_signature_passed": signature >= gate,
                "cell_gaps": cell_gaps,
            }

    checks.update(
        {
            "all_registered_metrics_finite": metrics_finite,
            "all_registered_metrics_publicly_observed": metrics_observed,
            "load_axis_observable": any(
                report["load_axis_observable"] for report in channel_reports.values()
            ),
            "phase_volume_axis_observable": any(
                report["phase_volume_axis_observable"]
                for report in channel_reports.values()
            ),
            "functional_form_signature_resolved": any(
                report["functional_form_signature_passed"]
                for report in channel_reports.values()
            ),
            "at_least_two_supporting_cells": len(supporting_cells) >= 2,
            "participant_visible_leakage_free": not leakage_matches,
        }
    )
    return {
        "task_id": TASK_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": denominators(rows),
        "constitutive_audit": dict(constitutive_audit),
        "channel_reports": channel_reports,
        "supporting_cells": sorted(supporting_cells),
        "leakage_matches": leakage_matches,
    }


def analyze_nominal_pairs(
    rows: Sequence[Mapping[str, Any]],
    constitutive_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze the independently frozen nominal-pair constitutive Q0."""

    checks: dict[str, bool] = {
        "fixed_execution_denominator": len(rows) == 32,
        "all_outcomes_classified": all(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in rows
        ),
        "all_executions_completed": all(
            row.get("status") == "completed" for row in rows
        ),
        "registered_task_world_and_cell_bindings": (
            _nominal_pair_registered_row_bindings(rows)
        ),
        "all_exact_replay": all(row.get("exact_replay") is True for row in rows),
        "zero_platform_failures": not any(
            row.get("status") == "platform_failure" for row in rows
        ),
        "all_completed_executions_safe": all(
            row.get("status") != "completed" or row.get("safe") is True
            for row in rows
        ),
        "paired_action_plans": _nominal_pair_paired_equal(
            rows, "action_plan_sha256"
        ),
        "frozen_action_plans_bound": _nominal_pair_frozen_action_plans_bound(rows),
        "paired_hplc_noise": _nominal_pair_paired_noise_equal(rows, "hplc"),
        "paired_final_assay_noise": _nominal_pair_paired_noise_equal(
            rows, "final_assay"
        ),
        "frozen_noise_coordinates_bound": (
            _nominal_pair_frozen_noise_coordinates_bound(rows)
        ),
        "reaction_network_unchanged": (
            constitutive_audit.get("reaction_network_unchanged") is True
        ),
        "public_contract_unchanged": (
            constitutive_audit.get("public_contract_unchanged") is True
        ),
        "baseline_exponent_is_frozen": math.isclose(
            float(constitutive_audit.get("baseline_exponent", math.nan)),
            BASELINE_EXPONENT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "power_response_exponent_is_frozen": math.isclose(
            float(constitutive_audit.get("power_response_exponent", math.nan)),
            POWER_RESPONSE_EXPONENT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "only_registered_constitutive_parameter_changed": (
            constitutive_audit.get("only_registered_constitutive_parameter_changed")
            is True
            and constitutive_audit.get("changed_domain_parameter_keys")
            == ["partition_coefficient_exponent"]
        ),
        "intervention_binding_deterministic": (
            constitutive_audit.get("intervention_binding_deterministic") is True
        ),
        "execution_constitutive_binding_matches": (
            constitutive_audit.get("execution_constitutive_binding_matches") is True
            and _execution_constitutive_binding_matches(rows, constitutive_audit)
        ),
    }
    if not all(checks.values()):
        return _nominal_pair_early_result(rows, constitutive_audit, checks)

    completed = [row for row in rows if row.get("status") == "completed"]
    metrics_finite = all(
        all(
            math.isfinite(float(row["measurements"][instrument][metric]))
            for instrument in INSTRUMENTS
            for metric in METRICS
        )
        for row in completed
    )
    metrics_observed = all(
        all(
            row["observed_masks"][instrument].get(metric) is True
            for instrument in INSTRUMENTS
            for metric in METRICS
        )
        for row in completed
    )
    leakage_matches = sorted(
        {
            str(token)
            for row in rows
            for token in row.get("participant_visible_leakage_matches", [])
        }
        | {
            token
            for row in rows
            for token in FORBIDDEN_PUBLIC_TOKENS
            if token
            in json.dumps(row.get("participant_visible_payload", {}), sort_keys=True)
        }
    )
    pairs = _nominal_pair_paired_rows(rows)
    channel_reports: dict[str, dict[str, Any]] = {}
    channels_with_eight_supporting_pairs = 0
    for instrument in INSTRUMENTS:
        for metric in METRICS:
            channel_id = f"{instrument}:{metric}"
            gate = effect_gate(instrument, metric)
            pair_gaps = []
            for pair in pairs:
                baseline = float(
                    pair[LAW_IDS[0]]["measurements"][instrument][metric]
                )
                power = float(pair[LAW_IDS[1]]["measurements"][instrument][metric])
                pair_gaps.append(
                    {
                        "cell_id": pair["cell_id"],
                        "solvent": pair["solvent"],
                        "extractant": pair["extractant"],
                        LAW_IDS[0]: baseline,
                        LAW_IDS[1]: power,
                        "signed_gap": power - baseline,
                    }
                )
            solvent_ranges = _nominal_pair_axis_ranges(
                pairs, instrument, metric, "solvent"
            )
            extractant_ranges = _nominal_pair_axis_ranges(
                pairs, instrument, metric, "extractant"
            )
            supporting_count = sum(
                abs(float(item["signed_gap"])) >= gate for item in pair_gaps
            )
            if metric in {"product_in_organic", "product_in_aqueous"} and (
                supporting_count >= 8
            ):
                channels_with_eight_supporting_pairs += 1
            channel_reports[channel_id] = {
                "instrument": instrument,
                "metric": metric,
                "declared_sigma": DECLARED_SIGMA[instrument][metric],
                "effect_gate": gate,
                "max_absolute_paired_gap": max(
                    abs(float(item["signed_gap"])) for item in pair_gaps
                ),
                "supporting_pair_count": supporting_count,
                "solvent_identity_max_range": max(solvent_ranges),
                "solvent_identity_axis_observable": max(solvent_ranges) >= gate,
                "extractant_identity_max_range": max(extractant_ranges),
                "extractant_identity_axis_observable": max(extractant_ranges) >= gate,
                "pair_gaps": pair_gaps,
            }

    slope_reports: dict[str, dict[str, Any]] = {}
    allocation_positive = True
    for instrument in INSTRUMENTS:
        try:
            slope_reports[instrument] = _public_log_ratio_slope_report(
                pairs, instrument
            )
        except ValueError:
            allocation_positive = False

    checks.update(
        {
            "all_registered_metrics_finite": metrics_finite,
            "all_registered_metrics_publicly_observed": metrics_observed,
            "public_allocation_components_strictly_positive": allocation_positive,
            "solvent_identity_axis_observable": any(
                report["solvent_identity_axis_observable"]
                for report in channel_reports.values()
            ),
            "extractant_identity_axis_observable": any(
                report["extractant_identity_axis_observable"]
                for report in channel_reports.values()
            ),
            "at_least_two_product_channels_support_eight_pairs": (
                channels_with_eight_supporting_pairs >= 2
            ),
            "public_log_ratio_slope_signature_resolved": (
                allocation_positive
                and any(
                    report["slope_signature_passed"]
                    for report in slope_reports.values()
                )
            ),
            "participant_visible_leakage_free": not leakage_matches,
        }
    )
    return {
        "task_id": TASK_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": nominal_pair_denominators(rows),
        "constitutive_audit": dict(constitutive_audit),
        "channel_reports": channel_reports,
        "slope_reports": slope_reports,
        "channels_with_eight_supporting_pairs": (
            channels_with_eight_supporting_pairs
        ),
        "leakage_matches": leakage_matches,
    }


def _public_log_ratio_slope_report(
    pairs: Sequence[Mapping[str, Any]], instrument: str
) -> dict[str, Any]:
    baseline_log_ratios: list[float] = []
    power_log_ratios: list[float] = []
    components: list[tuple[float, float, float, float]] = []
    for pair in pairs:
        baseline = pair[LAW_IDS[0]]["measurements"][instrument]
        power = pair[LAW_IDS[1]]["measurements"][instrument]
        organic_0 = float(baseline["product_in_organic"])
        aqueous_0 = float(baseline["product_in_aqueous"])
        organic_1 = float(power["product_in_organic"])
        aqueous_1 = float(power["product_in_aqueous"])
        if min(organic_0, aqueous_0, organic_1, aqueous_1) <= 0.0:
            raise ValueError("public allocation log-ratio requires positive components")
        baseline_log_ratios.append(math.log(organic_0 / aqueous_0))
        power_log_ratios.append(math.log(organic_1 / aqueous_1))
        components.append((organic_0, aqueous_0, organic_1, aqueous_1))

    mean_x = sum(baseline_log_ratios) / len(baseline_log_ratios)
    mean_y = sum(power_log_ratios) / len(power_log_ratios)
    centered_x = [value - mean_x for value in baseline_log_ratios]
    centered_y = [value - mean_y for value in power_log_ratios]
    sum_xx = sum(value * value for value in centered_x)
    if sum_xx <= 0.0:
        raise ValueError("baseline public allocation log-ratio has zero variance")
    sum_xy = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    slope = sum_xy / sum_xx
    sigma_organic = DECLARED_SIGMA[instrument]["product_in_organic"]
    sigma_aqueous = DECLARED_SIGMA[instrument]["product_in_aqueous"]
    slope_variance = 0.0
    for index, (organic_0, aqueous_0, organic_1, aqueous_1) in enumerate(components):
        derivative_y = centered_x[index] / sum_xx
        derivative_x = (
            centered_y[index] * sum_xx
            - 2.0 * centered_x[index] * sum_xy
        ) / (sum_xx * sum_xx)
        derivative_organic_noise = (
            derivative_x / organic_0 + derivative_y / organic_1
        )
        derivative_aqueous_noise = -(
            derivative_x / aqueous_0 + derivative_y / aqueous_1
        )
        slope_variance += (
            sigma_organic * derivative_organic_noise
        ) ** 2 + (sigma_aqueous * derivative_aqueous_noise) ** 2
    slope_standard_error = math.sqrt(slope_variance)
    slope_gate = max(
        NOMINAL_PAIR_SLOPE_MINIMUM_DEVIATION, 6.0 * slope_standard_error
    )
    slope_deviation = abs(slope - 1.0)
    return {
        "instrument": instrument,
        "derived_quantity": "log(product_in_organic/product_in_aqueous)",
        "pair_count": len(pairs),
        "slope": slope,
        "slope_standard_error": slope_standard_error,
        "slope_minimum_deviation": NOMINAL_PAIR_SLOPE_MINIMUM_DEVIATION,
        "slope_gate": slope_gate,
        "absolute_slope_deviation_from_one": slope_deviation,
        "slope_signature_passed": slope_deviation >= slope_gate,
        "noise_propagation": (
            "first_order_paired_metric_noise_shared_between_laws_"
            "independent_across_cells_and_metrics"
        ),
    }


def _axis_ranges(
    pairs: Sequence[Mapping[str, Any]],
    instrument: str,
    metric: str,
    axis: str,
) -> list[float]:
    other = "phase_volume_index" if axis == "load_index" else "load_index"
    ranges = []
    for law_id in LAW_IDS:
        for other_index in range(3):
            values = [
                float(pair[law_id]["measurements"][instrument][metric])
                for pair in pairs
                if int(pair[other]) == other_index
            ]
            ranges.append(max(values) - min(values))
    return ranges


def _nominal_pair_axis_ranges(
    pairs: Sequence[Mapping[str, Any]],
    instrument: str,
    metric: str,
    axis: str,
) -> list[float]:
    other = "extractant" if axis == "solvent" else "solvent"
    return [
        max(values) - min(values)
        for law_id in LAW_IDS
        for other_identity in NOMINAL_IDENTITIES
        if (
            values := [
                float(pair[law_id]["measurements"][instrument][metric])
                for pair in pairs
                if int(pair[other]) == other_identity
            ]
        )
    ]


def _nominal_pair_paired_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for cell in registered_nominal_pair_cells():
        selected = [row for row in rows if row.get("cell_id") == cell["cell_id"]]
        if len(selected) != len(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} has the wrong paired denominator")
        laws = {str(row.get("law_id")): row for row in selected}
        if set(laws) != set(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} lacks its paired laws")
        pairs.append({**cell, **laws})
    return pairs


def _nominal_pair_paired_equal(
    rows: Sequence[Mapping[str, Any]], field: str
) -> bool:
    try:
        return all(
            pair[LAW_IDS[0]].get(field) == pair[LAW_IDS[1]].get(field)
            and pair[LAW_IDS[0]].get(field) is not None
            for pair in _nominal_pair_paired_rows(rows)
        )
    except (KeyError, ValueError):
        return False


def _nominal_pair_registered_row_bindings(
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    registered = {cell["cell_id"]: cell for cell in registered_nominal_pair_cells()}
    try:
        return all(
            row.get("task_id") == TASK_ID
            and row.get("world_seed") == WORLD_SEED
            and row.get("cell_id") in registered
            and all(
                row.get(key) == registered[str(row["cell_id"])][key]
                for key in registered[str(row["cell_id"])]
            )
            for row in rows
        )
    except (KeyError, TypeError):
        return False


def _nominal_pair_frozen_action_plans_bound(
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    registered = {cell["cell_id"]: cell for cell in registered_nominal_pair_cells()}
    return all(
        row.get("cell_id") in registered
        and row.get("action_plan_sha256")
        == canonical_json_sha256(
            frozen_nominal_pair_action_plan(registered[str(row["cell_id"])])
        )
        for row in rows
    )


def _nominal_pair_frozen_noise_coordinates_bound(
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    try:
        return all(
            row["noise_key_sha256"].get(instrument)
            == nominal_pair_noise_coordinate(
                str(row["cell_id"]), instrument
            ).key_sha256
            and row["observation_coordinate_sha256"].get(instrument)
            == canonical_json_sha256(
                nominal_pair_noise_coordinate(
                    str(row["cell_id"]), instrument
                ).to_audit_dict()
            )
            for row in rows
            for instrument in INSTRUMENTS
        )
    except (KeyError, TypeError, ValueError):
        return False


def _nominal_pair_paired_noise_equal(
    rows: Sequence[Mapping[str, Any]], instrument: str
) -> bool:
    try:
        return all(
            pair[LAW_IDS[0]]["noise_key_sha256"].get(instrument)
            == pair[LAW_IDS[1]]["noise_key_sha256"].get(instrument)
            and pair[LAW_IDS[0]]["noise_key_sha256"].get(instrument) is not None
            for pair in _nominal_pair_paired_rows(rows)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for cell in registered_cells():
        selected = [row for row in rows if row.get("cell_id") == cell["cell_id"]]
        if len(selected) != len(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} has the wrong paired denominator")
        laws = {str(row.get("law_id")): row for row in selected}
        if set(laws) != set(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} lacks its paired laws")
        pairs.append({**cell, **laws})
    return pairs


def _paired_equal(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    try:
        return all(
            pair[LAW_IDS[0]].get(field) == pair[LAW_IDS[1]].get(field)
            and pair[LAW_IDS[0]].get(field) is not None
            for pair in _paired_rows(rows)
        )
    except (KeyError, ValueError):
        return False


def _registered_row_bindings(rows: Sequence[Mapping[str, Any]]) -> bool:
    registered = {cell["cell_id"]: cell for cell in registered_cells()}
    try:
        return all(
            row.get("task_id") == TASK_ID
            and row.get("world_seed") == WORLD_SEED
            and row.get("cell_id") in registered
            and all(
                row.get(key) == registered[str(row["cell_id"])][key]
                for key in registered[str(row["cell_id"])]
            )
            for row in rows
        )
    except (KeyError, TypeError):
        return False


def _frozen_action_plans_bound(rows: Sequence[Mapping[str, Any]]) -> bool:
    registered = {cell["cell_id"]: cell for cell in registered_cells()}
    return all(
        row.get("cell_id") in registered
        and row.get("action_plan_sha256")
        == canonical_json_sha256(frozen_action_plan(registered[str(row["cell_id"])]))
        for row in rows
    )


def _frozen_noise_coordinates_bound(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return all(
            row["noise_key_sha256"].get(instrument)
            == noise_coordinate(str(row["cell_id"]), instrument).key_sha256
            and row["observation_coordinate_sha256"].get(instrument)
            == canonical_json_sha256(
                noise_coordinate(str(row["cell_id"]), instrument).to_audit_dict()
            )
            for row in rows
            for instrument in INSTRUMENTS
        )
    except (KeyError, TypeError, ValueError):
        return False


def _execution_constitutive_binding_matches(
    rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> bool:
    baseline_hashes = {
        row.get("constitutive_intervention_hash")
        for row in rows
        if row.get("law_id") == LAW_IDS[0]
    }
    power_hashes = {
        row.get("constitutive_intervention_hash")
        for row in rows
        if row.get("law_id") == LAW_IDS[1]
    }
    mechanism_hashes = {row.get("mechanism_hash") for row in rows}
    task_contract_hashes = {row.get("task_contract_hash") for row in rows}
    return (
        baseline_hashes == {None}
        and power_hashes == {audit.get("power_response_intervention_hash")}
        and mechanism_hashes
        == {
            audit.get("baseline_mechanism_hash"),
            audit.get("power_response_mechanism_hash"),
        }
        and None not in mechanism_hashes
        and task_contract_hashes
        == {
            audit.get("baseline_public_task_contract_hash"),
            audit.get("power_response_public_task_contract_hash"),
        }
        and None not in task_contract_hashes
    )


def _paired_noise_equal(rows: Sequence[Mapping[str, Any]], instrument: str) -> bool:
    try:
        return all(
            pair[LAW_IDS[0]]["noise_key_sha256"].get(instrument)
            == pair[LAW_IDS[1]]["noise_key_sha256"].get(instrument)
            and pair[LAW_IDS[0]]["noise_key_sha256"].get(instrument) is not None
            for pair in _paired_rows(rows)
        )
    except (KeyError, TypeError, ValueError):
        return False


def denominators(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "planned": 18,
        "attempted": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "exact_replay": sum(row.get("exact_replay") is True for row in rows),
        "physical_failures": sum(row.get("status") == "physical_failure" for row in rows),
        "platform_failures": sum(row.get("status") == "platform_failure" for row in rows),
        "unsafe_completed": sum(
            row.get("status") == "completed" and row.get("safe") is False for row in rows
        ),
    }


def nominal_pair_denominators(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "planned": 32,
        "attempted": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "exact_replay": sum(row.get("exact_replay") is True for row in rows),
        "physical_failures": sum(
            row.get("status") == "physical_failure" for row in rows
        ),
        "platform_failures": sum(
            row.get("status") == "platform_failure" for row in rows
        ),
        "unsafe_completed": sum(
            row.get("status") == "completed" and row.get("safe") is False
            for row in rows
        ),
    }


def _early_result(
    rows: Sequence[Mapping[str, Any]],
    constitutive_audit: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": denominators(rows),
        "constitutive_audit": dict(constitutive_audit),
        "channel_reports": None,
        "supporting_cells": [],
        "leakage_matches": [],
    }


def _nominal_pair_early_result(
    rows: Sequence[Mapping[str, Any]],
    constitutive_audit: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": nominal_pair_denominators(rows),
        "constitutive_audit": dict(constitutive_audit),
        "channel_reports": None,
        "slope_reports": None,
        "channels_with_eight_supporting_pairs": 0,
        "leakage_matches": [],
    }


def task_report_sha256(report: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def summary_sha256(summary: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )


def _execution_errors(
    payload: Mapping[str, Any],
    *,
    root: Path | None,
    expected_execution_context: WorkIIExecutionContext | None,
) -> tuple[list[str], str | None]:
    envelope = payload.get("execution_context")
    if not isinstance(envelope, Mapping):
        return ["partition constitutive Q0 lacks an execution context"], None
    if root is None:
        mode = envelope.get("execution_mode")
        if mode not in {item.value for item in ExecutionMode}:
            return ["partition constitutive Q0 has an invalid execution mode"], None
        return [], str(mode)
    return (
        validate_execution_envelope(
            root, envelope, expected_context=expected_execution_context
        ),
        str(envelope.get("execution_mode")),
    )


def validate_task_report(
    report: Mapping[str, Any],
    *,
    root: Path | None = None,
    expected_execution_context: WorkIIExecutionContext | None = None,
) -> list[str]:
    errors: list[str] = []
    execution_errors, _mode = _execution_errors(
        report,
        root=root,
        expected_execution_context=expected_execution_context,
    )
    errors.extend(execution_errors)
    if report.get("schema_version") != TASK_REPORT_VERSION:
        errors.append("unexpected partition constitutive Q0 task-report schema")
    if report.get("qualification_schema_version") != QUALIFICATION_VERSION:
        errors.append("partition constitutive Q0 qualification schema mismatch")
    if report.get("formal_result") is not False:
        errors.append("partition constitutive Q0 must not be marked formal")
    if report.get("provider_call_count") != 0 or report.get("participant_session_count") != 0:
        errors.append("partition constitutive Q0 must be provider-free")
    if report.get("task_id") != TASK_ID or report.get("world_seed") != WORLD_SEED:
        errors.append("partition constitutive Q0 task/world binding mismatch")
    if report.get("frozen_exponents") != {
        LAW_IDS[0]: BASELINE_EXPONENT,
        LAW_IDS[1]: POWER_RESPONSE_EXPONENT,
    }:
        errors.append("partition constitutive Q0 frozen exponent contract mismatch")
    if report.get("report_sha256") != task_report_sha256(report):
        errors.append("partition constitutive Q0 task-report self-hash mismatch")
    rows = report.get("rows")
    audit = report.get("constitutive_audit")
    if not isinstance(rows, list) or not isinstance(audit, Mapping):
        errors.append("partition constitutive Q0 task report lacks rows or audit")
    else:
        try:
            rebuilt = analyze(rows, audit)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"partition constitutive Q0 analysis cannot be rebuilt: {error}")
        else:
            if report.get("analysis") != rebuilt:
                errors.append("partition constitutive Q0 task analysis mismatch")
    return errors


def validate_summary(
    summary: Mapping[str, Any],
    *,
    root: Path | None = None,
    expected_execution_context: WorkIIExecutionContext | None = None,
) -> list[str]:
    errors: list[str] = []
    execution_errors, _mode = _execution_errors(
        summary,
        root=root,
        expected_execution_context=expected_execution_context,
    )
    errors.extend(execution_errors)
    if summary.get("schema_version") != SUMMARY_VERSION:
        errors.append("unexpected partition constitutive Q0 summary schema")
    if summary.get("qualification_schema_version") != QUALIFICATION_VERSION:
        errors.append("partition constitutive Q0 summary qualification schema mismatch")
    if summary.get("summary_sha256") != summary_sha256(summary):
        errors.append("partition constitutive Q0 summary self-hash mismatch")
    if summary.get("formal_result") is not False:
        errors.append("partition constitutive Q0 summary must not be formal")
    if summary.get("task_id") != TASK_ID or summary.get("world_seed") != WORLD_SEED:
        errors.append("partition constitutive Q0 summary task/world binding mismatch")
    if summary.get("provider_call_count") != 0 or summary.get("participant_session_count") != 0:
        errors.append("partition constitutive Q0 summary must be provider-free")
    analysis = summary.get("analysis")
    if not isinstance(analysis, Mapping):
        errors.append("partition constitutive Q0 summary lacks analysis")
    else:
        passed = analysis.get("passed") is True
        if summary.get("denominators") != analysis.get("denominators"):
            errors.append("partition constitutive Q0 summary denominator mismatch")
        if summary.get("five_world_provider_free_expansion_authorized") is not passed:
            errors.append("partition constitutive Q0 expansion authorization mismatch")
        expected_decision = (
            "platform_defect_stop_and_rerun_whole_block_after_fix"
            if summary.get("platform_stop_triggered") is True
            else "proceed_to_unchanged_five_world_provider_free_qualification"
            if passed
            else "retain_q0_scientific_rejection_and_do_not_expand"
        )
        if summary.get("decision") != expected_decision:
            errors.append("partition constitutive Q0 decision mismatch")
    if summary.get("participant_d1_authorized") is not False:
        errors.append("partition constitutive Q0 must not authorize participant D1")
    if summary.get("provider_execution_authorized") is not False:
        errors.append("partition constitutive Q0 must not authorize provider execution")
    expected_coverage = {
        "law_ids": list(LAW_IDS),
        "grid_axes": {
            "aqueous_load_volume_L": list(LOAD_LEVELS_L),
            "extractant_phase_volume_L": list(PHASE_VOLUME_LEVELS_L),
        },
        "grid_cell_count": len(registered_cells()),
        "planned_execution_count": 18,
        "attempted_execution_count": (
            analysis.get("denominators", {}).get("attempted")
            if isinstance(analysis, Mapping)
            else None
        ),
    }
    if summary.get("coverage") != expected_coverage:
        errors.append("partition constitutive Q0 summary coverage mismatch")
    binding = summary.get("raw_binding")
    if root is not None:
        if not isinstance(binding, Mapping):
            errors.append("partition constitutive Q0 raw binding is missing")
        else:
            path_value = binding.get("path")
            try:
                path = (root / str(path_value)).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError("raw task report is outside the repository or missing")
                if binding.get("sha256") != file_sha256(path):
                    errors.append("partition constitutive Q0 raw file hash mismatch")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("raw task report is not an object")
                errors.extend(
                    validate_task_report(
                        payload,
                        root=root,
                        expected_execution_context=expected_execution_context,
                    )
                )
                if payload.get("execution_context") != summary.get("execution_context"):
                    errors.append("partition constitutive Q0 raw/execution context mismatch")
                if binding.get("report_sha256") != payload.get("report_sha256"):
                    errors.append("partition constitutive Q0 embedded report hash mismatch")
                if summary.get("analysis") != payload.get("analysis"):
                    errors.append("partition constitutive Q0 summary/raw analysis mismatch")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"partition constitutive Q0 raw binding cannot be read: {error}")
    return errors


def validate_nominal_pair_task_report(
    report: Mapping[str, Any],
    *,
    root: Path | None = None,
    expected_execution_context: WorkIIExecutionContext | None = None,
) -> list[str]:
    errors: list[str] = []
    execution_errors, _mode = _execution_errors(
        report,
        root=root,
        expected_execution_context=expected_execution_context,
    )
    errors.extend(execution_errors)
    if report.get("schema_version") != NOMINAL_PAIR_TASK_REPORT_VERSION:
        errors.append("unexpected partition nominal-pair Q0 task-report schema")
    if (
        report.get("qualification_schema_version")
        != NOMINAL_PAIR_QUALIFICATION_VERSION
    ):
        errors.append("partition nominal-pair Q0 qualification schema mismatch")
    if report.get("formal_result") is not False:
        errors.append("partition nominal-pair Q0 must not be marked formal")
    if report.get("provider_call_count") != 0 or report.get(
        "participant_session_count"
    ) != 0:
        errors.append("partition nominal-pair Q0 must be provider-free")
    if report.get("task_id") != TASK_ID or report.get("world_seed") != WORLD_SEED:
        errors.append("partition nominal-pair Q0 task/world binding mismatch")
    if report.get("frozen_exponents") != {
        LAW_IDS[0]: BASELINE_EXPONENT,
        LAW_IDS[1]: POWER_RESPONSE_EXPONENT,
    }:
        errors.append("partition nominal-pair Q0 frozen exponent contract mismatch")
    if report.get("report_sha256") != task_report_sha256(report):
        errors.append("partition nominal-pair Q0 task-report self-hash mismatch")
    rows = report.get("rows")
    audit = report.get("constitutive_audit")
    if not isinstance(rows, list) or not isinstance(audit, Mapping):
        errors.append("partition nominal-pair Q0 task report lacks rows or audit")
    else:
        try:
            rebuilt = analyze_nominal_pairs(rows, audit)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            errors.append(f"partition nominal-pair Q0 analysis cannot be rebuilt: {error}")
        else:
            if report.get("analysis") != rebuilt:
                errors.append("partition nominal-pair Q0 task analysis mismatch")
    return errors


def validate_nominal_pair_summary(
    summary: Mapping[str, Any],
    *,
    root: Path | None = None,
    expected_execution_context: WorkIIExecutionContext | None = None,
) -> list[str]:
    errors: list[str] = []
    execution_errors, _mode = _execution_errors(
        summary,
        root=root,
        expected_execution_context=expected_execution_context,
    )
    errors.extend(execution_errors)
    if summary.get("schema_version") != NOMINAL_PAIR_SUMMARY_VERSION:
        errors.append("unexpected partition nominal-pair Q0 summary schema")
    if (
        summary.get("qualification_schema_version")
        != NOMINAL_PAIR_QUALIFICATION_VERSION
    ):
        errors.append("partition nominal-pair Q0 summary qualification schema mismatch")
    if summary.get("summary_sha256") != summary_sha256(summary):
        errors.append("partition nominal-pair Q0 summary self-hash mismatch")
    if summary.get("formal_result") is not False:
        errors.append("partition nominal-pair Q0 summary must not be formal")
    if summary.get("task_id") != TASK_ID or summary.get("world_seed") != WORLD_SEED:
        errors.append("partition nominal-pair Q0 summary task/world binding mismatch")
    if summary.get("provider_call_count") != 0 or summary.get(
        "participant_session_count"
    ) != 0:
        errors.append("partition nominal-pair Q0 summary must be provider-free")
    analysis = summary.get("analysis")
    if not isinstance(analysis, Mapping):
        errors.append("partition nominal-pair Q0 summary lacks analysis")
    else:
        passed = analysis.get("passed") is True
        if summary.get("denominators") != analysis.get("denominators"):
            errors.append("partition nominal-pair Q0 summary denominator mismatch")
        if summary.get("five_world_provider_free_expansion_authorized") is not passed:
            errors.append("partition nominal-pair Q0 expansion authorization mismatch")
        expected_decision = (
            "platform_defect_stop_and_rerun_whole_block_after_fix"
            if summary.get("platform_stop_triggered") is True
            else "proceed_to_unchanged_five_world_provider_free_qualification"
            if passed
            else "retain_q0_scientific_rejection_and_do_not_expand"
        )
        if summary.get("decision") != expected_decision:
            errors.append("partition nominal-pair Q0 decision mismatch")
    if summary.get("participant_d1_authorized") is not False:
        errors.append("partition nominal-pair Q0 must not authorize participant D1")
    if summary.get("provider_execution_authorized") is not False:
        errors.append("partition nominal-pair Q0 must not authorize provider execution")
    expected_coverage = {
        "law_ids": list(LAW_IDS),
        "grid_axes": {
            "solvent": list(NOMINAL_IDENTITIES),
            "extractant": list(NOMINAL_IDENTITIES),
        },
        "fixed_coordinates": {
            "aqueous_volume_L": NOMINAL_PAIR_AQUEOUS_VOLUME_L,
            "extractant_volume_L": NOMINAL_PAIR_EXTRACTANT_VOLUME_L,
            "solvent_volume_L": NOMINAL_PAIR_SOLVENT_VOLUME_L,
        },
        "grid_cell_count": len(registered_nominal_pair_cells()),
        "planned_execution_count": 32,
        "attempted_execution_count": (
            analysis.get("denominators", {}).get("attempted")
            if isinstance(analysis, Mapping)
            else None
        ),
    }
    if summary.get("coverage") != expected_coverage:
        errors.append("partition nominal-pair Q0 summary coverage mismatch")
    binding = summary.get("raw_binding")
    if root is not None:
        if not isinstance(binding, Mapping):
            errors.append("partition nominal-pair Q0 raw binding is missing")
        else:
            path_value = binding.get("path")
            try:
                path = (root / str(path_value)).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError("raw task report is outside the repository or missing")
                if binding.get("sha256") != file_sha256(path):
                    errors.append("partition nominal-pair Q0 raw file hash mismatch")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("raw task report is not an object")
                errors.extend(
                    validate_nominal_pair_task_report(
                        payload,
                        root=root,
                        expected_execution_context=expected_execution_context,
                    )
                )
                if payload.get("execution_context") != summary.get("execution_context"):
                    errors.append(
                        "partition nominal-pair Q0 raw/execution context mismatch"
                    )
                if binding.get("report_sha256") != payload.get("report_sha256"):
                    errors.append("partition nominal-pair Q0 embedded report hash mismatch")
                if summary.get("analysis") != payload.get("analysis"):
                    errors.append("partition nominal-pair Q0 summary/raw analysis mismatch")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(
                    f"partition nominal-pair Q0 raw binding cannot be read: {error}"
                )
    return errors


__all__ = [
    "BASELINE_EXPONENT",
    "DECLARED_SIGMA",
    "INSTRUMENTS",
    "LAW_IDS",
    "METRICS",
    "NOMINAL_IDENTITIES",
    "NOMINAL_PAIR_AQUEOUS_VOLUME_L",
    "NOMINAL_PAIR_EXTRACTANT_VOLUME_L",
    "NOMINAL_PAIR_QUALIFICATION_VERSION",
    "NOMINAL_PAIR_SLOPE_MINIMUM_DEVIATION",
    "NOMINAL_PAIR_SOLVENT_VOLUME_L",
    "NOMINAL_PAIR_SUMMARY_VERSION",
    "NOMINAL_PAIR_TASK_REPORT_VERSION",
    "POWER_RESPONSE_EXPONENT",
    "QUALIFICATION_VERSION",
    "SUMMARY_VERSION",
    "TASK_ID",
    "TASK_REPORT_VERSION",
    "WORLD_SEED",
    "analyze",
    "analyze_nominal_pairs",
    "constitutive_intervention",
    "denominators",
    "effect_gate",
    "frozen_action_plan",
    "frozen_nominal_pair_action_plan",
    "noise_coordinate",
    "nominal_pair_denominators",
    "nominal_pair_noise_coordinate",
    "nominal_pair_observation_binding",
    "observation_binding",
    "registered_cells",
    "registered_nominal_pair_cells",
    "summary_sha256",
    "task_report_sha256",
    "validate_nominal_pair_summary",
    "validate_nominal_pair_task_report",
    "validate_summary",
    "validate_task_report",
]
