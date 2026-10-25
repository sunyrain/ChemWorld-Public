"""Fail-closed analysis for the frozen Work I latent-terminal audit.

This module is deliberately pure analysis.  It neither reconstructs a chemical
state nor executes a terminal branch.  L05 may pass evaluator receipts to this
surface only after its separate replay and execution gates have passed.  L04's
qualification uses synthetic receipts exclusively.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from chemworld.eval.latent_terminal_contract import (
    EXPECTED_ASSAY_COUNT,
    EXPECTED_CELL_COUNT,
    EXPECTED_DISCARD_COUNT,
    EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT,
    EXPECTED_LIFECYCLE_COUNT,
    PRIMARY_RELATIVE_THRESHOLD,
    REGISTERED_TASK_THRESHOLD,
    RELATIVE_THRESHOLD_SENSITIVITY,
    validate_latent_terminal_contract,
)
from chemworld.eval.provenance import canonical_json_sha256

ANALYSIS_SCHEMA_ID = "chemworld.latent_terminal_analysis"
ANALYSIS_SCHEMA_VERSION = "0.1.0"
FROZEN_CONTRACT_SHA256 = (
    "55a0d6a7cb983ce4099dbea24586ea63ccc9433e106d36011d349472809efe30"
)
ALLOWED_MODES = {"synthetic_qualification", "formal_shadow_analysis"}
UNRESOLVED_CATEGORIES = {
    "prefix",
    "identity",
    "evaluator",
    "resource",
    "nonfinite_score",
}

AnalysisMode = Literal["synthetic_qualification", "formal_shadow_analysis"]


class LatentTerminalAnalysisError(ValueError):
    """Raised when analysis cannot preserve the frozen population identity."""


@dataclass(frozen=True)
class _Unit:
    discard_id: str
    cell_id: str
    world_seed: int
    information_arm: str
    lifecycle_index: int
    terminal_step: int
    campaign_best: float
    prior_incumbent: float | None
    score: float | None
    unresolved_category: str | None
    unresolved_reason: str | None
    binding_errors: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        return self.score is not None


@dataclass(frozen=True)
class _ThresholdSpec:
    threshold_id: str
    kind: Literal["relative", "absolute"]
    value: float
    primary: bool


def finite_population_fraction(numerator: int, denominator: int) -> dict[str, Any]:
    """Return an auditable fixed-denominator fraction, null at zero denominator."""

    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise LatentTerminalAnalysisError("invalid finite-population fraction")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def _as_finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _linear_percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _empirical_cdf(values: Sequence[float]) -> list[dict[str, float | int]]:
    counts = Counter(values)
    running = 0
    result: list[dict[str, float | int]] = []
    for value in sorted(counts):
        running += counts[value]
        result.append(
            {
                "value": value,
                "cumulative_count": running,
                "cumulative_fraction": running / len(values),
            }
        )
    return result


def _summary(values: Sequence[float]) -> dict[str, Any]:
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "25th_percentile_linear": None,
            "median_linear": None,
            "75th_percentile_linear": None,
            "maximum": None,
            "empirical_cdf": [],
        }
    mean = statistics.fmean(values)
    standard_deviation = math.sqrt(
        statistics.fmean((value - mean) ** 2 for value in values)
    )
    return {
        "count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": min(values),
        "25th_percentile_linear": _linear_percentile(values, 0.25),
        "median_linear": _linear_percentile(values, 0.50),
        "75th_percentile_linear": _linear_percentile(values, 0.75),
        "maximum": max(values),
        "empirical_cdf": _empirical_cdf(values),
    }


def _cdf_band(intervals: Sequence[tuple[float, float]]) -> list[dict[str, Any]]:
    if not intervals:
        return []
    support = sorted({value for interval in intervals for value in interval})
    result: list[dict[str, Any]] = []
    denominator = len(intervals)
    for value in support:
        definitely_at_or_below = sum(upper <= value for _, upper in intervals)
        possibly_at_or_below = sum(lower <= value for lower, _ in intervals)
        result.append(
            {
                "value": value,
                "lower_cumulative_fraction": definitely_at_or_below / denominator,
                "upper_cumulative_fraction": possibly_at_or_below / denominator,
            }
        )
    return result


def _interval_summary(intervals: Sequence[tuple[float, float]]) -> dict[str, Any]:
    for lower, upper in intervals:
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise LatentTerminalAnalysisError("invalid estimand support interval")
    lowers = [item[0] for item in intervals]
    uppers = [item[1] for item in intervals]
    lower_summary = _summary(lowers)
    upper_summary = _summary(uppers)
    order_fields = (
        "mean",
        "minimum",
        "25th_percentile_linear",
        "median_linear",
        "75th_percentile_linear",
        "maximum",
    )
    return {
        "fixed_denominator": len(intervals),
        "mean_and_order_statistic_bounds": {
            field: {
                "lower": lower_summary[field],
                "upper": upper_summary[field],
            }
            for field in order_fields
        },
        "empirical_cdf_band": _cdf_band(intervals),
        "all_unresolved_score_zero_endpoint": lower_summary,
        "all_unresolved_score_one_endpoint": upper_summary,
        "standard_deviation_bound_registered": False,
    }


def _contract_cells(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    population = contract.get("population")
    if not isinstance(population, Mapping):
        raise LatentTerminalAnalysisError("contract population is missing")
    cells = population.get("cells")
    if not isinstance(cells, list) or not all(isinstance(cell, Mapping) for cell in cells):
        raise LatentTerminalAnalysisError("contract cells are invalid")
    return list(cells)


def _validate_contract_binding(contract: Mapping[str, Any]) -> None:
    if contract.get("contract_sha256") != FROZEN_CONTRACT_SHA256:
        raise LatentTerminalAnalysisError("contract is not the frozen L01 binding")
    errors = validate_latent_terminal_contract(contract)
    if errors:
        raise LatentTerminalAnalysisError("invalid frozen contract: " + "; ".join(errors))


def _prior_incumbent(cell: Mapping[str, Any], terminal_step: int) -> float | None:
    assays = cell.get("observed_assays")
    if not isinstance(assays, list):
        raise LatentTerminalAnalysisError("cell observed assays are invalid")
    prior: list[float] = []
    for assay in assays:
        if not isinstance(assay, Mapping):
            raise LatentTerminalAnalysisError("observed assay row is invalid")
        step = assay.get("terminal_step")
        score = _as_finite_score(assay.get("score"))
        if isinstance(step, bool) or not isinstance(step, int) or score is None:
            raise LatentTerminalAnalysisError("observed assay identity is invalid")
        if step < terminal_step:
            prior.append(score)
    return max(prior) if prior else None


def _expected_units(
    contract: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for cell in _contract_cells(contract):
        units = cell.get("discard_units")
        if not isinstance(units, list):
            raise LatentTerminalAnalysisError("contract discard units are invalid")
        for unit in units:
            if not isinstance(unit, Mapping):
                raise LatentTerminalAnalysisError("contract discard unit is invalid")
            discard_id = unit.get("discard_id")
            if not isinstance(discard_id, str) or discard_id in result:
                raise LatentTerminalAnalysisError("discard identity is missing or duplicated")
            result[discard_id] = (cell, unit)
    if len(result) != EXPECTED_DISCARD_COUNT:
        raise LatentTerminalAnalysisError("contract does not enumerate 36 discards")
    return result


def _unresolved_unit(
    cell: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    category: str,
    reason: str,
    binding_errors: Sequence[str] = (),
) -> _Unit:
    terminal_step = int(expected["terminal_step"])
    return _Unit(
        discard_id=str(expected["discard_id"]),
        cell_id=str(cell["cell_id"]),
        world_seed=int(cell["world_seed"]),
        information_arm=str(cell["information_arm"]),
        lifecycle_index=int(expected["lifecycle_index"]),
        terminal_step=terminal_step,
        campaign_best=float(cell["campaign_best_assayed_score"]),
        prior_incumbent=_prior_incumbent(cell, terminal_step),
        score=None,
        unresolved_category=category,
        unresolved_reason=reason,
        binding_errors=tuple(binding_errors),
    )


def _parse_unit(
    contract: Mapping[str, Any],
    cell: Mapping[str, Any],
    expected: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    *,
    mode: AnalysisMode,
) -> _Unit:
    if receipt is None:
        return _unresolved_unit(
            cell,
            expected,
            category="identity",
            reason="receipt_missing",
            binding_errors=("receipt_missing",),
        )
    population = contract["population"]
    fixture_kind = (
        "synthetic_qualification"
        if mode == "synthetic_qualification"
        else "formal_shadow_receipt"
    )
    required = {
        "contract_sha256": contract["contract_sha256"],
        "population_manifest_sha256": population["population_manifest_sha256"],
        "fixture_kind": fixture_kind,
        "discard_id": expected["discard_id"],
        "cell_id": cell["cell_id"],
        "world_seed": cell["world_seed"],
        "information_arm": cell["information_arm"],
        "lifecycle_index": expected["lifecycle_index"],
        "terminal_step": expected["terminal_step"],
        "public_prefix_sha256": expected["public_prefix_sha256"],
        "terminal_action_sha256": expected["terminal_action_sha256"],
    }
    binding_errors = tuple(
        f"{field}_mismatch"
        for field, expected_value in required.items()
        if receipt.get(field) != expected_value
    )
    if binding_errors:
        return _unresolved_unit(
            cell,
            expected,
            category="identity",
            reason="receipt_binding_mismatch",
            binding_errors=binding_errors,
        )
    status = receipt.get("outcome_status")
    if status == "unresolved":
        category = receipt.get("failure_category")
        if category not in UNRESOLVED_CATEGORIES:
            category = "evaluator"
            binding_errors = ("unregistered_failure_category",)
        reason = receipt.get("failure_reason")
        if not isinstance(reason, str) or not reason:
            reason = "unresolved_without_failure_reason"
            binding_errors = (*binding_errors, "failure_reason_missing")
        if receipt.get("score") is not None:
            reason = "forbidden_score_on_unresolved_receipt"
            binding_errors = (*binding_errors, "forbidden_unresolved_score")
        return _unresolved_unit(
            cell,
            expected,
            category=str(category),
            reason=reason,
            binding_errors=binding_errors,
        )
    if status != "resolved":
        return _unresolved_unit(
            cell,
            expected,
            category="evaluator",
            reason="invalid_outcome_status",
            binding_errors=("invalid_outcome_status",),
        )
    score = _as_finite_score(receipt.get("score"))
    if score is None:
        return _unresolved_unit(
            cell,
            expected,
            category="nonfinite_score",
            reason="score_is_missing_nonfinite_or_out_of_range",
            binding_errors=("invalid_score",),
        )
    terminal_step = int(expected["terminal_step"])
    return _Unit(
        discard_id=str(expected["discard_id"]),
        cell_id=str(cell["cell_id"]),
        world_seed=int(cell["world_seed"]),
        information_arm=str(cell["information_arm"]),
        lifecycle_index=int(expected["lifecycle_index"]),
        terminal_step=terminal_step,
        campaign_best=float(cell["campaign_best_assayed_score"]),
        prior_incumbent=_prior_incumbent(cell, terminal_step),
        score=score,
        unresolved_category=None,
        unresolved_reason=None,
        binding_errors=(),
    )


def _parse_receipts(
    contract: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    mode: AnalysisMode,
) -> list[_Unit]:
    expected = _expected_units(contract)
    supplied: dict[str, Mapping[str, Any]] = {}
    unknown: list[str] = []
    for receipt in receipts:
        discard_id = receipt.get("discard_id")
        if not isinstance(discard_id, str) or discard_id not in expected:
            unknown.append(str(discard_id))
            continue
        if discard_id in supplied:
            raise LatentTerminalAnalysisError(f"duplicate receipt: {discard_id}")
        supplied[discard_id] = receipt
    if unknown:
        raise LatentTerminalAnalysisError(
            "receipts contain unknown discard identities: " + ", ".join(sorted(unknown))
        )
    return [
        _parse_unit(contract, cell, unit, supplied.get(discard_id), mode=mode)
        for discard_id, (cell, unit) in expected.items()
    ]


def _threshold_specs() -> list[_ThresholdSpec]:
    result = [
        _ThresholdSpec(
            threshold_id=f"relative_{fraction:.2f}",
            kind="relative",
            value=fraction,
            primary=fraction == PRIMARY_RELATIVE_THRESHOLD,
        )
        for fraction in RELATIVE_THRESHOLD_SENSITIVITY
    ]
    result.append(
        _ThresholdSpec(
            threshold_id=f"absolute_{REGISTERED_TASK_THRESHOLD:.2f}",
            kind="absolute",
            value=REGISTERED_TASK_THRESHOLD,
            primary=False,
        )
    )
    return result


def _threshold(campaign_best: float, spec: _ThresholdSpec) -> float:
    return campaign_best * spec.value if spec.kind == "relative" else spec.value


def _assay_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in _contract_cells(contract):
        assays = cell.get("observed_assays")
        if not isinstance(assays, list):
            raise LatentTerminalAnalysisError("observed assay rows are invalid")
        for assay in assays:
            if not isinstance(assay, Mapping):
                raise LatentTerminalAnalysisError("observed assay row is invalid")
            score = _as_finite_score(assay.get("score"))
            if score is None:
                raise LatentTerminalAnalysisError("observed assay score is invalid")
            rows.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "world_seed": int(cell["world_seed"]),
                    "information_arm": str(cell["information_arm"]),
                    "score": score,
                    "campaign_best": float(cell["campaign_best_assayed_score"]),
                }
            )
    if len(rows) != EXPECTED_ASSAY_COUNT:
        raise LatentTerminalAnalysisError("contract does not enumerate 24 assays")
    return rows


def _table_metrics(table: Mapping[str, int]) -> dict[str, Any]:
    tp = table["TP"]
    fp = table["FP"]
    fn = table["FN"]
    tn = table["TN"]
    return {
        "false_discard_fraction": finite_population_fraction(fn, fn + tn),
        "assay_commitment_precision": finite_population_fraction(tp, tp + fp),
        "assay_commitment_recall": finite_population_fraction(tp, tp + fn),
    }


def _selection_subset(
    assays: Sequence[Mapping[str, Any]],
    units: Sequence[_Unit],
    spec: _ThresholdSpec,
    *,
    point_estimates_allowed: bool,
) -> dict[str, Any]:
    tp = sum(
        float(row["score"]) >= _threshold(float(row["campaign_best"]), spec)
        for row in assays
    )
    fp = len(assays) - tp
    resolved_fn = sum(
        unit.resolved
        and unit.score is not None
        and unit.score >= _threshold(unit.campaign_best, spec)
        for unit in units
    )
    resolved_tn = sum(unit.resolved for unit in units) - resolved_fn
    unresolved = sum(not unit.resolved for unit in units)
    all_zero_fn = resolved_fn + sum(
        _threshold(unit.campaign_best, spec) <= 0.0
        for unit in units
        if not unit.resolved
    )
    all_one_fn = resolved_fn + sum(
        _threshold(unit.campaign_best, spec) <= 1.0
        for unit in units
        if not unit.resolved
    )
    all_zero = {
        "TP": tp,
        "FP": fp,
        "FN": all_zero_fn,
        "TN": len(units) - all_zero_fn,
    }
    all_one = {
        "TP": tp,
        "FP": fp,
        "FN": all_one_fn,
        "TN": len(units) - all_one_fn,
    }
    point_table = None
    point_metrics = None
    if point_estimates_allowed:
        point_table = {
            "TP": tp,
            "FP": fp,
            "FN": resolved_fn,
            "TN": resolved_tn,
        }
        point_metrics = _table_metrics(point_table)
    zero_metrics = _table_metrics(all_zero)
    one_metrics = _table_metrics(all_one)
    return {
        "fixed_lifecycle_denominator": len(assays) + len(units),
        "assay_denominator": len(assays),
        "discard_denominator": len(units),
        "resolved_counts": {
            "TP": tp,
            "FP": fp,
            "FN": resolved_fn,
            "TN": resolved_tn,
            "unresolved_discards": unresolved,
        },
        "point_table": point_table,
        "point_metrics": point_metrics,
        "all_unresolved_score_zero_table": all_zero,
        "all_unresolved_score_one_table": all_one,
        "bounds": {
            "false_discard_fraction": {
                "lower": zero_metrics["false_discard_fraction"],
                "upper": one_metrics["false_discard_fraction"],
            },
            "assay_commitment_precision": {
                "lower": zero_metrics["assay_commitment_precision"],
                "upper": one_metrics["assay_commitment_precision"],
            },
            "assay_commitment_recall": {
                "lower": one_metrics["assay_commitment_recall"],
                "upper": zero_metrics["assay_commitment_recall"],
            },
        },
    }


def _in_stratum(item: Mapping[str, Any] | _Unit, stratum: str) -> bool:
    if stratum == "overall":
        return True
    field, value = stratum.split("/", 1)
    source_field = "cell_id" if field == "campaign_cell" else field
    observed = (
        getattr(item, source_field) if isinstance(item, _Unit) else item[source_field]
    )
    return str(observed) == value


def _strata(contract: Mapping[str, Any]) -> list[str]:
    cells = _contract_cells(contract)
    arms = sorted({str(cell["information_arm"]) for cell in cells})
    worlds = sorted({int(cell["world_seed"]) for cell in cells})
    cell_ids = [str(cell["cell_id"]) for cell in cells]
    return [
        "overall",
        *(f"information_arm/{arm}" for arm in arms),
        *(f"world_seed/{world}" for world in worlds),
        *(f"campaign_cell/{cell_id}" for cell_id in cell_ids),
    ]


def _selection_analysis(
    contract: Mapping[str, Any],
    units: Sequence[_Unit],
    *,
    all_resolved: bool,
) -> list[dict[str, Any]]:
    assays = _assay_rows(contract)
    result: list[dict[str, Any]] = []
    for spec in _threshold_specs():
        rows: dict[str, Any] = {}
        for stratum in _strata(contract):
            stratum_assays = [item for item in assays if _in_stratum(item, stratum)]
            stratum_units = [item for item in units if _in_stratum(item, stratum)]
            rows[stratum] = _selection_subset(
                stratum_assays,
                stratum_units,
                spec,
                point_estimates_allowed=all_resolved,
            )
        result.append(
            {
                "threshold_id": spec.threshold_id,
                "kind": spec.kind,
                "value": spec.value,
                "primary": spec.primary,
                "positive_comparator": ">=",
                "strata": rows,
            }
        )
    return result


def _unit_strata_values(
    contract: Mapping[str, Any],
    values: Mapping[str, tuple[float, float]],
    units: Sequence[_Unit],
    *,
    point_estimates_allowed: bool,
) -> dict[str, Any]:
    by_id = {unit.discard_id: unit for unit in units}
    strata: dict[str, Any] = {}
    for stratum in _strata(contract):
        selected = [
            interval
            for discard_id, interval in values.items()
            if _in_stratum(by_id[discard_id], stratum)
        ]
        exact_values = [lower for lower, upper in selected if lower == upper]
        strata[stratum] = {
            "fixed_denominator": len(selected),
            "point_summary": (
                _summary(exact_values)
                if point_estimates_allowed and len(exact_values) == len(selected)
                else None
            ),
            "bounds": _interval_summary(selected),
            "observed_only_diagnostic": _summary(exact_values),
            "observed_only_is_primary": False,
        }
    cell_rows = [
        row
        for key, row in strata.items()
        if key.startswith("campaign_cell/") and row["fixed_denominator"] > 0
    ]
    cell_mean_intervals = [
        (
            float(row["bounds"]["mean_and_order_statistic_bounds"]["mean"]["lower"]),
            float(row["bounds"]["mean_and_order_statistic_bounds"]["mean"]["upper"]),
        )
        for row in cell_rows
    ]
    macro_point = None
    if point_estimates_allowed and cell_rows:
        macro_point = statistics.fmean(
            float(row["point_summary"]["mean"]) for row in cell_rows
        )
    paired: list[dict[str, Any]] = []
    for world in sorted({unit.world_seed for unit in units}):
        opaque = [
            values[unit.discard_id]
            for unit in units
            if unit.world_seed == world and unit.information_arm == "opaque_codes"
        ]
        nominal = [
            values[unit.discard_id]
            for unit in units
            if unit.world_seed == world
            and unit.information_arm == "anonymous_nominal_properties"
        ]
        if not opaque or not nominal:
            paired.append(
                {
                    "world_seed": world,
                    "defined": False,
                    "reason": "one arm has zero estimand denominator",
                    "nominal_minus_opaque": None,
                    "bounds": None,
                }
            )
            continue
        opaque_lower = statistics.fmean(item[0] for item in opaque)
        opaque_upper = statistics.fmean(item[1] for item in opaque)
        nominal_lower = statistics.fmean(item[0] for item in nominal)
        nominal_upper = statistics.fmean(item[1] for item in nominal)
        point = None
        if point_estimates_allowed:
            point = nominal_lower - opaque_lower
        paired.append(
            {
                "world_seed": world,
                "defined": True,
                "nominal_minus_opaque": point,
                "bounds": {
                    "lower": nominal_lower - opaque_upper,
                    "upper": nominal_upper - opaque_lower,
                },
            }
        )
    return {
        "finite_population_micro": strata,
        "cell_macro_average": {
            "defined_cell_count": len(cell_rows),
            "point_estimate": macro_point,
            "bounds": (
                None
                if not cell_mean_intervals
                else {
                    "lower": statistics.fmean(item[0] for item in cell_mean_intervals),
                    "upper": statistics.fmean(item[1] for item in cell_mean_intervals),
                }
            ),
            "replaces_micro_primary": False,
        },
        "paired_arm_contrasts": paired,
    }


def _continuous_values(
    units: Sequence[_Unit],
    transform: Callable[[_Unit, float], float],
    unresolved_interval: Callable[[_Unit], tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for unit in units:
        if unit.score is None:
            result[unit.discard_id] = unresolved_interval(unit)
        else:
            value = transform(unit, unit.score)
            result[unit.discard_id] = (value, value)
    return result


def _campaign_oracle(
    contract: Mapping[str, Any], units: Sequence[_Unit], *, all_resolved: bool
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    intervals: list[tuple[float, float]] = []
    for cell in _contract_cells(contract):
        cell_id = str(cell["cell_id"])
        campaign_best = float(cell["campaign_best_assayed_score"])
        selected = [unit for unit in units if unit.cell_id == cell_id]
        if not selected:
            rows.append(
                {
                    "cell_id": cell_id,
                    "opportunity": False,
                    "point_estimate": None,
                    "bounds": None,
                    "null_rule": "zero committed discards; excluded, never assigned zero",
                }
            )
            continue
        resolved_regret = [
            max(0.0, float(unit.score) - campaign_best)
            for unit in selected
            if unit.score is not None
        ]
        lower = max(resolved_regret, default=0.0)
        upper = max(
            [lower]
            + [1.0 - campaign_best for unit in selected if unit.score is None]
        )
        interval = (lower, upper)
        intervals.append(interval)
        rows.append(
            {
                "cell_id": cell_id,
                "opportunity": True,
                "point_estimate": lower if all_resolved else None,
                "bounds": {"lower": lower, "upper": upper},
            }
        )
    if len(intervals) != EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT:
        raise LatentTerminalAnalysisError("campaign oracle denominator is not nine cells")
    return {
        "denominator": EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT,
        "point_estimate_status": "available" if all_resolved else "withheld",
        "point_summary": _summary([item[0] for item in intervals]) if all_resolved else None,
        "bounds": _interval_summary(intervals),
        "cells": rows,
    }


def _decision_time_values(
    units: Sequence[_Unit],
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    values: dict[str, tuple[float, float]] = {}
    null_ids: list[str] = []
    for unit in units:
        if unit.prior_incumbent is None:
            null_ids.append(unit.discard_id)
            continue
        if unit.score is None:
            values[unit.discard_id] = (0.0, 1.0 - unit.prior_incumbent)
        else:
            regret = max(0.0, unit.score - unit.prior_incumbent)
            values[unit.discard_id] = (regret, regret)
    return values, null_ids


def _primary_selection(threshold_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in threshold_rows:
        if row.get("primary") is True:
            strata = row.get("strata")
            if not isinstance(strata, Mapping) or not isinstance(strata.get("overall"), Mapping):
                break
            return strata["overall"]
    raise LatentTerminalAnalysisError("primary selection table is missing")


def _estimands(
    contract: Mapping[str, Any],
    units: Sequence[_Unit],
    threshold_rows: Sequence[Mapping[str, Any]],
    *,
    all_resolved: bool,
) -> dict[str, Any]:
    latent = _continuous_values(units, lambda _unit, score: score, lambda _unit: (0.0, 1.0))
    delta = _continuous_values(
        units,
        lambda unit, score: score - unit.campaign_best,
        lambda unit: (-unit.campaign_best, 1.0 - unit.campaign_best),
    )
    regret = _continuous_values(
        units,
        lambda unit, score: max(0.0, score - unit.campaign_best),
        lambda unit: (0.0, 1.0 - unit.campaign_best),
    )
    primary = _primary_selection(threshold_rows)
    point_metrics = primary["point_metrics"]
    bounds = primary["bounds"]
    decision, null_ids = _decision_time_values(units)
    decision_complete = all(lower == upper for lower, upper in decision.values())
    return {
        "latent_terminal_score": {
            "denominator": EXPECTED_DISCARD_COUNT,
            "point_estimate_status": "available" if all_resolved else "withheld",
            "aggregation": _unit_strata_values(
                contract, latent, units, point_estimates_allowed=all_resolved
            ),
        },
        "discard_to_observed_best_delta": {
            "denominator": EXPECTED_DISCARD_COUNT,
            "point_estimate_status": "available" if all_resolved else "withheld",
            "aggregation": _unit_strata_values(
                contract, delta, units, point_estimates_allowed=all_resolved
            ),
        },
        "positive_discard_regret": {
            "denominator": EXPECTED_DISCARD_COUNT,
            "point_estimate_status": "available" if all_resolved else "withheld",
            "aggregation": _unit_strata_values(
                contract, regret, units, point_estimates_allowed=all_resolved
            ),
        },
        "campaign_oracle_regret": _campaign_oracle(
            contract, units, all_resolved=all_resolved
        ),
        "false_discard_fraction": {
            "denominator": EXPECTED_DISCARD_COUNT,
            "point_estimate_status": "available" if all_resolved else "withheld",
            "point_estimate": (
                None if point_metrics is None else point_metrics["false_discard_fraction"]
            ),
            "bounds": bounds["false_discard_fraction"],
        },
        "assay_commitment_precision": {
            "denominator": EXPECTED_ASSAY_COUNT,
            "point_estimate_status": "available_observed_exact",
            "point_estimate": _table_metrics(
                primary["all_unresolved_score_zero_table"]
            )["assay_commitment_precision"],
            "bounds": bounds["assay_commitment_precision"],
            "main_text_promotable_while_shadow_unresolved": False,
        },
        "assay_commitment_recall": {
            "denominator": "TP + FN among the frozen 60 lifecycles",
            "point_estimate_status": "available" if all_resolved else "withheld",
            "point_estimate": (
                None if point_metrics is None else point_metrics["assay_commitment_recall"]
            ),
            "bounds": bounds["assay_commitment_recall"],
        },
        "decision_time_discard_regret": {
            "denominator": len(decision),
            "null_count": len(null_ids),
            "null_discard_ids": null_ids,
            "point_estimate_status": "available" if decision_complete else "withheld",
            "aggregation": _unit_strata_values(
                contract,
                decision,
                [unit for unit in units if unit.discard_id in decision],
                point_estimates_allowed=decision_complete,
            ),
            "future_assay_imputed": False,
        },
    }


def _missingness(
    contract: Mapping[str, Any], units: Sequence[_Unit]
) -> dict[str, Any]:
    unresolved = [unit for unit in units if not unit.resolved]

    def count_by(
        field: str, groups: Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        if groups is None:
            groups = sorted({str(getattr(unit, field)) for unit in units})
        result: dict[str, dict[str, Any]] = {}
        for group in groups:
            denominator = sum(str(getattr(unit, field)) == group for unit in units)
            count = sum(str(getattr(unit, field)) == group for unit in unresolved)
            result[group] = {
                "unresolved_count": count,
                "fixed_denominator": denominator,
                "unresolved_fraction": finite_population_fraction(count, denominator),
            }
        return result

    return {
        "fixed_discard_denominator": EXPECTED_DISCARD_COUNT,
        "unresolved_count": len(unresolved),
        "unresolved_fraction": finite_population_fraction(
            len(unresolved), EXPECTED_DISCARD_COUNT
        ),
        "by_information_arm": count_by("information_arm"),
        "by_world_seed": count_by("world_seed"),
        "by_campaign_cell": count_by(
            "cell_id", [str(cell["cell_id"]) for cell in _contract_cells(contract)]
        ),
        "by_registered_reason": {
            category: sum(unit.unresolved_category == category for unit in unresolved)
            for category in sorted(UNRESOLVED_CATEGORIES)
        },
        "binding_error_count": sum(bool(unit.binding_errors) for unit in unresolved),
        "complete_case_primary_allowed": False,
        "complete_case_primary_used": False,
        "observed_only_rows_are_diagnostic_not_primary": True,
    }


def _unit_rows(units: Sequence[_Unit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        threshold = PRIMARY_RELATIVE_THRESHOLD * unit.campaign_best
        rows.append(
            {
                "discard_id": unit.discard_id,
                "cell_id": unit.cell_id,
                "world_seed": unit.world_seed,
                "information_arm": unit.information_arm,
                "lifecycle_index": unit.lifecycle_index,
                "terminal_step": unit.terminal_step,
                "campaign_best_assayed_score": unit.campaign_best,
                "primary_threshold": threshold,
                "prior_assayed_incumbent": unit.prior_incumbent,
                "decision_time_null": unit.prior_incumbent is None,
                "outcome_status": "resolved" if unit.resolved else "unresolved",
                "score": unit.score,
                "primary_classification": (
                    None
                    if unit.score is None
                    else ("FN" if unit.score >= threshold else "TN")
                ),
                "unresolved_category": unit.unresolved_category,
                "unresolved_reason": unit.unresolved_reason,
                "binding_errors": list(unit.binding_errors),
            }
        )
    return rows


def _entry_gate(
    mode: AnalysisMode,
    all_resolved: bool,
    execution_gates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required = {
        "exact_prefix_reconstruction_count": EXPECTED_DISCARD_COUNT,
        "valid_shadow_score_count": EXPECTED_DISCARD_COUNT,
        "exact_same_identity_replay_count": EXPECTED_DISCARD_COUNT,
        "agent_provider_calls": 0,
        "original_trajectory_mutated": False,
        "original_resource_ledger_mutated": False,
    }
    evaluated = mode == "formal_shadow_analysis"
    supplied = dict(execution_gates or {})
    gates_match = evaluated and all(
        supplied.get(field) == value for field, value in required.items()
    )
    return {
        "formal_gate_evaluated": evaluated,
        "required": required,
        "supplied": supplied if evaluated else {},
        "all_36_scores_resolved": all_resolved,
        "main_text_eligible": bool(gates_match and all_resolved),
        "result_direction_gate": False,
        "significance_gate": False,
        "arm_difference_gate": False,
        "threshold_selected_after_outcomes": False,
        "synthetic_qualification_never_main_text": mode == "synthetic_qualification",
    }


def latent_terminal_analysis_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an analysis while excluding its embedded digest."""

    candidate = dict(payload)
    candidate.pop("analysis_sha256", None)
    return canonical_json_sha256(candidate)


def analyze_latent_terminal_population(
    contract: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    mode: AnalysisMode,
    execution_gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze all frozen terminal decisions with registered fail-closed rules."""

    if mode not in ALLOWED_MODES:
        raise LatentTerminalAnalysisError(f"unsupported analysis mode: {mode}")
    _validate_contract_binding(contract)
    units = _parse_receipts(contract, receipts, mode=mode)
    all_resolved = all(unit.resolved for unit in units)
    threshold_rows = _selection_analysis(contract, units, all_resolved=all_resolved)
    estimands = _estimands(
        contract,
        units,
        threshold_rows,
        all_resolved=all_resolved,
    )
    result: dict[str, Any] = {
        "schema_id": ANALYSIS_SCHEMA_ID,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_mode": mode,
        "status": "complete" if all_resolved else "incomplete_full_report_required",
        "evidence_bindings": {
            "latent_terminal_contract_sha256": contract["contract_sha256"],
            "population_manifest_sha256": contract["population"][
                "population_manifest_sha256"
            ],
        },
        "census": {
            "campaign_cells": EXPECTED_CELL_COUNT,
            "closed_lifecycles": EXPECTED_LIFECYCLE_COUNT,
            "observed_assays": EXPECTED_ASSAY_COUNT,
            "observed_discards": EXPECTED_DISCARD_COUNT,
            "resolved_shadow_receipts": sum(unit.resolved for unit in units),
            "unresolved_shadow_receipts": sum(not unit.resolved for unit in units),
        },
        "estimands": estimands,
        "selection_and_threshold_sensitivity": threshold_rows,
        "missingness_and_censoring": _missingness(contract, units),
        "unit_rows": _unit_rows(units),
        "entry_gate": _entry_gate(mode, all_resolved, execution_gates),
        "scientific_boundary": {
            "finite_population_primary": True,
            "complete_case_primary_used": False,
            "shadow_evaluations_executed_by_analyzer": 0,
            "agent_provider_calls_by_analyzer": 0,
            "formal_shadow_outcomes_accessed": mode == "formal_shadow_analysis",
            "shadow_branch_is_agent_choice": False,
            "counts_as_original_agent_experiment": False,
        },
    }
    result["analysis_sha256"] = latent_terminal_analysis_sha256(result)
    return result


def validate_latent_terminal_analysis(payload: Mapping[str, Any]) -> list[str]:
    """Return deterministic structural errors for an analysis report."""

    errors: list[str] = []
    if payload.get("schema_id") != ANALYSIS_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("analysis_sha256") != latent_terminal_analysis_sha256(payload):
        errors.append("analysis self-hash mismatch")
    bindings = payload.get("evidence_bindings")
    if not isinstance(bindings, Mapping) or bindings.get(
        "latent_terminal_contract_sha256"
    ) != FROZEN_CONTRACT_SHA256:
        errors.append("frozen contract binding mismatch")
    census = payload.get("census")
    if not isinstance(census, Mapping):
        errors.append("census missing")
    else:
        resolved = census.get("resolved_shadow_receipts")
        unresolved = census.get("unresolved_shadow_receipts")
        valid_receipt_counts = (
            isinstance(resolved, int)
            and not isinstance(resolved, bool)
            and isinstance(unresolved, int)
            and not isinstance(unresolved, bool)
            and resolved + unresolved == EXPECTED_DISCARD_COUNT
        )
        if (
            census.get("observed_assays") != EXPECTED_ASSAY_COUNT
            or census.get("observed_discards") != EXPECTED_DISCARD_COUNT
            or not valid_receipt_counts
        ):
            errors.append("fixed terminal census mismatch")
    estimands = payload.get("estimands")
    expected_estimands = {
        "latent_terminal_score",
        "discard_to_observed_best_delta",
        "positive_discard_regret",
        "campaign_oracle_regret",
        "false_discard_fraction",
        "assay_commitment_precision",
        "assay_commitment_recall",
        "decision_time_discard_regret",
    }
    if not isinstance(estimands, Mapping) or set(estimands) != expected_estimands:
        errors.append("eight frozen estimands are not present")
    missingness = payload.get("missingness_and_censoring")
    if not isinstance(missingness, Mapping) or (
        missingness.get("complete_case_primary_used") is not False
    ):
        errors.append("complete-case primary prohibition missing")
    rows = payload.get("unit_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_DISCARD_COUNT:
        errors.append("36-row audit is incomplete")
    threshold_rows = payload.get("selection_and_threshold_sensitivity")
    if not isinstance(threshold_rows, list) or len(threshold_rows) != 4:
        errors.append("registered threshold sensitivity surface is incomplete")
    return errors


__all__ = [
    "ALLOWED_MODES",
    "ANALYSIS_SCHEMA_ID",
    "ANALYSIS_SCHEMA_VERSION",
    "FROZEN_CONTRACT_SHA256",
    "UNRESOLVED_CATEGORIES",
    "LatentTerminalAnalysisError",
    "analyze_latent_terminal_population",
    "finite_population_fraction",
    "latent_terminal_analysis_sha256",
    "validate_latent_terminal_analysis",
]
