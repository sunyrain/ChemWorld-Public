"""Participant-visible identifiability audit for the completed Work II A-S B2 study."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.world.phase_kernel import (
    PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION,
    nominal_partition_pair_tables,
)

B2_IDENTIFIABILITY_AUDIT_VERSION = (
    "chemworld-work-ii-b2-participant-visible-identifiability-audit-0.1"
)
_METRICS = ("product_in_organic", "product_in_aqueous", "phase_ratio")


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _query_pair(query: Mapping[str, Any]) -> tuple[int, int]:
    features = query.get("feature_values")
    if not isinstance(features, Mapping):
        raise ValueError("B2 query lacks feature_values")
    solvent = features.get("solvent")
    extractant = features.get("extractant")
    if (
        isinstance(solvent, bool)
        or not isinstance(solvent, int)
        or isinstance(extractant, bool)
        or not isinstance(extractant, int)
    ):
        raise ValueError("B2 nominal pair must be integer-valued")
    return solvent, extractant


def _constant_baseline(
    evidence: Sequence[Mapping[str, Any]],
    scoring_truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    means = {
        metric: statistics.mean(
            _finite(row["observations"][metric], field=f"evidence.{metric}")
            for row in evidence
        )
        for metric in _METRICS
    }
    terms = [
        abs(means[metric] - _finite(values[metric], field=f"truth.{query_id}.{metric}"))
        for query_id, values in scoring_truth.items()
        for metric in _METRICS
    ]
    return {
        "metric_means": means,
        "term_count": len(terms),
        "mean_normalized_absolute_error": statistics.mean(terms),
    }


def audit_b2_participant_visible_identifiability(
    manifest: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
    b2_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise ValueError("B2 manifest must contain 15 cells")
    completed = [row for row in cell_results if row.get("status") == "completed"]
    if len(completed) != 15:
        raise ValueError("B2 identifiability audit requires 15 completed cells")

    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        cluster_id = str(cell["cluster_id"])
        by_cluster.setdefault(cluster_id, []).append(cell)
    if len(by_cluster) != 5:
        raise ValueError("B2 identifiability audit requires five clusters")

    all_pairs: set[tuple[int, int]] = set()
    world_rows: list[dict[str, Any]] = []
    completed_by_id = {str(row["cell_id"]): row for row in completed}
    for cluster_id, cluster_cells in sorted(by_cluster.items()):
        reference = cluster_cells[0]
        packet = reference.get("public_packet")
        scoring_truth = reference.get("scoring_truth")
        if not isinstance(packet, Mapping) or not isinstance(scoring_truth, Mapping):
            raise ValueError(f"{cluster_id} lacks packet or scoring truth")
        evidence = packet.get("evidence")
        scoring = packet.get("scoring_queries")
        if not isinstance(evidence, list) or not isinstance(scoring, list):
            raise ValueError(f"{cluster_id} has malformed evidence/scoring rosters")
        pairs = {_query_pair(row) for row in [*evidence, *scoring]}
        all_pairs.update(pairs)
        post_errors = {
            str(cell["arm"]): _finite(
                completed_by_id[str(cell["cell_id"])]["scores"]["post"][
                    "mean_normalized_absolute_error"
                ],
                field=f"{cell['cell_id']}.post_error",
            )
            for cell in cluster_cells
        }
        world_rows.append(
            {
                "cluster_id": cluster_id,
                "world_seed": int(reference["world_seed"]),
                "nominal_pairs": [list(pair) for pair in sorted(pairs)],
                "constant_endpoint_baseline": _constant_baseline(evidence, scoring_truth),
                "post_error_by_arm": post_errors,
            }
        )

    if len(all_pairs) != 1:
        raise ValueError("completed B2 packet unexpectedly varies its nominal pair")
    solvent, extractant = next(iter(all_pairs))
    public_tables = nominal_partition_pair_tables()
    coefficient_table = public_tables["product_distribution_coefficients"]
    base = (
        PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION
        * float(coefficient_table[solvent][extractant])
    )
    exponent = 1.75
    alias_multiplier = base ** (exponent - 1.0)

    public_audit = b2_analysis.get("public_summary_audit")
    if not isinstance(public_audit, Mapping):
        raise ValueError("B2 analysis lacks public_summary_audit")
    by_arm = public_audit.get("by_arm")
    if not isinstance(by_arm, Mapping):
        raise ValueError("B2 public summary audit lacks arm rows")
    aligned = by_arm.get("aligned_nominal")
    if not isinstance(aligned, Mapping):
        raise ValueError("B2 public summary audit lacks aligned arm")

    constant_errors = [
        float(row["constant_endpoint_baseline"]["mean_normalized_absolute_error"])
        for row in world_rows
    ]
    return {
        "schema_version": B2_IDENTIFIABILITY_AUDIT_VERSION,
        "study_id": manifest.get("study_id"),
        "world_count": len(world_rows),
        "participant_session_count": len(completed),
        "provider_call_count": 0,
        "new_participant_session_count": 0,
        "participant_visible_design": {
            "nominal_pair_count": len(all_pairs),
            "nominal_pairs": [list(pair) for pair in sorted(all_pairs)],
            "reference_partition_coefficient_supplied": any(
                "reference_partition" in str(key)
                for key in cells[0]["public_packet"]
            ),
            "free_text_structural_fields": ["model_summary", "evidence_assessment"],
            "typed_family_or_exponent_field": False,
        },
        "exact_alias": {
            "present": True,
            "reason": (
                "All evidence and scoring queries hold the nominal solvent/extractant pair fixed. "
                "The executable kernel uses D_base**alpha multiplied by the same process factors, "
                "so alpha=1 with a free multiplier D_base**(1.75-1) is observationally identical "
                "to alpha=1.75 on this participant-visible surface."
            ),
            "solvent": solvent,
            "extractant": extractant,
            "effective_reference_partition_coefficient": base,
            "registered_power_exponent": exponent,
            "linear_alias_coefficient_multiplier": alias_multiplier,
        },
        "positive_control": {
            "aligned_exact_1_75_count": int(
                aligned["exact_1_75_power_law_recovery_count"]
            ),
            "aligned_power_compatible_count": int(
                aligned["power_compatible_language_count"]
            ),
            "world_count": int(aligned["world_count"]),
            "readout_positive_control_passed": (
                int(aligned["exact_1_75_power_law_recovery_count"])
                == int(aligned["world_count"])
            ),
        },
        "empirical_alternative": {
            "model": "evidence_metric_mean_constant_endpoint",
            "mean_scoring_error": statistics.mean(constant_errors),
            "minimum_scoring_error": min(constant_errors),
            "maximum_scoring_error": max(constant_errors),
            "world_rows": world_rows,
        },
        "decision": {
            "structural_family_identification_supported": False,
            "prediction_updating_result_retained": True,
            "allowed_wording": (
                "The fixed B2 packet drove strong numerical revision, but exact 1.75-law "
                "expression was not recovered in the misindexed public summaries."
            ),
            "forbidden_wording": (
                "B2 uniquely localized the remaining failure to internal structural-law "
                "identification."
            ),
            "follow_up_required_for_structural_identification": (
                "Vary or publicly anchor the base partition coefficient, require typed family and "
                "exponent commitments, and score family-diagnostic unseen queries."
            ),
        },
    }


__all__ = [
    "B2_IDENTIFIABILITY_AUDIT_VERSION",
    "audit_b2_participant_visible_identifiability",
]

