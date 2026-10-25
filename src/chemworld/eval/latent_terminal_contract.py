"""Outcome-blind contract for the Work I discarded-state terminal audit.

The contract enumerates the already observed DeepSeek G2 terminal population
and freezes every counterfactual, estimand, denominator, sensitivity, and
reporting rule before any discarded state is evaluated.  It reads public
terminal trajectories only; hidden pre-discard states and shadow scores remain
outside this task.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.data.records import trajectory_value
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.tasks import get_task

CONTRACT_SCHEMA_ID = "chemworld.latent_terminal_estimand_contract"
CONTRACT_SCHEMA_VERSION = "0.1.0"
CONTRACT_ID = "work-i-deepseek-discarded-state-latent-terminal-v0.1"

EXPECTED_CELL_COUNT = 10
EXPECTED_LIFECYCLE_COUNT = 60
EXPECTED_ASSAY_COUNT = 24
EXPECTED_DISCARD_COUNT = 36
EXPECTED_OPERATION_COUNT = 889
EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT = 9
EXPECTED_ARM_COUNTS = {
    "opaque_codes": {"cells": 5, "assays": 8, "discards": 22},
    "anonymous_nominal_properties": {
        "cells": 5,
        "assays": 16,
        "discards": 14,
    },
}

PRIMARY_RELATIVE_THRESHOLD = 0.90
RELATIVE_THRESHOLD_SENSITIVITY = (0.80, 0.90, 1.00)
REGISTERED_TASK_THRESHOLD = 0.58

FROZEN_CAMPAIGN_AUDIT_SHA256 = (
    "74b08ec6cf318f8fa7739ba133fa3f09d69964d40b3a6279cd82e40b91ba5d6a"
)
FROZEN_MATRIX_MANIFEST_SHA256 = (
    "0b0ebae45e7f269a3e1ab268d06a90c996347b98071a0fb19240d47fd00bfa1d"
)
FROZEN_PUBLIC_ARCHIVE_SHA256 = (
    "3362ea0a2f6349e6528fde3e2ac23f4de3580ae4d8ce750163dc4e181498a3f6"
)
FROZEN_COMPARISON_SHA256 = (
    "5d534615aa0eb070b1a8ddf7cf123c2548bc8e4c948a98ffe2eafb0b545ef93e"
)
FROZEN_TERMINAL_INDEX_SHA256 = (
    "6c4c9a933e1a3cc0c6ead749892bf90b0abf2e3fc33fb796497d7bd3a99f82b3"
)

TERMINAL_INDEX_PATH = Path(
    "benchmark/releases/chemworld-serious-v1/"
    "g2-deepseek-v0.6-terminal-file-index.json"
)
PUBLIC_ARCHIVE_ROOT = Path(
    "benchmark/releases/chemworld-serious-v1/"
    "g2-deepseek-v0.6-public-trajectory-archive"
)
PUBLIC_ARCHIVE_MANIFEST_PATH = PUBLIC_ARCHIVE_ROOT / "manifest.json"
COMPARISON_PATH = Path(
    "workstreams/arxiv_v1/reports/g2-agent-system-comparison-v0.1.json"
)
EXPERIMENT_LEDGER_PATH = Path(
    "workstreams/arxiv_v1/reports/"
    "experimental-intelligence-experiment-ledger-v0.1.json"
)
CONTRACT_IMPLEMENTATION_PATH = Path("src/chemworld/eval/latent_terminal_contract.py")

SOURCE_PATHS = (
    TERMINAL_INDEX_PATH,
    PUBLIC_ARCHIVE_MANIFEST_PATH,
    COMPARISON_PATH,
    EXPERIMENT_LEDGER_PATH,
    Path("src/chemworld/tasks.py"),
    Path("src/chemworld/world/scoring.py"),
    CONTRACT_IMPLEMENTATION_PATH,
    Path("scripts/freeze_work_i_latent_terminal_contract.py"),
)


class LatentTerminalContractError(ValueError):
    """Raised when frozen source evidence or the estimand contract is invalid."""


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentTerminalContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise LatentTerminalContractError(f"{label} must be a JSON object")
    return payload


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop(field, None)
    return result


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, expected: str, label: str
) -> None:
    supplied = payload.get(field)
    rebuilt = canonical_json_sha256(_without(payload, field))
    if supplied != rebuilt or supplied != expected:
        raise LatentTerminalContractError(f"{label} self-hash is stale")


def _finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LatentTerminalContractError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise LatentTerminalContractError(f"{label} must be finite in [0, 1]")
    return score


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise LatentTerminalContractError(
                        f"{path}:{line_number} must be a JSON object"
                    )
                records.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentTerminalContractError(f"cannot read public trajectory: {path}") from exc
    return records


def _terminal_kind(record: Mapping[str, Any]) -> str | None:
    action = record.get("action")
    if not isinstance(action, Mapping):
        return None
    if action.get("operation") == "discard_batch":
        return "discard"
    if (
        action.get("operation") == "measure"
        and action.get("instrument") == "final_assay"
    ):
        return "assay"
    return None


def _raw_trajectory_index(index: Mapping[str, Any]) -> dict[str, str]:
    files = index.get("files")
    if not isinstance(files, list):
        raise LatentTerminalContractError("terminal file index lacks files")
    result: dict[str, str] = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            raise LatentTerminalContractError("terminal file index entry is invalid")
        path = str(raw.get("path", ""))
        if path.endswith("/trajectory.jsonl"):
            digest = str(raw.get("sha256", ""))
            result[path] = digest
    if len(result) != EXPECTED_CELL_COUNT:
        raise LatentTerminalContractError("terminal index lacks ten raw trajectories")
    return result


def _source_manifest(root: Path) -> dict[str, str]:
    resolved = root.resolve()
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = (resolved / relative).resolve()
        if not path.is_relative_to(resolved) or not path.is_file():
            raise LatentTerminalContractError(f"missing source artifact: {relative}")
        result[relative.as_posix()] = file_sha256(path)
    return result


def _population_manifest(root: Path) -> dict[str, Any]:
    index = _read_json_object(root / TERMINAL_INDEX_PATH, label="terminal file index")
    archive = _read_json_object(
        root / PUBLIC_ARCHIVE_MANIFEST_PATH,
        label="public trajectory archive manifest",
    )
    comparison = _read_json_object(
        root / COMPARISON_PATH,
        label="complete-system comparison",
    )
    _verify_self_hash(
        index,
        field="index_sha256",
        expected=FROZEN_TERMINAL_INDEX_SHA256,
        label="terminal file index",
    )
    _verify_self_hash(
        archive,
        field="archive_sha256",
        expected=FROZEN_PUBLIC_ARCHIVE_SHA256,
        label="public trajectory archive",
    )
    _verify_self_hash(
        comparison,
        field="comparison_sha256",
        expected=FROZEN_COMPARISON_SHA256,
        label="complete-system comparison",
    )
    for payload, label in (
        (index, "terminal file index"),
        (archive, "public trajectory archive"),
    ):
        if payload.get("campaign_audit_sha256") != FROZEN_CAMPAIGN_AUDIT_SHA256:
            raise LatentTerminalContractError(f"{label} campaign audit binding is stale")
        if payload.get("matrix_manifest_sha256") != FROZEN_MATRIX_MANIFEST_SHA256:
            raise LatentTerminalContractError(f"{label} matrix binding is stale")
    deepseek = comparison.get("systems", {}).get("deepseek_v4_flash_direct", {})
    if not isinstance(deepseek, Mapping):
        raise LatentTerminalContractError("comparison lacks the DeepSeek system")
    expected_summary = {
        "cell_count": EXPECTED_CELL_COUNT,
        "closed_batch_count": EXPECTED_LIFECYCLE_COUNT,
        "final_assay_count": EXPECTED_ASSAY_COUNT,
        "discarded_batch_count": EXPECTED_DISCARD_COUNT,
        "operation_attempt_count": EXPECTED_OPERATION_COUNT,
        "source_audit_sha256": FROZEN_CAMPAIGN_AUDIT_SHA256,
    }
    if any(deepseek.get(key) != value for key, value in expected_summary.items()):
        raise LatentTerminalContractError("comparison population summary is stale")

    raw_index = _raw_trajectory_index(index)
    raw_cells = archive.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != EXPECTED_CELL_COUNT:
        raise LatentTerminalContractError("public archive must contain ten cells")
    cells: list[dict[str, Any]] = []
    run_identity: dict[str, Any] | None = None
    for expected_number, raw_cell in enumerate(raw_cells, start=1):
        if not isinstance(raw_cell, Mapping):
            raise LatentTerminalContractError("public archive cell is invalid")
        cell_id = f"cell-{expected_number:02d}"
        if raw_cell.get("cell_id") != cell_id:
            raise LatentTerminalContractError("public archive cell order is not frozen")
        compact_relative = PUBLIC_ARCHIVE_ROOT / str(raw_cell.get("compact_path"))
        compact_path = root / compact_relative
        if file_sha256(compact_path) != raw_cell.get("compact_sha256"):
            raise LatentTerminalContractError(f"{cell_id} compact trajectory hash mismatch")
        records = _read_jsonl(compact_path)
        if len(records) != raw_cell.get("record_count"):
            raise LatentTerminalContractError(f"{cell_id} record count mismatch")
        if raw_cell.get("closed_batch_count") != 6:
            raise LatentTerminalContractError(f"{cell_id} must close six lifecycles")
        exact = raw_cell.get("exact_replay")
        if not isinstance(exact, Mapping) or exact.get("verified") is not True:
            raise LatentTerminalContractError(f"{cell_id} exact replay is not verified")
        raw_path = f"{cell_id}/attempt-01/trajectory.jsonl"
        if raw_index.get(raw_path) != raw_cell.get("source_trajectory_sha256"):
            raise LatentTerminalContractError(f"{cell_id} raw trajectory binding mismatch")

        terminals: list[tuple[int, Mapping[str, Any], str]] = []
        for record_index, record in enumerate(records):
            step = record.get("step")
            if step != record_index + 1:
                raise LatentTerminalContractError(f"{cell_id} step ordinals are not contiguous")
            kind = _terminal_kind(record)
            if kind is not None:
                if record.get("transaction_status") != "committed":
                    raise LatentTerminalContractError(
                        f"{cell_id} contains a non-committed terminal"
                    )
                terminals.append((record_index, record, kind))
        if len(terminals) != 6:
            raise LatentTerminalContractError(f"{cell_id} must contain six terminals")
        lifecycle_indices = [int(item[1].get("experiment_index", -1)) for item in terminals]
        if lifecycle_indices != list(range(6)):
            raise LatentTerminalContractError(
                f"{cell_id} terminal lifecycle indices are not 0..5"
            )

        assays: list[dict[str, Any]] = []
        discards: list[dict[str, Any]] = []
        terminal_sequence: list[str] = []
        for record_index, terminal_record, kind in terminals:
            lifecycle_index = int(terminal_record["experiment_index"])
            terminal_sequence.append(kind)
            if kind == "assay":
                assays.append(
                    {
                        "lifecycle_index": lifecycle_index,
                        "terminal_step": int(terminal_record["step"]),
                        "score": _finite_score(
                            trajectory_value(terminal_record, 'leaderboard_score', None),
                            label=f"{cell_id} assay score",
                        ),
                    }
                )
            else:
                action = terminal_record["action"]
                discards.append(
                    {
                        "discard_id": (
                            f"{cell_id}:lifecycle-{lifecycle_index:02d}:"
                            f"terminal-step-{int(terminal_record['step']):03d}"
                        ),
                        "lifecycle_index": lifecycle_index,
                        "terminal_step": int(terminal_record["step"]),
                        "terminal_action_sha256": canonical_json_sha256(action),
                        "public_prefix_sha256": canonical_json_sha256(
                            records[:record_index]
                        ),
                        "shadow_outcome_status_at_freeze": "unobserved",
                    }
                )
        if not assays:
            raise LatentTerminalContractError(
                f"{cell_id} lacks an observed assay benchmark"
            )
        first = records[0]
        observed_identity = {
            "task_id": first.get("task_id"),
            "task_contract_hash": first.get("task_contract_hash"),
            "scoring_contract_id": first.get("scoring_contract_id"),
            "scoring_contract_hash": first.get("scoring_contract_hash"),
            "observation_contract_hash": first.get("observation_contract_hash"),
            "world_family_version": first.get("world_family_version"),
            "material_family_id": first.get("electrochemical_material_family_id"),
            "workflow_mode": first.get("electrochemical_workflow_mode"),
            "observation_noise_mode": first.get("observation_noise_mode"),
            "observation_noise_namespace": first.get("observation_noise_namespace"),
            "campaign_resource_card_sha256": first.get("campaign_resource_card", {}).get(
                "card_sha256"
            ),
        }
        if run_identity is None:
            run_identity = observed_identity
        elif run_identity != observed_identity:
            raise LatentTerminalContractError("public cells do not share run contracts")
        cells.append(
            {
                "cell_id": cell_id,
                "world_seed": int(raw_cell["world_seed"]),
                "information_arm": str(raw_cell["condition_id"]),
                "compact_path": compact_relative.as_posix(),
                "compact_sha256": str(raw_cell["compact_sha256"]),
                "source_trajectory_sha256": str(
                    raw_cell["source_trajectory_sha256"]
                ),
                "record_count": len(records),
                "terminal_sequence": terminal_sequence,
                "observed_assays": assays,
                "observed_assay_count": len(assays),
                "observed_discard_count": len(discards),
                "campaign_best_assayed_score": max(item["score"] for item in assays),
                "discard_units": discards,
            }
        )

    assay_count = sum(int(cell["observed_assay_count"]) for cell in cells)
    discard_count = sum(int(cell["observed_discard_count"]) for cell in cells)
    if (assay_count, discard_count) != (
        EXPECTED_ASSAY_COUNT,
        EXPECTED_DISCARD_COUNT,
    ):
        raise LatentTerminalContractError("terminal population is not 24 assays + 36 discards")
    for arm, expected in EXPECTED_ARM_COUNTS.items():
        arm_cells = [cell for cell in cells if cell["information_arm"] == arm]
        observed = {
            "cells": len(arm_cells),
            "assays": sum(int(cell["observed_assay_count"]) for cell in arm_cells),
            "discards": sum(int(cell["observed_discard_count"]) for cell in arm_cells),
        }
        if observed != expected:
            raise LatentTerminalContractError(f"{arm} terminal counts are stale")

    task = get_task("electrochemical-conversion")
    if task.threshold != REGISTERED_TASK_THRESHOLD:
        raise LatentTerminalContractError("registered task threshold is stale")
    if run_identity is None or run_identity["task_contract_hash"] != task.contract_hash:
        raise LatentTerminalContractError("run/task contract binding is stale")
    return {
        "population_id": "deepseek-v4-flash-g2-v0.6-terminal-census",
        "selection_rule": (
            "Every committed discard_batch terminal in the frozen ten-cell DeepSeek "
            "G2 v0.6 matrix; no sampling, score filtering, or outcome-dependent exclusion."
        ),
        "system_id": "deepseek_v4_flash_direct",
        "world_seeds": [0, 1, 2, 3, 4],
        "information_arms": [
            "opaque_codes",
            "anonymous_nominal_properties",
        ],
        "counts": {
            "cells": EXPECTED_CELL_COUNT,
            "closed_lifecycles": EXPECTED_LIFECYCLE_COUNT,
            "observed_assays": EXPECTED_ASSAY_COUNT,
            "observed_discards": EXPECTED_DISCARD_COUNT,
            "accepted_primitive_operations": EXPECTED_OPERATION_COUNT,
            "shadow_evaluations_planned": EXPECTED_DISCARD_COUNT,
            "agent_provider_calls_planned": 0,
        },
        "arm_counts": deepcopy(EXPECTED_ARM_COUNTS),
        "run_contracts": run_identity,
        "cells": cells,
        "population_manifest_sha256": canonical_json_sha256(cells),
        "latent_outcomes_accessed": False,
        "hidden_states_accessed": False,
    }


def _estimands() -> list[dict[str, Any]]:
    return [
        {
            "estimand_id": "latent_terminal_score",
            "role": "primary_continuous",
            "unit": "discarded lifecycle",
            "definition": (
                "The evaluator-bound final-assay leaderboard score of the exact "
                "pre-discard hidden state under the frozen counterfactual terminal rule."
            ),
            "formula": "S_i",
            "denominator": "all 36 valid shadow evaluations",
            "range": [0.0, 1.0],
        },
        {
            "estimand_id": "discard_to_observed_best_delta",
            "role": "primary_continuous",
            "unit": "discarded lifecycle",
            "definition": (
                "Signed latent improvement over the best score that the same campaign "
                "actually committed to assay."
            ),
            "formula": "Delta_i = S_i - B_c",
            "denominator": "all 36 valid shadow evaluations",
            "range": [-1.0, 1.0],
        },
        {
            "estimand_id": "positive_discard_regret",
            "role": "primary_continuous",
            "unit": "discarded lifecycle",
            "definition": "Missed improvement above the campaign's observed assayed best.",
            "formula": "R_i = max(0, S_i - B_c)",
            "denominator": "all 36 valid shadow evaluations",
            "range": [0.0, 1.0],
        },
        {
            "estimand_id": "campaign_oracle_regret",
            "role": "primary_campaign",
            "unit": "campaign cell",
            "definition": (
                "Improvement available from the best discarded state over the campaign's "
                "best actually assayed state, among cells where the agent made at least "
                "one discard decision."
            ),
            "formula": "R_c = max(0, max_{i in discarded(c)} S_i - B_c)",
            "denominator": (
                "the 9 frozen campaign cells with at least one committed discard"
            ),
            "null_rule": (
                "A campaign with zero committed discards is retained in the 10-cell "
                "census with null campaign_oracle_regret and is excluded from this "
                "estimand denominator by the pre-outcome opportunity rule. It is never "
                "assigned zero regret."
            ),
            "range": [0.0, 1.0],
        },
        {
            "estimand_id": "false_discard_fraction",
            "role": "primary_classification",
            "unit": "discarded lifecycle",
            "definition": (
                "Fraction of discard decisions whose latent score reaches the frozen "
                "near-best threshold q_c = 0.90 B_c."
            ),
            "formula": "FN / (FN + TN)",
            "denominator": "all 36 discard decisions",
            "range": [0.0, 1.0],
        },
        {
            "estimand_id": "assay_commitment_precision",
            "role": "primary_classification",
            "unit": "assayed lifecycle",
            "definition": (
                "Fraction of actual assay commitments whose observed score reaches the "
                "same frozen campaign-relative near-best threshold."
            ),
            "formula": "TP / (TP + FP)",
            "denominator": "all 24 observed assay decisions",
            "range": [0.0, 1.0],
        },
        {
            "estimand_id": "assay_commitment_recall",
            "role": "secondary_classification",
            "unit": "high-value lifecycle",
            "definition": (
                "Fraction of all observed-or-latent near-best lifecycles that the agent "
                "committed to assay."
            ),
            "formula": "TP / (TP + FN)",
            "denominator": "all near-best lifecycles among the frozen 60",
            "range": [0.0, 1.0],
        },
        {
            "estimand_id": "decision_time_discard_regret",
            "role": "secondary_temporal",
            "unit": "discarded lifecycle with a prior assayed incumbent",
            "definition": (
                "Latent improvement over the best assay observed strictly before the "
                "discard decision."
            ),
            "formula": "max(0, S_i - I_i^-)",
            "denominator": (
                "discard decisions with at least one earlier assay in the same campaign"
            ),
            "null_rule": "null when no prior assay exists; never impute a future assay",
            "range": [0.0, 1.0],
        },
    ]


def _aggregation() -> dict[str, Any]:
    return {
        "finite_population_primary": True,
        "primary_overall_units": {
            "discarded_lifecycles": EXPECTED_DISCARD_COUNT,
            "all_lifecycles_for_classification": EXPECTED_LIFECYCLE_COUNT,
            "campaign_cells": EXPECTED_CELL_COUNT,
            "campaign_cells_with_discard_opportunity": (
                EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT
            ),
        },
        "campaign_oracle_opportunity_rule": {
            "defined_when": "observed_discard_count >= 1",
            "defined_cell_count": EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT,
            "no_opportunity_cell_ids": ["cell-02"],
            "no_opportunity_value": None,
            "exclude_no_opportunity_from_denominator": True,
            "freeze_timing": "before latent outcomes",
        },
        "continuous_summary": [
            "count",
            "mean",
            "standard_deviation",
            "minimum",
            "25th_percentile_linear",
            "median_linear",
            "75th_percentile_linear",
            "maximum",
            "empirical_cdf",
        ],
        "required_strata": [
            "overall",
            "information_arm",
            "world_seed",
            "campaign_cell",
        ],
        "micro_average": (
            "Lifecycle-level overall and arm fractions use their exact lifecycle "
            "denominators."
        ),
        "cell_macro_average": (
            "Report separately across cells with defined denominators; do not replace "
            "the finite-population micro estimate."
        ),
        "paired_arm_contrast": (
            "Descriptive nominal-minus-opaque world-paired contrasts are secondary "
            "and defined only where both arm-specific denominators exist."
        ),
        "uncertainty": (
            "Counts and fractions describe the complete frozen population; no "
            "super-population p-values or confidence intervals are primary."
        ),
    }


def _missingness_and_failure() -> dict[str, Any]:
    return {
        "complete_case_primary_allowed": False,
        "all_36_required_for_primary_point_estimates": True,
        "nonfinite_score_policy": "invalid shadow evaluation; never clamp or impute",
        "prefix_mismatch_policy": "fail closed and retain the mismatch receipt",
        "resource_or_precondition_failure_policy": (
            "retain as an unresolved shadow evaluation; do not change the state or "
            "counterfactual semantics to obtain a score"
        ),
        "retry_policy": (
            "Only an exact replay under the same source, identity, and noise bindings "
            "is allowed; retries never replace a valid first result."
        ),
        "zero_denominator_policy": "return null with the exact denominator disclosed",
        "unresolved_unit_policy": (
            "Retain every unresolved unit in its frozen denominator and publish its "
            "failure reason; observed-only estimates are descriptive diagnostics and "
            "never substitute for the registered finite-population point estimate."
        ),
        "point_estimate_policy_by_estimand": {
            "latent_terminal_score": "withhold if any of 36 shadow scores is unresolved",
            "discard_to_observed_best_delta": (
                "withhold if any of 36 shadow scores is unresolved"
            ),
            "positive_discard_regret": (
                "withhold if any of 36 shadow scores is unresolved"
            ),
            "campaign_oracle_regret": (
                "withhold if any discard in any of the 9 opportunity cells is unresolved"
            ),
            "false_discard_fraction": (
                "withhold if any of 36 shadow classifications is unresolved"
            ),
            "assay_commitment_precision": (
                "compute exactly from the 24 observed assays, but do not promote the "
                "terminal-selection result to main text while the 36-score gate fails"
            ),
            "assay_commitment_recall": (
                "withhold if any of 36 shadow classifications is unresolved"
            ),
            "decision_time_discard_regret": (
                "withhold if any eligible shadow score is unresolved"
            ),
        },
        "unresolved_bounds": {
            "latent_terminal_score": (
                "assign each unresolved S_i its sharp support [0,1]; report mean and "
                "order-statistic bounds on the fixed 36-unit denominator"
            ),
            "discard_to_observed_best_delta": (
                "for unresolved unit i in cell c use [-B_c, 1-B_c], then report sharp "
                "fixed-denominator aggregate bounds"
            ),
            "positive_discard_regret": (
                "for unresolved unit i in cell c use [0, 1-B_c], then report sharp "
                "fixed-denominator aggregate bounds"
            ),
            "campaign_oracle_regret": (
                "for each of the 9 opportunity cells, the lower endpoint is the best "
                "resolved positive regret or 0 and the upper endpoint additionally lets "
                "each unresolved discard attain 1-B_c; aggregate over the same 9 cells"
            ),
            "false_discard_fraction": (
                "lower/upper bounds assign every unresolved discard below/at-or-above "
                "the frozen threshold on the fixed 36-decision denominator"
            ),
            "assay_commitment_precision": (
                "exact from the frozen 24 observed assays; unresolved shadows do not "
                "change TP/(TP+FP)"
            ),
            "assay_commitment_recall": (
                "hold observed-assay TP fixed and assign every unresolved discard to "
                "TN/FN extremes before evaluating TP/(TP+FN)"
            ),
            "decision_time_discard_regret": (
                "for each unresolved eligible unit with prior incumbent I_i^- use "
                "[0, 1-I_i^-]; units without a prior assay remain null in both bounds"
            ),
        },
        "formal_status_if_any_unresolved": "incomplete_full_report_required",
    }


def _sensitivity_analysis() -> dict[str, Any]:
    return {
        "relative_near_best_fractions": list(RELATIVE_THRESHOLD_SENSITIVITY),
        "registered_absolute_score_threshold": REGISTERED_TASK_THRESHOLD,
        "decision_time_incumbent_analysis": True,
        "first_discard_without_prior_assay": "retain as null, never use a future assay",
        "censoring_definition": (
            "Censoring in this audit means an unresolved shadow evaluation; the frozen "
            "60 original lifecycles are all closed and are not right-censored."
        ),
        "mandatory_censoring_rows": [
            "unresolved count and fraction overall, by information arm, and by campaign cell",
            "unresolved reasons by prefix, identity, evaluator, resource, and nonfinite-score gate",
            "worst-case assignment S_i=0 for every unresolved shadow evaluation",
            "best-case assignment S_i=1 for every unresolved shadow evaluation",
            "sharp estimand-specific bounds from missingness_and_failure.unresolved_bounds",
        ],
        "censoring_rows_apply_to": [
            "primary threshold",
            "relative threshold sensitivities 0.80, 0.90, and 1.00",
            "registered absolute threshold 0.58",
            "decision-time incumbent analysis",
        ],
        "observed_only_rows_are_diagnostic_not_primary": True,
        "all_sensitivity_rows_mandatory": True,
        "primary_threshold_may_not_change": True,
    }


def _entry_rules() -> dict[str, Any]:
    return {
        "complete_report": (
            "Always publish all 36 unit rows, execution gates, continuous summaries, "
            "classification tables, sensitivity rows, and unresolved receipts."
        ),
        "main_text_requires": [
            "36/36 exact pre-discard prefix reconstructions",
            "36/36 valid evaluator-only shadow scores",
            "36/36 exact same-identity shadow replays",
            "zero agent/provider calls",
            "no mutation of original trajectories or resource ledgers",
        ],
        "main_text_items_if_gate_passes": [
            "latent-terminal score distribution",
            "false-discard fraction at q_c = 0.90 B_c",
            "campaign oracle-regret distribution over 9 discard-opportunity cells",
            "60-lifecycle assay/discard selection table",
            "assay commitment precision and recall",
        ],
        "main_text_if_gate_fails": (
            "State that terminal quality remains unresolved and report only the frozen "
            "failure status and pre-registered bounds; do not publish a complete-case "
            "latent-dependent point estimate or a favorable censoring assignment."
        ),
        "result_direction_gate": False,
        "significance_gate": False,
        "arm_difference_gate": False,
        "threshold_selection_after_outcomes": False,
        "failure_handling": (
            "If the gate fails, publish the complete bounded audit and mark the "
            "terminal-quality result unresolved; do not substitute a favorable subset."
        ),
    }


def build_latent_terminal_contract(root: Path) -> dict[str, Any]:
    """Build the complete latent-outcome-blind L01 contract."""

    resolved = root.resolve()
    sources = _source_manifest(resolved)
    population = _population_manifest(resolved)
    contract: dict[str, Any] = {
        "schema_id": CONTRACT_SCHEMA_ID,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "purpose": {
            "question": (
                "What terminal quality was present in states that the complete agent "
                "system chose to discard, and what does that reveal about its terminal "
                "selection policy beyond lifecycle completion counts?"
            ),
            "design": "finite-population evaluator-only counterfactual terminal census",
            "freeze_timing": "before any of the 36 latent terminal scores is read",
            "not_a_model_leaderboard": True,
        },
        "evidence_bindings": {
            "campaign_audit_sha256": FROZEN_CAMPAIGN_AUDIT_SHA256,
            "matrix_manifest_sha256": FROZEN_MATRIX_MANIFEST_SHA256,
            "public_archive_sha256": FROZEN_PUBLIC_ARCHIVE_SHA256,
            "terminal_file_index_sha256": FROZEN_TERMINAL_INDEX_SHA256,
            "complete_system_comparison_sha256": FROZEN_COMPARISON_SHA256,
            "source_manifest": sources,
            "source_manifest_sha256": canonical_json_sha256(sources),
        },
        "population": population,
        "counterfactual_terminal_rule": {
            "unit": "one original committed discard terminal",
            "branch_origin": (
                "The immutable hidden state and resource ledger immediately before the "
                "original discard_batch attempt."
            ),
            "prefix_identity_required": [
                "world and mechanism identity",
                "material instance and information-arm identity",
                "all committed prefix actions in order",
                "all public prefix observations and keyed-noise receipts",
                "hidden state immediately before discard",
                "campaign resource ledger immediately before discard",
                "lifecycle and operation ordinals",
            ],
            "intervention": (
                "Suppress only the original discard terminal and evaluate the same hidden "
                "state with the frozen final-assay observation and scoring contracts."
            ),
            "evaluator_only": True,
            "public_agent_action": False,
            "additional_process_operations_allowed": [],
            "workflow_gate_policy": (
                "Evaluator scoring bypasses agent-facing workflow readiness only; it may "
                "not advance chemistry, repair state, add material, or alter the prefix."
            ),
            "score_field": "leaderboard_score",
            "score_contract_hash": population["run_contracts"][
                "scoring_contract_hash"
            ],
            "terminal_noise": {
                "prefix_noise": "must remain byte-identical to the original prefix",
                "shadow_namespace_template": (
                    "{original_namespace}::latent-terminal-final-assay-v0.1::"
                    "{cell_id}::lifecycle-{lifecycle_index:02d}"
                ),
                "observation_seed": "retain the original campaign observation seed",
                "reuse_across_replay": True,
                "borrow_noise_from_observed_assays": False,
            },
            "branch_accounting": {
                "original_trajectory_mutated": False,
                "original_resource_ledger_mutated": False,
                "shadow_branch_receipt_required": True,
                "shadow_evaluations": EXPECTED_DISCARD_COUNT,
                "agent_provider_calls": 0,
                "count_as_original_agent_experiment": False,
                "count_as_agent_assay_decision": False,
            },
        },
        "quality_reference": {
            "campaign_benchmark_symbol": "B_c",
            "campaign_benchmark_definition": (
                "Maximum observed leaderboard_score among the original final-assay "
                "decisions in campaign cell c."
            ),
            "primary_near_best_fraction": PRIMARY_RELATIVE_THRESHOLD,
            "primary_threshold_formula": "q_c = 0.90 B_c",
            "positive_comparator": ">=",
            "rationale": (
                "The 0.90 fraction is inherited from the frozen Work I retention rule "
                "and is fixed before latent outcomes; it normalizes across physical worlds."
            ),
            "registered_absolute_threshold": REGISTERED_TASK_THRESHOLD,
            "absolute_threshold_role": "sensitivity only",
        },
        "classification_table": {
            "population": "all 60 original lifecycles",
            "assay_and_score_at_least_q_c": "TP",
            "assay_and_score_below_q_c": "FP",
            "discard_and_shadow_score_at_least_q_c": "FN",
            "discard_and_shadow_score_below_q_c": "TN",
            "equality_rule": "scores exactly equal to q_c are classified as near-best",
        },
        "estimands": _estimands(),
        "aggregation": _aggregation(),
        "missingness_and_failure": _missingness_and_failure(),
        "sensitivity_analysis": _sensitivity_analysis(),
        "entry_rules": _entry_rules(),
        "claim_boundary": {
            "allowed": [
                "quality of discarded states in this frozen complete-system demonstration",
                "whether lifecycle completion masks distinct terminal selection policies",
                "whether the best available state in a campaign was committed to assay",
                "descriptive differences between the two fixed information arms",
            ],
            "forbidden": [
                "the shadow assay was chosen or observed by the agent",
                "discarding saved real laboratory resources",
                "the complete system is generally rational or irrational",
                "a causal model-backend or general material-information effect",
                "superiority over another agent system",
                "real-laboratory executability or safety",
                "counting shadow evaluations as original agent experiments",
            ],
        },
        "freeze": {
            "latent_outcomes_read": False,
            "hidden_pre_discard_states_read": False,
            "formal_execution_authorized": False,
            "owned_next_steps": {
                "L02": "discarded-state reconstructability audit",
                "L03": "prefix-identity replay and terminal branch replacement",
                "L04": "estimand and commitment audit implementation",
                "L05": "qualification, final freeze, and 36 shadow evaluations",
                "L06": "final analysis and threshold/censoring sensitivity report",
            },
            "immutable_after_L05_freeze": [
                "population",
                "counterfactual terminal rule",
                "primary threshold",
                "estimands",
                "denominators",
                "aggregation",
                "missingness rules",
                "entry rules",
            ],
        },
    }
    contract["contract_sha256"] = latent_terminal_contract_sha256(contract)
    return contract


def latent_terminal_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a contract while excluding its embedded digest."""

    return canonical_json_sha256(_without(payload, "contract_sha256"))


def _deterministic_rebuild_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove non-scientific hashes that may advance after the L01 freeze.

    L01 froze before the integration ledger was complete.  The ledger is useful
    provenance, but it is not an input to the population, thresholds, estimands,
    or censoring rules.  This validator's own file hash must also be excluded to
    avoid a self-referential maintenance lock.  Comparing the remaining payload
    still keeps all scientific fields and every independent source binding exact.
    """

    view = deepcopy(dict(payload))
    view.pop("contract_sha256", None)
    bindings = view.get("evidence_bindings")
    if isinstance(bindings, dict):
        sources = bindings.get("source_manifest")
        if isinstance(sources, dict):
            for relative in (EXPERIMENT_LEDGER_PATH, CONTRACT_IMPLEMENTATION_PATH):
                sources.pop(relative.as_posix(), None)
        bindings.pop("source_manifest_sha256", None)
    return view


def validate_latent_terminal_contract(
    payload: Mapping[str, Any], *, root: Path | None = None
) -> list[str]:
    """Return deterministic errors for a candidate L01 contract."""

    errors: list[str] = []
    if payload.get("schema_id") != CONTRACT_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("contract_id") != CONTRACT_ID:
        errors.append("contract_id mismatch")
    if payload.get("contract_sha256") != latent_terminal_contract_sha256(payload):
        errors.append("contract self-hash mismatch")
    bindings = payload.get("evidence_bindings")
    if not isinstance(bindings, Mapping):
        errors.append("evidence_bindings must be an object")
    else:
        sources = bindings.get("source_manifest")
        if not isinstance(sources, Mapping):
            errors.append("source_manifest must be an object")
        elif bindings.get("source_manifest_sha256") != canonical_json_sha256(sources):
            errors.append("source manifest hash mismatch")
    population = payload.get("population")
    if not isinstance(population, Mapping):
        errors.append("population must be an object")
    else:
        counts = population.get("counts")
        expected_counts = {
            "cells": EXPECTED_CELL_COUNT,
            "closed_lifecycles": EXPECTED_LIFECYCLE_COUNT,
            "observed_assays": EXPECTED_ASSAY_COUNT,
            "observed_discards": EXPECTED_DISCARD_COUNT,
            "accepted_primitive_operations": EXPECTED_OPERATION_COUNT,
            "shadow_evaluations_planned": EXPECTED_DISCARD_COUNT,
            "agent_provider_calls_planned": 0,
        }
        if counts != expected_counts:
            errors.append("population counts are not the frozen census")
        if population.get("latent_outcomes_accessed") is not False:
            errors.append("L01 may not access latent outcomes")
        cells = population.get("cells")
        if not isinstance(cells, list) or len(cells) != EXPECTED_CELL_COUNT:
            errors.append("population cells are incomplete")
        elif population.get("population_manifest_sha256") != canonical_json_sha256(
            cells
        ):
            errors.append("population manifest hash mismatch")
    reference = payload.get("quality_reference")
    if not isinstance(reference, Mapping):
        errors.append("quality_reference must be an object")
    else:
        if reference.get("primary_near_best_fraction") != PRIMARY_RELATIVE_THRESHOLD:
            errors.append("primary near-best threshold changed")
        if reference.get("registered_absolute_threshold") != REGISTERED_TASK_THRESHOLD:
            errors.append("registered absolute threshold changed")
    if payload.get("estimands") != _estimands():
        errors.append("estimand definitions or denominators changed")
    if payload.get("aggregation") != _aggregation():
        errors.append("aggregation or campaign opportunity rules changed")
    if payload.get("missingness_and_failure") != _missingness_and_failure():
        errors.append("missingness or estimand-bound rules changed")
    if payload.get("sensitivity_analysis") != _sensitivity_analysis():
        errors.append("threshold or censoring sensitivity rules changed")
    if payload.get("entry_rules") != _entry_rules():
        errors.append("evidence-entry rules changed")
    freeze = payload.get("freeze")
    if not isinstance(freeze, Mapping):
        errors.append("freeze must be an object")
    elif (
        freeze.get("latent_outcomes_read") is not False
        or freeze.get("hidden_pre_discard_states_read") is not False
        or freeze.get("formal_execution_authorized") is not False
    ):
        errors.append("L01 freeze boundary was crossed")
    if root is not None:
        try:
            expected = build_latent_terminal_contract(root)
        except LatentTerminalContractError as exc:
            errors.append(f"source evidence invalid: {exc}")
        else:
            if _deterministic_rebuild_view(payload) != _deterministic_rebuild_view(
                expected
            ):
                errors.append("contract differs from deterministic outcome-blind rebuild")
    return errors


__all__ = [
    "COMPARISON_PATH",
    "CONTRACT_ID",
    "CONTRACT_IMPLEMENTATION_PATH",
    "CONTRACT_SCHEMA_ID",
    "CONTRACT_SCHEMA_VERSION",
    "EXPECTED_ASSAY_COUNT",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_DISCARD_COUNT",
    "EXPECTED_DISCARD_OPPORTUNITY_CELL_COUNT",
    "EXPECTED_LIFECYCLE_COUNT",
    "EXPECTED_OPERATION_COUNT",
    "EXPERIMENT_LEDGER_PATH",
    "FROZEN_CAMPAIGN_AUDIT_SHA256",
    "FROZEN_COMPARISON_SHA256",
    "FROZEN_MATRIX_MANIFEST_SHA256",
    "FROZEN_PUBLIC_ARCHIVE_SHA256",
    "FROZEN_TERMINAL_INDEX_SHA256",
    "PRIMARY_RELATIVE_THRESHOLD",
    "PUBLIC_ARCHIVE_MANIFEST_PATH",
    "PUBLIC_ARCHIVE_ROOT",
    "REGISTERED_TASK_THRESHOLD",
    "RELATIVE_THRESHOLD_SENSITIVITY",
    "SOURCE_PATHS",
    "TERMINAL_INDEX_PATH",
    "LatentTerminalContractError",
    "build_latent_terminal_contract",
    "latent_terminal_contract_sha256",
    "validate_latent_terminal_contract",
]
