"""Frozen five-world paired-law qualification for the Work II A-S locus.

This module deliberately contains no fitted generic response surrogate.  The two
candidate hypotheses are executable ChemWorld laws: the registered partition
coefficient power transform and the registered reversible target-pathway
topology transform.  Qualification compares their paired provider-free
executions at an outcome-blind held-out roster.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.verify import verify_records
from chemworld.eval.work_ii_crystallization_reversible_q0 import (
    validate_summary as validate_crystallization_q0,
)
from chemworld.eval.work_ii_execution_mode import (
    ExecutionMode,
    WorkIIExecutionContext,
    validate_execution_envelope,
)
from chemworld.eval.work_ii_formal import build_checkpoint_contract
from chemworld.eval.work_ii_partition_constitutive_q0 import (
    validate_nominal_pair_summary as validate_partition_q0,
)
from chemworld.tasks import get_task

QUALIFICATION_VERSION = "chemworld-work-ii-constitutive-structural-q1-q2-0.1"
PLAN_VERSION = "chemworld-work-ii-constitutive-structural-plan-0.1"
RECEIPT_VERSION = "chemworld-work-ii-constitutive-structural-receipt-0.1"
WORLD_REPORT_VERSION = "chemworld-work-ii-constitutive-structural-world-report-0.1"
PACKAGE_VERSION = "chemworld-work-ii-constitutive-structural-q2-package-0.1"
SUMMARY_VERSION = "chemworld-work-ii-constitutive-structural-five-world-summary-0.1"
EXPERIMENT_NOTE_PATH = (
    "workstreams/flagship_tasks/WORK_II_CONSTITUTIVE_STRUCTURAL_Q1_Q2_EXPERIMENT_NOTE.md"
)
WORLD_SEEDS = (0, 1, 2, 3, 4)
COORDINATES_PER_CANDIDATE_WORLD = 512
LAWS_PER_COORDINATE = 2
PRIMARY_EXECUTIONS_PER_CANDIDATE_WORLD = 1024
PRIMARY_EXECUTIONS_TOTAL = 10_240
EXACT_REPLAYS_TOTAL = 10_240
Q1_COORDINATES_PER_FAMILY = 192
Q2_COORDINATES_PER_FAMILY = 64
Q2_QUERY_COUNT_PER_FAMILY = 8
Q2_QUERY_COUNT_PER_CANDIDATE = 16
MINIMUM_SUPPORT_PER_Q2_FAMILY = 4
MINIMUM_RESOLVED_METRICS_PER_WORLD = 2
PARTITION_NOMINAL_PAIR_STRATA = tuple(
    (solvent, extractant) for solvent in range(4) for extractant in range(4)
)

PARTITION_CANDIDATE_ID = "partition_power_response"
CRYSTALLIZATION_CANDIDATE_ID = "crystallization_reversible_topology"
CANDIDATE_IDS = (PARTITION_CANDIDATE_ID, CRYSTALLIZATION_CANDIDATE_ID)
D1_CLOSEOUT_OPERATION_CLASSES = (
    "discard_batch",
    "final_assay",
    "quench",
    "terminate",
    "transfer",
)
CRYSTALLIZATION_RESOURCE_DESIGN_VERSION = (
    "crystallization-as-k12-thermal-recovery-15-start-v0.1"
)
CRYSTALLIZATION_TARGET_COMPLETE_EXPERIMENTS = 12
CRYSTALLIZATION_MAX_STARTED_BATCHES = 15


def crystallization_recovery_resource_design() -> dict[str, Any]:
    """Return the finite A-S crystallization exploration and recovery envelope."""

    target = CRYSTALLIZATION_TARGET_COMPLETE_EXPERIMENTS
    maximum_starts = CRYSTALLIZATION_MAX_STARTED_BATCHES
    start_ratio = maximum_starts / target
    required_stage_max_s = 148_800.0 * start_ratio
    repeat_allowance_s = 29_760.0 * start_ratio
    protected_reserve_s = (required_stage_max_s + repeat_allowance_s) * 0.20
    implicit_stage_reserve_s = 4_800.0 * start_ratio
    quench_transfer_allowance_s = 1_440.0 * start_ratio
    process_time_limit_s = (
        required_stage_max_s
        + repeat_allowance_s
        + protected_reserve_s
        + implicit_stage_reserve_s
        + quench_transfer_allowance_s
    )
    return {
        "card_id": CRYSTALLIZATION_RESOURCE_DESIGN_VERSION,
        "complete_experiments": target,
        "checkpoint_complete_experiments": [0, 3, 6, 9, 12],
        "final_assay_limit": target,
        "nonfinal_instrument_use_limit": 36,
        "operation_attempt_limit": 210,
        "operation_repeat_limits": {
            "heat": 24,
            "cool_crystallize": 24,
            "seed_crystals": 24,
            "filter_crystals": maximum_starts,
            "quench": maximum_starts,
        },
        "process_time_limit_s": process_time_limit_s,
        "process_time_policy": {
            "pattern_id": CRYSTALLIZATION_RESOURCE_DESIGN_VERSION,
            "formula": (
                "12 completed experiments, up to 15 started batches, twofold "
                "thermal/seed action counts, and 20% protected closeout reserve"
            ),
            "target_complete_experiments": target,
            "maximum_started_batches": maximum_starts,
            "started_batch_capacity_ratio": start_ratio,
            "required_stage_max_s": required_stage_max_s,
            "repeat_allowance_s": repeat_allowance_s,
            "protected_reserve_s": protected_reserve_s,
            "protected_reserve_fraction": 0.20,
            "implicit_stage_reserve_s": implicit_stage_reserve_s,
            "quench_transfer_allowance_s": quench_transfer_allowance_s,
            "implicit_operation_time_s": {
                "filter_crystals": 480.0,
                "quench": 120.0,
            },
            "resource_status": "participant_recovery_envelope_v0.1",
        },
        "stock_limits": {
            "reagent_mol": 0.288 * start_ratio,
            "solvent_L": 0.36 * start_ratio,
            "catalyst_mol": 0.004536 * start_ratio,
            "seed_g": 0.1152 * start_ratio,
        },
        "vessel_start_limit": maximum_starts,
        "implicit_operation_time_s": {
            "filter_crystals": 480.0,
            "quench": 120.0,
        },
        "closeout_policy": {
            "policy": "protected_closeout_reserve_enforced",
            "allowed_operation_classes": list(D1_CLOSEOUT_OPERATION_CLASSES),
            "automatic_action_repair": False,
            "automatic_closeout": False,
            "planned_batches": maximum_starts,
            "final_assay_path_operations_per_batch": 2,
            "discard_path_operations_per_batch": 1,
            "final_assay_path_total_operation_reserve": 2 * maximum_starts,
            "discard_path_total_operation_reserve": maximum_starts,
            "resource_status": "participant_recovery_envelope_v0.1",
        },
    }


def apply_crystallization_recovery_resource_design(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the current recovery envelope without changing scientific content."""

    config = copy.deepcopy(dict(source))
    if config.get("task_id") not in {None, "reaction-to-crystallization"}:
        raise ValueError("crystallization recovery resources require the crystallization task")
    campaign = config.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("crystallization recovery resources require a campaign config")
    if int(campaign.get("complete_experiments", -1)) != 12:
        raise ValueError("crystallization recovery resources require 12 complete experiments")
    design = crystallization_recovery_resource_design()
    config["campaign"] = copy.deepcopy(design)
    resources = config.get("method_resources")
    if not isinstance(resources, Mapping):
        raise ValueError("crystallization recovery resources require method resources")
    config["method_resources"] = {
        **copy.deepcopy(dict(resources)),
        "operation_limit": int(design["operation_attempt_limit"]),
    }
    qualification = config.get("qualification")
    if isinstance(qualification, Mapping):
        config["qualification"] = {
            **copy.deepcopy(dict(qualification)),
            "resource_calibration_status": "participant_recovery_envelope_v0.1",
        }
    identity = config.get("w2_26_runtime_identity")
    if isinstance(identity, Mapping):
        config["w2_26_runtime_identity"] = {
            **copy.deepcopy(dict(identity)),
            "resource_design_version": CRYSTALLIZATION_RESOURCE_DESIGN_VERSION,
        }
    return config


def partition_intervention() -> dict[str, Any]:
    return {
        "kind": "mechanism_family",
        "mode": "constitutive_law_family",
        "severity": 1.0,
        "constitutive_law_change": {
            "transform_id": "partition_power_response_stress_v1",
            "partition_coefficient_exponent_at_full_severity": 1.75,
        },
    }


def crystallization_intervention() -> dict[str, Any]:
    return {
        "kind": "mechanism_family",
        "mode": "topology_family",
        "severity": 0.8,
        "topology_change": {
            "reaction_role": "primary_target_pathway",
            "transform_id": "reversible_target_pathway_stress_v1",
            "reverse_rate_constant_s_inv_at_full_severity": 0.000625,
        },
    }


def candidate_specs() -> dict[str, dict[str, Any]]:
    """Return the immutable executable-law and measurement contracts."""

    return {
        PARTITION_CANDIDATE_ID: {
            "task_id": "partition-discovery",
            "law_ids": ("linear_response", "power_response"),
            "altered_law_id": "power_response",
            "world_intervention": partition_intervention(),
            "intervention_families": ("identity", "phase_process"),
            "metric_ids": (
                "product_in_organic",
                "product_in_aqueous",
                "phase_ratio",
            ),
            "declared_sigma": {
                "product_in_organic": 0.010,
                "product_in_aqueous": 0.010,
                "phase_ratio": 0.012,
            },
            "effect_floor": 0.03,
            "noise_multiplier": 6.0,
            "allowed_feature_ids": (
                "solvent",
                "aqueous_phase_volume_L",
                "extractant",
                "extractant_volume_L",
                "mix_duration_s",
                "settle_duration_s",
                "stirring_speed_rpm",
            ),
            "allowed_prior_fields": (
                "partition_law_family",
                "partition_coefficient_exponent",
            ),
        },
        CRYSTALLIZATION_CANDIDATE_ID: {
            "task_id": "reaction-to-crystallization",
            "law_ids": ("baseline", "reversible_target_pathway"),
            "altered_law_id": "reversible_target_pathway",
            "world_intervention": crystallization_intervention(),
            "intervention_families": ("temperature", "duration"),
            "metric_ids": ("yield", "conversion", "selectivity"),
            "declared_sigma": {
                "yield": 0.012,
                "conversion": 0.012,
                "selectivity": 0.018,
            },
            "effect_floor": 0.05,
            "noise_multiplier": 3.0,
            "allowed_feature_ids": (
                "catalyst",
                "solvent",
                "reagent_amount_mol",
                "reaction_temperature_K",
                "reaction_duration_s",
                "stirring_speed_rpm",
                "catalyst_amount_mol",
                "seed_mass_g",
                "crystallization_temperature_K",
                "crystallization_duration_s",
            ),
            "allowed_prior_fields": (
                "target_pathway_topology",
                "reverse_rate_constant_s_inv",
            ),
        },
    }


def materialize_d1_resource_design(
    source: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    """Apply the current A-S 12-round resource design to a qualified D1 config.

    This is intentionally independent of Q1/Q2 outcomes.  A frozen development
    qualification can therefore be integrated against the current downstream
    resource semantics without changing its scientific evidence.
    """

    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"unknown A-S candidate: {candidate_id}")
    config = copy.deepcopy(dict(source))
    if candidate_id == PARTITION_CANDIDATE_ID:
        operation_limit = 144
        process_time_limit_s = 38_880.0
        stock_limits = {
            "solvent_L": 0.288,
            "phase_liquid_L": 0.3456,
            "extractant_L": 0.432,
        }
        repeat_limits = {"mix": 12, "settle": 12, "separate_phase": 12}
        policy = {
            "pattern_id": "partition-as-k12-ten-unique-two-repeat-planning",
            "formula": "10 unique + 2 exact-repeat partition stages + 20% protected reserve",
            "required_stage_max_s": 27_000.0,
            "repeat_allowance_s": 5_400.0,
            "protected_reserve_s": 6_480.0,
            "protected_reserve_fraction": 0.20,
            "implicit_stage_reserve_s": 0.0,
            "resource_status": "planning_envelope_pending_w2_26_calibration",
        }
    else:
        design = crystallization_recovery_resource_design()
        operation_limit = int(design["operation_attempt_limit"])
        process_time_limit_s = float(design["process_time_limit_s"])
        stock_limits = copy.deepcopy(design["stock_limits"])
        repeat_limits = copy.deepcopy(design["operation_repeat_limits"])
        policy = copy.deepcopy(design["process_time_policy"])
    config["campaign"] = {
        "card_id": policy["pattern_id"],
        "checkpoint_complete_experiments": [0, 3, 6, 9, 12],
        "complete_experiments": 12,
        "final_assay_limit": 12,
        "nonfinal_instrument_use_limit": (
            int(design["nonfinal_instrument_use_limit"])
            if candidate_id == CRYSTALLIZATION_CANDIDATE_ID
            else 36
        ),
        "operation_attempt_limit": operation_limit,
        "operation_repeat_limits": repeat_limits,
        "process_time_limit_s": process_time_limit_s,
        "process_time_policy": policy,
        "stock_limits": stock_limits,
        "vessel_start_limit": (
            int(design["vessel_start_limit"])
            if candidate_id == CRYSTALLIZATION_CANDIDATE_ID
            else 12
        ),
        "implicit_operation_time_s": dict(policy.get("implicit_operation_time_s", {})),
        "closeout_policy": {
            "policy": "protected_closeout_reserve_enforced",
            "allowed_operation_classes": list(D1_CLOSEOUT_OPERATION_CLASSES),
            "automatic_action_repair": False,
            "automatic_closeout": False,
            "planned_batches": (
                int(design["closeout_policy"]["planned_batches"])
                if candidate_id == CRYSTALLIZATION_CANDIDATE_ID
                else 12
            ),
            "final_assay_path_operations_per_batch": 2,
            "discard_path_operations_per_batch": 1,
            "final_assay_path_total_operation_reserve": (
                int(
                    design["closeout_policy"][
                        "final_assay_path_total_operation_reserve"
                    ]
                )
                if candidate_id == CRYSTALLIZATION_CANDIDATE_ID
                else 24
            ),
            "discard_path_total_operation_reserve": (
                int(
                    design["closeout_policy"][
                        "discard_path_total_operation_reserve"
                    ]
                )
                if candidate_id == CRYSTALLIZATION_CANDIDATE_ID
                else 12
            ),
            "resource_status": (
                "participant_recovery_envelope_v0.1"
                if candidate_id == CRYSTALLIZATION_CANDIDATE_ID
                else "planning_envelope_pending_w2_26_calibration"
            ),
        },
    }
    config["method_resources"] = {
        "checkpoint_complete_experiments": [3, 6, 9, 12],
        "complete_experiment_limit": 12,
        "operation_limit": operation_limit,
        "model_call_limit": 1,
        "input_token_limit": 5_760_000,
        "uncached_input_token_limit": 768_000,
        "output_token_limit": 57_600,
        "wall_time_limit_s": 9_000.0,
        "training_environment_step_limit": 0,
        "resource_status": "development_d1_envelope_pending_w2_26_calibration",
    }
    source_qualification = config.get("qualification")
    source_qualification = (
        source_qualification if isinstance(source_qualification, Mapping) else {}
    )
    config["qualification"] = {
        "q0_q1_q2_passed": True,
        "q2_passed": True,
        # These bindings are scientific qualification evidence, not resource
        # envelope fields.  Preserve them when upgrading a completed D1 config
        # to the current 12-round downstream design.
        **(
            {"q2_package_sha256": source_qualification["q2_package_sha256"]}
            if isinstance(source_qualification.get("q2_package_sha256"), str)
            else {}
        ),
        **(
            {"plan_sha256": source_qualification["plan_sha256"]}
            if isinstance(source_qualification.get("plan_sha256"), str)
            else {}
        ),
        "minimum_unique_recipes": 10,
        "maximum_exact_repeats": 2,
        "execution_authorized": False,
        "formal_r5_authorized": False,
        "resource_calibration_status": "pending_w2_26",
    }
    return config


def _design_seed(candidate_id: str) -> int:
    digest = sha256(f"{QUALIFICATION_VERSION}:{candidate_id}:roster".encode()).hexdigest()
    return int(digest[:8], 16)


def _category(value: float, count: int) -> int:
    return min(int(float(value) * count), count - 1)


def _scale(value: float, low: float, high: float) -> float:
    return round(low + float(value) * (high - low), 12)


def _partition_features(family: str, vector: Sequence[float]) -> dict[str, Any]:
    if family == "identity":
        raise ValueError("identity features require an explicit categorical stratum")
    return {
        "solvent": 0,
        "aqueous_phase_volume_L": _scale(vector[0], 0.006, 0.024),
        "extractant": 1,
        "extractant_volume_L": _scale(vector[1], 0.008, 0.030),
        "mix_duration_s": _scale(vector[2], 120.0, 900.0),
        "settle_duration_s": _scale(vector[3], 420.0, 1800.0),
        "stirring_speed_rpm": _scale(vector[4], 400.0, 1100.0),
    }


def _partition_identity_features(family_index: int) -> dict[str, Any]:
    """Assign the nominal pair as an explicit balanced categorical stratum."""

    solvent, extractant = PARTITION_NOMINAL_PAIR_STRATA[
        family_index % len(PARTITION_NOMINAL_PAIR_STRATA)
    ]
    return {
        "solvent": solvent,
        "aqueous_phase_volume_L": 0.015,
        "extractant": extractant,
        "extractant_volume_L": 0.019,
        "mix_duration_s": 420.0,
        "settle_duration_s": 900.0,
        "stirring_speed_rpm": 800.0,
    }


def _crystallization_features(family: str, vector: Sequence[float]) -> dict[str, Any]:
    temperature = _scale(vector[0], 350.0, 420.0) if family == "temperature" else 385.0
    duration = _scale(vector[1], 1200.0, 7200.0) if family == "duration" else 3600.0
    return {
        "catalyst": _category(vector[2], 4),
        "solvent": _category(vector[3], 4),
        "reagent_amount_mol": _scale(vector[4], 0.010, 0.020),
        "reaction_temperature_K": temperature,
        "reaction_duration_s": duration,
        "stirring_speed_rpm": 675.0,
        "catalyst_amount_mol": 0.000315,
        "seed_mass_g": 0.008,
        "crystallization_temperature_K": 290.0,
        "crystallization_duration_s": 7200.0,
    }


def registered_coordinates(candidate_id: str) -> list[dict[str, Any]]:
    """Return 512 immutable coordinates, split evenly across two families.

    The roster is independent of world outcomes and is identical across all five
    worlds.  Within each family the first 192 positions are Q1 coverage and the
    remaining 64 are the frozen Q2 held-out pool.
    """

    spec = candidate_specs()[candidate_id]
    design = qmc.Sobol(d=5, scramble=True, seed=_design_seed(candidate_id)).random_base2(m=9)
    rows: list[dict[str, Any]] = []
    family_counts = dict.fromkeys(spec["intervention_families"], 0)
    for coordinate_index, vector in enumerate(design):
        family = str(spec["intervention_families"][coordinate_index % 2])
        family_index = int(family_counts[family])
        family_counts[family] += 1
        phase = "q1_coverage" if family_index < Q1_COORDINATES_PER_FAMILY else "q2_heldout"
        features = (
            (
                _partition_identity_features(family_index)
                if family == "identity"
                else _partition_features(family, vector)
            )
            if candidate_id == PARTITION_CANDIDATE_ID
            else _crystallization_features(family, vector)
        )
        rows.append(
            {
                "coordinate_id": f"c{coordinate_index:03d}",
                "coordinate_index": coordinate_index,
                "family_index": family_index,
                "phase": phase,
                "intervention_family": family,
                "feature_values": features,
                "coordinate_sha256": canonical_json_sha256(
                    {
                        "candidate_id": candidate_id,
                        "coordinate_index": coordinate_index,
                        "phase": phase,
                        "intervention_family": family,
                        "feature_values": features,
                    }
                ),
            }
        )
    if len(rows) != COORDINATES_PER_CANDIDATE_WORLD:
        raise AssertionError("paired-law coordinate denominator drifted")
    return rows


def selected_q2_queries(candidate_id: str) -> list[dict[str, Any]]:
    """Select 16 held-out queries using coordinates only, never outcomes."""

    rows = registered_coordinates(candidate_id)
    selected: list[dict[str, Any]] = []
    for family in candidate_specs()[candidate_id]["intervention_families"]:
        pool = [
            row
            for row in rows
            if row["phase"] == "q2_heldout" and row["intervention_family"] == family
        ]
        indices = np.linspace(0, len(pool) - 1, Q2_QUERY_COUNT_PER_FAMILY, dtype=int)
        selected.extend(pool[int(index)] for index in indices)
    return [dict(row) for row in selected]


def observation_binding(candidate_id: str, world_seed: int, coordinate_id: str) -> tuple[int, str]:
    digest = sha256(
        f"{QUALIFICATION_VERSION}:{candidate_id}:{world_seed}:{coordinate_id}".encode()
    ).hexdigest()
    return (
        int(digest[:8], 16) % 2_147_483_647,
        f"work-ii-as-{candidate_id}-w{world_seed}-{digest[:12]}",
    )


def build_prior_arms(candidate_id: str) -> dict[str, dict[str, Any]]:
    spec = candidate_specs()[candidate_id]
    if candidate_id == PARTITION_CANDIDATE_ID:
        aligned_claim = (
            "Extraction follows a nonlinear power response in the solvent-extractant partition "
            "coefficient; phase balance and mixing conditions modulate the observable separation."
        )
        misspecified_claim = (
            "Extraction follows a linear reference response in the solvent-extractant partition "
            "coefficient; phase balance and mixing conditions modulate the observable separation."
        )
        aligned_law = {
            "law_id": "power_response",
            "world_interventions": [partition_intervention()],
        }
        misspecified_law = {"law_id": "linear_response", "world_interventions": []}
    else:
        aligned_claim = (
            "The primary target pathway is reversible; temperature and reaction duration jointly "
            "expose accumulated reverse flux before crystallization and terminal assay."
        )
        misspecified_claim = (
            "The primary target pathway is irreversible; temperature and reaction duration jointly "
            "expose accumulated forward flux before crystallization and terminal assay."
        )
        aligned_law = {
            "law_id": "reversible_target_pathway",
            "world_interventions": [crystallization_intervention()],
        }
        misspecified_law = {"law_id": "baseline", "world_interventions": []}
    common = {
        "schema_version": "chemworld-work-ii-initial-world-model-0.4",
        "locus": "structural_mechanistic",
        "confidence": 0.70,
        "intervention_families": list(spec["intervention_families"]),
        "scope_limit": (
            "This is an incomplete local law. Public experimental evidence is authoritative."
        ),
    }
    return {
        "opaque": {
            **common,
            "availability": "opaque_for_target_locus",
            "claim": None,
            "executable_law": None,
        },
        "aligned_nominal": {
            **common,
            "availability": "supplied_incomplete_executable_law",
            "claim": aligned_claim,
            "executable_law": aligned_law,
        },
        "misindexed_nominal": {
            **common,
            "availability": "supplied_incomplete_executable_law",
            "claim": misspecified_claim,
            "executable_law": misspecified_law,
        },
    }


def effect_gate(candidate_id: str, metric: str) -> float:
    spec = candidate_specs()[candidate_id]
    return max(
        float(spec["effect_floor"]),
        float(spec["noise_multiplier"]) * float(spec["declared_sigma"][metric]),
    )


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def plan_sha256(plan: Mapping[str, Any]) -> str:
    return _self_hash(plan, "plan_sha256")


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return _self_hash(receipt, "receipt_sha256")


def package_sha256(package: Mapping[str, Any]) -> str:
    return _self_hash(package, "package_sha256")


def build_q2_package(
    reports: Sequence[Mapping[str, Any]],
    *,
    q0_bindings: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    all_passed = len(reports) == len(CANDIDATE_IDS) * len(WORLD_SEEDS) and all(
        report.get("analysis", {}).get("passed") is True for report in reports
    )
    package: dict[str, Any] = {
        "schema_version": PACKAGE_VERSION,
        "qualification_schema_version": QUALIFICATION_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "execution_context": dict(execution_context),
        "plan_binding": dict(plan_binding),
        "q0_bindings": dict(q0_bindings),
        "candidate_laws": {
            candidate_id: {
                "task_id": candidate_specs()[candidate_id]["task_id"],
                "law_ids": list(candidate_specs()[candidate_id]["law_ids"]),
                "registered_truth_law_id": candidate_specs()[candidate_id]["altered_law_id"],
                "world_intervention": candidate_specs()[candidate_id]["world_intervention"],
                "prior_arms": build_prior_arms(candidate_id),
                "outcome_blind_q2_queries": selected_q2_queries(candidate_id),
                "world_evidence": [
                    {
                        "world_seed": report["world_seed"],
                        "passed": report["analysis"]["passed"],
                        "q2": report["analysis"]["q2"],
                    }
                    for report in reports
                    if report.get("candidate_id") == candidate_id
                ],
            }
            for candidate_id in CANDIDATE_IDS
        },
        "all_five_world_cohorts_passed": all_passed,
    }
    package["package_sha256"] = package_sha256(package)
    return package


def build_qualification_plan(
    root: Path,
    *,
    q0_bindings: Mapping[str, Any],
    execution_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the frozen A-S question without hashing unrelated repository files."""

    note_path = (root / EXPERIMENT_NOTE_PATH).resolve()
    note_path.relative_to(root.resolve())
    candidates = {}
    for candidate_id in CANDIDATE_IDS:
        spec = json.loads(json.dumps(candidate_specs()[candidate_id], sort_keys=True))
        roster = registered_coordinates(candidate_id)
        candidates[candidate_id] = {
            "spec": spec,
            "effect_gates": {
                metric: effect_gate(candidate_id, metric) for metric in spec["metric_ids"]
            },
            "coordinate_roster_sha256": canonical_json_sha256(roster),
            "q2_query_roster_sha256": canonical_json_sha256(selected_q2_queries(candidate_id)),
        }
    plan: dict[str, Any] = {
        "schema_version": PLAN_VERSION,
        "qualification_schema_version": QUALIFICATION_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "execution_context": dict(execution_context),
        "experiment_note_binding": {
            "path": EXPERIMENT_NOTE_PATH,
            "sha256": file_sha256(note_path),
        },
        "q0_bindings": dict(q0_bindings),
        "world_seeds": list(WORLD_SEEDS),
        "coverage": {
            "coordinates_per_candidate_world": COORDINATES_PER_CANDIDATE_WORLD,
            "laws_per_coordinate": LAWS_PER_COORDINATE,
            "primary_executions_per_candidate_world": PRIMARY_EXECUTIONS_PER_CANDIDATE_WORLD,
            "primary_execution_count": PRIMARY_EXECUTIONS_TOTAL,
            "exact_replay_count": EXACT_REPLAYS_TOTAL,
            "q1_coordinates_per_family": Q1_COORDINATES_PER_FAMILY,
            "q2_coordinates_per_family": Q2_COORDINATES_PER_FAMILY,
            "q2_queries_per_family": Q2_QUERY_COUNT_PER_FAMILY,
        },
        "pass_rules": {
            "minimum_support_per_q2_family": MINIMUM_SUPPORT_PER_Q2_FAMILY,
            "minimum_resolved_metrics_per_world": MINIMUM_RESOLVED_METRICS_PER_WORLD,
            "all_candidate_worlds_must_pass": True,
            "tolerance_zero_exact_replay_required": True,
        },
        "candidates": candidates,
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def _validated_q0_bindings(
    root: Path,
    bindings: Mapping[str, Any],
    *,
    expected_execution_context: WorkIIExecutionContext | None,
) -> list[str]:
    errors: list[str] = []
    if set(bindings) != set(CANDIDATE_IDS):
        return ["A-S plan Q0 binding roster mismatch"]
    for candidate_id in CANDIDATE_IDS:
        binding = bindings.get(candidate_id)
        if not isinstance(binding, Mapping):
            errors.append(f"A-S plan Q0 binding is malformed: {candidate_id}")
            continue
        try:
            path = (root / str(binding["path"])).resolve()
            path.relative_to(root.resolve())
            if not path.is_file():
                raise ValueError("summary is missing")
            if binding.get("sha256") != file_sha256(path):
                errors.append(f"A-S plan Q0 file hash mismatch: {candidate_id}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("summary is not an object")
            q0_errors = (
                validate_partition_q0(
                    payload,
                    root=root,
                    expected_execution_context=expected_execution_context,
                )
                if candidate_id == PARTITION_CANDIDATE_ID
                else validate_crystallization_q0(
                    root,
                    payload,
                    expected_execution_context=expected_execution_context,
                )
            )
            errors.extend(f"A-S {candidate_id} Q0: {error}" for error in q0_errors)
            if payload.get("analysis", {}).get("passed") is not True:
                errors.append(f"A-S Q0 did not pass: {candidate_id}")
            if binding.get("summary_sha256") != payload.get("summary_sha256"):
                errors.append(f"A-S plan Q0 summary hash mismatch: {candidate_id}")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"A-S plan Q0 cannot be read: {candidate_id}: {error}")
    return errors


def validate_qualification_plan(
    root: Path,
    plan: Mapping[str, Any],
    *,
    expected_execution_context: WorkIIExecutionContext | None = None,
) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != PLAN_VERSION:
        errors.append("unexpected A-S qualification-plan schema")
    if plan.get("qualification_schema_version") != QUALIFICATION_VERSION:
        errors.append("A-S qualification-plan version mismatch")
    if plan.get("plan_sha256") != plan_sha256(plan):
        errors.append("A-S qualification-plan self-hash mismatch")
    if plan.get("formal_result") is not False or plan.get("provider_call_count") != 0:
        errors.append("A-S qualification plan crossed its provider-free boundary")
    envelope = plan.get("execution_context")
    if not isinstance(envelope, Mapping):
        errors.append("A-S qualification plan lacks an execution context")
    else:
        errors.extend(
            validate_execution_envelope(root, envelope, expected_context=expected_execution_context)
        )
    note = plan.get("experiment_note_binding")
    if not isinstance(note, Mapping) or note.get("path") != EXPERIMENT_NOTE_PATH:
        errors.append("A-S qualification plan lacks its experiment-note binding")
    else:
        try:
            note_path = (root / str(note["path"])).resolve()
            note_path.relative_to(root.resolve())
            if not note_path.is_file() or note.get("sha256") != file_sha256(note_path):
                errors.append("A-S qualification experiment-note binding is stale")
        except (KeyError, OSError, TypeError, ValueError):
            errors.append("A-S qualification experiment-note binding is invalid")
    q0_bindings = plan.get("q0_bindings")
    if not isinstance(q0_bindings, Mapping):
        errors.append("A-S qualification plan lacks Q0 bindings")
    else:
        errors.extend(
            _validated_q0_bindings(
                root,
                q0_bindings,
                expected_execution_context=expected_execution_context,
            )
        )
    expected = build_qualification_plan(
        root,
        q0_bindings=q0_bindings if isinstance(q0_bindings, Mapping) else {},
        execution_context=envelope if isinstance(envelope, Mapping) else {},
    )
    if plan != expected:
        errors.append("A-S qualification plan differs from the frozen spec or roster")
    return errors


def _pairs(candidate_id: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    spec = candidate_specs()[candidate_id]
    pairs = []
    for coordinate in registered_coordinates(candidate_id):
        selected = [row for row in rows if row.get("coordinate_id") == coordinate["coordinate_id"]]
        laws = {str(row.get("law_id")): row for row in selected}
        if len(selected) != LAWS_PER_COORDINATE or set(laws) != set(spec["law_ids"]):
            raise ValueError(
                f"{candidate_id}/{coordinate['coordinate_id']} lacks exactly two registered laws"
            )
        pairs.append({**coordinate, "laws": laws})
    return pairs


def denominators(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "planned_primary_executions": PRIMARY_EXECUTIONS_PER_CANDIDATE_WORLD,
        "attempted_primary_executions": len(rows),
        "completed_primary_executions": sum(row.get("status") == "completed" for row in rows),
        "physical_failures": sum(row.get("status") == "physical_failure" for row in rows),
        "platform_failures": sum(row.get("status") == "platform_failure" for row in rows),
        "unsafe_completed": sum(
            row.get("status") == "completed" and row.get("safe") is False for row in rows
        ),
        "exact_replays": sum(row.get("exact_replay") is True for row in rows),
    }


def _registered_bindings(
    candidate_id: str, world_seed: int, rows: Sequence[Mapping[str, Any]]
) -> bool:
    coordinates = {row["coordinate_id"]: row for row in registered_coordinates(candidate_id)}
    try:
        return all(
            row.get("candidate_id") == candidate_id
            and row.get("task_id") == candidate_specs()[candidate_id]["task_id"]
            and row.get("world_seed") == world_seed
            and row.get("coordinate_id") in coordinates
            and row.get("coordinate_sha256")
            == coordinates[str(row["coordinate_id"])]["coordinate_sha256"]
            and row.get("feature_values")
            == coordinates[str(row["coordinate_id"])]["feature_values"]
            for row in rows
        )
    except (KeyError, TypeError):
        return False


def _paired_binding_checks(pairs: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {
        "paired_action_plans": all(
            len({law.get("action_plan_sha256") for law in pair["laws"].values()}) == 1
            and next(iter(pair["laws"].values())).get("action_plan_sha256") is not None
            for pair in pairs
        ),
        "paired_observation_noise": all(
            len({law.get("observation_coordinate_sha256") for law in pair["laws"].values()}) == 1
            and next(iter(pair["laws"].values())).get("observation_coordinate_sha256") is not None
            for pair in pairs
        ),
        "all_trajectories_hash_bound": all(
            isinstance(law.get("trajectory"), Mapping)
            and isinstance(law["trajectory"].get("path"), str)
            and isinstance(law["trajectory"].get("sha256"), str)
            for pair in pairs
            for law in pair["laws"].values()
        ),
        "paired_safety_classified": all(
            all(isinstance(law.get("safe"), bool) for law in pair["laws"].values())
            for pair in pairs
        ),
    }


def _law_binding_check(
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
    law_audit: Mapping[str, Any],
) -> bool:
    spec = candidate_specs()[candidate_id]
    baseline, altered = spec["law_ids"]
    baseline_rows = [row for row in rows if row.get("law_id") == baseline]
    altered_rows = [row for row in rows if row.get("law_id") == altered]
    common = (
        law_audit.get("altered_hash_deterministic") is True
        and law_audit.get("world_intervention") == spec["world_intervention"]
        and law_audit.get("registered_law_ids") == list(spec["law_ids"])
    )
    if candidate_id == PARTITION_CANDIDATE_ID:
        baseline_mechanism_hash = law_audit.get("baseline_mechanism_hash")
        altered_intervention_hash = law_audit.get("altered_intervention_hash")
        return (
            common
            and law_audit.get("mechanism_hash_changed") is False
            and isinstance(baseline_mechanism_hash, str)
            and isinstance(altered_intervention_hash, str)
            and law_audit.get("altered_mechanism_hash") == baseline_mechanism_hash
            and law_audit.get("only_registered_constitutive_parameter_changed") is True
            and law_audit.get("changed_domain_parameter_keys") == ["partition_coefficient_exponent"]
            and {row.get("intervention_hash") for row in baseline_rows} == {None}
            and {row.get("intervention_hash") for row in altered_rows}
            == {altered_intervention_hash}
            and {row.get("mechanism_hash") for row in rows} == {baseline_mechanism_hash}
        )
    return (
        common
        and law_audit.get("mechanism_hash_changed") is True
        and law_audit.get("baseline_mechanism_hash") != law_audit.get("altered_mechanism_hash")
        and law_audit.get("transform_id") == "reversible_target_pathway_stress_v1"
        and {row.get("mechanism_hash") for row in baseline_rows}
        == {law_audit.get("baseline_mechanism_hash")}
        and {row.get("mechanism_hash") for row in altered_rows}
        == {law_audit.get("altered_mechanism_hash")}
        and law_audit.get("added_reaction_count") == 1
    )


def analyze_candidate_world(
    candidate_id: str,
    world_seed: int,
    rows: Sequence[Mapping[str, Any]],
    law_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply Q1 coverage and Q2 actual-law identifiability gates."""

    spec = candidate_specs()[candidate_id]
    checks: dict[str, bool] = {
        "registered_world": world_seed in WORLD_SEEDS,
        "fixed_primary_denominator": len(rows) == PRIMARY_EXECUTIONS_PER_CANDIDATE_WORLD,
        "all_outcomes_classified": all(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in rows
        ),
        "all_primary_executions_completed": all(row.get("status") == "completed" for row in rows),
        "zero_physical_failures": not any(row.get("status") == "physical_failure" for row in rows),
        "zero_platform_failures": not any(row.get("status") == "platform_failure" for row in rows),
        "all_exact_replays": all(row.get("exact_replay") is True for row in rows),
        "registered_coordinate_bindings": _registered_bindings(candidate_id, world_seed, rows),
        "executable_law_binding": _law_binding_check(candidate_id, rows, law_audit),
        "participant_visible_leakage_free": not any(
            row.get("participant_visible_leakage_matches") for row in rows
        ),
    }
    try:
        pairs = _pairs(candidate_id, rows)
    except (KeyError, TypeError, ValueError):
        pairs = []
        checks["complete_paired_law_roster"] = False
    else:
        checks["complete_paired_law_roster"] = len(pairs) == COORDINATES_PER_CANDIDATE_WORLD
        checks.update(_paired_binding_checks(pairs))
    if not all(checks.values()):
        return {
            "candidate_id": candidate_id,
            "world_seed": world_seed,
            "passed": False,
            "checks": checks,
            "failures": sorted(key for key, passed in checks.items() if not passed),
            "denominators": denominators(rows),
            "law_audit": dict(law_audit),
            "q1": None,
            "q2": None,
        }

    baseline_law, altered_law = spec["law_ids"]
    q1_pairs = [pair for pair in pairs if pair["phase"] == "q1_coverage"]
    q2_ids = {row["coordinate_id"] for row in selected_q2_queries(candidate_id)}
    q2_pairs = [pair for pair in pairs if pair["coordinate_id"] in q2_ids]
    q1_family_counts = {
        family: sum(pair["intervention_family"] == family for pair in q1_pairs)
        for family in spec["intervention_families"]
    }
    finite = all(
        all(math.isfinite(float(law["metrics"][metric])) for metric in spec["metric_ids"])
        for pair in pairs
        for law in pair["laws"].values()
    )
    family_reports: dict[str, Any] = {}
    resolved_metrics: set[str] = set()
    for family in spec["intervention_families"]:
        selected = [pair for pair in q2_pairs if pair["intervention_family"] == family]
        query_reports = []
        supporting = 0
        for pair in selected:
            metric_gaps = {
                metric: float(pair["laws"][altered_law]["metrics"][metric])
                - float(pair["laws"][baseline_law]["metrics"][metric])
                for metric in spec["metric_ids"]
            }
            passed_metrics = [
                metric
                for metric, gap in metric_gaps.items()
                if abs(gap) >= effect_gate(candidate_id, metric)
            ]
            resolved_metrics.update(passed_metrics)
            is_supporting = bool(passed_metrics)
            supporting += is_supporting
            query_reports.append(
                {
                    "coordinate_id": pair["coordinate_id"],
                    "coordinate_sha256": pair["coordinate_sha256"],
                    "metric_gaps": metric_gaps,
                    "passed_metrics": passed_metrics,
                    "supports_law_contrast": is_supporting,
                    "candidate_predictions": {
                        "blind_law_a": dict(pair["laws"][baseline_law]["metrics"]),
                        "blind_law_b": dict(pair["laws"][altered_law]["metrics"]),
                    },
                    "altered_world_observation": dict(pair["laws"][altered_law]["metrics"]),
                }
            )
        family_reports[family] = {
            "selected_query_count": len(selected),
            "supporting_query_count": supporting,
            "passed": len(selected) == Q2_QUERY_COUNT_PER_FAMILY
            and supporting >= MINIMUM_SUPPORT_PER_Q2_FAMILY,
            "queries": query_reports,
        }

    q2_roster = selected_q2_queries(candidate_id)
    q2 = {
        "selection_policy": "coordinate_only_even_spread_within_each_frozen_heldout_family",
        "selection_reads_outcomes": False,
        "query_count": len(q2_pairs),
        "query_roster_sha256": canonical_json_sha256(q2_roster),
        "candidate_laws": {
            "blind_law_a": {
                "registered_law_id": baseline_law,
                "world_interventions": [],
                "prediction_source": "direct_provider_free_execution",
            },
            "blind_law_b": {
                "registered_law_id": altered_law,
                "world_interventions": [spec["world_intervention"]],
                "prediction_source": "direct_provider_free_execution",
            },
        },
        "truth_law_id": altered_law,
        "blind_identified_truth_law": "blind_law_b",
        "family_reports": family_reports,
        "resolved_metrics": sorted(resolved_metrics),
    }
    q1 = {
        "coverage_coordinate_count": len(q1_pairs),
        "family_coordinate_counts": q1_family_counts,
        "all_metrics_finite": finite,
        "paired_law_execution_count": 2 * len(q1_pairs),
    }
    checks.update(
        {
            "q1_fixed_coverage": len(q1_pairs) == 384
            and set(q1_family_counts.values()) == {Q1_COORDINATES_PER_FAMILY},
            "all_registered_metrics_finite": finite,
            "q2_outcome_blind_selection": q2["selection_reads_outcomes"] is False,
            "q2_fixed_query_denominator": len(q2_pairs) == Q2_QUERY_COUNT_PER_CANDIDATE,
            "both_intervention_families_resolve_laws": all(
                report["passed"] for report in family_reports.values()
            ),
            "multiple_metrics_resolve_laws": len(resolved_metrics)
            >= MINIMUM_RESOLVED_METRICS_PER_WORLD,
            "actual_registered_laws_compared": all(
                candidate["prediction_source"] == "direct_provider_free_execution"
                for candidate in q2["candidate_laws"].values()
            ),
        }
    )
    return {
        "candidate_id": candidate_id,
        "world_seed": world_seed,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, passed in checks.items() if not passed),
        "denominators": denominators(rows),
        "law_audit": dict(law_audit),
        "q1": q1,
        "q2": q2,
    }


def _trajectory_metrics(
    candidate_id: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    spec = candidate_specs()[candidate_id]
    if candidate_id == PARTITION_CANDIDATE_ID:
        measurements = [
            row
            for row in records
            if row.get("transaction_status") == "committed"
            and row.get("operation_type") == "measure"
            and row.get("instrument") == "final_assay"
        ]
        if len(measurements) != 1:
            raise ValueError("partition evidence lacks one committed final assay")
    else:
        measurements = [
            row
            for row in records
            if row.get("transaction_status") == "committed"
            and row.get("operation_type") == "measure"
            and row.get("instrument") == "hplc"
        ]
        if len(measurements) != 2:
            raise ValueError("crystallization evidence lacks two committed HPLC assays")
    payload = measurements[0].get("processed_estimate")
    if not isinstance(payload, Mapping):
        raise ValueError("registered measurement lacks processed estimates")
    metrics: dict[str, float] = {}
    for metric in spec["metric_ids"]:
        value = payload.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"registered measurement lacks numeric {metric}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"registered measurement {metric} is not finite")
        metrics[str(metric)] = number
    return metrics


def _trajectory_leakage(records: Sequence[Mapping[str, Any]]) -> list[str]:
    tokens = (
        "mechanism_family",
        "world_intervention",
        "private_seed",
        "hidden_state",
        "evaluator_truth",
    )
    matches = set()
    for row in records:
        public = {
            key: row.get(key)
            for key in (
                "observation",
                "observed_mask",
                "processed_estimate",
                "raw_signal",
                "agent_visible_observation",
                "agent_view",
            )
        }
        rendered = json.dumps(public, ensure_ascii=False, sort_keys=True)
        matches.update(token for token in tokens if token in rendered)
    return sorted(matches)


def validate_execution_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    candidate_id: str,
    world_seed: int,
) -> list[str]:
    """Reopen one trajectory and prove the cached receipt from its bytes."""

    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_VERSION:
        errors.append("A-S execution receipt schema mismatch")
    if receipt.get("receipt_sha256") != receipt_sha256(receipt):
        errors.append("A-S execution receipt self-hash mismatch")
    coordinate_id = str(receipt.get("coordinate_id", ""))
    registered = {row["coordinate_id"]: row for row in registered_coordinates(candidate_id)}
    coordinate = registered.get(coordinate_id)
    spec = candidate_specs()[candidate_id]
    law_id = receipt.get("law_id")
    if (
        coordinate is None
        or law_id not in spec["law_ids"]
        or receipt.get("candidate_id") != candidate_id
        or receipt.get("task_id") != spec["task_id"]
        or receipt.get("world_seed") != world_seed
        or any(receipt.get(key) != value for key, value in coordinate.items())
    ):
        errors.append("A-S execution receipt differs from the frozen coordinate roster")
        return errors
    trajectory = receipt.get("trajectory")
    if not isinstance(trajectory, Mapping):
        errors.append("A-S execution receipt lacks a trajectory binding")
        return errors
    try:
        trajectory_path = (root / str(trajectory["path"])).resolve()
        trajectory_path.relative_to(root.resolve())
        if not trajectory_path.is_file():
            raise ValueError("trajectory is missing")
        if trajectory.get("sha256") != file_sha256(trajectory_path):
            errors.append("A-S execution trajectory file hash mismatch")
        records = load_jsonl(trajectory_path)
        expected_actions = _expected_actions(candidate_id, coordinate["feature_values"])
        if [row.get("action") for row in records] != expected_actions:
            errors.append("A-S execution trajectory differs from the frozen action plan")
        expected_action_hash = canonical_json_sha256({"actions": expected_actions})
        if receipt.get("action_plan_sha256") != expected_action_hash:
            errors.append("A-S execution receipt action hash mismatch")
        observation_seed, namespace = observation_binding(candidate_id, world_seed, coordinate_id)
        if any(
            row.get("observation_seed") != observation_seed
            or row.get("observation_noise_mode") != "keyed"
            or row.get("observation_noise_namespace") != namespace
            for row in records
        ):
            errors.append("A-S trajectory observation coordinate mismatch")
        if receipt.get("observation_coordinate_sha256") != canonical_json_sha256(
            {
                "observation_seed": observation_seed,
                "observation_noise_namespace": namespace,
            }
        ):
            errors.append("A-S execution receipt observation hash mismatch")
        interventions = [] if law_id == spec["law_ids"][0] else [spec["world_intervention"]]
        replay = verify_records(records, tolerance=0.0, world_interventions=interventions).to_dict()
        if replay.get("verified") is not True:
            errors.append("A-S execution trajectory failed tolerance-zero replay")
        if receipt.get("replay") != replay or receipt.get("exact_replay") is not True:
            errors.append("A-S execution replay receipt mismatch")
        noncommitted = [row for row in records if row.get("transaction_status") != "committed"]
        if noncommitted and not all(
            row.get("rollback_reason") == "constitution_failed" for row in noncommitted
        ):
            errors.append("A-S trajectory contains a non-constitution rollback")
        expected_physical_failure = (
            {
                "rollback_count": len(noncommitted),
                "operations": sorted({str(row.get("operation_type")) for row in noncommitted}),
                "attribution": "protocol_owned_physical_boundary",
            }
            if noncommitted
            else None
        )
        metrics = None if noncommitted else _trajectory_metrics(candidate_id, records)
        finals = [
            row
            for row in records
            if row.get("transaction_status") == "committed"
            and row.get("operation_type") == "measure"
            and row.get("instrument") == "final_assay"
        ]
        if noncommitted:
            safe = None
        else:
            risk = (
                trajectory_value(finals[0], "observation", {}).get("safety_risk")
                if len(finals) == 1 else None
            )
            if isinstance(risk, bool) or not isinstance(risk, int | float):
                raise ValueError("trajectory lacks one finite final safety assay")
            safe = float(risk) < float(get_task(str(spec["task_id"])).safety_limit)
        mechanism_hashes = {
            row.get("mechanism_hash")
            for row in records
            if isinstance(row.get("mechanism_hash"), str)
        }
        intervention_hashes = {
            row.get("mechanism_family_intervention_hash")
            for row in records
            if isinstance(row.get("mechanism_family_intervention_hash"), str)
        }
        expected_intervention = (
            next(iter(intervention_hashes)) if len(intervention_hashes) == 1 else None
        )
        if law_id == spec["law_ids"][0] and intervention_hashes:
            errors.append("A-S baseline trajectory carries an intervention hash")
        if law_id != spec["law_ids"][0] and len(intervention_hashes) != 1:
            errors.append("A-S altered trajectory lacks one intervention hash")
        if (
            receipt.get("status") != ("physical_failure" if noncommitted else "completed")
            or receipt.get("physical_failure") != expected_physical_failure
            or receipt.get("platform_failure") is not None
            or receipt.get("metrics") != metrics
            or receipt.get("safe") is not safe
            or receipt.get("participant_visible_leakage_matches") != _trajectory_leakage(records)
            or len(mechanism_hashes) != 1
            or receipt.get("mechanism_hash") != next(iter(mechanism_hashes), None)
            or receipt.get("intervention_hash") != expected_intervention
        ):
            errors.append("A-S execution receipt differs from trajectory evidence")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"A-S execution trajectory validation failed: {error}")
    return errors


def _expected_actions(candidate_id: str, features: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pure copy of the frozen protocol compiler, kept with evidence validation."""

    if candidate_id == PARTITION_CANDIDATE_ID:
        return [
            {"operation": "add_solvent", "volume_L": 0.020, "solvent": int(features["solvent"])},
            {
                "operation": "add_phase",
                "phase": "aqueous",
                "volume_L": float(features["aqueous_phase_volume_L"]),
            },
            {
                "operation": "add_extractant",
                "extractant": int(features["extractant"]),
                "volume_L": float(features["extractant_volume_L"]),
            },
            {
                "operation": "mix",
                "duration_s": float(features["mix_duration_s"]),
                "stirring_speed_rpm": float(features["stirring_speed_rpm"]),
            },
            {"operation": "settle", "duration_s": float(features["settle_duration_s"])},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "separate_phase", "target_phase": "organic"},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
    return [
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": int(features["solvent"])},
        {"operation": "add_reagent", "amount_mol": float(features["reagent_amount_mol"])},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": float(features["catalyst_amount_mol"]),
            "catalyst": int(features["catalyst"]),
        },
        {
            "operation": "heat",
            "target_temperature_K": float(features["reaction_temperature_K"]),
            "duration_s": float(features["reaction_duration_s"]),
            "stirring_speed_rpm": float(features["stirring_speed_rpm"]),
        },
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "seed_crystals", "seed_mass_g": float(features["seed_mass_g"])},
        {
            "operation": "cool_crystallize",
            "target_temperature_K": float(features["crystallization_temperature_K"]),
            "duration_s": float(features["crystallization_duration_s"]),
        },
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "filter_crystals"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def report_sha256(report: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def summary_sha256(summary: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )


def validate_world_report(
    report: Mapping[str, Any],
    *,
    root: Path | None = None,
    expected_execution_context: WorkIIExecutionContext | None = None,
    evidence_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    errors: list[str] = []
    envelope = report.get("execution_context")
    if not isinstance(envelope, Mapping):
        errors.append("A-S world report lacks an execution context")
        mode = None
    else:
        mode = envelope.get("execution_mode")
        if root is not None:
            errors.extend(
                validate_execution_envelope(
                    root, envelope, expected_context=expected_execution_context
                )
            )
        elif mode not in {item.value for item in ExecutionMode}:
            errors.append("A-S world report has an invalid execution context")
    if report.get("schema_version") != WORLD_REPORT_VERSION:
        errors.append("unexpected A-S world-report schema")
    if report.get("qualification_schema_version") != QUALIFICATION_VERSION:
        errors.append("A-S qualification schema mismatch")
    if report.get("formal_result") is not False:
        errors.append("A-S qualification must not be formal")
    if report.get("provider_call_count") != 0 or report.get("participant_session_count") != 0:
        errors.append("A-S qualification must remain provider-free")
    if report.get("report_sha256") != report_sha256(report):
        errors.append("A-S world-report self-hash mismatch")
    plan_binding = report.get("plan_binding")
    if not isinstance(plan_binding, Mapping) or root is None:
        errors.append("A-S world report lacks its qualification-plan binding")
    else:
        try:
            plan_path = (root / str(plan_binding["path"])).resolve()
            plan_path.relative_to(root.resolve())
            if not plan_path.is_file() or plan_binding.get("sha256") != file_sha256(plan_path):
                raise ValueError("plan file binding is stale")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(plan, dict):
                raise TypeError("plan is not an object")
            if plan_binding.get("plan_sha256") != plan.get("plan_sha256"):
                raise ValueError("embedded plan hash mismatch")
            errors.extend(
                validate_qualification_plan(
                    root,
                    plan,
                    expected_execution_context=expected_execution_context,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"A-S world report plan binding cannot be read: {error}")
    candidate_id = report.get("candidate_id")
    world_seed = report.get("world_seed")
    rows = report.get("rows")
    audit = report.get("law_audit")
    if (
        candidate_id not in CANDIDATE_IDS
        or world_seed not in WORLD_SEEDS
        or not isinstance(rows, list)
        or not isinstance(audit, Mapping)
    ):
        errors.append("A-S world report lacks a registered candidate/world/rows/audit")
    else:
        try:
            validated_rows: list[dict[str, Any]] = []
            for receipt_index, embedded in enumerate(rows, start=1):
                if not isinstance(embedded, Mapping):
                    raise TypeError("embedded row is not an object")
                receipt_binding = embedded.get("receipt")
                if not isinstance(receipt_binding, Mapping):
                    raise ValueError("embedded row lacks its receipt binding")
                receipt_path = (root / str(receipt_binding["path"])).resolve() if root else None
                if receipt_path is None:
                    raise ValueError("evidence root is required")
                receipt_path.relative_to(root.resolve())
                if not receipt_path.is_file():
                    raise ValueError("execution receipt is missing")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if not isinstance(receipt, dict):
                    raise TypeError("execution receipt is not an object")
                if receipt_binding.get("sha256") != file_sha256(
                    receipt_path
                ) or receipt_binding.get("receipt_sha256") != receipt.get("receipt_sha256"):
                    raise ValueError("execution receipt binding is stale")
                receipt_errors = validate_execution_receipt(
                    root,
                    receipt,
                    candidate_id=str(candidate_id),
                    world_seed=int(world_seed),
                )
                if receipt_errors:
                    raise ValueError("; ".join(receipt_errors))
                cached = {key: value for key, value in embedded.items() if key != "receipt"}
                if cached != receipt:
                    raise ValueError("embedded row differs from bound execution receipt")
                validated_rows.append(receipt)
                if evidence_progress is not None and (
                    receipt_index == 1 or receipt_index % 32 == 0 or receipt_index == len(rows)
                ):
                    evidence_progress(receipt_index, len(rows))
            rebuilt = analyze_candidate_world(
                str(candidate_id), int(world_seed), validated_rows, audit
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"A-S world analysis cannot be rebuilt: {error}")
        else:
            if report.get("analysis") != rebuilt:
                errors.append("A-S world analysis mismatch")
    return errors


def validate_summary(
    root: Path,
    summary: Mapping[str, Any],
    *,
    expected_execution_context: WorkIIExecutionContext | None = None,
    evidence_progress: Callable[[str, int, int], None] | None = None,
    deep_validate_world_reports: bool = True,
) -> list[str]:
    errors: list[str] = []
    envelope = summary.get("execution_context")
    if not isinstance(envelope, Mapping):
        errors.append("A-S five-world summary lacks an execution context")
    else:
        errors.extend(
            validate_execution_envelope(root, envelope, expected_context=expected_execution_context)
        )
    if summary.get("schema_version") != SUMMARY_VERSION:
        errors.append("unexpected A-S five-world summary schema")
    if summary.get("qualification_schema_version") != QUALIFICATION_VERSION:
        errors.append("A-S five-world qualification schema mismatch")
    if summary.get("summary_sha256") != summary_sha256(summary):
        errors.append("A-S five-world summary self-hash mismatch")
    if summary.get("formal_result") is not False:
        errors.append("A-S five-world summary must not be formal")
    if summary.get("provider_call_count") != 0 or summary.get("participant_session_count") != 0:
        errors.append("A-S five-world summary must remain provider-free")
    expected_coverage = {
        "candidate_count": 2,
        "worlds_per_candidate": 5,
        "coordinates_per_candidate_world": COORDINATES_PER_CANDIDATE_WORLD,
        "laws_per_coordinate": LAWS_PER_COORDINATE,
        "primary_executions_per_candidate_world": PRIMARY_EXECUTIONS_PER_CANDIDATE_WORLD,
        "planned_primary_execution_count": PRIMARY_EXECUTIONS_TOTAL,
        "planned_exact_replay_count": EXACT_REPLAYS_TOTAL,
        "q2_queries_per_candidate_world": Q2_QUERY_COUNT_PER_CANDIDATE,
    }
    if summary.get("coverage") != expected_coverage:
        errors.append("A-S five-world coverage mismatch")
    generated_package = summary.get("generated_package")
    package: dict[str, Any] | None = None
    if not isinstance(generated_package, Mapping):
        errors.append("A-S summary lacks its generated Q2 package binding")
    else:
        try:
            package_path = (root / str(generated_package["path"])).resolve()
            package_path.relative_to(root.resolve())
            if not package_path.is_file():
                raise ValueError("Q2 package is missing")
            if generated_package.get("sha256") != file_sha256(package_path):
                errors.append("A-S generated Q2 package file hash mismatch")
            loaded_package = json.loads(package_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_package, dict):
                raise ValueError("Q2 package is not an object")
            package = loaded_package
            if package.get("package_sha256") != package_sha256(package):
                errors.append("A-S generated Q2 package self-hash mismatch")
            if generated_package.get("package_sha256") != package.get("package_sha256"):
                errors.append("A-S embedded Q2 package hash mismatch")
            if package.get("execution_context") != summary.get("execution_context"):
                errors.append("A-S Q2 package/execution context mismatch")
            if package.get("formal_result") is not False or package.get("provider_call_count") != 0:
                errors.append("A-S Q2 package crossed its provider-free boundary")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"A-S generated Q2 package cannot be read: {error}")
    raw_bindings = summary.get("raw_bindings")
    validated_reports: list[dict[str, Any]] = []
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 10:
        errors.append("A-S summary must bind all ten candidate-world reports")
    else:
        seen: set[tuple[str, int]] = set()
        for binding in raw_bindings:
            try:
                path = (root / str(binding["path"])).resolve()
                path.relative_to(root.resolve())
                if not path.is_file():
                    raise ValueError("world report is missing")
                if binding.get("sha256") != file_sha256(path):
                    errors.append("A-S raw world-report file hash mismatch")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("world report is not an object")
                label = f"{payload.get('candidate_id')}:world-{payload.get('world_seed')}"
                if deep_validate_world_reports:
                    errors.extend(
                        validate_world_report(
                            payload,
                            root=root,
                            expected_execution_context=expected_execution_context,
                            evidence_progress=(
                                (
                                    lambda completed, total, label=label: evidence_progress(
                                        label, completed, total
                                    )
                                )
                                if evidence_progress is not None
                                else None
                            ),
                        )
                    )
                validated_reports.append(payload)
                if payload.get("execution_context") != summary.get("execution_context"):
                    errors.append("A-S raw/execution context mismatch")
                if binding.get("report_sha256") != payload.get("report_sha256"):
                    errors.append("A-S raw embedded world-report hash mismatch")
                seen.add((str(payload["candidate_id"]), int(payload["world_seed"])))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"A-S raw binding cannot be read: {error}")
        expected = {(candidate, world) for candidate in CANDIDATE_IDS for world in WORLD_SEEDS}
        if seen != expected:
            errors.append("A-S raw report roster mismatch")
    if validated_reports:
        plan_bindings = {
            canonical_json_sha256(report.get("plan_binding")) for report in validated_reports
        }
        if len(plan_bindings) != 1:
            errors.append("A-S world reports do not share one qualification plan")
        expected_denominators = {
            "planned_primary_executions": PRIMARY_EXECUTIONS_TOTAL,
            "attempted_primary_executions": sum(
                len(report["rows"]) for report in validated_reports
            ),
            "completed_primary_executions": sum(
                report["analysis"]["denominators"]["completed_primary_executions"]
                for report in validated_reports
            ),
            "planned_exact_replays": EXACT_REPLAYS_TOTAL,
            "completed_exact_replays": sum(
                report["analysis"]["denominators"]["exact_replays"] for report in validated_reports
            ),
            "physical_failures": sum(
                report["analysis"]["denominators"]["physical_failures"]
                for report in validated_reports
            ),
            "platform_failures": sum(
                report["analysis"]["denominators"]["platform_failures"]
                for report in validated_reports
            ),
            "unsafe_completed": sum(
                report["analysis"]["denominators"]["unsafe_completed"]
                for report in validated_reports
            ),
        }
        if summary.get("denominators") != expected_denominators:
            errors.append("A-S summary denominators differ from validated world reports")
        expected_passed = len(validated_reports) == 10 and all(
            report["analysis"]["passed"] is True for report in validated_reports
        )
        if summary.get("all_candidates_passed") is not expected_passed:
            errors.append("A-S summary pass decision differs from validated world reports")
        if isinstance(generated_package, Mapping) and package is not None:
            expected_package = build_q2_package(
                validated_reports,
                q0_bindings=summary.get("q0_bindings", {}),
                execution_context=summary.get("execution_context", {}),
                plan_binding=validated_reports[0].get("plan_binding", {}),
            )
            if package != expected_package:
                errors.append("A-S Q2 package differs from validated world reports")
        plan_binding = validated_reports[0].get("plan_binding")
        if isinstance(plan_binding, Mapping):
            try:
                plan_path = (root / str(plan_binding["path"])).resolve()
                plan_path.relative_to(root.resolve())
                bound_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if not isinstance(bound_plan, Mapping) or summary.get(
                    "q0_bindings"
                ) != bound_plan.get("q0_bindings"):
                    errors.append("A-S summary Q0 bindings differ from qualification plan")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"A-S summary plan cannot be read: {error}")
        expected_candidates = {
            candidate_id: {
                "task_id": candidate_specs()[candidate_id]["task_id"],
                "passed_world_count": sum(
                    report["analysis"]["passed"]
                    for report in validated_reports
                    if report["candidate_id"] == candidate_id
                ),
                "qualification_passed": all(
                    report["analysis"]["passed"]
                    for report in validated_reports
                    if report["candidate_id"] == candidate_id
                ),
                "worlds": [
                    {
                        "world_seed": report["world_seed"],
                        "passed": report["analysis"]["passed"],
                        "failures": report["analysis"]["failures"],
                        "q1": report["analysis"]["q1"],
                        "q2": report["analysis"]["q2"],
                    }
                    for report in validated_reports
                    if report["candidate_id"] == candidate_id
                ],
            }
            for candidate_id in CANDIDATE_IDS
        }
        if summary.get("candidates") != expected_candidates:
            errors.append("A-S summary candidates differ from validated world reports")
        expected_decision = (
            "generate_locked_d1_readiness_for_both_candidates"
            if expected_passed
            else "retain_full_five_world_qualification_and_do_not_generate_d1"
        )
        if summary.get("decision") != expected_decision:
            errors.append("A-S summary decision differs from validated world reports")
    denominators_value = summary.get("denominators")
    if not isinstance(denominators_value, Mapping):
        errors.append("A-S summary lacks denominators")
    else:
        if denominators_value.get("planned_primary_executions") != PRIMARY_EXECUTIONS_TOTAL:
            errors.append("A-S primary denominator mismatch")
        if denominators_value.get("planned_exact_replays") != EXACT_REPLAYS_TOTAL:
            errors.append("A-S replay denominator mismatch")
    passed = summary.get("all_candidates_passed") is True
    generated_d1 = summary.get("participant_d1_configs_generated")
    if generated_d1 not in ({}, None) and not passed:
        errors.append("A-S summary generated D1 configs without complete qualification")
    if passed and (
        not isinstance(generated_d1, Mapping) or set(generated_d1) != set(CANDIDATE_IDS)
    ):
        errors.append("A-S complete qualification lacks both D1 configurations")
    elif isinstance(generated_d1, Mapping):
        for candidate_id, binding in generated_d1.items():
            if candidate_id not in CANDIDATE_IDS or not isinstance(binding, Mapping):
                errors.append("A-S generated D1 roster is malformed")
                continue
            try:
                d1_path = (root / str(binding["path"])).resolve()
                d1_path.relative_to(root.resolve())
                if not d1_path.is_file():
                    raise ValueError("D1 configuration is missing")
                if binding.get("sha256") != file_sha256(d1_path):
                    errors.append(f"A-S generated D1 file hash mismatch: {candidate_id}")
                d1 = json.loads(d1_path.read_text(encoding="utf-8"))
                if not isinstance(d1, dict):
                    raise ValueError("D1 configuration is not an object")
                if d1.get("execution_context") != summary.get("execution_context"):
                    errors.append(f"A-S D1/execution context mismatch: {candidate_id}")
                qualification = d1.get("qualification")
                intervention = d1.get("intervention")
                if (
                    not isinstance(qualification, Mapping)
                    or not isinstance(intervention, Mapping)
                    or qualification.get("execution_authorized") is not False
                    or qualification.get("formal_r5_authorized") is not False
                    or binding.get("execution_authorized") is not False
                    or package is None
                    or qualification.get("q2_package_sha256") != package.get("package_sha256")
                    or qualification.get("plan_sha256")
                    != package.get("plan_binding", {}).get("plan_sha256")
                    or intervention.get("q2_package_sha256") != package.get("package_sha256")
                    or intervention.get("candidate_id") != candidate_id
                    or intervention.get("registered_truth_law_id")
                    != candidate_specs()[candidate_id]["altered_law_id"]
                ):
                    errors.append(f"A-S D1 was prematurely authorized: {candidate_id}")
                else:
                    expected_queries = [
                        {
                            "query_id": row["coordinate_id"],
                            "feature_values": row["feature_values"],
                            "metric_ids": list(candidate_specs()[candidate_id]["metric_ids"]),
                            "intervention_family": row["intervention_family"],
                            "q2_coordinate_sha256": row["coordinate_sha256"],
                        }
                        for row in selected_q2_queries(candidate_id)
                    ]
                    checkpoint = d1.get("belief_checkpoint")
                    if (
                        not isinstance(checkpoint, Mapping)
                        or checkpoint.get("held_out_queries") != expected_queries
                    ):
                        errors.append(f"A-S D1 Q2 query binding mismatch: {candidate_id}")
                    try:
                        for arm in ("opaque", "aligned_nominal", "misindexed_nominal"):
                            build_checkpoint_contract(d1, arm)
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(f"A-S D1 contract is not runnable: {candidate_id}: {error}")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"A-S generated D1 cannot be read: {candidate_id}: {error}")
    if summary.get("provider_execution_authorized") is not False:
        errors.append("A-S qualification cannot authorize provider execution")
    if summary.get("formal_r5_authorized") is not False:
        errors.append("A-S qualification cannot authorize formal R5")
    return errors


__all__ = [
    "CANDIDATE_IDS",
    "COORDINATES_PER_CANDIDATE_WORLD",
    "CRYSTALLIZATION_CANDIDATE_ID",
    "EXACT_REPLAYS_TOTAL",
    "PACKAGE_VERSION",
    "PARTITION_CANDIDATE_ID",
    "PARTITION_NOMINAL_PAIR_STRATA",
    "PRIMARY_EXECUTIONS_PER_CANDIDATE_WORLD",
    "PRIMARY_EXECUTIONS_TOTAL",
    "Q2_QUERY_COUNT_PER_CANDIDATE",
    "QUALIFICATION_VERSION",
    "SUMMARY_VERSION",
    "WORLD_REPORT_VERSION",
    "WORLD_SEEDS",
    "analyze_candidate_world",
    "apply_crystallization_recovery_resource_design",
    "build_prior_arms",
    "candidate_specs",
    "crystallization_intervention",
    "crystallization_recovery_resource_design",
    "denominators",
    "effect_gate",
    "observation_binding",
    "partition_intervention",
    "registered_coordinates",
    "report_sha256",
    "selected_q2_queries",
    "summary_sha256",
    "validate_summary",
    "validate_world_report",
]
