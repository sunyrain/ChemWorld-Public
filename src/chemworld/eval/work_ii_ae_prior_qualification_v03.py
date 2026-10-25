"""Development-only A-E v0.3 locus qualification.

The implementation deliberately separates physical execution, classifier fitting,
untouched classifier validation, prospective screening, and truth-only scoring.  It
has no provider or release mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath
from statistics import fmean, variance
from typing import Any

import numpy as np
from scipy.stats import beta

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.task_recipes import task_recipe_dimension, task_recipe_from_unit_vector
from chemworld.data.logging import load_jsonl
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.materials import static_material_information_dossier
from chemworld.tasks import get_task

CONTRACT_VERSION = "chemworld-work-ii-ae-locus-qualification-contract-0.3"
PLAN_VERSION = "chemworld-work-ii-ae-locus-plan-0.3"
REPORT_VERSION = "chemworld-work-ii-ae-locus-report-0.3"
SELECTION_VERSION = "chemworld-work-ii-ae-locus-selection-0.3"
SUMMARY_VERSION = "chemworld-work-ii-ae-locus-summary-0.3"
RECEIPT_VERSION = "chemworld-work-ii-ae-locus-receipt-0.3"
MODEL_VERSION = "chemworld-work-ii-ae-calibrated-residual-model-0.3"

EXPECTED_TASKS = (
    "electrochemical-conversion",
    "reaction-to-crystallization",
    "reaction-to-distillation",
    "partition-discovery",
    "reaction-safety-constrained",
)
PHASES = (
    "classifier_fit",
    "classifier_validation",
    "prospective_screen",
    "confirmation",
)
COHORT_INTERVALS = {
    "classifier_fit": (1_000_000_000, 1_199_999_999),
    "classifier_validation": (1_200_000_000, 1_399_999_999),
    "prospective_screen": (1_400_000_000, 1_599_999_999),
    "confirmation": (1_600_000_000, 1_799_999_999),
    "future_formal_reserved": (1_800_000_000, 1_999_999_999),
}
HYPOTHESES = (
    "H0",
    "swap-0-1",
    "swap-0-2",
    "swap-0-3",
    "swap-1-2",
    "swap-1-3",
    "swap-2-3",
)
TRANSPOSITIONS = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
HELMERT = np.asarray(
    (
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0, 0.0),
        (1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0), 0.0),
        (
            1.0 / math.sqrt(12.0),
            1.0 / math.sqrt(12.0),
            1.0 / math.sqrt(12.0),
            -3.0 / math.sqrt(12.0),
        ),
    ),
    dtype=float,
)
FORBIDDEN_CLASSIFIER_KEYS = frozenset(
    {
        "descriptor_permutation",
        "selected_pair",
        "private_scoring_pair",
        "target_pair",
        "true_pair",
        "truth_hypothesis",
        "world_parameters",
        "world_seed",
        "hidden_outcomes",
        "candidate_name",
        "observation_sigma",
    }
)
CLASSIFIER_INPUT_KEYS = frozenset(
    {
        "dossier",
        "task_id",
        "locus_id",
        "target_field",
        "anchor_ids",
        "anchor_recipes",
        "registered_observations",
        "calibration_model",
    }
)
EXPECTED_LOCUS_SEMANTICS = (
    (
        "electrochemical-conversion",
        "ae-locus-electrolyte-profile",
        1,
        "electrolyte_profile",
        "final-assay",
        "transport_efficiency",
        ("ohmic_efficiency",),
        ("faradaic_efficiency", "energy_efficiency"),
        (
            "acid_concentration_mol_L",
            "acid_pKa",
            "bulk_conductivity_S_m",
            "diffusion_layer_thickness_mm",
            "diffusivity_m2_s",
            "double_layer_capacitance_F_m2",
            "precipitating_salt_concentration_mol_L",
            "precipitation_log10_Ksp",
            "standard_potential_shift_V",
            "supporting_electrolyte_concentration_mol_L",
        ),
    ),
    (
        "electrochemical-conversion",
        "ae-locus-electrochemical-solvent",
        2,
        "solvent",
        "final-assay",
        "energy_efficiency",
        ("faradaic_efficiency", "electrochemical_selectivity"),
        ("transport_efficiency", "ohmic_efficiency"),
        (
            "relative_conductivity",
            "relative_cost_index",
            "relative_diffusivity",
            "relative_double_layer_capacitance",
            "relative_proton_activity",
            "relative_solubility_product",
            "standard_potential_shift_V",
        ),
    ),
    (
        "reaction-to-crystallization",
        "ae-locus-crystallization-catalyst",
        1,
        "catalyst",
        "reaction-post-quench-hplc",
        "conversion",
        ("yield", "selectivity"),
        ("byproduct_signal", "safety_risk"),
        (
            "reference_panel_activity_ceiling",
            "reference_panel_activity_floor",
            "reference_panel_activity_geomean",
            "reference_panel_log_variability",
        ),
    ),
    (
        "reaction-to-crystallization",
        "ae-locus-crystallization-solvent",
        2,
        "solvent",
        "final-assay",
        "crystal_yield",
        ("crystal_purity", "crystal_csd_quality"),
        ("yield", "selectivity"),
        (
            "reference_panel_activity_ceiling",
            "reference_panel_activity_floor",
            "reference_panel_activity_geomean",
            "reference_panel_log_variability",
            "relative_crystal_growth",
            "relative_impurity_occlusion",
            "relative_nucleation_tendency",
            "relative_solubility",
        ),
    ),
    (
        "reaction-to-distillation",
        "ae-locus-distillation-catalyst",
        1,
        "catalyst",
        "reaction-post-quench-hplc",
        "conversion",
        ("yield", "selectivity"),
        ("byproduct_signal", "safety_risk"),
        (
            "reference_panel_activity_ceiling",
            "reference_panel_activity_floor",
            "reference_panel_activity_geomean",
            "reference_panel_log_variability",
        ),
    ),
    (
        "reaction-to-distillation",
        "ae-locus-distillation-solvent",
        2,
        "solvent",
        "reaction-post-quench-hplc",
        "conversion",
        ("yield", "selectivity"),
        ("byproduct_signal", "safety_risk"),
        (
            "reference_panel_activity_ceiling",
            "reference_panel_activity_floor",
            "reference_panel_activity_geomean",
            "reference_panel_log_variability",
        ),
    ),
    (
        "partition-discovery",
        "ae-locus-partition-extractant",
        1,
        "extractant",
        "partition-post-settle-pre-separation-hplc",
        "product_in_organic",
        ("product_in_aqueous", "phase_ratio"),
        ("purity", "recovery"),
        (
            "partner_panel_impurity_distribution_geomean",
            "partner_panel_product_distribution_geomean",
            "partner_panel_selectivity_ceiling",
            "partner_panel_selectivity_geomean",
            "partner_panel_selectivity_log_variability",
        ),
    ),
    (
        "partition-discovery",
        "ae-locus-partition-solvent",
        2,
        "solvent",
        "partition-post-settle-pre-separation-hplc",
        "product_in_organic",
        ("product_in_aqueous", "phase_ratio"),
        ("purity", "recovery"),
        (
            "partner_panel_impurity_distribution_geomean",
            "partner_panel_product_distribution_geomean",
            "partner_panel_selectivity_ceiling",
            "partner_panel_selectivity_geomean",
            "partner_panel_selectivity_log_variability",
        ),
    ),
    (
        "reaction-safety-constrained",
        "ae-locus-safety-catalyst",
        1,
        "catalyst",
        "reaction-post-quench-hplc",
        "selectivity",
        ("yield", "conversion", "byproduct_signal"),
        ("safety_risk",),
        (
            "reference_panel_activity_ceiling",
            "reference_panel_activity_floor",
            "reference_panel_activity_geomean",
            "reference_panel_log_variability",
        ),
    ),
    (
        "reaction-safety-constrained",
        "ae-locus-safety-solvent",
        2,
        "solvent",
        "reaction-post-quench-hplc",
        "selectivity",
        ("yield", "conversion", "byproduct_signal"),
        ("safety_risk",),
        (
            "reference_panel_activity_ceiling",
            "reference_panel_activity_floor",
            "reference_panel_activity_geomean",
            "reference_panel_log_variability",
        ),
    ),
)


class AEPriorQualificationV03Error(ValueError):
    """Raised when the development qualification contract is violated."""


class _FrozenRecipeAgent(BaseAgent):
    name = "work_ii_ae_locus_v03_frozen_recipe"

    def __init__(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._actions = [deepcopy(dict(action)) for action in actions]

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._index = 0

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if self._index >= len(self._actions):
            raise RuntimeError("frozen v0.3 recipe exhausted")
        action = deepcopy(self._actions[self._index])
        self._index += 1
        return action

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        del action, observation, reward, info


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AEPriorQualificationV03Error(f"{path} must contain one JSON object")
    return value


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return 1_000_000_000 + int.from_bytes(digest[:8], "big") % 900_000_000


def _execution_noise_seed(
    namespace: str, coordinate: Sequence[object], occupied: set[int]
) -> int:
    """Derive a coordinate-stable noise seed with deterministic collision repair."""

    for counter in range(1024):
        seed = _stable_seed("observation", namespace, *coordinate, counter)
        if seed not in occupied:
            return seed
    raise AEPriorQualificationV03Error("could not derive a unique observation seed")


def _cohort_seed(
    phase: str,
    namespace: str,
    locus_id: str,
    index: int,
    occupied: set[int] | None = None,
) -> int:
    """Derive a phase-disjoint seed; collision resolution is deterministic."""

    low, high = COHORT_INTERVALS[phase]
    width = high - low + 1
    for counter in range(1024):
        digest = hashlib.sha256(
            f"world:{phase}:{namespace}:{locus_id}:{index}:{counter}".encode()
        ).digest()
        seed = low + int.from_bytes(digest[:8], "big") % width
        # A counter is kept in the derivation so future registries can fail closed and
        # deterministically re-draw collisions without changing other coordinates.
        if low <= seed <= high and seed not in (occupied or set()):
            return seed
    raise AEPriorQualificationV03Error("could not derive a cohort seed")


def _cohort_seed_map(
    phase: str, namespace: str, locus_ids: Sequence[str], count: int
) -> dict[tuple[str, int], int]:
    output: dict[tuple[str, int], int] = {}
    occupied: set[int] = set()
    for locus_id in locus_ids:
        for index in range(count):
            seed = _cohort_seed(phase, namespace, locus_id, index, occupied)
            output[(locus_id, index)] = seed
            occupied.add(seed)
    return output


def _target_coordinate(task_id: str, target_field: str) -> int:
    fields = {
        "electrochemical-conversion": {"electrolyte_profile": 0, "solvent": 1},
        "reaction-to-crystallization": {"catalyst": 4, "solvent": 6},
        "reaction-to-distillation": {"catalyst": 4, "solvent": 6},
        "partition-discovery": {"solvent": 0, "extractant": 3},
        "reaction-safety-constrained": {"catalyst": 4, "solvent": 6},
    }
    try:
        return fields[task_id][target_field]
    except KeyError as error:
        raise AEPriorQualificationV03Error(
            f"unsupported locus: {task_id}.{target_field}"
        ) from error


def _material_family_id(task_id: str, config: Mapping[str, Any]) -> object | None:
    if task_id == "electrochemical-conversion":
        return config.get("electrochemical_material_family_id")
    if task_id == "reaction-to-crystallization":
        return config.get("crystallization_material_family_id")
    return None


def _aligned_dossier(
    task_id: str, target_field: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    full = static_material_information_dossier(
        {"mode": "anonymous_nominal_properties"},
        task_id=task_id,
        material_family_id=_material_family_id(task_id, config),
    )
    if not isinstance(full, Mapping):
        raise AEPriorQualificationV03Error("aligned dossier is unavailable")
    choices = full.get("choices")
    rows = choices.get(target_field) if isinstance(choices, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 4:
        raise AEPriorQualificationV03Error(
            f"aligned dossier lacks four {task_id}.{target_field} choices"
        )
    return deepcopy(dict(full))


def _descriptor_matrix(
    dossier: Mapping[str, Any], target_field: str, whitelist: Sequence[str]
) -> np.ndarray:
    choices = dossier.get("choices")
    rows = choices.get(target_field) if isinstance(choices, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 4:
        raise AEPriorQualificationV03Error("classifier dossier lacks four target choices")
    if len(set(whitelist)) != len(whitelist) or not whitelist:
        raise AEPriorQualificationV03Error("descriptor whitelist is empty or duplicated")
    matrix: list[list[float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("action_value") != index:
            raise AEPriorQualificationV03Error("dossier category order is invalid")
        properties = row.get("nominal_properties")
        if not isinstance(properties, Mapping) or set(properties) != set(whitelist):
            raise AEPriorQualificationV03Error(
                "dossier properties differ from the frozen descriptor whitelist"
            )
        values = [properties.get(key) for key in whitelist]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise AEPriorQualificationV03Error("dossier descriptor is non-finite")
        matrix.append([float(value) for value in values])
    return np.asarray(matrix, dtype=float)


def _redacted_dossier(
    aligned: Mapping[str, Any], target_field: str, whitelist: Sequence[str]
) -> dict[str, Any]:
    matrix = _descriptor_matrix(aligned, target_field, whitelist)
    rows = aligned["choices"][target_field]
    return {
        "choices": {
            target_field: [
                {
                    "action_value": index,
                    "anonymous_material_id": str(rows[index]["anonymous_material_id"]),
                    "nominal_properties": {
                        key: float(matrix[index, column])
                        for column, key in enumerate(whitelist)
                    },
                }
                for index in range(4)
            ]
        }
    }


def hypothesis_permutation(hypothesis: str) -> tuple[int, int, int, int]:
    if hypothesis == "H0":
        return (0, 1, 2, 3)
    if hypothesis not in HYPOTHESES:
        raise AEPriorQualificationV03Error(f"unknown hypothesis {hypothesis}")
    _, left, right = hypothesis.split("-")
    permutation = [0, 1, 2, 3]
    i, j = int(left), int(right)
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return tuple(permutation)


def dossier_variant(
    aligned: Mapping[str, Any],
    target_field: str,
    whitelist: Sequence[str],
    hypothesis: str,
) -> dict[str, Any]:
    """Construct one visible dossier; truth is supplied only by the caller/scorer."""

    redacted = _redacted_dossier(aligned, target_field, whitelist)
    original = redacted["choices"][target_field]
    permutation = hypothesis_permutation(hypothesis)
    varied = deepcopy(redacted)
    for destination, source in enumerate(permutation):
        varied["choices"][target_field][destination]["nominal_properties"] = deepcopy(
            original[source]["nominal_properties"]
        )
    return varied


def select_descriptor_pair(
    dossier: Mapping[str, Any], target_field: str, whitelist: Sequence[str]
) -> tuple[int, int]:
    """Select the physical gate pair before outcomes by a frozen descriptor rule."""

    matrix = _descriptor_matrix(dossier, target_field, whitelist)
    scale = matrix.std(axis=0, ddof=0)
    keep = scale > 0.0
    if not bool(np.any(keep)):
        raise AEPriorQualificationV03Error("descriptor geometry has no varying column")
    standardized = (matrix[:, keep] - matrix[:, keep].mean(axis=0)) / scale[keep]
    ranked = sorted(
        (
            (
                -round(
                    float(np.linalg.norm(standardized[left] - standardized[right])), 12
                ),
                left,
                right,
            )
            for left, right in TRANSPOSITIONS
        )
    )
    return ranked[0][1], ranked[0][2]


def _known_earlier_seeds(root: Path) -> set[int]:
    seeds = {0, 1, 2, 3, 4}
    v01 = _load_object(root / "configs/benchmark/work_ii_formal_design_v0.1.json")
    for values in v01["world_cohort"]["public_formal"]["task_world_seeds"].values():
        seeds.update(int(value) for value in values)
    v02 = _load_object(
        root / "configs/benchmark/work_ii_ae_prior_distinguishability_v0.2.json"
    )
    for cohort in v02["cohorts"].values():
        for values in cohort["task_world_seeds"].values():
            seeds.update(int(value) for value in values)
    return seeds


def _candidate_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [deepcopy(candidate) for task in contract["tasks"] for candidate in task["loci"]]


def _candidate_index(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for task in contract["tasks"]:
        for candidate in task["loci"]:
            row = deepcopy(candidate)
            row["task_id"] = task["task_id"]
            row["campaign_config"] = task["campaign_config"]
            output[row["locus_id"]] = row
    return output


def validate_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    scalar = {
        "schema_version": CONTRACT_VERSION,
        "contract_id": "work-ii-ae-v0.3-locus-development",
        "status": "design_only_not_executed",
        "development_only": True,
        "participant_provider_calls": 0,
        "participant_outcomes_read": False,
    }
    for key, expected in scalar.items():
        if contract.get(key) != expected:
            errors.append(f"invalid scalar contract field {key}")
    expected_top = {
        *scalar,
        "experiment_note",
        "coverage",
        "classifier",
        "thresholds",
        "selection_rule",
        "noise",
        "cohorts",
        "denominators",
        "tasks",
    }
    if set(contract) != expected_top:
        errors.append("contract top-level fields are not exact")
    note = contract.get("experiment_note")
    if not isinstance(note, str) or not (root / note).is_file():
        errors.append("experiment note is missing")
    coverage = contract.get("coverage")
    expected_coverage = {
        "anchors_per_locus_world": 2,
        "categories_per_anchor": 4,
        "independent_replicates_per_anchor_category": 3,
        "fit_worlds_per_locus": 60,
        "validation_worlds_per_locus": 60,
        "screen_worlds_per_locus": 5,
        "confirmation_worlds_per_selected_task": 5,
        "loci_per_task": 2,
        "anchor_algorithm": "task-locus-complement-v1",
        "anchor_namespace": "work-ii-ae-v0.3-anchors-20260813",
        "anchor_lower_bound": 0.15,
        "anchor_upper_bound": 0.85,
    }
    if coverage != expected_coverage:
        errors.append("coverage contract changed")
    classifier = contract.get("classifier")
    expected_classifier = {
        "classifier_id": "signed-ridge-calibrated-weighted-residual-v0.3",
        "allowed_inputs": sorted(CLASSIFIER_INPUT_KEYS),
        "forbidden_inputs": sorted(FORBIDDEN_CLASSIFIER_KEYS),
        "hypotheses": list(HYPOTHESES),
        "contrast": "orthonormal-helmert-3x4",
        "ridge_alpha_grid": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "hyperparameter_selection": "nested-world-blocked-loo-on-fit-only",
        "covariance": "fit-only-diagonal-shrinkage-plus-replicate-variance",
        "target_nuisance": "anchor-metric-intercept-eliminated-by-helmert",
        "score_name": "calibrated_weighted_residual_score",
        "decision_states": ["H0", "swap", "abstain"],
        "threshold_source": "fit-only-crossfit-world-clusters",
        "model_scope": "four-category-reference-response-fingerprint-not-transfer-physics",
    }
    if classifier != expected_classifier:
        errors.append("classifier semantic contract changed")
    thresholds = contract.get("thresholds")
    expected_thresholds = {
        "minimum_absolute_gate_separation": 0.05,
        "minimum_gate_signal_to_noise_ratio": 2.0,
        "validation_any_definite_wrong_max": 0,
        "validation_any_definite_wrong_worlds": 60,
        "validation_any_definite_wrong_cp95_upper": 0.0487029133101,
        "validation_per_class_correct_min": 56,
        "validation_per_class_denominator": 60,
        "validation_per_class_bonferroni_cp_lower_min": 0.80,
        "validation_all_seven_correct_worlds_min": 54,
        "validation_all_seven_correct_denominator": 60,
        "validation_all_seven_cp95_lower_min": 0.80,
        "screen_all_seven_correct_worlds_required": 5,
        "screen_physical_gate_worlds_required": 5,
        "confirmation_all_seven_correct_worlds_required": 5,
        "confirmation_physical_gate_worlds_required": 5,
        "exact_replay_required": True,
    }
    if thresholds != expected_thresholds:
        errors.append("scientific or statistical thresholds changed")
    if contract.get("selection_rule") != {
        "pair": "max-zscored-descriptor-euclidean-then-pair-lexicographic-before-data",
        "locus_order": ["scientific_priority_ascending", "locus_id_lexicographic"],
        "selected_per_task_max": 1,
        "effect_size_used_for_locus_ranking": False,
        "claim_if_some_tasks_pass": "qualified-locus",
        "universal_claim_requires_tasks": 5,
    }:
        errors.append("selection rule changed")
    if contract.get("noise") != {
        "mode": "keyed",
        "seed_namespace": "work-ii-ae-v0.3-independent-observation-20260813",
        "distinct_execution_coordinates_have_independent_streams": True,
        "target_data_may_not_refit_scale_or_covariance": True,
    }:
        errors.append("noise contract changed")
    expected_denominators = {
        "tasks": 5,
        "loci": 10,
        "classifier_fit_primary": 14400,
        "classifier_fit_exact_replay": 14400,
        "classifier_validation_primary": 14400,
        "classifier_validation_exact_replay": 14400,
        "prospective_screen_primary": 1200,
        "prospective_screen_exact_replay": 1200,
        "confirmation_primary_per_selected_task": 120,
        "confirmation_exact_replay_per_selected_task": 120,
        "confirmation_selected_task_range": [0, 5],
        "confirmation_primary_range": [0, 600],
    }
    if contract.get("denominators") != expected_denominators:
        errors.append("denominator contract changed")
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or [row.get("task_id") for row in tasks] != list(
        EXPECTED_TASKS
    ):
        errors.append("exact five-task roster is required")
        tasks = []
    locus_ids: set[str] = set()
    observed_semantics: list[tuple[Any, ...]] = []
    for task in tasks:
        try:
            task_id = str(task["task_id"])
            if set(task) != {"task_id", "campaign_config", "loci"}:
                raise AEPriorQualificationV03Error("task fields are not exact")
            config_path = (root / str(task["campaign_config"])).resolve()
            if not config_path.is_relative_to(root.resolve()) or not config_path.is_file():
                raise AEPriorQualificationV03Error("campaign config missing or outside root")
            config = _load_object(config_path)
            if config.get("task_id") != task_id:
                raise AEPriorQualificationV03Error("campaign task mismatch")
            loci = task["loci"]
            if not isinstance(loci, list) or len(loci) != 2:
                raise AEPriorQualificationV03Error("task requires two loci")
            if {int(row["scientific_priority"]) for row in loci} != {1, 2}:
                raise AEPriorQualificationV03Error("locus priorities must be 1 and 2")
            for locus in loci:
                expected_fields = {
                    "locus_id",
                    "scientific_priority",
                    "target_field",
                    "measurement_stage_id",
                    "gate_endpoint_id",
                    "classifier_secondary_endpoint_ids",
                    "non_gating_secondary_metric_ids",
                    "descriptor_whitelist",
                }
                if set(locus) != expected_fields:
                    raise AEPriorQualificationV03Error("locus fields are not exact")
                locus_id = str(locus["locus_id"])
                if "swap" in locus_id or not locus_id.startswith("ae-locus-"):
                    raise AEPriorQualificationV03Error("locus ID is not neutral")
                if locus_id in locus_ids:
                    raise AEPriorQualificationV03Error("locus ID is duplicated")
                locus_ids.add(locus_id)
                target_field = str(locus["target_field"])
                _target_coordinate(task_id, target_field)
                aligned = _redacted_dossier(
                    _aligned_dossier(task_id, target_field, config),
                    target_field,
                    locus["descriptor_whitelist"],
                )
                select_descriptor_pair(aligned, target_field, locus["descriptor_whitelist"])
                classifier_metrics = [
                    str(locus["gate_endpoint_id"]),
                    *[str(x) for x in locus["classifier_secondary_endpoint_ids"]],
                ]
                secondary = [str(x) for x in locus["non_gating_secondary_metric_ids"]]
                if len(set(classifier_metrics + secondary)) != len(
                    classifier_metrics + secondary
                ):
                    raise AEPriorQualificationV03Error("metric roles overlap")
                observed_semantics.append(
                    (
                        task_id,
                        locus_id,
                        int(locus["scientific_priority"]),
                        target_field,
                        str(locus["measurement_stage_id"]),
                        str(locus["gate_endpoint_id"]),
                        tuple(str(x) for x in locus["classifier_secondary_endpoint_ids"]),
                        tuple(str(x) for x in locus["non_gating_secondary_metric_ids"]),
                        tuple(str(x) for x in locus["descriptor_whitelist"]),
                    )
                )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid locus design for {task.get('task_id')}: {error}")
    if tuple(observed_semantics) != EXPECTED_LOCUS_SEMANTICS:
        errors.append("exact task/locus/stage/endpoint/descriptor table changed")
    cohorts = contract.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {
        *PHASES,
        "future_formal_reserved",
    }:
        errors.append("cohort registry is malformed")
        cohorts = {}
    counts = {
        "classifier_fit": 60,
        "classifier_validation": 60,
        "prospective_screen": 5,
        "confirmation": 5,
        "future_formal_reserved": 5,
    }
    all_new: set[int] = set()
    earlier = _known_earlier_seeds(root)
    for phase, count in counts.items():
        row = cohorts.get(phase)
        if not isinstance(row, Mapping) or set(row) != {"namespace", "worlds_per_locus"}:
            errors.append(f"{phase} cohort fields are invalid")
            continue
        if row.get("worlds_per_locus") != count or not isinstance(row.get("namespace"), str):
            errors.append(f"{phase} cohort count/namespace changed")
            continue
        seeds = set(
            _cohort_seed_map(
                phase, str(row["namespace"]), sorted(locus_ids), count
            ).values()
        )
        if len(seeds) != len(locus_ids) * count:
            errors.append(f"{phase} seeds are not globally unique")
        if seeds & earlier or seeds & all_new:
            errors.append(f"{phase} seeds collide with an earlier or another cohort")
        all_new.update(seeds)
    return errors


def _anchors(
    task_id: str,
    locus_id: str,
    target_coordinate: int,
    dimension: int,
    coverage: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    lower = float(coverage["anchor_lower_bound"])
    upper = float(coverage["anchor_upper_bound"])
    namespace = str(coverage["anchor_namespace"])
    first = np.empty(dimension, dtype=float)
    for coordinate in range(dimension):
        digest = hashlib.sha256(
            f"{namespace}:{task_id}:{locus_id}:{coordinate}".encode()
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64)
        first[coordinate] = round(lower + (upper - lower) * unit, 9)
    second = np.asarray(
        [round(lower + upper - float(value), 9) for value in first], dtype=float
    )
    first[target_coordinate] = second[target_coordinate] = 0.5
    return first, second


def build_locus_schedule(
    *,
    task_id: str,
    locus_id: str,
    target_field: str,
    coverage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_coordinate = _target_coordinate(task_id, target_field)
    task_info = get_task(task_id).to_dict()
    anchors = _anchors(
        task_id,
        locus_id,
        target_coordinate,
        task_recipe_dimension(task_info),
        coverage,
    )
    output: list[dict[str, Any]] = []
    for anchor_id, vector in enumerate(anchors):
        for category in range(4):
            candidate = vector.copy()
            candidate[target_coordinate] = (category + 0.5) / 4.0
            output.append(
                {
                    "anchor_id": anchor_id,
                    "target_category": category,
                    "anchor_recipe": [float(x) for x in vector],
                    "recipe_id": f"{task_id}:{locus_id}:a{anchor_id}:c{category}",
                    "recipe": task_recipe_from_unit_vector(task_info, candidate),
                }
            )
    return output


def _phase_count(contract: Mapping[str, Any], phase: str) -> int:
    return int(contract["cohorts"][phase]["worlds_per_locus"])


def _build_plan(
    root: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    phase: str,
    selected: Mapping[str, str] | None = None,
    upstream: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AEPriorQualificationV03Error("unknown phase")
    index = _candidate_index(contract)
    if phase == "confirmation":
        selected = selected or {}
        locus_ids = [selected[task] for task in EXPECTED_TASKS if task in selected]
    else:
        locus_ids = list(index)
    executions: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    occupied_observation_seeds: set[int] = set()
    namespace = str(contract["cohorts"][phase]["namespace"])
    phase_seeds = _cohort_seed_map(
        phase, namespace, locus_ids, _phase_count(contract, phase)
    )
    for locus_id in locus_ids:
        locus = index[locus_id]
        task_id = str(locus["task_id"])
        config = _load_object(root / locus["campaign_config"])
        aligned = _redacted_dossier(
            _aligned_dossier(task_id, locus["target_field"], config),
            locus["target_field"],
            locus["descriptor_whitelist"],
        )
        pair = select_descriptor_pair(
            aligned, locus["target_field"], locus["descriptor_whitelist"]
        )
        schedule = build_locus_schedule(
            task_id=task_id,
            locus_id=locus_id,
            target_field=locus["target_field"],
            coverage=contract["coverage"],
        )
        classifier_metrics = [
            locus["gate_endpoint_id"], *locus["classifier_secondary_endpoint_ids"]
        ]
        measured_metrics = [
            *classifier_metrics, *locus["non_gating_secondary_metric_ids"]
        ]
        bindings.append(
            {
                "task_id": task_id,
                "locus_id": locus_id,
                "campaign_config": locus["campaign_config"],
                "campaign_config_sha256": canonical_json_sha256(config),
                "aligned_dossier_sha256": canonical_json_sha256(aligned),
                "private_scoring_pair": list(pair),
            }
        )
        for world_index in range(_phase_count(contract, phase)):
            world_seed = phase_seeds[(locus_id, world_index)]
            for replicate in range(3):
                for item in schedule:
                    coordinate = (
                        phase,
                        locus_id,
                        world_index,
                        item["anchor_id"],
                        item["target_category"],
                        replicate,
                    )
                    noise_namespace = str(contract["noise"]["seed_namespace"])
                    observation_seed = _execution_noise_seed(
                        noise_namespace, coordinate, occupied_observation_seeds
                    )
                    occupied_observation_seeds.add(observation_seed)
                    executions.append(
                        {
                            "execution_index": len(executions),
                            "execution_id": "v0.3:" + ":".join(map(str, coordinate)),
                            "phase": phase,
                            "task_id": task_id,
                            "locus_id": locus_id,
                            "scientific_priority": int(locus["scientific_priority"]),
                            "target_field": locus["target_field"],
                            "measurement_stage_id": locus["measurement_stage_id"],
                            "gate_endpoint_id": locus["gate_endpoint_id"],
                            "classification_metric_ids": classifier_metrics,
                            "non_gating_secondary_metric_ids": list(
                                locus["non_gating_secondary_metric_ids"]
                            ),
                            "descriptor_whitelist": list(locus["descriptor_whitelist"]),
                            "aligned_dossier": aligned,
                            "world_index": world_index,
                            "world_seed": world_seed,
                            "anchor_id": item["anchor_id"],
                            "target_category": item["target_category"],
                            "replicate": replicate,
                            "anchor_recipe": item["anchor_recipe"],
                            "observation_seed": observation_seed,
                            "observation_noise_namespace": (
                                f"{noise_namespace}:" + ":".join(map(str, coordinate))
                            ),
                            "recipe_id": item["recipe_id"],
                            "recipe": item["recipe"],
                            "measured_metric_ids": measured_metrics,
                        }
                    )
    exact_expected = len(locus_ids) * _phase_count(contract, phase) * 24
    if len(executions) != exact_expected:
        raise AEPriorQualificationV03Error("plan construction denominator mismatch")
    payload = {
        "schema_version": PLAN_VERSION,
        "development_only": True,
        "phase": phase,
        "contract_binding": {
            "path": contract_path.relative_to(root).as_posix(),
            "canonical_sha256": canonical_json_sha256(contract),
        },
        "participant_provider_calls": 0,
        "selected_locus_ids": dict(selected or {}),
        "upstream_bindings": dict(upstream or {}),
        "task_locus_bindings": bindings,
        "denominators": {
            "included_loci": len(locus_ids),
            "worlds": len(locus_ids) * _phase_count(contract, phase),
            "primary_executions": exact_expected,
            "exact_replays": exact_expected,
        },
        "executions": executions,
    }
    payload["plan_sha256"] = canonical_json_sha256(payload)
    return payload


def build_phase_plan(
    root: Path,
    contract_path: Path,
    phase: str,
    *,
    fit_report: Mapping[str, Any] | None = None,
    fit_plan: Mapping[str, Any] | None = None,
    fit_receipts: Sequence[Mapping[str, Any]] | None = None,
    validation_report: Mapping[str, Any] | None = None,
    validation_plan: Mapping[str, Any] | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] | None = None,
    selection: Mapping[str, Any] | None = None,
    screen_report: Mapping[str, Any] | None = None,
    screen_plan: Mapping[str, Any] | None = None,
    screen_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = _load_object(contract_path)
    errors = validate_contract(root, contract)
    if errors:
        raise AEPriorQualificationV03Error("invalid contract: " + "; ".join(errors))
    upstream: dict[str, str] = {}
    selected: dict[str, str] | None = None
    if phase == "classifier_validation":
        _require_upstream_chain(fit_report=fit_report)
        _require_phase_evidence(
            root=root,
            contract_path=contract_path,
            contract=contract,
            phase="classifier_fit",
            plan=fit_plan,
            receipts=fit_receipts,
            report=fit_report,
        )
        upstream["fit_report_sha256"] = str(fit_report["report_sha256"])
    elif phase == "prospective_screen":
        _require_upstream_chain(
            fit_report=fit_report, validation_report=validation_report
        )
        _require_phase_evidence(
            root=root,
            contract_path=contract_path,
            contract=contract,
            phase="classifier_fit",
            plan=fit_plan,
            receipts=fit_receipts,
            report=fit_report,
        )
        _require_phase_evidence(
            root=root,
            contract_path=contract_path,
            contract=contract,
            phase="classifier_validation",
            plan=validation_plan,
            receipts=validation_receipts,
            report=validation_report,
            fit_report=fit_report,
        )
        if validation_report.get("status") != "passed":
            raise AEPriorQualificationV03Error(
                "prospective screen requires all ten untouched validations to pass"
            )
        upstream = {
            "fit_report_sha256": str(fit_report["report_sha256"]),
            "validation_report_sha256": str(validation_report["report_sha256"]),
        }
    elif phase == "confirmation":
        _require_upstream_chain(
            fit_report=fit_report,
            validation_report=validation_report,
            screen_report=screen_report,
        )
        _require_phase_evidence(
            root=root,
            contract_path=contract_path,
            contract=contract,
            phase="classifier_fit",
            plan=fit_plan,
            receipts=fit_receipts,
            report=fit_report,
        )
        _require_phase_evidence(
            root=root,
            contract_path=contract_path,
            contract=contract,
            phase="classifier_validation",
            plan=validation_plan,
            receipts=validation_receipts,
            report=validation_report,
            fit_report=fit_report,
        )
        _require_phase_evidence(
            root=root,
            contract_path=contract_path,
            contract=contract,
            phase="prospective_screen",
            plan=screen_plan,
            receipts=screen_receipts,
            report=screen_report,
            fit_report=fit_report,
            validation_report=validation_report,
        )
        if validation_report.get("status") != "passed":
            raise AEPriorQualificationV03Error("confirmation requires passed validation")
        if not isinstance(selection, Mapping):
            raise AEPriorQualificationV03Error("confirmation requires selection")
        expected = select_screen_loci(contract, screen_report["locus_results"])
        if dict(selection) != expected:
            raise AEPriorQualificationV03Error("selection differs from frozen screen rule")
        selected = {str(k): str(v) for k, v in selection["selected_locus_ids"].items()}
        upstream = {
            "fit_report_sha256": str(fit_report["report_sha256"]),
            "validation_report_sha256": str(validation_report["report_sha256"]),
            "screen_report_sha256": str(screen_report["report_sha256"]),
            "selection_sha256": str(selection["selection_sha256"]),
        }
    return _build_plan(root, contract_path, contract, phase, selected, upstream)


def _require_phase_evidence(
    *,
    root: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    phase: str,
    plan: Mapping[str, Any] | None,
    receipts: Sequence[Mapping[str, Any]] | None,
    report: Mapping[str, Any] | None,
    fit_report: Mapping[str, Any] | None = None,
    validation_report: Mapping[str, Any] | None = None,
) -> None:
    """Admit an upstream phase only from its deterministic plan and raw receipts."""

    if (
        not isinstance(plan, Mapping)
        or not isinstance(receipts, Sequence)
        or isinstance(receipts, str | bytes)
    ):
        raise AEPriorQualificationV03Error(f"{phase} raw evidence bundle is required")
    upstream: dict[str, str] = {}
    if phase in {"classifier_validation", "prospective_screen"}:
        if not isinstance(fit_report, Mapping):
            raise AEPriorQualificationV03Error(f"{phase} evidence lacks fit report")
        upstream["fit_report_sha256"] = str(fit_report.get("report_sha256"))
    if phase == "prospective_screen":
        if not isinstance(validation_report, Mapping):
            raise AEPriorQualificationV03Error("screen evidence lacks validation report")
        upstream["validation_report_sha256"] = str(
            validation_report.get("report_sha256")
        )
    expected_plan = _build_plan(
        root, contract_path, contract, phase, selected=None, upstream=upstream
    )
    if dict(plan) != expected_plan:
        raise AEPriorQualificationV03Error(
            f"{phase} plan differs from deterministic reconstruction"
        )
    errors = validate_report(
        contract,
        plan,
        receipts,
        report or {},
        fit_report=fit_report if phase != "classifier_fit" else None,
    )
    if errors:
        raise AEPriorQualificationV03Error(
            f"invalid {phase} raw evidence: " + "; ".join(errors)
        )


def build_screen_plan(
    root: Path,
    contract_path: Path,
    fit_report: Mapping[str, Any],
    fit_plan: Mapping[str, Any],
    fit_receipts: Sequence[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return build_phase_plan(
        root,
        contract_path,
        "prospective_screen",
        fit_report=fit_report,
        fit_plan=fit_plan,
        fit_receipts=fit_receipts,
        validation_report=validation_report,
        validation_plan=validation_plan,
        validation_receipts=validation_receipts,
    )


def build_confirmation_plan(
    root: Path,
    contract_path: Path,
    fit_report: Mapping[str, Any],
    fit_plan: Mapping[str, Any],
    fit_receipts: Sequence[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
    selection: Mapping[str, Any],
    screen_report: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    screen_plan: Mapping[str, Any],
    screen_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return build_phase_plan(
        root,
        contract_path,
        "confirmation",
        fit_report=fit_report,
        fit_plan=fit_plan,
        fit_receipts=fit_receipts,
        validation_report=validation_report,
        selection=selection,
        screen_report=screen_report,
        validation_plan=validation_plan,
        validation_receipts=validation_receipts,
        screen_plan=screen_plan,
        screen_receipts=screen_receipts,
    )


def _require_report(report: Mapping[str, Any] | None, phase: str) -> None:
    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != REPORT_VERSION
        or report.get("phase") != phase
        or report.get("development_only") is not True
        or report.get("report_sha256")
        != canonical_json_sha256({k: v for k, v in report.items() if k != "report_sha256"})
    ):
        raise AEPriorQualificationV03Error(f"invalid or unbound {phase} report")
    denominators = report.get("denominators")
    if not isinstance(denominators, Mapping):
        raise AEPriorQualificationV03Error(f"{phase} report lacks denominators")
    expected_primary = {
        "classifier_fit": 14400,
        "classifier_validation": 14400,
        "prospective_screen": 1200,
    }.get(phase)
    if expected_primary is not None and any(
        denominators.get(key) != expected_primary
        for key in (
            "primary_executions",
            "completed_primary_executions",
            "exact_replays_verified",
        )
    ):
        raise AEPriorQualificationV03Error(f"{phase} report denominator is invalid")
    if report.get("failures") != []:
        raise AEPriorQualificationV03Error(f"{phase} report contains platform failures")
    if phase == "classifier_fit":
        if report.get("status") != "completed":
            raise AEPriorQualificationV03Error("fit report status is invalid")
        models = report.get("models")
        if not isinstance(models, list) or len(models) != 10:
            raise AEPriorQualificationV03Error("fit report requires ten models")
        if {row.get("locus_id") for row in models} != {
            row[1] for row in EXPECTED_LOCUS_SEMANTICS
        }:
            raise AEPriorQualificationV03Error("fit report locus roster is invalid")
        for model in models:
            if (
                model.get("schema_version") != MODEL_VERSION
                or model.get("fit_world_clusters") != 60
                or model.get("hypotheses") != list(HYPOTHESES)
                or model.get("target_fit_parameters") != []
                or model.get("model_sha256")
                != canonical_json_sha256(
                    {key: value for key, value in model.items() if key != "model_sha256"}
                )
            ):
                raise AEPriorQualificationV03Error("fit model structure/binding is invalid")
            _validate_model_structure(model)
    elif phase == "classifier_validation":
        loci = report.get("locus_results")
        if (
            not isinstance(loci, list)
            or len(loci) != 10
            or {row.get("locus_id") for row in loci}
            != {row[1] for row in EXPECTED_LOCUS_SEMANTICS}
        ):
            raise AEPriorQualificationV03Error("validation locus roster is invalid")
        for row in loci:
            if (
                row.get("world_clusters") != 60
                or row.get("offline_cases") != 420
                or not isinstance(row.get("confusion"), Mapping)
                or set(row["confusion"]) != set(HYPOTHESES)
            ):
                raise AEPriorQualificationV03Error("validation evidence structure is invalid")
        expected_status = (
            "passed"
            if all(row.get("passed") is True for row in loci)
            else "scientifically_rejected"
        )
        if report.get("status") != expected_status:
            raise AEPriorQualificationV03Error("validation status contradicts locus results")
    elif phase == "prospective_screen":
        loci = report.get("locus_results")
        expected_loci = {row[1]: row[0] for row in EXPECTED_LOCUS_SEMANTICS}
        if (
            not isinstance(loci, list)
            or len(loci) != 10
            or {row.get("locus_id") for row in loci} != set(expected_loci)
        ):
            raise AEPriorQualificationV03Error("screen report locus roster is invalid")
        for row in loci:
            _validate_five_world_locus(row, expected_loci[str(row["locus_id"])])
        if report.get("status") != "completed":
            raise AEPriorQualificationV03Error("screen report status is invalid")
    elif phase == "confirmation":
        loci = report.get("locus_results")
        if not isinstance(loci, list) or not 0 <= len(loci) <= 5:
            raise AEPriorQualificationV03Error("confirmation locus denominator is invalid")
        expected_loci = {row[1]: row[0] for row in EXPECTED_LOCUS_SEMANTICS}
        locus_ids = [row.get("locus_id") for row in loci]
        task_ids = [row.get("task_id") for row in loci]
        if (
            len(locus_ids) != len(set(locus_ids))
            or len(task_ids) != len(set(task_ids))
            or any(locus_id not in expected_loci for locus_id in locus_ids)
        ):
            raise AEPriorQualificationV03Error("confirmation locus roster is invalid")
        for row in loci:
            _validate_five_world_locus(row, expected_loci[str(row["locus_id"])])
        expected = 120 * len(loci)
        if any(
            denominators.get(key) != expected
            for key in (
                "primary_executions",
                "completed_primary_executions",
                "exact_replays_verified",
            )
        ):
            raise AEPriorQualificationV03Error("confirmation execution denominator is invalid")
        expected_status = (
            "no_eligible_tasks"
            if not loci
            else "passed"
            if all(row.get("passed") is True for row in loci)
            else "scientifically_rejected"
        )
        if report.get("status") != expected_status:
            raise AEPriorQualificationV03Error("confirmation status contradicts results")


def _validate_five_world_locus(row: Mapping[str, Any], task_id: str) -> None:
    classification = row.get("classification_all_seven_correct_worlds")
    physical = row.get("physical_gate_worlds")
    if (
        row.get("task_id") != task_id
        or row.get("worlds_total") != 5
        or isinstance(classification, bool)
        or not isinstance(classification, int)
        or not 0 <= classification <= 5
        or isinstance(physical, bool)
        or not isinstance(physical, int)
        or not 0 <= physical <= 5
        or row.get("passed") is not (classification == physical == 5)
    ):
        raise AEPriorQualificationV03Error("five-world locus result is inconsistent")


def _validate_model_structure(model: Mapping[str, Any]) -> None:
    descriptors = model.get("descriptor_whitelist")
    metrics = model.get("classification_metric_ids")
    center = model.get("descriptor_center")
    scale = model.get("descriptor_scale")
    keep = model.get("descriptor_keep")
    if (
        not isinstance(descriptors, list)
        or not descriptors
        or any(not isinstance(value, str) or not value for value in descriptors)
        or len(set(descriptors)) != len(descriptors)
        or not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(value, str) or not value for value in metrics)
        or len(set(metrics)) != len(metrics)
        or not isinstance(center, list)
        or not isinstance(scale, list)
        or not isinstance(keep, list)
        or not len(center) == len(scale) == len(keep) == len(descriptors)
        or any(type(value) is not bool for value in keep)
        or not any(keep)
    ):
        raise AEPriorQualificationV03Error("calibration model descriptors are invalid")
    numeric = [*center, *scale]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in numeric
    ) or any(float(value) < 0.0 for value in scale) or any(
        bool(float(value) > 0.0) is not selected
        for value, selected in zip(scale, keep, strict=True)
    ):
        raise AEPriorQualificationV03Error("calibration model preprocessing is invalid")
    try:
        coefficients = np.asarray(model.get("anchor_coefficients"), dtype=float)
        covariance = np.asarray(
            model.get("predictive_covariance_diagonal"), dtype=float
        )
    except (TypeError, ValueError) as error:
        raise AEPriorQualificationV03Error("calibration model arrays are invalid") from error
    expected_coefficients = (2, sum(keep), len(metrics))
    if coefficients.shape != expected_coefficients or not np.all(np.isfinite(coefficients)):
        raise AEPriorQualificationV03Error("calibration coefficient shape/values are invalid")
    if (
        covariance.shape != (2 * 3 * len(metrics),)
        or not np.all(np.isfinite(covariance))
        or not np.all(covariance > 0.0)
    ):
        raise AEPriorQualificationV03Error("calibration covariance is invalid")
    thresholds = model.get("decision_thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "swap_evidence_min",
        "h0_evidence_min",
        "pair_evidence_min",
    }:
        raise AEPriorQualificationV03Error("calibration thresholds are invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in thresholds.values()
    ):
        raise AEPriorQualificationV03Error("calibration thresholds are invalid")


def _require_upstream_chain(
    *,
    fit_report: Mapping[str, Any],
    validation_report: Mapping[str, Any] | None = None,
    screen_report: Mapping[str, Any] | None = None,
) -> None:
    _require_report(fit_report, "classifier_fit")
    if validation_report is not None:
        _require_report(validation_report, "classifier_validation")
        if validation_report.get("fit_report_sha256") != fit_report.get("report_sha256"):
            raise AEPriorQualificationV03Error(
                "validation report is not bound to the supplied fit report"
            )
        if validation_report.get("models_sha256") != canonical_json_sha256(
            fit_report.get("models")
        ):
            raise AEPriorQualificationV03Error("validation model binding is invalid")
    if screen_report is not None:
        _require_report(screen_report, "prospective_screen")
        if validation_report is None:
            raise AEPriorQualificationV03Error("screen chain lacks validation report")
        if (
            screen_report.get("fit_report_sha256") != fit_report.get("report_sha256")
            or screen_report.get("validation_report_sha256")
            != validation_report.get("report_sha256")
        ):
            raise AEPriorQualificationV03Error("screen report upstream chain is invalid")


def _find_forbidden_key(value: object, path: str = "classifier_input") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_CLASSIFIER_KEYS:
                return f"{path}.{key}"
            found = _find_forbidden_key(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            found = _find_forbidden_key(item, f"{path}[{index}]")
            if found:
                return found
    return None


def _world_tensor(
    observations: Sequence[Mapping[str, Any]], metric_ids: Sequence[str]
) -> np.ndarray:
    expected = {(a, c, r) for a in range(2) for c in range(4) for r in range(3)}
    coordinates = {
        (int(row["anchor_id"]), int(row["target_category"]), int(row["replicate"]))
        for row in observations
    }
    if len(observations) != 24 or coordinates != expected:
        raise AEPriorQualificationV03Error("world observations are not exact 2x4x3")
    tensor = np.empty((2, 4, 3, len(metric_ids)), dtype=float)
    for row in observations:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(metric_ids):
            raise AEPriorQualificationV03Error("classifier metrics are not exact")
        values = [metrics[key] for key in metric_ids]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in values
        ):
            raise AEPriorQualificationV03Error("classifier metric is invalid")
        tensor[
            int(row["anchor_id"]),
            int(row["target_category"]),
            int(row["replicate"]),
            :,
        ] = values
    return tensor


def _descriptor_preprocessing(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    keep = scale > 0.0
    if not bool(np.any(keep)):
        raise AEPriorQualificationV03Error("no varying descriptor survives preprocessing")
    return center, scale, keep


def _fit_coefficients(
    tensors: Sequence[np.ndarray], z: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    metrics = tensors[0].shape[-1]
    coefficients = np.empty((2, z.shape[1], metrics), dtype=float)
    residuals: list[np.ndarray] = []
    for anchor in range(2):
        design = np.tile(z, (len(tensors), 1))
        response = np.concatenate(
            [HELMERT @ tensor[anchor].mean(axis=1) for tensor in tensors], axis=0
        )
        coefficients[anchor] = np.linalg.solve(
            design.T @ design + alpha * np.eye(z.shape[1]), design.T @ response
        )
    for tensor in tensors:
        observed = np.stack(
            [HELMERT @ tensor[anchor].mean(axis=1) for anchor in range(2)]
        )
        predicted = np.stack([z @ coefficients[anchor] for anchor in range(2)])
        residuals.append((observed - predicted).reshape(-1))
    residual_matrix = np.stack(residuals)
    if len(tensors) > 1:
        residual_variance = residual_matrix.var(axis=0, ddof=1)
    else:
        residual_variance = np.full(residual_matrix.shape[1], 1.0e-6)
    measurement_variances: list[np.ndarray] = []
    for tensor in tensors:
        cell = tensor.var(axis=2, ddof=1) / 3.0
        contrast = np.stack(
            [((HELMERT**2) @ cell[anchor]) for anchor in range(2)]
        )
        measurement_variances.append(contrast.reshape(-1))
    measurement = np.mean(np.stack(measurement_variances), axis=0)
    raw = residual_variance + measurement
    positive = raw[raw > 0.0]
    global_scale = float(np.median(positive)) if positive.size else 1.0e-6
    covariance = np.maximum(0.75 * raw + 0.25 * global_scale, 1.0e-8)
    return coefficients, covariance


def _raw_scores(
    *,
    dossier: Mapping[str, Any],
    target_field: str,
    whitelist: Sequence[str],
    tensor: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    keep: np.ndarray,
    coefficients: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    visible = _descriptor_matrix(dossier, target_field, whitelist)
    observed = np.stack(
        [HELMERT @ tensor[anchor].mean(axis=1) for anchor in range(2)]
    )
    scores: dict[str, float] = {}
    for hypothesis in HYPOTHESES:
        permutation = hypothesis_permutation(hypothesis)
        physical = visible[np.asarray(permutation)]
        standardized = (physical[:, keep] - center[keep]) / scale[keep]
        z = HELMERT @ standardized
        predicted = np.stack([z @ coefficients[anchor] for anchor in range(2)])
        residual = (observed - predicted).reshape(-1)
        scores[hypothesis] = float(
            0.5 * np.sum((residual**2) / covariance + np.log(covariance))
        )
    return scores


def _choose_alpha(tensors: Sequence[np.ndarray], z: np.ndarray, grid: Sequence[float]) -> float:
    if len(tensors) < 3:
        raise AEPriorQualificationV03Error("world-blocked ridge selection needs >=3 worlds")
    losses: list[tuple[float, float]] = []
    for alpha in grid:
        loss = 0.0
        for heldout in range(len(tensors)):
            training = [tensor for i, tensor in enumerate(tensors) if i != heldout]
            coefficients, covariance = _fit_coefficients(training, z, float(alpha))
            observed = np.stack(
                [HELMERT @ tensors[heldout][anchor].mean(axis=1) for anchor in range(2)]
            )
            predicted = np.stack([z @ coefficients[anchor] for anchor in range(2)])
            residual = (observed - predicted).reshape(-1)
            loss += float(np.sum((residual**2) / covariance))
        losses.append((loss, float(alpha)))
    return min(losses)[1]


def _score_evidence(scores: Mapping[str, float]) -> dict[str, Any]:
    swaps = sorted((float(scores[h]), h) for h in HYPOTHESES if h != "H0")
    best_score, best = swaps[0]
    second_score = swaps[1][0]
    return {
        "best_swap": best,
        "swap_evidence": float(scores["H0"]) - best_score,
        "h0_evidence": best_score - float(scores["H0"]),
        "pair_evidence": second_score - best_score,
    }


def _derive_thresholds(cases: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    h0_swap = [
        _score_evidence(case["scores"])["swap_evidence"]
        for case in cases
        if case["truth_hypothesis"] == "H0"
    ]
    swap_h0 = [
        _score_evidence(case["scores"])["h0_evidence"]
        for case in cases
        if case["truth_hypothesis"] != "H0"
    ]
    wrong_pair = []
    for case in cases:
        evidence = _score_evidence(case["scores"])
        if (
            case["truth_hypothesis"] != "H0"
            and evidence["best_swap"] != case["truth_hypothesis"]
        ):
            wrong_pair.append(evidence["pair_evidence"])
    if not h0_swap or not swap_h0:
        raise AEPriorQualificationV03Error("fit cases do not cover H0 and all swaps")
    epsilon = 1.0e-12
    return {
        "swap_evidence_min": max(0.0, max(h0_swap)) + epsilon,
        "h0_evidence_min": max(0.0, max(swap_h0)) + epsilon,
        "pair_evidence_min": max(0.0, max(wrong_pair) if wrong_pair else 0.0)
        + epsilon,
    }


def _raw_model(
    dossier: Mapping[str, Any],
    target_field: str,
    whitelist: Sequence[str],
    metric_ids: Sequence[str],
    tensors: Sequence[np.ndarray],
    alpha_grid: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix = _descriptor_matrix(dossier, target_field, whitelist)
    center, scale, keep = _descriptor_preprocessing(matrix)
    standardized = (matrix[:, keep] - center[keep]) / scale[keep]
    z = HELMERT @ standardized
    crossfit: list[dict[str, Any]] = []
    for heldout in range(len(tensors)):
        training = [tensor for index, tensor in enumerate(tensors) if index != heldout]
        inner_alpha = _choose_alpha(training, z, alpha_grid)
        coefficients, covariance = _fit_coefficients(training, z, inner_alpha)
        for truth in HYPOTHESES:
            varied = dossier_variant(dossier, target_field, whitelist, truth)
            crossfit.append(
                {
                    "world_index": heldout,
                    "truth_hypothesis": truth,
                    "scores": _raw_scores(
                        dossier=varied,
                        target_field=target_field,
                        whitelist=whitelist,
                        tensor=tensors[heldout],
                        center=center,
                        scale=scale,
                        keep=keep,
                        coefficients=coefficients,
                        covariance=covariance,
                    ),
                }
            )
    alpha = _choose_alpha(tensors, z, alpha_grid)
    coefficients, covariance = _fit_coefficients(tensors, z, alpha)
    thresholds = _derive_thresholds(crossfit)
    model = {
        "schema_version": MODEL_VERSION,
        "classifier_id": "signed-ridge-calibrated-weighted-residual-v0.3",
        "descriptor_whitelist": list(whitelist),
        "descriptor_center": [float(x) for x in center],
        "descriptor_scale": [float(x) for x in scale],
        "descriptor_keep": [bool(x) for x in keep],
        "ridge_alpha": alpha,
        "anchor_coefficients": coefficients.tolist(),
        "predictive_covariance_diagonal": covariance.tolist(),
        "classification_metric_ids": list(metric_ids),
        "hypotheses": list(HYPOTHESES),
        "decision_thresholds": thresholds,
        "fit_world_clusters": len(tensors),
        "target_fit_parameters": [],
        "score_name": "calibrated_weighted_residual_score",
        "scope": "four-category-reference-response-fingerprint-not-transfer-physics",
    }
    model["model_sha256"] = canonical_json_sha256(model)
    return model, crossfit


def fit_calibration_model(
    *,
    locus_id: str,
    task_id: str,
    target_field: str,
    descriptor_whitelist: Sequence[str],
    classification_metric_ids: Sequence[str],
    aligned_dossier: Mapping[str, Any],
    worlds: Sequence[Sequence[Mapping[str, Any]]],
    alpha_grid: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(worlds) != 60:
        raise AEPriorQualificationV03Error("classifier fit requires exactly 60 worlds")
    tensors = [_world_tensor(world, classification_metric_ids) for world in worlds]
    model, crossfit = _raw_model(
        aligned_dossier,
        target_field,
        descriptor_whitelist,
        classification_metric_ids,
        tensors,
        alpha_grid,
    )
    model.update({"locus_id": locus_id, "task_id": task_id, "target_field": target_field})
    model["model_sha256"] = canonical_json_sha256(
        {k: v for k, v in model.items() if k != "model_sha256"}
    )
    return model, crossfit


def classify_blind(classifier_input: Mapping[str, Any]) -> dict[str, Any]:
    if set(classifier_input) != CLASSIFIER_INPUT_KEYS:
        raise AEPriorQualificationV03Error("classifier input fields are not exact")
    forbidden = _find_forbidden_key(classifier_input)
    if forbidden:
        raise AEPriorQualificationV03Error(f"forbidden classifier input: {forbidden}")
    if classifier_input.get("anchor_ids") != [0, 1]:
        raise AEPriorQualificationV03Error("classifier requires anchor IDs [0,1]")
    model = classifier_input.get("calibration_model")
    if not isinstance(model, Mapping):
        raise AEPriorQualificationV03Error("calibration model is missing")
    if model.get("model_sha256") != canonical_json_sha256(
        {k: v for k, v in model.items() if k != "model_sha256"}
    ):
        raise AEPriorQualificationV03Error("calibration model binding is invalid")
    _validate_model_structure(model)
    for key in ("locus_id", "task_id", "target_field"):
        if classifier_input.get(key) != model.get(key):
            raise AEPriorQualificationV03Error(f"classifier {key} differs from model")
    metrics = model["classification_metric_ids"]
    tensor = _world_tensor(classifier_input["registered_observations"], metrics)
    center = np.asarray(model["descriptor_center"], dtype=float)
    scale = np.asarray(model["descriptor_scale"], dtype=float)
    keep = np.asarray(model["descriptor_keep"], dtype=bool)
    coefficients = np.asarray(model["anchor_coefficients"], dtype=float)
    covariance = np.asarray(model["predictive_covariance_diagonal"], dtype=float)
    scores = _raw_scores(
        dossier=classifier_input["dossier"],
        target_field=str(model["target_field"]),
        whitelist=model["descriptor_whitelist"],
        tensor=tensor,
        center=center,
        scale=scale,
        keep=keep,
        coefficients=coefficients,
        covariance=covariance,
    )
    evidence = _score_evidence(scores)
    thresholds = model["decision_thresholds"]
    if (
        evidence["swap_evidence"] > thresholds["swap_evidence_min"]
        and evidence["pair_evidence"] > thresholds["pair_evidence_min"]
    ):
        decision = str(evidence["best_swap"])
        state = "swap"
    elif evidence["h0_evidence"] > thresholds["h0_evidence_min"]:
        decision = "H0"
        state = "H0"
    else:
        decision = "abstain"
        state = "abstain"
    return {
        "classifier_id": model["classifier_id"],
        "decision_state": state,
        "predicted_hypothesis": decision,
        "hypothesis_scores": scores,
        "evidence": evidence,
        "model_sha256": model["model_sha256"],
    }


def _classifier_input(
    *,
    dossier: Mapping[str, Any],
    candidate: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_recipes: dict[str, list[float]] = {}
    for row in observations:
        anchor_recipes[str(row["anchor_id"])] = [float(x) for x in row["anchor_recipe"]]
    return {
        "dossier": deepcopy(dossier),
        "task_id": candidate["task_id"],
        "locus_id": candidate["locus_id"],
        "target_field": candidate["target_field"],
        "anchor_ids": [0, 1],
        "anchor_recipes": anchor_recipes,
        "registered_observations": [
            {
                "anchor_id": int(row["anchor_id"]),
                "target_category": int(row["target_category"]),
                "replicate": int(row["replicate"]),
                "metrics": {
                    key: float(row["metrics"][key])
                    for key in model["classification_metric_ids"]
                },
            }
            for row in observations
        ],
        "calibration_model": deepcopy(model),
    }


def score_all_hypotheses(
    *,
    candidate: Mapping[str, Any],
    aligned_dossier: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate predictions first, then score them against private offline truth."""

    predictions: list[dict[str, Any]] = []
    for truth in HYPOTHESES:
        visible = dossier_variant(
            aligned_dossier,
            str(candidate["target_field"]),
            candidate["descriptor_whitelist"],
            truth,
        )
        prediction = classify_blind(
            _classifier_input(
                dossier=visible,
                candidate=candidate,
                observations=observations,
                model=model,
            )
        )
        # Truth is attached only after classify_blind has returned.
        predictions.append(
            {
                "truth_hypothesis": truth,
                "prediction": prediction,
                "correct": prediction["predicted_hypothesis"] == truth,
                "definite_wrong": prediction["predicted_hypothesis"]
                not in {truth, "abstain"},
            }
        )
    return {
        "cases": predictions,
        "all_seven_correct": all(row["correct"] for row in predictions),
        "any_definite_wrong": any(row["definite_wrong"] for row in predictions),
    }


def _physical_gate(
    observations: Sequence[Mapping[str, Any]],
    gate_endpoint_id: str,
    pair: Sequence[int],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for anchor in range(2):
        left = [
            float(row["metrics"][gate_endpoint_id])
            for row in observations
            if int(row["anchor_id"]) == anchor
            and int(row["target_category"]) == int(pair[0])
        ]
        right = [
            float(row["metrics"][gate_endpoint_id])
            for row in observations
            if int(row["anchor_id"]) == anchor
            and int(row["target_category"]) == int(pair[1])
        ]
        if len(left) != 3 or len(right) != 3:
            raise AEPriorQualificationV03Error("physical gate lacks three replicates")
        separation = abs(fmean(right) - fmean(left))
        standard_error = math.sqrt(variance(left) / 3.0 + variance(right) / 3.0)
        snr = separation / max(standard_error, 1.0e-12)
        anchors.append(
            {
                "anchor_id": anchor,
                "absolute_separation": separation,
                "welch_standard_error": standard_error,
                "signal_to_noise_ratio": snr,
                "passed": separation
                >= float(thresholds["minimum_absolute_gate_separation"])
                and snr
                >= float(thresholds["minimum_gate_signal_to_noise_ratio"]),
            }
        )
    return {"anchor_results": anchors, "passed": all(row["passed"] for row in anchors)}


def _receipt_observations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "anchor_id": int(row["anchor_id"]),
            "target_category": int(row["target_category"]),
            "replicate": int(row["replicate"]),
            "anchor_recipe": list(row["anchor_recipe"]),
            "metrics": {
                **dict(row["classification_metrics"]),
                **dict(row["non_gating_secondary_metrics"]),
            },
        }
        for row in rows
    ]


def _project_classifier_metrics(
    observations: Sequence[Mapping[str, Any]], metric_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Keep complete receipts while exposing only registered classifier metrics."""

    return [
        {
            **{key: deepcopy(value) for key, value in row.items() if key != "metrics"},
            "metrics": {key: row["metrics"][key] for key in metric_ids},
        }
        for row in observations
    ]


def _group_receipts(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    planned = {row["execution_id"]: row for row in plan["executions"]}
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        row = planned[receipt["execution_id"]]
        grouped[(str(row["locus_id"]), int(row["world_index"]))].append(receipt)
    return grouped


def build_phase_report(
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    fit_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors = validate_receipt_denominator(plan, receipts)
    if errors:
        raise AEPriorQualificationV03Error("invalid phase evidence: " + "; ".join(errors))
    phase = str(plan["phase"])
    grouped = _group_receipts(plan, receipts)
    candidate_index = _candidate_index(contract)
    bindings = {row["locus_id"]: row for row in plan["task_locus_bindings"]}
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "development_only": True,
        "phase": phase,
        "plan_sha256": plan["plan_sha256"],
        "denominators": {
            **deepcopy(plan["denominators"]),
            "completed_primary_executions": len(receipts),
            "exact_replays_verified": sum(
                1 for row in receipts if row["exact_replay"]["verified"] is True
            ),
        },
        "failures": [
            {
                "execution_id": row.get("execution_id"),
                "status": row.get("status"),
                "failure": deepcopy(row.get("failure")),
            }
            for row in receipts
            if row.get("status") != "completed"
        ],
    }
    if phase == "classifier_fit":
        models: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for locus_id, candidate in candidate_index.items():
            classifier_metric_ids = [
                candidate["gate_endpoint_id"],
                *candidate["classifier_secondary_endpoint_ids"],
            ]
            worlds = [
                _project_classifier_metrics(
                    _receipt_observations(grouped[(locus_id, world)]),
                    classifier_metric_ids,
                )
                for world in range(60)
            ]
            model, crossfit = fit_calibration_model(
                locus_id=locus_id,
                task_id=candidate["task_id"],
                target_field=candidate["target_field"],
                descriptor_whitelist=candidate["descriptor_whitelist"],
                classification_metric_ids=classifier_metric_ids,
                aligned_dossier=plan["executions"][
                    next(
                        i
                        for i, row in enumerate(plan["executions"])
                        if row["locus_id"] == locus_id
                    )
                ]["aligned_dossier"],
                worlds=worlds,
                alpha_grid=contract["classifier"]["ridge_alpha_grid"],
            )
            models.append(model)
            diagnostics.append(
                {
                    "locus_id": locus_id,
                    "fit_world_clusters": 60,
                    "offline_cases": len(crossfit),
                    "offline_cases_are_clustered_by_world": True,
                }
            )
        report.update(
            {
                "status": "completed",
                "models": models,
                "fit_diagnostics": diagnostics,
                "locus_results": [],
            }
        )
    else:
        _require_upstream_chain(fit_report=fit_report)
        expected_fit = plan.get("upstream_bindings", {}).get("fit_report_sha256")
        if expected_fit != fit_report.get("report_sha256"):
            raise AEPriorQualificationV03Error(
                "phase plan is not bound to the supplied fit report"
            )
        models = {row["locus_id"]: row for row in fit_report["models"]}
        world_results: list[dict[str, Any]] = []
        for (locus_id, world_index), rows in sorted(grouped.items()):
            candidate = candidate_index[locus_id]
            observations = _receipt_observations(rows)
            aligned = next(
                row["aligned_dossier"]
                for row in plan["executions"]
                if row["locus_id"] == locus_id
            )
            classification = score_all_hypotheses(
                candidate=candidate,
                aligned_dossier=aligned,
                observations=observations,
                model=models[locus_id],
            )
            row_result: dict[str, Any] = {
                "locus_id": locus_id,
                "task_id": candidate["task_id"],
                "world_index": world_index,
                "classification": classification,
            }
            if phase in {"prospective_screen", "confirmation"}:
                physical = _physical_gate(
                    observations,
                    candidate["gate_endpoint_id"],
                    bindings[locus_id]["private_scoring_pair"],
                    contract["thresholds"],
                )
                row_result["physical_gate"] = physical
                row_result["passed"] = (
                    classification["all_seven_correct"] and physical["passed"]
                )
            world_results.append(row_result)
        locus_results = _summarize_loci(contract, phase, world_results)
        report.update(
            {
                "world_results": world_results,
                "locus_results": locus_results,
                "models_sha256": canonical_json_sha256(fit_report["models"]),
                "fit_report_sha256": fit_report["report_sha256"],
            }
        )
        if phase in {"prospective_screen", "confirmation"}:
            report["validation_report_sha256"] = plan["upstream_bindings"][
                "validation_report_sha256"
            ]
        if phase == "confirmation":
            report["screen_report_sha256"] = plan["upstream_bindings"][
                "screen_report_sha256"
            ]
            report["selection_sha256"] = plan["upstream_bindings"]["selection_sha256"]
        if phase == "classifier_validation":
            report["status"] = (
                "passed"
                if len(locus_results) == 10 and all(row["passed"] for row in locus_results)
                else "scientifically_rejected"
            )
        elif phase == "prospective_screen":
            report["status"] = "completed"
        else:
            report["status"] = (
                "no_eligible_tasks"
                if not locus_results
                else "passed"
                if all(row["passed"] for row in locus_results)
                else "scientifically_rejected"
            )
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def _summarize_loci(
    contract: Mapping[str, Any], phase: str, world_results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in world_results:
        grouped[str(row["locus_id"])].append(row)
    for locus_id, worlds in sorted(grouped.items()):
        if phase == "classifier_validation":
            class_correct = Counter()
            confusion = {truth: Counter() for truth in HYPOTHESES}
            any_wrong = 0
            global_correct = 0
            for world in worlds:
                classification = world["classification"]
                any_wrong += int(classification["any_definite_wrong"])
                global_correct += int(classification["all_seven_correct"])
                for case in classification["cases"]:
                    truth = case["truth_hypothesis"]
                    predicted = case["prediction"]["predicted_hypothesis"]
                    confusion[truth][predicted] += 1
                    class_correct[truth] += int(case["correct"])
            world_count = len(worlds)
            any_wrong_upper = (
                1.0
                if world_count == 0 or any_wrong == world_count
                else float(beta.ppf(0.95, any_wrong + 1, world_count - any_wrong))
            )
            per_class_lower = {
                hypothesis: float(
                    beta.ppf(
                        0.05 / 7.0,
                        class_correct[hypothesis],
                        world_count - class_correct[hypothesis] + 1,
                    )
                )
                if class_correct[hypothesis] > 0
                else 0.0
                for hypothesis in HYPOTHESES
            }
            all_correct_lower = (
                float(
                    beta.ppf(
                        0.05, global_correct, world_count - global_correct + 1
                    )
                )
                if global_correct > 0
                else 0.0
            )
            passed = (
                len(worlds) == 60
                and any_wrong == 0
                and any_wrong_upper
                <= float(
                    contract["thresholds"][
                        "validation_any_definite_wrong_cp95_upper"
                    ]
                )
                and all(class_correct[h] >= 56 for h in HYPOTHESES)
                and all(
                    per_class_lower[h]
                    > float(
                        contract["thresholds"][
                            "validation_per_class_bonferroni_cp_lower_min"
                        ]
                    )
                    for h in HYPOTHESES
                )
                and global_correct >= 54
                and all_correct_lower
                > float(
                    contract["thresholds"][
                        "validation_all_seven_cp95_lower_min"
                    ]
                )
            )
            output.append(
                {
                    "locus_id": locus_id,
                    "world_clusters": len(worlds),
                    "offline_cases": len(worlds) * 7,
                    "any_definite_wrong_worlds": any_wrong,
                    "any_definite_wrong_cp95_upper": any_wrong_upper,
                    "per_class_correct": {h: class_correct[h] for h in HYPOTHESES},
                    "per_class_bonferroni_cp_lower": per_class_lower,
                    "all_seven_correct_worlds": global_correct,
                    "all_seven_correct_cp95_lower": all_correct_lower,
                    "confusion": {
                        truth: {
                            prediction: confusion[truth][prediction]
                            for prediction in (*HYPOTHESES, "abstain")
                        }
                        for truth in HYPOTHESES
                    },
                    "passed": passed,
                }
            )
        else:
            output.append(
                {
                    "locus_id": locus_id,
                    "task_id": worlds[0]["task_id"],
                    "worlds_total": len(worlds),
                    "classification_all_seven_correct_worlds": sum(
                        int(row["classification"]["all_seven_correct"]) for row in worlds
                    ),
                    "physical_gate_worlds": sum(
                        int(row["physical_gate"]["passed"]) for row in worlds
                    ),
                    "passed": len(worlds) == 5 and all(row["passed"] for row in worlds),
                }
            )
    return output


def select_screen_loci(
    contract: Mapping[str, Any], locus_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    results = {str(row["locus_id"]): row for row in locus_results}
    selected: dict[str, str] = {}
    disposition: list[dict[str, Any]] = []
    for task in contract["tasks"]:
        ordered = sorted(
            task["loci"],
            key=lambda row: (int(row["scientific_priority"]), str(row["locus_id"])),
        )
        eligible = [row for row in ordered if results.get(row["locus_id"], {}).get("passed")]
        if eligible:
            selected[str(task["task_id"])] = str(eligible[0]["locus_id"])
        for locus in ordered:
            disposition.append(
                {
                    "task_id": task["task_id"],
                    "locus_id": locus["locus_id"],
                    "eligible": bool(results.get(locus["locus_id"], {}).get("passed")),
                }
            )
    payload = {
        "schema_version": SELECTION_VERSION,
        "development_only": True,
        "selected_locus_ids": selected,
        "selected_task_count": len(selected),
        "candidate_disposition": disposition,
        "screen_locus_results_sha256": canonical_json_sha256(list(locus_results)),
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    return payload


def build_development_summary(
    contract: Mapping[str, Any],
    fit_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    screen_report: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
    confirmation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_upstream_chain(
        fit_report=fit_report,
        validation_report=validation_report,
        screen_report=screen_report,
    )
    selected_count = int(selection.get("selected_task_count", 0)) if selection else 0
    if screen_report is not None:
        expected_selection = select_screen_loci(contract, screen_report["locus_results"])
        if not isinstance(selection, Mapping) or dict(selection) != expected_selection:
            raise AEPriorQualificationV03Error("summary selection differs from screen")
    if confirmation_report is not None:
        _require_report(confirmation_report, "confirmation")
        if confirmation_report["denominators"]["primary_executions"] != 120 * selected_count:
            raise AEPriorQualificationV03Error("confirmation denominator differs from selection")
        if (
            confirmation_report.get("fit_report_sha256") != fit_report.get("report_sha256")
            or confirmation_report.get("validation_report_sha256")
            != validation_report.get("report_sha256")
            or confirmation_report.get("screen_report_sha256")
            != screen_report.get("report_sha256")
            or confirmation_report.get("selection_sha256")
            != selection.get("selection_sha256")
            or {
                row.get("task_id"): row.get("locus_id")
                for row in confirmation_report["locus_results"]
            }
            != dict(selection["selected_locus_ids"])
        ):
            raise AEPriorQualificationV03Error("confirmation upstream binding is invalid")
    confirmed = (
        sum(1 for row in confirmation_report["locus_results"] if row["passed"])
        if confirmation_report
        else 0
    )
    summary = {
        "schema_version": SUMMARY_VERSION,
        "development_only": True,
        "status": (
            "classifier_validation_rejected"
            if validation_report["status"] != "passed"
            else "awaiting_prospective_screen"
            if screen_report is None
            else "awaiting_confirmation"
            if confirmation_report is None
            else "completed"
        ),
        "phase_report_bindings": {
            "classifier_fit": fit_report["report_sha256"],
            "classifier_validation": validation_report["report_sha256"],
            "prospective_screen": screen_report.get("report_sha256")
            if screen_report
            else None,
            "confirmation": confirmation_report.get("report_sha256")
            if confirmation_report
            else None,
        },
        "denominators": {
            "classifier_fit_primary": fit_report["denominators"]["primary_executions"],
            "classifier_validation_primary": validation_report["denominators"][
                "primary_executions"
            ],
            "prospective_screen_primary": screen_report["denominators"][
                "primary_executions"
            ]
            if screen_report
            else 0,
            "selected_tasks": selected_count,
            "confirmation_primary": confirmation_report["denominators"][
                "primary_executions"
            ]
            if confirmation_report
            else 0,
        },
        "qualified_task_count": confirmed,
        "claim_scope": (
            "universal-five-task" if confirmed == 5 else "qualified-locus" if confirmed else "none"
        ),
        "all_failures": [
            failure
            for report in (fit_report, validation_report, screen_report, confirmation_report)
            if report
            for failure in report.get("failures", [])
        ],
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return summary


def _operation(record: Mapping[str, Any]) -> str | None:
    action = record.get("action")
    return str(action.get("operation")) if isinstance(action, Mapping) else None


def extract_registered_measurement(
    records: Sequence[Mapping[str, Any]],
    task_id: str,
    stage_id: str,
    metric_ids: Sequence[str],
) -> dict[str, Any]:
    """Resolve a registered measurement by event ordering, never by fixed step index."""

    if not records:
        raise AEPriorQualificationV03Error("trajectory is empty")
    operations = [_operation(row) for row in records]
    if stage_id == "final-assay":
        terminate = [i for i, operation in enumerate(operations) if operation == "terminate"]
        matched = [
            i
            for i, row in enumerate(records)
            if _operation(row) == "measure" and row.get("instrument") == "final_assay"
        ]
        if (
            len(terminate) != 1
            or len(matched) != 1
            or matched[0] <= terminate[0]
            or matched[0] != len(records) - 1
        ):
            raise AEPriorQualificationV03Error("final-assay event window is not unique/terminal")
    elif stage_id == "reaction-post-quench-hplc":
        quench = [i for i, operation in enumerate(operations) if operation == "quench"]
        if len(quench) != 1:
            raise AEPriorQualificationV03Error("post-quench stage requires exactly one quench")
        downstream_by_task = {
            "reaction-to-crystallization": {
                "seed_crystals",
                "cool_crystallize",
                "filter_crystals",
            },
            "reaction-to-distillation": {"evaporate", "distill", "collect_fraction"},
            "reaction-safety-constrained": {"terminate"},
        }
        if task_id not in downstream_by_task:
            raise AEPriorQualificationV03Error("post-quench stage task is invalid")
        boundaries = [
            i
            for i in range(quench[0] + 1, len(records))
            if operations[i] in downstream_by_task[task_id]
        ]
        if not boundaries:
            raise AEPriorQualificationV03Error("required downstream boundary is missing")
        boundary = boundaries[0]
        matched = [
            i
            for i in range(quench[0] + 1, boundary)
            if operations[i] == "measure" and records[i].get("instrument") == "hplc"
        ]
        if len(matched) != 1:
            raise AEPriorQualificationV03Error(
                "post-quench/pre-downstream HPLC is not unique"
            )
    elif stage_id == "partition-post-settle-pre-separation-hplc":
        settle = [i for i, operation in enumerate(operations) if operation == "settle"]
        separate = [i for i, operation in enumerate(operations) if operation == "separate_phase"]
        if len(settle) != 1 or len(separate) != 1 or separate[0] <= settle[0]:
            raise AEPriorQualificationV03Error("partition settle/separation window is invalid")
        matched = [
            i
            for i in range(settle[0] + 1, separate[0])
            if operations[i] == "measure" and records[i].get("instrument") == "hplc"
        ]
        if len(matched) != 1:
            raise AEPriorQualificationV03Error(
                "post-settle/pre-separation HPLC is not unique"
            )
    else:
        raise AEPriorQualificationV03Error(f"unknown measurement stage {stage_id}")
    index = matched[0]
    record = records[index]
    if record.get("transaction_status") != "committed":
        raise AEPriorQualificationV03Error("registered measurement is not committed")
    visible = record.get("agent_visible_observation")
    views = visible.get("views") if isinstance(visible, Mapping) else None
    tool_json = views.get("tool_json") if isinstance(views, Mapping) else None
    observation = tool_json.get("observation") if isinstance(tool_json, Mapping) else None
    mask = tool_json.get("observed_mask") if isinstance(tool_json, Mapping) else None
    if not isinstance(observation, Mapping) or not isinstance(mask, Mapping):
        raise AEPriorQualificationV03Error("measurement observation/mask is missing")
    metrics: dict[str, float] = {}
    for metric_id in metric_ids:
        value = observation.get(metric_id)
        if (
            mask.get(metric_id) is not True
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise AEPriorQualificationV03Error(
                f"registered measurement lacks valid observed {metric_id}"
            )
        metrics[str(metric_id)] = float(value)
    return {
        "measurement_stage_id": stage_id,
        "matched_step_index": index,
        "log_step": record.get("step"),
        "operation": _operation(record),
        "operation_id": record.get("operation_id"),
        "instrument": record.get("instrument"),
        "observed_mask": dict.fromkeys(metric_ids, True),
        "metrics": metrics,
    }


def execute_one(
    root: Path,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    output_root: Path,
    *,
    execution_root: Path | None = None,
) -> dict[str, Any]:
    binding = next(
        item for item in plan["task_locus_bindings"] if item["locus_id"] == row["locus_id"]
    )
    config = _load_object(root / binding["campaign_config"])
    execution_root = execution_root or (
        output_root / "executions" / str(row["execution_index"])
    )
    execution_root = execution_root.resolve()
    if not execution_root.is_relative_to(output_root.resolve()):
        raise AEPriorQualificationV03Error("execution directory escapes output root")
    execution_root.mkdir(parents=True, exist_ok=False)
    trajectory = execution_root / "trajectory.jsonl"
    receipt = {
        key: deepcopy(row[key])
        for key in (
            "execution_index",
            "execution_id",
            "phase",
            "task_id",
            "locus_id",
            "world_index",
            "world_seed",
            "anchor_id",
            "target_category",
            "replicate",
            "anchor_recipe",
            "measurement_stage_id",
        )
    }
    receipt.update(
        {
            "schema_version": RECEIPT_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "provider_call_count": 0,
        }
    )
    try:
        actions = row["recipe"]["steps"]
        run_agent(
            env_id=get_task(str(row["task_id"])).env_id,
            agent=_FrozenRecipeAgent(actions),
            world_split=str(config["world_split"]),
            budget=len(actions),
            objective=str(config["objective"]),
            seed=int(row["world_seed"]),
            observation_seed=int(row["observation_seed"]),
            task_id=str(row["task_id"]),
            output_path=trajectory,
            budget_override=len(actions),
            episode_mode_override="single_experiment",
            material_information={"mode": "anonymous_nominal_properties"},
            electrochemical_material_family_id=config.get("electrochemical_material_family_id"),
            crystallization_material_family_id=config.get("crystallization_material_family_id"),
            electrochemical_workflow_mode=config.get("electrochemical_workflow_mode"),
            scoring_contract_id=config.get("scoring_contract_id"),
            observation_noise_mode="keyed",
            observation_noise_namespace=str(row["observation_noise_namespace"]),
            world_interventions=config.get("world_interventions", []),
        )
        records = load_jsonl(trajectory)
        if [record.get("action") for record in records] != actions:
            raise AEPriorQualificationV03Error("trajectory differs from frozen recipe")
        if any(record.get("transaction_status") != "committed" for record in records):
            raise AEPriorQualificationV03Error(
                "deterministic qualification recipe contains a noncommitted operation"
            )
        measurement = extract_registered_measurement(
            records,
            str(row["task_id"]),
            row["measurement_stage_id"],
            row["measured_metric_ids"],
        )
        replay = verify_records(
            records, tolerance=0.0, world_interventions=config.get("world_interventions", [])
        ).to_dict()
        if replay.get("verified") is not True:
            raise AEPriorQualificationV03Error("full-trajectory exact replay failed")
        receipt.update(
            {
                "status": "completed",
                "measurement": measurement,
                "classification_metrics": {
                    key: measurement["metrics"][key]
                    for key in row["classification_metric_ids"]
                },
                "non_gating_secondary_metrics": {
                    key: measurement["metrics"][key]
                    for key in row["non_gating_secondary_metric_ids"]
                },
                "exact_replay": replay,
                "trajectory": {
                    "path": trajectory.relative_to(output_root).as_posix(),
                    "sha256": file_sha256(trajectory),
                },
                "failure": None,
            }
        )
    except Exception as error:
        receipt.update(
            {
                "status": "platform_failure",
                "measurement": None,
                "classification_metrics": None,
                "non_gating_secondary_metrics": None,
                "exact_replay": None,
                "trajectory": {
                    "path": trajectory.relative_to(output_root).as_posix(),
                    "sha256": file_sha256(trajectory),
                }
                if trajectory.is_file()
                else None,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def validate_plan(
    plan: Mapping[str, Any],
    *,
    root: Path | None = None,
    contract_path: Path | None = None,
    upstream_reports: Mapping[str, Mapping[str, Any]] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != PLAN_VERSION or plan.get("development_only") is not True:
        errors.append("plan is not development-only v0.3")
    if plan.get("phase") not in PHASES:
        errors.append("plan phase is invalid")
    if plan.get("plan_sha256") != canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        errors.append("plan self binding is invalid")
    executions = plan.get("executions")
    executions = executions if isinstance(executions, list) else []
    denominator = plan.get("denominators")
    if not isinstance(denominator, Mapping) or len(executions) != denominator.get(
        "primary_executions"
    ):
        errors.append("plan denominator is invalid")
    ids = [row.get("execution_id") for row in executions]
    if len(ids) != len(set(ids)):
        errors.append("execution IDs are duplicated")
    coordinates = [
        (
            row.get("phase"),
            row.get("locus_id"),
            row.get("world_index"),
            row.get("anchor_id"),
            row.get("target_category"),
            row.get("replicate"),
        )
        for row in executions
    ]
    if len(coordinates) != len(set(coordinates)):
        errors.append("execution coordinates are duplicated")
    noise = [row.get("observation_seed") for row in executions]
    namespaces = [row.get("observation_noise_namespace") for row in executions]
    if len(noise) != len(set(noise)) or len(namespaces) != len(set(namespaces)):
        errors.append("keyed observation noise is not coordinate-unique")
    for row in executions:
        if row.get("measurement_stage_id") not in {
            "final-assay",
            "reaction-post-quench-hplc",
            "partition-post-settle-pre-separation-hplc",
        }:
            errors.append("execution measurement stage is invalid")
            break
        if row.get("gate_endpoint_id") not in row.get("classification_metric_ids", []):
            errors.append("gate endpoint is absent from classifier metrics")
            break
    if root is not None and contract_path is not None:
        try:
            phase = str(plan["phase"])
            reports = upstream_reports or {}
            rebuilt = build_phase_plan(
                root,
                contract_path,
                phase,
                fit_report=reports.get("classifier_fit"),
                validation_report=reports.get("classifier_validation"),
                selection=selection,
                screen_report=reports.get("prospective_screen"),
            )
            if rebuilt != plan:
                errors.append("plan differs from deterministic reconstruction")
        except Exception as error:
            errors.append(f"plan cannot be reconstructed: {error}")
    return errors


def _receipt_errors(
    planned: Mapping[str, Any], receipt: Mapping[str, Any], *, completed: bool
) -> list[str]:
    errors: list[str] = []
    if receipt.get("receipt_sha256") != canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ):
        errors.append("receipt self binding is invalid")
    for key in (
        "execution_index",
        "execution_id",
        "phase",
        "task_id",
        "locus_id",
        "world_index",
        "world_seed",
        "anchor_id",
        "target_category",
        "replicate",
        "anchor_recipe",
        "measurement_stage_id",
    ):
        if receipt.get(key) != planned.get(key):
            errors.append(f"receipt differs from plan at {key}")
    if completed:
        if receipt.get("status") != "completed" or receipt.get("failure") is not None:
            errors.append("completed receipt status/failure is invalid")
        measurement = receipt.get("measurement")
        if not isinstance(measurement, Mapping):
            errors.append("completed receipt lacks measurement")
        else:
            if measurement.get("measurement_stage_id") != planned.get(
                "measurement_stage_id"
            ):
                errors.append("receipt measurement stage differs from plan")
            if set(measurement.get("metrics", {})) != set(planned["measured_metric_ids"]):
                errors.append("receipt measured metrics differ from plan")
        if set(receipt.get("classification_metrics", {})) != set(
            planned["classification_metric_ids"]
        ):
            errors.append("receipt classifier metrics differ from plan")
        if receipt.get("exact_replay", {}).get("verified") is not True:
            errors.append("receipt exact replay is not verified")
        if not isinstance(receipt.get("trajectory", {}).get("sha256"), str):
            errors.append("receipt trajectory binding is missing")
    elif receipt.get("status") != "platform_failure" or not isinstance(
        receipt.get("failure"), Mapping
    ):
        errors.append("platform failure receipt is malformed")
    return errors


def validate_phase_progress(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> list[str]:
    errors = validate_plan(plan)
    planned = plan.get("executions", [])
    if len(receipts) > len(planned):
        return [*errors, "receipt prefix exceeds plan"]
    if [row.get("execution_id") for row in receipts] != [
        row.get("execution_id") for row in planned[: len(receipts)]
    ]:
        errors.append("receipts are not the immutable plan prefix")
    platform_failed = [
        index for index, row in enumerate(receipts) if row.get("status") == "platform_failure"
    ]
    if platform_failed and platform_failed != [len(receipts) - 1]:
        errors.append("phase continued after its first platform failure")
    for index, receipt in enumerate(receipts):
        errors.extend(
            _receipt_errors(
                planned[index], receipt, completed=receipt.get("status") == "completed"
            )
        )
    return errors


def validate_next_receipt(
    plan: Mapping[str, Any], completed_count: int, receipt: Mapping[str, Any]
) -> list[str]:
    """Validate one append without rescanning an already validated receipt prefix."""

    executions = plan.get("executions")
    if not isinstance(executions, list) or not 0 <= completed_count < len(executions):
        return ["next receipt is outside the immutable plan"]
    planned = executions[completed_count]
    if receipt.get("execution_id") != planned.get("execution_id"):
        return ["next receipt is not the immutable plan continuation"]
    return _receipt_errors(
        planned, receipt, completed=receipt.get("status") == "completed"
    )


def load_resume_prefix(
    output_root: Path,
    expected_plan: Mapping[str, Any],
    *,
    minimum_quiescent_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Load a fail-closed, missing-only phase prefix from an interrupted output."""

    output_root = output_root.resolve()
    if not output_root.is_dir():
        raise AEPriorQualificationV03Error("resume output root does not exist")
    allowed_root_members = {
        "plan.json",
        "receipts",
        "executions",
        "resume-executions",
        ".runner.lock",
    }
    unexpected = sorted(
        path.name for path in output_root.iterdir() if path.name not in allowed_root_members
    )
    if unexpected:
        raise AEPriorQualificationV03Error(
            "resume output contains terminal or unexpected members: "
            + ", ".join(unexpected)
        )
    materialized = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != ".runner.lock"
    ]
    if materialized and time.time() - max(path.stat().st_mtime for path in materialized) < float(
        minimum_quiescent_seconds
    ):
        raise AEPriorQualificationV03Error(
            "resume output is not quiescent; the original runner may still be active"
        )
    stored_plan = _load_object(output_root / "plan.json")
    if stored_plan != expected_plan:
        raise AEPriorQualificationV03Error(
            "resume plan differs from deterministic reconstruction"
        )
    receipt_root = output_root / "receipts"
    if receipt_root.exists() and not receipt_root.is_dir():
        raise AEPriorQualificationV03Error("resume receipt root is not a directory")
    receipt_files = list(receipt_root.iterdir()) if receipt_root.is_dir() else []
    if any(not path.is_file() or path.suffix != ".json" for path in receipt_files):
        raise AEPriorQualificationV03Error("resume receipts contain unexpected members")
    try:
        receipt_files.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise AEPriorQualificationV03Error(
            "resume receipt filenames must be numeric execution indexes"
        ) from error
    if [int(path.stem) for path in receipt_files] != list(range(len(receipt_files))):
        raise AEPriorQualificationV03Error(
            "resume receipt indexes must be a complete prefix from zero"
        )
    receipts = [_load_object(path) for path in receipt_files]
    progress_errors = validate_phase_progress(expected_plan, receipts)
    if progress_errors:
        raise AEPriorQualificationV03Error(
            "invalid resume receipt prefix: " + "; ".join(progress_errors)
        )
    if any(receipt.get("status") == "platform_failure" for receipt in receipts):
        raise AEPriorQualificationV03Error(
            "a platform-defective phase must restart from execution zero"
        )
    total = len(expected_plan["executions"])
    prefix_length = len(receipts)
    for receipt in receipts:
        trajectory_binding = receipt.get("trajectory")
        if not isinstance(trajectory_binding, Mapping):
            raise AEPriorQualificationV03Error("resume receipt lacks trajectory binding")
        relative = PurePosixPath(str(trajectory_binding.get("path", "")))
        index = int(receipt["execution_index"])
        canonical = relative.parts == ("executions", str(index), "trajectory.jsonl")
        resumed = (
            len(relative.parts) == 4
            and relative.parts[0] == "resume-executions"
            and relative.parts[1].isdigit()
            and relative.parts[2] == str(index)
            and relative.parts[3] == "trajectory.jsonl"
        )
        if not canonical and not resumed:
            raise AEPriorQualificationV03Error("resume trajectory path is not registered")
        trajectory = (output_root / Path(*relative.parts)).resolve()
        if (
            not trajectory.is_relative_to(output_root)
            or not trajectory.is_file()
            or file_sha256(trajectory) != trajectory_binding.get("sha256")
        ):
            raise AEPriorQualificationV03Error("resume trajectory binding is invalid")

    def validate_execution_tree(root: Path, *, nested: bool) -> None:
        if not root.exists():
            return
        if not root.is_dir():
            raise AEPriorQualificationV03Error("resume execution root is not a directory")
        parents = list(root.iterdir())
        if nested:
            if any(not path.is_dir() or not path.name.isdigit() for path in parents):
                raise AEPriorQualificationV03Error(
                    "resume attempt directories are malformed"
                )
            execution_roots = [child for parent in parents for child in parent.iterdir()]
        else:
            execution_roots = parents
        for execution_root in execution_roots:
            if not execution_root.is_dir() or not execution_root.name.isdigit():
                raise AEPriorQualificationV03Error(
                    "resume execution directories are malformed"
                )
            index = int(execution_root.name)
            if index < 0 or index > prefix_length or index >= total:
                raise AEPriorQualificationV03Error(
                    "resume found an execution outside the completed/next prefix"
                )
            members = list(execution_root.iterdir())
            if any(path.name != "trajectory.jsonl" or not path.is_file() for path in members):
                raise AEPriorQualificationV03Error(
                    "resume execution directory contains unexpected members"
                )

    validate_execution_tree(output_root / "executions", nested=False)
    validate_execution_tree(output_root / "resume-executions", nested=True)
    return receipts


def validate_receipt_denominator(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> list[str]:
    errors = validate_phase_progress(plan, receipts)
    if len(receipts) != len(plan.get("executions", [])):
        errors.append("receipt count differs from exact denominator")
    if any(row.get("status") == "platform_failure" for row in receipts):
        errors.append("platform-defective phase cannot produce a scientific report")
    return errors


def validate_report(
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    fit_report: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        rebuilt = build_phase_report(contract, plan, receipts, fit_report=fit_report)
        if rebuilt != report:
            errors.append("report differs from deterministic reconstruction")
    except Exception as error:
        errors.append(f"report cannot be reconstructed: {error}")
    return errors


__all__ = [
    "CONTRACT_VERSION",
    "PLAN_VERSION",
    "REPORT_VERSION",
    "SELECTION_VERSION",
    "SUMMARY_VERSION",
    "AEPriorQualificationV03Error",
    "build_confirmation_plan",
    "build_development_summary",
    "build_locus_schedule",
    "build_phase_plan",
    "build_phase_report",
    "build_screen_plan",
    "classify_blind",
    "dossier_variant",
    "execute_one",
    "extract_registered_measurement",
    "fit_calibration_model",
    "hypothesis_permutation",
    "load_resume_prefix",
    "score_all_hypotheses",
    "select_descriptor_pair",
    "select_screen_loci",
    "validate_contract",
    "validate_next_receipt",
    "validate_phase_progress",
    "validate_plan",
    "validate_receipt_denominator",
    "validate_report",
]
