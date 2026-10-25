"""Development/release A-E prior-distinguishability qualification v0.2.

The blind policy is intentionally separate from the hidden-pair analyzer.  It receives
neither descriptor permutations nor outcomes and makes no provider calls.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from statistics import fmean, variance
from typing import Any

import numpy as np

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.task_recipes import (
    task_recipe_dimension,
    task_recipe_from_unit_vector,
)
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.eval.work_ii_execution_mode import (
    ExecutionMode,
    build_execution_envelope,
    prepare_execution_context,
    release_manifest_sha256,
    validate_execution_envelope,
)
from chemworld.eval.work_ii_formal import build_checkpoint_contract
from chemworld.tasks import get_task

CONTRACT_VERSION = "chemworld-work-ii-ae-prior-distinguishability-contract-0.2"
LEGACY_DEVELOPMENT_PLAN_VERSION = "chemworld-work-ii-ae-prior-distinguishability-plan-0.2"
PLAN_VERSION = "chemworld-work-ii-ae-prior-distinguishability-plan-0.3"
LEGACY_DEVELOPMENT_REPORT_VERSION = "chemworld-work-ii-ae-prior-distinguishability-report-0.2"
REPORT_VERSION = "chemworld-work-ii-ae-prior-distinguishability-report-0.3"
RECEIPT_VERSION = "chemworld-work-ii-ae-prior-distinguishability-receipt-0.2"
PARTIAL_AUDIT_VERSION = "chemworld-work-ii-ae-prior-partial-audit-0.2"
EXPECTED_TASKS = (
    "electrochemical-conversion",
    "reaction-to-crystallization",
    "reaction-to-distillation",
    "partition-discovery",
    "reaction-safety-constrained",
)
EXPECTED_PHASES = ("construction", "heldout_qualification")
EXPECTED_NOTE_PATH = (
    "workstreams/flagship_tasks/WORK_II_AE_PRIOR_DISTINGUISHABILITY_V02_EXPERIMENT_NOTE.md"
)
EXPECTED_NOTE_SHA256 = "e470fd2d3191d6d7ed2a44cc1573152c48429c90a958aea2317d18ac5222b98e"
RELEASE_EXECUTION_PROTOCOL_VERSION = "chemworld-work-ii-ae-prior-release-execution-protocol-0.1"
RUNTIME_ENVIRONMENT_FINGERPRINT_VERSION = "chemworld-work-ii-ae-runtime-environment-fingerprint-0.1"
RELEASE_ATTEMPT_BINDING_VERSION = "chemworld-work-ii-release-attempt-binding-0.1"
RELEASE_EXPERIMENT_ID = "work-ii-ae-prior-qualification-v0.2"
RELEASE_CANONICAL_OUTPUT_PATH = "runs/work-ii-release/work-ii-ae-prior-qualification-v0.2"
EXPECTED_POLICY = {
    "policy_id": "blind-two-anchor-four-category-sweep-v0.2",
    "inputs": ["task_id", "target_field"],
    "forbidden_inputs": [
        "target_pair",
        "descriptor_permutation",
        "observations",
        "outcomes",
        "metric_values",
        "favorable_region",
    ],
    "policy_replicates_per_world": 3,
    "rounds_per_policy_replicate": 8,
    "minimum_unique_recipes_per_policy_replicate": 6,
    "planned_unique_recipes_per_policy_replicate": 8,
    "category_order_by_anchor": [[0, 1, 2, 3], [0, 1, 2, 3]],
    "nuisance_design": {
        "algorithm": "sha256_hash_uniform_complement_v1",
        "namespace": "work-ii-ae-prior-v0.2-nuisance-coverage-20260812",
        "lower_bound": 0.15,
        "upper_bound": 0.85,
        "anchor_count": 2,
        "round_decimal_places": 9,
    },
}

EXPECTED_NOISE = {
    "mode": "keyed",
    "covariance_between_distinct_recipe_executions": 0.0,
    "seed_namespace": "work-ii-ae-prior-v0.2-independent-observation-20260812",
    "seed_coordinate_fields": [
        "phase",
        "task_id",
        "world_seed",
        "policy_replicate",
        "nuisance_anchor",
        "target_category",
    ],
    "left_right_seed_and_namespace_must_differ": True,
    "replicates_per_anchor_category": 3,
    "contrast_standard_error": (
        "sqrt(sample_variance_ddof1_left/3 + sample_variance_ddof1_right/3)"
    ),
}
EXPECTED_THRESHOLDS = {
    "minimum_mean_support_separation": 0.05,
    "minimum_single_support_metric_separation": 0.03,
    "minimum_support_signal_to_noise_ratio": 2.0,
    "all_allowed_metrics_finite_in_unit_interval": True,
    "all_primary_executions_completed": True,
    "all_tolerance_zero_exact_replays_verified": True,
    "both_nuisance_anchors_must_pass": True,
    "all_five_heldout_worlds_per_task_must_pass": True,
    "all_five_tasks_must_pass": True,
}
EXPECTED_DENOMINATORS = {
    "tasks": 5,
    "task_worlds_total": 50,
    "construction_task_worlds": 25,
    "heldout_qualification_task_worlds": 25,
    "policy_replicates_total": 150,
    "primary_executions_total": 1200,
    "construction_primary_executions": 600,
    "heldout_qualification_primary_executions": 600,
    "tolerance_zero_exact_replay_checks": 1200,
}
EXPECTED_CONSTRUCTION_SEEDS = {
    "electrochemical-conversion": [672326802, 263752154, 254732618, 553482792, 789741083],
    "reaction-to-crystallization": [128467214, 876914043, 166055883, 964375871, 218451485],
    "reaction-to-distillation": [897463930, 294959649, 617827675, 102623012, 705786312],
    "partition-discovery": [958536734, 274543076, 887544358, 579145448, 328656968],
    "reaction-safety-constrained": [709004002, 312314252, 762339748, 247136763, 930008953],
}
EXPECTED_HELDOUT_SEEDS = {
    "electrochemical-conversion": [934334899, 222130288, 187256385, 779398037, 533253734],
    "reaction-to-crystallization": [981471142, 371545319, 680821974, 854364962, 297088702],
    "reaction-to-distillation": [439344905, 353545270, 305419816, 301573033, 510396964],
    "partition-discovery": [595257646, 913561854, 392161417, 114255949, 308641243],
    "reaction-safety-constrained": [581283898, 413319517, 311564803, 267586659, 854968543],
}
EXPECTED_TASK_SPECS = (
    {
        "task_id": "electrochemical-conversion",
        "campaign_config": "configs/benchmark/work_ii_campaign_pilot.json",
        "campaign_config_sha256": (
            "5b3dd3d6c6e9933b6dc1974ad32076508ca06ba020277607e783a820aaf0fd24"
        ),
        "target_field": "solvent",
        "descriptor_permutation": [0, 3, 2, 1],
        "support_metric_ids": ["selective_product_yield", "energy_efficiency"],
        "negative_control_metric_ids": ["safety_risk"],
    },
    {
        "task_id": "reaction-to-crystallization",
        "campaign_config": "configs/benchmark/work_ii_crystallization_campaign.json",
        "campaign_config_sha256": (
            "571035bc2acef138a76ed220f05a9759697eb31d0e3fc4132aee04a6bbebe5d5"
        ),
        "target_field": "solvent",
        "descriptor_permutation": [0, 3, 2, 1],
        "support_metric_ids": ["crystal_yield", "crystal_csd_quality"],
        "negative_control_metric_ids": ["crystal_purity"],
    },
    {
        "task_id": "reaction-to-distillation",
        "campaign_config": "configs/benchmark/work_ii_distillation_campaign.json",
        "campaign_config_sha256": (
            "8faebc2f892f8da2ee764ef6ec12d889aa96d9d440ea4ab5eaa1df007fbb7bf2"
        ),
        "target_field": "solvent",
        "descriptor_permutation": [0, 3, 2, 1],
        "support_metric_ids": ["distillate_purity", "distillate_recovery"],
        "negative_control_metric_ids": ["solvent_loss", "score"],
    },
    {
        "task_id": "partition-discovery",
        "campaign_config": "configs/benchmark/work_ii_partition_campaign.json",
        "campaign_config_sha256": (
            "0acea167934582b40b274b1b9c156e0f3b94188bde657c0d8500c2914ae6831b"
        ),
        "target_field": "extractant",
        "descriptor_permutation": [3, 1, 2, 0],
        "support_metric_ids": ["product_in_organic"],
        "negative_control_metric_ids": ["phase_ratio", "product_in_aqueous"],
    },
    {
        "task_id": "reaction-safety-constrained",
        "campaign_config": "configs/benchmark/work_ii_safety_campaign.json",
        "campaign_config_sha256": (
            "e45a048388496c95b1fc66574a802a0ca4d6e86a04f6dca455a82d18668772b2"
        ),
        "target_field": "catalyst",
        "descriptor_permutation": [0, 2, 1, 3],
        "support_metric_ids": ["yield", "selectivity", "safety_risk"],
        "negative_control_metric_ids": ["score"],
    },
)

RELEASE_EXECUTION_REQUIRED_PATHS = (
    "configs/benchmark/work_ii_ae_prior_distinguishability_v0.2.json",
    *tuple(str(row["campaign_config"]) for row in EXPECTED_TASK_SPECS),
    "configs/benchmark/resource_limits.json",
    "configs/scenarios",
    "configs/mechanisms",
    EXPECTED_NOTE_PATH,
    "pyproject.toml",
    "uv.lock",
    "scripts/run_work_ii_ae_prior_qualification_v02.py",
    "src/chemworld",
)
RELEASE_EXECUTION_PROTOCOL = {
    "schema_version": RELEASE_EXECUTION_PROTOCOL_VERSION,
    "science_contract_version": CONTRACT_VERSION,
    "science_contract_remains_development_only": True,
    "release_mode_reexecutes_identical_scientific_contract": True,
    "required_execution_surface_paths": list(RELEASE_EXECUTION_REQUIRED_PATHS),
}


class AEPriorQualificationV02Error(ValueError):
    """Raised when the v0.2 frozen design or evidence is malformed."""


def runtime_environment_fingerprint() -> dict[str, Any]:
    """Return the lightweight runtime identity that can affect numerical evidence."""

    payload: dict[str, Any] = {
        "schema_version": RUNTIME_ENVIRONMENT_FINGERPRINT_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {package: metadata.version(package) for package in ("numpy", "scipy")},
    }
    payload["fingerprint_sha256"] = canonical_json_sha256(payload)
    return payload


def release_attempt_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one canonical A-E attempt bound to a release freeze."""

    freeze_id = manifest.get("freeze_id")
    if not isinstance(freeze_id, str) or re.fullmatch(r"[0-9a-f]{64}", freeze_id) is None:
        raise AEPriorQualificationV02Error(
            "release manifest lacks a valid freeze ID for its A-E attempt"
        )
    identity = {
        "schema_version": RELEASE_ATTEMPT_BINDING_VERSION,
        "experiment_id": RELEASE_EXPERIMENT_ID,
        "freeze_id": freeze_id,
        "canonical_output_root": RELEASE_CANONICAL_OUTPUT_PATH,
        "single_use": True,
    }
    attempt_id = canonical_json_sha256(identity)
    return {
        **identity,
        "attempt_id": attempt_id,
        "canonical_output_path": f"{RELEASE_CANONICAL_OUTPUT_PATH}/{attempt_id}",
    }


def bind_release_attempt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the canonical write-once A-E attempt into a release manifest."""

    bound = deepcopy(dict(manifest))
    attempts = bound.get("release_attempts")
    if attempts is None:
        attempts = {}
    if not isinstance(attempts, Mapping):
        raise AEPriorQualificationV02Error("release manifest attempts must be an object")
    attempts = deepcopy(dict(attempts))
    expected = release_attempt_binding(bound)
    existing = attempts.get(RELEASE_EXPERIMENT_ID)
    if existing is not None:
        raise AEPriorQualificationV02Error("release manifest A-E attempt is already claimed")
    attempts[RELEASE_EXPERIMENT_ID] = expected
    bound["release_attempts"] = attempts
    bound["manifest_sha256"] = release_manifest_sha256(bound)
    return bound


def _trajectory_release_errors(
    records: Sequence[Mapping[str, Any]], execution_context: Mapping[str, Any]
) -> list[str]:
    if execution_context.get("execution_mode") != ExecutionMode.RELEASE.value:
        return []
    tested_commit = execution_context.get("tested_commit")
    observed = {
        metadata.get("git_commit")
        for row in records
        for metadata in (row.get("agent_metadata"),)
        if isinstance(metadata, Mapping)
    }
    if observed != {tested_commit} or any(
        not isinstance(row.get("agent_metadata"), Mapping)
        or row["agent_metadata"].get("git_commit") != tested_commit
        for row in records
    ):
        return ["trajectory commit does not match the v0.2 release execution commit"]
    return []


def _legacy_development_plan(plan: Mapping[str, Any]) -> bool:
    """Recognize the completed pre-release-envelope development schema only."""

    return (
        plan.get("schema_version") == LEGACY_DEVELOPMENT_PLAN_VERSION
        and plan.get("development_only") is True
        and "execution_context" not in plan
        and "release_execution_protocol" not in plan
        and "release_manifest_binding" not in plan
        and "runtime_environment_fingerprint" not in plan
    )


def _plan_execution_context(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    context = plan.get("execution_context")
    return context if isinstance(context, Mapping) else {}


class _FrozenRecipeAgent(BaseAgent):
    name = "work_ii_ae_prior_distinguishability_v02_frozen_recipe"

    def __init__(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._actions = [deepcopy(dict(action)) for action in actions]

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._index = 0

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if self._index >= len(self._actions):
            raise RuntimeError("frozen v0.2 qualification recipe exhausted")
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
        raise AEPriorQualificationV02Error(f"{path} must contain an object")
    return value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _contained_path(root: Path, relative: object, *, must_exist: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AEPriorQualificationV02Error("artifact path must be nonempty and relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise AEPriorQualificationV02Error("artifact path escapes its evidence root")
    if must_exist and not resolved.is_file():
        raise AEPriorQualificationV02Error(f"artifact file is missing: {relative}")
    return resolved


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _heldout_seed(namespace: str, task_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{task_id}:{index}".encode()).digest()
    return 100_000_000 + int.from_bytes(digest[:8], "big") % 900_000_000


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
        raise AEPriorQualificationV02Error(
            f"unsupported target coordinate: {task_id}.{target_field}"
        ) from error


def _moved_pair(permutation: Sequence[Any]) -> tuple[int, int]:
    values = [int(value) for value in permutation]
    moved = [index for index, source in enumerate(values) if index != source]
    if (
        len(values) != 4
        or len(moved) != 2
        or values[moved[0]] != moved[1]
        or values[moved[1]] != moved[0]
    ):
        raise AEPriorQualificationV02Error(
            "descriptor_permutation must be exactly one four-category transposition"
        )
    return moved[0], moved[1]


def _nuisance_vectors(
    task_id: str,
    target_coordinate: int,
    dimension: int,
    nuisance_design: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Create two frozen outcome-blind complementary nuisance anchors."""

    namespace = str(nuisance_design["namespace"])
    lower = float(nuisance_design["lower_bound"])
    upper = float(nuisance_design["upper_bound"])
    digits = int(nuisance_design["round_decimal_places"])
    anchor = np.empty(dimension, dtype=float)
    for coordinate in range(dimension):
        digest = hashlib.sha256(f"{namespace}:{task_id}:coordinate-{coordinate}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64)
        anchor[coordinate] = round(lower + (upper - lower) * unit, digits)
    complement = np.asarray(
        [round(lower + upper - float(value), digits) for value in anchor],
        dtype=float,
    )
    # The policy controls this coordinate categorically, so its nuisance value is unused.
    anchor[target_coordinate] = 0.5
    complement[target_coordinate] = 0.5
    return anchor, complement


def build_blind_policy_schedule(
    *,
    task_id: str,
    target_field: str,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the eight recipes without accepting a target pair or any outcomes."""

    task_info = get_task(task_id).to_dict()
    target_coordinate = _target_coordinate(task_id, target_field)
    dimension = task_recipe_dimension(task_info)
    anchors = _nuisance_vectors(
        task_id,
        target_coordinate,
        dimension,
        policy["nuisance_design"],
    )
    category_orders = policy["category_order_by_anchor"]
    schedule: list[dict[str, Any]] = []
    for anchor_index, categories in enumerate(category_orders):
        for category in categories:
            vector = anchors[anchor_index].copy()
            vector[target_coordinate] = (int(category) + 0.5) / 4.0
            recipe = task_recipe_from_unit_vector(task_info, vector)
            schedule.append(
                {
                    "round_index": len(schedule),
                    "nuisance_anchor": anchor_index,
                    "target_category": int(category),
                    "target_coordinate": target_coordinate,
                    "recipe_id": f"{task_id}:anchor-{anchor_index}:category-{category}",
                    "recipe": recipe,
                }
            )
    return schedule


def validate_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_scalar = {
        "schema_version": CONTRACT_VERSION,
        "contract_id": "work-ii-ae-prior-distinguishability-v0.2",
        "status": "design_frozen_before_execution",
        "development_only": True,
        "experiment_note": EXPECTED_NOTE_PATH,
        "experiment_note_sha256": EXPECTED_NOTE_SHA256,
        "participant_provider_calls": 0,
        "participant_outcomes_read": False,
    }
    for key, expected in expected_scalar.items():
        if contract.get(key) != expected:
            errors.append(f"semantic contract field changed: {key}")
    try:
        note_path = _contained_path(root, contract.get("experiment_note"))
        if file_sha256(note_path) != contract.get("experiment_note_sha256"):
            errors.append("v0.2 experiment note hash mismatch")
    except AEPriorQualificationV02Error as error:
        errors.append(str(error))
    if contract.get("policy") != EXPECTED_POLICY:
        errors.append("blind policy semantic contract changed")
    policy = contract.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    if contract.get("noise") != EXPECTED_NOISE:
        errors.append("independent noise semantic contract changed")
    if contract.get("thresholds") != EXPECTED_THRESHOLDS:
        errors.append("scientific threshold semantic contract changed")
    if contract.get("denominators") != EXPECTED_DENOMINATORS:
        errors.append("v0.2 denominators are not exactly frozen")

    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or tasks != list(EXPECTED_TASK_SPECS):
        errors.append("v0.2 task/support/control semantic contract changed")
        tasks = []
    for row in tasks:
        try:
            config_path = _contained_path(root, row.get("campaign_config"))
        except AEPriorQualificationV02Error as error:
            errors.append(str(error))
            continue
        config = _load_object(config_path)
        if canonical_json_sha256(config) != row.get("campaign_config_sha256"):
            errors.append(f"campaign config hash mismatch: {row.get('task_id')}")
        allowed = build_checkpoint_contract(config, "aligned_nominal")["allowed_metric_ids"]
        support = row.get("support_metric_ids")
        controls = row.get("negative_control_metric_ids")
        if (
            not isinstance(support, list)
            or not support
            or not isinstance(controls, list)
            or set(support) & set(controls)
            or set(support) | set(controls) != set(allowed)
        ):
            errors.append(
                f"support/control metrics do not partition allowed metrics: {row.get('task_id')}"
            )
        try:
            _moved_pair(row.get("descriptor_permutation", []))
            schedule = build_blind_policy_schedule(
                task_id=str(row["task_id"]),
                target_field=str(row["target_field"]),
                policy=policy,
            )
            if len(schedule) != 8 or len({item["recipe_id"] for item in schedule}) != 8:
                errors.append(f"blind schedule is not eight unique recipes: {row.get('task_id')}")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid task policy contract {row.get('task_id')}: {error}")

    cohorts = contract.get("cohorts")
    cohorts = cohorts if isinstance(cohorts, Mapping) else {}
    expected_cohorts = {
        "construction": {
            "role": "frozen_descriptive_construction_only",
            "scientific_admission_denominator": False,
            "results_may_change_v0_2_rules": False,
            "task_world_seeds": EXPECTED_CONSTRUCTION_SEEDS,
        },
        "heldout_qualification": {
            "role": "only_scientific_admission_denominator",
            "scientific_admission_denominator": True,
            "selection_algorithm": "sha256_first8_modulo_namespace_v1",
            "selection_namespace": ("work-ii-ae-prior-v0.2-heldout-qualification-20260812"),
            "namespace_start": 100_000_000,
            "namespace_size": 900_000_000,
            "worlds_per_task": 5,
            "task_world_seeds": EXPECTED_HELDOUT_SEEDS,
        },
    }
    if cohorts != expected_cohorts:
        errors.append("construction/held-out cohort semantic contract changed")
    seed_sets: dict[str, set[int]] = {}
    for phase in EXPECTED_PHASES:
        cohort = cohorts.get(phase)
        cohort = cohort if isinstance(cohort, Mapping) else {}
        by_task = cohort.get("task_world_seeds")
        by_task = by_task if isinstance(by_task, Mapping) else {}
        flat: set[int] = set()
        for task_id in EXPECTED_TASKS:
            seeds = by_task.get(task_id)
            if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
                errors.append(f"{phase} does not freeze five unique worlds for {task_id}")
                continue
            flat.update(int(seed) for seed in seeds)
        seed_sets[phase] = flat
    if seed_sets.get("construction", set()) & seed_sets.get("heldout_qualification", set()):
        errors.append("construction and held-out worlds overlap")
    heldout = cohorts.get("heldout_qualification")
    heldout = heldout if isinstance(heldout, Mapping) else {}
    namespace = str(heldout.get("selection_namespace", ""))
    by_task = heldout.get("task_world_seeds")
    by_task = by_task if isinstance(by_task, Mapping) else {}
    for task_id in EXPECTED_TASKS:
        expected = [_heldout_seed(namespace, task_id, index) for index in range(5)]
        if by_task.get(task_id) != expected:
            errors.append(f"held-out namespace derivation mismatch: {task_id}")
    construction = cohorts.get("construction")
    construction = construction if isinstance(construction, Mapping) else {}
    if construction.get("results_may_change_v0_2_rules") is not False:
        errors.append("construction results are allowed to change v0.2 rules")
    if construction.get("scientific_admission_denominator") is not False:
        errors.append("construction is incorrectly an admission denominator")
    if heldout.get("scientific_admission_denominator") is not True:
        errors.append("held-out is not the only admission denominator")
    return errors


def _build_plan_payload(
    root: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    release_manifest_binding: Mapping[str, Any] | None,
    *,
    legacy_development: bool = False,
) -> dict[str, Any]:
    policy = contract["policy"]
    noise_namespace = str(contract["noise"]["seed_namespace"])
    tasks = {str(row["task_id"]): row for row in contract["tasks"]}
    task_bindings: list[dict[str, Any]] = []
    allowed_by_task: dict[str, list[str]] = {}
    for task_id in EXPECTED_TASKS:
        task_row = tasks[task_id]
        config_path = _contained_path(root, task_row["campaign_config"])
        config = _load_object(config_path)
        allowed = list(build_checkpoint_contract(config, "aligned_nominal")["allowed_metric_ids"])
        allowed_by_task[task_id] = allowed
        task_bindings.append(
            {
                "task_id": task_id,
                "campaign_config": str(task_row["campaign_config"]),
                "campaign_config_sha256": canonical_json_sha256(config),
                "target_field": str(task_row["target_field"]),
                "allowed_metric_ids": allowed,
                "support_metric_ids": list(task_row["support_metric_ids"]),
                "negative_control_metric_ids": list(task_row["negative_control_metric_ids"]),
            }
        )

    executions: list[dict[str, Any]] = []
    for phase in EXPECTED_PHASES:
        seeds_by_task = contract["cohorts"][phase]["task_world_seeds"]
        for task_id in EXPECTED_TASKS:
            task_row = tasks[task_id]
            schedule = build_blind_policy_schedule(
                task_id=task_id,
                target_field=str(task_row["target_field"]),
                policy=policy,
            )
            for world_seed in seeds_by_task[task_id]:
                for policy_replicate in range(3):
                    for item in schedule:
                        coordinate = (
                            phase,
                            task_id,
                            int(world_seed),
                            policy_replicate,
                            item["nuisance_anchor"],
                            item["target_category"],
                        )
                        coordinate_text = ":".join(str(value) for value in coordinate)
                        execution_id = f"v0.2:{coordinate_text}"
                        observation_namespace = f"{noise_namespace}:{coordinate_text}"
                        execution = {
                            "execution_index": len(executions),
                            "execution_id": execution_id,
                            "phase": phase,
                            "task_id": task_id,
                            "world_seed": int(world_seed),
                            "policy_replicate": policy_replicate,
                            "round_index": int(item["round_index"]),
                            "nuisance_anchor": int(item["nuisance_anchor"]),
                            "target_category": int(item["target_category"]),
                            "target_field": str(task_row["target_field"]),
                            "target_coordinate": int(item["target_coordinate"]),
                            "recipe_id": str(item["recipe_id"]),
                            "allowed_metric_ids": allowed_by_task[task_id],
                            "support_metric_ids": list(task_row["support_metric_ids"]),
                            "negative_control_metric_ids": list(
                                task_row["negative_control_metric_ids"]
                            ),
                            "observation_seed": _stable_seed(noise_namespace, *coordinate),
                            "observation_noise_namespace": observation_namespace,
                            "recipe": item["recipe"],
                            "recipe_sha256": canonical_json_sha256(item["recipe"]),
                        }
                        executions.append(execution)
    plan = {
        "schema_version": (LEGACY_DEVELOPMENT_PLAN_VERSION if legacy_development else PLAN_VERSION),
        "development_only": (
            True if legacy_development else execution_context.get("execution_mode") != "release"
        ),
        "contract_binding": {
            "path": contract_path.relative_to(root).as_posix(),
            "canonical_sha256": canonical_json_sha256(contract),
        },
        "experiment_note_binding": {
            "path": str(contract["experiment_note"]),
            "sha256": str(contract["experiment_note_sha256"]),
        },
        "participant_provider_calls": 0,
        "participant_outcomes_read": False,
        "denominators": deepcopy(contract["denominators"]),
        "task_bindings": task_bindings,
        "executions": executions,
    }
    if not legacy_development:
        plan["execution_context"] = dict(execution_context)
        if execution_context.get("execution_mode") == ExecutionMode.RELEASE.value:
            plan.update(
                {
                    "release_execution_protocol": deepcopy(RELEASE_EXECUTION_PROTOCOL),
                    "runtime_environment_fingerprint": runtime_environment_fingerprint(),
                    "release_manifest_binding": (
                        dict(release_manifest_binding)
                        if release_manifest_binding is not None
                        else None
                    ),
                }
            )
    plan["plan_sha256"] = _self_hash(plan, "plan_sha256")
    return plan


def build_qualification_plan(
    root: Path,
    contract_path: Path,
    *,
    execution_context: Mapping[str, Any] | None = None,
    release_manifest_path: Path | None = None,
    release_output_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = _load_object(contract_path)
    errors = validate_contract(root, contract)
    if errors:
        raise AEPriorQualificationV02Error("invalid v0.2 contract: " + "; ".join(errors))
    if execution_context is None:
        execution_context = build_execution_envelope(
            prepare_execution_context(root, mode=ExecutionMode.DEVELOPMENT)
        )
    release_manifest_binding = _release_manifest_binding(
        root, execution_context, release_manifest_path, release_output_root
    )
    plan = _build_plan_payload(
        root,
        contract_path,
        contract,
        execution_context,
        release_manifest_binding,
    )
    errors = validate_qualification_plan(root, plan, contract)
    if errors:
        raise AEPriorQualificationV02Error("invalid generated plan: " + "; ".join(errors))
    return plan


def validate_qualification_plan(
    root: Path, plan: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[str]:
    errors = validate_contract(root, contract)
    if errors:
        return errors
    binding = plan.get("contract_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    try:
        contract_path = _contained_path(root, binding.get("path"))
    except AEPriorQualificationV02Error as error:
        errors.append(str(error))
        return errors
    disk_contract = _load_object(contract_path)
    if disk_contract != dict(contract):
        errors.append("plan contract binding does not match supplied contract")
    if binding.get("canonical_sha256") != canonical_json_sha256(contract):
        errors.append("plan contract canonical hash mismatch")
    if plan.get("plan_sha256") != _self_hash(plan, "plan_sha256"):
        errors.append("plan self-hash mismatch")
    legacy_development = _legacy_development_plan(plan)
    if plan.get("schema_version") not in {
        PLAN_VERSION,
        LEGACY_DEVELOPMENT_PLAN_VERSION,
    }:
        errors.append("unexpected v0.2 plan schema")
    execution_context = _plan_execution_context(plan)
    release_mode = False
    if not legacy_development:
        errors.extend(validate_execution_envelope(root, execution_context))
        release_mode = execution_context.get("execution_mode") == "release"
        if plan.get("development_only") is release_mode:
            errors.append("plan development/release boundary is inconsistent")
        if release_mode:
            if plan.get("release_execution_protocol") != RELEASE_EXECUTION_PROTOCOL:
                errors.append("v0.2 plan release execution protocol changed")
        elif any(
            field in plan
            for field in (
                "release_execution_protocol",
                "runtime_environment_fingerprint",
                "release_manifest_binding",
            )
        ):
            errors.append("development plan contains release-only bindings")
    if plan.get("participant_provider_calls") != 0:
        errors.append("plan permits provider calls")
    if plan.get("participant_outcomes_read") is not False:
        errors.append("plan permits participant outcomes")
    if (
        release_mode
        and plan.get("runtime_environment_fingerprint") != runtime_environment_fingerprint()
    ):
        errors.append("plan runtime environment fingerprint is stale")
    if plan.get("denominators") != contract.get("denominators"):
        errors.append("plan denominators differ from contract")
    note_binding = plan.get("experiment_note_binding")
    if note_binding != {
        "path": contract.get("experiment_note"),
        "sha256": contract.get("experiment_note_sha256"),
    }:
        errors.append("plan experiment-note binding mismatch")
    release_binding = plan.get("release_manifest_binding")
    release_binding = release_binding if isinstance(release_binding, Mapping) else None
    release_manifest_path: Path | None = None
    if legacy_development:
        release_binding = None
    elif release_mode:
        if release_binding is None:
            errors.append("release plan lacks its release manifest binding")
        else:
            try:
                release_manifest_path = _contained_path(root, release_binding.get("path"))
                expected_release_binding = _release_manifest_binding(
                    root,
                    execution_context,
                    release_manifest_path,
                    None,
                )
                if dict(release_binding) != expected_release_binding:
                    errors.append("release manifest binding is stale or malformed")
            except (AEPriorQualificationV02Error, OSError, ValueError) as error:
                errors.append(f"release manifest binding is invalid: {error}")
    elif "release_manifest_binding" in plan:
        errors.append("development plan unexpectedly binds a release manifest")
    expected = _build_plan_payload(
        root,
        contract_path,
        contract,
        execution_context,
        release_binding,
        legacy_development=legacy_development,
    )
    if dict(plan) != expected:
        errors.append("plan does not exactly reconstruct from frozen contract")
        return errors
    executions = plan.get("executions")
    if not isinstance(executions, list) or len(executions) != 1200:
        errors.append("plan must contain exactly 1200 primary executions")
        return errors
    ids: set[str] = set()
    seeds: set[int] = set()
    namespaces: set[str] = set()
    grouped: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    phase_counts: dict[str, int] = defaultdict(int)
    for row in executions:
        if not isinstance(row, Mapping):
            errors.append("execution row is not an object")
            continue
        execution_id = str(row.get("execution_id", ""))
        seed = int(row.get("observation_seed", -1))
        namespace = str(row.get("observation_noise_namespace", ""))
        if execution_id in ids or seed in seeds or namespace in namespaces:
            errors.append("execution id, noise seed, or noise namespace is duplicated")
        ids.add(execution_id)
        seeds.add(seed)
        namespaces.add(namespace)
        phase = str(row.get("phase", ""))
        phase_counts[phase] += 1
        key = (
            phase,
            str(row.get("task_id", "")),
            int(row.get("world_seed", -1)),
            int(row.get("policy_replicate", -1)),
        )
        grouped[key].append(row)
        if set(row.get("support_metric_ids", [])) & set(row.get("negative_control_metric_ids", [])):
            errors.append(f"support/control overlap: {execution_id}")
        if set(row.get("support_metric_ids", [])) | set(
            row.get("negative_control_metric_ids", [])
        ) != set(row.get("allowed_metric_ids", [])):
            errors.append(f"support/control coverage mismatch: {execution_id}")
        if row.get("recipe_sha256") != canonical_json_sha256(row.get("recipe")):
            errors.append(f"recipe hash mismatch: {execution_id}")
    if dict(phase_counts) != {"construction": 600, "heldout_qualification": 600}:
        errors.append("plan cohort execution counts are not 600/600")
    if len(grouped) != 150:
        errors.append("plan must contain 150 policy replicates")
    for key, rows in grouped.items():
        if (
            len(rows) != 8
            or {int(row["round_index"]) for row in rows} != set(range(8))
            or len({str(row["recipe_id"]) for row in rows}) != 8
            or {(int(row["nuisance_anchor"]), int(row["target_category"])) for row in rows}
            != {(anchor, category) for anchor in range(2) for category in range(4)}
        ):
            errors.append(f"blind eight-round coverage mismatch: {key}")
    return errors


def _release_manifest_binding(
    root: Path,
    execution_context: Mapping[str, Any],
    release_manifest_path: Path | None,
    release_output_root: Path | None,
) -> dict[str, Any] | None:
    release_mode = execution_context.get("execution_mode") == ExecutionMode.RELEASE.value
    if not release_mode:
        if release_manifest_path is not None:
            raise AEPriorQualificationV02Error("development plan must not bind a release manifest")
        return None
    if release_manifest_path is None:
        raise AEPriorQualificationV02Error(
            "release plan requires its validated release manifest path"
        )
    path = release_manifest_path.resolve()
    try:
        relative = path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise AEPriorQualificationV02Error(
            "release manifest path escapes the repository"
        ) from error
    manifest = _load_object(path)
    context = prepare_execution_context(
        root,
        mode=ExecutionMode.RELEASE,
        release_manifest=path,
    )
    if dict(execution_context) != build_execution_envelope(context):
        raise AEPriorQualificationV02Error(
            "release execution context differs from the bound release manifest"
        )
    coverage_errors = _release_surface_coverage_errors(root, manifest)
    if coverage_errors:
        raise AEPriorQualificationV02Error("; ".join(coverage_errors))
    attempt = release_attempt_binding(manifest)
    attempts = manifest.get("release_attempts")
    attempts = attempts if isinstance(attempts, Mapping) else {}
    if attempts.get(RELEASE_EXPERIMENT_ID) != attempt:
        raise AEPriorQualificationV02Error(
            "release manifest lacks the canonical write-once A-E attempt"
        )
    expected_output = _contained_path(
        root,
        attempt["canonical_output_path"],
        must_exist=False,
    )
    if release_output_root is not None and release_output_root.resolve() != expected_output:
        raise AEPriorQualificationV02Error(
            "release output differs from the canonical A-E attempt path"
        )
    return {
        "path": relative,
        "file_sha256": file_sha256(path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "freeze_id": manifest.get("freeze_id"),
        "tested_commit": manifest.get("tested_commit"),
        "execution_protocol_sha256": canonical_json_sha256(RELEASE_EXECUTION_PROTOCOL),
        "required_execution_surface_paths": list(RELEASE_EXECUTION_REQUIRED_PATHS),
        "validated_execution_surface_roots": list(
            manifest.get("execution_surface", {}).get("relative_roots", [])
        ),
        "attempt": attempt,
    }


def _release_surface_coverage_errors(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Require the release freeze to cover every A-E execution dependency."""

    surface = manifest.get("execution_surface")
    surface = surface if isinstance(surface, Mapping) else {}
    raw_roots = surface.get("relative_roots")
    if not isinstance(raw_roots, list) or not all(isinstance(item, str) for item in raw_roots):
        return ["A-E release manifest lacks canonical execution-surface roots"]
    repository = root.resolve()
    covered_roots = {
        (repository / item).resolve().relative_to(repository).as_posix() for item in raw_roots
    }
    required_roots = set(RELEASE_EXECUTION_REQUIRED_PATHS)
    errors: list[str] = []
    for relative in RELEASE_EXECUTION_REQUIRED_PATHS:
        if relative not in covered_roots:
            errors.append("A-E release execution surface does not cover required path: " + relative)
    unexpected = sorted(covered_roots - required_roots)
    if unexpected:
        errors.append(
            "A-E release execution surface contains non-required paths: " + ", ".join(unexpected)
        )
    return errors


def _config_for_task(root: Path, plan: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    matches = [row for row in plan["task_bindings"] if row.get("task_id") == task_id]
    if len(matches) != 1:
        raise AEPriorQualificationV02Error(f"{task_id} lacks one task binding")
    config_path = _contained_path(root, matches[0]["campaign_config"])
    config = _load_object(config_path)
    if canonical_json_sha256(config) != matches[0].get("campaign_config_sha256"):
        raise AEPriorQualificationV02Error(f"{task_id} campaign config binding is stale")
    return config


def execute_one(
    root: Path,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Execute and tolerance-zero replay one provider-free recipe."""

    task_id = str(row["task_id"])
    config = _config_for_task(root, plan, task_id)
    execution_root = output_root / "executions" / str(row["execution_index"])
    execution_root.mkdir(parents=True, exist_ok=False)
    trajectory_path = execution_root / "trajectory.jsonl"
    actions = row["recipe"]["steps"]
    receipt = deepcopy(dict(row))
    receipt.update(
        {
            "schema_version": RECEIPT_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "provider_call_count": 0,
        }
    )
    try:
        run_agent(
            env_id=get_task(task_id).env_id,
            agent=_FrozenRecipeAgent(actions),
            world_split=str(config["world_split"]),
            budget=len(actions),
            objective=str(config["objective"]),
            seed=int(row["world_seed"]),
            observation_seed=int(row["observation_seed"]),
            task_id=task_id,
            output_path=trajectory_path,
            budget_override=len(actions),
            episode_mode_override="single_experiment",
            electrochemical_material_family_id=config.get("electrochemical_material_family_id"),
            crystallization_material_family_id=config.get("crystallization_material_family_id"),
            electrochemical_workflow_mode=config.get("electrochemical_workflow_mode"),
            scoring_contract_id=config.get("scoring_contract_id"),
            observation_noise_mode="keyed",
            observation_noise_namespace=str(row["observation_noise_namespace"]),
            world_interventions=config.get("world_interventions", []),
        )
        records = load_jsonl(trajectory_path)
        release_errors = _trajectory_release_errors(records, _plan_execution_context(plan))
        if release_errors:
            raise AEPriorQualificationV02Error("; ".join(release_errors))
        if [record.get("action") for record in records] != actions:
            raise AEPriorQualificationV02Error("trajectory differs from frozen recipe")
        if any(record.get("transaction_status") != "committed" for record in records):
            raise AEPriorQualificationV02Error("physical execution contains a noncommitted action")
        final_rows = [
            record
            for record in records
            if record.get("instrument") == "final_assay"
            and record.get("transaction_status") == "committed"
        ]
        if len(final_rows) != 1:
            raise AEPriorQualificationV02Error("execution lacks exactly one committed final assay")
        observation = trajectory_value(final_rows[0], "observation", None)
        observation = observation if isinstance(observation, Mapping) else {}
        metrics: dict[str, float] = {}
        for metric_id in row["allowed_metric_ids"]:
            value = observation.get(metric_id)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise AEPriorQualificationV02Error(f"missing allowed metric {metric_id}")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise AEPriorQualificationV02Error(
                    f"allowed metric {metric_id} is outside finite [0,1] bounds"
                )
            metrics[str(metric_id)] = number
        replay = verify_records(
            records,
            tolerance=0.0,
            world_interventions=config.get("world_interventions", []),
        ).to_dict()
        if replay.get("verified") is not True:
            raise AEPriorQualificationV02Error("tolerance-zero exact replay did not verify")
        receipt.update(
            {
                "status": "completed",
                "allowed_metrics": metrics,
                "support_metrics": {
                    metric: metrics[metric] for metric in row["support_metric_ids"]
                },
                "negative_control_metrics": {
                    metric: metrics[metric] for metric in row["negative_control_metric_ids"]
                },
                "exact_replay": replay,
                "trajectory": {
                    "path": trajectory_path.relative_to(output_root).as_posix(),
                    "sha256": file_sha256(trajectory_path),
                },
                "failure": None,
            }
        )
    except Exception as error:
        receipt.update(
            {
                "status": "failed",
                "allowed_metrics": None,
                "support_metrics": None,
                "negative_control_metrics": None,
                "exact_replay": None,
                "trajectory": (
                    trajectory_path.relative_to(output_root).as_posix()
                    if trajectory_path.is_file()
                    else None
                ),
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        if trajectory_path.is_file():
            receipt["trajectory"] = {
                "path": trajectory_path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(trajectory_path),
            }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return receipt


def _contrast_summary(
    rows_by_category: Mapping[int, list[Mapping[str, Any]]],
    left_category: int,
    right_category: int,
    metric_ids: Sequence[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric_id in metric_ids:
        left = [float(row["allowed_metrics"][metric_id]) for row in rows_by_category[left_category]]
        right = [
            float(row["allowed_metrics"][metric_id]) for row in rows_by_category[right_category]
        ]
        standard_error = math.sqrt(variance(left) / 3.0 + variance(right) / 3.0)
        contrast = fmean(right) - fmean(left)
        metrics[metric_id] = {
            "left_mean": fmean(left),
            "right_mean": fmean(right),
            "contrast": contrast,
            "absolute_contrast": abs(contrast),
            "welch_standard_error": standard_error,
        }
    return metrics


def build_qualification_report(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze both cohorts while admitting only universal held-out success."""

    if len(receipts) != 1200:
        raise AEPriorQualificationV02Error("report requires exactly 1200 receipts")
    receipt_by_id = {str(row.get("execution_id")): row for row in receipts}
    if len(receipt_by_id) != 1200:
        raise AEPriorQualificationV02Error("receipt IDs are missing or duplicated")
    plan_by_id = {str(row["execution_id"]): row for row in plan["executions"]}
    if set(receipt_by_id) != set(plan_by_id):
        raise AEPriorQualificationV02Error("receipt coverage differs from plan")
    task_contract = {str(row["task_id"]): row for row in contract["tasks"]}
    thresholds = contract["thresholds"]

    execution_failures: list[dict[str, Any]] = []
    valid_receipts: dict[str, Mapping[str, Any]] = {}
    for execution_id, row in receipt_by_id.items():
        planned = plan_by_id[execution_id]
        metadata_matches = all(row.get(key) == value for key, value in planned.items())
        allowed = row.get("allowed_metrics")
        metrics_valid = (
            isinstance(allowed, Mapping)
            and set(allowed) == set(planned["allowed_metric_ids"])
            and all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in allowed.values()
            )
        )
        replay = row.get("exact_replay")
        replay_ok = isinstance(replay, Mapping) and replay.get("verified") is True
        support = row.get("support_metrics")
        controls = row.get("negative_control_metrics")
        valid = (
            metadata_matches
            and row.get("schema_version") == RECEIPT_VERSION
            and row.get("plan_sha256") == plan.get("plan_sha256")
            and row.get("receipt_sha256") == _self_hash(row, "receipt_sha256")
            and row.get("provider_call_count") == 0
            and row.get("status") == "completed"
            and metrics_valid
            and support == {metric: allowed[metric] for metric in planned["support_metric_ids"]}
            and controls
            == {metric: allowed[metric] for metric in planned["negative_control_metric_ids"]}
            and replay_ok
        )
        if valid:
            valid_receipts[execution_id] = row
        else:
            execution_failures.append(
                {
                    "level": "execution",
                    "phase": planned["phase"],
                    "execution_id": execution_id,
                    "reason": "incomplete_invalid_metric_or_exact_replay_failure",
                }
            )

    reachability_results: list[dict[str, Any]] = []
    anchor_results: list[dict[str, Any]] = []
    world_results: list[dict[str, Any]] = []
    all_failures = list(execution_failures)
    for phase in EXPECTED_PHASES:
        for task_id in EXPECTED_TASKS:
            task_row = task_contract[task_id]
            left_category, right_category = _moved_pair(task_row["descriptor_permutation"])
            seeds = contract["cohorts"][phase]["task_world_seeds"][task_id]
            for world_seed in seeds:
                world_plan = [
                    row
                    for row in plan["executions"]
                    if row["phase"] == phase
                    and row["task_id"] == task_id
                    and row["world_seed"] == world_seed
                ]
                world_receipts = [
                    valid_receipts[row["execution_id"]]
                    for row in world_plan
                    if row["execution_id"] in valid_receipts
                ]
                reach_rows: list[dict[str, Any]] = []
                for policy_replicate in range(3):
                    replicate_rows = [
                        row for row in world_receipts if row["policy_replicate"] == policy_replicate
                    ]
                    visited = {
                        (int(row["nuisance_anchor"]), int(row["target_category"]))
                        for row in replicate_rows
                    }
                    unique_recipes = len(
                        {
                            plan_by_id[str(row["execution_id"])]["recipe_id"]
                            for row in replicate_rows
                        }
                    )
                    reached_pair_both_anchors = all(
                        (anchor, category) in visited
                        for anchor in range(2)
                        for category in (left_category, right_category)
                    )
                    passed = (
                        len(replicate_rows) == 8
                        and unique_recipes >= 6
                        and reached_pair_both_anchors
                    )
                    reach = {
                        "phase": phase,
                        "task_id": task_id,
                        "world_seed": world_seed,
                        "policy_replicate": policy_replicate,
                        "completed_rounds": len(replicate_rows),
                        "unique_recipes": unique_recipes,
                        "reached_hidden_pair_at_both_anchors": reached_pair_both_anchors,
                        "passed": passed,
                    }
                    reachability_results.append(reach)
                    reach_rows.append(reach)
                    if not passed:
                        all_failures.append(
                            {
                                "level": "reachability",
                                "phase": phase,
                                "task_id": task_id,
                                "world_seed": world_seed,
                                "policy_replicate": policy_replicate,
                            }
                        )

                world_anchor_rows: list[dict[str, Any]] = []
                for anchor in range(2):
                    anchor_receipts = [
                        row for row in world_receipts if row["nuisance_anchor"] == anchor
                    ]
                    by_category: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
                    for row in anchor_receipts:
                        by_category[int(row["target_category"])].append(row)
                    complete = len(anchor_receipts) == 12 and all(
                        len(by_category[category]) == 3 for category in range(4)
                    )
                    support_metrics: dict[str, Any] = {}
                    control_metrics: dict[str, Any] = {}
                    if complete:
                        support_metrics = _contrast_summary(
                            by_category,
                            left_category,
                            right_category,
                            task_row["support_metric_ids"],
                        )
                        control_metrics = _contrast_summary(
                            by_category,
                            left_category,
                            right_category,
                            task_row["negative_control_metric_ids"],
                        )
                    separations = [
                        float(row["absolute_contrast"]) for row in support_metrics.values()
                    ]
                    standard_errors = [
                        float(row["welch_standard_error"]) for row in support_metrics.values()
                    ]
                    mean_support = fmean(separations) if separations else 0.0
                    max_support = max(separations, default=0.0)
                    rms_uncertainty: float | None = (
                        math.sqrt(fmean(value * value for value in standard_errors))
                        if standard_errors
                        else None
                    )
                    snr = (
                        mean_support / max(rms_uncertainty, 1.0e-12)
                        if rms_uncertainty is not None
                        else 0.0
                    )
                    passed = (
                        complete
                        and mean_support >= float(thresholds["minimum_mean_support_separation"])
                        and max_support
                        >= float(thresholds["minimum_single_support_metric_separation"])
                        and snr >= float(thresholds["minimum_support_signal_to_noise_ratio"])
                    )
                    anchor_result = {
                        "phase": phase,
                        "task_id": task_id,
                        "world_seed": world_seed,
                        "nuisance_anchor": anchor,
                        "completed_executions": len(anchor_receipts),
                        "support_metric_ids": list(task_row["support_metric_ids"]),
                        "negative_control_metric_ids": list(
                            task_row["negative_control_metric_ids"]
                        ),
                        "support_metric_results": support_metrics,
                        "negative_control_metric_results": control_metrics,
                        "mean_support_separation": mean_support,
                        "maximum_support_separation": max_support,
                        "support_contrast_rms_standard_error": rms_uncertainty,
                        "support_signal_to_noise_ratio": snr,
                        "passed": passed,
                    }
                    anchor_results.append(anchor_result)
                    world_anchor_rows.append(anchor_result)
                    if not passed:
                        all_failures.append(
                            {
                                "level": "anchor",
                                "phase": phase,
                                "task_id": task_id,
                                "world_seed": world_seed,
                                "nuisance_anchor": anchor,
                            }
                        )
                world_passed = all(row["passed"] for row in reach_rows) and all(
                    row["passed"] for row in world_anchor_rows
                )
                world_result = {
                    "phase": phase,
                    "task_id": task_id,
                    "world_seed": world_seed,
                    "passed_policy_replicates": sum(int(row["passed"]) for row in reach_rows),
                    "passed_nuisance_anchors": sum(int(row["passed"]) for row in world_anchor_rows),
                    "passed": world_passed,
                }
                world_results.append(world_result)
                if not world_passed:
                    all_failures.append(
                        {
                            "level": "world",
                            "phase": phase,
                            "task_id": task_id,
                            "world_seed": world_seed,
                        }
                    )

    task_results: list[dict[str, Any]] = []
    for task_id in EXPECTED_TASKS:
        construction_worlds = [
            row
            for row in world_results
            if row["phase"] == "construction" and row["task_id"] == task_id
        ]
        heldout_worlds = [
            row
            for row in world_results
            if row["phase"] == "heldout_qualification" and row["task_id"] == task_id
        ]
        task_results.append(
            {
                "task_id": task_id,
                "construction_passed_worlds": sum(
                    int(row["passed"]) for row in construction_worlds
                ),
                "construction_status": (
                    "passed" if all(row["passed"] for row in construction_worlds) else "failed"
                ),
                "heldout_passed_worlds": sum(int(row["passed"]) for row in heldout_worlds),
                "heldout_status": (
                    "passed" if all(row["passed"] for row in heldout_worlds) else "failed"
                ),
                "admission_passed": all(row["passed"] for row in heldout_worlds),
            }
        )
    heldout_passed = all(row["admission_passed"] for row in task_results)
    legacy_development = _legacy_development_plan(plan)
    report = {
        "schema_version": (
            LEGACY_DEVELOPMENT_REPORT_VERSION if legacy_development else REPORT_VERSION
        ),
        "development_only": bool(plan["development_only"]),
        "status": "passed" if heldout_passed else "failed",
        "admission_basis": "heldout_qualification_only",
        "construction_can_change_v0_2_rules": False,
        "plan_sha256": plan["plan_sha256"],
        "denominators": deepcopy(plan["denominators"]),
        "completed_primary_executions": len(valid_receipts),
        "verified_tolerance_zero_exact_replays": sum(
            int(
                isinstance(row.get("exact_replay"), Mapping)
                and row["exact_replay"].get("verified") is True
            )
            for row in receipts
        ),
        "reachability_results": reachability_results,
        "anchor_results": anchor_results,
        "world_results": world_results,
        "task_results": task_results,
        "failures": all_failures,
        "receipt_bindings": [
            {
                "execution_index": int(row["execution_index"]),
                "execution_id": str(row["execution_id"]),
                "receipt_sha256": str(row.get("receipt_sha256", "")),
                "trajectory_sha256": (
                    str(row["trajectory"].get("sha256", ""))
                    if isinstance(row.get("trajectory"), Mapping)
                    else None
                ),
            }
            for row in sorted(receipts, key=lambda item: int(item["execution_index"]))
        ],
    }
    if not legacy_development:
        report["execution_context"] = deepcopy(_plan_execution_context(plan))
    if _plan_execution_context(plan).get("execution_mode") == ExecutionMode.RELEASE.value:
        report["runtime_environment_fingerprint"] = deepcopy(
            plan["runtime_environment_fingerprint"]
        )
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def validate_qualification_report(
    root: Path,
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[str]:
    errors = validate_qualification_plan(root, plan, contract)
    expected_report_version = (
        LEGACY_DEVELOPMENT_REPORT_VERSION if _legacy_development_plan(plan) else REPORT_VERSION
    )
    if report.get("schema_version") != expected_report_version:
        errors.append("unexpected v0.2 report schema")
    if report.get("report_sha256") != _self_hash(report, "report_sha256"):
        errors.append("v0.2 report self-hash mismatch")
    expected = build_qualification_report(plan, receipts, contract)
    if dict(report) != expected:
        errors.append("v0.2 report does not reproduce from plan and receipts")
    if report.get("admission_basis") != "heldout_qualification_only":
        errors.append("report admission basis is not held-out only")
    if report.get("construction_can_change_v0_2_rules") is not False:
        errors.append("report permits construction-driven rule changes")
    if (
        _plan_execution_context(plan).get("execution_mode") == ExecutionMode.RELEASE.value
        and report.get("runtime_environment_fingerprint") != runtime_environment_fingerprint()
    ):
        errors.append("report runtime environment fingerprint is stale")
    return errors


def _metrics_from_records(
    records: Sequence[Mapping[str, Any]], metric_ids: Sequence[str]
) -> dict[str, float]:
    final_rows = [
        row
        for row in records
        if row.get("instrument") == "final_assay" and row.get("transaction_status") == "committed"
    ]
    if len(final_rows) != 1:
        raise AEPriorQualificationV02Error("trajectory lacks exactly one committed final assay")
    observation = trajectory_value(final_rows[0], "observation", None)
    if not isinstance(observation, Mapping):
        raise AEPriorQualificationV02Error("final assay observation is missing")
    metrics: dict[str, float] = {}
    for metric_id in metric_ids:
        value = observation.get(metric_id)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise AEPriorQualificationV02Error(f"trajectory is missing allowed metric {metric_id}")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise AEPriorQualificationV02Error(
                f"trajectory metric {metric_id} is outside finite [0,1]"
            )
        metrics[metric_id] = number
    return metrics


def _audit_disk_receipt(
    root: Path,
    output_root: Path,
    plan: Mapping[str, Any],
    planned: Mapping[str, Any],
    receipt_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        receipt_path = receipt_path.resolve()
        if not receipt_path.is_relative_to(output_root.resolve()):
            raise AEPriorQualificationV02Error("receipt path escapes output root")
        receipt = _load_object(receipt_path)
        if receipt.get("receipt_sha256") != _self_hash(receipt, "receipt_sha256"):
            errors.append("receipt self-hash mismatch")
        if receipt.get("schema_version") != RECEIPT_VERSION:
            errors.append("receipt schema mismatch")
        if receipt.get("plan_sha256") != plan.get("plan_sha256"):
            errors.append("receipt plan binding mismatch")
        for key, value in planned.items():
            if receipt.get(key) != value:
                errors.append(f"receipt immutable plan field mismatch: {key}")
        if receipt.get("provider_call_count") != 0:
            errors.append("receipt provider call count is nonzero")
        if receipt.get("status") != "completed":
            errors.append("receipt is not completed")
            return receipt, errors
        trajectory = receipt.get("trajectory")
        if not isinstance(trajectory, Mapping):
            errors.append("completed receipt lacks trajectory binding")
            return receipt, errors
        try:
            trajectory_path = _contained_path(output_root, trajectory.get("path"))
        except AEPriorQualificationV02Error as error:
            errors.append(str(error))
            return receipt, errors
        if file_sha256(trajectory_path) != trajectory.get("sha256"):
            errors.append("trajectory SHA-256 mismatch")
            return receipt, errors
        records = load_jsonl(trajectory_path)
        errors.extend(_trajectory_release_errors(records, _plan_execution_context(plan)))
        if [row.get("action") for row in records] != planned["recipe"]["steps"]:
            errors.append("trajectory actions differ from frozen recipe")
        if any(row.get("transaction_status") != "committed" for row in records):
            errors.append("trajectory contains a noncommitted action")
        config = _config_for_task(root, plan, str(planned["task_id"]))
        replay = verify_records(
            records,
            tolerance=0.0,
            world_interventions=config.get("world_interventions", []),
        ).to_dict()
        if replay.get("verified") is not True:
            errors.append("fresh tolerance-zero exact replay failed")
        metrics = _metrics_from_records(records, planned["allowed_metric_ids"])
        support = {metric: metrics[metric] for metric in planned["support_metric_ids"]}
        controls = {metric: metrics[metric] for metric in planned["negative_control_metric_ids"]}
        if receipt.get("allowed_metrics") != metrics:
            errors.append("receipt allowed metrics differ from trajectory")
        if receipt.get("support_metrics") != support:
            errors.append("receipt support metrics differ from trajectory")
        if receipt.get("negative_control_metrics") != controls:
            errors.append("receipt control metrics differ from trajectory")
        if receipt.get("exact_replay") != replay:
            errors.append("receipt replay summary differs from fresh replay")
    except (AEPriorQualificationV02Error, OSError, ValueError) as error:
        return None, [str(error)]
    return receipt, errors


def validate_qualification_output(root: Path, output_root: Path, contract_path: Path) -> list[str]:
    """Reopen and independently validate a complete output directory."""

    root = root.resolve()
    output_root = output_root.resolve()
    errors: list[str] = []
    try:
        contract = _load_object(contract_path.resolve())
        plan_path = _contained_path(output_root, "plan.json")
        report_path = _contained_path(output_root, "report.json")
        disk_plan = _load_object(plan_path)
        disk_report = _load_object(report_path)
    except (AEPriorQualificationV02Error, OSError, ValueError) as error:
        return [str(error)]
    errors.extend(validate_qualification_plan(root, disk_plan, contract))
    planned_rows = disk_plan.get("executions")
    if not isinstance(planned_rows, list) or len(planned_rows) != 1200:
        return [*errors, "disk plan does not contain 1200 executions"]
    receipts: list[dict[str, Any]] = []
    for planned in planned_rows:
        index = int(planned["execution_index"])
        receipt_path = output_root / "receipts" / f"{index:04d}.json"
        receipt, receipt_errors = _audit_disk_receipt(
            root, output_root, disk_plan, planned, receipt_path
        )
        errors.extend(f"execution {index}: {error}" for error in receipt_errors)
        if receipt is not None:
            receipts.append(receipt)
    receipt_files = sorted((output_root / "receipts").glob("*.json"))
    if len(receipt_files) != 1200:
        errors.append("disk receipt denominator is not exactly 1200")
    if len(receipts) == 1200 and not errors:
        expected_report = build_qualification_report(disk_plan, receipts, contract)
        if disk_report != expected_report:
            errors.append("disk report differs from fresh trajectory-derived report")
        errors.extend(
            validate_qualification_report(root, disk_report, disk_plan, receipts, contract)
        )
    return errors


def validate_formal_qualification_output(
    root: Path, report_path: Path, contract_path: Path
) -> list[str]:
    """Validate a complete on-disk v0.2 report and require release eligibility."""

    root = root.resolve()
    report_path = report_path.resolve()
    try:
        report_path.relative_to(root)
    except ValueError:
        return ["A-E v0.2 report path escapes the repository"]
    if report_path.name != "report.json":
        return ["A-E v0.2 formal validator requires the canonical report.json"]
    output_root = report_path.parent
    errors: list[str] = []
    try:
        errors.extend(
            validate_qualification_output(root, output_root, contract_path.resolve())
        )
    except Exception as error:  # fail closed on arbitrarily damaged disk evidence
        errors.append(
            "A-E v0.2 formal output validation failed closed: "
            f"{type(error).__name__}: {error}"
        )
    if not report_path.is_file():
        return [*errors, "A-E v0.2 formal report is missing"]
    try:
        report = _load_object(report_path)
    except (AEPriorQualificationV02Error, OSError, TypeError, ValueError) as error:
        errors.append(
            "A-E v0.2 formal report is unreadable: "
            f"{type(error).__name__}: {error}"
        )
        return sorted(set(errors))
    context = report.get("execution_context")
    context = context if isinstance(context, Mapping) else {}
    errors.extend(validate_execution_envelope(root, context))
    if report.get("runtime_environment_fingerprint") != runtime_environment_fingerprint():
        errors.append("A-E v0.2 formal runtime environment fingerprint changed")
    try:
        plan = _load_object(output_root / "plan.json")
        release_binding = plan.get("release_manifest_binding")
        release_binding = release_binding if isinstance(release_binding, Mapping) else {}
        attempt = release_binding.get("attempt")
        attempt = attempt if isinstance(attempt, Mapping) else {}
        expected_output = _contained_path(
            root,
            attempt.get("canonical_output_path"),
            must_exist=False,
        )
        if output_root != expected_output:
            errors.append("A-E v0.2 formal report is not at its canonical attempt path")
    except (AEPriorQualificationV02Error, OSError, TypeError, ValueError):
        errors.append("A-E v0.2 formal report lacks its canonical attempt path")
    if (
        report.get("development_only") is not False
        or context.get("execution_mode") != ExecutionMode.RELEASE.value
        or context.get("release_eligible") is not True
        or context.get("c2_admission_authorized") is not True
    ):
        errors.append("A-E v0.2 qualification is not release-authorized")
    failures = report.get("failures")
    failures = failures if isinstance(failures, list) else [None]
    if report.get("status") != "passed" or any(
        not isinstance(failure, Mapping) or failure.get("phase") != "construction"
        for failure in failures
    ):
        errors.append("A-E v0.2 held-out qualification did not pass cleanly")
    return sorted(set(errors))


def build_partial_audit(root: Path, output_root: Path, contract_path: Path) -> dict[str, Any]:
    """Return a readable fail-closed audit of an interrupted run; never resumes it."""

    root = root.resolve()
    output_root = output_root.resolve()
    contract = _load_object(contract_path.resolve())
    plan_path = _contained_path(output_root, "plan.json")
    plan = _load_object(plan_path)
    plan_errors = validate_qualification_plan(root, plan, contract)
    receipt_dir = output_root / "receipts"
    receipt_files = sorted(receipt_dir.glob("*.json")) if receipt_dir.is_dir() else []
    valid = 0
    completed = 0
    failed = 0
    audit_errors = list(plan_errors)
    seen_indexes: set[int] = set()
    for receipt_path in receipt_files:
        try:
            index = int(receipt_path.stem)
        except ValueError:
            audit_errors.append(f"unexpected receipt filename: {receipt_path.name}")
            continue
        if index not in range(1200) or index in seen_indexes:
            audit_errors.append(f"invalid or duplicate receipt index: {index}")
            continue
        seen_indexes.add(index)
        planned = plan["executions"][index]
        receipt, errors = _audit_disk_receipt(root, output_root, plan, planned, receipt_path)
        audit_errors.extend(f"execution {index}: {error}" for error in errors)
        if receipt is not None:
            completed += int(receipt.get("status") == "completed")
            failed += int(receipt.get("status") == "failed")
            valid += int(not errors)
    payload = {
        "schema_version": PARTIAL_AUDIT_VERSION,
        "status": "complete" if len(receipt_files) == 1200 else "interrupted",
        "resume_allowed": False,
        "planned_primary_executions": 1200,
        "materialized_receipts": len(receipt_files),
        "completed_receipts": completed,
        "failed_receipts": failed,
        "independently_valid_receipts": valid,
        "missing_receipts": 1200 - len(seen_indexes),
        "plan_valid": not plan_errors,
        "errors": audit_errors,
    }
    payload["partial_audit_sha256"] = _self_hash(payload, "partial_audit_sha256")
    return payload


def markdown_summary(report: Mapping[str, Any]) -> str:
    execution_context = report.get("execution_context")
    execution_context = execution_context if isinstance(execution_context, Mapping) else {}
    lines = [
        "# Work II A-E prior distinguishability v0.2",
        "",
        (
            f"Status: **{report['status']}** "
            f"({execution_context.get('execution_mode', 'development')} mode)"
        ),
        "",
        "The admission decision uses held-out qualification worlds only; construction "
        "failures are retained below and cannot change v0.2 rules.",
        "",
        f"Primary executions: {report['completed_primary_executions']}/1200; "
        f"tolerance-zero exact replays: {report['verified_tolerance_zero_exact_replays']}/1200.",
        "",
        "| Task | Construction worlds | Held-out worlds | Admission |",
        "|---|---:|---:|---|",
    ]
    for row in report["task_results"]:
        lines.append(
            f"| {row['task_id']} | {row['construction_passed_worlds']}/5 | "
            f"{row['heldout_passed_worlds']}/5 | {row['heldout_status']} |"
        )
    lines.extend(
        [
            "",
            f"Retained failures: {len(report['failures'])}.",
            "",
            "Support and negative/control metric contrasts and Welch standard errors are "
            "reported separately in report.json.",
        ]
    )
    return "\n".join(lines) + "\n"


def execute_qualification(
    root: Path,
    contract_path: Path,
    output_root: Path,
    *,
    execution_mode: ExecutionMode | str = ExecutionMode.DEVELOPMENT,
    release_manifest: Path | None = None,
) -> dict[str, Any]:
    """Execute one fixed provider-free block in development or release mode."""

    root = root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite v0.2 output: {output_root}")
    resolved_mode = ExecutionMode(execution_mode)
    release_manifest_path = release_manifest.resolve() if release_manifest is not None else None
    if resolved_mode is ExecutionMode.DEVELOPMENT and release_manifest_path is not None:
        raise ValueError("development mode must not bind a release manifest")
    contract = _load_object(contract_path.resolve())
    contract_errors = validate_contract(root, contract)
    if contract_errors:
        raise AEPriorQualificationV02Error("invalid v0.2 contract: " + "; ".join(contract_errors))
    if resolved_mode is ExecutionMode.RELEASE:
        if release_manifest_path is None:
            raise ValueError("release mode requires a release manifest")
        manifest = _load_object(release_manifest_path)
        attempt = release_attempt_binding(manifest)
        expected_output = _contained_path(
            root,
            attempt["canonical_output_path"],
            must_exist=False,
        )
        if output_root != expected_output:
            raise AEPriorQualificationV02Error(
                "release output differs from the canonical A-E attempt path"
            )
    context = prepare_execution_context(
        root,
        mode=resolved_mode,
        release_manifest=release_manifest_path,
    )
    plan = build_qualification_plan(
        root,
        contract_path.resolve(),
        execution_context=build_execution_envelope(context),
        release_manifest_path=release_manifest_path,
        release_output_root=(output_root if resolved_mode is ExecutionMode.RELEASE else None),
    )
    output_root.mkdir(parents=True)
    write_json_atomic(output_root / "plan.json", plan)
    receipts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in plan["executions"]:
        receipt = execute_one(root, plan, row, output_root)
        receipts.append(receipt)
        write_json_atomic(output_root / "receipts" / f"{row['execution_index']:04d}.json", receipt)
        completed = len(receipts)
        elapsed = time.perf_counter() - started
        throughput = completed / elapsed if elapsed else 0.0
        eta = (1200 - completed) / throughput if throughput else None
        print(
            json.dumps(
                {
                    "stage": f"ae_prior_v02_{row['phase']}",
                    "completed": completed,
                    "total": 1200,
                    "throughput_executions_per_minute": round(throughput * 60.0, 3),
                    "eta_s": round(eta, 3) if eta is not None else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    report = build_qualification_report(plan, receipts, contract)
    errors = validate_qualification_report(root, report, plan, receipts, contract)
    if errors:
        raise AEPriorQualificationV02Error("v0.2 report validation failed: " + "; ".join(errors))
    write_json_atomic(output_root / "report.json", report)
    (output_root / "summary.md").write_text(markdown_summary(report), encoding="utf-8")
    disk_errors = validate_qualification_output(root, output_root, contract_path)
    if disk_errors:
        raise AEPriorQualificationV02Error(
            "v0.2 disk evidence validation failed: " + "; ".join(disk_errors)
        )
    return report


__all__ = [
    "CONTRACT_VERSION",
    "LEGACY_DEVELOPMENT_PLAN_VERSION",
    "LEGACY_DEVELOPMENT_REPORT_VERSION",
    "PARTIAL_AUDIT_VERSION",
    "PLAN_VERSION",
    "RECEIPT_VERSION",
    "RELEASE_ATTEMPT_BINDING_VERSION",
    "RELEASE_CANONICAL_OUTPUT_PATH",
    "RELEASE_EXECUTION_PROTOCOL",
    "RELEASE_EXECUTION_PROTOCOL_VERSION",
    "RELEASE_EXECUTION_REQUIRED_PATHS",
    "RELEASE_EXPERIMENT_ID",
    "REPORT_VERSION",
    "RUNTIME_ENVIRONMENT_FINGERPRINT_VERSION",
    "AEPriorQualificationV02Error",
    "bind_release_attempt",
    "build_blind_policy_schedule",
    "build_partial_audit",
    "build_qualification_plan",
    "build_qualification_report",
    "execute_one",
    "execute_qualification",
    "markdown_summary",
    "release_attempt_binding",
    "runtime_environment_fingerprint",
    "validate_contract",
    "validate_formal_qualification_output",
    "validate_qualification_output",
    "validate_qualification_plan",
    "validate_qualification_report",
]
