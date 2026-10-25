"""Provider-free materialization contract for five-seed Work II A-S D1.

This module turns the two locked W2-37 seed-0 D1 templates into a static
parent/child schedule.  It deliberately does not authorize or launch a
provider and does not create release or audit artifacts.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AS_D1_PRODUCTION_VERSION = "chemworld-work-ii-as-d1-production-plan-0.1"
AS_D1_CHILD_VERSION = "chemworld-work-ii-as-d1-campaign-child-0.1"
AS_D1_Q2_PACKAGE_VERSION = "chemworld-work-ii-constitutive-structural-q2-package-0.1"
AS_D1_WORLD_SEEDS = tuple(range(5))
AS_D1_ARMS = frozenset({"opaque", "aligned_nominal", "misindexed_nominal"})
AS_D1_EXPERIMENTS_PER_CELL = 12
AS_D1_CHECKPOINTS = [0, 3, 6, 9, 12]
AS_D1_PARENT_ID = "work-ii-as-d1-five-seed-production"
AS_D1_Q2_PACKAGE = Path(
    "configs/benchmark/work_ii_as_paired_law_q2_package_v0.1.json"
)
AS_D1_TASK_SPECS = {
    "reaction-to-crystallization": {
        "candidate_id": "crystallization_reversible_topology",
        "source": "configs/benchmark/work_ii_as_crystallization_d1_v0.1.json",
        "slug": "crystallization",
    },
    "partition-discovery": {
        "candidate_id": "partition_power_response",
        "source": "configs/benchmark/work_ii_as_partition_d1_v0.1.json",
        "slug": "partition",
    },
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative_inside(root: Path, path: Path, *, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside the repository")
    return resolved.relative_to(root.resolve()).as_posix()


def _candidate_rows(package: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    candidates = package.get("candidate_laws")
    if (
        package.get("schema_version") != AS_D1_Q2_PACKAGE_VERSION
        or package.get("provider_call_count") != 0
        or package.get("all_five_world_cohorts_passed") is not True
        or not isinstance(candidates, Mapping)
    ):
        raise ValueError("W2-37 Q2 package is not a provider-free five-world pass")
    observed = {str(key): value for key, value in candidates.items()}
    expected = {str(spec["candidate_id"]) for spec in AS_D1_TASK_SPECS.values()}
    if set(observed) != expected or any(
        not isinstance(value, Mapping) for value in observed.values()
    ):
        raise ValueError("W2-37 Q2 package does not contain exactly the two locked candidates")
    return observed  # type: ignore[return-value]


def _validate_candidate(
    package: Mapping[str, Any],
    *,
    task_id: str,
    candidate_id: str,
) -> Mapping[str, Any]:
    candidate = _candidate_rows(package)[candidate_id]
    evidence = candidate.get("world_evidence")
    rows = evidence if isinstance(evidence, list) else []
    seeds = {
        row.get("world_seed")
        for row in rows
        if isinstance(row, Mapping) and row.get("passed") is True
    }
    if (
        candidate.get("task_id") != task_id
        or len(rows) != len(AS_D1_WORLD_SEEDS)
        or seeds != set(AS_D1_WORLD_SEEDS)
        or any(not isinstance(row, Mapping) or row.get("passed") is not True for row in rows)
        or len(candidate.get("outcome_blind_q2_queries", [])) != 16
    ):
        raise ValueError(f"W2-37 candidate is not a complete five-world pass: {candidate_id}")
    return candidate


def _validate_source(
    source: Mapping[str, Any],
    package: Mapping[str, Any],
    *,
    task_id: str,
    candidate_id: str,
) -> None:
    campaign = source.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    qualification = source.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    execution_context = source.get("execution_context")
    execution_context = execution_context if isinstance(execution_context, Mapping) else {}
    intervention = source.get("intervention")
    intervention = intervention if isinstance(intervention, Mapping) else {}
    candidate = _validate_candidate(
        package,
        task_id=task_id,
        candidate_id=candidate_id,
    )
    checkpoint = source.get("belief_checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    analysis = source.get("analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    metric_ids = analysis.get("final_metric_ids")
    registered_queries = candidate.get("outcome_blind_q2_queries")
    registered_queries = registered_queries if isinstance(registered_queries, list) else []
    expected_queries = [
        {
            "feature_values": copy.deepcopy(query.get("feature_values")),
            "intervention_family": query.get("intervention_family"),
            "metric_ids": copy.deepcopy(metric_ids),
            "q2_coordinate_sha256": query.get("coordinate_sha256"),
            "query_id": query.get("coordinate_id"),
        }
        for query in registered_queries
        if isinstance(query, Mapping)
    ]
    if (
        source.get("task_id") != task_id
        or source.get("world_seed") != 0
        or intervention.get("candidate_id") != candidate_id
        or set(source.get("prior_arms", {})) != AS_D1_ARMS
        or campaign.get("complete_experiments") != AS_D1_EXPERIMENTS_PER_CELL
        or campaign.get("checkpoint_complete_experiments") != AS_D1_CHECKPOINTS
        or source.get("snapshot_stages")
        != [
            "pre_evidence",
            "after_experiment_3",
            "after_experiment_6",
            "after_experiment_9",
            "final",
        ]
        or checkpoint.get("held_out_queries") != expected_queries
        or qualification.get("q0_q1_q2_passed") is not True
        or qualification.get("q2_passed") is not True
        or qualification.get("execution_authorized") is not False
        or qualification.get("formal_r5_authorized") is not False
        or source.get("formal_result") is not False
        or execution_context.get("execution_mode") != "development"
        or execution_context.get("release_eligible") is not False
        or execution_context.get("c2_admission_authorized") is not False
        or qualification.get("q2_package_sha256") != package.get("package_sha256")
    ):
        raise ValueError(f"locked A-S D1 source contract drifted: {task_id}")


def _child_path(output_directory: Path, *, task_slug: str, seed: int) -> Path:
    return output_directory / f"work-ii-as-{task_slug}-d1-seed{seed}.json"


def _build_child(
    source: Mapping[str, Any],
    *,
    task_id: str,
    candidate_id: str,
    source_path: str,
    seed: int,
    child_path: str,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(source))
    namespace = f"{AS_D1_PARENT_ID}--{candidate_id}--seed{seed}"
    config["world_seed"] = seed
    config["pilot_id"] = namespace
    config["observation_noise_namespace"] = namespace
    config["as_d1_production"] = {
        "schema_version": AS_D1_CHILD_VERSION,
        "parent_id": AS_D1_PARENT_ID,
        "task_id": task_id,
        "candidate_id": candidate_id,
        "world_seed": seed,
        "source_static_config": source_path,
        "child_config": child_path,
        "provider_call_count": 0,
        "provider_execution_authorized": False,
        "formal_result": False,
    }
    return config


def validate_as_d1_child(
    child: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    task_id: str,
    candidate_id: str,
    source_path: str,
    child_path: str,
    seed: int,
) -> list[str]:
    """Validate that a child changes only seed/namespace and parent linkage."""

    expected = _build_child(
        source,
        task_id=task_id,
        candidate_id=candidate_id,
        source_path=source_path,
        seed=seed,
        child_path=child_path,
    )
    return [] if dict(child) == expected else ["A-S D1 child differs from its locked source"]


def build_as_d1_production_materialization(
    root: Path,
    *,
    output_directory: Path,
    package_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build one provider-blocked parent plan and its ten campaign children."""

    root = root.resolve()
    output_directory = output_directory.resolve()
    output_relative = _relative_inside(
        root,
        output_directory,
        label="A-S D1 materialization output",
    )
    package_path = (package_path or root / AS_D1_Q2_PACKAGE).resolve()
    package_relative = _relative_inside(root, package_path, label="W2-37 Q2 package")
    package = _load(package_path)
    _candidate_rows(package)

    children: dict[str, dict[str, Any]] = {}
    schedule: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for task_id, spec in AS_D1_TASK_SPECS.items():
        candidate_id = str(spec["candidate_id"])
        source_path = str(spec["source"])
        source = _load(root / source_path)
        _validate_source(
            source,
            package,
            task_id=task_id,
            candidate_id=candidate_id,
        )
        source_paths.append(source_path)
        for seed in AS_D1_WORLD_SEEDS:
            path = _child_path(
                output_directory,
                task_slug=str(spec["slug"]),
                seed=seed,
            )
            child_relative = _relative_inside(root, path, label="A-S D1 child")
            child = _build_child(
                source,
                task_id=task_id,
                candidate_id=candidate_id,
                source_path=source_path,
                seed=seed,
                child_path=child_relative,
            )
            children[child_relative] = child
            schedule.append(
                {
                    "schedule_index": len(schedule),
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "world_seed": seed,
                    "campaign_child_config": child_relative,
                    "prior_arm_count": len(AS_D1_ARMS),
                    "complete_experiments_per_arm": AS_D1_EXPERIMENTS_PER_CELL,
                    "provider_execution_authorized": False,
                }
            )

    parent = {
        "schema_version": AS_D1_PRODUCTION_VERSION,
        "parent_id": AS_D1_PARENT_ID,
        "status": "ready_static_materialization_provider_execution_blocked",
        "formal_result": False,
        "provider_execution_authorized": False,
        "provider_call_count": 0,
        "selection_reads_participant_outcomes": False,
        "q2_package": package_relative,
        "source_static_configs": source_paths,
        "output_directory": output_relative,
        "world_seeds": list(AS_D1_WORLD_SEEDS),
        "task_count": len(AS_D1_TASK_SPECS),
        "campaign_child_count": len(schedule),
        "participant_cell_count": len(schedule) * len(AS_D1_ARMS),
        "complete_experiment_count": (
            len(schedule) * len(AS_D1_ARMS) * AS_D1_EXPERIMENTS_PER_CELL
        ),
        "schedule": schedule,
    }
    return parent, children


__all__ = [
    "AS_D1_ARMS",
    "AS_D1_CHECKPOINTS",
    "AS_D1_EXPERIMENTS_PER_CELL",
    "AS_D1_PARENT_ID",
    "AS_D1_PRODUCTION_VERSION",
    "AS_D1_TASK_SPECS",
    "AS_D1_WORLD_SEEDS",
    "build_as_d1_production_materialization",
    "validate_as_d1_child",
]
