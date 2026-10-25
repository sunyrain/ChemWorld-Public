"""Evaluator-only held-out truth execution for the Work II formal matrix.

The participant predicts the registered query metrics at every belief checkpoint.
This module compiles each public query into one frozen complete experiment and
executes it outside the participant session.  Truth executions make no provider
calls, emit no participant feedback, and are shared by the three prior arms in
the same task/world cluster.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.static_optimization import (
    StaticOptimizationValidator,
    compile_static_optimization_plan,
)
from chemworld.agents.task_recipes import (
    electrochemical_recipe_parameter_schema,
    task_recipe_coordinate_schema,
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
from chemworld.eval.work_ii_formal import build_checkpoint_contract
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
)
from chemworld.tasks import get_task

WORK_II_TRUTH_PLAN_VERSION = "chemworld-work-ii-evaluator-truth-plan-0.1"
WORK_II_TRUTH_REPORT_VERSION = "chemworld-work-ii-evaluator-truth-report-0.1"

# Query fields may be narrower than complete executable recipes. These defaults
# fill only controls omitted by the registered query; an explicitly frozen query
# value is authoritative and remains bound by the campaign-config hash.
_FROZEN_RECIPE_DEFAULTS: dict[str, dict[str, int | float]] = {
    "electrochemical-conversion": {},
    "reaction-to-crystallization": {
        "stirring_speed_rpm": 675.0,
        "catalyst_amount_mol": 0.000315,
    },
    "reaction-to-distillation": {
        "stirring_speed_rpm": 675.0,
        "catalyst_amount_mol": 0.000315,
        "evaporation_temperature_K": 332.5,
        "evaporation_duration_s": 900.0,
        "transfer_fraction": 0.77,
    },
    "partition-discovery": {
        "solvent_volume_L": 0.020,
    },
    "reaction-safety-constrained": {
        "stirring_speed_rpm": 675.0,
    },
}

_QUERY_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "partition-discovery": {
        "aqueous_phase_volume_L": "aqueous_volume_L",
    }
}


class _FrozenTruthReplayAgent(BaseAgent):
    name = "work_ii_frozen_evaluator_truth_replay"

    def __init__(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._frozen_actions = [deepcopy(dict(action)) for action in actions]

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._index = 0
        self._pending: dict[str, Any] | None = None

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if self._pending is not None:
            raise RuntimeError("truth replay requested a new action before its outcome")
        if self._index >= len(self._frozen_actions):
            raise RuntimeError("truth replay exhausted its frozen action plan")
        self._pending = deepcopy(self._frozen_actions[self._index])
        self._index += 1
        return deepcopy(self._pending)

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        del observation, reward, info
        if self._pending is None or action != self._pending:
            raise RuntimeError("truth replay outcome differs from its frozen action")
        self._pending = None

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "requires_online_model": False,
                "execution_role": "evaluator_held_out_truth_replay",
                "participant_feedback": False,
                "frozen_action_count": len(self._frozen_actions),
                "frozen_action_plan_sha256": canonical_json_sha256(self._frozen_actions),
            }
        )
        return manifest


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _workflow_mode(
    config: Mapping[str, Any],
    feature_values: Mapping[str, Any],
) -> str:
    if config.get("task_id") == "electrochemical-conversion":
        compiled_fields = set(
            _compiled_control_values("electrochemical-conversion", feature_values)
        )
        if compiled_fields != set(electrochemical_recipe_parameter_schema()):
            return ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
    return str(config.get("electrochemical_workflow_mode", "static_single_stage"))


def _compiled_control_values(
    task_id: str,
    feature_values: Mapping[str, Any],
) -> dict[str, int | float]:
    if task_id not in _FROZEN_RECIPE_DEFAULTS:
        raise ValueError(f"Work II truth compiler does not support task {task_id}")
    aliases = _QUERY_FIELD_ALIASES.get(task_id, {})
    values = dict(_FROZEN_RECIPE_DEFAULTS[task_id])
    for field, raw_value in feature_values.items():
        compiled_field = aliases.get(str(field), str(field))
        values[compiled_field] = raw_value
    return values


def _unit_vector(
    task_id: str,
    feature_values: Mapping[str, Any],
) -> list[float]:
    task_info = get_task(task_id).to_dict()
    values = _compiled_control_values(task_id, feature_values)
    vector: list[float] = []
    schema = task_recipe_coordinate_schema(task_info)
    expected_fields = {str(item["control_id"]) for item in schema}
    if set(values) != expected_fields:
        raise ValueError(
            "held-out query does not compile to the exact executable controls: "
            f"missing={sorted(expected_fields - set(values))}, "
            f"extra={sorted(set(values) - expected_fields)}"
        )
    for coordinate in schema:
        field = str(coordinate["control_id"])
        value = values[field]
        if coordinate.get("kind") == "categorical":
            count = int(coordinate["category_count"])
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an integer category")
            if not 0 <= value < count:
                raise ValueError(f"{field} category is outside its frozen domain")
            vector.append((float(value) + 0.5) / count)
            continue
        low, high = coordinate["physical_bounds"]
        number = _finite_number(value, field=field)
        if not float(low) <= number <= float(high):
            raise ValueError(f"{field} is outside its physical bounds")
        vector.append((number - float(low)) / (float(high) - float(low)))
    return vector


def _reaction_safety_pattern_actions(
    feature_values: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {
        "reaction_temperature_K",
        "reaction_duration_s",
        "reagent_amount_mol",
        "stirring_speed_rpm",
        "catalyst",
        "catalyst_amount_mol",
        "solvent",
        "solvent_volume_L",
    }
    if set(feature_values) != expected:
        raise ValueError(
            "reaction-safety held-out query does not match its executable pattern: "
            f"missing={sorted(expected - set(feature_values))}, "
            f"extra={sorted(set(feature_values) - expected)}"
        )
    physical = dict(feature_values)
    numeric_bounds = {
        "reaction_temperature_K": (250.0, 470.0),
        "reaction_duration_s": (1.0, 14_400.0),
        "reagent_amount_mol": (0.003, 0.030),
        "stirring_speed_rpm": (100.0, 1200.0),
        "catalyst_amount_mol": (0.00008, 0.00055),
        "solvent_volume_L": (0.005, 0.050),
    }
    for field, (low, high) in numeric_bounds.items():
        value = _finite_number(physical[field], field=field)
        if not low <= value <= high:
            raise ValueError(f"{field} is outside its reaction-safety pattern bounds")
        physical[field] = value
    for field in ("catalyst", "solvent"):
        value = physical[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
            raise ValueError(f"{field} is outside its reaction-safety categorical domain")
    actions = [
        {
            "operation": "add_solvent",
            "volume_L": physical["solvent_volume_L"],
            "solvent": physical["solvent"],
        },
        {"operation": "add_reagent", "amount_mol": physical["reagent_amount_mol"]},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": physical["catalyst_amount_mol"],
            "catalyst": physical["catalyst"],
        },
        {
            "operation": "heat",
            "target_temperature_K": physical["reaction_temperature_K"],
            "duration_s": physical["reaction_duration_s"],
            "stirring_speed_rpm": physical["stirring_speed_rpm"],
        },
        {"operation": "quench"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    return actions, {
        "schema_version": "chemworld-work-ii-reaction-safety-truth-plan-0.1",
        "recipe_contract": "work-ii-reaction-safety-q1-executable-control-envelope-0.2",
        "physical_controls": physical,
    }


def _partition_pattern_actions(
    feature_values: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile registered physical partition queries outside the agent search box.

    The phase-process transfer family intentionally includes held-out settling
    durations beyond the participant's static search-vector range.  Those values
    remain inside the runtime operation domain and must not be clipped or rejected
    merely because they test extrapolation.
    """

    expected = {
        "solvent",
        "aqueous_phase_volume_L",
        "extractant",
        "extractant_volume_L",
        "mix_duration_s",
        "settle_duration_s",
        "stirring_speed_rpm",
    }
    if set(feature_values) != expected:
        raise ValueError(
            "partition held-out query does not match its executable pattern: "
            f"missing={sorted(expected - set(feature_values))}, "
            f"extra={sorted(set(feature_values) - expected)}"
        )
    physical = dict(feature_values)
    numeric_bounds = {
        "aqueous_phase_volume_L": (1.0e-6, 0.060),
        "extractant_volume_L": (1.0e-6, 0.060),
        "mix_duration_s": (1.0, 14_400.0),
        "settle_duration_s": (1.0, 14_400.0),
        "stirring_speed_rpm": (0.0, 2000.0),
    }
    for field, (low, high) in numeric_bounds.items():
        value = _finite_number(physical[field], field=field)
        if not low <= value <= high:
            raise ValueError(f"{field} is outside its partition operation bounds")
        physical[field] = value
    for field in ("solvent", "extractant"):
        value = physical[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
            raise ValueError(f"{field} is outside its partition categorical domain")
    physical["solvent_volume_L"] = float(
        _FROZEN_RECIPE_DEFAULTS["partition-discovery"]["solvent_volume_L"]
    )
    actions = [
        {
            "operation": "add_solvent",
            "volume_L": physical["solvent_volume_L"],
            "solvent": physical["solvent"],
        },
        {
            "operation": "add_phase",
            "phase": "aqueous",
            "volume_L": physical["aqueous_phase_volume_L"],
        },
        {
            "operation": "add_extractant",
            "extractant": physical["extractant"],
            "volume_L": physical["extractant_volume_L"],
        },
        {
            "operation": "mix",
            "duration_s": physical["mix_duration_s"],
            "stirring_speed_rpm": physical["stirring_speed_rpm"],
        },
        {"operation": "settle", "duration_s": physical["settle_duration_s"]},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    return actions, {
        "schema_version": "chemworld-work-ii-partition-truth-plan-0.1",
        "recipe_contract": "work-ii-partition-physical-query-envelope-0.1",
        "physical_controls": physical,
        "participant_search_box_required": False,
    }


def compile_evaluator_truth_query(
    config: Mapping[str, Any],
    query: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one registered held-out query into an exact action plan."""

    task_id = str(config["task_id"])
    task_info = get_task(task_id).to_dict()
    feature_values = query.get("feature_values")
    if not isinstance(feature_values, Mapping):
        raise ValueError("held-out query feature_values must be an object")
    workflow_mode = _workflow_mode(config, feature_values)
    validator = StaticOptimizationValidator(
        task_info,
        electrochemical_workflow_mode=workflow_mode,
    )
    common = {
        "experiment_intent": "evaluator-only Work II held-out truth query",
        "requested_measurement_slots": list(validator.measurement_slot_ids),
        "measurement_objective": "score the frozen held-out query metrics",
        "expected_effect": "produce evaluator truth without participant feedback",
        "uncertainty": 0.0,
    }
    if task_id == "reaction-safety-constrained" and "solvent_volume_L" in feature_values:
        actions, compiled_plan = _reaction_safety_pattern_actions(feature_values)
    elif task_id == "partition-discovery":
        actions, compiled_plan = _partition_pattern_actions(feature_values)
    elif task_id in {"electrochemical-conversion", "reaction-to-crystallization"}:
        payload = {
            **common,
            "recipe_parameters": _compiled_control_values(task_id, feature_values),
        }
        plan = validator.validate(payload)
        compiled_plan = plan.to_dict()
        recipe = compile_static_optimization_plan(
            task_info,
            plan,
            electrochemical_workflow_mode=workflow_mode,
        )
        actions = [deepcopy(dict(item)) for item in recipe["steps"]]
    else:
        payload = {**common, "search_vector": _unit_vector(task_id, feature_values)}
        plan = validator.validate(payload)
        compiled_plan = plan.to_dict()
        recipe = compile_static_optimization_plan(
            task_info,
            plan,
            electrochemical_workflow_mode=workflow_mode,
        )
        actions = [deepcopy(dict(item)) for item in recipe["steps"]]
    if (
        not actions
        or actions[-1].get("operation") != "measure"
        or actions[-1].get("instrument") != "final_assay"
    ):
        raise ValueError("compiled held-out query does not end in final_assay")
    return {
        "query_id": str(query["query_id"]),
        "feature_values": dict(feature_values),
        "metric_ids": [str(item) for item in query["metric_ids"]],
        "workflow_mode": workflow_mode,
        "compiled_plan": compiled_plan,
        "compiled_plan_sha256": canonical_json_sha256(compiled_plan),
        "action_plan": actions,
        "action_plan_sha256": canonical_json_sha256(actions),
    }


def _noise_binding(
    *,
    formal_preflight_sha256: str | None,
    world_cluster_id: str,
    query_id: str,
) -> dict[str, Any]:
    scope = formal_preflight_sha256 or "development"
    payload = f"work-ii-truth-v0.1:{scope}:{world_cluster_id}:{query_id}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "observation_coordinate_sha256": digest,
        "observation_seed": int(digest[:8], 16) % 2_147_483_647,
        "observation_noise_namespace": (f"work-ii-truth-v0.1-{world_cluster_id}-{digest[:12]}"),
    }


def build_evaluator_truth_plan(
    cluster: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    formal_result: bool,
    formal_preflight_sha256: str | None,
) -> dict[str, Any]:
    """Build one pattern-owned evaluator plan shared by a task/world arm triplet."""

    if formal_result:
        if not isinstance(formal_preflight_sha256, str) or len(formal_preflight_sha256) != 64:
            raise ValueError("formal truth plan requires its preflight binding")
    elif formal_preflight_sha256 is not None:
        raise ValueError("development truth plan cannot carry a formal binding")
    task_id = str(cluster["task_id"])
    if task_id != config.get("task_id"):
        raise ValueError("truth-plan task differs from its campaign config")
    world_cluster_id = str(cluster["world_cluster_id"])
    world_seed = int(cluster["world_seed"])
    checkpoint = build_checkpoint_contract(config, "opaque")
    queries: list[dict[str, Any]] = []
    for index, raw_query in enumerate(checkpoint["held_out_queries"], start=1):
        query = compile_evaluator_truth_query(config, raw_query)
        query.update(
            {
                "execution_index": index,
                "execution_id": f"{world_cluster_id}--truth-{index:02d}",
                **_noise_binding(
                    formal_preflight_sha256=formal_preflight_sha256,
                    world_cluster_id=world_cluster_id,
                    query_id=str(raw_query["query_id"]),
                ),
            }
        )
        queries.append(query)
    plan: dict[str, Any] = {
        "schema_version": WORK_II_TRUTH_PLAN_VERSION,
        "formal_result": bool(formal_result),
        "formal_preflight_sha256": formal_preflight_sha256,
        "world_cluster_id": world_cluster_id,
        "task_id": task_id,
        "world_seed": world_seed,
        "campaign_config_sha256": canonical_json_sha256(config),
        "truth_query_count": len(queries),
        "truth_query_metric_count": sum(len(item["metric_ids"]) for item in queries),
        "evaluator_provider_call_count": 0,
        "participant_operation_denominator_impact": 0,
        "participant_feedback_allowed": False,
        "shared_across_prior_arms": True,
        "law_summary_contract": {
            "allowed_feature_ids": list(checkpoint["allowed_feature_ids"]),
            "allowed_metric_ids": list(checkpoint["allowed_metric_ids"]),
            "required_metric_ids": list(checkpoint["allowed_metric_ids"]),
            "evidence_catalog": list(checkpoint["evidence_catalog"]),
        },
        "queries": queries,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return plan


def validate_evaluator_truth_plan(plan: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != WORK_II_TRUTH_PLAN_VERSION:
        errors.append("unexpected evaluator truth plan schema")
    if plan.get("formal_result") is True:
        digest = plan.get("formal_preflight_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("formal evaluator truth plan lacks its preflight binding")
    elif plan.get("formal_preflight_sha256") is not None:
        errors.append("development evaluator truth plan carries a formal binding")
    expected_hash = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected_hash:
        errors.append("evaluator truth plan self-hash mismatch")
    queries = plan.get("queries")
    declared_query_count = plan.get("truth_query_count")
    if (
        isinstance(declared_query_count, bool)
        or not isinstance(declared_query_count, int)
        or declared_query_count < 1
    ):
        errors.append("evaluator truth plan has an invalid query denominator")
        declared_query_count = 0
    if not isinstance(queries, list) or len(queries) != declared_query_count:
        errors.append("evaluator truth plan query denominator mismatch")
        queries = []
    query_ids: list[str] = []
    metric_count = 0
    for query in queries:
        if not isinstance(query, Mapping):
            errors.append("evaluator truth plan contains a malformed query")
            continue
        query_ids.append(str(query.get("query_id")))
        metrics = query.get("metric_ids")
        if not isinstance(metrics, list) or not metrics or len(set(metrics)) != len(metrics):
            errors.append("evaluator truth query has invalid metric IDs")
        else:
            metric_count += len(metrics)
        actions = query.get("action_plan")
        if not isinstance(actions, list) or canonical_json_sha256(actions) != query.get(
            "action_plan_sha256"
        ):
            errors.append("evaluator truth action-plan binding mismatch")
        if (
            not isinstance(actions, list)
            or not actions
            or not isinstance(actions[-1], Mapping)
            or actions[-1].get("operation") != "measure"
            or actions[-1].get("instrument") != "final_assay"
        ):
            errors.append("evaluator truth action plan lacks terminal final_assay")
    if len(set(query_ids)) != len(query_ids):
        errors.append("evaluator truth query IDs are not unique")
    law_contract = plan.get("law_summary_contract")
    law_contract = law_contract if isinstance(law_contract, Mapping) else {}
    feature_ids = law_contract.get("allowed_feature_ids")
    metric_ids = law_contract.get("allowed_metric_ids")
    required_metric_ids = law_contract.get("required_metric_ids")
    evidence_catalog = law_contract.get("evidence_catalog")
    query_feature_ids = {
        str(field)
        for query in queries
        if isinstance(query, Mapping)
        for field in (
            query.get("feature_values", {}).keys()
            if isinstance(query.get("feature_values"), Mapping)
            else ()
        )
    }
    query_metric_ids = {
        str(metric)
        for query in queries
        if isinstance(query, Mapping)
        for metric in (
            query.get("metric_ids", []) if isinstance(query.get("metric_ids"), list) else ()
        )
    }
    if (
        not isinstance(feature_ids, list)
        or {str(item) for item in feature_ids} != query_feature_ids
        or not isinstance(metric_ids, list)
        or {str(item) for item in metric_ids} != query_metric_ids
        or required_metric_ids != metric_ids
        or not isinstance(evidence_catalog, list)
        or not evidence_catalog
        or evidence_catalog
        != [f"experiment-{index}-final-assay" for index in range(1, len(evidence_catalog) + 1)]
    ):
        errors.append("evaluator truth law-summary contract is invalid")
    if declared_query_count != len(queries):
        errors.append("evaluator truth query denominator mismatch")
    if plan.get("truth_query_metric_count") != metric_count:
        errors.append("evaluator truth metric denominator mismatch")
    if (
        plan.get("evaluator_provider_call_count") != 0
        or plan.get("participant_operation_denominator_impact") != 0
        or plan.get("participant_feedback_allowed") is not False
        or plan.get("shared_across_prior_arms") is not True
    ):
        errors.append("evaluator truth isolation invariant failed")
    return errors


def _truth_metrics(
    final_row: Mapping[str, Any],
    metric_ids: Sequence[str],
) -> dict[str, float]:
    observation = trajectory_value(final_row, "observation", None)
    observation = observation if isinstance(observation, Mapping) else {}
    truth: dict[str, float] = {}
    for metric_id in metric_ids:
        value = observation.get(metric_id)
        if value is None and metric_id == "score":
            value = trajectory_value(final_row, "leaderboard_score", None)
        truth[metric_id] = _finite_number(
            value,
            field=f"evaluator_truth.{metric_id}",
        )
    return truth


def execute_evaluator_truth_plan(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Execute each truth query once and retain every success or failure."""

    errors = validate_evaluator_truth_plan(plan)
    if errors:
        raise ValueError("invalid evaluator truth plan: " + "; ".join(errors))
    if plan.get("campaign_config_sha256") != canonical_json_sha256(config):
        raise ValueError("evaluator truth campaign config binding drifted")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluator truth output: {output_root}")
    output_root.mkdir(parents=True)
    write_json_atomic(output_root / "plan.json", dict(plan))
    receipts: list[dict[str, Any]] = []
    truth: dict[str, dict[str, float]] = {}
    for query in plan["queries"]:
        query_id = str(query["query_id"])
        execution_root = output_root / "queries" / query_id
        execution_root.mkdir(parents=True, exist_ok=False)
        trajectory_path = execution_root / "trajectory.jsonl"
        actions = query["action_plan"]
        receipt: dict[str, Any] = {
            "execution_index": query["execution_index"],
            "execution_id": query["execution_id"],
            "query_id": query_id,
            "metric_ids": list(query["metric_ids"]),
            "action_plan_sha256": query["action_plan_sha256"],
            "observation_coordinate_sha256": query["observation_coordinate_sha256"],
            "evaluator_provider_call_count": 0,
            "participant_operation_denominator_impact": 0,
            "participant_feedback_emitted": False,
        }
        try:
            run_agent(
                env_id=get_task(str(plan["task_id"])).env_id,
                agent=_FrozenTruthReplayAgent(actions),
                world_split=str(config["world_split"]),
                budget=len(actions),
                objective=str(config["objective"]),
                seed=int(plan["world_seed"]),
                agent_seed=0,
                observation_seed=int(query["observation_seed"]),
                task_id=str(plan["task_id"]),
                output_path=trajectory_path,
                budget_override=len(actions),
                episode_mode_override="single_experiment",
                world_interventions=config.get("world_interventions"),
                electrochemical_material_family_id=config.get("electrochemical_material_family_id"),
                crystallization_material_family_id=config.get("crystallization_material_family_id"),
                electrochemical_workflow_mode=str(query["workflow_mode"]),
                scoring_contract_id=config.get("scoring_contract_id"),
                observation_noise_mode=str(config["observation_noise_mode"]),
                observation_noise_namespace=str(query["observation_noise_namespace"]),
            )
            records = load_jsonl(trajectory_path)
            if [record.get("action") for record in records] != actions:
                raise ValueError("evaluator truth trajectory differs from its action plan")
            final_rows = [
                record
                for record in records
                if record.get("transaction_status") == "committed"
                and record.get("operation_type") == "measure"
                and record.get("instrument") == "final_assay"
            ]
            if len(final_rows) != 1:
                raise ValueError("evaluator truth trajectory lacks one final assay")
            query_truth = _truth_metrics(final_rows[0], query["metric_ids"])
            replay = verify_records(
                records,
                tolerance=0.0,
                world_interventions=config.get("world_interventions"),
            ).to_dict()
            if replay.get("verified") is not True:
                raise ValueError("evaluator truth trajectory does not replay exactly")
            truth[query_id] = query_truth
            receipt.update(
                {
                    "status": "completed",
                    "truth": query_truth,
                    "operation_attempt_count": len(records),
                    "trajectory": {
                        "path": trajectory_path.relative_to(output_root).as_posix(),
                        "sha256": file_sha256(trajectory_path),
                    },
                    "exact_replay": replay,
                }
            )
        except Exception as error:  # retain the frozen denominator without replacement
            receipt.update(
                {
                    "status": "failed",
                    "truth": None,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error)[:1000],
                    "trajectory": (
                        {
                            "path": trajectory_path.relative_to(output_root).as_posix(),
                            "sha256": file_sha256(trajectory_path),
                        }
                        if trajectory_path.is_file()
                        else None
                    ),
                }
            )
        receipts.append(receipt)
    completed = sum(item["status"] == "completed" for item in receipts)
    report: dict[str, Any] = {
        "schema_version": WORK_II_TRUTH_REPORT_VERSION,
        "formal_result": plan.get("formal_result") is True,
        "formal_preflight_sha256": plan.get("formal_preflight_sha256"),
        "plan_sha256": plan["plan_sha256"],
        "world_cluster_id": plan["world_cluster_id"],
        "task_id": plan["task_id"],
        "world_seed": plan["world_seed"],
        "status": "completed" if completed == len(receipts) else "failed",
        "truth_query_count": len(receipts),
        "completed_truth_query_count": completed,
        "failed_truth_query_count": len(receipts) - completed,
        "truth_query_metric_count": sum(len(item["metric_ids"]) for item in receipts),
        "completed_truth_query_metric_count": sum(
            len(item["metric_ids"]) for item in receipts if item["status"] == "completed"
        ),
        "evaluator_provider_call_count": 0,
        "participant_operation_denominator_impact": 0,
        "participant_feedback_emitted": False,
        "truth": truth,
        "receipts": receipts,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json_atomic(output_root / "report.json", report)
    return report


def validate_evaluator_truth_report(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[str]:
    errors = validate_evaluator_truth_plan(plan)
    if report.get("schema_version") != WORK_II_TRUTH_REPORT_VERSION:
        errors.append("unexpected evaluator truth report schema")
    expected_hash = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    if report.get("report_sha256") != expected_hash:
        errors.append("evaluator truth report self-hash mismatch")
    if report.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("evaluator truth report plan binding mismatch")
    expected_query_count = plan.get("truth_query_count")
    if (
        isinstance(expected_query_count, bool)
        or not isinstance(expected_query_count, int)
        or expected_query_count < 1
    ):
        errors.append("evaluator truth report plan denominator is invalid")
        expected_query_count = 0
    receipts = report.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != expected_query_count:
        errors.append("evaluator truth report receipt denominator mismatch")
        receipts = []
    completed = [
        receipt
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("status") == "completed"
    ]
    if report.get("completed_truth_query_count") != len(completed):
        errors.append("evaluator truth completed denominator mismatch")
    if report.get("truth_query_count") != expected_query_count:
        errors.append("evaluator truth report query denominator mismatch")
    if report.get("status") == "completed" and len(completed) != expected_query_count:
        errors.append("completed evaluator truth report is incomplete")
    if (
        report.get("evaluator_provider_call_count") != 0
        or report.get("participant_operation_denominator_impact") != 0
        or report.get("participant_feedback_emitted") is not False
    ):
        errors.append("evaluator truth report isolation invariant failed")
    return errors


__all__ = [
    "WORK_II_TRUTH_PLAN_VERSION",
    "WORK_II_TRUTH_REPORT_VERSION",
    "build_evaluator_truth_plan",
    "compile_evaluator_truth_query",
    "execute_evaluator_truth_plan",
    "validate_evaluator_truth_plan",
    "validate_evaluator_truth_report",
]
