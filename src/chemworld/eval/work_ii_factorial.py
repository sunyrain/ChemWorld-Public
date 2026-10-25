"""Public-data-only representation-by-decision experiment primitives.

This small development protocol shares the production ActionPlan compiler and
executor. Candidate outcomes enter only ``score_slots``, after choices are sealed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from chemworld.eval.work_ii_truth import compile_evaluator_truth_query

TASKS = ("electrochemical-conversion", "reaction-to-crystallization")
MODELS = ("deepseek", "gpt")
CONDITIONS = ("L-A", "L-X", "F-A", "F-X")
BASIS = ["1", "x", "y", "x*x", "x*y", "y*y"]
INTERVENTION = [
    {
        "kind": "material_law_counterfactual",
        "material_field": "electrolyte_profile",
        "public_to_baseline": [2, 1, 0, 3],
    }
]


def normalized_design(count: int, seed: int) -> list[list[float]]:
    """Outcome-blind LHS rejection preserves strata and the two-task pairing."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        points = np.column_stack(
            [(rng.permutation(count) + rng.random(count)) / count for _ in range(2)]
        )
        potential = 0.65 + points[:, 0]
        current = 20 + 100 * points[:, 1]
        if np.all(np.abs(potential - 1.18) >= 0.02) and np.all(np.abs(current - 70) >= 1):
            return points.tolist()
    raise ValueError("no valid design within the fixed outcome-blind search limit")


def development_protocol() -> dict[str, Any]:
    tasks = {
        TASKS[0]: {
            "axes": [
                ["controlled_potential_V", 0.65, 1.65],
                ["controlled_current_mA", 20.0, 120.0],
            ],
            "fixed_controls": {
                "controlled_duration_s": 3540.0,
                "electrolyte_profile": 2,
                "probe_current_mA": 70.0,
                "probe_duration_s": 630.0,
                "probe_potential_V": 1.18,
                "reagent_amount_mol": 0.004,
                "solvent": 1,
            },
            "electrochemical_workflow_mode": "autonomous_open_v1",
            "scoring_contract_id": "electrochemical-s0-balanced-efficiency-v2",
        },
        TASKS[1]: {
            "axes": [
                ["reaction_temperature_K", 350.0, 405.0],
                ["crystallization_temperature_K", 275.0, 295.0],
            ],
            "fixed_controls": {
                "catalyst": 0,
                "catalyst_amount_mol": 0.000315,
                "crystallization_duration_s": 7200.0,
                "reaction_duration_s": 3600.0,
                "reagent_amount_mol": 0.015,
                "seed_mass_g": 0.008,
                "solvent": 1,
                "stirring_speed_rpm": 675.0,
            },
        },
    }
    return {
        "version": "work-ii-m0-m1-development-1",
        "formal_result": False,
        "world_split": "public-test",
        "world_seed": 0,
        "objective": "balanced",
        "observation_noise_mode": "keyed",
        "observation_seed_base": 90500,
        "noise_namespace": "work-ii-m0-m1-development-20260905",
        "tasks": tasks,
        "evidence_xy": normalized_design(12, 90512),
        "candidate_xy": normalized_design(8, 90508),
        "basis": BASIS,
        "ridge": 1e-6,
        "score_scale": 1.0,
        "epsilon": 0.01,
        "failure_regret": 1.0,
        "provider_timeout_s": 600,
        "requested_output_tokens": 2048,
        "provider_call_opportunities": 12,
        "physical_execution_count": 42,
        "intervention": INTERVENTION,
    }


def compile_design(protocol: Mapping[str, Any], task: str) -> dict[str, Any]:
    spec = protocol["tasks"][task]
    config = {**spec, "task_id": task}
    packet: dict[str, Any] = {
        "task_id": task,
        "axes": spec["axes"],
        "basis": protocol["basis"],
        "utility": "balanced leaderboard score; larger is better; range [0,1]",
        "evidence": [],
        "candidates": [],
    }
    for group, prefix, xy_key in (
        ("evidence", "e", "evidence_xy"),
        ("candidates", "c", "candidate_xy"),
    ):
        for index, xy in enumerate(protocol[xy_key]):
            controls = dict(spec["fixed_controls"])
            controls.update(
                {
                    axis[0]: axis[1] + value * (axis[2] - axis[1])
                    for axis, value in zip(spec["axes"], xy, strict=True)
                }
            )
            compiled = compile_evaluator_truth_query(
                config,
                {
                    "query_id": f"{prefix}{index + 1:02d}",
                    "feature_values": controls,
                    "metric_ids": ["score"],
                },
            )
            packet[group].append(
                {
                    "id": compiled["query_id"],
                    "xy": list(xy),
                    "controls": controls,
                    "action_plan": compiled["action_plan"],
                    "workflow_mode": compiled["workflow_mode"],
                }
            )
    return packet


def public_packet(packet: Mapping[str, Any], *, candidates: bool) -> dict[str, Any]:
    """Explicit projection: trajectories, world identity and hidden scores cannot enter."""
    result = {key: deepcopy(packet[key]) for key in ("task_id", "axes", "basis", "utility")}
    result["evidence"] = [
        {key: deepcopy(row[key]) for key in ("id", "xy", "controls", "action_plan", "score")}
        for row in packet["evidence"]
    ]
    if candidates:
        result["candidates"] = [
            {key: deepcopy(row[key]) for key in ("id", "xy", "controls", "action_plan")}
            for row in packet["candidates"]
        ]
    return result


def design_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    xy = np.asarray([row["xy"] for row in rows], dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([np.ones(len(rows)), x, y, x * x, x * y, y * y])


def fit_public_law(evidence: Sequence[Mapping[str, Any]], ridge: float = 1e-6) -> list[float]:
    matrix = design_matrix(evidence)
    scores = np.asarray([row["score"] for row in evidence], dtype=float)
    penalty = np.diag([0.0, ridge, ridge, ridge, ridge, ridge])
    return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ scores).tolist()


def output_schema(stage: str, candidate_ids: Sequence[str] = ()) -> dict[str, Any]:
    key = "coefficients" if stage == "source" else "candidate_id"
    value: dict[str, Any] = (
        {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6}
        if stage == "source"
        else {"type": "string", "enum": list(candidate_ids)}
    )
    return {
        "type": "object",
        "properties": {key: value},
        "required": [key],
        "additionalProperties": False,
    }


def validate_payload(payload: Any, stage: str, candidate_ids: Sequence[str] = ()) -> None:
    key = "coefficients" if stage == "source" else "candidate_id"
    if not isinstance(payload, dict) or set(payload) != {key}:
        raise ValueError(f"expected only {key}")
    if stage == "source":
        values = payload[key]
        if (
            not isinstance(values, list)
            or len(values) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError("coefficients must be six finite numbers")
    elif payload[key] not in candidate_ids:
        raise ValueError("unknown candidate ID")


def maximize(coefficients: Sequence[float], candidates: Sequence[Mapping[str, Any]]) -> str:
    predictions = design_matrix(candidates) @ np.asarray(coefficients)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("non-finite artifact predictions")
    # Original candidate-ID order breaks exact ties; no clipping of the fitted surface.
    return str(candidates[int(np.argmax(predictions))]["id"])


def participant_prompt(packet: Mapping[str, Any], *, coefficients: Sequence[float] | None) -> str:
    source = coefficients is None
    public = public_packet(packet, candidates=not source)
    if not source:
        public["artifact"] = {"coefficients": list(coefficients), "basis": BASIS}
    task = (
        "Fit a useful local response surface from the public experiments. Return only "
        '{"coefficients":[b0,b1,b2,b3,b4,b5]}. '
        if source
        else "Choose one candidate to maximize utility, using the supplied public evidence and "
        'artifact. Return only {"candidate_id":"..."}. '
    )
    return (
        "You are a scientific participant. Use only INPUT; no tools, shell, files, web, "
        "apps or external data. Give only the minimal JSON, within 2048 output tokens. "
        "x and y are the supplied normalized coordinates; "
        "u=b0+b1*x+b2*y+b3*x*x+b4*x*y+b5*y*y. "
        "The artifact is an unclipped quadratic approximation, not a guaranteed true law. "
        + task
        + "\nINPUT:\n"
        + json.dumps(public, ensure_ascii=False, separators=(",", ":"))
    )


def nearest_public_choice(packet: Mapping[str, Any]) -> str:
    xy = np.asarray([row["xy"] for row in packet["evidence"]])
    scores = [row["score"] for row in packet["evidence"]]
    predicted = [
        scores[int(np.argmin(np.sum((xy - row["xy"]) ** 2, axis=1)))]
        for row in packet["candidates"]
    ]
    return str(packet["candidates"][int(np.argmax(predicted))]["id"])


def score_slots(slots: Sequence[Mapping[str, Any]], truth: Mapping[str, float]) -> list[dict]:
    best = max(truth.values())
    result = []
    for slot in slots:
        row = dict(slot)
        completed = row["status"] == "completed" and row.get("candidate_id") in truth
        regret = best - truth[row["candidate_id"]] if completed else None
        row.update(
            {
                "raw_regret": regret,
                "failure_aware_regret": regret if completed else 1.0,
                "near_optimal": completed and regret <= 0.01,
                "top1": completed and regret <= 1e-12,
            }
        )
        result.append(row)
    return result
