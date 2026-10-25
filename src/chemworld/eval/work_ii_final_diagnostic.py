"""Minimal B3 submission, balanced tool intervention, and failure-aware summaries."""

from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from statistics import mean
from typing import Any

import numpy as np

from chemworld.eval.work_ii_reviewer_followup import evaluate_b3_selected_action

MODELS = ("gpt", "deepseek")
ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
TOOLS = ("off", "on")
METRICS = ("product_in_organic", "product_in_aqueous", "phase_ratio", "score")
FAMILIES = ("FAMILY_A_LINEAR", "FAMILY_B_POWER", "FAMILY_C_SATURATING", "FAMILY_D_CONSTANT")


def schema(cell: dict, stage: str) -> dict:
    number = {"type": "number", "minimum": 0, "maximum": 1}
    predictions = {}
    for query in cell["public_packet"]["scoring_action_queries"]:
        predictions[query["query_id"]] = {
            "type": "object",
            "properties": dict.fromkeys(METRICS, number),
            "required": list(METRICS),
            "additionalProperties": False,
        }
    properties = {
        "family": {"type": "string", "enum": list(FAMILIES)},
        "exponent": {"type": "number", "minimum": 0.25, "maximum": 3},
        "predictions": {
            "type": "object",
            "properties": predictions,
            "required": list(predictions),
            "additionalProperties": False,
        },
    }
    if stage == "post":
        properties["selected_query_id"] = {"type": "string", "enum": list(predictions)}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def validate(payload: Any, cell: dict, stage: str) -> dict:
    contract = schema(cell, stage)
    if not isinstance(payload, dict) or set(payload) != set(contract["required"]):
        raise ValueError("minimal submission fields mismatch")
    if payload["family"] not in FAMILIES:
        raise ValueError("invalid family")
    exponent = payload["exponent"]
    if (
        type(exponent) not in (int, float)
        or not math.isfinite(exponent)
        or not 0.25 <= exponent <= 3
    ):
        raise ValueError("invalid exponent")
    predictions = payload["predictions"]
    ids = contract["properties"]["predictions"]["required"]
    if not isinstance(predictions, dict) or set(predictions) != set(ids):
        raise ValueError("prediction query coverage mismatch")
    for values in predictions.values():
        if not isinstance(values, dict) or set(values) != set(METRICS):
            raise ValueError("prediction metric coverage mismatch")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values.values()
        ):
            raise ValueError("prediction must be finite and in [0,1]")
    if stage == "post" and payload["selected_query_id"] not in ids:
        raise ValueError("invalid candidate")
    return {
        "law_type": "reference_coefficient_power",
        "mechanism_family": payload["family"],
        "reference_exponent": exponent,
    }


def prompt(cell: dict, stage: str, tool: str) -> str:
    packet = cell["public_packet"]
    public = {
        key: packet[key]
        for key in (
            "task_id",
            "metric_range",
            "candidate_mechanism_families",
            "scoring_action_queries",
        )
    }
    if stage == "pre":
        public["initial_world_model"] = cell["initial_world_model"]
        instruction = "No target measurements are available yet. No tools are available."
    else:
        public["evidence"] = packet["evidence"]
        instruction = (
            "Here is the fixed evidence: public linear-reference calibration "
            "and target observations "
            "under matched controls. Update your answer and select one unseen scoring query ID. "
            + (
                "You may use public_numerics.calculate for up to 8 numerical calculations. "
                "It accepts literal expressions: +,-,*,/,**, "
                "array,log,exp,sqrt,abs,mean,sum,min,max,"
                "clip,linspace(start,stop,count),lstsq(design_matrix,target_vector). "
                "It has no simulator or hidden information. Tool use is optional."
                if tool == "on"
                else "No tools are available."
            )
        )
    return (
        "You are a scientific participant in a two-turn constitutive-law study. "
        + instruction
        + " Use only the supplied public information. "
        "No shell, files, web, other tools, or agents. "
        "Return only JSON with family, exponent, predictions keyed by query ID"
        + (", and selected_query_id" if stage == "post" else "")
        + ". Declare family/exponent once; the host maps these fields directly to the original "
        "typed-law commitment without fitting or correcting them. Predictions contain all four "
        "metric values in [0,1] for each query. "
        "Do not include status, explanations or extra fields.\n"
        + json.dumps(public, sort_keys=True)
    )


def schedule(cells: list[dict], *, development: bool) -> list[dict]:
    unique: dict[tuple[str, str], dict] = {}
    for cell in cells:
        unique.setdefault((cell["cluster_id"], cell["arm"]), cell)
    worlds = list(dict.fromkeys(key[0] for key in unique))
    rows = []
    conditions = [(model, tool) for model in MODELS for tool in TOOLS]
    for wi, world in enumerate(worlds):
        for ai, arm in enumerate(ARMS):
            for repeat in range(1, 2 if development else 3):
                offset = (wi + ai + repeat - 1) % 4
                for model, tool in conditions[offset:] + conditions[:offset]:
                    cell = deepcopy(unique[(world, arm)])
                    cell.update(model=model, tool=tool, repeat=repeat)
                    cell["cell_id"] = f"{world}--{arm}--r{repeat}--{model}--{tool}"
                    rows.append(cell)
    return rows


def score(result: dict, cell: dict) -> dict:
    row = {key: cell[key] for key in ("cell_id", "cluster_id", "arm", "model", "tool", "repeat")}
    row.update(
        status=result["status"],
        failure=result.get("failure"),
        joint_recovery=0,
        normalized_regret=1.0,
        top1=0,
        near_optimal=0,
        useful_gain=0,
        action_opportunity=bool(cell["action_opportunity_eligible"]),
    )
    for stage in ("pre", "post"):
        payload = result.get(stage)
        if payload is None:
            continue
        try:
            law = validate(payload, cell, stage)
        except ValueError:
            continue
        errors = [
            abs(payload["predictions"][query][metric] - truth[metric])
            for query, truth in cell["scoring_truth"].items()
            for metric in METRICS
        ]
        row[stage + "_mae"] = mean(errors)
        row[stage + "_family"] = law["mechanism_family"]
        row[stage + "_exponent"] = law["reference_exponent"]
        row[stage + "_exponent_abs_error"] = abs(law["reference_exponent"] - 1.75)
        row[stage + "_family_correct"] = int(law["mechanism_family"] == "FAMILY_B_POWER")
    if result["status"] == "completed":
        payload = result["post"]
        row["joint_recovery"] = int(
            payload["family"] == "FAMILY_B_POWER" and abs(payload["exponent"] - 1.75) <= 0.10
        )
        action = evaluate_b3_selected_action(cell, payload["selected_query_id"])
        row.update(
            normalized_regret=action["normalized_regret"],
            raw_regret=action["raw_regret"],
            top1=int(action["top1_selected"]),
            near_optimal=int(action["raw_regret"] <= 0.01),
            useful_gain=int(
                action["gain_over_evidence_incumbent"] >= 0.02
                and action["action_opportunity_eligible"]
            ),
            action=action,
        )
    return row


def summarize(results: list[dict], cells: list[dict], *, formal: bool) -> dict:
    by_id = {row["cell_id"]: row for row in results}
    if len(by_id) != len(results) or not set(by_id) <= {c["cell_id"] for c in cells}:
        raise ValueError("duplicate or unexpected results")
    rows = [score(by_id.get(cell["cell_id"], {"status": "unstarted"}), cell) for cell in cells]
    summaries = []
    for model in MODELS:
        for tool in TOOLS:
            selected = [r for r in rows if r["model"] == model and r["tool"] == tool]
            summaries.append(
                {
                    "model": model,
                    "tool": tool,
                    "scheduled": len(selected),
                    "attempted": sum(r["status"] != "unstarted" for r in selected),
                    "failed": sum(r["status"] == "failed" for r in selected),
                    "unstarted": sum(r["status"] == "unstarted" for r in selected),
                    "completed": sum(r["status"] == "completed" for r in selected),
                    "joint_recovery": sum(r["joint_recovery"] for r in selected),
                    "mean_regret": mean(r["normalized_regret"] for r in selected),
                    "top1": sum(r["top1"] for r in selected),
                    "near_optimal": sum(r["near_optimal"] for r in selected),
                    "useful_gain": sum(r["useful_gain"] for r in selected),
                    "action_opportunities": sum(r["action_opportunity"] for r in selected),
                    "mean_post_mae_available": mean(
                        r["post_mae"] for r in selected if "post_mae" in r
                    )
                    if any("post_mae" in r for r in selected)
                    else None,
                    "post_mae_available": sum("post_mae" in r for r in selected),
                }
            )
    by_prior = []
    for model in MODELS:
        for tool in TOOLS:
            for arm in ARMS:
                selected = [
                    r for r in rows if (r["model"], r["tool"], r["arm"]) == (model, tool, arm)
                ]
                by_prior.append(
                    {
                        "model": model,
                        "tool": tool,
                        "arm": arm,
                        "scheduled": len(selected),
                        "completed": sum(r["status"] == "completed" for r in selected),
                        "joint_recovery": sum(r["joint_recovery"] for r in selected),
                    }
                )
    contrasts = []
    for world in dict.fromkeys(r["cluster_id"] for r in rows):
        world_rows = [r for r in rows if r["cluster_id"] == world]
        contrasts.append(
            {
                "cluster_id": world,
                "tool_on_minus_off": mean(
                    r["joint_recovery"] for r in world_rows if r["tool"] == "on"
                )
                - mean(r["joint_recovery"] for r in world_rows if r["tool"] == "off"),
            }
        )
    values = np.array([r["tool_on_minus_off"] for r in contrasts])
    rng = np.random.default_rng(90770)
    interval = np.quantile(rng.choice(values, (20_000, len(values))).mean(axis=1), [0.025, 0.975])
    resources = []
    for model in MODELS:
        for tool in TOOLS:
            selected = [r for r in results if r.get("model") == model and r.get("tool") == tool]
            receipts = [turn for r in selected for turn in r.get("receipts", [])]
            resources.append(
                {
                    "model": model,
                    "tool": tool,
                    "attempted_sessions": len(selected),
                    "turns": len(receipts),
                    "usage_available_turns": sum(bool(t.get("usage")) for t in receipts),
                    "usage_missing_turns": sum(not bool(t.get("usage")) for t in receipts),
                    "provider_error_turns": sum(bool(t.get("provider_errors")) for t in receipts),
                    "usage_potentially_incomplete_turns": sum(
                        not bool(t.get("usage"))
                        or bool(t.get("provider_errors"))
                        or t.get("failure") is not None
                        for t in receipts
                    ),
                    "input_tokens": sum(
                        t.get("usage", {}).get("input_tokens", 0) for t in receipts
                    ),
                    "output_tokens": sum(
                        t.get("usage", {}).get("output_tokens", 0) for t in receipts
                    ),
                    "cached_input_tokens": sum(
                        t.get("usage", {}).get("cached_input_tokens", 0) for t in receipts
                    ),
                    "reasoning_output_tokens_reported": sum(
                        t.get("usage", {}).get("reasoning_output_tokens", 0) for t in receipts
                    ),
                    "reasoning_usage_available_turns": sum(
                        "reasoning_output_tokens" in t.get("usage", {}) for t in receipts
                    ),
                    "wall_seconds": sum(r.get("elapsed_s", 0) for r in selected),
                    "tool_attempts": sum(len(r.get("tool_audit", [])) for r in selected),
                    "tool_compute_seconds": sum(
                        t["elapsed_s"] for r in selected for t in r.get("tool_audit", [])
                    ),
                    "tool_rejections": sum(
                        t["status"] != "completed"
                        for r in selected
                        for t in r.get("tool_audit", [])
                    ),
                }
            )
    model_contrasts = []
    for model in MODELS:
        paired = []
        for world in dict.fromkeys(r["cluster_id"] for r in rows):
            subset = [r for r in rows if r["cluster_id"] == world and r["model"] == model]
            paired.append(
                {
                    "cluster_id": world,
                    "tool_on_minus_off": mean(
                        r["joint_recovery"] for r in subset if r["tool"] == "on"
                    )
                    - mean(r["joint_recovery"] for r in subset if r["tool"] == "off"),
                }
            )
        model_contrasts.append(
            {"model": model, "worlds": paired, "mean": mean(r["tool_on_minus_off"] for r in paired)}
        )
    return {
        "schema_version": "work-ii-final-diagnostic-1",
        "formal_result": formal,
        "counts": dict(Counter(r["status"] for r in rows)),
        "scheduled": len(cells),
        "worlds": len(contrasts),
        "additional_independent_worlds": 0,
        "participant_physics": 0,
        "new_truth_executions": 0,
        "new_replays": 0,
        "primary": {
            "contrast": "tool_on_minus_off_joint_recovery",
            "mean": float(values.mean()),
            "approximate_world_bootstrap_95": interval.tolist() if len(contrasts) > 1 else None,
            "interval_status": "small_sample_approximation"
            if len(contrasts) > 1
            else "not_estimated_single_development_world",
            "bootstrap_seed": 90770,
            "bootstrap_draws": 20_000,
        },
        "by_model_tool": summaries,
        "by_prior": by_prior,
        "model_contrasts": model_contrasts,
        "world_contrasts": contrasts,
        "resources": resources,
        "failures": [
            {
                "cell_id": r["cell_id"],
                "status": r["status"],
                "failure": r.get("failure"),
                "provider_failure_kinds": sorted(
                    {
                        kind
                        for receipt in by_id.get(r["cell_id"], {}).get("receipts", [])
                        for kind in receipt.get("provider_failure_kinds", [])
                    }
                ),
            }
            for r in rows
            if r["status"] != "completed"
        ],
        "rows": rows,
    }
