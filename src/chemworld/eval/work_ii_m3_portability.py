"""Fixed-information projections and paired inference for same-world portability."""

from __future__ import annotations

import json
from copy import deepcopy

from chemworld.eval.work_ii_factorial import (
    BASIS,
    MODELS,
    TASKS,
    normalized_design,
    validate_payload,
)
from chemworld.eval.work_ii_factorial_replication import source_schedule, summarize_factorial

CONDITIONS = ("none", "raw", "L", "F")
CONTRASTS = {
    "L_minus_none": {"L": 1, "none": -1},
    "raw_minus_none": {"raw": 1, "none": -1},
    "F_minus_none": {"F": 1, "none": -1},
    "L_minus_raw": {"L": 1, "raw": -1},
    "F_minus_raw": {"F": 1, "raw": -1},
    "F_minus_L": {"F": 1, "L": -1},
}


def portability_protocol(source_protocol: dict, source_binding: dict) -> dict:
    protocol = deepcopy(source_protocol)
    protocol.update(
        version="work-ii-m3-portability-1",
        source_binding={
            key: source_binding[key] for key in ("report", "report_sha256", "protocol")
        },
        noise_namespace="work-ii-m3-portability-20260905",
        observation_seed_base=90600,
        candidate_xy=normalized_design(8, 90608),
        provider_call_opportunities=160,
        physical_execution_count=80,
        primary_contrast="L_minus_none",
        bootstrap_seed=20260908,
        secondary_interval_level=0.99,
        source_schedule_seed=20260907,
        evidence_reuse_count=120,
        scope="same-world context portability on new candidates; no new independent worlds",
    )
    old = {tuple(xy) for key in ("evidence_xy", "candidate_xy") for xy in source_protocol[key]}
    if old & {tuple(xy) for xy in protocol["candidate_xy"]}:
        raise ValueError("new candidates overlap source evidence or old candidates")
    return protocol


def recipient_schedule(protocol: dict) -> list[dict]:
    result = []
    for state in source_schedule(protocol):
        offset = (
            2 * MODELS.index(state["model"])
            + state["repeat"]
            - 1
            + state["world_index"]
            + TASKS.index(state["task"])
        ) % len(CONDITIONS)
        order = CONDITIONS[offset:] + CONDITIONS[:offset]
        result.extend(
            {
                **{
                    key: state[key] for key in ("state_id", "cluster_id", "task", "model", "repeat")
                },
                "condition": condition,
                "serial_position": position + 1,
                "call_id": state["state_id"] + "--" + condition,
            }
            for position, condition in enumerate(order)
        )
    return result


def recipient_input(packet: dict, condition: str, coefficients: list | None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("unknown information condition")
    public = {key: deepcopy(packet[key]) for key in ("task_id", "axes", "basis", "utility")}
    public["candidates"] = [
        {key: deepcopy(row[key]) for key in ("id", "xy", "controls", "action_plan")}
        for row in packet["candidates"]
    ]
    if condition == "raw":
        public["evidence"] = [
            {key: deepcopy(row[key]) for key in ("id", "xy", "controls", "action_plan", "score")}
            for row in packet["evidence"]
        ]
    if condition in ("L", "F"):
        validate_payload({"coefficients": coefficients}, "source")
        public["artifact"] = {"coefficients": list(coefficients), "basis": BASIS}
    return public


def recipient_prompt(packet: dict, condition: str, coefficients: list | None) -> str:
    return (
        "You are a scientific participant. Use only INPUT; no tools, shell, files, web, "
        "apps or external data. Give only the minimal JSON, within 2048 output tokens. "
        "x and y are the supplied normalized coordinates. If a quadratic artifact is supplied, "
        "u=b0+b1*x+b2*y+b3*x*x+b4*x*y+b5*y*y; it is an unclipped approximation, "
        "not a guaranteed true law. Choose one candidate to maximize utility using the "
        'information provided. Return only {"candidate_id":"..."}. '
        "\nINPUT:\n"
        + json.dumps(
            recipient_input(packet, condition, coefficients),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def summarize_portability(rows: list[dict], protocol: dict) -> dict:
    result = summarize_factorial(rows, protocol, conditions=CONDITIONS, contrasts=CONTRASTS)
    result["inference_limit"] += (
        " These are the same ten M1 worlds with new candidate plans, not ten additional "
        "replication worlds. No equivalence or experimental-savings inference."
    )
    return result
