"""Information-completeness contrast, separated from the historical tool contrast."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from statistics import mean

import numpy as np

from chemworld.eval.work_ii_final_diagnostic import ARMS, METRICS, validate
from chemworld.eval.work_ii_final_diagnostic import prompt as minimal_prompt

INFORMATION = ("original", "complete")


def prompt(cell: dict, stage: str, tool: str) -> str:
    instruction, serialized = minimal_prompt(cell, stage, tool).rsplit("\n", 1)
    public = json.loads(serialized)
    public["candidate_parameter_domains"] = cell["public_packet"]["candidate_parameter_domains"]
    if stage == "post" and cell["information"] == "complete":
        public["complete_observation_contract"] = cell["public_packet"][
            "complete_observation_contract"
        ]
    return instruction + "\n" + json.dumps(public, sort_keys=True)


def cells_for_world(world: dict, model: str, world_index: int) -> list[dict]:
    cells = []
    for ai, arm in enumerate(ARMS):
        prior = {
            "availability": "opaque_for_target_locus"
            if arm == "opaque"
            else "supplied_incomplete_executable_law",
            "mechanism_family": None
            if arm == "opaque"
            else "FAMILY_B_POWER"
            if arm == "aligned_nominal"
            else "FAMILY_A_LINEAR",
            "reference_exponent": None
            if arm == "opaque"
            else world["target_exponent"]
            if arm == "aligned_nominal"
            else 1.0,
            "confidence": 0.7,
            "scope_limit": "This is an incomplete local law. Public evidence is authoritative.",
        }
        conditions = INFORMATION if (world_index + ai) % 2 == 0 else INFORMATION[::-1]
        for information in conditions:
            cell = deepcopy(world)
            cell.update(
                arm=arm,
                information=information,
                initial_world_model=prior,
                model=model,
                tool="on",
                repeat=1,
                cell_id=f"{world['cluster_id']}--{arm}--{model}--{information}",
            )
            cells.append(cell)
    return cells


def score(result: dict, cell: dict) -> dict:
    row = {k: cell[k] for k in ("cell_id", "cluster_id", "arm", "model", "information")}
    row.update(
        status=result["status"],
        failure=result.get("failure"),
        joint_recovery=0,
        normalized_regret=1.0,
        top1=0,
        tool_used=bool(result.get("tool_audit")),
    )
    truth = cell["scoring_truth"]
    for stage in ("pre", "post"):
        payload = result.get(stage)
        if payload is None:
            continue
        try:
            validate(payload, cell, stage)
        except ValueError:
            continue
        row[stage + "_exponent_abs_error"] = abs(payload["exponent"] - cell["target_exponent"])
        row[stage + "_family"] = payload["family"]
        row[stage + "_exponent"] = payload["exponent"]
        row[stage + "_mae"] = mean(
            abs(payload["predictions"][q][m] - truth[q][m]) for q in truth for m in METRICS
        )
    if result["status"] == "completed":
        payload = result["post"]
        validate(payload, cell, "post")
        row["joint_recovery"] = int(
            payload["family"] == "FAMILY_B_POWER"
            and abs(payload["exponent"] - cell["target_exponent"]) <= 0.1
        )
        scores = [t["score"] for t in truth.values()]
        raw = max(scores) - truth[payload["selected_query_id"]]["score"]
        row.update(
            raw_regret=raw,
            normalized_regret=raw / max(max(scores) - min(scores), 1e-12),
            top1=int(raw <= 1e-8),
            near_optimal=int(raw <= 0.01),
        )
    return row


def summarize(results: list[dict], cells: list[dict], *, formal: bool) -> dict:
    by_id = {r["cell_id"]: r for r in results}
    if len(by_id) != len(results) or not set(by_id) <= {c["cell_id"] for c in cells}:
        raise ValueError("duplicate or unexpected result")
    rows = [score(by_id.get(c["cell_id"], {"status": "unstarted"}), c) for c in cells]
    groups = []
    for info in INFORMATION:
        for prior in ("recovery", "retention"):
            selected = [
                r
                for r in rows
                if r["information"] == info
                and (r["arm"] == "aligned_nominal") == (prior == "retention")
            ]
            groups.append(
                {
                    "information": info,
                    "analysis": prior,
                    "scheduled": len(selected),
                    "counts": dict(Counter(r["status"] for r in selected)),
                    "joint_recovery": sum(r["joint_recovery"] for r in selected),
                    "tool_use_sessions": sum(r["tool_used"] for r in selected),
                    "mean_normalized_regret_failure_aware": mean(
                        r["normalized_regret"] for r in selected
                    ),
                }
            )
    contrasts = []
    for world in dict.fromkeys(c["cluster_id"] for c in cells):
        selected = [r for r in rows if r["cluster_id"] == world and r["arm"] != "aligned_nominal"]
        contrasts.append(
            {
                "world": world,
                "complete_minus_original": mean(
                    r["joint_recovery"] for r in selected if r["information"] == "complete"
                )
                - mean(r["joint_recovery"] for r in selected if r["information"] == "original"),
            }
        )
    values = np.array([r["complete_minus_original"] for r in contrasts])
    terminal = len(results) == len(cells)
    interval = None
    if terminal and len(values) > 1:
        interval = np.quantile(
            np.random.default_rng(90870).choice(values, (20000, len(values))).mean(axis=1),
            [0.025, 0.975],
        ).tolist()
    receipts = [t for r in results for t in r.get("receipts", [])]
    resources = {
        "attempted_sessions": len(results),
        "turns": len(receipts),
        "scheduled_turns": 2 * len(cells),
        "usage_available_turns": sum(bool(t.get("usage")) for t in receipts),
        "usage_missing_turns": sum(not bool(t.get("usage")) for t in receipts),
        "provider_error_turns": sum(bool(t.get("provider_errors")) for t in receipts),
        "wall_seconds": sum(r.get("elapsed_s", 0) for r in results),
        "wall_seconds_incomplete_sessions": sum(
            any(t.get("elapsed_unavailable", False) for t in r.get("receipts", []))
            for r in results
        ),
        "tool_attempts": sum(len(r.get("tool_audit", [])) for r in results),
        "tool_rejections": sum(
            t["status"] != "completed" for r in results for t in r.get("tool_audit", [])
        ),
        "tool_compute_seconds": sum(
            t["elapsed_s"] for r in results for t in r.get("tool_audit", [])
        ),
    }
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"):
        total = 0
        for result in results:
            cumulative_by_thread = {}
            for receipt in result.get("receipts", []):
                usage = receipt.get("usage", {})
                if key in usage:
                    cumulative_by_thread[receipt.get("thread_id")] = usage[key]
            total += sum(cumulative_by_thread.values())
        resources[key] = total
    return {
        "schema_version": "work-ii-information-intervention-1",
        "formal_result": formal,
        "status": "terminal" if terminal else "incomplete",
        "scheduled": len(cells),
        "counts": dict(Counter(r["status"] for r in rows)),
        "worlds": len(contrasts),
        "models": list(dict.fromkeys(c["model"] for c in cells)),
        "groups": groups,
        "primary": {
            "contrast": "complete_minus_original_joint_recovery_unknown_and_wrong_priors",
            "mean": float(values.mean()) if terminal else None,
            "approximate_world_bootstrap_95": interval,
            "bootstrap_seed": 90870,
            "bootstrap_draws": 20000,
        },
        "world_contrasts": contrasts,
        "rows": rows,
        "resources": resources,
        "failures": [r for r in rows if r["status"] != "completed"],
        "resource_accounting": "CLI usage is cumulative within each thread. Sum the last available "
        "cumulative value per thread and session; missing/interrupted usage is a lower bound.",
        "interpretation": "Aligned-prior retention is separate from recovery. Units are worlds, "
        "not queries, priors or models. Information length is part of the disclosure treatment. "
        "Current-v3 data are not a reanalysis of historical B3 runtime semantics.",
    }
