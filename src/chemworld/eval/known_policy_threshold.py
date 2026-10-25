"""Disjoint-world threshold qualification for Work I known-policy controls."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import to_builtin
from chemworld.eval.known_policy_contract import (
    FORMAL_WORLD_SEEDS,
    INFORMATION_ARMS,
    PROBE_SCHEDULE,
    known_policy_contract_sha256,
)
from chemworld.eval.provenance import canonical_json_sha256, file_sha256

QUALIFICATION_SCHEMA_ID = "chemworld.known_policy_threshold_qualification"
QUALIFICATION_SCHEMA_VERSION = "0.1.0"
THRESHOLD_BINDING_SCHEMA_ID = "chemworld.known_policy_threshold_binding"
THRESHOLD_BINDING_SCHEMA_VERSION = "0.1.0"
QUALIFICATION_WORLD_SEEDS = (1000, 1001, 1002, 1003, 1004)
PROTOCOL_BASE_COMMIT = "acf89124715577ea743beb15731891bfb411fe73"
OBSERVATION_SEED_OFFSET = 200_000
NOISE_NAMESPACE_PREFIX = "work-i-known-policy-threshold-qualification-v0.1"

SOURCE_PATHS = (
    "configs/benchmark/work_i_known_policy_contract_v0.1.json",
    "src/chemworld/campaign_resources.py",
    "src/chemworld/envs/chemworld_env.py",
    "src/chemworld/envs/observation_noise.py",
    "src/chemworld/eval/known_policy_contract.py",
    "src/chemworld/eval/known_policy_threshold.py",
    "src/chemworld/operation_validator.py",
    "src/chemworld/runtime/electrochemical_services.py",
    "src/chemworld/world/electrochemical_material_family.py",
    "src/chemworld/world/observation_kernel.py",
    "src/chemworld/world/scoring.py",
    "scripts/qualify_work_i_known_policy_threshold.py",
)
ARTIFACT_FLOAT_SIGNIFICANT_DIGITS = 12
ARTIFACT_FLOAT_ABSOLUTE_ZERO_TOLERANCE = 1e-15


def qualification_resource_card() -> CampaignResourceCard:
    """Return the exact qualification-only campaign resource card."""

    return CampaignResourceCard(
        card_id="work-i-known-policy-threshold-qualification-k6-v1",
        operation_attempt_limit=36,
        vessel_start_limit=6,
        # The campaign scheduler uses the final-assay ceiling as part of its
        # future-lifecycle availability envelope even though this qualification
        # closes every vessel by discard.
        final_assay_limit=6,
        nonfinal_instrument_use_limit=6,
        stock_limits={"reagent_mol": 0.10, "solvent_L": 0.16},
        per_instrument_limits={"uvvis": 6},
        metadata={
            "task_id": "electrochemical-conversion",
            "role": "threshold_qualification_only",
            "planned_lifecycles": 6,
            "terminal_policy": "measure_then_discard",
        },
    )


def source_manifest(root: Path) -> dict[str, str]:
    """Hash the complete declared qualification source surface."""

    return {path: file_sha256(root / path) for path in SOURCE_PATHS}


def stable_numeric_payload(value: Any) -> Any:
    """Normalize negligible libm/runtime float tails for artifact identity.

    Qualification decisions retain their raw public diagnostic values.  This
    normalization is used only for state/resource evidence hashes and report-
    only ledger values so Python runtimes with immaterial libm/root-solver
    differences rebuild the same audit artifact.
    """

    if isinstance(value, Mapping):
        return {str(key): stable_numeric_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable_numeric_payload(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("qualification artifacts cannot contain non-finite floats")
        if abs(value) < ARTIFACT_FLOAT_ABSOLUTE_ZERO_TOLERANCE:
            return 0.0
        normalized = float(format(value, f".{ARTIFACT_FLOAT_SIGNIFICANT_DIGITS}g"))
        return 0.0 if normalized == 0.0 else normalized
    return value


def _probe_prefix(probe: Any) -> list[dict[str, Any]]:
    return [
        {
            "operation": "add_solvent",
            "volume_L": 0.025,
            "solvent": probe.solvent,
        },
        {
            "operation": "add_reagent",
            "amount_mol": probe.reagent_amount_mol,
        },
        {
            "operation": "set_potential",
            "potential_V": probe.potential_V,
            "current_mA": probe.current_mA,
            "electrolyte_profile": probe.electrolyte_profile,
        },
        {"operation": "electrolyze", "duration_s": probe.probe_duration_s},
    ]


def _scalar(observation: Mapping[str, Any], key: str) -> float:
    values = np.asarray(observation[key], dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{key} must be a finite scalar observation")
    return float(values[0])


def _physical_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "world_seed",
        "world_id",
        "mechanism_id",
        "mechanism_hash",
        "observation_seed",
        "observation_noise_mode",
        "observation_noise_namespace",
        "campaign_resource_card_sha256",
        "electrochemical_material_family_id",
        "electrochemical_material_family_sha256",
        "electrochemical_material_instance_sha256",
    )
    return {field: to_builtin(provenance.get(field)) for field in fields}


def execute_qualification_campaign(world_seed: int, information_arm: str) -> dict[str, Any]:
    """Run six fixed diagnostic probes, closing each vessel by discard."""

    if world_seed in FORMAL_WORLD_SEEDS:
        raise ValueError("qualification world overlaps the formal world set")
    if information_arm not in INFORMATION_ARMS:
        raise ValueError("unknown information arm")
    namespace = f"{NOISE_NAMESPACE_PREFIX}-world-{world_seed:04d}"
    card = qualification_resource_card()
    env = gym.make(
        "ChemWorld",
        task_id="electrochemical-conversion",
        seed=world_seed,
        budget_override=36,
        episode_mode_override="campaign",
        electrochemical_workflow_mode="autonomous_open_v1",
        electrochemical_material_family_id="nominal-prior-latent-v2",
        material_information={"mode": information_arm},
        observation_seed_override=world_seed + OBSERVATION_SEED_OFFSET,
        observation_noise_mode="keyed",
        observation_noise_namespace=namespace,
        campaign_resource_card=card,
    )
    action_trace: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    last_info: dict[str, Any] | None = None
    try:
        _, reset_info = env.reset(seed=world_seed)
        base_env = cast(Any, env.unwrapped)
        for lifecycle_index, probe in enumerate(PROBE_SCHEDULE):
            actions = [
                *_probe_prefix(probe),
                {"operation": "measure", "instrument": "uvvis"},
                {
                    "operation": "discard_batch",
                    "reason": "known_policy_threshold_qualification",
                },
            ]
            diagnostic: float | None = None
            diagnostic_noise_key: str | None = None
            for within_lifecycle_index, action in enumerate(actions):
                observation, _, terminated, truncated, info = env.step(action)
                last_info = dict(info)
                status = str(info.get("transaction_status"))
                status_counts[status] = status_counts.get(status, 0) + 1
                provenance = base_env.evaluator_provenance()
                state_payload = base_env._state.to_dict(include_hidden=True)
                trace_entry = {
                    "lifecycle_index": lifecycle_index,
                    "within_lifecycle_index": within_lifecycle_index,
                    "action": to_builtin(action),
                    "transaction_status": status,
                    "operation_type": info.get("operation_type"),
                    "instrument": info.get("instrument"),
                    "kernel_id": info.get("kernel_id"),
                    "state_sha256": canonical_json_sha256(
                        stable_numeric_payload(to_builtin(state_payload))
                    ),
                    "public_observation_sha256": canonical_json_sha256(
                        stable_numeric_payload(to_builtin(observation))
                    ),
                    "campaign_resource_state_sha256": canonical_json_sha256(
                        stable_numeric_payload(info["campaign_resources"]["state"])
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                action_trace.append(trace_entry)
                if action.get("operation") == "measure":
                    diagnostic = _scalar(observation, "conversion")
                    diagnostic_noise_key = str(
                        provenance["last_observation_noise"]["noise_key_sha256"]
                    )
            if diagnostic is None or diagnostic_noise_key is None:
                raise RuntimeError("diagnostic measurement was not recorded")
            signals.append(
                {
                    "lifecycle_index": lifecycle_index,
                    "probe_id": probe.probe_id,
                    "conversion": diagnostic,
                    "noise_key_sha256": diagnostic_noise_key,
                }
            )
        if last_info is None:
            raise RuntimeError("qualification campaign executed no operations")
        provenance = base_env.evaluator_provenance()
        resource_state = stable_numeric_payload(
            to_builtin(last_info["campaign_resources"]["state"])
        )
        report = {
            "world_seed": world_seed,
            "information_arm": information_arm,
            "material_information_sha256": reset_info.get(
                "material_information_sha256"
            ),
            "physical_identity": _physical_identity(provenance),
            "signals": signals,
            "status_counts": status_counts,
            "resource_state": resource_state,
            "action_trace_sha256": canonical_json_sha256(action_trace),
            "action_count": len(action_trace),
            "closed_lifecycle_count": int(resource_state["closed_batches"]),
            "provider_call_count": 0,
        }
        report["campaign_sha256"] = canonical_json_sha256(report)
        return report
    finally:
        env.close()


def midpoint_candidates(signals: Sequence[float]) -> tuple[float, ...]:
    """Return midpoints between adjacent unique finite signals."""

    if not signals or any(not math.isfinite(float(value)) for value in signals):
        raise ValueError("candidate signals must be a non-empty finite sequence")
    unique = sorted({float(value) for value in signals})
    if len(unique) < 2:
        return ()
    return tuple((left + right) / 2.0 for left, right in pairwise(unique))


def branch_counts(signals: Sequence[float], threshold: float) -> dict[str, int]:
    """Count frozen comparator branches for a sequence of signals."""

    return {
        "discard": sum(float(value) < threshold for value in signals),
        "continue_and_assay": sum(float(value) >= threshold for value in signals),
    }


def select_threshold(
    signals_by_arm: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Apply the exact V02 candidate, admissibility, and tie-breaking rules."""

    if set(signals_by_arm) != set(INFORMATION_ARMS):
        raise ValueError("qualification signals must contain both frozen arms")
    pooled = [
        float(value)
        for arm in INFORMATION_ARMS
        for value in signals_by_arm[arm]
    ]
    candidates = midpoint_candidates(pooled)
    admissible: list[dict[str, Any]] = []
    for candidate in candidates:
        counts = {
            arm: branch_counts(signals_by_arm[arm], candidate)
            for arm in INFORMATION_ARMS
        }
        if all(
            arm_counts["discard"] > 0
            and arm_counts["continue_and_assay"] > 0
            for arm_counts in counts.values()
        ):
            admissible.append(
                {
                    "threshold": candidate,
                    "branch_counts_by_arm": counts,
                }
            )
    if not admissible:
        raise ValueError("no candidate produces both branches in every qualification arm")
    pooled_median = float(statistics.median(pooled))
    selected = min(
        admissible,
        key=lambda item: (abs(float(item["threshold"]) - pooled_median), item["threshold"]),
    )
    return {
        "pooled_median": pooled_median,
        "unique_signal_count": len(set(pooled)),
        "candidate_count": len(candidates),
        "admissible_candidate_count": len(admissible),
        "selected_threshold": float(selected["threshold"]),
        "selected_branch_counts_by_arm": selected["branch_counts_by_arm"],
        "candidate_sha256": canonical_json_sha256(list(candidates)),
        "selection_rule": (
            "admissible midpoint closest to pooled median; lower threshold breaks "
            "equal-distance ties"
        ),
    }


def _without_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(field, None)
    return result


def qualification_report_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_without_hash(payload, "report_sha256"))


def threshold_binding_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_without_hash(payload, "binding_sha256"))


def build_qualification_report(root: Path) -> dict[str, Any]:
    """Execute all original/replay campaigns and select the threshold."""

    manifest = source_manifest(root)
    originals: list[dict[str, Any]] = []
    replays: list[dict[str, Any]] = []
    for world_seed in QUALIFICATION_WORLD_SEEDS:
        for arm in INFORMATION_ARMS:
            originals.append(execute_qualification_campaign(world_seed, arm))
            replays.append(execute_qualification_campaign(world_seed, arm))

    signals_by_arm = {
        arm: [
            float(signal["conversion"])
            for campaign in originals
            if campaign["information_arm"] == arm
            for signal in campaign["signals"]
        ]
        for arm in INFORMATION_ARMS
    }
    selection = select_threshold(signals_by_arm)
    original_by_cell = {
        (int(item["world_seed"]), str(item["information_arm"])): item
        for item in originals
    }
    replay_by_cell = {
        (int(item["world_seed"]), str(item["information_arm"])): item
        for item in replays
    }
    replay_matches = {
        f"seed-{seed}:{arm}": (
            original_by_cell[(seed, arm)] == replay_by_cell[(seed, arm)]
        )
        for seed in QUALIFICATION_WORLD_SEEDS
        for arm in INFORMATION_ARMS
    }
    arm_matches = {}
    for seed in QUALIFICATION_WORLD_SEEDS:
        left = original_by_cell[(seed, INFORMATION_ARMS[0])]
        right = original_by_cell[(seed, INFORMATION_ARMS[1])]
        arm_matches[f"seed-{seed}"] = {
            "physical_identity_match": (
                left["physical_identity"] == right["physical_identity"]
            ),
            "signal_vector_match": (
                [item["conversion"] for item in left["signals"]]
                == [item["conversion"] for item in right["signals"]]
            ),
            "noise_coordinate_match": (
                [item["noise_key_sha256"] for item in left["signals"]]
                == [item["noise_key_sha256"] for item in right["signals"]]
            ),
            "action_trace_match": (
                left["action_trace_sha256"] == right["action_trace_sha256"]
            ),
            "resource_state_match": left["resource_state"] == right["resource_state"],
        }

    all_campaigns = [*originals, *replays]
    checks = {
        "qualification_worlds_disjoint_from_formal_worlds": set(
            QUALIFICATION_WORLD_SEEDS
        ).isdisjoint(FORMAL_WORLD_SEEDS),
        "ten_original_and_ten_replay_campaigns": (
            len(originals) == 10 and len(replays) == 10
        ),
        "all_120_original_and_replay_signals_finite": all(
            math.isfinite(float(signal["conversion"]))
            for campaign in all_campaigns
            for signal in campaign["signals"]
        ),
        "all_720_actions_committed": all(
            campaign["status_counts"] == {"committed": 36}
            for campaign in all_campaigns
        ),
        "all_20_campaigns_close_six_lifecycles": all(
            campaign["closed_lifecycle_count"] == 6 for campaign in all_campaigns
        ),
        "all_exact_replays_match": all(replay_matches.values()),
        "all_matched_information_arms_preserve_physical_trace": all(
            all(gates.values()) for gates in arm_matches.values()
        ),
        "selected_threshold_has_both_branches_in_every_arm": all(
            counts["discard"] > 0 and counts["continue_and_assay"] > 0
            for counts in selection["selected_branch_counts_by_arm"].values()
        ),
        "provider_call_count_is_zero": all(
            campaign["provider_call_count"] == 0 for campaign in all_campaigns
        ),
    }
    report: dict[str, Any] = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "status": "qualified_and_frozen" if all(checks.values()) else "failed",
        "formal_result": False,
        "protocol_base_commit": PROTOCOL_BASE_COMMIT,
        "known_policy_contract_sha256": known_policy_contract_sha256(),
        "qualification_world_seeds": list(QUALIFICATION_WORLD_SEEDS),
        "formal_world_seeds_excluded": list(FORMAL_WORLD_SEEDS),
        "information_arms": list(INFORMATION_ARMS),
        "diagnostic_signal": "observation.conversion",
        "comparator": ">=",
        "artifact_float_canonicalization": {
            "scope": "state/resource evidence hashes and report-only ledger values",
            "significant_digits": ARTIFACT_FLOAT_SIGNIFICANT_DIGITS,
            "absolute_zero_tolerance": ARTIFACT_FLOAT_ABSOLUTE_ZERO_TOLERANCE,
            "threshold_selection_uses_raw_diagnostic_values": True,
        },
        "resource_card": qualification_resource_card().to_dict(),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "selection": selection,
        "checks": checks,
        "replay_matches": replay_matches,
        "matched_arm_audit": arm_matches,
        "original_campaigns": originals,
        "replay_campaigns": replays,
        "counts": {
            "original_campaigns": len(originals),
            "replay_campaigns": len(replays),
            "original_signals": sum(len(item["signals"]) for item in originals),
            "replay_signals": sum(len(item["signals"]) for item in replays),
            "original_actions": sum(item["action_count"] for item in originals),
            "replay_actions": sum(item["action_count"] for item in replays),
            "provider_calls": 0,
        },
        "claim_boundary": (
            "This qualification binds one deterministic diagnostic threshold for the "
            "known-policy construct-validity control. It is not an agent-performance "
            "result, an endpoint comparison, or evidence from the formal five worlds."
        ),
    }
    report["report_sha256"] = qualification_report_sha256(report)
    return report


def build_threshold_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    """Create the compact machine binding consumed by V04 and later tasks."""

    if report.get("status") != "qualified_and_frozen":
        raise ValueError("cannot bind a threshold from a failed qualification")
    selection = report["selection"]
    binding: dict[str, Any] = {
        "schema_id": THRESHOLD_BINDING_SCHEMA_ID,
        "schema_version": THRESHOLD_BINDING_SCHEMA_VERSION,
        "status": "frozen",
        "known_policy_contract_sha256": report["known_policy_contract_sha256"],
        "qualification_report_sha256": qualification_report_sha256(report),
        "source_manifest_sha256": report["source_manifest_sha256"],
        "qualification_world_seeds": report["qualification_world_seeds"],
        "formal_world_seeds_excluded": report["formal_world_seeds_excluded"],
        "information_arms": report["information_arms"],
        "diagnostic_signal": report["diagnostic_signal"],
        "comparator": report["comparator"],
        "threshold": selection["selected_threshold"],
        "qualification_branch_counts_by_arm": selection[
            "selected_branch_counts_by_arm"
        ],
        "selection_rule": selection["selection_rule"],
        "formal_retuning_forbidden": True,
        "provider_call_count": 0,
    }
    binding["binding_sha256"] = threshold_binding_sha256(binding)
    return binding


def validate_qualification_report(report: Mapping[str, Any]) -> list[str]:
    """Return deterministic integrity errors for a frozen qualification report."""

    errors: list[str] = []
    if report.get("schema_id") != QUALIFICATION_SCHEMA_ID:
        errors.append("qualification schema_id mismatch")
    if report.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        errors.append("qualification schema_version mismatch")
    if report.get("known_policy_contract_sha256") != known_policy_contract_sha256():
        errors.append("known-policy contract binding is stale")
    if set(report.get("qualification_world_seeds", [])) & set(FORMAL_WORLD_SEEDS):
        errors.append("qualification worlds overlap formal worlds")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        errors.append("one or more qualification gates failed")
    if report.get("status") != "qualified_and_frozen":
        errors.append("qualification status is not frozen-pass")
    if report.get("report_sha256") != qualification_report_sha256(report):
        errors.append("qualification report hash mismatch")
    if report.get("source_manifest_sha256") != canonical_json_sha256(
        report.get("source_manifest")
    ):
        errors.append("source manifest hash mismatch")
    return errors


def validate_threshold_binding(
    binding: Mapping[str, Any], report: Mapping[str, Any]
) -> list[str]:
    """Return deterministic integrity errors for a threshold binding."""

    errors: list[str] = []
    if binding.get("schema_id") != THRESHOLD_BINDING_SCHEMA_ID:
        errors.append("threshold binding schema_id mismatch")
    if binding.get("schema_version") != THRESHOLD_BINDING_SCHEMA_VERSION:
        errors.append("threshold binding schema_version mismatch")
    if binding.get("status") != "frozen":
        errors.append("threshold binding status is not frozen")
    if binding.get("qualification_report_sha256") != qualification_report_sha256(
        report
    ):
        errors.append("threshold binding report hash mismatch")
    if binding.get("threshold") != report.get("selection", {}).get(
        "selected_threshold"
    ):
        errors.append("threshold value disagrees with qualification selection")
    if binding.get("formal_retuning_forbidden") is not True:
        errors.append("formal retuning prohibition is missing")
    if binding.get("binding_sha256") != threshold_binding_sha256(binding):
        errors.append("threshold binding hash mismatch")
    return errors


__all__ = [
    "ARTIFACT_FLOAT_ABSOLUTE_ZERO_TOLERANCE",
    "ARTIFACT_FLOAT_SIGNIFICANT_DIGITS",
    "INFORMATION_ARMS",
    "NOISE_NAMESPACE_PREFIX",
    "OBSERVATION_SEED_OFFSET",
    "PROTOCOL_BASE_COMMIT",
    "QUALIFICATION_SCHEMA_ID",
    "QUALIFICATION_SCHEMA_VERSION",
    "QUALIFICATION_WORLD_SEEDS",
    "SOURCE_PATHS",
    "THRESHOLD_BINDING_SCHEMA_ID",
    "THRESHOLD_BINDING_SCHEMA_VERSION",
    "branch_counts",
    "build_qualification_report",
    "build_threshold_binding",
    "execute_qualification_campaign",
    "midpoint_candidates",
    "qualification_report_sha256",
    "qualification_resource_card",
    "select_threshold",
    "source_manifest",
    "stable_numeric_payload",
    "threshold_binding_sha256",
    "validate_qualification_report",
    "validate_threshold_binding",
]
