"""Outcome-blind reconstructability audit for the 36 frozen discard units.

This module replays only the already recorded deterministic environment actions.
It captures hashes of evaluator-private state immediately before each original
``discard_batch`` action, but never serializes those private states into a report,
replaces a terminal action, or evaluates a discarded state.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.campaign_resources import CampaignResourceCard, CampaignResourceLedger
from chemworld.data.records import trajectory_value
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.latent_terminal_contract import (
    CONTRACT_ID,
    EXPECTED_CELL_COUNT,
    EXPECTED_DISCARD_COUNT,
    FROZEN_MATRIX_MANIFEST_SHA256,
    FROZEN_TERMINAL_INDEX_SHA256,
    TERMINAL_INDEX_PATH,
    latent_terminal_contract_sha256,
    validate_latent_terminal_contract,
)
from chemworld.eval.provenance import canonical_json_sha256, file_sha256

REPORT_SCHEMA_ID = "chemworld.latent_terminal_reconstructability_audit"
REPORT_SCHEMA_VERSION = "0.1.0"
REPORT_ID = "work-i-deepseek-discarded-state-reconstructability-v0.1"

CONTRACT_PATH = Path("configs/benchmark/work_i_latent_terminal_contract_v0.1.json")
SOURCE_PATHS = (
    CONTRACT_PATH,
    TERMINAL_INDEX_PATH,
    Path("src/chemworld/eval/latent_terminal_reconstructability.py"),
    Path("scripts/audit_work_i_latent_terminal_reconstructability.py"),
)


class LatentTerminalReconstructabilityError(RuntimeError):
    """Raised when the frozen source cannot support an exact checkpoint audit."""


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentTerminalReconstructabilityError(
            f"cannot read {label}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise LatentTerminalReconstructabilityError(f"{label} must be an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise LatentTerminalReconstructabilityError(
                        f"{path}:{line_number} must be an object"
                    )
                records.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentTerminalReconstructabilityError(
            f"cannot read trajectory: {path}"
        ) from exc
    if not records:
        raise LatentTerminalReconstructabilityError(f"empty trajectory: {path}")
    return records


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop(field, None)
    return result


def reconstructability_report_sha256(payload: Mapping[str, Any]) -> str:
    """Return the report digest while excluding its embedded self-hash."""

    return canonical_json_sha256(_without(payload, "report_sha256"))


def _git_common_root(root: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    )
    common = Path(completed.stdout.strip())
    if not common.is_absolute():
        common = (root / common).resolve()
    return common.parent


def discover_run_root(root: Path) -> Path:
    """Resolve the raw root by frozen manifest identity, never by version-looking name."""

    bases = {(root / "runs").resolve(), (_git_common_root(root) / "runs").resolve()}
    matches: set[Path] = set()
    for base in sorted(bases):
        if not base.is_dir():
            continue
        for manifest_path in base.rglob("matrix_manifest.json"):
            try:
                manifest = _read_json_object(manifest_path, label="matrix manifest")
            except LatentTerminalReconstructabilityError:
                continue
            if manifest.get("manifest_sha256") == FROZEN_MATRIX_MANIFEST_SHA256:
                matches.add(manifest_path.parent.resolve())
    if len(matches) != 1:
        raise LatentTerminalReconstructabilityError(
            "expected exactly one local raw root with the frozen matrix manifest; "
            f"found {len(matches)}"
        )
    return next(iter(matches))


def _source_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    resolved = root.resolve()
    for relative in SOURCE_PATHS:
        path = (resolved / relative).resolve()
        if not path.is_relative_to(resolved) or not path.is_file():
            raise LatentTerminalReconstructabilityError(
                f"missing source artifact: {relative}"
            )
        result[relative.as_posix()] = file_sha256(path)
    return result


def _validate_indexed_root(
    run_root: Path, index: Mapping[str, Any]
) -> dict[str, Any]:
    if index.get("index_sha256") != FROZEN_TERMINAL_INDEX_SHA256:
        raise LatentTerminalReconstructabilityError("terminal index identity changed")
    if canonical_json_sha256(_without(index, "index_sha256")) != (
        FROZEN_TERMINAL_INDEX_SHA256
    ):
        raise LatentTerminalReconstructabilityError("terminal index self-hash is stale")
    raw_files = index.get("files")
    if not isinstance(raw_files, list):
        raise LatentTerminalReconstructabilityError("terminal index files are invalid")
    expected_paths: set[str] = set()
    byte_count = 0
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise LatentTerminalReconstructabilityError("terminal index row is invalid")
        relative = Path(str(raw.get("path", "")))
        path = (run_root / relative).resolve()
        if not path.is_relative_to(run_root.resolve()) or not path.is_file():
            raise LatentTerminalReconstructabilityError(
                f"missing indexed raw file: {relative.as_posix()}"
            )
        expected_paths.add(relative.as_posix())
        size = path.stat().st_size
        if size != raw.get("bytes") or file_sha256(path) != raw.get("sha256"):
            raise LatentTerminalReconstructabilityError(
                f"indexed raw file mismatch: {relative.as_posix()}"
            )
        byte_count += size
    actual_paths = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise LatentTerminalReconstructabilityError(
            "raw root file membership differs from the frozen terminal index"
        )
    if len(expected_paths) != index.get("file_count") or byte_count != index.get(
        "byte_count"
    ):
        raise LatentTerminalReconstructabilityError(
            "raw root count or byte total differs from the terminal index"
        )
    return {
        "file_count": len(expected_paths),
        "byte_count": byte_count,
        "all_paths_sizes_and_hashes_match": True,
        "unindexed_file_count": 0,
    }


def _make_replay_env(first: Mapping[str, Any]) -> gym.Env[Any, Any]:
    benchmark_task_id = trajectory_value(first, 'benchmark_task_id', None)
    if not benchmark_task_id:
        raise LatentTerminalReconstructabilityError(
            "raw trajectory lacks benchmark_task_id"
        )
    env_kwargs: dict[str, Any] = {
        "task_id": str(benchmark_task_id),
        "seed": int(first["seed"]),
    }
    if first.get("contract_profile") == "extended-research":
        env_kwargs["budget_override"] = int(first["budget"])
        env_kwargs["episode_mode_override"] = str(first["episode_mode"])
    optional_string_kwargs = {
        "electrochemical_workflow_mode": "electrochemical_workflow_mode",
        "electrochemical_material_family_id": "electrochemical_material_family_id",
        "crystallization_material_family_id": "crystallization_material_family_id",
        "scoring_contract_id": "scoring_contract_id",
        "observation_noise_mode": "observation_noise_mode",
        "observation_noise_namespace": "observation_noise_namespace",
    }
    for record_key, env_key in optional_string_kwargs.items():
        value = first.get(record_key)
        if isinstance(value, str) and value:
            env_kwargs[env_key] = value
    observation_seed = first.get("observation_seed")
    if isinstance(observation_seed, int) and not isinstance(observation_seed, bool):
        env_kwargs["observation_seed_override"] = observation_seed
    material_information = first.get("material_information_config")
    if not isinstance(material_information, Mapping):
        legacy = first.get("material_information")
        material_information = (
            {"mode": legacy.get("mode")}
            if isinstance(legacy, Mapping) and isinstance(legacy.get("mode"), str)
            else None
        )
    if isinstance(material_information, Mapping):
        env_kwargs["material_information"] = dict(material_information)
    card = first.get("campaign_resource_card")
    if isinstance(card, Mapping):
        env_kwargs["campaign_resource_card"] = dict(card)
    return gym.make(str(first["env_id"]), **env_kwargs)


def _dig(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _scalar_observation(observation: Mapping[str, Any]) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    for key, value in observation.items():
        scalar = float(value.reshape(-1)[0])
        payload[key] = scalar if math.isfinite(scalar) else None
    return payload


def _jsonish_mismatches(
    *,
    step: int,
    field: str,
    recorded: Any,
    replayed: Any,
    tolerance: float,
) -> list[dict[str, Any]]:
    if isinstance(recorded, bool) or isinstance(replayed, bool):
        return [] if recorded == replayed else [
            {"step": step, "field": field, "recorded": recorded, "replayed": replayed}
        ]
    if isinstance(recorded, int | float) and isinstance(replayed, int | float):
        error = abs(float(recorded) - float(replayed))
        return [] if error <= tolerance else [
            {
                "step": step,
                "field": field,
                "recorded": recorded,
                "replayed": replayed,
                "abs_error": error,
            }
        ]
    if isinstance(recorded, Mapping) and isinstance(replayed, Mapping):
        mismatches: list[dict[str, Any]] = []
        for key in sorted(set(recorded) | set(replayed)):
            if key not in recorded or key not in replayed:
                mismatches.append(
                    {
                        "step": step,
                        "field": f"{field}.{key}",
                        "recorded": recorded.get(key),
                        "replayed": replayed.get(key),
                    }
                )
            else:
                mismatches.extend(
                    _jsonish_mismatches(
                        step=step,
                        field=f"{field}.{key}",
                        recorded=recorded[key],
                        replayed=replayed[key],
                        tolerance=tolerance,
                    )
                )
        return mismatches
    if isinstance(recorded, list) and isinstance(replayed, list):
        mismatches = []
        if len(recorded) != len(replayed):
            mismatches.append(
                {
                    "step": step,
                    "field": f"{field}.length",
                    "recorded": len(recorded),
                    "replayed": len(replayed),
                }
            )
        for index, item in enumerate(recorded[: len(replayed)]):
            mismatches.extend(
                _jsonish_mismatches(
                    step=step,
                    field=f"{field}.{index}",
                    recorded=item,
                    replayed=replayed[index],
                    tolerance=tolerance,
                )
            )
        return mismatches
    return [] if recorded == replayed else [
        {"step": step, "field": field, "recorded": recorded, "replayed": replayed}
    ]


def _recorded_resource_ledger_sha256(record: Mapping[str, Any]) -> str | None:
    view = trajectory_value(record, "agent_view", {})
    if not isinstance(view, Mapping):
        return None
    value = _dig(view, ("lab_report", "campaign_state", "campaign_resources", "ledger_sha256"))
    return value if isinstance(value, str) and value else None


def _checkpoint_capture(base: ChemWorldEnv, *, next_step: int) -> dict[str, Any]:
    resources = base.campaign_resource_snapshot()
    if not isinstance(resources, Mapping):
        raise LatentTerminalReconstructabilityError(
            f"step {next_step} lacks a campaign resource snapshot"
        )
    state_payload = base._state.to_dict(include_hidden=True)
    provenance = base.evaluator_provenance()
    baseline = base._current_batch_resource_baseline
    checkpoint = {
        "next_step": next_step,
        "experiment_index": base._experiment_index,
        "campaign_resource_current_vessel_started": (
            base._campaign_resource_current_vessel_started
        ),
        "campaign_terminal": base._campaign_terminal,
        "campaign_terminal_reason": base._campaign_terminal_reason,
        "hidden_state_sha256": canonical_json_sha256(state_payload),
        "campaign_resource_state_sha256": canonical_json_sha256(resources["state"]),
        "campaign_resource_event_count": len(resources["events"]),
        "evaluator_provenance_sha256": canonical_json_sha256(provenance),
        "current_batch_resource_baseline_sha256": (
            None if baseline is None else canonical_json_sha256(baseline)
        ),
        "last_observation_noise_sha256": canonical_json_sha256(
            base.observation_noise_provenance()
        ),
        "world_id": provenance.get("world_id"),
        "mechanism_hash": provenance.get("mechanism_hash"),
        "material_instance_sha256": provenance.get(
            "electrochemical_material_instance_sha256"
        ),
        "observation_seed": provenance.get("observation_seed"),
        "observation_noise_mode": provenance.get("observation_noise_mode"),
        "observation_noise_namespace": provenance.get(
            "observation_noise_namespace"
        ),
        "campaign_resource_card_sha256": provenance.get(
            "campaign_resource_card_sha256"
        ),
    }
    checkpoint["checkpoint_identity_sha256"] = canonical_json_sha256(checkpoint)
    return checkpoint


def _historical_resource_prefixes(
    ledger_payload: Mapping[str, Any], target_steps: set[int]
) -> dict[int, dict[str, Any]]:
    card = ledger_payload.get("card")
    events = ledger_payload.get("events")
    if not isinstance(card, Mapping) or not isinstance(events, list):
        raise LatentTerminalReconstructabilityError(
            "historical campaign resource ledger is invalid"
        )
    ledger = CampaignResourceLedger(CampaignResourceCard.from_dict(card))
    prefixes: dict[int, dict[str, Any]] = {}
    for step, raw_event in enumerate(events, start=1):
        if step in target_steps:
            prefixes[step] = ledger.snapshot()
        if not isinstance(raw_event, Mapping):
            raise LatentTerminalReconstructabilityError(
                f"campaign resource event {step} is invalid"
            )
        event_id = str(raw_event.get("event_id", ""))
        action = raw_event.get("action")
        if not isinstance(action, Mapping):
            raise LatentTerminalReconstructabilityError(
                f"campaign resource event {step} action is invalid"
            )
        starts_vessel = raw_event.get("starts_vessel") is True
        replayed_preflight = ledger.preflight(
            event_id,
            dict(action),
            starts_vessel=starts_vessel,
        ).to_dict()
        if replayed_preflight != raw_event.get("preflight"):
            raise LatentTerminalReconstructabilityError(
                f"campaign resource preflight mismatch at step {step}"
            )
        outcome = raw_event.get("outcome")
        if not isinstance(outcome, Mapping):
            raise LatentTerminalReconstructabilityError(
                f"campaign resource outcome missing at step {step}"
            )
        delta = outcome.get("delta")
        if not isinstance(delta, Mapping):
            raise LatentTerminalReconstructabilityError(
                f"campaign resource delta missing at step {step}"
            )
        replayed_delta = ledger.record_outcome(
            event_id,
            dict(action),
            {
                "operation_committed": outcome.get("committed") is True,
                "campaign_resource_report_delta": dict(
                    delta.get("report_only", {})
                ),
            },
            starts_vessel=starts_vessel,
        ).to_dict()
        if replayed_delta != dict(delta):
            raise LatentTerminalReconstructabilityError(
                f"campaign resource outcome mismatch at step {step}"
            )
    if set(prefixes) != target_steps:
        raise LatentTerminalReconstructabilityError(
            "not all historical resource prefixes were reconstructed"
        )
    if ledger.snapshot() != dict(ledger_payload):
        raise LatentTerminalReconstructabilityError(
            "historical campaign resource ledger does not reconstruct exactly"
        )
    return prefixes


def _capture_pre_discard_checkpoints(
    records: list[dict[str, Any]],
    target_steps: set[int],
    *,
    verify_trajectory: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    env = _make_replay_env(records[0])
    captures: dict[int, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    max_abs_error = 0.0
    tolerance = 1.0e-5
    try:
        _, reset_info = env.reset(seed=int(records[0]["seed"]))
        base = cast(ChemWorldEnv, env.unwrapped)
        provenance = base.evaluator_provenance()
        if verify_trajectory:
            first = records[0]
            for field, recorded, replayed in (
                (
                    "task_contract_hash",
                    first.get("task_contract_hash"),
                    reset_info.get("task_contract_hash"),
                ),
                (
                    "runtime_profile_hash",
                    first.get("runtime_profile_hash"),
                    reset_info.get("runtime_profile_hash"),
                ),
                (
                    "mechanism_hash",
                    first.get("mechanism_hash"),
                    provenance.get("mechanism_hash"),
                ),
                (
                    "scoring_contract_hash",
                    first.get("scoring_contract_hash"),
                    reset_info.get("scoring_contract_hash"),
                ),
                (
                    "observation_contract_hash",
                    first.get("observation_contract_hash"),
                    reset_info.get("observation_contract_hash"),
                ),
            ):
                if recorded and recorded != replayed:
                    mismatches.append(
                        {
                            "step": 0,
                            "field": field,
                            "recorded": recorded,
                            "replayed": replayed,
                        }
                    )
        for record in records:
            step = int(record["step"])
            if step in target_steps:
                if record.get("action", {}).get("operation") != "discard_batch":
                    raise LatentTerminalReconstructabilityError(
                        f"target step {step} is not discard_batch"
                    )
                captures[step] = _checkpoint_capture(base, next_step=step)
            observation, reward, terminated, truncated, info = env.step(
                record["action"]
            )
            if verify_trajectory:
                reward_error = abs(float(trajectory_value(record, 'reward')) - float(reward))
                max_abs_error = max(max_abs_error, reward_error)
                if reward_error > tolerance:
                    mismatches.append(
                        {
                            "step": step,
                            "field": "reward",
                            "recorded": trajectory_value(record, 'reward'),
                            "replayed": reward,
                            "abs_error": reward_error,
                        }
                    )
                replay_observation = _scalar_observation(observation)
                for key, replayed_value in replay_observation.items():
                    recorded_value = trajectory_value(record, 'observation')[key]
                    if replayed_value is None or recorded_value is None:
                        if replayed_value is not None or recorded_value is not None:
                            mismatches.append(
                                {
                                    "step": step,
                                    "field": f"observation.{key}",
                                    "recorded": recorded_value,
                                    "replayed": replayed_value,
                                }
                            )
                        continue
                    error = abs(float(recorded_value) - replayed_value)
                    max_abs_error = max(max_abs_error, error)
                    if error > tolerance:
                        mismatches.append(
                            {
                                "step": step,
                                "field": f"observation.{key}",
                                "recorded": recorded_value,
                                "replayed": replayed_value,
                                "abs_error": error,
                            }
                        )
                for field, recorded, replayed in (
                    ("terminated", bool(record["terminated"]), terminated),
                    ("truncated", bool(record["truncated"]), truncated),
                    (
                        "operation_type",
                        record.get("operation_type"),
                        info.get("operation_type"),
                    ),
                ):
                    if recorded != replayed:
                        mismatches.append(
                            {
                                "step": step,
                                "field": field,
                                "recorded": recorded,
                                "replayed": replayed,
                            }
                        )
                replay_audit_info = {**info, **provenance}
                for field in (
                    "task_contract_hash",
                    "runtime_profile_hash",
                    "mechanism_id",
                    "mechanism_hash",
                    "scoring_contract_hash",
                    "observation_contract_hash",
                    "kernel_id",
                    "kernel_version",
                    "affected_ledgers",
                    "world_events",
                    "state_patches_summary",
                    "transaction_status",
                    "rollback_reason",
                    "state_delta_summary",
                ):
                    if field in record:
                        mismatches.extend(
                            _jsonish_mismatches(
                                step=step,
                                field=field,
                                recorded=record[field],
                                replayed=replay_audit_info.get(field),
                                tolerance=tolerance,
                            )
                        )
    finally:
        env.close()
    if set(captures) != target_steps:
        raise LatentTerminalReconstructabilityError(
            "not all target discard checkpoints were captured"
        )
    verification = (
        {
            "verified": not mismatches,
            "checked_steps": len(records),
            "max_abs_error": max_abs_error,
            "mismatches": mismatches,
        }
        if verify_trajectory
        else None
    )
    return captures, verification


def _unit_identity_matches(
    record: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> bool:
    expected = {
        "world_id": record.get("world_id"),
        "mechanism_hash": record.get("mechanism_hash"),
        "material_instance_sha256": record.get(
            "electrochemical_material_instance_sha256"
        ),
        "observation_seed": record.get("observation_seed"),
        "observation_noise_mode": record.get("observation_noise_mode"),
        "observation_noise_namespace": record.get("observation_noise_namespace"),
        "campaign_resource_card_sha256": record.get(
            "campaign_resource_card_sha256"
        ),
    }
    return all(checkpoint.get(key) == value for key, value in expected.items())


def _audit_cell(
    *,
    root: Path,
    run_root: Path,
    manifest_cell: Mapping[str, Any],
    contract_cell: Mapping[str, Any],
) -> dict[str, Any]:
    cell_id = str(contract_cell["cell_id"])
    attempt_dir = manifest_cell.get("authoritative_attempt_dir")
    if not isinstance(attempt_dir, str) or not attempt_dir:
        raise LatentTerminalReconstructabilityError(
            f"{cell_id} lacks an authoritative attempt"
        )
    raw_path = (run_root / attempt_dir / "trajectory.jsonl").resolve()
    if not raw_path.is_relative_to(run_root.resolve()):
        raise LatentTerminalReconstructabilityError(f"unsafe path for {cell_id}")
    if file_sha256(raw_path) != contract_cell.get("source_trajectory_sha256"):
        raise LatentTerminalReconstructabilityError(
            f"{cell_id} raw trajectory hash changed"
        )
    raw_records = _read_jsonl(raw_path)
    ledger_payload = _read_json_object(
        raw_path.parent / "campaign_resource_ledger.json",
        label=f"{cell_id} campaign resource ledger",
    )
    compact_path = root / str(contract_cell["compact_path"])
    compact_records = _read_jsonl(compact_path)
    if len(raw_records) != len(compact_records):
        raise LatentTerminalReconstructabilityError(
            f"{cell_id} raw/compact record counts differ"
        )
    units = contract_cell.get("discard_units")
    if not isinstance(units, list):
        raise LatentTerminalReconstructabilityError(
            f"{cell_id} discard units are invalid"
        )
    target_steps = {int(unit["terminal_step"]) for unit in units}
    historical_prefixes = _historical_resource_prefixes(
        ledger_payload,
        target_steps,
    )
    first_capture, verification = _capture_pre_discard_checkpoints(
        raw_records,
        target_steps,
        verify_trajectory=True,
    )
    if not isinstance(verification, Mapping) or verification.get("verified") is not True:
        raise LatentTerminalReconstructabilityError(
            f"{cell_id} exact full-trajectory replay failed"
        )
    second_capture, _ = _capture_pre_discard_checkpoints(
        raw_records,
        target_steps,
        verify_trajectory=False,
    )
    rows: list[dict[str, Any]] = []
    for raw_unit in units:
        if not isinstance(raw_unit, Mapping):
            raise LatentTerminalReconstructabilityError(
                f"{cell_id} discard unit is invalid"
            )
        step = int(raw_unit["terminal_step"])
        index = step - 1
        raw_record = raw_records[index]
        compact_record = compact_records[index]
        checkpoint = first_capture[step]
        repeat = second_capture[step]
        historical_resources = historical_prefixes[step]
        recorded_ledger = _recorded_resource_ledger_sha256(raw_records[index - 1])
        historical_resource_state_sha256 = canonical_json_sha256(
            historical_resources["state"]
        )
        checkpoint_identity_sha256 = canonical_json_sha256(
            {
                **checkpoint,
                "historical_campaign_resource_snapshot_sha256": (
                    historical_resources["ledger_sha256"]
                ),
            }
        )
        gates = {
            "step_and_lifecycle_identity_match": (
                int(raw_record.get("step", -1)) == step
                and int(raw_record.get("experiment_index", -1))
                == int(raw_unit["lifecycle_index"])
                and checkpoint.get("experiment_index")
                == int(raw_unit["lifecycle_index"])
            ),
            "terminal_action_hash_match": (
                canonical_json_sha256(raw_record.get("action"))
                == raw_unit.get("terminal_action_sha256")
                == canonical_json_sha256(compact_record.get("action"))
            ),
            "public_prefix_hash_match": (
                canonical_json_sha256(compact_records[:index])
                == raw_unit.get("public_prefix_sha256")
            ),
            "world_material_contract_identity_match": _unit_identity_matches(
                raw_record, checkpoint
            ),
            "campaign_resource_ledger_matches_recorded_prefix": (
                recorded_ledger is not None
                and historical_resources.get("ledger_sha256")
                == recorded_ledger
            ),
            "historical_resource_prefix_replays_exactly": (
                len(historical_resources["events"]) == index
                and historical_resources["last_event_id"]
                == ledger_payload["events"][index - 1]["event_id"]
            ),
            "runtime_resource_state_matches_historical_prefix": (
                checkpoint.get("campaign_resource_state_sha256")
                == historical_resource_state_sha256
            ),
            "independent_hidden_state_replay_match": (
                checkpoint.get("hidden_state_sha256")
                == repeat.get("hidden_state_sha256")
            ),
            "independent_resource_replay_match": (
                checkpoint.get("campaign_resource_state_sha256")
                == repeat.get("campaign_resource_state_sha256")
            ),
            "independent_checkpoint_identity_match": checkpoint == repeat,
            "captured_before_original_discard": (
                checkpoint.get("next_step") == step
                and raw_record.get("action", {}).get("operation") == "discard_batch"
            ),
            "source_trajectory_unchanged": (
                file_sha256(raw_path) == contract_cell.get("source_trajectory_sha256")
            ),
        }
        row = {
            "discard_id": raw_unit["discard_id"],
            "cell_id": cell_id,
            "world_seed": contract_cell["world_seed"],
            "information_arm": contract_cell["information_arm"],
            "lifecycle_index": raw_unit["lifecycle_index"],
            "terminal_step": step,
            "terminal_action_sha256": raw_unit["terminal_action_sha256"],
            "public_prefix_sha256": raw_unit["public_prefix_sha256"],
            "raw_prefix_sha256": canonical_json_sha256(raw_records[:index]),
            "hidden_state_sha256": checkpoint["hidden_state_sha256"],
            "campaign_resource_snapshot_sha256": historical_resources[
                "ledger_sha256"
            ],
            "campaign_resource_state_sha256": historical_resource_state_sha256,
            "campaign_resource_ledger_sha256": historical_resources[
                "ledger_sha256"
            ],
            "checkpoint_identity_sha256": checkpoint_identity_sha256,
            "gates": gates,
            "reconstructable": all(gates.values()),
            "shadow_terminal_executed": False,
            "latent_discard_score_accessed": False,
            "agent_provider_calls": 0,
        }
        rows.append(row)
    return {
        "cell_id": cell_id,
        "world_seed": contract_cell["world_seed"],
        "information_arm": contract_cell["information_arm"],
        "source_trajectory_sha256": contract_cell["source_trajectory_sha256"],
        "record_count": len(raw_records),
        "discard_count": len(rows),
        "exact_full_trajectory_replay": dict(verification),
        "discard_units": rows,
        "all_discard_checkpoints_reconstructable": all(
            row["reconstructable"] for row in rows
        ),
    }


def build_reconstructability_report(root: Path, run_root: Path) -> dict[str, Any]:
    """Build the complete 36-unit outcome-blind reconstructability report."""

    resolved = root.resolve()
    raw_root = run_root.resolve()
    contract = _read_json_object(resolved / CONTRACT_PATH, label="L01 contract")
    contract_errors = validate_latent_terminal_contract(contract, root=resolved)
    if contract_errors:
        raise LatentTerminalReconstructabilityError(
            "L01 contract invalid: " + "; ".join(contract_errors)
        )
    if contract.get("contract_id") != CONTRACT_ID:
        raise LatentTerminalReconstructabilityError("unexpected L01 contract identity")
    index = _read_json_object(resolved / TERMINAL_INDEX_PATH, label="terminal index")
    raw_root_audit = _validate_indexed_root(raw_root, index)
    matrix_manifest = _read_json_object(
        raw_root / "matrix_manifest.json", label="matrix manifest"
    )
    if matrix_manifest.get("manifest_sha256") != FROZEN_MATRIX_MANIFEST_SHA256:
        raise LatentTerminalReconstructabilityError("matrix manifest identity changed")
    manifest_cells = matrix_manifest.get("cells")
    if not isinstance(manifest_cells, list):
        raise LatentTerminalReconstructabilityError("matrix cells are invalid")
    manifest_by_id = {
        str(cell.get("cell_id")): cell
        for cell in manifest_cells
        if isinstance(cell, Mapping)
    }
    population = contract.get("population")
    if not isinstance(population, Mapping):
        raise LatentTerminalReconstructabilityError("L01 population is invalid")
    contract_cells = population.get("cells")
    if not isinstance(contract_cells, list) or len(contract_cells) != EXPECTED_CELL_COUNT:
        raise LatentTerminalReconstructabilityError("L01 population cells are invalid")
    cells = [
        _audit_cell(
            root=resolved,
            run_root=raw_root,
            manifest_cell=manifest_by_id[str(cell["cell_id"])],
            contract_cell=cell,
        )
        for cell in contract_cells
        if isinstance(cell, Mapping)
    ]
    unit_rows = [unit for cell in cells for unit in cell["discard_units"]]
    if len(unit_rows) != EXPECTED_DISCARD_COUNT:
        raise LatentTerminalReconstructabilityError(
            "reconstructability census is not 36 units"
        )
    source_manifest = _source_manifest(resolved)
    gates = {
        "L01_contract_valid_and_outcome_blind": (
            contract.get("freeze", {}).get("latent_outcomes_read") is False
            and contract.get("freeze", {}).get("hidden_pre_discard_states_read") is False
            and contract.get("freeze", {}).get("formal_execution_authorized") is False
        ),
        "raw_root_exactly_matches_terminal_index": raw_root_audit[
            "all_paths_sizes_and_hashes_match"
        ],
        "ten_source_trajectories_exactly_replay": all(
            cell["exact_full_trajectory_replay"]["verified"] for cell in cells
        ),
        "all_36_frozen_discards_enumerated": len(unit_rows)
        == EXPECTED_DISCARD_COUNT,
        "all_36_pre_discard_checkpoints_reconstructable": all(
            row["reconstructable"] for row in unit_rows
        ),
        "all_36_resource_ledgers_match_recorded_prefix": all(
            row["gates"]["campaign_resource_ledger_matches_recorded_prefix"]
            for row in unit_rows
        ),
        "all_36_hidden_states_match_independent_replay": all(
            row["gates"]["independent_hidden_state_replay_match"]
            for row in unit_rows
        ),
        "original_source_bytes_unchanged": all(
            row["gates"]["source_trajectory_unchanged"] for row in unit_rows
        ),
        "shadow_terminal_evaluations_executed_zero": all(
            row["shadow_terminal_executed"] is False for row in unit_rows
        ),
        "latent_discard_scores_accessed_zero": all(
            row["latent_discard_score_accessed"] is False for row in unit_rows
        ),
        "agent_provider_calls_zero": all(
            row["agent_provider_calls"] == 0 for row in unit_rows
        ),
    }
    report: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "status": "reconstructable" if all(gates.values()) else "failed_closed",
        "purpose": (
            "Prove whether every frozen pre-discard evaluator checkpoint can be "
            "reconstructed exactly without evaluating a discarded state."
        ),
        "counting_rule": (
            "One audit unit per committed discard_batch enumerated by the frozen L01 "
            "contract; no sampling, deduplication, score filtering, or retries that "
            "replace a first valid reconstruction."
        ),
        "evidence_bindings": {
            "latent_terminal_contract_sha256": latent_terminal_contract_sha256(
                contract
            ),
            "population_manifest_sha256": population.get(
                "population_manifest_sha256"
            ),
            "matrix_manifest_sha256": FROZEN_MATRIX_MANIFEST_SHA256,
            "terminal_file_index_sha256": FROZEN_TERMINAL_INDEX_SHA256,
            "source_manifest": source_manifest,
            "source_manifest_sha256": canonical_json_sha256(source_manifest),
        },
        "raw_root_audit": raw_root_audit,
        "census": {
            "cell_count": len(cells),
            "discard_unit_count": len(unit_rows),
            "reconstructable_unit_count": sum(
                int(row["reconstructable"]) for row in unit_rows
            ),
            "unresolved_unit_count": sum(
                int(not row["reconstructable"]) for row in unit_rows
            ),
            "shadow_terminal_evaluations_executed": 0,
            "latent_discard_scores_accessed": 0,
            "agent_provider_calls": 0,
        },
        "checkpoint_contract": {
            "capture_point": "immediately before the original discard_batch action",
            "private_state_publication": "hashes only; hidden payloads are not emitted",
            "independent_replays_per_cell": 2,
            "original_action_after_capture": (
                "executed only to replay the already observed source trajectory and "
                "reach later frozen discard checkpoints"
            ),
            "terminal_replacement_or_shadow_assay": False,
        },
        "historical_identity_limit": (
            "The historical raw trajectory did not persist a pre-discard hidden-state "
            "digest. This audit therefore binds deterministic independent reconstruction "
            "to the frozen configuration, exact recorded prefix, public observations, "
            "resource-ledger hash, and source bytes; it does not claim comparison with "
            "a previously published hidden-state digest."
        ),
        "gates": gates,
        "cells": cells,
    }
    report["report_sha256"] = reconstructability_report_sha256(report)
    return report


def validate_reconstructability_report(
    payload: Mapping[str, Any], *, root: Path | None = None
) -> list[str]:
    """Return fail-closed deterministic errors for a candidate L02 report."""

    errors: list[str] = []
    if payload.get("schema_id") != REPORT_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("report_id") != REPORT_ID:
        errors.append("report_id mismatch")
    if payload.get("report_sha256") != reconstructability_report_sha256(payload):
        errors.append("report self-hash mismatch")
    census = payload.get("census")
    expected_census = {
        "cell_count": EXPECTED_CELL_COUNT,
        "discard_unit_count": EXPECTED_DISCARD_COUNT,
        "reconstructable_unit_count": EXPECTED_DISCARD_COUNT,
        "unresolved_unit_count": 0,
        "shadow_terminal_evaluations_executed": 0,
        "latent_discard_scores_accessed": 0,
        "agent_provider_calls": 0,
    }
    if census != expected_census:
        errors.append("census is incomplete or outcome boundary was crossed")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(
        value is not True for value in gates.values()
    ):
        errors.append("one or more reconstructability gates failed")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CELL_COUNT:
        errors.append("cell reports are incomplete")
    else:
        units = [
            unit
            for cell in cells
            if isinstance(cell, Mapping)
            for unit in cell.get("discard_units", [])
            if isinstance(unit, Mapping)
        ]
        if len(units) != EXPECTED_DISCARD_COUNT:
            errors.append("discard-unit reports are incomplete")
        elif any(
            unit.get("reconstructable") is not True
            or unit.get("shadow_terminal_executed") is not False
            or unit.get("latent_discard_score_accessed") is not False
            or unit.get("agent_provider_calls") != 0
            for unit in units
        ):
            errors.append("discard-unit boundary or reconstructability failure")
        forbidden = {"latent_terminal_score", "leaderboard_score", "truth"}
        if any(forbidden & set(unit) for unit in units):
            errors.append("report leaks a latent score or hidden-state payload")
    if payload.get("status") != "reconstructable":
        errors.append("report status is not reconstructable")
    if root is not None:
        bindings = payload.get("evidence_bindings")
        if not isinstance(bindings, Mapping):
            errors.append("evidence_bindings must be an object")
        else:
            manifest = _source_manifest(root.resolve())
            if bindings.get("source_manifest") != manifest or bindings.get(
                "source_manifest_sha256"
            ) != canonical_json_sha256(manifest):
                errors.append("source manifest is stale")
    return errors


__all__ = [
    "CONTRACT_PATH",
    "REPORT_ID",
    "REPORT_SCHEMA_ID",
    "REPORT_SCHEMA_VERSION",
    "SOURCE_PATHS",
    "LatentTerminalReconstructabilityError",
    "build_reconstructability_report",
    "discover_run_root",
    "reconstructability_report_sha256",
    "validate_reconstructability_report",
]
