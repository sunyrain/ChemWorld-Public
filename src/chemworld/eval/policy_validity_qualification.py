"""Outcome-free runner qualification and formal protocol freeze for Work I.

The qualification has two deliberately separate evidence paths:

* a complete V05 matrix driven by an injected synthetic executor and consumed
  by the V06 auditor; and
* a fixed nonformal-world smoke using the released V04 controllers.

Neither path invokes a formal world.  The resulting receipt authorizes the
separate W1-V08 formal task only when every frozen gate passes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.campaign_resources import (
    CampaignResourceLedger,
    campaign_resource_event_id,
)
from chemworld.eval.known_policy_contract import (
    FORMAL_WORLD_SEEDS,
    INFORMATION_ARMS,
    LIFECYCLES_PER_CELL,
    POLICY_IDS,
    PROBE_SCHEDULE,
    known_policy_contract_sha256,
)
from chemworld.eval.policy_validity_audit import audit_policy_validity_manifest
from chemworld.eval.policy_validity_matrix import (
    FORMAL_QUALIFICATION_RECEIPT_SCHEMA_ID,
    FORMAL_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    MatrixCell,
    PolicyMatrixError,
    assert_execution_apparatus,
    build_cell_bundle,
    build_preflight,
    build_profile_record_from_execution,
    campaign_resource_card,
    dependency_bindings,
    execute_known_policy_campaign,
    run_matrix,
    semantic_sha256,
    validate_formal_qualification_receipt,
    validate_preflight,
)
from chemworld.eval.provenance import file_sha256, write_json_atomic

QUALIFICATION_SCHEMA_ID = "chemworld.policy_control_runner_qualification"
QUALIFICATION_SCHEMA_VERSION = "0.1.0"
QUALIFICATION_PROTOCOL_SCHEMA_ID = (
    "chemworld.policy_control_runner_qualification_protocol"
)
QUALIFICATION_PROTOCOL_SCHEMA_VERSION = "0.1.0"
LIVE_MANIFEST_SCHEMA_ID = "chemworld.policy_control_live_qualification_manifest"
LIVE_MANIFEST_SCHEMA_VERSION = "0.1.0"
DELIVERY_MANIFEST_SCHEMA_ID = (
    "chemworld.policy_control_runner_qualification_delivery_manifest"
)
DELIVERY_MANIFEST_SCHEMA_VERSION = "0.1.0"
QUALIFICATION_VERSION = "work-i-policy-control-runner-qualification-0.1"
_COMPATIBLE_POST_FREEZE_PREFLIGHT_HASHES = {
    "src/chemworld/agents/known_policy.py": (
        "1abca1369e0c12cc9bea7b7cdfca4a589d676b85b2a4b54d17b4ca7e7a8dd330"
    ),
    "src/chemworld/runtime/electrochemical_services.py": (
        "e6c6f9a9ad6cc39ef7838d16ec50adaf107079f986d86f0fb599bb7e559ab46b"
    ),
}


class PolicyQualificationError(RuntimeError):
    """Qualification evidence is missing, stale, or fails a frozen gate."""


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop(field, None)
    return result


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyQualificationError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PolicyQualificationError(f"{label} must be a JSON object")
    return payload


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyQualificationError(f"{label} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise PolicyQualificationError(f"{label} must be finite")
    return result


def validate_qualification_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Return deterministic errors for the V07-only qualification protocol."""

    errors: list[str] = []
    if protocol.get("schema_id") != QUALIFICATION_PROTOCOL_SCHEMA_ID:
        errors.append("qualification protocol schema_id mismatch")
    if protocol.get("schema_version") != QUALIFICATION_PROTOCOL_SCHEMA_VERSION:
        errors.append("qualification protocol schema_version mismatch")
    if tuple(protocol.get("formal_world_seeds_excluded", ())) != FORMAL_WORLD_SEEDS:
        errors.append("formal-world exclusion does not match the frozen matrix")

    synthetic = protocol.get("synthetic_matrix")
    if not isinstance(synthetic, Mapping):
        errors.append("synthetic_matrix must be an object")
    else:
        if synthetic.get("execution_mode") != "injected_test":
            errors.append("synthetic qualification must use injected_test mode")
        signals = synthetic.get("threshold_signals")
        if not isinstance(signals, list) or len(signals) != LIFECYCLES_PER_CELL:
            errors.append("synthetic threshold signal vector must contain six values")
        else:
            try:
                values = [
                    _finite_number(value, label="synthetic threshold signal")
                    for value in signals
                ]
            except PolicyQualificationError as exc:
                errors.append(str(exc))
            else:
                if not any(value < 0.007984561379998922 for value in values) or not any(
                    value >= 0.007984561379998922 for value in values
                ):
                    errors.append("synthetic threshold vector must exercise both branches")
        expected_synthetic = {
            "expected_primary_campaigns": 30,
            "expected_primary_closed_lifecycles": 180,
            "expected_retest_campaigns": 30,
            "expected_retest_closed_lifecycles": 180,
            "provider_calls": 0,
        }
        for field, expected in expected_synthetic.items():
            if synthetic.get(field) != expected:
                errors.append(f"synthetic matrix count is stale: {field}")

    live = protocol.get("live_smoke")
    if not isinstance(live, Mapping):
        errors.append("live_smoke must be an object")
    else:
        seed = live.get("world_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            errors.append("live qualification seed must be an integer")
        elif seed in FORMAL_WORLD_SEEDS:
            errors.append("live qualification seed overlaps a formal world")
        if tuple(live.get("information_arms", ())) != INFORMATION_ARMS:
            errors.append("live qualification information arms are stale")
        if tuple(live.get("policy_ids", ())) != POLICY_IDS:
            errors.append("live qualification policy IDs are stale")
        expected_live = {
            "expected_primary_campaigns": 6,
            "expected_primary_closed_lifecycles": 36,
            "expected_retest_campaigns": 6,
            "expected_retest_closed_lifecycles": 36,
            "provider_calls": 0,
        }
        for field, expected in expected_live.items():
            if live.get(field) != expected:
                errors.append(f"live smoke count is stale: {field}")

    source_paths = protocol.get("source_paths")
    if (
        not isinstance(source_paths, list)
        or not source_paths
        or any(not isinstance(path, str) or not path for path in source_paths)
        or len(source_paths) != len(set(source_paths))
    ):
        errors.append("source_paths must be a non-empty unique string list")
    output_paths = protocol.get("output_paths")
    required_outputs = {
        "artifact_root",
        "report",
        "markdown",
        "receipt",
        "delivery_manifest",
    }
    if not isinstance(output_paths, Mapping) or set(output_paths) != required_outputs:
        errors.append("qualification output paths are incomplete")
    elif any(
        not isinstance(output_paths[field], str) or not output_paths[field]
        for field in required_outputs
    ):
        errors.append("qualification output paths must be non-empty strings")
    freeze = protocol.get("freeze_rules")
    required_freeze = (
        "formal_environment_execution_forbidden",
        "formal_outcome_read_forbidden",
        "formal_retuning_forbidden",
        "qualification_failure_is_terminal",
        "seed_substitution_forbidden",
        "threshold_change_forbidden",
        "acceptance_rule_change_forbidden",
    )
    if not isinstance(freeze, Mapping) or any(
        freeze.get(field) is not True for field in required_freeze
    ):
        errors.append("qualification freeze rules are incomplete")
    return errors


def load_qualification_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json_object(path, label="V07 qualification protocol")
    errors = validate_qualification_protocol(protocol)
    if errors:
        raise PolicyQualificationError("; ".join(errors))
    return protocol


def qualification_source_manifest(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, str]:
    resolved = root.resolve()
    result: dict[str, str] = {}
    for relative in protocol["source_paths"]:
        path = (resolved / str(relative)).resolve()
        if not path.is_relative_to(resolved) or not path.is_file():
            raise PolicyQualificationError(f"missing qualification source: {relative}")
        result[str(relative)] = file_sha256(path)
    return result


def assert_qualification_outputs_absent(
    root: Path, protocol: Mapping[str, Any]
) -> None:
    """Fail closed before a non-check run can overwrite any frozen deliverable."""

    paths = protocol["output_paths"]
    candidates = {
        str(paths[field]): root / str(paths[field])
        for field in (
            "artifact_root",
            "report",
            "markdown",
            "receipt",
            "delivery_manifest",
        )
    }
    occupied = sorted(relative for relative, path in candidates.items() if path.exists())
    if occupied:
        raise PolicyQualificationError(
            "refusing to overwrite qualification outputs: " + ", ".join(occupied)
        )


def _synthetic_actions(
    policy_id: str, lifecycle_index: int, threshold_signal: float
) -> list[dict[str, Any]]:
    probe = PROBE_SCHEDULE[lifecycle_index]
    prefix = [
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": probe.solvent},
        {"operation": "add_reagent", "amount_mol": probe.reagent_amount_mol},
        {
            "operation": "set_potential",
            "potential_V": probe.potential_V,
            "current_mA": probe.current_mA,
            "electrolyte_profile": probe.electrolyte_profile,
        },
        {"operation": "electrolyze", "duration_s": probe.probe_duration_s},
    ]
    if policy_id == "assay_all":
        return [
            *prefix,
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
    if policy_id == "start_then_discard":
        return [
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": probe.solvent},
            {"operation": "discard_batch", "reason": "known_policy_immediate_discard"},
        ]
    if threshold_signal < 0.007984561379998922:
        return [
            *prefix,
            {"operation": "measure", "instrument": "uvvis"},
            {"operation": "discard_batch", "reason": "known_policy_below_threshold"},
        ]
    return [
        *prefix,
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "electrolyze", "duration_s": probe.post_measure_duration_s},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def _synthetic_terminal_kind(action: Mapping[str, Any]) -> str | None:
    if action.get("operation") == "discard_batch":
        return "discard"
    if action.get("operation") == "measure" and action.get("instrument") == "final_assay":
        return "assay"
    return None


def _synthetic_controller_manifest(policy_id: str) -> dict[str, Any]:
    identity = semantic_sha256(
        {
            "role": "injected-synthetic-v07-qualification",
            "policy_id": policy_id,
        }
    )
    return {
        "schema_id": "chemworld.known_policy_controller",
        "schema_version": "0.1.0",
        "policy_id": policy_id,
        "artifact_bindings": {
            "known_policy_contract_sha256": known_policy_contract_sha256(),
            "threshold_binding_sha256": (
                "8a55b713ca900a644a301ecd8e83a0a686f9a28b3f60a18a85a4e57b66288c6a"
            ),
            "threshold": 0.007984561379998922,
            "diagnostic_signal": "observation.conversion",
            "comparator": ">=",
        },
        "reads_material_information": False,
        "provider_call_count": 0,
        "controller_sha256": identity,
        "qualification_only": True,
    }


def synthetic_qualification_execution(
    cell: MatrixCell,
    matrix_protocol: Mapping[str, Any],
    *,
    execution_role: str,
    qualification_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic non-chemical evidence through the V05 contract."""

    if execution_role not in {"original", "retest"}:
        raise PolicyQualificationError("synthetic execution role is invalid")
    signals = tuple(
        float(value)
        for value in qualification_protocol["synthetic_matrix"]["threshold_signals"]
    )
    synthetic = qualification_protocol["synthetic_matrix"]
    card = campaign_resource_card(matrix_protocol)
    ledger = CampaignResourceLedger(card)
    records: list[dict[str, Any]] = []
    terminals: list[dict[str, Any]] = []
    operation_attempt = 0
    logical_campaign_id = (
        f"{qualification_protocol['protocol_id']}--synthetic-resource--"
        f"world-{cell.world_seed:04d}-{cell.policy_id}"
    )
    for lifecycle_index, signal in enumerate(signals):
        measurement_seen = False
        actions = _synthetic_actions(cell.policy_id, lifecycle_index, signal)
        for within_lifecycle_index, action in enumerate(actions):
            operation_attempt += 1
            event_id = campaign_resource_event_id(logical_campaign_id, operation_attempt)
            starts_vessel = within_lifecycle_index == 0
            preflight = ledger.preflight(event_id, action, starts_vessel=starts_vessel)
            if not preflight.allowed:
                raise PolicyQualificationError("synthetic resource preflight rejected action")
            ledger.record_outcome(
                event_id,
                action,
                {
                    "transaction_status": "committed",
                    "campaign_resource_report_delta": {
                        "physical_cost": 0.1,
                        "accumulated_risk": 0.01,
                    },
                },
                starts_vessel=starts_vessel,
            )
            operation = str(action["operation"])
            instrument = action.get("instrument")
            observation: dict[str, Any] = {}
            if operation == "measure" and instrument == "uvvis":
                observation = {"conversion": signal}
            terminal_kind = _synthetic_terminal_kind(action)
            terminal_score = (
                0.5 + lifecycle_index * 0.01 if terminal_kind == "assay" else None
            )
            if terminal_score is not None:
                observation = {"score": terminal_score}
            state = {
                "qualification_identity": str(synthetic["physics_identity"]),
                "schedule_coordinate": cell.world_seed,
                "policy_id": cell.policy_id,
                "lifecycle_index": lifecycle_index,
                "within_lifecycle_index": within_lifecycle_index,
            }
            resource_state = deepcopy(ledger.snapshot()["state"])
            decision_audit = {
                "action": deepcopy(action),
                "adaptation_source": "measurement" if measurement_seen else "none",
                "status": "provided",
                "material_information_accessed": False,
                "provider_call_count": 0,
            }
            if terminal_kind is not None:
                terminals.append(
                    {
                        "lifecycle_index": lifecycle_index,
                        "terminal_kind": terminal_kind,
                        "terminal_score": terminal_score,
                        "terminal_event_index": operation_attempt,
                    }
                )
            record = {
                "event_index": operation_attempt,
                "lifecycle_index": lifecycle_index,
                "action": deepcopy(action),
                "observation": observation,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "event_type": (
                    "experiment_end"
                    if terminal_kind == "assay"
                    else "batch_discard"
                    if terminal_kind == "discard"
                    else "operation_result"
                ),
                "info": {
                    "transaction_status": "committed",
                    "experiment_ended": terminal_kind is not None,
                },
                "state": state,
                "campaign_resource_state": resource_state,
                "decision_audit": decision_audit,
            }
            record["observation_sha256"] = semantic_sha256(observation)
            record["state_sha256"] = semantic_sha256(state)
            record["campaign_resource_state_sha256"] = semantic_sha256(resource_state)
            records.append(record)
            if operation == "measure" and instrument == "uvvis":
                measurement_seen = True

    snapshot = ledger.snapshot()
    identity_prefix = str(synthetic["world_identity_prefix"])
    world_id = f"{identity_prefix}-{cell.world_seed:04d}"
    profile = build_profile_record_from_execution(
        cell=cell,
        world_id=world_id,
        resource_card_sha256=card.card_sha256,
        trajectory_records=records,
        campaign_resource_ledger=snapshot,
        lifecycle_terminals=terminals,
        threshold=0.007984561379998922,
    )
    return {
        "schema_id": "chemworld.policy_control_campaign_execution",
        "schema_version": "0.1.0",
        "execution_role": execution_role,
        "identity": {
            "campaign_id": cell.cell_id,
            "cell_id": cell.cell_id,
            "world_seed": cell.world_seed,
            "world_id": world_id,
            "information_arm": cell.information_arm,
            "policy_id": cell.policy_id,
            "resource_card_sha256": card.card_sha256,
            "observation_noise_namespace": (
                f"{synthetic['noise_identity']}--world-{cell.world_seed:04d}"
            ),
            "physical_identity": {
                "world_seed_schedule_coordinate": cell.world_seed,
                "world_id": world_id,
                "physics": str(synthetic["physics_identity"]),
                "formal_chemical_world": False,
            },
            "material_information_sha256": semantic_sha256(cell.material_information),
            "qualification_only": True,
            "qualification_namespace": qualification_protocol["protocol_id"],
            "qualification_role": "injected_synthetic_runner_audit",
        },
        "controller_manifest": _synthetic_controller_manifest(cell.policy_id),
        "trajectory_records": records,
        "campaign_resource_ledger": snapshot,
        "lifecycle_terminals": terminals,
        "profile_record": profile,
        "decision_audits": [record["decision_audit"] for record in records],
        "counts": {
            **profile["counts"],
            "attempted_operation_count": len(records),
            "committed_operation_count": len(records),
            "provider_call_count": 0,
        },
    }


def _artifact_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "file_sha256": file_sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    return entries


def _rendered_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def build_qualification_delivery_manifest(
    *,
    qualification_protocol: Mapping[str, Any],
    artifact_root: Path,
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    markdown: str,
) -> dict[str, Any]:
    """Bind every payload artifact plus the three top-level deliverables by byte."""

    artifact_entries = _artifact_entries(artifact_root)
    if artifact_entries != report.get("artifact_manifest"):
        raise PolicyQualificationError(
            "qualification artifact payload differs from report inventory"
        )
    paths = qualification_protocol["output_paths"]
    artifact_root_path = str(paths["artifact_root"]).replace("\\", "/").rstrip("/")
    entries: list[dict[str, Any]] = []
    for entry in artifact_entries:
        entries.append(
            {
                "role": "qualification_artifact",
                "path": f"{artifact_root_path}/{entry['path']}",
                "file_sha256": entry["file_sha256"],
                "byte_count": entry["byte_count"],
            }
        )
    top_level = (
        ("qualification_report", str(paths["report"]), _rendered_json_bytes(report)),
        (
            "formal_qualification_receipt",
            str(paths["receipt"]),
            _rendered_json_bytes(receipt),
        ),
        ("qualification_markdown", str(paths["markdown"]), markdown.encode("utf-8")),
    )
    for role, path, content in top_level:
        entries.append(
            {
                "role": role,
                "path": path.replace("\\", "/"),
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    entries.sort(key=lambda entry: str(entry["path"]))
    manifest: dict[str, Any] = {
        "schema_id": DELIVERY_MANIFEST_SCHEMA_ID,
        "schema_version": DELIVERY_MANIFEST_SCHEMA_VERSION,
        "task_id": "W1-V07",
        "status": "complete",
        "immutable": True,
        "formal_result": False,
        "entry_count": len(entries),
        "entries": entries,
        "bindings": {
            "qualification_report_sha256": report["report_sha256"],
            "formal_qualification_receipt_sha256": receipt["receipt_sha256"],
            "qualification_artifact_manifest_sha256": report[
                "artifact_manifest_sha256"
            ],
        },
        "counting_rule": (
            "Every generated qualification payload file and each top-level report, "
            "receipt, and Markdown deliverable appears exactly once. This manifest "
            "excludes only its own file to avoid a circular byte hash."
        ),
    }
    manifest["delivery_manifest_sha256"] = semantic_sha256(manifest)
    return manifest


def _live_pair_audits(
    bundles: Mapping[str, Mapping[str, Any]], cells: Sequence[MatrixCell]
) -> list[dict[str, Any]]:
    by_key = {
        (cell.information_arm, cell.policy_id): bundles[cell.cell_id]
        for cell in cells
    }
    audits: list[dict[str, Any]] = []
    for policy_id in POLICY_IDS:
        left = by_key[(INFORMATION_ARMS[0], policy_id)]["original"]
        right = by_key[(INFORMATION_ARMS[1], policy_id)]["original"]
        left_profile = left["profile_record"]
        right_profile = right["profile_record"]
        checks = {
            "physical_identity": left["identity"]["physical_identity"]
            == right["identity"]["physical_identity"],
            "distinct_material_information": left["identity"][
                "material_information_sha256"
            ]
            != right["identity"]["material_information_sha256"],
            "actions": semantic_sha256(
                [record["action"] for record in left["trajectory_records"]]
            )
            == semantic_sha256(
                [record["action"] for record in right["trajectory_records"]]
            ),
            "observations": semantic_sha256(
                [record["observation"] for record in left["trajectory_records"]]
            )
            == semantic_sha256(
                [record["observation"] for record in right["trajectory_records"]]
            ),
            "profile_values": semantic_sha256(
                {
                    "counts": left_profile["counts"],
                    "construct_axes": left_profile["construct_axes"],
                    "endpoint_context": left_profile["endpoint_context"],
                }
            )
            == semantic_sha256(
                {
                    "counts": right_profile["counts"],
                    "construct_axes": right_profile["construct_axes"],
                    "endpoint_context": right_profile["endpoint_context"],
                }
            ),
            "resource_state": left["campaign_resource_ledger"]["state"]
            == right["campaign_resource_ledger"]["state"],
        }
        audits.append(
            {
                "policy_id": policy_id,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return audits


def derive_live_smoke_protocol(
    matrix_protocol: Mapping[str, Any],
    qualification_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the self-describing 1 x 2 x 3 nonformal live protocol."""

    live = qualification_protocol["live_smoke"]
    protocol = deepcopy(dict(matrix_protocol))
    protocol["protocol_id"] = (
        f"{qualification_protocol['protocol_id']}-live-smoke"
    )
    protocol["matrix"] = {
        "world_seeds": [int(live["world_seed"])],
        "information_arms": list(live["information_arms"]),
        "policy_ids": list(live["policy_ids"]),
        "lifecycles_per_campaign": LIFECYCLES_PER_CELL,
        "primary_campaign_count": int(live["expected_primary_campaigns"]),
        "primary_closed_lifecycle_count": int(
            live["expected_primary_closed_lifecycles"]
        ),
        "provider_call_count": int(live["provider_calls"]),
        "schedule_order": "world_then_arm_then_policy",
    }
    protocol["task"]["observation_seed_offset"] = int(
        live["observation_seed_offset"]
    )
    protocol["task"]["observation_noise_namespace_template"] = str(
        live["observation_noise_namespace_template"]
    )
    protocol["qualification_context"] = {
        "formal_result": False,
        "qualification_only": True,
        "role": "fixed_nonformal_controller_runtime_smoke",
        "source_protocol_id": matrix_protocol["protocol_id"],
    }
    errors = validate_live_smoke_protocol(protocol, qualification_protocol)
    if errors:
        raise PolicyQualificationError(
            "invalid derived live-smoke protocol: " + "; ".join(errors)
        )
    return protocol


def validate_live_smoke_protocol(
    protocol: Mapping[str, Any], qualification_protocol: Mapping[str, Any]
) -> list[str]:
    """Validate that a derived live protocol exactly describes its 1 x 2 x 3 run."""

    errors: list[str] = []
    live = qualification_protocol["live_smoke"]
    matrix = protocol.get("matrix")
    if not isinstance(matrix, Mapping):
        return ["derived live protocol matrix must be an object"]
    expected_matrix = {
        "world_seeds": [int(live["world_seed"])],
        "information_arms": list(INFORMATION_ARMS),
        "policy_ids": list(POLICY_IDS),
        "lifecycles_per_campaign": LIFECYCLES_PER_CELL,
        "primary_campaign_count": 6,
        "primary_closed_lifecycle_count": 36,
        "provider_call_count": 0,
        "schedule_order": "world_then_arm_then_policy",
    }
    if dict(matrix) != expected_matrix:
        errors.append("derived live protocol matrix is not the frozen 1 x 2 x 3 smoke")
    if int(live["world_seed"]) in FORMAL_WORLD_SEEDS:
        errors.append("derived live protocol overlaps a formal world")
    task = protocol.get("task")
    if not isinstance(task, Mapping):
        errors.append("derived live protocol task must be an object")
    else:
        if task.get("observation_seed_offset") != live["observation_seed_offset"]:
            errors.append("derived live protocol observation seed offset is stale")
        if (
            task.get("observation_noise_namespace_template")
            != live["observation_noise_namespace_template"]
        ):
            errors.append("derived live protocol noise namespace is stale")
    context = protocol.get("qualification_context")
    if not isinstance(context, Mapping) or (
        context.get("formal_result") is not False
        or context.get("qualification_only") is not True
        or context.get("role") != "fixed_nonformal_controller_runtime_smoke"
    ):
        errors.append("derived live protocol qualification context is incomplete")
    return errors


def run_live_smoke(
    *,
    root: Path,
    matrix_protocol: Mapping[str, Any],
    qualification_protocol: Mapping[str, Any],
    output_root: Path,
    qualification_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Run the released controllers on one fixed, explicitly nonformal world."""

    live = qualification_protocol["live_smoke"]
    world_seed = int(live["world_seed"])
    if world_seed in FORMAL_WORLD_SEEDS:
        raise PolicyQualificationError("live smoke seed overlaps formal worlds")
    live_protocol = derive_live_smoke_protocol(
        matrix_protocol, qualification_protocol
    )
    card = campaign_resource_card(live_protocol)
    dependencies = dependency_bindings(root, matrix_protocol)
    qualification_source_sha256 = semantic_sha256(qualification_sources)
    cells: list[MatrixCell] = []
    arm_payloads = matrix_protocol["material_information_by_arm"]
    for arm in INFORMATION_ARMS:
        arm_slug = "opaque" if arm == INFORMATION_ARMS[0] else "anonymous-nominal"
        for policy_id in POLICY_IDS:
            ordinal = len(cells) + 1
            cells.append(
                MatrixCell(
                    ordinal=ordinal,
                    cell_id=(
                        f"qualification-live-cell-{ordinal:02d}-world-{world_seed:05d}-"
                        f"{arm_slug}-{policy_id.replace('_', '-')}"
                    ),
                    world_seed=world_seed,
                    information_arm=arm,
                    policy_id=policy_id,
                    material_information=deepcopy(dict(arm_payloads[arm])),
                )
            )

    bundle_root = output_root / "bundles"
    bundle_root.mkdir(parents=True, exist_ok=False)
    bundles: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    live_protocol_sha256 = semantic_sha256(live_protocol)
    for cell in cells:
        original = execute_known_policy_campaign(
            cell, live_protocol, execution_role="original"
        )
        retest = execute_known_policy_campaign(
            cell, live_protocol, execution_role="retest"
        )
        bundle = build_cell_bundle(
            cell=cell,
            protocol_sha256=live_protocol_sha256,
            source_manifest_sha256=qualification_source_sha256,
            dependency_identity=dependencies,
            card_sha256=card.card_sha256,
            original=original,
            retest=retest,
        )
        path = bundle_root / f"{cell.cell_id}.json"
        write_json_atomic(path, bundle)
        bundles[cell.cell_id] = bundle
        entries.append(
            {
                "ordinal": cell.ordinal,
                "cell_id": cell.cell_id,
                "world_seed": cell.world_seed,
                "information_arm": cell.information_arm,
                "policy_id": cell.policy_id,
                "bundle_path": path.relative_to(output_root).as_posix(),
                "bundle_sha256": bundle["bundle_sha256"],
                "file_sha256": file_sha256(path),
                "byte_count": path.stat().st_size,
                "primary_closed_lifecycles": LIFECYCLES_PER_CELL,
                "retest_closed_lifecycles": LIFECYCLES_PER_CELL,
                "provider_calls": 0,
            }
        )

    pair_audits = _live_pair_audits(bundles, cells)
    controller_source_sha256 = dependencies["controller"]["sha256"]
    controller_bindings_match = all(
        bundle["original"]["controller_manifest"].get(
            "controller_source_sha256"
        )
        == controller_source_sha256
        for bundle in bundles.values()
    )
    gates = {
        "fixed_nonformal_seed": world_seed == 20_000 and world_seed not in FORMAL_WORLD_SEEDS,
        "self_consistent_derived_protocol": not validate_live_smoke_protocol(
            live_protocol, qualification_protocol
        ),
        "six_primary_campaigns": len(entries) == 6,
        "all_36_primary_lifecycles_closed": all(
            bundle["original"]["counts"]["closed_lifecycle_count"] == 6
            for bundle in bundles.values()
        ),
        "all_36_retest_lifecycles_closed": all(
            bundle["retest"]["counts"]["closed_lifecycle_count"] == 6
            for bundle in bundles.values()
        ),
        "all_original_retest_components_match": all(
            bundle["retest_audit"]["same_identity"]
            and bundle["retest_audit"]["all_components_match"]
            for bundle in bundles.values()
        ),
        "matched_arm_invariance": all(audit["passed"] for audit in pair_audits),
        "released_controller_source_bound": controller_bindings_match,
        "zero_provider_calls": all(
            bundle[role]["counts"]["provider_call_count"] == 0
            for bundle in bundles.values()
            for role in ("original", "retest")
        ),
    }
    manifest: dict[str, Any] = {
        "schema_id": LIVE_MANIFEST_SCHEMA_ID,
        "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "passed": all(gates.values()),
        "formal_result": False,
        "formal_environment_execution_count": 0,
        "world_seed": world_seed,
        "formal_world_seeds_excluded": list(FORMAL_WORLD_SEEDS),
        "live_protocol_sha256": live_protocol_sha256,
        "live_protocol": live_protocol,
        "qualification_source_manifest_sha256": qualification_source_sha256,
        "controller_source_sha256": controller_source_sha256,
        "campaign_resource_card_sha256": card.card_sha256,
        "counts": {
            "primary_campaigns": len(entries),
            "primary_closed_lifecycles": len(entries) * LIFECYCLES_PER_CELL,
            "retest_campaigns": len(entries),
            "retest_closed_lifecycles": len(entries) * LIFECYCLES_PER_CELL,
            "formal_campaigns": 0,
            "formal_closed_lifecycles": 0,
            "provider_calls": 0,
        },
        "gates": gates,
        "matched_arm_audits": pair_audits,
        "cells": entries,
        "counting_rules": deepcopy(qualification_protocol["counting_rules"]),
        "claim_boundary": (
            "This fixed seed-20000 smoke qualifies controller/runtime integration only. "
            "It is not a formal result or an endpoint-performance comparison."
        ),
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    write_json_atomic(output_root / "live_smoke_manifest.json", manifest)
    if not manifest["passed"]:
        raise PolicyQualificationError("one or more live-smoke gates failed")
    return manifest


def _v06_done(root: Path) -> bool:
    claim = (root / "workstreams/arxiv_v1/claims/W1-V06--codex.md").read_text(
        encoding="utf-8"
    )
    return "\nstatus: DONE\n" in claim and "\nfinal_commit: null\n" not in claim


def _matches_compatible_historical_preflight(
    committed: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Accept only the two reviewed post-freeze, non-scientific source revisions."""

    if validate_preflight(committed):
        return False
    normalized = deepcopy(dict(current))
    committed_sources = committed.get("source_manifest")
    current_sources = normalized.get("source_manifest")
    if not isinstance(committed_sources, Mapping) or not isinstance(
        current_sources, dict
    ):
        return False
    for path, expected_current_hash in _COMPATIBLE_POST_FREEZE_PREFLIGHT_HASHES.items():
        if current_sources.get(path) != expected_current_hash or path not in committed_sources:
            return False
        current_sources[path] = committed_sources[path]

    committed_dependencies = committed.get("dependency_bindings")
    current_dependencies = normalized.get("dependency_bindings")
    if not isinstance(committed_dependencies, Mapping) or not isinstance(
        current_dependencies, dict
    ):
        return False
    committed_controller = committed_dependencies.get("controller")
    current_controller = current_dependencies.get("controller")
    if not isinstance(committed_controller, Mapping) or not isinstance(
        current_controller, dict
    ):
        return False
    controller_path = "src/chemworld/agents/known_policy.py"
    if current_controller.get("sha256") != _COMPATIBLE_POST_FREEZE_PREFLIGHT_HASHES[
        controller_path
    ]:
        return False
    current_controller["sha256"] = committed_controller.get("sha256")
    normalized["source_manifest_sha256"] = committed.get("source_manifest_sha256")
    normalized["preflight_sha256"] = committed.get("preflight_sha256")
    return normalized == dict(committed)


def build_qualification(
    *, root: Path, protocol_path: Path, artifact_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Execute both nonformal qualification paths and return report/receipt/Markdown."""

    resolved_root = root.resolve()
    protocol = load_qualification_protocol(protocol_path)
    matrix_protocol_path = resolved_root / str(protocol["formal_matrix_protocol_path"])
    preflight_path = resolved_root / str(protocol["formal_preflight_path"])
    matrix_protocol = _read_json_object(
        matrix_protocol_path, label="formal matrix protocol"
    )
    try:
        apparatus = assert_execution_apparatus(resolved_root, matrix_protocol)
    except PolicyMatrixError as exc:
        raise PolicyQualificationError(
            f"execution apparatus qualification failed: {exc}"
        ) from exc
    preflight = build_preflight(resolved_root, matrix_protocol_path)
    preflight_errors = validate_preflight(preflight)
    if preflight_errors:
        raise PolicyQualificationError("invalid formal preflight: " + "; ".join(preflight_errors))
    committed_preflight = _read_json_object(
        preflight_path, label="committed formal preflight"
    )
    if committed_preflight != preflight and not _matches_compatible_historical_preflight(
        committed_preflight, preflight
    ):
        raise PolicyQualificationError("formal preflight does not rebuild exactly")
    if not _v06_done(resolved_root):
        raise PolicyQualificationError("W1-V06 is not DONE")

    sources = qualification_source_manifest(resolved_root, protocol)
    source_sha256 = semantic_sha256(sources)
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise PolicyQualificationError("refusing to overwrite qualification artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)

    synthetic_root = artifact_root / "synthetic-matrix"

    def executor(
        cell: MatrixCell, runner_protocol: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            synthetic_qualification_execution(
                cell,
                runner_protocol,
                execution_role="original",
                qualification_protocol=protocol,
            ),
            synthetic_qualification_execution(
                cell,
                runner_protocol,
                execution_role="retest",
                qualification_protocol=protocol,
            ),
        )

    synthetic_manifest = run_matrix(
        root=resolved_root,
        protocol_path=matrix_protocol_path,
        output_root=synthetic_root,
        executor=executor,
        resume=False,
        execution_mode="injected_test",
        allow_formal_execution=False,
        formal_qualification_receipt=None,
    )
    synthetic_audit = audit_policy_validity_manifest(
        synthetic_root / MANIFEST_FILENAME
    )
    write_json_atomic(artifact_root / "synthetic_audit.json", synthetic_audit)

    live_manifest = run_live_smoke(
        root=resolved_root,
        matrix_protocol=matrix_protocol,
        qualification_protocol=protocol,
        output_root=artifact_root / "live-smoke",
        qualification_sources=sources,
    )
    synthetic_config = protocol["synthetic_matrix"]
    synthetic_bundles = [
        _read_json_object(
            synthetic_root / str(entry["bundle_path"]),
            label="synthetic bundle",
        )
        for entry in synthetic_manifest["cells"]
    ]
    synthetic_executions = [
        (bundle, role, bundle[role])
        for bundle in synthetic_bundles
        for role in ("original", "retest")
    ]
    noise_by_match: dict[tuple[int, str, str], set[str]] = {}
    for bundle, role, execution in synthetic_executions:
        cell = bundle["cell"]
        key = (int(cell["world_seed"]), str(cell["policy_id"]), role)
        noise_by_match.setdefault(key, set()).add(
            str(execution["identity"]["observation_noise_namespace"])
        )
    synthetic_identity_gates = {
        "execution_mode_injected_test": synthetic_manifest.get("execution_mode")
        == "injected_test",
        "formal_result_false": synthetic_manifest.get("formal_result") is False,
        "all_original_retest_identities_match": all(
            bundle["original"]["identity"] == bundle["retest"]["identity"]
            for bundle in synthetic_bundles
        ),
        "all_roles_are_explicit": all(
            execution["execution_role"] == role
            for _, role, execution in synthetic_executions
        ),
        "all_qualification_markers_are_explicit": all(
            execution["identity"].get("qualification_only") is True
            and execution["identity"].get("qualification_namespace")
            == protocol["protocol_id"]
            and execution["identity"].get("qualification_role")
            == "injected_synthetic_runner_audit"
            for _, _, execution in synthetic_executions
        ),
        "all_world_ids_are_synthetic": all(
            str(execution["identity"]["world_id"]).startswith(
                str(synthetic_config["world_identity_prefix"])
            )
            for _, _, execution in synthetic_executions
        ),
        "all_noise_namespaces_are_synthetic": all(
            str(execution["identity"]["observation_noise_namespace"]).startswith(
                f"{synthetic_config['noise_identity']}--world-"
            )
            for _, _, execution in synthetic_executions
        ),
        "matched_arm_noise_namespaces_are_paired": all(
            len(namespaces) == 1 for namespaces in noise_by_match.values()
        )
        and len(noise_by_match) == len(FORMAL_WORLD_SEEDS) * len(POLICY_IDS) * 2,
        "campaign_profile_ids_are_schedule_coordinate_exception": all(
            execution["identity"]["campaign_id"] == bundle["cell"]["cell_id"]
            and execution["identity"]["cell_id"] == bundle["cell"]["cell_id"]
            and execution["profile_record"]["identity"]["campaign_id"]
            == bundle["cell"]["cell_id"]
            and str(execution["profile_record"]["identity"]["world_id"]).startswith(
                str(synthetic_config["world_identity_prefix"])
            )
            for bundle, _, execution in synthetic_executions
        ),
        "all_physics_identities_are_nonformal": all(
            execution["identity"]["physical_identity"].get("formal_chemical_world")
            is False
            and execution["identity"]["physical_identity"].get("physics")
            == synthetic_config["physics_identity"]
            for _, _, execution in synthetic_executions
        ),
    }
    gates = {
        "v06_dependency_done": True,
        "formal_preflight_exact_rebuild": True,
        "formal_outcomes_not_read": True,
        "formal_environment_not_executed": True,
        "synthetic_matrix_complete": synthetic_manifest["materialized_counts"]
        == synthetic_manifest["expected_counts"],
        "synthetic_identity_separation": all(synthetic_identity_gates.values()),
        "v06_all_gates_pass": synthetic_audit["passed"] is True
        and all(synthetic_audit["gates"].values()),
        "live_nonformal_smoke_pass": live_manifest["passed"] is True
        and all(live_manifest["gates"].values()),
        "zero_provider_calls": synthetic_audit["counts"]["provider_calls"] == 0
        and live_manifest["counts"]["provider_calls"] == 0,
        "formal_retuning_forbidden": protocol["freeze_rules"][
            "formal_retuning_forbidden"
        ]
        is True,
    }
    if not all(gates.values()):
        raise PolicyQualificationError("one or more V07 qualification gates failed")

    artifact_entries = _artifact_entries(artifact_root)
    report: dict[str, Any] = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "task_id": "W1-V07",
        "status": "qualified_and_frozen",
        "formal_result": False,
        "formal_environment_execution_count": 0,
        "formal_outcome_read_count": 0,
        "formal_world_seeds_excluded": list(FORMAL_WORLD_SEEDS),
        "qualification_protocol_sha256": semantic_sha256(protocol),
        "qualification_source_manifest": sources,
        "qualification_source_manifest_sha256": source_sha256,
        "formal_bindings": {
            "matrix_protocol_sha256": preflight["protocol_sha256"],
            "matrix_source_manifest_sha256": preflight["source_manifest_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "controller_sha256": preflight["dependency_bindings"]["controller"][
                "sha256"
            ],
            "execution_apparatus": apparatus,
            "execution_apparatus_sha256": preflight[
                "execution_apparatus_sha256"
            ],
            "auditor_sha256": sources["src/chemworld/eval/policy_validity_audit.py"],
        },
        "synthetic_matrix": {
            "role": "injected qualification evidence; no chemical world execution",
            "manifest_sha256": synthetic_manifest["manifest_sha256"],
            "audit_sha256": synthetic_audit["audit_sha256"],
            "counts": synthetic_audit["counts"],
            "gates": synthetic_audit["gates"],
            "identity_gates": synthetic_identity_gates,
        },
        "live_smoke": {
            "role": "fixed nonformal controller/runtime interface smoke",
            "manifest_sha256": live_manifest["manifest_sha256"],
            "world_seed": live_manifest["world_seed"],
            "counts": live_manifest["counts"],
            "gates": live_manifest["gates"],
        },
        "qualification_gates": gates,
        "counting_rules": deepcopy(protocol["counting_rules"]),
        "artifact_manifest": artifact_entries,
        "artifact_manifest_sha256": semantic_sha256(artifact_entries),
        "outer_delivery_manifest": {
            "path": protocol["output_paths"]["delivery_manifest"],
            "self_hash_field": "delivery_manifest_sha256",
            "self_excluded_to_avoid_circular_byte_hash": True,
        },
        "claim_boundary": (
            "V07 qualifies and freezes the runner protocol. Synthetic matrix evidence "
            "and seed-20000 live smoke are excluded from the W1-V08 formal estimand and "
            "support no endpoint-performance or chemical-world outcome claim."
        ),
    }
    report["report_sha256"] = semantic_sha256(report)

    receipt: dict[str, Any] = {
        "schema_id": FORMAL_QUALIFICATION_RECEIPT_SCHEMA_ID,
        "schema_version": FORMAL_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        "task_id": "W1-V07",
        "qualification_gates": {
            "runner_qualified": True,
            "protocol_frozen": True,
            "formal_outcomes_not_read": True,
            "formal_environment_not_executed": True,
            "v06_all_gates_pass": True,
            "live_nonformal_smoke_pass": True,
            "zero_provider_calls": True,
        },
        "bindings": {
            "matrix_protocol_sha256": preflight["protocol_sha256"],
            "source_manifest_sha256": preflight["source_manifest_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "controller_sha256": preflight["dependency_bindings"]["controller"][
                "sha256"
            ],
            "execution_apparatus_sha256": preflight[
                "execution_apparatus_sha256"
            ],
            "auditor_sha256": sources["src/chemworld/eval/policy_validity_audit.py"],
            "qualification_protocol_sha256": report[
                "qualification_protocol_sha256"
            ],
            "qualification_source_manifest_sha256": source_sha256,
            "qualification_report_sha256": report["report_sha256"],
            "qualification_artifact_manifest_sha256": report[
                "artifact_manifest_sha256"
            ],
            "synthetic_matrix_manifest_sha256": synthetic_manifest[
                "manifest_sha256"
            ],
            "synthetic_audit_sha256": synthetic_audit["audit_sha256"],
            "live_smoke_manifest_sha256": live_manifest["manifest_sha256"],
        },
    }
    receipt["receipt_sha256"] = semantic_sha256(receipt)
    receipt_errors = validate_formal_qualification_receipt(
        receipt, preflight=preflight
    )
    if receipt_errors:
        raise PolicyQualificationError(
            "formal qualification receipt is invalid: " + "; ".join(receipt_errors)
        )
    markdown = build_qualification_markdown(report, receipt)
    return report, receipt, markdown


def build_qualification_markdown(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> str:
    """Render a compact human audit record from frozen machine artifacts."""

    synthetic = report["synthetic_matrix"]
    live = report["live_smoke"]
    lines = [
        "# Work I Policy-Control Runner Qualification and Protocol Freeze",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Qualification report SHA-256: `{report['report_sha256']}`",
        "",
        f"Formal-entry receipt SHA-256: `{receipt['receipt_sha256']}`",
        "",
        "## Outcome firewall",
        "",
        "No formal chemical world was instantiated and no formal outcome was read. "
        "Seeds 0--4 occur only as frozen V05 schedule coordinates attached to explicitly "
        "synthetic world, physics, and noise identities.",
        "",
        "## Qualification evidence",
        "",
        f"- Synthetic V05 matrix: {synthetic['counts']['campaigns']} campaigns, "
        f"{synthetic['counts']['closed_lifecycles']} closed lifecycles, zero provider calls.",
        f"- Fixed nonformal live smoke: seed {live['world_seed']}, "
        f"{live['counts']['primary_campaigns']} original campaigns plus "
        f"{live['counts']['retest_campaigns']} retests, zero provider calls.",
        f"- V06 audit receipt: `{synthetic['audit_sha256']}`.",
        "",
        "## Frozen entry gates",
        "",
        "| Gate | Pass |",
        "| --- | --- |",
    ]
    for gate, passed in report["qualification_gates"].items():
        lines.append(f"| `{gate}` | {passed} |")
    lines.extend(
        [
            "",
            "## Counting boundary",
            "",
            "All qualification executions are excluded from the W1-V08 formal "
            "30-campaign/180-lifecycle estimand. A failed frozen gate must be reported "
            "without changing seeds, threshold, estimands, or acceptance rules.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DELIVERY_MANIFEST_SCHEMA_ID",
    "DELIVERY_MANIFEST_SCHEMA_VERSION",
    "LIVE_MANIFEST_SCHEMA_ID",
    "LIVE_MANIFEST_SCHEMA_VERSION",
    "QUALIFICATION_PROTOCOL_SCHEMA_ID",
    "QUALIFICATION_PROTOCOL_SCHEMA_VERSION",
    "QUALIFICATION_SCHEMA_ID",
    "QUALIFICATION_SCHEMA_VERSION",
    "PolicyQualificationError",
    "assert_qualification_outputs_absent",
    "build_qualification",
    "build_qualification_delivery_manifest",
    "build_qualification_markdown",
    "derive_live_smoke_protocol",
    "load_qualification_protocol",
    "qualification_source_manifest",
    "run_live_smoke",
    "synthetic_qualification_execution",
    "validate_live_smoke_protocol",
    "validate_qualification_protocol",
]
