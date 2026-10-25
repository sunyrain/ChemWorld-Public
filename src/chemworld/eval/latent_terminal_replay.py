"""Fail-closed terminal replacement for the Work I discard audit.

The primitive in this module evaluates one copied pre-discard state directly
through the frozen final-assay observation kernel.  It never calls ``env.step``
for the replacement branch and therefore cannot advance chemistry or charge the
original campaign resource ledger.  Formal execution remains the responsibility
of W1-L05; W1-L03 only supplies and synthetically qualifies the mechanism.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.campaign_resources import (
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
)
from chemworld.data.logging import to_builtin
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.observation_noise import (
    ObservationNoiseCoordinate,
    keyed_noise_provenance,
    keyed_observation_rng,
)
from chemworld.eval.latent_terminal_contract import (
    CONTRACT_ID,
    latent_terminal_contract_sha256,
    validate_latent_terminal_contract,
)
from chemworld.eval.provenance import canonical_json_sha256

REPLAY_SCHEMA_ID = "chemworld.latent_terminal_replay_receipt"
REPLAY_SCHEMA_VERSION = "0.1.0"
REPLAY_IMPLEMENTATION_ID = "work-i-latent-terminal-replay-v0.1"
FROZEN_CONTRACT_SHA256 = (
    "55a0d6a7cb983ce4099dbea24586ea63ccc9433e106d36011d349472809efe30"
)
CONTRACT_PATH = Path("configs/benchmark/work_i_latent_terminal_contract_v0.1.json")
FINAL_ASSAY_ACTION = {"operation": "measure", "instrument": "final_assay"}
SHADOW_NAMESPACE_SUFFIX = "latent-terminal-final-assay-v0.1"
ALLOWED_WORKFLOW_BYPASS_REASONS = frozenset({"measure_final_requires_terminated"})

PREFIX_IDENTITY_FIELDS = (
    "discard_id",
    "cell_id",
    "world_seed",
    "information_arm",
    "lifecycle_index",
    "terminal_step",
    "operation_ordinal",
    "experiment_index",
    "terminal_action_sha256",
    "public_prefix_sha256",
    "hidden_state_sha256",
    "campaign_resource_snapshot_sha256",
    "campaign_resource_state_sha256",
    "world_id",
    "mechanism_hash",
    "material_instance_sha256",
    "observation_seed",
    "observation_noise_mode",
    "observation_noise_namespace",
    "campaign_resource_card_sha256",
    "task_contract_hash",
    "scoring_contract_hash",
    "observation_contract_hash",
)


class LatentTerminalReplayError(RuntimeError):
    """Raised before scoring when any frozen identity or branch gate fails."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentTerminalReplayError(f"cannot read frozen contract: {path}") from exc
    if not isinstance(payload, dict):
        raise LatentTerminalReplayError("frozen contract must be a JSON object")
    return payload


def load_frozen_terminal_contract(root: Path) -> dict[str, Any]:
    """Load the exact L01 contract accepted by this replay implementation."""

    contract = _read_json_object(root.resolve() / CONTRACT_PATH)
    errors = validate_latent_terminal_contract(contract, root=root.resolve())
    computed = latent_terminal_contract_sha256(contract)
    if errors:
        raise LatentTerminalReplayError("invalid L01 contract: " + "; ".join(errors))
    if contract.get("contract_id") != CONTRACT_ID:
        raise LatentTerminalReplayError("unexpected L01 contract ID")
    if contract.get("contract_sha256") != computed or computed != FROZEN_CONTRACT_SHA256:
        raise LatentTerminalReplayError("L01 contract identity is not frozen v0.1")
    return contract


def shadow_noise_namespace(
    original_namespace: str,
    cell_id: str,
    lifecycle_index: int,
) -> str:
    """Return the registered, unit-specific terminal-noise namespace."""

    if not original_namespace.strip() or not cell_id.strip() or lifecycle_index < 0:
        raise LatentTerminalReplayError("invalid shadow-noise namespace coordinate")
    return (
        f"{original_namespace}::{SHADOW_NAMESPACE_SUFFIX}::"
        f"{cell_id}::lifecycle-{lifecycle_index:02d}"
    )


def _resource_snapshot(base: ChemWorldEnv) -> dict[str, Any]:
    snapshot = base.campaign_resource_snapshot()
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("state"), Mapping):
        raise LatentTerminalReplayError("campaign resource snapshot is unavailable")
    return deepcopy(dict(snapshot))


def _environment_mutation_surface(base: ChemWorldEnv) -> dict[str, Any]:
    """Capture every mutable surface the evaluator-only branch could disturb."""

    last_operation = base._last_operation_record
    return to_builtin(
        {
            "hidden_state": base._state.to_dict(include_hidden=True),
            "campaign_resources": _resource_snapshot(base),
            "step_count": base._step_count,
            "operation_id": base._operation_id,
            "experiment_index": base._experiment_index,
            "done": base._done,
            "campaign_terminal": base._campaign_terminal,
            "campaign_terminal_reason": base._campaign_terminal_reason,
            "right_censored_open_batch": base._right_censored_open_batch,
            "campaign_resource_current_vessel_started": (
                base._campaign_resource_current_vessel_started
            ),
            "current_batch_resource_baseline": base._current_batch_resource_baseline,
            "observation_occurrences": [
                {"coordinate": list(key), "count": value}
                for key, value in sorted(base._observation_occurrences.items())
            ],
            "sequential_rng_state": deepcopy(base._rng.bit_generator.state),
            "last_observation": deepcopy(base._last_observation),
            "last_operation_record": (
                None if last_operation is None else last_operation.to_dict()
            ),
            "last_info": deepcopy(base._last_info),
            "experiment_summaries": deepcopy(base._experiment_summaries),
            "observation_noise_provenance": base.observation_noise_provenance(),
            "observation_kernel_last_provider_execution": deepcopy(
                base.observation_kernel.last_provider_execution
            ),
        }
    )


def capture_prefix_identity(
    base: ChemWorldEnv,
    *,
    cell_id: str,
    lifecycle_index: int,
    terminal_step: int,
    original_discard_action: Mapping[str, Any],
    public_prefix_records: Sequence[Mapping[str, Any]],
    authoritative_resource_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture the complete replay identity without exposing hidden payloads."""

    if original_discard_action.get("operation") != "discard_batch":
        raise LatentTerminalReplayError("original terminal action is not discard_batch")
    if lifecycle_index < 0 or terminal_step <= 0:
        raise LatentTerminalReplayError("invalid lifecycle or terminal ordinal")
    if len(public_prefix_records) + 1 != terminal_step:
        raise LatentTerminalReplayError("public prefix length does not match terminal step")
    if base._step_count + 1 != terminal_step or base._operation_id + 1 != terminal_step:
        raise LatentTerminalReplayError("environment ordinal does not match terminal step")
    if base._experiment_index != lifecycle_index:
        raise LatentTerminalReplayError("environment lifecycle ordinal mismatch")

    resources = _resource_snapshot(base)
    historical_resources = deepcopy(dict(authoritative_resource_snapshot))
    historical_state = historical_resources.get("state")
    if not isinstance(historical_state, Mapping):
        raise LatentTerminalReplayError("authoritative resource snapshot lacks state")
    try:
        CampaignResourceLedger.from_snapshot(historical_resources)
    except (CampaignResourceIntegrityError, TypeError, ValueError) as exc:
        raise LatentTerminalReplayError(
            "authoritative resource snapshot is not canonically replayable"
        ) from exc
    historical_snapshot_sha256 = historical_resources.get("ledger_sha256")
    if not isinstance(historical_snapshot_sha256, str):
        raise LatentTerminalReplayError("authoritative resource snapshot lacks hash")
    runtime_state_sha256 = canonical_json_sha256(resources["state"])
    historical_state_sha256 = canonical_json_sha256(historical_state)
    if runtime_state_sha256 != historical_state_sha256:
        raise LatentTerminalReplayError(
            "runtime resource state differs from authoritative prefix ledger"
        )

    provenance = base.evaluator_provenance()
    task_info = base.task_info()
    material_config = provenance.get("material_information_config")
    information_arm = (
        material_config.get("mode") if isinstance(material_config, Mapping) else None
    )
    identity = {
        "discard_id": (
            f"{cell_id}:lifecycle-{lifecycle_index:02d}:"
            f"terminal-step-{terminal_step:03d}"
        ),
        "cell_id": cell_id,
        "world_seed": provenance.get("world_seed"),
        "information_arm": information_arm,
        "lifecycle_index": lifecycle_index,
        "terminal_step": terminal_step,
        "operation_ordinal": base._operation_id + 1,
        "experiment_index": base._experiment_index,
        "terminal_action_sha256": canonical_json_sha256(
            to_builtin(dict(original_discard_action))
        ),
        "public_prefix_sha256": canonical_json_sha256(
            to_builtin(list(public_prefix_records))
        ),
        "hidden_state_sha256": canonical_json_sha256(
            to_builtin(base._state.to_dict(include_hidden=True))
        ),
        "campaign_resource_snapshot_sha256": historical_snapshot_sha256,
        "campaign_resource_state_sha256": runtime_state_sha256,
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
        "task_contract_hash": task_info.get("task_contract_hash"),
        "scoring_contract_hash": base.scoring_contract.contract_hash,
        "observation_contract_hash": base.observation_contract.contract_hash,
    }
    identity["prefix_identity_sha256"] = canonical_json_sha256(identity)
    return identity


def assert_exact_prefix_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    """Fail closed unless every registered prefix-identity field matches."""

    missing = [field for field in PREFIX_IDENTITY_FIELDS if field not in expected]
    if missing:
        raise LatentTerminalReplayError(
            "expected prefix identity lacks fields: " + ", ".join(missing)
        )
    mismatches = [
        field
        for field in PREFIX_IDENTITY_FIELDS
        if expected.get(field) != actual.get(field)
    ]
    expected_digest = expected.get("prefix_identity_sha256")
    if expected_digest is not None and expected_digest != actual.get(
        "prefix_identity_sha256"
    ):
        mismatches.append("prefix_identity_sha256")
    if mismatches:
        raise LatentTerminalReplayError(
            "exact prefix identity mismatch: " + ", ".join(dict.fromkeys(mismatches))
        )


def _validate_contract_bindings(
    base: ChemWorldEnv,
    contract: Mapping[str, Any],
) -> None:
    rule = contract.get("counterfactual_terminal_rule")
    population = contract.get("population")
    if not isinstance(rule, Mapping) or not isinstance(population, Mapping):
        raise LatentTerminalReplayError("L01 terminal rule is missing")
    run_contracts = population.get("run_contracts")
    if not isinstance(run_contracts, Mapping):
        raise LatentTerminalReplayError("L01 run-contract bindings are missing")
    required = {
        "task_contract_hash": (
            None if base.task_spec is None else base.task_spec.contract_hash
        ),
        "scoring_contract_hash": base.scoring_contract.contract_hash,
        "observation_contract_hash": base.observation_contract.contract_hash,
        "workflow_mode": base.electrochemical_workflow_mode,
        "material_family_id": base.electrochemical_material_family_id,
        "observation_noise_mode": base.observation_noise_mode,
        "campaign_resource_card_sha256": base.evaluator_provenance().get(
            "campaign_resource_card_sha256"
        ),
    }
    mismatches = [
        key for key, value in required.items() if run_contracts.get(key) != value
    ]
    if rule.get("score_contract_hash") != base.scoring_contract.contract_hash:
        mismatches.append("counterfactual_terminal_rule.score_contract_hash")
    if mismatches:
        raise LatentTerminalReplayError(
            "frozen contract/runtime mismatch: " + ", ".join(mismatches)
        )


def evaluate_terminal_replacement(
    base: ChemWorldEnv,
    *,
    expected_identity: Mapping[str, Any],
    original_discard_action: Mapping[str, Any],
    public_prefix_records: Sequence[Mapping[str, Any]],
    authoritative_resource_snapshot: Mapping[str, Any],
    frozen_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace one discard with one evaluator-only final assay.

    The function returns hashes plus the scalar score.  It never returns hidden
    state or resource payloads and never mutates the supplied environment.
    """

    if latent_terminal_contract_sha256(frozen_contract) != FROZEN_CONTRACT_SHA256:
        raise LatentTerminalReplayError("terminal replacement received a stale contract")
    _validate_contract_bindings(base, frozen_contract)
    prefix = capture_prefix_identity(
        base,
        cell_id=str(expected_identity.get("cell_id", "")),
        lifecycle_index=int(expected_identity.get("lifecycle_index", -1)),
        terminal_step=int(expected_identity.get("terminal_step", -1)),
        original_discard_action=original_discard_action,
        public_prefix_records=public_prefix_records,
        authoritative_resource_snapshot=authoritative_resource_snapshot,
    )
    assert_exact_prefix_identity(expected_identity, prefix)

    validation = base.operation_validator.validate(FINAL_ASSAY_ACTION, base._state)
    bypass_reasons = frozenset(validation.invalid_reasons)
    if not validation.dispatchable_to_runtime:
        raise LatentTerminalReplayError("final assay is not runtime-dispatchable")
    if bypass_reasons != ALLOWED_WORKFLOW_BYPASS_REASONS:
        rendered = ", ".join(sorted(bypass_reasons)) or "none"
        raise LatentTerminalReplayError(
            "terminal replacement may bypass only workflow readiness; observed: "
            + rendered
        )

    ledger = base._campaign_resource_ledger
    if ledger is None:
        raise LatentTerminalReplayError("campaign resource ledger is unavailable")
    shadow_event_id = "shadow-" + str(prefix["prefix_identity_sha256"])
    preflight = ledger.preflight(
        shadow_event_id,
        FINAL_ASSAY_ACTION,
        starts_vessel=False,
    )
    if not preflight.allowed:
        raise LatentTerminalReplayError(
            "shadow final assay lacks frozen campaign resources: "
            + ", ".join(preflight.rejection_reasons)
        )

    before_surface = _environment_mutation_surface(base)
    before_sha256 = canonical_json_sha256(before_surface)
    prefix_input_sha256 = canonical_json_sha256(to_builtin(list(public_prefix_records)))
    resource_input_sha256 = canonical_json_sha256(
        to_builtin(dict(authoritative_resource_snapshot))
    )

    lifecycle_index = int(prefix["lifecycle_index"])
    coordinate = ObservationNoiseCoordinate(
        namespace=shadow_noise_namespace(
            str(prefix["observation_noise_namespace"]),
            str(prefix["cell_id"]),
            lifecycle_index,
        ),
        base_observation_seed=int(prefix["observation_seed"]),
        experiment_index=lifecycle_index,
        operation_type="measure",
        instrument="final_assay",
        replicate_index=0,
    )
    isolated_kernel = deepcopy(base.observation_kernel)
    observation = isolated_kernel.observe(
        deepcopy(base._state),
        FINAL_ASSAY_ACTION,
        keyed_observation_rng(coordinate),
    )
    observation_report = base.constitution.check_observation(
        observation,
        debug_truth=False,
    )
    if not observation_report.passed:
        raise LatentTerminalReplayError("shadow observation failed constitution checks")
    score = observation.values.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise LatentTerminalReplayError("shadow leaderboard score is not finite in [0,1]")

    after_surface = _environment_mutation_surface(base)
    after_sha256 = canonical_json_sha256(after_surface)
    if before_surface != after_surface:
        raise LatentTerminalReplayError("evaluator-only branch mutated the original environment")
    if prefix_input_sha256 != canonical_json_sha256(
        to_builtin(list(public_prefix_records))
    ):
        raise LatentTerminalReplayError("evaluator-only branch mutated the public prefix")
    if resource_input_sha256 != canonical_json_sha256(
        to_builtin(dict(authoritative_resource_snapshot))
    ):
        raise LatentTerminalReplayError("evaluator-only branch mutated the resource prefix")

    observation_payload = to_builtin(observation.to_dict())
    result_identity = {
        "discard_id": prefix["discard_id"],
        "prefix_identity_sha256": prefix["prefix_identity_sha256"],
        "replacement_action_sha256": canonical_json_sha256(FINAL_ASSAY_ACTION),
        "shadow_observation_sha256": canonical_json_sha256(observation_payload),
        "leaderboard_score": float(score),
        "noise_key_sha256": coordinate.key_sha256,
        "scoring_contract_hash": prefix["scoring_contract_hash"],
        "observation_contract_hash": prefix["observation_contract_hash"],
    }
    result_identity["terminal_evaluation_identity_sha256"] = canonical_json_sha256(
        result_identity
    )
    return {
        "schema_id": REPLAY_SCHEMA_ID,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "implementation_id": REPLAY_IMPLEMENTATION_ID,
        **result_identity,
        "terminal_action_replacement": {
            "suppressed_original_action_sha256": prefix["terminal_action_sha256"],
            "replacement_action": deepcopy(FINAL_ASSAY_ACTION),
            "additional_process_operations": 0,
            "env_step_calls": 0,
            "workflow_readiness_bypassed": sorted(bypass_reasons),
            "all_other_preconditions_passed": True,
        },
        "noise_provenance": keyed_noise_provenance(coordinate),
        "resource_preflight": {
            "allowed": preflight.allowed,
            "rejection_reasons": list(preflight.rejection_reasons),
            "proposed_delta_sha256": canonical_json_sha256(
                to_builtin(preflight.proposed_delta.to_dict())
            ),
            "charged_to_original_ledger": False,
        },
        "observation_constitution_checks": observation_report.to_list(),
        "original_environment_before_sha256": before_sha256,
        "original_environment_after_sha256": after_sha256,
        "original_environment_mutated": False,
        "original_prefix_mutated": False,
        "original_resource_ledger_mutated": False,
        "count_as_original_agent_experiment": False,
        "count_as_agent_assay_decision": False,
        "agent_provider_calls": 0,
        "local_instrument_runtime_evaluations": 1,
        "hidden_state_payload_emitted": False,
    }


__all__ = [
    "ALLOWED_WORKFLOW_BYPASS_REASONS",
    "CONTRACT_PATH",
    "FINAL_ASSAY_ACTION",
    "FROZEN_CONTRACT_SHA256",
    "PREFIX_IDENTITY_FIELDS",
    "REPLAY_IMPLEMENTATION_ID",
    "REPLAY_SCHEMA_ID",
    "REPLAY_SCHEMA_VERSION",
    "LatentTerminalReplayError",
    "assert_exact_prefix_identity",
    "capture_prefix_identity",
    "evaluate_terminal_replacement",
    "load_frozen_terminal_contract",
    "shadow_noise_namespace",
]
