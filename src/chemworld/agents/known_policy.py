"""Frozen deterministic controls for experimental-agency construct validity.

These agents are measurement instruments, not optimization baselines.  They
execute the three policies frozen by W1-V02 and bind the conditional policy to
the disjoint-world threshold frozen by W1-V03.  Every action is selected at the
primitive-operation boundary; no action repair or provider call is available.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.interaction import InteractionCapabilities
from chemworld.eval.known_policy_contract import (
    INFORMATION_ARMS,
    KNOWN_POLICY_SCHEMA_ID,
    KNOWN_POLICY_SCHEMA_VERSION,
    LIFECYCLES_PER_CELL,
    POLICY_IDS,
    PROBE_SCHEDULE,
    known_policy_contract_sha256,
    validate_known_policy_contract,
)
from chemworld.eval.known_policy_threshold import (
    qualification_report_sha256,
    threshold_binding_sha256,
    validate_qualification_report,
    validate_threshold_binding,
)
from chemworld.eval.known_policy_threshold import (
    source_manifest as qualification_source_manifest,
)
from chemworld.eval.provenance import canonical_json_sha256, file_sha256

KNOWN_POLICY_AGENT_SCHEMA_ID = "chemworld.known_policy_agent"
KNOWN_POLICY_AGENT_SCHEMA_VERSION = "0.1.0"
KNOWN_POLICY_IMPLEMENTATION_VERSION = "chemworld-known-policy-agent-0.1"
FROZEN_KNOWN_POLICY_CONTRACT_SHA256 = (
    "79681abfa92af758af8326db1727b865376ad0da192ea13552b68fd94a66dd45"
)
FROZEN_QUALIFICATION_REPORT_SHA256 = (
    "9a928c28862099049c560b7135067ea86dc6535a7077926b66f39221abbe924e"
)
FROZEN_THRESHOLD_BINDING_SHA256 = (
    "8a55b713ca900a644a301ecd8e83a0a686f9a28b3f60a18a85a4e57b66288c6a"
)
DEFAULT_CONTRACT_PATH = Path("configs/benchmark/work_i_known_policy_contract_v0.1.json")
DEFAULT_THRESHOLD_BINDING_PATH = Path(
    "configs/benchmark/work_i_known_policy_threshold_v0.1.json"
)
DEFAULT_QUALIFICATION_REPORT_PATH = Path(
    "workstreams/arxiv_v1/reports/"
    "work-i-known-policy-threshold-qualification-v0.1.json"
)
_COMPATIBLE_POST_FREEZE_SOURCE_HASHES = {
    # AQ-03 added type annotations only; the executable statements and frozen
    # V03 scientific payload are unchanged.  Pin the reviewed replacement hash
    # so any later runtime edit still fails closed.
    "src/chemworld/runtime/electrochemical_services.py": (
        "e6c6f9a9ad6cc39ef7838d16ec50adaf107079f986d86f0fb599bb7e559ab46b"
    ),
}


class KnownPolicyContractError(ValueError):
    """A frozen policy artifact is missing, stale, or internally inconsistent."""


class KnownPolicyExecutionError(RuntimeError):
    """A deterministic policy can no longer follow its frozen execution path."""


@dataclass(frozen=True)
class KnownPolicyArtifacts:
    """Validated immutable identities consumed by the policy implementation."""

    contract_sha256: str
    threshold_binding_sha256: str
    qualification_report_sha256: str
    threshold: float
    diagnostic_signal: str
    comparator: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_policy_contract_sha256": self.contract_sha256,
            "threshold_binding_sha256": self.threshold_binding_sha256,
            "qualification_report_sha256": self.qualification_report_sha256,
            "threshold": self.threshold,
            "diagnostic_signal": self.diagnostic_signal,
            "comparator": self.comparator,
        }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnownPolicyContractError(f"cannot read frozen artifact {path}") from exc
    if not isinstance(payload, dict):
        raise KnownPolicyContractError(f"frozen artifact must be a JSON object: {path}")
    return payload


def _qualification_sources_match(
    frozen: object, current: Mapping[str, str]
) -> bool:
    if not isinstance(frozen, Mapping) or set(frozen) != set(current):
        return False
    for path, current_hash in current.items():
        if frozen.get(path) == current_hash:
            continue
        if _COMPATIBLE_POST_FREEZE_SOURCE_HASHES.get(path) == current_hash:
            continue
        return False
    return True


def load_known_policy_artifacts(root: Path | None = None) -> KnownPolicyArtifacts:
    """Load and validate the complete V02/V03 artifact chain."""

    resolved_root = _repository_root() if root is None else Path(root).resolve()
    contract = _read_json_object(resolved_root / DEFAULT_CONTRACT_PATH)
    report = _read_json_object(resolved_root / DEFAULT_QUALIFICATION_REPORT_PATH)
    binding = _read_json_object(resolved_root / DEFAULT_THRESHOLD_BINDING_PATH)

    errors = validate_known_policy_contract(contract)
    computed_contract_hash = known_policy_contract_sha256(contract)
    if contract.get("schema_id") != KNOWN_POLICY_SCHEMA_ID:
        errors.append("frozen contract schema ID is not supported")
    if contract.get("schema_version") != KNOWN_POLICY_SCHEMA_VERSION:
        errors.append("frozen contract schema version is not supported")
    if contract.get("contract_sha256") != computed_contract_hash:
        errors.append("frozen contract content hash mismatch")
    if computed_contract_hash != FROZEN_KNOWN_POLICY_CONTRACT_SHA256:
        errors.append("known-policy contract is not the V02 release")
    errors.extend(validate_qualification_report(report))
    errors.extend(validate_threshold_binding(binding, report))
    computed_report_hash = qualification_report_sha256(report)
    computed_binding_hash = threshold_binding_sha256(binding)
    if computed_report_hash != FROZEN_QUALIFICATION_REPORT_SHA256:
        errors.append("qualification report is not the V03 release")
    if computed_binding_hash != FROZEN_THRESHOLD_BINDING_SHA256:
        errors.append("threshold binding is not the V03 release")
    if binding.get("known_policy_contract_sha256") != computed_contract_hash:
        errors.append("threshold binding targets a different known-policy contract")
    if binding.get("qualification_report_sha256") != computed_report_hash:
        errors.append("threshold binding targets a different qualification report")
    if binding.get("binding_sha256") != computed_binding_hash:
        errors.append("threshold binding content hash mismatch")
    try:
        current_source_manifest = qualification_source_manifest(resolved_root)
    except OSError as exc:
        raise KnownPolicyContractError(
            "cannot verify the V03 qualification source manifest"
        ) from exc
    if not _qualification_sources_match(
        report.get("source_manifest"), current_source_manifest
    ):
        errors.append("qualification source files differ from the V03 release")
    if tuple(binding.get("formal_world_seeds_excluded", ())) != (0, 1, 2, 3, 4):
        errors.append("formal-world exclusion set is not the frozen five-world set")
    if tuple(binding.get("information_arms", ())) != INFORMATION_ARMS:
        errors.append("threshold binding information arms are stale")
    if binding.get("diagnostic_signal") != "observation.conversion":
        errors.append("threshold binding diagnostic signal is unsupported")
    if binding.get("comparator") != ">=":
        errors.append("threshold binding comparator is unsupported")
    if binding.get("formal_retuning_forbidden") is not True:
        errors.append("threshold binding permits formal-world retuning")
    if binding.get("provider_call_count") != 0:
        errors.append("known-policy threshold binding used provider calls")
    raw_threshold = binding.get("threshold")
    if (
        isinstance(raw_threshold, bool)
        or not isinstance(raw_threshold, int | float)
        or not math.isfinite(float(raw_threshold))
    ):
        errors.append("threshold must be a finite scalar")
        threshold = 0.0
    else:
        threshold = float(raw_threshold)
    if errors:
        raise KnownPolicyContractError("; ".join(dict.fromkeys(errors)))

    return KnownPolicyArtifacts(
        contract_sha256=computed_contract_hash,
        threshold_binding_sha256=str(binding["binding_sha256"]),
        qualification_report_sha256=str(binding["qualification_report_sha256"]),
        threshold=threshold,
        diagnostic_signal=str(binding["diagnostic_signal"]),
        comparator=str(binding["comparator"]),
    )


def _probe_prefix(probe: Any) -> list[dict[str, Any]]:
    return [
        {
            "operation": "add_solvent",
            "volume_L": 0.025,
            "solvent": probe.solvent,
        },
        {"operation": "add_reagent", "amount_mol": probe.reagent_amount_mol},
        {
            "operation": "set_potential",
            "potential_V": probe.potential_V,
            "current_mA": probe.current_mA,
            "electrolyte_profile": probe.electrolyte_profile,
        },
        {"operation": "electrolyze", "duration_s": probe.probe_duration_s},
    ]


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class KnownPolicyAgent(BaseAgent):
    """Operation-level state machine for one frozen known policy."""

    name = "known_policy"

    def __init__(
        self,
        policy_id: str,
        *,
        artifact_root: Path | None = None,
    ) -> None:
        if policy_id not in POLICY_IDS:
            allowed = ", ".join(POLICY_IDS)
            raise KnownPolicyContractError(
                f"unknown known-policy ID {policy_id!r}; expected one of {allowed}"
            )
        self.policy_id = policy_id
        self.artifacts = load_known_policy_artifacts(artifact_root)
        self._source_sha256 = file_sha256(Path(__file__))
        self._controller_identity_sha256 = canonical_json_sha256(
            {
                "schema_id": KNOWN_POLICY_AGENT_SCHEMA_ID,
                "schema_version": KNOWN_POLICY_AGENT_SCHEMA_VERSION,
                "implementation_version": KNOWN_POLICY_IMPLEMENTATION_VERSION,
                "policy_id": self.policy_id,
                "controller_source_sha256": self._source_sha256,
                "artifact_bindings": self.artifacts.to_dict(),
            }
        )

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        """Reset from a deliberately narrow task-contract view.

        In particular, this method never reads or stores ``material_information``.
        """

        if task_info.get("task_id") != "electrochemical-conversion":
            raise KnownPolicyExecutionError(
                "known policies require task_id=electrochemical-conversion"
            )
        if task_info.get("episode_mode") != "campaign":
            raise KnownPolicyExecutionError("known policies require campaign mode")
        if task_info.get("electrochemical_workflow_mode") != "autonomous_open_v1":
            raise KnownPolicyExecutionError(
                "known policies require autonomous_open_v1 workflow semantics"
            )
        allowed_operations = frozenset(
            str(item) for item in task_info.get("allowed_operations", ())
        )
        required_operations = {
            "add_solvent",
            "discard_batch",
        }
        if self.policy_id != "start_then_discard":
            required_operations.update(
                {"add_reagent", "set_potential", "electrolyze", "terminate", "measure"}
            )
        if not required_operations.issubset(allowed_operations):
            missing = sorted(required_operations - allowed_operations)
            raise KnownPolicyExecutionError(
                f"task interface is missing required operations: {missing}"
            )
        allowed_instruments = frozenset(
            str(item) for item in task_info.get("allowed_instruments", ())
        )
        required_instruments = (
            set()
            if self.policy_id == "start_then_discard"
            else {"final_assay"}
        )
        if self.policy_id == "measure_then_threshold":
            required_instruments.add("uvvis")
        if not required_instruments.issubset(allowed_instruments):
            missing = sorted(required_instruments - allowed_instruments)
            raise KnownPolicyExecutionError(
                f"task interface is missing required instruments: {missing}"
            )

        # Do not call BaseAgent.reset: retaining the complete task_info object
        # would retain the material dossier despite the frozen no-read contract.
        self.seed = int(seed)
        self._lifecycle_index = 0
        self._pending_action: dict[str, Any] | None = None
        self._action_queue: list[dict[str, Any]] = []
        self._last_decision_audit: dict[str, Any] | None = None
        self._active_branch: str | None = None
        self._active_branch_signal: float | None = None
        self._active_branch_signal_status: str | None = None
        self._decision_ordinal = 0
        self._lifecycle_action_ordinal = 0
        self._controller_trace: list[dict[str, Any]] = []
        self._failed = False
        self._prepare_lifecycle()

    def _prepare_lifecycle(self) -> None:
        if self._lifecycle_index >= LIFECYCLES_PER_CELL:
            self._action_queue = []
            return
        probe = PROBE_SCHEDULE[self._lifecycle_index]
        self._active_branch = None
        self._active_branch_signal = None
        self._active_branch_signal_status = None
        self._lifecycle_action_ordinal = 0
        if self.policy_id == "assay_all":
            self._action_queue = [
                *_probe_prefix(probe),
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        elif self.policy_id == "start_then_discard":
            self._action_queue = [
                {
                    "operation": "add_solvent",
                    "volume_L": 0.025,
                    "solvent": probe.solvent,
                },
                {
                    "operation": "discard_batch",
                    "reason": "known_policy_immediate_discard",
                },
            ]
        else:
            self._action_queue = [
                *_probe_prefix(probe),
                {"operation": "measure", "instrument": "uvvis"},
            ]

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if self._failed:
            raise KnownPolicyExecutionError("policy is faulted after an invalid outcome")
        if self._pending_action is not None:
            raise KnownPolicyExecutionError(
                "act called before the preceding operation outcome was supplied"
            )
        if self._lifecycle_index >= LIFECYCLES_PER_CELL:
            raise KnownPolicyExecutionError("all six frozen lifecycles are complete")
        if not self._action_queue:
            raise KnownPolicyExecutionError(
                "threshold branch is unresolved despite a completed diagnostic"
            )
        action = deepcopy(self._action_queue.pop(0))
        self._pending_action = deepcopy(action)
        self._last_decision_audit = self._build_decision_audit(action)
        self._record_action_decision(action)
        return action

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        del reward
        if self._pending_action is None or action != self._pending_action:
            self._failed = True
            raise KnownPolicyExecutionError(
                "operation outcome does not match the pending frozen action"
            )
        if info.get("transaction_status") != "committed":
            self._failed = True
            raise KnownPolicyExecutionError(
                "known policy fails closed after a non-committed transaction"
            )

        completed_action = self._pending_action
        self._pending_action = None
        if (
            self.policy_id == "measure_then_threshold"
            and completed_action == {"operation": "measure", "instrument": "uvvis"}
        ):
            self._resolve_threshold_branch(observation)
        terminal_kind = self._terminal_kind(completed_action)
        if terminal_kind is not None:
            if not info.get("experiment_ended"):
                self._failed = True
                raise KnownPolicyExecutionError(
                    "terminal policy action did not close the lifecycle"
                )
            if terminal_kind == "discard" and not info.get("batch_discarded"):
                self._failed = True
                raise KnownPolicyExecutionError(
                    "discard action did not produce a discarded lifecycle"
                )
            if terminal_kind == "assay" and info.get("batch_discarded"):
                self._failed = True
                raise KnownPolicyExecutionError(
                    "final assay unexpectedly produced a discarded lifecycle"
                )
            if self._action_queue:
                self._failed = True
                raise KnownPolicyExecutionError(
                    "lifecycle closed before the frozen action queue was exhausted"
                )
            self._lifecycle_index += 1
            self._prepare_lifecycle()

    def _resolve_threshold_branch(
        self, observation: Mapping[str, float | None]
    ) -> None:
        if self._action_queue:
            self._failed = True
            raise KnownPolicyExecutionError(
                "diagnostic result arrived before the fixed prefix was exhausted"
            )
        signal = _finite_scalar(observation.get("conversion"))
        probe = PROBE_SCHEDULE[self._lifecycle_index]
        if signal is None:
            branch = "discard_diagnostic_unavailable"
            self._action_queue = [
                {
                    "operation": "discard_batch",
                    "reason": "known_policy_diagnostic_unavailable",
                }
            ]
        elif signal >= self.artifacts.threshold:
            branch = "continue_and_assay"
            self._action_queue = [
                {
                    "operation": "electrolyze",
                    "duration_s": probe.post_measure_duration_s,
                },
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        else:
            branch = "discard_below_threshold"
            self._action_queue = [
                {
                    "operation": "discard_batch",
                    "reason": "known_policy_below_threshold",
                }
            ]
        self._active_branch = branch
        self._active_branch_signal = signal
        self._active_branch_signal_status = (
            "finite" if signal is not None else "missing_or_nonfinite"
        )
        self._append_trace_event(
            {
                **self._trace_identity(),
                "event_type": "known_policy_threshold_decision",
                "decision_ordinal": self._decision_ordinal,
                "lifecycle_index": self._lifecycle_index,
                "probe_id": probe.probe_id,
                "diagnostic_signal": self.artifacts.diagnostic_signal,
                "diagnostic_value": signal,
                "signal_status": self._active_branch_signal_status,
                "comparator": self.artifacts.comparator,
                "threshold": self.artifacts.threshold,
                "branch": branch,
                "selected_action_plan": deepcopy(self._action_queue),
                "selected_action_plan_sha256": canonical_json_sha256(
                    self._action_queue
                ),
                "observation_fields_read": ["conversion"],
                "material_information_read": False,
                "provider_call_count": 0,
            }
        )

    def _trace_identity(self) -> dict[str, Any]:
        return {
            "schema_id": "chemworld.known_policy_controller_trace",
            "schema_version": "0.1.0",
            "controller_schema_id": KNOWN_POLICY_AGENT_SCHEMA_ID,
            "controller_schema_version": KNOWN_POLICY_AGENT_SCHEMA_VERSION,
            "controller_identity_sha256": self._controller_identity_sha256,
            "controller_source_sha256": self._source_sha256,
            "policy_id": self.policy_id,
            "known_policy_contract_sha256": self.artifacts.contract_sha256,
            "threshold_binding_sha256": self.artifacts.threshold_binding_sha256,
        }

    def _append_trace_event(self, event: dict[str, Any]) -> None:
        payload = deepcopy(event)
        payload["trace_event_sha256"] = canonical_json_sha256(payload)
        self._controller_trace.append(payload)

    def _record_action_decision(self, action: dict[str, Any]) -> None:
        probe = PROBE_SCHEDULE[self._lifecycle_index]
        self._decision_ordinal += 1
        self._lifecycle_action_ordinal += 1
        expected_action = deepcopy(action)
        issued_action = deepcopy(action)
        reads_threshold_signal = (
            self.policy_id == "measure_then_threshold"
            and self._active_branch is not None
        )
        self._append_trace_event(
            {
                **self._trace_identity(),
                "event_type": "known_policy_action_decision",
                "decision_ordinal": self._decision_ordinal,
                "lifecycle_index": self._lifecycle_index,
                "lifecycle_action_ordinal": self._lifecycle_action_ordinal,
                "probe_id": probe.probe_id,
                "branch": self._active_branch,
                "expected_action": expected_action,
                "expected_action_sha256": canonical_json_sha256(expected_action),
                "issued_action": issued_action,
                "issued_action_sha256": canonical_json_sha256(issued_action),
                "action_sha256": canonical_json_sha256(issued_action),
                "actions_match": expected_action == issued_action,
                "adaptation_source": (
                    "measurement" if reads_threshold_signal else "none"
                ),
                "observation_fields_read": (
                    ["conversion"] if reads_threshold_signal else []
                ),
                "observed_signal_access": reads_threshold_signal,
                "diagnostic_value": (
                    self._active_branch_signal if reads_threshold_signal else None
                ),
                "diagnostic_signal": (
                    self._active_branch_signal if reads_threshold_signal else None
                ),
                "signal_status": (
                    self._active_branch_signal_status
                    if reads_threshold_signal
                    else "not_read"
                ),
                "material_information_read": False,
                "material_information_accessed": False,
                "provider_call_count": 0,
            }
        )

    @staticmethod
    def _terminal_kind(action: Mapping[str, Any]) -> str | None:
        if action.get("operation") == "discard_batch":
            return "discard"
        if action == {"operation": "measure", "instrument": "final_assay"}:
            return "assay"
        return None

    def _build_decision_audit(self, action: dict[str, Any]) -> dict[str, Any]:
        operation = str(action["operation"])
        measurement_branch = (
            self.policy_id == "measure_then_threshold"
            and self._active_branch is not None
        )
        if operation == "measure" and action.get("instrument") == "uvvis":
            expected_effect = "Acquire the frozen public conversion diagnostic."
            diagnostic_target = "observation.conversion"
            information_gain = 1.0
            update_rule = {
                "if_supported": "continue only when finite conversion meets the frozen threshold",
                "if_not_supported": "discard on a below-threshold or unavailable diagnostic",
            }
        elif measurement_branch:
            expected_effect = f"Execute the frozen {self._active_branch} branch."
            diagnostic_target = "previous public conversion diagnostic"
            information_gain = 0.0
            update_rule = {
                "if_supported": "preserve the already selected terminal branch",
                "if_not_supported": "fail closed; the branch is immutable after selection",
            }
        else:
            expected_effect = f"Execute frozen {self.policy_id} operation {operation}."
            diagnostic_target = "frozen probe schedule; no observation-dependent choice"
            information_gain = 0.0
            update_rule = {
                "if_supported": "advance to the next frozen operation after commit",
                "if_not_supported": "fail closed without action repair",
            }
        return {
            "action": deepcopy(action),
            "expected_effect": expected_effect,
            "diagnostic_target": diagnostic_target,
            "expected_information_gain": information_gain,
            "belief_update_rule": update_rule,
            "uncertainty": 0.0,
            "adaptation_source": "measurement" if measurement_branch else "none",
            "status": "provided",
        }

    def decision_audit(self) -> dict[str, Any] | None:
        return deepcopy(self._last_decision_audit)

    def agent_trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._controller_trace)

    def interaction_capabilities(self) -> InteractionCapabilities:
        conditional = self.policy_id == "measure_then_threshold"
        return InteractionCapabilities(
            decision_scope="operation",
            consumes_intermediate_observations=conditional,
            consumes_spectra=False,
            adapts_within_experiment=conditional,
            adapts_across_experiments=False,
            emits_structured_decision_audit=True,
        )

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "schema_id": KNOWN_POLICY_AGENT_SCHEMA_ID,
                "schema_version": KNOWN_POLICY_AGENT_SCHEMA_VERSION,
                "implementation_version": KNOWN_POLICY_IMPLEMENTATION_VERSION,
                "controller_sha256": self._controller_identity_sha256,
                "controller_source_sha256": self._source_sha256,
                "role_id": "experimental_agency_construct_validity_control",
                "policy_id": self.policy_id,
                "deterministic": True,
                "provider_call_count": 0,
                "reads_material_information": False,
                "planned_lifecycle_count": LIFECYCLES_PER_CELL,
                "probe_ids": [probe.probe_id for probe in PROBE_SCHEDULE],
                "artifact_bindings": self.artifacts.to_dict(),
                "input_access_contract": {
                    "reads_material_information": False,
                    "reads_history_in_act": False,
                    "observation_fields_read": (
                        ["conversion"]
                        if self.policy_id == "measure_then_threshold"
                        else []
                    ),
                    "reads_spectra": False,
                    "formal_world_retuning": False,
                },
                "failure_policy": "fail_closed_without_action_repair",
            }
        )
        return payload


def make_known_policy_agent(
    policy_id: str,
    *,
    artifact_root: Path | None = None,
) -> KnownPolicyAgent:
    """Construct one contract-validated known-policy controller."""

    return KnownPolicyAgent(policy_id, artifact_root=artifact_root)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DEFAULT_QUALIFICATION_REPORT_PATH",
    "DEFAULT_THRESHOLD_BINDING_PATH",
    "FROZEN_KNOWN_POLICY_CONTRACT_SHA256",
    "FROZEN_QUALIFICATION_REPORT_SHA256",
    "FROZEN_THRESHOLD_BINDING_SHA256",
    "KNOWN_POLICY_AGENT_SCHEMA_ID",
    "KNOWN_POLICY_AGENT_SCHEMA_VERSION",
    "KNOWN_POLICY_IMPLEMENTATION_VERSION",
    "KnownPolicyAgent",
    "KnownPolicyArtifacts",
    "KnownPolicyContractError",
    "KnownPolicyExecutionError",
    "load_known_policy_artifacts",
    "make_known_policy_agent",
]
