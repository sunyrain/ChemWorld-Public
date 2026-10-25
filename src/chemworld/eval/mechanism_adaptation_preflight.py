"""Preflight report for the frozen mechanism-adaptation confirmation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chemworld.eval.mechanism_adaptation import (
    evaluate_protocol_gates,
    load_mechanism_adaptation_protocol,
    validate_mechanism_adaptation_protocol,
)
from chemworld.eval.mechanism_adaptation_execution import load_json_object
from chemworld.eval.mechanism_gate_decision import (
    gate_a_certificate_decision,
    gate_a_execution_contract_binding,
)
from chemworld.eval.mechanism_preregistration import (
    validate_mechanism_preregistration,
)
from chemworld.eval.mechanism_relation_graph import (
    validate_diagnostic_relation_graph,
)
from chemworld.physchem.mechanism_library import configuration_root

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    configuration_root()
    / "benchmark/mechanism_adaptation_v0.3.0_rc28.json"
)
DEFAULT_GATE_A_PLAN = (
    configuration_root()
    / "benchmark/mechanism_adaptation_gate_a_v0.3.0_rc28.json"
)
DEFAULT_SEMANTICS_AUDIT = (
    ROOT
    / "workstreams/flagship_tasks/reports/"
    "confirmatory-task-semantics-audit-rc28.json"
)
REQUIRED_IMPLEMENTATION_ARTIFACTS = (
    "configs/benchmark/mechanism_adaptation_participant_preregistration_rc28.json",
    "src/chemworld/agents/mechanism_adaptation_live_llm.py",
    "src/chemworld/eval/mechanism_adaptation.py",
    "src/chemworld/eval/confirmatory_task_semantics_audit.py",
    "src/chemworld/eval/mechanism_gate_decision.py",
    "src/chemworld/eval/mechanism_design_audit.py",
    "src/chemworld/eval/mechanism_relation_graph.py",
    "src/chemworld/eval/mechanism_adaptation_execution.py",
    "src/chemworld/eval/mechanism_release.py",
    "src/chemworld/eval/trial_store.py",
    "scripts/audit_mechanism_adaptation_design.py",
    "scripts/audit_confirmatory_task_semantics.py",
    "scripts/build_mechanism_diagnostic_relation_graph.py",
    "scripts/build_mechanism_adaptation_preregistration.py",
    "scripts/audit_mechanism_adaptation_sample_size.py",
    "scripts/run_mechanism_adaptation.py",
    "scripts/qualify_mechanism_adaptation_release.py",
    "scripts/plan_mechanism_adaptation_matrix.py",
    "tests/test_mechanism_adaptation.py",
    "workstreams/flagship_tasks/reports/mechanism-adaptation-v0.3.0-public-matrix.json",
)


def build_mechanism_adaptation_preflight(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    gate_a_plan_path: Path = DEFAULT_GATE_A_PLAN,
    semantics_audit_path: Path = DEFAULT_SEMANTICS_AUDIT,
) -> dict[str, Any]:
    """Check protocol completeness without treating unrun empirical gates as failures."""

    protocol = load_mechanism_adaptation_protocol(protocol_path)
    validation_errors = validate_mechanism_adaptation_protocol(protocol)
    gate_a_plan = load_json_object(gate_a_plan_path)
    design_audit_relative = str(
        gate_a_plan["design_validity_precondition"]["report"]
    )
    design_audit_path = ROOT / design_audit_relative
    relation_graph_path = ROOT / gate_a_plan["diagnostic_relation_graph"]["report"]
    relation_graph = (
        json.loads(relation_graph_path.read_text(encoding="utf-8"))
        if relation_graph_path.is_file()
        else {}
    )
    relation_graph_errors = (
        validate_diagnostic_relation_graph(protocol, gate_a_plan, relation_graph)
        if relation_graph
        else ["diagnostic relation graph is missing"]
    )
    sample_size_path = ROOT / gate_a_plan["sample_size_audit"]["report"]
    sample_size = (
        json.loads(sample_size_path.read_text(encoding="utf-8"))
        if sample_size_path.is_file()
        else {}
    )
    sample_size_errors = []
    if (
        sample_size.get("pass") is not True
        or sample_size.get("protocol_sha256")
        != _canonical_sha256(protocol)
        or sample_size.get("gate_a_plan_sha256")
        != _canonical_sha256(gate_a_plan)
    ):
        sample_size_errors.append(
            "sample-size audit is missing, failed, or stale"
        )
    preregistration_path = ROOT / gate_a_plan["preregistration"]["manifest"]
    preregistration = (
        json.loads(preregistration_path.read_text(encoding="utf-8"))
        if preregistration_path.is_file()
        else {}
    )
    preregistration_errors = (
        validate_mechanism_preregistration(
            preregistration,
            repository_root=ROOT,
            protocol=protocol,
            plan=gate_a_plan,
            relation_graph=relation_graph,
            sample_size_audit=sample_size,
        )
        if preregistration
        else ["preregistration manifest is missing"]
    )
    semantics_path = semantics_audit_path
    semantics = (
        json.loads(semantics_path.read_text(encoding="utf-8"))
        if semantics_path.is_file()
        else {}
    )
    semantics_errors = []
    if (
        semantics.get("pass") is not True
        or semantics.get("protocol_sha256") != _canonical_sha256(protocol)
        or semantics.get("gate_a_plan_sha256")
        != _canonical_sha256(gate_a_plan)
    ):
        semantics_errors.append(
            "confirmatory-task semantics audit is missing, failed, or stale"
        )
    design_audit = (
        json.loads(design_audit_path.read_text(encoding="utf-8"))
        if design_audit_path.is_file()
        else {}
    )
    design_audit_pass = bool(
        design_audit.get("pass")
        and design_audit.get("protocol_sha256") == _canonical_sha256(protocol)
        and design_audit.get("gate_a_plan_sha256") == _canonical_sha256(gate_a_plan)
    )
    artifacts = []
    required_artifacts = (
        *REQUIRED_IMPLEMENTATION_ARTIFACTS,
        _relative_path(gate_a_plan_path),
        _relative_path(semantics_path),
        design_audit_relative,
        str(gate_a_plan["diagnostic_relation_graph"]["report"]),
        str(gate_a_plan["sample_size_audit"]["report"]),
        str(gate_a_plan["preregistration"]["manifest"]),
    )
    for relative in required_artifacts:
        path = ROOT / relative
        artifacts.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    gate_status = (
        evaluate_protocol_gates(
            protocol,
            {"gate_a": {"design_validity_audit": design_audit}},
            gate_a_plan=gate_a_plan,
        )
        if not validation_errors
        else None
    )
    implementation_complete = (
        not validation_errors
        and not relation_graph_errors
        and not semantics_errors
        and not sample_size_errors
        and not preregistration_errors
        and design_audit_pass
        and all(item["exists"] for item in artifacts)
    )
    return {
        "schema_version": "chemworld-mechanism-adaptation-preflight-0.2",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": _relative_path(protocol_path),
        "protocol_source_sha256": _sha256(protocol_path),
        "protocol_sha256": _canonical_sha256(protocol),
        "status": (
            "protocol_complete_empirical_execution_pending"
            if implementation_complete
            else "preflight_incomplete"
        ),
        "implementation_complete": implementation_complete,
        "method_freeze_decision_blocker_count": (
            len(validation_errors)
            + len(relation_graph_errors)
            + len(semantics_errors)
            + len(sample_size_errors)
            + len(preregistration_errors)
        ),
        "method_freeze_decision_blockers": [
            *validation_errors,
            *relation_graph_errors,
            *semantics_errors,
            *sample_size_errors,
            *preregistration_errors,
        ],
        "diagnostic_relation_graph": relation_graph,
        "confirmatory_task_semantics_audit": semantics,
        "sample_size_audit": sample_size,
        "preregistration_manifest": preregistration,
        "design_validity_audit_pass": design_audit_pass,
        "design_validity_audit": design_audit,
        "external_empirical_run_completed": False,
        "empirical_gate_status": dict.fromkeys(
            ("gate_0", "gate_a", "gate_b", "gate_c", "gate_d", "gate_e"),
            "not_evaluated",
        ),
        "gate_evaluator_without_evidence": gate_status,
        "publication_ready": False,
        "artifacts": artifacts,
        "interpretation": (
            "There are no unresolved method-freeze choices when blocker_count is zero. "
            "A1 design validity may be complete while A2 controlled and A3 online-"
            "attainability certificates remain explicitly pending. Participant Agent/"
            "provider campaigns remain separate Gates B--E."
        ),
    }


def build_mechanism_adaptation_pending_gate_state(
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build explicit non-empirical A3 and Gate A pending-state artifacts."""

    protocol = load_mechanism_adaptation_protocol(protocol_path)
    gate_a_plan_path = (
        configuration_root()
        / "benchmark/mechanism_adaptation_gate_a_v0.3.0_rc28.json"
    )
    plan = load_json_object(gate_a_plan_path)
    design_path = ROOT / plan["design_validity_precondition"]["report"]
    design = (
        json.loads(design_path.read_text(encoding="utf-8"))
        if design_path.is_file()
        else {}
    )
    binding = gate_a_execution_contract_binding(protocol, plan)
    requirement = plan["online_attainability_certificate"]
    online_state = {
        "schema_version": requirement["certificate_schema_version"],
        "certificate_scope": requirement["certificate_scope"],
        "status": "pending_execution",
        "gate_pass": False,
        "formal_benchmark_result": False,
        "certification_subject": "frozen_reference_diagnostic_policy",
        "participant_agent_evaluation": False,
        "certificate_present": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _canonical_sha256(protocol),
        "gate_a_plan_id": plan["plan_id"],
        "gate_a_plan_sha256": _canonical_sha256(plan),
        "execution_contract_binding": binding,
        "execution_contract_binding_sha256": binding["binding_sha256"],
        "controlled_matched_primary_budget": plan["held_out_certificate"][
            "primary_gate_budget"
        ],
        "online_policy_gate_budget": requirement["online_policy_gate_budget"],
        "evaluation_track_id": requirement["evaluation_track_id"],
        "minimum_stable_prefix_experiments": requirement[
            "minimum_stable_prefix_experiments"
        ],
        "hidden_change_time": True,
        "truth_change_time_support": requirement["truth_change_time_support"],
        "changepoint_semantics": requirement["changepoint_semantics"],
        "policy_received_change_time_support": False,
        "policy_received_minimum_stable_prefix": False,
        "policy_received_reference_certificate": False,
        "policy_received_phase_or_reset_indicator": False,
        "uses_actual_available_pre_change_history": True,
        "uses_actual_action_measurement_and_budget_contract": True,
        "reference_acquisition_certificate": {
            "status": "pending_execution",
            "gate_pass": False,
        },
        "online_capability_chain_certificate": {
            "status": "pending_execution",
            "gate_pass": False,
        },
        "interpretation": (
            "This is a deterministic readiness state, not an empirical online "
            "certificate. A new untouched execution is required."
        ),
        "publication_ready": False,
    }
    decision = gate_a_certificate_decision(
        protocol,
        plan,
        physical_intervention_validity_pass=bool(design.get("pass")),
        controlled_gate_pass=False,
        controlled_certificate_present=False,
    )
    online_reference = {
        **decision["online_attainability_certificate"],
        "report": (
            "workstreams/flagship_tasks/reports/"
            "mechanism-adaptation-online-attainability-certificate-v0.9-rc28-pending.json"
        ),
        "certificate_sha256": _canonical_sha256(online_state),
        "certificate_hash_source": "standalone_pending_state_canonical_json",
    }
    decision["online_attainability_certificate"] = online_reference
    gate_state = {
        "schema_version": "chemworld-mechanism-adaptation-gate-a-report-0.3.0",
        "status": decision["status"],
        "formal_benchmark_result": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _canonical_sha256(protocol),
        "gate_a_plan_id": plan["plan_id"],
        "gate_a_plan_sha256": _canonical_sha256(plan),
        "execution_contract_binding": binding,
        "design_validity_audit": {
            "report": plan["design_validity_precondition"]["report"],
            "pass": bool(design.get("pass")),
        },
        "certificate_decision": decision,
        "controlled_matched_certificate": {
            "status": "pending_execution",
            "gate_pass": False,
            "certificate_present": False,
        },
        "online_attainability_certificate": online_reference,
        "gate_a_pass": False,
        "publication_ready": False,
        "interpretation": (
            "A1 design validity is recorded separately. A2 controlled and A3 "
            "online attainability of the frozen reference diagnostic policy have "
            "not been executed for v0.3; participant Agents remain Gates B--E."
        ),
    }
    return online_state, gate_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_PROTOCOL",
    "REQUIRED_IMPLEMENTATION_ARTIFACTS",
    "build_mechanism_adaptation_pending_gate_state",
    "build_mechanism_adaptation_preflight",
]
