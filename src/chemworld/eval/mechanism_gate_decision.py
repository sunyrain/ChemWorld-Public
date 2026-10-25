"""Authoritative certificate composition for mechanism-adaptation gates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.eval.mechanism_relation_graph import (
    build_diagnostic_relation_graph,
)
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    repository_tree_sha256,
)
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.tasks import get_task
from chemworld.world.operations import operation_contracts

ONLINE_ATTAINABILITY_CERTIFICATE_VERSION = (
    "chemworld-mechanism-adaptation-online-attainability-certificate-0.9"
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def gate_a_execution_contract_binding(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every Gate A certificate to the runtime semantics it evaluated."""

    config_paths = {
        "backend_contract": configuration_root() / "foundation/backend_v0.5.json",
        "evaluation_contract": configuration_root() / "benchmark/evaluation_vnext.json",
        "public_boundary_contract": (
            configuration_root() / "foundation/public_boundary_security_vnext.json"
        ),
    }
    binding: dict[str, Any] = {
        "schema_version": "chemworld-gate-a-execution-binding-0.1",
        "runtime_source_tree_sha256": repository_tree_sha256(
            PACKAGE_ROOT,
            relative_roots=(".",),
        ),
        "task_contract_hashes": {
            str(task_id): get_task(str(task_id)).contract_hash
            for task_id in protocol["design"]["tasks"]
        },
        "operation_contract_sha256": canonical_json_sha256(
            {key: value.to_dict() for key, value in sorted(operation_contracts().items())}
        ),
        "bound_config_sha256": {
            key: file_sha256(path) for key, path in sorted(config_paths.items())
        },
        "protocol_sha256": canonical_json_sha256(protocol),
        "gate_a_plan_sha256": canonical_json_sha256(plan),
        "diagnostic_relation_graph_sha256": (
            build_diagnostic_relation_graph(protocol)["graph_sha256"]
        ),
    }
    binding["binding_sha256"] = canonical_json_sha256(binding)
    return binding


def gate_a_certificate_decision(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    physical_intervention_validity_pass: bool,
    controlled_gate_pass: bool,
    controlled_certificate_present: bool = True,
    online_attainability_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose Gate A1/A2/A3 from independently bound certificates.

    A1 establishes that the declared intervention changes the intended physical law.
    The A2 controlled matched oracle establishes that the public experiment contract
    contains diagnostic information. It is not a substitute for demonstrating that
    a frozen A3 reference diagnostic policy can first acquire a sufficient
    old-world reference and then diagnose a hidden change online. A3 certifies
    benchmark attainability, not any participant Agent; those Agents are evaluated
    only in Gates B--E. Missing, stale, or malformed evidence fails closed.
    """

    requirement = plan.get("online_attainability_certificate")
    if (
        not isinstance(requirement, Mapping)
        or requirement.get("required_before_formal_mechanism_claim") is not True
    ):
        raise ValueError("Gate A plan must require an online-attainability certificate")

    expected_protocol_sha = canonical_json_sha256(protocol)
    expected_plan_sha = canonical_json_sha256(plan)
    expected_execution_binding = gate_a_execution_contract_binding(protocol, plan)
    controlled_primary_budget = int(plan["held_out_certificate"]["primary_gate_budget"])
    online_gate_budget = int(requirement["online_policy_gate_budget"])
    expected_certificate_version = str(
        requirement.get(
            "certificate_schema_version",
            ONLINE_ATTAINABILITY_CERTIFICATE_VERSION,
        )
    )
    expected_certificate_scope = str(
        requirement.get(
            "certificate_scope",
            "online_attainability_of_frozen_reference_diagnostic_policy",
        )
    )
    if online_attainability_certificate is None:
        online_summary = {
            "schema_version": expected_certificate_version,
            "status": "pending_execution",
            "gate_pass": False,
            "required": True,
            "certificate_present": False,
            "controlled_matched_primary_budget": controlled_primary_budget,
            "online_policy_gate_budget": online_gate_budget,
            "certification_subject": "frozen_reference_diagnostic_policy",
            "participant_agent_evaluation": False,
        }
    else:
        certificate = dict(online_attainability_certificate)
        errors: list[str] = []
        expected_values = {
            "schema_version": expected_certificate_version,
            "certificate_scope": expected_certificate_scope,
            "protocol_sha256": expected_protocol_sha,
            "gate_a_plan_sha256": expected_plan_sha,
            "controlled_matched_primary_budget": controlled_primary_budget,
            "online_policy_gate_budget": online_gate_budget,
            "certification_subject": "frozen_reference_diagnostic_policy",
            "participant_agent_evaluation": False,
            "hidden_change_time": True,
            "policy_received_change_time_support": False,
            "policy_received_minimum_stable_prefix": False,
            "policy_received_reference_certificate": False,
            "policy_received_phase_or_reset_indicator": False,
            "uses_actual_available_pre_change_history": True,
            "uses_actual_action_measurement_and_budget_contract": True,
            "execution_contract_binding_sha256": expected_execution_binding["binding_sha256"],
        }
        if "evaluation_track_id" in requirement:
            expected_values["evaluation_track_id"] = requirement[
                "evaluation_track_id"
            ]
        if "minimum_stable_prefix_experiments" in requirement:
            expected_values["minimum_stable_prefix_experiments"] = int(
                requirement["minimum_stable_prefix_experiments"]
            )
        if "truth_change_time_support" in requirement:
            expected_values["truth_change_time_support"] = requirement[
                "truth_change_time_support"
            ]
        if "changepoint_semantics" in requirement:
            expected_values["changepoint_semantics"] = requirement[
                "changepoint_semantics"
            ]
        for field, expected in expected_values.items():
            if certificate.get(field) != expected:
                errors.append(f"{field} must equal {expected!r}")
        gate_pass = certificate.get("gate_pass")
        if not isinstance(gate_pass, bool):
            errors.append("gate_pass must be boolean")
        expected_status = "passed" if gate_pass is True else "failed"
        if certificate.get("status") != expected_status:
            errors.append(f"status must equal {expected_status!r}")
        if (
            expected_certificate_scope
            == "online_attainability_of_frozen_reference_diagnostic_policy"
        ):
            reference_certificate = certificate.get(
                "reference_acquisition_certificate"
            )
            if not isinstance(reference_certificate, Mapping):
                errors.append(
                    "reference_acquisition_certificate must be present for calibrated "
                    "online change"
                )
            elif not isinstance(reference_certificate.get("gate_pass"), bool):
                errors.append(
                    "reference_acquisition_certificate.gate_pass must be boolean"
                )
            elif (
                gate_pass is True
                and reference_certificate.get("gate_pass") is not True
            ):
                errors.append(
                    "a passing calibrated certificate requires reference acquisition"
                )
            capability_certificate = certificate.get(
                "online_capability_chain_certificate"
            )
            if not isinstance(capability_certificate, Mapping):
                errors.append(
                    "online_capability_chain_certificate must be present for calibrated "
                    "online change"
                )
            elif not isinstance(capability_certificate.get("gate_pass"), bool):
                errors.append(
                    "online_capability_chain_certificate.gate_pass must be boolean"
                )
            elif gate_pass is True and capability_certificate.get("gate_pass") is not True:
                errors.append(
                    "a passing calibrated certificate requires the end-to-end "
                    "capability chain"
                )
        if errors:
            raise ValueError(
                "invalid online-attainability certificate: "
                + "; ".join(errors)
            )
        online_summary = {
            **certificate,
            "required": True,
            "certificate_present": True,
            "certificate_sha256": canonical_json_sha256(certificate),
        }

    online_gate_pass = online_summary["gate_pass"] is True
    physical_gate_pass = bool(physical_intervention_validity_pass)
    controlled_present = bool(controlled_certificate_present)
    combined_pass = bool(
        physical_gate_pass
        and controlled_present
        and controlled_gate_pass
        and online_gate_pass
    )
    if combined_pass:
        status = "gate_a_passed"
    elif not physical_gate_pass:
        status = "gate_a_failed_physical_intervention_validity"
    elif not controlled_present:
        status = "gate_a_blocked_controlled_matched_certificate_pending"
    elif not controlled_gate_pass:
        status = "gate_a_failed_controlled_matched_certificate"
    elif online_summary["certificate_present"] is not True:
        status = "gate_a_blocked_online_attainability_certificate_pending"
    else:
        status = "gate_a_failed_online_attainability_certificate"
    return {
        "schema_version": "chemworld-mechanism-adaptation-gate-a-decision-0.3",
        "status": status,
        "required_certificates": [
            "a1_physical_intervention_validity",
            "a2_controlled_matched_identifiability",
            "a3_online_attainability_frozen_reference_diagnostic_policy",
        ],
        "a1_physical_intervention_validity_pass": physical_gate_pass,
        "a2_controlled_matched_identifiability_pass": bool(
            controlled_gate_pass
        ),
        "a2_controlled_matched_certificate_present": controlled_present,
        "a3_online_attainability_pass": online_gate_pass,
        "controlled_matched_gate_pass": bool(controlled_gate_pass),
        "online_attainability_gate_pass": online_gate_pass,
        "online_attainability_certificate": online_summary,
        "gate_a_pass": combined_pass,
    }


__all__ = [
    "ONLINE_ATTAINABILITY_CERTIFICATE_VERSION",
    "gate_a_certificate_decision",
    "gate_a_execution_contract_binding",
]
