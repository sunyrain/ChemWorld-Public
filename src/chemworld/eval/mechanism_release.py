"""Embargo, go/no-go, and readiness semantics for mechanism adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256

STRUCTURAL_RECEIPT_VERSION = "chemworld-mechanism-metric-embargo-receipt-0.2"
PUBLIC_DECISION_VERSION = "chemworld-mechanism-public-decision-0.1"


def build_metric_embargo_receipt(
    report: Mapping[str, Any],
    *,
    stage: str,
    expected_trial_count: int,
) -> dict[str, Any]:
    """Expose binding/completeness only, never A2/A3 scientific metrics."""

    if stage not in {"a2", "a3"}:
        raise ValueError("metric embargo receipt stage must be a2 or a3")
    manifests = report.get("trial_manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("scientific report lacks trial manifests")
    flattened = list(_leaf_manifests(manifests))
    observed = sum(int(item.get("completed_count", 0)) for item in flattened)
    manifest_complete = bool(flattened) and all(
        item.get("complete") is True
        and item.get("duplicate_count") == 0
        and not item.get("missing_trial_key_sha256")
        and not item.get("unexpected_trial_key_sha256")
        and not item.get("invalid_receipts")
        for item in flattened
    )
    execution_binding = report.get("execution_contract_binding")
    if not isinstance(execution_binding, Mapping):
        raise ValueError("scientific report lacks its Gate A execution binding")
    execution_binding_sha256 = execution_binding.get("binding_sha256")
    runtime_source_tree_sha256 = execution_binding.get("runtime_source_tree_sha256")
    for label, digest in (
        ("execution contract binding", execution_binding_sha256),
        ("runtime source tree", runtime_source_tree_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"scientific report has an invalid {label} SHA-256")
    recorded_binding_sha256 = report.get("execution_contract_binding_sha256")
    if (
        recorded_binding_sha256 is not None
        and recorded_binding_sha256 != execution_binding_sha256
    ):
        raise ValueError("scientific report Gate A execution binding hashes disagree")
    return {
        "schema_version": STRUCTURAL_RECEIPT_VERSION,
        "stage": stage,
        "metric_embargo": "active",
        "scientific_metrics_disclosed": False,
        "source_report_sha256": canonical_json_sha256(report),
        "source_schema_version": report.get("schema_version"),
        "protocol_sha256": report.get("protocol_sha256"),
        "gate_a_plan_sha256": report.get("gate_a_plan_sha256"),
        "execution_contract_binding_sha256": execution_binding_sha256,
        "runtime_source_tree_sha256": runtime_source_tree_sha256,
        "expected_trial_count": int(expected_trial_count),
        "observed_completed_trial_count": observed,
        "trial_manifest_count": len(flattened),
        "trial_manifests_sha256": canonical_json_sha256(manifests),
        "structurally_complete": (
            manifest_complete and observed == int(expected_trial_count)
        ),
    }


def gate_a_go_no_go(*, a2_pass: bool, a3_pass: bool) -> dict[str, Any]:
    if not a2_pass:
        return {
            "branch": "a2_failed",
            "formal_gates_b_to_d": "stop",
            "gate_e": "autonomy_only",
            "private_environment_confirmation": "negative_result_reportable",
            "private_agent_mechanism_claim": "sealed",
            "interpretation": "controlled identifiability was not established",
        }
    if not a3_pass:
        return {
            "branch": "a2_pass_a3_failed",
            "formal_gates_b_to_d": "exploratory_only",
            "gate_e": "autonomy_only",
            "private_environment_confirmation": "negative_result_reportable",
            "private_agent_mechanism_claim": "sealed",
            "interpretation": (
                "controlled information exists but online attainability was not established"
            ),
        }
    return {
        "branch": "a2_a3_passed",
        "formal_gates_b_to_d": "eligible",
        "gate_e": "eligible",
        "private_environment_confirmation": "eligible",
        "private_agent_mechanism_claim": "eligible_after_participant_freeze",
        "interpretation": "benchmark prerequisites passed",
    }


def derive_readiness(
    *,
    a1_pass: bool,
    a2_pass: bool,
    a3_pass: bool,
    runner_validated: bool,
    statistics_validated: bool,
    replay_validated: bool,
    gates_b_to_e_executed: bool,
    public_results_frozen: bool,
    private_e_reported: bool,
    private_a_reported: bool,
    claims_match_results: bool,
    participant_performance_pass: bool | None,
) -> dict[str, Any]:
    """Keep performance outcomes separate from publication evidence state."""

    benchmark_ready = all(
        (
            a1_pass,
            a2_pass,
            a3_pass,
            runner_validated,
            statistics_validated,
            replay_validated,
        )
    )
    evidence_complete = all(
        (
            benchmark_ready,
            gates_b_to_e_executed,
            public_results_frozen,
            private_e_reported,
            private_a_reported,
        )
    )
    return {
        "benchmark_ready": benchmark_ready,
        "evidence_complete": evidence_complete,
        "publication_ready": evidence_complete and claims_match_results,
        "participant_performance_pass": participant_performance_pass,
        "performance_is_result_not_readiness_requirement": True,
    }


def build_public_gate_a_decision(
    gate_a_report: Mapping[str, Any],
    *,
    a3_report: Mapping[str, Any],
    a2_structural_receipt: Mapping[str, Any],
    a3_structural_receipt: Mapping[str, Any],
    release_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Release one joint A2/A3 decision only after both structural receipts exist."""

    for stage, receipt in (
        ("a2", a2_structural_receipt),
        ("a3", a3_structural_receipt),
    ):
        if (
            receipt.get("stage") != stage
            or receipt.get("metric_embargo") != "active"
            or receipt.get("structurally_complete") is not True
        ):
            raise ValueError(f"{stage.upper()} structural receipt is incomplete")
    if a2_structural_receipt.get(
        "source_report_sha256"
    ) != canonical_json_sha256(gate_a_report):
        raise ValueError("A2 structural receipt does not bind the Gate A report")
    if a3_structural_receipt.get(
        "source_report_sha256"
    ) != canonical_json_sha256(a3_report):
        raise ValueError("A3 structural receipt does not bind the A3 report")
    decision = gate_a_report.get("certificate_decision")
    if not isinstance(decision, Mapping):
        raise ValueError("Gate A report lacks authoritative certificate decision")
    a2_pass = decision.get("a2_controlled_matched_identifiability_pass") is True
    a3_pass = decision.get("a3_online_attainability_pass") is True
    qualification_pass = release_qualification.get("qualified") is True
    readiness = derive_readiness(
        a1_pass=decision.get("a1_physical_intervention_validity_pass") is True,
        a2_pass=a2_pass,
        a3_pass=a3_pass,
        runner_validated=qualification_pass,
        statistics_validated=qualification_pass,
        replay_validated=qualification_pass,
        gates_b_to_e_executed=False,
        public_results_frozen=False,
        private_e_reported=False,
        private_a_reported=False,
        claims_match_results=False,
        participant_performance_pass=None,
    )
    payload = {
        "schema_version": PUBLIC_DECISION_VERSION,
        "metric_embargo": "released_for_joint_a2_a3_decision",
        "gate_a_report_sha256": canonical_json_sha256(gate_a_report),
        "a3_report_sha256": canonical_json_sha256(a3_report),
        "a2_structural_receipt_sha256": canonical_json_sha256(
            a2_structural_receipt
        ),
        "a3_structural_receipt_sha256": canonical_json_sha256(
            a3_structural_receipt
        ),
        "release_qualification_sha256": canonical_json_sha256(
            release_qualification
        ),
        "a1_pass": decision.get("a1_physical_intervention_validity_pass")
        is True,
        "a2_pass": a2_pass,
        "a3_pass": a3_pass,
        "gate_a_pass": decision.get("gate_a_pass") is True,
        "go_no_go": gate_a_go_no_go(a2_pass=a2_pass, a3_pass=a3_pass),
        "readiness": readiness,
        "public_scientific_tables": {
            "a2_controlled_identifiability": _without_trial_payloads(
                {
                    "primary_gate_budget": gate_a_report.get(
                        "primary_gate_budget"
                    ),
                    "active_oracle": gate_a_report.get("active_oracle"),
                    "fixed_trajectory_decoder": gate_a_report.get(
                        "fixed_trajectory_decoder"
                    ),
                    "task_reports": gate_a_report.get("task_reports"),
                }
            ),
            "a3_online_attainability": _without_trial_payloads(
                {
                    key: a3_report.get(key)
                    for key in (
                        "reference_acquisition_certificate",
                        "keyed_noise_pairing_audit",
                        "online_capability_chain_certificate",
                        "online_capability_chain_by_post_change_budget",
                        "online_capability_chain_by_task_and_post_change_budget",
                        "online_capability_chain_by_family_and_post_change_budget",
                        "primary_task_intersection_pass",
                        "primary_family_intersection_pass",
                        "family_macro_average",
                        "pooled_micro_average",
                        "identifiability_by_post_change_budget",
                    )
                }
            ),
        },
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    return payload


def _leaf_manifests(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("schema_version") == "chemworld-confirmatory-trial-manifest-0.1":
        return [value]
    result: list[Mapping[str, Any]] = []
    for item in value.values():
        if isinstance(item, Mapping):
            result.extend(_leaf_manifests(item))
    return result


def _without_trial_payloads(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_trial_payloads(item)
            for key, item in value.items()
            if key
            not in {
                "trials",
                "pre_change_updates",
                "post_change_updates",
                "predictives",
            }
        }
    if isinstance(value, list):
        return [_without_trial_payloads(item) for item in value]
    return value


__all__ = [
    "PUBLIC_DECISION_VERSION",
    "STRUCTURAL_RECEIPT_VERSION",
    "build_metric_embargo_receipt",
    "build_public_gate_a_decision",
    "derive_readiness",
    "gate_a_go_no_go",
]
