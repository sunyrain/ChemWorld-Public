"""Deterministic reporting for the frozen Work I known-policy controls.

The reporter is intentionally read-only.  It delegates evidence reconstruction to
the V06 auditor, compares that reconstruction with the immutable V08 audit receipt,
and never imports or invokes a world, controller, or provider entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.eval.policy_validity_audit import (
    PolicyValidityAuditError,
    audit_policy_validity_manifest,
)
from chemworld.eval.provenance import canonical_json_sha256, file_sha256

REPORT_SCHEMA_ID = "chemworld.known_policy_validity_report"
REPORT_SCHEMA_VERSION = "0.1.0"
DELIVERY_SCHEMA_ID = "chemworld.known_policy_validity_report_delivery"
DELIVERY_SCHEMA_VERSION = "0.1.0"

POLICY_IDS = ("assay_all", "start_then_discard", "measure_then_threshold")
SUMMARY_METRICS = (
    "assay_fraction",
    "discard_fraction",
    "measured_lifecycle_fraction",
    "nonfinal_instrument_uses_per_closed_lifecycle",
    "continued_after_measurement_fraction",
    "attempted_operations_per_closed_lifecycle",
    "committed_operations_per_closed_lifecycle",
)
FROZEN_GATES = (
    "matrix_complete",
    "all_180_lifecycles_closed",
    "all_profiles_rebuilt",
    "all_resource_ledgers_replayed",
    "all_exact_replays_and_retests_match",
    "matched_arm_invariance",
    "zero_provider_calls",
    "threshold_non_degenerate",
    "exact_policy_signatures",
    "conditional_null_rules",
    "six_partial_orderings",
    "resource_expectations",
)


class PolicyValidityReportError(ValueError):
    """Raised when V08 evidence cannot support a bounded V09 report."""


def rendered_json_text(payload: Mapping[str, Any]) -> str:
    """Render one deterministic, human-readable JSON artifact."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyValidityReportError(f"invalid_evidence: {label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PolicyValidityReportError(f"invalid_evidence: {label} must be an array")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyValidityReportError(
            f"invalid_evidence: {label} must be a lowercase SHA-256 digest"
        )
    return value


def _git_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyValidityReportError(
            f"invalid_evidence: {label} must be a lowercase 40-character Git commit"
        )
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyValidityReportError(f"invalid_evidence: cannot read {label}: {path}") from exc
    return dict(_mapping(value, label))


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop(field, None)
    return result


def _claim_value(text: str, field: str) -> str:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(?:\"([^\"]+)\"|([^\s]+))\s*$", text)
    if match is None:
        raise PolicyValidityReportError(f"invalid_evidence: V08 claim lacks {field}")
    return match.group(1) or match.group(2)


def _load_v08_claim(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyValidityReportError(f"invalid_evidence: cannot read V08 claim: {path}") from exc
    task_id = _claim_value(text, "task_id")
    status = _claim_value(text, "status")
    final_commit = _git_commit(_claim_value(text, "final_commit"), "V08 claim final_commit")
    if task_id != "W1-V08" or status != "DONE":
        raise PolicyValidityReportError("invalid_evidence: V08 claim is not DONE")
    return {
        "path": path.as_posix(),
        "file_sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
        "task_id": task_id,
        "status": status,
        "final_commit": final_commit,
    }


def _validate_audit_receipt(receipt: Mapping[str, Any]) -> None:
    declared = _digest(receipt.get("audit_sha256"), "formal audit audit_sha256")
    if declared != canonical_json_sha256(_without(receipt, "audit_sha256")):
        raise PolicyValidityReportError("invalid_evidence: formal audit self-hash mismatch")
    if receipt.get("formal_execution_performed_by_auditor") is not False:
        raise PolicyValidityReportError(
            "invalid_evidence: formal audit executed the formal environment"
        )
    gates = _mapping(receipt.get("gates"), "formal audit gates")
    if set(gates) != set(FROZEN_GATES) or not all(isinstance(gates[name], bool) for name in gates):
        raise PolicyValidityReportError("invalid_evidence: formal audit frozen gate set mismatch")
    passed = all(bool(gates[name]) for name in FROZEN_GATES)
    expected_status = "passed" if passed else "positive_control_unestablished"
    if receipt.get("passed") is not passed or receipt.get("status") != expected_status:
        raise PolicyValidityReportError(
            "invalid_evidence: formal audit status disagrees with gates"
        )


def _profile_metric(profile: Mapping[str, Any], metric_id: str) -> float:
    axes = _mapping(profile.get("construct_axes"), "campaign profile construct_axes")
    for raw_axis in axes.values():
        axis = _mapping(raw_axis, "campaign profile axis")
        if metric_id in axis:
            value = axis[metric_id]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise PolicyValidityReportError(
                    f"invalid_evidence: summary metric {metric_id} is not numeric"
                )
            result = float(value)
            if not math.isfinite(result):
                raise PolicyValidityReportError(
                    f"invalid_evidence: summary metric {metric_id} is not finite"
                )
            return result
    raise PolicyValidityReportError(f"invalid_evidence: campaign profile lacks {metric_id}")


def _equal_weight_summaries(cells: Sequence[Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for policy_id in POLICY_IDS:
        policy_cells = [
            _mapping(cell, "audited campaign")
            for cell in cells
            if _mapping(
                _mapping(cell, "audited campaign").get("identity"), "campaign identity"
            ).get("policy_id")
            == policy_id
        ]
        if len(policy_cells) != 10:
            raise PolicyValidityReportError(
                f"invalid_evidence: policy {policy_id} must contain ten campaign profiles"
            )
        summaries[policy_id] = {
            "campaign_count": 10,
            "weighting": "one_campaign_one_equal_weight",
            "metrics": {
                metric_id: statistics.fmean(
                    _profile_metric(_mapping(cell.get("profile"), "campaign profile"), metric_id)
                    for cell in policy_cells
                )
                for metric_id in SUMMARY_METRICS
            },
        }
    return summaries


def _validate_equal_weight_summaries(
    summaries: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
    for policy_id in POLICY_IDS:
        observed = _mapping(_mapping(summaries[policy_id], policy_id).get("metrics"), policy_id)
        expected = _mapping(frozen.get(policy_id), f"formal audit summary {policy_id}")
        for metric_id in SUMMARY_METRICS:
            if not math.isclose(
                float(observed[metric_id]),
                float(expected[metric_id]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise PolicyValidityReportError(
                    f"invalid_evidence: campaign-equal summary mismatch for {policy_id}:{metric_id}"
                )


def _failed_checks(audit: Mapping[str, Any]) -> list[str]:
    sections = (
        "exact_signature_checks",
        "conditional_null_checks",
        "partial_ordering_checks",
        "resource_expectation_checks",
    )
    failures: list[str] = []
    for section in sections:
        checks = _mapping(audit.get(section), f"formal audit {section}")
        failures.extend(f"{section}:{name}" for name, value in checks.items() if value is not True)
    return failures


def _artifact_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.as_posix(),
        "file_sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def build_policy_validity_report(
    matrix_manifest_path: Path,
    formal_audit_path: Path,
    *,
    v08_claim_path: Path,
    v08_done_commit: str,
    analyzer_source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the bounded V09 report from immutable V08 evidence."""

    v08_done_commit = _git_commit(v08_done_commit, "V08 DONE integration commit")
    receipt = _load_object(formal_audit_path, "formal audit")
    _validate_audit_receipt(receipt)
    try:
        rebuilt = audit_policy_validity_manifest(matrix_manifest_path)
    except (OSError, PolicyValidityAuditError) as exc:
        raise PolicyValidityReportError(
            f"invalid_evidence: V06 reconstruction failed: {exc}"
        ) from exc
    if rebuilt != receipt:
        raise PolicyValidityReportError(
            "invalid_evidence: formal audit receipt differs from the V06 reconstruction"
        )

    manifest = _load_object(matrix_manifest_path, "formal matrix manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("execution_mode") != "formal"
        or manifest.get("formal_result") is not True
    ):
        raise PolicyValidityReportError(
            "invalid_evidence: matrix is not immutable-complete formal evidence"
        )
    manifest_sha256 = _digest(manifest.get("manifest_sha256"), "matrix manifest_sha256")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise PolicyValidityReportError(
            "invalid_evidence: formal audit is bound to another manifest"
        )
    expected_counts = {
        "primary_campaigns": 30,
        "primary_closed_lifecycles": 180,
        "retest_campaigns": 30,
        "retest_closed_lifecycles": 180,
        "provider_calls": 0,
    }
    if manifest.get("materialized_counts") != expected_counts:
        raise PolicyValidityReportError(
            "invalid_evidence: matrix materialized counts are not frozen"
        )
    counting_rules = _mapping(manifest.get("counting_rules"), "matrix counting_rules")
    if counting_rules.get("retest_in_primary_estimand") is not False:
        raise PolicyValidityReportError("invalid_evidence: retest entered the primary estimand")

    cells = _sequence(receipt.get("cells"), "formal audit cells")
    references = [
        dict(_mapping(value, f"matrix cell reference {index}"))
        for index, value in enumerate(_sequence(manifest.get("cells"), "matrix cells"))
    ]
    if len(cells) != 30 or len(references) != 30:
        raise PolicyValidityReportError("invalid_evidence: report requires exactly 30 campaigns")
    references_by_cell = {str(reference.get("cell_id")): reference for reference in references}
    if len(references_by_cell) != 30:
        raise PolicyValidityReportError("invalid_evidence: matrix bundle references are not unique")

    campaign_profiles: list[dict[str, Any]] = []
    for raw_cell in cells:
        cell = _mapping(raw_cell, "formal audit cell")
        cell_id = str(cell.get("cell_id"))
        if cell_id not in references_by_cell:
            raise PolicyValidityReportError("invalid_evidence: audit cell lacks a bundle reference")
        reference = references_by_cell[cell_id]
        campaign_profiles.append(
            {
                "cell_id": cell_id,
                "identity": deepcopy(cell.get("identity")),
                "profile": deepcopy(cell.get("profile")),
                "exact_replay": deepcopy(cell.get("exact_replay")),
                "test_retest": deepcopy(cell.get("test_retest")),
                "bundle": {
                    field: reference[field]
                    for field in (
                        "ordinal",
                        "bundle_path",
                        "bundle_sha256",
                        "file_sha256",
                        "byte_count",
                    )
                },
            }
        )

    summaries = _equal_weight_summaries(cells)
    _validate_equal_weight_summaries(
        summaries,
        _mapping(receipt.get("policy_summaries"), "formal audit policy_summaries"),
    )
    gates = dict(_mapping(receipt.get("gates"), "formal audit gates"))
    established = all(gates.values())
    status = "positive_control_established" if established else "positive_control_unestablished"
    failed_gates = [name for name in FROZEN_GATES if gates[name] is not True]
    failed_checks = _failed_checks(receipt)
    conclusion = (
        "Across the frozen five-world, two-arm matrix, the experimental-agency profile "
        "recovered the prespecified signatures, nulls, partial orderings, matched-arm "
        "invariance, resource expectations, and exact deterministic retest behavior for "
        "the three known policies. This establishes the bounded construct/discriminant-"
        "validity positive control for this simulated apparatus."
        if established
        else "The frozen known-policy positive control was not established. All failed "
        "gates and checks are retained without retuning or replacement."
    )

    dependencies = deepcopy(_mapping(manifest.get("dependency_bindings"), "dependencies"))
    report: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "W1-V09",
        "status": status,
        "read_only_analysis": True,
        "formal_world_controller_or_provider_execution": False,
        "estimand": {
            "unit": "one primary campaign profile",
            "weighting": "equal weight across ten world-arm campaigns per policy",
            "lifecycle_rows_pooled_before_profile_construction": False,
            "primary_campaigns": 30,
            "primary_closed_lifecycles": 180,
            "retest_campaigns": 30,
            "retest_closed_lifecycles": 180,
            "retest_in_primary_estimand": False,
            "provider_calls": 0,
        },
        "evidence_validity": {
            "status": "valid",
            "v06_reconstruction_exactly_matches_receipt": True,
            "all_bundle_hashes_and_byte_counts_verified": True,
            "all_profiles_rebuilt": bool(gates["all_profiles_rebuilt"]),
            "all_resource_ledgers_replayed": bool(gates["all_resource_ledgers_replayed"]),
            "formal_execution_performed_by_reporter": False,
        },
        "scientific_status": {
            "status": status,
            "established": established,
            "frozen_gates": gates,
            "failed_gates": failed_gates,
            "failed_checks": failed_checks,
            "threshold_assays": receipt["counts"]["threshold_assays"],
            "threshold_discards": receipt["counts"]["threshold_discards"],
            "threshold_non_degenerate": gates["threshold_non_degenerate"],
        },
        "policy_summaries": summaries,
        "frozen_checks": {
            "exact_signatures": deepcopy(receipt.get("exact_signature_checks")),
            "conditional_nulls": deepcopy(receipt.get("conditional_null_checks")),
            "partial_orderings": deepcopy(receipt.get("partial_ordering_checks")),
            "resource_expectations": deepcopy(receipt.get("resource_expectation_checks")),
            "matched_arm_invariance": deepcopy(receipt.get("arm_invariance")),
            "explicit_non_orderings": deepcopy(receipt.get("explicit_non_orderings")),
        },
        "test_retest_reliability": {
            "pair_count": 30,
            "same_identity_deterministic_pairs": bool(gates["all_exact_replays_and_retests_match"]),
            "all_component_hashes_match": bool(gates["all_exact_replays_and_retests_match"]),
            "excluded_from_primary_estimand": True,
        },
        "campaign_profiles": campaign_profiles,
        "input_bindings": {
            "v08_claim": _load_v08_claim(v08_claim_path),
            "v08_done_integration_commit": v08_done_commit,
            "matrix_manifest": {
                **_artifact_entry(matrix_manifest_path, "v08_matrix_manifest"),
                "manifest_sha256": manifest_sha256,
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "schedule_sha256": manifest["schedule_sha256"],
                "formal_qualification_receipt_sha256": manifest[
                    "formal_qualification_receipt_sha256"
                ],
                "bundle_count": len(references),
                "bundles": deepcopy(references),
            },
            "formal_audit": {
                **_artifact_entry(formal_audit_path, "v08_formal_audit"),
                "audit_sha256": receipt["audit_sha256"],
            },
            "v01_v03_contracts": {
                field: dependencies[field]
                for field in (
                    "profile_contract_sha256",
                    "known_policy_contract_sha256",
                    "threshold_binding_sha256",
                    "qualification_report_sha256",
                )
            },
            "resource_card": deepcopy(manifest.get("campaign_resource_card")),
            "counting_rules": deepcopy(counting_rules),
            "dependency_bindings": dependencies,
            "analyzer_sources": [
                _artifact_entry(path, role) for role, path in sorted(analyzer_source_paths.items())
            ],
        },
        "claim_boundary": {
            "allowed": (
                "bounded known-policy construct/discriminant-validity and deterministic "
                "reliability positive control"
            ),
            "endpoint_performance_ranking": False,
            "causal_material_information_null_effect": False,
            "model_or_provider_capability": False,
            "scalar_experimental_intelligence": False,
            "real_laboratory_generalization": False,
            "formal_retuning_or_result_replacement": False,
        },
        "conclusion": conclusion,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def render_policy_validity_markdown(report: Mapping[str, Any]) -> str:
    """Render the bounded human report from the self-hashed machine report."""

    science = _mapping(report["scientific_status"], "scientific_status")
    evidence = _mapping(report["evidence_validity"], "evidence_validity")
    reliability = _mapping(report["test_retest_reliability"], "test_retest_reliability")
    retest_statement = (
        "All 30 original/retest pairs matched in controller, trajectory identity, profile, "
        "and component hashes."
        if reliability["same_identity_deterministic_pairs"] is True
        else "The frozen same-identity deterministic retest gate did not pass for all 30 pairs."
    )
    profile_statement = (
        "V06 independently rebuilt all campaign profiles."
        if evidence["all_profiles_rebuilt"] is True
        else "V06 did not establish complete reconstruction of all campaign profiles."
    )
    resource_statement = (
        "V06 independently replayed all campaign resource ledgers."
        if evidence["all_resource_ledgers_replayed"] is True
        else "V06 did not establish complete replay of all campaign resource ledgers."
    )
    lines = [
        "# Work I known-policy measurement-validity report",
        "",
        f"Status: **{report['status']}**.",
        "",
        str(report["conclusion"]),
        "",
        "## Frozen design and counts",
        "",
        "The primary estimand gives one equal weight to each campaign profile: ten "
        "world-arm campaigns per policy, 30 campaigns and 180 closed lifecycles total. "
        "The 30 same-identity retest campaigns and 180 retest lifecycles are reliability "
        "evidence only and are excluded from the primary estimand. Provider calls: 0.",
        "",
        "## Campaign-equal policy summaries",
        "",
        "| Policy | Assay | Discard | Measured | Continued | Non-final instruments | Operations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = _mapping(report["policy_summaries"], "policy_summaries")
    for policy_id in POLICY_IDS:
        metrics = _mapping(_mapping(summaries[policy_id], policy_id)["metrics"], "metrics")
        lines.append(
            f"| `{policy_id}` | {float(metrics['assay_fraction']):.3f} | "
            f"{float(metrics['discard_fraction']):.3f} | "
            f"{float(metrics['measured_lifecycle_fraction']):.3f} | "
            f"{float(metrics['continued_after_measurement_fraction']):.3f} | "
            f"{float(metrics['nonfinal_instrument_uses_per_closed_lifecycle']):.3f} | "
            f"{float(metrics['attempted_operations_per_closed_lifecycle']):.3f} |"
        )
    gates = _mapping(science["frozen_gates"], "frozen_gates")
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
            *[f"- `{name}`: {'PASS' if gates[name] else 'FAIL'}" for name in FROZEN_GATES],
            "",
            "The frozen threshold policy produced "
            f"{science['threshold_assays']} assays and {science['threshold_discards']} "
            "discards; both branches were observed. All V01 conditional nulls, V02 exact "
            "signatures and six prespecified partial orderings are published in the JSON report.",
            "",
            "Explicit non-orderings remain descriptive and are not promoted to gates: "
            + "; ".join(str(value) for value in report["frozen_checks"]["explicit_non_orderings"])
            + ".",
            "",
            "## Reliability and evidence",
            "",
            f"{retest_statement} {profile_statement} {resource_statement} The V06 "
            "reconstruction exactly matched the immutable formal audit receipt.",
            "",
            "## Interpretation boundary",
            "",
            "This is a bounded construct/discriminant-validity positive control for three "
            "deterministic policies in five simulated worlds and two information arms. It is "
            "not an endpoint ranking, causal information-null result, provider/model capability "
            "claim, scalar intelligence score, or real-laboratory generalization.",
            "",
            f"Machine report SHA-256: `{report['report_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_delivery_manifest(
    *,
    report: Mapping[str, Any],
    markdown: str,
    report_path: Path,
    markdown_path: Path,
    matrix_manifest_path: Path,
    formal_audit_path: Path,
    v08_claim_path: Path,
    analyzer_source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Bind reports, immutable inputs, and analyzers without a self-reference cycle."""

    report_bytes = rendered_json_text(report).encode("utf-8")
    markdown_bytes = markdown.encode("utf-8")
    entries = [
        {
            "role": "machine_report",
            "path": report_path.as_posix(),
            "file_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "byte_count": len(report_bytes),
        },
        {
            "role": "human_report",
            "path": markdown_path.as_posix(),
            "file_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            "byte_count": len(markdown_bytes),
        },
        _artifact_entry(matrix_manifest_path, "v08_matrix_manifest"),
        _artifact_entry(formal_audit_path, "v08_formal_audit"),
        _artifact_entry(v08_claim_path, "v08_done_claim"),
        *[_artifact_entry(path, role) for role, path in sorted(analyzer_source_paths.items())],
    ]
    entries.sort(key=lambda entry: str(entry["path"]))
    manifest: dict[str, Any] = {
        "schema_id": DELIVERY_SCHEMA_ID,
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "task_id": "W1-V09",
        "status": "complete",
        "immutable": True,
        "entry_count": len(entries),
        "entries": entries,
        "bindings": {
            "report_sha256": report["report_sha256"],
            "matrix_manifest_sha256": report["input_bindings"]["matrix_manifest"][
                "manifest_sha256"
            ],
            "formal_audit_sha256": report["input_bindings"]["formal_audit"]["audit_sha256"],
            "v08_done_integration_commit": report["input_bindings"]["v08_done_integration_commit"],
        },
        "counting_rule": (
            "The machine report, Markdown report, two immutable V08 scientific inputs, "
            "V08 DONE claim, and each analyzer source appear exactly once. This delivery "
            "manifest excludes its own file to avoid a circular byte hash."
        ),
    }
    manifest["delivery_manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


__all__ = [
    "DELIVERY_SCHEMA_ID",
    "DELIVERY_SCHEMA_VERSION",
    "FROZEN_GATES",
    "REPORT_SCHEMA_ID",
    "REPORT_SCHEMA_VERSION",
    "SUMMARY_METRICS",
    "PolicyValidityReportError",
    "build_delivery_manifest",
    "build_policy_validity_report",
    "render_policy_validity_markdown",
    "rendered_json_text",
]
