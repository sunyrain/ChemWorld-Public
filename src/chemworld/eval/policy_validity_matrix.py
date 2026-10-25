"""Outcome-blind matrix orchestration for Work I known-policy controls.

This module owns scheduling, immutable cell bundles, content-addressed matrix
manifests, and fail-closed resume.  The chemistry/controller execution surface
is injected so the orchestration can be qualified without reading formal-world
outcomes before W1-V07 freezes the protocol.
"""

from __future__ import annotations

import json
import math
import sys
import sysconfig
import tomllib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceLedger,
    campaign_resource_event_id,
)
from chemworld.eval.known_policy_contract import (
    FORMAL_WORLD_SEEDS,
    INFORMATION_ARMS,
    LIFECYCLES_PER_CELL,
    POLICY_IDS,
    known_policy_contract_sha256,
)
from chemworld.eval.known_policy_threshold import (
    stable_numeric_payload,
    threshold_binding_sha256,
)
from chemworld.eval.policy_validity_contract import (
    METRICS,
    PROFILE_SCHEMA_ID,
    PROFILE_SCHEMA_VERSION,
    profile_contract_sha256,
    validate_profile_record,
)
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)

PROTOCOL_SCHEMA_ID = "chemworld.policy_control_matrix_protocol"
PROTOCOL_SCHEMA_VERSION = "0.1.0"
PREFLIGHT_SCHEMA_ID = "chemworld.policy_control_matrix_preflight"
PREFLIGHT_SCHEMA_VERSION = "0.1.0"
FORMAL_QUALIFICATION_RECEIPT_SCHEMA_ID = (
    "chemworld.policy_control_matrix_formal_qualification_receipt"
)
FORMAL_QUALIFICATION_RECEIPT_SCHEMA_VERSION = "0.1.0"
EXECUTION_SCHEMA_ID = "chemworld.policy_control_campaign_execution"
EXECUTION_SCHEMA_VERSION = "0.1.0"
CELL_BUNDLE_SCHEMA_ID = "chemworld.policy_control_cell_bundle"
CELL_BUNDLE_SCHEMA_VERSION = "0.1.0"
PROGRESS_SCHEMA_ID = "chemworld.policy_control_matrix_progress"
PROGRESS_SCHEMA_VERSION = "0.1.0"
MANIFEST_SCHEMA_ID = "chemworld.policy_control_matrix_manifest"
MANIFEST_SCHEMA_VERSION = "0.1.0"
RUNNER_VERSION = "chemworld-policy-control-matrix-runner-0.1"
EXECUTION_APPARATUS_FIELDS = (
    "python_version",
    "python_implementation",
    "python_cache_tag",
    "python_abi",
    "numpy_version",
    "numpy_wheel_filename",
    "numpy_wheel_tag",
    "numpy_wheel_sha256",
    "scipy_version",
    "scipy_wheel_filename",
    "scipy_wheel_tag",
    "scipy_wheel_sha256",
    "sysconfig_platform",
    "uv_lock_path",
    "uv_lock_sha256",
)

PRIMARY_CAMPAIGN_COUNT = 30
PRIMARY_LIFECYCLE_COUNT = 180
PROVIDER_CALL_COUNT = 0

PROGRESS_FILENAME = "matrix_progress.json"
MANIFEST_FILENAME = "matrix_manifest.json"
BUNDLE_DIRECTORY = "bundles"


class PolicyMatrixError(RuntimeError):
    """The matrix protocol, artifact chain, or resume state is invalid."""


@dataclass(frozen=True)
class MatrixCell:
    """One campaign in the frozen world-major factorial schedule."""

    ordinal: int
    cell_id: str
    world_seed: int
    information_arm: str
    policy_id: str
    material_information: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CellExecutor = Callable[
    [MatrixCell, Mapping[str, Any]],
    tuple[Mapping[str, Any], Mapping[str, Any]],
]


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop(field, None)
    return result


def semantic_sha256(payload: Any) -> str:
    """Hash audit payloads after the frozen V03 numeric normalization."""

    return canonical_json_sha256(stable_numeric_payload(deepcopy(payload)))


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return semantic_sha256(_without(payload, field))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyMatrixError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PolicyMatrixError(f"{label} must be a JSON object: {path}")
    return payload


def _locked_wheel_identity(
    lock_payload: Mapping[str, Any],
    *,
    package_name: str,
    package_version: str,
    python_abi: str,
    platform: str,
) -> dict[str, str]:
    """Resolve one exact runtime wheel from the committed uv lock."""

    wheel_tag = f"{python_abi}-{python_abi}-{platform.replace('-', '_')}"
    matches: list[dict[str, str]] = []
    packages = lock_payload.get("package")
    if isinstance(packages, list):
        for package in packages:
            if not isinstance(package, Mapping):
                continue
            if (
                package.get("name") != package_name
                or package.get("version") != package_version
            ):
                continue
            wheels = package.get("wheels")
            if not isinstance(wheels, list):
                continue
            for wheel in wheels:
                if not isinstance(wheel, Mapping):
                    continue
                url = wheel.get("url")
                digest = wheel.get("hash")
                if not isinstance(url, str) or not isinstance(digest, str):
                    continue
                filename = Path(urlsplit(url).path).name
                if filename.endswith(f"-{wheel_tag}.whl"):
                    matches.append(
                        {
                            "filename": filename,
                            "tag": wheel_tag,
                            "sha256": digest.removeprefix("sha256:"),
                        }
                    )
    if len(matches) != 1:
        raise PolicyMatrixError(
            f"uv.lock must select exactly one {package_name} {wheel_tag} wheel"
        )
    return matches[0]


def observed_execution_apparatus(root: Path) -> dict[str, Any]:
    """Return the path-independent runtime identity that controls exact bytes."""

    lock_path = root.resolve() / "uv.lock"
    try:
        numpy_version = metadata.version("numpy")
        scipy_version = metadata.version("scipy")
    except metadata.PackageNotFoundError as exc:
        raise PolicyMatrixError(
            f"required execution package is not installed: {exc.name}"
        ) from exc
    try:
        with lock_path.open("rb") as stream:
            lock_payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PolicyMatrixError("uv.lock is missing or invalid") from exc
    python_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform()
    numpy_wheel = _locked_wheel_identity(
        lock_payload,
        package_name="numpy",
        package_version=numpy_version,
        python_abi=python_abi,
        platform=platform,
    )
    scipy_wheel = _locked_wheel_identity(
        lock_payload,
        package_name="scipy",
        package_version=scipy_version,
        python_abi=python_abi,
        platform=platform,
    )
    return {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "python_implementation": sys.implementation.name,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_abi": python_abi,
        "numpy_version": numpy_version,
        "numpy_wheel_filename": numpy_wheel["filename"],
        "numpy_wheel_tag": numpy_wheel["tag"],
        "numpy_wheel_sha256": numpy_wheel["sha256"],
        "scipy_version": scipy_version,
        "scipy_wheel_filename": scipy_wheel["filename"],
        "scipy_wheel_tag": scipy_wheel["tag"],
        "scipy_wheel_sha256": scipy_wheel["sha256"],
        "sysconfig_platform": platform,
        "uv_lock_path": "uv.lock",
        "uv_lock_sha256": file_sha256(lock_path),
    }


def execution_apparatus_sha256(apparatus: Mapping[str, Any]) -> str:
    """Hash the exact frozen execution apparatus identity."""

    return semantic_sha256(dict(apparatus))


def validate_execution_apparatus(
    expected: Any, observed: Any | None = None
) -> list[str]:
    """Return one error for every frozen runtime field that differs."""

    if not isinstance(expected, Mapping):
        return ["frozen execution apparatus must be an object"]
    if set(expected) != set(EXECUTION_APPARATUS_FIELDS):
        return ["frozen execution apparatus fields are incomplete"]
    errors: list[str] = []
    for field in EXECUTION_APPARATUS_FIELDS:
        value = expected.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"frozen execution apparatus field is invalid: {field}")
        elif field.endswith("sha256") and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            errors.append(f"frozen execution apparatus hash is invalid: {field}")
    if expected.get("uv_lock_path") != "uv.lock":
        errors.append("frozen execution apparatus uv_lock_path must be uv.lock")
    if observed is None:
        return errors
    if not isinstance(observed, Mapping):
        errors.append("observed execution apparatus must be an object")
        return errors
    for field in EXECUTION_APPARATUS_FIELDS:
        value = expected.get(field)
        if isinstance(value, str) and observed.get(field) != value:
            errors.append(f"execution apparatus mismatch: {field}")
    return errors


def assert_execution_apparatus(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail before output/world construction unless the exact apparatus matches."""

    observed = observed_execution_apparatus(root)
    errors = validate_execution_apparatus(
        protocol.get("execution_apparatus"), observed
    )
    if errors:
        raise PolicyMatrixError("; ".join(errors))
    return observed


def load_matrix_protocol(path: Path) -> dict[str, Any]:
    """Load and semantically validate the V05 matrix protocol."""

    protocol = _read_json_object(path, label="policy-control matrix protocol")
    errors = validate_matrix_protocol(protocol)
    if errors:
        raise PolicyMatrixError("; ".join(errors))
    return protocol


def matrix_protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return semantic_sha256(protocol)


def campaign_resource_card(protocol: Mapping[str, Any]) -> CampaignResourceCard:
    raw = protocol.get("campaign_resource_card")
    if not isinstance(raw, Mapping):
        raise PolicyMatrixError("campaign_resource_card must be an object")
    try:
        return CampaignResourceCard.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyMatrixError("campaign_resource_card is invalid") from exc


def validate_matrix_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Return deterministic validation errors for the outcome-blind protocol."""

    errors: list[str] = []
    if protocol.get("schema_id") != PROTOCOL_SCHEMA_ID:
        errors.append("protocol schema_id mismatch")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        errors.append("protocol schema_version mismatch")
    errors.extend(validate_execution_apparatus(protocol.get("execution_apparatus")))

    matrix = protocol.get("matrix")
    if not isinstance(matrix, Mapping):
        errors.append("matrix must be an object")
    else:
        if tuple(matrix.get("world_seeds", ())) != FORMAL_WORLD_SEEDS:
            errors.append("matrix world seeds do not match V02")
        if tuple(matrix.get("information_arms", ())) != INFORMATION_ARMS:
            errors.append("matrix information arms do not match V02")
        if tuple(matrix.get("policy_ids", ())) != POLICY_IDS:
            errors.append("matrix policy IDs do not match V02")
        if matrix.get("lifecycles_per_campaign") != LIFECYCLES_PER_CELL:
            errors.append("matrix lifecycle count does not match V02")
        if matrix.get("primary_campaign_count") != PRIMARY_CAMPAIGN_COUNT:
            errors.append("matrix campaign count must be 30")
        if matrix.get("primary_closed_lifecycle_count") != PRIMARY_LIFECYCLE_COUNT:
            errors.append("matrix lifecycle count must be 180")
        if matrix.get("provider_call_count") != PROVIDER_CALL_COUNT:
            errors.append("known-policy controls must make zero provider calls")
        if matrix.get("schedule_order") != "world_then_arm_then_policy":
            errors.append("matrix schedule order is not canonical")

    bindings = protocol.get("dependency_bindings")
    if not isinstance(bindings, Mapping):
        errors.append("dependency_bindings must be an object")
    else:
        if bindings.get("profile_contract_sha256") != profile_contract_sha256():
            errors.append("profile contract binding is stale")
        if bindings.get("known_policy_contract_sha256") != known_policy_contract_sha256():
            errors.append("known-policy contract binding is stale")

    arm_payloads = protocol.get("material_information_by_arm")
    if not isinstance(arm_payloads, Mapping) or set(arm_payloads) != set(
        INFORMATION_ARMS
    ):
        errors.append("material-information payloads must cover the frozen arms")
    elif any(not isinstance(arm_payloads[arm], Mapping) for arm in INFORMATION_ARMS):
        errors.append("each material-information payload must be an object")

    source_paths = protocol.get("source_paths")
    if (
        not isinstance(source_paths, list)
        or not source_paths
        or any(not isinstance(path, str) or not path for path in source_paths)
        or len(source_paths) != len(set(source_paths))
    ):
        errors.append("source_paths must be a non-empty unique string list")

    resume = protocol.get("resume_policy")
    required_resume = {
        "canonical_prefix_only": True,
        "accepted_bundles_are_immutable": True,
        "identity_drift_fails_closed": True,
        "corruption_fails_closed": True,
        "unexpected_files_fail_closed": True,
        "atomic_bundle_publication": True,
        "atomic_progress_publication": True,
        "completed_manifest_is_terminal": True,
    }
    if not isinstance(resume, Mapping) or any(
        resume.get(key) is not value for key, value in required_resume.items()
    ):
        errors.append("resume policy is weaker than the frozen fail-closed contract")

    counting = protocol.get("counting_rules")
    if not isinstance(counting, Mapping):
        errors.append("counting_rules must be an object")
    else:
        if counting.get("retest_in_primary_estimand") is not False:
            errors.append("retest executions must not enter the primary estimand")
        if counting.get("primary_campaign_unit") != "one scheduled cell":
            errors.append("primary campaign counting unit is not frozen")

    try:
        card = campaign_resource_card(protocol)
    except PolicyMatrixError as exc:
        errors.append(str(exc))
    else:
        if card.operation_attempt_limit != 48:
            errors.append("resource card must admit the maximum 48-operation path")
        if card.vessel_start_limit != LIFECYCLES_PER_CELL:
            errors.append("resource card must admit six vessel starts")
        if card.final_assay_limit != LIFECYCLES_PER_CELL:
            errors.append("resource card must admit six final assays")
        if card.nonfinal_instrument_use_limit != LIFECYCLES_PER_CELL:
            errors.append("resource card must admit six non-final measurements")
    return errors


def canonical_schedule(protocol: Mapping[str, Any]) -> tuple[MatrixCell, ...]:
    """Materialize the immutable world-major V02 schedule."""

    errors = validate_matrix_protocol(protocol)
    if errors:
        raise PolicyMatrixError("; ".join(errors))
    matrix = protocol["matrix"]
    arm_payloads = protocol["material_information_by_arm"]
    cells: list[MatrixCell] = []
    for world_seed in matrix["world_seeds"]:
        for information_arm in matrix["information_arms"]:
            arm_slug = (
                "opaque"
                if information_arm == "opaque_codes"
                else "anonymous-nominal"
            )
            for policy_id in matrix["policy_ids"]:
                ordinal = len(cells) + 1
                cells.append(
                    MatrixCell(
                        ordinal=ordinal,
                        cell_id=(
                            f"cell-{ordinal:02d}-world-{int(world_seed):04d}-"
                            f"{arm_slug}-{policy_id.replace('_', '-')}"
                        ),
                        world_seed=int(world_seed),
                        information_arm=str(information_arm),
                        policy_id=str(policy_id),
                        material_information=deepcopy(
                            dict(arm_payloads[information_arm])
                        ),
                    )
                )
    if len(cells) != PRIMARY_CAMPAIGN_COUNT:
        raise PolicyMatrixError("canonical schedule does not contain 30 cells")
    return tuple(cells)


def source_manifest(root: Path, protocol: Mapping[str, Any]) -> dict[str, str]:
    """Hash the complete declared V05 source surface using relative paths."""

    resolved = root.resolve()
    result: dict[str, str] = {}
    for relative in protocol["source_paths"]:
        path = (resolved / str(relative)).resolve()
        if not path.is_relative_to(resolved) or not path.is_file():
            raise PolicyMatrixError(f"missing or invalid source path: {relative}")
        result[str(relative)] = file_sha256(path)
    return result


def dependency_bindings(
    root: Path,
    protocol: Mapping[str, Any],
    *,
    execution_apparatus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate V01-V03 and disclose the separately owned V04 availability."""

    expected = protocol["dependency_bindings"]
    apparatus = (
        dict(execution_apparatus)
        if execution_apparatus is not None
        else assert_execution_apparatus(root, protocol)
    )
    profile_path = root / str(expected["profile_contract_path"])
    policy_path = root / str(expected["known_policy_contract_path"])
    threshold_path = root / str(expected["threshold_binding_path"])
    qualification_path = root / str(expected["qualification_report_path"])
    profile = _read_json_object(profile_path, label="V01 profile contract")
    policy = _read_json_object(policy_path, label="V02 known-policy contract")
    threshold = _read_json_object(threshold_path, label="V03 threshold binding")
    qualification = _read_json_object(
        qualification_path,
        label="V03 qualification report",
    )
    observed = {
        "profile_contract_sha256": profile.get("contract_sha256"),
        "known_policy_contract_sha256": policy.get("contract_sha256"),
        "threshold_binding_sha256": threshold.get("binding_sha256"),
        "qualification_report_sha256": qualification.get("report_sha256"),
    }
    for key, value in observed.items():
        if value != expected.get(key):
            raise PolicyMatrixError(f"stale dependency binding: {key}")
    if profile_contract_sha256(_without(profile, "contract_sha256")) != observed[
        "profile_contract_sha256"
    ]:
        raise PolicyMatrixError("V01 profile contract content hash mismatch")
    if known_policy_contract_sha256(policy) != observed["known_policy_contract_sha256"]:
        raise PolicyMatrixError("V02 known-policy contract content hash mismatch")
    if threshold_binding_sha256(threshold) != observed["threshold_binding_sha256"]:
        raise PolicyMatrixError("V03 threshold binding content hash mismatch")
    if threshold.get("qualification_report_sha256") != observed[
        "qualification_report_sha256"
    ]:
        raise PolicyMatrixError("V03 threshold/report binding mismatch")

    controller = protocol.get("controller_dependency", {})
    controller_path = root / str(controller.get("path", ""))
    controller_available = controller_path.is_file()
    return {
        **observed,
        "execution_apparatus": apparatus,
        "execution_apparatus_sha256": execution_apparatus_sha256(apparatus),
        "threshold": threshold.get("threshold"),
        "threshold_comparator": threshold.get("comparator"),
        "formal_retuning_forbidden": threshold.get("formal_retuning_forbidden"),
        "controller": {
            "path": str(controller.get("path")),
            "required_task": "W1-V04",
            "status": "available" if controller_available else "pending_W1-V04_merge",
            "sha256": file_sha256(controller_path) if controller_available else None,
        },
    }


def build_preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    """Build an outcome-blind schedule and source qualification artifact."""

    protocol = load_matrix_protocol(protocol_path)
    apparatus = assert_execution_apparatus(root, protocol)
    schedule = canonical_schedule(protocol)
    sources = source_manifest(root, protocol)
    dependencies = dependency_bindings(
        root, protocol, execution_apparatus=apparatus
    )
    card = campaign_resource_card(protocol)
    checks = {
        "protocol_valid": not validate_matrix_protocol(protocol),
        "schedule_has_30_unique_cells": (
            len(schedule) == PRIMARY_CAMPAIGN_COUNT
            and len({cell.cell_id for cell in schedule}) == PRIMARY_CAMPAIGN_COUNT
        ),
        "schedule_has_180_primary_lifecycles": (
            len(schedule) * LIFECYCLES_PER_CELL == PRIMARY_LIFECYCLE_COUNT
        ),
        "provider_calls_are_zero": protocol["matrix"]["provider_call_count"] == 0,
        "source_manifest_complete": len(sources) == len(protocol["source_paths"]),
        "frozen_contract_chain_matches": all(
            dependencies[key] == protocol["dependency_bindings"][key]
            for key in (
                "profile_contract_sha256",
                "known_policy_contract_sha256",
                "threshold_binding_sha256",
                "qualification_report_sha256",
            )
        ),
        "execution_apparatus_matches": (
            dependencies["execution_apparatus"] == protocol["execution_apparatus"]
            and dependencies["execution_apparatus_sha256"]
            == execution_apparatus_sha256(protocol["execution_apparatus"])
        ),
        "formal_outcomes_not_read": True,
    }
    report: dict[str, Any] = {
        "schema_id": PREFLIGHT_SCHEMA_ID,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "status": "outcome_blind_preflight_passed" if all(checks.values()) else "failed",
        "formal_result": False,
        "formal_execution_authorized": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": matrix_protocol_sha256(protocol),
        "execution_apparatus": apparatus,
        "execution_apparatus_sha256": execution_apparatus_sha256(apparatus),
        "source_manifest": sources,
        "source_manifest_sha256": semantic_sha256(sources),
        "dependency_bindings": dependencies,
        "campaign_resource_card": card.to_dict(),
        "schedule": [cell.to_dict() for cell in schedule],
        "schedule_sha256": semantic_sha256(
            [cell.to_dict() for cell in schedule]
        ),
        "expected_counts": {
            "primary_campaigns": PRIMARY_CAMPAIGN_COUNT,
            "primary_closed_lifecycles": PRIMARY_LIFECYCLE_COUNT,
            "retest_campaigns": PRIMARY_CAMPAIGN_COUNT,
            "retest_closed_lifecycles": PRIMARY_LIFECYCLE_COUNT,
            "provider_calls": PROVIDER_CALL_COUNT,
        },
        "artifact_schemas": {
            "formal_qualification_receipt": (
                f"{FORMAL_QUALIFICATION_RECEIPT_SCHEMA_ID}@"
                f"{FORMAL_QUALIFICATION_RECEIPT_SCHEMA_VERSION}"
            ),
            "execution": f"{EXECUTION_SCHEMA_ID}@{EXECUTION_SCHEMA_VERSION}",
            "cell_bundle": f"{CELL_BUNDLE_SCHEMA_ID}@{CELL_BUNDLE_SCHEMA_VERSION}",
            "progress": f"{PROGRESS_SCHEMA_ID}@{PROGRESS_SCHEMA_VERSION}",
            "manifest": f"{MANIFEST_SCHEMA_ID}@{MANIFEST_SCHEMA_VERSION}",
        },
        "resume_policy": deepcopy(protocol["resume_policy"]),
        "counting_rules": deepcopy(protocol["counting_rules"]),
        "checks": checks,
        "claim_boundary": (
            "This artifact validates only the frozen schedule, dependency identities, "
            "artifact schemas, and resume contract. It reads no formal-world outcome "
            "and does not qualify or execute the formal matrix."
        ),
    }
    report["preflight_sha256"] = _self_hash(report, "preflight_sha256")
    return report


def validate_preflight(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_id") != PREFLIGHT_SCHEMA_ID:
        errors.append("preflight schema_id mismatch")
    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        errors.append("preflight schema_version mismatch")
    if report.get("formal_result") is not False:
        errors.append("preflight must not be a formal result")
    if report.get("formal_execution_authorized") is not False:
        errors.append("preflight must not authorize formal execution")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        errors.append("one or more preflight gates failed")
    if report.get("preflight_sha256") != _self_hash(
        report,
        "preflight_sha256",
    ):
        errors.append("preflight self-hash mismatch")
    if report.get("source_manifest_sha256") != semantic_sha256(
        report.get("source_manifest")
    ):
        errors.append("preflight source-manifest hash mismatch")
    apparatus = report.get("execution_apparatus")
    if validate_execution_apparatus(apparatus):
        errors.append("preflight execution apparatus is invalid")
    if report.get("execution_apparatus_sha256") != (
        execution_apparatus_sha256(apparatus)
        if isinstance(apparatus, Mapping)
        else None
    ):
        errors.append("preflight execution-apparatus hash mismatch")
    dependencies = report.get("dependency_bindings")
    if not isinstance(dependencies, Mapping) or (
        dependencies.get("execution_apparatus") != apparatus
        or dependencies.get("execution_apparatus_sha256")
        != report.get("execution_apparatus_sha256")
    ):
        errors.append("preflight execution-apparatus dependency binding mismatch")
    return errors


def load_formal_qualification_receipt(path: Path) -> dict[str, Any]:
    """Load a separately owned W1-V07 formal-execution receipt."""

    return _read_json_object(path, label="W1-V07 formal qualification receipt")


def formal_qualification_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the receipt self-hash after excluding its hash field."""

    return _self_hash(receipt, "receipt_sha256")


def validate_formal_qualification_receipt(
    receipt: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
) -> list[str]:
    """Validate that W1-V07 qualified and froze the exact current runner."""

    errors: list[str] = []
    if receipt.get("schema_id") != FORMAL_QUALIFICATION_RECEIPT_SCHEMA_ID:
        errors.append("formal qualification receipt schema_id mismatch")
    if (
        receipt.get("schema_version")
        != FORMAL_QUALIFICATION_RECEIPT_SCHEMA_VERSION
    ):
        errors.append("formal qualification receipt schema_version mismatch")
    if receipt.get("task_id") != "W1-V07":
        errors.append("formal qualification receipt task_id must be W1-V07")

    gates = receipt.get("qualification_gates")
    if not isinstance(gates, Mapping):
        errors.append("formal qualification receipt gates must be an object")
    else:
        if gates.get("runner_qualified") is not True:
            errors.append("formal qualification receipt runner_qualified gate is false")
        if gates.get("protocol_frozen") is not True:
            errors.append("formal qualification receipt protocol_frozen gate is false")

    preflight_errors = validate_preflight(preflight)
    if preflight_errors:
        errors.append("formal qualification receipt is bound to an invalid preflight")
    dependency_bindings = preflight.get("dependency_bindings")
    controller = (
        dependency_bindings.get("controller")
        if isinstance(dependency_bindings, Mapping)
        else None
    )
    if not isinstance(controller, Mapping) or controller.get("status") != "available":
        errors.append("formal qualification receipt requires the available V04 controller")
        controller_sha256 = None
    else:
        controller_sha256 = controller.get("sha256")

    expected_bindings = {
        "matrix_protocol_sha256": preflight.get("protocol_sha256"),
        "source_manifest_sha256": preflight.get("source_manifest_sha256"),
        "preflight_sha256": preflight.get("preflight_sha256"),
        "controller_sha256": controller_sha256,
        "execution_apparatus_sha256": preflight.get(
            "execution_apparatus_sha256"
        ),
    }
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        errors.append("formal qualification receipt bindings must be an object")
    else:
        for key, expected in expected_bindings.items():
            if not isinstance(expected, str) or bindings.get(key) != expected:
                errors.append(f"formal qualification receipt binding is stale: {key}")

    if receipt.get("receipt_sha256") != formal_qualification_receipt_sha256(receipt):
        errors.append("formal qualification receipt self-hash mismatch")
    return errors


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def execution_component_hashes(execution: Mapping[str, Any]) -> dict[str, str]:
    """Rebuild the independently addressable V06 consumer components."""

    records = execution.get("trajectory_records")
    states = [record.get("state") for record in records] if isinstance(records, list) else []
    profile = execution.get("profile_record")
    endpoint = profile.get("endpoint_context") if isinstance(profile, Mapping) else None
    return {
        "event_sha256": semantic_sha256(records),
        "state_sha256": semantic_sha256(states),
        "resource_sha256": semantic_sha256(execution.get("campaign_resource_ledger")),
        "terminal_sha256": semantic_sha256(execution.get("lifecycle_terminals")),
        "profile_sha256": semantic_sha256(profile),
        "endpoint_sha256": semantic_sha256(endpoint),
        "controller_sha256": semantic_sha256(execution.get("controller_manifest")),
        "decision_audit_sha256": semantic_sha256(execution.get("decision_audits")),
    }


def finalize_execution_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach producer component hashes and a self-hash to one execution."""

    result = deepcopy(dict(payload))
    result["component_hashes"] = execution_component_hashes(result)
    result["execution_sha256"] = _self_hash(result, "execution_sha256")
    return result


def _validate_ledger(
    ledger: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    card_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if ledger.get("ledger_sha256") != canonical_json_sha256(
        _without(ledger, "ledger_sha256")
    ):
        errors.append("campaign resource ledger self-hash mismatch")
    card = ledger.get("card")
    if not isinstance(card, Mapping) or card.get("card_sha256") != card_sha256:
        errors.append("campaign resource ledger card binding mismatch")
    events = ledger.get("events")
    if not isinstance(events, list) or len(events) != len(records):
        errors.append("campaign resource events do not align with trajectory records")
    elif all(isinstance(event, Mapping) for event in events):
        for index, (event, record) in enumerate(zip(events, records, strict=True), start=1):
            if event.get("action") != record.get("action"):
                errors.append(f"resource event {index} action mismatch")
                break
    state = ledger.get("state")
    if not isinstance(state, Mapping):
        errors.append("campaign resource ledger state must be an object")
    return errors


def validate_execution_record(
    execution: Mapping[str, Any],
    *,
    cell: MatrixCell,
    execution_role: str,
    card_sha256: str,
) -> list[str]:
    """Validate a complete producer execution without trusting its hashes."""

    errors: list[str] = []
    if execution.get("schema_id") != EXECUTION_SCHEMA_ID:
        errors.append("execution schema_id mismatch")
    if execution.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        errors.append("execution schema_version mismatch")
    if execution.get("execution_role") != execution_role:
        errors.append("execution role mismatch")
    identity = execution.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("execution identity must be an object")
    else:
        expected = {
            "cell_id": cell.cell_id,
            "world_seed": cell.world_seed,
            "information_arm": cell.information_arm,
            "policy_id": cell.policy_id,
            "resource_card_sha256": card_sha256,
            "material_information_sha256": semantic_sha256(
                cell.material_information
            ),
        }
        for field, value in expected.items():
            if identity.get(field) != value:
                errors.append(f"execution identity mismatch: {field}")
        for field in (
            "campaign_id",
            "world_id",
            "observation_noise_namespace",
        ):
            if not isinstance(identity.get(field), str) or not identity[field]:
                errors.append(f"execution identity missing: {field}")

    records = execution.get("trajectory_records")
    if not isinstance(records, list) or not records:
        errors.append("trajectory_records must be a non-empty list")
        typed_records: list[Mapping[str, Any]] = []
    else:
        typed_records = [record for record in records if isinstance(record, Mapping)]
        if len(typed_records) != len(records):
            errors.append("every trajectory record must be an object")
        for index, record in enumerate(typed_records, start=1):
            if record.get("event_index") != index:
                errors.append("trajectory event indices are not contiguous")
                break
            required = ("lifecycle_index", "action", "observation", "info", "state")
            if any(field not in record for field in required):
                errors.append(f"trajectory record {index} is incomplete")
                break

    audits = execution.get("decision_audits")
    if not isinstance(audits, list) or len(audits) != len(typed_records):
        errors.append("decision audits do not align with trajectory records")
    controller = execution.get("controller_manifest")
    if not isinstance(controller, Mapping):
        errors.append("controller manifest must be an object")
    else:
        if controller.get("policy_id") != cell.policy_id:
            errors.append("controller manifest policy mismatch")
        if controller.get("provider_call_count") != 0:
            errors.append("controller manifest reports provider calls")

    counts = execution.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("execution counts must be an object")
    else:
        closed = counts.get("closed_lifecycle_count")
        assays = counts.get("final_assay_count")
        discards = counts.get("discard_count")
        if closed != LIFECYCLES_PER_CELL:
            errors.append("execution did not close six lifecycles")
        if not isinstance(assays, int) or not isinstance(discards, int):
            errors.append("terminal counts must be integers")
        elif assays + discards != closed:
            errors.append("terminal counts do not reconcile")
        if counts.get("provider_call_count") != 0:
            errors.append("execution reports provider calls")
        if counts.get("attempted_operation_count") != len(typed_records):
            errors.append("attempted operation count does not match trajectory")

    terminals = execution.get("lifecycle_terminals")
    if not isinstance(terminals, list) or len(terminals) != LIFECYCLES_PER_CELL:
        errors.append("lifecycle terminals must contain six entries")
    else:
        for index, terminal in enumerate(terminals):
            if not isinstance(terminal, Mapping) or terminal.get("lifecycle_index") != index:
                errors.append("lifecycle terminal indices are not canonical")
                break
            if terminal.get("terminal_kind") not in {"assay", "discard"}:
                errors.append("lifecycle terminal kind is invalid")
                break
            score = terminal.get("terminal_score")
            if terminal.get("terminal_kind") == "assay" and _finite_number(score) is None:
                errors.append("assayed lifecycle requires a finite terminal score")
                break
            if terminal.get("terminal_kind") == "discard" and score is not None:
                errors.append("discarded lifecycle terminal score must be null")
                break

    ledger = execution.get("campaign_resource_ledger")
    if not isinstance(ledger, Mapping):
        errors.append("campaign_resource_ledger must be an object")
    else:
        errors.extend(
            _validate_ledger(
                ledger,
                records=typed_records,
                card_sha256=card_sha256,
            )
        )

    profile = execution.get("profile_record")
    if not isinstance(profile, Mapping):
        errors.append("profile_record must be an object")
    else:
        errors.extend(f"profile: {error}" for error in validate_profile_record(profile))
        profile_identity = profile.get("identity")
        if isinstance(profile_identity, Mapping):
            campaign_identity = (
                identity if isinstance(identity, Mapping) else {}
            )
            for field, value in (
                ("campaign_id", campaign_identity.get("campaign_id")),
                ("world_id", campaign_identity.get("world_id")),
                ("information_arm", cell.information_arm),
                ("policy_id", cell.policy_id),
                ("resource_card_sha256", card_sha256),
            ):
                if profile_identity.get(field) != value:
                    errors.append(f"profile identity mismatch: {field}")
        profile_counts = profile.get("counts")
        if isinstance(profile_counts, Mapping) and isinstance(counts, Mapping):
            for field in (
                "planned_lifecycle_count",
                "closed_lifecycle_count",
                "final_assay_count",
                "discard_count",
                "measured_lifecycle_count",
                "threshold_eligible_lifecycle_count",
            ):
                if profile_counts.get(field) != counts.get(field):
                    errors.append(f"profile/execution count mismatch: {field}")

    declared_components = execution.get("component_hashes")
    observed_components = execution_component_hashes(execution)
    if declared_components != observed_components:
        errors.append("execution component hashes do not rebuild")
    if execution.get("execution_sha256") != _self_hash(execution, "execution_sha256"):
        errors.append("execution self-hash mismatch")
    return errors


def build_cell_bundle(
    *,
    cell: MatrixCell,
    protocol_sha256: str,
    source_manifest_sha256: str,
    dependency_identity: Mapping[str, Any],
    card_sha256: str,
    original: Mapping[str, Any],
    retest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate two executions and return one immutable cell bundle."""

    original_record = finalize_execution_record(original)
    retest_record = finalize_execution_record(retest)
    errors = validate_execution_record(
        original_record,
        cell=cell,
        execution_role="original",
        card_sha256=card_sha256,
    )
    errors.extend(
        validate_execution_record(
            retest_record,
            cell=cell,
            execution_role="retest",
            card_sha256=card_sha256,
        )
    )
    components = tuple(execution_component_hashes(original_record))
    matches = {
        component: (
            original_record["component_hashes"][component]
            == retest_record["component_hashes"][component]
        )
        for component in components
    }
    original_identity = original_record.get("identity", {})
    retest_identity = retest_record.get("identity", {})
    same_identity = original_identity == retest_identity
    if not same_identity:
        errors.append("original/retest campaign identities differ")
    if not all(matches.values()):
        errors.append("original/retest component hashes differ")
    if errors:
        raise PolicyMatrixError("; ".join(dict.fromkeys(errors)))

    bundle: dict[str, Any] = {
        "schema_id": CELL_BUNDLE_SCHEMA_ID,
        "schema_version": CELL_BUNDLE_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "cell": cell.to_dict(),
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "dependency_bindings": deepcopy(dict(dependency_identity)),
        "campaign_resource_card_sha256": card_sha256,
        "original": original_record,
        "retest": retest_record,
        "retest_audit": {
            "same_identity": same_identity,
            "component_matches": matches,
            "all_components_match": all(matches.values()),
        },
        "counting": {
            "primary_campaigns": 1,
            "primary_closed_lifecycles": LIFECYCLES_PER_CELL,
            "retest_campaigns": 1,
            "retest_closed_lifecycles": LIFECYCLES_PER_CELL,
            "retest_in_primary_estimand": False,
            "provider_calls": 0,
        },
    }
    bundle["bundle_sha256"] = _self_hash(bundle, "bundle_sha256")
    return bundle


def validate_cell_bundle(
    bundle: Mapping[str, Any],
    *,
    cell: MatrixCell,
    protocol_sha256: str,
    source_manifest_sha256: str,
    dependency_identity: Mapping[str, Any],
    card_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if bundle.get("schema_id") != CELL_BUNDLE_SCHEMA_ID:
        errors.append("bundle schema_id mismatch")
    if bundle.get("schema_version") != CELL_BUNDLE_SCHEMA_VERSION:
        errors.append("bundle schema_version mismatch")
    if bundle.get("runner_version") != RUNNER_VERSION:
        errors.append("bundle runner version mismatch")
    if bundle.get("cell") != cell.to_dict():
        errors.append("bundle cell identity mismatch")
    if bundle.get("protocol_sha256") != protocol_sha256:
        errors.append("bundle protocol binding mismatch")
    if bundle.get("source_manifest_sha256") != source_manifest_sha256:
        errors.append("bundle source binding mismatch")
    if bundle.get("dependency_bindings") != dict(dependency_identity):
        errors.append("bundle dependency binding mismatch")
    if bundle.get("campaign_resource_card_sha256") != card_sha256:
        errors.append("bundle resource-card binding mismatch")
    for role in ("original", "retest"):
        execution = bundle.get(role)
        if not isinstance(execution, Mapping):
            errors.append(f"bundle {role} execution is missing")
        else:
            errors.extend(
                f"{role}: {error}"
                for error in validate_execution_record(
                    execution,
                    cell=cell,
                    execution_role=role,
                    card_sha256=card_sha256,
                )
            )
    original = bundle.get("original")
    retest = bundle.get("retest")
    if isinstance(original, Mapping) and isinstance(retest, Mapping):
        component_matches = {
            key: original.get("component_hashes", {}).get(key)
            == retest.get("component_hashes", {}).get(key)
            for key in execution_component_hashes(original)
        }
        expected_retest = {
            "same_identity": original.get("identity") == retest.get("identity"),
            "component_matches": component_matches,
            "all_components_match": all(component_matches.values()),
        }
        if bundle.get("retest_audit") != expected_retest:
            errors.append("bundle retest audit does not rebuild")
        if not expected_retest["same_identity"] or not expected_retest[
            "all_components_match"
        ]:
            errors.append("bundle retest gate failed")
    if bundle.get("bundle_sha256") != _self_hash(bundle, "bundle_sha256"):
        errors.append("bundle self-hash mismatch")
    return errors


def _bundle_relative_path(cell: MatrixCell) -> str:
    return f"{BUNDLE_DIRECTORY}/{cell.cell_id}.json"


def _bundle_entry(path: Path, bundle: Mapping[str, Any], cell: MatrixCell) -> dict[str, Any]:
    return {
        "ordinal": cell.ordinal,
        "cell_id": cell.cell_id,
        "world_seed": cell.world_seed,
        "information_arm": cell.information_arm,
        "policy_id": cell.policy_id,
        "bundle_path": _bundle_relative_path(cell),
        "bundle_sha256": bundle["bundle_sha256"],
        "file_sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
        "primary_closed_lifecycles": LIFECYCLES_PER_CELL,
        "retest_closed_lifecycles": LIFECYCLES_PER_CELL,
        "provider_calls": 0,
    }


def _common_matrix_payload(
    *,
    protocol: Mapping[str, Any],
    sources: Mapping[str, str],
    dependencies: Mapping[str, Any],
    schedule: Sequence[MatrixCell],
    card: CampaignResourceCard,
    execution_mode: str,
    formal_qualification_receipt_sha256: str | None,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": matrix_protocol_sha256(protocol),
        "source_manifest": deepcopy(dict(sources)),
        "source_manifest_sha256": semantic_sha256(sources),
        "dependency_bindings": deepcopy(dict(dependencies)),
        "schedule_sha256": semantic_sha256([cell.to_dict() for cell in schedule]),
        "campaign_resource_card": card.to_dict(),
        "execution_mode": execution_mode,
        "formal_qualification_receipt_sha256": (
            formal_qualification_receipt_sha256
        ),
        "formal_result": execution_mode == "formal",
        "expected_counts": {
            "primary_campaigns": PRIMARY_CAMPAIGN_COUNT,
            "primary_closed_lifecycles": PRIMARY_LIFECYCLE_COUNT,
            "retest_campaigns": PRIMARY_CAMPAIGN_COUNT,
            "retest_closed_lifecycles": PRIMARY_LIFECYCLE_COUNT,
            "provider_calls": 0,
        },
        "materialized_counts": {
            "primary_campaigns": len(entries),
            "primary_closed_lifecycles": len(entries) * LIFECYCLES_PER_CELL,
            "retest_campaigns": len(entries),
            "retest_closed_lifecycles": len(entries) * LIFECYCLES_PER_CELL,
            "provider_calls": 0,
        },
        "counting_rules": deepcopy(protocol["counting_rules"]),
        "cells": [deepcopy(dict(entry)) for entry in entries],
    }


def _progress_payload(
    *,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_id": PROGRESS_SCHEMA_ID,
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "in_progress",
        **deepcopy(dict(common)),
    }
    payload["progress_sha256"] = _self_hash(payload, "progress_sha256")
    return payload


def _pair_audits(
    bundles: Mapping[str, Mapping[str, Any]],
    schedule: Sequence[MatrixCell],
) -> list[dict[str, Any]]:
    cells = {(cell.world_seed, cell.information_arm, cell.policy_id): cell for cell in schedule}
    audits: list[dict[str, Any]] = []
    for world_seed in FORMAL_WORLD_SEEDS:
        for policy_id in POLICY_IDS:
            left_cell = cells[(world_seed, INFORMATION_ARMS[0], policy_id)]
            right_cell = cells[(world_seed, INFORMATION_ARMS[1], policy_id)]
            left = bundles[left_cell.cell_id]["original"]
            right = bundles[right_cell.cell_id]["original"]
            left_identity = left["identity"]
            right_identity = right["identity"]
            physical_match = left_identity.get("physical_identity") == right_identity.get(
                "physical_identity"
            )
            left_profile = left["profile_record"]
            right_profile = right["profile_record"]
            left_records = left["trajectory_records"]
            right_records = right["trajectory_records"]
            component_matches = {
                "action_trace": semantic_sha256(
                    [record["action"] for record in left_records]
                )
                == semantic_sha256([record["action"] for record in right_records]),
                "public_observations": semantic_sha256(
                    [record["observation"] for record in left_records]
                )
                == semantic_sha256(
                    [record["observation"] for record in right_records]
                ),
                "state": left["component_hashes"]["state_sha256"]
                == right["component_hashes"]["state_sha256"],
                "resource_state": semantic_sha256(
                    left["campaign_resource_ledger"]["state"]
                )
                == semantic_sha256(right["campaign_resource_ledger"]["state"]),
                "terminal": left["component_hashes"]["terminal_sha256"]
                == right["component_hashes"]["terminal_sha256"],
                "profile_values": semantic_sha256(
                    {
                        "counts": left_profile["counts"],
                        "construct_axes": left_profile["construct_axes"],
                    }
                )
                == semantic_sha256(
                    {
                        "counts": right_profile["counts"],
                        "construct_axes": right_profile["construct_axes"],
                    }
                ),
                "endpoint": left["component_hashes"]["endpoint_sha256"]
                == right["component_hashes"]["endpoint_sha256"],
                "controller": left["component_hashes"]["controller_sha256"]
                == right["component_hashes"]["controller_sha256"],
            }
            audits.append(
                {
                    "world_seed": world_seed,
                    "policy_id": policy_id,
                    "left_cell_id": left_cell.cell_id,
                    "right_cell_id": right_cell.cell_id,
                    "physical_identity_match": physical_match,
                    "component_matches": component_matches,
                    "passed": physical_match and all(component_matches.values()),
                }
            )
    return audits


def _manifest_payload(
    *,
    common: Mapping[str, Any],
    bundles: Mapping[str, Mapping[str, Any]],
    schedule: Sequence[MatrixCell],
) -> dict[str, Any]:
    pair_audits = _pair_audits(bundles, schedule)
    payload = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        **deepcopy(dict(common)),
        "matched_arm_audits": pair_audits,
        "all_matched_arm_audits_passed": all(item["passed"] for item in pair_audits),
        "immutable": True,
    }
    payload["manifest_sha256"] = _self_hash(payload, "manifest_sha256")
    return payload


def _validate_matrix_identity(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    sources: Mapping[str, str],
    dependencies: Mapping[str, Any],
    schedule: Sequence[MatrixCell],
    card: CampaignResourceCard,
    execution_mode: str,
    formal_qualification_receipt_sha256: str | None,
) -> list[str]:
    checks = {
        "runner_version": payload.get("runner_version") == RUNNER_VERSION,
        "protocol_id": payload.get("protocol_id") == protocol["protocol_id"],
        "protocol_sha256": payload.get("protocol_sha256")
        == matrix_protocol_sha256(protocol),
        "source_manifest": payload.get("source_manifest") == dict(sources),
        "source_manifest_sha256": payload.get("source_manifest_sha256")
        == semantic_sha256(sources),
        "dependency_bindings": payload.get("dependency_bindings")
        == dict(dependencies),
        "schedule_sha256": payload.get("schedule_sha256")
        == semantic_sha256([cell.to_dict() for cell in schedule]),
        "campaign_resource_card": payload.get("campaign_resource_card")
        == card.to_dict(),
        "execution_mode": payload.get("execution_mode") == execution_mode,
        "formal_qualification_receipt_sha256": payload.get(
            "formal_qualification_receipt_sha256"
        )
        == formal_qualification_receipt_sha256,
    }
    return [f"matrix identity mismatch: {key}" for key, passed in checks.items() if not passed]


def _load_and_validate_bundle(
    output_root: Path,
    entry: Mapping[str, Any],
    cell: MatrixCell,
    *,
    protocol_sha256: str,
    source_manifest_sha256: str,
    dependencies: Mapping[str, Any],
    card_sha256: str,
) -> dict[str, Any]:
    expected_relative = _bundle_relative_path(cell)
    if entry.get("bundle_path") != expected_relative:
        raise PolicyMatrixError(f"resume bundle path mismatch: {cell.cell_id}")
    path = output_root / expected_relative
    if not path.is_file():
        raise PolicyMatrixError(f"resume bundle is missing: {cell.cell_id}")
    if path.stat().st_size != entry.get("byte_count"):
        raise PolicyMatrixError(f"resume bundle byte count mismatch: {cell.cell_id}")
    if file_sha256(path) != entry.get("file_sha256"):
        raise PolicyMatrixError(f"resume bundle file hash mismatch: {cell.cell_id}")
    bundle = _read_json_object(path, label=f"bundle {cell.cell_id}")
    errors = validate_cell_bundle(
        bundle,
        cell=cell,
        protocol_sha256=protocol_sha256,
        source_manifest_sha256=source_manifest_sha256,
        dependency_identity=dependencies,
        card_sha256=card_sha256,
    )
    if bundle.get("bundle_sha256") != entry.get("bundle_sha256"):
        errors.append("bundle entry semantic hash mismatch")
    if errors:
        raise PolicyMatrixError(
            f"resume bundle validation failed for {cell.cell_id}: "
            + "; ".join(errors)
        )
    return bundle


def _load_resume_prefix(
    output_root: Path,
    *,
    protocol: Mapping[str, Any],
    sources: Mapping[str, str],
    dependencies: Mapping[str, Any],
    schedule: Sequence[MatrixCell],
    card: CampaignResourceCard,
    execution_mode: str,
    formal_qualification_receipt_sha256: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], bool]:
    manifest_path = output_root / MANIFEST_FILENAME
    progress_path = output_root / PROGRESS_FILENAME
    allowed_root_names = {
        BUNDLE_DIRECTORY,
        MANIFEST_FILENAME,
        PROGRESS_FILENAME,
    }
    unexpected_root = sorted(
        path.name for path in output_root.iterdir() if path.name not in allowed_root_names
    )
    if unexpected_root:
        raise PolicyMatrixError(
            "resume found unexpected output-root members: "
            + ", ".join(unexpected_root)
        )
    expected_bundle_names = {f"{cell.cell_id}.json" for cell in schedule}
    bundle_root = output_root / BUNDLE_DIRECTORY
    if bundle_root.is_dir():
        invalid_bundle_members = sorted(
            path.name
            for path in bundle_root.iterdir()
            if not path.is_file()
            or path.suffix != ".json"
            or path.name not in expected_bundle_names
        )
        if invalid_bundle_members:
            raise PolicyMatrixError(
                "resume found unexpected bundle members: "
                + ", ".join(invalid_bundle_members)
            )
    if manifest_path.is_file():
        manifest = _read_json_object(manifest_path, label="completed matrix manifest")
        errors = _validate_matrix_identity(
            manifest,
            protocol=protocol,
            sources=sources,
            dependencies=dependencies,
            schedule=schedule,
            card=card,
            execution_mode=execution_mode,
            formal_qualification_receipt_sha256=(
                formal_qualification_receipt_sha256
            ),
        )
        if manifest.get("schema_id") != MANIFEST_SCHEMA_ID:
            errors.append("completed manifest schema mismatch")
        if manifest.get("status") != "complete" or manifest.get("immutable") is not True:
            errors.append("completed manifest is not immutable-complete")
        if manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256"):
            errors.append("completed manifest self-hash mismatch")
        manifest_entries_raw = manifest.get("cells")
        if not isinstance(manifest_entries_raw, list) or len(manifest_entries_raw) != len(
            schedule
        ):
            errors.append("completed manifest does not contain the full schedule")
            typed_entries: list[dict[str, Any]] = []
        else:
            typed_entries = [
                dict(entry)
                for entry in manifest_entries_raw
                if isinstance(entry, Mapping)
            ]
            if len(typed_entries) != len(manifest_entries_raw):
                errors.append("completed manifest cell entry is not an object")
        if errors:
            raise PolicyMatrixError("; ".join(errors))
        completed_bundles = {
            cell.cell_id: _load_and_validate_bundle(
                output_root,
                entry,
                cell,
                protocol_sha256=matrix_protocol_sha256(protocol),
                source_manifest_sha256=semantic_sha256(sources),
                dependencies=dependencies,
                card_sha256=card.card_sha256,
            )
            for cell, entry in zip(schedule, typed_entries, strict=True)
        }
        expected_manifest = _manifest_payload(
            common=_common_matrix_payload(
                protocol=protocol,
                sources=sources,
                dependencies=dependencies,
                schedule=schedule,
                card=card,
                execution_mode=execution_mode,
                formal_qualification_receipt_sha256=(
                    formal_qualification_receipt_sha256
                ),
                entries=typed_entries,
            ),
            bundles=completed_bundles,
            schedule=schedule,
        )
        if manifest != expected_manifest:
            raise PolicyMatrixError("completed matrix manifest does not rebuild")
        return typed_entries, completed_bundles, True

    entries: list[dict[str, Any]] = []
    if progress_path.is_file():
        progress = _read_json_object(progress_path, label="matrix progress")
        errors = _validate_matrix_identity(
            progress,
            protocol=protocol,
            sources=sources,
            dependencies=dependencies,
            schedule=schedule,
            card=card,
            execution_mode=execution_mode,
            formal_qualification_receipt_sha256=(
                formal_qualification_receipt_sha256
            ),
        )
        if progress.get("schema_id") != PROGRESS_SCHEMA_ID:
            errors.append("progress schema mismatch")
        if progress.get("status") != "in_progress":
            errors.append("progress status mismatch")
        if progress.get("progress_sha256") != _self_hash(progress, "progress_sha256"):
            errors.append("progress self-hash mismatch")
        raw_entries = progress.get("cells")
        if not isinstance(raw_entries, list):
            errors.append("progress cells must be a list")
        else:
            entries = [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
            if len(entries) != len(raw_entries):
                errors.append("progress cell entry is not an object")
        if errors:
            raise PolicyMatrixError("; ".join(errors))

    if len(entries) > len(schedule):
        raise PolicyMatrixError("resume prefix is longer than the frozen schedule")
    bundles: dict[str, dict[str, Any]] = {}
    for cell, entry in zip(schedule, entries, strict=False):
        if entry.get("cell_id") != cell.cell_id or entry.get("ordinal") != cell.ordinal:
            raise PolicyMatrixError("resume entries are not the canonical schedule prefix")
        bundles[cell.cell_id] = _load_and_validate_bundle(
            output_root,
            entry,
            cell,
            protocol_sha256=matrix_protocol_sha256(protocol),
            source_manifest_sha256=semantic_sha256(sources),
            dependencies=dependencies,
            card_sha256=card.card_sha256,
        )

    present_members = sorted(bundle_root.iterdir()) if bundle_root.is_dir() else []
    invalid_members = [path.name for path in present_members if not path.is_file()]
    invalid_members.extend(
        path.name for path in present_members if path.is_file() and path.suffix != ".json"
    )
    if invalid_members:
        raise PolicyMatrixError(
            "resume found unexpected bundle members: "
            + ", ".join(sorted(invalid_members))
        )
    present = [path for path in present_members if path.is_file()]
    unexpected = sorted(
        path.name for path in present if path.name not in expected_bundle_names
    )
    if unexpected:
        raise PolicyMatrixError("resume found unexpected bundles: " + ", ".join(unexpected))
    bound_names = {f"{cell.cell_id}.json" for cell in schedule[: len(entries)]}
    extra = [path for path in present if path.name not in bound_names]
    if extra:
        if len(extra) != 1 or len(entries) >= len(schedule):
            raise PolicyMatrixError("resume found non-prefix or multiple unbound bundles")
        next_cell = schedule[len(entries)]
        if extra[0].name != f"{next_cell.cell_id}.json":
            raise PolicyMatrixError("resume found an unbound bundle outside the next cell")
        bundle = _read_json_object(extra[0], label=f"orphan bundle {next_cell.cell_id}")
        errors = validate_cell_bundle(
            bundle,
            cell=next_cell,
            protocol_sha256=matrix_protocol_sha256(protocol),
            source_manifest_sha256=semantic_sha256(sources),
            dependency_identity=dependencies,
            card_sha256=card.card_sha256,
        )
        if errors:
            raise PolicyMatrixError(
                f"unbound bundle validation failed for {next_cell.cell_id}: "
                + "; ".join(errors)
            )
        entries.append(_bundle_entry(extra[0], bundle, next_cell))
        bundles[next_cell.cell_id] = bundle
        common = _common_matrix_payload(
            protocol=protocol,
            sources=sources,
            dependencies=dependencies,
            schedule=schedule,
            card=card,
            execution_mode=execution_mode,
            formal_qualification_receipt_sha256=(
                formal_qualification_receipt_sha256
            ),
            entries=entries,
        )
        write_json_atomic(progress_path, _progress_payload(common=common))
    return entries, bundles, False


def run_matrix(
    *,
    root: Path,
    protocol_path: Path,
    output_root: Path,
    executor: CellExecutor,
    resume: bool,
    execution_mode: str = "injected_test",
    allow_formal_execution: bool = False,
    formal_qualification_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume the immutable schedule through an injected executor.

    Formal execution requires explicit opt-in plus a current, self-hashed W1-V07
    qualification receipt. W1-V05 tests use an injected synthetic executor.
    """

    if execution_mode not in {"injected_test", "formal"}:
        raise PolicyMatrixError("execution_mode must be injected_test or formal")
    protocol = load_matrix_protocol(protocol_path)
    apparatus = assert_execution_apparatus(root, protocol)
    sources = source_manifest(root, protocol)
    dependencies = dependency_bindings(
        root, protocol, execution_apparatus=apparatus
    )
    formal_receipt_sha256: str | None = None
    if execution_mode == "formal":
        if not allow_formal_execution:
            raise PolicyMatrixError("formal matrix execution requires explicit authorization")
        if dependencies["controller"]["status"] != "available":
            raise PolicyMatrixError("formal matrix execution requires merged W1-V04")
        if formal_qualification_receipt is None:
            raise PolicyMatrixError(
                "formal matrix execution requires a W1-V07 qualification receipt"
            )
        current_preflight = build_preflight(root, protocol_path)
        receipt_errors = validate_formal_qualification_receipt(
            formal_qualification_receipt,
            preflight=current_preflight,
        )
        if receipt_errors:
            raise PolicyMatrixError("; ".join(receipt_errors))
        formal_receipt_sha256 = str(
            formal_qualification_receipt["receipt_sha256"]
        )
    elif formal_qualification_receipt is not None:
        raise PolicyMatrixError(
            "formal qualification receipt is valid only for formal execution"
        )
    schedule = canonical_schedule(protocol)
    card = campaign_resource_card(protocol)
    resolved_output = output_root.resolve()
    if resume and not resolved_output.exists():
        raise PolicyMatrixError("resume output root does not exist")
    if resolved_output.exists() and not resume:
        if any(resolved_output.iterdir()):
            raise PolicyMatrixError("refusing to overwrite a non-empty output root")
    else:
        resolved_output.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}
    complete = False
    if resume:
        entries, bundles, complete = _load_resume_prefix(
            resolved_output,
            protocol=protocol,
            sources=sources,
            dependencies=dependencies,
            schedule=schedule,
            card=card,
            execution_mode=execution_mode,
            formal_qualification_receipt_sha256=formal_receipt_sha256,
        )
    if complete:
        return _read_json_object(
            resolved_output / MANIFEST_FILENAME,
            label="completed matrix manifest",
        )

    bundle_root = resolved_output / BUNDLE_DIRECTORY
    bundle_root.mkdir(parents=True, exist_ok=True)
    for cell in schedule[len(entries) :]:
        original, retest = executor(cell, protocol)
        bundle = build_cell_bundle(
            cell=cell,
            protocol_sha256=matrix_protocol_sha256(protocol),
            source_manifest_sha256=semantic_sha256(sources),
            dependency_identity=dependencies,
            card_sha256=card.card_sha256,
            original=original,
            retest=retest,
        )
        bundle_path = resolved_output / _bundle_relative_path(cell)
        if bundle_path.exists():
            raise PolicyMatrixError(f"refusing to overwrite accepted bundle: {cell.cell_id}")
        write_json_atomic(bundle_path, bundle)
        entry = _bundle_entry(bundle_path, bundle, cell)
        entries.append(entry)
        bundles[cell.cell_id] = bundle
        common = _common_matrix_payload(
            protocol=protocol,
            sources=sources,
            dependencies=dependencies,
            schedule=schedule,
            card=card,
            execution_mode=execution_mode,
            formal_qualification_receipt_sha256=formal_receipt_sha256,
            entries=entries,
        )
        write_json_atomic(
            resolved_output / PROGRESS_FILENAME,
            _progress_payload(common=common),
        )

    common = _common_matrix_payload(
        protocol=protocol,
        sources=sources,
        dependencies=dependencies,
        schedule=schedule,
        card=card,
        execution_mode=execution_mode,
        formal_qualification_receipt_sha256=formal_receipt_sha256,
        entries=entries,
    )
    manifest = _manifest_payload(common=common, bundles=bundles, schedule=schedule)
    if not manifest["all_matched_arm_audits_passed"]:
        raise PolicyMatrixError("matched information-arm invariance gate failed")
    write_json_atomic(resolved_output / MANIFEST_FILENAME, manifest)
    return manifest


def empty_profile_axes() -> dict[str, dict[str, float | None]]:
    """Return a complete metric container for producer-side profile builders."""

    axes: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        axes.setdefault(metric.axis_id, {})[metric.metric_id] = None
    return axes


def profile_record_identity(
    *,
    campaign_id: str,
    world_id: str,
    information_arm: str,
    policy_id: str,
    resource_card_sha256: str,
    trajectory_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the required V01 identity surface for injected/live executors."""

    return {
        "schema_id": PROFILE_SCHEMA_ID,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "contract_sha256": profile_contract_sha256(),
        "identity": {
            "campaign_id": campaign_id,
            "world_id": world_id,
            "information_arm": information_arm,
            "policy_id": policy_id,
            "resource_card_sha256": resource_card_sha256,
            "trajectory_manifest_sha256": semantic_sha256(trajectory_records),
        },
    }


def _trajectory_profile(scores: Sequence[float]) -> dict[str, float | None]:
    if not scores:
        return {
            "global_best_discovery_fraction": None,
            "online_incumbent_retention_rate": None,
            "maximum_absolute_incumbent_drawdown": None,
            "loss_episode_recovery_rate": None,
            "terminal_to_global_best_ratio": None,
        }
    best = max(scores)
    first_best = next(index for index, score in enumerate(scores) if score == best)
    discovery = first_best / (len(scores) - 1) if len(scores) > 1 else 0.0
    incumbent = scores[0]
    retained = 0
    drawdowns: list[float] = []
    loss_episodes = 0
    recovered_episodes = 0
    open_recovery_threshold: float | None = None
    for score in scores[1:]:
        pre_incumbent = incumbent
        threshold = 0.9 * pre_incumbent
        if score + 1e-12 >= threshold:
            retained += 1
        drawdowns.append(max(0.0, pre_incumbent - score))
        if open_recovery_threshold is not None and score + 1e-12 >= open_recovery_threshold:
            recovered_episodes += 1
            open_recovery_threshold = None
        elif open_recovery_threshold is None and score + 1e-12 < threshold:
            loss_episodes += 1
            open_recovery_threshold = threshold
        incumbent = max(incumbent, score)
    return {
        "global_best_discovery_fraction": discovery,
        "online_incumbent_retention_rate": (
            retained / (len(scores) - 1) if len(scores) > 1 else None
        ),
        "maximum_absolute_incumbent_drawdown": (
            max(drawdowns) if drawdowns else None
        ),
        "loss_episode_recovery_rate": (
            recovered_episodes / loss_episodes if loss_episodes else None
        ),
        "terminal_to_global_best_ratio": (
            scores[-1] / best if best > 0.0 else None
        ),
    }


def build_profile_record_from_execution(
    *,
    cell: MatrixCell,
    world_id: str,
    resource_card_sha256: str,
    trajectory_records: Sequence[Mapping[str, Any]],
    campaign_resource_ledger: Mapping[str, Any],
    lifecycle_terminals: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Construct the producer-side V01 profile from immutable raw evidence."""

    grouped: dict[int, list[Mapping[str, Any]]] = {
        index: [] for index in range(LIFECYCLES_PER_CELL)
    }
    for record in trajectory_records:
        lifecycle_index = record.get("lifecycle_index")
        if lifecycle_index not in grouped:
            raise PolicyMatrixError("trajectory lifecycle index is outside 0..5")
        grouped[int(lifecycle_index)].append(record)

    assays = sum(
        terminal.get("terminal_kind") == "assay" for terminal in lifecycle_terminals
    )
    discards = sum(
        terminal.get("terminal_kind") == "discard" for terminal in lifecycle_terminals
    )
    measured_lifecycles = 0
    eligible_lifecycles = 0
    nonfinal_measurements = 0
    first_measurement_fractions: list[float] = []
    continued_lifecycles = 0
    post_measure_operations = 0
    concordant = 0
    process_operations = {
        "add_solvent",
        "add_reagent",
        "set_potential",
        "electrolyze",
    }
    for lifecycle_index, records in grouped.items():
        terminal = lifecycle_terminals[lifecycle_index]
        measurement_indices = [
            index
            for index, record in enumerate(records)
            if record.get("info", {}).get("transaction_status") == "committed"
            and record.get("action", {}).get("operation") == "measure"
            and record.get("action", {}).get("instrument") != "final_assay"
        ]
        nonfinal_measurements += len(measurement_indices)
        if not measurement_indices:
            continue
        measured_lifecycles += 1
        first_index = measurement_indices[0]
        first_measurement_fractions.append(first_index / len(records))
        post_measure = [
            record
            for record in records[first_index + 1 : -1]
            if record.get("info", {}).get("transaction_status") == "committed"
            and record.get("action", {}).get("operation") in process_operations
        ]
        post_measure_operations += len(post_measure)
        continued_lifecycles += bool(post_measure)
        diagnostic = records[first_index].get("observation", {}).get("conversion")
        value = _finite_number(diagnostic)
        if value is not None:
            eligible_lifecycles += 1
            expected = "assay" if value >= threshold else "discard"
            concordant += terminal.get("terminal_kind") == expected

    closed = assays + discards
    attempted = len(trajectory_records)
    committed = sum(
        record.get("info", {}).get("transaction_status") == "committed"
        for record in trajectory_records
    )
    state = campaign_resource_ledger.get("state")
    if not isinstance(state, Mapping):
        raise PolicyMatrixError("campaign ledger state is missing")
    report_only = state.get("report_only")
    report_values = report_only if isinstance(report_only, Mapping) else {}
    scores = [
        float(terminal["terminal_score"])
        for terminal in lifecycle_terminals
        if terminal.get("terminal_kind") == "assay"
    ]
    trajectory = _trajectory_profile(scores)
    axes = empty_profile_axes()
    axes["terminal_commitment"].update(
        {
            "closed_lifecycle_fraction": closed / LIFECYCLES_PER_CELL,
            "assay_fraction": assays / closed if closed else None,
            "discard_fraction": discards / closed if closed else None,
        }
    )
    axes["evidence_acquisition"].update(
        {
            "measured_lifecycle_fraction": (
                measured_lifecycles / closed if closed else None
            ),
            "nonfinal_instrument_uses_per_closed_lifecycle": (
                nonfinal_measurements / closed if closed else None
            ),
            "mean_first_measurement_operation_fraction": (
                sum(first_measurement_fractions) / len(first_measurement_fractions)
                if first_measurement_fractions
                else None
            ),
        }
    )
    axes["evidence_conditioned_action"].update(
        {
            "continued_after_measurement_fraction": (
                continued_lifecycles / closed if closed else None
            ),
            "post_measure_process_operations_per_closed_lifecycle": (
                post_measure_operations / closed if closed else None
            ),
            "threshold_eligible_fraction": (
                eligible_lifecycles / closed if closed else None
            ),
            "threshold_decision_concordance": (
                concordant / eligible_lifecycles if eligible_lifecycles else None
            ),
        }
    )
    axes["resource_deployment"].update(
        {
            "attempted_operations_per_closed_lifecycle": (
                attempted / closed if closed else None
            ),
            "committed_operations_per_closed_lifecycle": (
                committed / closed if closed else None
            ),
            "total_cost_per_closed_lifecycle": (
                float(report_values.get("physical_cost", 0.0)) / closed
                if closed
                else None
            ),
            "total_risk_per_closed_lifecycle": (
                float(report_values.get("accumulated_risk", 0.0)) / closed
                if closed
                else None
            ),
        }
    )
    axes["outcome_trajectory"].update(trajectory)
    profile = profile_record_identity(
        campaign_id=cell.cell_id,
        world_id=world_id,
        information_arm=cell.information_arm,
        policy_id=cell.policy_id,
        resource_card_sha256=resource_card_sha256,
        trajectory_records=trajectory_records,
    )
    profile.update(
        {
            "counts": {
                "planned_lifecycle_count": LIFECYCLES_PER_CELL,
                "closed_lifecycle_count": closed,
                "final_assay_count": assays,
                "discard_count": discards,
                "measured_lifecycle_count": measured_lifecycles,
                "threshold_eligible_lifecycle_count": eligible_lifecycles,
            },
            "construct_axes": axes,
            "endpoint_context": {
                "mean_assayed_score": sum(scores) / len(scores) if scores else None,
                "best_assayed_score": max(scores) if scores else None,
            },
            "reliability": {
                "trajectory_exact_replay_match": True,
                "profile_exact_rebuild_match": True,
                "provider_call_count": 0,
            },
        }
    )
    errors = validate_profile_record(profile)
    if errors:
        raise PolicyMatrixError("producer profile is invalid: " + "; ".join(errors))
    return profile


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
    return {field: deepcopy(provenance.get(field)) for field in fields}


_STABLE_STEP_INFO_FIELDS = (
    "step",
    "budget",
    "remaining_budget",
    "episode_mode",
    "experiment_index",
    "experiment_ended",
    "task_id",
    "scenario_id",
    "initial_state_id",
    "world_law_id",
    "world_split",
    "objective",
    "operation_type",
    "instrument",
    "instrument_source",
    "preconditions",
    "state_delta_summary",
    "observed_keys",
    "observed_mask",
    "measurement_cost",
    "sample_consumed",
    "leaderboard_score",
    "reward_source",
    "cost",
    "cost_components",
    "error_message",
    "constraint_flags",
    "kernel_id",
    "kernel_version",
    "affected_ledgers",
    "state_patches_summary",
    "cost_delta",
    "risk_delta",
    "sample_delta",
    "transaction_status",
    "rollback_reason",
    "campaign_resource_outcome_delta",
    "environment_reward",
)


def _stable_step_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Project step metadata onto replay-stable evaluator evidence.

    Environment campaign and operation UUIDs deliberately identify a live
    process, not a deterministic scientific trajectory.  They are therefore
    excluded from content identity; the matrix cell and event ordinal provide
    the immutable audit coordinates instead.
    """

    return {
        field: deepcopy(info[field])
        for field in _STABLE_STEP_INFO_FIELDS
        if field in info
    }


def _canonical_resource_ledger(
    snapshot: Mapping[str, Any], *, cell: MatrixCell
) -> dict[str, Any]:
    """Replace live campaign UUID-derived receipts with deterministic IDs.

    The transformed snapshot remains a fully replayable
    :class:`CampaignResourceLedger`; only receipt names change.  Using a
    world-policy logical campaign identity also makes the two information arms
    share resource evidence when their action paths are identical.
    """

    payload = deepcopy(dict(snapshot))
    events = payload.get("events")
    if not isinstance(events, list):
        raise PolicyMatrixError("campaign resource ledger events are missing")
    logical_campaign_id = (
        f"work-i-known-policy-world-{cell.world_seed:04d}-{cell.policy_id}"
    )
    last_event_id: str | None = None
    for ordinal, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, dict):
            raise PolicyMatrixError("campaign resource ledger event is invalid")
        event_id = campaign_resource_event_id(logical_campaign_id, ordinal)
        raw_event["event_id"] = event_id
        preflight = raw_event.get("preflight")
        if not isinstance(preflight, dict):
            raise PolicyMatrixError("campaign resource preflight receipt is missing")
        preflight["event_id"] = event_id
        last_event_id = event_id
    payload["last_event_id"] = last_event_id
    payload.pop("ledger_sha256", None)
    payload["ledger_sha256"] = canonical_json_sha256(payload)
    try:
        rebuilt = CampaignResourceLedger.from_snapshot(payload).snapshot()
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyMatrixError(
            "canonical campaign resource ledger is not exactly replayable"
        ) from exc
    if rebuilt != payload:
        raise PolicyMatrixError("canonical campaign resource ledger changed on replay")
    return payload


def execute_known_policy_campaign(
    cell: MatrixCell,
    protocol: Mapping[str, Any],
    *,
    execution_role: str,
) -> dict[str, Any]:
    """Execute one live V04 controller campaign and retain complete V06 inputs."""

    if execution_role not in {"original", "retest"}:
        raise PolicyMatrixError("execution_role must be original or retest")
    import gymnasium as gym

    import chemworld  # noqa: F401
    from chemworld.agents.base import HistoryRecord
    from chemworld.agents.known_policy import make_known_policy_agent
    from chemworld.data.logging import observation_to_json, to_builtin

    task = protocol["task"]
    card = campaign_resource_card(protocol)
    namespace = str(task["observation_noise_namespace_template"]).format(
        world_seed=cell.world_seed
    )
    env = gym.make(
        str(task["env_id"]),
        task_id=str(task["task_id"]),
        world_split=str(task["world_split"]),
        objective=str(task["objective"]),
        seed=cell.world_seed,
        budget=int(task["budget_override"]),
        budget_override=int(task["budget_override"]),
        episode_mode_override=str(task["episode_mode"]),
        electrochemical_workflow_mode=str(task["workflow_mode"]),
        electrochemical_material_family_id=str(task["material_family_id"]),
        material_information=deepcopy(cell.material_information),
        observation_seed_override=(
            cell.world_seed + int(task["observation_seed_offset"])
        ),
        observation_noise_mode=str(task["observation_noise_mode"]),
        observation_noise_namespace=namespace,
        campaign_resource_card=card,
    )
    history: list[HistoryRecord] = []
    records: list[dict[str, Any]] = []
    decision_audits: list[dict[str, Any]] = []
    terminals: list[dict[str, Any]] = []
    lifecycle_index = 0
    try:
        env.reset(seed=cell.world_seed)
        base_env: Any = env.unwrapped
        task_info = base_env.task_info()
        agent = make_known_policy_agent(cell.policy_id)
        agent.reset(task_info, int(task["agent_seed"]))
        controller_manifest = deepcopy(agent.manifest())
        for event_index in range(1, int(task["budget_override"]) + 1):
            action = agent.act(history)
            decision_audit = deepcopy(agent.decision_audit() or {})
            observation, reward, terminated, truncated, info = env.step(action)
            observation_json = observation_to_json(observation)
            raw_info_json = to_builtin(info)
            if raw_info_json.get("transaction_status") != "committed":
                raise PolicyMatrixError(
                    f"known-policy action did not commit: {cell.cell_id} event {event_index}"
                )
            info_json = _stable_step_info(raw_info_json)
            agent.update(action, observation_json, float(reward), info)
            event_type = "operation_result"
            if info.get("experiment_ended"):
                terminal_kind = (
                    "discard"
                    if action.get("operation") == "discard_batch"
                    else "assay"
                )
                terminal_score = (
                    None
                    if terminal_kind == "discard"
                    else _finite_number(info.get("leaderboard_score"))
                )
                if terminal_kind == "assay" and terminal_score is None:
                    raise PolicyMatrixError(
                        f"final assay lacks a finite score: {cell.cell_id} event {event_index}"
                    )
                terminals.append(
                    {
                        "lifecycle_index": lifecycle_index,
                        "terminal_kind": terminal_kind,
                        "terminal_score": terminal_score,
                        "terminal_event_index": event_index,
                    }
                )
                event_type = "batch_discard" if terminal_kind == "discard" else "experiment_end"
            record = {
                "event_index": event_index,
                "lifecycle_index": lifecycle_index,
                "action": to_builtin(action),
                "observation": observation_json,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "event_type": event_type,
                "info": info_json,
                "state": to_builtin(base_env._state.to_dict(include_hidden=True)),
                "campaign_resource_state": deepcopy(
                    raw_info_json["campaign_resources"]["state"]
                ),
                "decision_audit": decision_audit,
            }
            record["observation_sha256"] = semantic_sha256(record["observation"])
            record["state_sha256"] = semantic_sha256(record["state"])
            record["campaign_resource_state_sha256"] = semantic_sha256(
                record["campaign_resource_state"]
            )
            records.append(record)
            decision_audits.append(decision_audit)
            history.append(
                HistoryRecord(
                    step=event_index,
                    action=deepcopy(dict(action)),
                    observation=deepcopy(observation_json),
                    reward=float(reward),
                    info=deepcopy(dict(info_json)),
                    decision_audit=deepcopy(decision_audit),
                    event_type=event_type,
                )
            )
            if info.get("experiment_ended"):
                lifecycle_index += 1
            if lifecycle_index == LIFECYCLES_PER_CELL:
                break
        if lifecycle_index != LIFECYCLES_PER_CELL:
            raise PolicyMatrixError(
                f"live controller did not close six lifecycles: {cell.cell_id}"
            )
        raw_ledger = base_env.campaign_resource_snapshot()
        if not isinstance(raw_ledger, Mapping):
            raise PolicyMatrixError("live execution did not expose a resource ledger")
        ledger = _canonical_resource_ledger(raw_ledger, cell=cell)
        provenance = base_env.evaluator_provenance()
        world_id = str(provenance["world_id"])
        profile = build_profile_record_from_execution(
            cell=cell,
            world_id=world_id,
            resource_card_sha256=card.card_sha256,
            trajectory_records=records,
            campaign_resource_ledger=ledger,
            lifecycle_terminals=terminals,
            threshold=float(agent.artifacts.threshold),
        )
        counts = profile["counts"]
        usage = agent.method_resource_usage()
        if usage.get("model_call_count") != 0:
            raise PolicyMatrixError("known-policy controller made provider calls")
        return {
            "schema_id": EXECUTION_SCHEMA_ID,
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "execution_role": execution_role,
            "identity": {
                "campaign_id": cell.cell_id,
                "cell_id": cell.cell_id,
                "world_seed": cell.world_seed,
                "world_id": world_id,
                "information_arm": cell.information_arm,
                "policy_id": cell.policy_id,
                "resource_card_sha256": card.card_sha256,
                "observation_noise_namespace": namespace,
                "physical_identity": _physical_identity(provenance),
                "material_information_sha256": semantic_sha256(
                    cell.material_information
                ),
            },
            "controller_manifest": controller_manifest,
            "trajectory_records": records,
            "campaign_resource_ledger": deepcopy(dict(ledger)),
            "lifecycle_terminals": terminals,
            "profile_record": profile,
            "decision_audits": decision_audits,
            "counts": {
                **counts,
                "attempted_operation_count": len(records),
                "committed_operation_count": sum(
                    record["info"].get("transaction_status") == "committed"
                    for record in records
                ),
                "provider_call_count": 0,
            },
        }
    finally:
        env.close()


def known_policy_cell_executor(
    cell: MatrixCell,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the frozen controller twice under one physical/noise identity."""

    original = execute_known_policy_campaign(cell, protocol, execution_role="original")
    retest = execute_known_policy_campaign(cell, protocol, execution_role="retest")
    return original, retest


__all__ = [
    "BUNDLE_DIRECTORY",
    "CELL_BUNDLE_SCHEMA_ID",
    "CELL_BUNDLE_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_ID",
    "EXECUTION_SCHEMA_VERSION",
    "FORMAL_QUALIFICATION_RECEIPT_SCHEMA_ID",
    "FORMAL_QUALIFICATION_RECEIPT_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_ID",
    "MANIFEST_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_ID",
    "PREFLIGHT_SCHEMA_VERSION",
    "PRIMARY_CAMPAIGN_COUNT",
    "PRIMARY_LIFECYCLE_COUNT",
    "PROGRESS_FILENAME",
    "PROGRESS_SCHEMA_ID",
    "PROGRESS_SCHEMA_VERSION",
    "PROTOCOL_SCHEMA_ID",
    "PROTOCOL_SCHEMA_VERSION",
    "RUNNER_VERSION",
    "MatrixCell",
    "PolicyMatrixError",
    "assert_execution_apparatus",
    "build_cell_bundle",
    "build_preflight",
    "build_profile_record_from_execution",
    "campaign_resource_card",
    "canonical_schedule",
    "dependency_bindings",
    "empty_profile_axes",
    "execute_known_policy_campaign",
    "execution_apparatus_sha256",
    "execution_component_hashes",
    "finalize_execution_record",
    "formal_qualification_receipt_sha256",
    "known_policy_cell_executor",
    "load_formal_qualification_receipt",
    "load_matrix_protocol",
    "matrix_protocol_sha256",
    "observed_execution_apparatus",
    "profile_record_identity",
    "run_matrix",
    "semantic_sha256",
    "source_manifest",
    "validate_cell_bundle",
    "validate_execution_apparatus",
    "validate_execution_record",
    "validate_formal_qualification_receipt",
    "validate_matrix_protocol",
    "validate_preflight",
]
