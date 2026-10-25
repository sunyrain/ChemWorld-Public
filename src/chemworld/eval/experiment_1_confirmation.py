"""Sealed Experiment 1 process-isolated confirmation preparation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_ae_prior_qualification_v02 import build_blind_policy_schedule
from chemworld.eval.work_ii_structural_candidate_qualification import candidate_specs

CONTRACT_SCHEMA = "chemworld-experiment-1-confirmation-contract-1.0"
PREFLIGHT_SCHEMA = "chemworld-experiment-1-confirmation-preflight-1.0"
PLAN_SCHEMA = "chemworld-experiment-1-confirmation-secret-plan-1.0"
GENERATOR_VERSION = "chemworld-experiment-1-confirmation-coordinate-generator-1.0"
NOISE_VERSION = "chemworld-experiment-1-confirmation-keyed-noise-1.0"
ELIGIBLE_LOCI = ("EC-E", "EC-P", "EC-S", "RX-P", "PA-E", "C-E", "C-P")
EXCLUDED_LOCI = ("RX-S", "PA-P", "PA-S")
WORLDS = tuple(f"W{index:02d}" for index in range(1, 6))
GATES = tuple(
    f"Q{index}_{name}"
    for index, name in enumerate(
        (
            "world_integrity",
            "task_accessibility",
            "public_contract_invariance",
            "prior_symmetry",
            "identifiability",
            "budgeted_falsifiability",
            "behavioral_relevance",
            "noise_robustness",
        ),
        1,
    )
)

SOURCE_CONTRACTS = {
    "EC-E": "configs/benchmark/experiment_1_ec_qualification_repair_v1.0.2.json",
    "EC-P": "configs/benchmark/experiment_1_ec_qualification_v1.0.1.json",
    "EC-S": "configs/benchmark/experiment_1_ec_qualification_repair_v1.0.2.json",
    "RX-P": "configs/benchmark/experiment_1_rx_qualification_v1.0.1.json",
    "PA-E": "configs/benchmark/experiment_1_pa_qualification_v1.0.1.json",
    "C-E": "configs/benchmark/experiment_1_c_qualification_v1.0.1.json",
    "C-P": "configs/benchmark/experiment_1_c_parametric_repair_v1.0.2.json",
}

PUBLIC_GENERATOR_METADATA: dict[str, Any] = {
    "generator_version": GENERATOR_VERSION,
    "outcome_blind": True,
    "same_rule_across_five_worlds": True,
    "development_coordinate_overlap_forbidden": True,
    "minimum_outside_development_regions": 2,
    "loci": {
        "EC-E": {
            "coordinate_family": "hash-uniform complementary nuisance anchors",
            "legal_domain": [0.15, 0.85],
            "regions": ["lower-complement", "upper-complement"],
            "target_categories": [0, 1, 2, 3],
        },
        "EC-P": {
            "coordinate_family": "jittered normalized potential-current surface",
            "legal_domain": [0.245, 0.755],
            "regions": ["below-center", "above-center"],
            "axis_points": [11, 11],
        },
        "EC-S": {
            "coordinate_family": "three-by-three potential-current surface",
            "potential_regions_V": [[0.78, 0.92], [1.08, 1.20], [1.25, 1.32]],
            "current_regions_mA": [[25.0, 55.0], [105.0, 135.0], [155.0, 180.0]],
        },
        "RX-P": {
            "coordinate_family": "interstitial temperature-duration surface",
            "temperature_regions_K": [[372.0, 418.0], [422.0, 468.0]],
            "duration_regions_s": [[450.0, 3150.0], [3450.0, 6150.0]],
            "axis_points": [11, 11],
        },
        "PA-E": {
            "coordinate_family": "held-out solvent anchors",
            "legal_solvents": [0, 1, 2, 3],
            "development_anchors": [0, 2],
            "confirmation_anchor_domain": [1, 3],
        },
        "C-E": {
            "coordinate_family": "held-out cooling anchors",
            "cooling_regions_K": [[275.0, 285.0], [295.0, 305.0]],
            "solvent_targets": [1, 3],
        },
        "C-P": {
            "coordinate_family": "nearby held-out cooling response",
            "cooling_regions_K": [[306.0, 309.0], [286.0, 289.0], [271.0, 274.0]],
            "preserved_order": "high-mid-low",
        },
    },
}

PUBLIC_NOISE_METADATA: dict[str, Any] = {
    "noise_version": NOISE_VERSION,
    "mode": "keyed",
    "namespace_prefix": "experiment-1-confirmation-v1",
    "distinct_from_development": True,
    "seed_derivation": "HMAC-SHA256(secret-salt, locus/world/coordinate/replicate)",
    "replicates": {
        "EC-E": 3,
        "EC-P": 1,
        "EC-S": {"main_grid": 1, "validation": 3},
        "RX-P": 1,
        "PA-E": 3,
        "C-E": 3,
        "C-P": 3,
    },
    "parametric_overlay": (
        "registered development validation sigma applied with a keyed normal deviate; "
        "primary and tolerance-zero replay regenerate the identical deviate"
    ),
}

EXECUTION_COUNTS = {
    "EC-E": 120,
    "EC-P": 605,
    "EC-S": 90,
    "RX-P": 605,
    "PA-E": 60,
    "C-E": 60,
    "C-P": 45,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(left))
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_source_contract(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    value = _load(path)
    extends = value.get("extends")
    if not isinstance(extends, Mapping):
        return value
    parent_path = root / str(extends.get("path", ""))
    if file_sha256(parent_path) != extends.get("sha256"):
        raise ValueError(f"source overlay parent digest mismatch: {relative}")
    return _deep_merge(_load(parent_path), dict(value.get("overrides", {})))


def _locus_name(block: str) -> str:
    return {"E": "entity", "P": "parametric", "S": "structural"}[block[-1]]


def _development_coordinates(root: Path) -> dict[str, Any]:
    resolved = {
        block: resolve_source_contract(root, path) for block, path in SOURCE_CONTRACTS.items()
    }
    structural = candidate_specs()["electrochemical_transport"]
    coordinates: dict[str, Any] = {
        "EC-E": {
            "nuisance_design": resolved["EC-E"]["loci"]["entity"]["nuisance_design"],
            "category_order_by_anchor": resolved["EC-E"]["loci"]["entity"][
                "category_order_by_anchor"
            ],
        },
        "EC-P": {
            "potential_coordinates": resolved["EC-P"]["loci"]["parametric"]["grid_coordinates"],
            "current_coordinates": resolved["EC-P"]["loci"]["parametric"]["grid_coordinates"],
        },
        "EC-S": {
            "axis_names": list(structural["axis_names"]),
            "axis_levels": [list(values) for values in structural["axis_levels"]],
            "fixed_context": structural["fixed_context"],
            "validation_groups": resolved["EC-S"]["loci"]["structural"]["validation_groups"],
        },
        "RX-P": {
            "temperature_grid_K": resolved["RX-P"]["loci"]["parametric"]["temperature_grid_K"],
            "duration_grid_s": resolved["RX-P"]["loci"]["parametric"]["duration_grid_s"],
        },
        "PA-E": {
            "solvent_anchors": resolved["PA-E"]["loci"]["entity"]["solvent_anchors"],
            "extractant_targets": resolved["PA-E"]["loci"]["entity"]["extractant_targets"],
        },
        "C-E": {
            "cooling_anchors_K": resolved["C-E"]["loci"]["entity"]["cooling_anchors_K"],
            "solvent_targets": resolved["C-E"]["loci"]["entity"]["solvent_targets"],
        },
        "C-P": {
            "temperature_levels_K": resolved["C-P"]["loci"]["parametric"]["temperature_levels_K"]
        },
    }
    return coordinates


def _hmac_digest(salt: bytes, label: str) -> bytes:
    return hmac.new(salt, label.encode("utf-8"), hashlib.sha256).digest()


def _unit(salt: bytes, label: str) -> float:
    return int.from_bytes(_hmac_digest(salt, label)[:8], "big") / float(2**64)


def _between(salt: bytes, label: str, low: float, high: float, digits: int) -> float:
    return round(low + (high - low) * _unit(salt, label), digits)


def _namespace(salt: bytes, block: str) -> str:
    token = _hmac_digest(salt, f"noise:{block}").hex()[:20]
    return f"experiment-1-confirmation-v1-{block.lower()}-{token}"


def _interstitial_axis(
    salt: bytes,
    label: str,
    development: Sequence[float],
    *,
    low: float,
    high: float,
    digits: int,
) -> list[float]:
    values: list[float] = []
    for index, value in enumerate(development):
        lower = low if index == 0 else (float(development[index - 1]) + float(value)) / 2.0
        upper = (
            high
            if index == len(development) - 1
            else (float(value) + float(development[index + 1])) / 2.0
        )
        realized = _between(salt, f"{label}:{index}", lower, upper, digits)
        if realized == float(value):
            realized = round(realized + 10 ** (-digits), digits)
        values.append(realized)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} generator produced duplicate coordinates")
    return values


def realize_secret_plan(root: Path, salt: bytes, *, source_commit: str) -> dict[str, Any]:
    if len(salt) < 32:
        raise ValueError("confirmation salt must contain at least 256 bits")
    development = _development_coordinates(root)
    ec_grid = [float(value) for value in development["EC-P"]["potential_coordinates"]]
    rx_temperatures = [float(value) for value in development["RX-P"]["temperature_grid_K"]]
    rx_durations = [float(value) for value in development["RX-P"]["duration_grid_s"]]
    ec_entity_namespace = f"confirmation-coordinate-{_hmac_digest(salt, 'coord:EC-E').hex()[:24]}"
    ec_entity_policy = {
        "nuisance_design": {
            **development["EC-E"]["nuisance_design"],
            "namespace": ec_entity_namespace,
        },
        "category_order_by_anchor": development["EC-E"]["category_order_by_anchor"],
    }
    schedule = build_blind_policy_schedule(
        task_id="electrochemical-conversion",
        target_field="electrolyte_profile",
        policy=ec_entity_policy,
    )
    loci: dict[str, Any] = {
        "EC-E": {
            "nuisance_namespace": ec_entity_namespace,
            "coordinate_sha256": [
                canonical_json_sha256(
                    {
                        "anchor": row["nuisance_anchor"],
                        "category": row["target_category"],
                        "recipe": row["recipe"],
                    }
                )
                for row in schedule
            ],
        },
        "EC-P": {
            "potential_coordinates": _interstitial_axis(
                salt, "EC-P:potential", ec_grid, low=0.245, high=0.755, digits=6
            ),
            "current_coordinates": _interstitial_axis(
                salt, "EC-P:current", ec_grid, low=0.245, high=0.755, digits=6
            ),
        },
        "EC-S": {
            "potential_levels_V": [
                _between(salt, f"EC-S:potential:{index}", low, high, 6)
                for index, (low, high) in enumerate(((0.78, 0.92), (1.08, 1.20), (1.25, 1.32)))
            ],
            "current_levels_mA": [
                _between(salt, f"EC-S:current:{index}", low, high, 6)
                for index, (low, high) in enumerate(((25.0, 55.0), (105.0, 135.0), (155.0, 180.0)))
            ],
        },
        "RX-P": {
            "temperature_grid_K": _interstitial_axis(
                salt, "RX-P:temperature", rx_temperatures, low=365.0, high=475.0, digits=3
            ),
            "duration_grid_s": _interstitial_axis(
                salt, "RX-P:duration", rx_durations, low=150.0, high=6450.0, digits=3
            ),
        },
        "PA-E": {"solvent_anchors": [1, 3] if _unit(salt, "PA-E:order") < 0.5 else [3, 1]},
        "C-E": {
            "cooling_anchors_K": [
                _between(salt, "C-E:low", 275.0, 285.0, 3),
                _between(salt, "C-E:high", 295.0, 305.0, 3),
            ]
        },
        "C-P": {
            "temperature_levels_K": [
                _between(salt, "C-P:high", 306.0, 309.0, 3),
                _between(salt, "C-P:mid", 286.0, 289.0, 3),
                _between(salt, "C-P:low", 271.0, 274.0, 3),
            ]
        },
    }
    for block in ELIGIBLE_LOCI:
        loci[block]["observation_noise_namespace"] = _namespace(salt, block)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "source_commit": source_commit,
        "generator_version": GENERATOR_VERSION,
        "noise_version": NOISE_VERSION,
        "eligible_loci": list(ELIGIBLE_LOCI),
        "excluded_loci": list(EXCLUDED_LOCI),
        "loci": loci,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return plan


def verify_self_hash(payload: Mapping[str, Any], key: str) -> str:
    declared = payload.get(key)
    if not isinstance(declared, str):
        raise ValueError(f"missing {key}")
    unhashed = dict(payload)
    unhashed.pop(key, None)
    if canonical_json_sha256(unhashed) != declared:
        raise ValueError(f"{key} canonical self-hash mismatch")
    return declared


def validate_secret_plan(root: Path, plan: Mapping[str, Any]) -> None:
    verify_self_hash(plan, "plan_sha256")
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected secret plan schema")
    if tuple(plan.get("eligible_loci", ())) != ELIGIBLE_LOCI:
        raise ValueError("secret plan changed the seven-locus denominator")
    if tuple(plan.get("excluded_loci", ())) != EXCLUDED_LOCI:
        raise ValueError("secret plan changed the excluded loci")
    loci = plan.get("loci")
    if not isinstance(loci, Mapping) or set(loci) != set(ELIGIBLE_LOCI):
        raise ValueError("secret plan locus content changed")
    development = _development_coordinates(root)
    for block in ELIGIBLE_LOCI:
        row = loci[block]
        if not isinstance(row, Mapping):
            raise ValueError(f"{block}: malformed secret coordinates")
        namespace = str(row.get("observation_noise_namespace", ""))
        if not namespace.startswith(f"experiment-1-confirmation-v1-{block.lower()}-"):
            raise ValueError(f"{block}: confirmation noise namespace mismatch")
    for key in ("potential_coordinates", "current_coordinates"):
        values = [float(value) for value in loci["EC-P"][key]]
        if len(values) != 11 or set(values) & {float(value) for value in development["EC-P"][key]}:
            raise ValueError(f"EC-P {key} repeats development coordinates")
    for key, dev_key in (
        ("temperature_grid_K", "temperature_grid_K"),
        ("duration_grid_s", "duration_grid_s"),
    ):
        values = [float(value) for value in loci["RX-P"][key]]
        if len(values) != 11 or set(values) & {
            float(value) for value in development["RX-P"][dev_key]
        }:
            raise ValueError(f"RX-P {key} repeats development coordinates")
    if {int(value) for value in loci["PA-E"]["solvent_anchors"]} != {1, 3}:
        raise ValueError("PA-E must use exactly the two held-out solvent anchors")
    for block, key, expected_count in (
        ("EC-S", "potential_levels_V", 3),
        ("EC-S", "current_levels_mA", 3),
        ("C-E", "cooling_anchors_K", 2),
        ("C-P", "temperature_levels_K", 3),
    ):
        values = [float(value) for value in loci[block][key]]
        if len(values) != expected_count or len(set(values)) != expected_count:
            raise ValueError(f"{block}: {key} denominator changed")


def _resolve_evidence(relative: str, roots: Sequence[Path]) -> Path:
    matches = [(root / relative).resolve() for root in roots if (root / relative).is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one development report for {relative}; found {len(matches)}"
        )
    return matches[0]


def _prior_digest(report: Mapping[str, Any], registry_row: Mapping[str, Any]) -> tuple[str, str]:
    registered = registry_row.get("prior_hashes")
    if isinstance(registered, Mapping) and registered:
        return canonical_json_sha256(registered), "convergence_registry.prior_hashes"
    prior_audit = report.get("prior_audit")
    if isinstance(prior_audit, Mapping):
        selected = {
            key: prior_audit.get(key) for key in ("aligned_sha256", "misspecified_sha256", "opaque")
        }
        return canonical_json_sha256(selected), "development_report.prior_audit"
    legacy = report.get("legacy_analysis")
    if isinstance(legacy, Mapping):
        for key in ("public_priors", "prior_arms"):
            value = legacy.get(key)
            if isinstance(value, Mapping):
                return canonical_json_sha256(value), f"development_report.legacy_analysis.{key}"
    raise ValueError(f"cannot derive public prior digest for {registry_row.get('unit_id')}")


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_public_contract(
    root: Path,
    *,
    source_commit: str,
    salt_file_sha256: str,
    realized_plan_file_sha256: str,
    realized_plan_sha256: str,
    evidence_roots: Sequence[Path],
) -> dict[str, Any]:
    registry_path = root / (
        "workstreams/flagship_tasks/experiment_1/results/EXPERIMENT_1_CONVERGENCE_REGISTRY.json"
    )
    challenge_path = root / (
        "workstreams/flagship_tasks/experiment_1/results/EXPERIMENT_1_CHALLENGE_PROBES_V1_2.json"
    )
    audit_path = root / (
        "workstreams/flagship_tasks/experiment_1/results/EXPERIMENT_1_CHALLENGE_AUDIT_V1_2.json"
    )
    registry = _load(registry_path)
    challenge = _load(challenge_path)
    audit = _load(audit_path)
    verify_self_hash(registry, "registry_sha256")
    verify_self_hash(challenge, "challenge_probe_sha256")
    verify_self_hash(audit, "audit_sha256")
    eligible = tuple(
        row["block"] for row in audit.get("loci", []) if row.get("confirmation_eligible") is True
    )
    if eligible != ELIGIBLE_LOCI:
        raise ValueError("challenge audit does not authorize exactly the frozen seven loci")
    configs = {
        block: resolve_source_contract(root, relative)
        for block, relative in SOURCE_CONTRACTS.items()
    }
    source_contracts = {
        relative: {
            "path": relative,
            "file_sha256": file_sha256(root / relative),
            "resolved_sha256": canonical_json_sha256(resolve_source_contract(root, relative)),
        }
        for relative in dict.fromkeys(SOURCE_CONTRACTS.values())
    }
    units: list[dict[str, Any]] = []
    rows = registry.get("rows")
    if not isinstance(rows, list):
        raise ValueError("convergence registry rows are missing")
    for registry_row in rows:
        if not isinstance(registry_row, Mapping):
            continue
        system_id = registry_row.get("system_id")
        locus_id = str(registry_row.get("prior_locus", ""))[:1].upper()
        block = f"{system_id}-{locus_id}"
        if block not in ELIGIBLE_LOCI:
            continue
        if registry_row.get("current_status") != "qualified-development":
            raise ValueError(f"{registry_row.get('unit_id')}: development status changed")
        evidence = registry_row["qualification_run"]["evidence"]
        report_path = _resolve_evidence(str(evidence["path"]), evidence_roots)
        if file_sha256(report_path) != evidence["sha256"]:
            raise ValueError(f"{registry_row.get('unit_id')}: development report digest mismatch")
        report = _load(report_path)
        prior_sha, prior_source = _prior_digest(report, registry_row)
        locus_name = _locus_name(block)
        locus_contract = configs[block]["loci"][locus_name]
        units.append(
            {
                "unit_id": registry_row["unit_id"],
                "block": block,
                "system_id": registry_row["system_id"],
                "prior_locus": locus_name,
                "world_id": registry_row["world_id"],
                "world_seed": registry_row["world_seed"],
                "truth_sha256": registry_row["truth_sha256"],
                "public_prior_sha256": prior_sha,
                "public_prior_digest_source": prior_source,
                "development_report": {
                    "path": evidence["path"],
                    "file_sha256": evidence["sha256"],
                    "report_sha256": evidence["report_sha256"],
                },
                "public_task_contract_sha256": canonical_json_sha256(configs[block]["task"]),
                "locus_contract_sha256": canonical_json_sha256(locus_contract),
                "participant_unique_experiment_budget": locus_contract[
                    "participant_unique_experiment_budget"
                ],
                "q_thresholds": list(configs[block]["common_gates"]),
            }
        )
    units.sort(key=lambda row: (ELIGIBLE_LOCI.index(row["block"]), row["world_seed"]))
    if len(units) != 35 or len({row["unit_id"] for row in units}) != 35:
        raise ValueError("confirmation denominator must contain exactly 35 unique units")
    development = _development_coordinates(root)
    implementation_files = (
        "src/chemworld/eval/experiment_1_confirmation.py",
        "scripts/freeze_experiment_1_confirmation.py",
        "scripts/run_experiment_1_confirmation.py",
        "scripts/run_experiment_1_confirmation_locus.py",
        "scripts/run_work_ii_structural_candidate_qualification.py",
    )
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "contract_id": "experiment-1-seven-locus-process-isolated-confirmation-v1.0",
        "status": "frozen_before_confirmation_execution",
        "formal_result": False,
        "process_isolated": True,
        "one_shot": True,
        "source_commit": source_commit,
        "eligible_loci": list(ELIGIBLE_LOCI),
        "excluded_loci": list(EXCLUDED_LOCI),
        "denominators": {
            "loci": 7,
            "worlds_per_locus": 5,
            "atomic_units": 35,
            "planned_primary_executions": sum(EXECUTION_COUNTS.values()),
            "planned_tolerance_zero_replays": sum(EXECUTION_COUNTS.values()),
            "planned_executions_by_locus": EXECUTION_COUNTS,
            "missing_row_exclusion_allowed": False,
        },
        "protocol": {
            "path": (
                "workstreams/flagship_tasks/experiment_1/EXPERIMENT_1_CONFIRMATION_PROTOCOL_V1_0.md"
            ),
            "file_sha256": file_sha256(
                root / "workstreams/flagship_tasks/experiment_1/"
                "EXPERIMENT_1_CONFIRMATION_PROTOCOL_V1_0.md"
            ),
        },
        "experiment_note": {
            "path": (
                "workstreams/flagship_tasks/experiment_1/EXPERIMENT_1_CONFIRMATION_NOTE_V1_0.md"
            ),
            "file_sha256": file_sha256(
                root
                / "workstreams/flagship_tasks/experiment_1/EXPERIMENT_1_CONFIRMATION_NOTE_V1_0.md"
            ),
        },
        "challenge": {
            "probe_path": challenge_path.relative_to(root).as_posix(),
            "probe_file_sha256": file_sha256(challenge_path),
            "challenge_probe_sha256": challenge["challenge_probe_sha256"],
            "audit_path": audit_path.relative_to(root).as_posix(),
            "audit_file_sha256": file_sha256(audit_path),
            "audit_sha256": audit["audit_sha256"],
        },
        "convergence_registry": {
            "path": registry_path.relative_to(root).as_posix(),
            "file_sha256": file_sha256(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "source_contracts": source_contracts,
        "implementation_files": [
            {"path": path, "file_sha256": file_sha256(root / path)} for path in implementation_files
        ],
        "development_coordinate_contract": development,
        "development_coordinate_contract_sha256": canonical_json_sha256(development),
        "generator": {
            **PUBLIC_GENERATOR_METADATA,
            "secret_salt_file_sha256": salt_file_sha256,
            "realized_plan_file_sha256": realized_plan_file_sha256,
            "realized_plan_sha256": realized_plan_sha256,
        },
        "noise": PUBLIC_NOISE_METADATA,
        "units": units,
        "execution": {
            "worker_count": 1,
            "locus_process_count": 7,
            "output_overwrite_forbidden": True,
            "resume_forbidden": True,
            "all_terminal_rows_retained": True,
            "confirmation_secret_material_in_git_forbidden": True,
        },
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def validate_public_contract(root: Path, contract: Mapping[str, Any]) -> None:
    verify_self_hash(contract, "contract_sha256")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unexpected confirmation contract schema")
    if tuple(contract.get("eligible_loci", ())) != ELIGIBLE_LOCI:
        raise ValueError("confirmation contract changed the seven-locus denominator")
    if tuple(contract.get("excluded_loci", ())) != EXCLUDED_LOCI:
        raise ValueError("confirmation contract changed excluded loci")
    denominators = contract.get("denominators")
    if not isinstance(denominators, Mapping) or denominators.get("atomic_units") != 35:
        raise ValueError("confirmation contract denominator is not 35 units")
    if denominators.get("planned_primary_executions") != sum(EXECUTION_COUNTS.values()):
        raise ValueError("confirmation execution denominator changed")
    for label in ("protocol", "experiment_note"):
        binding = contract.get(label)
        if not isinstance(binding, Mapping):
            raise ValueError(f"confirmation {label} binding is missing")
        path = root / str(binding.get("path", ""))
        if not path.is_file() or file_sha256(path) != binding.get("file_sha256"):
            raise ValueError(f"confirmation {label} binding mismatch")
    challenge = contract.get("challenge")
    if not isinstance(challenge, Mapping):
        raise ValueError("confirmation challenge binding is missing")
    probe_path = root / str(challenge.get("probe_path", ""))
    audit_path = root / str(challenge.get("audit_path", ""))
    if not probe_path.is_file() or file_sha256(probe_path) != challenge.get("probe_file_sha256"):
        raise ValueError("confirmation challenge probe file binding mismatch")
    if not audit_path.is_file() or file_sha256(audit_path) != challenge.get("audit_file_sha256"):
        raise ValueError("confirmation challenge audit file binding mismatch")
    probe = _load(probe_path)
    audit = _load(audit_path)
    if verify_self_hash(probe, "challenge_probe_sha256") != challenge.get("challenge_probe_sha256"):
        raise ValueError("confirmation challenge probe self-hash mismatch")
    if verify_self_hash(audit, "audit_sha256") != challenge.get("audit_sha256"):
        raise ValueError("confirmation challenge audit self-hash mismatch")
    registry_binding = contract.get("convergence_registry")
    if not isinstance(registry_binding, Mapping):
        raise ValueError("confirmation convergence registry binding is missing")
    registry_path = root / str(registry_binding.get("path", ""))
    if not registry_path.is_file() or file_sha256(registry_path) != registry_binding.get(
        "file_sha256"
    ):
        raise ValueError("confirmation convergence registry file binding mismatch")
    registry = _load(registry_path)
    if verify_self_hash(registry, "registry_sha256") != registry_binding.get("registry_sha256"):
        raise ValueError("confirmation convergence registry self-hash mismatch")
    for binding in contract.get("implementation_files", []):
        path = root / str(binding.get("path", ""))
        if not path.is_file() or file_sha256(path) != binding.get("file_sha256"):
            raise ValueError("confirmation implementation binding mismatch")
    for binding in contract.get("source_contracts", {}).values():
        path = root / str(binding.get("path", ""))
        if not path.is_file() or file_sha256(path) != binding.get("file_sha256"):
            raise ValueError("confirmation source contract binding mismatch")
        resolved = resolve_source_contract(root, str(binding["path"]))
        if canonical_json_sha256(resolved) != binding.get("resolved_sha256"):
            raise ValueError("confirmation resolved source contract mismatch")
    units = contract.get("units", [])
    if len(units) != 35:
        raise ValueError("confirmation contract unit table is incomplete")
    identities = {
        (str(row.get("block")), str(row.get("world_id")))
        for row in units
        if isinstance(row, Mapping)
    }
    expected = {
        (block, f"{block.split('-')[0]}-{world}") for block in ELIGIBLE_LOCI for world in WORLDS
    }
    if identities != expected:
        raise ValueError("confirmation contract unit identities changed")
    for row in units:
        if not isinstance(row, Mapping):
            raise ValueError("confirmation contract contains a malformed unit")
        if not isinstance(row.get("truth_sha256"), str) or not isinstance(
            row.get("public_prior_sha256"), str
        ):
            raise ValueError("confirmation unit lacks truth or prior digest")
        if tuple(row.get("q_thresholds", ())) != GATES:
            raise ValueError("confirmation unit changed Q1-Q8 thresholds")
    development = contract.get("development_coordinate_contract")
    if not isinstance(development, Mapping) or canonical_json_sha256(development) != contract.get(
        "development_coordinate_contract_sha256"
    ):
        raise ValueError("confirmation development coordinate binding mismatch")
    generator = contract.get("generator")
    noise = contract.get("noise")
    if (
        not isinstance(generator, Mapping)
        or generator.get("generator_version") != GENERATOR_VERSION
    ):
        raise ValueError("confirmation coordinate generator version mismatch")
    if not isinstance(noise, Mapping) or noise.get("noise_version") != NOISE_VERSION:
        raise ValueError("confirmation noise version mismatch")


def build_preflight(
    root: Path,
    contract: Mapping[str, Any],
    *,
    secret_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate_public_contract(root, contract)
    resolved_secret = secret_dir.resolve()
    if resolved_secret.is_relative_to(root.resolve()):
        raise ValueError("confirmation secret directory must remain outside Git")
    if not resolved_secret.is_dir() or _mode(resolved_secret) != 0o700:
        raise ValueError("confirmation secret directory must have mode 0700")
    salt_path = resolved_secret / "salt.bin"
    plan_path = resolved_secret / "realized-plan.json"
    if not salt_path.is_file() or _mode(salt_path) != 0o600:
        raise ValueError("confirmation salt must have mode 0600")
    if not plan_path.is_file() or _mode(plan_path) != 0o600:
        raise ValueError("confirmation realized plan must have mode 0600")
    generator = contract["generator"]
    if file_sha256(salt_path) != generator["secret_salt_file_sha256"]:
        raise ValueError("confirmation salt commitment mismatch")
    if file_sha256(plan_path) != generator["realized_plan_file_sha256"]:
        raise ValueError("confirmation realized plan file commitment mismatch")
    plan = _load(plan_path)
    validate_secret_plan(root, plan)
    if plan["plan_sha256"] != generator["realized_plan_sha256"]:
        raise ValueError("confirmation realized plan self commitment mismatch")
    if plan["source_commit"] != contract["source_commit"]:
        raise ValueError("confirmation plan source commit mismatch")
    if output_root.exists():
        raise ValueError("one-shot confirmation output already exists")
    source_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{contract['source_commit']}^{{commit}}"],
            cwd=root,
            check=False,
        ).returncode
        == 0
    )
    if not source_exists:
        raise ValueError("frozen confirmation source commit is unavailable")
    preflight: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready-but-not-executed",
        "confirmation_execution_authorized": False,
        "provider_call_count": 0,
        "contract_sha256": contract["contract_sha256"],
        "source_commit": contract["source_commit"],
        "eligible_loci": list(ELIGIBLE_LOCI),
        "excluded_loci": list(EXCLUDED_LOCI),
        "atomic_units": 35,
        "planned_primary_executions": sum(EXECUTION_COUNTS.values()),
        "planned_tolerance_zero_replays": sum(EXECUTION_COUNTS.values()),
        "worker_count": 1,
        "locus_process_count": 7,
        "secret_permissions_verified": True,
        "salt_commitment_verified": True,
        "realized_plan_commitment_verified": True,
        "output_absent": True,
        "all_missing_rows_fail_closed": True,
    }
    preflight["preflight_sha256"] = canonical_json_sha256(preflight)
    return preflight


def write_secret_once(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)
