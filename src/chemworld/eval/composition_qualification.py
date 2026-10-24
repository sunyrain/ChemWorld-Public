"""Deterministic public composition-qualification runner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import gymnasium as gym

import chemworld
from chemworld.agent_interface import agent_view_bundle
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.eval.composition_qualification_design import (
    EXPECTED_GENERATED_CASE_COUNT,
    EXPECTED_PATTERN_CASE_COUNTS,
    QUALIFICATION_DESIGN_VERSION,
    UNSEEN_PATTERN_ID,
    build_generated_suites,
)
from chemworld.eval.cross_world_infrastructure_qualification import (
    recipe_cases,
    run_task_world_unit,
)
from chemworld.eval.report_io import git_output as _git
from chemworld.eval.verify import verify_records
from chemworld.foundation import equipment_settings
from chemworld.foundation.public_leakage import audit_public_payload
from chemworld.tasks import list_tasks
from chemworld.world.composition import WorldCompositionError

REPORT_SCHEMA_VERSION = "chemworld-composition-qualification-report-0.3"
QUALIFICATION_ID = "composition-qualification-v1"
EXPECTED_REFERENCE_UNIT_COUNT = 64
EXPECTED_REFERENCE_RECIPE_COUNT = 1786
EXPECTED_NEGATIVE_PROBE_COUNT = 192
EXPECTED_COMPILE_MUTANT_COUNT = 7
EXPECTED_MODULE_PROBE_COUNT = 32
EXPECTED_INTERFACE_PATH_COUNT = 7

PROTOCOL = Path("protocols/composition-qualification.md")

_REFERENCE_RESOURCE_CARD = {
    "operation_attempt_limit": 8,
    "vessel_start_limit": 2,
    "final_assay_limit": 2,
    "nonfinal_instrument_use_limit": 2,
    "stock_limits": {"reagent_mol": 0.08, "solvent_L": 0.026},
    "per_instrument_limits": {},
}

_OPERATION_COMPONENTS = {
    "add_solvent": "reaction",
    "add_reagent": "reaction",
    "add_catalyst": "reaction",
    "quench": "reaction",
    "heat": "thermal",
    "wait": "thermal",
    "add_phase": "phase",
    "add_extractant": "separation",
    "mix": "separation",
    "settle": "separation",
    "separate_phase": "separation",
    "wash": "separation",
    "dry": "separation",
    "concentrate": "separation",
    "transfer": "separation",
    "seed_crystals": "crystallization",
    "cool_crystallize": "crystallization",
    "filter_crystals": "crystallization",
    "evaporate": "distillation",
    "distill": "distillation",
    "collect_fraction": "distillation",
    "set_flow_rate": "continuous_flow",
    "run_flow": "continuous_flow",
    "set_potential": "electrochemistry",
    "electrolyze": "electrochemistry",
    "measure": "observation",
}

_MODULE_CONFIGS = {
    "reaction": {
        "pattern": "reaction-thermal-observation",
        "operation": "add_reagent",
        "field": "amount_mol",
        "low": 0.003,
        "high": 0.030,
        "metric": "total_species_mol",
        "maturity_module_id": "reaction_kinetics",
        "reference_fixture": "exact_reagent_amount_accounting",
        "reference_relation": "delta_equals_input",
        "reference_tolerance": 1.0e-12,
    },
    "thermal": {
        "pattern": "reaction-thermal-observation",
        "operation": "heat",
        "field": "target_temperature_K",
        "low": 350.0,
        "high": 390.0,
        "metric": "temperature_K",
        "maturity_module_id": "reactors",
        "reference_fixture": "thermal_energy_balance_closure",
        "reference_relation": "diagnostic_equals_zero",
        "reference_diagnostic": "energy_balance_residual_J",
        "reference_tolerance": 1.0e-8,
    },
    "phase": {
        "pattern": "phase-separation-observation",
        "operation": "add_phase",
        "field": "volume_L",
        "low": 0.010,
        "high": 0.020,
        "metric": "volume_L",
        "maturity_module_id": "phase_equilibrium",
        "reference_fixture": "exact_phase_volume_accounting",
        "reference_relation": "delta_equals_input",
        "reference_tolerance": 1.0e-12,
    },
    "separation": {
        "pattern": "phase-separation-observation",
        "operation": "mix",
        "field": "duration_s",
        "low": 60.0,
        "high": 300.0,
        "metric": "time_s",
        "maturity_module_id": "phase_equilibrium",
        "reference_fixture": "exact_operation_time_accounting",
        "reference_relation": "delta_equals_input",
        "reference_tolerance": 1.0e-12,
    },
    "crystallization": {
        "pattern": "reaction-crystallization-observation",
        "operation": "seed_crystals",
        "field": "seed_mass_g",
        "low": 0.002,
        "high": 0.010,
        "metric": "cost",
        "maturity_module_id": "crystallization",
    },
    "distillation": {
        "pattern": "reaction-distillation-observation",
        "operation": "distill",
        "field": "duration_s",
        "low": 900.0,
        "high": 2400.0,
        "metric": "time_s",
        "maturity_module_id": "distillation",
        "reference_fixture": "exact_operation_time_accounting",
        "reference_relation": "delta_equals_input",
        "reference_tolerance": 1.0e-12,
    },
    "continuous_flow": {
        "pattern": "reaction-continuous-flow-observation",
        "operation": "run_flow",
        "field": "duration_s",
        "low": 1200.0,
        "high": 3600.0,
        "metric": "time_s",
        "maturity_module_id": "continuous_flow",
        "reference_fixture": "exact_operation_time_accounting",
        "reference_relation": "delta_equals_input",
        "reference_tolerance": 1.0e-12,
    },
    "electrochemistry": {
        "pattern": "reaction-electrochemistry-observation",
        "operation": "electrolyze",
        "field": "duration_s",
        "low": 300.0,
        "high": 1800.0,
        "metric": "time_s",
        "maturity_module_id": "electrochemistry",
        "reference_fixture": "exact_operation_time_accounting",
        "reference_relation": "delta_equals_input",
        "reference_tolerance": 1.0e-12,
    },
}

_INTERFACE_PATHS = {
    "reaction--thermal": "reaction-thermal-observation",
    "reaction--phase--separation": "reaction-phase-separation-observation",
    "phase--separation": "phase-separation-observation",
    "reaction--crystallization": "reaction-crystallization-observation",
    "reaction--distillation": "reaction-distillation-observation",
    "reaction--continuous-flow": "reaction-continuous-flow-observation",
    "reaction--electrochemistry": "reaction-electrochemistry-observation",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_protocol() -> dict[str, Any]:
    return {
        "protocol_id": QUALIFICATION_ID,
        "intervention": {
            "negative_probes_per_unit": {
                "resource_exhaustion": {"resource_card": _REFERENCE_RESOURCE_CARD}
            }
        },
    }


def _leakage_findings(env: Any, payload: Any, surface: str) -> list[dict[str, Any]]:
    hidden_species = set(env.unwrapped._state.species_amounts)
    return [
        {"surface": surface, **finding.to_dict()}
        for finding in audit_public_payload(payload, hidden_species_ids=hidden_species)
    ]


def _physical_snapshot(env: Any) -> dict[str, Any]:
    state = env.unwrapped._state.to_dict(include_hidden=True)
    state.pop("ledger", None)
    state.pop("process", None)
    return state


def _ledger_snapshot(env: Any) -> dict[str, float]:
    ledger = env.unwrapped._state.ledger
    return {
        "time_s": float(ledger.time_s),
        "cost": float(ledger.cost),
        "risk": float(ledger.risk),
        "sample_volume_L": float(ledger.sample_consumed_L),
        "energy_jacket_J": float(ledger.energy_jacket_J),
        "heat_reaction_J": float(ledger.heat_reaction_J),
        "heat_loss_J": float(ledger.heat_loss_J),
    }


def _ledger_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in before}


def _summarize_constitution_checks(
    aggregate: dict[str, dict[str, Any]],
    checks: list[Any],
) -> None:
    for raw in checks:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            continue
        name = str(raw["name"])
        row = aggregate.setdefault(
            name,
            {
                "name": name,
                "observed_count": 0,
                "passed_count": 0,
                "tolerances": [],
                "max_abs_value": None,
            },
        )
        row["observed_count"] += 1
        row["passed_count"] += int(raw.get("passed") is True)
        tolerance = raw.get("tolerance")
        if isinstance(tolerance, int | float) and math.isfinite(float(tolerance)):
            row["tolerances"].append(float(tolerance))
        value = raw.get("value")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            absolute = abs(float(value))
            current = row["max_abs_value"]
            row["max_abs_value"] = absolute if current is None else max(float(current), absolute)


def _constitution_receipt(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for name in sorted(aggregate):
        row = copy.deepcopy(aggregate[name])
        row["tolerances"] = sorted(set(row["tolerances"]))
        row["passed"] = row["passed_count"] == row["observed_count"]
        rows.append(row)
    return {
        "named_check_count": len(rows),
        "observation_count": sum(int(row["observed_count"]) for row in rows),
        "failed_check_names": [str(row["name"]) for row in rows if not row["passed"]],
        "checks": rows,
        "passed": bool(rows) and all(bool(row["passed"]) for row in rows),
    }


def _resource_receipt(
    *,
    declared: dict[str, Any],
    minimum: dict[str, Any],
    before: dict[str, float],
    after: dict[str, float],
    step_receipts: list[dict[str, Any]],
    committed_count: int,
    final_assay_count: int,
    terminal_cost: float | None,
) -> dict[str, Any]:
    state_net_delta = _ledger_delta(before, after)
    instrument_uses = sum(
        int(row.get("transaction_status") == "committed" and row.get("operation") == "measure")
        for row in step_receipts
    )
    outcome = {
        key: sum(float(row["resource_delta"][key]) for row in step_receipts)
        for key in state_net_delta
    }
    preflight_checks = {
        key: (
            key not in minimum
            or key not in declared
            or float(declared[key]) + 1.0e-12 >= float(minimum[key])
        )
        for key in sorted(set(minimum) | set(declared))
        if isinstance(minimum.get(key, declared.get(key)), int | float)
    }
    limit_checks = {
        "operation_budget": committed_count
        <= int(declared.get("operation_budget", committed_count)),
        "sample_volume_L": outcome["sample_volume_L"]
        <= float(declared.get("sample_volume_L", outcome["sample_volume_L"])) + 1.0e-12,
        "instrument_uses": instrument_uses <= int(declared.get("instrument_uses", instrument_uses)),
        "final_assays": final_assay_count <= int(declared.get("final_assays", final_assay_count)),
        "time_s": outcome["time_s"] <= float(declared.get("time_s", outcome["time_s"])) + 1.0e-9,
    }
    terminal_cost_reconciled = (
        abs(
            float(outcome["cost"])
            - (terminal_cost if terminal_cost is not None else float(state_net_delta["cost"]))
        )
        <= 1.0e-12
    )
    state_net_reconciled = terminal_cost is not None or all(
        abs(float(outcome[key]) - float(state_net_delta[key]))
        <= (1.0e-8 if key.endswith("_J") else 1.0e-12)
        for key in state_net_delta
    )
    delta_reconciled = (
        terminal_cost_reconciled
        and state_net_reconciled
        and all(
            math.isfinite(float(row["resource_delta"][key]))
            for row in step_receipts
            for key in state_net_delta
        )
    )
    finite = all(math.isfinite(float(value)) for value in outcome.values())
    nonnegative = all(
        float(outcome[key]) >= -1.0e-12 for key in ("time_s", "cost", "sample_volume_L")
    )
    return {
        "preflight": {
            "minimum_required": copy.deepcopy(minimum),
            "declared_limits": copy.deepcopy(declared),
            "checks": preflight_checks,
            "passed": all(preflight_checks.values()),
        },
        "outcome_delta": {
            **outcome,
            "operation_attempts": len(step_receipts),
            "committed_operations": committed_count,
            "instrument_uses": instrument_uses,
            "final_assays": final_assay_count,
        },
        "state_net_delta_after_campaign_transition": state_net_delta,
        "terminal_cost": terminal_cost,
        "terminal_cost_reconciled": terminal_cost_reconciled,
        "state_net_reconciled": state_net_reconciled,
        "limit_checks": limit_checks,
        "delta_reconciled": delta_reconciled,
        "finite": finite,
        "nonnegative_accounting": nonnegative,
        "passed": (
            all(preflight_checks.values())
            and all(limit_checks.values())
            and delta_reconciled
            and finite
            and nonnegative
        ),
    }


def _named_checks(
    constitution: dict[str, Any],
    predicate: Any,
) -> list[dict[str, Any]]:
    return [
        row
        for row in constitution.get("checks", [])
        if isinstance(row, dict) and predicate(str(row.get("name", "")))
    ]


def _check_group(
    constitution: dict[str, Any],
    predicate: Any,
    *,
    applicable: bool = True,
) -> dict[str, Any]:
    rows = _named_checks(constitution, predicate)
    passed = bool(rows) and all(bool(row.get("passed")) for row in rows) if applicable else True
    return {
        "applicable": applicable,
        "named_checks": [str(row["name"]) for row in rows],
        "passed": passed,
    }


def _case_interface_receipt(
    *,
    component_kinds: set[str],
    constitution: dict[str, Any],
    step_receipts: list[dict[str, Any]],
    process_metrics: dict[str, Any],
) -> dict[str, Any]:
    material = _check_group(
        constitution,
        lambda name: (
            name == "species_registry_membership"
            or "material_single_source" in name
            or "phase_attached_vessel_exists" in name
            or "phase_listed_in_attached_vessel" in name
        ),
    )
    units = _check_group(
        constitution,
        lambda name: name.startswith("unit:") or name.startswith("observation_unit:"),
    )
    nonnegative = _check_group(
        constitution,
        lambda name: (
            name.startswith("nonnegative:")
            or "finite_nonnegative" in name
            or "amount_nonnegative" in name
        ),
    )
    material_balance = _check_group(
        constitution,
        lambda name: name == "material_conservation" or "material_balance" in name,
    )
    phase_applicable = bool(
        component_kinds & {"phase", "separation", "crystallization", "distillation"}
    )
    phase_balance = _check_group(
        constitution,
        lambda name: name.startswith("phase_") or name.startswith("vessel_phase_reverse_index"),
        applicable=phase_applicable,
    )
    energy_applicable = bool(
        component_kinds
        & {"thermal", "crystallization", "distillation", "continuous_flow", "electrochemistry"}
    )
    energy_checks = _check_group(
        constitution,
        lambda name: (
            name == "operation_energy_conservation" or name.startswith("thermal_value_finite:")
        ),
        applicable=energy_applicable,
    )
    energy_residual = process_metrics.get("energy_balance_residual_J")
    if energy_applicable and isinstance(energy_residual, int | float):
        energy_checks.update(
            {
                "residual": float(energy_residual),
                "tolerance": 1.0e-8,
                "passed": bool(energy_checks["passed"]) and abs(float(energy_residual)) <= 1.0e-8,
            }
        )
    charge_applicable = "electrochemistry" in component_kinds
    charge_values = {
        key: float(process_metrics[key])
        for key in ("charge_balance_residual_C", "electrolyte_charge_balance_error_eq")
        if isinstance(process_metrics.get(key), int | float)
    }
    charge_passed = not charge_applicable or (
        bool(charge_values)
        and abs(charge_values.get("charge_balance_residual_C", 0.0)) <= 1.0e-9
        and abs(charge_values.get("electrolyte_charge_balance_error_eq", 0.0)) <= 1.0e-10
    )
    state_identity = _check_group(
        constitution,
        lambda name: (
            "single_source" in name
            or "attached_vessel_exists" in name
            or "reverse_index" in name
            or "metadata_no_primary" in name
        ),
    )
    committed_steps = [row for row in step_receipts if row.get("transaction_status") == "committed"]
    event_passed = bool(committed_steps) and all(
        bool(row.get("event_propagation_matches_operation")) for row in committed_steps
    )
    groups = {
        "material_identity": material,
        "unit": units,
        "nonnegative_amount": nonnegative,
        "material_balance": material_balance,
        "charge_balance": {
            "applicable": charge_applicable,
            "residuals": charge_values,
            "tolerances": {
                "charge_balance_residual_C": 1.0e-9,
                "electrolyte_charge_balance_error_eq": 1.0e-10,
            },
            "passed": charge_passed,
        },
        "energy_balance": energy_checks,
        "phase_balance": phase_balance,
        "state_identity": state_identity,
        "event_propagation": {
            "applicable": True,
            "committed_step_count": len(committed_steps),
            "matched_step_count": sum(
                bool(row.get("event_propagation_matches_operation")) for row in committed_steps
            ),
            "passed": event_passed,
        },
    }
    return {
        "checks": groups,
        "passed": all(bool(row["passed"]) for row in groups.values()),
    }


def _depth_summary(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    summaries: dict[str, list[dict[str, Any]]] = {}
    for field in ("component_count", "workflow_stage_count", "action_count"):
        rows = []
        for value in sorted({int(case[field]) for case in cases}):
            group = [case for case in cases if int(case[field]) == value]
            elapsed = [float(case["elapsed_s"]) for case in group]
            sizes = [int(case["trajectory_bytes"]) for case in group]
            rows.append(
                {
                    field: value,
                    "passed": sum(bool(case["passed"]) for case in group),
                    "denominator": len(group),
                    "elapsed_s": {
                        "mean": sum(elapsed) / len(elapsed),
                        "min": min(elapsed),
                        "max": max(elapsed),
                    },
                    "trajectory_bytes": {
                        "mean": sum(sizes) / len(sizes),
                        "min": min(sizes),
                        "max": max(sizes),
                    },
                }
            )
        summaries[field] = rows
    return summaries


def validate_launch_preconditions(
    repository_root: str | Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    errors: list[str] = []
    tasks = list_tasks()
    reference_units = sum(len(task.seeds) for task in tasks)
    reference_recipes = sum(len(task.seeds) * len(recipe_cases(task)) for task in tasks)
    if len(tasks) != 15:
        errors.append("registered task denominator drifted")
    if reference_units != EXPECTED_REFERENCE_UNIT_COUNT:
        errors.append(f"reference unit denominator drifted: {reference_units}")
    if reference_recipes != EXPECTED_REFERENCE_RECIPE_COUNT:
        errors.append(f"reference recipe denominator drifted: {reference_recipes}")
    suites = build_generated_suites()
    generated_count = sum(len(suite.cases) for suite in suites)
    if generated_count != EXPECTED_GENERATED_CASE_COUNT:
        errors.append(f"generated composition denominator drifted: {generated_count}")
    protocol_path = root / PROTOCOL
    if not protocol_path.is_file():
        errors.append("frozen public protocol is missing")
    else:
        protocol_text = protocol_path.read_text(encoding="utf-8")
        required = (
            "frozen before qualification execution",
            "64 registered task-world units",
            "52 coverage-generated compositions",
            "192 invalid-action probes",
        )
        missing = [fragment for fragment in required if fragment not in protocol_text]
        if missing:
            errors.append("public protocol drifted: " + ", ".join(missing))
    if require_clean and _git(root, "status", "--porcelain", "--untracked-files=all"):
        errors.append("formal execution requires a clean worktree")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "status": "passed",
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "protocol": PROTOCOL.as_posix(),
        "protocol_sha256": _sha256_path(protocol_path),
        "qualification_design_version": QUALIFICATION_DESIGN_VERSION,
    }


def build_task_structure_baseline(repository_root: str | Path) -> dict[str, Any]:
    del repository_root
    rows = []
    all_components: set[str] = set()
    all_interfaces: set[str] = set()
    all_operations: set[str] = set()
    all_instruments: set[str] = set()
    all_observations: set[str] = set()
    all_metrics: set[str] = set()
    for task in list_tasks():
        env = gym.make(task.env_id, **task.env_kwargs(seed=int(task.seeds[0])))
        try:
            env.reset(seed=int(task.seeds[0]))
            base: Any = env.unwrapped
            task_info = base.task_info()
        finally:
            env.close()
        components = sorted(
            {
                _OPERATION_COMPONENTS[operation]
                for operation in task.allowed_operations
                if operation in _OPERATION_COMPONENTS
            }
        )
        operation_contracts = task_info.get("operation_contracts", {})
        interfaces = sorted(
            {
                "state_identity",
                "units",
                "resources",
                "termination",
                "evaluation",
                "events",
                "public_observation",
                *(
                    str(contract.get("module"))
                    for contract in operation_contracts.values()
                    if isinstance(contract, dict) and contract.get("module")
                ),
            }
        )
        observation_keys = sorted(str(value) for value in task_info.get("observation_keys", []))
        all_components.update(components)
        all_interfaces.update(interfaces)
        all_operations.update(task.allowed_operations)
        all_instruments.update(task.allowed_instruments)
        all_observations.update(observation_keys)
        all_metrics.update(task.success_metrics)
        rows.append(
            {
                "task_id": task.task_id,
                "world_seed_count": len(task.seeds),
                "components": components,
                "interfaces": interfaces,
                "operations": list(task.allowed_operations),
                "instruments": list(task.allowed_instruments),
                "observations": observation_keys,
                "resources": {
                    "operation_budget": task.budget,
                    "sample_accounting": "per-instrument",
                    "time_accounting": "per-operation-and-instrument",
                },
                "termination": task.termination_policy,
                "evaluation_metrics": list(task.success_metrics),
            }
        )
    return {
        "status": "passed",
        "registered_task_count": len(rows),
        "world_unit_count": sum(row["world_seed_count"] for row in rows),
        "coverage": {
            "components": sorted(all_components),
            "interfaces": sorted(all_interfaces),
            "operations": sorted(all_operations),
            "instruments": sorted(all_instruments),
            "observations": sorted(all_observations),
            "evaluation_metrics": sorted(all_metrics),
        },
        "tasks": rows,
    }


def _run_generated_case(
    case: Any,
    *,
    world_seed: int,
    generation_seed: int,
    generation_index: int,
    scratch_dir: Path,
) -> dict[str, Any]:
    env = gym.make("ChemWorld", composition=case.request, seed=world_seed)
    trajectory_path = scratch_dir / f"{case.case_id}.jsonl"
    failures: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    step_receipts: list[dict[str, Any]] = []
    constitution_checks: dict[str, dict[str, Any]] = {}
    committed_count = 0
    final_assay_count = 0
    terminate_count = 0
    constitution_failure_count = 0
    event_count = 0
    post_termination_validation: dict[str, Any] = {
        "checked": False,
        "passed": False,
        "reason": "terminate action was not reached",
    }
    evaluation_receipt: dict[str, Any] = {}
    process_metrics: dict[str, Any] = {}
    replay: dict[str, Any] = {
        "verified": False,
        "checked_steps": 0,
        "max_abs_error": None,
        "mismatches": [],
    }
    initial_public_view_sha256 = ""
    resource_before: dict[str, float] = {}
    resource_after: dict[str, float] = {}
    compiled_surface = case.compiled.to_public_dict()
    started = time.perf_counter()
    try:
        observation, reset_info = env.reset(seed=world_seed)
        base: Any = env.unwrapped
        logging_task_info = {**base.task_info(), **base.evaluator_provenance()}
        resource_before = _ledger_snapshot(env)
        leakage.extend(_leakage_findings(env, reset_info, "reset_info"))
        initial_public_view = agent_view_bundle(env, observation, {})
        initial_public_view_sha256 = _sha256_value(initial_public_view)
        leakage.extend(_leakage_findings(env, initial_public_view, "initial_agent_view"))
        with TrajectoryLogger(trajectory_path) as logger:
            for step, action in enumerate(case.actions, start=1):
                ledger_before = _ledger_snapshot(env)
                validation = base.validate_action(action)
                if not bool(validation.get("valid")):
                    failures.append(
                        {
                            "step": step,
                            "class": "generated_action_prevalidation_failed",
                            "invalid_reasons": validation.get("invalid_reasons", []),
                        }
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                ledger_after = _ledger_snapshot(env)
                current_metrics = base._state.process.metrics
                for metric_name in (
                    "charge_balance_residual_C",
                    "electrolyte_charge_balance_error_eq",
                    "energy_balance_residual_J",
                ):
                    metric_value = current_metrics.get(metric_name)
                    if isinstance(metric_value, int | float):
                        process_metrics[metric_name] = float(metric_value)
                committed = info.get("transaction_status") == "committed"
                committed_count += int(committed)
                if not committed:
                    failures.append(
                        {
                            "step": step,
                            "class": "generated_transaction_not_committed",
                            "operation": action.get("operation"),
                            "transaction_status": info.get("transaction_status"),
                            "preconditions": info.get("preconditions", {}),
                        }
                    )
                failed_checks = [
                    check
                    for check in info.get("constitution_checks", [])
                    if isinstance(check, dict) and check.get("passed") is False
                ]
                _summarize_constitution_checks(
                    constitution_checks,
                    cast(list[Any], info.get("constitution_checks", [])),
                )
                constitution_failure_count += len(failed_checks)
                if failed_checks:
                    failures.append(
                        {
                            "step": step,
                            "class": "constitution_check_failed",
                            "checks": failed_checks,
                        }
                    )
                world_events = [
                    event for event in info.get("world_events", []) if isinstance(event, dict)
                ]
                event_count += len(world_events)
                event_matches = any(
                    event.get("event_type") == "operation_applied"
                    and event.get("operation_type") == action.get("operation")
                    for event in world_events
                )
                if (
                    action.get("operation") == "measure"
                    and action.get("instrument") == "final_assay"
                ):
                    final_assay_count += int(committed)
                    evaluation_receipt = {
                        "reward": float(reward),
                        "environment_reward": info.get("environment_reward"),
                        "observed_reward": info.get("observed_reward"),
                        "leaderboard_score": info.get("leaderboard_score"),
                        "experiment_completed": info.get("experiment_completed"),
                        "experiment_ended": info.get("experiment_ended"),
                        "transaction_status": info.get("transaction_status"),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "last_terminal_summary": copy.deepcopy(info.get("last_terminal_summary")),
                    }
                if action.get("operation") == "terminate" and committed:
                    terminate_count += 1
                    probe_action = copy.deepcopy(
                        next(
                            candidate
                            for candidate in case.actions
                            if candidate.get("operation") not in {"terminate", "measure"}
                        )
                    )
                    rejection = base.validate_action(probe_action)
                    post_termination_validation = {
                        "checked": True,
                        "probe_action": probe_action,
                        "valid": bool(rejection.get("valid")),
                        "invalid_reasons": rejection.get("invalid_reasons", []),
                        "not_terminated_precondition": rejection.get("preconditions", {}).get(
                            "not_terminated"
                        ),
                        "will_mutate_state": rejection.get("will_mutate_state"),
                        "passed": (
                            rejection.get("valid") is False
                            and rejection.get("preconditions", {}).get("not_terminated") is False
                            and rejection.get("will_mutate_state") is False
                        ),
                    }
                public_view = agent_view_bundle(env, observation, info)
                step_leakage = _leakage_findings(env, public_view, f"agent_view.step-{step}")
                leakage.extend(step_leakage)
                step_receipts.append(
                    {
                        "step": step,
                        "action": copy.deepcopy(action),
                        "action_sha256": _sha256_value(action),
                        "operation": action.get("operation"),
                        "schema_validation": {
                            "valid": bool(validation.get("valid")),
                            "invalid_reasons": validation.get("invalid_reasons", []),
                            "canonical_action_sha256": _sha256_value(
                                validation.get("canonical_action")
                            ),
                        },
                        "transaction_status": info.get("transaction_status"),
                        "failed_preconditions": sorted(
                            str(name)
                            for name, passed in info.get("preconditions", {}).items()
                            if passed is False
                        ),
                        "state_transition_sha256": _sha256_value(
                            {
                                "delta": info.get("state_delta_summary"),
                                "patches": info.get("state_patches_summary"),
                            }
                        ),
                        "resource_delta": {
                            "time_s": float(
                                info.get("state_delta_summary", {}).get("delta_time_s", 0.0)
                            ),
                            "cost": float(info.get("cost_delta", 0.0)),
                            "risk": float(info.get("risk_delta", 0.0)),
                            "sample_volume_L": float(info.get("sample_delta", 0.0)),
                            **{
                                key: value
                                for key, value in _ledger_delta(
                                    ledger_before,
                                    ledger_after,
                                ).items()
                                if key.endswith("_J")
                            },
                        },
                        "remaining_budget": info.get("remaining_budget"),
                        "affected_ledgers": sorted(
                            str(value) for value in info.get("affected_ledgers", [])
                        ),
                        "constitution_check_count": len(info.get("constitution_checks", [])),
                        "constitution_failed_check_names": sorted(
                            str(check.get("name")) for check in failed_checks
                        ),
                        "world_event_count": len(world_events),
                        "world_event_types": sorted(
                            str(event.get("event_type")) for event in world_events
                        ),
                        "event_propagation_matches_operation": event_matches,
                        "public_observation_sha256": _sha256_value(
                            observation_to_json(observation)
                        ),
                        "public_private_leakage_count": len(step_leakage),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                logger.log(
                    task_info=logging_task_info,
                    step=step,
                    action=action,
                    observation=observation_to_json(observation),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    agent_metadata={"agent_id": "composition-qualification-reference"},
                    agent_view=public_view,
                )
                if (terminated or truncated) and step != len(case.actions):
                    failures.append(
                        {
                            "step": step,
                            "class": "generated_lifecycle_ended_before_frozen_workflow",
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    break
        resource_after = _ledger_snapshot(env)
        records = load_jsonl(trajectory_path)
        replay = verify_records(records, tolerance=0.0).to_dict()
        if not replay["verified"]:
            failures.append({"class": "exact_replay_failed", "mismatches": replay["mismatches"]})
    except Exception as exc:
        failures.append(
            {
                "class": "execution_exception",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    finally:
        env.close()
    constitution = _constitution_receipt(constitution_checks)
    declared_resources = cast(dict[str, Any], compiled_surface["task"]["resources"])
    minimum_resources = cast(dict[str, Any], compiled_surface["compatibility"]["minimum_resources"])
    zero_ledger = {
        "time_s": 0.0,
        "cost": 0.0,
        "risk": 0.0,
        "sample_volume_L": 0.0,
        "energy_jacket_J": 0.0,
        "heat_reaction_J": 0.0,
        "heat_loss_J": 0.0,
    }
    resources = _resource_receipt(
        declared=declared_resources,
        minimum=minimum_resources,
        before=resource_before or zero_ledger,
        after=resource_after or resource_before or zero_ledger,
        step_receipts=step_receipts,
        committed_count=committed_count,
        final_assay_count=final_assay_count,
        terminal_cost=(
            float(evaluation_receipt["last_terminal_summary"]["cost"])
            if isinstance(evaluation_receipt.get("last_terminal_summary"), dict)
            and isinstance(
                evaluation_receipt["last_terminal_summary"].get("cost"),
                int | float,
            )
            else None
        ),
    )
    interfaces = _case_interface_receipt(
        component_kinds={str(item["kind"]) for item in case.request["components"]},
        constitution=constitution,
        step_receipts=step_receipts,
        process_metrics=process_metrics,
    )
    if leakage:
        failures.append({"class": "public_private_leakage", "finding_count": len(leakage)})
    if final_assay_count != 1 or terminate_count != 1:
        failures.append(
            {
                "class": "lifecycle_not_closed",
                "committed_terminate_count": terminate_count,
                "committed_final_assay_count": final_assay_count,
            }
        )
    if not bool(post_termination_validation.get("passed")):
        failures.append(
            {
                "class": "post_termination_operation_not_rejected",
                "receipt": post_termination_validation,
            }
        )
    if not resources["passed"]:
        failures.append({"class": "resource_reconciliation_failed", "receipt": resources})
    if not interfaces["passed"]:
        failures.append({"class": "interface_receipt_failed", "receipt": interfaces})
    elapsed = time.perf_counter() - started
    return {
        "case_id": case.case_id,
        "composition_id": case.request["composition_id"],
        "composition_request": copy.deepcopy(case.request),
        "composition_request_sha256": _sha256_value(case.request),
        "pattern": str(case.compiled.compatibility.pattern),
        "generation_seed": generation_seed,
        "generation_index": generation_index,
        "discrete_levels": copy.deepcopy(case.discrete_levels),
        "continuous_coordinates": copy.deepcopy(case.continuous_coordinates),
        "workflow_id": case.workflow_id,
        "compile_receipt": {
            "compatible": bool(case.compiled.compatibility.compatible),
            "diagnostics": [item.to_dict() for item in case.compiled.compatibility.diagnostics],
            "compiled_public_surface_sha256": _sha256_value(compiled_surface),
            "task_contract_sha256": _sha256_value(compiled_surface["task"]),
        },
        "component_count": len(case.request["components"]),
        "workflow_stage_count": len(case.actions),
        "action_count": len(case.actions),
        "committed_action_count": committed_count,
        "constitution_failure_count": constitution_failure_count,
        "world_event_count": event_count,
        "committed_final_assay_count": final_assay_count,
        "execution_receipt": {
            "compiled": bool(case.compiled.compatibility.compatible),
            "executed": len(step_receipts) == len(case.actions),
            "closed": terminate_count == 1 and final_assay_count == 1,
            "resource_reconciled": bool(resources["passed"]),
        },
        "termination_receipt": {
            "committed_terminate_count": terminate_count,
            "committed_final_assay_count": final_assay_count,
            "post_termination_validation": post_termination_validation,
            "closed": terminate_count == 1 and final_assay_count == 1,
        },
        "evaluation_receipt": evaluation_receipt,
        "resource_receipt": resources,
        "constitution_receipt": constitution,
        "interface_receipt": interfaces,
        "step_receipts": step_receipts,
        "public_observation_receipt": {
            "initial_public_view_sha256": initial_public_view_sha256,
            "step_surface_count": len(step_receipts),
            "leakage_findings": leakage,
        },
        "public_private_leakage_count": len(leakage),
        "exact_replay": replay,
        "elapsed_s": elapsed,
        "trajectory_bytes": trajectory_path.stat().st_size if trajectory_path.exists() else 0,
        "passed": not failures,
        "failures": failures,
    }


def run_generated_qualification(*, scratch_dir: Path) -> dict[str, Any]:
    suites = build_generated_suites()
    receipts = [
        _run_generated_case(
            case,
            world_seed=0,
            generation_seed=int(suite.target.seed),
            generation_index=generation_index,
            scratch_dir=scratch_dir,
        )
        for suite in suites
        for generation_index, case in enumerate(suite.cases)
    ]
    pattern_rows = []
    for pattern, denominator in EXPECTED_PATTERN_CASE_COUNTS.items():
        rows = [row for row in receipts if row["pattern"] == pattern]
        pattern_rows.append(
            {
                "pattern": pattern,
                "passed": sum(bool(row["passed"]) for row in rows),
                "denominator": denominator,
                "coverage_report": next(
                    suite.report
                    for suite in suites
                    if str(suite.cases[0].compiled.compatibility.pattern) == pattern
                ),
            }
        )
    unseen_cases = [row for row in receipts if row["pattern"] == UNSEEN_PATTERN_ID]
    registered_task_ids = {task.task_id for task in list_tasks()}
    unseen_reference_task_id_overlap = sorted(
        str(row["composition_id"])
        for row in unseen_cases
        if row["composition_id"] in registered_task_ids
    )
    return {
        "passed": sum(bool(row["passed"]) for row in receipts),
        "denominator": len(receipts),
        "unseen_pattern": UNSEEN_PATTERN_ID,
        "unseen_passed": sum(bool(row["passed"]) for row in unseen_cases),
        "unseen_denominator": EXPECTED_PATTERN_CASE_COUNTS[UNSEEN_PATTERN_ID],
        "unseen_reference_task_id_overlap": unseen_reference_task_id_overlap,
        "pattern_matrix": pattern_rows,
        "depth_summary": _depth_summary(receipts),
        "cases": receipts,
    }


def _mutant_requests() -> list[dict[str, Any]]:
    by_pattern = {
        str(suite.cases[0].compiled.compatibility.pattern): copy.deepcopy(suite.cases[0].request)
        for suite in build_generated_suites()
    }
    missing_observation = copy.deepcopy(by_pattern["reaction-thermal-observation"])
    missing_observation["composition_id"] = "qualification-mutant-missing-observation"
    missing_observation["components"] = [
        item for item in missing_observation["components"] if item["kind"] != "observation"
    ]
    missing_thermal = copy.deepcopy(by_pattern["reaction-crystallization-observation"])
    missing_thermal["composition_id"] = "qualification-mutant-crystallization-without-thermal"
    missing_thermal["components"] = [
        item for item in missing_thermal["components"] if item["kind"] != "thermal"
    ]
    conflicting = copy.deepcopy(by_pattern["reaction-crystallization-observation"])
    conflicting["composition_id"] = "qualification-mutant-conflicting-phase-owner"
    conflicting["components"].insert(2, {"kind": "phase", "role": "phase", "parameters": {}})
    wrong_unit = copy.deepcopy(by_pattern["reaction-thermal-observation"])
    wrong_unit["composition_id"] = "qualification-mutant-thermal-unit"
    next(item for item in wrong_unit["components"] if item["kind"] == "thermal")["parameters"] = {
        "temperature_range_K": {"value": [350.0, 390.0], "unit": "L"}
    }
    invalid_fraction = copy.deepcopy(by_pattern["reaction-distillation-observation"])
    invalid_fraction["composition_id"] = "qualification-mutant-fraction-count"
    next(item for item in invalid_fraction["components"] if item["kind"] == "distillation")[
        "parameters"
    ]["fraction_count"] = 0
    lifecycle = copy.deepcopy(by_pattern["reaction-thermal-observation"])
    lifecycle["composition_id"] = "qualification-mutant-lifecycle-hole"
    lifecycle["task"]["operations"] = [
        item
        for item in lifecycle["task"].get(
            "operations",
            [
                "add_solvent",
                "add_reagent",
                "add_catalyst",
                "heat",
                "wait",
                "measure",
                "terminate",
            ],
        )
        if item != "terminate"
    ]
    impossible = copy.deepcopy(by_pattern["reaction-thermal-observation"])
    impossible["composition_id"] = "qualification-mutant-resource-impossibility"
    impossible["task"]["budget"] = 3
    impossible["task"]["resources"]["operation_budget"] = 3
    return [
        {
            "mutant_id": "missing-observation",
            "expected": "missing_dependency",
            "request": missing_observation,
        },
        {
            "mutant_id": "crystallization-missing-thermal",
            "expected": "missing_dependency",
            "request": missing_thermal,
        },
        {
            "mutant_id": "conflicting-phase-owner",
            "expected": "conflicting_state_owner",
            "request": conflicting,
        },
        {"mutant_id": "thermal-unit-mismatch", "expected": "unit_mismatch", "request": wrong_unit},
        {
            "mutant_id": "invalid-fraction-count",
            "expected": "invalid_parameter",
            "request": invalid_fraction,
        },
        {"mutant_id": "missing-terminate", "expected": "lifecycle_hole", "request": lifecycle},
        {
            "mutant_id": "insufficient-budget",
            "expected": "resource_impossibility",
            "request": impossible,
        },
    ]


def run_compile_mutants() -> dict[str, Any]:
    receipts = []
    for mutant in _mutant_requests():
        codes: list[str] = []
        constructed = False
        try:
            chemworld.compile_world_composition(mutant["request"])
            constructed = True
        except WorldCompositionError as exc:
            codes = sorted({item.code for item in exc.diagnostics})
        passed = not constructed and mutant["expected"] in codes
        receipts.append(
            {
                "mutant_id": mutant["mutant_id"],
                "expected_diagnostic": mutant["expected"],
                "observed_diagnostics": codes,
                "environment_constructed": constructed,
                "passed": passed,
                "failures": [] if passed else ["compile mutant did not fail closed as specified"],
            }
        )
    return {
        "passed": sum(bool(row["passed"]) for row in receipts),
        "denominator": len(receipts),
        "mutants": receipts,
    }


def _state_metric(env: Any, metric: str) -> float:
    state = env.unwrapped._state
    if metric == "total_species_mol":
        return float(sum(state.species_amounts.values()))
    if metric == "temperature_K":
        return float(state.temperature_K)
    if metric == "volume_L":
        return float(state.volume_L)
    if metric == "time_s":
        return float(state.ledger.time_s)
    if metric == "cost":
        return float(state.ledger.cost)
    raise ValueError(f"unknown module probe metric: {metric}")


def _module_action_receipt(case: Any, config: dict[str, Any], value: float) -> dict[str, Any]:
    env = gym.make("ChemWorld", composition=case.request, seed=0)
    operation = str(config["operation"])
    try:
        env.reset(seed=0)
        focus_index = next(
            index for index, action in enumerate(case.actions) if action["operation"] == operation
        )
        for setup_action in case.actions[:focus_index]:
            _observation, _reward, _terminated, _truncated, info = env.step(setup_action)
            if info.get("transaction_status") != "committed":
                return {"passed": False, "failure": "module setup action did not commit"}
        action = copy.deepcopy(case.actions[focus_index])
        action[str(config["field"])] = value
        before_physical = _physical_snapshot(env)
        before_metric = _state_metric(env, str(config["metric"]))
        base: Any = env.unwrapped
        validation = base.validate_action(action)
        _observation, _reward, _terminated, _truncated, info = env.step(action)
        after_metric = _state_metric(env, str(config["metric"]))
        committed = info.get("transaction_status") == "committed"
        failed_checks = [
            check
            for check in info.get("constitution_checks", [])
            if isinstance(check, dict) and check.get("passed") is False
        ]
        relation = config.get("reference_relation")
        reference_fixture: dict[str, Any] | None = None
        if committed and isinstance(relation, str):
            if relation == "diagnostic_equals_zero":
                settings = equipment_settings(
                    base._state.equipment,
                    "batch_reactor",
                )
                observed = float(settings[str(config["reference_diagnostic"])])
                expected = 0.0
            else:
                observed = (
                    after_metric
                    if relation == "after_equals_input"
                    else after_metric - before_metric
                )
                expected = float(value)
            tolerance = float(config["reference_tolerance"])
            error = abs(float(observed) - expected)
            reference_fixture = {
                "fixture_id": config["reference_fixture"],
                "relation": relation,
                "expected": expected,
                "observed": float(observed),
                "absolute_error": error,
                "tolerance": tolerance,
                "within_tolerance": error <= tolerance,
            }
        return {
            "passed": committed and bool(validation.get("valid")) and not failed_checks,
            "validation_passed": bool(validation.get("valid")),
            "transaction_status": info.get("transaction_status"),
            "metric_before": before_metric,
            "metric_after": after_metric,
            "metric_delta": after_metric - before_metric,
            "constitution_failure_count": len(failed_checks),
            "constitution_check_names": sorted(
                str(check.get("name"))
                for check in info.get("constitution_checks", [])
                if isinstance(check, dict)
            ),
            "reference_fixture": reference_fixture,
            "physical_state_preserved": _physical_snapshot(env) == before_physical,
        }
    finally:
        env.close()


def run_module_probes() -> dict[str, Any]:
    suites = {
        str(suite.cases[0].compiled.compatibility.pattern): suite
        for suite in build_generated_suites()
    }
    receipts = []
    maturity_rows = []
    for module_id, config in _MODULE_CONFIGS.items():
        case = suites[str(config["pattern"])].cases[0]
        zero = _module_action_receipt(case, config, 0.0)
        zero_delta = zero.get("metric_delta")
        zero_passed = zero.get("transaction_status") != "committed" or (
            isinstance(zero_delta, int | float) and abs(float(zero_delta)) <= 1.0e-12
        )
        low = _module_action_receipt(case, config, cast(float, config["low"]))
        high = _module_action_receipt(case, config, cast(float, config["high"]))
        low_delta = low.get("metric_delta")
        high_delta = high.get("metric_delta")
        direction_passed = (
            bool(low.get("passed"))
            and bool(high.get("passed"))
            and isinstance(low_delta, int | float)
            and isinstance(high_delta, int | float)
            and float(high_delta) > float(low_delta)
        )
        invariant_passed = (
            int(low.get("constitution_failure_count", 1)) == 0
            and int(high.get("constitution_failure_count", 1)) == 0
        )
        fixture_rows = [
            cast(dict[str, Any], row["reference_fixture"])
            for row in (low, high)
            if isinstance(row.get("reference_fixture"), dict)
        ]
        fixture_passed = bool(fixture_rows) and all(
            bool(row.get("within_tolerance")) for row in fixture_rows
        )
        receipts.extend(
            [
                {
                    "module_id": module_id,
                    "probe_id": "zero_input",
                    "classification": "bounded_runtime_probe",
                    "passed": zero_passed,
                    "receipt": zero,
                    "failures": [] if zero_passed else ["zero-input behavior drifted"],
                },
                {
                    "module_id": module_id,
                    "probe_id": "legal_low_high",
                    "classification": (
                        "numerical_reference_fixture" if fixture_rows else "conceptual_or_synthetic"
                    ),
                    "passed": (
                        bool(low.get("passed"))
                        and bool(high.get("passed"))
                        and (fixture_passed if fixture_rows else True)
                    ),
                    "receipt": {"low": low, "high": high},
                    "failures": (
                        []
                        if (
                            bool(low.get("passed"))
                            and bool(high.get("passed"))
                            and (fixture_passed if fixture_rows else True)
                        )
                        else ["legal low/high boundary or numerical fixture failed"]
                    ),
                },
                {
                    "module_id": module_id,
                    "probe_id": "directionality_pair",
                    "classification": "conceptual_or_synthetic",
                    "passed": direction_passed,
                    "receipt": {
                        "metric": config["metric"],
                        "low_delta": low.get("metric_delta"),
                        "high_delta": high.get("metric_delta"),
                    },
                    "failures": ([] if direction_passed else ["declared directional pair failed"]),
                },
                {
                    "module_id": module_id,
                    "probe_id": "conservation_invariant",
                    "classification": "runtime_constitution",
                    "passed": invariant_passed,
                    "receipt": {
                        "low_constitution_failure_count": low.get("constitution_failure_count"),
                        "high_constitution_failure_count": high.get("constitution_failure_count"),
                    },
                    "failures": (
                        []
                        if invariant_passed
                        else ["module conservation or invariant check failed"]
                    ),
                },
            ]
        )
        env = gym.make("ChemWorld", composition=case.request, seed=0)
        try:
            env.reset(seed=0)
            base: Any = env.unwrapped
            kernel_maturity = base.task_info()["kernel_maturity"]
        finally:
            env.close()
        focus_module_id = str(config["maturity_module_id"])
        focus_modules = [
            row
            for row in kernel_maturity.get("modules", [])
            if row.get("module_id") == focus_module_id
        ]
        if len(focus_modules) != 1:
            raise RuntimeError(
                f"expected one maturity record for {module_id}/{focus_module_id}, "
                f"got {len(focus_modules)}"
            )
        maturity_rows.append(
            {
                "module_id": module_id,
                "domain": str(config["pattern"]),
                "focus_runtime_module_id": focus_module_id,
                "focus_maturity": focus_modules[0],
                "task_runtime_maturity_context": kernel_maturity,
                "claim_boundary": "virtual_instrument_qualification_only",
            }
        )
    return {
        "passed": sum(bool(row["passed"]) for row in receipts),
        "denominator": len(receipts),
        "probes": receipts,
        "model_boundaries": maturity_rows,
    }


def build_interface_receipts(generated: dict[str, Any]) -> dict[str, Any]:
    receipts = []
    for path_id, pattern in _INTERFACE_PATHS.items():
        cases = [row for row in generated["cases"] if row["pattern"] == pattern]
        check_ids = (
            "material_identity",
            "unit",
            "nonnegative_amount",
            "material_balance",
            "charge_balance",
            "energy_balance",
            "phase_balance",
            "state_identity",
            "event_propagation",
        )
        checks = {
            check_id: all(
                bool(row["interface_receipt"]["checks"][check_id]["passed"]) for row in cases
            )
            for check_id in check_ids
        }
        checks["transaction_atomicity"] = all(
            int(row["committed_action_count"]) == int(row["action_count"]) for row in cases
        )
        checks["resource_reconciliation"] = all(
            bool(row["resource_receipt"]["passed"]) for row in cases
        )
        checks["lifecycle_closure"] = all(
            bool(row["termination_receipt"]["closed"])
            and bool(row["termination_receipt"]["post_termination_validation"]["passed"])
            for row in cases
        )
        checks["public_private_boundary"] = all(
            int(row["public_private_leakage_count"]) == 0 for row in cases
        )
        checks["exact_replay"] = all(bool(row["exact_replay"]["verified"]) for row in cases)
        receipts.append(
            {
                "path_id": path_id,
                "pattern": pattern,
                "case_count": len(cases),
                "checks": checks,
                "case_receipts": [
                    {
                        "case_id": row["case_id"],
                        "composition_request_sha256": row["composition_request_sha256"],
                        "interface_receipt": row["interface_receipt"],
                        "resource_reconciled": row["resource_receipt"]["passed"],
                        "lifecycle_closed": row["termination_receipt"]["closed"],
                        "exact_replay": row["exact_replay"]["verified"],
                    }
                    for row in cases
                ],
                "passed": bool(cases) and all(checks.values()),
                "failures": (
                    []
                    if bool(cases) and all(checks.values())
                    else ["cross-component interface path failed"]
                ),
            }
        )
    return {
        "passed": sum(bool(row["passed"]) for row in receipts),
        "denominator": len(receipts),
        "paths": receipts,
    }


def _receipt_completeness_errors(
    *,
    task_structure: dict[str, Any],
    reference: dict[str, Any],
    generated: dict[str, Any],
    modules: dict[str, Any],
    interfaces: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def missing(surface: str, identity: str, requirement: str) -> None:
        errors.append(
            {
                "class": "missing_receipt",
                "surface": surface,
                "identity": identity,
                "requirement": requirement,
            }
        )

    coverage = task_structure.get("coverage", {})
    for field in ("components", "interfaces", "operations", "instruments", "observations"):
        if not isinstance(coverage.get(field), list) or not coverage[field]:
            missing("task_structure", "coverage", field)
    for task in task_structure.get("tasks", []):
        task_id = str(task.get("task_id"))
        for field in (
            "components",
            "interfaces",
            "operations",
            "instruments",
            "observations",
            "resources",
            "termination",
            "evaluation_metrics",
        ):
            if field not in task or task[field] in (None, [], {}):
                missing("task_structure", task_id, field)

    for unit in reference.get("units", []):
        unit_id = str(unit.get("unit_id"))
        for case in unit.get("valid_recipe_cases", []):
            case_id = f"{unit_id}/{case.get('case_id')}"
            execution = case.get("execution_receipt", {})
            if not all(
                execution.get(field) is True
                for field in ("compiled", "executed", "closed", "resource_reconciled")
            ):
                missing("reference_recipe", case_id, "execution_receipt")
            if len(case.get("step_receipts", [])) != int(case.get("compiled_operation_count", -1)):
                missing("reference_recipe", case_id, "step_receipts")
            if not case.get("resource_preflight") or not case.get("resource_outcome_delta"):
                missing("reference_recipe", case_id, "resource_preflight_and_outcome")
            if case.get("resource_reconciled") is not True:
                missing("reference_recipe", case_id, "resource_reconciled")
            post_termination = case.get("lifecycle_receipt", {}).get(
                "post_termination_nonfinal_validation", {}
            )
            if post_termination.get("passed") is not True:
                missing("reference_recipe", case_id, "post_termination_rejection")
            if case.get("constitution_receipt", {}).get("passed") is not True:
                missing("reference_recipe", case_id, "constitution_receipt")
            if case.get("public_observation_receipt", {}).get("step_surface_count") != len(
                case.get("step_receipts", [])
            ):
                missing("reference_recipe", case_id, "public_observation_receipt")
            if not case.get("evaluation_receipt"):
                missing("reference_recipe", case_id, "evaluation_receipt")
            elapsed = case.get("elapsed_s")
            if not isinstance(elapsed, int | float) or not math.isfinite(float(elapsed)):
                missing("reference_recipe", case_id, "elapsed_s")
            if int(case.get("trajectory_bytes", 0)) <= 0:
                missing("reference_recipe", case_id, "trajectory_bytes")
        for probe in unit.get("negative_probes", []):
            probe_id = f"{unit_id}/{probe.get('probe_id')}"
            if not probe.get("expected_rejection") or not probe.get("observed_rejection"):
                missing("negative_probe", probe_id, "expected_and_observed_rejection")
            ghost = probe.get("ghost_state", {})
            if ghost.get("ghost_state_preserved") is not True:
                missing("negative_probe", probe_id, "ghost_state")
            if not probe.get("resource_outcome_delta"):
                missing("negative_probe", probe_id, "resource_outcome_delta")

    for case in generated.get("cases", []):
        case_id = str(case.get("case_id"))
        for field in (
            "composition_request",
            "composition_request_sha256",
            "generation_seed",
            "generation_index",
            "discrete_levels",
            "continuous_coordinates",
            "compile_receipt",
        ):
            if field not in case:
                missing("generated_case", case_id, field)
        execution = case.get("execution_receipt", {})
        if not all(
            execution.get(field) is True
            for field in ("compiled", "executed", "closed", "resource_reconciled")
        ):
            missing("generated_case", case_id, "execution_receipt")
        if len(case.get("step_receipts", [])) != int(case.get("action_count", -1)):
            missing("generated_case", case_id, "step_receipts")
        if case.get("resource_receipt", {}).get("passed") is not True:
            missing("generated_case", case_id, "resource_receipt")
        if (
            case.get("termination_receipt", {}).get("post_termination_validation", {}).get("passed")
            is not True
        ):
            missing("generated_case", case_id, "post_termination_rejection")
        if case.get("constitution_receipt", {}).get("passed") is not True:
            missing("generated_case", case_id, "constitution_receipt")
        if case.get("interface_receipt", {}).get("passed") is not True:
            missing("generated_case", case_id, "interface_receipt")
        if not case.get("evaluation_receipt"):
            missing("generated_case", case_id, "evaluation_receipt")
        if case.get("public_observation_receipt", {}).get("step_surface_count") != len(
            case.get("step_receipts", [])
        ):
            missing("generated_case", case_id, "public_observation_receipt")

    for probe in modules.get("probes", []):
        if probe.get("probe_id") != "legal_low_high":
            continue
        probe_id = str(probe.get("module_id"))
        classification = probe.get("classification")
        if classification == "numerical_reference_fixture":
            for boundary in ("low", "high"):
                fixture = probe.get("receipt", {}).get(boundary, {}).get("reference_fixture")
                if not isinstance(fixture, dict) or not all(
                    field in fixture
                    for field in (
                        "expected",
                        "observed",
                        "absolute_error",
                        "tolerance",
                        "within_tolerance",
                    )
                ):
                    missing("module_probe", probe_id, f"{boundary}_reference_fixture")
        elif classification != "conceptual_or_synthetic":
            missing("module_probe", probe_id, "reference_classification")

    required_interface_checks = {
        "material_identity",
        "unit",
        "nonnegative_amount",
        "material_balance",
        "charge_balance",
        "energy_balance",
        "phase_balance",
        "state_identity",
        "event_propagation",
        "transaction_atomicity",
        "resource_reconciliation",
        "lifecycle_closure",
        "public_private_boundary",
        "exact_replay",
    }
    for path in interfaces.get("paths", []):
        path_id = str(path.get("path_id"))
        if set(path.get("checks", {})) != required_interface_checks:
            missing("interface_path", path_id, "named_interface_checks")
        if len(path.get("case_receipts", [])) != int(path.get("case_count", -1)):
            missing("interface_path", path_id, "case_receipts")

    depth = generated.get("depth_summary", {})
    if set(depth) != {"component_count", "workflow_stage_count", "action_count"}:
        missing("generated_qualification", "depth_summary", "all_depth_axes")
    return errors


def _failure_counts(report_parts: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("passed") is False:
                failures = value.get("failures")
                if isinstance(failures, list) and failures:
                    for failure in failures:
                        if isinstance(failure, dict):
                            counts[str(failure.get("class", "unclassified"))] += 1
                        else:
                            counts[str(failure)] += 1
                elif "failure" in value:
                    counts[str(value["failure"])] += 1
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for part in report_parts:
        visit(part)
    return dict(sorted(counts.items()))


def build_report(
    *,
    repository_root: str | Path,
    require_clean: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    binding = validate_launch_preconditions(root, require_clean=require_clean)
    task_structure = build_task_structure_baseline(root)
    reference_units = []
    with tempfile.TemporaryDirectory(prefix="chemworld-composition-qualification-") as tmp:
        scratch = Path(tmp)
        protocol = _reference_protocol()
        for task in list_tasks():
            for world_seed in task.seeds:
                reference_units.append(
                    run_task_world_unit(
                        task,
                        int(world_seed),
                        protocol,
                        scratch_dir=scratch,
                    )
                )
        generated = run_generated_qualification(scratch_dir=scratch)
    valid_cases = [case for unit in reference_units for case in unit["valid_recipe_cases"]]
    negative_probes = [probe for unit in reference_units for probe in unit["negative_probes"]]
    reference = {
        "unit_passed": sum(bool(unit["passed"]) for unit in reference_units),
        "unit_denominator": len(reference_units),
        "recipe_passed": sum(bool(case["passed"]) for case in valid_cases),
        "recipe_denominator": len(valid_cases),
        "negative_probe_passed": sum(bool(probe["passed"]) for probe in negative_probes),
        "negative_probe_denominator": len(negative_probes),
        "units": reference_units,
    }
    mutants = run_compile_mutants()
    modules = run_module_probes()
    interfaces = build_interface_receipts(generated)
    completeness_errors = _receipt_completeness_errors(
        task_structure=task_structure,
        reference=reference,
        generated=generated,
        modules=modules,
        interfaces=interfaces,
    )
    completeness = {
        "passed": not completeness_errors,
        "error_count": len(completeness_errors),
        "errors": completeness_errors,
        "failures": completeness_errors,
    }
    failure_counts = _failure_counts(
        [reference, generated, mutants, modules, interfaces, completeness]
    )
    denominator_receipt_gaps = sum(
        abs(expected - actual)
        for expected, actual in (
            (EXPECTED_REFERENCE_UNIT_COUNT, cast(int, reference["unit_denominator"])),
            (EXPECTED_REFERENCE_RECIPE_COUNT, cast(int, reference["recipe_denominator"])),
            (EXPECTED_NEGATIVE_PROBE_COUNT, cast(int, reference["negative_probe_denominator"])),
            (EXPECTED_GENERATED_CASE_COUNT, generated["denominator"]),
            (EXPECTED_COMPILE_MUTANT_COUNT, mutants["denominator"]),
            (EXPECTED_MODULE_PROBE_COUNT, modules["denominator"]),
            (EXPECTED_INTERFACE_PATH_COUNT, interfaces["denominator"]),
        )
    )
    missing_receipt_count = denominator_receipt_gaps + len(completeness_errors)
    leakage_count = (
        sum(int(case["public_private_leakage_count"]) for case in generated["cases"])
        + sum(
            int(case["public_private_leakage_count"])
            for unit in reference_units
            for case in unit["valid_recipe_cases"]
        )
        + sum(
            int(probe["public_private_leakage_count"])
            for unit in reference_units
            for probe in unit["negative_probes"]
        )
    )
    overall_pass = (
        reference["unit_passed"] == EXPECTED_REFERENCE_UNIT_COUNT
        and reference["recipe_passed"] == EXPECTED_REFERENCE_RECIPE_COUNT
        and reference["negative_probe_passed"] == EXPECTED_NEGATIVE_PROBE_COUNT
        and generated["passed"] == EXPECTED_GENERATED_CASE_COUNT
        and generated["unseen_passed"] == generated["unseen_denominator"] == 8
        and not generated["unseen_reference_task_id_overlap"]
        and mutants["passed"] == EXPECTED_COMPILE_MUTANT_COUNT
        and modules["passed"] == EXPECTED_MODULE_PROBE_COUNT
        and interfaces["passed"] == EXPECTED_INTERFACE_PATH_COUNT
        and leakage_count == 0
        and missing_receipt_count == 0
        and not failure_counts
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "status": "passed" if overall_pass else "failed",
        "source_binding": binding,
        "claim_boundary": [
            "Deterministic virtual-instrument qualification only.",
            "Finite v1 component and interface coverage; not exhaustive task-space validation.",
            "No physical-laboratory external-validity or agent-intelligence claim.",
        ],
        "counting_rule": {
            "reference_independent_unit": "registered task by registered world seed",
            "generated_independent_unit": "frozen generated composition case",
            "statistics": "deterministic exact counts only",
        },
        "summary": {
            "reference_units": {
                "passed": reference["unit_passed"],
                "denominator": reference["unit_denominator"],
            },
            "reference_recipes": {
                "passed": reference["recipe_passed"],
                "denominator": reference["recipe_denominator"],
            },
            "negative_probes": {
                "passed": reference["negative_probe_passed"],
                "denominator": reference["negative_probe_denominator"],
            },
            "generated_compositions": {
                "passed": generated["passed"],
                "denominator": generated["denominator"],
            },
            "unseen_distillation_compositions": {
                "passed": generated["unseen_passed"],
                "denominator": generated["unseen_denominator"],
            },
            "compile_mutants": {"passed": mutants["passed"], "denominator": mutants["denominator"]},
            "module_probes": {"passed": modules["passed"], "denominator": modules["denominator"]},
            "interface_paths": {
                "passed": interfaces["passed"],
                "denominator": interfaces["denominator"],
            },
            "public_private_leakage_count": leakage_count,
            "missing_receipt_count": missing_receipt_count,
            "failure_class_counts": failure_counts,
        },
        "task_structure": task_structure,
        "receipt_completeness": completeness,
        "reference_qualification": reference,
        "generated_qualification": generated,
        "compile_mutants": mutants,
        "module_qualification": modules,
        "interface_qualification": interfaces,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = [
        ("Registered task/world units", summary["reference_units"]),
        ("Valid midpoint/boundary/category recipes", summary["reference_recipes"]),
        ("Negative runtime probes", summary["negative_probes"]),
        ("Generated compositions", summary["generated_compositions"]),
        ("Frozen unseen reaction--distillation cases", summary["unseen_distillation_compositions"]),
        ("Compile-time mutants", summary["compile_mutants"]),
        ("Module probes", summary["module_probes"]),
        ("Cross-component interface paths", summary["interface_paths"]),
    ]
    lines = [
        "# First-paper composition qualification",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        "## Exact deterministic counts",
        "",
        "| Quantity | Passed | Denominator |",
        "| --- | ---: | ---: |",
        *[f"| {label} | {value['passed']} | {value['denominator']} |" for label, value in rows],
        "",
        f"Public/private leakage findings: `{summary['public_private_leakage_count']}`.",
        f"Missing receipts: `{summary['missing_receipt_count']}`.",
        "",
        "## Generated pattern matrix",
        "",
        "| Pattern | Passed | Denominator |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['pattern']} | {row['passed']} | {row['denominator']} |"
        for row in report["generated_qualification"]["pattern_matrix"]
    )
    lines.extend(["", "## Composition depth and runtime envelope", ""])
    for field, depth_rows in report["generated_qualification"]["depth_summary"].items():
        lines.extend(
            [
                f"### `{field}`",
                "",
                "| Value | Passed | Denominator | Mean elapsed (s) | Mean bytes |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            (
                f"| {row[field]} | {row['passed']} | {row['denominator']} | "
                f"{row['elapsed_s']['mean']:.6f} | "
                f"{row['trajectory_bytes']['mean']:.1f} |"
            )
            for row in depth_rows
        )
        lines.append("")
    lines.extend(
        [
            "## Module reference boundary",
            "",
            "| Module | Low/high classification | Maximum absolute error | Tolerance |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for probe in report["module_qualification"]["probes"]:
        if probe["probe_id"] != "legal_low_high":
            continue
        fixtures = [
            probe["receipt"][boundary].get("reference_fixture") for boundary in ("low", "high")
        ]
        numerical = [fixture for fixture in fixtures if isinstance(fixture, dict)]
        max_error = (
            f"{max(float(fixture['absolute_error']) for fixture in numerical):.6g}"
            if numerical
            else "n/a"
        )
        tolerance = (
            f"{max(float(fixture['tolerance']) for fixture in numerical):.6g}"
            if numerical
            else "n/a"
        )
        lines.append(
            f"| {probe['module_id']} | {probe['classification']} | {max_error} | {tolerance} |"
        )
    lines.extend(
        [
            "",
            "## Interface receipts",
            "",
            "| Path | Cases | Named checks passed |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {path['path_id']} | {path['case_count']} | "
        f"{sum(bool(value) for value in path['checks'].values())}/{len(path['checks'])} |"
        for path in report["interface_qualification"]["paths"]
    )
    completeness = report["receipt_completeness"]
    lines.extend(
        [
            "",
            "## Receipt completeness",
            "",
            f"Status: `{'passed' if completeness['passed'] else 'failed'}`; "
            f"errors: `{completeness['error_count']}`.",
        ]
    )
    if completeness["errors"]:
        lines.extend(
            f"- `{row['surface']}/{row['identity']}`: {row['requirement']}"
            for row in completeness["errors"]
        )
    lines.extend(["", "## Failure classes", ""])
    failures = summary["failure_class_counts"]
    if failures:
        lines.extend(f"- `{name}`: {count}" for name, count in failures.items())
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(report: dict[str, Any], *, output_path: str | Path) -> tuple[Path, Path]:
    report_path = Path(output_path)
    markdown_path = report_path.with_suffix(".md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report_path, markdown_path


__all__ = [
    "EXPECTED_COMPILE_MUTANT_COUNT",
    "EXPECTED_INTERFACE_PATH_COUNT",
    "EXPECTED_MODULE_PROBE_COUNT",
    "EXPECTED_NEGATIVE_PROBE_COUNT",
    "EXPECTED_REFERENCE_RECIPE_COUNT",
    "EXPECTED_REFERENCE_UNIT_COUNT",
    "PROTOCOL",
    "QUALIFICATION_ID",
    "REPORT_SCHEMA_VERSION",
    "build_interface_receipts",
    "build_report",
    "build_task_structure_baseline",
    "render_markdown",
    "run_compile_mutants",
    "run_generated_qualification",
    "run_module_probes",
    "validate_launch_preconditions",
    "write_outputs",
]
