"""Fixed-step parameter responses of the original finite gas-network advance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from math import exp, isclose, isfinite
from typing import Any

from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.gas_network_energy import gas_link_heat_key, gas_loss_key
from chemworld.physchem.gas_reactor_network import (
    advance_gas_reactor_network,
    gas_bath_key,
    gas_enthalpy_key,
    gas_node_pressure,
    gas_node_view,
    gas_transfer_key,
)
from chemworld.physchem.gas_thermochemistry_parameters import (
    GasThermochemistryParameter,
    species_enthalpy_parameter,
    with_species_enthalpy,
)
from chemworld.physchem.reaction_rate_laws import (
    positive_reaction_parameter,
    with_reaction_parameter,
)


@dataclass(frozen=True)
class GasSensitivityParameter:
    kind: str
    owner_id: str
    parameter_name: str

    def __post_init__(self) -> None:
        allowed = {
            "reaction": {"k", "A", "A_reverse"},
            "flow": {"coefficient"},
            "heat_link": {"conductance_W_K"},
            "loss": {"loss_conductance_W_K"},
        }
        if self.kind not in allowed or self.parameter_name not in allowed[self.kind]:
            raise ValueError("unsupported gas-network sensitivity parameter")
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise ValueError("sensitivity parameter requires a named original owner")

    @property
    def parameter_id(self) -> str:
        return json.dumps([self.kind, self.owner_id, self.parameter_name], separators=(",", ":"))


@dataclass(frozen=True)
class GasSensitivityPolicy:
    log_step: float = 0.01
    max_advances: int = 1000

    def __post_init__(self) -> None:
        if (
            isinstance(self.log_step, bool)
            or not isfinite(self.log_step)
            or not 0 < self.log_step <= 0.1
        ):
            raise ValueError("log step must be positive finite and at most .1")
        if type(self.max_advances) is not int or self.max_advances < 1:
            raise ValueError("advance budget must be a positive integer")


def _parameter(
    spec: Any,
    network: Any,
    parameter: GasSensitivityParameter | GasThermochemistryParameter,
    value: float | None = None,
) -> tuple[Any, Any, float]:
    if isinstance(parameter, GasThermochemistryParameter):
        original = species_enthalpy_parameter(network, parameter)
        return (
            spec,
            network if value is None else with_species_enthalpy(network, parameter, value),
            original,
        )
    if parameter.kind == "reaction":
        reaction = next((r for r in network.reactions if r.reaction_id == parameter.owner_id), None)
        if (
            reaction is None
            or reaction.rate_law.uses_activities
            or reaction.rate_law.equation_id
            not in {
                "mass_action",
                "arrhenius",
                "modified_arrhenius",
                "reversible_arrhenius",
                "third_body_arrhenius",
            }
        ):
            raise ValueError("reaction response requires an explicit concentration-based prefactor")
        original = positive_reaction_parameter(
            network, parameter.owner_id, parameter.parameter_name
        )
        return (
            spec,
            (
                network
                if value is None
                else with_reaction_parameter(
                    network, parameter.owner_id, parameter.parameter_name, value
                )
            ),
            original,
        )
    attribute, identity = {
        "flow": ("connections", "connection_id"),
        "heat_link": ("heat_links", "connection_id"),
        "loss": ("energy_nodes", "vessel_id"),
    }[parameter.kind]
    entries = getattr(spec, attribute)
    matches = [item for item in entries if getattr(item, identity) == parameter.owner_id]
    if len(matches) != 1:
        raise ValueError("sensitivity parameter must name exactly one original owner")
    original = getattr(matches[0], parameter.parameter_name)
    if isinstance(original, bool) or not isfinite(original) or original <= 0:
        raise ValueError("log sensitivity requires a finite strictly positive parameter")
    if value is not None:
        entries = tuple(
            replace(item, **{parameter.parameter_name: value}) if item is matches[0] else item
            for item in entries
        )
        spec = replace(spec, **{attribute: entries})
    return spec, network, original


def _parameter_unit(
    spec: Any, network: Any, parameter: GasSensitivityParameter | GasThermochemistryParameter
) -> str:
    if isinstance(parameter, GasThermochemistryParameter):
        return "J/mol"
    if parameter.kind in {"heat_link", "loss"}:
        return "W/K"
    if parameter.kind == "flow":
        edge = next(e for e in spec.connections if e.connection_id == parameter.owner_id)
        return {"molar-flow": "mol/s", "mass-flow": "kg/s", "pressure-valve": "kg/(s Pa)"}[
            edge.mode
        ]
    reaction = next(r for r in network.reactions if r.reaction_id == parameter.owner_id)
    reverse = parameter.parameter_name == "A_reverse"
    orders = reaction.kinetic_reverse_orders if reverse else reaction.kinetic_forward_orders
    order = sum(orders.values()) + int(reaction.rate_law.equation_id == "third_body_arrhenius")
    exponent = (
        0.0
        if parameter.parameter_name == "k"
        else float(reaction.rate_law.parameters.get("b_reverse" if reverse else "b", 0))
    )
    return f"(mol/L)^({1 - order:g}) / s / K^({exponent:g})"


def gas_sensitivity_observables(
    spec: Any, network: Any, state: Any, origin: Any
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """Physical amounts/temperatures and receipt increments from one fixed origin."""
    values: dict[str, float] = {}
    units: dict[str, str] = {}

    def put(key: str, value: float, unit: str) -> None:
        if not isfinite(value):
            raise ValueError("nonfinite gas-network observable")
        values[key], units[key] = float(value), unit

    def receipt(key: str) -> float:
        return state.coordinates.get(key, 0.0) - origin.coordinates.get(key, 0.0)

    thermal = IdealGasThermodynamics(network.imported_gas_thermochemistry)
    pressures = {}
    for node in spec.nodes:
        name = node.vessel_id
        view, initial = gas_node_view(state, node, network), gas_node_view(origin, node, network)
        for species in network.species_ids:
            put(f"node:{name}:amount:{species}", view.amounts_mol[species], "mol")
        put(f"node:{name}:temperature", view.temperature_K, "K")
        pressure = gas_node_pressure(node, sum(view.amounts_mol.values()), view.temperature_K)
        pressures[name] = pressure
        put(f"node:{name}:pressure", pressure, "Pa")
        put(f"node:{name}:bath", receipt(gas_bath_key(name)), "J")
        put(f"node:{name}:loss", receipt(gas_loss_key(name)), "J")
        conducted = sum(
            (int(link.target_vessel_id == name) - int(link.source_vessel_id == name))
            * receipt(gas_link_heat_key(link.connection_id))
            for link in spec.heat_links
        )
        put(f"node:{name}:conducted", conducted, "J")
        energy = thermal.internal_energy_j(view) - thermal.internal_energy_j(initial)
        support = next(
            (
                item.support_heat_capacity_J_K
                for item in spec.energy_nodes
                if item.vessel_id == name
            ),
            0.0,
        )
        energy += support * (view.temperature_K - initial.temperature_K)
        put(f"node:{name}:delta_U", energy, "J")
    for edge in spec.connections:
        for species in network.species_ids:
            put(
                f"edge:{edge.connection_id}:amount:{species}",
                receipt(gas_transfer_key(edge.connection_id, species)),
                "mol",
            )
        put(
            f"edge:{edge.connection_id}:enthalpy",
            receipt(gas_enthalpy_key(edge.connection_id)),
            "J",
        )
    for link in spec.heat_links:
        put(f"link:{link.connection_id}:heat", receipt(gas_link_heat_key(link.connection_id)), "J")
    regimes = {}
    for edge in spec.connections:
        if edge.mode == "pressure-valve":
            delta = pressures[edge.source_vessel_id] - pressures[edge.target_vessel_id]
            regimes[edge.connection_id] = (
                "disabled"
                if not edge.enabled
                else ("open" if delta > 0 else "closed" if delta < 0 else "kink")
            )
    return values, units, regimes


def gas_network_trajectory_sensitivities(
    spec: Any,
    network: Any,
    state: Any,
    parameters: Any,
    times_s: Any,
    *,
    thermal_device: Any = None,
    policy: GasSensitivityPolicy | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Run 1+4P original trajectories; no analytical or differentiability claim.

    Times are offsets from the current state's clock. No event extrapolation or
    post-event fixed-time sensitivity is allowed. Returned numerical work and
    receipts belong to hypothetical analyses, never live laboratory charges.
    """
    policy = policy or GasSensitivityPolicy()
    if not isinstance(policy, GasSensitivityPolicy):
        raise ValueError("explicit GasSensitivityPolicy required")
    if spec.surfaces:
        raise ValueError("surface-network sensitivity requires declared slot response observables")
    if spec.spatial_transport is not None:
        raise ValueError("spatial networks are outside this finite-network sensitivity scope")
    parameters = tuple(parameters)
    if not parameters or any(
        not isinstance(p, (GasSensitivityParameter, GasThermochemistryParameter))
        for p in parameters
    ):
        raise ValueError("declare at least one gas sensitivity or thermochemistry parameter")
    if len({p.parameter_id for p in parameters}) != len(parameters):
        raise ValueError("sensitivity parameters must be unique")
    times = tuple(times_s)
    if (
        not times
        or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not isfinite(t) or t < 0
            for t in times
        )
        or any(a >= b for a, b in pairwise(times))
    ):
        raise ValueError("time offsets must be finite, nonnegative and strictly increasing")
    times = tuple(float(t) for t in times)
    if any(
        not isfinite(state.time_s + t) or (t > 0 and state.time_s + t == state.time_s)
        for t in times
    ):
        raise ValueError("analysis clock interval is unrepresentable")
    _, units, _ = gas_sensitivity_observables(spec, network, state, state)
    candidates: list[tuple[str, str | None, float, Any, Any]] = [
        ("baseline", None, 0.0, spec, network)
    ]
    bindings = []
    for parameter in parameters:
        _, _, value = _parameter(spec, network, parameter)
        additive = isinstance(parameter, GasThermochemistryParameter)
        bindings.append(
            {
                **asdict(parameter),
                "parameter_id": parameter.parameter_id,
                "baseline_value": value,
                "parameter_unit": _parameter_unit(spec, network, parameter),
                **(
                    {"coordinate": "additive_species_enthalpy", "step_J_mol": parameter.step_J_mol}
                    if isinstance(parameter, GasThermochemistryParameter)
                    else {}
                ),
            }
        )
        full_step = (
            parameter.step_J_mol
            if isinstance(parameter, GasThermochemistryParameter)
            else policy.log_step
        )
        for label, step in (
            ("plus", full_step),
            ("minus", -full_step),
            ("plus_half", full_step / 2),
            ("minus_half", -full_step / 2),
        ):
            varied = value + step if additive else value * exp(step)
            if not isfinite(varied) or (not additive and varied <= 0) or varied == value:
                raise ValueError("parameter perturbation is unrepresentable")
            altered_spec, altered_network, _ = _parameter(spec, network, parameter, varied)
            candidates.append(
                (
                    parameter.parameter_id + ":" + label,
                    parameter.parameter_id,
                    step,
                    altered_spec,
                    altered_network,
                )
            )
    piecewise = any(e.enabled and e.mode == "pressure-valve" for e in spec.connections) or (
        thermal_device is not None and thermal_device.enabled
    )
    report: dict[str, Any] = {
        "schema_version": "chemworld-gas-network-sensitivity-1",
        "start_time_s": state.time_s,
        "times_s": list(times),
        "policy": asdict(policy),
        "parameters": bindings,
        "units": units,
        "trajectories": [],
        "responses": [],
        "failures": [],
        "method": "two central log finite differences of original native network trajectories",
        "smoothness": "piecewise_not_certified" if piecewise else "differentiability_not_certified",
        "error_estimate": "abs(D_half-D_full)/3; not a guaranteed error bound",
        "resource_scope": (
            "hypothetical original solver work; no live operations or resource charges"
        ),
        "work": {
            "planned_trajectories": len(candidates),
            "planned_time_slots": len(candidates) * len(times),
            "planned_advances": len(candidates) * sum(t > 0 for t in times),
            "attempted_advances": 0,
            "completed_advances": 0,
            "failed_advances": 0,
            "reported_nfev": 0,
            "reported_njev": 0,
            "reported_nlu": 0,
            "unknown_work_failures": 0,
        },
    }
    work = report["work"]
    additive_ids = {
        p.parameter_id for p in parameters if isinstance(p, GasThermochemistryParameter)
    }
    if additive_ids:
        report["method"] = (
            "two central differences: explicit additive species enthalpy; "
            "original positive log parameters"
        )
        report["thermochemistry_semantics"] = (
            "fixed initialized n/T; original source H/G integration constants shifted, "
            "Cp/S unchanged"
        )
    for candidate_id, parameter_id, step, candidate_spec, candidate_network in candidates:
        candidate: dict[str, Any] = {
            "candidate_id": candidate_id,
            "parameter_id": parameter_id,
            "log_offset": step,
            "status": "not_attempted",
            "stopping_event": None,
            "actual_end_time_s": state.time_s,
            "advances": [],
            "records": [
                {"offset_s": t, "status": "not_attempted", "values": None, "regimes": None}
                for t in times
            ],
        }
        if parameter_id in additive_ids:
            candidate["enthalpy_offset_J_mol"] = candidate.pop("log_offset")
        report["trajectories"].append(candidate)
        current, halted = state, None
        for record in candidate["records"]:
            offset = record["offset_s"]
            if halted:
                record.update(status="unavailable", reason=halted)
                continue
            target = state.time_s + offset
            if target > current.time_s:
                if work["attempted_advances"] >= policy.max_advances:
                    halted = "advance_budget_exhausted"
                    candidate["status"] = "budget_exhausted"
                    record.update(status="not_attempted", reason=halted)
                    continue
                work["attempted_advances"] += 1
                attempted: dict[str, Any] = {
                    "start_time_s": current.time_s,
                    "requested_end_time_s": target,
                    "status": "attempted",
                    "solver": None,
                }
                candidate["advances"].append(attempted)
                try:
                    result, diagnostics = advance_gas_reactor_network(
                        candidate_spec,
                        candidate_network,
                        current,
                        target - current.time_s,
                        thermal_device=thermal_device,
                    )
                    current = result.final_state
                    attempted.update(
                        status="completed",
                        actual_end_time_s=current.time_s,
                        solver=dict(result.solver_diagnostic),
                        diagnostics=diagnostics,
                    )
                    for field in ("nfev", "njev", "nlu"):
                        work["reported_" + field] += result.solver_diagnostic.get(field, 0) or 0
                    candidate["actual_end_time_s"] = current.time_s
                    if result.stopping_event is not None:
                        candidate.update(status="stopped", stopping_event=result.stopping_event)
                        halted = "native_stopping_event"
                        record.update(
                            status="unavailable",
                            reason=halted,
                            stopped_at_time_s=current.time_s,
                            endpoint_values=gas_sensitivity_observables(
                                candidate_spec, candidate_network, current, state
                            )[0],
                        )
                        work["completed_advances"] += 1
                        continue
                    if not isclose(current.time_s, target, rel_tol=1e-12, abs_tol=1e-12):
                        raise RuntimeError("original advance did not reach the requested time")
                    work["completed_advances"] += 1
                except (ValueError, ArithmeticError, RuntimeError) as error:
                    attempted.update(status="failed", failure=f"{type(error).__name__}: {error}")
                    work["failed_advances"] += 1
                    if attempted["solver"] is None:
                        work["unknown_work_failures"] += 1
                    report["failures"].append(
                        {
                            "candidate_id": candidate_id,
                            "offset_s": offset,
                            "failure": attempted["failure"],
                        }
                    )
                    candidate["status"] = "failed"
                    halted = "native_forward_failed"
                    record.update(status="unavailable", reason=halted)
                    continue
                finally:
                    if progress is not None:
                        progress(
                            {
                                "candidate": candidate_id,
                                "attempted_advances": work["attempted_advances"],
                                "planned_advances": work["planned_advances"],
                                "offset_s": offset,
                                "status": attempted["status"],
                            }
                        )
            values, _, regimes = gas_sensitivity_observables(
                candidate_spec, candidate_network, current, state
            )
            if thermal_device is not None and thermal_device.enabled:
                temperature = gas_node_view(
                    current,
                    next(n for n in spec.nodes if n.vessel_id == spec.anchor_vessel_id),
                    network,
                ).temperature_K
                demand = (
                    thermal_device.conductance_W_K
                    * thermal_device.calibration_factor
                    * (thermal_device.setpoint_K - temperature)
                )
                regimes["anchor_thermal_device"] = (
                    "high_limit"
                    if demand >= thermal_device.maximum_power_W
                    else (
                        "low_limit"
                        if demand <= -thermal_device.maximum_cooling_power_W
                        else "interior"
                    )
                )
            record.update(status="complete", values=values, regimes=regimes)
        if halted is None:
            candidate["status"] = "complete"
    base = report["trajectories"][0]
    for index, binding in enumerate(bindings):
        quartet = report["trajectories"][1 + 4 * index : 5 + 4 * index]
        response = {"parameter_id": binding["parameter_id"], "records": []}
        report["responses"].append(response)
        for position, offset in enumerate(times):
            records = [base["records"][position], *(c["records"][position] for c in quartet)]
            row: dict[str, Any] = {
                "offset_s": offset,
                "status": "unavailable",
                "coefficients": None,
            }
            response["records"].append(row)
            if any(record["status"] != "complete" for record in records):
                row["reason"] = "incomplete_or_event_limited_candidates"
                continue
            sampled_regimes = [record["regimes"] for record in records]
            branch_change = any(
                regime != sampled_regimes[0] for regime in sampled_regimes[1:]
            ) or any(value == "kink" for regime in sampled_regimes for value in regime.values())
            row.update(
                status="secant_only_branch_boundary"
                if branch_change
                else (
                    "finite_difference_piecewise_uncertified" if piecewise else "finite_difference"
                ),
                coefficients={},
            )
            for key in units:
                y, plus, minus, half_plus, half_minus = [
                    record["values"][key] for record in records
                ]
                if binding["parameter_id"] in additive_ids:
                    step = binding["step_J_mol"]
                    coarse, fine = (plus - minus) / (2 * step), (half_plus - half_minus) / step
                    row["coefficients"][key] = {
                        "baseline": y,
                        "coarse_dy_dp": coarse,
                        "fine_dy_dp": fine,
                        "dy_dp": fine,
                        "normalized": None
                        if y == 0 or binding["baseline_value"] == 0
                        else fine * binding["baseline_value"] / y,
                        "estimated_discretization_error": abs(fine - coarse) / 3,
                        "unit_dy_dp": f"{units[key]} / (J/mol)",
                    }
                    continue
                coarse = (plus - minus) / (2 * policy.log_step)
                fine = (half_plus - half_minus) / policy.log_step
                row["coefficients"][key] = {
                    "baseline": y,
                    "coarse_dy_dlnp": coarse,
                    "fine_dy_dlnp": fine,
                    "dy_dp": fine / binding["baseline_value"],
                    "normalized": None if y == 0 else fine / y,
                    "estimated_discretization_error": abs(fine - coarse) / 3,
                    "unit_dy_dlnp": units[key],
                    "unit_dy_dp": f"{units[key]} / ({binding['parameter_unit']})",
                }
    work["completed_trajectories"] = sum(c["status"] == "complete" for c in report["trajectories"])
    work["evaluated_time_slots"] = sum(
        r["status"] == "complete" for c in report["trajectories"] for r in c["records"]
    )
    work["unavailable_time_slots"] = work["planned_time_slots"] - work["evaluated_time_slots"]
    work["planned_response_slots"] = len(parameters) * len(times)
    work["evaluated_response_slots"] = sum(
        r["coefficients"] is not None
        for response in report["responses"]
        for r in response["records"]
    )
    report["status"] = (
        "complete_finite_difference"
        if work["completed_trajectories"] == len(candidates)
        else "incomplete_finite_difference"
    )
    if piecewise and report["status"] == "complete_finite_difference":
        report["status"] = "piecewise_finite_difference_uncertified"
    json.dumps(report, allow_nan=False)
    return report


__all__ = [
    "GasSensitivityParameter",
    "GasSensitivityPolicy",
    "gas_network_trajectory_sensitivities",
]
