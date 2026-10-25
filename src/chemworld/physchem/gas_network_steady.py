"""Read-only steady roots of the original connected gas/surface/energy equations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from fractions import Fraction
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.catalysis_network import catalytic_inventory_key
from chemworld.physchem.coupled_system import assemble_coupled_rates
from chemworld.physchem.gas_network_energy import gas_temperature_key
from chemworld.physchem.gas_network_surface import GasSurfaceContext, gas_phase_network
from chemworld.physchem.gas_reactor_network import gas_inventory_key, gas_reactor_network_system
from chemworld.physchem.residual_system import (
    DEFAULT_ROOT_POLICY,
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)
from chemworld.physchem.stoichiometry import _rref


def _geometry(
    spec: Any, network: Any, surface: Any, keys: Sequence[str], metric: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original reaction and actual enabled transport stoichiometry, not a fitted Jacobian."""
    gas = gas_phase_network(network)
    columns: list[dict[str, float]] = []
    for node in spec.nodes:
        if node.chemistry_enabled:
            columns.extend(
                {gas_inventory_key(node.vessel_id, s): v for s, v in r.stoichiometry.items()}
                for r in gas.reactions
            )
        if surface is not None and node.vessel_id in surface.attachments:
            attachment = surface.attachments[node.vessel_id]
            slot = surface.slots[node.vessel_id]
            columns.extend(
                {
                    catalytic_inventory_key(slot, s)
                    if s in attachment.mechanism.adsorbates
                    else gas_inventory_key(node.vessel_id, s): v
                    for s, v in r.stoichiometry.items()
                }
                for r in attachment.mechanism.reactions
            )
    for edge in spec.connections:
        if edge.enabled and edge.coefficient > 0:
            columns.extend(
                {
                    gas_inventory_key(edge.source_vessel_id, s): -1,
                    gas_inventory_key(edge.target_vessel_id, s): 1,
                }
                for s in gas.species_ids
            )
    matrix = np.asarray([[c.get(k, 0.0) for c in columns] for k in keys], dtype=np.float64)
    _, pivots = _rref([[Fraction(float(v)) for v in row] for row in matrix], width=len(columns))
    rank = len(pivots)
    weighted = matrix / metric[:, None]
    vectors, singular, _ = np.linalg.svd(weighted, full_matrices=True)
    if rank and singular[rank - 1] <= 64 * max(weighted.shape) * np.finfo(float).eps * singular[0]:
        raise ValueError("original exact stoichiometric rank is unresolved in the specified metric")
    return vectors[:, :rank], vectors[:, rank:].T, singular


def _thermal_components(spec: Any) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {n.vessel_id: set() for n in spec.nodes}
    for edge in (*spec.connections, *spec.heat_links):
        if (hasattr(edge, "enabled") and edge.enabled and edge.coefficient > 0) or (
            hasattr(edge, "conductance_W_K") and edge.conductance_W_K > 0
        ):
            adjacency[edge.source_vessel_id].add(edge.target_vessel_id)
            adjacency[edge.target_vessel_id].add(edge.source_vessel_id)
    groups = []
    unseen = set(adjacency)
    while unseen:
        pending, group = [min(unseen)], set()
        while pending:
            name = pending.pop()
            if name not in group:
                group.add(name)
                pending.extend(adjacency[name] - group)
        unseen -= group
        groups.append(sorted(group))
    return groups


def gas_network_steady_state(
    spec: Any,
    network: Any,
    state: Any,
    *,
    amount_scales_mol: Mapping[str, float],
    held_vessel_ids: Sequence[str] = (),
    temperature_scale_K: float = 300.0,
    thermal_device: Any = None,
    surface_slots: Sequence[Any] = (),
    root_policy: RootPolicy = DEFAULT_ROOT_POLICY,
    derivative_step: float = 6e-6,
    material_tolerance_mol_s: float = 1e-9,
    temperature_tolerance_K_s: float = 1e-7,
    maximum_unknowns: int = 128,
    maximum_rate_evaluations: int = 20000,
    progress: Any = None,
) -> dict[str, Any]:
    """Find a local root without advancing the original world.

    Held vessels explicitly define a hypothetical boundary at their current n/T;
    their actual unheld derivatives remain in the report. Finite source/receiver
    inventories are never silently converted to live infinite reservoirs.
    """
    if any(
        x is not None
        for x in (
            spec.spatial_transport,
            spec.pressure_control,
            spec.thermal_wall,
            spec.shaft_machine,
        )
    ):
        raise ValueError(
            "steady query currently uses fixed geometry gas networks and fixed surfaces"
        )
    if not isinstance(root_policy, RootPolicy):
        raise ValueError("steady query requires the shared RootPolicy")
    if (
        any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (
                temperature_scale_K,
                derivative_step,
                material_tolerance_mol_s,
                temperature_tolerance_K_s,
            )
        )
        or type(maximum_unknowns) is not int
        or maximum_unknowns < 1
        or type(maximum_rate_evaluations) is not int
        or maximum_rate_evaluations < 1
    ):
        raise ValueError("steady scales, tolerances and dimension budget must be positive finite")
    nodes = {n.vessel_id: n for n in spec.nodes}
    held = tuple(held_vessel_ids)
    if len(set(held)) != len(held) or set(held) - set(nodes) or len(held) == len(nodes):
        raise ValueError("held boundary must name distinct existing vessels and leave active nodes")
    if any(nodes[k].chemistry_enabled for k in held) or any(
        a.vessel_id in held for a in spec.surfaces
    ):
        raise ValueError(
            "held boundary vessels must be explicitly nonreacting and without surfaces"
        )
    active = sorted(set(nodes) - set(held))
    surface = GasSurfaceContext(spec, network, surface_slots) if spec.surfaces else None
    gas = gas_phase_network(network)
    owners = {gas_inventory_key(n, s): n for n in nodes for s in gas.species_ids}
    if surface:
        owners.update({key: n for n, row in surface.identities.items() for key in row.values()})
    system = gas_reactor_network_system(
        spec, network, thermal_device=thermal_device, surface_slots=surface_slots
    )
    system.validate_state(state)
    keys = tuple(k for k in system.thermodynamics.inventory_keys if owners[k] in active)
    if set(amount_scales_mol) != set(keys) or any(
        isinstance(v, bool) or not isfinite(v) or v <= 0 for v in amount_scales_mol.values()
    ):
        raise ValueError(
            "positive mol scales must cover exactly the active original material coordinates"
        )
    metric = np.asarray([amount_scales_mol[k] for k in keys], dtype=float)
    initial = np.asarray([state.amounts_mol[k] for k in keys])
    basis, conserved, singular = _geometry(spec, network, surface, keys, metric)
    rank = basis.shape[1]
    thermal_names = active if spec.energy_nodes else []
    energy = {e.vessel_id: e for e in spec.energy_nodes}

    def temperature(current: Any, name: str) -> float:
        return float(
            current.temperature_K
            if name == spec.anchor_vessel_id
            else current.coordinates[gas_temperature_key(name)]
        )

    def component_u(current: Any, names: Sequence[str]) -> float:
        thermo: Any = system.thermodynamics
        return float(
            sum(
                thermo.local_models[n].internal_energy_j(thermo.node_view(current, nodes[n]))
                for n in names
            )
        )

    isolated: list[dict[str, Any]] = []
    if thermal_names:
        for group in _thermal_components(spec):
            if (
                not set(group) & set(held)
                and all(energy[n].loss_conductance_W_K == 0 for n in group)
                and not (
                    spec.anchor_vessel_id in group
                    and thermal_device is not None
                    and thermal_device.enabled
                )
            ):
                isolated.append(
                    {"nodes": group, "row": group[0], "initial_U_J": component_u(state, group)}
                )
    if rank + len(thermal_names) > maximum_unknowns:
        raise ValueError("steady unknown count exceeds its declared budget")
    variables = [ResidualVariable(f"reachable_amount:{i}", "1", 1) for i in range(rank)]
    equations = [ResidualEquation(f"reachable_rate:{i}", "1/s", 1) for i in range(rank)]
    for name in thermal_names:
        variables.append(
            ResidualVariable(
                "temperature:" + name,
                "K",
                temperature_scale_K,
                lower=energy[name].minimum_temperature_K,
                upper=energy[name].maximum_temperature_K,
            )
        )
        conserved_group = next((g for g in isolated if g["row"] == name), None)
        equations.append(
            ResidualEquation(
                "energy:" + name,
                "J" if conserved_group else "K/s",
                max(1.0, abs(conserved_group["initial_U_J"]))
                if conserved_group
                else temperature_scale_K,
            )
        )
    guess = np.r_[np.zeros(rank), [temperature(state, n) for n in thermal_names]]
    calls = 0

    def candidate(vector: np.ndarray) -> Any:
        amounts = {
            **state.amounts_mol,
            **dict(zip(keys, initial + metric * (basis @ vector[:rank]), strict=True)),
        }
        if any(not isfinite(v) or v < 0 for v in amounts.values()):
            raise ResidualDomainError(
                "steady candidate has a negative/nonfinite original inventory"
            )
        coordinates = dict(state.coordinates)
        anchor_t = state.temperature_K
        for name, value in zip(thermal_names, vector[rank:], strict=True):
            if name == spec.anchor_vessel_id:
                anchor_t = float(value)
            else:
                coordinates[gas_temperature_key(name)] = float(value)
        current = replace(
            state, amounts_mol=amounts, temperature_K=anchor_t, coordinates=coordinates
        )
        try:
            system.validate_state(current)
            if any(e.value(current) < 0 for e in system.all_events):
                raise ValueError("steady candidate violates an original finite operating limit")
        except (ValueError, ArithmeticError) as error:
            raise ResidualDomainError(str(error)) from error
        return current

    def rates(current: Any) -> Any:
        nonlocal calls
        if calls >= maximum_rate_evaluations:
            raise RuntimeError("original network rate evaluation budget exhausted")
        calls += 1
        if progress is not None and (calls == 1 or calls % 100 == 0):
            progress({"stage": "original_network_steady_residual", "rate_evaluations": calls})
        return assemble_coupled_rates(
            current,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            heat_flux=system.heat_flux,
            ports=system.ports,
            energy_ports=system.energy_ports,
            isothermal=system.isothermal,
        )

    def t_rate(result: Any, name: str) -> float:
        return float(
            result.temperature_rate_K_s
            if name == spec.anchor_vessel_id
            else result.coordinate_rates[gas_temperature_key(name)]
        )

    def energy_rates(current: Any, result: Any) -> dict[str, float]:
        if not spec.energy_nodes:
            return {}
        thermo: Any = system.thermodynamics
        output = {}
        for name, node in nodes.items():
            model = thermo.local_models[name]
            local = thermo.node_view(current, node)
            mapping = (
                surface.identities[name]
                if surface
                else {s: gas_inventory_key(name, s) for s in gas.species_ids}
            )
            dn = {s: result.amount_rates_mol_s[k] for s, k in mapping.items()}
            output[name] = float(
                model.chemical_energy_rate_w(local, dn, ())
                + model.heat_capacity_j_k(local) * t_rate(result, name)
            )
        return output

    def residual(_time: float, vector: np.ndarray, _rate: Any, _parameters: Any) -> np.ndarray:
        current = candidate(vector)
        result = rates(current)
        values = list(basis.T @ (np.asarray([result.amount_rates_mol_s[k] for k in keys]) / metric))
        for name in thermal_names:
            group = next((g for g in isolated if g["row"] == name), None)
            values.append(
                component_u(current, group["nodes"]) - group["initial_U_J"]
                if group
                else t_rate(result, name)
            )
        return np.asarray(values)

    report: dict[str, Any] = {
        "schema": "gas-network-steady-v1",
        "scope": "held-boundary-internal-steady" if held else "closed-full-network-steady",
        "active_vessel_ids": active,
        "held_vessel_ids": list(held),
        "start_time_s": state.time_s,
        "initial_amounts_mol": dict(state.amounts_mol),
        "initial_temperature_K": state.temperature_K,
        "initial_coordinates": dict(state.coordinates),
        "active_material_keys": list(keys),
        "amount_scales_mol": dict(amount_scales_mol),
        "reachable_dimension": rank,
        "conservation_rows_scaled_amounts": conserved.tolist(),
        "initial_conserved_totals": (conserved @ (initial / metric)).tolist(),
        "stoichiometric_singular_values": singular.tolist(),
        "conserved_energy_components": isolated,
        "root_policy": asdict(root_policy),
        "status": "failed",
        "failure": None,
        "material_tolerance_mol_s": material_tolerance_mol_s,
        "temperature_tolerance_K_s": temperature_tolerance_K_s,
        "derivative_step": derivative_step,
        "maximum_unknowns": maximum_unknowns,
        "maximum_rate_evaluations": maximum_rate_evaluations,
        "world_advanced": False,
        "live_resources_charged": False,
        "interpretation": (
            "local root with explicit held boundary; no global uniqueness, "
            "stability or infinite reservoir simulation"
        ),
    }
    try:
        if variables:
            system_r = ResidualSystem(
                "original-gas-network-steady",
                tuple(variables),
                tuple(equations),
                residual,
                ResidualDerivatives("finite_difference", relative_step=derivative_step),
            )
            solved = solve_steady_residual(system_r, guess, time_s=state.time_s, policy=root_policy)
            report.update(
                root_diagnostic=solved.root.diagnostics, derivative_diagnostic=solved.diagnostics
            )
            if not solved.root.success:
                report["failure"] = solved.root.failure_reason
                return report
            final = candidate(solved.root.value)
        else:
            final = candidate(guess)
        final_rates = rates(final)
        du = energy_rates(final, final_rates)
        report.update(
            amounts_mol=dict(final.amounts_mol),
            temperature_K=final.temperature_K,
            coordinates=dict(final.coordinates),
            material_rates_mol_s=dict(final_rates.amount_rates_mol_s),
            temperature_rates_K_s={n: t_rate(final_rates, n) for n in nodes}
            if spec.energy_nodes
            else dict.fromkeys(nodes, 0.0),
            original_receipt_rates=dict(final_rates.coordinate_rates),
            heater_W=final_rates.heater_W,
            loss_W=final_rates.loss_W,
            original_internal_energy_rates_W=du,
            energy_residual_W=max((abs(du[n]) for n in thermal_names), default=0.0),
            material_residual_mol_s=max(
                (abs(final_rates.amount_rates_mol_s[k]) for k in keys), default=0.0
            ),
            temperature_residual_K_s=max(
                (abs(t_rate(final_rates, n)) for n in thermal_names), default=0.0
            ),
            conservation_error=(
                conserved @ ((np.asarray([final.amounts_mol[k] for k in keys]) - initial) / metric)
            ).tolist(),
            energy_invariant_errors_J=[
                component_u(final, g["nodes"]) - g["initial_U_J"] for g in isolated
            ],
        )
        report["status"] = (
            "steady_state_found"
            if report["material_residual_mol_s"] <= material_tolerance_mol_s
            and report["temperature_residual_K_s"] <= temperature_tolerance_K_s
            else "failed"
        )
        if report["status"] == "failed":
            report["failure"] = "full_original_active_residual_exceeds_tolerance"
    except (ValueError, RuntimeError, ArithmeticError) as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    finally:
        report["original_rate_evaluations"] = calls
    return report
