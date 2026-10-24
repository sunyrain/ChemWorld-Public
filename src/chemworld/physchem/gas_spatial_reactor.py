"""Original coupled solver for isothermal, pressure-equilibrated gas grids."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.physchem.coupled_system import (
    DEFAULT_COUPLED_ODE_POLICY,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledStateView,
    CoupledSystem,
    InternalTransfer,
    ProcessContribution,
)
from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    _key,
    gas_bath_key,
    gas_enthalpy_key,
    gas_inventory_key,
    gas_material_topology,
    gas_network_coordinates,
    gas_node_pressure,
    gas_node_view,
    gas_transfer_key,
)
from chemworld.physchem.gas_spatial_transport import spatial_gas_faces
from chemworld.physchem.gas_transport import gas_transport_response
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_solvers import reaction_network_process


def spatial_bulk_key(face: str) -> str:
    return _key("spatial_bulk_mol", face)


def spatial_diffusion_key(face: str, species: str) -> str:
    return _key("spatial_diffusion_mol", face, species)


def spatial_diffusion_heat_key(face: str) -> str:
    return _key("spatial_diffusion_H_J", face)


def spatial_coordinates(
    spec: GasReactorNetworkSpec, network: ReactionNetworkSpec
) -> tuple[CoupledCoordinate, ...]:
    spatial = spec.spatial_transport
    if spatial is None:
        raise ValueError("missing spatial gas declaration")
    return tuple(
        c
        for face, _, _ in gas_material_topology(spec)
        for c in (
            CoupledCoordinate(spatial_bulk_key(face), "mol", 0.001),
            CoupledCoordinate(spatial_diffusion_heat_key(face), "J", 1),
            *(
                CoupledCoordinate(spatial_diffusion_key(face, s), "mol", 0.001)
                for s in network.species_ids
            ),
        )
    )


def spatial_rates(
    spec: GasReactorNetworkSpec, network: ReactionNetworkSpec, state: CoupledStateView
) -> tuple[dict[str, ProcessContribution], dict[str, Any]]:
    spatial = spec.spatial_transport
    if spatial is None:
        raise ValueError("missing spatial gas declaration")
    nodes = {n.vessel_id: n for n in spec.nodes}
    reaction = reaction_network_process(
        network, species_thermo=network.imported_gas_thermochemistry
    )
    views = {name: gas_node_view(state, node, network) for name, node in nodes.items()}
    chemistry = {
        name: reaction.evaluate(views[name])
        if nodes[name].chemistry_enabled
        else ProcessContribution()
        for name in nodes
    }
    boundary_rates = {}
    if spatial.boundary_mode == "finite-flow":
        from chemworld.physchem.gas_spatial_flow import spatial_boundary_rates

        inlet, outlet = spatial_boundary_rates(spec)
        boundary_rates = {"inlet_mol_s": inlet, "outlet_mol_s": outlet}
    packet = spatial_gas_faces(
        spatial,
        network,
        spec.temperature_K,
        [views[name].amounts_mol for name in spatial.cell_vessel_ids],
        [chemistry[name].amount_rates_mol_s for name in spatial.cell_vessel_ids],
        **boundary_rates,
    )
    if spatial.boundary_mode == "finite-flow":
        from chemworld.physchem.gas_spatial_flow import add_spatial_boundary_faces

        add_spatial_boundary_faces(spec, network, views, packet)
    return chemistry, packet


def gas_spatial_system(spec: GasReactorNetworkSpec, network: ReactionNetworkSpec) -> CoupledSystem:
    spatial = spec.spatial_transport
    if spatial is None:
        raise ValueError("missing spatial gas declaration")
    thermo = network.imported_gas_thermochemistry
    if set(thermo) != set(network.species_ids) or any(s.phase != "gas" for s in network.species):
        raise ValueError("spatial gas requires original gas species and declared thermochemistry")
    # Validate transport/species/T even when a one-cell mesh has no interior
    # face. Actual faces still evaluate at their current pressure/composition.
    gas_transport_response(
        spatial.transport,
        network,
        temperature_K=spec.temperature_K,
        pressure_Pa=min(100000.0, spatial.transport.maximum_pressure_Pa),
        mole_fractions=dict.fromkeys(network.species_ids, 1 / len(network.species_ids)),
    )
    local_thermal = IdealGasThermodynamics(thermo)
    thermal = IdealGasThermodynamics(
        {
            gas_inventory_key(n.vessel_id, s): thermo[s]
            for n in spec.nodes
            for s in network.species_ids
        }
    )
    internal = {
        s: local_thermal.species_energy_j_mol(s, spec.temperature_K) for s in network.species_ids
    }
    nodes = {n.vessel_id: n for n in spec.nodes}

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        chemistry, packet = spatial_rates(spec, network, state)
        counters: dict[str, float] = {}
        derivatives = {
            name: {s: c.amount_rates_mol_s.get(s, 0.0) for s in network.species_ids}
            for name, c in chemistry.items()
        }
        net_h = dict.fromkeys(nodes, 0.0)
        transfers = []
        for face, row in packet["faces"].items():
            left, right = row["source"], row["target"]
            for species, rate in row["amount_rates_mol_s"].items():
                source, target = (left, right) if rate >= 0 else (right, left)
                transfers.append(
                    InternalTransfer(
                        _key("spatial", face, species),
                        gas_inventory_key(source, species),
                        gas_inventory_key(target, species),
                        abs(rate),
                    )
                )
                derivatives[left][species] -= rate
                derivatives[right][species] += rate
                counters[gas_transfer_key(face, species)] = rate
                counters[spatial_diffusion_key(face, species)] = row["diffusion_mol_s"][species]
            net_h[left] -= row["enthalpy_W"]
            net_h[right] += row["enthalpy_W"]
            counters[gas_enthalpy_key(face)] = row["enthalpy_W"]
            counters[spatial_bulk_key(face)] = row["bulk_mol_s"]
            counters[spatial_diffusion_heat_key(face)] = row["diffusion_enthalpy_W"]
        for name, rates in derivatives.items():
            counters[gas_bath_key(name)] = (
                sum(internal[s] * v for s, v in rates.items()) - net_h[name]
            )
        return ProcessContribution(
            {
                gas_inventory_key(name, s): rate
                for name, c in chemistry.items()
                for s, rate in c.amount_rates_mol_s.items()
            },
            {
                _key("reaction", name, r): v
                for name, c in chemistry.items()
                for r, v in c.reaction_progress_mol_s.items()
            },
            tuple(transfers),
            coordinate_rates=counters,
        )

    def validate(state: CoupledStateView) -> None:
        if state.temperature_K != spec.temperature_K or not isclose(
            state.volume_L, sum(n.volume_L for n in spec.nodes), rel_tol=1e-12
        ):
            raise ValueError("spatial gas requires its declared fixed temperature and mesh volume")
        if set(state.coordinates) != {c.name for c in gas_network_coordinates(spec, network)}:
            raise ValueError("spatial gas accounting layout differs")
        cell_volume = sum(nodes[name].volume_L for name in spatial.cell_vessel_ids)
        total = (
            sum(state.amounts_mol.values())
            if spatial.boundary_mode == "sealed"
            else sum(
                sum(views.amounts_mol.values())
                for name in spatial.cell_vessel_ids
                for views in [gas_node_view(state, nodes[name], network)]
            )
        )
        for node in nodes.values():
            view = gas_node_view(state, node, network)
            n = sum(view.amounts_mol.values())
            if n <= 0 or n < node.minimum_amount_mol - 1e-14:
                raise ValueError("spatial gas below positive cell inventory floor")
            expected = (
                total
                * node.volume_L
                / (state.volume_L if spatial.boundary_mode == "sealed" else cell_volume)
            )
            if (
                node.vessel_id in spatial.cell_vessel_ids
                and abs(n - expected) > 1e-14 + 1e-9 * expected
            ):
                raise ValueError("spatial cells must lie on the common-pressure manifold")
            if spatial.boundary_mode == "finite-flow" and gas_node_pressure(
                node, n, spec.temperature_K
            ) > node.maximum_pressure_Pa * (1 + 1e-10):
                raise ValueError("finite spatial gas node exceeds its declared pressure ceiling")
            local_thermal.validate_state(view)
        _, packet = spatial_rates(spec, network, state)
        ceiling = min(
            spatial.transport.maximum_pressure_Pa,
            *(nodes[name].maximum_pressure_Pa for name in spatial.cell_vessel_ids),
        )
        if packet["pressure_Pa"] > ceiling * (1 + 1e-10):
            raise ValueError("spatial gas exceeds declared pressure ceiling")
        if packet["maximum_mach"] > spatial.maximum_mach * (1 + 1e-9):
            raise ValueError("spatial gas exceeds mass-average low-Mach domain")

    def pressure_event(state: CoupledStateView) -> float:
        from chemworld.physchem.reaction_network import R_J_PER_MOL_K

        if spatial.boundary_mode == "finite-flow":
            grid_n = sum(
                sum(gas_node_view(state, nodes[name], network).amounts_mol.values())
                for name in spatial.cell_vessel_ids
            )
            grid_volume = sum(nodes[name].volume_L for name in spatial.cell_vessel_ids)
            grid_pressure = grid_n * R_J_PER_MOL_K * spec.temperature_K * 1000 / grid_volume
            return min(
                spatial.transport.maximum_pressure_Pa - grid_pressure,
                *(
                    n.maximum_pressure_Pa
                    - gas_node_pressure(
                        n,
                        sum(gas_node_view(state, n, network).amounts_mol.values()),
                        spec.temperature_K,
                    )
                    for n in spec.nodes
                ),
            )
        p = (
            sum(state.amounts_mol.values())
            * R_J_PER_MOL_K
            * spec.temperature_K
            * 1000
            / state.volume_L
        )
        return (
            min(spatial.transport.maximum_pressure_Pa, *(n.maximum_pressure_Pa for n in spec.nodes))
            - p
        )

    def mach_event(state: CoupledStateView) -> float:
        # The shared event backend passes signed trial inventories to locate
        # depletion. Mach, unlike the depletion root itself, needs the same
        # boundary extension as the shared RHS. Accepted amounts stay checked.
        trial = replace(state, amounts_mol={s: max(n, 0.0) for s, n in state.amounts_mol.items()})
        return spatial.maximum_mach - spatial_rates(spec, network, trial)[1]["maximum_mach"]

    def amount_event(state: CoupledStateView) -> float:
        return min(
            sum(gas_node_view(state, n, network).amounts_mol.values()) - n.minimum_amount_mol
            for n in spec.nodes
        )

    elements = {e for s in network.species for e in s.composition}
    return CoupledSystem(
        thermal,
        (
            CoupledProcess(
                "sealed-spatial-gas"
                if spatial.boundary_mode == "sealed"
                else "finite-flow-spatial-gas",
                evaluate,
                validate_state=validate,
            ),
        ),
        isothermal=True,
        coordinates=gas_network_coordinates(spec, network),
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-9),
        events=(
            CoupledEvent("spatial:maximum_pressure", pressure_event),
            CoupledEvent("spatial:maximum_mach", mach_event),
            CoupledEvent("spatial:minimum_amount", amount_event),
        ),
        conservation_laws={
            e: {
                gas_inventory_key(n.vessel_id, s.species_id): s.composition.get(e, 0)
                for n in spec.nodes
                for s in network.species
            }
            for e in elements
        },
    )
