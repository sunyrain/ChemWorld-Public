"""Multi-temperature energy closure on the original finite-gas CoupledSystem.

The solver's temperature is the actual anchor temperature. Other node T values
are temporary coordinates reconstructed from their original WorldState fields.
The aggregate caloric derivative includes those off-anchor sensible directions;
that coordinate derivative is not reported as physical reaction heat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isclose
from typing import Any

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.coupled_system import (
    DEFAULT_COUPLED_ODE_POLICY,
    CoupledAdvanceResult,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    InternalTransfer,
    MaterialPort,
    ProcessContribution,
    assemble_coupled_rates,
)
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.gas_network_pressure import (
    gas_work_key,
    piston_events,
    piston_work_rate,
    pressure_control_for,
    project_gas_volumes,
    validate_piston_volume,
)
from chemworld.physchem.gas_network_surface import GasSurfaceContext
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    _key,
    gas_bath_key,
    gas_connection_molar_rate,
    gas_enthalpy_key,
    gas_inventory_key,
    gas_network_coordinates,
    gas_node_pressure,
    gas_node_view,
    gas_transfer_key,
)
from chemworld.physchem.gas_shaft_machine import gas_shaft_operating_point, shaft_key
from chemworld.physchem.gas_thermal_wall import wall_key
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_solvers import reaction_network_process


def gas_temperature_key(vessel_id: str) -> str:
    return _key("node_temperature_K", vessel_id)


def gas_loss_key(vessel_id: str) -> str:
    return _key("environment_loss_J", vessel_id)


def gas_reaction_energy_key(vessel_id: str) -> str:
    return _key("physical_reaction_energy_J", vessel_id)


def gas_link_heat_key(connection_id: str) -> str:
    return _key("conducted_heat_J", connection_id)


def gas_energy_coordinates(spec: GasReactorNetworkSpec) -> tuple[CoupledCoordinate, ...]:
    return (
        *(spec.thermal_wall.coordinates() if spec.thermal_wall else ()),
        *(
            CoupledCoordinate(gas_temperature_key(n.vessel_id), "K", 500)
            for n in spec.nodes
            if n.vessel_id != spec.anchor_vessel_id
        ),
        *(
            coordinate
            for n in spec.nodes
            for coordinate in (
                CoupledCoordinate(gas_loss_key(n.vessel_id), "J", 1),
                CoupledCoordinate(gas_reaction_energy_key(n.vessel_id), "J", 1),
            )
        ),
        *(CoupledCoordinate(gas_link_heat_key(e.connection_id), "J", 1) for e in spec.heat_links),
    )


@dataclass(frozen=True)
class _NetworkCalorics:
    spec: GasReactorNetworkSpec
    network: ReactionNetworkSpec
    surface: GasSurfaceContext | None = None

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        if self.surface is not None:
            return self.surface.inventory_keys
        return tuple(
            gas_inventory_key(n.vessel_id, s)
            for n in self.spec.nodes
            for s in self.network.species_ids
        )

    @property
    def local_models(self) -> dict[str, Any]:
        if self.surface is not None:
            return self.surface.models
        return {
            e.vessel_id: IdealGasThermodynamics(
                self.network.imported_gas_thermochemistry,
                e.support_heat_capacity_J_K,
                e.reference_temperature_K,
            )
            for e in self.spec.energy_nodes
        }

    def node_view(self, state: CoupledStateView, node: GasReactorNodeSpec) -> CoupledStateView:
        return (
            self.surface.view(state, node)
            if self.surface
            else gas_node_view(
                state, node, self.network, pressure_control=self.spec.pressure_control
            )
        )

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        if not self.spec.energy_nodes:
            return sum(
                self.local_models[n.vessel_id].heat_capacity_j_k(self.node_view(state, n))
                for n in self.spec.nodes
            )
        anchor = next(n for n in self.spec.nodes if n.vessel_id == self.spec.anchor_vessel_id)
        return self.local_models[anchor.vessel_id].heat_capacity_j_k(self.node_view(state, anchor))

    def internal_energy_j(self, state: CoupledStateView) -> float:
        models = self.local_models
        return sum(
            models[n.vessel_id].internal_energy_j(self.node_view(state, n)) for n in self.spec.nodes
        ) + (
            self.spec.thermal_wall.heat_capacity_J_K
            * (
                state.coordinates[wall_key(self.spec.thermal_wall.owner_id, "temperature_K")]
                - self.spec.thermal_wall.reference_temperature_K
            )
            if self.spec.thermal_wall
            else 0.0
        )

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        # Complete dU except the true anchor Cv*dT. Original derivative naming
        # predates multiple physical temperatures; never book this as reaction heat.
        models, rate = self.local_models, 0.0
        for node in self.spec.nodes:
            local = self.node_view(state, node)
            model = models[node.vessel_id]
            local_dn = (
                {
                    key: derivatives[global_key]
                    for key, global_key in self.surface.identities[node.vessel_id].items()
                }
                if self.surface is not None
                else {
                    s: derivatives[gas_inventory_key(node.vessel_id, s)]
                    for s in self.network.species_ids
                }
            )
            rate += model.chemical_energy_rate_w(local, local_dn, contributions)
            if self.spec.energy_nodes and node.vessel_id != self.spec.anchor_vessel_id:
                rate += model.heat_capacity_j_k(local) * sum(
                    c.coordinate_rates.get(gas_temperature_key(node.vessel_id), 0)
                    for c in contributions
                )
        if self.spec.thermal_wall is not None:
            wall = self.spec.thermal_wall
            rate += wall.heat_capacity_J_K * sum(
                c.coordinate_rates.get(wall_key(wall.owner_id, "temperature_K"), 0)
                for c in contributions
            )
        return rate


def _device_domain(spec: GasReactorNetworkSpec, thermal_device: ThermalDeviceSpec | None) -> None:
    if thermal_device is not None:
        anchor = next(e for e in spec.energy_nodes if e.vessel_id == spec.anchor_vessel_id)
        if (
            thermal_device.mode != "finite_power"
            or not anchor.minimum_temperature_K
            <= thermal_device.setpoint_K
            <= anchor.maximum_temperature_K
        ):
            raise ValueError(
                "thermal gas network requires a finite-power anchor device within its domain"
            )


def gas_energy_system(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
    surface_slots: Sequence[StateSlot] = (),
) -> CoupledSystem:
    surface = GasSurfaceContext(spec, network, surface_slots) if spec.surfaces else None
    if surface is not None:
        network = surface.gas
    elif surface_slots:
        raise ValueError("undeclared surface slots cannot be ignored")
    if not spec.energy_nodes or any(s.phase != "gas" for s in network.species):
        raise ValueError(
            "multi-temperature gas network requires node calorics and original gas species"
        )
    thermo = network.imported_gas_thermochemistry
    if set(thermo) != set(network.species_ids):
        raise ValueError("every gas species needs declared thermochemistry")
    _device_domain(spec, thermal_device)
    thermal = _NetworkCalorics(spec, network, surface)
    models = thermal.local_models
    energy_nodes = {e.vessel_id: e for e in spec.energy_nodes}
    nodes = {n.vessel_id: n for n in spec.nodes}
    for energy in spec.energy_nodes:
        model = models[energy.vessel_id]
        getattr(model, "base", model).validate_temperature_interval(
            energy.minimum_temperature_K, energy.maximum_temperature_K
        )
    source_bounds = {
        name: (
            max(s.min_temperature_K for s in thermo.values()),
            min(s.max_temperature_K for s in thermo.values()),
        )
        for name in nodes
    }
    if surface is not None:
        for name, attachment in surface.attachments.items():
            low, high = source_bounds[name]
            source_bounds[name] = (
                max(low, attachment.mechanism.minimum_temperature_K),
                min(high, attachment.mechanism.maximum_temperature_K),
            )

    def extend_source_boundary_trial(state: CoupledStateView) -> CoupledStateView:
        # Only the RHS uses this continuous boundary extension. Original event
        # callbacks and every accepted state retain their raw temperatures.
        coordinates = dict(state.coordinates)
        anchor_t = state.temperature_K
        changed = False
        for name, (low, high) in source_bounds.items():
            value = (
                anchor_t
                if name == spec.anchor_vessel_id
                else coordinates[gas_temperature_key(name)]
            )
            bounded = max(low, min(high, value))
            if bounded != value:
                changed = True
                if name == spec.anchor_vessel_id:
                    anchor_t = bounded
                else:
                    coordinates[gas_temperature_key(name)] = bounded
        return replace(state, temperature_K=anchor_t, coordinates=coordinates) if changed else state

    reaction = reaction_network_process(network, species_thermo=thermo)
    masses = {s.species_id: molecular_weight(s.composition) / 1000 for s in network.species}
    ports = {
        name: tuple(
            MaterialPort(
                _key("local", edge.connection_id, s, direction),
                s,
                edge.source_vessel_id,
                edge.target_vessel_id,
                direction,
            )
            for edge in spec.connections
            for direction, endpoint in (
                ("out", edge.source_vessel_id),
                ("in", edge.target_vessel_id),
            )
            if endpoint == name
            for s in network.species_ids
        )
        for name in nodes
    }

    def machine_point(state: CoupledStateView) -> dict[str, Any]:
        machine = spec.shaft_machine
        if machine is None:
            return {}
        edge = spec.connections[0]
        views = {name: gas_node_view(state, node, network) for name, node in nodes.items()}
        source, target = views[edge.source_vessel_id], views[edge.target_vessel_id]
        total, target_total = sum(source.amounts_mol.values()), sum(target.amounts_mol.values())
        if total <= 0 or target_total <= 0:
            # Numerical trial extension only; finite positive-stock events own
            # every accepted state. No substitute gas is introduced.
            return {
                "status": "empty_trial",
                "molar_rate_mol_s": 0.0,
                "shaft_power_W": 0.0,
                "entropy_generation_W_K": 0.0,
                "pressure_ratio": 0.0,
                "outlet_temperature_K": source.temperature_K,
            }
        return gas_shaft_operating_point(
            machine,
            thermo,
            inlet_temperature_K=source.temperature_K,
            inlet_pressure_Pa=gas_node_pressure(
                nodes[edge.source_vessel_id], total, source.temperature_K
            ),
            outlet_pressure_Pa=gas_node_pressure(
                nodes[edge.target_vessel_id], target_total, target.temperature_K
            ),
            mole_fractions={s: n / total for s, n in source.amounts_mol.items()},
            requested_molar_rate_mol_s=gas_connection_molar_rate(edge, nodes, views, masses),
            enabled=edge.enabled,
            check_limits=False,
        )

    def fluxes(state: CoupledStateView) -> tuple[dict[str, float], dict[str, float]]:
        external, loss = {}, {}
        for name, node in nodes.items():
            t = gas_node_view(
                state, node, network, pressure_control=spec.pressure_control
            ).temperature_K
            energy = energy_nodes[name]
            external[name] = (
                thermal_device.delivered_power_w(t)
                if name == spec.anchor_vessel_id and thermal_device is not None
                else 0.0
            )
            loss[name] = energy.loss_conductance_W_K * (t - energy.environment_temperature_K)
        return external, loss

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        views = {
            name: gas_node_view(state, node, network, pressure_control=spec.pressure_control)
            for name, node in nodes.items()
        }
        external, loss = fluxes(state)
        link_net = dict.fromkeys(nodes, 0.0)
        counters: dict[str, float] = {}
        if spec.thermal_wall is not None:
            wall = spec.thermal_wall
            wall_temperature = state.coordinates[wall_key(wall.owner_id, "temperature_K")]
            conduction, radiation = wall.environment_losses(wall_temperature)
            incoming = 0.0
            for contact in wall.contacts:
                q = contact.conductance_W_K * (
                    views[contact.vessel_id].temperature_K - wall_temperature
                )
                incoming += q
                link_net[contact.vessel_id] -= q
                counters[wall_key(wall.owner_id, "contact_J", contact.vessel_id)] = q
            counters[wall_key(wall.owner_id, "environment_conduction_J")] = conduction
            counters[wall_key(wall.owner_id, "environment_radiation_J")] = radiation
            counters[wall_key(wall.owner_id, "temperature_K")] = (
                incoming - conduction - radiation
            ) / wall.heat_capacity_J_K
        for heat_link in spec.heat_links:
            heat = heat_link.conductance_W_K * (
                views[heat_link.source_vessel_id].temperature_K
                - views[heat_link.target_vessel_id].temperature_K
            )
            link_net[heat_link.source_vessel_id] -= heat
            link_net[heat_link.target_vessel_id] += heat
            counters[gas_link_heat_key(heat_link.connection_id)] = heat
        port_rates: dict[str, dict[str, float]] = {name: {} for name in nodes}
        port_enthalpies: dict[str, dict[str, float]] = {name: {} for name in nodes}
        transfers, amounts, progress = [], {}, {}
        for edge in spec.connections:
            source = views[edge.source_vessel_id]
            total = sum(source.amounts_mol.values())
            flow = gas_connection_molar_rate(edge, nodes, views, masses)
            machine = machine_point(state) if spec.shaft_machine else None
            if machine is not None:
                flow = machine["molar_rate_mol_s"]
            transported_h = 0.0
            delivered_h = 0.0
            for species in network.species_ids:
                rate = flow * source.amounts_mol[species] / total if total > 0 else 0.0
                h = thermo[species].evaluate(source.temperature_K).enthalpy_J_mol
                outlet_h = (
                    thermo[species].evaluate(machine["outlet_temperature_K"]).enthalpy_J_mol
                    if machine is not None and flow > 0
                    else h
                )
                transfers.append(
                    InternalTransfer(
                        _key("flow", edge.connection_id, species),
                        gas_inventory_key(edge.source_vessel_id, species),
                        gas_inventory_key(edge.target_vessel_id, species),
                        rate,
                    )
                )
                counters[gas_transfer_key(edge.connection_id, species)] = rate
                transported_h += rate * h
                delivered_h += rate * outlet_h
                for direction, endpoint in (
                    ("out", edge.source_vessel_id),
                    ("in", edge.target_vessel_id),
                ):
                    key = _key("local", edge.connection_id, species, direction)
                    port_rates[endpoint][key] = rate
                    port_enthalpies[endpoint][key] = outlet_h if direction == "in" else h
            counters[gas_enthalpy_key(edge.connection_id)] = transported_h
            if machine is not None:
                counters[shaft_key(edge.connection_id, "delivered_enthalpy_J")] = delivered_h
                counters[shaft_key(edge.connection_id, "shaft_work_J")] = (
                    delivered_h - transported_h
                )
                counters[shaft_key(edge.connection_id, "entropy_generated_J_K")] = machine[
                    "entropy_generation_W_K"
                ]
        for name, node in nodes.items():
            view = views[name]
            chemistry = reaction.evaluate(view) if node.chemistry_enabled else ProcessContribution()
            if surface is not None:
                chemistry = surface.chemistry(state, node, chemistry)
                view = surface.view(state, node)
            amounts.update(
                {
                    (surface.identities[name][s] if surface else gas_inventory_key(name, s)): rate
                    for s, rate in chemistry.amount_rates_mol_s.items()
                }
            )
            progress.update(
                {
                    _key("reaction", name, key): rate
                    for key, rate in chemistry.reaction_progress_mol_s.items()
                }
            )
            contribution = ProcessContribution(
                amount_rates_mol_s=chemistry.amount_rates_mol_s,
                port_rates_mol_s=port_rates[name],
                port_enthalpies_J_mol=port_enthalpies[name],
            )

            def local_contribution(
                _: CoupledStateView, value: ProcessContribution = contribution
            ) -> ProcessContribution:
                return value

            def local_flux(
                _: CoupledStateView,
                q: float = external[name] + link_net[name],
                loss_w: float = loss[name],
                work_w: float = piston_work_rate(
                    view,
                    models[name],
                    contribution,
                    ports[name],
                    net_heat_W=external[name] + link_net[name] - loss[name],
                )
                if pressure_control_for(node, spec.pressure_control)
                else 0.0,
            ) -> HeatFlux:
                return HeatFlux(q, loss_w, work_w)

            local = assemble_coupled_rates(
                view,
                processes=(CoupledProcess("node-balance", local_contribution),),
                thermodynamics=models[name],
                ports=ports[name],
                heat_flux=local_flux,
            )
            if name != spec.anchor_vessel_id:
                counters[gas_temperature_key(name)] = local.temperature_rate_K_s
            counters[gas_bath_key(name)] = external[name]
            counters[gas_loss_key(name)] = loss[name]
            counters[gas_reaction_energy_key(name)] = models[name].chemical_energy_rate_w(
                view, chemistry.amount_rates_mol_s, (chemistry,)
            )
            if pressure_control_for(node, spec.pressure_control):
                counters[gas_work_key(name)] = local.work_W
        return ProcessContribution(amounts, progress, tuple(transfers), coordinate_rates=counters)

    def validate(state: CoupledStateView) -> None:
        if spec.shaft_machine is not None:
            machine = machine_point(state)
            declaration = spec.shaft_machine
            if spec.connections[0].enabled and spec.connections[0].coefficient > 0:
                if machine["pressure_ratio"] > declaration.maximum_pressure_ratio * (1 + 1e-9):
                    raise ValueError("shaft pressure ratio exceeds its operating envelope")
                if machine["status"] == "running" and not (
                    declaration.minimum_outlet_temperature_K - 1e-6
                    <= machine["outlet_temperature_K"]
                    <= declaration.maximum_outlet_temperature_K + 1e-6
                ):
                    raise ValueError("shaft outlet temperature exceeds its operating envelope")
        if spec.thermal_wall is not None:
            wall = spec.thermal_wall
            temperature = state.coordinates[wall_key(wall.owner_id, "temperature_K")]
            if (
                not wall.minimum_temperature_K - 1e-7
                <= temperature
                <= wall.maximum_temperature_K + 1e-7
            ):
                raise ValueError("wall temperature exceeds its declared domain")
        if surface is not None:
            surface.validate(state)
        if (
            spec.pressure_control is None
            and not isclose(state.volume_L, sum(n.volume_L for n in spec.nodes), rel_tol=1e-12)
        ) or set(state.coordinates) != {c.name for c in gas_network_coordinates(spec, network)}:
            raise ValueError("thermal gas assembly volume or coordinate layout differs")
        for name, node in nodes.items():
            local = gas_node_view(state, node, network, pressure_control=spec.pressure_control)
            total = sum(local.amounts_mol.values())
            energy = energy_nodes[name]
            if total <= 0 or total < node.minimum_amount_mol - (
                1e-14 + 1e-10 * max(total, node.minimum_amount_mol)
            ):
                raise ValueError("gas node below declared minimum amount")
            if not (
                energy.minimum_temperature_K - 1e-7
                <= local.temperature_K
                <= energy.maximum_temperature_K + 1e-7
            ):
                raise ValueError("gas node outside temperature limits")
            if gas_node_pressure(
                node, total, local.temperature_K, spec.pressure_control
            ) > node.maximum_pressure_Pa * (1 + 1e-10):
                raise ValueError("gas node exceeds pressure ceiling")
            control = pressure_control_for(node, spec.pressure_control)
            if control is not None:
                validate_piston_volume(control, control.volume_L(total, local.temperature_K))
            models[name].validate_state(surface.view(state, node) if surface else local)

    def node_events(node: GasReactorNodeSpec) -> tuple[CoupledEvent, ...]:
        name = node.vessel_id
        energy = energy_nodes[name]

        def minimum(s: CoupledStateView) -> float:
            return (
                sum(
                    gas_node_view(
                        s, node, network, pressure_control=spec.pressure_control
                    ).amounts_mol.values()
                )
                - node.minimum_amount_mol
            )

        def pressure(s: CoupledStateView) -> float:
            local = gas_node_view(s, node, network, pressure_control=spec.pressure_control)
            return node.maximum_pressure_Pa - gas_node_pressure(
                node, sum(local.amounts_mol.values()), local.temperature_K, spec.pressure_control
            )

        def low(s: CoupledStateView) -> float:
            return (
                gas_node_view(
                    s, node, network, pressure_control=spec.pressure_control
                ).temperature_K
                - energy.minimum_temperature_K
            )

        def high(s: CoupledStateView) -> float:
            return (
                energy.maximum_temperature_K
                - gas_node_view(
                    s, node, network, pressure_control=spec.pressure_control
                ).temperature_K
            )

        return (
            CoupledEvent(f"gas:{name}:minimum_amount", minimum),
            CoupledEvent(f"gas:{name}:maximum_pressure", pressure),
            CoupledEvent(f"gas:{name}:minimum_temperature", low),
            CoupledEvent(f"gas:{name}:maximum_temperature", high),
        )

    events = tuple(e for node in nodes.values() for e in node_events(node)) + piston_events(
        spec, network
    )
    if spec.shaft_machine and spec.connections[0].enabled and spec.connections[0].coefficient > 0:
        machine = spec.shaft_machine
        identity = machine.connection_id
        events += (
            CoupledEvent(
                f"shaft:{identity}:pressure_direction",
                lambda s: machine_point(s)["pressure_ratio"] - 1.0,
            ),
            CoupledEvent(
                f"shaft:{identity}:maximum_pressure_ratio",
                lambda s: machine.maximum_pressure_ratio - machine_point(s)["pressure_ratio"],
            ),
            CoupledEvent(
                f"shaft:{identity}:minimum_outlet_temperature",
                lambda s: (
                    machine_point(s)["outlet_temperature_K"] - machine.minimum_outlet_temperature_K
                ),
            ),
            CoupledEvent(
                f"shaft:{identity}:maximum_outlet_temperature",
                lambda s: (
                    machine.maximum_outlet_temperature_K - machine_point(s)["outlet_temperature_K"]
                ),
            ),
        )
    elements = {element for s in network.species for element in s.composition}

    def external_flux(state: CoupledStateView) -> HeatFlux:
        heater, loss = fluxes(state)
        wall_loss = 0.0
        if spec.thermal_wall is not None:
            wall = spec.thermal_wall
            wall_loss = sum(
                wall.environment_losses(state.coordinates[wall_key(wall.owner_id, "temperature_K")])
            )
        return HeatFlux(
            sum(heater.values()),
            sum(loss.values()) + wall_loss,
            machine_point(state)["shaft_power_W"]
            if spec.shaft_machine
            else evaluate(state).coordinate_rates[gas_work_key(spec.pressure_control.vessel_id)]
            if spec.pressure_control
            else 0.0,
        )

    if spec.thermal_wall is not None:
        wall = spec.thermal_wall
        temperature_key = wall_key(wall.owner_id, "temperature_K")
        events += (
            CoupledEvent(
                "wall:minimum_temperature",
                lambda s: s.coordinates[temperature_key] - wall.minimum_temperature_K,
            ),
            CoupledEvent(
                "wall:maximum_temperature",
                lambda s: wall.maximum_temperature_K - s.coordinates[temperature_key],
            ),
        )
    return CoupledSystem(
        thermal,
        (CoupledProcess("gas-network-energy", evaluate, validate_state=validate),),
        heat_flux=external_flux,
        coordinates=gas_network_coordinates(spec, network),
        rate_independent_coordinates=tuple(
            c.name
            for c in gas_network_coordinates(spec, network)
            if c.name not in {gas_temperature_key(n) for n in nodes}
            and (
                spec.thermal_wall is None
                or c.name != wall_key(spec.thermal_wall.owner_id, "temperature_K")
            )
        ),
        trial_state_extension=extend_source_boundary_trial,
        # Several nonlinear local energies cancel in the aggregate first law.
        # Resolve that cancellation without loosening the common energy check.
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-10),
        events=tuple(events),
        conservation_laws=surface.laws
        if surface is not None
        else {
            element: {
                gas_inventory_key(n.vessel_id, s.species_id): s.composition.get(element, 0)
                for n in spec.nodes
                for s in network.species
            }
            for element in elements
        },
    )


def advance_gas_energy(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    state: CoupledState,
    duration_s: float,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
    surface_slots: Sequence[StateSlot] = (),
) -> tuple[CoupledAdvanceResult, dict[str, Any]]:
    system = gas_energy_system(
        spec, network, thermal_device=thermal_device, surface_slots=surface_slots
    )
    surface = GasSurfaceContext(spec, network, surface_slots) if spec.surfaces else None
    if surface is not None:
        network = surface.gas
    result = project_gas_volumes(system.advance(state, duration_s), spec, network)
    final = result.final_state
    delta = {key: final.coordinates[key] - state.coordinates[key] for key in state.coordinates}
    flows: dict[str, dict[str, Any]] = {
        e.connection_id: {
            "source": e.source_vessel_id,
            "target": e.target_vessel_id,
            "amounts_mol": {
                s: delta[gas_transfer_key(e.connection_id, s)] for s in network.species_ids
            },
            "enthalpy_J": delta[gas_enthalpy_key(e.connection_id)],
            **(
                {
                    "delivered_enthalpy_J": delta[
                        shaft_key(e.connection_id, "delivered_enthalpy_J")
                    ],
                    "shaft_work_J": delta[shaft_key(e.connection_id, "shaft_work_J")],
                    "entropy_generated_J_K": delta[
                        shaft_key(e.connection_id, "entropy_generated_J_K")
                    ],
                }
                if spec.shaft_machine
                else {}
            ),
        }
        for e in spec.connections
    }
    links: dict[str, dict[str, Any]] = {
        e.connection_id: {
            "source": e.source_vessel_id,
            "target": e.target_vessel_id,
            "heat_J": delta[gas_link_heat_key(e.connection_id)],
        }
        for e in spec.heat_links
    }
    wall_details: dict[str, Any] | None = None
    if spec.thermal_wall is not None:
        wall = spec.thermal_wall
        contact_heat = {
            c.vessel_id: delta[wall_key(wall.owner_id, "contact_J", c.vessel_id)]
            for c in wall.contacts
        }
        conduction = delta[wall_key(wall.owner_id, "environment_conduction_J")]
        radiation = delta[wall_key(wall.owner_id, "environment_radiation_J")]
        wall_energy = wall.heat_capacity_J_K * delta[wall_key(wall.owner_id, "temperature_K")]
        residual = wall_energy - sum(contact_heat.values()) + conduction + radiation
        if abs(residual) > 2e-6 + 5e-6 * (
            abs(wall_energy)
            + sum(map(abs, contact_heat.values()))
            + abs(conduction)
            + abs(radiation)
        ):
            raise RuntimeError(f"thermal wall energy failed to close: {residual} J")
        wall_details = {
            "owner_id": wall.owner_id,
            "temperature_K": final.coordinates[wall_key(wall.owner_id, "temperature_K")],
            "heat_capacity_J_K": wall.heat_capacity_J_K,
            "contact_heat_from_gas_J": contact_heat,
            "environment_conduction_J": conduction,
            "environment_radiation_J": radiation,
            "heat_loss_J": conduction + radiation,
            "internal_energy_change_J": wall_energy,
            "energy_residual_J": residual,
        }
    thermal = _NetworkCalorics(spec, network, surface)
    models = thermal.local_models
    details: dict[str, Any] = {}
    for node in spec.nodes:
        name = node.vessel_id
        after = gas_node_view(final, node, network, pressure_control=spec.pressure_control)
        energy = models[name].internal_energy_j(thermal.node_view(final, node)) - models[
            name
        ].internal_energy_j(thermal.node_view(state, node))
        incoming = sum(
            f.get("delivered_enthalpy_J", f["enthalpy_J"])
            for f in flows.values()
            if f["target"] == name
        )
        outgoing = sum(f["enthalpy_J"] for f in flows.values() if f["source"] == name)
        conducted = sum(
            ((1 if f["target"] == name else 0) - (1 if f["source"] == name else 0)) * f["heat_J"]
            for f in links.values()
        )
        heater, loss = delta[gas_bath_key(name)], delta[gas_loss_key(name)]
        work = delta.get(gas_work_key(name), 0.0)
        wall_heat = -wall_details["contact_heat_from_gas_J"].get(name, 0.0) if wall_details else 0.0
        residual = energy - heater + loss - incoming + outgoing - conducted - work - wall_heat
        if abs(residual) > 2e-6 + 5e-6 * sum(
            abs(x) for x in (energy, heater, loss, incoming, outgoing, conducted)
        ):
            raise RuntimeError(f"thermal gas node energy failed to close: {name}: {residual} J")
        details[name] = {
            **(
                {
                    "surface_amounts_mol": {
                        s: final.amounts_mol[s]
                        for s in surface.identities[name]
                        if s not in network.species_ids
                    }
                }
                if surface is not None
                else {}
            ),
            "amounts_mol": dict(after.amounts_mol),
            "temperature_K": after.temperature_K,
            "volume_L": after.volume_L,
            "pressure_Pa": gas_node_pressure(
                node, sum(after.amounts_mol.values()), after.temperature_K, spec.pressure_control
            ),
            **({"boundary_work_on_J": work} if spec.pressure_control else {}),
            "bath_heat_J": heater,
            "heat_loss_J": loss,
            "enthalpy_in_J": incoming,
            "enthalpy_out_J": outgoing,
            "conducted_heat_J": conducted,
            **({"wall_heat_J": wall_heat} if wall_details is not None else {}),
            "internal_energy_change_J": energy,
            "chemical_energy_change_J": delta[gas_reaction_energy_key(name)],
            "energy_residual_J": residual,
        }
    heater, loss = (
        sum(d["bath_heat_J"] for d in details.values()),
        sum(d["heat_loss_J"] for d in details.values())
        + (wall_details["heat_loss_J"] if wall_details else 0.0),
    )
    work_sum = sum(d.get("boundary_work_on_J", 0.0) for d in details.values())
    shaft_work = sum(f.get("shaft_work_J", 0.0) for f in flows.values())
    if spec.shaft_machine:
        for flow in flows.values():
            if not isclose(
                flow["delivered_enthalpy_J"] - flow["enthalpy_J"],
                flow["shaft_work_J"],
                abs_tol=2e-7,
                rel_tol=2e-7,
            ):
                raise RuntimeError("shaft device inlet/outlet enthalpy and work disagree")
        if not isclose(shaft_work, final.work_J - state.work_J, abs_tol=2e-7, rel_tol=2e-7):
            raise RuntimeError("shaft work differs from original common energy ledger")
    if spec.pressure_control and not isclose(
        work_sum, final.work_J - state.work_J, abs_tol=2e-7, rel_tol=2e-7
    ):
        raise RuntimeError("piston work differs from the original common energy ledger")
    if not isclose(
        heater, final.boundary_heat_J - state.boundary_heat_J, abs_tol=2e-7, rel_tol=2e-7
    ) or not isclose(loss, final.heat_loss_J - state.heat_loss_J, abs_tol=2e-7, rel_tol=2e-7):
        raise RuntimeError("node thermal receipts differ from the common energy ledger")
    return result, {
        "nodes": details,
        "connections": flows,
        "heat_links": links,
        **({"thermal_wall": wall_details} if wall_details is not None else {}),
        "bath_heat_J": heater,
        "heat_loss_J": loss,
        **({"boundary_work_on_J": work_sum} if spec.pressure_control else {}),
        **({"shaft_work_J": shaft_work} if spec.shaft_machine else {}),
        "actual_duration_s": result.actual_duration_s,
        "stopping_event": result.stopping_event,
        "energy_residual_J": result.energy_balance_residual_J,
        "element_residuals_mol": dict(result.conservation_residuals),
        "energy_scope": (
            "two original gas+support U inventories; actual inlet/outlet H; "
            "adiabatic shaft work once, positive into gas"
            if spec.shaft_machine
            else "node gas+support U; independent wall C/T; internal contacts; "
            "external grey radiation/conduction"
            if spec.thermal_wall
            else "node gas+support U; source H; actual signed constant-load piston work"
            if spec.pressure_control
            else "node gas+declared common-temperature support U; source H advection; no wall work"
        ),
    }
