"""Finite isothermal gas networks compiled into the existing CoupledSystem.

Node/species keys are numerical addresses, never new SpeciesSpec identities.
One common candidate is used for every reaction and connection. Each reaction
reads its node's volume; the sum-of-volumes field is only the assembly extent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Any

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.coupled_system import (
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
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec
from chemworld.physchem.gas_network_pressure import (
    GasPressureControlSpec,
    gas_node_volume,
    gas_work_key,
    piston_events,
    piston_work_rate,
    pressure_control_for,
    project_gas_volumes,
    validate_piston_volume,
)
from chemworld.physchem.gas_network_surface import (
    GasSurfaceAttachment,
    GasSurfaceContext,
    gas_phase_network,
)
from chemworld.physchem.gas_spatial_transport import GasSpatialTransportSpec
from chemworld.physchem.gas_thermal_wall import GasThermalWallSpec, wall_key
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_solvers import reaction_network_process


def _positive(value: float, label: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def _key(kind: str, *parts: str) -> str:
    return json.dumps([kind, *parts], separators=(",", ":"))


def gas_inventory_key(vessel: str, species: str) -> str:
    return _key("inventory", vessel, species)


def gas_bath_key(vessel: str) -> str:
    return _key("bath_J", vessel)


def gas_transfer_key(edge: str, species: str) -> str:
    return _key("transferred_mol", edge, species)


def gas_enthalpy_key(edge: str) -> str:
    return _key("transferred_enthalpy_J", edge)


@dataclass(frozen=True)
class GasReactorNodeSpec:
    """Finite gas node; chemistry_enabled controls gas reactions only.

    Fixed surfaces are enabled by their separate attachment declaration. Omit
    an attachment to author a node without a catalyst; no live switch is implied.
    """

    vessel_id: str
    volume_L: float
    chemistry_enabled: bool = True
    minimum_amount_mol: float = 1e-12
    maximum_pressure_Pa: float = 1e8

    def __post_init__(self) -> None:
        if not isinstance(self.vessel_id, str) or not self.vessel_id:
            raise ValueError("gas node requires a vessel identity")
        for name in ("volume_L", "minimum_amount_mol", "maximum_pressure_Pa"):
            _positive(getattr(self, name), name)
        if type(self.chemistry_enabled) is not bool:
            raise ValueError("chemistry_enabled must be boolean")


@dataclass(frozen=True)
class GasFlowConnectionSpec:
    """Molar flow mol/s, mass flow kg/s, or one-way valve kg/(s Pa).

    Prescribed flow is an ideal external control law, not a compressor model;
    pressure-valve flow is passive and zero against an adverse pressure gradient.
    """

    connection_id: str
    source_vessel_id: str
    target_vessel_id: str
    mode: str = "molar-flow"
    coefficient: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(v, str) or not v
                for v in (self.connection_id, self.source_vessel_id, self.target_vessel_id)
            )
            or self.source_vessel_id == self.target_vessel_id
        ):
            raise ValueError("gas connection needs distinct declared endpoints and an identity")
        if self.mode not in {"molar-flow", "mass-flow", "pressure-valve"}:
            raise ValueError("unknown gas flow law")
        if (
            isinstance(self.coefficient, bool)
            or not isfinite(self.coefficient)
            or self.coefficient < 0
            or type(self.enabled) is not bool
        ):
            raise ValueError("flow coefficient must be finite nonnegative and enabled boolean")


@dataclass(frozen=True)
class GasReactorNetworkSpec:
    nodes: tuple[GasReactorNodeSpec, ...]
    connections: tuple[GasFlowConnectionSpec, ...]
    temperature_K: float
    anchor_vessel_id: str
    energy_nodes: tuple[GasNodeEnergySpec, ...] = ()
    heat_links: tuple[GasThermalLinkSpec, ...] = ()
    spatial_transport: GasSpatialTransportSpec | None = None
    surfaces: tuple[GasSurfaceAttachment, ...] = ()
    pressure_control: GasPressureControlSpec | None = None
    thermal_wall: GasThermalWallSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda x: x.vessel_id)))
        object.__setattr__(
            self, "connections", tuple(sorted(self.connections, key=lambda x: x.connection_id))
        )
        object.__setattr__(
            self, "energy_nodes", tuple(sorted(self.energy_nodes, key=lambda x: x.vessel_id))
        )
        object.__setattr__(
            self, "heat_links", tuple(sorted(self.heat_links, key=lambda x: x.connection_id))
        )
        _positive(self.temperature_K, "temperature_K")
        names = [n.vessel_id for n in self.nodes]
        if self.thermal_wall is not None:
            wall = self.thermal_wall
            if (
                not self.energy_nodes
                or self.surfaces
                or self.spatial_transport is not None
                or self.pressure_control is not None
                or {c.vessel_id for c in wall.contacts} - set(names)
                or wall.owner_id
                in {f"{self.anchor_vessel_id}:gas-network", f"{self.anchor_vessel_id}:heater"}
            ):
                raise ValueError(
                    "thermal wall requires declared constant-volume energy nodes "
                    "without surface, spatial or piston coupling"
                )
        if self.pressure_control is not None:
            control = self.pressure_control
            if (
                self.surfaces
                or self.spatial_transport is not None
                or control.vessel_id not in names
            ):
                raise ValueError(
                    "piston control requires one original gas node "
                    "without surfaces or spatial coupling"
                )
            node = next(n for n in self.nodes if n.vessel_id == control.vessel_id)
            if (
                control.pressure_Pa >= node.maximum_pressure_Pa
                or not control.minimum_volume_L < node.volume_L < control.maximum_volume_L
            ):
                raise ValueError(
                    "piston reference volume/pressure must lie inside "
                    "its stroke and pressure ceiling"
                )
        object.__setattr__(
            self, "surfaces", tuple(sorted(self.surfaces, key=lambda a: a.vessel_id))
        )
        if self.surfaces and (
            self.spatial_transport is not None
            or len({a.vessel_id for a in self.surfaces}) != len(self.surfaces)
            or len({a.owner_id for a in self.surfaces}) != len(self.surfaces)
            or {a.vessel_id for a in self.surfaces} - set(names)
        ):
            raise ValueError(
                "fixed surfaces need distinct declared nodes/owners and no spatial model"
            )
        if self.spatial_transport is not None:
            spatial = self.spatial_transport
            if self.energy_nodes or self.heat_links:
                raise ValueError("isothermal spatial gas cannot combine heaters or energy nodes")
            volumes = dict(zip(spatial.cell_vessel_ids, spatial.mesh.cell_volumes_m3, strict=True))
            if spatial.boundary_mode == "sealed":
                if self.connections or set(volumes) != set(names):
                    raise ValueError(
                        "sealed spatial gas requires only mesh nodes and no imposed flow"
                    )
            else:
                from chemworld.physchem.gas_spatial_flow import spatial_boundary_connections

                spatial_boundary_connections(self)
            if set(volumes) - set(names) or any(
                not isclose(n.volume_L, volumes[n.vessel_id] * 1000, rel_tol=1e-12)
                for n in self.nodes
                if n.vessel_id in volumes
            ):
                raise ValueError("spatial cells must match original vessel identities and volumes")
        if (
            not names
            or len(set(names)) != len(names)
            or self.anchor_vessel_id not in names
            or len({e.connection_id for e in self.connections}) != len(self.connections)
        ):
            raise ValueError("gas network requires unique nodes/edges and a declared anchor")
        if any({e.source_vessel_id, e.target_vessel_id} - set(names) for e in self.connections):
            raise ValueError("gas connection references an undeclared node")
        if self.energy_nodes:
            if {e.vessel_id for e in self.energy_nodes} != set(names) or len(
                self.energy_nodes
            ) != len(names):
                raise ValueError("thermal gas nodes must cover the original vessels exactly once")
            if self.initial_temperatures_K[self.anchor_vessel_id] != self.temperature_K:
                raise ValueError("network temperature is the actual anchor initial temperature")
        elif self.heat_links:
            raise ValueError("conductive links require explicit node energy models")
        if (
            len({e.connection_id for e in self.heat_links}) != len(self.heat_links)
            or {e.connection_id for e in self.heat_links}
            & {e.connection_id for e in self.connections}
            or any({e.source_vessel_id, e.target_vessel_id} - set(names) for e in self.heat_links)
        ):
            raise ValueError("thermal links require unique identities and declared endpoints")
        if any(
            n.minimum_amount_mol * R * self.initial_temperatures_K[n.vessel_id] * 1000 / n.volume_L
            >= n.maximum_pressure_Pa
            for n in self.nodes
        ):
            raise ValueError("node cutoff and pressure ceiling leave no operating domain")

    @property
    def initial_temperatures_K(self) -> dict[str, float]:  # noqa: N802 - unit-explicit property
        return (
            {n.vessel_id: n.initial_temperature_K for n in self.energy_nodes}
            if self.energy_nodes
            else {n.vessel_id: self.temperature_K for n in self.nodes}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": (
                "finite-thermal-wall-gas-network-v1"
                if self.thermal_wall is not None
                else "finite-thermal-piston-gas-network-v1"
                if self.pressure_control is not None and self.energy_nodes
                else "finite-isothermal-piston-gas-network-v1"
                if self.pressure_control is not None
                else "finite-thermal-surface-gas-network-v1"
                if self.surfaces and self.energy_nodes
                else "finite-isothermal-surface-gas-network-v1"
                if self.surfaces
                else "sealed-spatial-gas-network-v1"
                if self.spatial_transport is not None
                and self.spatial_transport.boundary_mode == "sealed"
                else "finite-flow-spatial-gas-network-v1"
                if self.spatial_transport is not None
                else "finite-thermal-gas-network-v1"
                if self.energy_nodes
                else "finite-isothermal-gas-network-v1"
            ),
            "nodes": [asdict(n) for n in self.nodes],
            "connections": [asdict(e) for e in self.connections],
            "temperature_K": self.temperature_K,
            "anchor_vessel_id": self.anchor_vessel_id,
            **(
                {"thermal_wall": self.thermal_wall.to_dict()}
                if self.thermal_wall is not None
                else {}
            ),
            **({"surfaces": [a.to_dict() for a in self.surfaces]} if self.surfaces else {}),
            **(
                {"pressure_control": self.pressure_control.to_dict()}
                if self.pressure_control is not None
                else {}
            ),
            **(
                {"spatial_transport": self.spatial_transport.to_dict()}
                if self.spatial_transport
                else {}
            ),
            **(
                {
                    "energy_nodes": [asdict(n) for n in self.energy_nodes],
                    "heat_links": [asdict(e) for e in self.heat_links],
                }
                if self.energy_nodes
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasReactorNetworkSpec:
        if value.get("model") not in {
            "finite-thermal-wall-gas-network-v1",
            "finite-isothermal-gas-network-v1",
            "finite-thermal-gas-network-v1",
            "sealed-spatial-gas-network-v1",
            "finite-flow-spatial-gas-network-v1",
            "finite-isothermal-surface-gas-network-v1",
            "finite-thermal-surface-gas-network-v1",
            "finite-thermal-piston-gas-network-v1",
            "finite-isothermal-piston-gas-network-v1",
        }:
            raise ValueError("unsupported gas network declaration")
        thermal = value["model"] in {
            "finite-thermal-wall-gas-network-v1",
            "finite-thermal-gas-network-v1",
            "finite-thermal-surface-gas-network-v1",
            "finite-thermal-piston-gas-network-v1",
        }
        if (value["model"] == "finite-thermal-wall-gas-network-v1") != bool(
            value.get("thermal_wall")
        ):
            raise ValueError("thermal wall declaration differs from its model version")
        if ("piston-gas" in value["model"]) != bool(value.get("pressure_control")):
            raise ValueError("piston declaration differs from its model version")
        if ("surface-gas" in value["model"]) != bool(value.get("surfaces")):
            raise ValueError("surface declaration differs from its model version")
        if (
            value["model"]
            in {"sealed-spatial-gas-network-v1", "finite-flow-spatial-gas-network-v1"}
        ) != bool(value.get("spatial_transport")):
            raise ValueError("spatial gas declaration differs from its model version")
        if value.get("spatial_transport"):
            mode = GasSpatialTransportSpec.from_dict(value["spatial_transport"]).boundary_mode
            if (value["model"] == "finite-flow-spatial-gas-network-v1") != (mode == "finite-flow"):
                raise ValueError("spatial boundary mode differs from its model version")
        if thermal != bool(value.get("energy_nodes")) or (not thermal and value.get("heat_links")):
            raise ValueError("gas energy declaration differs from its model version")
        return cls(
            tuple(GasReactorNodeSpec(**n) for n in value["nodes"]),
            tuple(GasFlowConnectionSpec(**e) for e in value["connections"]),
            float(value["temperature_K"]),
            str(value["anchor_vessel_id"]),
            tuple(GasNodeEnergySpec(**n) for n in value.get("energy_nodes", ())),
            tuple(GasThermalLinkSpec(**e) for e in value.get("heat_links", ())),
            GasSpatialTransportSpec.from_dict(value["spatial_transport"])
            if value.get("spatial_transport")
            else None,
            tuple(GasSurfaceAttachment.from_dict(a) for a in value.get("surfaces", ())),
            GasPressureControlSpec.from_dict(value["pressure_control"])
            if value.get("pressure_control")
            else None,
            GasThermalWallSpec.from_dict(value["thermal_wall"])
            if value.get("thermal_wall")
            else None,
        )


def gas_material_topology(spec: GasReactorNetworkSpec) -> tuple[tuple[str, str, str], ...]:
    if spec.spatial_transport is not None:
        return (
            *spec.spatial_transport.topology,
            *((e.connection_id, e.source_vessel_id, e.target_vessel_id) for e in spec.connections),
        )
    return tuple(
        (e.connection_id, e.source_vessel_id, e.target_vessel_id) for e in spec.connections
    )


def gas_network_coordinates(
    spec: GasReactorNetworkSpec, network: ReactionNetworkSpec
) -> tuple[CoupledCoordinate, ...]:
    if spec.surfaces:
        network = gas_phase_network(network)
    original = (
        *(
            (CoupledCoordinate(gas_work_key(spec.pressure_control.vessel_id), "J", 1),)
            if spec.pressure_control
            else ()
        ),
        *(CoupledCoordinate(gas_bath_key(n.vessel_id), "J", 1) for n in spec.nodes),
        *(
            coordinate
            for face, _, _ in gas_material_topology(spec)
            for coordinate in (
                *(
                    CoupledCoordinate(gas_transfer_key(face, s), "mol", 0.01)
                    for s in network.species_ids
                ),
                CoupledCoordinate(gas_enthalpy_key(face), "J", 1),
            )
        ),
    )
    if spec.spatial_transport is not None:
        from chemworld.physchem.gas_spatial_reactor import spatial_coordinates

        return (*original, *spatial_coordinates(spec, network))
    if not spec.energy_nodes:
        return original
    from chemworld.physchem.gas_network_energy import gas_energy_coordinates

    return (*original, *gas_energy_coordinates(spec))


def gas_accounting_coordinates(
    spec: GasReactorNetworkSpec, network: ReactionNetworkSpec
) -> tuple[CoupledCoordinate, ...]:
    """Persistent receipts exclude the temporary copies of original node temperatures."""
    return tuple(c for c in gas_network_coordinates(spec, network) if c.unit != "K")


def pack_gas_reactor_network_state(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
    *,
    time_s: float = 0,
    counters: Mapping[str, float] | None = None,
    temperatures_by_vessel_K: Mapping[str, float] | None = None,
    surface_slots: Sequence[StateSlot] = (),
    wall_temperature_K: float | None = None,
) -> CoupledState:
    surface = GasSurfaceContext(spec, network, surface_slots) if spec.surfaces else None
    if surface is not None:
        network = surface.gas
    elif surface_slots:
        raise ValueError("undeclared surface slots cannot be ignored")
    if set(amounts_by_vessel) != {n.vessel_id for n in spec.nodes}:
        raise ValueError("gas inventory must contain exactly the declared nodes")
    flat = {}
    for node in spec.nodes:
        amounts = amounts_by_vessel[node.vessel_id]
        if set(amounts) - set(network.species_ids):
            raise ValueError("unknown species in gas inventory")
        flat.update(
            {
                gas_inventory_key(node.vessel_id, s): float(amounts.get(s, 0))
                for s in network.species_ids
            }
        )
    coordinates = {c.name: 0.0 for c in gas_network_coordinates(spec, network)}
    if spec.thermal_wall is not None:
        wall = spec.thermal_wall
        coordinates[wall_key(wall.owner_id, "temperature_K")] = (
            wall.initial_temperature_K if wall_temperature_K is None else wall_temperature_K
        )
    elif wall_temperature_K is not None:
        raise ValueError("wall temperature requires a declared thermal wall")
    if surface is not None:
        flat.update(surface.amounts())
    if counters is not None:
        if set(counters) != {c.name for c in gas_accounting_coordinates(spec, network)}:
            raise ValueError("gas accounting counters must match the compiled layout")
        coordinates.update(counters)
    temperatures = (
        spec.initial_temperatures_K
        if temperatures_by_vessel_K is None
        else dict(temperatures_by_vessel_K)
    )
    if set(temperatures) != {n.vessel_id for n in spec.nodes}:
        raise ValueError("gas temperatures must cover exactly the original nodes")
    if not spec.energy_nodes and any(t != spec.temperature_K for t in temperatures.values()):
        raise ValueError("isothermal gas nodes must retain their declared temperature")
    for name, temperature in temperatures.items():
        _positive(temperature, "node temperature")
        if spec.energy_nodes and name != spec.anchor_vessel_id:
            coordinates[_key("node_temperature_K", name)] = temperature
    return CoupledState(
        flat,
        temperatures[spec.anchor_vessel_id],
        sum(
            gas_node_volume(
                n,
                sum(amounts_by_vessel[n.vessel_id].values()),
                temperatures[n.vessel_id],
                spec.pressure_control,
            )
            for n in spec.nodes
        ),
        time_s,
        boundary_heat_J=sum(coordinates[gas_bath_key(n.vessel_id)] for n in spec.nodes),
        heat_loss_J=(
            sum(coordinates[_key("environment_loss_J", n.vessel_id)] for n in spec.nodes)
            + (
                sum(
                    coordinates[wall_key(spec.thermal_wall.owner_id, kind)]
                    for kind in ("environment_conduction_J", "environment_radiation_J")
                )
                if spec.thermal_wall
                else 0.0
            )
            if spec.energy_nodes
            else 0.0
        ),
        coordinates=coordinates,
        work_J=coordinates[gas_work_key(spec.pressure_control.vessel_id)]
        if spec.pressure_control
        else 0.0,
    )


def gas_node_view(
    state: CoupledStateView,
    node: GasReactorNodeSpec,
    network: ReactionNetworkSpec,
    *,
    pressure_control: GasPressureControlSpec | None = None,
) -> CoupledStateView:
    network = gas_phase_network(network)
    amounts = {
        s: state.amounts_mol[gas_inventory_key(node.vessel_id, s)] for s in network.species_ids
    }
    temperature = state.coordinates.get(
        _key("node_temperature_K", node.vessel_id), state.temperature_K
    )
    volume = gas_node_volume(node, sum(amounts.values()), temperature, pressure_control)
    return CoupledStateView(
        amounts,
        temperature,
        # Signed event trials can cross zero n; rates use the same zero-stock
        # extension as the original solver. Accepted volume is checked separately.
        volume if volume > 0 else node.volume_L,
        state.time_s,
    )


def gas_node_pressure(
    node: GasReactorNodeSpec,
    total_mol: float,
    temperature_K: float,
    pressure_control: GasPressureControlSpec | None = None,
) -> float:
    control = pressure_control_for(node, pressure_control)
    if control is not None:
        return control.pressure_Pa
    return total_mol * R * temperature_K * 1000 / node.volume_L


def gas_connection_molar_rate(
    edge: GasFlowConnectionSpec,
    nodes: Mapping[str, GasReactorNodeSpec],
    views: Mapping[str, CoupledStateView],
    masses: Mapping[str, float],
) -> float:
    """One original finite flow law, using each endpoint's actual pressure."""
    source = views[edge.source_vessel_id]
    target = views[edge.target_vessel_id]
    total = sum(source.amounts_mol.values())
    rate = edge.coefficient if edge.enabled and total > 0 else 0.0
    if edge.mode == "pressure-valve":
        p1 = total * R * source.temperature_K * 1000 / source.volume_L
        p2 = sum(target.amounts_mol.values()) * R * target.temperature_K * 1000 / target.volume_L
        rate *= max(p1 - p2, 0.0)
    if edge.mode != "molar-flow" and total > 0:
        rate /= sum(source.amounts_mol[s] * masses[s] for s in source.amounts_mol) / total
    return rate


def gas_reactor_network_system(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
    surface_slots: Sequence[StateSlot] = (),
) -> CoupledSystem:
    """Compile native processes; no solver, clock or persistent inventory is added."""
    if spec.energy_nodes:
        from chemworld.physchem.gas_network_energy import gas_energy_system

        return gas_energy_system(
            spec, network, thermal_device=thermal_device, surface_slots=surface_slots
        )
    if thermal_device is not None:
        raise ValueError("finite thermal devices require the gas energy model")
    if spec.spatial_transport is not None:
        from chemworld.physchem.gas_spatial_reactor import gas_spatial_system

        return gas_spatial_system(spec, network)
    surface = GasSurfaceContext(spec, network, surface_slots) if spec.surfaces else None
    if surface is not None:
        network = surface.gas
    elif surface_slots:
        raise ValueError("undeclared surface slots cannot be ignored")
    if any(s.phase != "gas" for s in network.species):
        raise ValueError("gas network requires original gas-phase species")
    thermo = network.imported_gas_thermochemistry
    if set(thermo) != set(network.species_ids):
        raise ValueError("every gas species needs declared thermochemistry")
    for item in thermo.values():
        item.evaluate(spec.temperature_K)
    thermal = IdealGasThermodynamics(
        {
            gas_inventory_key(n.vessel_id, s): thermo[s]
            for n in spec.nodes
            for s in network.species_ids
        }
    )
    local_thermal = IdealGasThermodynamics(thermo)
    if surface is not None:
        from chemworld.physchem.gas_network_energy import _NetworkCalorics

        thermal = _NetworkCalorics(spec, network, surface)  # type: ignore[assignment]
    reactions = reaction_network_process(network, species_thermo=thermo)
    nodes = {n.vessel_id: n for n in spec.nodes}
    masses = {s.species_id: molecular_weight(s.composition) / 1000 for s in network.species}
    enthalpies = {
        s: thermo[s].evaluate(spec.temperature_K).enthalpy_J_mol for s in network.species_ids
    }
    local_ports = {
        n.vessel_id: tuple(
            MaterialPort(
                _key("local", edge.connection_id, s, direction),
                s,
                edge.source_vessel_id,
                edge.target_vessel_id,
                direction,
            )
            for edge in spec.connections
            for direction, vessel in (("out", edge.source_vessel_id), ("in", edge.target_vessel_id))
            if vessel == n.vessel_id
            for s in network.species_ids
        )
        for n in spec.nodes
    }

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        views = {
            key: gas_node_view(state, node, network, pressure_control=spec.pressure_control)
            for key, node in nodes.items()
        }
        local_reactions = {
            key: reactions.evaluate(view)
            if nodes[key].chemistry_enabled and sum(view.amounts_mol.values()) > 0
            else ProcessContribution()
            for key, view in views.items()
        }
        amounts, progress, transfers, counters = {}, {}, [], {}
        port_rates: dict[str, dict[str, float]] = {key: {} for key in nodes}
        for edge in spec.connections:
            source = views[edge.source_vessel_id].amounts_mol
            total = sum(source.values())
            # The common solver extends trial amounts to the nonnegative domain.
            # An empty trial has no outflow. Accepted nodes still stop at their
            # strictly positive declared floor; no empty state is committed.
            rate = gas_connection_molar_rate(edge, nodes, views, masses)
            energy = 0.0
            for species in network.species_ids:
                flow = rate * source[species] / total if total > 0 else 0.0
                transfers.append(
                    InternalTransfer(
                        _key("flow", edge.connection_id, species),
                        gas_inventory_key(edge.source_vessel_id, species),
                        gas_inventory_key(edge.target_vessel_id, species),
                        flow,
                    )
                )
                counters[gas_transfer_key(edge.connection_id, species)] = flow
                energy += flow * enthalpies[species]
                for direction, vessel in (
                    ("out", edge.source_vessel_id),
                    ("in", edge.target_vessel_id),
                ):
                    port_rates[vessel][_key("local", edge.connection_id, species, direction)] = flow
            counters[gas_enthalpy_key(edge.connection_id)] = energy
        for key, view in views.items():
            reaction = local_reactions[key]
            model = local_thermal
            if surface is not None:
                reaction = surface.chemistry(state, nodes[key], reaction)
                view = surface.view(state, nodes[key])
                model = surface.models[key]
            amounts.update(
                {
                    (surface.identities[key][s] if surface else gas_inventory_key(key, s)): rate
                    for s, rate in reaction.amount_rates_mol_s.items()
                }
            )
            progress.update(
                {
                    _key("reaction", key, r): rate
                    for r, rate in reaction.reaction_progress_mol_s.items()
                }
            )
            contribution = ProcessContribution(
                amount_rates_mol_s=reaction.amount_rates_mol_s,
                port_rates_mol_s=port_rates[key],
                port_enthalpies_J_mol={
                    p.port_id: enthalpies[p.inventory_key] for p in local_ports[key]
                },
            )
            work = (
                piston_work_rate(view, model, contribution, local_ports[key], isothermal=True)
                if pressure_control_for(nodes[key], spec.pressure_control)
                else 0.0
            )
            if pressure_control_for(nodes[key], spec.pressure_control):
                counters[gas_work_key(key)] = work
            if surface is None and sum(view.amounts_mol.values()) == 0:
                # Exact zero-inventory isothermal trial extension: no sensible
                # heat capacity is needed because dT=0. Use the same caloric
                # adapter and declared port enthalpies; never accept this state.
                derivatives = {
                    s: sum(
                        p.sign * port_rates[key][p.port_id]
                        for p in local_ports[key]
                        if p.inventory_key == s
                    )
                    for s in network.species_ids
                }
                counters[gas_bath_key(key)] = (
                    local_thermal.chemical_energy_rate_w(view, derivatives, ())
                    - sum(
                        p.sign * port_rates[key][p.port_id] * enthalpies[p.inventory_key]
                        for p in local_ports[key]
                    )
                    - work
                )
                continue

            def local_contribution(
                _: CoupledStateView,
                value: ProcessContribution = contribution,
            ) -> ProcessContribution:
                return value

            def local_flux(_: CoupledStateView, value: float = work) -> HeatFlux:
                return HeatFlux(work_W=value)

            local = assemble_coupled_rates(
                view,
                processes=(CoupledProcess("node-balance", local_contribution),),
                thermodynamics=model,
                heat_flux=local_flux,
                ports=local_ports[key],
                isothermal=True,
            )
            counters[gas_bath_key(key)] = local.heater_W
            if pressure_control_for(nodes[key], spec.pressure_control):
                counters[gas_work_key(key)] = work
        return ProcessContribution(amounts, progress, tuple(transfers), coordinate_rates=counters)

    def validate(state: CoupledStateView) -> None:
        if surface is not None:
            surface.validate(state)
        if state.temperature_K != spec.temperature_K:
            raise ValueError("network nodes must share the declared fixed temperature")
        if spec.pressure_control is None and not isclose(
            state.volume_L, sum(n.volume_L for n in spec.nodes), rel_tol=1e-12
        ):
            raise ValueError("assembly volume must equal declared node volumes")
        if set(state.coordinates) != {c.name for c in gas_network_coordinates(spec, network)}:
            raise ValueError("gas network accounting layout differs from declaration")
        for node in spec.nodes:
            total = sum(
                gas_node_view(
                    state, node, network, pressure_control=spec.pressure_control
                ).amounts_mol.values()
            )
            # Endpoint arithmetic is allowed; the solver does not accept negative inventory.
            if total <= 0 or total < node.minimum_amount_mol - (
                1e-14 + 1e-10 * max(total, node.minimum_amount_mol)
            ):
                raise ValueError("gas node below declared minimum amount")
            pressure = gas_node_pressure(node, total, spec.temperature_K, spec.pressure_control)
            if pressure > node.maximum_pressure_Pa * (1 + 1e-10):
                raise ValueError("gas node exceeds pressure ceiling")
            control = pressure_control_for(node, spec.pressure_control)
            if control is not None:
                validate_piston_volume(control, control.volume_L(total, spec.temperature_K))

    def node_events(node: GasReactorNodeSpec) -> tuple[CoupledEvent, CoupledEvent]:
        def minimum(state: CoupledStateView) -> float:
            return (
                sum(
                    gas_node_view(
                        state, node, network, pressure_control=spec.pressure_control
                    ).amounts_mol.values()
                )
                - node.minimum_amount_mol
            )

        def maximum(state: CoupledStateView) -> float:
            return node.maximum_pressure_Pa - gas_node_pressure(
                node,
                sum(
                    gas_node_view(
                        state, node, network, pressure_control=spec.pressure_control
                    ).amounts_mol.values()
                ),
                spec.temperature_K,
                spec.pressure_control,
            )

        return (
            CoupledEvent(f"gas:{node.vessel_id}:minimum_amount", minimum),
            CoupledEvent(f"gas:{node.vessel_id}:maximum_pressure", maximum),
        )

    events = tuple(event for node in spec.nodes for event in node_events(node)) + piston_events(
        spec, network
    )

    def boundary_work(state: CoupledStateView) -> HeatFlux:
        control = spec.pressure_control
        if control is None:
            return HeatFlux()
        return HeatFlux(work_W=evaluate(state).coordinate_rates[gas_work_key(control.vessel_id)])

    elements = {element for s in network.species for element in s.composition}
    return CoupledSystem(
        thermal,
        (CoupledProcess("gas-network", evaluate, validate_state=validate),),
        events=events,
        isothermal=True,
        heat_flux=boundary_work,
        coordinates=gas_network_coordinates(spec, network),
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


def advance_gas_reactor_network(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    state: CoupledState,
    duration_s: float,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
    surface_slots: Sequence[StateSlot] = (),
) -> tuple[CoupledAdvanceResult, dict[str, Any]]:
    if spec.energy_nodes:
        from chemworld.physchem.gas_network_energy import advance_gas_energy

        return advance_gas_energy(
            spec,
            network,
            state,
            duration_s,
            thermal_device=thermal_device,
            surface_slots=surface_slots,
        )
    system = gas_reactor_network_system(
        spec, network, thermal_device=thermal_device, surface_slots=surface_slots
    )
    surface = GasSurfaceContext(spec, network, surface_slots) if spec.surfaces else None
    if surface is not None:
        network = surface.gas
    result = project_gas_volumes(system.advance(state, duration_s), spec, network)
    final = result.final_state
    increments = {key: final.coordinates[key] - state.coordinates[key] for key in state.coordinates}
    flows: dict[str, dict[str, Any]] = {
        face: {
            "source": source,
            "target": target,
            "amounts_mol": {s: increments[gas_transfer_key(face, s)] for s in network.species_ids},
            "enthalpy_J": increments[gas_enthalpy_key(face)],
        }
        for face, source, target in gas_material_topology(spec)
    }
    thermal = IdealGasThermodynamics(network.imported_gas_thermochemistry)
    details: dict[str, dict[str, Any]] = {}
    for node in spec.nodes:
        name = node.vessel_id
        before, after = (
            gas_node_view(state, node, network, pressure_control=spec.pressure_control),
            gas_node_view(final, node, network, pressure_control=spec.pressure_control),
        )
        if surface is not None:
            model = surface.models[name]
            delta_u = model.internal_energy_j(surface.view(final, node)) - model.internal_energy_j(
                surface.view(state, node)
            )
        else:
            delta_u = thermal.internal_energy_j(after) - thermal.internal_energy_j(before)
        transferred_u = {
            key: sum(
                thermal.species_energy_j_mol(s, spec.temperature_K) * n
                for s, n in flow["amounts_mol"].items()
            )
            for key, flow in flows.items()
        }
        chemical_delta = delta_u - sum(
            transferred_u[key]
            * ((1 if flow["target"] == name else 0) - (1 if flow["source"] == name else 0))
            for key, flow in flows.items()
        )
        energy_in = sum(f["enthalpy_J"] for f in flows.values() if f["target"] == name)
        energy_out = sum(f["enthalpy_J"] for f in flows.values() if f["source"] == name)
        bath = increments[gas_bath_key(name)]
        work = increments.get(gas_work_key(name), 0.0)
        residual = delta_u - bath - energy_in + energy_out - work
        tolerance = 1e-7 + 2e-6 * (abs(bath) + abs(energy_in) + abs(energy_out))
        if abs(residual) > tolerance:
            raise RuntimeError(f"gas node energy failed to close: {name}: {residual} J")
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
                node, sum(after.amounts_mol.values()), spec.temperature_K, spec.pressure_control
            ),
            **({"boundary_work_on_J": work} if spec.pressure_control else {}),
            "bath_heat_J": bath,
            "enthalpy_in_J": energy_in,
            "enthalpy_out_J": energy_out,
            "internal_energy_change_J": delta_u,
            "chemical_energy_change_J": chemical_delta,
            "energy_residual_J": residual,
        }
    bath_sum = sum(v["bath_heat_J"] for v in details.values())
    work_sum = sum(v.get("boundary_work_on_J", 0.0) for v in details.values())
    if spec.pressure_control and not isclose(
        work_sum, final.work_J - state.work_J, abs_tol=1e-7, rel_tol=1e-7
    ):
        raise RuntimeError("piston work differs from the original common energy ledger")
    spatial_diagnostic = {}
    if spec.spatial_transport is not None:
        from chemworld.physchem.gas_spatial_reactor import (
            spatial_bulk_key,
            spatial_diffusion_heat_key,
            spatial_diffusion_key,
            spatial_rates,
        )

        for face, _, _ in gas_material_topology(spec):
            flows[face].update(
                {
                    "bulk_mol": increments[spatial_bulk_key(face)],
                    "diffusion_mol": {
                        s: increments[spatial_diffusion_key(face, s)] for s in network.species_ids
                    },
                    "diffusion_enthalpy_J": increments[spatial_diffusion_heat_key(face)],
                }
            )
        spatial_diagnostic = {"spatial": spatial_rates(spec, network, final)[1]}
    if not isclose(
        bath_sum, final.boundary_heat_J - state.boundary_heat_J, abs_tol=1e-7, rel_tol=1e-7
    ):
        raise RuntimeError("node thermostat duties differ from the common energy ledger")
    return result, {
        **spatial_diagnostic,
        "nodes": details,
        "connections": flows,
        "bath_heat_J": bath_sum,
        **({"boundary_work_on_J": work_sum} if spec.pressure_control else {}),
        "actual_duration_s": result.actual_duration_s,
        "stopping_event": result.stopping_event,
        "energy_residual_J": result.energy_balance_residual_J,
        "element_residuals_mol": dict(result.conservation_residuals),
    }
