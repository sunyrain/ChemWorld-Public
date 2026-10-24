"""Finite boundary ports for the original isothermal low-Mach gas grid.

Only the grid shares pressure. The two original reservoir vessels retain their
own finite inventories and pressures. Prescribed flow is a kinematic control,
not a pressure-drop, compressor or momentum model.
"""

from __future__ import annotations

from math import fsum, sqrt
from typing import TYPE_CHECKING, Any

from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chemworld.physchem.coupled_system import CoupledStateView
    from chemworld.physchem.gas_reactor_network import GasFlowConnectionSpec, GasReactorNetworkSpec
    from chemworld.physchem.reaction_network import ReactionNetworkSpec


def spatial_boundary_connections(
    spec: GasReactorNetworkSpec,
) -> tuple[GasFlowConnectionSpec, GasFlowConnectionSpec]:
    """Resolve the two original connections without duplicating their declaration."""
    spatial = spec.spatial_transport
    if spatial is None or spatial.boundary_mode != "finite-flow" or len(spec.connections) != 2:
        raise ValueError("finite spatial flow needs exactly two declared boundary connections")
    cells = set(spatial.cell_vessel_ids)
    left, right = spatial.cell_vessel_ids[0], spatial.cell_vessel_ids[-1]
    inlet = [
        e
        for e in spec.connections
        if e.source_vessel_id not in cells and e.target_vessel_id == left
    ]
    outlet = [
        e
        for e in spec.connections
        if e.source_vessel_id == right and e.target_vessel_id not in cells
    ]
    if len(inlet) != 1 or len(outlet) != 1:
        raise ValueError("finite spatial connections must feed the first cell and drain the last")
    feed, drain = inlet[0], outlet[0]
    tanks = {feed.source_vessel_id, drain.target_vessel_id}
    nodes = {n.vessel_id: n for n in spec.nodes}
    if len(tanks) != 2 or set(nodes) != cells | tanks:
        raise ValueError("finite spatial flow needs two distinct declared finite tanks")
    if any(nodes[name].chemistry_enabled for name in tanks):
        raise ValueError("finite spatial boundary tanks must explicitly disable gas chemistry")
    if any(e.mode != "molar-flow" for e in (feed, drain)):
        raise ValueError("finite spatial boundary flow supports prescribed molar flow only")
    identities = [e.connection_id for e in spec.connections]
    identities.extend(face for face, _, _ in spatial.topology)
    if len(identities) != len(set(identities)):
        raise ValueError("spatial boundary and interior face identities must be distinct")
    return feed, drain


def spatial_boundary_rates(spec: GasReactorNetworkSpec) -> tuple[float, float]:
    return tuple(  # type: ignore[return-value]
        e.coefficient if e.enabled else 0.0 for e in spatial_boundary_connections(spec)
    )


def add_spatial_boundary_faces(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    views: Mapping[str, CoupledStateView],
    packet: dict[str, Any],
) -> None:
    """Add actual donor-composition boundary fluxes to a transient face packet."""
    spatial = spec.spatial_transport
    assert spatial is not None
    thermo = network.imported_gas_thermochemistry
    masses = {s.species_id: molecular_weight(s.composition) / 1000 for s in network.species}
    boundary = {}
    rates = spatial_boundary_rates(spec)
    for index, (edge, rate) in enumerate(
        zip(spatial_boundary_connections(spec), rates, strict=True)
    ):
        donor = views[edge.source_vessel_id].amounts_mol
        total = fsum(donor.values())
        if total <= 0:
            raise ValueError("finite spatial donor requires positive actual inventory")
        composition = {s: donor[s] / total for s in network.species_ids}
        amounts = {s: rate * x for s, x in composition.items()}
        enthalpy = fsum(
            thermo[s].evaluate(spec.temperature_K).enthalpy_J_mol * n for s, n in amounts.items()
        )
        area = float(spatial.mesh.face_areas_m2[0 if index == 0 else -1])
        concentration = packet["pressure_Pa"] / (R * spec.temperature_K)
        velocity = rate / (area * concentration)
        cp = fsum(
            x * thermo[s].evaluate(spec.temperature_K).cp_J_mol_K for s, x in composition.items()
        )
        mass = fsum(x * masses[s] for s, x in composition.items())
        sound = sqrt(cp / (cp - R) * R * spec.temperature_K / mass)
        packet["faces"][edge.connection_id] = {
            "source": edge.source_vessel_id,
            "target": edge.target_vessel_id,
            "bulk_mol_s": rate,
            "diffusion_mol_s": dict.fromkeys(network.species_ids, 0.0),
            "convection_mol_s": amounts,
            "amount_rates_mol_s": amounts,
            "diffusion_enthalpy_W": 0.0,
            "convection_enthalpy_W": enthalpy,
            "enthalpy_W": enthalpy,
            "molar_velocity_m_s": velocity,
            "mass_velocity_m_s": velocity,
            "mach": abs(velocity) / sound,
            "pressure_Pa": packet["pressure_Pa"],
            "temperature_K": spec.temperature_K,
            "mole_fractions": composition,
            "diffusive_molar_residual_mol_s": 0.0,
        }
        boundary["left" if index == 0 else "right"] = amounts
    packet["boundary_species_rates_mol_s"] = boundary
    packet["maximum_mach"] = max(f["mach"] for f in packet["faces"].values())
    feed, drain = spatial_boundary_connections(spec)
    packet["tank_total_rates_mol_s"] = {
        feed.source_vessel_id: -rates[0],
        drain.target_vessel_id: rates[1],
    }
