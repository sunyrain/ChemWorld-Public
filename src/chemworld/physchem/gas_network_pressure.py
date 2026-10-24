"""A finite-stroke, constant-load ideal-gas piston in the original gas network.

Amounts and temperature remain the only evolving physical stock. Volume follows
the EOS; this declaration stores neither volume history nor mechanical energy.
Work is positive into the gas and has its own original accounting coordinate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any

from chemworld.physchem.coupled_system import (
    CoupledAdvanceResult,
    CoupledEvent,
    CoupledState,
    CoupledStateView,
    MaterialPort,
    ProcessContribution,
)
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class GasPressureControlSpec:
    """Constant external mechanical pressure with explicit finite stroke limits."""

    vessel_id: str
    pressure_Pa: float
    minimum_volume_L: float
    maximum_volume_L: float

    def __post_init__(self) -> None:
        if not isinstance(self.vessel_id, str) or not self.vessel_id:
            raise ValueError("pressure control needs an original vessel identity")
        for value in (self.pressure_Pa, self.minimum_volume_L, self.maximum_volume_L):
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError("piston pressure and finite stroke bounds must be positive")
        if self.minimum_volume_L >= self.maximum_volume_L:
            raise ValueError("piston stroke requires distinct ordered volume bounds")

    def volume_L(self, total_mol: float, temperature_K: float) -> float:  # noqa: N802
        return total_mol * R * temperature_K * 1000 / self.pressure_Pa

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasPressureControlSpec:
        return cls(**dict(value))


def gas_work_key(vessel_id: str) -> str:
    return json.dumps(["boundary_work_on_J", vessel_id], separators=(",", ":"))


def pressure_control_for(
    node: Any, control: GasPressureControlSpec | None
) -> GasPressureControlSpec | None:
    return control if control is not None and control.vessel_id == node.vessel_id else None


def gas_node_volume(
    node: Any, total_mol: float, temperature_K: float, control: GasPressureControlSpec | None = None
) -> float:
    piston = pressure_control_for(node, control)
    return node.volume_L if piston is None else piston.volume_L(total_mol, temperature_K)


def piston_work_rate(
    view: CoupledStateView,
    model: Any,
    contribution: ProcessContribution,
    ports: Sequence[MaterialPort],
    *,
    net_heat_W: float = 0.0,
    isothermal: bool = False,
) -> float:
    """Eliminate constant-P dV/dt without changing the original U/Cv adapter."""
    derivatives = dict.fromkeys(view.amounts_mol, 0.0)
    derivatives.update(contribution.amount_rates_mol_s)
    flow_H = 0.0
    for port in ports:
        rate = port.sign * contribution.port_rates_mol_s[port.port_id]
        derivatives[port.inventory_key] += rate
        flow_H += rate * contribution.port_enthalpies_J_mol[port.port_id]
    total = sum(view.amounts_mol.values())
    amount_rate = sum(derivatives.values())
    temperature_rate = 0.0
    if not isothermal:
        capacity_p = model.heat_capacity_j_k(view) + total * R
        chemical_h = (
            model.chemical_energy_rate_w(view, derivatives, (contribution,))
            + R * view.temperature_K * amount_rate
        )
        temperature_rate = (net_heat_W + flow_H - chemical_h) / capacity_p
    return -R * (view.temperature_K * amount_rate + total * temperature_rate)


def piston_events(spec: Any, network: Any) -> tuple[CoupledEvent, ...]:
    from chemworld.physchem.gas_reactor_network import gas_node_view

    control = spec.pressure_control
    if control is None:
        return ()
    node = next(n for n in spec.nodes if n.vessel_id == control.vessel_id)

    def volume(state: CoupledStateView) -> float:
        local = gas_node_view(state, node, network, pressure_control=control)
        return control.volume_L(sum(local.amounts_mol.values()), local.temperature_K)

    return (
        CoupledEvent(
            f"gas:{node.vessel_id}:minimum_volume", lambda s: volume(s) - control.minimum_volume_L
        ),
        CoupledEvent(
            f"gas:{node.vessel_id}:maximum_volume", lambda s: control.maximum_volume_L - volume(s)
        ),
    )


def validate_piston_volume(control: GasPressureControlSpec, volume_L: float) -> None:
    tolerance = 1e-10 * control.maximum_volume_L
    if not control.minimum_volume_L - tolerance <= volume_L <= control.maximum_volume_L + tolerance:
        raise ValueError("gas piston lies outside its declared finite stroke")


def project_gas_volumes(
    result: CoupledAdvanceResult, spec: Any, network: Any
) -> CoupledAdvanceResult:
    """Project the public transient result; no extra integrated volume coordinate."""
    if spec.pressure_control is None:
        return result
    from chemworld.physchem.gas_reactor_network import gas_node_view

    def project(state: CoupledState) -> CoupledState:
        return replace(
            state,
            volume_L=sum(
                gas_node_view(state, node, network, pressure_control=spec.pressure_control).volume_L
                for node in spec.nodes
            ),
        )

    return replace(
        result,
        initial_state=project(result.initial_state),
        final_state=project(result.final_state),
        trajectory=tuple(project(s) for s in result.trajectory),
    )
