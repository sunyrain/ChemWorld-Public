"""Finite vacuum pot/bottle model using the shared n/H phase strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from chemworld.physchem.flash_units import FlashSystemSpec, VLEPropertyPackage, phase_advection_rate
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.phase_inventory import (
    PhaseInventoryAdvance,
    PhaseInventoryRates,
    integrate_phase_inventory,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K

if TYPE_CHECKING:
    from chemworld.physchem.concentration_units import VacuumConcentratorSpec

COUPLED_CONCENTRATION_MODEL_ID = "state_coupled_vacuum_concentration_v1"


@dataclass(frozen=True)
class CoupledConcentrationSpec:
    properties: VLEPropertyPackage
    equipment: VacuumConcentratorSpec
    solvent_component_ids: tuple[str, ...]
    target_component_id: str | None = None
    pressure_Pa: float = 1e5
    maximum_pressure_Pa: float = 1e5
    pot_capacity_J_K: float = 1.0
    bottle_capacity_J_K: float = 0.2
    reference_temperature_K: float = 298.15
    coolant_temperature_K: float = 270.0
    condenser_conductance_W_K: float = 0.5
    maximum_condenser_power_W: float = 15.0
    collector_capacity_mol: float = 0.03
    condenser_gas_limit_mol: float = 2e-4
    vapor_withdrawal_rate_s_inv: float = 2.0
    liquid_molar_holdup_L_mol: float = 0.05
    target_solvent_remaining_fraction: float = 0.4
    minimum_target_recovery: float = 0.95
    minimum_liquid_solvent_mol: float = 1e-7

    def __post_init__(self):
        from chemworld.physchem.concentration_units import VacuumConcentratorSpec

        object.__setattr__(self, "solvent_component_ids", tuple(self.solvent_component_ids))
        ids = set(self.properties.component_ids)
        if (
            not isinstance(self.equipment, VacuumConcentratorSpec)
            or self.properties.vapor_eos is not None
            or self.properties.activity_model.model != "ideal"
            or not self.solvent_component_ids
            or len(set(self.solvent_component_ids)) != len(self.solvent_component_ids)
            or not set(self.solvent_component_ids) <= set(self.properties.gas_references)
            or (
                self.target_component_id is not None
                and (
                    self.target_component_id not in ids
                    or self.target_component_id in self.solvent_component_ids
                )
            )
        ):
            raise ValueError(
                "coupled concentration needs ideal properties and explicit "
                "volatile/target identities"
            )
        for name in (
            "pressure_Pa",
            "maximum_pressure_Pa",
            "pot_capacity_J_K",
            "bottle_capacity_J_K",
            "reference_temperature_K",
            "coolant_temperature_K",
            "condenser_conductance_W_K",
            "collector_capacity_mol",
            "condenser_gas_limit_mol",
            "vapor_withdrawal_rate_s_inv",
            "liquid_molar_holdup_L_mol",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"coupled concentration requires finite positive {name}")
        if (
            not self.equipment.minimum_pressure_Pa <= self.pressure_Pa <= self.maximum_pressure_Pa
            or not isfinite(self.maximum_condenser_power_W)
            or self.maximum_condenser_power_W < 0
            or not isfinite(self.minimum_liquid_solvent_mol)
            or self.minimum_liquid_solvent_mol < 0
            or any(
                not isfinite(v) or not 0 <= v <= 1
                for v in (self.target_solvent_remaining_fraction, self.minimum_target_recovery)
            )
        ):
            raise ValueError("coupled concentration controls leave their finite equipment domain")
        lower, upper = self.properties.temperature_bounds_K
        if not (lower < self.equipment.maximum_temperature_K < upper
                and lower <= self.coolant_temperature_K < self.equipment.maximum_temperature_K
                and lower <= self.reference_temperature_K <= upper):
            raise ValueError("concentrator thermal stop must lie inside its property domain")
        if any(
            abs(ref.pressure_Pa - self.pressure_Pa) > 1e-8
            for ref in self.properties.nonvolatile_references.values()
        ):
            raise ValueError("liquid-only caloric references must match the fixed pressure")

    @property
    def capacities_j_k(self):
        return self.pot_capacity_J_K, self.bottle_capacity_J_K

    @property
    def cell_count(self):
        return 2

    @cached_property
    def _models(self):
        return tuple(
            FlashSystemSpec(
                self.properties,
                pressure_Pa=self.pressure_Pa,
                support_heat_capacity_J_K=capacity,
                support_reference_temperature_K=self.reference_temperature_K,
                environment_temperature_K=self.coolant_temperature_K,
                loss_conductance_W_K=0,
                stop_temperature_K=self.equipment.maximum_temperature_K,
                liquid_molar_holdup_L_mol=self.liquid_molar_holdup_L_mol,
            )
            for capacity in self.capacities_j_k
        )

    def at_tp(self, index, amounts, temperature):
        return self._models[index].at_tp(amounts, temperature)

    def at_ph(self, index, amounts, enthalpy):
        return self._models[index].at_ph(amounts, enthalpy)

    def enthalpy_bounds(self, index, amounts):
        bounds = self.properties.ideal_enthalpy_bounds(
            amounts, self.pressure_Pa, self.capacities_j_k[index], self.reference_temperature_K
        )
        return (
            bounds
            if bounds is not None
            else tuple(
                self.at_tp(index, amounts, t).total_enthalpy_j
                for t in self.properties.temperature_bounds_K
            )
        )

    def volume_l(self, phase):
        return (
            sum(phase.liquid_amounts_mol.values()) * self.liquid_molar_holdup_L_mol
            + sum(phase.vapor_amounts_mol.values())
            * R_J_PER_MOL_K
            * phase.temperature_K
            / self.pressure_Pa
            * 1000
        )

    def to_dict(self):
        from dataclasses import fields

        return {
            f.name: self.properties.to_dict()
            if f.name == "properties"
            else asdict(self.equipment)
            if f.name == "equipment"
            else getattr(self, f.name)
            for f in fields(self)
        }

    @classmethod
    def from_dict(cls, payload):
        from chemworld.physchem.concentration_units import VacuumConcentratorSpec

        values = dict(payload)
        values["properties"] = VLEPropertyPackage.from_dict(values["properties"])
        values["equipment"] = VacuumConcentratorSpec(**values["equipment"])
        return cls(**values)


@dataclass(frozen=True)
class CoupledConcentrationState:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    time_s: float = 0.0
    heater_energy_J: float = 0.0
    condenser_energy_J: float = 0.0
    reference_amounts_mol: dict[str, float] = field(default_factory=dict)
    outlet_amounts_mol: dict[str, float] = field(default_factory=dict)
    outlet_enthalpy_J: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "amounts_mol", tuple(dict(n) for n in self.amounts_mol))
        object.__setattr__(self, "enthalpies_J", tuple(self.enthalpies_J))
        object.__setattr__(self, "outlet_amounts_mol", dict(self.outlet_amounts_mol))
        if (
            len(self.amounts_mol) != 2
            or len(self.enthalpies_J) != 2
            or any(not isfinite(v) or v < 0 for n in self.amounts_mol for v in n.values())
            or any(not isfinite(v) for v in (*self.enthalpies_J, self.outlet_enthalpy_J))
            or any(
                not isfinite(v) or v < 0
                for v in (
                    self.time_s,
                    self.heater_energy_J,
                    self.condenser_energy_J,
                    *self.outlet_amounts_mol.values(),
                )
            )
        ):
            raise ValueError("concentration state needs finite material/H and nonnegative counters")
        reference = self.reference_amounts_mol or {
            key: sum(n.get(key, 0) for n in self.amounts_mol) + self.outlet_amounts_mol.get(key, 0)
            for key in self.amounts_mol[0]
        }
        if set(reference) != set(self.amounts_mol[0]) or any(
            not isfinite(v) or v < 0 for v in reference.values()
        ):
            raise ValueError("concentration references must identify original material amounts")
        object.__setattr__(self, "reference_amounts_mol", dict(reference))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CoupledConcentrationResult(PhaseInventoryAdvance):
    spec: CoupledConcentrationSpec = field(kw_only=True)
    requested_duration_s: float = field(kw_only=True)

    @property
    def phases(self):
        return tuple(
            self.spec.at_ph(i, n, h)
            for i, (n, h) in enumerate(
                zip(self.final_state.amounts_mol, self.final_state.enthalpies_J, strict=True)
            )
        )

    @property
    def solvent_remaining_fraction(self):
        initial = sum(
            self.initial_state.reference_amounts_mol[k] for k in self.spec.solvent_component_ids
        )
        return (
            sum(self.phases[0].liquid_amounts_mol[k] for k in self.spec.solvent_component_ids)
            / initial
            if initial
            else None
        )

    @property
    def endpoint_met(self):
        value = self.solvent_remaining_fraction
        return value is None or value <= self.spec.target_solvent_remaining_fraction + 1e-10

    @property
    def vent_amounts_mol(self):
        return {
            k: self.final_state.outlet_amounts_mol.get(k, 0)
            - self.initial_state.outlet_amounts_mol.get(k, 0)
            for k in self.spec.properties.component_ids
        }

    def to_dict(self):
        pot, bottle = self.phases
        return {
            "model_id": COUPLED_CONCENTRATION_MODEL_ID,
            "state": self.final_state.to_dict(),
            "requested_duration_s": self.requested_duration_s,
            "elapsed_time_s": self.actual_duration_s,
            "pot_liquid_amounts_mol": pot.liquid_amounts_mol,
            "pot_vapor_amounts_mol": pot.vapor_amounts_mol,
            "collector_liquid_amounts_mol": bottle.liquid_amounts_mol,
            "collector_vapor_amounts_mol": bottle.vapor_amounts_mol,
            "vent_amounts_mol": self.vent_amounts_mol,
            "temperature_K": [pot.temperature_K, bottle.temperature_K],
            "solvent_remaining_fraction": self.solvent_remaining_fraction,
            "endpoint_met": self.endpoint_met,
            "stopping_event": self.stopping_event,
            "component_residuals_mol": self.component_residuals_mol,
            "enthalpy_residual_J": self.enthalpy_residual_J,
            "solver_diagnostic": self.solver_diagnostic,
            "heater_energy_J": self.final_state.heater_energy_J
            - self.initial_state.heater_energy_J,
            "cooling_energy_J": self.final_state.condenser_energy_J
            - self.initial_state.condenser_energy_J,
            "vent_enthalpy_J": self.final_state.outlet_enthalpy_J
            - self.initial_state.outlet_enthalpy_J,
        }


def concentration_rates(spec, state, heater: ThermalDeviceSpec, *, flashes, transport_enabled=True):
    ids = spec.properties.component_ids
    pot, bottle = flashes
    dn, dh = np.zeros((2, len(ids))), np.zeros(2)
    rate = (
        min(
            spec.vapor_withdrawal_rate_s_inv * sum(pot.vapor_amounts_mol.values()),
            spec.equipment.max_evaporation_rate_mol_s,
        )
        if transport_enabled
        else 0.0
    )
    outgoing, power = phase_advection_rate(pot.vapor_amounts_mol, pot.vapor_enthalpy_J, rate)
    capture = spec.equipment.condenser_recovery_fraction
    flux = np.array([outgoing[k] for k in ids])
    dn[0] -= flux
    dn[1] += capture * flux
    dh[0] -= power
    dh[1] += capture * power
    heat = heater.delivered_power_w(pot.temperature_K)
    cool = min(
        spec.maximum_condenser_power_W,
        spec.condenser_conductance_W_K * max(bottle.temperature_K - spec.coolant_temperature_K, 0),
    )
    dh[0] += heat
    dh[1] -= cool
    return PhaseInventoryRates(
        dn,
        dh,
        heat,
        cool,
        (
            {
                "source_cell": 0,
                "target_cell": 1,
                "phase": "gas",
                "amounts_mol_s": {k: capture * outgoing[k] for k in ids},
                "enthalpy_W": capture * power,
            },
            {
                "source_cell": 0,
                "target": "vent",
                "phase": "gas",
                "amounts_mol_s": {k: (1 - capture) * outgoing[k] for k in ids},
                "enthalpy_W": (1 - capture) * power,
            },
        ),
        outlet_amounts_mol_s={k: (1 - capture) * outgoing[k] for k in ids},
        outlet_enthalpy_W=(1 - capture) * power,
    )


def advance_coupled_concentration(
    spec,
    state,
    *,
    heater,
    duration_s,
    transport_enabled=True,
    enforce_endpoint=True,
    evaluation_times_s=None,
    max_step_s=None,
    rtol=1e-8,
    progress=None,
    maximum_volume_L=None,
):
    if (
        len(state.amounts_mol) != 2
        or any(set(n) != set(spec.properties.component_ids) for n in state.amounts_mol)
        or set(state.outlet_amounts_mol) - set(spec.properties.component_ids)
        or not isfinite(duration_s)
        or not 0 <= duration_s <= spec.equipment.maximum_duration_s
        or heater.mode != "finite_power"
        or heater.maximum_power_W > spec.equipment.max_heater_power_W
    ):
        raise ValueError("concentration controls/inventories leave the declared equipment domain")
    initial_phases = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True))
    )
    if (
        spec.volume_l(initial_phases[0]) > spec.equipment.max_working_volume_L + 1e-12
        or sum(state.amounts_mol[1].values()) > spec.collector_capacity_mol + 1e-12
    ):
        raise ValueError("concentration state exceeds its pot or bottle capacity")
    reference_solvent = sum(state.reference_amounts_mol[k] for k in spec.solvent_component_ids)
    events = []

    def event(name, value):
        events.append((name, value))

    if transport_enabled:
        if enforce_endpoint and spec.target_solvent_remaining_fraction > 0:
            event(
                "solvent_endpoint",
                lambda candidate, flashes: (
                    sum(flashes[0].liquid_amounts_mol[k] for k in spec.solvent_component_ids)
                    - reference_solvent * spec.target_solvent_remaining_fraction
                ),
            )
        event(
            "liquid_solvent_exhausted",
            lambda candidate, flashes: (
                sum(flashes[0].liquid_amounts_mol[k] for k in spec.solvent_component_ids)
                - spec.minimum_liquid_solvent_mol
            ),
        )
        event(
            "collector_capacity",
            lambda candidate, flashes: (
                spec.collector_capacity_mol - sum(candidate.amounts_mol[1].values())
            ),
        )
        event(
            "condenser_saturation",
            lambda candidate, flashes: (
                spec.condenser_gas_limit_mol - sum(flashes[1].vapor_amounts_mol.values())
            ),
        )
        if spec.equipment.minimum_residual_volume_L > 0:
            event(
                "minimum_residual_volume",
                lambda candidate, flashes: (
                    sum(flashes[0].liquid_amounts_mol.values()) * spec.liquid_molar_holdup_L_mol
                    - spec.equipment.minimum_residual_volume_L
                ),
            )
        if (
            spec.target_component_id is not None
            and spec.minimum_target_recovery > 0
            and state.reference_amounts_mol[spec.target_component_id] > 0
        ):
            key = spec.target_component_id
            event(
                "target_recovery_limit",
                lambda candidate, flashes: (
                    sum((flashes[0].liquid_amounts_mol[key], flashes[0].vapor_amounts_mol[key]))
                    - state.reference_amounts_mol[key] * spec.minimum_target_recovery
                ),
            )
    # With closed transport and disabled heat the pot is held, while the bottle
    # may safely cool from an already reached operating boundary.
    for i in range(2):
        if transport_enabled or (heater.enabled and heater.maximum_power_W):
            event(
                f"temperature_limit_{i}",
                lambda candidate, flashes, index=i: (
                    spec.equipment.maximum_temperature_K - flashes[index].temperature_K
                ),
            )
            event(
                f"minimum_temperature_{i}",
                lambda candidate, flashes, index=i: (
                    candidate.enthalpies_J[index]
                    - spec.enthalpy_bounds(index, candidate.amounts_mol[index])[0]
                ),
            )
    if transport_enabled or (heater.enabled and heater.maximum_power_W):
        event(
            "pot_capacity",
            lambda candidate, flashes: (
                spec.equipment.max_working_volume_L - spec.volume_l(flashes[0])
            ),
        )
    if transport_enabled and maximum_volume_L is not None:
        event(
            "vessel_volume",
            lambda candidate, flashes: (
                maximum_volume_L - sum(spec.volume_l(phase) for phase in flashes)
            ),
        )
    return integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=lambda candidate, flashes: concentration_rates(
            spec, candidate, heater, flashes=flashes, transport_enabled=transport_enabled
        ),
        closure_indices=range(2),
        stop_functions=events,
        enthalpy_bounds=spec.enthalpy_bounds,
        result_type=lambda *args: CoupledConcentrationResult(
            *args, spec=spec, requested_duration_s=duration_s
        ),
        evaluation_times_s=evaluation_times_s,
        max_step_s=max_step_s,
        rtol=rtol,
        progress=progress,
        solver_id="coupled_concentration_enthalpy_bdf_v1",
        open_outlet=True,
        initial_boundary="stop",
    )


@dataclass(frozen=True)
class CoupledConcentrationRequest:
    spec: CoupledConcentrationSpec
    state: CoupledConcentrationState
    duration_s: float
    heater: ThermalDeviceSpec
    transport_enabled: bool = True
    enforce_endpoint: bool = True


def simulate_coupled_concentration(request):
    """Standalone request and native session select the same contribution/solve."""
    return advance_coupled_concentration(
        request.spec,
        request.state,
        heater=request.heater,
        duration_s=request.duration_s,
        transport_enabled=request.transport_enabled,
        enforce_endpoint=request.enforce_endpoint,
    )
