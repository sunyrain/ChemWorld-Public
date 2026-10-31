"""Finite equilibrium compartments with one material/enthalpy advance.

The reboiler, trays, condenser and finite collector exchange actual phase
streams. PH is algebraic; no compartment owns a clock or a second material
inventory. This numerical strategy is independent of the session/transactions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
from scipy.sparse import lil_matrix

from chemworld.physchem.distillation_units import ShortcutColumnSpec
from chemworld.physchem.flash_units import (
    VLEPropertyPackage,
    ph_flash,
    phase_advection_rate,
    tp_flash,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.phase_inventory import (
    PhaseInventoryAdvance,
    PhaseInventoryRates,
    integrate_phase_inventory,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K


class ColumnOperatingLimitError(ValueError):
    """The requested advance starts at an already reached equipment limit."""


@dataclass(frozen=True)
class DynamicColumnSpec:
    properties: VLEPropertyPackage
    equipment: ShortcutColumnSpec
    pressure_Pa: float = 1e5
    liquid_overflow_rate_s_inv: float = 1.0
    vapor_withdrawal_rate_s_inv: float = 2.0
    liquid_setpoint_mol: float = 0.002
    tray_maximum_amount_mol: float = 0.02
    collector_capacity_mol: float = 0.01
    reboiler_capacity_J_K: float = 1.0
    tray_capacity_J_K: float = 0.2
    condenser_capacity_J_K: float = 0.2
    reference_temperature_K: float = 298.15
    coolant_temperature_K: float = 300.0
    condenser_conductance_W_K: float = 0.5
    liquid_molar_holdup_L_mol: float = 0.05

    def __post_init__(self) -> None:
        n = self.equipment.theoretical_stages
        if (
            n != int(n)
            or not 2 <= n <= 8
            or self.equipment.stage_efficiency != 1
            or len(self.properties.component_ids) != 2
            or not self.equipment.minimum_pressure_Pa
            <= self.pressure_Pa
            <= (self.equipment.maximum_pressure_Pa)
        ):
            raise ValueError("dynamic column requires 2-8 equilibrium trays and declared binary P")
        for name in (
            "pressure_Pa",
            "liquid_overflow_rate_s_inv",
            "vapor_withdrawal_rate_s_inv",
            "liquid_setpoint_mol",
            "tray_maximum_amount_mol",
            "collector_capacity_mol",
            "reboiler_capacity_J_K",
            "tray_capacity_J_K",
            "condenser_capacity_J_K",
            "reference_temperature_K",
            "coolant_temperature_K",
            "condenser_conductance_W_K",
            "liquid_molar_holdup_L_mol",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"dynamic column requires finite positive {name}")
        if self.liquid_setpoint_mol >= self.tray_maximum_amount_mol:
            raise ValueError("liquid setpoint must leave finite tray capacity")
        if not (
            self.properties.temperature_bounds_K[0]
            < self.equipment.maximum_temperature_K
            < self.properties.temperature_bounds_K[1]
        ):
            raise ValueError("column thermal stop must lie within its property domain")

    @property
    def tray_count(self):
        return int(self.equipment.theoretical_stages)

    @property
    def condenser_index(self):
        return self.tray_count + 1

    @property
    def collector_index(self):
        return self.tray_count + 2

    @property
    def cell_count(self):
        return self.tray_count + 3

    @property
    def capacities_j_k(self):
        return (
            self.reboiler_capacity_J_K,
            *((self.tray_capacity_J_K,) * self.tray_count),
            self.condenser_capacity_J_K,
            0.0,
        )

    def at_tp(self, index, amounts, temperature):
        return tp_flash(
            self.properties,
            amounts,
            temperature_K=temperature,
            pressure_Pa=self.pressure_Pa,
            support_heat_capacity_J_K=self.capacities_j_k[index],
            support_reference_temperature_K=self.reference_temperature_K,
        )

    def at_ph(self, index, amounts, enthalpy):
        if index == self.collector_index and sum(amounts.values()) == 0:
            if abs(enthalpy) > 1e-8:
                raise ValueError("empty collector cannot own a material enthalpy")
            return None
        return ph_flash(
            self.properties,
            amounts,
            pressure_Pa=self.pressure_Pa,
            enthalpy_J=enthalpy,
            support_heat_capacity_J_K=self.capacities_j_k[index],
            support_reference_temperature_K=self.reference_temperature_K,
        )

    def to_dict(self):
        from dataclasses import fields

        return {
            f.name: self.properties.to_dict()
            if f.name == "properties"
            else self.equipment.to_dict()
            if f.name == "equipment"
            else getattr(self, f.name)
            for f in fields(self)
        }

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["properties"] = VLEPropertyPackage.from_dict(values["properties"])
        values["equipment"] = ShortcutColumnSpec(**values["equipment"])
        return cls(**values)


@dataclass(frozen=True)
class DynamicColumnState:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    time_s: float = 0.0
    heater_energy_J: float = 0.0
    condenser_energy_J: float = 0.0
    withdrawn_product_mol: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "amounts_mol", tuple(dict(v) for v in self.amounts_mol))
        object.__setattr__(self, "enthalpies_J", tuple(self.enthalpies_J))
        if (
            len(self.amounts_mol) != len(self.enthalpies_J)
            or any(not isfinite(v) or v < 0 for p in self.amounts_mol for v in p.values())
            or any(not isfinite(v) for v in self.enthalpies_J)
            or any(
                not isfinite(v) or v < 0
                for v in (
                    self.time_s,
                    self.heater_energy_J,
                    self.condenser_energy_J,
                    self.withdrawn_product_mol,
                )
            )
        ):
            raise ValueError("column state requires finite inventories, H and resource counters")


@dataclass(frozen=True)
class DynamicColumnRates(PhaseInventoryRates):
    """Column port contributions to the common phase-inventory strategy."""


def column_rates(spec, state, heater, reflux_ratio, *, flashes=None):
    """Evaluate port contributions; each stream debits and credits the same flux."""
    ids = spec.properties.component_ids
    dn, dh = np.zeros((spec.cell_count, len(ids))), np.zeros(spec.cell_count)
    if flashes is None:
        flashes = tuple(
            spec.at_ph(i, n, h)
            for i, (n, h) in enumerate(
                zip(state.amounts_mol[:-1], state.enthalpies_J[:-1], strict=True)
            )
        )
    streams = []

    def stream(source, target, phase, rate):
        report = flashes[source]
        amounts = report.vapor_amounts_mol if phase == "gas" else report.liquid_amounts_mol
        total = sum(amounts.values())
        if not total or not rate:
            return
        energy = report.vapor_enthalpy_J if phase == "gas" else report.liquid_enthalpy_J
        component_rates, power = phase_advection_rate(amounts, energy, rate)
        flux = np.array([component_rates[key] for key in ids])
        dn[source] -= flux
        dn[target] += flux
        dh[source] -= power
        dh[target] += power
        streams.append(
            {
                "source_cell": source,
                "target_cell": target,
                "phase": phase,
                "amounts_mol_s": dict(zip(ids, map(float, flux), strict=True)),
                "enthalpy_W": power,
            }
        )

    for index in range(spec.condenser_index):
        vapor = sum(flashes[index].vapor_amounts_mol.values())
        rate = min(
            spec.vapor_withdrawal_rate_s_inv * vapor,
            spec.equipment.maximum_internal_vapor_rate_mol_s,
        )
        stream(index, index + 1, "gas", rate)
    for index in range(1, spec.condenser_index):
        liquid = sum(flashes[index].liquid_amounts_mol.values())
        stream(
            index,
            index - 1,
            "liquid",
            spec.liquid_overflow_rate_s_inv * max(liquid - spec.liquid_setpoint_mol, 0),
        )
    liquid = sum(flashes[spec.condenser_index].liquid_amounts_mol.values())
    overflow = spec.liquid_overflow_rate_s_inv * max(liquid - spec.liquid_setpoint_mol, 0)
    product = 0 if reflux_ratio is None else overflow / (1 + reflux_ratio)
    stream(spec.condenser_index, spec.condenser_index - 1, "liquid", overflow - product)
    stream(spec.condenser_index, spec.collector_index, "liquid", product)
    power = heater.delivered_power_w(flashes[0].temperature_K)
    cooling = min(
        spec.equipment.maximum_condenser_power_W,
        spec.condenser_conductance_W_K
        * max(flashes[spec.condenser_index].temperature_K - spec.coolant_temperature_K, 0),
    )
    dh[0] += power
    dh[spec.condenser_index] -= cooling
    return DynamicColumnRates(dn, dh, power, cooling, tuple(streams))


@dataclass(frozen=True)
class DynamicColumnResult(PhaseInventoryAdvance[DynamicColumnState]):
    """Common numerical report with actual collector increments."""

    @property
    def fractions(self):
        return tuple(
            {
                "start_s": before.time_s,
                "end_s": after.time_s,
                "amounts_mol": {
                    key: after.amounts_mol[-1][key] - before.amounts_mol[-1][key]
                    for key in before.amounts_mol[-1]
                },
                "enthalpy_J": after.enthalpies_J[-1] - before.enthalpies_J[-1],
            }
            for before, after in zip(self.trajectory[:-1], self.trajectory[1:], strict=True)
        )


def advance_dynamic_column(
    spec: DynamicColumnSpec,
    state: DynamicColumnState,
    *,
    heater: ThermalDeviceSpec,
    duration_s: float,
    reflux_ratio: float | None,
    evaluation_times_s=None,
    max_step_s: float | None = None,
    rtol: float = 1e-8,
    progress: Callable | None = None,
    maximum_volume_L: float | None = None,
    occupied_volume_L: float = 0.0,
) -> DynamicColumnResult:
    ids, count = spec.properties.component_ids, spec.cell_count
    if (
        len(state.amounts_mol) != count
        or any(set(n) != set(ids) for n in state.amounts_mol)
        or not isfinite(duration_s)
        or not 0 < duration_s <= spec.equipment.maximum_duration_s
        or (
            reflux_ratio is not None
            and (
                not isfinite(reflux_ratio)
                or not 0 <= reflux_ratio <= spec.equipment.maximum_reflux_ratio
            )
        )
        or heater.maximum_power_W > spec.equipment.maximum_reboiler_power_W
        or heater.mode != "finite_power"
    ):
        raise ValueError("dynamic column controls/inventory leave the declared equipment domain")
    if (
        not isfinite(occupied_volume_L)
        or occupied_volume_L < 0
        or (
            maximum_volume_L is not None
            and (not isfinite(maximum_volume_L) or maximum_volume_L <= occupied_volume_L)
        )
    ):
        raise ValueError("column volume boundary requires finite positive free volume")
    initial_total = sum(sum(n.values()) for n in state.amounts_mol) + state.withdrawn_product_mol
    if initial_total > spec.equipment.maximum_batch_amount_mol:
        raise ValueError("column batch exceeds equipment amount capacity")
    collector_limit = min(
        spec.collector_capacity_mol,
        spec.equipment.maximum_distillate_cut_fraction * initial_total
        - state.withdrawn_product_mol,
    )
    if collector_limit <= sum(state.amounts_mol[-1].values()):
        raise ColumnOperatingLimitError("column collection capacity is exhausted")
    counters = {"trial_domain_extensions": 0}

    def enthalpy_bounds(index, amounts):
        bounds = spec.properties.ideal_enthalpy_bounds(
            amounts, spec.pressure_Pa, spec.capacities_j_k[index], spec.reference_temperature_K
        )
        if bounds is None:
            bounds = tuple(
                spec.at_tp(index, amounts, t).total_enthalpy_j
                for t in spec.properties.temperature_bounds_K
            )
        return bounds

    events = []

    def event(name, value):
        events.append((name, value))

    for index in range(count - 1):
        minimum = spec.equipment.minimum_bottoms_amount_mol if index == 0 else 1e-6
        event(
            f"dry_cell_{index}",
            lambda candidate, flashes, i=index, floor=minimum: (
                sum(flashes[i].liquid_amounts_mol.values()) - floor
            ),
        )
        if index:
            event(
                f"flood_cell_{index}",
                lambda candidate, flashes, i=index: (
                    spec.tray_maximum_amount_mol - sum(candidate.amounts_mol[i].values())
                ),
            )
        event(
            f"hot_cell_{index}",
            lambda candidate, flashes, i=index: (
                spec.equipment.maximum_temperature_K - flashes[i].temperature_K
            ),
        )
        event(
            f"cold_cell_{index}",
            lambda candidate, flashes, i=index: (
                candidate.enthalpies_J[i] - enthalpy_bounds(i, candidate.amounts_mol[i])[0]
            ),
        )
    event(
        "collector_capacity",
        lambda candidate, flashes: collector_limit - sum(candidate.amounts_mol[-1].values()),
    )
    if maximum_volume_L is not None:

        def free_volume(candidate, flashes):
            total = occupied_volume_L
            for phase in flashes:
                if phase is None:
                    continue
                total += (
                    sum(phase.liquid_amounts_mol.values()) * spec.liquid_molar_holdup_L_mol
                    + sum(phase.vapor_amounts_mol.values())
                    * R_J_PER_MOL_K
                    * phase.temperature_K
                    / spec.pressure_Pa
                    * 1000
                )
            n, h = candidate.amounts_mol[-1], candidate.enthalpies_J[-1]
            if sum(n.values()):
                low, high = enthalpy_bounds(spec.collector_index, n)
                if not low <= h <= high:
                    counters["trial_domain_extensions"] += 1
                collector = spec.at_ph(spec.collector_index, n, min(max(h, low), high))
                total += (
                    sum(collector.liquid_amounts_mol.values()) * spec.liquid_molar_holdup_L_mol
                    + sum(collector.vapor_amounts_mol.values())
                    * R_J_PER_MOL_K
                    * collector.temperature_K
                    / spec.pressure_Pa
                    * 1000
                )
            return maximum_volume_L - total

        event("vessel_volume", free_volume)
    vector_length = count * (len(ids) + 1) + 2
    mask = lil_matrix((vector_length, vector_length), dtype=bool)
    for cell in range(count):
        rows = [cell * 2, cell * 2 + 1, count * 2 + cell]
        for neighbor in range(max(0, cell - 1), min(count - 1, cell + 1) + 1):
            columns = [neighbor * 2, neighbor * 2 + 1, count * 2 + neighbor]
            for row in rows:
                mask[row, columns] = True
    mask[-2, [0, 1, count * 2]] = True
    c = spec.condenser_index
    mask[-1, [c * 2, c * 2 + 1, count * 2 + c]] = True
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=lambda candidate, flashes: column_rates(
            spec, candidate, heater, reflux_ratio, flashes=flashes
        ),
        closure_indices=range(count - 1),
        stop_functions=events,
        enthalpy_bounds=enthalpy_bounds,
        result_type=DynamicColumnResult,
        evaluation_times_s=evaluation_times_s,
        max_step_s=max_step_s,
        rtol=rtol,
        progress=progress,
        jac_sparsity=mask.tocsr(),
        solver_id="finite_equilibrium_column_bdf_v1",
        balance_scale_mol=initial_total,
        diagnostic_counters=counters,
        operating_limit_error=ColumnOperatingLimitError,
        reuse_cell_closures=type(spec) is DynamicColumnSpec,
    )
    return cast(DynamicColumnResult, result)
