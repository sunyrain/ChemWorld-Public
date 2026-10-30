"""Declared thermal fields on original fixed-bed gas and sorbent owners."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec
from chemworld.physchem.fixed_bed import FixedBedSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

FIXED_BED_THERMAL_STATE = "finite_fixed_bed_thermal_fields_v1"


def bed_thermal_key(kind: str, identity: str = "") -> str:
    return json.dumps(["fixed_bed_thermal", kind, identity], separators=(",", ":"))


@dataclass(frozen=True)
class FixedBedThermalSpec:
    """Two-temperature constant-Cv model; initial temperatures are authored once.

    Contact conductance is per bulk bed volume. Axial gas/solid conduction uses
    the corresponding void/solid cross section and original mesh distances.
    A shared finite device is distributed by bulk cell volume, including its
    heating and cooling limits; refinement cannot manufacture extra power.
    """

    bed: FixedBedSpec
    initial_gas_temperatures_K: Mapping[str, float]
    initial_solid_temperatures_K: tuple[float, ...]
    gas_solid_heat_transfer_W_m3_K: float
    gas_thermal_conductivity_W_m_K: float = 0.0
    solid_thermal_conductivity_W_m_K: float = 0.0
    gas_temperature_bounds_K: tuple[float, float] = (200.0, 600.0)
    jacket: ThermalDeviceSpec | None = None
    _gas_ids: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bed, FixedBedSpec):
            raise ValueError("thermal bed requires the original fixed-bed declaration")
        initial = dict(self.initial_gas_temperatures_K)
        if (not initial or any(not isinstance(key, str) or not key for key in initial)
                or set(self.bed.cell_phase_ids) - set(initial)):
            raise ValueError("thermal fields require named original gas cells and boundaries")
        object.__setattr__(self, "_gas_ids", tuple(sorted(initial)))
        object.__setattr__(self, "initial_gas_temperatures_K",
                           MappingProxyType({key: initial[key] for key in self._gas_ids}))
        object.__setattr__(self, "initial_solid_temperatures_K",
                           tuple(self.initial_solid_temperatures_K))
        object.__setattr__(self, "gas_temperature_bounds_K", tuple(self.gas_temperature_bounds_K))
        bounds = self.gas_temperature_bounds_K
        if (len(bounds) != 2 or any(type(t) not in {int, float} or not isfinite(t) for t in bounds)
                or not 0 < bounds[0] < bounds[1]):
            raise ValueError("gas thermal domain must be a finite positive interval")
        if (len(self.initial_solid_temperatures_K) != len(self.bed.cell_phase_ids)
                or any(type(t) not in {int, float} or not isfinite(t)
                       or not bounds[0] < t < bounds[1] for t in initial.values())
                or any(type(t) not in {int, float} or not isfinite(t)
                       or not self.bed.sorbent.temperature_bounds_K[0]
                       < t < self.bed.sorbent.temperature_bounds_K[1]
                       for t in self.initial_solid_temperatures_K)):
            raise ValueError("every initial temperature must be interior to its declared domain")
        for name in ("gas_solid_heat_transfer_W_m3_K", "gas_thermal_conductivity_W_m_K",
                     "solid_thermal_conductivity_W_m_K"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not isfinite(value) or value < 0:
                raise ValueError("thermal conductances must be finite and nonnegative")
        if (self.bed.sorbent.carrier_heat_capacity_J_kg_K <= 0
                or any(law.cv_J_mol_K <= 0 for law in self.bed.gas_calorics.values())):
            raise ValueError("separate gas and solid fields require positive declared Cv")
        if self.jacket is not None and (
            not isinstance(self.jacket, ThermalDeviceSpec) or self.jacket.mode != "finite_power"
            or not self.bed.sorbent.temperature_bounds_K[0] <= self.jacket.setpoint_K
            <= self.bed.sorbent.temperature_bounds_K[1]
        ):
            raise ValueError("thermal bed needs a finite-power device in the solid property domain")

    @property
    def gas_phase_ids(self) -> tuple[str, ...]:
        return self._gas_ids

    @property
    def anchor_phase_id(self) -> str:
        return self.bed.cell_phase_ids[-1]

    @property
    def slot_id(self) -> str:
        return f"{self.bed.bed_id}:thermal_fields"

    def make_slot(self, vessel_id: str) -> StateSlot:
        fields = (
            *(StateFieldSpec(bed_thermal_key("gas_K", phase), "K", mixing="equal", public=True)
              for phase in self.gas_phase_ids),
            *(StateFieldSpec(bed_thermal_key("solid_K", phase), "K", mixing="equal", public=True)
              for phase in self.bed.cell_phase_ids),
            StateFieldSpec(bed_thermal_key("heat_import_J"), "J"),
            StateFieldSpec(bed_thermal_key("heat_export_J"), "J"),
        )
        values = {
            **{bed_thermal_key("gas_K", phase): value
               for phase, value in self.initial_gas_temperatures_K.items()},
            **{bed_thermal_key("solid_K", phase): value for phase, value in zip(
                self.bed.cell_phase_ids, self.initial_solid_temperatures_K, strict=True)},
            bed_thermal_key("heat_import_J"): 0.0,
            bed_thermal_key("heat_export_J"): 0.0,
        }
        return StateSlot(StateSlotSpec(
            self.slot_id, "equipment", self.bed.bed_id, FIXED_BED_THERMAL_STATE,
            fields, batch_reset="preserve"), vessel_id, values, dict(values))

    def validate_slot(self, slot: StateSlot, vessel_id: str) -> None:
        expected = self.make_slot(vessel_id)
        if (slot.spec != expected.spec or slot.location_id != vessel_id
                or slot.initial_values != expected.initial_values
                or set(slot.values) != set(expected.values)):
            raise ValueError("thermal fields must retain original owner, layout and initial state")
        for key, value in slot.values.items():
            if type(value) not in {int, float} or not isfinite(value):
                raise ValueError("thermal state must contain finite real values")
            kind = json.loads(key)[1]
            bounds = (self.gas_temperature_bounds_K if kind == "gas_K"
                      else self.bed.sorbent.temperature_bounds_K)
            if kind.endswith("_K"):
                if not bounds[0] - 1e-7 <= value <= bounds[1] + 1e-7:
                    raise ValueError("thermal state temperature leaves its declared domain")
            elif value < 0:
                raise ValueError("cumulative heat receipts must be nonnegative")

    def project_slot(self, slot: StateSlot, temperatures: Mapping[str, float]) -> StateSlot:
        if set(temperatures) - set(slot.values):
            raise ValueError("thermal projection references undeclared original fields")
        projected = replace(slot, values={**slot.values, **temperatures})
        self.validate_slot(projected, slot.location_id)
        return projected

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "finite-two-temperature-fixed-bed-v1",
            "bed": self.bed.to_dict(),
            "gas_component_ids": list(self.bed.component_ids),
            "adsorbate_ids": list(self.bed.sorbent.component_ids),
            "initial_gas_temperatures_K": dict(self.initial_gas_temperatures_K),
            "initial_solid_temperatures_K": list(self.initial_solid_temperatures_K),
            "gas_solid_heat_transfer_W_m3_K": self.gas_solid_heat_transfer_W_m3_K,
            "gas_thermal_conductivity_W_m_K": self.gas_thermal_conductivity_W_m_K,
            "solid_thermal_conductivity_W_m_K": self.solid_thermal_conductivity_W_m_K,
            "gas_temperature_bounds_K": list(self.gas_temperature_bounds_K),
            "jacket": None if self.jacket is None else self.jacket.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FixedBedThermalSpec:
        values = dict(raw)
        if values.pop("model", None) != "finite-two-temperature-fixed-bed-v1":
            raise ValueError("unknown thermal fixed-bed declaration")
        base = dict(values.pop("bed"))
        gas_order, bound_order = values.pop("gas_component_ids"), values.pop("adsorbate_ids")
        for name in ("molecular_weights_kg_mol", "gas_calorics"):
            mapping = base[name]
            if len(gas_order) != len(mapping) or set(gas_order) != set(mapping):
                raise ValueError("thermal bed must retain its complete gas component order")
            base[name] = {key: mapping[key] for key in gas_order}
        sorbent = dict(base["sorbent"])
        for name in ("reference_affinity_L_per_mol", "mass_transfer_rates_s_inv",
                     "adsorption_energy_J_mol"):
            mapping = sorbent[name]
            if len(bound_order) != len(mapping) or set(bound_order) != set(mapping):
                raise ValueError("thermal bed must retain its complete adsorbate order")
            sorbent[name] = {key: mapping[key] for key in bound_order}
        base["sorbent"] = sorbent
        values["bed"] = FixedBedSpec.from_dict(base)
        if values["jacket"] is not None:
            values["jacket"] = ThermalDeviceSpec.from_dict(values["jacket"])
        return cls(**values)
