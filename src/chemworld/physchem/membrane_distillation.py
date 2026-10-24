"""Finite direct-contact thermal membrane on the original aqueous n/H engine.

Dry pure-vapor Knudsen pores, well-mixed interfaces and negligible membrane
holdup are explicit approximations. Mean-T vapor H carries signed water flux;
conduction is additional. This is neither a vacuum nor a noncondensable-air
model, and pressure-driven liquid leakage is outside its domain.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite, pi, sqrt
from typing import Any, cast

import numpy as np

from chemworld.physchem.aqueous_database import AqueousSpecies
from chemworld.physchem.aqueous_transport import BinaryAqueousProperties
from chemworld.physchem.enthalpy import caloric_temperature
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.transport_devices import TransportInventory


@dataclass(frozen=True)
class MembraneDistillationSpec:
    properties: BinaryAqueousProperties
    vapor: AqueousSpecies
    source_id: str
    membrane_area_m2: float
    pore_radius_m: float
    thickness_m: float
    porosity: float
    tortuosity: float
    effective_conductivity_W_m_K: float
    molecular_collision_diameter_m: float
    liquid_entry_pressure_Pa: float
    capacities_L: tuple[float, float]
    pressures_Pa: tuple[float, float] = (100000.0, 100000.0)
    temperature_bounds_K: tuple[float, float] = (280.0, 340.0)
    minimum_knudsen_number: float = 10.0
    maximum_relative_temperature_difference: float = 0.1
    maximum_concentration_mol_m3: float = 200.0
    minimum_water_mol: float = 0.01
    support_heat_capacities_J_K: tuple[float, float] = (2.0, 2.0)
    reference_temperature_K: float = 298.15
    cold_vessel_id: str = "cold"

    def __post_init__(self) -> None:
        for name in (
            "capacities_L",
            "pressures_Pa",
            "temperature_bounds_K",
            "support_heat_capacities_J_K",
        ):
            value = tuple(getattr(self, name))
            if len(value) != 2:
                raise ValueError("thermal membrane needs exactly two finite liquid sides")
            object.__setattr__(self, name, value)
        positive = (
            self.membrane_area_m2,
            self.pore_radius_m,
            self.thickness_m,
            self.porosity,
            self.tortuosity,
            self.molecular_collision_diameter_m,
            self.liquid_entry_pressure_Pa,
            *self.capacities_L,
            *self.pressures_Pa,
            *self.temperature_bounds_K,
            self.minimum_knudsen_number,
            self.maximum_relative_temperature_difference,
            self.maximum_concentration_mol_m3,
            self.minimum_water_mol,
            self.reference_temperature_K,
        )
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive):
            raise ValueError("thermal membrane needs finite positive geometry/domain declarations")
        if (
            any(
                not isfinite(v) or v < 0
                for v in (self.effective_conductivity_W_m_K, *self.support_heat_capacities_J_K)
            )
            or not self.cold_vessel_id
        ):
            raise ValueError("thermal membrane needs finite conductance, support and vessel")
        if (
            self.porosity > 1
            or self.tortuosity < 1
            or self.minimum_knudsen_number < 10
            or self.maximum_relative_temperature_difference > 0.1
            or not self.temperature_bounds_K[0] < self.temperature_bounds_K[1]
            or not self.temperature_bounds_K[0]
            <= self.reference_temperature_K
            <= self.temperature_bounds_K[1]
        ):
            raise ValueError("thermal membrane leaves its declared reduced Knudsen domain")
        water = self.properties.database.solvent_species
        sources = {s.source_id for s in self.properties.database.sources}
        if (
            self.vapor.phase != "gas"
            or self.vapor.charge != 0
            or dict(self.vapor.composition) != dict(water.composition)
            or self.vapor.standard_state.reference_basis_id
            != water.standard_state.reference_basis_id
            or self.vapor.source_id not in sources
            or self.vapor.standard_state.source_id not in sources
            or self.source_id not in sources
        ):
            raise ValueError(
                "liquid/vapor require identical composition and same reference basis/source"
            )
        if abs(self.pressures_Pa[0] - self.pressures_Pa[1]) >= self.liquid_entry_pressure_Pa:
            raise ValueError("liquid pressure difference reaches the declared entry pressure")
        for t in self.temperature_bounds_K:
            for p in self.pressures_Pa:
                self.properties.liquid(self.properties.amounts(1, 0), t, p)
                self.vapor.standard_state.evaluate(t, p)

    def enthalpy(self, index: int, amounts: Mapping[str, float], temperature: float) -> float:
        return self.properties.enthalpy(amounts, temperature, self.pressures_Pa[index]) + (
            self.support_heat_capacities_J_K[index] * (temperature - self.reference_temperature_K)
        )

    def at_ph(self, index: int, amounts: Mapping[str, float], enthalpy: float) -> float:
        return caloric_temperature(
            lambda t: self.enthalpy(index, amounts, t),
            enthalpy,
            self.temperature_bounds_K,
            energy_basis="enthalpy",
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    **{name: getattr(self, name) for name in self.__dataclass_fields__},
                    "properties": self.properties.to_dict(),
                    "vapor": self.vapor.to_dict(),
                }
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MembraneDistillationSpec":
        return cls(
            **{
                **payload,
                "properties": BinaryAqueousProperties.from_dict(payload["properties"]),
                "vapor": AqueousSpecies.from_dict(payload["vapor"]),
            }
        )


def _point(spec: MembraneDistillationSpec, amounts: Any, temperatures: Any) -> dict[str, Any]:
    vapor_pressures, volumes, concentrations = [], [], []
    for n, t, p in zip(amounts, temperatures, spec.pressures_Pa, strict=True):
        report = spec.properties.database.chemical_potentials(n, temperature_K=t, pressure_Pa=p)
        mu = cast(float, report.chemical_potentials_J_mol[spec.properties.water_id])
        gas = spec.vapor.standard_state.evaluate(t, p)
        vapor_pressures.append(gas.reference_pressure_Pa * exp((mu - gas.gibbs_J_mol) / (R * t)))
        volumes.append(1000 * spec.properties.volume(n, t, p))
        concentrations.append(spec.properties.concentration(n))
    mean = sum(temperatures) / 2
    coefficient = (
        2
        * spec.porosity
        * spec.pore_radius_m
        / (3 * spec.tortuosity * spec.thickness_m)
        * sqrt(8 / (pi * spec.properties.water_mass_kg_mol * R * mean))
    )
    flux = coefficient * (vapor_pressures[0] - vapor_pressures[1])
    vapor_h = spec.vapor.standard_state.evaluate(mean, spec.pressures_Pa[0]).enthalpy_J_mol
    conduction = (
        spec.membrane_area_m2
        * spec.effective_conductivity_W_m_K
        / spec.thickness_m
        * (temperatures[0] - temperatures[1])
    )
    # Hard-sphere mean free path uses pure-vapor pressure, not liquid pressure.
    mean_free_path = (
        1.380649e-23
        * min(temperatures)
        / (sqrt(2) * pi * spec.molecular_collision_diameter_m**2 * max(vapor_pressures))
    )
    return {
        "temperatures_K": list(temperatures),
        "volumes_L": volumes,
        "concentrations_mol_m3": concentrations,
        "vapor_pressures_Pa": vapor_pressures,
        "water_flux_mol_m2_s": flux,
        "water_transfer_mol_s": spec.membrane_area_m2 * flux,
        "knudsen_coefficient_mol_m2_s_Pa": coefficient,
        "knudsen_number": mean_free_path / (2 * spec.pore_radius_m),
        "vapor_molar_enthalpy_J_mol": vapor_h,
        "conductive_heat_W": conduction,
        "vapor_enthalpy_flow_W": spec.membrane_area_m2 * flux * vapor_h,
        "membrane_energy_flow_W": spec.membrane_area_m2 * flux * vapor_h + conduction,
        "relative_temperature_difference": abs(temperatures[0] - temperatures[1]) / mean,
        "maximum_liquid_vapor_pressure_difference_Pa": max(
            abs(p - v) for p, v in zip(spec.pressures_Pa, vapor_pressures, strict=True)
        ),
    }


def membrane_distillation_point(
    spec: MembraneDistillationSpec, amounts: Any, temperatures: Any
) -> dict[str, Any]:
    if len(amounts) != 2 or len(temperatures) != 2:
        raise ValueError("thermal membrane requires exactly two liquid sides")
    for n, t in zip(amounts, temperatures, strict=True):
        spec.properties.validate_amounts(n)
        if not spec.temperature_bounds_K[0] <= t <= spec.temperature_bounds_K[1]:
            raise ValueError("thermal membrane temperature is outside declared domain")
    row = _point(spec, amounts, temperatures)
    if (
        row["knudsen_number"] < spec.minimum_knudsen_number
        or row["relative_temperature_difference"]
        > spec.maximum_relative_temperature_difference * (1 + 1e-10)
        or max(row["concentrations_mol_m3"]) > spec.maximum_concentration_mol_m3 * (1 + 1e-10)
        or row["maximum_liquid_vapor_pressure_difference_Pa"] >= spec.liquid_entry_pressure_Pa
        or any(v >= p for v, p in zip(row["vapor_pressures_Pa"], spec.pressures_Pa, strict=True))
    ):
        raise ValueError("thermal membrane leaves Knudsen/temperature/salinity/entry domain")
    return row


def membrane_distillation_inventory(
    spec: MembraneDistillationSpec,
    hot_amounts: Mapping[str, float],
    cold_amounts: Mapping[str, float],
    hot_temperature_K: float,
    cold_temperature_K: float,
) -> TransportInventory:
    amounts = (dict(hot_amounts), dict(cold_amounts))
    temperatures = (hot_temperature_K, cold_temperature_K)
    row = membrane_distillation_point(spec, amounts, temperatures)
    if any(n[spec.properties.water_id] < spec.minimum_water_mol for n in amounts) or any(
        v > capacity for v, capacity in zip(row["volumes_L"], spec.capacities_L, strict=True)
    ):
        raise ValueError("thermal membrane inventory exceeds finite water/capacity bounds")
    return TransportInventory(
        amounts,
        tuple(
            spec.enthalpy(i, n, t)
            for i, (n, t) in enumerate(zip(amounts, temperatures, strict=True))
        ),
        0.0,
    )


def validate_membrane_thermal_devices(heater: ThermalDeviceSpec, cooler: ThermalDeviceSpec) -> None:
    if (
        heater.mode != "finite_power"
        or cooler.mode != "finite_power"
        or heater.maximum_cooling_power_W != 0
        or cooler.maximum_power_W != 0
    ):
        raise ValueError("thermal membrane requires heat-only hot and cool-only cold utilities")


def advance_membrane_distillation(
    spec: MembraneDistillationSpec,
    state: TransportInventory,
    *,
    heater: ThermalDeviceSpec,
    cooler: ThermalDeviceSpec,
    duration_s: float,
    rtol: float = 1e-8,
    max_step_s: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    validate_membrane_thermal_devices(heater, cooler)
    initial_t = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True))
    )
    membrane_distillation_point(spec, state.amounts_mol, initial_t)
    ids, water = spec.properties.component_ids, spec.properties.water_id

    def rates(candidate: Any, temperatures: Any) -> PhaseInventoryRates:
        point = _point(spec, candidate.amounts_mol, temperatures)
        dn = np.zeros((2, len(ids)))
        dn[:, ids.index(water)] = (-point["water_transfer_mol_s"], point["water_transfer_mol_s"])
        qh, qc = (
            heater.delivered_power_w(temperatures[0]),
            -cooler.delivered_power_w(temperatures[1]),
        )
        q = point["membrane_energy_flow_W"]
        return PhaseInventoryRates(dn, np.array([qh - q, q - qc]), qh, qc, ())

    def bounds(index: int, n: Any) -> tuple[float, float]:
        return tuple(spec.enthalpy(index, n, t) for t in spec.temperature_bounds_K)  # type: ignore[return-value]

    stops: list[Any] = []
    for i in range(2):
        stops.extend(
            (
                (
                    f"water_pickup_{i}",
                    lambda s, t, i=i: s.amounts_mol[i][water] - spec.minimum_water_mol,
                ),
                (
                    f"capacity_{i}",
                    lambda s, t, i=i: (
                        spec.capacities_L[i]
                        - 1000
                        * spec.properties.volume(s.amounts_mol[i], t[i], spec.pressures_Pa[i])
                    ),
                ),
                (
                    f"salinity_{i}",
                    lambda s, t, i=i: (
                        spec.maximum_concentration_mol_m3
                        - spec.properties.concentration(s.amounts_mol[i])
                    ),
                ),
                (
                    f"minimum_temperature_{i}",
                    lambda s, t, i=i: s.enthalpies_J[i] - bounds(i, s.amounts_mol[i])[0],
                ),
                (
                    f"maximum_temperature_{i}",
                    lambda s, t, i=i: bounds(i, s.amounts_mol[i])[1] - s.enthalpies_J[i],
                ),
            )
        )
    stops.extend(
        (
            (
                "temperature_span",
                lambda s, t: (
                    spec.maximum_relative_temperature_difference - abs(t[0] - t[1]) / (sum(t) / 2)
                ),
            ),
            (
                "knudsen_domain",
                lambda s, t: (
                    _point(spec, s.amounts_mol, t)["knudsen_number"] - spec.minimum_knudsen_number
                ),
            ),
            (
                "liquid_entry",
                lambda s, t: (
                    spec.liquid_entry_pressure_Pa
                    - _point(spec, s.amounts_mol, t)["maximum_liquid_vapor_pressure_difference_Pa"]
                ),
            ),
            (
                "bulk_boiling",
                lambda s, t: min(
                    p - v
                    for p, v in zip(
                        spec.pressures_Pa,
                        _point(spec, s.amounts_mol, t)["vapor_pressures_Pa"],
                        strict=True,
                    )
                ),
            ),
        )
    )
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=rates,
        closure_indices=(0, 1),
        stop_functions=stops,
        enthalpy_bounds=bounds,
        rtol=rtol,
        max_step_s=max_step_s,
        initial_boundary="stop",
        solver_id="finite_direct_contact_membrane_bdf_v1",
    )
    final = result.final_state
    final_t = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(final.amounts_mol, final.enthalpies_J, strict=True))
    )
    for n in final.amounts_mol:
        spec.properties.validate_amounts(n)
    qh = final.heater_energy_J - state.heater_energy_J
    qc = final.condenser_energy_J - state.condenser_energy_J
    transfer = qh - (final.enthalpies_J[0] - state.enthalpies_J[0])
    return result, {
        "initial": _point(spec, state.amounts_mol, initial_t),
        "final": _point(spec, final.amounts_mol, final_t),
        "heater_energy_into_hot_J": qh,
        "cooler_energy_out_of_cold_J": qc,
        "membrane_energy_hot_to_cold_J": transfer,
        "membrane_energy_closure_J": transfer
        - (final.enthalpies_J[1] - state.enthalpies_J[1] + qc),
        "water_hot_to_cold_mol": final.amounts_mol[1][water] - state.amounts_mol[1][water],
        "actual_duration_s": result.actual_duration_s,
        "quantity_kind": "accepted_advance_receipt",
        "energy_scope": "signed mean-T vapor enthalpy plus membrane conduction; finite utilities",
    }
