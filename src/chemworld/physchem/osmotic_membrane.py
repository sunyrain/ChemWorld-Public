"""Finite isothermal osmosis on the original aqueous n/H integration engine.

Perfect salt rejection, homogeneous interfaces and two independently fixed
pressures are explicit approximations. Positive mobility multiplies the signed
water chemical-potential difference; FO, PRO and reverse flow share one law.
The nominal volume flux uses standard water volume. Actual tank V, H, U and G
come from the same original aqueous property table, including mixing terms.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isclose, isfinite
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from chemworld.physchem.aqueous_transport import AqueousTransportLaw, BinaryAqueousProperties
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.transport_devices import TransportInventory

SCHEMA = "finite-isothermal-osmotic-membrane-v1"


@dataclass(frozen=True)
class OsmoticMembraneSpec:
    properties: BinaryAqueousProperties
    water_permeability: AqueousTransportLaw
    bath: ThermalDeviceSpec
    membrane_area_m2: float
    pressures_Pa: tuple[float, float]
    capacities_L: tuple[float, float]
    minimum_water_mol: float = 0.005
    maximum_concentration_mol_m3: float = 200.0
    maximum_pressure_difference_Pa: float = 1e6
    right_vessel_id: str = "draw"

    def __post_init__(self) -> None:
        for name in ("pressures_Pa", "capacities_L"):
            values = tuple(getattr(self, name))
            if len(values) != 2:
                raise ValueError("osmotic membrane requires two finite liquid sides")
            object.__setattr__(self, name, values)
        values = (
            self.membrane_area_m2,
            *self.pressures_Pa,
            *self.capacities_L,
            self.minimum_water_mol,
            self.maximum_concentration_mol_m3,
            self.maximum_pressure_difference_Pa,
        )
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in values):
            raise ValueError("osmotic membrane requires positive finite geometry and bounds")
        if not self.right_vessel_id or self.bath.mode != "isothermal" or not self.bath.enabled:
            raise ValueError("osmotic membrane requires an enabled finite isothermal bath")
        if self.water_permeability.unit != "m/(Pa*s)" or self.water_permeability.source_id not in {
            source.source_id for source in self.properties.database.sources
        }:
            raise ValueError("osmotic permeability requires m/(Pa*s) and a database source")
        if abs(self.pressures_Pa[0] - self.pressures_Pa[1]) > self.maximum_pressure_difference_Pa:
            raise ValueError("osmotic membrane exceeds declared pressure-difference rating")
        # The constitutive query retains the original property's explicit domain.
        # Qualification below covers ideal and Davies; this is not a new data fit.
        for pressure in self.pressures_Pa:
            self.properties.liquid(self.properties.amounts(1, 0), self.temperature_K, pressure)
        self.water_permeability.evaluate(self.temperature_K, sum(self.pressures_Pa) / 2, 0)
        if self.maximum_concentration_mol_m3 > self.water_permeability.maximum_concentration_mol_m3:
            raise ValueError("osmotic salinity limit exceeds the permeability law domain")

    @property
    def temperature_K(self) -> float:  # noqa: N802
        return self.bath.setpoint_K

    def enthalpy(self, index: int, amounts: Mapping[str, float]) -> float:
        return self.properties.enthalpy(amounts, self.temperature_K, self.pressures_Pa[index])

    def to_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            json.loads(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        **{key: getattr(self, key) for key in self.__dataclass_fields__},
                        "properties": self.properties.to_dict(),
                        "water_permeability": self.water_permeability.to_dict(),
                        "bath": self.bath.to_dict(),
                    }
                )
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OsmoticMembraneSpec":
        values = dict(payload)
        if values.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported osmotic membrane schema")
        return cls(
            **{
                **values,
                "properties": BinaryAqueousProperties.from_dict(values["properties"]),
                "water_permeability": AqueousTransportLaw(**values["water_permeability"]),
                "bath": ThermalDeviceSpec.from_dict(values["bath"]),
            }
        )


def _point(spec: OsmoticMembraneSpec, amounts: Any) -> dict[str, Any]:
    prop, t = spec.properties, spec.temperature_K
    concentrations = [prop.concentration(n) for n in amounts]
    potentials = [
        cast(
            float,
            prop.database.chemical_potentials(
                n,
                temperature_K=t,
                pressure_Pa=p,
            ).chemical_potentials_J_mol[prop.water_id],
        )
        for n, p in zip(amounts, spec.pressures_Pa, strict=True)
    ]
    delta = potentials[0] - potentials[1]
    permeability = spec.water_permeability.evaluate(
        t,
        sum(spec.pressures_Pa) / 2,
        sum(concentrations) / 2,
    )
    flux = permeability * delta / prop.water_volume_m3_mol
    transfer = spec.membrane_area_m2 * flux / prop.water_volume_m3_mol
    rates = []
    for i, sign in enumerate((-1, 1)):
        dn = {key: sign * transfer if key == prop.water_id else 0.0 for key in prop.component_ids}
        rates.append(prop.enthalpy_rate(amounts[i], dn, t, spec.pressures_Pa[i]))
    return {
        "temperature_K": t,
        "pressures_Pa": list(spec.pressures_Pa),
        "concentrations_mol_m3": concentrations,
        "volumes_L": [
            1000 * prop.volume(n, t, p) for n, p in zip(amounts, spec.pressures_Pa, strict=True)
        ],
        "water_chemical_potentials_J_mol": potentials,
        "water_chemical_potential_difference_J_mol": delta,
        "water_permeability_m_Pa_s": permeability,
        "nominal_standard_water_volume_flux_m_s": flux,
        "water_transfer_mol_s": transfer,
        "enthalpy_rates_W": rates,
        "bath_heat_into_liquids_W": sum(rates),
        "entropy_production_W_K": transfer * delta / t,
        "transport_evaluation": "common T; arithmetic mean P and salt concentration",
    }


def osmotic_membrane_point(spec: OsmoticMembraneSpec, amounts: Any) -> dict[str, Any]:
    """Read current original inventories; no state or volume-flow clipping."""
    if len(amounts) != 2:
        raise ValueError("osmotic membrane requires exactly two liquid inventories")
    for n in amounts:
        spec.properties.validate_amounts(n)
    result = _point(spec, amounts)
    if max(result["concentrations_mol_m3"]) > spec.maximum_concentration_mol_m3 * (1 + 1e-10):
        raise ValueError("osmotic membrane leaves its declared salinity domain")
    return result


def osmotic_membrane_inventory(
    spec: OsmoticMembraneSpec,
    left_amounts: Mapping[str, float],
    right_amounts: Mapping[str, float],
) -> TransportInventory:
    amounts = (dict(left_amounts), dict(right_amounts))
    point = osmotic_membrane_point(spec, amounts)
    if any(n[spec.properties.water_id] < spec.minimum_water_mol * (1 - 1e-10) for n in amounts):
        raise ValueError("osmotic membrane inventory is below its finite water reserve")
    if any(v > c * (1 + 1e-10) for v, c in zip(point["volumes_L"], spec.capacities_L, strict=True)):
        raise ValueError("osmotic membrane inventory exceeds actual finite capacity")
    return TransportInventory(amounts, tuple(spec.enthalpy(i, n) for i, n in enumerate(amounts)), 0)


def advance_osmotic_membrane(
    spec: OsmoticMembraneSpec,
    state: TransportInventory,
    *,
    duration_s: float,
    rtol: float = 1e-10,
    max_step_s: float | None = None,
    evaluation_times_s: Any = None,
    progress: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Integrate two water coordinates and their original full enthalpies.

    Salt has identically zero derivatives. Its exact elimination here prevents
    numerical Jacobian probes from inventing non-electroneutral salt. The view
    is temporary; full original inventories remain the only persisted truth.
    """
    validated = osmotic_membrane_inventory(spec, *state.amounts_mol)
    if len(state.enthalpies_J) != 2 or any(
        not isclose(h, expected, rel_tol=1e-8, abs_tol=3e-6)
        for h, expected in zip(state.enthalpies_J, validated.enthalpies_J, strict=True)
    ):
        raise ValueError("osmotic membrane original H disagrees with common isothermal state")
    initial_point = osmotic_membrane_point(spec, state.amounts_mol)
    q0 = initial_point["bath_heat_into_liquids_W"]
    if q0 > spec.bath.maximum_power_W + 1e-12 or -q0 > spec.bath.maximum_cooling_power_W + 1e-12:
        raise ValueError("osmotic membrane isothermal bath power rating is insufficient")
    water = spec.properties.water_id
    coordinate_spec = SimpleNamespace(properties=SimpleNamespace(component_ids=(water,)))
    coordinates = replace(state, amounts_mol=tuple({water: n[water]} for n in state.amounts_mol))

    def amounts(candidate: Any) -> tuple[dict[str, float], ...]:
        return tuple(
            {**original, water: current[water]}
            for original, current in zip(state.amounts_mol, candidate.amounts_mol, strict=True)
        )

    def rates(candidate: Any, _: Any) -> PhaseInventoryRates:
        point = _point(spec, amounts(candidate))
        flow, q = point["water_transfer_mol_s"], point["bath_heat_into_liquids_W"]
        return PhaseInventoryRates(
            np.array([[-flow], [flow]]),
            np.array(point["enthalpy_rates_W"]),
            max(q, 0),
            max(-q, 0),
            (),
        )

    # A scalar autonomous water-transfer ODE cannot cross an equilibrium.
    # Apply water/salinity limits to its donor and capacity to both sides;
    # an initially full tank moving inward must not block a valid reverse flow.
    forward = initial_point["water_transfer_mol_s"] >= 0
    donor = 0 if forward else 1
    stops: list[Any] = [
        (
            f"water_reserve_{donor}",
            lambda s, _: s.amounts_mol[donor][water] - spec.minimum_water_mol,
        ),
        (
            f"salinity_{donor}",
            lambda s, _: (
                spec.maximum_concentration_mol_m3 - spec.properties.concentration(amounts(s)[donor])
            ),
        ),
    ]
    for i in range(2):

        def capacity(s: Any, _: Any, i: int = i) -> float:
            n = amounts(s)[i]
            margin = spec.capacities_L[i] - 1000 * spec.properties.volume(
                n,
                spec.temperature_K,
                spec.pressures_Pa[i],
            )
            # At the donor's exact initial full boundary, moving water out is
            # admissible. Later positive-time crossings are still monitored.
            if s.time_s == state.time_s and i == donor and abs(margin) < 1e-12:
                return 1e-12
            return margin

        stops.append((f"capacity_{i}", capacity))
    stops.extend(
        (
            (
                "bath_heating_power",
                lambda s, _: (
                    spec.bath.maximum_power_W
                    + 1e-12
                    - _point(spec, amounts(s))["bath_heat_into_liquids_W"]
                ),
            ),
            (
                "bath_cooling_power",
                lambda s, _: (
                    spec.bath.maximum_cooling_power_W
                    + 1e-12
                    + _point(spec, amounts(s))["bath_heat_into_liquids_W"]
                ),
            ),
        )
    )
    result = integrate_phase_inventory(
        coordinate_spec,
        coordinates,
        duration_s=duration_s,
        rates=rates,
        closure_indices=(),
        stop_functions=stops,
        enthalpy_bounds=None,
        evaluation_times_s=evaluation_times_s,
        max_step_s=max_step_s,
        rtol=rtol,
        progress=progress,
        initial_boundary="stop",
        solver_id="finite_isothermal_osmotic_bdf_v1",
    )
    final = replace(result.final_state, amounts_mol=amounts(result.final_state))
    # Preserve integrated H; validate against its constitutive endpoint, never
    # replace the owned energy to conceal an integration/closure discrepancy.
    for i, n in enumerate(final.amounts_mol):
        spec.properties.validate_amounts(n)
        if not isclose(final.enthalpies_J[i], spec.enthalpy(i, n), rel_tol=1e-8, abs_tol=3e-6):
            raise RuntimeError("osmotic integration failed common-temperature H closure")
    residuals = {
        key: sum(n[key] for n in final.amounts_mol) - sum(n[key] for n in state.amounts_mol)
        for key in spec.properties.component_ids
    }
    result = replace(
        result,
        initial_state=state,
        final_state=final,
        trajectory=tuple(replace(s, amounts_mol=amounts(s)) for s in result.trajectory),
        component_residuals_mol=residuals,
        solver_diagnostic={**result.solver_diagnostic, "static_salt_eliminated": True},
    )
    initial_props, final_props = [
        [
            spec.properties.liquid(n, spec.temperature_K, p)
            for n, p in zip(s.amounts_mol, spec.pressures_Pa, strict=True)
        ]
        for s in (state, final)
    ]
    dh = sum(final.enthalpies_J) - sum(state.enthalpies_J)
    du = sum(
        b.internal_energy_J - a.internal_energy_J
        for a, b in zip(initial_props, final_props, strict=True)
    )
    dg = sum(
        b.gibbs_energy_J - a.gibbs_energy_J for a, b in zip(initial_props, final_props, strict=True)
    )
    work = sum(
        p * (b.volume_m3 - a.volume_m3)
        for p, a, b in zip(spec.pressures_Pa, initial_props, final_props, strict=True)
    )
    heating, cooling = (
        final.heater_energy_J - state.heater_energy_J,
        final.condenser_energy_J - state.condenser_energy_J,
    )
    return result, {
        "initial": initial_point,
        "final": _point(spec, final.amounts_mol),
        "water_left_to_right_mol": final.amounts_mol[1][water] - state.amounts_mol[1][water],
        "bath_heating_J": heating,
        "bath_cooling_J": cooling,
        "bath_heat_into_liquids_J": heating - cooling,
        "enthalpy_change_J": dh,
        "internal_energy_change_J": du,
        "boundary_work_J": work,
        "internal_energy_closure_J": du - dh + work,
        "entropy_production_J_K": -dg / spec.temperature_K,
        "actual_duration_s": result.actual_duration_s,
        "quantity_kind": "accepted_advance_receipt",
        "energy_scope": (
            "finite isothermal bath Q=delta H; fixed-pressure moving boundaries W=sum P delta V"
        ),
        "shaft_and_electrical_output": "none; pressure boundary work is not a generator",
    }
